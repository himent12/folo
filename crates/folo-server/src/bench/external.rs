//! External entries: predictions made anywhere else, checked strictly and gathered into what an entry is graded
//! on. Every error names the first record that is wrong, in plain words.
//!
//! An entry arrives in one request or in parts ([`Filling`]: an open entry). Each part is checked whole before
//! any of it is taken, so a part that is refused can be fixed and sent again. A record that was already received
//! with exactly the same content is accepted and ignored (a client retrying a part whose answer it lost gets no
//! `400`); only a hash of each received record is kept for that. A (condition, severity) is graded (its mAP pair,
//! all the result needs of it) as soon as its last image has arrived and its boxes are dropped, so a client that
//! sends pass by pass costs the server one pass of memory; the unchanged images' detections stay until the entry
//! is finished (the evaluation and the per-image outcomes need them). What an open entry holds is capped
//! ([`MAX_HELD_BOXES`]), and one that gets no part for a while expires ([`expire`]).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{BenchContestant, BenchDetection, BenchEntry, BenchEntryStatus, BenchEvent, BenchPrediction};
use folo_bench::score::{self, Map, Sample};
use folo_bench::{Condition, conditions};

use super::model::{self, write_closed};
use super::results;
use super::state::{BenchFile, Slot};
use super::{add_entry, new_entry, status_name};
use crate::error::{Error, Result};
use crate::util::{rfc3339, unix_now};

/// Most boxes one record carries.
pub const MAX_BOXES: usize = 1000;
/// Most boxes an open entry holds in memory at once (the unchanged images' and those of variants still
/// incomplete): a client that sends pass by pass never comes near it.
pub const MAX_HELD_BOXES: u64 = 20_000_000;
/// How long an open entry waits for its next part before it expires.
pub const OPEN_TIMEOUT: Duration = Duration::from_secs(30 * 60);

fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", message)
}

/// How a record is shown in a message: `prediction 3 (image '17', haze 3)`.
fn record(i: usize, p: &BenchPrediction) -> String {
    match (&p.condition, p.severity) {
        (Some(c), Some(s)) => format!("prediction {} (image '{}', {c} {s})", i + 1, p.image),
        (Some(c), None) => format!("prediction {} (image '{}', {c})", i + 1, p.image),
        _ => format!("prediction {} (image '{}', unchanged)", i + 1, p.image),
    }
}

/// A record's content as a 64-bit FNV-1a digest: its boxes (classes, coordinates, and scores when the contestant
/// gives confidences), in order. Its image, condition and severity are where it is kept.
fn content(p: &BenchPrediction, scored: bool) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01B3);
        }
    };
    eat(p.boxes.len() as u64);
    for d in &p.boxes {
        eat(d.class as u64);
        eat(if scored { d.score.to_bits() as u64 } else { 0 });
        for v in [d.x1, d.y1, d.x2, d.y2] {
            eat(v.to_bits() as u64);
        }
    }
    h
}

/// The answers of one pass (the unchanged images, or one planned variant) so far.
struct PassFill {
    /// Per image: the content hash of its record, once received.
    seen: Vec<Option<u64>>,
    /// Per image: its detections, until the pass is graded.
    boxes: Vec<Vec<BenchDetection>>,
    count: usize,
    /// The variant's mAP pair, once every image has arrived (its boxes are then dropped).
    map: Option<Map>,
}

impl PassFill {
    fn new(n: usize) -> PassFill {
        PassFill { seen: vec![None; n], boxes: vec![Vec::new(); n], count: 0, map: None }
    }
}

/// What a finished external entry is graded on.
pub struct Finished {
    /// The unchanged images, in the benchmark's order.
    pub clean: Vec<Sample>,
    /// The mAP pair of every (condition, severity) of the conditions that were answered, in plan order.
    pub runs: Vec<(Condition, u8, Map)>,
    /// Every condition of the plan was answered.
    pub complete: bool,
}

/// An external entry being filled: what has arrived, checked against the benchmark. See the module docs.
pub struct Filling {
    file: BenchFile,
    scored: bool,
    variants: Vec<(Condition, u8)>,
    index: HashMap<String, usize>,
    clean: PassFill,
    passes: Vec<PassFill>,
    /// Records received.
    pub received: u64,
    /// Boxes held in memory now.
    held: u64,
    /// The most boxes it may hold ([`MAX_HELD_BOXES`]).
    max_held: u64,
    /// The entry was finished or stopped: nothing more is taken.
    pub closed: bool,
    /// When its progress was last sent.
    pub last_event: Option<Instant>,
    /// When it was created or last took a part.
    pub last_part: Instant,
}

impl Filling {
    /// An empty entry of the benchmark `file` (its images, classes and plan).
    pub fn new(file: BenchFile, scored: bool) -> Filling {
        let variants = folo_bench::variants(&file.plan);
        let n = file.images.len();
        let index = file.images.iter().enumerate().map(|(i, im)| (im.id.clone(), i)).collect();
        Filling {
            passes: variants.iter().map(|_| PassFill::new(n)).collect(),
            clean: PassFill::new(n),
            variants,
            index,
            file,
            scored,
            received: 0,
            held: 0,
            max_held: MAX_HELD_BOXES,
            closed: false,
            last_event: None,
            last_part: Instant::now(),
        }
    }

    /// Whether the contestant gives confidences.
    pub fn scored(&self) -> bool {
        self.scored
    }

    /// The benchmark the entry is of.
    pub fn file(&self) -> &BenchFile {
        &self.file
    }

    /// The most records the entry can have: every image unchanged and under every planned variant.
    pub fn capacity(&self) -> u64 {
        (self.file.images.len() * (1 + self.variants.len())) as u64
    }

    /// Takes a part: every record is checked first (`400 invalid_body` for the first that is wrong, and then none
    /// of the part is taken): an image the benchmark does not have, a condition that is not in the catalog or a
    /// (condition, severity) the plan does not run, a severity without its condition or the other way round, more
    /// than [`MAX_BOXES`] boxes, a class outside the benchmark's, a score outside `0..=1` (when scored), a record
    /// that repeats one of this part or one received before with other boxes. A repeat with the same boxes is
    /// ignored. More than [`MAX_HELD_BOXES`] held at once is refused too. Variants whose last image arrived are
    /// graded.
    pub fn add(&mut self, predictions: Vec<BenchPrediction>) -> Result<()> {
        let classes = self.file.classes.len();
        // The pass (`None`: unchanged) and image of each record to take (`None`: a repeat to ignore).
        let mut targets: Vec<Option<(Option<usize>, usize)>> = Vec::with_capacity(predictions.len());
        let mut here: HashMap<(Option<usize>, usize), (usize, u64)> = HashMap::new();
        let mut new_boxes = 0u64;
        for (i, p) in predictions.iter().enumerate() {
            let what = record(i, p);
            let Some(&image) = self.index.get(p.image.as_str()) else {
                return Err(invalid(format!("{what}: the benchmark has no image '{}'", p.image))
                    .hint("GET /api/bench/benchmarks/{id}/images lists its image ids"));
            };
            let pass = match (&p.condition, p.severity) {
                (None, None) => None,
                (None, Some(_)) => return Err(invalid(format!("{what}: a severity needs its condition"))),
                (Some(c), None) => {
                    return Err(invalid(format!("{what}: condition '{c}' needs its severity")));
                }
                (Some(c), Some(s)) => {
                    let Some(cond) = Condition::parse(c) else {
                        return Err(invalid(format!("{what}: '{c}' is not a condition of the catalog"))
                            .hint("GET /api/bench/catalog lists the conditions"));
                    };
                    let Some(v) = self.variants.iter().position(|&(vc, vs)| vc == cond && vs == s) else {
                        let planned: Vec<String> = self
                            .file
                            .plan
                            .conditions
                            .iter()
                            .map(|pc| format!("{} {:?}", pc.condition, pc.severities))
                            .collect();
                        let runs = if planned.is_empty() {
                            "the unchanged images only".to_string()
                        } else {
                            planned.join(", ")
                        };
                        return Err(invalid(format!(
                            "{what}: the benchmark's plan does not run {c} at severity {s} (it runs {runs})"
                        )));
                    };
                    Some(v)
                }
            };
            let hash = content(p, self.scored);
            if let Some(&(first, h)) = here.get(&(pass, image)) {
                if h != hash {
                    return Err(invalid(format!(
                        "{what} repeats prediction {} with other boxes: answer each image and condition once",
                        first + 1
                    )));
                }
                targets.push(None);
                continue;
            }
            here.insert((pass, image), (i, hash));
            let seen = match pass {
                None => self.clean.seen[image],
                Some(v) => self.passes[v].seen[image],
            };
            match seen {
                Some(h) if h == hash => {
                    // Sent again as it was (a retry): nothing to do.
                    targets.push(None);
                    continue;
                }
                Some(_) => {
                    return Err(invalid(format!(
                        "{what} was received in an earlier part with other boxes: answer each image and \
                         condition once"
                    ))
                    .hint("send only the records not sent yet; this part was not taken"));
                }
                None => {}
            }
            if p.boxes.len() > MAX_BOXES {
                return Err(invalid(format!(
                    "{what} has {} boxes; a record carries at most {MAX_BOXES}",
                    p.boxes.len()
                )));
            }
            for (b, d) in p.boxes.iter().enumerate() {
                if d.class as usize >= classes {
                    return Err(invalid(format!(
                        "{what}, box {}: class {} is not one of the benchmark's {classes} classes (0 to {})",
                        b + 1,
                        d.class,
                        classes.saturating_sub(1)
                    )));
                }
                // A box the scoring would take as a wrong guess without a word: an overflowed coordinate, or
                // corners the wrong way round (most often (x, y, width, height) sent as corners).
                if ![d.x1, d.y1, d.x2, d.y2].iter().all(|v| v.is_finite()) {
                    return Err(invalid(format!(
                        "{what}, box {}: a coordinate is not a finite number",
                        b + 1
                    )));
                }
                if d.x2 < d.x1 || d.y2 < d.y1 {
                    return Err(invalid(format!(
                        "{what}, box {}: x2 {} or y2 {} is less than x1 {} or y1 {}",
                        b + 1,
                        d.x2,
                        d.y2,
                        d.x1,
                        d.y1
                    ))
                    .hint("a box is its corners in pixels: top-left (x1, y1), bottom-right (x2, y2), not x, y, width, height"));
                }
                if self.scored && !(0.0..=1.0).contains(&d.score) {
                    return Err(invalid(format!(
                        "{what}, box {}: score {} is outside 0..=1",
                        b + 1,
                        d.score
                    )));
                }
            }
            new_boxes += p.boxes.len() as u64;
            targets.push(Some((pass, image)));
        }
        if self.held + new_boxes > self.max_held {
            return Err(invalid(format!(
                "this part would make the entry hold {} boxes in memory, more than its {}: send it pass by pass \
                 (the unchanged images, then each condition and severity), so each condition's severity is graded \
                 and let go of as soon as all its images are in",
                self.held + new_boxes,
                self.max_held
            )));
        }
        // Every record is good: take them all (repeats of what is there are left out).
        let mut touched = HashSet::new();
        for (p, target) in predictions.into_iter().zip(targets) {
            let Some((pass, image)) = target else { continue };
            let hash = content(&p, self.scored);
            let fill = match pass {
                None => &mut self.clean,
                Some(v) => {
                    touched.insert(v);
                    &mut self.passes[v]
                }
            };
            fill.seen[image] = Some(hash);
            self.held += p.boxes.len() as u64;
            fill.boxes[image] = p.boxes;
            fill.count += 1;
            self.received += 1;
        }
        for v in touched {
            if self.passes[v].count == self.file.images.len() && self.passes[v].map.is_none() {
                self.grade(v);
            }
        }
        self.last_part = Instant::now();
        Ok(())
    }

    /// Grades variant `v` (every image has arrived) and drops its boxes.
    fn grade(&mut self, v: usize) {
        let (c, s) = self.variants[v];
        let boxes = std::mem::take(&mut self.passes[v].boxes);
        self.held -= boxes.iter().map(|b| b.len() as u64).sum::<u64>();
        let samples: Vec<Sample> = boxes
            .into_iter()
            .zip(&self.file.images)
            .map(|(detections, im)| {
                // The truth under the condition, as the image was served.
                let truths = conditions::moved_boxes(c, s, im.width, im.height, &im.boxes)
                    .expect("planned severities are valid");
                Sample { truths, detections }
            })
            .collect();
        self.passes[v].map = Some(score::map(&samples, self.file.classes.len(), self.scored));
    }

    /// Checks that the entry is whole and hands out what it is graded on: `400 invalid_body` for an image without
    /// its unchanged record, and for a condition answered for some images or severities but not all (the entry
    /// then stays as it is, to be completed).
    pub fn finish(&mut self) -> Result<Finished> {
        if let Some(missing) = self.clean.seen.iter().position(Option::is_none) {
            return Err(invalid(format!(
                "no prediction for image '{}' (unchanged): every image of the benchmark must be answered for, an \
                 image with nothing found with an empty boxes list",
                self.file.images[missing].id
            )));
        }
        // A condition counts when every image is answered at every one of its severities; one that is partly
        // answered is an error, so a robustness number is never made of a part of a condition.
        let mut complete = true;
        let mut runs = Vec::new();
        for pc in &self.file.plan.conditions {
            let Some(cond) = Condition::parse(&pc.condition) else { continue };
            let of: Vec<usize> =
                self.variants.iter().enumerate().filter(|(_, v)| v.0 == cond).map(|(i, _)| i).collect();
            if !of.iter().any(|&v| self.passes[v].count > 0) {
                complete = false;
                continue;
            }
            for &v in &of {
                if let Some(missing) = self.passes[v].seen.iter().position(Option::is_none) {
                    return Err(invalid(format!(
                        "condition '{}' is answered, but not for image '{}' at severity {}: answer a condition \
                         for every image at every planned severity ({:?}), or leave it out",
                        pc.condition, self.file.images[missing].id, self.variants[v].1, pc.severities
                    )));
                }
            }
            for &v in &of {
                let (c, s) = self.variants[v];
                runs.push((c, s, self.passes[v].map.expect("a whole variant is graded")));
            }
        }
        let boxes = std::mem::take(&mut self.clean.boxes);
        self.held = 0;
        let clean = boxes
            .into_iter()
            .zip(&self.file.images)
            .map(|(detections, im)| Sample { truths: im.boxes.clone(), detections })
            .collect();
        self.closed = true;
        Ok(Finished { clean, runs, complete })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------------------------------------------

/// A filling's lock (what is in it stays usable after a panic elsewhere: plain counters and lists).
fn lock(fill: &Mutex<Filling>) -> std::sync::MutexGuard<'_, Filling> {
    fill.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `POST …/entries` for an external contestant: in one request, checked, scored and added `done`; `open`, added
/// `running` with the predictions so far (checked), to be filled by [`add`] and scored by [`finish`].
pub(super) fn start(
    slot: &Slot,
    name: String,
    scored: bool,
    open: bool,
    predictions: Vec<BenchPrediction>,
) -> Result<BenchEntry> {
    let (file, fingerprint) = {
        let state = slot.live()?;
        (state.file.clone(), state.fingerprint.clone())
    };
    let mut fill = Filling::new(file.clone(), scored);
    fill.add(predictions)?;
    // The plan is fixed once an entry is done or active; before that it may change under the check.
    let changed = || {
        Error::conflict("bench_frozen", "the benchmark's plan changed while the entry was checked")
            .hint("send the request again")
    };
    if open {
        let mut state = slot.live()?;
        if state.fingerprint != fingerprint {
            return Err(changed());
        }
        let mut entry = new_entry(
            &state,
            name,
            BenchContestant::External,
            BenchEntryStatus::Running,
            file.images.len(),
            false,
            scored,
            fill.capacity(),
            Vec::new(),
        );
        entry.progress.done = fill.received;
        fill.last_event = Some(Instant::now());
        let cancel = Arc::new(AtomicBool::new(false));
        add_entry(slot, &mut state, entry.clone(), cancel, None, Some(Arc::new(Mutex::new(fill))))?;
        return Ok(entry);
    }
    let finished = fill.finish()?;
    let graded = grade(&file, scored, &finished);
    let mut state = slot.live()?;
    if state.fingerprint != fingerprint {
        return Err(changed());
    }
    let mut entry = new_entry(
        &state,
        name,
        BenchContestant::External,
        BenchEntryStatus::Done,
        file.images.len(),
        false,
        scored,
        0,
        Vec::new(),
    );
    entry.finished = Some(entry.created.clone());
    done(&mut entry, &finished, &graded, fill.received);
    let cancel = Arc::new(AtomicBool::new(false));
    add_entry(slot, &mut state, entry.clone(), cancel, Some((&graded.parts, &graded.images)), None)?;
    Ok(entry)
}

/// The filling of the open external entry `entry_id`, or `409 bench_entry_not_open`.
fn open_entry(slot: &Slot, entry_id: &str) -> Result<Arc<Mutex<Filling>>> {
    let state = slot.live()?;
    let e = state.entry(entry_id)?;
    e.open.clone().ok_or_else(|| not_open(entry_id, &e.entry))
}

/// `POST …/entries/{entry_id}/predictions`: a part of an open external entry, checked whole (a refused part
/// takes none of its records) and taken.
pub(super) fn add(slot: &Slot, entry_id: &str, predictions: Vec<BenchPrediction>) -> Result<BenchEntry> {
    let fill = open_entry(slot, entry_id)?;
    // One part at a time per entry; never under the benchmark's lock (a variant may be graded here). The two
    // locks are only ever taken benchmark first (stop, shutdown, expiry), so the filling's is let go before the
    // benchmark's is asked for.
    let taken = {
        let mut f = lock(&fill);
        if f.closed {
            None
        } else {
            f.add(predictions)?;
            let emit = f.last_event.is_none_or(|t| t.elapsed() >= model::PROGRESS_EVERY);
            if emit {
                f.last_event = Some(Instant::now());
            }
            Some((f.received, emit))
        }
    };
    let Some((received, emit)) = taken else {
        let state = slot.live()?;
        return Err(not_open(entry_id, &state.entry(entry_id)?.entry));
    };
    let mut state = slot.live()?;
    let e = state.entry_mut(entry_id).ok_or_else(|| super::state::entry_not_found(&slot.id, entry_id))?;
    e.entry.progress.done = e.entry.progress.done.max(received);
    let entry = e.entry.clone();
    if emit {
        slot.emit(BenchEvent::Entry(Box::new(entry.clone())));
    }
    Ok(entry)
}

/// `POST …/entries/{entry_id}/finish`: checks that the open entry is whole and scores it (`done`); a `400`
/// saying what is missing leaves it open.
pub(super) fn finish(slot: &Slot, entry_id: &str) -> Result<BenchEntry> {
    let fill = open_entry(slot, entry_id)?;
    // As in `add`: the filling's lock is let go before the benchmark's is asked for.
    let taken = {
        let mut f = lock(&fill);
        if f.closed { None } else { Some((f.finish()?, f.scored(), f.received, f.file().clone())) }
    };
    let Some((finished, scored, received, file)) = taken else {
        let state = slot.live()?;
        return Err(not_open(entry_id, &state.entry(entry_id)?.entry));
    };
    let graded = grade(&file, scored, &finished);
    let mut state = slot.live()?;
    let e = state.entry(entry_id)?;
    if e.open.as_ref().is_none_or(|o| !Arc::ptr_eq(o, &fill)) {
        // Stopped, expired or the server shut down while it was scored.
        return Err(not_open(entry_id, &e.entry));
    }
    let mut entry = e.entry.clone();
    entry.status = BenchEntryStatus::Done;
    entry.finished = Some(rfc3339(unix_now()));
    done(&mut entry, &finished, &graded, received);
    // Written before it changes in memory: a result that cannot be written leaves the entry failed.
    let entry = write_closed(slot, &state, entry, Some((&graded.parts, &graded.images)));
    let e = state.entry_mut(entry_id).expect("looked up above, under the same lock");
    e.open = None;
    e.entry = entry.clone();
    slot.emit(BenchEvent::Entry(Box::new(entry.clone())));
    slot.emit_bench(&state);
    Ok(entry)
}

/// How long `d` is, in words: "30 minutes", "2 seconds".
fn span(d: Duration) -> String {
    match d.as_secs() {
        s if s >= 120 => format!("{} minutes", s / 60),
        s if s >= 2 => format!("{s} seconds"),
        _ => format!("{} ms", d.as_millis()),
    }
}

/// Fails every open entry of `slot` that got no part for `timeout` ("no predictions arrived for …"): what it held
/// is dropped, and the benchmark (its `active`) is sent when that changed.
pub(super) fn expire(slot: &Slot, timeout: Duration) {
    let mut state = slot.lock();
    if state.deleted {
        return;
    }
    // Checked and closed in one hold of the filling's lock: a part that got in first has moved `last_part` (the
    // entry lives on), one that comes after finds the filling closed. One that is being finished is closed
    // already and is left to its finish.
    let stale: Vec<String> = state
        .entries
        .iter_mut()
        .filter_map(|e| {
            {
                let mut f = lock(e.open.as_ref()?);
                if f.closed || f.last_part.elapsed() < timeout {
                    return None;
                }
                f.closed = true;
            }
            e.open = None;
            Some(e.entry.id.clone())
        })
        .collect();
    for id in &stale {
        let e = state.entry_mut(id).expect("listed above");
        let mut entry = e.entry.clone();
        entry.status = BenchEntryStatus::Failed;
        entry.finished = Some(rfc3339(unix_now()));
        entry.error = Some(format!("no predictions arrived for {}", span(timeout)));
        entry.hint = Some(
            "start it again (POST /api/bench/benchmarks/{id}/entries with contestant.open: true) and send its \
             parts without long pauses"
                .into(),
        );
        let entry = write_closed(slot, &state, entry, None);
        if let Some(e) = state.entry_mut(id) {
            e.entry = entry.clone();
        }
        slot.emit(BenchEvent::Entry(Box::new(entry)));
    }
    if !stale.is_empty() {
        slot.emit_bench(&state);
    }
}

/// `409 bench_entry_not_open` for `entry`, saying what it is instead.
pub(super) fn not_open(id: &str, entry: &BenchEntry) -> Error {
    let what = match (&entry.contestant, entry.status) {
        (BenchContestant::External, BenchEntryStatus::Running) => "being finished".to_string(),
        (BenchContestant::External, s) => format!("an external entry that is {}", status_name(s)),
        (BenchContestant::Model { .. }, _) => "a model entry".to_string(),
        (BenchContestant::Agent { .. }, _) => "an agent entry".to_string(),
    };
    Error::conflict(
        "bench_entry_not_open",
        format!("entry '{id}' is not an open external entry: it is {what}"),
    )
    .hint("POST /api/bench/benchmarks/{id}/entries with contestant.open: true starts one to fill in parts")
}

/// The grade of a finished external entry.
fn grade(file: &BenchFile, scored: bool, finished: &Finished) -> results::Graded {
    let ids: Vec<String> = file.images.iter().map(|i| i.id.clone()).collect();
    let spec = results::Spec {
        classes: file.classes.len(),
        scored,
        plan: &file.plan,
        seed: file.seed,
        complete: finished.complete,
    };
    results::grade_maps(&finished.clean, &ids, &finished.runs, &spec)
}

/// Fills in a finished external entry: its score and what it ran (`received` records).
fn done(entry: &mut BenchEntry, finished: &Finished, graded: &results::Graded, received: u64) {
    entry.conditions = !finished.runs.is_empty();
    entry.progress = model::progress(received);
    entry.progress.done = received;
    entry.ci95 = Some(graded.parts.accuracy.ci95);
    entry.conditions_run = graded.conditions_run();
    entry.score = Some(graded.score.clone());
    entry.map50_95 = Some(graded.map50_95);
}

#[cfg(test)]
mod tests {
    use folo_api::{BBox, BenchPlan, BenchPlanCondition, BenchSource};

    use super::super::state::{ImageRecord, VERSION};
    use super::*;

    fn file() -> BenchFile {
        let image = |id: &str| ImageRecord {
            id: id.into(),
            file: format!("{id}.jpg"),
            width: 100,
            height: 100,
            bytes: 1,
            sha256: id.into(),
            boxes: vec![BBox { class: 0, x1: 10.0, y1: 10.0, x2: 50.0, y2: 50.0 }],
        };
        BenchFile {
            version: VERSION,
            id: "bn_1".into(),
            name: "b".into(),
            description: String::new(),
            classes: vec!["a".into(), "b".into()],
            created: String::new(),
            source: BenchSource::LabelJob { job_id: "lb_1".into() },
            source_name: String::new(),
            seed: 0,
            plan: BenchPlan {
                conditions: vec![
                    BenchPlanCondition {
                        condition: "haze".into(),
                        severities: vec![1, 3],
                        weight: 1.0,
                        why: None,
                    },
                    BenchPlanCondition {
                        condition: "distance".into(),
                        severities: vec![2],
                        weight: 1.0,
                        why: None,
                    },
                ],
                rationale: None,
                designed_by: None,
            },
            images: vec![image("1"), image("2")],
        }
    }

    fn p(image: &str, cond: Option<(&str, u8)>) -> BenchPrediction {
        BenchPrediction {
            image: image.into(),
            condition: cond.map(|c| c.0.into()),
            severity: cond.map(|c| c.1),
            boxes: vec![BenchDetection { class: 0, score: 0.5, x1: 10.0, y1: 10.0, x2: 50.0, y2: 50.0 }],
        }
    }

    /// Every record of [`file`]'s benchmark.
    fn all() -> Vec<BenchPrediction> {
        let mut all = vec![p("2", None), p("1", None)];
        for im in ["1", "2"] {
            all.extend([p(im, Some(("haze", 1))), p(im, Some(("haze", 3))), p(im, Some(("distance", 2)))]);
        }
        all
    }

    fn check(f: &BenchFile, scored: bool, preds: &[BenchPrediction]) -> Result<Finished> {
        let mut fill = Filling::new(f.clone(), scored);
        fill.add(preds.to_vec())?;
        fill.finish()
    }

    #[test]
    fn complete_partial_and_moved_truth() {
        let f = file();
        let c = check(&f, true, &all()).unwrap();
        assert!(c.complete);
        assert_eq!(
            c.runs.iter().map(|r| (r.0.id(), r.1)).collect::<Vec<_>>(),
            [("haze", 1), ("haze", 3), ("distance", 2)]
        );
        assert_eq!(c.clean.len(), 2);
        // Haze keeps the truth (a perfect answer), distance moves it (the same boxes are wrong there).
        assert_eq!(c.runs[0].2.map50, 1.0);
        assert_eq!(c.runs[2].2.map50, 0.0);
        let only_clean = check(&f, true, &all()[..2]).unwrap();
        assert!(!only_clean.complete && only_clean.runs.is_empty());
    }

    #[test]
    fn parts_in_any_order_come_to_what_one_request_does() {
        let f = file();
        let whole = check(&f, true, &all()).unwrap();
        let mut fill = Filling::new(f.clone(), true);
        let mut records = all();
        records.reverse();
        // Variants first (each graded as its last image arrives), the unchanged images last.
        for part in records.chunks(2) {
            fill.add(part.to_vec()).unwrap();
        }
        assert!(fill.passes.iter().all(|v| v.map.is_some() && v.boxes.is_empty()), "graded and dropped");
        assert_eq!(fill.received, 8);
        let parts = fill.finish().unwrap();
        assert_eq!(parts.clean, whole.clean);
        assert_eq!(parts.runs, whole.runs);
        assert!(fill.closed);
    }

    #[test]
    fn a_part_is_all_or_nothing_and_repeats_are_refused_across_parts() {
        let f = file();
        let mut fill = Filling::new(f.clone(), true);
        fill.add(vec![p("1", None)]).unwrap();
        // A part with a bad record takes none of its records.
        let e = fill.add(vec![p("2", None), p("9", None)]).unwrap_err();
        assert!(e.body.message.contains("no image '9'"), "{}", e.body.message);
        assert_eq!(fill.received, 1);
        // The same record again (a retry) is taken as it is; with other boxes it is refused.
        fill.add(vec![p("1", None)]).unwrap();
        assert_eq!(fill.received, 1);
        let mut other = p("1", None);
        other.boxes[0].x2 = 51.0;
        let e = fill.add(vec![other.clone()]).unwrap_err();
        assert!(
            e.body.message.contains("was received in an earlier part with other boxes"),
            "{}",
            e.body.message
        );
        // A score is part of the content only with confidences.
        let mut rescored = p("1", None);
        rescored.boxes[0].score = 0.7;
        assert!(fill.add(vec![rescored.clone()]).is_err());
        let mut unscored = Filling::new(f.clone(), false);
        unscored.add(vec![p("1", None)]).unwrap();
        unscored.add(vec![rescored]).unwrap();
        assert_eq!(unscored.received, 1);
        // Unfinished: it says what is missing and stays open.
        let e = fill.finish().err().unwrap();
        assert!(e.body.message.contains("no prediction for image '2'"), "{}", e.body.message);
        assert!(!fill.closed);
        fill.add(vec![p("2", None)]).unwrap();
        assert!(!fill.finish().unwrap().complete);
    }

    #[test]
    fn identical_repeats_in_one_part_count_once_and_what_is_held_is_capped() {
        let f = file();
        let mut fill = Filling::new(f.clone(), true);
        fill.add(vec![p("1", None), p("1", None)]).unwrap();
        assert_eq!((fill.received, fill.held), (1, 1));
        fill.max_held = 4;
        // Haze 1 for both images (2 boxes) fits and is graded at once, which lets its boxes go.
        fill.add(vec![p("1", Some(("haze", 1))), p("2", Some(("haze", 1)))]).unwrap();
        assert_eq!(fill.held, 1);
        let many = BenchPrediction { boxes: vec![p("2", None).boxes[0]; 4], ..p("2", None) };
        let e = fill.add(vec![many]).unwrap_err();
        assert!(
            e.body.message.contains("would make the entry hold 5 boxes in memory, more than its 4"),
            "{}",
            e.body.message
        );
        assert!(e.body.message.contains("pass by pass"));
        assert_eq!(fill.received, 3);
    }

    #[test]
    fn every_wrong_record_is_named() {
        let f = file();
        let base = vec![p("1", None), p("2", None)];
        let cases: Vec<(Vec<BenchPrediction>, &str)> = vec![
            (
                vec![p("1", None), p("9", None)],
                "prediction 2 (image '9', unchanged): the benchmark has no image '9'",
            ),
            (vec![p("1", Some(("fog", 1)))], "'fog' is not a condition of the catalog"),
            (vec![p("1", Some(("jpeg", 1)))], "the benchmark's plan does not run jpeg at severity 1"),
            (vec![p("1", Some(("haze", 2)))], "does not run haze at severity 2"),
            (
                vec![p("1", None), BenchPrediction { boxes: vec![], ..p("1", None) }],
                "prediction 2 (image '1', unchanged) repeats prediction 1 with other boxes",
            ),
            (vec![BenchPrediction { severity: Some(1), ..p("1", None) }], "a severity needs its condition"),
            (
                vec![BenchPrediction { severity: None, ..p("1", Some(("haze", 1))) }],
                "condition 'haze' needs its severity",
            ),
            (vec![p("1", None)], "no prediction for image '2' (unchanged)"),
            (
                [
                    base.clone(),
                    vec![p("1", Some(("haze", 1))), p("2", Some(("haze", 1))), p("1", Some(("haze", 3)))],
                ]
                .concat(),
                "condition 'haze' is answered, but not for image '2' at severity 3",
            ),
        ];
        for (preds, want) in cases {
            let e = check(&f, true, &preds).err().expect(want);
            assert_eq!(e.body.code, "invalid_body");
            assert!(e.body.message.contains(want), "{} / {want}", e.body.message);
        }
        let mut bad_class = base.clone();
        bad_class[1].boxes[0].class = 2;
        let e = check(&f, true, &bad_class).err().unwrap();
        assert!(
            e.body.message.contains("prediction 2 (image '2', unchanged), box 1: class 2"),
            "{}",
            e.body.message
        );
        let mut bad_score = base.clone();
        bad_score[0].boxes[0].score = 1.5;
        assert!(check(&f, true, &bad_score).err().unwrap().body.message.contains("score 1.5"));
        // Without confidences the scores are not read.
        assert!(check(&f, false, &bad_score).is_ok());
        // Corners the wrong way round (an (x, y, w, h) box sent as corners) and overflowed coordinates are named,
        // not scored as wrong guesses; a box of no area is a box.
        let mut swapped = base.clone();
        swapped[1].boxes[0].x2 = 5.0;
        let e = check(&f, true, &swapped).err().unwrap();
        assert!(
            e.body
                .message
                .contains("prediction 2 (image '2', unchanged), box 1: x2 5 or y2 50 is less than x1 10"),
            "{}",
            e.body.message
        );
        assert!(e.body.hint.as_deref().unwrap_or("").contains("not x, y, width, height"));
        let mut upside = base.clone();
        upside[0].boxes[0].y1 = 60.0;
        assert!(check(&f, true, &upside).is_err());
        let mut overflow = base.clone();
        overflow[0].boxes[0].x2 = f32::INFINITY;
        assert!(check(&f, true, &overflow).err().unwrap().body.message.contains("not a finite number"));
        let mut flat = base.clone();
        flat[0].boxes[0].x2 = flat[0].boxes[0].x1;
        assert!(check(&f, true, &flat).is_ok());
        let mut many = base.clone();
        many[0].boxes = vec![many[0].boxes[0]; MAX_BOXES + 1];
        let e = check(&f, true, &many).err().unwrap();
        assert!(
            e.body.message.contains("has 1001 boxes; a record carries at most 1000"),
            "{}",
            e.body.message
        );
        many[0].boxes.truncate(MAX_BOXES);
        assert!(check(&f, true, &many).is_ok());
    }
}
