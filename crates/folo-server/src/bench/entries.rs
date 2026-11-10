//! Entries: starting them (model, agent and external), stopping, results, deleting, and their shared helpers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use folo_api::{
    BenchContestant, BenchContestantSpec, BenchEntry, BenchEntryImage, BenchEntryImagePage, BenchEntryStatus,
    BenchEvent, BenchRemoved, BenchResult, LabelUsage, StartBenchEntry,
};

use crate::error::{Error, Result};
use crate::label::{check_model, check_name};
use crate::store::Store;
use crate::util::{fnv1a, parse_rfc3339_millis, rfc3339, rfc3339_millis, unique_id, unix_now};

use super::benchmarks::bench_image;
use super::model::{Engine, Job, Resolved};
use super::state::{EntryState, Slot, State};
use super::{AGENT_IMAGES, Benches, agent, external, invalid, model, state};

/// Starts a model entry of `resolved` on `slot`: `queued` until the worker takes it. `name` defaults to the
/// model's, `max_images` to every image.
pub(super) fn enqueue_model(
    engine: &Engine,
    slot: &Arc<Slot>,
    resolved: Resolved,
    name: Option<String>,
    max_images: Option<u32>,
    conditions: bool,
) -> Result<BenchEntry> {
    let mut state = slot.live()?;
    let images = entry_images(max_images, state.file.images.len(), state.file.images.len())?;
    let passes = model::passes(&state.file.plan, conditions).len();
    let entry = new_entry(
        &state,
        name.unwrap_or(resolved.name),
        resolved.contestant,
        BenchEntryStatus::Queued,
        images,
        conditions,
        true,
        model::total(images, passes),
        resolved.missing,
    );
    let cancel = Arc::new(AtomicBool::new(false));
    add_entry(slot, &mut state, entry.clone(), cancel.clone(), None, None)?;
    drop(state);
    engine.push(Job {
        slot: slot.clone(),
        entry: entry.id.clone(),
        model: resolved.spec,
        images,
        conditions,
        cancel,
    });
    Ok(entry)
}

/// Stops entry `id` of `slot` if it is active: one that waits is taken out of the queue and is `stopped` at once,
/// and so is an open external entry (what it received is dropped); one that runs is `stopping` until its work
/// ends. Returns the entry, or `409 bench_entry_not_active`. The entry's file is written before it changes in
/// memory.
pub(super) fn stop_entry_in(engine: &Engine, slot: &Slot, id: &str) -> Result<BenchEntry> {
    let mut state = slot.live()?;
    let e = state.entry(id)?;
    let mut entry = e.entry.clone();
    let ends = if e.open.is_some() {
        entry.status = BenchEntryStatus::Stopped;
        true
    } else {
        match e.entry.status {
            BenchEntryStatus::Queued if engine.unqueue(&slot.id, id) => {
                entry.status = BenchEntryStatus::Stopped;
                true
            }
            BenchEntryStatus::Queued | BenchEntryStatus::Running => {
                entry.status = BenchEntryStatus::Stopping;
                false
            }
            BenchEntryStatus::Stopping => return Ok(entry),
            s => {
                return Err(Error::conflict(
                    "bench_entry_not_active",
                    format!("entry '{id}' has already ended ({})", status_name(s)),
                )
                .hint("nothing to stop; GET /api/bench/benchmarks/{id}/entries/{entry_id} has its result"));
            }
        }
    };
    if ends {
        entry.finished = Some(rfc3339(unix_now()));
    }
    slot.save_entry(&state, &entry, None)?;
    let e = state.entry_mut(id).expect("looked up above, under the same lock");
    e.cancel.store(true, Ordering::SeqCst);
    if let Some(fill) = e.open.take() {
        fill.lock().unwrap_or_else(std::sync::PoisonError::into_inner).closed = true;
    }
    e.entry = entry.clone();
    slot.emit(BenchEvent::Entry(Box::new(entry.clone())));
    if ends {
        slot.emit_bench(&state);
    }
    Ok(entry)
}

pub(super) fn status_name(s: BenchEntryStatus) -> &'static str {
    match s {
        BenchEntryStatus::Queued => "queued",
        BenchEntryStatus::Running => "running",
        BenchEntryStatus::Stopping => "stopping",
        BenchEntryStatus::Done => "done",
        BenchEntryStatus::Failed => "failed",
        BenchEntryStatus::Stopped => "stopped",
    }
}

/// How many images an entry runs: `max_images` (at least 1, at most the benchmark's), else `default` (capped).
fn entry_images(max_images: Option<u32>, images: usize, default: usize) -> Result<usize> {
    match max_images {
        Some(0) => Err(invalid("max_images must be at least 1")),
        Some(n) => Ok((n as usize).min(images)),
        None => Ok(default.min(images)),
    }
}

/// A new entry of the benchmark in `state` with a fresh id.
#[allow(clippy::too_many_arguments)]
pub(super) fn new_entry(
    state: &State,
    name: String,
    contestant: BenchContestant,
    status: BenchEntryStatus,
    images: usize,
    conditions: bool,
    scored: bool,
    total: u64,
    missing: Vec<String>,
) -> BenchEntry {
    let id = unique_id("be", fnv1a(name.as_bytes()) ^ unix_now().to_bits(), |id| {
        state.entries.iter().any(|e| e.entry.id == id)
    });
    // To the millisecond and later than every entry the benchmark has: `created` (then the id) is the order of
    // the leaderboard's ties, and it has to be the order the entries were made in, also after a restart.
    let last = state.entries.iter().filter_map(|e| parse_rfc3339_millis(&e.entry.created)).max();
    let now = (unix_now() * 1000.0) as i64;
    let created = rfc3339_millis(last.map_or(now, |l| now.max(l + 1)));
    BenchEntry {
        id,
        bench_id: state.file.id.clone(),
        name,
        contestant,
        status,
        created,
        finished: None,
        progress: model::progress(total),
        images: images as u64,
        conditions,
        scored,
        fingerprint: state.fingerprint.clone(),
        missing_classes: missing,
        score: None,
        map50_95: None,
        ci95: None,
        conditions_run: 0,
        speed: None,
        usage: None,
        error: None,
        hint: None,
    }
}

/// Adds `entry` to the benchmark (written, sent, the benchmark's count sent too). `parts` for an entry that is
/// done already (an external one); `open` for an open external entry.
pub(super) fn add_entry(
    slot: &Slot,
    state: &mut State,
    entry: BenchEntry,
    cancel: Arc<AtomicBool>,
    parts: Option<(&state::ResultParts, &[state::ImageOutcome])>,
    open: Option<Arc<Mutex<external::Filling>>>,
) -> Result<()> {
    slot.save_entry(state, &entry, parts)?;
    state.entries.push(EntryState { entry: entry.clone(), cancel, open });
    slot.emit(BenchEvent::Entry(Box::new(entry)));
    slot.emit_bench(state);
    Ok(())
}

impl Benches {
    /// `GET /api/bench/benchmarks/{id}/entries`: the leaderboard.
    pub fn entries(&self, id: &str) -> Result<Vec<BenchEntry>> {
        Ok(self.slot(id)?.live()?.leaderboard())
    }

    /// `GET /api/bench/benchmarks/{id}/entries/{entry_id}`.
    pub fn entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry> {
        Ok(self.slot(id)?.live()?.entry(entry_id)?.entry.clone())
    }

    /// `POST /api/bench/benchmarks/{id}/entries`.
    pub fn start_entry(&self, store: &dyn Store, id: &str, req: StartBenchEntry) -> Result<BenchEntry> {
        let slot = self.slot(id)?;
        let name =
            req.name.as_deref().map(str::trim).filter(|n| !n.is_empty()).map(check_name).transpose()?;
        match req.contestant {
            BenchContestantSpec::External { scored, open, predictions } => {
                if req.max_images.is_some() || req.conditions.is_some() {
                    return Err(invalid(
                        "max_images and conditions are not for an external contestant: its predictions say what \
                         it ran",
                    ));
                }
                let name = name.ok_or_else(|| invalid("name is required for an external contestant"))?;
                external::start(&slot, name, scored, open, predictions)
            }
            BenchContestantSpec::Model { model_id } => {
                let classes = slot.live()?.file.classes.clone();
                let m = store.get_model(&model_id)?;
                let resolved = model::resolve(&m, &classes)?;
                enqueue_model(
                    &self.engine,
                    &slot,
                    resolved,
                    name,
                    req.max_images,
                    req.conditions.unwrap_or(true),
                )
            }
            BenchContestantSpec::Agent { agent: agent_id, model, effort } => {
                let model = check_model(model.as_deref())?;
                let agent = store.labeler()?.agent(agent_id)?;
                let info = agent.info();
                let model = model.or(info.default_model);
                let effort = effort.unwrap_or(info.default_effort);
                let conditions = req.conditions.unwrap_or(false);
                let name = name.unwrap_or_else(|| match &model {
                    Some(m) => format!("{} ({m})", agent_id.display_name()),
                    None => agent_id.display_name().to_string(),
                });
                let mut state = slot.live()?;
                let images = entry_images(req.max_images, state.file.images.len(), AGENT_IMAGES as usize)?;
                let passes = model::passes(&state.file.plan, conditions).len();
                let mut entry = new_entry(
                    &state,
                    name,
                    BenchContestant::Agent { agent: agent_id, model: model.clone(), effort },
                    BenchEntryStatus::Running,
                    images,
                    conditions,
                    false,
                    model::total(images, passes),
                    Vec::new(),
                );
                entry.usage = Some(LabelUsage::default());
                let cancel = Arc::new(AtomicBool::new(false));
                add_entry(&slot, &mut state, entry.clone(), cancel.clone(), None, None)?;
                drop(state);
                let job = agent::AgentJob {
                    slot: slot.clone(),
                    entry: entry.id.clone(),
                    agent,
                    model,
                    effort,
                    images,
                    conditions,
                    cancel,
                };
                let engine = self.engine.clone();
                let thread = std::thread::Builder::new()
                    .name(format!("folo-bench-agent-{}", entry.id))
                    .spawn(move || agent::run(job, &engine.closing));
                match thread {
                    Ok(t) => self.keep(t),
                    Err(e) => {
                        let error = Error::internal(format!("could not start the entry's thread: {e}"));
                        model::finish_entry(&slot, &entry.id, false, Err(error.clone()));
                        return Err(error);
                    }
                }
                Ok(entry)
            }
        }
    }

    /// `POST …/entries/{entry_id}/predictions` ([`external::add`]).
    pub fn add_predictions(
        &self,
        id: &str,
        entry_id: &str,
        predictions: Vec<folo_api::BenchPrediction>,
    ) -> Result<BenchEntry> {
        external::add(&*self.slot(id)?, entry_id, predictions)
    }

    /// `POST …/entries/{entry_id}/finish` ([`external::finish`]).
    pub fn finish_open(&self, id: &str, entry_id: &str) -> Result<BenchEntry> {
        external::finish(&*self.slot(id)?, entry_id)
    }

    /// `GET …/entries/{entry_id}/result`.
    pub fn result(&self, id: &str, entry_id: &str) -> Result<BenchResult> {
        let slot = self.slot(id)?;
        let entry = done_entry(&slot, entry_id)?;
        let file = slot.read_entry(entry_id)?;
        let r = file
            .result
            .ok_or_else(|| Error::internal(format!("entry '{entry_id}' has no result on disk")))?;
        Ok(BenchResult {
            entry,
            accuracy: r.accuracy,
            robustness: r.robustness,
            calibration: r.calibration,
            localization: r.localization,
            self_labeling: r.self_labeling,
        })
    }

    /// `GET …/entries/{entry_id}/images`: per image the truth, the detections and what was missed or wrong;
    /// `worst` puts the images with the most `missed + wrong` first.
    pub fn entry_images(
        &self,
        id: &str,
        entry_id: &str,
        worst: bool,
        offset: u64,
        limit: u64,
    ) -> Result<BenchEntryImagePage> {
        let slot = self.slot(id)?;
        done_entry(&slot, entry_id)?;
        let file = slot.read_entry(entry_id)?;
        let mut outcomes = file.images;
        if worst {
            outcomes.sort_by_key(|o| std::cmp::Reverse(o.missed + o.wrong));
        }
        let state = slot.live()?;
        let images = outcomes
            .into_iter()
            .skip(offset.min(usize::MAX as u64) as usize)
            .take(limit.min(usize::MAX as u64) as usize)
            .filter_map(|o| {
                let im = state.file.image(&o.image)?;
                Some(BenchEntryImage {
                    image: bench_image(im, None).ok()?,
                    detections: o.detections,
                    missed: o.missed,
                    wrong: o.wrong,
                })
            })
            .collect();
        Ok(BenchEntryImagePage { total: file.entry.images, images })
    }

    /// `POST …/entries/{entry_id}/stop`.
    pub fn stop_entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry> {
        stop_entry_in(&self.engine, &*self.slot(id)?, entry_id)
    }

    /// `DELETE …/entries/{entry_id}`.
    pub fn delete_entry(&self, id: &str, entry_id: &str) -> Result<()> {
        let slot = self.slot(id)?;
        let mut state = slot.live()?;
        let e = state.entry(entry_id)?;
        if e.entry.status.is_active() {
            return Err(Error::conflict(
                "bench_entry_active",
                format!("entry '{entry_id}' is still {}", status_name(e.entry.status)),
            )
            .hint("stop it first: POST /api/bench/benchmarks/{id}/entries/{entry_id}/stop"));
        }
        if let Some(l) = state.loops.iter().find(|l| {
            l.lp.status.is_active()
                && l.lp.generations.iter().any(|g| g.entry_id.as_deref() == Some(entry_id))
        }) {
            return Err(Error::conflict(
                "bench_entry_in_loop",
                format!("loop '{}' is running and uses entry '{entry_id}'", l.lp.id),
            )
            .hint("delete it once the loop has ended, or stop the loop"));
        }
        slot.remove_entry(entry_id)?;
        state.entries.retain(|e| e.entry.id != entry_id);
        slot.emit(BenchEvent::Removed(BenchRemoved { id: entry_id.to_string() }));
        slot.emit_bench(&state);
        Ok(())
    }
}

/// A finished entry, or `409 bench_entry_not_done`.
fn done_entry(slot: &Slot, entry_id: &str) -> Result<BenchEntry> {
    let state = slot.live()?;
    let e = state.entry(entry_id)?;
    if e.entry.status != BenchEntryStatus::Done {
        return Err(Error::conflict(
            "bench_entry_not_done",
            format!("entry '{entry_id}' is {}: it has a result once it is done", status_name(e.entry.status)),
        ));
    }
    Ok(e.entry.clone())
}
