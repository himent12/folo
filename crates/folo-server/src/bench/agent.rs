//! Agent entries: a coding agent (Claude Code, Codex, Antigravity) asked for the boxes and calibrated confidences
//! of a benchmark's images ([`crate::label::prompt::BENCH_SYSTEM`]: the benchmark's classes and its description as
//! the instructions, the same reply checks, batching, parallelism and failure policy as a labeling run). When the
//! agent reports valid confidences on its detections, the entry is graded as scored (populating the calibration axis
//! and confidence-thresholded self-labeling curve); if an agent omits confidences on any detection (legacy or mixed
//! replies), the entry falls back cleanly to unscored grading so fallback constant scores never fabricate
//! calibration. Agents are IO-bound, so every agent entry has a thread of its own and never waits for the model
//! queue.
//!
//! **Failures.** A call that cannot be made (the CLI missing or signed out, an unknown model, a rate limit) ends
//! the entry at once, as it ends a labeling run, and so do three calls in a row that time out or return garbage.
//! A call the provider failed on a passing fault (overloaded, a server error) is asked once more after a pause,
//! as in a labeling run, and fails its images when that fails too. An image whose call failed otherwise (its reply could not be read even when asked once more, or left the image
//! out) has no answer. Scoring such an image as one where the agent found nothing would understate the agent, so
//! the entry is scored only when at most [`allowed_failures`] of its images failed (2%, at least one); with more,
//! the entry fails and says how many.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_api::{BenchDetection, BenchEvent, LabelEffort, LabelUsage};
use folo_bench::score::Sample;
use folo_bench::{Condition, conditions};

use super::model::{PROGRESS_EVERY, finish_entry, passes};
use super::results::{self, Pass};
use super::state::{BenchFile, Slot};
use crate::error::{Error, Result};
use crate::label::agent::{Agent, AgentRequest, BatchError, CALL_TIMEOUT, Event, ask, failure_hint};
use crate::label::{AgentImage, MAX_SOFT_FAILURES, agent_jpeg, agent_jpeg_of, prompt};

/// Agent calls in flight at once (as a labeling run's default).
pub const PARALLEL: usize = crate::label::DEFAULT_PARALLEL as usize;

/// Images of an entry of `total` image answers that may fail before the entry is not scored: 2%, at least one.
pub fn allowed_failures(total: usize) -> usize {
    (total / 50).max(1)
}

/// An agent entry to run.
pub struct AgentJob {
    pub slot: Arc<Slot>,
    pub entry: String,
    pub agent: Arc<dyn Agent>,
    pub model: Option<String>,
    pub effort: LabelEffort,
    pub images: usize,
    pub conditions: bool,
    pub cancel: Arc<AtomicBool>,
    /// The wait before a call that failed on a passing fault is asked again, as a labeling run's
    /// ([`crate::label::Labeler::transient_backoff`]).
    pub backoff: Duration,
}

/// One call's worth of images: of pass `pass`, the images at these positions.
struct Unit {
    n: usize,
    pass: usize,
    images: Vec<usize>,
}

struct Run<'a> {
    job: &'a AgentJob,
    file: &'a BenchFile,
    passes: Vec<Option<(Condition, u8)>>,
    schema: serde_json::Value,
    queue: Mutex<VecDeque<Unit>>,
    /// Per pass, per image: the parsed detections (with optional reported confidences), once answered.
    answers: Mutex<Vec<Vec<Option<Vec<prompt::ScoredBox>>>>>,
    /// Images whose call failed, with the first reason.
    failed: Mutex<(usize, Option<String>)>,
    soft_failures: AtomicU32,
    /// Why the entry gave up, and what to do about it when that is known then.
    fatal: Mutex<Option<(String, Option<String>)>>,
    done: Mutex<(u64, Option<Instant>)>,
}

/// Runs an agent entry to its end (the body of its thread) and closes it. `closing` says the server shuts down.
pub fn run(job: AgentJob, closing: &AtomicBool) {
    let file = job.slot.lock().file.clone();
    let passes = passes(&file.plan, job.conditions);
    let batch = job.agent.default_batch().max(1);
    let mut units = VecDeque::new();
    for pass in 0..passes.len() {
        for chunk in (0..job.images).collect::<Vec<_>>().chunks(batch) {
            units.push_back(Unit { n: units.len() + 1, pass, images: chunk.to_vec() });
        }
    }
    let workers = PARALLEL.clamp(1, units.len().max(1));
    let run = Run {
        job: &job,
        file: &file,
        schema: prompt::bench_reply_schema(&file.classes),
        answers: Mutex::new(vec![vec![None; job.images]; passes.len()]),
        passes,
        queue: Mutex::new(units),
        failed: Mutex::new((0, None)),
        soft_failures: AtomicU32::new(0),
        fatal: Mutex::new(None),
        done: Mutex::new((0, None)),
    };
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| run.work());
            }
        });
        run.outcome()
    }));
    let _ = std::fs::remove_dir_all(job.slot.dir.join("tmp").join(&job.entry));
    let outcome = match ran {
        Ok(outcome) => outcome,
        Err(_) => Err(Error::internal("the entry stopped on an internal error (see the server log)")),
    };
    finish_entry(
        &job.slot,
        &job.entry,
        closing.load(Ordering::SeqCst),
        outcome.map(|o| o.map(|g| (g, None))),
    );
}

impl Run<'_> {
    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::Relaxed)
    }

    fn work(&self) {
        while !self.cancelled() {
            let Some(unit) = self.queue.lock().unwrap().pop_front() else { break };
            self.call(unit);
        }
    }

    /// Gives up, with `hint` as the caller resolved it ([`failure_hint`], [`Agent::hint`]); `None` stays none.
    fn fail(&self, why: String, hint: Option<String>) {
        self.fatal.lock().unwrap().get_or_insert((why, hint));
        self.job.cancel.store(true, Ordering::Relaxed);
    }

    fn soft_failure(&self, why: &str, hint: Option<String>) {
        if self.soft_failures.fetch_add(1, Ordering::Relaxed) + 1 >= MAX_SOFT_FAILURES {
            self.fail(format!("{MAX_SOFT_FAILURES} calls in a row failed; the last one: {why}"), hint);
        }
    }

    /// The image at `i` as the agent gets it under the unit's pass.
    fn jpeg(&self, pass: usize, i: usize) -> std::result::Result<AgentImage, String> {
        let im = &self.file.images[i];
        let path = self.job.slot.image_path(im);
        let side = self.job.agent.image_side();
        match self.passes[pass] {
            None => agent_jpeg(&path, side),
            Some((c, s)) => {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                let seed = conditions::variant_seed(self.file.seed, &im.id, c, s);
                let r = conditions::render(&bytes, c, s, &im.boxes, seed).map_err(|e| e.to_string())?;
                agent_jpeg_of(folo_data::decode(&r.jpeg).map_err(|e| e.to_string())?, side)
            }
        }
    }

    /// One call: the unit's images through the agent (a second call when the first reply cannot be read).
    fn call(&self, unit: Unit) {
        let mut jpegs = Vec::with_capacity(unit.images.len());
        let mut sizes = Vec::with_capacity(unit.images.len());
        for &i in &unit.images {
            match self.jpeg(unit.pass, i) {
                Ok(image) => {
                    let file = (self.file.images[i].width, self.file.images[i].height);
                    sizes.push(prompt::ImageSize { sent: image.size(), file });
                    jpegs.push(image.jpeg);
                }
                Err(why) => {
                    // A benchmark's files are its own: one that cannot be read is a broken benchmark.
                    self.fail(format!("image '{}': {why}", self.file.images[i].id), None);
                    return;
                }
            }
        }
        let scratch = self.job.slot.dir.join("tmp").join(&self.job.entry).join(format!("call-{}", unit.n));
        if let Err(e) = std::fs::create_dir_all(&scratch) {
            self.fail(format!("creating {}: {e}", scratch.display()), None);
            return;
        }
        let sent: Vec<(u32, u32)> = sizes.iter().map(|s| s.sent).collect();
        let text = prompt::user_prompt(&self.file.classes, &self.file.description, "", &sent);
        let request = AgentRequest {
            system: prompt::BENCH_SYSTEM,
            prompt: &text,
            images: &jpegs,
            schema: &self.schema,
            model: self.job.model.as_deref(),
            effort: self.job.effort,
            fast: false,
            scratch: &scratch,
            cancel: &self.job.cancel,
        };
        let mut usage = LabelUsage::default();
        // `Some(labels per image)` on a reply that was read, `Err(why)` when the images failed, `None` untouched.
        let read = |text: &str| prompt::parse_scored_reply(text, &self.file.classes, &sizes);
        // Every unreadable reply counts towards the cutoff, as in a labeling run.
        let on = |event: Event<'_>| {
            if let Event::Unreadable(why) = event {
                self.soft_failure(why, self.job.agent.hint(why));
            }
        };
        let agent = &*self.job.agent;
        let result = match ask(agent, &request, self.job.backoff, &mut usage, read, on) {
            Ok(labels) => {
                self.soft_failures.store(0, Ordering::Relaxed);
                Some(Ok(labels))
            }
            Err(BatchError::Cancelled) => None,
            Err(BatchError::Timeout) => {
                let why = format!(
                    "{} did not answer within {} s",
                    agent.id().display_name(),
                    CALL_TIMEOUT.as_secs()
                );
                self.soft_failure(&why, agent.hint(&why));
                Some(Err(why))
            }
            Err(BatchError::Transient(failure)) => {
                self.soft_failure(&failure.shown(), failure_hint(agent, &failure));
                Some(Err(failure.shown()))
            }
            Err(BatchError::Unreadable(why)) => Some(Err(why)),
            Err(BatchError::Failed(failure)) => {
                self.fail(failure.shown(), failure_hint(agent, &failure));
                None
            }
            Err(BatchError::Unavailable(why)) => {
                let hint = agent.hint(&why);
                self.fail(why, hint);
                None
            }
        };
        let _ = std::fs::remove_dir_all(&scratch);
        self.settle(&unit, result, &usage);
    }

    /// Writes a call's answers and usage, and the entry's progress.
    fn settle(
        &self,
        unit: &Unit,
        result: Option<std::result::Result<Vec<Option<prompt::ScoredLabels>>, String>>,
        usage: &LabelUsage,
    ) {
        let mut answered = 0u64;
        match result {
            None => {}
            Some(Err(why)) => {
                let mut f = self.failed.lock().unwrap();
                f.0 += unit.images.len();
                f.1.get_or_insert(why);
                answered = unit.images.len() as u64;
            }
            Some(Ok(labels)) => {
                let mut answers = self.answers.lock().unwrap();
                for (&i, labels) in unit.images.iter().zip(labels) {
                    answered += 1;
                    // As a labeling run judges an image: one left out, or whose every box was dropped, failed.
                    match prompt::scored_image_answer(labels) {
                        Ok(l) => answers[unit.pass][i] = Some(l.boxes),
                        Err(why) => {
                            let mut f = self.failed.lock().unwrap();
                            f.0 += 1;
                            f.1.get_or_insert(why);
                        }
                    }
                }
            }
        }
        let mut done = self.done.lock().unwrap();
        done.0 += answered;
        let emit = done.1.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY);
        if emit {
            done.1 = Some(Instant::now());
        }
        let mut state = self.job.slot.lock();
        let Some(e) = state.entry_mut(&self.job.entry) else { return };
        e.entry.progress.done = done.0;
        let variant = self.passes[unit.pass];
        e.entry.progress.condition = variant.map(|v| v.0.id().to_string());
        e.entry.progress.severity = variant.map(|v| v.1);
        e.entry.usage.get_or_insert_with(LabelUsage::default).add(usage);
        if emit {
            self.job.slot.emit(BenchEvent::Entry(Box::new(e.entry.clone())));
        }
    }

    /// The entry's grade, `Ok(None)` when it was stopped, or why it failed.
    fn outcome(&self) -> Result<Option<results::Graded>> {
        if let Some((why, hint)) = self.fatal.lock().unwrap().take() {
            let e = Error::new(axum::http::StatusCode::BAD_GATEWAY, "agent_failed", why);
            return Err(match hint {
                Some(h) => e.hint(h),
                None => e,
            });
        }
        if self.cancelled() {
            return Ok(None);
        }
        let total = self.job.images * self.passes.len();
        let (failed, why) = self.failed.lock().unwrap().clone();
        if failed > allowed_failures(total) {
            return Err(Error::new(
                axum::http::StatusCode::BAD_GATEWAY,
                "agent_failed",
                format!(
                    "{failed} of {total} images got no usable answer from the agent ({}); scoring them as images \
                     where it found nothing would not be honest",
                    why.unwrap_or_default()
                ),
            )
            .hint("start the entry again; a smaller max_images costs less to try"));
        }
        let answers = std::mem::take(&mut *self.answers.lock().unwrap());
        let scored = answers_scored(&answers);
        let mut clean = Vec::new();
        let mut variants = Vec::new();
        for (pass, answers) in self.passes.iter().zip(answers) {
            let samples: Vec<Sample> = answers
                .into_iter()
                .zip(&self.file.images)
                .map(|(dets, im)| {
                    let truths = match pass {
                        None => im.boxes.clone(),
                        Some((c, s)) => conditions::moved_boxes(*c, *s, im.width, im.height, &im.boxes)
                            .expect("planned severities are valid"),
                    };
                    let detections = dets
                        .unwrap_or_default()
                        .into_iter()
                        .map(|b| BenchDetection {
                            class: b.bbox.class,
                            score: if scored {
                                b.confidence.expect("every box has a confidence when scored")
                            } else {
                                1.0
                            },
                            x1: b.bbox.x1,
                            y1: b.bbox.y1,
                            x2: b.bbox.x2,
                            y2: b.bbox.y2,
                        })
                        .collect();
                    Sample { truths, detections }
                })
                .collect();
            match pass {
                None => clean = samples,
                Some(v) => variants.push(Pass { variant: Some(*v), samples }),
            }
        }
        let ids: Vec<String> = self.file.images[..self.job.images].iter().map(|i| i.id.clone()).collect();
        let complete = self.job.images == self.file.images.len()
            && (self.job.conditions || self.file.plan.conditions.is_empty());
        let spec = results::Spec {
            classes: self.file.classes.len(),
            scored,
            plan: &self.file.plan,
            seed: self.file.seed,
            complete,
        };
        Ok(Some(results::grade(&clean, &ids, &variants, &spec)))
    }
}

/// Whether an agent entry's collected answers should be graded as scored (with confidences): true only when no
/// detection omitted a confidence. If an agent omits confidence on any detection (legacy or mixed replies), the
/// whole entry falls back to unscored grading so fallback `1.0` constants never fabricate calibration scores or
/// skew AP/self-labeling rankings. When there are zero detections across every pass, `folo-bench` reports
/// `calibration: None` and `partial: true` regardless, matching model entries that find nothing.
fn answers_scored(answers: &[Vec<Option<Vec<prompt::ScoredBox>>>]) -> bool {
    answers.iter().flat_map(|pass| pass.iter().flatten()).flatten().all(|b| b.confidence.is_some())
}

#[cfg(test)]
mod tests {
    use folo_api::BBox;

    use super::*;

    fn box_with(confidence: Option<f32>) -> prompt::ScoredBox {
        prompt::ScoredBox { bbox: BBox { class: 0, x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 }, confidence }
    }

    #[test]
    fn a_few_failures_are_allowed() {
        assert_eq!(allowed_failures(4), 1);
        assert_eq!(allowed_failures(48), 1);
        assert_eq!(allowed_failures(100), 2);
        assert_eq!(allowed_failures(500), 10);
    }

    #[test]
    fn answers_scored_requires_confidence_on_every_detection() {
        // All detections report confidences -> scored.
        let scored = vec![vec![Some(vec![box_with(Some(0.9)), box_with(Some(0.4))]), Some(vec![])]];
        assert!(answers_scored(&scored));
        // Zero detections across all images (or failed images left None) -> scored (calibration is None in scorer).
        let empty = vec![vec![Some(vec![]), None]];
        assert!(answers_scored(&empty));
        // Legacy omitted confidences -> unscored.
        let legacy = vec![vec![Some(vec![box_with(None)])]];
        assert!(!answers_scored(&legacy));
        // Mixed confidences across images or passes -> unscored so fallback 1.0 never corrupts calibration.
        let mixed = vec![
            vec![Some(vec![box_with(Some(0.85))])],
            vec![Some(vec![box_with(Some(0.7)), box_with(None)])],
        ];
        assert!(!answers_scored(&mixed));
    }
}
