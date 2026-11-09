//! An agent run: the images of a job, batch by batch, through an [`Agent`].
//!
//! A run has one thread that starts `parallel` workers. Each worker takes a batch off the queue, encodes its
//! images, makes one agent call, checks the reply and writes the labels back under the job's lock. Three rules
//! keep a run from wasting tokens: a call is one model request with the images inline; a reply that is not the
//! JSON asked for is retried once and no more; and a failure that another call would only repeat (the CLI
//! missing or signed out, an unknown model, a rate limit, calls timing out in a row) ends the run at once, killing
//! the calls in flight.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use folo_api::{LabelEffort, LabelEvent, LabelImageStatus, LabelOutcome, LabelSource, LabelUsage, LogLevel};

use super::agent::{Agent, AgentError, AgentRequest, CALL_TIMEOUT};
use super::job::Job;
use super::prompt::{self, Labels};
use crate::util::{rfc3339, unix_now};

/// The longest side, in pixels, of an image as an agent sees it. Larger images are scaled down: about 1000
/// input tokens per image, and box coordinates are relative, so nothing is lost but detail the models do not use.
pub const AGENT_IMAGE_SIDE: u32 = 1024;
/// JPEG quality of the images sent.
const JPEG_QUALITY: u8 = 85;
/// Calls in a row that may time out or return garbage before the run gives up.
const MAX_SOFT_FAILURES: u32 = 3;

/// What a run does, decided when it starts.
pub struct Plan {
    pub agent: Arc<dyn Agent>,
    pub model: Option<String>,
    pub effort: LabelEffort,
    /// The extra instruction of this run.
    pub note: String,
    /// The image ids of every call, in order.
    pub batches: Vec<Vec<String>>,
    /// Calls in flight at once.
    pub parallel: usize,
    /// The run's stop switch ([`super::job::Active::cancel`]).
    pub cancel: Arc<AtomicBool>,
}

/// What a call did to one image.
enum Outcome {
    Labeled(Labels),
    Failed(String),
    /// The call did not happen or was cut short: the image stays as it was.
    Untouched,
}

/// An image of a call, ready to send.
struct Sent {
    id: String,
    size: (u32, u32),
    jpeg: Vec<u8>,
}

struct Run<'a> {
    job: &'a Job,
    plan: &'a Plan,
    classes: Vec<String>,
    instructions: String,
    schema: serde_json::Value,
    queue: Mutex<VecDeque<(usize, Vec<String>)>>,
    /// Calls in a row that timed out or returned garbage.
    soft_failures: AtomicU32,
    /// Why the run gave up, once it has.
    fatal: Mutex<Option<String>>,
}

/// Runs `plan` on `job` to its end (the body of the run's thread). The job's `active` and `run` are set by the
/// caller; this clears `active` and closes `run` when it is done.
pub fn run(job: &Job, plan: &Plan) {
    let (classes, instructions) = {
        let state = job.lock();
        (state.file.classes.clone(), state.file.instructions.clone())
    };
    let run = Run {
        job,
        plan,
        schema: prompt::reply_schema(&classes),
        classes,
        instructions,
        queue: Mutex::new(plan.batches.iter().cloned().enumerate().map(|(i, b)| (i + 1, b)).collect()),
        soft_failures: AtomicU32::new(0),
        fatal: Mutex::new(None),
    };
    let workers = plan.parallel.clamp(1, plan.batches.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| run.work());
        }
    });
    run.finish();
}

impl Run<'_> {
    fn agent_name(&self) -> &'static str {
        self.plan.agent.id().display_name()
    }

    fn cancelled(&self) -> bool {
        self.plan.cancel.load(Ordering::Relaxed)
    }

    fn work(&self) {
        while !self.cancelled() {
            let Some((n, ids)) = self.queue.lock().unwrap().pop_front() else { break };
            self.call(n, ids);
        }
    }

    /// Gives up: no further calls, and the ones in flight are ended.
    fn fail(&self, why: String) {
        self.fatal.lock().unwrap().get_or_insert(why);
        self.plan.cancel.store(true, Ordering::Relaxed);
    }

    /// Call number `n`: the images `ids` through the agent and their labels into the job.
    fn call(&self, n: usize, ids: Vec<String>) {
        let taken: Vec<_> = {
            let mut state = self.job.lock();
            let mut taken = Vec::new();
            for id in ids {
                let Some(active) = state.active.as_mut() else { break };
                active.waiting.remove(&id);
                active.in_call.insert(id.clone());
                if let Ok(r) = state.record(&id) {
                    self.job.emit(LabelEvent::Image(state.image(r)));
                    taken.push((id, self.job.image_path(r), (r.width, r.height)));
                }
            }
            taken
        };
        let (mut sent, mut outcomes) = (Vec::new(), Vec::new());
        for (id, path, size) in taken {
            match agent_jpeg(&path) {
                Ok(jpeg) => sent.push(Sent { id, size, jpeg }),
                Err(why) => outcomes.push((id, Outcome::Failed(why))),
            }
        }
        let mut usage = LabelUsage::default();
        if !sent.is_empty() {
            outcomes.extend(self.ask(n, &sent, &mut usage));
        }
        self.settle(n, outcomes, &usage);
    }

    /// One agent call for `sent` (a second one when the first reply cannot be read): what became of each image.
    fn ask(&self, n: usize, sent: &[Sent], usage: &mut LabelUsage) -> Vec<(String, Outcome)> {
        let all = |outcome: fn(String) -> Outcome, why: &str| -> Vec<(String, Outcome)> {
            sent.iter().map(|s| (s.id.clone(), outcome(why.to_string()))).collect()
        };
        let untouched = |_| Outcome::Untouched;
        let scratch = self.job.dir.join("tmp").join(format!("call-{n}"));
        if let Err(e) = std::fs::create_dir_all(&scratch) {
            self.fail(format!("creating {}: {e}", scratch.display()));
            return all(untouched, "");
        }
        let images: Vec<Vec<u8>> = sent.iter().map(|s| s.jpeg.clone()).collect();
        let sizes: Vec<(u32, u32)> = sent.iter().map(|s| s.size).collect();
        let prompt = prompt::user_prompt(&self.classes, &self.instructions, &self.plan.note, sent.len());
        let request = AgentRequest {
            system: prompt::SYSTEM,
            prompt: &prompt,
            images: &images,
            schema: &self.schema,
            model: self.plan.model.as_deref(),
            effort: self.plan.effort,
            scratch: &scratch,
            cancel: &self.plan.cancel,
        };
        let mut retried = false;
        let outcomes = loop {
            let reply = match self.plan.agent.label(&request) {
                Ok(reply) => reply,
                Err(AgentError::Cancelled) => break all(untouched, ""),
                Err(AgentError::Timeout) => {
                    let why =
                        format!("{} did not answer within {} s", self.agent_name(), CALL_TIMEOUT.as_secs());
                    self.soft_failure(&why);
                    break all(Outcome::Failed, &why);
                }
                Err(AgentError::Unavailable(why) | AgentError::Failed(why)) => {
                    self.fail(why);
                    break all(untouched, "");
                }
            };
            usage.add(&reply.usage);
            match prompt::parse_reply(&reply.text, &self.classes, &sizes) {
                Ok(labels) => {
                    self.soft_failures.store(0, Ordering::Relaxed);
                    break sent
                        .iter()
                        .zip(labels)
                        .map(|(s, labels)| {
                            let outcome = match labels {
                                Some(labels) => Outcome::Labeled(labels),
                                None => Outcome::Failed("the agent's reply left this image out".into()),
                            };
                            (s.id.clone(), outcome)
                        })
                        .collect();
                }
                Err(why) if !retried && !self.cancelled() => {
                    retried = true;
                    let mut state = self.job.lock();
                    self.job.log(&mut state, LogLevel::Warn, format!("call {n}: {why}; asking once more"));
                }
                Err(why) => {
                    self.soft_failure(&why);
                    break all(Outcome::Failed, &why);
                }
            }
        };
        let _ = std::fs::remove_dir_all(&scratch);
        outcomes
    }

    /// Counts a call that timed out or returned garbage; too many in a row end the run.
    fn soft_failure(&self, why: &str) {
        if self.soft_failures.fetch_add(1, Ordering::Relaxed) + 1 >= MAX_SOFT_FAILURES {
            self.fail(format!("{MAX_SOFT_FAILURES} calls in a row failed; the last one: {why}"));
        }
    }

    /// Writes a call's results into the job: labels, statuses, the run's progress and usage.
    fn settle(&self, n: usize, outcomes: Vec<(String, Outcome)>, usage: &LabelUsage) {
        let mut state = self.job.lock();
        let (mut labeled, mut failed, mut boxes, mut dropped) = (0u64, 0u64, 0usize, 0usize);
        let mut failures: Vec<String> = Vec::new();
        for (id, outcome) in outcomes {
            if let Some(active) = state.active.as_mut() {
                active.in_call.remove(&id);
            }
            let Some(record) = state.record_mut(&id) else { continue };
            match outcome {
                Outcome::Labeled(labels) => {
                    labeled += 1;
                    boxes += labels.boxes.len();
                    dropped += labels.dropped;
                    record.boxes = labels.boxes;
                    record.status = LabelImageStatus::Labeled;
                    record.source = Some(LabelSource::Agent);
                    record.error = None;
                }
                Outcome::Failed(why) => {
                    failed += 1;
                    record.status = LabelImageStatus::Failed;
                    record.error = Some(why.clone());
                    if !failures.contains(&why) {
                        failures.push(why);
                    }
                }
                Outcome::Untouched => {}
            }
            let record = state.record(&id).expect("the record was just changed");
            self.job.emit(LabelEvent::Image(state.image(record)));
        }
        state.file.usage.add(usage);
        if let Some(run) = state.file.run.as_mut() {
            run.done += labeled;
            run.failed += failed;
            run.usage.add(usage);
        }
        if usage.calls > 0 {
            let message = format!(
                "call {n}: {labeled} {} labeled, {boxes} {} · {} tokens in, {} out",
                plural(labeled, "image"),
                plural(boxes as u64, "box"),
                usage.input_tokens,
                usage.output_tokens
            );
            self.job.log(&mut state, LogLevel::Info, message);
        }
        if dropped > 0 {
            let message = format!(
                "call {n}: {dropped} {} dropped (a label that is not a class, or a box that is not one)",
                plural(dropped as u64, "box")
            );
            self.job.log(&mut state, LogLevel::Warn, message);
        }
        for why in failures {
            self.job.log(&mut state, LogLevel::Warn, format!("call {n}: {why}"));
        }
        if let Err(e) = self.job.save(&state) {
            tracing::warn!(job = %self.job.id, "{e}");
        }
        self.job.emit_job(&state);
    }

    /// Closes the run: its outcome, the job idle again.
    fn finish(&self) {
        let fatal = self.fatal.lock().unwrap().take();
        let _ = std::fs::remove_dir_all(self.job.dir.join("tmp"));
        let mut state = self.job.lock();
        state.active = None;
        let name = self.agent_name();
        let hint = fatal.as_deref().and_then(|why| self.plan.agent.hint(why));
        let Some(run) = state.file.run.as_mut() else { return };
        run.finished = Some(rfc3339(unix_now()));
        run.outcome = Some(match &fatal {
            Some(_) => LabelOutcome::Failed,
            None if self.plan.cancel.load(Ordering::Relaxed) => LabelOutcome::Stopped,
            None => LabelOutcome::Completed,
        });
        run.error = fatal;
        run.hint = hint;
        let summary = format!(
            "{} of {} {} labeled, {} failed · {} {}, {} tokens in, {} out",
            run.done,
            run.total,
            plural(run.total, "image"),
            run.failed,
            run.usage.calls,
            plural(run.usage.calls, "call"),
            run.usage.input_tokens,
            run.usage.output_tokens
        );
        let (level, message) = match (run.outcome, &run.error) {
            (Some(LabelOutcome::Failed), Some(why)) => {
                (LogLevel::Error, format!("{name} stopped: {why} ({summary})"))
            }
            (Some(LabelOutcome::Stopped), _) => (LogLevel::Info, format!("{name} was stopped: {summary}")),
            _ => (LogLevel::Info, format!("{name} is done: {summary}")),
        };
        self.job.log(&mut state, level, message);
        if let Err(e) = self.job.save(&state) {
            tracing::warn!(job = %self.job.id, "{e}");
        }
        self.job.emit_job(&state);
    }
}

fn plural(n: u64, word: &'static str) -> String {
    match (n, word) {
        (1, _) => word.to_string(),
        (_, "box") => "boxes".to_string(),
        _ => format!("{word}s"),
    }
}

/// The image at `path` as an agent gets it: upright, at most [`AGENT_IMAGE_SIDE`] on its long side, JPEG.
pub fn agent_jpeg(path: &Path) -> Result<Vec<u8>, String> {
    let image = folo_data::read_image(path).map_err(|e| e.to_string())?;
    let long = image.width.max(image.height);
    let image = if long > AGENT_IMAGE_SIDE {
        let scale = f64::from(AGENT_IMAGE_SIDE) / f64::from(long);
        let side = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
        folo_data::resize::resize(&image, side(image.width), side(image.height))
    } else {
        image
    };
    folo_data::encode(&image, folo_data::ImageFormat::Jpeg(JPEG_QUALITY)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::*;

    #[test]
    fn large_images_are_scaled_down_and_small_ones_kept() {
        let dir = TestDir::new("label-agent-jpeg");
        let big = dir.join("big.png");
        folo_data::write_image(
            &folo_data::Image::new(3000, 1500, [9, 9, 9]),
            &big,
            folo_data::ImageFormat::Png,
        )
        .unwrap();
        let small = dir.join("small.png");
        folo_data::write_image(
            &folo_data::Image::new(320, 200, [9, 9, 9]),
            &small,
            folo_data::ImageFormat::Png,
        )
        .unwrap();
        let size = |path: &Path| {
            let jpeg = agent_jpeg(path).unwrap();
            assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
            let image = folo_data::decode(&jpeg).unwrap();
            (image.width, image.height)
        };
        assert_eq!(size(&big), (1024, 512));
        assert_eq!(size(&small), (320, 200));
        assert!(agent_jpeg(&dir.join("missing.png")).is_err());
    }

    #[test]
    fn plurals() {
        assert_eq!(plural(1, "box"), "box");
        assert_eq!(plural(2, "box"), "boxes");
        assert_eq!(plural(0, "image"), "images");
        assert_eq!(plural(1, "call"), "call");
    }
}
