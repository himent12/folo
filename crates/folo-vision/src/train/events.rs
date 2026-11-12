//! The trainer's event stream: docs/API.md's `status`, `step`, `epoch` and `log` objects, as [`RunEvent`]s.

use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use folo_api::{LogLevel, LogLine, RunEvent, RunStatus, StatusEvent};
use serde_json::Value;

/// Receives the trainer's events. Steps arrive rate-limited (at most one per [`super::TrainOptions::step_interval`],
/// plus the last batch of every epoch); everything else arrives as it happens. Called on the training thread, so
/// it should be quick (forward to a channel for slow consumers).
pub type EventSink = Box<dyn FnMut(&RunEvent) + Send>;

/// An [`EventSink`] that forwards every event into a channel (dropped silently once the receiver is gone).
pub fn channel_sink(tx: std::sync::mpsc::Sender<RunEvent>) -> EventSink {
    Box::new(move |e| {
        let _ = tx.send(e.clone());
    })
}

/// An event as one JSON object: its payload with an `"event"` field (`status`, `step`, `epoch`, `log`) first. This
/// is the line format of a run's `metrics.jsonl` and of `folo train --json`.
pub fn event_record(e: &RunEvent) -> Value {
    let payload: Value = serde_json::from_str(&e.data_json()).expect("event payloads are JSON");
    let mut map = serde_json::Map::new();
    map.insert("event".into(), Value::String(e.name().into()));
    if let Value::Object(o) = payload {
        map.extend(o);
    }
    Value::Object(map)
}

/// Seconds since the Unix epoch as RFC 3339 UTC with second precision (`2026-11-08T06:00:00Z`).
pub fn rfc3339_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, tod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", tod / 3600, (tod / 60) % 60, tod % 60)
}

/// Fans events out to the sink (rate-limited steps) and to `metrics.jsonl` (every event, every step).
pub(crate) struct Emitter {
    sink: Option<EventSink>,
    file: Option<std::io::BufWriter<std::fs::File>>,
    step_interval: Duration,
    last_step: Option<Instant>,
    /// The run's length, `(epochs, steps_per_epoch)`, carried by every `status` event.
    length: (u32, u64),
    /// While a checkpoint write runs (`train/writer.rs`): the events emitted meanwhile, in order, delivered after
    /// the epoch event that waits for the write.
    held: Option<Vec<Held>>,
}

/// An event held back by [`Emitter::hold`].
enum Held {
    Emit(RunEvent),
    Step(RunEvent, bool),
}

impl Emitter {
    pub(crate) fn new(
        sink: Option<EventSink>,
        file: Option<std::io::BufWriter<std::fs::File>>,
        step_interval: Duration,
        length: (u32, u64),
    ) -> Self {
        Emitter { sink, file, step_interval, last_step: None, length, held: None }
    }

    /// Holds every event from now on until [`Emitter::release`] (a checkpoint write is running).
    pub(crate) fn hold(&mut self) {
        self.held.get_or_insert_with(Vec::new);
    }

    /// Emits `first` (the epoch event the held ones waited for), then the held events in order, and stops holding.
    pub(crate) fn release(&mut self, first: Option<RunEvent>) {
        let held = self.held.take().unwrap_or_default();
        if let Some(e) = first {
            self.emit(e);
        }
        for h in held {
            match h {
                Held::Emit(e) => self.emit(e),
                Held::Step(e, force) => self.step(e, force),
            }
        }
    }

    /// Stops holding and drops the held events (the write they waited for failed, so the run fails at that
    /// epoch's end).
    pub(crate) fn release_dropping(&mut self) {
        self.held = None;
    }

    fn write(&mut self, e: &RunEvent) {
        if let Some(f) = &mut self.file {
            // The metrics file is a convenience copy of the stream: an IO error there must not stop training.
            if serde_json::to_writer(&mut *f, &event_record(e)).is_err() || writeln!(f).is_err() {
                self.file = None;
            }
        }
    }

    /// Emit a non-step event (and flush the metrics file, so readers see whole epochs).
    pub(crate) fn emit(&mut self, e: RunEvent) {
        if let Some(h) = &mut self.held {
            h.push(Held::Emit(e));
            return;
        }
        self.write(&e);
        if let Some(f) = &mut self.file {
            let _ = f.flush();
        }
        if let Some(s) = &mut self.sink {
            s(&e);
        }
    }

    /// Emit a step: always to the metrics file, to the sink at most once per interval unless `force`.
    pub(crate) fn step(&mut self, e: RunEvent, force: bool) {
        if let Some(h) = &mut self.held {
            h.push(Held::Step(e, force));
            return;
        }
        self.write(&e);
        let now = Instant::now();
        let due = force || self.last_step.is_none_or(|t| now.duration_since(t) >= self.step_interval);
        if due {
            self.last_step = Some(now);
            if let Some(s) = &mut self.sink {
                s(&e);
            }
        }
    }

    pub(crate) fn status(&mut self, status: RunStatus, error: Option<String>) {
        let (epochs, spe) = self.length;
        self.emit(RunEvent::Status(StatusEvent::new(status, error).with_length(epochs, spe)));
    }

    /// The final `failed` status, with the error's message and stable code.
    pub(crate) fn failed(&mut self, code: &str, message: String) {
        let (epochs, spe) = self.length;
        let ev = StatusEvent::new(RunStatus::Failed, Some(message))
            .with_length(epochs, spe)
            .with_error_code(Some(code.to_string()));
        self.emit(RunEvent::Status(ev));
    }

    pub(crate) fn log(&mut self, level: LogLevel, message: impl Into<String>) {
        self.emit(RunEvent::Log(LogLine { t: rfc3339_now(), level, message: message.into() }));
    }
}
