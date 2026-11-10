//! The training slot that runs created through the API share.

use std::sync::Arc;
use std::time::{Duration, Instant};

use folo_api::RunStatus;

use crate::store::QueueMove;

use super::{SimRun, Timeline};

/// Whether `run` waits in the training queue now (queued, not stopped).
pub fn is_waiting(run: &SimRun) -> bool {
    run.snap().queue.is_some()
}

/// Plans the training slot shared by the runs in `queue` (front first) as of `now`: runs that started keep their
/// timeline; each waiting run starts when the slot frees (not before its `min_start`), in queue order. Every
/// waiting run also learns the starts of the runs ahead of it, so its queue position follows from the clock.
pub fn reschedule(queue: &[Arc<SimRun>], now: Instant) {
    let mut free_at = now;
    let mut starts: Vec<(Arc<SimRun>, Instant)> = Vec::new();
    for run in queue {
        let Timeline::Live(c) = &run.timeline else { continue };
        let mut c = c.lock().unwrap();
        let snap = c.snap(&run.curve, now);
        match snap.status {
            // A run stopped while it waited never held the slot.
            RunStatus::Running | RunStatus::Stopping if snap.started.is_some() => {
                free_at = free_at.max(c.ends_at(&run.curve));
                c.queue = None;
            }
            RunStatus::Queued => {
                let start = free_at.max(c.origin + Duration::from_secs_f64(c.min_start.max(0.0)));
                c.queue_s = start.saturating_duration_since(c.origin).as_secs_f64();
                free_at = start + Duration::from_secs_f64(run.curve.epochs as f64 * c.epoch_s);
                starts.push((run.clone(), start));
            }
            _ => c.queue = None,
        }
    }
    let all: Vec<Instant> = starts.iter().map(|s| s.1).collect();
    for (i, (run, _)) in starts.iter().enumerate() {
        if let Timeline::Live(c) = &run.timeline {
            c.lock().unwrap().queue = Some((all[..i].to_vec(), all.clone()));
        }
    }
}

/// Moves run `id` among the waiting runs of `queue` (runs that are not waiting keep their order, ahead of the
/// waiting ones) and plans the slot again. `start` also lets the run start at once when the slot is free. Returns
/// `(from, to, waiting)` (0-based places), or `None` when the run does not wait in the queue.
pub fn move_waiting(queue: &mut Vec<Arc<SimRun>>, id: &str, mv: QueueMove) -> Option<(usize, usize, usize)> {
    let now = Instant::now();
    reschedule(queue, now);
    let (mut waiting, rest): (Vec<Arc<SimRun>>, Vec<Arc<SimRun>>) =
        queue.drain(..).partition(|r| is_waiting(r));
    let from = waiting.iter().position(|r| r.id == id);
    let moved = from.map(|from| {
        let to = mv.target(from, waiting.len());
        let run = waiting.remove(from);
        if mv == QueueMove::Start
            && let Timeline::Live(c) = &run.timeline
        {
            let mut c = c.lock().unwrap();
            c.min_start = now.saturating_duration_since(c.origin).as_secs_f64();
        }
        waiting.insert(to, run);
        (from, to, waiting.len())
    });
    queue.extend(rest);
    queue.extend(waiting);
    reschedule(queue, now);
    moved
}
