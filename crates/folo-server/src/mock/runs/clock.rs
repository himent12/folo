//! The wall-clock timeline of a live run: queue wait, epochs, stops and looping.

use std::time::{Duration, Instant};

use folo_api::RunStatus;

use crate::util::unix_now;

use super::{Curve, LiveClock, STOP_S, Snap};

impl LiveClock {
    /// A clock that started `offset_s` seconds ago.
    pub fn new(epoch_s: f64, queue_s: f64, hold_s: f64, looping: bool, offset_s: f64) -> LiveClock {
        let now = Instant::now();
        let origin = now.checked_sub(Duration::from_secs_f64(offset_s)).unwrap_or(now);
        let origin_wall = unix_now() - now.duration_since(origin).as_secs_f64();
        LiveClock {
            origin,
            origin_wall,
            epoch_s,
            queue_s,
            hold_s,
            looping,
            stop_at: None,
            cycle: 0,
            min_start: queue_s,
            queue: None,
            tail_s: 0.0,
            lost_at: None,
        }
    }

    /// The same clock, `running` for `tail_s` more seconds after the last epoch (a cloud run's collecting and
    /// releasing).
    pub fn with_tail(mut self, tail_s: f64) -> LiveClock {
        self.tail_s = tail_s;
        self
    }

    /// Seconds per epoch.
    pub fn epoch_s(&self) -> f64 {
        self.epoch_s
    }

    /// When a started run frees the training slot.
    pub(super) fn ends_at(&self, curve: &Curve) -> Instant {
        let end = match self.stop_at {
            Some(sa) => sa + STOP_S,
            None => self.queue_s + curve.epochs as f64 * self.epoch_s,
        };
        self.origin + Duration::from_secs_f64(end.max(0.0))
    }

    /// The clock as if `d` had passed: its origin and the planned starts it knows move back by `d`.
    pub(super) fn pass(&mut self, d: Duration) {
        self.origin -= d;
        self.origin_wall -= d.as_secs_f64();
        if let Some((ahead, all)) = &mut self.queue {
            for start in ahead.iter_mut().chain(all.iter_mut()) {
                *start -= d;
            }
        }
    }

    fn advance(&mut self, by: f64) {
        self.origin += Duration::from_secs_f64(by);
        self.origin_wall += by;
        self.stop_at = None;
        self.cycle += 1;
    }

    pub(super) fn snap(&mut self, curve: &Curve, now: Instant) -> Snap {
        let step_s = self.epoch_s / curve.spe as f64;
        let total = curve.total_steps();
        let run_end = self.queue_s + curve.epochs as f64 * self.epoch_s + self.tail_s;
        let queue_s = self.queue_s;
        let steps_at = |t: f64| (((t - queue_s) / step_s).floor().max(0.0) as u64).min(total);
        loop {
            let t = now.saturating_duration_since(self.origin).as_secs_f64();
            let base = Snap {
                status: RunStatus::Queued,
                step: 0,
                created: self.origin_wall,
                started: None,
                finished: None,
                eta_s: None,
                cycle: self.cycle,
                stop_requested: self.stop_at.map(|sa| self.origin_wall + sa),
                queue: None,
                at: self.origin_wall + t,
            };
            let started = (t >= self.queue_s).then_some(self.origin_wall + self.queue_s);
            if let Some(lost) = self.lost_at.filter(|&l| t >= l) {
                let started = (lost >= self.queue_s).then_some(self.origin_wall + self.queue_s);
                let finished = Some(self.origin_wall + lost);
                return Snap { status: RunStatus::Failed, step: steps_at(lost), started, finished, ..base };
            }
            if let Some(sa) = self.stop_at {
                let started = (sa >= self.queue_s).then_some(self.origin_wall + self.queue_s);
                let step = steps_at(sa);
                if t < sa + STOP_S {
                    return Snap { status: RunStatus::Stopping, step, started, ..base };
                }
                if self.looping && t >= sa + STOP_S + self.hold_s {
                    self.advance(sa + STOP_S + self.hold_s);
                    continue;
                }
                let finished = Some(self.origin_wall + sa + STOP_S);
                return Snap { status: RunStatus::Stopped, step, started, finished, ..base };
            }
            if t < self.queue_s {
                let queue = self.queue.as_ref().map(|(ahead, all)| {
                    let later = |v: &[Instant]| v.iter().filter(|&&s| s > now).count() as u32;
                    (1 + later(ahead), later(all).max(1))
                });
                return Snap { queue, ..base };
            }
            if t < run_end {
                let step = steps_at(t);
                return Snap { status: RunStatus::Running, step, started, eta_s: Some(run_end - t), ..base };
            }
            if self.looping && t >= run_end + self.hold_s {
                self.advance(run_end + self.hold_s);
                continue;
            }
            let finished = Some(self.origin_wall + run_end);
            return Snap { status: RunStatus::Completed, step: total, started, finished, ..base };
        }
    }
}
