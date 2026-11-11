//! `HeadlessDispatcher`: the fake clock and the queues behind headless apps. Owner tag: `"elements"`.
//!
//! - **Main thread**: runnables wait in a queue that `run_until_idle` runs on the calling thread.
//! - **Background**: runnables run on real threads (a pool that grows on demand; idle threads exit), since
//!   the app blocks background tasks in HTTP calls and mock delays. `run_until_idle` waits for every
//!   outstanding job, with a timeout that panics with a clear message (a job that never ends is a bug in a
//!   test or a `--shot` scenario). A job that panics fails the next `run_until_idle` with its message.
//! - **Delayed** (`dispatch_after`, i.e. timers): due when the fake clock passes them. The clock moves only
//!   in `advance_clock`, which steps from deadline to deadline, running the due timers on the calling thread
//!   in deadline order (then insertion order) and pumping until idle after each step, so a task that re-arms
//!   a timer from the time it woke behaves as on a real clock. A timer's runnable is an empty future, so
//!   running it here instead of on a pool thread only changes which thread wakes the waiting task.

use std::{
    any::Any,
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    thread::ThreadId,
    time::{Duration, Instant},
};

use async_task::Runnable;
use parking_lot::{Condvar, Mutex};

use crate::PlatformDispatcher;

/// How long `run_until_idle` waits for background jobs before it panics.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(20);

/// How long an idle pool thread waits for work before it exits.
const WORKER_IDLE: Duration = Duration::from_millis(500);

/// The queues and clock behind `HeadlessDispatcher`.
pub(crate) struct DispatcherState {
    /// The fake time.
    pub(crate) now: Instant,
    /// Main-thread runnables.
    pub(crate) foreground: VecDeque<Runnable>,
    /// Background runnables waiting for a pool thread.
    pub(crate) background: VecDeque<Runnable>,
    /// Delayed runnables and when they are due, in insertion order.
    pub(crate) delayed: Vec<(Instant, Runnable)>,
    /// Background jobs queued or running.
    pub(crate) outstanding: usize,
    /// Pool threads waiting for work.
    pub(crate) idle_workers: usize,
    /// The first panic message of a background job since the last `run_until_idle`.
    pub(crate) panic: Option<String>,
    /// Set when the dispatcher is dropped: pool threads exit.
    pub(crate) shut_down: bool,
}

/// What the dispatcher and its pool threads share.
pub(crate) struct Shared {
    /// The queues and clock.
    pub(crate) state: Mutex<DispatcherState>,
    /// Pool threads wait on it for work.
    work: Condvar,
    /// The pumping thread waits on it for main-thread work or a finished job.
    progress: Condvar,
}

/// A dispatcher whose main queue the test pumps, whose clock the test moves, and whose background jobs run
/// on real threads.
pub struct HeadlessDispatcher {
    /// The queues.
    pub(crate) shared: Arc<Shared>,
    /// The thread that made it (the "main" thread).
    main_thread: ThreadId,
    /// `run_until_idle`'s limit for background jobs.
    idle_timeout: Mutex<Duration>,
}

impl Default for HeadlessDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl HeadlessDispatcher {
    /// A dispatcher whose main thread is the calling thread.
    pub fn new() -> Self {
        HeadlessDispatcher {
            shared: Arc::new(Shared {
                state: Mutex::new(DispatcherState {
                    now: Instant::now(),
                    foreground: VecDeque::new(),
                    background: VecDeque::new(),
                    delayed: Vec::new(),
                    outstanding: 0,
                    idle_workers: 0,
                    panic: None,
                    shut_down: false,
                }),
                work: Condvar::new(),
                progress: Condvar::new(),
            }),
            main_thread: std::thread::current().id(),
            idle_timeout: Mutex::new(DEFAULT_IDLE_TIMEOUT),
        }
    }

    /// Sets how long `run_until_idle` waits for background jobs before it panics.
    pub fn set_idle_timeout(&self, timeout: Duration) {
        *self.idle_timeout.lock() = timeout;
    }

    /// Runs main-thread runnables and due timers until none is left and no background job is outstanding.
    /// Returns whether anything ran on this thread. Panics if background jobs are still running after the
    /// idle timeout, or if one panicked. Call it while the app is not borrowed.
    pub fn run_until_idle(&self) -> bool {
        let timeout = *self.idle_timeout.lock();
        let deadline = Instant::now() + timeout;
        let mut ran = false;
        let mut state = self.shared.state.lock();
        loop {
            if let Some(message) = state.panic.take() {
                drop(state);
                panic!("a background job panicked: {message}");
            }
            let now = state.now;
            let runnable = state.foreground.pop_front().or_else(|| take_next_due(&mut state.delayed, now));
            if let Some(runnable) = runnable {
                drop(state);
                runnable.run();
                ran = true;
                state = self.shared.state.lock();
                continue;
            }
            if state.outstanding == 0 {
                return ran;
            }
            if self.shared.progress.wait_until(&mut state, deadline).timed_out()
                && state.foreground.is_empty()
                && state.outstanding > 0
            {
                let outstanding = state.outstanding;
                drop(state);
                panic!("run_until_idle: {outstanding} background job(s) still running after {timeout:?}");
            }
        }
    }

    /// Moves the clock forward by `duration`, stepping through each timer deadline on the way: at each, the
    /// due timers run and the queues are pumped until idle.
    pub fn advance_clock(&self, duration: Duration) {
        let target = self.shared.state.lock().now + duration;
        loop {
            self.run_until_idle();
            let mut state = self.shared.state.lock();
            let next = state.delayed.iter().map(|(due, _)| *due).filter(|due| *due <= target).min();
            match next {
                Some(due) => state.now = state.now.max(due),
                None => {
                    state.now = target;
                    return;
                }
            }
        }
    }

    /// Whether nothing is queued, due or running in the background.
    pub fn is_idle(&self) -> bool {
        let state = self.shared.state.lock();
        state.foreground.is_empty()
            && state.outstanding == 0
            && !state.delayed.iter().any(|(due, _)| *due <= state.now)
    }

    /// When the next timer is due, if one is pending.
    pub fn next_timer(&self) -> Option<Instant> {
        self.shared.state.lock().delayed.iter().map(|(due, _)| *due).min()
    }
}

/// Removes and returns the earliest due runnable (insertion order among equal deadlines).
fn take_next_due(delayed: &mut Vec<(Instant, Runnable)>, now: Instant) -> Option<Runnable> {
    let (ix, _) = delayed
        .iter()
        .enumerate()
        .filter(|(_, (due, _))| *due <= now)
        .min_by_key(|(ix, (due, _))| (*due, *ix))?;
    Some(delayed.remove(ix).1)
}

impl Drop for HeadlessDispatcher {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.shut_down = true;
        // Dropping queued runnables cancels their tasks; do it outside the lock.
        let foreground = std::mem::take(&mut state.foreground);
        let delayed = std::mem::take(&mut state.delayed);
        drop(state);
        self.shared.work.notify_all();
        drop(foreground);
        drop(delayed);
    }
}

impl PlatformDispatcher for HeadlessDispatcher {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread
    }

    fn dispatch(&self, runnable: Runnable) {
        let mut state = self.shared.state.lock();
        if state.shut_down {
            return;
        }
        state.background.push_back(runnable);
        state.outstanding += 1;
        // One pool thread per queued job: a job may block until another one finishes.
        let spawn = state.background.len() > state.idle_workers;
        drop(state);
        if spawn {
            let shared = self.shared.clone();
            std::thread::Builder::new()
                .name("folo-ui-headless".into())
                .spawn(move || worker(shared))
                .expect("spawn a headless pool thread");
        } else {
            self.shared.work.notify_one();
        }
    }

    fn dispatch_on_main_thread(&self, runnable: Runnable) {
        let mut state = self.shared.state.lock();
        if state.shut_down {
            return;
        }
        state.foreground.push_back(runnable);
        drop(state);
        self.shared.progress.notify_all();
    }

    fn dispatch_after(&self, duration: Duration, runnable: Runnable) {
        let mut state = self.shared.state.lock();
        if state.shut_down {
            return;
        }
        let due = state.now + duration;
        state.delayed.push((due, runnable));
        drop(state);
        self.shared.progress.notify_all();
    }

    fn now(&self) -> Instant {
        self.shared.state.lock().now
    }
}

/// A pool thread: runs background runnables until it has been idle for `WORKER_IDLE`.
fn worker(shared: Arc<Shared>) {
    let mut state = shared.state.lock();
    loop {
        if let Some(runnable) = state.background.pop_front() {
            drop(state);
            let result = catch_unwind(AssertUnwindSafe(|| runnable.run()));
            state = shared.state.lock();
            state.outstanding -= 1;
            if let Err(payload) = result {
                state.panic.get_or_insert_with(|| panic_message(payload.as_ref()));
            }
            shared.progress.notify_all();
            continue;
        }
        if state.shut_down {
            return;
        }
        state.idle_workers += 1;
        let timed_out = shared.work.wait_for(&mut state, WORKER_IDLE).timed_out();
        state.idle_workers -= 1;
        if timed_out && state.background.is_empty() {
            return;
        }
    }
}

/// The message of a caught panic.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "(non-string panic payload)".into())
}
