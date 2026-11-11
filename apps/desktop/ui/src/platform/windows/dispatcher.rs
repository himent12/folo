//! `WindowsDispatcher`: main-thread tasks (a queue plus one coalesced wake-up message to the platform's
//! message window), a background pool that grows on demand and lets idle threads go, and one timer
//! thread for `dispatch_after`. Nothing here polls: idle threads block on a condition variable.

use super::WM_FOLO_MAIN_TASKS;
use crate::PlatformDispatcher;
use async_task::Runnable;
use parking_lot::{Condvar, Mutex};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::thread::ThreadId;
use std::time::{Duration, Instant};
use ::windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use ::windows::Win32::UI::WindowsAndMessaging::PostMessageW;

/// Most background threads alive at once. The app blocks pool threads in HTTP calls, so the pool must
/// be generous; idle threads exit after `IDLE_EXIT`.
pub const MAX_POOL_THREADS: usize = 256;
/// How long an idle pool thread waits for work before it exits.
pub const IDLE_EXIT: Duration = Duration::from_secs(5);

/// The main thread's task queue. Producers push and post one wake-up message while none is pending.
pub(crate) struct MainQueue {
    tasks: Mutex<VecDeque<Runnable>>,
    wake_pending: AtomicBool,
    /// The platform's message-only window (0 until it exists).
    hwnd: AtomicIsize,
}

impl MainQueue {
    fn new() -> Self {
        Self { tasks: Mutex::new(VecDeque::new()), wake_pending: AtomicBool::new(false), hwnd: AtomicIsize::new(0) }
    }

    fn push(&self, runnable: Runnable) {
        self.tasks.lock().push_back(runnable);
        if !self.wake_pending.swap(true, Ordering::AcqRel) {
            self.post_wake();
        }
    }

    fn post_wake(&self) {
        let hwnd = self.hwnd.load(Ordering::Acquire);
        if hwnd != 0 {
            // SAFETY: posting to a window handle; a stale handle only makes the call fail.
            let _ = unsafe { PostMessageW(Some(HWND(hwnd as _)), WM_FOLO_MAIN_TASKS, WPARAM(0), LPARAM(0)) };
        }
    }

    /// Runs the tasks queued when it was called, one at a time (a task may open a modal dialog whose
    /// own loop calls this again, and the rest must not wait for the dialog). Main thread only.
    pub(crate) fn run_pending(&self) {
        self.wake_pending.store(false, Ordering::Release);
        let count = self.tasks.lock().len();
        for _ in 0..count {
            let Some(runnable) = self.tasks.lock().pop_front() else { break };
            runnable.run();
        }
        if !self.tasks.lock().is_empty() && !self.wake_pending.swap(true, Ordering::AcqRel) {
            self.post_wake();
        }
    }

    /// Drops whatever is left (on the main thread, where the tasks' futures live).
    pub(crate) fn clear(&self) {
        let left = std::mem::take(&mut *self.tasks.lock());
        drop(left);
    }
}

#[derive(Default)]
struct PoolState {
    queue: VecDeque<Runnable>,
    threads: usize,
    idle: usize,
}

/// The background pool: a thread is added whenever queued work outnumbers idle threads.
struct Pool {
    state: Mutex<PoolState>,
    work: Condvar,
}

impl Pool {
    fn dispatch(self: &Arc<Self>, runnable: Runnable) {
        let mut state = self.state.lock();
        state.queue.push_back(runnable);
        if state.queue.len() > state.idle && state.threads < MAX_POOL_THREADS {
            state.threads += 1;
            drop(state);
            let pool = self.clone();
            let spawned = std::thread::Builder::new().name("folo-ui-pool".into()).spawn(move || pool.worker());
            if spawned.is_err() {
                self.state.lock().threads -= 1;
            }
        } else {
            drop(state);
            self.work.notify_one();
        }
    }

    fn worker(&self) {
        loop {
            let runnable = {
                let mut state = self.state.lock();
                loop {
                    if let Some(runnable) = state.queue.pop_front() {
                        break runnable;
                    }
                    state.idle += 1;
                    let timed_out = self.work.wait_for(&mut state, IDLE_EXIT).timed_out();
                    state.idle -= 1;
                    if timed_out && state.queue.is_empty() {
                        state.threads -= 1;
                        return;
                    }
                }
            };
            runnable.run();
        }
    }

    fn threads(&self) -> usize {
        self.state.lock().threads
    }
}

struct Timer {
    due: Instant,
    seq: u64,
    runnable: Runnable,
}

impl PartialEq for Timer {
    fn eq(&self, other: &Self) -> bool {
        (self.due, self.seq) == (other.due, other.seq)
    }
}
impl Eq for Timer {}
impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Timer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.due, self.seq).cmp(&(other.due, other.seq))
    }
}

#[derive(Default)]
struct TimerState {
    heap: BinaryHeap<Reverse<Timer>>,
    seq: u64,
    running: bool,
}

/// One thread sleeps until the earliest deadline (or forever while there is none), then hands the
/// due runnables to the pool in deadline order (equal deadlines in dispatch order).
struct Timers {
    state: Mutex<TimerState>,
    changed: Condvar,
}

impl Timers {
    fn add(self: &Arc<Self>, due: Instant, runnable: Runnable, pool: &Arc<Pool>) {
        let mut state = self.state.lock();
        let seq = state.seq;
        state.seq += 1;
        state.heap.push(Reverse(Timer { due, seq, runnable }));
        if !state.running {
            state.running = true;
            let timers = self.clone();
            let pool = pool.clone();
            let _ = std::thread::Builder::new().name("folo-ui-timers".into()).spawn(move || timers.run(&pool));
        }
        drop(state);
        self.changed.notify_one();
    }

    fn run(&self, pool: &Arc<Pool>) {
        let mut state = self.state.lock();
        loop {
            let now = Instant::now();
            while state.heap.peek().is_some_and(|t| t.0.due <= now) {
                let Reverse(timer) = state.heap.pop().unwrap();
                pool.dispatch(timer.runnable);
            }
            match state.heap.peek() {
                Some(next) => {
                    let wait = next.0.due.saturating_duration_since(now);
                    self.changed.wait_for(&mut state, wait);
                }
                None => self.changed.wait(&mut state),
            }
        }
    }
}

/// The platform's `PlatformDispatcher`.
pub struct WindowsDispatcher {
    main_thread: ThreadId,
    pub(crate) main: Arc<MainQueue>,
    pool: Arc<Pool>,
    timers: Arc<Timers>,
}

impl WindowsDispatcher {
    /// A dispatcher whose main thread is the calling thread. Main-thread tasks wait until
    /// `set_main_window` names the window their wake-up message goes to.
    pub(crate) fn new() -> Self {
        Self {
            main_thread: std::thread::current().id(),
            main: Arc::new(MainQueue::new()),
            pool: Arc::new(Pool { state: Mutex::new(PoolState::default()), work: Condvar::new() }),
            timers: Arc::new(Timers { state: Mutex::new(TimerState::default()), changed: Condvar::new() }),
        }
    }

    /// Names the message window main-thread wake-ups are posted to, and posts one if work is waiting.
    pub(crate) fn set_main_window(&self, hwnd: HWND) {
        self.main.hwnd.store(hwnd.0 as isize, Ordering::Release);
        if !self.main.tasks.lock().is_empty() && !self.main.wake_pending.swap(true, Ordering::AcqRel) {
            self.main.post_wake();
        }
    }

    /// How many background pool threads are alive (for tests and diagnostics).
    pub fn pool_threads(&self) -> usize {
        self.pool.threads()
    }
}

impl PlatformDispatcher for WindowsDispatcher {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread
    }

    fn dispatch(&self, runnable: Runnable) {
        self.pool.dispatch(runnable);
    }

    fn dispatch_on_main_thread(&self, runnable: Runnable) {
        self.main.push(runnable);
    }

    fn dispatch_after(&self, duration: Duration, runnable: Runnable) {
        self.timers.add(Instant::now() + duration, runnable, &self.pool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn spawn_with(dispatcher: Arc<WindowsDispatcher>, f: impl FnOnce() + Send + 'static) {
        let (runnable, task) = async_task::spawn(async move { f() }, move |r| dispatcher.dispatch(r));
        task.detach();
        runnable.schedule();
    }

    #[test]
    fn the_pool_grows_past_blocked_threads() {
        let d = Arc::new(WindowsDispatcher::new());
        // 40 jobs that all wait for each other: they finish only if 40 threads run at once.
        let barrier = Arc::new(std::sync::Barrier::new(40));
        let (tx, rx) = mpsc::channel();
        for _ in 0..40 {
            let (barrier, tx) = (barrier.clone(), tx.clone());
            spawn_with(d.clone(), move || {
                barrier.wait();
                tx.send(()).unwrap();
            });
        }
        for _ in 0..40 {
            rx.recv_timeout(Duration::from_secs(5)).expect("a job starved");
        }
        assert!(d.pool_threads() >= 40);
    }

    #[test]
    fn timers_fire_in_deadline_order() {
        let d = Arc::new(WindowsDispatcher::new());
        let (tx, rx) = mpsc::channel();
        for (i, ms) in [(0, 60u64), (1, 20), (2, 40), (3, 80)] {
            let tx = tx.clone();
            let dd = d.clone();
            let (runnable, task) =
                async_task::spawn(async move { tx.send(i).unwrap() }, move |r| dd.dispatch_after(Duration::from_millis(ms), r));
            task.detach();
            runnable.schedule();
        }
        let start = Instant::now();
        let got: Vec<i32> = (0..4).map(|_| rx.recv_timeout(Duration::from_secs(2)).unwrap()).collect();
        assert_eq!(got, [1, 2, 0, 3]);
        assert!(start.elapsed() >= Duration::from_millis(55));
    }
}
