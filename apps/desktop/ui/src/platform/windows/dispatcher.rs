//! `WindowsDispatcher`: main-thread tasks (a queue plus one coalesced wake-up message to the platform's
//! message window), a background pool that grows on demand and lets idle threads go, and one timer
//! thread for `dispatch_after`. Nothing here polls: idle threads block on a condition variable.
//! `shut_down` (once the message loop has ended) drops what is queued and lets the threads exit.

use super::WM_FOLO_MAIN_TASKS;
use crate::PlatformDispatcher;
use ::windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use ::windows::Win32::UI::WindowsAndMessaging::PostMessageW;
use async_task::Runnable;
use parking_lot::{Condvar, Mutex};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

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
    /// Set (under the tasks' lock) by `shut_down`: nothing runs main-thread tasks any more.
    shut_down: AtomicBool,
}

impl MainQueue {
    fn new() -> Self {
        Self {
            tasks: Mutex::new(VecDeque::new()),
            wake_pending: AtomicBool::new(false),
            hwnd: AtomicIsize::new(0),
            shut_down: AtomicBool::new(false),
        }
    }

    fn push(&self, runnable: Runnable) {
        let mut tasks = self.tasks.lock();
        if self.shut_down.load(Ordering::Acquire) {
            // Dropping it cancels its task, which may schedule another: outside the lock.
            drop(tasks);
            return drop(runnable);
        }
        tasks.push_back(runnable);
        drop(tasks);
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

    /// Drops whatever is left.
    pub(crate) fn clear(&self) {
        let left = std::mem::take(&mut *self.tasks.lock());
        drop(left);
    }

    /// Stops taking tasks and drops the queued ones. A task woken later, on any thread, is dropped there:
    /// a main-thread runnable holds nothing of its future (`executor`), which the dispatcher's `shut_down`
    /// dropped on the main thread.
    fn shut_down(&self) {
        let left = {
            let mut tasks = self.tasks.lock();
            self.shut_down.store(true, Ordering::Release);
            std::mem::take(&mut *tasks)
        };
        drop(left);
    }
}

#[derive(Default)]
struct PoolState {
    queue: VecDeque<Runnable>,
    threads: usize,
    idle: usize,
    /// Set by `shut_down`: runnables are dropped, threads exit.
    shut_down: bool,
}

/// The background pool: a thread is added whenever queued work outnumbers idle threads.
struct Pool {
    state: Mutex<PoolState>,
    work: Condvar,
}

impl Pool {
    fn dispatch(self: &Arc<Self>, runnable: Runnable) {
        let mut state = self.state.lock();
        if state.shut_down {
            // Dropping it cancels its task, which may schedule another: outside the lock.
            drop(state);
            return drop(runnable);
        }
        state.queue.push_back(runnable);
        if state.queue.len() > state.idle && state.threads < MAX_POOL_THREADS {
            state.threads += 1;
            drop(state);
            let pool = self.clone();
            let spawned =
                std::thread::Builder::new().name("folo-ui-pool".into()).spawn(move || pool.worker());
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
                    if state.shut_down {
                        state.threads -= 1;
                        return;
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

    fn shut_down(&self) {
        let queue = {
            let mut state = self.state.lock();
            state.shut_down = true;
            std::mem::take(&mut state.queue)
        };
        self.work.notify_all();
        drop(queue);
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
    /// Set by `shut_down`: the thread exits, later timers are dropped.
    stopped: bool,
}

impl TimerState {
    fn push(&mut self, due: Instant, runnable: Runnable) {
        let seq = self.seq;
        self.seq += 1;
        self.heap.push(Reverse(Timer { due, seq, runnable }));
    }

    /// The timers due at `now`, in deadline order (equal deadlines in dispatch order).
    fn pop_due(&mut self, now: Instant) -> Vec<Runnable> {
        let mut due = Vec::new();
        while self.heap.peek().is_some_and(|t| t.0.due <= now) {
            if let Some(Reverse(timer)) = self.heap.pop() {
                due.push(timer.runnable);
            }
        }
        due
    }
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
        if state.stopped {
            drop(state);
            return drop(runnable);
        }
        state.push(due, runnable);
        if !state.running {
            state.running = true;
            let timers = self.clone();
            let pool = pool.clone();
            let _ =
                std::thread::Builder::new().name("folo-ui-timers".into()).spawn(move || timers.run(&pool));
        }
        drop(state);
        self.changed.notify_one();
    }

    fn run(&self, pool: &Arc<Pool>) {
        let mut state = self.state.lock();
        loop {
            if state.stopped {
                return;
            }
            let now = Instant::now();
            for runnable in state.pop_due(now) {
                pool.dispatch(runnable);
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

    fn shut_down(&self) {
        let heap = {
            let mut state = self.state.lock();
            state.stopped = true;
            std::mem::take(&mut state.heap)
        };
        self.changed.notify_all();
        drop(heap);
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
        let main_thread = std::thread::current().id();
        Self {
            main_thread,
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

    /// Stops the dispatcher once the message loop has ended: queued runnables are dropped (cancelling
    /// their tasks), later ones are dropped as they arrive, and the timer and pool threads exit (a pool
    /// thread once its current job ends). A queued runnable's scheduler holds the dispatcher, so
    /// without this the queues would keep it, the tasks' state and those threads alive. Main thread.
    pub(crate) fn shut_down(&self) {
        self.main.shut_down();
        // The main-thread tasks that are waiting, not queued: dropped here, on their thread.
        crate::executor::drop_foreground_futures(self);
        // The timer first: it hands due runnables to the pool.
        self.timers.shut_down();
        self.pool.shut_down();
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
    use std::rc::Rc;
    use std::sync::mpsc;

    fn spawn_with(dispatcher: Arc<WindowsDispatcher>, f: impl FnOnce() + Send + 'static) {
        let (runnable, task) = async_task::spawn(async move { f() }, move |r| dispatcher.dispatch(r));
        task.detach();
        runnable.schedule();
    }

    /// Waits (polling every millisecond) until `done`, up to a limit no loaded machine reaches.
    fn wait_until(done: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !done() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        done()
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
            rx.recv_timeout(Duration::from_secs(30)).expect("a job starved");
        }
        assert!(d.pool_threads() >= 40);
        d.shut_down();
    }

    /// The timer thread hands due runnables over in deadline order, equal deadlines in dispatch order.
    /// Tested on the handover itself with the deadlines given, never waited for: once handed over they
    /// run on pool threads, in whatever order those get the CPU.
    #[test]
    fn timers_are_handed_over_in_deadline_order() {
        let base = Instant::now();
        let ran = Arc::new(Mutex::new(Vec::new()));
        let mut state = TimerState::default();
        for (i, ms) in [(0, 60u64), (1, 20), (2, 40), (3, 80), (4, 20)] {
            let ran = ran.clone();
            let (runnable, task) = async_task::spawn(async move { ran.lock().push(i) }, |_| {});
            task.detach();
            state.push(base + Duration::from_millis(ms), runnable);
        }
        let mut run_due = |at: u64| {
            for runnable in state.pop_due(base + Duration::from_millis(at)) {
                runnable.run();
            }
            std::mem::take(&mut *ran.lock())
        };
        assert_eq!(run_due(19), Vec::<i32>::new());
        assert_eq!(run_due(40), [1, 4, 2]);
        assert_eq!(run_due(79), [0]);
        assert_eq!(run_due(1000), [3]);
    }

    /// A real timer runs on a pool thread, never before its deadline (a bound that holds however loaded
    /// the machine is: the deadline is taken after `start`).
    #[test]
    fn a_timer_runs_on_the_pool_not_early() {
        let d = Arc::new(WindowsDispatcher::new());
        let (tx, rx) = mpsc::channel();
        let start = Instant::now();
        let dd = d.clone();
        let (runnable, task) = async_task::spawn(
            async move { tx.send((Instant::now(), std::thread::current().name().map(String::from))).unwrap() },
            move |r| dd.dispatch_after(Duration::from_millis(30), r),
        );
        task.detach();
        runnable.schedule();
        let (at, thread) = rx.recv_timeout(Duration::from_secs(30)).expect("the timer fired");
        assert!(at.duration_since(start) >= Duration::from_millis(30));
        assert_eq!(thread.as_deref(), Some("folo-ui-pool"));
        d.shut_down();
    }

    #[test]
    fn shut_down_drops_queued_work_and_lets_its_threads_exit() {
        let d = Arc::new(WindowsDispatcher::new());
        let weak = Arc::downgrade(&d);
        // A pool thread, a main-thread runnable that never runs (no message window), and a timer an
        // hour away: each queued runnable's scheduler holds the dispatcher.
        let (tx, rx) = mpsc::channel();
        spawn_with(d.clone(), move || tx.send(()).unwrap());
        rx.recv_timeout(Duration::from_secs(30)).expect("a background job ran");
        let dd = d.clone();
        let (runnable, task) = async_task::spawn(async {}, move |r| dd.dispatch_on_main_thread(r));
        task.detach();
        runnable.schedule();
        let dd = d.clone();
        let (runnable, task) =
            async_task::spawn(async {}, move |r| dd.dispatch_after(Duration::from_secs(3600), r));
        task.detach();
        runnable.schedule();
        let timers = Arc::downgrade(&d.timers);
        assert!(d.timers.state.lock().running, "the timer thread started");

        d.shut_down();
        assert!(wait_until(|| d.pool_threads() == 0), "pool threads exit");
        spawn_with(d.clone(), || {});
        let dd = d.clone();
        let (runnable, task) =
            async_task::spawn(async {}, move |r| dd.dispatch_after(Duration::from_secs(1), r));
        task.detach();
        runnable.schedule();
        assert_eq!(d.pool_threads(), 0, "work after shutdown is dropped, no thread starts");
        assert!(d.main.tasks.lock().is_empty() && d.timers.state.lock().heap.is_empty());
        drop(d);
        assert!(wait_until(|| timers.upgrade().is_none()), "the timer thread exits");
        assert!(weak.upgrade().is_none(), "nothing keeps the dispatcher alive");
    }

    /// A main-thread task waiting when the dispatcher shuts down: its future is dropped then, on the main
    /// thread, and its runnable, scheduled later on another thread, is dropped there without a panic (it
    /// holds nothing of the future) and keeps nothing alive (it used to be leaked, with the dispatcher).
    #[test]
    fn a_main_thread_task_woken_elsewhere_after_shut_down_is_dropped_there() {
        let d = Arc::new(WindowsDispatcher::new());
        let weak = Arc::downgrade(&d);
        let state = Rc::new(());
        let (tx, rx) = crate::oneshot::channel::<()>();
        let held = state.clone();
        let executor = crate::ForegroundExecutor::new(d.clone());
        executor
            .spawn(async move {
                rx.await.ok();
                drop(held);
            })
            .detach();
        d.main.run_pending();
        assert_eq!(Rc::strong_count(&state), 2, "the task waits, holding its state");
        d.shut_down();
        assert_eq!(Rc::strong_count(&state), 1, "its state is dropped at the shutdown, on this thread");
        // Waking it on another thread schedules its runnable there, after the shutdown.
        std::thread::spawn(move || tx.send(()).ok()).join().expect("no panic off the main thread");
        drop((executor, d));
        assert!(weak.upgrade().is_none(), "nothing keeps the dispatcher alive");
    }
}
