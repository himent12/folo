//! The winit backend's `PlatformDispatcher`: main-thread runnables go through one coalesced wake-up of
//! the event loop, background runnables to a pool that grows on demand and lets idle threads exit, and
//! delayed runnables to one timer thread. `shut_down` (once the loop has ended) drops what is queued and
//! lets those threads exit.

use crate::PlatformDispatcher;
use async_task::Runnable;
use parking_lot::{Condvar, Mutex};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// The most background threads the pool runs at once.
pub const MAX_BACKGROUND_THREADS: usize = 64;
/// How long a background thread waits for work before it exits.
const IDLE_EXIT: Duration = Duration::from_secs(10);

/// Wakes the event loop so it drains the main-thread queue; `false` once the loop is gone.
pub(crate) type MainWaker = Box<dyn Fn() -> bool + Send + Sync>;

/// The dispatcher (see the module docs). Main-thread runnables run when the loop calls `drain_main`.
pub struct WinitDispatcher {
    main_thread: ThreadId,
    main: Mutex<MainQueue>,
    waker: MainWaker,
    pool: Arc<Pool>,
    timer: OnceLock<Arc<Timer>>,
}

#[derive(Default)]
struct MainQueue {
    runnables: VecDeque<Runnable>,
    wake_pending: bool,
    shut_down: bool,
}

impl WinitDispatcher {
    /// A dispatcher whose main thread is the calling thread.
    pub(crate) fn new(waker: MainWaker) -> Self {
        Self {
            main_thread: thread::current().id(),
            main: Mutex::new(MainQueue::default()),
            waker,
            pool: Arc::new(Pool::default()),
            timer: OnceLock::new(),
        }
    }

    /// Runs the main-thread runnables queued so far (the loop calls this on its wake-up event). Returns
    /// how many ran. Runnables they queue wait for the next wake-up.
    pub fn drain_main(&self) -> usize {
        let runnables = {
            let mut main = self.main.lock();
            main.wake_pending = false;
            std::mem::take(&mut main.runnables)
        };
        let count = runnables.len();
        for runnable in runnables {
            runnable.run();
        }
        count
    }

    /// Background threads alive now (busy or idle).
    pub fn background_threads(&self) -> usize {
        self.pool.state.lock().threads
    }

    /// Stops the dispatcher once the loop has ended: queued runnables are dropped (cancelling their tasks),
    /// later ones are dropped as they arrive, and the timer and pool threads exit (a pool thread once its
    /// current job ends). A queued runnable's scheduler holds the dispatcher, so without this the queues
    /// would keep it, the tasks' state and those threads alive. Main thread.
    pub fn shut_down(&self) {
        let runnables = {
            let mut main = self.main.lock();
            main.shut_down = true;
            std::mem::take(&mut main.runnables)
        };
        self.discard_main(runnables);
        // The timer first: it hands due runnables to the pool under its lock.
        if let Some(timer) = self.timer.get() {
            timer.shut_down();
        }
        self.pool.shut_down();
    }

    /// Drops main-thread runnables, cancelling their tasks. Off the main thread (a pool thread waking a task
    /// after the loop ended) it leaks them instead: dropping one there panics inside async-task.
    fn discard_main(&self, runnables: VecDeque<Runnable>) {
        if self.is_main_thread() {
            drop(runnables);
        } else {
            runnables.into_iter().for_each(std::mem::forget);
        }
    }
}

impl PlatformDispatcher for WinitDispatcher {
    fn is_main_thread(&self) -> bool {
        thread::current().id() == self.main_thread
    }

    fn dispatch(&self, runnable: Runnable) {
        self.pool.dispatch(runnable);
    }

    fn dispatch_on_main_thread(&self, runnable: Runnable) {
        let wake = {
            let mut main = self.main.lock();
            if main.shut_down {
                drop(main);
                return self.discard_main(VecDeque::from([runnable]));
            }
            main.runnables.push_back(runnable);
            !std::mem::replace(&mut main.wake_pending, true)
        };
        if wake && !(self.waker)() {
            // The loop has exited: nothing will run these.
            let runnables = std::mem::take(&mut self.main.lock().runnables);
            self.discard_main(runnables);
        }
    }

    fn dispatch_after(&self, duration: Duration, runnable: Runnable) {
        if duration.is_zero() || self.pool.state.lock().shut_down {
            // After `shut_down` the pool drops it, and no timer thread starts.
            return self.pool.dispatch(runnable);
        }
        let timer = self.timer.get_or_init(|| Timer::start(self.pool.clone()));
        timer.schedule(Instant::now() + duration, runnable);
    }
}

#[derive(Default)]
struct Pool {
    state: Mutex<PoolState>,
    work: Condvar,
}

#[derive(Default)]
struct PoolState {
    queue: VecDeque<Runnable>,
    threads: usize,
    idle: usize,
    /// Idle threads already woken for queued work.
    wakeups: usize,
    /// Set by `shut_down`: runnables are dropped, threads exit.
    shut_down: bool,
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
        if state.idle > state.wakeups {
            state.wakeups += 1;
            self.work.notify_one();
        } else if state.threads < MAX_BACKGROUND_THREADS {
            let pool = self.clone();
            let spawned = thread::Builder::new().name("folo-ui-background".into()).spawn(move || pool.work());
            if spawned.is_ok() {
                state.threads += 1;
            } else if state.threads == 0 {
                // No thread at all: run here rather than never.
                let runnable = state.queue.pop_back();
                drop(state);
                if let Some(runnable) = runnable {
                    runnable.run();
                }
            }
        }
    }

    fn work(&self) {
        let mut state = self.state.lock();
        loop {
            if let Some(runnable) = state.queue.pop_front() {
                drop(state);
                runnable.run();
                state = self.state.lock();
                continue;
            }
            if state.shut_down {
                state.threads -= 1;
                return;
            }
            state.idle += 1;
            let timed_out = self.work.wait_for(&mut state, IDLE_EXIT).timed_out();
            state.idle -= 1;
            if state.wakeups > 0 {
                state.wakeups -= 1;
                continue;
            }
            if timed_out && state.queue.is_empty() {
                state.threads -= 1;
                return;
            }
        }
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
    queue: Mutex<BinaryHeap<Delayed>>,
    changed: Condvar,
    /// Set (under the queue's lock) by `shut_down`: the thread exits.
    stopped: AtomicBool,
}

struct Delayed {
    at: Instant,
    seq: u64,
    runnable: Runnable,
}

impl PartialEq for Delayed {
    fn eq(&self, other: &Self) -> bool {
        (self.at, self.seq) == (other.at, other.seq)
    }
}
impl Eq for Delayed {}
impl PartialOrd for Delayed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Delayed {
    /// Reversed, so the heap's top is the earliest (and among equals the first scheduled).
    fn cmp(&self, other: &Self) -> Ordering {
        (other.at, other.seq).cmp(&(self.at, self.seq))
    }
}

impl Timer {
    fn start(pool: Arc<Pool>) -> Arc<Self> {
        let timer = Arc::new(Timer {
            queue: Mutex::new(BinaryHeap::new()),
            changed: Condvar::new(),
            stopped: AtomicBool::new(false),
        });
        let worker = timer.clone();
        thread::Builder::new()
            .name("folo-ui-timer".into())
            .spawn(move || worker.run(&pool))
            .expect("failed to start the folo-ui timer thread");
        timer
    }

    fn schedule(&self, at: Instant, runnable: Runnable) {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut queue = self.queue.lock();
        if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
            drop(queue);
            return drop(runnable);
        }
        let earliest = queue.peek().is_none_or(|top| at < top.at);
        queue.push(Delayed { at, seq, runnable });
        if earliest {
            self.changed.notify_one();
        }
    }

    fn run(&self, pool: &Arc<Pool>) {
        let mut queue = self.queue.lock();
        loop {
            if self.stopped.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            let now = Instant::now();
            while queue.peek().is_some_and(|top| top.at <= now) {
                let due = queue.pop().expect("peeked");
                pool.dispatch(due.runnable);
            }
            match queue.peek().map(|top| top.at) {
                Some(at) => {
                    self.changed.wait_until(&mut queue, at);
                }
                None => self.changed.wait(&mut queue),
            }
        }
    }

    fn shut_down(&self) {
        let queue = {
            let mut queue = self.queue.lock();
            self.stopped.store(true, std::sync::atomic::Ordering::Relaxed);
            std::mem::take(&mut *queue)
        };
        self.changed.notify_all();
        drop(queue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    fn dispatcher() -> (Arc<WinitDispatcher>, Arc<AtomicUsize>) {
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = wakes.clone();
        let waker: MainWaker = Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            true
        });
        (Arc::new(WinitDispatcher::new(waker)), wakes)
    }

    fn spawn_on(d: &Arc<WinitDispatcher>, how: &str, f: impl FnOnce() + Send + 'static) {
        let d2 = d.clone();
        let how = how.to_owned();
        let (runnable, task) = async_task::spawn(async move { f() }, move |r| match how.as_str() {
            "main" => d2.dispatch_on_main_thread(r),
            _ => d2.dispatch(r),
        });
        runnable.schedule();
        task.detach();
    }

    #[test]
    fn main_thread_runnables_coalesce_one_wake_and_keep_order() {
        let (d, wakes) = dispatcher();
        let (tx, rx) = mpsc::channel();
        for i in 0..5 {
            let tx = tx.clone();
            spawn_on(&d, "main", move || tx.send(i).unwrap());
        }
        assert_eq!(wakes.load(Ordering::SeqCst), 1, "one wake-up for five runnables");
        assert!(d.is_main_thread());
        assert_eq!(d.drain_main(), 5);
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
        spawn_on(&d, "main", move || tx.send(5).unwrap());
        assert_eq!(wakes.load(Ordering::SeqCst), 2, "a new batch wakes again");
        assert_eq!(d.drain_main(), 1);
        assert_eq!(rx.try_recv(), Ok(5));
    }

    #[test]
    fn background_pool_runs_many_jobs_and_grows_past_32() {
        let (d, _) = dispatcher();
        let (tx, rx) = mpsc::channel();
        // 40 jobs that all wait on one barrier need 40 threads at once.
        let barrier = Arc::new(std::sync::Barrier::new(41));
        for _ in 0..40 {
            let barrier = barrier.clone();
            let tx = tx.clone();
            spawn_on(&d, "background", move || {
                barrier.wait();
                tx.send(()).unwrap();
            });
        }
        barrier.wait();
        for _ in 0..40 {
            rx.recv_timeout(Duration::from_secs(5)).expect("a background job ran");
        }
        assert!(d.background_threads() >= 40);
        assert!(!d.is_main_thread() || thread::current().id() == d.main_thread);

        let done = Arc::new(AtomicUsize::new(0));
        for _ in 0..1000 {
            let done = done.clone();
            spawn_on(&d, "background", move || {
                done.fetch_add(1, Ordering::SeqCst);
            });
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while done.load(Ordering::SeqCst) < 1000 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(done.load(Ordering::SeqCst), 1000);
        assert!(d.background_threads() <= MAX_BACKGROUND_THREADS);
    }

    #[test]
    fn timers_fire_in_deadline_order() {
        let (d, _) = dispatcher();
        let (tx, rx) = mpsc::channel();
        let start = Instant::now();
        for (i, ms) in [(0, 60u64), (1, 20), (2, 40), (3, 20)] {
            let tx = tx.clone();
            let d2 = d.clone();
            let (runnable, task) =
                async_task::spawn(async move { tx.send((i, Instant::now())).unwrap() }, move |r| {
                    d2.dispatch_after(Duration::from_millis(ms), r)
                });
            runnable.schedule();
            task.detach();
        }
        let got: Vec<(i32, Instant)> =
            (0..4).map(|_| rx.recv_timeout(Duration::from_secs(2)).unwrap()).collect();
        let order: Vec<i32> = got.iter().map(|g| g.0).collect();
        // 1 and 3 share a deadline and run on the pool, in either order; then 2, then 0.
        assert!(order[..2].contains(&1) && order[..2].contains(&3), "{order:?}");
        assert_eq!(&order[2..], [2, 0]);
        assert!(got.iter().all(|(i, at)| {
            let want = [60, 20, 40, 20][*i as usize];
            at.duration_since(start) >= Duration::from_millis(want)
        }));
    }

    #[test]
    fn shut_down_drops_queued_work_and_lets_its_threads_exit() {
        let (d, _) = dispatcher();
        let weak = Arc::downgrade(&d);
        // A pool thread, a main-thread runnable never drained, and a timer an hour away: each queued
        // runnable's scheduler holds the dispatcher.
        let (tx, rx) = mpsc::channel();
        spawn_on(&d, "background", move || tx.send(()).unwrap());
        rx.recv_timeout(Duration::from_secs(5)).expect("a background job ran");
        spawn_on(&d, "main", || {});
        let d2 = d.clone();
        let (runnable, task) =
            async_task::spawn(async {}, move |r| d2.dispatch_after(Duration::from_secs(3600), r));
        runnable.schedule();
        task.detach();
        let timer = Arc::downgrade(d.timer.get().expect("the timer started"));

        d.shut_down();
        let deadline = Instant::now() + Duration::from_secs(5);
        while d.background_threads() > 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(d.background_threads(), 0, "pool threads exit");
        spawn_on(&d, "main", || {});
        spawn_on(&d, "background", || {});
        assert_eq!(d.background_threads(), 0, "work after shutdown is dropped, no thread starts");
        drop(d);
        while timer.upgrade().is_some() && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(timer.upgrade().is_none(), "the timer thread exits");
        assert!(weak.upgrade().is_none(), "nothing keeps the dispatcher alive");
    }
}
