//! Executors: `BackgroundExecutor` (a thread pool through `PlatformDispatcher::dispatch`, plus timers),
//! `ForegroundExecutor` (the UI thread, `dispatch_on_main_thread`), and `Task<T>`. Owner tag: `"runtime"`.
//!
//! Tasks are `async-task` runnables: waking one schedules it on its dispatcher queue, so an idle app has no
//! timer and no thread awake. Dropping a `Task` cancels it; `detach` lets it run to the end.
//!
//! A foreground task is polled only on the thread that spawned it (the platform runs main-thread runnables
//! there). Its future lives in that thread's registry (`FOREGROUND`), and the async-task holds only a
//! handle to it that owns nothing of it, so a foreground runnable may be dropped on any thread: a waker on
//! a pool thread that schedules it after the dispatcher shut down drops it there. The future itself is
//! dropped only on its thread: when it finishes or its task is cancelled, or when the dispatcher shuts
//! down (`drop_foreground_futures`), which cancels the tasks that are waiting as well as the queued ones.
//! A task woken while the app is gone but its dispatcher is not just runs: its `AsyncApp` calls return
//! `Err`, nothing panics.

use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    fmt,
    future::Future,
    marker::PhantomData,
    pin::{Pin, pin},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread::{Thread, ThreadId},
    time::{Duration, Instant},
};

use crate::PlatformDispatcher;

/// Runs futures on background threads.
#[derive(Clone)]
pub struct BackgroundExecutor {
    /// The platform's queues.
    pub(crate) dispatcher: Arc<dyn PlatformDispatcher>,
}

/// Runs futures on the UI thread (not `Send`).
#[derive(Clone)]
pub struct ForegroundExecutor {
    /// The platform's queues.
    pub(crate) dispatcher: Arc<dyn PlatformDispatcher>,
    /// Keeps the executor on the thread that made it.
    not_send: PhantomData<Rc<()>>,
}

/// A spawned future's result. Await it, or `detach` it; dropping it cancels the future.
#[must_use]
pub struct Task<T>(TaskState<T>);

/// A task is either already finished or a running `async-task`.
enum TaskState<T> {
    /// `Task::ready`.
    Ready(Option<T>),
    /// Spawned on an executor.
    Spawned(async_task::Task<T>),
}

impl<T> Task<T> {
    /// A task that is already done.
    pub fn ready(value: T) -> Self {
        Task(TaskState::Ready(Some(value)))
    }

    /// Lets the task run to completion with nobody awaiting it.
    pub fn detach(self) {
        match self.0 {
            TaskState::Ready(_) => {}
            TaskState::Spawned(task) => task.detach(),
        }
    }
}

impl<T> Future for Task<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `TaskState` is never moved out of the pinned `Task`; both arms are `Unpin`.
        match unsafe { &mut self.get_unchecked_mut().0 } {
            TaskState::Ready(value) => Poll::Ready(value.take().expect("Task polled after completion")),
            TaskState::Spawned(task) => Pin::new(task).poll(cx),
        }
    }
}

impl<T> fmt::Debug for Task<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            TaskState::Ready(_) => f.write_str("Task::Ready"),
            TaskState::Spawned(_) => f.write_str("Task::Spawned"),
        }
    }
}

impl BackgroundExecutor {
    /// An executor over `dispatcher`.
    pub fn new(dispatcher: Arc<dyn PlatformDispatcher>) -> Self {
        BackgroundExecutor { dispatcher }
    }

    /// Runs `future` on a background thread.
    pub fn spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R>
    where
        R: Send + 'static,
    {
        let dispatcher = self.dispatcher.clone();
        let (runnable, task) = async_task::spawn(future, move |runnable| dispatcher.dispatch(runnable));
        runnable.schedule();
        Task(TaskState::Spawned(task))
    }

    /// Resolves after `duration` (through `dispatch_after`; no thread sleeps). A zero duration is ready
    /// at once, as GPUI's.
    pub fn timer(&self, duration: Duration) -> Task<()> {
        if duration.is_zero() {
            return Task::ready(());
        }
        let dispatcher = self.dispatcher.clone();
        let (runnable, task) =
            async_task::spawn(async {}, move |runnable| dispatcher.dispatch_after(duration, runnable));
        runnable.schedule();
        Task(TaskState::Spawned(task))
    }

    /// The dispatcher's clock (the headless clock in tests).
    pub fn now(&self) -> Instant {
        self.dispatcher.now()
    }

    /// Polls `future` on this thread until it is ready or `timeout` (real time) has passed; `None` on
    /// timeout. Nothing else runs on this thread meanwhile (GPUI's
    /// `block_with_timeout`, used by quit).
    pub fn block_with_timeout<F: Future>(&self, timeout: Duration, future: F) -> Option<F::Output> {
        let deadline = Instant::now() + timeout;
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut future = pin!(future);
        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return Some(output);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            std::thread::park_timeout(deadline - now);
        }
    }
}

/// Wakes a thread parked in `block_with_timeout`.
struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

impl ForegroundExecutor {
    /// An executor over `dispatcher`; call on the UI thread.
    pub fn new(dispatcher: Arc<dyn PlatformDispatcher>) -> Self {
        ForegroundExecutor { dispatcher, not_send: PhantomData }
    }

    /// Runs `future` on the UI thread.
    pub fn spawn<R>(&self, future: impl Future<Output = R> + 'static) -> Task<R>
    where
        R: 'static,
    {
        let future = ForegroundFuture::register(dispatcher_key(&*self.dispatcher), future);
        let dispatcher = self.dispatcher.clone();
        // SAFETY: async-task's conditions hold: the future (`ForegroundFuture`, ids only) is `Send` and
        // `'static`, and so is the schedule closure (an `Arc` of a `Send + Sync` dispatcher). What
        // `spawn` would add, a `Send` output, is not needed: the output is made only on this thread
        // (`ForegroundFuture::poll` asserts it) and is read through `Task<R>`, which is not `Send` unless
        // `R` is; a detached task's output is dropped by `Runnable::run`, here too.
        let (runnable, task) = unsafe {
            async_task::spawn_unchecked(future, move |runnable| dispatcher.dispatch_on_main_thread(runnable))
        };
        runnable.schedule();
        Task(TaskState::Spawned(task))
    }

    /// The dispatcher's clock.
    pub fn now(&self) -> Instant {
        self.dispatcher.now()
    }
}

/// The key of a dispatcher's foreground tasks in the registry: its address.
fn dispatcher_key(dispatcher: &dyn PlatformDispatcher) -> usize {
    dispatcher as *const dyn PlatformDispatcher as *const () as usize
}

thread_local! {
    /// This thread's live foreground futures.
    static FOREGROUND: RefCell<Registry> = RefCell::default();
}

/// Foreground futures by dispatcher (`dispatcher_key`) and task; a future is `None` while it is polled.
#[derive(Default)]
struct Registry {
    tasks: HashMap<usize, HashMap<u64, Option<Box<dyn Any>>>>,
}

impl Registry {
    /// Removes a task, returning its future if it is here (not being polled).
    fn remove(&mut self, dispatcher: usize, id: u64) -> Option<Box<dyn Any>> {
        let tasks = self.tasks.get_mut(&dispatcher)?;
        let future = tasks.remove(&id).flatten();
        if tasks.is_empty() {
            self.tasks.remove(&dispatcher);
        }
        future
    }
}

/// Drops the futures of every foreground task of `dispatcher` that lives on this thread, waiting or
/// queued (the platforms call it from the dispatcher's `shut_down`, on the main thread: nothing will run
/// them any more). Their runnables are left holding nothing, and their tasks never finish. A task being
/// polled right now is dropped when its poll returns.
pub(crate) fn drop_foreground_futures(dispatcher: &dyn PlatformDispatcher) {
    let key = dispatcher_key(dispatcher);
    let futures = FOREGROUND.try_with(|r| r.try_borrow_mut().ok().and_then(|mut r| r.tasks.remove(&key)));
    // Outside the borrow: dropping a future can spawn or end other tasks.
    drop(futures);
}

/// What a foreground task's async-task holds: the ids of its future in its thread's registry. So it is
/// `Send`, and dropping it on another thread drops nothing that belongs to the main thread.
struct ForegroundFuture<F> {
    dispatcher: usize,
    id: u64,
    thread: ThreadId,
    future: PhantomData<fn() -> F>,
}

impl<F: Future + 'static> ForegroundFuture<F> {
    /// Puts `future` in this thread's registry.
    fn register(dispatcher: usize, future: F) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let future: Box<dyn Any> = Box::new(Box::pin(future));
        FOREGROUND.with_borrow_mut(|r| r.tasks.entry(dispatcher).or_default().insert(id, Some(future)));
        ForegroundFuture { dispatcher, id, thread: std::thread::current().id(), future: PhantomData }
    }
}

impl<F: Future + 'static> Future for ForegroundFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        assert_eq!(std::thread::current().id(), self.thread, "a foreground task polled off its thread");
        let (dispatcher, id) = (self.dispatcher, self.id);
        // Taken out for the poll, which may spawn or end other foreground tasks.
        let taken = FOREGROUND.with_borrow_mut(|r| {
            r.tasks.get_mut(&dispatcher).and_then(|tasks| tasks.get_mut(&id)).and_then(Option::take)
        });
        // Gone: its dispatcher shut down, and the task never finishes.
        let Some(taken) = taken else { return Poll::Pending };
        let mut future = *taken.downcast::<Pin<Box<F>>>().expect("a foreground future of its own type");
        let poll = future.as_mut().poll(cx);
        // What is left to drop comes out of the closure: dropping a future runs its destructors, which may
        // spawn or end foreground tasks.
        let left = FOREGROUND.with_borrow_mut(|r| {
            if poll.is_ready() {
                r.remove(dispatcher, id);
                return Some(future);
            }
            match r.tasks.get_mut(&dispatcher).and_then(|tasks| tasks.get_mut(&id)) {
                Some(slot) => {
                    *slot = Some(Box::new(future));
                    None
                }
                // The dispatcher shut down during the poll.
                None => Some(future),
            }
        });
        // Outside the borrow (on this thread, where it belongs).
        drop(left);
        poll
    }
}

impl<F> Drop for ForegroundFuture<F> {
    fn drop(&mut self) {
        // On another thread (a runnable dropped there after the shutdown) the future is not this
        // thread's to drop: `drop_foreground_futures` dropped it on its own thread.
        if std::thread::current().id() != self.thread {
            return;
        }
        let future = FOREGROUND
            .try_with(|r| r.try_borrow_mut().ok().and_then(|mut r| r.remove(self.dispatcher, self.id)))
            .ok()
            .flatten();
        // Outside the borrow.
        drop(future);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::headless::HeadlessDispatcher;
    use std::cell::Cell;

    /// A foreground future that finishes on its first poll (or never) and whose destructor spawns another
    /// foreground task, which records that it ran; the spawned task's own state counts its drop.
    struct SpawnsOnDrop {
        executor: ForegroundExecutor,
        ready: bool,
        ran: Rc<Cell<bool>>,
        dropped: Rc<Cell<bool>>,
    }

    impl Future for SpawnsOnDrop {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            if self.ready { Poll::Ready(()) } else { Poll::Pending }
        }
    }

    impl Drop for SpawnsOnDrop {
        fn drop(&mut self) {
            let (ran, dropped) = (self.ran.clone(), DropFlag(self.dropped.clone()));
            self.executor
                .spawn(async move {
                    let _dropped = dropped;
                    ran.set(true);
                })
                .detach();
        }
    }

    struct DropFlag(Rc<Cell<bool>>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    /// The dispatcher, the future, and the flags "the spawned task ran" and "its state was dropped".
    type Setup = (Arc<HeadlessDispatcher>, SpawnsOnDrop, Rc<Cell<bool>>, Rc<Cell<bool>>);

    fn setup(ready: bool) -> Setup {
        let dispatcher = Arc::new(HeadlessDispatcher::new());
        let (ran, dropped) = (Rc::new(Cell::new(false)), Rc::new(Cell::new(false)));
        let executor = ForegroundExecutor::new(dispatcher.clone());
        let future = SpawnsOnDrop { executor, ready, ran: ran.clone(), dropped: dropped.clone() };
        (dispatcher, future, ran, dropped)
    }

    /// A finished future is dropped outside the registry's borrow: its destructor may spawn a foreground
    /// task, which then runs (it used to be dropped under the borrow, and the spawn panicked).
    #[test]
    fn a_future_that_spawns_when_dropped_after_finishing_does_not_panic() {
        let (dispatcher, future, ran, dropped) = setup(true);
        let executor = ForegroundExecutor::new(dispatcher.clone());
        executor.spawn(future).detach();
        dispatcher.run_until_idle();
        assert!(ran.get() && dropped.get(), "the task its destructor spawned ran");
    }

    /// The same when the task is cancelled (its `Task` dropped) while it waits.
    #[test]
    fn a_future_that_spawns_when_dropped_on_cancel_does_not_panic() {
        let (dispatcher, future, ran, dropped) = setup(false);
        let executor = ForegroundExecutor::new(dispatcher.clone());
        let task = executor.spawn(future);
        dispatcher.run_until_idle();
        drop(task);
        dispatcher.run_until_idle();
        assert!(ran.get() && dropped.get(), "the task its destructor spawned ran");
    }

    /// And when the dispatcher shuts down while it waits: the task its destructor spawns is dropped, never
    /// run, and nothing is left in the registry.
    #[test]
    fn a_future_that_spawns_when_dropped_at_shut_down_does_not_panic() {
        let (dispatcher, future, ran, dropped) = setup(false);
        let executor = ForegroundExecutor::new(dispatcher.clone());
        executor.spawn(future).detach();
        dispatcher.run_until_idle();
        dispatcher.shut_down();
        assert!(!ran.get() && dropped.get(), "the task its destructor spawned was dropped, not run");
        let key = dispatcher_key(&*dispatcher);
        assert!(FOREGROUND.with_borrow(|r| !r.tasks.contains_key(&key)), "the registry holds nothing of it");
    }
}
