//! Executors: `BackgroundExecutor` (a thread pool through `PlatformDispatcher::dispatch`, plus timers),
//! `ForegroundExecutor` (the UI thread, `dispatch_on_main_thread`), and `Task<T>`. Owner tag: `"runtime"`.
//!
//! Tasks are `async-task` runnables: waking one schedules it on its dispatcher queue, so an idle app has no
//! timer and no thread awake. Dropping a `Task` cancels it; `detach` lets it run to the end.
//!
//! A foreground task is an `async_task::spawn_local` task: it is polled only on the thread that spawned it
//! (the platform runs main-thread runnables there). A task woken after the app is gone just runs: its
//! `AsyncApp` calls return `Err`, nothing panics.

use std::{
    fmt,
    future::Future,
    marker::PhantomData,
    pin::{Pin, pin},
    rc::Rc,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread::Thread,
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
        let dispatcher = self.dispatcher.clone();
        let (runnable, task) =
            async_task::spawn_local(future, move |runnable| dispatcher.dispatch_on_main_thread(runnable));
        runnable.schedule();
        Task(TaskState::Spawned(task))
    }

    /// The dispatcher's clock.
    pub fn now(&self) -> Instant {
        self.dispatcher.now()
    }
}
