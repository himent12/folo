//! `AsyncApp`: the app as a future sees it across awaits (a weak reference; every call can fail once the
//! app is gone). Owner tag: `"runtime"`.

use std::{future::Future, rc::Weak};

use anyhow::anyhow;

use crate::{
    AnyView, AnyWindowHandle, App, AppCell, AppContext, BackgroundExecutor, Context, Entity,
    ForegroundExecutor, Global, Task, Window,
};

/// The app from inside a spawned future.
#[derive(Clone)]
pub struct AsyncApp {
    /// The app, if it still exists.
    pub(crate) app: Weak<AppCell>,
    /// Background threads.
    pub(crate) background_executor: BackgroundExecutor,
    /// The UI thread.
    pub(crate) foreground_executor: ForegroundExecutor,
}

impl AsyncApp {
    /// The background executor (`cx.background_executor().spawn(..)` / `.timer(..)` in the app's tasks).
    pub fn background_executor(&self) -> &BackgroundExecutor {
        &self.background_executor
    }

    /// The UI-thread executor.
    pub fn foreground_executor(&self) -> &ForegroundExecutor {
        &self.foreground_executor
    }

    /// Runs `f` with the app (and flushes its effects); `Err` once the app is gone.
    ///
    /// GPUI panics when the app is already borrowed (a future polled inside an update); this returns
    /// `Err` instead, so a task woken during quit cannot take the process down.
    pub fn update<R>(&self, f: impl FnOnce(&mut App) -> R) -> anyhow::Result<R> {
        let app = self.app.upgrade().ok_or_else(|| anyhow!("app was released"))?;
        let mut app = app.try_borrow_mut().ok_or_else(|| anyhow!("app is already borrowed"))?;
        Ok(app.update(f))
    }

    /// Reads the app; `Err` once it is gone (or mutably borrowed).
    fn read<R>(&self, f: impl FnOnce(&App) -> R) -> anyhow::Result<R> {
        let app = self.app.upgrade().ok_or_else(|| anyhow!("app was released"))?;
        let app = app.try_borrow().ok_or_else(|| anyhow!("app is already borrowed"))?;
        Ok(f(&app))
    }

    /// Spawns another future on the UI thread.
    pub fn spawn<AsyncFn, R>(&self, f: AsyncFn) -> Task<R>
    where
        AsyncFn: AsyncFnOnce(&mut AsyncApp) -> R + 'static,
        R: 'static,
    {
        let mut cx = self.clone();
        self.foreground_executor.spawn(async move { f(&mut cx).await })
    }
}

impl AppContext for AsyncApp {
    type Result<T> = anyhow::Result<T>;

    fn new<T: 'static>(
        &mut self,
        build_entity: impl FnOnce(&mut Context<T>) -> T,
    ) -> anyhow::Result<Entity<T>> {
        self.update(|cx| cx.new(build_entity))
    }

    fn update_entity<T, R>(
        &mut self,
        handle: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> anyhow::Result<R>
    where
        T: 'static,
    {
        self.update(|cx| cx.update_entity(handle, update))
    }

    fn read_entity<T, R>(&self, handle: &Entity<T>, read: impl FnOnce(&T, &App) -> R) -> anyhow::Result<R>
    where
        T: 'static,
    {
        self.read(|cx| cx.read_entity(handle, read))
    }

    fn update_window<T, F>(&mut self, window: AnyWindowHandle, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(AnyView, &mut Window, &mut App) -> T,
    {
        self.update(|cx| cx.update_window(window, f))?
    }

    fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R>
    where
        R: Send + 'static,
    {
        self.background_executor.spawn(future)
    }

    fn read_global<G, R>(&self, callback: impl FnOnce(&G, &App) -> R) -> anyhow::Result<R>
    where
        G: Global,
    {
        self.read(|cx| cx.read_global(callback))
    }
}
