//! `AsyncApp`: the app as a future sees it across awaits (a weak reference; every call can fail once the
//! app is gone). Owner tag: `"runtime"`.
#![allow(dead_code)] // skeleton

use std::{future::Future, rc::Weak};

use crate::{
    AnyView, AnyWindowHandle, App, AppCell, AppContext, BackgroundExecutor, Context, Entity, ForegroundExecutor,
    Global, Task, Window,
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
    pub fn update<R>(&self, _f: impl FnOnce(&mut App) -> R) -> anyhow::Result<R> {
        todo!("runtime")
    }

    /// Spawns another future on the UI thread.
    pub fn spawn<AsyncFn, R>(&self, _f: AsyncFn) -> Task<R>
    where
        AsyncFn: AsyncFnOnce(&mut AsyncApp) -> R + 'static,
        R: 'static,
    {
        todo!("runtime")
    }
}

impl AppContext for AsyncApp {
    type Result<T> = anyhow::Result<T>;

    fn new<T: 'static>(&mut self, _build_entity: impl FnOnce(&mut Context<T>) -> T) -> anyhow::Result<Entity<T>> {
        todo!("runtime")
    }

    fn update_entity<T, R>(
        &mut self,
        _handle: &Entity<T>,
        _update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> anyhow::Result<R>
    where
        T: 'static,
    {
        todo!("runtime")
    }

    fn read_entity<T, R>(&self, _handle: &Entity<T>, _read: impl FnOnce(&T, &App) -> R) -> anyhow::Result<R>
    where
        T: 'static,
    {
        todo!("runtime")
    }

    fn update_window<T, F>(&mut self, _window: AnyWindowHandle, _f: F) -> anyhow::Result<T>
    where
        F: FnOnce(AnyView, &mut Window, &mut App) -> T,
    {
        todo!("runtime")
    }

    fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R>
    where
        R: Send + 'static,
    {
        self.background_executor.spawn(future)
    }

    fn read_global<G, R>(&self, _callback: impl FnOnce(&G, &App) -> R) -> anyhow::Result<R>
    where
        G: Global,
    {
        todo!("runtime")
    }
}
