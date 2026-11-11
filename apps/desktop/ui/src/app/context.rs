//! `Context<T>`: the app plus the entity being updated, and `EventEmitter`. Owner tag: `"runtime"`.
//!
//! A `Context<T>` exists only inside `Entity<T>::update` / `AppContext::new` (the state is leased out of the
//! entity map meanwhile); it derefs to `App`.
#![allow(dead_code)] // skeleton

use std::{
    future::Future,
    ops::{Deref, DerefMut},
};

use crate::{AnyWindowHandle, App, AppContext, AsyncApp, Entity, EntityId, Global, Subscription, Task, WeakEntity, Window};

/// Declares that an entity of this type emits events of type `E` (`cx.emit`, `cx.subscribe`).
pub trait EventEmitter<E: 'static>: 'static {}

/// The app while one entity of type `T` is being updated.
pub struct Context<'a, T> {
    /// The app.
    pub(crate) app: &'a mut App,
    /// The entity being updated.
    pub(crate) entity_state: WeakEntity<T>,
}

impl<T> Deref for Context<'_, T> {
    type Target = App;

    fn deref(&self) -> &App {
        self.app
    }
}

impl<T> DerefMut for Context<'_, T> {
    fn deref_mut(&mut self) -> &mut App {
        self.app
    }
}

impl<'a, T: 'static> Context<'a, T> {
    /// A context for `entity_state` over `app`.
    pub(crate) fn new_context(app: &'a mut App, entity_state: WeakEntity<T>) -> Self {
        Context { app, entity_state }
    }

    /// The entity's id.
    pub fn entity_id(&self) -> EntityId {
        self.entity_state.entity_id()
    }

    /// A strong handle to the entity (panics while it is being built, as GPUI).
    pub fn entity(&self) -> Entity<T> {
        self.weak_entity().upgrade().expect("the entity is being built")
    }

    /// A weak handle to the entity.
    pub fn weak_entity(&self) -> WeakEntity<T> {
        self.entity_state.clone()
    }

    /// Says the entity changed: its observers run at the flush and every window redraws.
    pub fn notify(&mut self) {
        let entity_id = self.entity_id();
        self.app.push_effect(crate::app::Effect::Notify { emitter: entity_id });
    }

    /// Emits `event` to the entity's subscribers (at the flush).
    pub fn emit<Evt>(&mut self, _event: Evt)
    where
        T: EventEmitter<Evt>,
        Evt: 'static,
    {
        todo!("runtime")
    }

    /// Calls `on_event` with each event `entity` emits, while the subscription lives.
    pub fn subscribe<T2, Evt>(
        &mut self,
        _entity: &Entity<T2>,
        _on_event: impl FnMut(&mut T, Entity<T2>, &Evt, &mut Context<T>) + 'static,
    ) -> Subscription
    where
        T2: 'static + EventEmitter<Evt>,
        Evt: 'static,
    {
        todo!("runtime")
    }

    /// Calls `on_notify` whenever `entity` notifies, while the subscription lives.
    pub fn observe<W: 'static>(
        &mut self,
        _entity: &Entity<W>,
        _on_notify: impl FnMut(&mut T, Entity<W>, &mut Context<T>) + 'static,
    ) -> Subscription {
        todo!("runtime")
    }

    /// Calls `f` whenever global `G` changes, while the subscription lives.
    pub fn observe_global<G: Global>(&mut self, _f: impl FnMut(&mut T, &mut Context<T>) + 'static) -> Subscription {
        todo!("runtime")
    }

    /// Calls `on_quit` when the app quits; the quit waits briefly for the returned future.
    pub fn on_app_quit<Fut>(&self, _on_quit: impl FnMut(&mut T, &mut Context<T>) -> Fut + 'static) -> Subscription
    where
        Fut: 'static + Future<Output = ()>,
    {
        todo!("runtime")
    }

    /// Spawns `f` on the UI thread with a weak handle to this entity (GPUI 0.2.2's closure shape:
    /// `cx.spawn(async move |this, cx| ..)`).
    pub fn spawn<AsyncFn, R>(&self, _f: AsyncFn) -> Task<R>
    where
        AsyncFn: AsyncFnOnce(WeakEntity<T>, &mut AsyncApp) -> R + 'static,
        R: 'static,
    {
        todo!("runtime")
    }

    /// Wraps an entity method as an event listener (`.on_click(cx.listener(Self::clicked))`): the listener
    /// updates this entity, if it is alive, when called.
    pub fn listener<E: ?Sized>(
        &self,
        f: impl Fn(&mut T, &E, &mut Window, &mut Context<T>) + 'static,
    ) -> impl Fn(&E, &mut Window, &mut App) + 'static {
        let view = self.weak_entity();
        move |e: &E, window: &mut Window, cx: &mut App| {
            view.update(cx, |view, cx| f(view, e, window, cx)).ok();
        }
    }

    /// Like `listener`, for callbacks that take their argument by value and return a value
    /// (`uniform_list`'s row builder). Panics if the entity was released, as GPUI.
    pub fn processor<E, R>(
        &self,
        f: impl Fn(&mut T, E, &mut Window, &mut Context<T>) -> R + 'static,
    ) -> impl Fn(E, &mut Window, &mut App) -> R + 'static {
        let view = self.entity();
        move |e: E, window: &mut Window, cx: &mut App| view.update(cx, |view, cx| f(view, e, window, cx))
    }
}

impl<T> AppContext for Context<'_, T> {
    type Result<U> = U;

    fn new<U: 'static>(&mut self, build_entity: impl FnOnce(&mut Context<U>) -> U) -> Entity<U> {
        self.app.new(build_entity)
    }

    fn update_entity<U, R>(&mut self, handle: &Entity<U>, update: impl FnOnce(&mut U, &mut Context<U>) -> R) -> R
    where
        U: 'static,
    {
        self.app.update_entity(handle, update)
    }

    fn read_entity<U, R>(&self, handle: &Entity<U>, read: impl FnOnce(&U, &App) -> R) -> R
    where
        U: 'static,
    {
        self.app.read_entity(handle, read)
    }

    fn update_window<R, F>(&mut self, window: AnyWindowHandle, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(crate::AnyView, &mut Window, &mut App) -> R,
    {
        self.app.update_window(window, f)
    }

    fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> Task<R>
    where
        R: Send + 'static,
    {
        self.app.background_spawn(future)
    }

    fn read_global<G, R>(&self, callback: impl FnOnce(&G, &App) -> R) -> R
    where
        G: Global,
    {
        self.app.read_global(callback)
    }
}
