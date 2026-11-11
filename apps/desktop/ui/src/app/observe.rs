//! Observers and listeners registered on `App` itself: `notify`, `observe`, `subscribe`, `observe_global`,
//! the app-level `on_action` and `on_app_quit`. Owner tag: `"runtime"`. `Context<T>` wraps these so the
//! callbacks update the observing entity.
//!
//! As in GPUI, an entity observer or listener is activated by a deferred effect: it does not see the effect
//! cycle it was registered in. App-wide ones (`on_app_quit`, `on_window_closed`) are active at once.

use std::{any::TypeId, rc::Rc};

use crate::{
    Action, App, DispatchPhase, Entity, EntityId, EventEmitter, Global, Subscription,
    app::{Effect, LocalBoxFuture},
};

impl App {
    /// Says `entity_id` changed: its observers run at the flush, and every window redraws.
    pub fn notify(&mut self, entity_id: EntityId) {
        if self.drawing_window.is_none() {
            self.windows_need_refresh = true;
        }
        if self.pending_notifications.insert(entity_id) {
            self.push_effect(Effect::Notify { emitter: entity_id });
        }
    }

    /// Calls `on_notify` whenever `entity` notifies, while the subscription lives.
    pub fn observe<W: 'static>(
        &mut self,
        entity: &Entity<W>,
        mut on_notify: impl FnMut(Entity<W>, &mut App) + 'static,
    ) -> Subscription {
        self.observe_internal(entity, move |entity, cx| {
            on_notify(entity, cx);
            true
        })
    }

    /// `observe` whose callback returns whether to stay subscribed.
    pub(crate) fn observe_internal<W: 'static>(
        &mut self,
        entity: &Entity<W>,
        mut on_notify: impl FnMut(Entity<W>, &mut App) -> bool + 'static,
    ) -> Subscription {
        let handle = entity.downgrade();
        let (subscription, activate) = self.observers.insert(
            entity.entity_id(),
            Box::new(move |cx| match handle.upgrade() {
                Some(entity) => on_notify(entity, cx),
                None => false,
            }),
        );
        self.defer(move |_| activate());
        subscription
    }

    /// Calls `on_event` with each event of type `Evt` that `entity` emits, while the subscription lives.
    pub fn subscribe<T, Evt>(
        &mut self,
        entity: &Entity<T>,
        mut on_event: impl FnMut(Entity<T>, &Evt, &mut App) + 'static,
    ) -> Subscription
    where
        T: 'static + EventEmitter<Evt>,
        Evt: 'static,
    {
        self.subscribe_internal(entity, move |entity, event, cx| {
            on_event(entity, event, cx);
            true
        })
    }

    /// `subscribe` whose callback returns whether to stay subscribed.
    pub(crate) fn subscribe_internal<T, Evt>(
        &mut self,
        entity: &Entity<T>,
        mut on_event: impl FnMut(Entity<T>, &Evt, &mut App) -> bool + 'static,
    ) -> Subscription
    where
        T: 'static + EventEmitter<Evt>,
        Evt: 'static,
    {
        let handle = entity.downgrade();
        let (subscription, activate) = self.event_listeners.insert(
            entity.entity_id(),
            (
                TypeId::of::<Evt>(),
                Box::new(move |event, cx| {
                    let event: &Evt = event.downcast_ref().expect("event type");
                    match handle.upgrade() {
                        Some(entity) => on_event(entity, event, cx),
                        None => false,
                    }
                }),
            ),
        );
        self.defer(move |_| activate());
        subscription
    }

    /// Calls `f` whenever global `G` is set or updated, while the subscription lives.
    pub fn observe_global<G: Global>(&mut self, mut f: impl FnMut(&mut App) + 'static) -> Subscription {
        self.observe_global_internal::<G>(move |cx| {
            f(cx);
            true
        })
    }

    /// `observe_global` whose callback returns whether to stay subscribed.
    pub(crate) fn observe_global_internal<G: Global>(
        &mut self,
        f: impl FnMut(&mut App) -> bool + 'static,
    ) -> Subscription {
        let (subscription, activate) = self.global_observers.insert(TypeId::of::<G>(), Box::new(f));
        self.defer(move |_| activate());
        subscription
    }

    /// Registers an app-level handler for action `A`: it runs when no element on the dispatch path handled
    /// the action (the window calls `dispatch_global_action`), as GPUI's `App::on_action`.
    pub fn on_action<A: Action>(&mut self, listener: impl Fn(&A, &mut App) + 'static) {
        self.global_action_listeners.entry(TypeId::of::<A>()).or_default().push(Rc::new(
            move |action, phase, cx| {
                if phase == DispatchPhase::Bubble {
                    listener(action.downcast_ref().expect("action type"), cx)
                }
            },
        ));
    }

    /// Runs the app-level handlers of `action` for one phase, as GPUI's action dispatch does: the window
    /// calls `Capture` first (before its own capture listeners; `propagate_event` starts `true`), then its
    /// capture and bubble listeners along the path, then `Bubble` here last, where each handler starts
    /// with `propagate_event = false` (a handled action stops unless the handler calls `cx.propagate()`).
    /// Afterwards `propagate_event` says whether the action is still unhandled.
    #[allow(dead_code)] // seam: the window's action dispatch (window package) calls it
    pub(crate) fn dispatch_global_action(&mut self, action: &dyn Action, phase: DispatchPhase) {
        let type_id = action.as_any().type_id();
        if phase == DispatchPhase::Capture {
            self.propagate_event = true;
        }
        let Some(listeners) = self.global_action_listeners.get(&type_id).cloned() else { return };
        match phase {
            DispatchPhase::Capture => {
                for listener in &listeners {
                    listener(action.as_any(), phase, self);
                    if !self.propagate_event {
                        break;
                    }
                }
            }
            DispatchPhase::Bubble => {
                for listener in listeners.iter().rev() {
                    self.propagate_event = false;
                    listener(action.as_any(), phase, self);
                    if !self.propagate_event {
                        break;
                    }
                }
            }
        }
    }

    /// Registers `on_quit` to run when the app quits; the quit waits briefly for the returned future.
    pub fn on_app_quit<Fut>(&self, mut on_quit: impl FnMut(&mut App) -> Fut + 'static) -> Subscription
    where
        Fut: 'static + Future<Output = ()>,
    {
        let (subscription, activate) =
            self.quit_observers.insert((), Box::new(move |cx| Box::pin(on_quit(cx)) as LocalBoxFuture));
        activate();
        subscription
    }
}
