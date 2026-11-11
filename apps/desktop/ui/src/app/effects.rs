//! The effect queue: what an update schedules for its end, and the flush that runs it in GPUI's order.
//! Owner tag: `"runtime"`.
//!
//! `notify`, `emit`, `defer`, global changes and window refreshes are queued while an update runs and
//! flushed when the outermost update ends: effects in FIFO order (each may queue more), with released
//! entities and focus handles dropped before each one (as GPUI), then windows whose state changed are marked
//! dirty. **Any `notify` marks every window dirty** (one root view plus a few child views; no per-view
//! caching to preserve), and a dirty window asks its platform window for a frame (`request_frame`) instead of
//! drawing in the flush.

use std::any::{Any, TypeId};

use crate::{App, EntityId};

/// One queued effect.
pub(crate) enum Effect {
    /// `cx.notify()` on an entity: run its observers, mark every window dirty.
    Notify {
        /// The entity that changed.
        emitter: EntityId,
    },
    /// `cx.emit(event)`: run the emitter's subscribers for that event type.
    Emit {
        /// The emitting entity.
        emitter: EntityId,
        /// The event's type.
        event_type: TypeId,
        /// The event.
        event: Box<dyn Any>,
    },
    /// `refresh_windows()`: mark every window dirty.
    RefreshWindows,
    /// A global was set or updated: run its observers.
    NotifyGlobalObservers {
        /// The global's type.
        global_type: TypeId,
    },
    /// `cx.defer(f)`.
    Defer {
        /// The deferred call.
        callback: Box<dyn FnOnce(&mut App) + 'static>,
    },
}

impl App {
    /// Queues an effect; flushed when the outermost update ends.
    pub(crate) fn push_effect(&mut self, effect: Effect) {
        match &effect {
            Effect::NotifyGlobalObservers { global_type } => {
                // One per global per flush (GPUI's `pending_global_notifications`); a global change also
                // redraws every window, as everything reads the theme.
                self.windows_need_refresh = true;
                if !self.pending_global_notifications.insert(*global_type) {
                    return;
                }
            }
            Effect::RefreshWindows => self.windows_need_refresh = true,
            _ => {}
        }
        self.pending_effects.push_back(effect);
    }

    /// Runs every queued effect (and those they queue), releases dropped entities, marks windows dirty.
    pub(crate) fn flush_effects(&mut self) {
        loop {
            self.release_dropped_entities();
            self.release_dropped_focus_handles();
            if let Some(effect) = self.pending_effects.pop_front() {
                match effect {
                    Effect::Notify { emitter } => self.apply_notify_effect(emitter),
                    Effect::Emit { emitter, event_type, event } => {
                        self.apply_emit_effect(emitter, event_type, event)
                    }
                    Effect::RefreshWindows => self.windows_need_refresh = true,
                    Effect::NotifyGlobalObservers { global_type } => {
                        self.pending_global_notifications.remove(&global_type);
                        self.global_observers.clone().retain(&global_type, |observer| observer(self));
                    }
                    Effect::Defer { callback } => callback(self),
                }
            } else {
                if std::mem::take(&mut self.windows_need_refresh) {
                    self.mark_all_windows_dirty();
                }
                if self.pending_effects.is_empty() {
                    break;
                }
            }
        }
    }

    /// Runs the observers of `emitter` and marks every window dirty.
    pub(crate) fn apply_notify_effect(&mut self, emitter: EntityId) {
        self.pending_notifications.remove(&emitter);
        self.observers.clone().retain(&emitter, |handler| handler(self));
    }

    /// Runs the subscribers of `emitter` for `event_type`.
    pub(crate) fn apply_emit_effect(&mut self, emitter: EntityId, event_type: TypeId, event: Box<dyn Any>) {
        self.event_listeners.clone().retain(&emitter, |(stored_type, handler)| {
            if *stored_type == event_type { handler(event.as_ref(), self) } else { true }
        });
    }

    /// Drops the state of entities whose last handle went away, and their subscriptions.
    pub(crate) fn release_dropped_entities(&mut self) {
        loop {
            let dropped = self.entities.take_dropped();
            if dropped.is_empty() {
                break;
            }
            for (entity_id, state) in dropped {
                drop(self.observers.remove(&entity_id));
                drop(self.event_listeners.remove(&entity_id));
                // Dropping the state may drop the last handles of other entities: the loop takes them next.
                drop(state);
            }
        }
    }

    /// Forgets focus handles whose last strong handle dropped, blurring any window focused on one.
    pub(crate) fn release_dropped_focus_handles(&mut self) {
        let released: Vec<crate::FocusId> = {
            let mut handles = self.focus_handles.handles.write();
            let released = handles
                .iter()
                .filter(|(_, focus)| focus.ref_count.load(std::sync::atomic::Ordering::SeqCst) == 0)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            for id in &released {
                handles.remove(id);
            }
            released
        };
        if released.is_empty() {
            return;
        }
        for window in self.windows.values_mut().flatten() {
            if window.focus.is_some_and(|focus| released.contains(&focus)) {
                window.blur();
            }
        }
    }

    /// Marks every window dirty (each one calls `request_frame` once).
    pub(crate) fn mark_all_windows_dirty(&mut self) {
        // A window leased out by an update is back in its slot by the time the outermost update flushes.
        for window in self.windows.values_mut().flatten() {
            window.refresh();
        }
    }
}
