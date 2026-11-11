//! The effect queue: what an update schedules for its end, and the flush that runs it in GPUI's order.
//! Owner tag: `"runtime"`.
//!
//! `notify`, `emit`, `defer`, global changes and window refreshes are queued while an update runs and
//! flushed when the outermost update ends: effects in FIFO order (each may queue more), then released
//! entities are dropped, then windows whose state changed are marked dirty. **Any `notify` marks every
//! window dirty** (one root view plus a few child views; no per-view caching to preserve), and a dirty
//! window asks its platform window for a frame (`request_frame`) instead of drawing in the flush.
#![allow(dead_code)] // skeleton

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
        self.pending_effects.push_back(effect);
    }

    /// Runs every queued effect (and those they queue), releases dropped entities, marks windows dirty.
    pub(crate) fn flush_effects(&mut self) {
        todo!("runtime")
    }

    /// Runs the observers of `emitter` and marks every window dirty.
    pub(crate) fn apply_notify_effect(&mut self, _emitter: EntityId) {
        todo!("runtime")
    }

    /// Runs the subscribers of `emitter` for `event_type`.
    pub(crate) fn apply_emit_effect(
        &mut self,
        _emitter: EntityId,
        _event_type: TypeId,
        _event: Box<dyn Any>,
    ) {
        todo!("runtime")
    }

    /// Drops the state of entities whose last handle went away, and their subscriptions.
    pub(crate) fn release_dropped_entities(&mut self) {
        todo!("runtime")
    }

    /// Marks every window dirty (each one calls `request_frame` once).
    pub(crate) fn mark_all_windows_dirty(&mut self) {
        todo!("runtime")
    }
}
