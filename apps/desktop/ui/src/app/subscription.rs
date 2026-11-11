//! `SubscriberSet` (every kind of observer and listener the app registers, keyed by emitter) and the
//! `Subscription` guard. Owner tag: `"runtime"`.
//!
//! A subscriber is inserted inactive and activated after the current effect flush (GPUI's rule: a callback
//! registered during an effect does not see that effect). Dropping a `Subscription` removes its callback;
//! `detach` keeps it for the life of the emitter.
#![allow(dead_code)] // skeleton

use std::{cell::RefCell, collections::BTreeMap, fmt, rc::Rc};

/// Callbacks of one kind, keyed by emitter (`EntityId`, `TypeId`, or `()` for app-wide ones).
pub(crate) struct SubscriberSet<EmitterKey, Callback>(Rc<RefCell<SubscriberSetState<EmitterKey, Callback>>>);

impl<EmitterKey, Callback> Clone for SubscriberSet<EmitterKey, Callback> {
    fn clone(&self) -> Self {
        SubscriberSet(self.0.clone())
    }
}

/// The shared state of a `SubscriberSet`.
pub(crate) struct SubscriberSetState<EmitterKey, Callback> {
    /// Subscribers per emitter, in subscription order; `None` while a key's callbacks are being run.
    pub(crate) subscribers: BTreeMap<EmitterKey, Option<BTreeMap<usize, Subscriber<Callback>>>>,
    /// Subscriptions dropped while their key was being run.
    pub(crate) dropped_subscribers: Vec<(EmitterKey, usize)>,
    /// The next subscriber id.
    pub(crate) next_subscriber_id: usize,
}

/// One registered callback.
pub(crate) struct Subscriber<Callback> {
    /// Activated after the flush in which it was registered.
    pub(crate) active: Rc<std::cell::Cell<bool>>,
    /// The callback.
    pub(crate) callback: Callback,
}

impl<EmitterKey, Callback> SubscriberSet<EmitterKey, Callback>
where
    EmitterKey: 'static + Ord + Clone + fmt::Debug,
    Callback: 'static,
{
    /// An empty set.
    pub(crate) fn new() -> Self {
        SubscriberSet(Rc::new(RefCell::new(SubscriberSetState {
            subscribers: BTreeMap::new(),
            dropped_subscribers: Vec::new(),
            next_subscriber_id: 0,
        })))
    }

    /// Registers `callback` for `emitter_key`. The returned closure activates it; the subscription removes it.
    pub(crate) fn insert(
        &self,
        _emitter_key: EmitterKey,
        _callback: Callback,
    ) -> (Subscription, Box<dyn FnOnce()>) {
        todo!("runtime")
    }

    /// Removes every callback of `emitter`, returning them (entity release).
    pub(crate) fn remove(&self, _emitter: &EmitterKey) -> Vec<Callback> {
        todo!("runtime")
    }

    /// Calls each active callback of `emitter`; a callback returning `false` is removed.
    pub(crate) fn retain(&self, _emitter: &EmitterKey, _f: impl FnMut(&mut Callback) -> bool) {
        todo!("runtime")
    }
}

/// Keeps a callback registered; dropping it unregisters the callback.
#[must_use]
pub struct Subscription {
    /// Unregisters the callback; `None` once detached.
    pub(crate) unsubscribe: Option<Box<dyn FnOnce() + 'static>>,
}

impl Subscription {
    /// A subscription that runs `unsubscribe` when dropped.
    pub fn new(unsubscribe: impl 'static + FnOnce()) -> Self {
        Subscription { unsubscribe: Some(Box::new(unsubscribe)) }
    }

    /// Keeps the callback for the life of its emitter.
    pub fn detach(mut self) {
        self.unsubscribe.take();
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(unsubscribe) = self.unsubscribe.take() {
            unsubscribe();
        }
    }
}

impl fmt::Debug for Subscription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Subscription").finish_non_exhaustive()
    }
}
