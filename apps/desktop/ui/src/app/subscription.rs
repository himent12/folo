//! `SubscriberSet` (every kind of observer and listener the app registers, keyed by emitter) and the
//! `Subscription` guard. Owner tag: `"runtime"`.
//!
//! A subscriber is inserted inactive and activated after the current effect flush (GPUI's rule: a callback
//! registered during an effect does not see that effect). Dropping a `Subscription` removes its callback;
//! `detach` keeps it for the life of the emitter.
use std::{cell::RefCell, collections::BTreeMap, fmt, mem, rc::Rc};

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
    /// Keys whose callbacks `retain` has taken out and is running (innermost last).
    pub(crate) dispatching: Vec<EmitterKey>,
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
            dispatching: Vec::new(),
            next_subscriber_id: 0,
        })))
    }

    /// Registers `callback` for `emitter_key`. The returned closure activates it; the subscription removes it.
    pub(crate) fn insert(
        &self,
        emitter_key: EmitterKey,
        callback: Callback,
    ) -> (Subscription, Box<dyn FnOnce()>) {
        let active = Rc::new(std::cell::Cell::new(false));
        let mut lock = self.0.borrow_mut();
        let subscriber_id = lock.next_subscriber_id;
        lock.next_subscriber_id += 1;
        lock.subscribers
            .entry(emitter_key.clone())
            .or_default()
            .get_or_insert_with(BTreeMap::new)
            .insert(subscriber_id, Subscriber { active: active.clone(), callback });
        drop(lock);

        let this = Rc::downgrade(&self.0);
        let subscription = Subscription::new(move || {
            let Some(this) = this.upgrade() else { return };
            // Take the callback out under the borrow and drop it after: dropping a callback can drop
            // handles whose release touches this set again.
            let removed = {
                let mut lock = this.borrow_mut();
                let removed = match lock.subscribers.get_mut(&emitter_key) {
                    Some(Some(subscribers)) => {
                        let removed = subscribers.remove(&subscriber_id);
                        if subscribers.is_empty() {
                            lock.subscribers.remove(&emitter_key);
                        }
                        removed
                    }
                    _ => None,
                };
                // Not found while the key's callbacks are being run: it is one of them (the slot holds
                // only callbacks added meanwhile, if any). Remove it when they are put back.
                if removed.is_none() && lock.dispatching.contains(&emitter_key) {
                    lock.dropped_subscribers.push((emitter_key, subscriber_id));
                }
                removed
            };
            drop(removed);
        });
        (subscription, Box::new(move || active.set(true)))
    }

    /// Removes every callback of `emitter`, returning them (entity release). Inactive ones are dropped.
    pub(crate) fn remove(&self, emitter: &EmitterKey) -> Vec<Callback> {
        let subscribers = self.0.borrow_mut().subscribers.remove(emitter);
        subscribers
            .flatten()
            .map(|subscribers| {
                subscribers
                    .into_values()
                    .filter(|subscriber| subscriber.active.get())
                    .map(|subscriber| subscriber.callback)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Calls each active callback of `emitter`; a callback returning `false` is removed. Callbacks added
    /// for `emitter` meanwhile are kept (inactive until activated); ones dropped meanwhile are removed.
    pub(crate) fn retain(&self, emitter: &EmitterKey, mut f: impl FnMut(&mut Callback) -> bool) {
        let Some(mut subscribers) = ({
            let mut lock = self.0.borrow_mut();
            let taken = lock.subscribers.get_mut(emitter).and_then(|subscribers| subscribers.take());
            if taken.is_some() {
                lock.dispatching.push(emitter.clone());
            }
            taken
        }) else {
            return;
        };

        subscribers
            .retain(|_, subscriber| if subscriber.active.get() { f(&mut subscriber.callback) } else { true });

        let mut lock = self.0.borrow_mut();
        if let Some(ix) = lock.dispatching.iter().rposition(|key| key == emitter) {
            lock.dispatching.remove(ix);
        }
        if let Some(Some(new_subscribers)) = lock.subscribers.remove(emitter) {
            subscribers.extend(new_subscribers);
        }
        let mut removed = Vec::new();
        for (dropped_emitter, dropped_id) in mem::take(&mut lock.dropped_subscribers) {
            if dropped_emitter == *emitter {
                removed.extend(subscribers.remove(&dropped_id));
            } else {
                lock.dropped_subscribers.push((dropped_emitter, dropped_id));
            }
        }
        if !subscribers.is_empty() {
            lock.subscribers.insert(emitter.clone(), Some(subscribers));
        }
        drop(lock);
        drop(removed);
    }

    /// Whether `emitter` has any callback (tests).
    #[cfg(test)]
    pub(crate) fn has(&self, emitter: &EmitterKey) -> bool {
        self.0.borrow().subscribers.contains_key(emitter)
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
