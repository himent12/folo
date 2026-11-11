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
    /// If a callback panics, the callbacks are put back as they were and the key is no longer marked.
    pub(crate) fn retain(&self, emitter: &EmitterKey, mut f: impl FnMut(&mut Callback) -> bool) {
        let Some(subscribers) = ({
            let mut lock = self.0.borrow_mut();
            let taken = lock.subscribers.get_mut(emitter).and_then(|subscribers| subscribers.take());
            if taken.is_some() {
                lock.dispatching.push(emitter.clone());
            }
            taken
        }) else {
            return;
        };

        let mut dispatch = Dispatch { set: &self.0, emitter, subscribers };
        dispatch
            .subscribers
            .retain(|_, subscriber| if subscriber.active.get() { f(&mut subscriber.callback) } else { true });
    }

    /// Whether `emitter` has any callback (tests).
    #[cfg(test)]
    pub(crate) fn has(&self, emitter: &EmitterKey) -> bool {
        self.0.borrow().subscribers.contains_key(emitter)
    }
}

/// One `retain` of one key: the callbacks it took out. Dropping it (at the end, or while a callback's panic
/// unwinds) puts them back, unmarks the key and removes the ones dropped meanwhile.
struct Dispatch<'a, EmitterKey: Ord + Clone, Callback> {
    set: &'a RefCell<SubscriberSetState<EmitterKey, Callback>>,
    emitter: &'a EmitterKey,
    subscribers: BTreeMap<usize, Subscriber<Callback>>,
}

impl<EmitterKey: Ord + Clone, Callback> Drop for Dispatch<'_, EmitterKey, Callback> {
    fn drop(&mut self) {
        let mut subscribers = mem::take(&mut self.subscribers);
        let emitter = self.emitter;
        // Never borrowed here (callbacks run with the set unborrowed); a panic while unwinding would abort.
        let Ok(mut lock) = self.set.try_borrow_mut() else { return };
        if let Some(ix) = lock.dispatching.iter().rposition(|key| key == emitter) {
            lock.dispatching.remove(ix);
        }
        if let Some(Some(new_subscribers)) = lock.subscribers.remove(emitter) {
            subscribers.extend(new_subscribers);
        }
        // A cancellation this dispatch's callbacks do not hold belongs to an outer dispatch of the same
        // key (the callback was taken out by it): keep it for that one. With no outer dispatch left, it
        // is stale.
        let outer = lock.dispatching.contains(emitter);
        let mut removed = Vec::new();
        for (dropped_emitter, dropped_id) in mem::take(&mut lock.dropped_subscribers) {
            if dropped_emitter != *emitter {
                lock.dropped_subscribers.push((dropped_emitter, dropped_id));
            } else if let Some(subscriber) = subscribers.remove(&dropped_id) {
                removed.push(subscriber);
            } else if outer {
                lock.dropped_subscribers.push((dropped_emitter, dropped_id));
            }
        }
        if !subscribers.is_empty() {
            lock.subscribers.insert(emitter.clone(), Some(subscribers));
        }
        drop(lock);
        drop(removed);
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::{SubscriberSet, Subscription};

    /// The values of `emitter`'s active callbacks, in order.
    fn values(set: &SubscriberSet<u8, u32>, emitter: u8) -> Vec<u32> {
        let mut out = Vec::new();
        set.retain(&emitter, |value| {
            out.push(*value);
            true
        });
        out
    }

    #[test]
    fn a_nested_dispatch_keeps_the_outer_dispatchs_cancellations() {
        let set = SubscriberSet::<u8, u32>::new();
        let (a, activate) = set.insert(0, 10);
        activate();
        let a = RefCell::new(Some(a));
        let b: RefCell<Option<Subscription>> = RefCell::new(None);
        set.retain(&0, |value| {
            if *value == 10 {
                // While A runs: subscribe B, drop A, then dispatch the same emitter again.
                let (sub, activate) = set.insert(0, 20);
                activate();
                *b.borrow_mut() = Some(sub);
                drop(a.borrow_mut().take());
                set.retain(&0, |_| true);
            }
            true
        });
        assert_eq!(values(&set, 0), [20], "A was dropped during the outer dispatch");
        drop(b.take());
        assert!(!set.has(&0));
        assert!(set.0.borrow().dropped_subscribers.is_empty());
    }

    #[test]
    fn a_callback_dropped_during_a_nested_dispatch_is_removed() {
        let set = SubscriberSet::<u8, u32>::new();
        let (a, activate) = set.insert(0, 10);
        activate();
        let b: RefCell<Option<Subscription>> = RefCell::new(None);
        set.retain(&0, |value| {
            if *value == 10 {
                let (sub, activate) = set.insert(0, 20);
                activate();
                *b.borrow_mut() = Some(sub);
                // B is dropped while the nested dispatch runs it.
                set.retain(&0, |value| {
                    if *value == 20 {
                        drop(b.borrow_mut().take());
                    }
                    true
                });
            }
            true
        });
        assert_eq!(values(&set, 0), [10]);
        drop(a);
        assert!(!set.has(&0));
        assert!(set.0.borrow().dropped_subscribers.is_empty());
    }

    #[test]
    fn a_panicking_callback_leaves_no_key_marked_and_keeps_the_callbacks() {
        let set = SubscriberSet::<u8, u32>::new();
        let (a, activate) = set.insert(0, 10);
        activate();
        let (b, activate) = set.insert(0, 20);
        activate();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            set.retain(&0, |value| {
                if *value == 10 {
                    panic!("callback panics");
                }
                true
            })
        }));
        assert!(panicked.is_err());
        assert!(set.0.borrow().dispatching.is_empty());
        assert_eq!(values(&set, 0), [10, 20]);
        drop(a);
        // A drop after the panic is not taken for one during a dispatch.
        assert!(set.0.borrow().dropped_subscribers.is_empty());
        assert_eq!(values(&set, 0), [20]);
        drop(b);
        assert!(!set.has(&0));
    }
}
