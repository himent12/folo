//! Entities: the `EntityMap` that owns every entity's state, and the handles `Entity<T>`, `WeakEntity<T>`,
//! `AnyEntity`, `AnyWeakEntity`. Owner tag: `"runtime"`.
//!
//! The map owns one `Box<dyn Any>` per `EntityId`. `update` leases the state out of the map for the call
//! (GPUI's pattern: the slot is emptied, the closure gets `&mut T` and `&mut Context<T>`, the state goes back),
//! so `Context<T>` can deref to `&mut App` while `T` is borrowed. Strong handles share one ref count per entity
//! (`EntityRefCounts`); when it reaches zero the entity id is queued in `dropped_entity_ids` and the state is
//! released at the end of the next effect flush.
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst},
    },
};

use parking_lot::RwLock;

use crate::{App, AppContext, Context};

/// A process-unique entity id (never reused).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub(crate) u64);

impl EntityId {
    /// The id as a number (stable for the process).
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strong counts of live entities, shared by every handle (handles are `Send`, as GPUI's are).
pub(crate) struct EntityRefCounts {
    /// Strong handle count per live entity.
    pub(crate) counts: HashMap<EntityId, AtomicUsize>,
    /// Entities whose count reached zero, released at the next flush.
    pub(crate) dropped_entity_ids: Vec<EntityId>,
}

/// Owns the state of every entity.
pub(crate) struct EntityMap {
    /// The state, `None` while leased out by an `update`.
    pub(crate) entities: HashMap<EntityId, Option<Box<dyn Any>>>,
    /// Shared with every handle.
    pub(crate) ref_counts: Arc<RwLock<EntityRefCounts>>,
    /// The next id to hand out.
    pub(crate) next_id: AtomicU64,
}

/// An entity's state taken out of the map for the duration of an update.
pub(crate) struct Lease<T> {
    /// The leased state (put back by `EntityMap::end_lease`).
    pub(crate) entity: Option<Box<dyn Any>>,
    /// Which entity.
    pub(crate) id: EntityId,
    _marker: PhantomData<T>,
}

impl EntityMap {
    /// An empty map.
    pub(crate) fn new() -> Self {
        EntityMap {
            entities: HashMap::new(),
            ref_counts: Arc::new(RwLock::new(EntityRefCounts {
                counts: HashMap::new(),
                dropped_entity_ids: Vec::new(),
            })),
            next_id: AtomicU64::new(1),
        }
    }

    /// Reserves an id and a strong handle for an entity being built.
    pub(crate) fn reserve<T: 'static>(&self) -> Entity<T> {
        let entity_id = EntityId(self.next_id.fetch_add(1, SeqCst));
        self.ref_counts.write().counts.insert(entity_id, AtomicUsize::new(1));
        Entity {
            any_entity: AnyEntity {
                entity_id,
                entity_type: TypeId::of::<T>(),
                entity_map: Arc::downgrade(&self.ref_counts),
            },
            entity_type: PhantomData,
        }
    }

    /// Stores the built state of a reserved entity.
    pub(crate) fn insert<T: 'static>(&mut self, entity: &Entity<T>, state: T) {
        self.entities.insert(entity.entity_id(), Some(Box::new(state)));
    }

    /// Takes the state out for an update. Panics when it is already leased (a re-entrant update).
    #[track_caller]
    pub(crate) fn lease<T: 'static>(&mut self, entity: &Entity<T>) -> Lease<T> {
        let id = entity.entity_id();
        let state = match self.entities.get_mut(&id) {
            Some(slot) => slot.take().unwrap_or_else(|| double_lease_panic::<T>("update")),
            None => panic!("{} {id} was released or is still being built", std::any::type_name::<T>()),
        };
        Lease { entity: Some(state), id, _marker: PhantomData }
    }

    /// Puts leased state back.
    pub(crate) fn end_lease<T>(&mut self, mut lease: Lease<T>) {
        // A slot removed meanwhile (its last handle dropped during the update) drops the state instead.
        if let Some(slot) = self.entities.get_mut(&lease.id) {
            *slot = lease.entity.take();
        }
    }

    /// Reads an entity's state. Panics when it is leased out.
    #[track_caller]
    pub(crate) fn read<T: 'static>(&self, entity: &Entity<T>) -> &T {
        let id = entity.entity_id();
        match self.entities.get(&id) {
            Some(Some(state)) => state.downcast_ref::<T>().expect("entity state type"),
            Some(None) => double_lease_panic::<T>("read"),
            None => panic!("{} {id} was released or is still being built", std::any::type_name::<T>()),
        }
    }

    /// Removes and returns the states of entities whose last strong handle dropped.
    pub(crate) fn take_dropped(&mut self) -> Vec<(EntityId, Box<dyn Any>)> {
        let mut ref_counts = self.ref_counts.write();
        let dropped = std::mem::take(&mut ref_counts.dropped_entity_ids);
        dropped
            .into_iter()
            .filter_map(|id| {
                // Queued at zero; a handle can no longer come back (weak upgrades fail at zero).
                let count = ref_counts.counts.remove(&id)?;
                debug_assert_eq!(count.load(SeqCst), 0, "dropped entity {id} still has handles");
                self.entities.remove(&id).flatten().map(|state| (id, state))
            })
            .collect()
    }
}

#[track_caller]
fn double_lease_panic<T>(operation: &str) -> ! {
    panic!(
        "cannot {operation} {} while it is already being updated (a re-entrant update of the same entity)",
        std::any::type_name::<T>()
    )
}

impl<T: 'static> Lease<T> {
    /// The leased state.
    pub(crate) fn state_mut(&mut self) -> &mut T {
        self.entity.as_mut().and_then(|state| state.downcast_mut::<T>()).expect("entity state type")
    }
}

/// A strong, type-erased entity handle.
pub struct AnyEntity {
    /// The entity.
    pub(crate) entity_id: EntityId,
    /// The state's type.
    pub(crate) entity_type: TypeId,
    /// The shared counts (dropping the handle decrements its count).
    pub(crate) entity_map: Weak<RwLock<EntityRefCounts>>,
}

impl AnyEntity {
    /// The entity id.
    pub fn entity_id(&self) -> EntityId {
        self.entity_id
    }

    /// The state's type id.
    pub fn entity_type(&self) -> TypeId {
        self.entity_type
    }

    /// A weak handle to the same entity.
    pub fn downgrade(&self) -> AnyWeakEntity {
        AnyWeakEntity {
            entity_id: self.entity_id,
            entity_type: self.entity_type,
            entity_ref_counts: self.entity_map.clone(),
        }
    }

    /// The typed handle, if the state is a `T`.
    pub fn downcast<T: 'static>(self) -> Result<Entity<T>, AnyEntity> {
        if TypeId::of::<T>() == self.entity_type {
            Ok(Entity { any_entity: self, entity_type: PhantomData })
        } else {
            Err(self)
        }
    }
}

impl Clone for AnyEntity {
    fn clone(&self) -> Self {
        if let Some(ref_counts) = self.entity_map.upgrade() {
            let ref_counts = ref_counts.read();
            let count = ref_counts.counts.get(&self.entity_id).expect("cloned a released entity handle");
            let previous = count.fetch_add(1, SeqCst);
            assert_ne!(previous, 0, "cloned a released entity handle");
        }
        AnyEntity {
            entity_id: self.entity_id,
            entity_type: self.entity_type,
            entity_map: self.entity_map.clone(),
        }
    }
}

impl Drop for AnyEntity {
    fn drop(&mut self) {
        let Some(ref_counts) = self.entity_map.upgrade() else { return };
        let ref_counts = ref_counts.upgradable_read();
        let Some(count) = ref_counts.counts.get(&self.entity_id) else { return };
        let previous = count.fetch_sub(1, SeqCst);
        assert_ne!(previous, 0, "over-released an entity handle");
        if previous == 1 {
            // The last strong handle: release the state at the next flush.
            let mut ref_counts = parking_lot::RwLockUpgradableReadGuard::upgrade(ref_counts);
            ref_counts.dropped_entity_ids.push(self.entity_id);
        }
    }
}

impl PartialEq for AnyEntity {
    fn eq(&self, other: &Self) -> bool {
        self.entity_id == other.entity_id
    }
}

impl Eq for AnyEntity {}

impl Hash for AnyEntity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entity_id.hash(state)
    }
}

impl fmt::Debug for AnyEntity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyEntity").field("entity_id", &self.entity_id).finish()
    }
}

/// A strong handle to an entity of type `T`.
pub struct Entity<T> {
    /// The erased handle.
    pub(crate) any_entity: AnyEntity,
    pub(crate) entity_type: PhantomData<fn(T) -> T>,
}

impl<T: 'static> Entity<T> {
    /// The entity id.
    pub fn entity_id(&self) -> EntityId {
        self.any_entity.entity_id
    }

    /// A weak handle.
    pub fn downgrade(&self) -> WeakEntity<T> {
        WeakEntity { any_entity: self.any_entity.downgrade(), entity_type: PhantomData }
    }

    /// Erases the type.
    pub fn into_any(self) -> AnyEntity {
        self.any_entity
    }

    /// Reads the state.
    pub fn read<'a>(&self, cx: &'a App) -> &'a T {
        cx.entities.read(self)
    }

    /// Reads the state through a closure.
    pub fn read_with<R, C: AppContext>(&self, cx: &C, f: impl FnOnce(&T, &App) -> R) -> C::Result<R> {
        cx.read_entity(self, f)
    }

    /// Updates the state (the state is leased out of the map for the call).
    pub fn update<R, C: AppContext>(
        &self,
        cx: &mut C,
        update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> C::Result<R> {
        cx.update_entity(self, update)
    }
}

impl<T> Clone for Entity<T> {
    fn clone(&self) -> Self {
        Entity { any_entity: self.any_entity.clone(), entity_type: PhantomData }
    }
}

impl<T> PartialEq for Entity<T> {
    fn eq(&self, other: &Self) -> bool {
        self.any_entity == other.any_entity
    }
}

impl<T> Eq for Entity<T> {}

impl<T> Hash for Entity<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.any_entity.hash(state)
    }
}

impl<T> fmt::Debug for Entity<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entity").field("entity_id", &self.any_entity.entity_id).finish()
    }
}

/// A weak, type-erased entity handle.
#[derive(Clone)]
pub struct AnyWeakEntity {
    /// The entity.
    pub(crate) entity_id: EntityId,
    /// The state's type.
    pub(crate) entity_type: TypeId,
    /// The shared counts (to upgrade).
    pub(crate) entity_ref_counts: Weak<RwLock<EntityRefCounts>>,
}

impl AnyWeakEntity {
    /// The entity id.
    pub fn entity_id(&self) -> EntityId {
        self.entity_id
    }

    /// A strong handle, if the entity is still alive.
    pub fn upgrade(&self) -> Option<AnyEntity> {
        let ref_counts = self.entity_ref_counts.upgrade()?;
        let counts = ref_counts.read();
        let count = counts.counts.get(&self.entity_id)?;
        // Increment unless it is zero (released, or about to be).
        let mut current = count.load(SeqCst);
        loop {
            if current == 0 {
                return None;
            }
            match count.compare_exchange_weak(current, current + 1, SeqCst, SeqCst) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        drop(counts);
        Some(AnyEntity {
            entity_id: self.entity_id,
            entity_type: self.entity_type,
            entity_map: self.entity_ref_counts.clone(),
        })
    }
}

/// A weak handle to an entity of type `T`: what `cx.spawn` hands its future.
pub struct WeakEntity<T> {
    /// The erased handle.
    pub(crate) any_entity: AnyWeakEntity,
    pub(crate) entity_type: PhantomData<fn(T) -> T>,
}

impl<T: 'static> WeakEntity<T> {
    /// The entity id.
    pub fn entity_id(&self) -> EntityId {
        self.any_entity.entity_id
    }

    /// A strong handle, if the entity is still alive.
    pub fn upgrade(&self) -> Option<Entity<T>> {
        self.any_entity.upgrade().map(|any_entity| Entity { any_entity, entity_type: PhantomData })
    }

    /// Updates the entity if it is alive; `Err` when it was released (or, through `AsyncApp`, when the app
    /// is gone). `Result<C::Result<R>>` flattens to `Result<R>` as in GPUI.
    pub fn update<C, R>(
        &self,
        cx: &mut C,
        update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> anyhow::Result<R>
    where
        C: AppContext,
        anyhow::Result<C::Result<R>>: Flatten<R>,
    {
        Flatten::flatten(
            self.upgrade()
                .ok_or_else(|| anyhow::anyhow!("entity released"))
                .map(|this| cx.update_entity(&this, update)),
        )
    }

    /// Reads the entity if it is alive.
    pub fn read_with<C, R>(&self, cx: &C, read: impl FnOnce(&T, &App) -> R) -> anyhow::Result<R>
    where
        C: AppContext,
        anyhow::Result<C::Result<R>>: Flatten<R>,
    {
        Flatten::flatten(
            self.upgrade()
                .ok_or_else(|| anyhow::anyhow!("entity released"))
                .map(|this| cx.read_entity(&this, read)),
        )
    }
}

impl<T> Clone for WeakEntity<T> {
    fn clone(&self) -> Self {
        WeakEntity { any_entity: self.any_entity.clone(), entity_type: PhantomData }
    }
}

/// Flattens `Result<Result<T>>` and `Result<T>` to `Result<T>`, so `WeakEntity::update` has one return type
/// for every context (GPUI's `Flatten`).
pub trait Flatten<T> {
    /// Flattens.
    fn flatten(self) -> anyhow::Result<T>;
}

impl<T> Flatten<T> for anyhow::Result<anyhow::Result<T>> {
    fn flatten(self) -> anyhow::Result<T> {
        self?
    }
}

impl<T> Flatten<T> for anyhow::Result<T> {
    fn flatten(self) -> anyhow::Result<T> {
        self
    }
}
