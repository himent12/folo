//! Entities: the `EntityMap` that owns every entity's state, and the handles `Entity<T>`, `WeakEntity<T>`,
//! `AnyEntity`, `AnyWeakEntity`. Owner tag: `"runtime"`.
//!
//! The map owns one `Box<dyn Any>` per `EntityId`. `update` leases the state out of the map for the call
//! (GPUI's pattern: the slot is emptied, the closure gets `&mut T` and `&mut Context<T>`, the state goes back),
//! so `Context<T>` can deref to `&mut App` while `T` is borrowed. Strong handles share one ref count per entity
//! (`EntityRefCounts`); when it reaches zero the entity id is queued in `dropped_entity_ids` and the state is
//! released at the end of the next effect flush.
#![allow(dead_code)] // skeleton

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
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
        todo!("runtime")
    }

    /// Stores the built state of a reserved entity.
    pub(crate) fn insert<T: 'static>(&mut self, _entity: &Entity<T>, _state: T) {
        todo!("runtime")
    }

    /// Takes the state out for an update. Panics when it is already leased (a re-entrant update).
    pub(crate) fn lease<T: 'static>(&mut self, _entity: &Entity<T>) -> Lease<T> {
        todo!("runtime")
    }

    /// Puts leased state back.
    pub(crate) fn end_lease<T>(&mut self, _lease: Lease<T>) {
        todo!("runtime")
    }

    /// Reads an entity's state. Panics when it is leased out.
    pub(crate) fn read<T: 'static>(&self, _entity: &Entity<T>) -> &T {
        todo!("runtime")
    }

    /// Removes and returns the states of entities whose last strong handle dropped.
    pub(crate) fn take_dropped(&mut self) -> Vec<(EntityId, Box<dyn Any>)> {
        todo!("runtime")
    }
}

impl<T: 'static> Lease<T> {
    /// The leased state.
    pub(crate) fn state_mut(&mut self) -> &mut T {
        todo!("runtime")
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
        AnyWeakEntity { entity_id: self.entity_id, entity_type: self.entity_type, entity_ref_counts: self.entity_map.clone() }
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
        todo!("runtime")
    }
}

impl Drop for AnyEntity {
    fn drop(&mut self) {
        // skeleton: decrement the count and queue the id at zero.
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
        todo!("runtime")
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
