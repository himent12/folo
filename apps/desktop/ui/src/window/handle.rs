//! Window ids and handles: `WindowId`, `AnyWindowHandle`, `WindowHandle<V>`. Owner tag: `"window"`.
use std::{any::TypeId, marker::PhantomData};

use crate::{App, AppContext, Context, Entity, Window};

/// A window's id for this run of the app (never reused).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowId(pub(crate) u64);

impl WindowId {
    /// The id as a number.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A handle to a window whose root view type is erased.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnyWindowHandle {
    /// The window.
    pub(crate) id: WindowId,
    /// The root view's type.
    pub(crate) state_type: TypeId,
}

impl AnyWindowHandle {
    /// The window's id.
    pub fn window_id(&self) -> WindowId {
        self.id
    }

    /// Updates the window (`Err` when it has closed).
    pub fn update<C, R>(
        self,
        cx: &mut C,
        update: impl FnOnce(crate::AnyView, &mut Window, &mut App) -> R,
    ) -> anyhow::Result<R>
    where
        C: AppContext,
    {
        cx.update_window(self, update)
    }
}

/// A handle to a window whose root view is a `V` (what `open_window` returns).
pub struct WindowHandle<V> {
    /// The erased handle.
    pub(crate) any_handle: AnyWindowHandle,
    state_type: PhantomData<fn(V) -> V>,
}

impl<V: 'static> WindowHandle<V> {
    /// A handle to window `id`.
    pub(crate) fn new(id: WindowId) -> Self {
        WindowHandle {
            any_handle: AnyWindowHandle { id, state_type: TypeId::of::<V>() },
            state_type: PhantomData,
        }
    }

    /// The window's id.
    pub fn window_id(&self) -> WindowId {
        self.any_handle.id
    }

    /// Updates the root view with the window (`Err` when the window has closed).
    pub fn update<C, R>(
        &self,
        cx: &mut C,
        update: impl FnOnce(&mut V, &mut Window, &mut Context<V>) -> R,
    ) -> anyhow::Result<R>
    where
        C: AppContext,
    {
        cx.update_window(self.any_handle, |root, window, cx| {
            let view = root.downcast::<V>().map_err(|_| anyhow::anyhow!("the window's root view is not a {}", std::any::type_name::<V>()))?;
            Ok(view.update(cx, |view, cx| update(view, window, cx)))
        })?
    }

    /// The root view's entity.
    pub fn entity(&self, cx: &App) -> anyhow::Result<Entity<V>> {
        let window = cx
            .windows
            .get(&self.any_handle.id)
            .ok_or_else(|| anyhow::anyhow!("window closed"))?
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("window is being updated"))?;
        let root = window.root.clone().ok_or_else(|| anyhow::anyhow!("window has no root view"))?;
        root.downcast::<V>().map_err(|_| anyhow::anyhow!("the window's root view is not a {}", std::any::type_name::<V>()))
    }
}

impl<V> Clone for WindowHandle<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for WindowHandle<V> {}

impl<V: 'static> From<WindowHandle<V>> for AnyWindowHandle {
    fn from(handle: WindowHandle<V>) -> Self {
        handle.any_handle
    }
}
