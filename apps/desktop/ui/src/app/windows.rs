//! Opening, updating and closing windows from the app. Owner tag: `"runtime"`.
//!
//! A window update takes the `Box<Window>` out of its slot in `App.windows` for the call (the slot is `None`
//! meanwhile; a nested update of the same window is an error) and puts it back, or closes it if
//! `remove_window` was called.
#![allow(dead_code)] // skeleton

use crate::{AnyView, AnyWindowHandle, App, Entity, Render, Window, WindowHandle, WindowOptions};

impl App {
    /// Opens a window whose root view `build_root_view` makes. The first frame is drawn at the platform's
    /// first refresh, not here.
    pub fn open_window<V: 'static + Render>(
        &mut self,
        _options: WindowOptions,
        _build_root_view: impl FnOnce(&mut Window, &mut App) -> Entity<V>,
    ) -> anyhow::Result<WindowHandle<V>> {
        todo!("runtime")
    }

    /// Every open window, in the order opened.
    pub fn windows(&self) -> Vec<AnyWindowHandle> {
        todo!("runtime")
    }

    /// Leases window `handle` out for `f` (with its root view), then puts it back or closes it.
    pub(crate) fn update_window_id<T>(
        &mut self,
        _handle: AnyWindowHandle,
        _f: impl FnOnce(AnyView, &mut Window, &mut App) -> T,
    ) -> anyhow::Result<T> {
        todo!("runtime")
    }

    /// Closes a window: drops it, runs the `on_window_closed` observers.
    pub(crate) fn close_window(&mut self, _handle: AnyWindowHandle) {
        todo!("runtime")
    }
}
