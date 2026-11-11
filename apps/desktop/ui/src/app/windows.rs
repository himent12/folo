//! Opening, updating and closing windows from the app. Owner tag: `"runtime"`.
//!
//! A window update takes the `Box<Window>` out of its slot in `App.windows` for the call (the slot is `None`
//! meanwhile; a nested update of the same window is an error) and puts it back, or closes it if
//! `remove_window` was called.

use anyhow::{Context as _, anyhow};

use crate::{AnyView, AnyWindowHandle, App, Entity, Render, Window, WindowHandle, WindowId, WindowOptions};

impl App {
    /// Opens a window whose root view `build_root_view` makes. The first frame is drawn at the platform's
    /// first refresh, not here.
    pub fn open_window<V: 'static + Render>(
        &mut self,
        options: WindowOptions,
        build_root_view: impl FnOnce(&mut Window, &mut App) -> Entity<V>,
    ) -> anyhow::Result<WindowHandle<V>> {
        self.update(|cx| {
            let id = WindowId(cx.next_window_id);
            cx.next_window_id += 1;
            let handle = WindowHandle::<V>::new(id);
            // The slot exists (leased) while the window is built, so `update_window` on it fails cleanly.
            cx.windows.insert(id, None);
            match Window::new(handle.into(), options, cx) {
                Ok(mut window) => {
                    let root_view = build_root_view(&mut window, cx);
                    window.root = Some(root_view.into());
                    window.refresh();
                    cx.windows.insert(id, Some(Box::new(window)));
                    cx.window_order.push(id);
                    Ok(handle)
                }
                Err(error) => {
                    cx.windows.remove(&id);
                    Err(error)
                }
            }
        })
    }

    /// Every open window, in the order opened.
    pub fn windows(&self) -> Vec<AnyWindowHandle> {
        self.window_order
            .iter()
            .filter_map(|id| match self.windows.get(id) {
                Some(Some(window)) => Some(window.handle),
                _ => None,
            })
            .collect()
    }

    /// Leases window `handle` out for `f` (with its root view), then puts it back or closes it.
    pub(crate) fn update_window_id<T>(
        &mut self,
        handle: AnyWindowHandle,
        f: impl FnOnce(AnyView, &mut Window, &mut App) -> T,
    ) -> anyhow::Result<T> {
        self.update(|cx| {
            let id = handle.id;
            let slot = cx.windows.get_mut(&id).context("window not found")?;
            let mut window = slot.take().ok_or_else(|| anyhow!("window {id:?} is already being updated"))?;
            let Some(root_view) = window.root.clone() else {
                cx.windows.insert(id, Some(window));
                return Err(anyhow!("window {id:?} has no root view yet"));
            };

            let result = f(root_view, &mut window, cx);

            if window.removed {
                cx.windows.remove(&id);
                cx.window_order.retain(|w| *w != id);
                drop(window);
                cx.run_window_closed_observers();
            } else if let Some(slot) = cx.windows.get_mut(&id) {
                *slot = Some(window);
            }
            // Else the window was dropped meanwhile (quit): it closes here.
            Ok(result)
        })
    }

    /// Closes a window: drops it, runs the `on_window_closed` observers.
    #[allow(dead_code)] // seam: the window's `on_close` platform callback (window package) calls it
    pub(crate) fn close_window(&mut self, handle: AnyWindowHandle) {
        self.update(|cx| {
            // A leased window is closed by the update leasing it (`remove_window`).
            if let Some(Some(_)) = cx.windows.get(&handle.id) {
                let window = cx.windows.remove(&handle.id);
                cx.window_order.retain(|w| *w != handle.id);
                drop(window);
                cx.run_window_closed_observers();
            }
        })
    }

    fn run_window_closed_observers(&mut self) {
        self.window_closed_observers.clone().retain(&(), |callback| {
            callback(self);
            true
        });
    }
}
