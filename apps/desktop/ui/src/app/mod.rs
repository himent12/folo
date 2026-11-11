//! The app runtime: `Application`, `App` (in `Rc<AppCell>`), the `AppContext` trait. Owner tag: `"runtime"`.
//!
//! One UI thread. `App` lives in an `Rc<AppCell>` (a `RefCell<App>`); platform callbacks and task wake-ups
//! borrow it for their call and run the effect flush before returning. Files: `entity_map.rs` (entities and
//! handles), `context.rs` (`Context<T>`), `async_app.rs` (`AsyncApp`), `effects.rs` (the effect queue and its
//! flush), `subscription.rs` (`SubscriberSet`, `Subscription`), `global.rs` (globals), `windows.rs`
//! (opening, updating and closing windows), `platform_ops.rs` (clipboard, displays, prompts, URLs).
#![allow(dead_code)] // skeleton

mod async_app;
mod context;
mod effects;
mod entity_map;
mod global;
mod platform_ops;
mod subscription;
mod windows;

pub use async_app::*;
pub use context::*;
pub(crate) use effects::*;
pub use entity_map::*;
pub use global::*;
pub use subscription::*;

use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    collections::{HashMap, VecDeque},
    rc::{Rc, Weak},
    sync::Arc,
};

use crate::{
    AnyWindowHandle, AssetSource, BackgroundExecutor, DispatchPhase, FocusMap, ForegroundExecutor, Keymap,
    Platform, TextSystem, Window, WindowId,
};

/// The app before it runs: `Application::new().with_assets(..).run(|cx| ..)`.
pub struct Application(Rc<AppCell>);

impl Application {
    /// An app on the default platform: the Win32 backend on Windows; winit elsewhere, or on Windows when
    /// `FOLO_UI_PLATFORM=winit` and the `winit` feature is on.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        todo!("runtime")
    }

    /// An app on the headless platform (tests, `--shot`). Prefer `HeadlessApp`, which also drives it.
    pub fn headless(_options: crate::HeadlessOptions) -> Self {
        todo!("runtime")
    }

    /// Where `svg()` and other assets load from.
    pub fn with_assets(self, asset_source: impl AssetSource) -> Self {
        self.0.borrow_mut().asset_source = Arc::new(asset_source);
        self
    }

    /// Runs the platform's loop; `on_finish_launching` runs once it is up. Returns when the app quits.
    pub fn run<F>(self, _on_finish_launching: F)
    where
        F: 'static + FnOnce(&mut App),
    {
        todo!("runtime")
    }
}

/// The cell that holds the app: borrowed for each platform callback and task wake-up.
pub struct AppCell {
    app: RefCell<App>,
}

impl AppCell {
    /// Borrows the app.
    #[track_caller]
    pub fn borrow(&self) -> Ref<'_, App> {
        self.app.borrow()
    }

    /// Borrows the app mutably (panics on a re-entrant borrow, which is a bug in the caller).
    #[track_caller]
    pub fn borrow_mut(&self) -> RefMut<'_, App> {
        self.app.borrow_mut()
    }
}

/// A global action handler (`cx.on_action` at app level), called after the window's dispatch tree.
pub(crate) type GlobalActionListener = Rc<dyn Fn(&dyn Any, DispatchPhase, &mut App)>;
/// `observe` / `notify` callbacks: return `false` to unsubscribe.
pub(crate) type Handler = Box<dyn FnMut(&mut App) -> bool + 'static>;
/// `subscribe` / `emit` callbacks: `(emitter event type, callback)`.
pub(crate) type Listener = (TypeId, Box<dyn FnMut(&dyn Any, &mut App) -> bool + 'static>);
/// `on_app_quit` callbacks: each returns a future the quit waits for (briefly, as GPUI).
pub(crate) type QuitHandler = Box<dyn FnMut(&mut App) -> LocalBoxFuture + 'static>;
/// `on_window_closed` callbacks.
pub(crate) type WindowClosedHandler = Box<dyn FnMut(&mut App) + 'static>;

/// A boxed, non-`Send` future (no `futures` dependency).
pub(crate) type LocalBoxFuture = std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>;

/// The app: entities, windows, globals, keymap, executors, the platform. Every callback gets `&mut App`
/// (or a `Context<T>`, which derefs to it).
pub struct App {
    /// The cell this app lives in (for `AsyncApp` and platform callbacks).
    pub(crate) this: Weak<AppCell>,
    /// The platform backend.
    pub(crate) platform: Rc<dyn Platform>,
    /// Fonts, metrics and the app-wide font cache.
    pub(crate) text_system: Arc<TextSystem>,
    /// Where assets load from.
    pub(crate) asset_source: Arc<dyn AssetSource>,
    /// Background threads.
    pub(crate) background_executor: BackgroundExecutor,
    /// The UI thread.
    pub(crate) foreground_executor: ForegroundExecutor,
    /// Every entity's state.
    pub(crate) entities: EntityMap,
    /// Each window in a slot; `None` while leased out by a window update.
    pub(crate) windows: HashMap<WindowId, Option<Box<Window>>>,
    /// Window ids in the order opened (for `windows()`).
    pub(crate) window_order: Vec<WindowId>,
    /// The next window id.
    pub(crate) next_window_id: u64,
    /// Focus handles' ref counts (shared with every `FocusHandle`).
    pub(crate) focus_handles: Arc<FocusMap>,
    /// Globals by type.
    pub(crate) globals_by_type: HashMap<TypeId, Box<dyn Any>>,
    /// Key bindings.
    pub(crate) keymap: Rc<RefCell<Keymap>>,
    /// App-level action handlers by action type.
    pub(crate) global_action_listeners: HashMap<TypeId, Vec<GlobalActionListener>>,
    /// Effects queued by the current update, flushed when the outermost update ends.
    pub(crate) pending_effects: VecDeque<Effect>,
    /// Depth of nested updates; the flush runs when it returns to zero.
    pub(crate) pending_updates: usize,
    /// Whether a flush is running (effects queued now join it).
    pub(crate) flushing_effects: bool,
    /// `observe` callbacks by observed entity.
    pub(crate) observers: SubscriberSet<crate::EntityId, Handler>,
    /// `subscribe` callbacks by emitting entity.
    pub(crate) event_listeners: SubscriberSet<crate::EntityId, Listener>,
    /// `observe_global` callbacks by global type.
    pub(crate) global_observers: SubscriberSet<TypeId, Handler>,
    /// `on_app_quit` callbacks.
    pub(crate) quit_observers: SubscriberSet<(), QuitHandler>,
    /// `on_window_closed` callbacks.
    pub(crate) window_closed_observers: SubscriberSet<(), WindowClosedHandler>,
    /// Cleared by `stop_propagation` during mouse and key dispatch.
    pub(crate) propagate_event: bool,
    /// Set once quit starts.
    pub(crate) quitting: bool,
}

impl App {
    /// Builds the app around a platform (called by `Application::new` / `headless`).
    pub(crate) fn new_app(_platform: Rc<dyn Platform>, _asset_source: Arc<dyn AssetSource>) -> Rc<AppCell> {
        todo!("runtime")
    }

    /// Runs `f`, then flushes effects if this was the outermost update.
    pub(crate) fn update<R>(&mut self, _f: impl FnOnce(&mut Self) -> R) -> R {
        todo!("runtime")
    }

    /// The background executor.
    pub fn background_executor(&self) -> &BackgroundExecutor {
        &self.background_executor
    }

    /// The UI-thread executor.
    pub fn foreground_executor(&self) -> &ForegroundExecutor {
        &self.foreground_executor
    }

    /// Spawns `f` on the UI thread with an `AsyncApp`.
    pub fn spawn<AsyncFn, R>(&self, _f: AsyncFn) -> crate::Task<R>
    where
        AsyncFn: AsyncFnOnce(&mut AsyncApp) -> R + 'static,
        R: 'static,
    {
        todo!("runtime")
    }

    /// An `AsyncApp` for this app.
    pub fn to_async(&self) -> AsyncApp {
        AsyncApp {
            app: self.this.clone(),
            background_executor: self.background_executor.clone(),
            foreground_executor: self.foreground_executor.clone(),
        }
    }

    /// Runs `f` after the current effect flush.
    pub fn defer(&mut self, f: impl FnOnce(&mut App) + 'static) {
        self.push_effect(Effect::Defer { callback: Box::new(f) });
    }

    /// The app-wide text system (fonts, metrics).
    pub fn text_system(&self) -> &Arc<TextSystem> {
        &self.text_system
    }

    /// The asset source.
    pub fn asset_source(&self) -> &Arc<dyn AssetSource> {
        &self.asset_source
    }

    /// Adds key bindings.
    pub fn bind_keys(&mut self, bindings: impl IntoIterator<Item = KeyBinding>) {
        self.keymap.borrow_mut().add_bindings(bindings);
        self.refresh_windows();
    }

    /// The keymap.
    pub fn keymap(&self) -> Rc<RefCell<Keymap>> {
        self.keymap.clone()
    }

    /// Stops the mouse or key event being dispatched from reaching further listeners.
    pub fn stop_propagation(&mut self) {
        self.propagate_event = false;
    }

    /// Lets the event continue to further listeners (undoes `stop_propagation`; an action handler calls it
    /// to let the key go on).
    pub fn propagate(&mut self) {
        self.propagate_event = true;
    }

    /// Marks every window dirty (each draws at its next refresh).
    pub fn refresh_windows(&mut self) {
        self.push_effect(Effect::RefreshWindows);
    }

    /// Registers `f` to run when a window has closed.
    pub fn on_window_closed(&self, _on_closed: impl FnMut(&mut App) + 'static) -> Subscription {
        todo!("runtime")
    }

    /// Makes a new focus handle.
    pub fn focus_handle(&self) -> crate::FocusHandle {
        crate::FocusHandle::new(&self.focus_handles)
    }

    /// Frees an image's cached copies now (the renderer's resample cache and the window's masks). With no
    /// GPU atlas this only drops cache entries; `current_window` is the window being updated, if any.
    pub fn drop_image(&mut self, _image: Arc<crate::RenderImage>, _current_window: Option<&mut Window>) {
        todo!("runtime")
    }
}

use crate::KeyBinding;

/// What every context (`App`, `Context<T>`, `AsyncApp`) can do with entities. `Result<T>` is `T` for the
/// synchronous contexts and `anyhow::Result<T>` for `AsyncApp` (the app may be gone).
pub trait AppContext {
    /// The return wrapper of this context.
    type Result<T>;

    /// Builds a new entity.
    #[allow(clippy::wrong_self_convention)]
    fn new<T: 'static>(&mut self, build_entity: impl FnOnce(&mut Context<T>) -> T)
    -> Self::Result<Entity<T>>;

    /// Updates an entity.
    fn update_entity<T, R>(
        &mut self,
        handle: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> Self::Result<R>
    where
        T: 'static;

    /// Reads an entity.
    fn read_entity<T, R>(&self, handle: &Entity<T>, read: impl FnOnce(&T, &App) -> R) -> Self::Result<R>
    where
        T: 'static;

    /// Updates a window: the closure gets its root view, the window and the app.
    fn update_window<T, F>(&mut self, window: AnyWindowHandle, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(crate::AnyView, &mut Window, &mut App) -> T;

    /// Spawns a `Send` future on the background executor.
    fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> crate::Task<R>
    where
        R: Send + 'static;

    /// Reads a global.
    fn read_global<G, R>(&self, callback: impl FnOnce(&G, &App) -> R) -> Self::Result<R>
    where
        G: Global;
}

impl AppContext for App {
    type Result<T> = T;

    fn new<T: 'static>(&mut self, _build_entity: impl FnOnce(&mut Context<T>) -> T) -> Entity<T> {
        todo!("runtime")
    }

    fn update_entity<T, R>(
        &mut self,
        _handle: &Entity<T>,
        _update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> R
    where
        T: 'static,
    {
        todo!("runtime")
    }

    fn read_entity<T, R>(&self, handle: &Entity<T>, read: impl FnOnce(&T, &App) -> R) -> R
    where
        T: 'static,
    {
        read(self.entities.read(handle), self)
    }

    fn update_window<T, F>(&mut self, _window: AnyWindowHandle, _f: F) -> anyhow::Result<T>
    where
        F: FnOnce(crate::AnyView, &mut Window, &mut App) -> T,
    {
        todo!("runtime")
    }

    fn background_spawn<R>(&self, future: impl Future<Output = R> + Send + 'static) -> crate::Task<R>
    where
        R: Send + 'static,
    {
        self.background_executor.spawn(future)
    }

    fn read_global<G, R>(&self, callback: impl FnOnce(&G, &App) -> R) -> R
    where
        G: Global,
    {
        callback(self.global::<G>(), self)
    }
}
