//! The app runtime: `Application`, `App` (in `Rc<AppCell>`), the `AppContext` trait. Owner tag: `"runtime"`.
//!
//! One UI thread. `App` lives in an `Rc<AppCell>` (a `RefCell<App>`); platform callbacks and task wake-ups
//! borrow it for their call and run the effect flush before returning. Files: `entity_map.rs` (entities and
//! handles), `context.rs` (`Context<T>`), `async_app.rs` (`AsyncApp`), `effects.rs` (the effect queue and its
//! flush), `subscription.rs` (`SubscriberSet`, `Subscription`), `global.rs` (globals), `windows.rs`
//! (opening, updating and closing windows), `platform_ops.rs` (clipboard, displays, prompts, URLs),
//! `observe.rs` (observers, subscriptions and app-level handlers registered on `App`).

mod async_app;
mod context;
mod effects;
mod entity_map;
mod global;
mod observe;
mod platform_ops;
mod subscription;
#[cfg(test)]
mod tests;
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
    collections::{HashMap, HashSet, VecDeque},
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
        let platform = native_text_system()
            .and_then(default_platform)
            .unwrap_or_else(|error| panic!("cannot start the platform: {error:#}"));
        Application(App::new_app(platform, Arc::new(())))
    }

    /// An app on the headless platform (tests, `--shot`). Prefer `HeadlessApp`, which also drives it.
    pub fn headless(options: crate::HeadlessOptions) -> Self {
        let platform = crate::platform::headless::HeadlessPlatform::from_options(options);
        Application(App::new_app(Rc::new(platform), Arc::new(())))
    }

    /// Where `svg()` and other assets load from.
    pub fn with_assets(self, asset_source: impl AssetSource) -> Self {
        self.0.borrow_mut().asset_source = Arc::new(asset_source);
        self
    }

    /// Runs the platform's loop; `on_finish_launching` runs once it is up. Returns when the app quits.
    pub fn run<F>(self, on_finish_launching: F)
    where
        F: 'static + FnOnce(&mut App),
    {
        let this = self.0.clone();
        let platform = self.0.borrow().platform.clone();
        // GPUI runs this without an update around it; flushing its effects at once reaches the same state
        // sooner (defers and notifies do not wait for the first event).
        platform.run(Box::new(move || this.update(on_finish_launching)));
    }
}

/// The text backend the real platforms use: DirectWrite on Windows, cosmic-text elsewhere.
fn native_text_system() -> anyhow::Result<Arc<dyn crate::PlatformTextSystem>> {
    crate::platform::headless::text_backend(crate::TextBackend::Native)
}

/// The window backend for this OS: Win32 on Windows (winit when `FOLO_UI_PLATFORM=winit` and the `winit`
/// feature is on), winit elsewhere.
fn default_platform(text_system: Arc<dyn crate::PlatformTextSystem>) -> anyhow::Result<Rc<dyn Platform>> {
    #[cfg(windows)]
    {
        #[cfg(feature = "winit")]
        if std::env::var("FOLO_UI_PLATFORM").is_ok_and(|platform| platform == "winit") {
            return Ok(Rc::new(crate::platform::winit::WinitPlatform::new(text_system)?));
        }
        Ok(Rc::new(crate::platform::windows::WindowsPlatform::new(text_system)?))
    }
    #[cfg(not(windows))]
    {
        Ok(Rc::new(crate::platform::winit::WinitPlatform::new(text_system)?))
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

    /// Borrows the app unless it is mutably borrowed.
    pub(crate) fn try_borrow(&self) -> Option<Ref<'_, App>> {
        self.app.try_borrow().ok()
    }

    /// Borrows the app mutably unless it is borrowed already.
    pub(crate) fn try_borrow_mut(&self) -> Option<RefMut<'_, App>> {
        self.app.try_borrow_mut().ok()
    }

    /// Runs `f` with the app, then flushes its effects (the entry point of platform callbacks).
    #[track_caller]
    pub fn update<R>(&self, f: impl FnOnce(&mut App) -> R) -> R {
        self.borrow_mut().update(f)
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
    /// Entities notified and not yet flushed (one `Notify` effect each per flush, as GPUI).
    pub(crate) pending_notifications: HashSet<crate::EntityId>,
    /// Globals changed and not yet flushed (one effect each per flush, as GPUI).
    pub(crate) pending_global_notifications: HashSet<TypeId>,
    /// A notify, global change or refresh asked every window to redraw at the end of this flush.
    pub(crate) windows_need_refresh: bool,
    /// The window whose `draw` is running, if any. **`Window::draw` sets it for its duration**: a notify
    /// meanwhile still runs observers but does not dirty the windows (GPUI's invalidator ignores views that
    /// notify while their window draws), so a view that notifies in `render` does not redraw forever.
    pub(crate) drawing_window: Option<WindowId>,
}

impl App {
    /// Builds the app around a platform (called by `Application::new` / `headless`).
    pub(crate) fn new_app(platform: Rc<dyn Platform>, asset_source: Arc<dyn AssetSource>) -> Rc<AppCell> {
        let dispatcher = platform.dispatcher();
        let text_system = Arc::new(TextSystem::new(platform.text_system()));
        Rc::new_cyclic(|this| AppCell {
            app: RefCell::new(App {
                this: this.clone(),
                platform,
                text_system,
                asset_source,
                background_executor: BackgroundExecutor::new(dispatcher.clone()),
                foreground_executor: ForegroundExecutor::new(dispatcher),
                entities: EntityMap::new(),
                windows: HashMap::new(),
                window_order: Vec::new(),
                next_window_id: 1,
                focus_handles: Arc::new(FocusMap::default()),
                globals_by_type: HashMap::new(),
                keymap: Rc::new(RefCell::new(Keymap::default())),
                global_action_listeners: HashMap::new(),
                pending_effects: VecDeque::new(),
                pending_updates: 0,
                flushing_effects: false,
                observers: SubscriberSet::new(),
                event_listeners: SubscriberSet::new(),
                global_observers: SubscriberSet::new(),
                quit_observers: SubscriberSet::new(),
                window_closed_observers: SubscriberSet::new(),
                propagate_event: true,
                quitting: false,
                pending_notifications: HashSet::new(),
                pending_global_notifications: HashSet::new(),
                windows_need_refresh: false,
                drawing_window: None,
            }),
        })
    }

    /// Runs `f`, then flushes effects if this was the outermost update.
    pub(crate) fn update<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.pending_updates += 1;
        let result = f(self);
        if !self.flushing_effects && self.pending_updates == 1 {
            self.flushing_effects = true;
            self.flush_effects();
            self.flushing_effects = false;
        }
        self.pending_updates -= 1;
        result
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
    pub fn spawn<AsyncFn, R>(&self, f: AsyncFn) -> crate::Task<R>
    where
        AsyncFn: AsyncFnOnce(&mut AsyncApp) -> R + 'static,
        R: 'static,
    {
        let mut cx = self.to_async();
        self.foreground_executor.spawn(async move { f(&mut cx).await })
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
    pub fn on_window_closed(&self, on_closed: impl FnMut(&mut App) + 'static) -> Subscription {
        let (subscription, activate) = self.window_closed_observers.insert((), Box::new(on_closed));
        activate();
        subscription
    }

    /// Makes a new focus handle.
    pub fn focus_handle(&self) -> crate::FocusHandle {
        crate::FocusHandle::new(&self.focus_handles)
    }

    /// Frees an image's cached copies now (the renderer's resample cache and the window's masks). With no
    /// GPU atlas this only drops cache entries; `current_window` is the window being updated, if any.
    ///
    /// The renderer's caches key images by id, are bounded in bytes and drop what recent frames did not
    /// draw, so nothing has to be freed eagerly: this releases the caller's reference.
    pub fn drop_image(&mut self, image: Arc<crate::RenderImage>, _current_window: Option<&mut Window>) {
        drop(image);
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

    fn new<T: 'static>(&mut self, build_entity: impl FnOnce(&mut Context<T>) -> T) -> Entity<T> {
        self.update(|cx| {
            let handle = cx.entities.reserve::<T>();
            let state = build_entity(&mut Context::new_context(cx, handle.downgrade()));
            cx.entities.insert(&handle, state);
            handle
        })
    }

    fn update_entity<T, R>(
        &mut self,
        handle: &Entity<T>,
        update: impl FnOnce(&mut T, &mut Context<T>) -> R,
    ) -> R
    where
        T: 'static,
    {
        self.update(|cx| {
            let mut lease = cx.entities.lease(handle);
            let result = update(lease.state_mut(), &mut Context::new_context(cx, handle.downgrade()));
            cx.entities.end_lease(lease);
            result
        })
    }

    fn read_entity<T, R>(&self, handle: &Entity<T>, read: impl FnOnce(&T, &App) -> R) -> R
    where
        T: 'static,
    {
        read(self.entities.read(handle), self)
    }

    fn update_window<T, F>(&mut self, window: AnyWindowHandle, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(crate::AnyView, &mut Window, &mut App) -> T,
    {
        self.update_window_id(window, f)
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
