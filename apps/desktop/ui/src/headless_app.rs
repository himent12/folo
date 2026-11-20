//! The public harness over the headless platform: what the app's `--shot` mode and every test use.
//! Owner tag: `"elements"`.
//!
//! A frozen-clock screenshot settles like this: `update` builds the state, `run_until_idle` pumps the main
//! queue and waits for the background jobs (tasks that wait on timers stay parked: the clock does not move
//! by itself), then `frame` draws the window if it is dirty and returns the framebuffer. Animations see the
//! same clock, so a frame is a pure function of the state and the time the test advanced to.

use std::{
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    AnyWindowHandle, App, AppCell, AppContext, AssetSource, Framebuffer, Pixels, PlatformInput,
    RequestFrameOptions, Scene, Size,
    platform::headless::{HeadlessDispatcher, HeadlessPlatform},
};

/// Which text backend headless windows shape with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextBackend {
    /// The platform's own (DirectWrite on Windows, cosmic-text elsewhere): screenshots match the real app.
    #[default]
    Native,
    /// cosmic-text everywhere (feature `portable-text` on Windows): identical pixels on every OS.
    Portable,
    /// The deterministic fake (`FixedTextSystem`, feature `test-support`): fixed advances, no font files.
    Fixed,
}

/// How a headless app's windows look.
#[derive(Clone, Copy, Debug)]
pub struct HeadlessOptions {
    /// Every window's size (logical px).
    pub size: Size<Pixels>,
    /// Device pixels per logical pixel.
    pub scale_factor: f32,
    /// The text backend.
    pub text: TextBackend,
}

impl Default for HeadlessOptions {
    fn default() -> Self {
        HeadlessOptions {
            size: crate::size(crate::px(1280.), crate::px(800.)),
            scale_factor: 1.0,
            text: TextBackend::default(),
        }
    }
}

/// An app on the headless platform, driven by the test.
pub struct HeadlessApp {
    /// The app.
    pub(crate) app: Rc<AppCell>,
    /// The platform (for its dispatcher and windows).
    pub(crate) platform: Rc<HeadlessPlatform>,
    /// A copy of the last framebuffer `frame` returned.
    framebuffer: Framebuffer,
    /// A copy of the last scene `scene` returned.
    scene: Scene,
}

impl HeadlessApp {
    /// A headless app.
    pub fn new(options: HeadlessOptions) -> Self {
        let platform = Rc::new(HeadlessPlatform::from_options(options));
        let app = App::new_app(platform.clone(), Arc::new(()));
        HeadlessApp { app, platform, framebuffer: Framebuffer::default(), scene: Scene::default() }
    }

    /// A harness around an app already built on `platform` (what `--shot` runs).
    pub(crate) fn from_parts(app: Rc<AppCell>, platform: Rc<HeadlessPlatform>) -> Self {
        HeadlessApp { app, platform, framebuffer: Framebuffer::default(), scene: Scene::default() }
    }

    /// Sets the asset source.
    pub fn with_assets(self, asset_source: impl AssetSource) -> Self {
        self.app.borrow_mut().asset_source = Arc::new(asset_source);
        self
    }

    /// The dispatcher (its clock, `set_idle_timeout`, `next_timer`).
    pub fn dispatcher(&self) -> &Arc<HeadlessDispatcher> {
        self.platform.headless_dispatcher()
    }

    /// The headless platform (its clipboard, cursor style, prompt answers, opened URLs).
    pub fn platform(&self) -> &HeadlessPlatform {
        &self.platform
    }

    /// Runs `f` with the app and flushes its effects.
    pub fn update<R>(&mut self, f: impl FnOnce(&mut App) -> R) -> R {
        self.app.update(f)
    }

    /// Reads the app.
    pub fn read<R>(&self, f: impl FnOnce(&App) -> R) -> R {
        f(&self.app.borrow())
    }

    /// Pumps the main and background queues until both are empty and no background job is outstanding.
    pub fn run_until_idle(&mut self) {
        self.dispatcher().run_until_idle();
    }

    /// `run_until_idle`, but done once no more than `held` background jobs are left: those a test holds back
    /// to release later (a slow answer that lands after others).
    pub fn run_until_held(&mut self, held: usize) {
        self.dispatcher().run_until_held(held);
    }

    /// Advances the fake clock (due timers run, in order, each followed by a pump), then `run_until_idle`.
    pub fn advance_clock(&mut self, duration: Duration) {
        self.dispatcher().advance_clock(duration);
        self.run_until_idle();
    }

    /// `advance_clock` with `held` background jobs held back on purpose (`run_until_held`).
    pub fn advance_clock_held(&mut self, duration: Duration, held: usize) {
        self.dispatcher().advance_clock_held(duration, held);
        self.run_until_held(held);
    }

    /// The fake clock's time.
    pub fn now(&self) -> Instant {
        crate::PlatformDispatcher::now(self.dispatcher().as_ref())
    }

    /// Delivers `input` to `window` as the platform would, then `run_until_idle`.
    #[track_caller]
    pub fn simulate_input(&mut self, window: AnyWindowHandle, input: PlatformInput) {
        self.app
            .update(|cx| cx.update_window(window, |_, window, cx| window.dispatch_event(input, cx)))
            .expect("simulate_input: the window is not open");
        self.run_until_idle();
    }

    /// Whether `window` would draw at its next refresh.
    pub fn is_dirty(&self, window: AnyWindowHandle) -> bool {
        match self.app.borrow().windows.get(&window.id) {
            Some(Some(window)) => window.dirty.get(),
            _ => false,
        }
    }

    /// Whether `window` asks for its next refresh: it is dirty, or a paint animation
    /// (`with_opacity_animation`) is running.
    pub fn wants_frame(&self, window: AnyWindowHandle) -> bool {
        match self.app.borrow().windows.get(&window.id) {
            Some(Some(window)) => window.dirty.get() || window.has_running_paint_animations(),
            _ => false,
        }
    }

    /// Draws `window` if it is dirty (or advances its paint animations) and returns its framebuffer.
    #[track_caller]
    pub fn frame(&mut self, window: AnyWindowHandle) -> &Framebuffer {
        let framebuffer = &mut self.framebuffer;
        self.app
            .update(|cx| {
                cx.update_window(window, |_, window, cx| {
                    window.on_request_frame(RequestFrameOptions::default(), cx);
                    framebuffer.clone_from(&window.framebuffer);
                })
            })
            .expect("frame: the window is not open");
        &self.framebuffer
    }

    /// The last painted display list of `window` (exact assertions on primitives).
    #[track_caller]
    pub fn scene(&mut self, window: AnyWindowHandle) -> &Scene {
        let scene = &mut self.scene;
        self.app
            .update(|cx| {
                cx.update_window(window, |_, window, _| scene.clone_from(&window.rendered_frame.scene))
            })
            .expect("scene: the window is not open");
        &self.scene
    }

    /// Settles `window` for a frozen-clock screenshot: pumps until idle and draws until two consecutive
    /// frames are identical (or the window stops being dirty), at most `max_frames` frames. Returns whether
    /// it settled. The clock does not move.
    pub fn settle(&mut self, window: AnyWindowHandle, max_frames: usize) -> bool {
        self.run_until_idle();
        let mut previous: Option<Framebuffer> = None;
        for _ in 0..max_frames {
            let frame = self.frame(window).clone();
            self.run_until_idle();
            let same = previous.as_ref() == Some(&frame);
            if same || !self.wants_frame(window) {
                return true;
            }
            previous = Some(frame);
        }
        false
    }

    /// Settles `window` for a screenshot of the app as a person would see it a moment after launch: frames
    /// are drawn while the fake clock advances in `step`s, until the window stops asking for frames (every
    /// one-shot animation has finished) or `max_time` has passed (a repeating animation never finishes; it
    /// is shown at that time). Returns whether it went idle. Timers the app set fire as the clock passes
    /// them (the app's frozen scenarios set none).
    pub fn settle_advancing(&mut self, window: AnyWindowHandle, max_time: Duration, step: Duration) -> bool {
        self.run_until_idle();
        let mut elapsed = Duration::ZERO;
        loop {
            self.frame(window);
            self.run_until_idle();
            if !self.wants_frame(window) {
                return true;
            }
            if elapsed >= max_time {
                return false;
            }
            self.advance_clock(step);
            elapsed += step;
        }
    }

    /// The open windows, in the order opened.
    pub fn windows(&self) -> Vec<AnyWindowHandle> {
        self.app.borrow().windows()
    }

    /// The system turns `appearance` (light or dark), delivered to every window through the
    /// platform's appearance callback as a real OS change is, then `run_until_idle`.
    pub fn set_system_appearance(&mut self, appearance: crate::WindowAppearance) {
        self.platform.set_appearance(appearance);
        self.run_until_idle();
    }
}
