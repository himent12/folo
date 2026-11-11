//! The public harness over the headless platform: what the app's `--shot` mode and every test use.
//! Owner tag: `"elements"`.
//!
//! A frozen-clock screenshot settles like this: `update` builds the state, `run_until_idle` pumps the main
//! and background queues until both are empty (tasks that wait on timers stay parked: the clock does not
//! move by itself), then `frame` draws the window if it is dirty and returns the framebuffer. Animations see
//! the same clock, so a frame is a pure function of the state and the time the test advanced to.
#![allow(dead_code)] // skeleton

use std::{rc::Rc, sync::Arc, time::Duration};

use crate::{
    AnyWindowHandle, App, AppCell, AssetSource, Framebuffer, Pixels, PlatformInput, Scene, Size,
    platform::headless::HeadlessPlatform,
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
}

impl HeadlessApp {
    /// A headless app.
    pub fn new(_options: HeadlessOptions) -> Self {
        todo!("elements")
    }

    /// Sets the asset source.
    pub fn with_assets(self, asset_source: impl AssetSource) -> Self {
        self.app.borrow_mut().asset_source = Arc::new(asset_source);
        self
    }

    /// Runs `f` with the app and flushes its effects.
    pub fn update<R>(&mut self, _f: impl FnOnce(&mut App) -> R) -> R {
        todo!("elements")
    }

    /// Pumps the main and background queues until both are empty and no background job is outstanding.
    pub fn run_until_idle(&mut self) {
        todo!("elements")
    }

    /// Advances the fake clock (due timers run), then `run_until_idle`.
    pub fn advance_clock(&mut self, _duration: Duration) {
        todo!("elements")
    }

    /// Delivers `input` to `window` as the platform would.
    pub fn simulate_input(&mut self, _window: AnyWindowHandle, _input: PlatformInput) {
        todo!("elements")
    }

    /// Draws `window` if it is dirty and returns its framebuffer.
    pub fn frame(&mut self, _window: AnyWindowHandle) -> &Framebuffer {
        todo!("elements")
    }

    /// The last painted display list of `window` (exact assertions on primitives).
    pub fn scene(&mut self, _window: AnyWindowHandle) -> &Scene {
        todo!("elements")
    }
}
