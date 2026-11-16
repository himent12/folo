//! Window tests through the headless harness: they build a tree, drive synthetic input, and assert on the
//! scene and on callbacks (never on pixels). Expected bounds are worked out by hand from GPUI's rules.

mod frames;
mod keys;
mod layout;
mod layout_remeasure;
mod layout_reuse;
mod mouse;
mod realwin;

use std::{cell::RefCell, rc::Rc};

pub(super) use crate::AppContext as _;
use crate::{
    AnyElement, AnyWindowHandle, Bounds, Context, HeadlessApp, HeadlessOptions, Hsla, IntoElement, Modifiers,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, PlatformInput, Point, Primitive,
    Render, ScaledPixels, ScrollDelta, ScrollWheelEvent, Size, TextBackend, TouchPhase, Window,
    WindowOptions, hsla, point, px, size,
};

/// What a test view renders: a closure, so each test writes its tree inline.
pub(super) type RenderFn = Rc<dyn Fn(&mut Window, &mut Context<TestView>) -> AnyElement>;

/// A root view whose `render` is a closure.
pub(super) struct TestView {
    pub(super) render: RenderFn,
}

impl Render for TestView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let render = self.render.clone();
        render(window, cx)
    }
}

/// A headless app with one window of `size` at `scale`, rendering `render`; the first frame is drawn.
pub(super) fn open(
    size: Size<Pixels>,
    scale: f32,
    render: impl Fn(&mut Window, &mut Context<TestView>) -> AnyElement + 'static,
) -> (HeadlessApp, AnyWindowHandle) {
    let mut app = HeadlessApp::new(HeadlessOptions { size, scale_factor: scale, text: TextBackend::Fixed });
    let render: RenderFn = Rc::new(render);
    let handle = app
        .update(|cx| cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| TestView { render })))
        .expect("open the test window");
    let handle: AnyWindowHandle = handle.into();
    app.frame(handle);
    (app, handle)
}

/// A distinct opaque colour per index.
pub(super) fn color(ix: usize) -> Hsla {
    hsla(ix as f32 * 0.07 % 1., 0.8, 0.5, 1.)
}

/// The bounds of every quad whose background is `color`, in paint order.
pub(super) fn quads_of(
    app: &mut HeadlessApp,
    window: AnyWindowHandle,
    color: Hsla,
) -> Vec<Bounds<ScaledPixels>> {
    app.scene(window)
        .primitives
        .iter()
        .filter_map(|primitive| match primitive {
            Primitive::Quad(quad) if quad.background.solid == color && !color.is_transparent() => {
                Some(quad.bounds)
            }
            _ => None,
        })
        .collect()
}

/// The bounds of the single quad filled with `color`.
#[track_caller]
pub(super) fn quad_of(app: &mut HeadlessApp, window: AnyWindowHandle, color: Hsla) -> Bounds<ScaledPixels> {
    let quads = quads_of(app, window, color);
    assert_eq!(quads.len(), 1, "one quad of colour {color:?}, found {quads:?}");
    quads[0]
}

/// Logical bounds as device-pixel bounds at `scale` (what the scene holds).
pub(super) fn device(x: f32, y: f32, w: f32, h: f32, scale: f32) -> Bounds<ScaledPixels> {
    Bounds::new(point(px(x), px(y)), size(px(w), px(h))).scale(scale)
}

/// Asserts two device rectangles are equal within float noise.
#[track_caller]
pub(super) fn assert_bounds(actual: Bounds<ScaledPixels>, expected: Bounds<ScaledPixels>) {
    let close = |a: ScaledPixels, b: ScaledPixels| (a.0 - b.0).abs() < 1e-3;
    assert!(
        close(actual.origin.x, expected.origin.x)
            && close(actual.origin.y, expected.origin.y)
            && close(actual.size.width, expected.size.width)
            && close(actual.size.height, expected.size.height),
        "bounds {actual:?} != expected {expected:?}"
    );
}

/// A mouse move to `(x, y)` (logical px).
pub(super) fn mouse_move(app: &mut HeadlessApp, window: AnyWindowHandle, x: f32, y: f32) {
    let event = MouseMoveEvent {
        position: point(px(x), px(y)),
        pressed_button: None,
        modifiers: Modifiers::default(),
    };
    app.simulate_input(window, PlatformInput::MouseMove(event));
}

/// A left button press at `(x, y)`.
pub(super) fn mouse_down(app: &mut HeadlessApp, window: AnyWindowHandle, x: f32, y: f32) {
    let event = MouseDownEvent {
        button: MouseButton::Left,
        position: point(px(x), px(y)),
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    };
    app.simulate_input(window, PlatformInput::MouseDown(event));
}

/// A left button release at `(x, y)`.
pub(super) fn mouse_up(app: &mut HeadlessApp, window: AnyWindowHandle, x: f32, y: f32) {
    let event = MouseUpEvent {
        button: MouseButton::Left,
        position: point(px(x), px(y)),
        modifiers: Modifiers::default(),
        click_count: 1,
    };
    app.simulate_input(window, PlatformInput::MouseUp(event));
}

/// A wheel event at `(x, y)` with a pixel delta.
pub(super) fn wheel(app: &mut HeadlessApp, window: AnyWindowHandle, x: f32, y: f32, delta: Point<Pixels>) {
    let event = ScrollWheelEvent {
        position: point(px(x), px(y)),
        delta: ScrollDelta::Pixels(delta),
        modifiers: Modifiers::default(),
        touch_phase: TouchPhase::Moved,
    };
    app.simulate_input(window, PlatformInput::ScrollWheel(event));
}

/// A shared log tests push into from listeners.
pub(super) type Log = Rc<RefCell<Vec<String>>>;

/// A new empty log.
pub(super) fn log() -> Log {
    Rc::default()
}

/// Reads the window (its state) in a test.
pub(super) fn with_window<R>(
    app: &mut HeadlessApp,
    handle: AnyWindowHandle,
    f: impl FnOnce(&mut Window, &mut crate::App) -> R,
) -> R {
    app.update(|cx| cx.update_window(handle, |_, window, cx| f(window, cx))).expect("window open")
}
