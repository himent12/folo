//! Harness tests for the elements: a headless window with `FixedTextSystem`, asserting on the `Scene`.

mod layout;
mod lists;
mod opacity;
mod text;

use std::borrow::Cow;

use crate::{
    AnyElement, AnyWindowHandle, App, AppContext, Bounds, Context, HeadlessApp, HeadlessOptions, IntoElement,
    MonoSprite, Pixels, Primitive, Quad, Render, ScaledPixels, Scene, SharedString, TextBackend, Window,
    WindowOptions, point, px, size,
};

/// What a `TestView` renders with.
pub(super) type RenderFn = Box<dyn FnMut(&mut Window, &mut Context<TestView>) -> AnyElement>;

/// A view whose `render` calls a closure the test sets.
pub(super) struct TestView {
    pub(super) render: RenderFn,
}

impl Render for TestView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        (self.render)(window, cx)
    }
}

/// Test assets: a 16x16 square icon.
pub(super) struct Assets;

pub(super) const SQUARE_SVG: &str = r#"<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><rect x="0" y="0" width="16" height="16" fill="black"/></svg>"#;

impl crate::AssetSource for Assets {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        Ok((path == "icons/square.svg").then_some(Cow::Borrowed(SQUARE_SVG.as_bytes())))
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(Vec::new())
    }
}

/// A headless app (400x300 at scale 1, fixed text) with a window rendering `render`.
pub(super) fn open(
    render: impl FnMut(&mut Window, &mut Context<TestView>) -> AnyElement + 'static,
) -> (HeadlessApp, AnyWindowHandle) {
    open_sized(size(px(400.), px(300.)), 1.0, render)
}

/// `open` with a size and scale.
pub(super) fn open_sized(
    window_size: crate::Size<Pixels>,
    scale_factor: f32,
    render: impl FnMut(&mut Window, &mut Context<TestView>) -> AnyElement + 'static,
) -> (HeadlessApp, AnyWindowHandle) {
    let mut app =
        HeadlessApp::new(HeadlessOptions { size: window_size, scale_factor, text: TextBackend::Fixed })
            .with_assets(Assets);
    let window = app.update(|cx: &mut App| {
        let options = WindowOptions {
            window_bounds: Some(crate::WindowBounds::Windowed(Bounds::new(
                point(px(0.), px(0.)),
                window_size,
            ))),
            ..Default::default()
        };
        cx.open_window(options, |_, cx| cx.new(|_| TestView { render: Box::new(render) }))
            .expect("open window")
            .into()
    });
    (app, window)
}

/// Draws the window and returns a copy of its scene.
pub(super) fn draw(app: &mut HeadlessApp, window: AnyWindowHandle) -> Scene {
    app.frame(window);
    app.scene(window).clone()
}

/// The scene's quads.
pub(super) fn quads(scene: &Scene) -> Vec<Quad> {
    scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::Quad(q) => Some(q.clone()),
            _ => None,
        })
        .collect()
}

/// The scene's mono sprites (glyphs and icons).
pub(super) fn sprites(scene: &Scene) -> Vec<MonoSprite> {
    scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// Device-pixel bounds from logical numbers (scale 1).
pub(super) fn sb(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds::new(point(ScaledPixels(x), ScaledPixels(y)), size(ScaledPixels(w), ScaledPixels(h)))
}

/// The solid colour of a quad's background.
pub(super) fn solid(quad: &Quad) -> crate::Hsla {
    quad.background.solid
}
