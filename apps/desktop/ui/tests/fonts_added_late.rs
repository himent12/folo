//! Fonts added after a window has drawn: the next frame shows what a window drawn with the fonts from the
//! start shows (the text system's resolutions, the line layout cache, the line wrappers and the layout
//! tree's reuse all start again), on the cosmic-text backend and on DirectWrite.

use folo_ui::{
    AnyWindowHandle, AppContext as _, Context, Font, FontId, HeadlessApp, HeadlessOptions, IntoElement,
    ParentElement, PlatformTextSystem, Render, Styled, TextBackend, Window, WindowOptions, div, font, px,
    rgb, size,
};
use std::borrow::Cow;

const GEIST: &[u8] = include_bytes!("../../../../design/fonts/Geist-Regular.ttf");

/// Text in Geist: a line with a marker after it (where the marker lands is the text's measured width:
/// the layout tree's reuse), a line wrapped in a narrow box, and a truncated one (the line wrappers).
struct View;

impl Render for View {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(0xffffff))
            .font_family("Geist")
            .text_color(rgb(0x000000))
            .child(
                div().flex().child("alpha beta gamma WWW 01234").child(div().size(px(8.)).bg(rgb(0xff0000))),
            )
            .child(div().w(px(120.)).bg(rgb(0xddeeff)).child("wrapped words in a narrow box"))
            .child(div().w(px(90.)).truncate().child("a truncated line of text"))
    }
}

fn open(text: TextBackend) -> (HeadlessApp, AnyWindowHandle) {
    let mut app =
        HeadlessApp::new(HeadlessOptions { size: size(px(400.), px(160.)), scale_factor: 1., text });
    let handle = app.update(|cx| cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| View))).unwrap();
    (app, handle.into())
}

fn add_geist(app: &mut HeadlessApp) {
    app.update(|cx| cx.text_system().add_fonts(vec![Cow::Borrowed(GEIST)]).unwrap());
}

/// The reviewer's sequence: draw asking for Geist before it is added, add it, refresh, draw; against a
/// window whose app added Geist before its first frame.
fn assert_a_late_font_is_used(text: TextBackend) {
    let (mut app, window) = open(text);
    let before = app.frame(window).clone();
    let generation = app.read(|cx| cx.text_system().font_generation());
    add_geist(&mut app);
    assert_eq!(app.read(|cx| cx.text_system().font_generation()), generation + 1);
    app.update(|cx| cx.update_window(window, |_, window, _| window.refresh())).unwrap();
    let after = app.frame(window).clone();

    let (mut fresh, fresh_window) = open(text);
    add_geist(&mut fresh);
    let want = fresh.frame(fresh_window).clone();
    let differing = |a: &[u32], b: &[u32]| a.iter().zip(b).filter(|(a, b)| a != b).count();
    assert!(differing(&before.pixels, &want.pixels) > 0, "{text:?}: the fallback looks different from Geist");
    assert_eq!(differing(&after.pixels, &want.pixels), 0, "{text:?}: Geist after it was added");
}

#[cfg(any(not(windows), feature = "portable-text"))]
#[test]
fn a_font_added_after_the_first_frame_is_used_portable() {
    // The frame before Geist is added is drawn in a system font from the text system's fallback stack. A
    // machine with none of them (a bare CI image) cannot draw it, which says nothing about the code under
    // test: the backend's own test below still runs there.
    use folo_ui::text::portable::CosmicTextSystem;
    let system = CosmicTextSystem::new();
    let fallbacks =
        ["Helvetica", "Segoe UI", "Ubuntu", "Adwaita Sans", "Cantarell", "Noto Sans", "DejaVu Sans", "Arial"];
    if !fallbacks.iter().any(|family| system.font_id(&font(*family)).is_ok()) {
        eprintln!("skipped: none of the fallback families {fallbacks:?} is installed");
        return;
    }
    assert_a_late_font_is_used(TextBackend::Portable);
}

#[cfg(windows)]
#[test]
fn a_font_added_after_the_first_frame_is_used_direct_write() {
    assert_a_late_font_is_used(TextBackend::Native);
}

/// A face's metrics, to tell faces apart across text systems.
fn metrics(ts: &dyn PlatformTextSystem, id: FontId) -> (u32, f32, f32, f32, f32) {
    let m = ts.font_metrics(id);
    (m.units_per_em, m.ascent, m.descent, m.cap_height, m.x_height)
}

/// The backends' own family caches: a family asked for before its font was added resolves to that font
/// afterwards, and the id handed out before still names the face it named.
fn assert_the_backend_resolves_again(ts: &dyn PlatformTextSystem, fresh: &dyn PlatformTextSystem) {
    let geist: Font = font("Geist");
    let before = ts.font_id(&geist).ok().map(|id| (id, metrics(ts, id)));
    ts.add_fonts(vec![Cow::Borrowed(GEIST)]).unwrap();
    let after = ts.font_id(&geist).unwrap();
    fresh.add_fonts(vec![Cow::Borrowed(GEIST)]).unwrap();
    let want = fresh.font_id(&geist).unwrap();
    assert_eq!(metrics(ts, after), metrics(fresh, want), "Geist's metrics after it was added");
    if let Some((id, face)) = before {
        assert_ne!(face, metrics(ts, after), "the fallback is another face");
        assert_eq!(metrics(ts, id), face, "an old id names the same face");
    }
}

#[cfg(any(not(windows), feature = "portable-text"))]
#[test]
fn the_portable_backend_resolves_families_again_after_fonts_are_added() {
    use folo_ui::text::portable::CosmicTextSystem;
    assert_the_backend_resolves_again(&CosmicTextSystem::new(), &CosmicTextSystem::new());
}

#[cfg(windows)]
#[test]
fn direct_write_selects_fonts_again_after_fonts_are_added() {
    use folo_ui::text::dwrite::DirectWriteTextSystem;
    let ts = DirectWriteTextSystem::new().unwrap();
    assert_the_backend_resolves_again(&ts, &DirectWriteTextSystem::new().unwrap());
}
