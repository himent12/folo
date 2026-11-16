//! An image whose pixels failed is marked so where it is drawn, not left looking as if it loads
//! (d1-datasets, x4-soak; f2-desktop-siblings review round 1): painted, in the scene, on the
//! dataset grid, the Playground samples, a run's samples and a benchmark's Plan previews, while
//! a loading or loaded image paints no mark.

use std::sync::Arc;
use std::time::Duration;

use folo_ui::{HeadlessApp, Primitive, SpriteKind, WindowHandle};

use super::tests::drive;
use super::{FoloApp, Route};
use crate::api::mock::MockBackend;
use crate::store::{ImgKind, ImgSlot};

/// [`super::tests::open_with`], with the app's icons (so they paint) in a window tall enough for a
/// run's samples.
fn open_with_icons(api: Arc<MockBackend>) -> (HeadlessApp, WindowHandle<FoloApp>) {
    use folo_ui::{AppContext, Bounds, HeadlessOptions, WindowBounds, WindowOptions, point, px, size};
    let mut app = HeadlessApp::new(HeadlessOptions::default()).with_assets(crate::Assets);
    let window = app.update(|cx| {
        cx.set_global(crate::theme::Theme::new(crate::theme::Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(1280.), px(3200.)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
            .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    (app, window)
}

/// What the window paints, once what it asked for has come: its SVG icons and its text glyphs.
fn marks(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> (usize, usize) {
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    app.frame(window.into());
    let mut n = (0, 0);
    for p in &app.scene(window.into()).primitives {
        if let Primitive::MonoSprite(s) = p {
            match s.kind {
                SpriteKind::Svg => n.0 += 1,
                SpriteKind::Glyph => n.1 += 1,
            }
        }
    }
    n
}

/// The focus keys drawn in the last frame.
fn keys(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> Vec<String> {
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

/// What `route` paints with every image failed past its automatic retries, then with those images
/// loading again (held), then loaded. The app is left with them loaded.
struct Painted {
    failed: (usize, usize),
    failed_keys: Vec<String>,
    loading: (usize, usize),
    loaded: (usize, usize),
}

fn paint(
    route: Route,
    setup: impl Fn(&mut FoloApp),
) -> (HeadlessApp, WindowHandle<FoloApp>, Arc<MockBackend>, Painted) {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with_icons(api.clone());
    api.fail_pixels(true);
    drive(&mut app, window, |this, w, cx| this.navigate(route, w, cx));
    // `setup` again before each look (it may scroll to what the test reads, once laid out).
    let look = |app: &mut HeadlessApp| {
        for _ in 0..2 {
            drive(app, window, |this, _, cx| {
                setup(this);
                cx.notify();
            });
            marks(app, window);
        }
        marks(app, window)
    };
    // Each failure asks again 2, 8 then 32 s later; the fourth waits for a Retry.
    for _ in 0..6 {
        look(&mut app);
        app.advance_clock(Duration::from_secs(40));
        app.run_until_idle();
    }
    let failed = look(&mut app);
    let failed_keys = keys(&mut app, window);
    // The same images loading: held back, so none is in yet.
    drive(&mut app, window, |this, _, cx| {
        this.hold_loading = true;
        this.store.images_px.forget_failed();
        cx.notify();
    });
    let loading = look(&mut app);
    api.fail_pixels(false);
    drive(&mut app, window, |this, _, cx| {
        this.hold_loading = false;
        cx.notify();
    });
    let loaded = look(&mut app);
    (app, window, api, Painted { failed, failed_keys, loading, loaded })
}

/// The glyphs of "Image unavailable" (a space paints none).
fn words() -> usize {
    crate::screens::TILE_UNAVAILABLE.chars().filter(|c| !c.is_whitespace()).count()
}

#[test]
fn a_failed_dataset_tile_says_image_unavailable_and_a_loading_or_loaded_one_does_not() {
    let (_, _, _, p) = paint(Route::Dataset("ds_3b91".into()), |_| {});
    let tiles = p.failed.0 - p.loading.0;
    assert!(tiles > 0, "failed tiles paint their image glyph: {p:?}", p = (p.failed, p.loading));
    assert_eq!(p.failed.1 - p.loading.1, tiles * words(), "and say Image unavailable, each");
    assert_eq!(p.loaded, p.loading, "loaded tiles paint no mark, as loading ones");
}

#[test]
fn a_failed_run_sample_says_image_unavailable() {
    // The samples sit low on the page: scrolled to its foot, where they are painted.
    let (_, _, _, p) = paint(Route::Run("run_52e8".into()), |this| {
        let max = this.page_scroll.max_offset().height;
        this.page_scroll.set_offset(folo_ui::point(folo_ui::px(0.), -max));
    });
    let cells = p.failed.0 - p.loading.0;
    // (The headless window paints a row of four of the eight; the row below is past its edge.)
    assert!(cells >= 4, "each failed sample paints its image glyph: {p:?}", p = (p.failed, p.loading));
    assert_eq!(p.failed.1 - p.loading.1, cells * words());
    assert_eq!(p.loaded, p.loading);
}

#[test]
fn a_failed_bench_preview_is_marked_and_is_a_retry() {
    let (mut app, window, api, p) = paint(Route::Bench("bn_7a04".into()), |this| {
        this.bench.tab = crate::bench::BenchTab::Plan;
    });
    let previews = p.failed.0 - p.loading.0;
    assert!(previews > 0, "failed previews paint their image glyph: {p:?}", p = (p.failed, p.loading));
    assert_eq!(p.failed.1, p.loading.1, "a 72 px preview has room for the glyph only");
    assert_eq!(p.loaded, p.loading);
    let retries: Vec<&String> =
        p.failed_keys.iter().filter(|k| k.starts_with("bench-preview-retry")).collect();
    // Each marked preview is a Retry stop (the rows below the window's edge too).
    assert!(retries.len() >= previews, "{retries:?}");
    // Loaded, none is failed (so none is bound as a Retry; the focus map keeps past frames' stops).
    let ims: Vec<String> = drive(&mut app, window, |this, _, _| {
        this.bench.previews["bn_7a04"].ready().unwrap().images.iter().map(|i| i.id.clone()).collect()
    });
    assert!(drive(&mut app, window, |this, _, _| ims
        .iter()
        .all(|i| !this.bench_image_failed("bn_7a04", i, None))));

    // Failed again past its retries, then the server's images answer: its Retry loads it.
    api.fail_pixels(true);
    drive(&mut app, window, |this, _, cx| {
        this.store.images_px.forget_failed();
        this.store.images_px.clear();
        cx.notify();
    });
    for _ in 0..6 {
        marks(&mut app, window);
        app.advance_clock(Duration::from_secs(40));
        app.run_until_idle();
    }
    let (bench, image) = drive(&mut app, window, |this, _, _| {
        let im = this.bench.previews["bn_7a04"].ready().unwrap().images[0].id.clone();
        assert!(this.bench_image_failed("bn_7a04", &im, None), "the unchanged preview failed");
        ("bn_7a04".to_string(), im)
    });
    api.fail_pixels(false);
    app.advance_clock(Duration::from_secs(3600));
    assert!(drive(&mut app, window, |this, _, _| this.bench_image_failed(&bench, &image, None)), "it waits");

    // By keyboard (review round 2): the focused Retry's ring shows whole, not clipped by the
    // image's frame; Enter loads it; Tab then goes on to the next preview's Retry, not back to
    // the first control on the page.
    app.update(super::bind_keys);
    drive(&mut app, window, |this, w, cx| {
        this.kb = true;
        this.focus_key("bench-preview-retry", w, cx);
        cx.notify();
    });
    marks(&mut app, window);
    let focus = drive(&mut app, window, |_, _, cx| crate::theme::theme(cx).focus);
    let rings: Vec<folo_ui::Quad> = app
        .scene(window.into())
        .primitives
        .iter()
        .filter_map(|p| match p {
            // (2 px: the severity chips' 1 px accent edges share the colour.)
            Primitive::Quad(q)
                if q.border_color == focus && q.border_widths.top >= folo_ui::ScaledPixels::from(2.) =>
            {
                Some(q.clone())
            }
            _ => None,
        })
        .collect();
    // A border paints as one quad per edge, each clipped to its strip: together they cover the
    // ring's whole box unless something around the control clips it.
    let ring = rings.first().expect("the focused Retry paints its ring").bounds;
    assert!(rings.iter().all(|r| r.bounds == ring), "one ring: {rings:?}");
    let shown = rings.iter().skip(1).fold(rings[0].clip, |u, r| u.union(&r.clip));
    assert_eq!(shown, ring, "the ring shows whole, not clipped by the image's frame");
    press(&mut app, window, "enter");
    marks(&mut app, window);
    assert!(!drive(&mut app, window, |this, _, _| this.bench_image_failed(&bench, &image, None)), "loaded");
    press(&mut app, window, "tab");
    let focused = drive(&mut app, window, |this, w, _| {
        this.focus.borrow().iter().find(|(_, fh)| fh.is_focused(w)).map(|(k, _)| k.to_string())
    });
    assert!(
        focused.as_deref().is_some_and(|k| k.starts_with("bench-preview-retry-")),
        "Tab goes on to the next failed preview: {focused:?}"
    );
}

/// One press of `key`: down, then up, a frame drawn after each.
fn press(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
    use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};
    let keystroke = Keystroke::parse(key).unwrap();
    let down = KeyDownEvent { keystroke: keystroke.clone(), is_held: false };
    app.simulate_input(window.into(), PlatformInput::KeyDown(down));
    app.run_until_idle();
    app.frame(window.into());
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.run_until_idle();
    app.frame(window.into());
}

#[test]
fn a_failed_playground_sample_is_marked_and_opening_it_loads_its_thumbnail_again() {
    let (mut app, window, api, p) = paint(Route::Playground, |this| this.play_picker_open = true);
    let samples = p.failed.0 - p.loading.0;
    assert!(samples > 0, "failed samples paint their image glyph: {p:?}", p = (p.failed, p.loading));
    assert_eq!(p.failed.1, p.loading.1, "a 48 px sample has room for the glyph only");
    assert_eq!(p.loaded, p.loading);

    // Failed past its retries; the images answer again and the sample is opened (its full size is
    // fetched past the image cache): its thumbnail is asked for again, and shows.
    api.fail_pixels(true);
    drive(&mut app, window, |this, _, cx| {
        this.store.images_px.clear();
        cx.notify();
    });
    for _ in 0..6 {
        marks(&mut app, window);
        app.advance_clock(Duration::from_secs(40));
        app.run_until_idle();
    }
    let ds = drive(&mut app, window, |this, _, _| this.play_sample_dataset()).expect("a sample dataset");
    let im = keys(&mut app, window)
        .iter()
        .find_map(|k| k.strip_prefix("play-sample-").map(str::to_string))
        .expect("a sample drawn");
    assert!(drive(&mut app, window, |this, _, _| this.thumb_unavailable(&ds, &im)));
    api.fail_pixels(false);
    app.advance_clock(Duration::from_secs(3600));
    assert!(drive(&mut app, window, |this, _, _| this.thumb_unavailable(&ds, &im)), "it waits");
    drive(&mut app, window, |this, _, cx| this.play_dataset_image(&ds, &im, cx));
    marks(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(this.play_image.is_some(), "the sample opened");
        let key = crate::store::ImgKey { dataset: ds.clone(), image: im.clone(), kind: ImgKind::Thumb };
        assert!(matches!(this.store.images_px.get(&key), Some(ImgSlot::Ready(_))), "its thumbnail loaded");
    });
}
