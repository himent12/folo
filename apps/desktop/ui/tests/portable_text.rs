//! The cosmic-text backend (`text/portable`) with the bundled Geist fonts: layout covers the text, `tnum`,
//! reproducibility, masks and subpixel variants, and the lazy system-font load.
#![cfg(any(not(windows), feature = "portable-text"))]

use folo_ui::text::portable::CosmicTextSystem;
use folo_ui::{
    DevicePixels, Font, FontFeatures, FontId, FontRun, FontWeight, GlyphId, LineLayout, PlatformTextSystem,
    Point, RenderGlyphParams, font, px,
};
use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

const FONTS: [&[u8]; 6] = [
    include_bytes!("../../../../design/fonts/Geist-Regular.ttf"),
    include_bytes!("../../../../design/fonts/Geist-Medium.ttf"),
    include_bytes!("../../../../design/fonts/Geist-SemiBold.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-Regular.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-Medium.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-SemiBold.ttf"),
];

fn text_system() -> CosmicTextSystem {
    let ts = CosmicTextSystem::new();
    ts.add_fonts(FONTS.iter().map(|f| Cow::Borrowed(*f)).collect()).unwrap();
    ts
}

fn geist(weight: f32) -> Font {
    Font { weight: FontWeight(weight), ..font("Geist") }
}

fn tnum(mut f: Font) -> Font {
    f.features = FontFeatures(Arc::new(vec![("tnum".into(), 1)]));
    f
}

fn layout(ts: &CosmicTextSystem, f: &Font, text: &str, size: f32) -> LineLayout {
    let font_id = ts.font_id(f).unwrap();
    ts.layout_line(text, px(size), &[FontRun { len: text.len(), font_id }])
}

/// Glyph indices are char boundaries, ascending, start at 0, and every char starts some glyph's cluster
/// or lies inside one (no char is skipped past by the glyph stream).
fn assert_covers(text: &str, line: &LineLayout) {
    let indices: Vec<usize> = line.runs.iter().flat_map(|r| r.glyphs.iter().map(|g| g.index)).collect();
    assert!(!indices.is_empty(), "{text:?}: no glyphs");
    assert_eq!(indices[0], 0, "{text:?}: first glyph not at 0");
    assert!(indices.windows(2).all(|w| w[0] <= w[1]), "{text:?}: indices out of order {indices:?}");
    for &i in &indices {
        assert!(text.is_char_boundary(i), "{text:?}: index {i} is not a char boundary");
    }
    assert_eq!(line.len, text.len());
    // Every char start is at or after a glyph start, and the glyph starts reach the last char.
    let last_char = text.char_indices().last().unwrap().0;
    assert!(*indices.last().unwrap() <= last_char);
    let distinct: std::collections::BTreeSet<usize> = indices.iter().copied().collect();
    let chars = text.chars().count();
    assert!(distinct.len() * 2 >= chars, "{text:?}: {} clusters for {chars} chars", distinct.len());
}

fn glyph_ids(line: &LineLayout) -> Vec<GlyphId> {
    line.runs.iter().flat_map(|r| r.glyphs.iter().map(|g| g.id)).collect()
}

fn params(font_id: FontId, glyph_id: GlyphId, size: f32, scale: f32, variant_x: u8) -> RenderGlyphParams {
    RenderGlyphParams {
        font_id,
        glyph_id,
        font_size: px(size),
        subpixel_variant: Point { x: variant_x, y: 0 },
        scale_factor: scale,
        is_emoji: false,
    }
}

#[test]
fn runs_cover_the_text_with_bundled_fonts_only() {
    let ts = text_system();
    for text in
        ["Hello, world 123", "a · b", "3 × 4", "café", "1–2", "wait…", "Epoch 12 · loss 0.0451 × 3 – é…"]
    {
        for f in [geist(400.), geist(600.), font("Geist Mono")] {
            let line = layout(&ts, &f, text, 14.);
            assert_covers(text, &line);
            assert!(glyph_ids(&line).iter().all(|g| g.0 != 0), "{text:?} in {}: missing glyph", f.family);
            assert!(line.width > px(0.) && line.ascent > px(0.) && line.descent > px(0.));
            assert!(line.runs.iter().all(|r| r.glyphs.iter().all(|g| !g.is_emoji)));
        }
    }
    assert_eq!(ts.system_font_loads(), 0, "the bundled fonts cover these strings; no system scan");
}

#[test]
fn weights_and_families_resolve_as_css_matching() {
    let ts = text_system();
    let regular = ts.font_id(&geist(400.)).unwrap();
    let medium = ts.font_id(&geist(500.)).unwrap();
    let semibold = ts.font_id(&geist(600.)).unwrap();
    assert!(regular != medium && medium != semibold && regular != semibold);
    assert_eq!(ts.font_id(&geist(700.)).unwrap(), semibold, "bold falls back to the heaviest face");
    assert_eq!(ts.font_id(&geist(450.)).unwrap(), medium);
    assert_eq!(ts.font_id(&geist(400.)).unwrap(), regular, "ids are stable");
    let mono = ts.font_id(&font("Geist Mono")).unwrap();
    assert!(mono != regular);
    assert!(ts.font_id(&tnum(geist(400.))).unwrap() != regular, "a feature set is its own face entry");
    assert!(ts.all_font_names().iter().any(|n| n == "Geist Mono"));
    let metrics = ts.font_metrics(regular);
    assert!(metrics.units_per_em > 0 && metrics.ascent > 0. && metrics.descent < 0.);
    let m = ts.glyph_for_char(regular, 'm').unwrap();
    assert!(ts.advance(regular, m).unwrap().width > 0.);
    assert_eq!(ts.typographic_bounds(regular, m).unwrap().size.width, ts.advance(regular, m).unwrap().width);
    assert_eq!(ts.system_font_loads(), 0);
}

#[test]
fn tnum_digits_have_equal_advances() {
    let ts = text_system();
    let line = layout(&ts, &tnum(geist(400.)), "0123456789", 13.);
    let xs: Vec<f32> = line.runs[0].glyphs.iter().map(|g| f32::from(g.position.x)).collect();
    assert_eq!(xs.len(), 10);
    let step = xs[1] - xs[0];
    for w in xs.windows(2) {
        assert!((w[1] - w[0] - step).abs() < 1e-3, "unequal tnum advances {xs:?}");
    }
    let proportional = layout(&ts, &geist(400.), "1111", 13.).width;
    let tabular = layout(&ts, &tnum(geist(400.)), "1111", 13.).width;
    assert!(tabular > proportional, "tnum widens the 1s ({tabular:?} vs {proportional:?})");
}

#[test]
fn layout_is_reproducible_and_scales_with_size() {
    let text = "Training run 42 · mAP 0.512 – 3 × 640 px…";
    let a = text_system();
    let b = text_system();
    let first = layout(&a, &geist(500.), text, 14.);
    let again = layout(&a, &geist(500.), text, 14.);
    let other = layout(&b, &geist(500.), text, 14.);
    let positions = |l: &LineLayout| -> Vec<(u32, u32, usize, u32)> {
        l.runs
            .iter()
            .flat_map(|r| {
                r.glyphs.iter().map(|g| {
                    (g.id.0, f32::from(g.position.x).to_bits(), g.index, f32::from(g.position.y).to_bits())
                })
            })
            .collect()
    };
    assert_eq!(positions(&first), positions(&again));
    assert_eq!(positions(&first), positions(&other));
    assert_eq!(first.width, other.width);
    let double = layout(&a, &geist(500.), text, 28.);
    assert!(
        (f32::from(double.width) - 2. * f32::from(first.width)).abs() < 0.5,
        "{:?} vs 2 x {:?}",
        double.width,
        first.width
    );
}

#[test]
fn masks_are_non_empty_with_sane_bounds_and_reproducible() {
    let ts = text_system();
    let id = ts.font_id(&geist(400.)).unwrap();
    let a = ts.glyph_for_char(id, 'A').unwrap();
    let p = params(id, a, 13., 1.5, 0);
    let bounds = ts.glyph_raster_bounds(&p).unwrap();
    let (w, h) = (bounds.size.width.0, bounds.size.height.0);
    // 13 px at 1.5 is 19.5 device px per em: a cap is ~14 px tall and sits on the baseline.
    assert!((8..=20).contains(&h) && (8..=20).contains(&w), "bounds {bounds:?}");
    assert!(
        bounds.origin.y.0 < 0 && bounds.origin.y.0 + h >= -1 && bounds.origin.y.0 + h <= 2,
        "bounds {bounds:?}"
    );
    let (size, mask) = ts.rasterize_glyph(&p, bounds).unwrap();
    assert_eq!(size, bounds.size);
    assert_eq!(mask.len(), (w * h) as usize);
    assert!(mask.contains(&255) || mask.iter().filter(|&&c| c > 128).count() > 10);
    assert!(mask.contains(&0), "a glyph mask has empty corners");
    assert_eq!(ts.rasterize_glyph(&p, bounds).unwrap().1, mask, "rasterizing is reproducible");
    let fresh = text_system();
    let fresh_p = params(fresh.font_id(&geist(400.)).unwrap(), a, 13., 1.5, 0);
    assert_eq!(fresh.rasterize_glyph(&fresh_p, bounds).unwrap().1, mask, "across instances too");

    let space = ts.glyph_for_char(id, ' ').unwrap();
    let blank = params(id, space, 13., 1.5, 0);
    let blank_bounds = ts.glyph_raster_bounds(&blank).unwrap();
    assert!(blank_bounds.size.width.0 == 0 || blank_bounds.size.height.0 == 0);
    assert!(ts.rasterize_glyph(&blank, blank_bounds).is_err(), "empty bounds are an error, as GPUI's");
    assert!(ts.glyph_blend().apply, "GPUI's Linux renderer corrects glyph alpha");
}

#[test]
fn subpixel_variants_differ() {
    let ts = text_system();
    let id = ts.font_id(&geist(400.)).unwrap();
    let glyph = ts.glyph_for_char(id, 'l').unwrap();
    let masks: Vec<(folo_ui::Bounds<DevicePixels>, Vec<u8>)> = (0..folo_ui::SUBPIXEL_VARIANTS_X)
        .map(|v| {
            let p = params(id, glyph, 13., 1.0, v);
            let bounds = ts.glyph_raster_bounds(&p).unwrap();
            (bounds, ts.rasterize_glyph(&p, bounds).unwrap().1)
        })
        .collect();
    for i in 0..masks.len() {
        for j in i + 1..masks.len() {
            assert!(masks[i] != masks[j], "variants {i} and {j} rasterize the same");
        }
    }
}

#[test]
fn a_missing_glyph_loads_system_fonts_lazily() {
    let ts = text_system();
    let _ = layout(&ts, &geist(400.), "plain ASCII", 14.);
    assert_eq!(ts.system_font_loads(), 0);
    let text = "数据集 训练";
    let line = layout(&ts, &geist(400.), text, 14.);
    assert_eq!(ts.system_font_loads(), 1, "a CJK string needs a system font");
    assert_covers(text, &line);
    if cfg!(windows) {
        // Every Windows install has a CJK face (Microsoft YaHei / SimSun); Linux CI may not.
        let ids = glyph_ids(&line);
        assert!(ids.iter().filter(|g| g.0 != 0).count() >= 5, "fallback found no CJK face: {ids:?}");
        assert!(line.runs.len() >= 2 || line.runs[0].font_id != ts.font_id(&geist(400.)).unwrap());
    }
    // Once loaded, never again; and the bundled faces keep their ids.
    let before = ts.font_id(&geist(400.)).unwrap();
    let _ = layout(&ts, &geist(400.), "更多", 14.);
    assert_eq!(ts.system_font_loads(), 1);
    assert_eq!(ts.font_id(&geist(400.)).unwrap(), before);
}

#[test]
fn an_unknown_family_loads_system_fonts_and_errors_if_absent() {
    let ts = text_system();
    assert!(ts.font_id(&font("No Such Family 12345")).is_err());
    assert_eq!(ts.system_font_loads(), 1);
}

#[cfg(windows)]
#[test]
fn an_emoji_gives_a_colour_mask() {
    let ts = text_system();
    let line = layout(&ts, &geist(400.), "ok 😀", 14.);
    let glyph = line
        .runs
        .iter()
        .flat_map(|r| r.glyphs.iter().map(move |g| (r.font_id, g)))
        .find(|(_, g)| g.index == 3);
    let (font_id, glyph) = glyph.expect("a glyph for the emoji");
    assert!(glyph.is_emoji, "Segoe UI Emoji is a known emoji font");
    let p = RenderGlyphParams { is_emoji: true, ..params(font_id, glyph.id, 14., 1.0, 0) };
    let bounds = ts.glyph_raster_bounds(&p).unwrap();
    let (size, bgra) = ts.rasterize_glyph(&p, bounds).unwrap();
    assert_eq!(bgra.len(), (size.width.0 * size.height.0 * 4) as usize);
    let opaque = bgra.as_chunks::<4>().0.iter().filter(|p| p[3] > 200).count();
    let colourful =
        bgra.as_chunks::<4>().0.iter().filter(|p| p[3] > 200 && (p[0] != p[1] || p[1] != p[2])).count();
    assert!(opaque > 0 && colourful > 0, "not a colour image ({opaque} opaque, {colourful} coloured)");
    assert!(
        bgra.as_chunks::<4>().0.iter().all(|p| p[0] <= p[3] && p[1] <= p[3] && p[2] <= p[3]),
        "not premultiplied"
    );
}

/// Start-up cost: `cargo test -p folo-ui --release --features portable-text --test portable_text -- --ignored --nocapture`.
#[test]
#[ignore]
fn startup_cost() {
    let start = Instant::now();
    let ts = text_system();
    let line = layout(&ts, &geist(400.), "Epoch 12 / 300 · box 1.234 · cls 0.567 –", 13.);
    assert_eq!(ts.system_font_loads(), 0);
    let created = start.elapsed();
    let fallback_start = Instant::now();
    let _ = layout(&ts, &geist(400.), "数据集", 13.);
    let fallback = fallback_start.elapsed();
    assert_eq!(ts.system_font_loads(), 1);
    println!(
        "create + six Geist fonts + one {}-glyph line: {:.2} ms; first fallback (system font load + reshape): {:.1} ms",
        line.runs.iter().map(|r| r.glyphs.len()).sum::<usize>(),
        created.as_secs_f64() * 1e3,
        fallback.as_secs_f64() * 1e3
    );
}

#[test]
fn rtl_filename_with_english_reason_wraps_and_keeps_both_inks() {
    use folo_ui::text::GlyphPainter;
    use folo_ui::{
        Bounds, Hsla, Pixels, StrikethroughStyle, TextAlign, TextRun, TextSystem, UnderlineStyle,
        WindowTextSystem, blue, red, size,
    };

    struct Recorder {
        glyphs: Vec<(Point<Pixels>, Hsla)>,
    }

    impl GlyphPainter for Recorder {
        fn paint_glyph(
            &mut self,
            origin: Point<Pixels>,
            _font_id: FontId,
            _glyph_id: GlyphId,
            _font_size: Pixels,
            color: Hsla,
        ) -> anyhow::Result<()> {
            self.glyphs.push((origin, color));
            Ok(())
        }
        fn paint_emoji(&mut self, _: Point<Pixels>, _: FontId, _: GlyphId, _: Pixels) -> anyhow::Result<()> {
            Ok(())
        }
        fn paint_underline(&mut self, _: Point<Pixels>, _: Pixels, _: &UnderlineStyle) {}
        fn paint_strikethrough(&mut self, _: Point<Pixels>, _: Pixels, _: &StrikethroughStyle) {}
        fn paint_background(&mut self, _: Bounds<Pixels>, _: Hsla) {}
    }

    let ts = Arc::new(TextSystem::new(Arc::new(text_system())));
    let wts = WindowTextSystem::new(ts.clone());
    let file = "אx.json ";
    let reason = "invalid option with detailed reason text that wraps cleanly across multiple lines";
    let combined = format!("{file}{reason}");
    let name_color = blue();
    let error_color = red();
    let runs = [
        TextRun {
            len: file.len(),
            font: font("Geist Mono"),
            color: name_color,
            background_color: None,
            underline: None,
            strikethrough: None,
        },
        TextRun {
            len: reason.len(),
            font: geist(400.),
            color: error_color,
            background_color: None,
            underline: None,
            strikethrough: None,
        },
    ];
    let wrap_w = px(220.);
    let wrapped = wts.shape_text(combined.clone().into(), px(12.), &runs, Some(wrap_w), None).unwrap();
    assert_eq!(wrapped.len(), 1);
    let line = &wrapped[0];

    // Glyphs start cleanly at x=0 (never -epsilon) and are in ascending visual left-to-right order,
    // so wrapping triggers and nothing clips past wrap_w.
    let xs: Vec<Pixels> = line.runs().iter().flat_map(|r| r.glyphs.iter().map(|g| g.position.x)).collect();
    assert_eq!(xs.first().copied(), Some(px(0.)), "line must start at exact x=0: {xs:?}");
    assert!(xs.windows(2).all(|w| w[0] <= w[1]), "expected non-decreasing visual x positions: {xs:?}");
    assert!(!line.wrap_boundaries().is_empty(), "long reason after RTL filename must wrap");

    let mut rec = Recorder { glyphs: Vec::new() };
    folo_ui::text::paint_line(
        Point { x: px(0.), y: px(0.) },
        &line.unwrapped_layout,
        px(16.),
        TextAlign::Left,
        Some(wrap_w),
        line.decoration_runs(),
        line.wrap_boundaries(),
        &ts,
        Bounds::new(Point { x: px(0.), y: px(0.) }, size(px(1000.), px(1000.))),
        &mut rec,
    )
    .unwrap();

    // Every glyph from `file` (including 'א' at byte 0 and "x.json ") must paint in `name_color`,
    // and every glyph from `reason` in `error_color`.
    let expected_name =
        line.runs().iter().flat_map(|r| r.glyphs.iter()).filter(|g| g.index < file.len()).count();
    let expected_error =
        line.runs().iter().flat_map(|r| r.glyphs.iter()).filter(|g| g.index >= file.len()).count();
    let name_glyphs: Vec<_> = rec.glyphs.iter().filter(|(_, c)| *c == name_color).collect();
    let error_glyphs: Vec<_> = rec.glyphs.iter().filter(|(_, c)| *c == error_color).collect();
    assert_eq!(name_glyphs.len(), expected_name, "all filename glyphs (including א) must keep name_color");
    assert_eq!(error_glyphs.len(), expected_error, "all reason glyphs must keep error_color");
    // Row order: LTR paragraph base direction puts the entire filename and the start of the reason
    // on row 0 (same baseline y), and wraps the continuation of the reason onto later rows.
    let first_y = name_glyphs[0].0.y;
    assert!(
        name_glyphs.iter().all(|(p, _)| p.y == first_y),
        "RTL filename must sit on the first visual row: {name_glyphs:?}"
    );
    assert!(
        error_glyphs.iter().any(|(p, _)| p.y == first_y),
        "reason must begin on row 0 beside the filename"
    );
    assert!(error_glyphs.iter().any(|(p, _)| p.y > first_y), "long reason must wrap onto subsequent rows");
    let max_x = rec.glyphs.iter().map(|(p, _)| p.x).fold(px(0.), Pixels::max);
    assert!(
        max_x <= wrap_w + px(12.),
        "wrapped glyphs must stay within wrap_w ({wrap_w:?}), max_x={max_x:?}"
    );

    // Pure RTL text keeps distinct caret positions for every character boundary and starts at x=0.
    let rtl = wts.shape_line(
        "אבג".into(),
        px(14.),
        &[TextRun {
            len: "אבג".len(),
            font: geist(400.),
            color: name_color,
            background_color: None,
            underline: None,
            strikethrough: None,
        }],
        None,
    );
    let rtl_xs: Vec<Pixels> = rtl.runs.iter().flat_map(|r| r.glyphs.iter().map(|g| g.position.x)).collect();
    assert_eq!(rtl_xs.first().copied(), Some(px(0.)), "RTL line must start at exact x=0: {rtl_xs:?}");
    let positions = [rtl.x_for_index(0), rtl.x_for_index(2), rtl.x_for_index(4), rtl.x_for_index(6)];
    assert!(
        positions[0] != positions[1] && positions[1] != positions[2] && positions[2] != positions[3],
        "RTL caret positions must be distinct: {positions:?}"
    );
}
