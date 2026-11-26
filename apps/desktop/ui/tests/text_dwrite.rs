//! The DirectWrite backend against the app's bundled Geist fonts: layout covers the text, `tnum`,
//! stable and scaling widths, glyph masks and subpixel variants, colour emoji, and a start that never
//! touches the system font collection.
#![cfg(windows)]

use folo_ui::text::dwrite::DirectWriteTextSystem;
use folo_ui::{
    DevicePixels, Font, FontFeatures, FontId, FontRun, FontWeight, LineLayout, Pixels, PlatformTextSystem,
    Point, RenderGlyphParams, TextSystem, font, px,
};
use std::borrow::Cow;
use std::sync::Arc;

const FONTS: [&[u8]; 6] = [
    include_bytes!("../../../../design/fonts/Geist-Regular.ttf"),
    include_bytes!("../../../../design/fonts/Geist-Medium.ttf"),
    include_bytes!("../../../../design/fonts/Geist-SemiBold.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-Regular.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-Medium.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-SemiBold.ttf"),
];

fn f(pixels: Pixels) -> f32 {
    pixels.into()
}

fn system() -> DirectWriteTextSystem {
    let ts = DirectWriteTextSystem::new().expect("DirectWrite");
    ts.add_fonts(FONTS.iter().map(|f| Cow::Borrowed(*f)).collect()).expect("bundled fonts");
    ts
}

fn geist(weight: f32) -> Font {
    Font { weight: FontWeight(weight), ..font("Geist") }
}

fn tnum(mut font: Font) -> Font {
    font.features = FontFeatures(Arc::new(vec![("tnum".into(), 1)]));
    font
}

fn layout(ts: &DirectWriteTextSystem, font_id: FontId, text: &str, size: f32) -> LineLayout {
    ts.layout_line(text, px(size), &[FontRun { len: text.len(), font_id }])
}

/// Every glyph's index is a character boundary, indices never go back, the first is 0, and every
/// character starts a cluster or sits inside one (no character is skipped by more than a cluster).
fn assert_covers(text: &str, layout: &LineLayout) {
    let indices: Vec<usize> = layout.runs.iter().flat_map(|r| r.glyphs.iter().map(|g| g.index)).collect();
    assert!(!indices.is_empty(), "{text:?} has glyphs");
    assert_eq!(indices[0], 0, "{text:?} starts at 0");
    for pair in indices.windows(2) {
        assert!(pair[0] <= pair[1], "{text:?}: indices go back: {indices:?}");
    }
    for &ix in &indices {
        assert!(ix < text.len() && text.is_char_boundary(ix), "{text:?}: {ix} is not a character boundary");
    }
    assert_eq!(layout.len, text.len());
    assert!(layout.width > px(0.));
}

#[test]
fn layout_covers_text_on_character_boundaries() {
    let ts = system();
    let regular = ts.font_id(&geist(400.)).unwrap();
    for text in
        ["Hello, FOLO 0123", "best F1 · conf 0.412", "1024 × 640", "café crème", "epochs 91–100", "last…"]
    {
        let line = layout(&ts, regular, text, 13.);
        assert_covers(text, &line);
        assert!(line.runs.iter().all(|r| r.font_id == regular), "{text:?} needs no fallback");
        assert_eq!(line.runs.iter().map(|r| r.glyphs.len()).sum::<usize>(), text.chars().count());
        assert!(line.ascent > px(0.) && line.descent > px(0.));
    }
    assert_eq!(ts.system_collection_loads(), 0, "the bundled fonts cover Latin text and its punctuation");

    // Geist has no CJK: DirectWrite falls back to a system font, which gets a font id on the fly.
    let cjk = "漢字のラベル · 数据集";
    let line = layout(&ts, regular, cjk, 13.);
    assert_covers(cjk, &line);
    let fallback: Vec<_> = line.runs.iter().filter(|r| r.font_id != regular).collect();
    assert!(!fallback.is_empty(), "a fallback run");
    assert!(fallback.iter().flat_map(|r| &r.glyphs).all(|g| g.id.0 != 0), "no .notdef glyphs");
    assert!(ts.system_collection_loads() > 0, "the fallback face was looked up in the system collection");
}

#[test]
fn runs_of_several_fonts_cover_the_text() {
    let ts = system();
    let regular = ts.font_id(&geist(400.)).unwrap();
    let semibold = ts.font_id(&geist(600.)).unwrap();
    let mono = ts.font_id(&font("Geist Mono")).unwrap();
    assert!(regular != semibold && semibold != mono);
    let text = "é·loss 0.25 × lr";
    let split = [0, 2, 7, text.len()];
    let runs = [
        FontRun { len: split[1] - split[0], font_id: semibold },
        FontRun { len: split[2] - split[1], font_id: regular },
        FontRun { len: split[3] - split[2], font_id: mono },
    ];
    let line = ts.layout_line(text, px(13.), &runs);
    assert_covers(text, &line);
    let fonts: Vec<FontId> = line.runs.iter().map(|r| r.font_id).collect();
    assert_eq!(fonts, [semibold, regular, mono]);
    assert_eq!(line.runs[1].glyphs[0].index, 2);
    assert_eq!(line.runs[2].glyphs[0].index, 7);
}

#[test]
fn tnum_digits_have_equal_advances() {
    let ts = system();
    let id = ts.font_id(&tnum(geist(500.))).unwrap();
    let line = layout(&ts, id, "0123456789", 13.);
    let xs: Vec<f32> = line.runs[0].glyphs.iter().map(|g| f(g.position.x)).chain([f(line.width)]).collect();
    let advances: Vec<f32> = xs.windows(2).map(|w| w[1] - w[0]).collect();
    for advance in &advances {
        assert!((advance - advances[0]).abs() < 1e-4, "tabular digits: {advances:?}");
    }
    // Without the feature Geist's digits are proportional.
    let plain = ts.font_id(&geist(500.)).unwrap();
    assert_ne!(plain, id, "features make another font id");
    let one = layout(&ts, plain, "1", 13.).width;
    let zero = layout(&ts, plain, "0", 13.).width;
    assert_ne!(one, zero, "proportional 1 and 0");
}

#[test]
fn widths_are_stable_and_scale_with_the_size() {
    let ts = system();
    let id = ts.font_id(&geist(400.)).unwrap();
    let text = "Training run · epoch 12 of 300 — val mAP 0.412";
    let a = layout(&ts, id, text, 13.);
    let b = layout(&ts, id, text, 13.);
    assert_eq!(f(a.width).to_bits(), f(b.width).to_bits());
    let positions = |l: &LineLayout| -> Vec<u32> {
        l.runs.iter().flat_map(|r| r.glyphs.iter().map(|g| f(g.position.x).to_bits())).collect()
    };
    assert_eq!(positions(&a), positions(&b));
    for scale in [2., 1.5] {
        let big = layout(&ts, id, text, 13. * scale);
        assert!(
            (f(big.width) - f(a.width) * scale).abs() < 0.01,
            "{} vs {} x {scale}",
            f(big.width),
            f(a.width)
        );
    }
}

#[test]
fn metrics_and_the_text_system_helpers() {
    let platform = Arc::new(system());
    let ts = TextSystem::new(platform.clone());
    let id = ts.resolve_font(&geist(400.));
    let metrics = platform.font_metrics(id);
    assert_eq!(metrics.units_per_em, 1000);
    assert!(metrics.ascent > 0. && metrics.descent < 0. && metrics.cap_height > metrics.x_height);
    // A character's advance is its one-glyph line width.
    let advance = ts.advance(id, px(13.), 'm').unwrap().width;
    let width = layout(&platform, id, "m", 13.).width;
    assert!((f(advance) - f(width)).abs() < 1e-4, "{advance:?} vs {width:?}");
    assert_eq!(ts.em_advance(id, px(13.)).unwrap(), advance);
    assert!(ts.ch_width(id, px(13.)).unwrap() > px(0.));
    assert!(ts.bounding_box(id, px(13.)).size.height > ts.ascent(id, px(13.)));
    assert!(ts.glyph_blend().apply, "DirectWrite corrects glyph alpha");
    assert_eq!(platform.system_collection_loads(), 0);
}

fn params(
    ts: &DirectWriteTextSystem,
    font_id: FontId,
    ch: char,
    size: f32,
    scale: f32,
    x: u8,
) -> RenderGlyphParams {
    RenderGlyphParams {
        font_id,
        glyph_id: ts.glyph_for_char(font_id, ch).unwrap(),
        font_size: px(size),
        subpixel_variant: Point { x, y: 0 },
        scale_factor: scale,
        is_emoji: false,
    }
}

#[test]
fn a_glyph_mask_fills_its_raster_bounds() {
    let ts = system();
    let id = ts.font_id(&geist(400.)).unwrap();
    let p = params(&ts, id, 'A', 13., 1.5, 0);
    let bounds = ts.glyph_raster_bounds(&p).unwrap();
    assert!(bounds.size.width.0 > 0 && bounds.size.height.0 > 0);
    // "A" stands on the baseline: its raster starts above it and ends at about it.
    let cap = ts.font_metrics(id).cap_height / 1000. * 13. * 1.5;
    assert!(bounds.origin.y.0 < 0 && (bounds.origin.y.0 as f32 + cap).abs() <= 1.5, "{bounds:?}, cap {cap}");
    assert!((bounds.origin.y.0 + bounds.size.height.0).abs() <= 1);
    let (size, mask) = ts.rasterize_glyph(&p, bounds).unwrap();
    assert_eq!(size, bounds.size);
    let (w, h) = (size.width.0 as usize, size.height.0 as usize);
    assert_eq!(mask.len(), w * h);
    // The bounds are tight: coverage reaches the first and last rows and columns.
    let row = |y: usize| mask[y * w..(y + 1) * w].iter().any(|&a| a > 0);
    let col = |x: usize| (0..h).any(|y| mask[y * w + x] > 0);
    assert!(row(0) && row(h - 1) && col(0) && col(w - 1), "tight bounds");
    assert!(mask.contains(&255), "solid stems");
    // A space has no raster.
    let space = params(&ts, id, ' ', 13., 1.5, 0);
    assert_eq!(ts.glyph_raster_bounds(&space).unwrap().size.width, DevicePixels(0));
}

#[test]
fn subpixel_variants_give_different_masks() {
    let ts = system();
    let id = ts.font_id(&geist(400.)).unwrap();
    let masks: Vec<_> = (0..4)
        .map(|x| {
            let p = params(&ts, id, 'l', 13., 1.0, x);
            let bounds = ts.glyph_raster_bounds(&p).unwrap();
            (bounds, ts.rasterize_glyph(&p, bounds).unwrap().1)
        })
        .collect();
    for a in 0..4 {
        for b in a + 1..4 {
            assert_ne!(masks[a], masks[b], "variants {a} and {b} differ");
        }
    }
}

#[test]
fn an_emoji_gives_a_colour_mask() {
    let ts = system();
    let regular = ts.font_id(&geist(400.)).unwrap();
    let line = layout(&ts, regular, "ok 😀", 16.);
    let glyph = line
        .runs
        .iter()
        .flat_map(|r| r.glyphs.iter().map(move |g| (r.font_id, g)))
        .find(|(_, g)| g.index == 3);
    let (font_id, glyph) = glyph.expect("the emoji's glyph");
    assert!(glyph.is_emoji, "Segoe UI Emoji's glyph is a colour glyph");
    assert_ne!(font_id, regular);
    let p = RenderGlyphParams {
        font_id,
        glyph_id: glyph.id,
        font_size: px(16.),
        subpixel_variant: Point::default(),
        scale_factor: 1.5,
        is_emoji: true,
    };
    let bounds = ts.glyph_raster_bounds(&p).unwrap();
    let (size, bgra) = ts.rasterize_glyph(&p, bounds).unwrap();
    assert_eq!(bgra.len(), size.width.0 as usize * size.height.0 as usize * 4);
    let pixels: &[[u8; 4]] = bgra.as_chunks::<4>().0;
    assert!(pixels.iter().all(|p| p[0] <= p[3] && p[1] <= p[3] && p[2] <= p[3]), "premultiplied");
    assert!(pixels.iter().any(|p| p[3] > 200 && p[2] > p[0] + 40), "a warm, opaque face");
    assert!(pixels.iter().any(|p| p[3] == 0), "transparent corners");
}

#[test]
fn start_and_first_layout_do_not_touch_the_system_collection() {
    let ts = system();
    for f in [
        geist(400.),
        geist(500.),
        geist(600.),
        tnum(geist(500.)),
        font("Geist Mono"),
        tnum(font("Geist Mono")),
    ] {
        ts.font_id(&f).unwrap();
    }
    let id = ts.font_id(&geist(400.)).unwrap();
    let line = layout(&ts, id, "Datasets · 1,204 images — 98.2% labeled", 13.);
    for glyph in &line.runs[0].glyphs {
        let p = RenderGlyphParams { glyph_id: glyph.id, ..params(&ts, id, 'a', 13., 1.25, 1) };
        let bounds = ts.glyph_raster_bounds(&p).unwrap();
        if bounds.size.width.0 > 0 {
            ts.rasterize_glyph(&p, bounds).unwrap();
        }
    }
    ts.glyph_blend();
    assert_eq!(ts.system_collection_loads(), 0);
    // Asking for a family the app did not bundle loads it, once.
    let segoe = ts.font_id(&font("Segoe UI")).unwrap();
    assert_eq!(ts.system_collection_loads(), 1);
    assert!(layout(&ts, segoe, "Segoe", 13.).width > px(0.));
    assert!(ts.all_font_names().iter().any(|n| n == "Geist"));
    assert_eq!(ts.system_collection_loads(), 1);
}

#[test]
fn an_unknown_family_falls_back_to_the_system_ui_font() {
    let ts = system();
    let id = ts.font_id(&font("No Such Family 1234")).unwrap();
    assert!(layout(&ts, id, "fallback", 13.).width > px(0.));
    let ui = ts.font_id(&font(".SystemUIFont")).unwrap();
    assert_eq!(layout(&ts, id, "fallback", 13.).width, layout(&ts, ui, "fallback", 13.).width);
}

#[test]
fn rtl_and_bidi_runs_preserve_all_glyphs_and_visual_positions() {
    let ts = system();
    let regular = ts.font_id(&geist(400.)).unwrap();

    // Pure Hebrew ("שלום", 4 chars = 8 UTF-8 bytes at offsets 0, 2, 4, 6): all 4 glyphs present in
    // visual left-to-right order (ascending x, descending logical byte indices).
    let hebrew = "שלום";
    let line = layout(&ts, regular, hebrew, 14.);
    let glyphs: Vec<_> = line.runs.iter().flat_map(|r| r.glyphs.iter()).collect();
    assert_eq!(glyphs.len(), 4, "all 4 Hebrew letters must be shaped: {glyphs:?}");
    let indices: Vec<usize> = glyphs.iter().map(|g| g.index).collect();
    assert_eq!(indices, vec![6, 4, 2, 0], "RTL glyphs map to descending logical UTF-8 byte offsets");
    let xs: Vec<f32> = glyphs.iter().map(|g| f(g.position.x)).collect();
    assert!(xs.windows(2).all(|w| w[0] < w[1]), "visual positions must strictly increase: {xs:?}");
    assert!(line.width > px(0.));
    // RTL caret geometry: logical start (0) is the right edge, logical end (8) is the left edge,
    // and logical offsets 0, 2, 4, 6, 8 walk strictly right-to-left so selections span the right glyphs.
    let caret_xs: Vec<f32> = [0, 2, 4, 6, hebrew.len()].iter().map(|&ix| f(line.x_for_index(ix))).collect();
    assert_eq!(caret_xs[0], f(line.width), "RTL logical start sits at the right edge: {caret_xs:?}");
    assert_eq!(caret_xs[4], 0.0, "RTL logical end sits at the left edge: {caret_xs:?}");
    assert!(
        caret_xs.windows(2).all(|w| w[0] > w[1]),
        "RTL logical carets walk strictly right-to-left: {caret_xs:?}"
    );
    // Hit-testing round-trips with caret positioning on both sides of the RTL word.
    assert_eq!(line.closest_index_for_x(line.width), 0, "clicking right side of RTL maps to logical start");
    assert_eq!(
        line.closest_index_for_x(px(0.)),
        hebrew.len(),
        "clicking left side of RTL maps to logical end"
    );

    // Trailing RTL run ("hello שלום"): logical end sits at the left edge of the Hebrew run, and
    // Select All covers the full visual width [0..line.width].
    let trailing_rtl = "hello שלום";
    let trailing_line = layout(&ts, regular, trailing_rtl, 14.);
    let heb_left = trailing_line
        .runs
        .iter()
        .flat_map(|r| r.glyphs.iter())
        .find(|g| g.index >= 6)
        .map(|g| f(g.position.x))
        .unwrap();
    assert_eq!(
        f(trailing_line.x_for_index(trailing_rtl.len())),
        heb_left,
        "logical end of a trailing RTL run sits at its visual left edge"
    );
    let all_spans = trailing_line.selection_spans(0..trailing_rtl.len());
    assert_eq!(all_spans.len(), 1, "Select All on mixed text merges into one full-width span");
    assert_eq!(f(all_spans[0].0), 0.0);
    assert!((f(all_spans[0].1) - f(trailing_line.width)).abs() < 0.5);
    // Clicking the left half of 'ם' (leftmost Hebrew glyph) reaches logical end (14), while
    // clicking the right half of the preceding space reaches byte 6.
    assert_eq!(
        trailing_line.closest_index_for_x(px(heb_left + 0.5)),
        trailing_rtl.len(),
        "clicking left half of leftmost Hebrew glyph reaches logical end"
    );
    assert_eq!(
        trailing_line.closest_index_for_x(px(heb_left - 0.5)),
        6,
        "clicking right half of space before Hebrew reaches byte 6"
    );

    // Pure Arabic ("مرحبا", 5 chars): must not collapse to a single glyph; every index is a valid
    // char boundary and visual positions advance left-to-right.
    let arabic = "مرحبا";
    let line = layout(&ts, regular, arabic, 14.);
    let glyphs: Vec<_> = line.runs.iter().flat_map(|r| r.glyphs.iter()).collect();
    assert!(glyphs.len() >= 4, "Arabic word must retain its shaped glyphs, got {glyphs:?}");
    assert!(
        glyphs.iter().all(|g| g.index < arabic.len() && arabic.is_char_boundary(g.index)),
        "Arabic glyph indices must be valid char boundaries"
    );
    let xs: Vec<f32> = glyphs.iter().map(|g| f(g.position.x)).collect();
    assert!(xs.windows(2).all(|w| w[0] <= w[1]), "Arabic visual positions advance left-to-right: {xs:?}");
    assert!(line.width > px(0.));

    // Mixed LTR + RTL + LTR ("hello שלום world"): all 16 characters shaped, Hebrew middle run has
    // all 4 logical offsets (6, 8, 10, 12) in visual order [12, 10, 8, 6], and x increases monotonically.
    let mixed = "hello שלום world";
    let line = layout(&ts, regular, mixed, 14.);
    let glyphs: Vec<_> = line.runs.iter().flat_map(|r| r.glyphs.iter()).collect();
    assert_eq!(glyphs.len(), mixed.chars().count(), "mixed bidi text must keep every character");
    assert!(
        glyphs.iter().all(|g| g.index < mixed.len() && mixed.is_char_boundary(g.index)),
        "every mixed bidi glyph index must be a UTF-8 char boundary"
    );
    let hebrew_indices: Vec<usize> =
        glyphs.iter().map(|g| g.index).filter(|&ix| (6..14).contains(&ix)).collect();
    assert_eq!(hebrew_indices, vec![12, 10, 8, 6], "embedded Hebrew run is laid out visually RTL");
    let xs: Vec<f32> = glyphs.iter().map(|g| f(g.position.x)).collect();
    assert!(xs.windows(2).all(|w| w[0] < w[1]), "mixed bidi visual x must strictly increase: {xs:?}");
}
