//! Start-up cost of the DirectWrite backend, on a fresh process (this file is its own test binary with
//! one test): create the backend, add the six Geist fonts, lay out and rasterize one 40-character line.
//! Then the cost GPUI pays at start and this backend avoids: loading the system font collection.
//!
//! `cargo test -p folo-ui --test text_dwrite_startup -- --ignored --nocapture`
#![cfg(windows)]

use folo_ui::text::dwrite::DirectWriteTextSystem;
use folo_ui::{FontRun, PlatformTextSystem, Point, RenderGlyphParams, SUBPIXEL_VARIANTS_X, font, px};
use std::borrow::Cow;
use std::time::Instant;

const FONTS: [&[u8]; 6] = [
    include_bytes!("../../../../design/fonts/Geist-Regular.ttf"),
    include_bytes!("../../../../design/fonts/Geist-Medium.ttf"),
    include_bytes!("../../../../design/fonts/Geist-SemiBold.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-Regular.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-Medium.ttf"),
    include_bytes!("../../../../design/fonts/GeistMono-SemiBold.ttf"),
];

#[test]
#[ignore = "a manual measurement: run on a fresh process with --ignored --nocapture"]
fn startup_cost() {
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();

    let t = Instant::now();
    let ts = DirectWriteTextSystem::new().unwrap();
    let create = ms(t);

    let t = Instant::now();
    ts.add_fonts(FONTS.iter().map(|f| Cow::Borrowed(*f)).collect()).unwrap();
    let add_fonts = ms(t);

    let t = Instant::now();
    let id = ts.font_id(&font("Geist")).unwrap();
    let text = "Datasets · 1,204 images — 98.2% labeled.";
    assert_eq!(text.chars().count(), 40);
    let line = ts.layout_line(text, px(13.), &[FontRun { len: text.len(), font_id: id }]);
    let layout = ms(t);

    let t = Instant::now();
    let scale = 1.5;
    let mut pixels = 0;
    for glyph in &line.runs[0].glyphs {
        let x = f32::from(glyph.position.x) * scale;
        let params = RenderGlyphParams {
            font_id: id,
            glyph_id: glyph.id,
            font_size: px(13.),
            subpixel_variant: Point { x: (x.fract() * SUBPIXEL_VARIANTS_X as f32).floor() as u8, y: 0 },
            scale_factor: scale,
            is_emoji: false,
        };
        let bounds = ts.glyph_raster_bounds(&params).unwrap();
        if bounds.size.width.0 > 0 && bounds.size.height.0 > 0 {
            pixels += ts.rasterize_glyph(&params, bounds).unwrap().1.len();
        }
    }
    let raster = ms(t);
    let total = ms(start);
    assert_eq!(ts.system_collection_loads(), 0);

    let t = Instant::now();
    let names = ts.all_font_names().len();
    let system_collection = ms(t);

    println!(
        "dwrite start: create {create:.2} ms, add 6 fonts {add_fonts:.2} ms, font id + layout 40 chars {layout:.2} ms, \
         rasterize {} glyphs ({pixels} px) {raster:.2} ms; total {total:.2} ms. \
         Avoided at start: system font collection {system_collection:.2} ms ({names} families).",
        line.runs[0].glyphs.len()
    );
}
