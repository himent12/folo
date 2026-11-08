//! `folo://runs/{id}/samples/{epoch}`: the validation samples of an epoch as one annotated JPEG grid (ground truth
//! outlined white, predictions in the class colours of the design system), so a multimodal agent can eyeball a run.

use std::io::Cursor;

use folo_api::BBox;
use folo_server::Store;
use image::{Rgb, RgbImage};

use crate::errors::Error;

/// Cell size of the grid in pixels.
const TILE: u32 = 256;
/// Cells per row.
const COLS: u32 = 4;
/// Outline colour of ground-truth boxes.
const TRUTH: Rgb<u8> = Rgb([255, 255, 255]);
/// `--class-0` ... `--class-11` of `web/src/lib/design/tokens.css`.
const CLASS_COLORS: [[u8; 3]; 12] = [
    [0x4c, 0xc9, 0xf0],
    [0xf7, 0x25, 0x85],
    [0xff, 0xb7, 0x03],
    [0x80, 0xed, 0x99],
    [0x9d, 0x7c, 0xff],
    [0xff, 0x7b, 0x39],
    [0x2e, 0xc4, 0xb6],
    [0xe4, 0x5c, 0xff],
    [0xf2, 0xf2, 0xf2],
    [0x5b, 0x8c, 0xff],
    [0xff, 0xd1, 0x66],
    [0xff, 0x5d, 0x8f],
];

/// Draws a `thick`-pixel rectangle outline (clipped to the image).
fn outline(img: &mut RgbImage, b: &BBox, scale: f32, ox: u32, oy: u32, color: Rgb<u8>, thick: i64) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x1 = ox as i64 + (b.x1 * scale).round() as i64;
    let y1 = oy as i64 + (b.y1 * scale).round() as i64;
    let x2 = ox as i64 + (b.x2 * scale).round() as i64;
    let y2 = oy as i64 + (b.y2 * scale).round() as i64;
    let mut put = |x: i64, y: i64| {
        if (0..w).contains(&x) && (0..h).contains(&y) {
            img.put_pixel(x as u32, y as u32, color);
        }
    };
    for t in 0..thick {
        for x in x1..=x2 {
            put(x, y1 + t);
            put(x, y2 - t);
        }
        for y in y1..=y2 {
            put(x1 + t, y);
            put(x2 - t, y);
        }
    }
}

/// Renders the grid for `run_id` at `epoch` as JPEG bytes.
pub fn sample_grid(store: &dyn Store, run_id: &str, epoch: u32) -> Result<Vec<u8>, Error> {
    let samples = store.run_samples(run_id, Some(epoch))?.samples;
    let n = samples.len().max(1) as u32;
    let rows = n.div_ceil(COLS);
    let mut grid = RgbImage::from_pixel(COLS.min(n) * TILE, rows * TILE, Rgb([18, 18, 22]));
    for (i, s) in samples.iter().enumerate() {
        let (cx, cy) = ((i as u32 % COLS) * TILE, (i as u32 / COLS) * TILE);
        let scale = TILE as f32 / s.width.max(s.height).max(1) as f32;
        let (tw, th) = ((s.width as f32 * scale).round() as u32, (s.height as f32 * scale).round() as u32);
        let (ox, oy) = (cx + (TILE - tw.min(TILE)) / 2, cy + (TILE - th.min(TILE)) / 2);
        if let Ok(data) = store.image(&s.dataset_id, &s.image_id, Some(TILE))
            && let Ok(img) = image::load_from_memory(&data.bytes)
        {
            let img = img.resize_exact(tw.max(1), th.max(1), image::imageops::FilterType::Triangle).to_rgb8();
            image::imageops::replace(&mut grid, &img, ox as i64, oy as i64);
        }
        for b in &s.truth {
            outline(&mut grid, b, scale, ox, oy, TRUTH, 1);
        }
        for p in &s.pred {
            let b = BBox { class: p.class, x1: p.x1, y1: p.y1, x2: p.x2, y2: p.y2 };
            let color = Rgb(CLASS_COLORS[p.class as usize % CLASS_COLORS.len()]);
            outline(&mut grid, &b, scale, ox, oy, color, 2);
        }
    }
    let mut out = Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85)
        .encode_image(&grid)
        .map_err(|e| Error::internal(format!("encoding the sample grid: {e}")))?;
    Ok(out.into_inner())
}
