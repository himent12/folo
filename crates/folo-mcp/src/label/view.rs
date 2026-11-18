//! An image of a labeling job as an agent (or a person) checks it: its boxes outlined in class colours and
//! numbered from 1 in list order, and ticks every 100 units (of 1000) along the edges to read coordinates off.

use std::io::Cursor;

use image::{Rgb, RgbImage};

use crate::errors::Error;
use crate::types::label::RelBox;

/// `--class-0` ... `--class-11` of the design system (as the sample grid uses them).
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

/// 3 × 5 digits, one row per entry, bit 2 = the left column.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

/// The colour of class `c`.
pub fn class_color(c: u32) -> [u8; 3] {
    CLASS_COLORS[c as usize % CLASS_COLORS.len()]
}

fn fill(img: &mut RgbImage, x1: i64, y1: i64, x2: i64, y2: i64, c: [u8; 3]) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    for y in y1.max(0)..y2.min(h) {
        for x in x1.max(0)..x2.min(w) {
            img.put_pixel(x as u32, y as u32, Rgb(c));
        }
    }
}

/// Draws `boxes` (0..1000 units) on the JPEG `jpeg` and encodes the result as a JPEG.
pub fn render(jpeg: &[u8], boxes: &[RelBox]) -> Result<Vec<u8>, Error> {
    let mut img = image::load_from_memory(jpeg)
        .map_err(|e| Error::internal(format!("decoding the image to view: {e}")))?
        .to_rgb8();
    draw(&mut img, boxes);
    let mut out = Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 82)
        .encode_image(&img)
        .map_err(|e| Error::internal(format!("encoding the view: {e}")))?;
    Ok(out.into_inner())
}

/// Draws the ticks, then every box with its number (the last box on top).
pub fn draw(img: &mut RgbImage, boxes: &[RelBox]) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let long = w.max(h);
    let px = |v: u32, side: i64| (v.min(1000) as i64 * side + 500) / 1000;
    // Ticks: white on a dark edge, so they show on any background.
    let tick = (long / 60).max(4);
    for k in 1..10 {
        let (x, y) = (px(k * 100, w), px(k * 100, h));
        for (x1, y1, x2, y2) in [
            (x - 1, 0, x + 2, tick),
            (x - 1, h - tick, x + 2, h),
            (0, y - 1, tick, y + 2),
            (w - tick, y - 1, w, y + 2),
        ] {
            fill(img, x1 - 1, y1 - 1, x2 + 1, y2 + 1, [0; 3]);
            fill(img, x1, y1, x2, y2, [255; 3]);
        }
    }
    let t = (long / 400).max(2);
    let scale = (long / 256).clamp(2, 6);
    for (n, b) in boxes.iter().enumerate() {
        let c = class_color(b[0]);
        let (x1, y1, x2, y2) = (px(b[1], w), px(b[2], h), px(b[3], w), px(b[4], h));
        fill(img, x1, y1, x2, y1 + t, c);
        fill(img, x1, y2 - t, x2, y2, c);
        fill(img, x1, y1, x1 + t, y2, c);
        fill(img, x2 - t, y1, x2, y2, c);
        // The number in a tag at the top-left corner: above the box when there is room, else inside it.
        let label = (n + 1).to_string();
        let (gw, gh) = (4 * scale, 5 * scale);
        let (tag_w, tag_h) = (label.len() as i64 * gw + scale, gh + 2 * scale);
        let ty = if y1 - tag_h >= 0 { y1 - tag_h } else { y1 };
        let tx = x1.min(w - tag_w).max(0);
        fill(img, tx, ty, tx + tag_w, ty + tag_h, c);
        let ink = if c[0] as u32 * 299 + c[1] as u32 * 587 + c[2] as u32 * 114 > 140_000 {
            [0; 3]
        } else {
            [255; 3]
        };
        for (i, ch) in label.bytes().enumerate() {
            let g = DIGITS[(ch - b'0') as usize];
            let gx = tx + scale + i as i64 * gw;
            for (row, bits) in g.iter().enumerate() {
                for col in 0..3 {
                    if bits & (0b100 >> col) != 0 {
                        let (x, y) = (gx + col * scale, ty + scale + row as i64 * scale);
                        fill(img, x, y, x + scale, y + scale, ink);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_are_drawn_where_their_units_say() {
        let mut img = RgbImage::from_pixel(400, 200, Rgb([10, 10, 10]));
        draw(&mut img, &[[1, 250, 500, 750, 900]]);
        let c = Rgb(class_color(1));
        // The left edge at x = 0.25 × 400, half way down the box.
        assert_eq!(*img.get_pixel(100, 140), c);
        assert_eq!(*img.get_pixel(299, 140), c);
        // Inside the box nothing is drawn.
        assert_eq!(*img.get_pixel(200, 140), Rgb([10, 10, 10]));
        // The tag with the number sits above the top edge (y = 100).
        assert_eq!(*img.get_pixel(101, 98), c);
        // A tick at x = 100 units on the top edge.
        assert_eq!(*img.get_pixel(40, 1), Rgb([255; 3]));
        let jpeg = {
            let mut out = Cursor::new(Vec::new());
            image::codecs::jpeg::JpegEncoder::new(&mut out).encode_image(&img).unwrap();
            out.into_inner()
        };
        let drawn = render(&jpeg, &[]).unwrap();
        assert_eq!(&drawn[..2], &[0xff, 0xd8]);
        assert_eq!(image::load_from_memory(&drawn).unwrap().width(), 400);
    }
}
