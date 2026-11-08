//! Annotated images for `folo predict --save`: one colour per class, a box outline and a `class score` tag in a
//! tiny built-in digit font (no font files).

use folo_data::Image;
use folo_vision::Detection;

/// Distinct, saturated colours (cycled by class id).
const PALETTE: [[u8; 3]; 10] = [
    [255, 56, 56],
    [255, 157, 151],
    [255, 112, 31],
    [255, 178, 29],
    [207, 210, 49],
    [72, 249, 10],
    [26, 147, 52],
    [0, 212, 187],
    [44, 153, 168],
    [61, 219, 134],
];

/// 3×5 glyphs for `0-9` and `.`, one row per entry, bit 2 = left column.
const GLYPHS: [[u8; 5]; 11] = [
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
    [0b000, 0b000, 0b000, 0b000, 0b010],
];

/// The colour of class `c`.
pub fn class_color(c: u32) -> [u8; 3] {
    PALETTE[c as usize % PALETTE.len()]
}

/// Draw `dets` onto `img`.
pub fn annotate(img: &mut Image, dets: &[Detection]) {
    let short = img.width.min(img.height) as i64;
    let t = (short / 320).clamp(1, 6) + 1;
    let scale = (short / 400).clamp(1, 4);
    // Lowest score first, so the best boxes end up on top.
    for d in dets.iter().rev() {
        let c = class_color(d.class);
        let (x1, y1) = (d.bbox.x1.round() as i64, d.bbox.y1.round() as i64);
        let (x2, y2) = (d.bbox.x2.round() as i64, d.bbox.y2.round() as i64);
        img.fill_rect(x1, y1, x2, y1 + t, c);
        img.fill_rect(x1, y2 - t, x2, y2, c);
        img.fill_rect(x1, y1, x1 + t, y2, c);
        img.fill_rect(x2 - t, y1, x2, y2, c);
        let label = format!("{} {:.2}", d.class, d.score);
        let (gw, gh) = (4 * scale, 6 * scale);
        let tag_w = label.len() as i64 * gw + 2 * scale;
        let tag_h = gh + 2 * scale;
        let ty = if y1 - tag_h >= 0 { y1 - tag_h } else { y1 };
        img.fill_rect(x1, ty, x1 + tag_w, ty + tag_h, c);
        let ink = if c[0] as u32 * 299 + c[1] as u32 * 587 + c[2] as u32 * 114 > 140_000 {
            [0; 3]
        } else {
            [255; 3]
        };
        for (i, ch) in label.chars().enumerate() {
            let g = match ch {
                '0'..='9' => GLYPHS[ch as usize - '0' as usize],
                '.' => GLYPHS[10],
                _ => continue,
            };
            let gx = x1 + scale + i as i64 * gw;
            for (row, bits) in g.iter().enumerate() {
                for col in 0..3 {
                    if bits & (0b100 >> col) != 0 {
                        let px = gx + col * scale;
                        let py = ty + scale + row as i64 * scale;
                        img.fill_rect(px, py, px + scale, py + scale, ink);
                    }
                }
            }
        }
    }
}
