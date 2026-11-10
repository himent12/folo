//! Contact sheets of every condition, for looking at what the severities do to a real photo.
//!
//! ```sh
//! cargo run -p folo-bench --release --example contact_sheet -- <image> <out dir> [<COCO instances.json>]
//! ```
//!
//! Writes `<out dir>/overview.jpg` (one row per condition: severities 1, 3 and 5) and `<out dir>/<id>.jpg` per
//! condition (the unchanged image, then severities 1 to 5, in two rows of three). With a COCO instances file the
//! image's true boxes (found by file name) are drawn in green, as the condition returns them.

use folo_api::BBox;
use folo_bench::conditions::{self, Condition};
use image::RgbImage;
use image::imageops::{FilterType, resize};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [image, out, rest @ ..] = args.as_slice() else {
        eprintln!("usage: contact_sheet <image> <out dir> [<COCO instances.json>]");
        std::process::exit(2);
    };
    let bytes = std::fs::read(image).expect("read the image");
    let img = conditions::decode(&bytes).expect("decode the image");
    let boxes = rest.first().map(|ann| coco_boxes(Path::new(ann), Path::new(image))).unwrap_or_default();
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("create the output folder");
    let seed = 1;

    let variant = |c: Condition, s: u8| -> RgbImage {
        let r =
            conditions::render(&bytes, c, s, &boxes, conditions::variant_seed(seed, image, c, s)).unwrap();
        let mut v = conditions::decode(&r.jpeg).unwrap();
        draw(&mut v, &r.boxes);
        v
    };
    let mut clean = img.clone();
    draw(&mut clean, &boxes);

    // Per condition: unchanged, 1, 2 / 3, 4, 5 at 480 px wide.
    let tile_w = 480;
    let tile_h = (img.height() as f64 * tile_w as f64 / img.width() as f64).round() as u32;
    for c in Condition::ALL {
        let mut tiles = vec![clean.clone()];
        tiles.extend((1..=5).map(|s| variant(c, s)));
        let sheet = grid(&tiles, 3, tile_w, tile_h);
        let path = out.join(format!("{}.jpg", c.id()));
        sheet.save(&path).expect("write the sheet");
        println!("{}", path.display());
        // 1:1 crops of the centre: unchanged, 1, 3, 5.
        let (cw, ch) = (img.width().min(360), img.height().min(270));
        let (cx, cy) = ((img.width() - cw) / 2, (img.height() - ch) / 2);
        let crops: Vec<RgbImage> = [0, 1, 3, 5]
            .iter()
            .map(|&i| image::imageops::crop_imm(&tiles[i], cx, cy, cw, ch).to_image())
            .collect();
        let path = out.join(format!("{}_detail.jpg", c.id()));
        grid(&crops, 2, cw, ch).save(&path).expect("write the detail sheet");
    }
    // Overview: one row per condition, severities 1, 3, 5, at 320 px wide.
    let (ow, oh) = (320, (img.height() as f64 * 320.0 / img.width() as f64).round() as u32);
    let tiles: Vec<RgbImage> =
        Condition::ALL.iter().flat_map(|&c| [1, 3, 5].map(|s| variant(c, s))).collect();
    let path = out.join("overview.jpg");
    grid(&tiles, 3, ow, oh).save(&path).expect("write the overview");
    println!("{}", path.display());
}

/// Tiles in rows of `cols`, each resized to `w x h`, with a 4 px white gap.
fn grid(tiles: &[RgbImage], cols: u32, w: u32, h: u32) -> RgbImage {
    let rows = (tiles.len() as u32).div_ceil(cols);
    let gap = 4;
    let mut sheet =
        RgbImage::from_pixel(cols * (w + gap) + gap, rows * (h + gap) + gap, image::Rgb([255; 3]));
    for (i, t) in tiles.iter().enumerate() {
        let small = resize(t, w, h, FilterType::Triangle);
        let (x, y) = (gap + (i as u32 % cols) * (w + gap), gap + (i as u32 / cols) * (h + gap));
        image::imageops::overlay(&mut sheet, &small, x as i64, y as i64);
    }
    sheet
}

/// Green 2 px outlines.
fn draw(img: &mut RgbImage, boxes: &[BBox]) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    for b in boxes {
        let (x1, y1, x2, y2) = (b.x1 as i64, b.y1 as i64, b.x2 as i64, b.y2 as i64);
        for x in x1.max(0)..x2.min(w) {
            for y in [y1, y1 + 1, y2 - 2, y2 - 1] {
                if (0..h).contains(&y) {
                    img.put_pixel(x as u32, y as u32, image::Rgb([0, 255, 0]));
                }
            }
        }
        for y in y1.max(0)..y2.min(h) {
            for x in [x1, x1 + 1, x2 - 2, x2 - 1] {
                if (0..w).contains(&x) {
                    img.put_pixel(x as u32, y as u32, image::Rgb([0, 255, 0]));
                }
            }
        }
    }
}

/// The non-crowd boxes of the image with this file name in a COCO instances file.
fn coco_boxes(ann: &Path, image: &Path) -> Vec<BBox> {
    let gt = folo_data::read_coco_ground_truth(ann).expect("read the COCO file");
    let name = image.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let Some(im) = gt.images.iter().find(|i| i.file_name == name) else {
        eprintln!("{name} is not in {}", ann.display());
        return Vec::new();
    };
    im.annotations
        .iter()
        .filter(|a| !a.iscrowd)
        .map(|a| {
            let [x, y, w, h] = a.bbox.map(|v| v as f32);
            BBox { class: 0, x1: x, y1: y, x2: x + w, y2: y + h }
        })
        .collect()
}
