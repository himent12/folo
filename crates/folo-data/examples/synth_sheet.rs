//! Renders a contact sheet of synthetic shapes images (optionally with their boxes drawn) to a PNG.
//!
//! `cargo run --release -p folo-data --example synth_sheet -- out.png [seed] [--boxes] [--augment] [--all]`

use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{
    AugmentConfig, Dataset, Image, Rng, Sample, Split, TrainTransform, letterbox_sample, write_image,
};

fn draw_boxes(s: &mut Sample) {
    let palette = [[255, 64, 64], [64, 220, 64], [64, 128, 255], [255, 200, 0], [255, 0, 255], [0, 230, 230]];
    let (w, h) = (s.image.width, s.image.height);
    for b in &s.boxes {
        let [x1, y1, x2, y2] = b.pixel_xyxy(w, h).map(|v| v.round() as i64);
        let c = palette[b.class as usize % palette.len()];
        s.image.fill_rect(x1, y1, x2, y1 + 1, c);
        s.image.fill_rect(x1, y2 - 1, x2, y2, c);
        s.image.fill_rect(x1, y1, x1 + 1, y2, c);
        s.image.fill_rect(x2 - 1, y1, x2, y2, c);
    }
}

fn main() -> folo_data::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).map_or("synth_sheet.png", String::as_str);
    let seed = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let boxes = args.iter().any(|a| a == "--boxes");
    let augment = args.iter().any(|a| a == "--augment");
    let all = args.iter().any(|a| a == "--all");
    let mut cfg = SyntheticConfig { images: 16, imgsz: 320, seed, val_fraction: 0.0, ..Default::default() };
    if all {
        cfg.classes = folo_data::synthetic::SHAPES.iter().map(|s| s.to_string()).collect();
    }
    let ds = SyntheticDataset::new(cfg, Split::Train)?;
    let (cols, cell) = (4u32, 320u32);
    let mut sheet = Image::new(cols * cell, 4 * cell, [30, 30, 30]);
    let transform = TrainTransform::train(cell, AugmentConfig::default(), 0, 0);
    for i in 0..16 {
        let mut s = if augment {
            let mut rng = Rng::new(seed ^ i as u64);
            transform.apply(i, ds.len(), &|j| ds.get(j), &mut rng)?.sample
        } else {
            letterbox_sample(&ds.get(i)?, cell, cell, &Default::default()).0
        };
        if boxes {
            draw_boxes(&mut s);
        }
        let (ox, oy) = ((i as u32 % cols) * cell, (i as u32 / cols) * cell);
        for y in 0..cell {
            let src = &s.image.data[(y * cell * 3) as usize..((y + 1) * cell * 3) as usize];
            let o = (((oy + y) * sheet.width + ox) * 3) as usize;
            sheet.data[o..o + src.len()].copy_from_slice(src);
        }
    }
    write_image(&sheet, std::path::Path::new(out), folo_data::ImageFormat::Png)?;
    println!("wrote {out}");
    Ok(())
}
