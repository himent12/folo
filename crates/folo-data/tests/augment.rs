//! Every augmentation keeps boxes and pixels consistent: draw coloured boxes, augment, check the pixels under the
//! transformed boxes.

mod common;

use common::*;
use folo_data::augment::{
    BoxFilter, affine_sample, flip_lr, flip_ud, mixup, mosaic4, random_affine, random_hsv, resize_long_side,
    transform_boxes,
};
use folo_data::warp::Affine;
use folo_data::{
    AugmentConfig, BBox, InMemoryDataset, LetterboxOptions, Rng, Split, TrainTransform, letterbox_sample,
};

/// Keep every visible sliver so "every coloured pixel is inside a box" holds exactly.
fn permissive(cfg: AugmentConfig) -> AugmentConfig {
    AugmentConfig { min_box_px: 0.0, min_area_ratio: 0.0, max_aspect: 1e9, ..cfg }
}

fn geometric() -> AugmentConfig {
    permissive(AugmentConfig {
        degrees: 15.0,
        shear: 5.0,
        scale: 0.5,
        translate: 0.2,
        ..AugmentConfig::none()
    })
}

#[test]
fn letterbox_keeps_boxes_on_pixels() {
    for seed in 0..6 {
        let s = random_rect_sample(seed, 200 + 37 * seed as u32, 150, 5);
        for opts in [LetterboxOptions::default(), LetterboxOptions { stride: Some(32), ..Default::default() }]
        {
            let (out, lb) = letterbox_sample(&s, 256, 256, &opts);
            assert_eq!([out.image.width, out.image.height], lb.out);
            assert_consistent(&out, 1.5, 0.9, "letterbox");
        }
    }
}

#[test]
fn affine_without_rotation_is_tight() {
    let cfg = permissive(AugmentConfig { scale: 0.5, translate: 0.2, ..AugmentConfig::none() });
    for seed in 0..10 {
        let s = random_rect_sample(seed, 240, 180, 6);
        let mut rng = Rng::new(seed);
        let out = random_affine(&s, 200, 200, &cfg, &mut rng);
        assert_consistent(&out, 1.5, 0.85, &format!("affine seed {seed}"));
    }
}

#[test]
fn affine_with_rotation_and_shear() {
    for seed in 0..10 {
        let s = random_rect_sample(100 + seed, 240, 180, 6);
        let mut rng = Rng::new(seed);
        let out = random_affine(&s, 220, 200, &geometric(), &mut rng);
        // A rotated rectangle fills at least half of its bounding box (minus anti-aliased edges).
        assert_consistent(&out, 1.5, 0.35, &format!("rotated affine seed {seed}"));
    }
}

#[test]
fn explicit_rotation_by_90_degrees() {
    let s = rect_sample(100, 60, &[(0, [10, 10, 30, 50]), (2, [60, 5, 95, 20])]);
    let m = Affine::translate(-50.0, -30.0).then(&Affine::rotate(90.0)).then(&Affine::translate(30.0, 50.0));
    let out = affine_sample(&s, &m, 60, 100, &BoxFilter::default());
    assert_eq!(out.boxes.len(), 2);
    assert_consistent(&out, 1.0, 0.95, "rot90");
    // The 20x40 rect becomes 40x20.
    let [x1, y1, x2, y2] = out.boxes[0].pixel_xyxy(60, 100);
    assert!(((x2 - x1) - 40.0).abs() < 1e-3 && ((y2 - y1) - 20.0).abs() < 1e-3);
}

#[test]
fn box_filter_drops_tiny_thin_and_clipped_boxes() {
    let boxes = [
        BBox::from_pixel_xyxy(0, [10.0, 10.0, 50.0, 50.0], 100, 100), // kept
        BBox::from_pixel_xyxy(1, [10.0, 60.0, 11.5, 90.0], 100, 100), // narrower than 2 px
        BBox::from_pixel_xyxy(2, [96.0, 10.0, 140.0, 50.0], 200, 100), // mostly outside a 100-wide output
    ];
    let m = Affine::translate(4.0, 0.0);
    let out = transform_boxes(&boxes[..2], 100, 100, &m, 100, 100, &BoxFilter::default());
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].class, 0);
    // Box 2 rendered into a 100-wide output keeps 4 of its 44 px: area ratio < 0.1.
    let out = transform_boxes(&boxes[2..], 200, 100, &Affine::IDENTITY, 100, 100, &BoxFilter::default());
    assert!(out.is_empty());
    // Downscaling 10x makes a 15 px box 1.5 px: dropped.
    let small = [BBox::from_pixel_xyxy(0, [0.0, 0.0, 15.0, 15.0], 100, 100)];
    assert!(
        transform_boxes(&small, 100, 100, &Affine::scale(0.1, 0.1), 100, 100, &BoxFilter::default())
            .is_empty()
    );
    // Aspect ratio limit.
    let thin = [BBox::from_pixel_xyxy(0, [0.0, 0.0, 99.0, 3.0], 100, 100)];
    let f = BoxFilter { max_aspect: 20.0, ..BoxFilter::default() };
    assert!(transform_boxes(&thin, 100, 100, &Affine::IDENTITY, 100, 100, &f).is_empty());
}

#[test]
fn flips_mirror_boxes() {
    let s = random_rect_sample(7, 160, 120, 6);
    let mut a = s.clone();
    flip_lr(&mut a);
    assert_consistent(&a, 0.5, 0.99, "fliplr");
    let mut b = s.clone();
    flip_ud(&mut b);
    assert_consistent(&b, 0.5, 0.99, "flipud");
    flip_lr(&mut a);
    assert_eq!(a.image, s.image);
}

#[test]
fn mosaic_then_affine() {
    for seed in 0..12 {
        let parts: Vec<_> =
            (0..4).map(|k| resize_long_side(random_rect_sample(seed * 4 + k, 200, 150, 5), 128)).collect();
        let mut rng = Rng::new(seed);
        let canvas = mosaic4([&parts[0], &parts[1], &parts[2], &parts[3]], 128, &mut rng);
        assert_eq!((canvas.image.width, canvas.image.height), (256, 256));
        assert_consistent(&canvas, 1.5, 0.9, &format!("mosaic canvas {seed}"));
        let out = random_affine(&canvas, 128, 128, &geometric(), &mut rng);
        assert_consistent(&out, 1.5, 0.3, &format!("mosaic affine {seed}"));
    }
}

#[test]
fn mixup_blends_pixels_and_concatenates_boxes() {
    let a = random_rect_sample(1, 64, 64, 3);
    let b = random_rect_sample(2, 64, 64, 3);
    let mut m = a.clone();
    mixup(&mut m, &b, &mut Rng::new(5));
    assert_eq!(m.boxes.len(), a.boxes.len() + b.boxes.len());
    assert_eq!(&m.boxes[..a.boxes.len()], &a.boxes[..]);
    // Recover the blend weight from one pixel where the images differ, then check every pixel.
    let i = (0..a.image.data.len())
        .find(|&i| (a.image.data[i] as i32 - b.image.data[i] as i32).abs() > 200)
        .unwrap();
    let r =
        (m.image.data[i] as f32 - b.image.data[i] as f32) / (a.image.data[i] as f32 - b.image.data[i] as f32);
    assert!((0.2..0.8).contains(&r), "blend weight {r}");
    for k in 0..a.image.data.len() {
        let want = r * a.image.data[k] as f32 + (1.0 - r) * b.image.data[k] as f32;
        assert!((m.image.data[k] as f32 - want).abs() <= 2.0);
    }
}

#[test]
fn hsv_leaves_boxes_alone_and_is_deterministic() {
    let s = random_rect_sample(3, 64, 48, 4);
    let mut a = s.clone();
    let mut b = s.clone();
    random_hsv(&mut a.image, [0.015, 0.7, 0.4], &mut Rng::new(11));
    random_hsv(&mut b.image, [0.015, 0.7, 0.4], &mut Rng::new(11));
    assert_eq!(a, b);
    assert_eq!(a.boxes, s.boxes);
    assert_ne!(a.image, s.image);
}

#[test]
fn train_transform_end_to_end() {
    let samples: Vec<_> = (0..8).map(|k| random_rect_sample(50 + k, 180 + 20 * k as u32, 140, 5)).collect();
    let ds = InMemoryDataset {
        samples,
        classes: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        split: Split::Train,
    };
    let aug =
        permissive(AugmentConfig { hsv: [0.0; 3], degrees: 10.0, flipud: 0.5, ..AugmentConfig::default() });
    let t = TrainTransform::train(160, aug, 0, 0);
    let mut mosaics = 0;
    for seed in 0..12 {
        let out = t.apply_dataset(&ds, seed as usize % 8, &mut Rng::new(seed)).unwrap();
        assert_eq!((out.sample.image.width, out.sample.image.height), (160, 160));
        assert_consistent(&out.sample, 1.5, 0.3, &format!("pipeline seed {seed}"));
        mosaics += (out.letterbox.ratio == [1.0, 1.0]) as usize;
    }
    assert!(mosaics >= 10, "mosaic probability 1.0 should mosaic every time");
    // Without mosaic (closed): letterbox + affine.
    let t = TrainTransform::train(160, aug, 95, 100);
    assert!(!t.mosaic);
    let out = t.apply_dataset(&ds, 3, &mut Rng::new(1)).unwrap();
    assert_consistent(&out.sample, 1.5, 0.3, "closed mosaic");
    // Validation: letterbox only, geometry recorded.
    let v = TrainTransform::val(160).apply_dataset(&ds, 2, &mut Rng::new(0)).unwrap();
    assert_eq!(v.orig_size, [220, 140]);
    assert_eq!(v.letterbox.out, [160, 160]);
    assert_consistent(&v.sample, 1.5, 0.85, "val");
}

#[test]
fn same_seed_same_result() {
    let samples: Vec<_> = (0..4).map(|k| random_rect_sample(k, 100, 80, 3)).collect();
    let ds = InMemoryDataset { samples, classes: vec!["a".into(); 4], split: Split::Train };
    let aug = AugmentConfig { degrees: 5.0, shear: 2.0, mixup: 0.5, ..AugmentConfig::default() };
    let t = TrainTransform::train(96, aug, 0, 0);
    let a = t.apply_dataset(&ds, 1, &mut Rng::new(42)).unwrap();
    let b = t.apply_dataset(&ds, 1, &mut Rng::new(42)).unwrap();
    assert_eq!(a.sample, b.sample);
    let c = t.apply_dataset(&ds, 1, &mut Rng::new(43)).unwrap();
    assert_ne!(a.sample, c.sample);
}

#[test]
fn no_augmentation_matches_validation() {
    let mut s = rect_sample(120, 80, &[(0, [10, 10, 60, 50])]);
    s.boxes.push(BBox::from_pixel_xyxy(1, [80.0, 30.0, 81.0, 31.0], 120, 80)); // a 1 px box
    let ds = InMemoryDataset { samples: vec![s], classes: vec!["a".into(), "b".into()], split: Split::Train };
    let val = TrainTransform::val(96).apply_dataset(&ds, 0, &mut Rng::new(0)).unwrap();
    let none = TrainTransform::train(96, AugmentConfig::none(), 0, 0)
        .apply_dataset(&ds, 0, &mut Rng::new(0))
        .unwrap();
    assert_eq!(none.sample, val.sample);
    assert_eq!(none.sample.boxes.len(), 2);
}

/// A mosaic of size 0 is an empty image, alone and through its affine.
#[test]
fn an_empty_mosaic_is_empty() {
    let parts: Vec<_> = (0..4).map(|k| random_rect_sample(k, 40, 30, 2)).collect();
    let parts = [&parts[0], &parts[1], &parts[2], &parts[3]];
    let m = mosaic4(parts, 0, &mut Rng::new(1));
    assert_eq!((m.image.width, m.image.height, m.image.data.len()), (0, 0, 0));
    let m = folo_data::augment::mosaic_affine(parts, 0, &AugmentConfig::default(), &mut Rng::new(1));
    assert_eq!((m.image.width, m.image.height, m.image.data.len()), (0, 0, 0));
}

/// Edge-case coverage: 1x1 and odd-sized images, empty labels, out-of-bounds mosaic box containment,
/// grayscale/RGBA PNG decoding through `DataLoader`, and deterministic seeded RNG invariants.
#[test]
fn edge_sizes_empty_labels_oob_mosaic_and_formats_end_to_end() {
    use folo_data::{DataLoader, Image, LoaderConfig, Sample, decode};
    use std::sync::Arc;

    // 1. Out-of-bounds box in quadrant 0 must stay inside quadrant 0 (never bleed past xc/yc).
    let mut parts: Vec<Sample> = (0..4)
        .map(|k| Sample { image: Image::new(32, 32, [(k * 50) as u8, 20, 40]), boxes: vec![] })
        .collect();
    parts[0].boxes.push(BBox::new(0, 0.9, 0.9, 0.6, 0.6)); // extends to 1.2 in both axes
    let mut rng_peek = Rng::new(17);
    let xc = rng_peek.uniform(64.0 * 0.5, 64.0 * 1.5) as i64;
    let yc = rng_peek.uniform(64.0 * 0.5, 64.0 * 1.5) as i64;
    let m = mosaic4([&parts[0], &parts[1], &parts[2], &parts[3]], 64, &mut Rng::new(17));
    assert_eq!(m.boxes.len(), 1);
    let [_, _, x2, y2] = m.boxes[0].pixel_xyxy(128, 128);
    assert!(
        x2 <= xc as f32 + 1e-4 && y2 <= yc as f32 + 1e-4,
        "box bled past quadrant ({xc},{yc}): {x2},{y2}"
    );

    // 2. 1x1, odd sizes, and empty labels survive letterbox, mosaic, affine, and DataLoader cleanly.
    let mut odd_samples: Vec<Sample> = [(1u32, 1u32), (1, 3), (3, 1), (17, 19), (33, 65)]
        .into_iter()
        .enumerate()
        .map(|(k, (w, h))| Sample {
            image: Image::new(w, h, [(k * 40 + 10) as u8, 80, 160]),
            boxes: if k % 2 == 0 { vec![] } else { vec![BBox::new(0, 0.5, 0.5, 0.5, 0.5)] },
        })
        .collect();

    // 3. Grayscale (L8) and RGBA8 PNGs decode cleanly into upright RGB and flow through DataLoader.
    let encode_png = |buf: &[u8], w: u32, h: u32, ct: image::ExtendedColorType| {
        use image::ImageEncoder;
        let mut out = Vec::new();
        image::codecs::png::PngEncoder::new(&mut out).write_image(buf, w, h, ct).unwrap();
        out
    };
    let luma_img =
        decode(&encode_png(&[10, 50, 90, 130, 170, 210], 3, 2, image::ExtendedColorType::L8)).unwrap();
    assert_eq!(luma_img.pixel(2, 1), [210, 210, 210]);
    let rgba_img = decode(&encode_png(
        &[11, 22, 33, 255, 44, 55, 66, 0, 77, 88, 99, 128],
        3,
        1,
        image::ExtendedColorType::Rgba8,
    ))
    .unwrap();
    assert_eq!(rgba_img.pixel(1, 0), [44, 55, 66]);
    odd_samples.push(Sample { image: luma_img, boxes: vec![] });
    odd_samples.push(Sample { image: rgba_img, boxes: vec![BBox::new(0, 0.5, 0.5, 0.6, 0.8)] });

    for s in &odd_samples {
        let (lb, _) = letterbox_sample(s, 32, 32, &LetterboxOptions::default());
        assert_eq!((lb.image.width, lb.image.height), (32, 32));
    }

    let ds: Arc<dyn folo_data::Dataset> =
        Arc::new(InMemoryDataset { samples: odd_samples, classes: vec!["x".into()], split: Split::Train });
    let aug = AugmentConfig { degrees: 8.0, shear: 2.0, mixup: 0.5, flipud: 0.5, ..AugmentConfig::default() };
    for &u8_images in &[true, false] {
        let base = LoaderConfig {
            batch_size: 4,
            imgsz: 32,
            workers: 0,
            seed: 99,
            augment: Some(aug),
            u8_images,
            ..LoaderConfig::default()
        };
        let l0 = DataLoader::new(ds.clone(), base.clone()).unwrap();
        let l2 = DataLoader::new(ds.clone(), LoaderConfig { workers: 2, ..base }).unwrap();
        let b0: Vec<_> = l0.epoch(0).map(|b| b.unwrap()).collect();
        let b2: Vec<_> = l2.epoch(0).map(|b| b.unwrap()).collect();
        assert_eq!(b0, b2, "odd/empty/L8/RGBA determinism u8={u8_images}");
    }
}

/// Fused mixup + HSV + flip + planar DataLoader path matches `TrainTransform::apply` bit-for-bit
/// across both mosaic+mixup and non-mosaic+mixup (and both `u8` and `f32` batches).
#[test]
fn loader_mixup_fused_matches_apply_bit_exact() {
    use folo_data::{DataLoader, LoaderConfig};
    use std::sync::Arc;

    let samples: Vec<_> = (0..6).map(|k| random_rect_sample(200 + k, 150 + 17 * k as u32, 110, 4)).collect();
    let ds = Arc::new(InMemoryDataset { samples, classes: vec!["a".into(); 4], split: Split::Train });
    for &(mosaic_p, degrees) in &[(1.0f32, 0.0f32), (0.0, 0.0), (1.0, 12.0), (0.0, 12.0)] {
        let aug = AugmentConfig {
            mosaic: mosaic_p,
            mixup: 1.0,
            degrees,
            shear: if degrees > 0.0 { 3.0 } else { 0.0 },
            flipud: 0.5,
            fliplr: 0.5,
            ..AugmentConfig::default()
        };
        let t = TrainTransform::train(96, aug, 0, 0);
        for &u8_images in &[true, false] {
            let cfg = LoaderConfig {
                batch_size: 3,
                imgsz: 96,
                workers: 2,
                shuffle: true,
                seed: 77,
                augment: Some(aug),
                u8_images,
                ..LoaderConfig::default()
            };
            let loader = DataLoader::new(ds.clone(), cfg).unwrap();
            let order = loader.order(0);
            let mut pos = 0usize;
            for batch in loader.epoch(0) {
                let batch = batch.unwrap();
                let plane = 96 * 96;
                for k in 0..batch.len() {
                    let mut rng = Rng::derive(&[77, 0, pos as u64]);
                    let want = t.apply_dataset(ds.as_ref(), order[pos], &mut rng).unwrap();
                    let off = k * 3 * plane;
                    if u8_images {
                        let mut ref_u8 = vec![0u8; 3 * plane];
                        for (i, px) in want.sample.image.data.as_chunks::<3>().0.iter().enumerate() {
                            ref_u8[i] = px[0];
                            ref_u8[plane + i] = px[1];
                            ref_u8[2 * plane + i] = px[2];
                        }
                        assert_eq!(
                            &batch.images_u8[off..off + 3 * plane],
                            &ref_u8[..],
                            "u8 mixup parity mosaic={mosaic_p} deg={degrees} pos={pos}"
                        );
                    } else {
                        let mut ref_f32 = vec![0f32; 3 * plane];
                        for (i, px) in want.sample.image.data.as_chunks::<3>().0.iter().enumerate() {
                            ref_f32[i] = px[0] as f32 / 255.0;
                            ref_f32[plane + i] = px[1] as f32 / 255.0;
                            ref_f32[2 * plane + i] = px[2] as f32 / 255.0;
                        }
                        assert!(
                            batch.images[off..off + 3 * plane]
                                .iter()
                                .zip(&ref_f32)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "f32 mixup parity mosaic={mosaic_p} deg={degrees} pos={pos}"
                        );
                    }
                    pos += 1;
                }
            }
        }
    }
}
