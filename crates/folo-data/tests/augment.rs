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
    AugmentConfig, BBox, Dataset, InMemoryDataset, LetterboxOptions, Rng, Split, TrainTransform,
    letterbox_sample,
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
    let fetch = |i: usize| ds.get(i);
    let mut mosaics = 0;
    for seed in 0..12 {
        let out = t.apply(seed as usize % 8, ds.len(), &fetch, &mut Rng::new(seed)).unwrap();
        assert_eq!((out.sample.image.width, out.sample.image.height), (160, 160));
        assert_consistent(&out.sample, 1.5, 0.3, &format!("pipeline seed {seed}"));
        mosaics += (out.letterbox.ratio == [1.0, 1.0]) as usize;
    }
    assert!(mosaics >= 10, "mosaic probability 1.0 should mosaic every time");
    // Without mosaic (closed): letterbox + affine.
    let t = TrainTransform::train(160, aug, 95, 100);
    assert!(!t.mosaic);
    let out = t.apply(3, ds.len(), &fetch, &mut Rng::new(1)).unwrap();
    assert_consistent(&out.sample, 1.5, 0.3, "closed mosaic");
    // Validation: letterbox only, geometry recorded.
    let v = TrainTransform::val(160).apply(2, ds.len(), &fetch, &mut Rng::new(0)).unwrap();
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
    let fetch = |i: usize| ds.get(i);
    let a = t.apply(1, 4, &fetch, &mut Rng::new(42)).unwrap();
    let b = t.apply(1, 4, &fetch, &mut Rng::new(42)).unwrap();
    assert_eq!(a.sample, b.sample);
    let c = t.apply(1, 4, &fetch, &mut Rng::new(43)).unwrap();
    assert_ne!(a.sample, c.sample);
}
