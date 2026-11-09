//! Shared test helpers: temporary directories, fixture images and a pixel/box consistency checker.
#![allow(dead_code)]

use folo_data::{BBox, Image, ImageFormat, Sample, encode};
use folo_test_support::TestDir;
use std::path::{Path, PathBuf};

/// A unique `folo-data-<tag>-...` directory under `FOLO_TEST_TMP` (else the system temp dir), removed on drop.
pub struct TempDir(TestDir);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        TempDir(TestDir::new(&format!("data-{tag}")))
    }

    pub fn path(&self) -> &Path {
        self.0.path()
    }

    pub fn join(&self, p: impl AsRef<Path>) -> PathBuf {
        self.0.join(p)
    }
}

/// Writes `bytes` to `path`, creating parent directories.
pub fn write(path: &Path, bytes: impl AsRef<[u8]>) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// Writes a `w x h` PNG filled with `color`.
pub fn write_png(path: &Path, w: u32, h: u32, color: [u8; 3]) {
    write(path, encode(&Image::new(w, h, color), ImageFormat::Png).unwrap());
}

/// Pure, easily separable colours per class.
pub const CLASS_COLORS: [[u8; 3]; 4] = [[250, 0, 0], [0, 250, 0], [0, 0, 250], [250, 250, 0]];
pub const BACKGROUND: [u8; 3] = [20, 20, 20];

/// Which class colour (if any) a pixel clearly is.
pub fn class_of(p: [u8; 3]) -> Option<usize> {
    let hi = |v: u8| v > 190;
    let lo = |v: u8| v < 50;
    match (hi(p[0]), hi(p[1]), hi(p[2])) {
        (true, false, false) if lo(p[1]) && lo(p[2]) => Some(0),
        (false, true, false) if lo(p[0]) && lo(p[2]) => Some(1),
        (false, false, true) if lo(p[0]) && lo(p[1]) => Some(2),
        (true, true, false) if lo(p[2]) => Some(3),
        _ => None,
    }
}

/// A sample with axis-aligned coloured rectangles and their exact boxes.
pub fn rect_sample(w: u32, h: u32, rects: &[(u32, [i64; 4])]) -> Sample {
    let mut image = Image::new(w, h, BACKGROUND);
    let mut boxes = Vec::new();
    for &(class, [x1, y1, x2, y2]) in rects {
        image.fill_rect(x1, y1, x2, y2, CLASS_COLORS[class as usize]);
        boxes.push(BBox::from_pixel_xyxy(class, [x1 as f32, y1 as f32, x2 as f32, y2 as f32], w, h));
    }
    Sample { image, boxes }
}

/// A deterministic random rect sample (`seed` picks sizes, positions and classes; rects do not overlap).
pub fn random_rect_sample(seed: u64, w: u32, h: u32, n: usize) -> Sample {
    let mut rng = folo_data::Rng::new(seed);
    let mut rects: Vec<(u32, [i64; 4])> = Vec::new();
    for _ in 0..n * 10 {
        if rects.len() == n {
            break;
        }
        let rw = rng.range_inclusive(8, (w / 3) as i64);
        let rh = rng.range_inclusive(8, (h / 3) as i64);
        let x = rng.range_inclusive(0, w as i64 - rw);
        let y = rng.range_inclusive(0, h as i64 - rh);
        let r = [x, y, x + rw, y + rh];
        let overlaps = rects
            .iter()
            .any(|(_, o)| r[0] < o[2] + 2 && o[0] < r[2] + 2 && r[1] < o[3] + 2 && o[1] < r[3] + 2);
        if !overlaps {
            rects.push((rng.below(4) as u32, r));
        }
    }
    rect_sample(w, h, &rects)
}

/// Checks that boxes and pixels agree:
/// - every pixel that clearly has class `c`'s colour lies inside (within `tol` px of) a box of class `c`;
/// - every box of reasonable size contains a decent share of its class colour (at least `min_fill`).
pub fn assert_consistent(s: &Sample, tol: f32, min_fill: f32, what: &str) {
    let (w, h) = (s.image.width, s.image.height);
    let px_boxes: Vec<(usize, [f32; 4])> =
        s.boxes.iter().map(|b| (b.class as usize, b.pixel_xyxy(w, h))).collect();
    for b in &s.boxes {
        assert!(b.w > 0.0 && b.h > 0.0, "{what}: degenerate box {b:?}");
        let [x1, y1, x2, y2] = b.xyxy();
        assert!(
            x1 >= -1e-4 && y1 >= -1e-4 && x2 <= 1.0 + 1e-4 && y2 <= 1.0 + 1e-4,
            "{what}: box outside image {b:?}"
        );
    }
    let mut orphans = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let Some(c) = class_of(s.image.pixel(x, y)) else { continue };
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let inside = px_boxes.iter().any(|&(bc, [x1, y1, x2, y2])| {
                bc == c && fx >= x1 - tol && fx <= x2 + tol && fy >= y1 - tol && fy <= y2 + tol
            });
            if !inside {
                orphans.push((x, y, c));
            }
        }
    }
    assert!(
        orphans.is_empty(),
        "{what}: {} class pixels outside their boxes, e.g. {:?}",
        orphans.len(),
        &orphans[..orphans.len().min(5)]
    );
    for &(c, [x1, y1, x2, y2]) in &px_boxes {
        if x2 - x1 < 6.0 || y2 - y1 < 6.0 {
            continue;
        }
        let (mut n, mut hit) = (0usize, 0usize);
        for y in (y1.ceil() as u32 + 1)..(y2.floor() as u32).saturating_sub(1) {
            for x in (x1.ceil() as u32 + 1)..(x2.floor() as u32).saturating_sub(1) {
                n += 1;
                if class_of(s.image.pixel(x, y)) == Some(c) {
                    hit += 1;
                }
            }
        }
        if n > 0 {
            let fill = hit as f32 / n as f32;
            assert!(
                fill >= min_fill,
                "{what}: box of class {c} at {:?} is only {fill:.2} filled",
                [x1, y1, x2, y2]
            );
        }
    }
}
