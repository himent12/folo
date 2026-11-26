//! Box-aware occlusion and compositing augmentations: [`cutmix`], [`copy_paste`], and [`random_erasing`].
//!
//! Every op keeps bounding boxes strictly consistent with pixel edits, clips or trims boxes at patch boundaries,
//! drops boxes that become occluded or degenerate under [`BoxFilter`], and is deterministic for a given [`Rng`].

use super::BoxFilter;
use crate::bbox::BBox;
use crate::image::Sample;
use crate::letterbox::PAD_VALUE;
use crate::rng::Rng;
use crate::simd::assert_image_data;
use crate::simd::planar::copy_flip_lr_row;

/// Validates `rect = [x1, y1, x2, y2]` against `width x height` and returns clamped `[x1, y1, x2, y2]`
/// with `x1 <= x2 <= width` and `y1 <= y2 <= height`.
#[inline]
fn clamp_rect(rect: [u32; 4], width: u32, height: u32) -> [u32; 4] {
    let x1 = rect[0].min(width);
    let y1 = rect[1].min(height);
    let x2 = rect[2].min(width).max(x1);
    let y2 = rect[3].min(height).max(y1);
    [x1, y1, x2, y2]
}

/// Intersection area of two axis-aligned boxes `[x1, y1, x2, y2]`.
#[inline]
fn inter_area(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    iw * ih
}

/// Updates a host box after an opaque patch `rect = [rx1, ry1, rx2, ry2]` overwrites pixels in a `width x height`
/// image:
/// - Drops degenerate or fully occluded boxes.
/// - When the patch covers the full vertical span of the box and clips one horizontal edge (or covers the full
///   horizontal span and clips one vertical edge), trims the box cleanly to the single remaining visible strip and
///   checks `filter`.
/// - Otherwise (interior hole, corner bite, or split into two disjoint strips), retains the box only when the
///   remaining visible area is at least `filter.min_area_ratio * orig_area` and `filter.keep` passes.
fn filter_occluded_box(
    bx: &BBox,
    rect: [f32; 4],
    width: u32,
    height: u32,
    filter: &BoxFilter,
) -> Option<BBox> {
    if width == 0 || height == 0 {
        return None;
    }
    let (fw, fh) = (width as f32, height as f32);
    let [raw_x1, raw_y1, raw_x2, raw_y2] = bx.pixel_xyxy(width, height);
    let [x1, y1, x2, y2] =
        [raw_x1.clamp(0.0, fw), raw_y1.clamp(0.0, fh), raw_x2.clamp(0.0, fw), raw_y2.clamp(0.0, fh)];
    let (bw, bh) = (x2 - x1, y2 - y1);
    if !(bw > 0.0 && bh > 0.0) {
        return None;
    }
    let orig_area = bw * bh;
    let [rx1, ry1, rx2, ry2] = rect;
    let ov = inter_area([x1, y1, x2, y2], rect);
    if ov == 0.0 {
        return Some(*bx);
    }
    let visible = orig_area - ov;
    if visible <= 0.0 {
        return None;
    }
    // Check for a clean single-edge clip where the remaining visible region is one exact rectangle.
    let covers_y = ry1 <= y1 && ry2 >= y2;
    let covers_x = rx1 <= x1 && rx2 >= x2;
    let trimmed = if covers_y && rx1 <= x1 && rx2 < x2 {
        Some([rx2, y1, x2, y2])
    } else if covers_y && rx1 > x1 && rx2 >= x2 {
        Some([x1, y1, rx1, y2])
    } else if covers_x && ry1 <= y1 && ry2 < y2 {
        Some([x1, ry2, x2, y2])
    } else if covers_x && ry1 > y1 && ry2 >= y2 {
        Some([x1, y1, x2, ry1])
    } else {
        None
    };
    if let Some([nx1, ny1, nx2, ny2]) = trimmed {
        let (nw, nh) = (nx2 - nx1, ny2 - ny1);
        return filter
            .keep(nw, nh, orig_area)
            .then(|| BBox::from_pixel_xyxy(bx.class, [nx1, ny1, nx2, ny2], width, height));
    }
    // Partial interior/corner occlusion or split: check visible area ratio plus geometric filter bounds.
    if visible < filter.min_area_ratio * orig_area {
        return None;
    }
    filter.keep(bw, bh, orig_area).then_some(*bx)
}

/// Applies host-box occlusion filtering in place for an overwritten pixel rectangle `rect`.
fn apply_occlusion_to_boxes(
    boxes: &mut Vec<BBox>,
    rect: [u32; 4],
    width: u32,
    height: u32,
    filter: &BoxFilter,
) {
    if rect[2] <= rect[0] || rect[3] <= rect[1] || boxes.is_empty() {
        return;
    }
    let rf = [rect[0] as f32, rect[1] as f32, rect[2] as f32, rect[3] as f32];
    let mut write = 0usize;
    for read in 0..boxes.len() {
        if let Some(kept) = filter_occluded_box(&boxes[read], rf, width, height, filter) {
            boxes[write] = kept;
            write += 1;
        }
    }
    boxes.truncate(write);
}

/// Copies the rectangular patch `rect = [x1, y1, x2, y2]` from `other` onto `s` in place, filtering occluded
/// host boxes and appending clipped donor boxes that pass `filter`.
///
/// Both images must have the same dimensions.
pub fn cutmix_rect(s: &mut Sample, other: &Sample, rect: [u32; 4], filter: &BoxFilter) {
    assert_eq!(
        (s.image.width, s.image.height),
        (other.image.width, other.image.height),
        "cutmix needs images of the same size"
    );
    assert_image_data(&s.image);
    assert_image_data(&other.image);
    let (w, h) = (s.image.width, s.image.height);
    let [x1, y1, x2, y2] = clamp_rect(rect, w, h);
    if x2 <= x1 || y2 <= y1 {
        return;
    }

    let stride = w as usize * 3;
    let byte_lo = x1 as usize * 3;
    let byte_hi = x2 as usize * 3;
    for y in y1 as usize..y2 as usize {
        let row_off = y * stride;
        s.image.data[row_off + byte_lo..row_off + byte_hi]
            .copy_from_slice(&other.image.data[row_off + byte_lo..row_off + byte_hi]);
    }

    apply_occlusion_to_boxes(&mut s.boxes, [x1, y1, x2, y2], w, h, filter);

    let (rx1, ry1, rx2, ry2) = (x1 as f32, y1 as f32, x2 as f32, y2 as f32);
    for b in &other.boxes {
        let [bx1, by1, bx2, by2] = b.pixel_xyxy(w, h);
        let orig_w = bx2 - bx1;
        let orig_h = by2 - by1;
        if !(orig_w > 0.0 && orig_h > 0.0) {
            continue;
        }
        let orig_area = orig_w * orig_h;
        let cx1 = bx1.clamp(rx1, rx2);
        let cy1 = by1.clamp(ry1, ry2);
        let cx2 = bx2.clamp(rx1, rx2);
        let cy2 = by2.clamp(ry1, ry2);
        if filter.keep(cx2 - cx1, cy2 - cy1, orig_area) {
            s.boxes.push(BBox::from_pixel_xyxy(b.class, [cx1, cy1, cx2, cy2], w, h));
        }
    }
}

/// Box-aware CutMix: samples a random rectangular patch from `other` (with patch area ratio `1 - λ`,
/// `λ ~ Uniform(0, 1)`) and pastes it onto `s`, trimming/dropping occluded host boxes and clipping donor boxes
/// through `filter`. Returns the pasted pixel rectangle `[x1, y1, x2, y2]`.
///
/// Both images must have the same dimensions.
pub fn cutmix(s: &mut Sample, other: &Sample, filter: &BoxFilter, rng: &mut Rng) -> [u32; 4] {
    assert_eq!(
        (s.image.width, s.image.height),
        (other.image.width, other.image.height),
        "cutmix needs images of the same size"
    );
    let (w, h) = (s.image.width, s.image.height);
    if w == 0 || h == 0 {
        return [0, 0, 0, 0];
    }
    let lam = rng.f32();
    let cut_ratio = (1.0 - lam).sqrt();
    let cut_w = ((w as f32 * cut_ratio).round() as i64).clamp(1, w as i64);
    let cut_h = ((h as f32 * cut_ratio).round() as i64).clamp(1, h as i64);
    let cx = rng.below(w as usize) as i64;
    let cy = rng.below(h as usize) as i64;
    let x1 = (cx - cut_w / 2).clamp(0, w as i64) as u32;
    let y1 = (cy - cut_h / 2).clamp(0, h as i64) as u32;
    let x2 = (x1 as i64 + cut_w).min(w as i64) as u32;
    let y2 = (y1 as i64 + cut_h).min(h as i64) as u32;
    let rect = [x1, y1, x2, y2];
    cutmix_rect(s, other, rect, filter);
    rect
}

/// Parameters for box-level [`copy_paste`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopyPasteConfig {
    /// Per-donor-box selection probability in `[0, 1]` (default `0.5`).
    pub prob: f32,
    /// Maximum allowed intersection-over-existing-box-area (`IoA`) between a pasted box and any existing box
    /// in the destination sample (default `0.3`). Donor boxes that would occlude an existing object more than
    /// this threshold are skipped.
    pub max_ioa: f32,
    /// Horizontally mirror pasted crops into the mirrored x-position `[W - x2, y1, W - x1, y2]` (default `false`).
    pub flip_lr: bool,
}

impl Default for CopyPasteConfig {
    fn default() -> Self {
        CopyPasteConfig { prob: 0.5, max_ioa: 0.3, flip_lr: false }
    }
}

/// Box-level Copy-Paste augmentation: selects donor boxes from `other` with probability `cfg.prob`, rejects
/// any paste whose overlap with an existing destination box exceeds `cfg.max_ioa` (intersection over existing
/// box area), copies the RGB crop into `s` (horizontally mirrored via SIMD when `cfg.flip_lr` is true), and
/// appends the pasted box to `s.boxes`. Returns the number of boxes pasted.
///
/// Both images must have the same dimensions.
pub fn copy_paste(
    s: &mut Sample,
    other: &Sample,
    cfg: &CopyPasteConfig,
    filter: &BoxFilter,
    rng: &mut Rng,
) -> usize {
    assert_eq!(
        (s.image.width, s.image.height),
        (other.image.width, other.image.height),
        "copy_paste needs images of the same size"
    );
    assert_image_data(&s.image);
    assert_image_data(&other.image);
    let (w, h) = (s.image.width, s.image.height);
    if w == 0 || h == 0 || other.boxes.is_empty() || cfg.prob <= 0.0 {
        return 0;
    }
    let (fw, fh) = (w as f32, h as f32);
    let stride = w as usize * 3;
    let mut existing: Vec<[f32; 4]> = s.boxes.iter().map(|b| b.pixel_xyxy(w, h)).collect();
    let mut pasted = 0usize;

    for b in &other.boxes {
        if !rng.chance(cfg.prob) {
            continue;
        }
        let [bx1, by1, bx2, by2] = b.pixel_xyxy(w, h);
        let (orig_w, orig_h) = (bx2 - bx1, by2 - by1);
        if !(orig_w > 0.0 && orig_h > 0.0) {
            continue;
        }
        let orig_area = orig_w * orig_h;
        let sx1 = bx1.floor().clamp(0.0, fw) as u32;
        let sy1 = by1.floor().clamp(0.0, fh) as u32;
        let sx2 = bx2.ceil().clamp(0.0, fw) as u32;
        let sy2 = by2.ceil().clamp(0.0, fh) as u32;
        if sx2 <= sx1 || sy2 <= sy1 {
            continue;
        }
        let cw = (sx2 - sx1) as f32;
        let ch = (sy2 - sy1) as f32;
        if !filter.keep(cw, ch, orig_area) {
            continue;
        }
        let (dx1, dx2) = if cfg.flip_lr { (w - sx2, w - sx1) } else { (sx1, sx2) };
        let dst_rect = [dx1 as f32, sy1 as f32, dx2 as f32, sy2 as f32];

        let mut occluded = false;
        for &ex in &existing {
            let ex_area = (ex[2] - ex[0]).max(0.0) * (ex[3] - ex[1]).max(0.0);
            if ex_area > 0.0 && inter_area(dst_rect, ex) / ex_area > cfg.max_ioa {
                occluded = true;
                break;
            }
        }
        if occluded {
            continue;
        }

        let s_lo = sx1 as usize * 3;
        let s_hi = sx2 as usize * 3;
        let d_lo = dx1 as usize * 3;
        let d_hi = dx2 as usize * 3;
        for y in sy1 as usize..sy2 as usize {
            let row = y * stride;
            let src_slice = &other.image.data[row + s_lo..row + s_hi];
            let dst_slice = &mut s.image.data[row + d_lo..row + d_hi];
            if cfg.flip_lr {
                copy_flip_lr_row(src_slice, dst_slice);
            } else {
                dst_slice.copy_from_slice(src_slice);
            }
        }
        s.boxes.push(BBox::from_pixel_xyxy(b.class, dst_rect, w, h));
        existing.push(dst_rect);
        pasted += 1;
    }
    pasted
}

/// Pixel fill mode for [`random_erasing`] / [`erase_rect`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErasingFill {
    /// Fill erased patches with one constant RGB colour (e.g. `[PAD_VALUE; 3]`).
    Constant([u8; 3]),
    /// Fill erased patches with deterministic uniform random RGB bytes drawn from `rng`.
    RandomNoise,
}

/// Parameters for box-aware [`random_erasing`] (Cutout / Random Erasing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomErasingConfig {
    /// Probability of each erasing attempt (default `0.4`).
    pub prob: f32,
    /// Min and max fraction of total image area per erased patch (default `[0.02, 0.2]`).
    pub area: [f32; 2],
    /// Min and max aspect ratio `w / h` of an erased patch, sampled log-uniformly (default `[0.3, 3.3]`).
    pub aspect: [f32; 2],
    /// Number of independent erasing attempts per call (default `1`).
    pub count: usize,
    /// Fill mode for erased pixels (default `Constant([PAD_VALUE; 3])`).
    pub fill: ErasingFill,
}

impl Default for RandomErasingConfig {
    fn default() -> Self {
        RandomErasingConfig {
            prob: 0.4,
            area: [0.02, 0.2],
            aspect: [0.3, 3.3],
            count: 1,
            fill: ErasingFill::Constant([PAD_VALUE; 3]),
        }
    }
}

/// Fills a slice with deterministic random bytes drawn 8 bytes at a time from `rng`.
#[inline]
fn fill_random_bytes(dst: &mut [u8], rng: &mut Rng) {
    let (chunks, tail) = dst.as_chunks_mut::<8>();
    for c in chunks {
        *c = rng.next_u64().to_le_bytes();
    }
    if !tail.is_empty() {
        let bytes = rng.next_u64().to_le_bytes();
        tail.copy_from_slice(&bytes[..tail.len()]);
    }
}

/// Erases a specific pixel rectangle `rect = [x1, y1, x2, y2]` in `s` using `fill`, and updates `s.boxes`
/// with [`BoxFilter`] occlusion rules.
pub fn erase_rect(s: &mut Sample, rect: [u32; 4], fill: ErasingFill, filter: &BoxFilter, rng: &mut Rng) {
    assert_image_data(&s.image);
    let (w, h) = (s.image.width, s.image.height);
    let [x1, y1, x2, y2] = clamp_rect(rect, w, h);
    if x2 <= x1 || y2 <= y1 {
        return;
    }

    let stride = w as usize * 3;
    let byte_lo = x1 as usize * 3;
    let byte_hi = x2 as usize * 3;
    let row_bytes = byte_hi - byte_lo;
    match fill {
        ErasingFill::Constant([r, g, b]) if r == g && g == b => {
            for y in y1 as usize..y2 as usize {
                let off = y * stride;
                s.image.data[off + byte_lo..off + byte_hi].fill(r);
            }
        }
        ErasingFill::Constant(rgb) => {
            let mut template = vec![0u8; row_bytes];
            for px in template.as_chunks_mut::<3>().0 {
                *px = rgb;
            }
            for y in y1 as usize..y2 as usize {
                let off = y * stride;
                s.image.data[off + byte_lo..off + byte_hi].copy_from_slice(&template);
            }
        }
        ErasingFill::RandomNoise => {
            for y in y1 as usize..y2 as usize {
                let off = y * stride;
                fill_random_bytes(&mut s.image.data[off + byte_lo..off + byte_hi], rng);
            }
        }
    }

    apply_occlusion_to_boxes(&mut s.boxes, [x1, y1, x2, y2], w, h, filter);
}

/// Box-aware Random Erasing (Cutout): performs up to `cfg.count` independent erasing attempts (each with
/// probability `cfg.prob`), sampling patch area in `cfg.area` and log-uniform aspect ratio in `cfg.aspect`,
/// filling erased pixels according to `cfg.fill`, and trimming/dropping occluded boxes through `filter`.
/// Returns the number of patches actually erased.
pub fn random_erasing(s: &mut Sample, cfg: &RandomErasingConfig, filter: &BoxFilter, rng: &mut Rng) -> usize {
    assert_image_data(&s.image);
    let (w, h) = (s.image.width, s.image.height);
    if w == 0 || h == 0 || cfg.count == 0 || cfg.prob <= 0.0 {
        return 0;
    }
    let area_lo = cfg.area[0].clamp(0.0, 1.0);
    let area_hi = cfg.area[1].clamp(area_lo, 1.0);
    let asp_lo = cfg.aspect[0].max(1e-3);
    let asp_hi = cfg.aspect[1].max(asp_lo);
    if area_hi <= 0.0 {
        return 0;
    }
    let log_lo = asp_lo.ln();
    let log_hi = asp_hi.ln();
    let total_area = w as f32 * h as f32;
    let mut erased = 0usize;

    for _ in 0..cfg.count {
        if !rng.chance(cfg.prob) {
            continue;
        }
        for _ in 0..10 {
            let target_area = rng.uniform(area_lo, area_hi) * total_area;
            let aspect = rng.uniform(log_lo, log_hi).exp();
            let eh = (target_area / aspect).sqrt().round() as u32;
            let ew = (target_area * aspect).sqrt().round() as u32;
            if ew >= 1 && eh >= 1 && ew <= w && eh <= h {
                let x1 = rng.below((w - ew + 1) as usize) as u32;
                let y1 = rng.below((h - eh + 1) as usize) as u32;
                erase_rect(s, [x1, y1, x1 + ew, y1 + eh], cfg.fill, filter, rng);
                erased += 1;
                break;
            }
        }
    }
    erased
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    #[test]
    fn cutmix_rect_trims_host_and_clips_donor_boxes() {
        let mut host = Sample {
            image: Image::new(100, 100, [10, 20, 30]),
            boxes: vec![
                // Fully inside patch [20, 20, 80, 80] -> dropped.
                BBox::from_pixel_xyxy(0, [30.0, 30.0, 60.0, 60.0], 100, 100),
                // Right half covered by patch (patch covers full y-span [20..80] >= [25..75]) -> trimmed to [10, 25, 20, 75].
                BBox::from_pixel_xyxy(1, [10.0, 25.0, 40.0, 75.0], 100, 100),
                // Disjoint from patch -> kept untouched.
                BBox::from_pixel_xyxy(2, [0.0, 0.0, 15.0, 15.0], 100, 100),
            ],
        };
        let donor = Sample {
            image: Image::new(100, 100, [200, 210, 220]),
            boxes: vec![
                // Straddles patch right edge -> clipped to [60, 40, 80, 70].
                BBox::from_pixel_xyxy(3, [60.0, 40.0, 95.0, 70.0], 100, 100),
                // Outside patch -> dropped.
                BBox::from_pixel_xyxy(4, [0.0, 0.0, 15.0, 15.0], 100, 100),
            ],
        };
        cutmix_rect(&mut host, &donor, [20, 20, 80, 80], &BoxFilter::default());
        assert_eq!(host.image.pixel(50, 50), [200, 210, 220]);
        assert_eq!(host.image.pixel(5, 5), [10, 20, 30]);
        assert_eq!(host.boxes.len(), 3);
        assert_eq!(host.boxes[0].class, 1);
        let [x1, y1, x2, y2] = host.boxes[0].pixel_xyxy(100, 100);
        assert!(
            (x1 - 10.0).abs() < 1e-4
                && (x2 - 20.0).abs() < 1e-4
                && (y1 - 25.0).abs() < 1e-4
                && (y2 - 75.0).abs() < 1e-4
        );
        assert_eq!(host.boxes[1].class, 2);
        assert_eq!(host.boxes[2].class, 3);
        let [dx1, dy1, dx2, dy2] = host.boxes[2].pixel_xyxy(100, 100);
        assert!(
            (dx1 - 60.0).abs() < 1e-4
                && (dx2 - 80.0).abs() < 1e-4
                && (dy1 - 40.0).abs() < 1e-4
                && (dy2 - 70.0).abs() < 1e-4
        );
    }

    #[test]
    fn copy_paste_respects_max_ioa_and_flip_lr() {
        let mut dst = Sample {
            image: Image::new(64, 64, [0, 0, 0]),
            boxes: vec![BBox::from_pixel_xyxy(0, [10.0, 10.0, 30.0, 30.0], 64, 64)],
        };
        let mut donor_img = Image::new(64, 64, [0, 0, 0]);
        donor_img.fill_rect(10, 10, 30, 30, [255, 0, 0]); // overlaps dst box 100%
        donor_img.fill_rect(40, 40, 56, 56, [0, 255, 0]); // non-overlapping
        donor_img.set_pixel(40, 40, [0, 0, 255]); // marker at left edge of second box
        let donor = Sample {
            image: donor_img,
            boxes: vec![
                BBox::from_pixel_xyxy(1, [10.0, 10.0, 30.0, 30.0], 64, 64),
                BBox::from_pixel_xyxy(2, [40.0, 40.0, 56.0, 56.0], 64, 64),
            ],
        };
        let cfg = CopyPasteConfig { prob: 1.0, max_ioa: 0.2, flip_lr: false };
        let mut dst_plain = dst.clone();
        let n = copy_paste(&mut dst_plain, &donor, &cfg, &BoxFilter::default(), &mut Rng::new(7));
        assert_eq!(n, 1);
        assert_eq!(dst_plain.boxes.len(), 2);
        assert_eq!(dst_plain.boxes[1].class, 2);
        assert_eq!(dst_plain.image.pixel(40, 40), [0, 0, 255]);
        assert_eq!(dst_plain.image.pixel(15, 15), [0, 0, 0]);

        // With `flip_lr = true`, [10..30] mirrors to [34..54] (no overlap!) and [40..56] mirrors to [8..24]
        // (no overlap with [10..30] vertically since y=[40..56]). Both paste cleanly!
        let cfg_flip = CopyPasteConfig { prob: 1.0, max_ioa: 0.2, flip_lr: true };
        let n_flip = copy_paste(&mut dst, &donor, &cfg_flip, &BoxFilter::default(), &mut Rng::new(7));
        assert_eq!(n_flip, 2);
        // Marker at x=40 in [40..56) mirrors to x=64-40-1 = 23 in [8..24).
        assert_eq!(dst.image.pixel(23, 40), [0, 0, 255]);
        assert_eq!(dst.image.pixel(44, 15), [255, 0, 0]);
    }

    #[test]
    fn random_erasing_handles_edge_sizes_and_noise_deterministically() {
        // 0x0 and 1x1 never panic.
        let mut empty = Sample { image: Image::new(0, 0, [0, 0, 0]), boxes: vec![] };
        assert_eq!(
            random_erasing(
                &mut empty,
                &RandomErasingConfig { prob: 1.0, ..Default::default() },
                &BoxFilter::default(),
                &mut Rng::new(1)
            ),
            0
        );
        let mut one = Sample { image: Image::new(1, 1, [42, 42, 42]), boxes: vec![] };
        let cfg_one = RandomErasingConfig {
            prob: 1.0,
            area: [1.0, 1.0],
            aspect: [1.0, 1.0],
            count: 1,
            fill: ErasingFill::Constant([1, 2, 3]),
        };
        assert_eq!(random_erasing(&mut one, &cfg_one, &BoxFilter::default(), &mut Rng::new(1)), 1);
        assert_eq!(one.image.pixel(0, 0), [1, 2, 3]);

        // RandomNoise is bitwise deterministic for the same seed.
        let base = Sample {
            image: Image::new(64, 48, [100, 100, 100]),
            boxes: vec![BBox::from_pixel_xyxy(0, [10.0, 10.0, 50.0, 40.0], 64, 48)],
        };
        let cfg =
            RandomErasingConfig { prob: 1.0, count: 2, fill: ErasingFill::RandomNoise, ..Default::default() };
        let mut a = base.clone();
        let mut b = base;
        let na = random_erasing(&mut a, &cfg, &BoxFilter::default(), &mut Rng::new(99));
        let nb = random_erasing(&mut b, &cfg, &BoxFilter::default(), &mut Rng::new(99));
        assert_eq!(na, 2);
        assert_eq!(nb, 2);
        assert_eq!(a, b);
    }
}
