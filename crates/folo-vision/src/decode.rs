//! Decoding raw detection-head outputs into boxes and scores.
//!
//! # Layout
//!
//! The head predicts, for every anchor point on every level (strides 8 / 16 / 32), `4 * reg_max` box-distribution
//! logits and `num_classes` class logits. The functions here take them **anchor-major**, one row per anchor:
//!
//! - `reg`: `[n_anchors, 4 * reg_max]` row-major. Within a row the four sides come one after another, each as
//!   `reg_max` contiguous bins: `[l_0 .. l_{R-1}, t_0 .. t_{R-1}, r_0 .. r_{R-1}, b_0 .. b_{R-1}]`. This is the
//!   channel order of the head's box branch (channel `side * reg_max + bin`), so a `[4 * reg_max, A]` channel-major
//!   tensor transposed to `[A, 4 * reg_max]` gives exactly this layout.
//! - `cls`: `[n_anchors, num_classes]` row-major logits (pre-sigmoid).
//!
//! Anchors are ordered level by level (stride 8 first), and within a level row-major over the feature map (`y`
//! outer, `x` inner), matching the head's flatten + concat order and [`make_anchors`].
//!
//! # Units
//!
//! Each side's distance is the expectation of a softmax over the `reg_max` bins, `d = sum_i i * softmax(logits)_i`,
//! in **stride units** (feature-map cells). [`dist2bbox`] turns `ltrb` distances around an anchor point (also in
//! cells, cell centres at `+0.5`) into a box; [`decode_boxes`] multiplies by the level's stride to give boxes in
//! network-input pixels.

use crate::boxes::Xyxy;
use rayon::prelude::*;

/// Anchor count above which decoding runs in parallel.
const PAR_ANCHORS: usize = 4096;

/// One feature-map level of an anchor grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnchorLevel {
    /// Stride in input pixels.
    pub stride: u32,
    /// Feature-map height (cells).
    pub height: usize,
    /// Feature-map width (cells).
    pub width: usize,
    /// Index of this level's first anchor in the concatenated anchor list.
    pub start: usize,
}

/// Anchor points for all levels, concatenated in head order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Anchors {
    /// Anchor centres `(x, y)` in their level's grid units (cell centres are at `i + offset`).
    pub points: Vec<[f32; 2]>,
    /// Stride of each anchor, in input pixels.
    pub strides: Vec<f32>,
    /// Per-level grid descriptions.
    pub levels: Vec<AnchorLevel>,
}

impl Anchors {
    /// Number of anchors.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether there are no anchors.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Anchor `i`'s centre in input pixels.
    #[inline]
    pub fn point_px(&self, i: usize) -> [f32; 2] {
        let s = self.strides[i];
        [self.points[i][0] * s, self.points[i][1] * s]
    }

    /// All anchor centres in input pixels.
    pub fn points_px(&self) -> Vec<[f32; 2]> {
        (0..self.len()).map(|i| self.point_px(i)).collect()
    }
}

/// Anchor points for an `imgsz_h x imgsz_w` input and the given strides (Ultralytics' `make_anchors`).
///
/// A level with stride `s` has a `ceil(h / s) x ceil(w / s)` grid (the output size of the stride-2 `k=3, p=1` conv
/// chain); its points are `(x + offset, y + offset)` in grid units. Use `offset = 0.5` for cell centres. For a
/// 640x640 input and strides `[8, 16, 32]` that is `6400 + 1600 + 400 = 8400` anchors.
pub fn make_anchors(imgsz_h: u32, imgsz_w: u32, strides: &[u32], offset: f32) -> Anchors {
    let mut a = Anchors::default();
    for &s in strides {
        assert!(s > 0, "make_anchors: zero stride");
        let (h, w) = (imgsz_h.div_ceil(s) as usize, imgsz_w.div_ceil(s) as usize);
        a.levels.push(AnchorLevel { stride: s, height: h, width: w, start: a.points.len() });
        for y in 0..h {
            for x in 0..w {
                a.points.push([x as f32 + offset, y as f32 + offset]);
                a.strides.push(s as f32);
            }
        }
    }
    a
}

/// 2×8 AVX2 fast path for `reg_max == 16` DFL softmax statistics (degree-5 minimax `exp`, no libm calls).
#[cfg(target_arch = "x86_64")]
mod avx2_16 {
    use std::arch::x86_64::*;

    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const EXP5_C5: f32 = 8.290_312e-3;
    const EXP5_C4: f32 = 4.189_795_3e-2;
    const EXP5_C3: f32 = 1.666_763_6e-1;
    const EXP5_C2: f32 = 4.999_915e-1;
    const EXP5_C1: f32 = 9.999_997e-1;
    const EXP_MAGIC: f32 = 12_582_912.0;
    const EXP_MAGIC_BITS: i32 = 0x4B40_0000;

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmax8(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let m4 = _mm_max_ps(lo, hi);
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let m1 = _mm_max_ss(m2, _mm_shuffle_ps::<1>(m2, m2));
        _mm_cvtss_f32(m1)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum8(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s4 = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<1>(s2, s2));
        _mm_cvtss_f32(s1)
    }

    /// Vectorised `exp(u)` for `u <= 0` already clamped to `>= -104.0` (degree-5 minimax, max rel err `9.5e-8`).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn exp_neg8(uc: __m256) -> __m256 {
        let magic = _mm256_set1_ps(EXP_MAGIC);
        let z = _mm256_add_ps(_mm256_mul_ps(uc, _mm256_set1_ps(LOG2E)), magic);
        let n = _mm256_sub_ps(z, magic);
        let r = _mm256_sub_ps(
            _mm256_sub_ps(uc, _mm256_mul_ps(n, _mm256_set1_ps(LN2_HI))),
            _mm256_mul_ps(n, _mm256_set1_ps(LN2_LO)),
        );
        let mut p = _mm256_set1_ps(EXP5_C5);
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C4));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C3));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C2));
        p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP5_C1));
        let y = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.0));
        let ni = _mm256_sub_epi32(_mm256_castps_si256(z), _mm256_set1_epi32(EXP_MAGIC_BITS));
        let n1 = _mm256_srai_epi32::<1>(ni);
        let n2 = _mm256_sub_epi32(ni, n1);
        let bias = _mm256_set1_epi32(127);
        let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias)));
        let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias)));
        _mm256_mul_ps(_mm256_mul_ps(y, s1), s2)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn expectation(logits: &[f32; 16]) -> Option<f32> {
        let ptr = logits.as_ptr();
        let v0 = unsafe { _mm256_loadu_ps(ptr) };
        let v1 = unsafe { _mm256_loadu_ps(ptr.add(8)) };
        if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v1)) != 0 {
            return None;
        }
        let m = unsafe { hmax8(_mm256_max_ps(v0, v1)) };
        if !m.is_finite() {
            return None;
        }
        let vm = _mm256_set1_ps(m);
        let floor = _mm256_set1_ps(-104.0);
        let uc0 = _mm256_max_ps(_mm256_sub_ps(v0, vm), floor);
        let uc1 = _mm256_max_ps(_mm256_sub_ps(v1, vm), floor);
        let p0 = unsafe { exp_neg8(uc0) };
        let p1 = unsafe { exp_neg8(uc1) };
        let k0 = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let k1 = _mm256_setr_ps(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0);
        let z = unsafe { hsum8(_mm256_add_ps(p0, p1)) };
        let e = unsafe { hsum8(_mm256_add_ps(_mm256_mul_ps(p0, k0), _mm256_mul_ps(p1, k1))) };
        Some(e / z)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn moments(logits: &[f32; 16]) -> Option<(f32, f32)> {
        let ptr = logits.as_ptr();
        let v0 = unsafe { _mm256_loadu_ps(ptr) };
        let v1 = unsafe { _mm256_loadu_ps(ptr.add(8)) };
        if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v1)) != 0 {
            return None;
        }
        let m = unsafe { hmax8(_mm256_max_ps(v0, v1)) };
        if !m.is_finite() {
            return None;
        }
        let vm = _mm256_set1_ps(m);
        let floor = _mm256_set1_ps(-104.0);
        let uc0 = _mm256_max_ps(_mm256_sub_ps(v0, vm), floor);
        let uc1 = _mm256_max_ps(_mm256_sub_ps(v1, vm), floor);
        let p0 = unsafe { exp_neg8(uc0) };
        let p1 = unsafe { exp_neg8(uc1) };
        let k0 = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let k1 = _mm256_setr_ps(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0);
        let k0_sq = _mm256_setr_ps(0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0);
        let k1_sq = _mm256_setr_ps(64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0);
        let z = unsafe { hsum8(_mm256_add_ps(p0, p1)) };
        let e = unsafe { hsum8(_mm256_add_ps(_mm256_mul_ps(p0, k0), _mm256_mul_ps(p1, k1))) };
        let e2 = unsafe { hsum8(_mm256_add_ps(_mm256_mul_ps(p0, k0_sq), _mm256_mul_ps(p1, k1_sq))) };
        let mean = e / z;
        Some((mean, (e2 / z - mean * mean).max(0.0)))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn moments_and_entropy(logits: &[f32; 16]) -> Option<(f32, f32, f32)> {
        let ptr = logits.as_ptr();
        let v0 = unsafe { _mm256_loadu_ps(ptr) };
        let v1 = unsafe { _mm256_loadu_ps(ptr.add(8)) };
        if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v1)) != 0 {
            return None;
        }
        let m = unsafe { hmax8(_mm256_max_ps(v0, v1)) };
        if !m.is_finite() {
            return None;
        }
        let vm = _mm256_set1_ps(m);
        let floor = _mm256_set1_ps(-104.0);
        let uc0 = _mm256_max_ps(_mm256_sub_ps(v0, vm), floor);
        let uc1 = _mm256_max_ps(_mm256_sub_ps(v1, vm), floor);
        let p0 = unsafe { exp_neg8(uc0) };
        let p1 = unsafe { exp_neg8(uc1) };
        let k0 = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        let k1 = _mm256_setr_ps(8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0);
        let k0_sq = _mm256_setr_ps(0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0);
        let k1_sq = _mm256_setr_ps(64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0);
        let z = unsafe { hsum8(_mm256_add_ps(p0, p1)) };
        let e = unsafe { hsum8(_mm256_add_ps(_mm256_mul_ps(p0, k0), _mm256_mul_ps(p1, k1))) };
        let e2 = unsafe { hsum8(_mm256_add_ps(_mm256_mul_ps(p0, k0_sq), _mm256_mul_ps(p1, k1_sq))) };
        let su = unsafe { hsum8(_mm256_add_ps(_mm256_mul_ps(p0, uc0), _mm256_mul_ps(p1, uc1))) };
        let mean = e / z;
        let var = (e2 / z - mean * mean).max(0.0);
        let raw_ent = z.ln() - su / z;
        let ent = if raw_ent.is_finite() { raw_ent.max(0.0) } else { f32::NAN };
        Some((mean, var, ent))
    }
}

/// `sum_i i * softmax(logits)_i`, computed stably (max-subtracted). `0` for an empty slice.
#[inline]
pub fn softmax_expectation(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    if let Ok(arr) = <&[f32; 16]>::try_from(logits)
        && std::is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 support checked above; `arr` has 16 elements.
        if let Some(v) = unsafe { avx2_16::expectation(arr) } {
            return v;
        }
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (mut z, mut e) = (0.0f32, 0.0f32);
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp();
        z += p;
        e += p * i as f32;
    }
    e / z
}

/// Mean and variance (in bins and bins²) of `softmax(logits)` over the bin indices, computed stably
/// (max-subtracted); the mean is [`softmax_expectation`]'s. `(0, 0)` for an empty slice.
#[inline]
pub fn softmax_moments(logits: &[f32]) -> (f32, f32) {
    if logits.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if let Ok(arr) = <&[f32; 16]>::try_from(logits)
        && std::is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 support checked above; `arr` has 16 elements.
        if let Some(v) = unsafe { avx2_16::moments(arr) } {
            return v;
        }
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (mut z, mut e, mut e2) = (0.0f32, 0.0f32, 0.0f32);
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp();
        let k = i as f32;
        z += p;
        e += p * k;
        e2 += p * k * k;
    }
    let mean = e / z;
    (mean, (e2 / z - mean * mean).max(0.0))
}

/// Mean, variance (in bins and bins²) and Shannon entropy (in nats, `>= 0`) of `softmax(logits)` over the bin
/// indices, computed stably (max-subtracted) in a single pass. `(mean, var)` is bit-for-bit identical to
/// [`softmax_moments`]; `(0, 0, 0)` for an empty slice. `-∞` acts as a zero-mass bin when at least one finite
/// logit is present (and no logit is `+∞` or `NaN`); otherwise non-finite logits yield `NaN` entropy.
#[inline]
pub fn softmax_moments_and_entropy(logits: &[f32]) -> (f32, f32, f32) {
    if logits.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if let Ok(arr) = <&[f32; 16]>::try_from(logits)
        && std::is_x86_feature_detected!("avx2")
    {
        // SAFETY: AVX2 support checked above; `arr` has 16 elements.
        if let Some(v) = unsafe { avx2_16::moments_and_entropy(arr) } {
            return v;
        }
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (mut z, mut e, mut e2, mut su) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (i, &l) in logits.iter().enumerate() {
        let u = l - m;
        let p = u.exp();
        let k = i as f32;
        z += p;
        e += p * k;
        e2 += p * k * k;
        if p > 0.0 {
            su += p * u;
        }
    }
    let mean = e / z;
    let var = (e2 / z - mean * mean).max(0.0);
    let raw_ent = z.ln() - su / z;
    let ent = if raw_ent.is_finite() { raw_ent.max(0.0) } else { f32::NAN };
    (mean, var, ent)
}

/// The side precision of DFL evidence fusion (docs/MODEL.md §9.4) for a side whose distribution has `mean` and
/// `var` (bins, bins²) at an anchor of stride `stride`: `1 / (stride² · (max(var − f·(1 − f), 0) + ε))` with
/// `f = mean − floor(mean)` and `ε =` [`crate::fusion::DFL_FUSION_EPS`]. `f·(1 − f)` is the variance DFL trains a
/// perfectly certain edge at `mean` to (its two neighbouring bins), so only the excess is uncertainty.
#[inline]
pub fn side_precision(mean: f32, var: f32, stride: f32) -> f32 {
    if !mean.is_finite() || !var.is_finite() || !stride.is_finite() || stride <= 0.0 {
        return 0.0;
    }
    let f = mean - mean.floor();
    let excess = (var - f * (1.0 - f)).max(0.0);
    1.0 / (stride * stride * (excess + crate::fusion::DFL_FUSION_EPS))
}

/// Scale-free DFL bin precision `1 / (max(var − f·(1 − f), 0) + ε)` (in bins⁻²), where `f = mean − floor(mean)`
/// and `ε =` [`crate::fusion::DFL_FUSION_EPS`]: [`side_precision`] without the `1 / stride²` pixel scaling, used
/// by [`crate::fusion::ScoreRule::DflConsensus`] (docs/MODEL.md §9.4).
#[inline]
pub fn bin_precision(mean: f32, var: f32) -> f32 {
    if !mean.is_finite() || !var.is_finite() {
        return 0.0;
    }
    let f = mean - mean.floor();
    let excess = (var - f * (1.0 - f)).max(0.0);
    1.0 / (excess + crate::fusion::DFL_FUSION_EPS)
}

/// The four [`side_precision`]s (`l, t, r, b`) of one anchor from its `4 * reg_max` DFL logits (one row of
/// [`dfl_decode_into`]'s `reg`), at `stride`.
pub fn side_precisions(row: &[f32], reg_max: usize, stride: f32) -> [f32; 4] {
    std::array::from_fn(|side| {
        let (mean, var) = softmax_moments(&row[side * reg_max..(side + 1) * reg_max]);
        side_precision(mean, var, stride)
    })
}

/// The four [`bin_precision`]s (`l, t, r, b`) of one anchor from its `4 * reg_max` DFL logits (one row of
/// [`dfl_decode_into`]'s `reg`), in bins⁻².
pub fn bin_precisions(row: &[f32], reg_max: usize) -> [f32; 4] {
    std::array::from_fn(|side| {
        let (mean, var) = softmax_moments(&row[side * reg_max..(side + 1) * reg_max]);
        bin_precision(mean, var)
    })
}

/// Mean 4-side Shannon entropy (in nats) of one anchor from its `4 * reg_max` DFL logits (one row of
/// [`dfl_decode_into`]'s `reg`), used by [`crate::fusion::ScoreRule::DflRepulse`] (docs/MODEL.md §9.4).
pub fn dfl_entropy(row: &[f32], reg_max: usize) -> f32 {
    if reg_max == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for side in 0..4 {
        sum += softmax_moments_and_entropy(&row[side * reg_max..(side + 1) * reg_max]).2;
    }
    0.25 * sum
}

/// DFL decode: `reg` `[A, 4 * reg_max]` → `out` `[A, 4]` `ltrb` distances in stride units.
///
/// # Panics
/// If `reg_max == 0`, `reg.len()` is not a multiple of `4 * reg_max`, or `out.len() != 4 * A`.
pub fn dfl_decode_into(reg: &[f32], reg_max: usize, out: &mut [f32]) {
    assert!(reg_max > 0, "dfl_decode: reg_max must be > 0");
    assert_eq!(reg.len() % (4 * reg_max), 0, "dfl_decode: reg must be [A, 4 * reg_max]");
    assert_eq!(out.len(), reg.len() / reg_max, "dfl_decode: out must be [A, 4]");
    let row = |(r, o): (&[f32], &mut [f32; 4])| {
        for (side, o) in o.iter_mut().enumerate() {
            *o = softmax_expectation(&r[side * reg_max..(side + 1) * reg_max]);
        }
    };
    let rows = reg.len() / (4 * reg_max);
    if rows >= PAR_ANCHORS {
        reg.par_chunks_exact(4 * reg_max).zip(out.as_chunks_mut::<4>().0.par_iter_mut()).for_each(row);
    } else {
        reg.chunks_exact(4 * reg_max).zip(out.as_chunks_mut::<4>().0.iter_mut()).for_each(row);
    }
}

/// `ltrb` distances around `anchor` → box, in the same units as the inputs.
#[inline]
pub fn dist2bbox(ltrb: [f32; 4], anchor: [f32; 2]) -> Xyxy {
    Xyxy::new(anchor[0] - ltrb[0], anchor[1] - ltrb[1], anchor[0] + ltrb[2], anchor[1] + ltrb[3])
}

/// Box → `ltrb` distances from `anchor`, in the same units as the inputs (unclamped inverse of [`dist2bbox`]).
#[inline]
pub fn bbox2dist(anchor: [f32; 2], b: &Xyxy) -> [f32; 4] {
    [anchor[0] - b.x1, anchor[1] - b.y1, b.x2 - anchor[0], b.y2 - anchor[1]]
}

/// Decode DFL logits `[A, 4 * reg_max]` into boxes in input pixels.
///
/// # Panics
/// If `reg.len() != anchors.len() * 4 * reg_max`.
pub fn decode_boxes(reg: &[f32], anchors: &Anchors, reg_max: usize) -> Vec<Xyxy> {
    assert_eq!(reg.len(), anchors.len() * 4 * reg_max, "decode_boxes: reg must be [A, 4 * reg_max]");
    let one = |i: usize, r: &[f32]| {
        let mut d = [0.0f32; 4];
        for (side, d) in d.iter_mut().enumerate() {
            *d = softmax_expectation(&r[side * reg_max..(side + 1) * reg_max]);
        }
        let s = anchors.strides[i];
        dist2bbox(d, anchors.points[i]).scale(s, s)
    };
    if anchors.len() >= PAR_ANCHORS {
        reg.par_chunks_exact(4 * reg_max).enumerate().map(|(i, r)| one(i, r)).collect()
    } else {
        reg.chunks_exact(4 * reg_max).enumerate().map(|(i, r)| one(i, r)).collect()
    }
}

/// Logistic sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Elementwise sigmoid of class logits.
pub fn sigmoid_all(logits: &[f32]) -> Vec<f32> {
    if logits.len() >= PAR_ANCHORS * 16 {
        logits.par_iter().map(|&x| sigmoid(x)).collect()
    } else {
        logits.iter().map(|&x| sigmoid(x)).collect()
    }
}

/// Decode one image's raw head outputs: boxes in input pixels and post-sigmoid class scores `[A, num_classes]`,
/// ready for [`crate::nms::non_max_suppression`].
///
/// # Panics
/// If the shapes do not match `anchors.len()`.
pub fn decode_head(
    reg: &[f32],
    cls_logits: &[f32],
    anchors: &Anchors,
    reg_max: usize,
    num_classes: usize,
) -> (Vec<Xyxy>, Vec<f32>) {
    assert_eq!(cls_logits.len(), anchors.len() * num_classes, "decode_head: cls must be [A, num_classes]");
    (decode_boxes(reg, anchors, reg_max), sigmoid_all(cls_logits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_640() {
        let a = make_anchors(640, 640, &[8, 16, 32], 0.5);
        assert_eq!(a.len(), 8400);
        assert_eq!(
            a.levels.iter().map(|l| (l.height, l.start)).collect::<Vec<_>>(),
            [(80, 0), (40, 6400), (20, 8000)]
        );
        assert_eq!(a.points[0], [0.5, 0.5]);
        assert_eq!(a.points[1], [1.5, 0.5]);
        assert_eq!(a.points[80], [0.5, 1.5]);
        assert_eq!(a.point_px(6400), [8.0, 8.0]);
        assert_eq!(a.point_px(8399), [624.0, 624.0]);
        // MODEL.md §10.2: a rectangular 384x640 (H x W) input has 48·80 + 24·40 + 12·20 = 5040 anchors.
        let a = make_anchors(384, 640, &[8, 16, 32], 0.5);
        assert_eq!(a.len(), 5040);
        assert_eq!(a.levels[2], AnchorLevel { stride: 32, height: 12, width: 20, start: 4800 });
        assert_eq!(a.point_px(4800 + 20), [16.0, 48.0]);
        // Non-square, non-multiple input (outside the spec, which requires multiples of 32): ceil, like the convs.
        let a = make_anchors(100, 50, &[32], 0.5);
        assert_eq!((a.levels[0].height, a.levels[0].width), (4, 2));
    }

    #[test]
    fn softmax_expectation_values() {
        // Uniform over 4 bins: mean of 0..3 = 1.5.
        assert!((softmax_expectation(&[0.0; 4]) - 1.5).abs() < 1e-6);
        // Peaked at bin 2.
        assert!((softmax_expectation(&[-100.0, -100.0, 100.0, -100.0]) - 2.0).abs() < 1e-6);
        // Two equal peaks at bins 1 and 3 → 2. Large logits stay finite.
        assert!((softmax_expectation(&[0.0, 1e4, 0.0, 1e4]) - 2.0).abs() < 1e-6);
        assert_eq!(softmax_expectation(&[]), 0.0);
    }

    #[test]
    fn softmax_moments_and_entropy_values() {
        assert_eq!(softmax_moments_and_entropy(&[]), (0.0, 0.0, 0.0));
        // Uniform over 16 bins: H = ln(16).
        let u = [0.0f32; 16];
        let (m, v, h) = softmax_moments_and_entropy(&u);
        assert_eq!((m.to_bits(), v.to_bits()), {
            let (m0, v0) = softmax_moments(&u);
            (m0.to_bits(), v0.to_bits())
        });
        assert!((h - 16f32.ln()).abs() < 1e-5, "{h}");
        // Peaked at one bin: H = 0, even with -Inf elsewhere.
        let peaked = [f32::NEG_INFINITY, 10.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        assert_eq!(softmax_moments_and_entropy(&peaked), (1.0, 0.0, 0.0));
        // Two equal peaks: H = ln(2).
        let (m2, v2, h2) = softmax_moments_and_entropy(&[0.0, 1e4, 0.0, 1e4]);
        assert!((m2 - 2.0).abs() < 1e-6 && (v2 - 1.0).abs() < 1e-5 && (h2 - 2f32.ln()).abs() < 1e-5);
        // Non-finite logits yield NaN entropy (never mistaken for zero entropy).
        assert!(softmax_moments_and_entropy(&[f32::NAN; 4]).2.is_nan());
        assert!(softmax_moments_and_entropy(&[f32::INFINITY, 0.0, 0.0, 0.0]).2.is_nan());
        assert!(softmax_moments_and_entropy(&[f32::NEG_INFINITY; 4]).2.is_nan());
        // `dfl_entropy` averages the four sides.
        let mut row = vec![-100.0f32; 16];
        row[0..4].fill(0.0); // side 0 uniform (ln 4), sides 1..4 peaked at bin 0 (0)
        row[4] = 100.0;
        row[8] = 100.0;
        row[12] = 100.0;
        assert!((dfl_entropy(&row, 4) - 0.25 * 4f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn decode_round_trip() {
        let a = make_anchors(64, 64, &[8, 16], 0.5);
        let reg_max = 16;
        // Peaked logits: side k of anchor i at bin (i + k) % 16.
        let mut reg = vec![-50.0f32; a.len() * 4 * reg_max];
        for i in 0..a.len() {
            for k in 0..4 {
                reg[i * 4 * reg_max + k * reg_max + (i + k) % reg_max] = 50.0;
            }
        }
        let boxes = decode_boxes(&reg, &a, reg_max);
        let mut ltrb = vec![0.0; a.len() * 4];
        dfl_decode_into(&reg, reg_max, &mut ltrb);
        for i in 0..a.len() {
            let s = a.strides[i];
            let want: [f32; 4] = std::array::from_fn(|k| ((i + k) % reg_max) as f32);
            for k in 0..4 {
                assert!((ltrb[i * 4 + k] - want[k]).abs() < 1e-5);
            }
            let px = a.point_px(i);
            let d = bbox2dist(px, &boxes[i]);
            for k in 0..4 {
                assert!((d[k] - want[k] * s).abs() < 1e-3, "anchor {i} side {k}");
            }
        }
        let (bx, sc) = decode_head(&reg, &vec![0.0; a.len() * 3], &a, reg_max, 3);
        assert_eq!(bx, boxes);
        assert!(sc.iter().all(|&s| s == 0.5));
    }

    #[test]
    fn softmax_16_avx2_matches_scalar_reference() {
        fn ref_moments_ent(logits: &[f32]) -> (f32, f32, f32) {
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let (mut z, mut e, mut e2, mut su) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for (i, &l) in logits.iter().enumerate() {
                let u = l - m;
                let p = u.exp();
                let k = i as f32;
                z += p;
                e += p * k;
                e2 += p * k * k;
                if p > 0.0 {
                    su += p * u;
                }
            }
            let mean = e / z;
            let var = (e2 / z - mean * mean).max(0.0);
            let raw_ent = z.ln() - su / z;
            let ent = if raw_ent.is_finite() { raw_ent.max(0.0) } else { f32::NAN };
            (mean, var, ent)
        }
        let mut s = 0x1234_5678u64;
        for case in 0..200 {
            let mut logits = [0.0f32; 16];
            for x in &mut logits {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 40) as f32) / ((1u32 << 24) as f32);
                *x = (u * 2.0 - 1.0) * if case % 5 == 0 { 80.0 } else { 8.0 };
            }
            if case == 10 {
                logits[0] = f32::NEG_INFINITY;
                logits[15] = f32::NEG_INFINITY;
            }
            let (rm, rv, rh) = ref_moments_ent(&logits);
            let gm = softmax_expectation(&logits);
            let (gm2, gv2) = softmax_moments(&logits);
            let (gm3, gv3, gh3) = softmax_moments_and_entropy(&logits);
            assert_eq!(gm2.to_bits(), gm3.to_bits(), "moments vs moments_and_entropy mean");
            assert_eq!(gv2.to_bits(), gv3.to_bits(), "moments vs moments_and_entropy var");
            assert!((gm - rm).abs() <= 1e-5, "case {case} exp: {gm} vs {rm}");
            assert!((gm2 - rm).abs() <= 1e-5, "case {case} mean: {gm2} vs {rm}");
            assert!((gv2 - rv).abs() <= 1e-4, "case {case} var: {gv2} vs {rv}");
            assert!((gh3 - rh).abs() <= 2e-5, "case {case} ent: {gh3} vs {rh}");
        }
        // Non-finite 16-bin edge cases fall back cleanly.
        assert!(softmax_moments_and_entropy(&[f32::NAN; 16]).2.is_nan());
        let mut inf16 = [0.0f32; 16];
        inf16[7] = f32::INFINITY;
        assert!(softmax_moments_and_entropy(&inf16).2.is_nan());
        assert!(softmax_moments_and_entropy(&[f32::NEG_INFINITY; 16]).2.is_nan());
    }
}
