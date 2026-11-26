//! NMS suppression kernels: standard IoU NMS, DIoU-NMS, Soft-NMS (Linear / Gaussian), and batched
//! class-aware bucketing, with AVX2-vectorised SoA suppression loops on x86-64.

use crate::boxes::{IouKind, Xyxy, box_iou, iou};
use rayon::prelude::*;

use super::NmsKind;

/// Remaining-box count above which one suppression pass is split across threads.
const PAR_SUPPRESS: usize = 1 << 14;
/// Total candidate count above which class buckets are processed in parallel.
const PAR_BUCKETS: usize = 2048;

/// Greedy NMS over `boxes` sorted by descending score. Returns the kept positions in order, stopping after `limit`
/// survivors. A box is suppressed when its IoU with a kept box is `> iou_thr`.
pub fn nms_sorted(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> Vec<usize> {
    nms_sorted_inner(boxes, iou_thr, limit, false, false).0
}

/// [`nms_sorted`] also returning, for every input position `i`, the position in `keep` of the kept box that first
/// suppressed `i` (or `i`'s own position in `keep` when kept; `u32::MAX` if unsuppressed past `limit`).
pub fn nms_sorted_with_owner(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> (Vec<usize>, Vec<u32>) {
    nms_sorted_inner(boxes, iou_thr, limit, true, false)
}

/// Greedy DIoU-NMS (Zheng et al. 2020) over `boxes` sorted by descending score.
///
/// Suppresses a box when `IoU - rho² / c² > iou_thr`, preserving overlapping objects whose centres are distinct.
pub fn diou_nms_sorted(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> Vec<usize> {
    nms_sorted_inner(boxes, iou_thr, limit, false, true).0
}

/// [`diou_nms_sorted`] with per-input owner tracking (same convention as [`nms_sorted_with_owner`]).
pub fn diou_nms_sorted_with_owner(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> (Vec<usize>, Vec<u32>) {
    nms_sorted_inner(boxes, iou_thr, limit, true, true)
}

/// Soft-NMS (Bodla et al. 2017) over `boxes` and `scores` (not required to be pre-sorted).
///
/// Decays overlapping boxes' positive scores instead of hard-dropping them:
/// - `sigma == None`: Linear Soft-NMS — when `IoU > iou_thr` and `score > 0`, `score *= 1 - IoU`.
/// - `sigma == Some(s)`: Gaussian Soft-NMS — when `IoU > 0` and `score > 0`, `score *= exp(-IoU² / s)`.
///
/// Non-positive scores are never multiplied (so decay can never increase a score), preserving descending
/// output order for every finite `score_thr`.
///
/// Returns `(input_index, decayed_score)` pairs sorted by descending decayed score (ties: lower input index),
/// keeping only boxes whose final score is `> score_thr` and stopping at `limit`.
///
/// # Panics
/// If `boxes.len() != scores.len()`, or `sigma` is `Some(s)` with `!s.is_finite() || s <= 0.0`.
pub fn soft_nms_sorted(
    boxes: &[Xyxy],
    scores: &[f32],
    iou_thr: f32,
    score_thr: f32,
    sigma: Option<f32>,
    limit: usize,
) -> Vec<(usize, f32)> {
    assert_eq!(boxes.len(), scores.len(), "soft_nms_sorted: boxes/scores length mismatch");
    if let Some(s) = sigma {
        assert!(s.is_finite() && s > 0.0, "soft_nms_sorted: sigma must be finite and > 0, got {s}");
    }
    let n = boxes.len();
    if n == 0 || limit == 0 {
        return Vec::new();
    }
    let mut work_scores = scores.to_vec();
    let mut active: Vec<usize> = (0..n).filter(|&i| work_scores[i] > score_thr).collect();
    let mut kept: Vec<(usize, f32)> = Vec::with_capacity(active.len().min(limit));

    while kept.len() < limit && !active.is_empty() {
        // Select highest-scoring active box (ties: smaller input index).
        let mut best_pos = 0usize;
        let mut best_idx = active[0];
        let mut best_score = work_scores[best_idx];
        for (pos, &idx) in active.iter().enumerate().skip(1) {
            let s = work_scores[idx];
            if s > best_score || (s == best_score && idx < best_idx) {
                best_pos = pos;
                best_idx = idx;
                best_score = s;
            }
        }
        active.swap_remove(best_pos);
        if best_score <= score_thr || best_score.is_nan() {
            break;
        }
        kept.push((best_idx, best_score));
        if kept.len() == limit {
            break;
        }
        let best_box = &boxes[best_idx];
        let mut pos = active.len();
        while pos > 0 {
            pos -= 1;
            let idx = active[pos];
            let cur = work_scores[idx];
            if cur <= 0.0 {
                continue;
            }
            let ov = box_iou(best_box, &boxes[idx]);
            let decayed = match sigma {
                Some(s) => {
                    if ov > 0.0 {
                        cur * (-(ov * ov) / s).exp()
                    } else {
                        cur
                    }
                }
                None => {
                    if ov > iou_thr {
                        cur * (1.0 - ov)
                    } else {
                        cur
                    }
                }
            };
            work_scores[idx] = decayed;
            if decayed <= score_thr || decayed.is_nan() {
                active.swap_remove(pos);
            }
        }
    }
    kept
}

#[allow(clippy::too_many_arguments)]
fn suppress_iou_scalar(
    bx1: f32,
    by1: f32,
    bx2: f32,
    by2: f32,
    ba: f32,
    cx1: &[f32],
    cy1: &[f32],
    cx2: &[f32],
    cy2: &[f32],
    ca: &[f32],
    iou_thr: f32,
    kp: u32,
    sup: &mut [bool],
    own: Option<&mut [u32]>,
) {
    if let Some(own) = own {
        for k in 0..sup.len() {
            if sup[k] {
                continue;
            }
            let w = (bx2.min(cx2[k]) - bx1.max(cx1[k])).max(0.0);
            let h = (by2.min(cy2[k]) - by1.max(cy1[k])).max(0.0);
            let inter = w * h;
            let union = ba + ca[k] - inter;
            let ov = if union > 0.0 { inter / union } else { 0.0 };
            if ov > iou_thr {
                sup[k] = true;
                own[k] = kp;
            }
        }
    } else {
        for k in 0..sup.len() {
            // Same arithmetic as `box_iou`, so results match it bit for bit.
            let w = (bx2.min(cx2[k]) - bx1.max(cx1[k])).max(0.0);
            let h = (by2.min(cy2[k]) - by1.max(cy1[k])).max(0.0);
            let inter = w * h;
            let union = ba + ca[k] - inter;
            let ov = if union > 0.0 { inter / union } else { 0.0 };
            if ov > iou_thr {
                sup[k] = true;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn suppress_iou_avx2(
    bx1: f32,
    by1: f32,
    bx2: f32,
    by2: f32,
    ba: f32,
    cx1: &[f32],
    cy1: &[f32],
    cx2: &[f32],
    cy2: &[f32],
    ca: &[f32],
    iou_thr: f32,
    kp: u32,
    sup: &mut [bool],
    mut own: Option<&mut [u32]>,
) {
    use std::arch::x86_64::*;

    let n = sup.len();
    debug_assert_eq!(cx1.len(), n);
    debug_assert_eq!(cy1.len(), n);
    debug_assert_eq!(cx2.len(), n);
    debug_assert_eq!(cy2.len(), n);
    debug_assert_eq!(ca.len(), n);
    if let Some(ref o) = own {
        debug_assert_eq!(o.len(), n);
    }

    let vbx1 = _mm256_set1_ps(bx1);
    let vby1 = _mm256_set1_ps(by1);
    let vbx2 = _mm256_set1_ps(bx2);
    let vby2 = _mm256_set1_ps(by2);
    let vba = _mm256_set1_ps(ba);
    let vzero = _mm256_setzero_ps();
    let vthr = _mm256_set1_ps(iou_thr);

    // Branch-free 8-lane IoU hit mask (exact `vdivps`, bit-identical to `box_iou` on finite inputs).
    macro_rules! iou_hits_8 {
        ($off:expr) => {{
            let vx1 = unsafe { _mm256_loadu_ps(cx1.as_ptr().add($off)) };
            let vy1 = unsafe { _mm256_loadu_ps(cy1.as_ptr().add($off)) };
            let vx2 = unsafe { _mm256_loadu_ps(cx2.as_ptr().add($off)) };
            let vy2 = unsafe { _mm256_loadu_ps(cy2.as_ptr().add($off)) };
            let va = unsafe { _mm256_loadu_ps(ca.as_ptr().add($off)) };
            let iw = _mm256_max_ps(_mm256_sub_ps(_mm256_min_ps(vbx2, vx2), _mm256_max_ps(vbx1, vx1)), vzero);
            let ih = _mm256_max_ps(_mm256_sub_ps(_mm256_min_ps(vby2, vy2), _mm256_max_ps(vby1, vy1)), vzero);
            let inter = _mm256_mul_ps(iw, ih);
            let union = _mm256_sub_ps(_mm256_add_ps(vba, va), inter);
            let pos_u = _mm256_cmp_ps::<_CMP_GT_OQ>(union, vzero);
            let ov = _mm256_and_ps(_mm256_div_ps(inter, union), pos_u);
            _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(ov, vthr)) as u32
        }};
    }

    let mut k = 0usize;
    while k + 16 <= n {
        let hits = iou_hits_8!(k) | (iou_hits_8!(k + 8) << 8);
        if hits != 0 {
            let mut m = hits;
            if let Some(ref mut o) = own {
                while m != 0 {
                    let idx = k + m.trailing_zeros() as usize;
                    if !sup[idx] {
                        sup[idx] = true;
                        o[idx] = kp;
                    }
                    m &= m - 1;
                }
            } else {
                while m != 0 {
                    sup[k + m.trailing_zeros() as usize] = true;
                    m &= m - 1;
                }
            }
        }
        k += 16;
    }
    while k + 8 <= n {
        let mut hits = iou_hits_8!(k);
        if hits != 0 {
            if let Some(ref mut o) = own {
                while hits != 0 {
                    let idx = k + hits.trailing_zeros() as usize;
                    if !sup[idx] {
                        sup[idx] = true;
                        o[idx] = kp;
                    }
                    hits &= hits - 1;
                }
            } else {
                while hits != 0 {
                    sup[k + hits.trailing_zeros() as usize] = true;
                    hits &= hits - 1;
                }
            }
        }
        k += 8;
    }
    if k < n {
        suppress_iou_scalar(
            bx1,
            by1,
            bx2,
            by2,
            ba,
            &cx1[k..],
            &cy1[k..],
            &cx2[k..],
            &cy2[k..],
            &ca[k..],
            iou_thr,
            kp,
            &mut sup[k..],
            own.map(|o| &mut o[k..]),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn suppress_diou_scalar(
    a: &Xyxy,
    cx1: &[f32],
    cy1: &[f32],
    cx2: &[f32],
    cy2: &[f32],
    iou_thr: f32,
    kp: u32,
    sup: &mut [bool],
    own: Option<&mut [u32]>,
) {
    if let Some(own) = own {
        for k in 0..sup.len() {
            if sup[k] {
                continue;
            }
            let b = Xyxy::new(cx1[k], cy1[k], cx2[k], cy2[k]);
            if iou(a, &b, IouKind::Diou) > iou_thr {
                sup[k] = true;
                own[k] = kp;
            }
        }
    } else {
        for k in 0..sup.len() {
            if sup[k] {
                continue;
            }
            let b = Xyxy::new(cx1[k], cy1[k], cx2[k], cy2[k]);
            if iou(a, &b, IouKind::Diou) > iou_thr {
                sup[k] = true;
            }
        }
    }
}

fn nms_sorted_inner(
    boxes: &[Xyxy],
    iou_thr: f32,
    limit: usize,
    track_owner: bool,
    diou: bool,
) -> (Vec<usize>, Vec<u32>) {
    let n = boxes.len();
    let mut keep = Vec::new();
    let mut owner = if track_owner { vec![u32::MAX; n] } else { Vec::new() };
    if n == 0 || limit == 0 {
        return (keep, owner);
    }
    // Structure-of-arrays so the suppression pass vectorises cleanly.
    let mut x1 = Vec::with_capacity(n);
    let mut y1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y2 = Vec::with_capacity(n);
    let mut area = Vec::with_capacity(n);
    let mut all_finite = iou_thr.is_finite();
    for b in boxes {
        all_finite &= b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite();
        x1.push(b.x1);
        y1.push(b.y1);
        x2.push(b.x2);
        y2.push(b.y2);
        area.push(b.area());
    }
    let mut suppressed = vec![false; n];
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = !diou && all_finite && std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let _ = all_finite;

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let kp = keep.len() as u32;
        keep.push(i);
        if track_owner {
            owner[i] = kp;
        } else if keep.len() == limit {
            break;
        }
        let (bx1, by1, bx2, by2, ba) = (x1[i], y1[i], x2[i], y2[i], area[i]);
        let bi = boxes[i];
        let start = i + 1;
        let pass = |off: usize, sup: &mut [bool], own: Option<&mut [u32]>| {
            let lo = start + off;
            let hi = lo + sup.len();
            let (cx1, cy1, cx2, cy2, ca) =
                (&x1[lo..hi], &y1[lo..hi], &x2[lo..hi], &y2[lo..hi], &area[lo..hi]);
            if diou {
                suppress_diou_scalar(&bi, cx1, cy1, cx2, cy2, iou_thr, kp, sup, own);
            } else {
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    // SAFETY: AVX2 support detected above, and all inputs are finite.
                    unsafe {
                        suppress_iou_avx2(
                            bx1, by1, bx2, by2, ba, cx1, cy1, cx2, cy2, ca, iou_thr, kp, sup, own,
                        );
                    }
                    return;
                }
                suppress_iou_scalar(bx1, by1, bx2, by2, ba, cx1, cy1, cx2, cy2, ca, iou_thr, kp, sup, own);
            }
        };
        let rest_sup = &mut suppressed[start..];
        if track_owner {
            let rest_own = &mut owner[start..];
            if rest_sup.len() >= PAR_SUPPRESS {
                const CHUNK: usize = 4096;
                rest_sup
                    .par_chunks_mut(CHUNK)
                    .zip(rest_own.par_chunks_mut(CHUNK))
                    .enumerate()
                    .for_each(|(c, (sup, own))| pass(c * CHUNK, sup, Some(own)));
            } else {
                pass(0, rest_sup, Some(rest_own));
            }
            if keep.len() == limit {
                break;
            }
        } else if rest_sup.len() >= PAR_SUPPRESS {
            const CHUNK: usize = 4096;
            rest_sup.par_chunks_mut(CHUNK).enumerate().for_each(|(c, sup)| pass(c * CHUNK, sup, None));
        } else {
            pass(0, rest_sup, None);
        }
    }
    (keep, owner)
}

/// Order of indices by descending score, ties by ascending index. NaN scores sort last.
fn score_order(scores: &[f32], idx: &mut [u32]) {
    let key = |i: &u32| {
        let s = scores[*i as usize];
        if s.is_nan() { f32::NEG_INFINITY } else { s }
    };
    idx.sort_unstable_by(|a, b| key(b).total_cmp(&key(a)).then(a.cmp(b)));
}

/// Group `order` (already sorted by descending score) into per-class buckets preserving score order inside each
/// bucket. Uses counting sort for dense class ids and stable sort for sparse ids.
pub(super) fn bucket_by_class(order: &[u32], classes: &[u32]) -> (Vec<u32>, Vec<(usize, usize)>) {
    let n = order.len();
    let max_class = classes.iter().copied().max().unwrap_or(0) as usize;
    if max_class <= n.max(1024) {
        let mut starts = vec![0usize; max_class + 2];
        for &c in classes {
            starts[c as usize + 1] += 1;
        }
        for c in 0..=max_class {
            starts[c + 1] += starts[c];
        }
        let mut fill = starts.clone();
        let mut bucketed = vec![0u32; n];
        for &i in order {
            let c = classes[i as usize] as usize;
            bucketed[fill[c]] = i;
            fill[c] += 1;
        }
        let ranges = starts.windows(2).map(|w| (w[0], w[1])).filter(|r| r.0 < r.1).collect();
        (bucketed, ranges)
    } else {
        let mut bucketed = order.to_vec();
        bucketed.sort_by_key(|&i| classes[i as usize]);
        let mut ranges = Vec::new();
        let mut lo = 0;
        for hi in 1..=n {
            if hi == n || classes[bucketed[hi] as usize] != classes[bucketed[lo] as usize] {
                ranges.push((lo, hi));
                lo = hi;
            }
        }
        (bucketed, ranges)
    }
}

/// Class-aware NMS. Returns indices into the inputs of the kept boxes, sorted by descending score (ties: lower
/// index first), at most `max_det` of them. No confidence filter is applied here.
///
/// # Panics
/// If the three slices differ in length.
pub fn batched_nms(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    max_det: usize,
    agnostic: bool,
) -> Vec<usize> {
    batched_nms_inner(boxes, scores, classes, iou_thr, max_det, agnostic, false, false).0
}

/// Class-aware NMS supporting [`NmsKind::Standard`], [`NmsKind::Diou`], [`NmsKind::SoftLinear`],
/// [`NmsKind::SoftGaussian`], [`NmsKind::MatrixLinear`], and [`NmsKind::MatrixGaussian`]. Returns
/// `(input_index, final_score)` pairs sorted by descending score, at most `max_det` of them. `iou_thr` is
/// ignored by `SoftGaussian`, `MatrixLinear`, and `MatrixGaussian`; `score_thr` is used by Soft-NMS and
/// Matrix-NMS to drop boxes whose decayed score falls to or below it.
///
/// # Panics
/// If the three slices differ in length, or if `SoftGaussian` / `MatrixGaussian` has
/// `!sigma.is_finite() || sigma <= 0.0`.
#[allow(clippy::too_many_arguments)]
pub fn batched_nms_with_kind(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    score_thr: f32,
    max_det: usize,
    agnostic: bool,
    kind: NmsKind,
) -> Vec<(usize, f32)> {
    match kind {
        NmsKind::Standard => {
            batched_nms_inner(boxes, scores, classes, iou_thr, max_det, agnostic, false, false)
                .0
                .into_iter()
                .map(|i| (i, scores[i]))
                .collect()
        }
        NmsKind::Diou => batched_nms_inner(boxes, scores, classes, iou_thr, max_det, agnostic, false, true)
            .0
            .into_iter()
            .map(|i| (i, scores[i]))
            .collect(),
        NmsKind::SoftLinear => {
            batched_soft_nms(boxes, scores, classes, iou_thr, score_thr, None, max_det, agnostic)
        }
        NmsKind::SoftGaussian { sigma } => {
            batched_soft_nms(boxes, scores, classes, iou_thr, score_thr, Some(sigma), max_det, agnostic)
        }
        NmsKind::MatrixLinear => {
            super::matrix::batched_matrix_nms(boxes, scores, classes, score_thr, None, max_det, agnostic)
        }
        NmsKind::MatrixGaussian { sigma } => super::matrix::batched_matrix_nms(
            boxes,
            scores,
            classes,
            score_thr,
            Some(sigma),
            max_det,
            agnostic,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batched_nms_inner(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    max_det: usize,
    agnostic: bool,
    track_owner: bool,
    diou: bool,
) -> (Vec<usize>, Option<Vec<u32>>) {
    assert_eq!(boxes.len(), scores.len(), "batched_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_nms: boxes/classes length mismatch");
    let n = boxes.len();
    if n == 0 || max_det == 0 {
        return (Vec::new(), None);
    }
    assert!(n <= u32::MAX as usize, "batched_nms: too many boxes");
    let mut order: Vec<u32> = (0..n as u32).collect();
    score_order(scores, &mut order);

    if agnostic {
        let sorted: Vec<Xyxy> = order.iter().map(|&i| boxes[i as usize]).collect();
        if track_owner {
            let (kept_s, own_s) = nms_sorted_inner(&sorted, iou_thr, max_det, true, diou);
            let kept: Vec<usize> = kept_s.into_iter().map(|k| order[k] as usize).collect();
            let mut own = vec![u32::MAX; n];
            for (pos, &o) in own_s.iter().enumerate() {
                if o != u32::MAX {
                    own[order[pos] as usize] = o;
                }
            }
            return (kept, Some(own));
        }
        let kept = nms_sorted_inner(&sorted, iou_thr, max_det, false, diou)
            .0
            .into_iter()
            .map(|k| order[k] as usize)
            .collect();
        return (kept, None);
    }

    let (bucketed, ranges) = bucket_by_class(&order, classes);
    if !track_owner {
        let run_bucket = |&(lo, hi): &(usize, usize)| -> Vec<u32> {
            let ids = &bucketed[lo..hi];
            let sorted: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
            nms_sorted_inner(&sorted, iou_thr, max_det, false, diou).0.into_iter().map(|k| ids[k]).collect()
        };
        let mut kept: Vec<u32> = if n >= PAR_BUCKETS {
            ranges.par_iter().flat_map_iter(run_bucket).collect()
        } else {
            ranges.iter().flat_map(run_bucket).collect()
        };
        score_order(scores, &mut kept);
        kept.truncate(max_det);
        return (kept.into_iter().map(|i| i as usize).collect(), None);
    }
    let run_bucket = |&(lo, hi): &(usize, usize)| -> (Vec<u32>, Vec<u32>) {
        let ids = &bucketed[lo..hi];
        let sorted: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
        let (k, own) = nms_sorted_inner(&sorted, iou_thr, max_det, true, diou);
        (k.into_iter().map(|m| ids[m]).collect(), own)
    };
    let parts: Vec<(Vec<u32>, Vec<u32>)> = if n >= PAR_BUCKETS {
        ranges.par_iter().map(run_bucket).collect()
    } else {
        ranges.iter().map(run_bucket).collect()
    };
    let mut kept: Vec<u32> = parts.iter().flat_map(|(k, _)| k.iter().copied()).collect();
    score_order(scores, &mut kept);
    kept.truncate(max_det);
    let mut global_pos = vec![u32::MAX; n];
    for (kp, &cand_id) in kept.iter().enumerate() {
        global_pos[cand_id as usize] = kp as u32;
    }
    let mut owner = vec![u32::MAX; n];
    for (&(lo, hi), (bk_kept, bk_own)) in ranges.iter().zip(&parts) {
        let ids = &bucketed[lo..hi];
        for (&cand_id, &loc_kp) in ids.iter().zip(bk_own) {
            if loc_kp != u32::MAX {
                let win = bk_kept[loc_kp as usize] as usize;
                owner[cand_id as usize] = global_pos[win];
            }
        }
    }
    (kept.into_iter().map(|i| i as usize).collect(), Some(owner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn batched_soft_nms(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    score_thr: f32,
    sigma: Option<f32>,
    max_det: usize,
    agnostic: bool,
) -> Vec<(usize, f32)> {
    assert_eq!(boxes.len(), scores.len(), "batched_soft_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_soft_nms: boxes/classes length mismatch");
    if let Some(s) = sigma {
        assert!(s.is_finite() && s > 0.0, "batched_soft_nms: sigma must be finite and > 0, got {s}");
    }
    let n = boxes.len();
    if n == 0 || max_det == 0 {
        return Vec::new();
    }
    assert!(n <= u32::MAX as usize, "batched_soft_nms: too many boxes");
    if agnostic {
        return soft_nms_sorted(boxes, scores, iou_thr, score_thr, sigma, max_det);
    }
    let order: Vec<u32> = (0..n as u32).collect();
    let (bucketed, ranges) = bucket_by_class(&order, classes);
    let run_bucket = |&(lo, hi): &(usize, usize)| -> Vec<(usize, f32)> {
        let ids = &bucketed[lo..hi];
        let b: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
        let s: Vec<f32> = ids.iter().map(|&i| scores[i as usize]).collect();
        soft_nms_sorted(&b, &s, iou_thr, score_thr, sigma, max_det)
            .into_iter()
            .map(|(k, sc)| (ids[k] as usize, sc))
            .collect()
    };
    let mut kept: Vec<(usize, f32)> = if n >= PAR_BUCKETS {
        ranges.par_iter().flat_map_iter(run_bucket).collect()
    } else {
        ranges.iter().flat_map(run_bucket).collect()
    };
    kept.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    kept.truncate(max_det);
    kept
}
