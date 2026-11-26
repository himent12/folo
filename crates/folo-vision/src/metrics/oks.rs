//! COCO-style Object Keypoint Similarity (OKS) evaluation metric and keypoint AP.
//!
//! Implements the exact COCO person-keypoint OKS definition:
//! ```text
//! OKS = (sum_{k: v_k > 0} exp(-((x_hat_k - x_k)^2 + (y_hat_k - y_k)^2) / (8 * area * sigma_k^2)))
//!       / |{k: v_k > 0}|
//! ```
//! where `v_k in {0, 1, 2}` is the COCO visibility flag (`0` = unlabeled, `1` = labeled & occluded,
//! `2` = labeled & visible), `area = s^2` is the ground-truth object area, and `sigma_k > 0` is the
//! per-keypoint falloff constant (`kappa_k = 2 * sigma_k`; [`COCO_KEYPOINT_SIGMAS`] provides the
//! 17 official COCO person sigmas).

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::metrics::{NUM_IOU_THRESHOLDS, NUM_RECALL_POINTS, iou_thresholds, recall_points};

/// Official 17-keypoint COCO person falloff standard deviations `sigma_k`
/// (`nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist,
/// l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle`).
pub const COCO_KEYPOINT_SIGMAS: [f32; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087,
    0.089, 0.089,
];

/// COCO keypoint visibility flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum KeypointVisibility {
    /// `v = 0`: not labeled (excluded from OKS numerator and denominator).
    #[default]
    Unlabeled = 0,
    /// `v = 1`: labeled but occluded (included in OKS).
    Occluded = 1,
    /// `v = 2`: labeled and visible (included in OKS).
    Visible = 2,
}

impl KeypointVisibility {
    /// Parse a numeric COCO visibility flag (`0`, `1`, or `2`).
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Unlabeled),
            1 => Ok(Self::Occluded),
            2 => Ok(Self::Visible),
            _ => Err(Error::Config(format!("KeypointVisibility: expected 0, 1, or 2, got {v}"))),
        }
    }

    /// Whether this keypoint is labeled (`v > 0`, i.e., `Occluded` or `Visible`).
    #[inline]
    pub const fn is_labeled(self) -> bool {
        !matches!(self, Self::Unlabeled)
    }
}

/// One ground-truth 2D keypoint with its COCO visibility flag (`0` / `1` / `2`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    /// Horizontal pixel coordinate.
    pub x: f32,
    /// Vertical pixel coordinate.
    pub y: f32,
    /// COCO visibility flag: `0` = unlabeled, `1` = occluded, `2` = visible.
    pub visibility: u8,
}

impl Keypoint {
    /// Create a keypoint with explicit `(x, y, visibility)`.
    #[inline]
    pub const fn new(x: f32, y: f32, visibility: u8) -> Self {
        Self { x, y, visibility }
    }

    /// Create a visible (`v = 2`) keypoint.
    #[inline]
    pub const fn visible(x: f32, y: f32) -> Self {
        Self { x, y, visibility: 2 }
    }

    /// Create an occluded (`v = 1`) keypoint.
    #[inline]
    pub const fn occluded(x: f32, y: f32) -> Self {
        Self { x, y, visibility: 1 }
    }

    /// Create an unlabeled (`v = 0`) keypoint.
    #[inline]
    pub const fn unlabeled() -> Self {
        Self { x: 0.0, y: 0.0, visibility: 0 }
    }
}

fn validate_sigmas(sigmas: &[f32], op: &str) -> Result<()> {
    for (i, &s) in sigmas.iter().enumerate() {
        if !(s.is_finite() && s > 0.0) {
            return Err(Error::Config(format!("{op}: sigmas[{i}] must be finite and > 0, got {s}")));
        }
    }
    Ok(())
}

/// Compute exact COCO Object Keypoint Similarity in `f64` between predicted keypoints `pred` and
/// ground-truth keypoints `gt` with object `area` and per-keypoint `sigmas`.
///
/// Returns `0.0` when there are no labeled keypoints (`v > 0`), when `area == 0.0`, or when any
/// labeled keypoint coordinate is non-finite.
///
/// # Errors
/// Returns [`Error::Config`] if slice lengths mismatch, any `sigma` is non-finite or `<= 0`,
/// `area` is non-finite or `< 0`, or any ground-truth visibility flag is outside `{0, 1, 2}`.
pub fn oks_f64(pred: &[[f32; 2]], gt: &[Keypoint], area: f64, sigmas: &[f32]) -> Result<f64> {
    if pred.len() != gt.len() || pred.len() != sigmas.len() {
        return Err(Error::Config(format!(
            "oks: length mismatch pred={} gt={} sigmas={}",
            pred.len(),
            gt.len(),
            sigmas.len()
        )));
    }
    if !(area.is_finite() && area >= 0.0) {
        return Err(Error::Config(format!("oks: area must be finite and >= 0, got {area}")));
    }
    validate_sigmas(sigmas, "oks")?;
    for (i, g) in gt.iter().enumerate() {
        if g.visibility > 2 {
            return Err(Error::Config(format!(
                "oks: gt[{i}].visibility must be 0, 1, or 2, got {}",
                g.visibility
            )));
        }
    }
    if area == 0.0 {
        return Ok(0.0);
    }

    let mut sum = 0.0f64;
    let mut n_vis = 0usize;
    let sqrt_area = area.sqrt();
    let two_sqrt2 = 2.0 * std::f64::consts::SQRT_2;
    for ((p, g), &s) in pred.iter().zip(gt).zip(sigmas) {
        if g.visibility == 0 {
            continue;
        }
        let (px, py, gx, gy) = (p[0] as f64, p[1] as f64, g.x as f64, g.y as f64);
        if !(px.is_finite() && py.is_finite() && gx.is_finite() && gy.is_finite()) {
            return Ok(0.0);
        }
        let scale_k = two_sqrt2 * (sqrt_area * s as f64);
        if !(scale_k.is_finite() && scale_k > 0.0) {
            return Ok(0.0);
        }
        let ux = (px - gx) / scale_k;
        let uy = (py - gy) / scale_k;
        let e = ux * ux + uy * uy;
        sum += (-e).exp();
        n_vis += 1;
    }
    if n_vis == 0 { Ok(0.0) } else { Ok((sum / n_vis as f64).clamp(0.0, 1.0)) }
}

/// COCO zero-labeled-keypoint OKS against an instance's bounding box (expanded by 1x box size on
/// each side per `pycocotools/cocoeval.py` lines 208-215).
fn oks_unlabeled_bbox_f64(pred: &[[f32; 2]], bbox: &Xyxy, area: f64, sigmas: &[f32]) -> f64 {
    if pred.is_empty() || area <= 0.0 || !area.is_finite() {
        return 0.0;
    }
    if !(bbox.x1.is_finite() && bbox.y1.is_finite() && bbox.x2.is_finite() && bbox.y2.is_finite()) {
        return 0.0;
    }
    let bx1 = bbox.x1 as f64;
    let by1 = bbox.y1 as f64;
    let bx2 = bbox.x2 as f64;
    let by2 = bbox.y2 as f64;
    let bw = (bx2 - bx1).max(0.0);
    let bh = (by2 - by1).max(0.0);
    let x0 = bx1 - bw;
    let x1 = bx2 + bw;
    let y0 = by1 - bh;
    let y1 = by2 + bh;
    let sqrt_area = area.sqrt();
    let two_sqrt2 = 2.0 * std::f64::consts::SQRT_2;

    let mut sum = 0.0f64;
    for (p, &s) in pred.iter().zip(sigmas) {
        let (px, py) = (p[0] as f64, p[1] as f64);
        if !(px.is_finite() && py.is_finite()) {
            return 0.0;
        }
        let scale_k = two_sqrt2 * (sqrt_area * s as f64);
        if !(scale_k.is_finite() && scale_k > 0.0) {
            return 0.0;
        }
        let dx = (x0 - px).max(0.0) + (px - x1).max(0.0);
        let dy = (y0 - py).max(0.0) + (py - y1).max(0.0);
        let ux = dx / scale_k;
        let uy = dy / scale_k;
        sum += (-(ux * ux + uy * uy)).exp();
    }
    (sum / pred.len() as f64).clamp(0.0, 1.0)
}

/// Compute COCO Object Keypoint Similarity in `f32` (delegates to [`oks_f64`]).
pub fn oks(pred: &[[f32; 2]], gt: &[Keypoint], area: f32, sigmas: &[f32]) -> Result<f32> {
    Ok(oks_f64(pred, gt, area as f64, sigmas)? as f32)
}

/// Pairwise OKS matrix, row-major `[preds.len(), gts.len()]`, where each `gts[j]` is `(keypoints, area)`.
pub fn oks_matrix(preds: &[&[[f32; 2]]], gts: &[(&[Keypoint], f32)], sigmas: &[f32]) -> Result<Vec<f32>> {
    validate_sigmas(sigmas, "oks_matrix")?;
    let k_len = sigmas.len();
    for (i, p) in preds.iter().enumerate() {
        if p.len() != k_len {
            return Err(Error::Config(format!(
                "oks_matrix: preds[{i}] length {} != sigmas length {k_len}",
                p.len()
            )));
        }
    }
    for (j, &(g, area)) in gts.iter().enumerate() {
        if g.len() != k_len {
            return Err(Error::Config(format!(
                "oks_matrix: gts[{j}] length {} != sigmas length {k_len}",
                g.len()
            )));
        }
        if !(area.is_finite() && area >= 0.0) {
            return Err(Error::Config(format!(
                "oks_matrix: gts[{j}] area must be finite and >= 0, got {area}"
            )));
        }
        for (ki, kp) in g.iter().enumerate() {
            if kp.visibility > 2 {
                return Err(Error::Config(format!(
                    "oks_matrix: gts[{j}][{ki}].visibility must be 0, 1, or 2, got {}",
                    kp.visibility
                )));
            }
        }
    }
    let total = preds
        .len()
        .checked_mul(gts.len())
        .filter(|&n| n.checked_mul(std::mem::size_of::<f32>()).is_some_and(|b| b <= isize::MAX as usize))
        .ok_or_else(|| {
            Error::Config(format!("oks_matrix: matrix size {} x {} overflows", preds.len(), gts.len()))
        })?;
    let mut out = Vec::with_capacity(total);
    for p in preds {
        for &(g, area) in gts {
            out.push(oks_f64(p, g, area as f64, sigmas)? as f32);
        }
    }
    Ok(out)
}

/// Predicted pose instance for [`compute_oks_ap`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OksPred {
    /// Predicted `(x, y)` coordinates of length `K`.
    pub keypoints: Vec<[f32; 2]>,
    /// Instance confidence score.
    pub score: f32,
}

/// Ground-truth pose instance for [`compute_oks_ap`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OksTruth {
    /// Ground-truth keypoints of length `K`.
    pub keypoints: Vec<Keypoint>,
    /// Object scale area `s^2`.
    pub area: f32,
    /// Optional bounding box for COCO-style ignore matching when this instance has zero labeled
    /// keypoints (`v == 0` for all `k`).
    #[serde(default)]
    pub bbox: Option<Xyxy>,
    /// Mark this ground truth as ignored (e.g. crowd or out-of-area instance): matches to it are
    /// neither true positives nor false positives, and regular truths always take priority.
    #[serde(default)]
    pub ignore: bool,
}

impl OksTruth {
    /// Construct a regular ground-truth pose instance with no bounding box and `ignore = false`.
    pub fn new(keypoints: Vec<Keypoint>, area: f32) -> Self {
        Self { keypoints, area, bbox: None, ignore: false }
    }

    /// Attach a bounding box (enabling COCO bounding-box ignore OKS when all keypoints are unlabeled).
    pub fn with_bbox(mut self, bbox: Xyxy) -> Self {
        self.bbox = Some(bbox);
        self
    }

    /// Mark this instance as ignored during evaluation.
    pub fn with_ignore(mut self, ignore: bool) -> Self {
        self.ignore = ignore;
        self
    }
}

/// COCO-style keypoint Average Precision summary across OKS thresholds `0.50:0.05:0.95`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OksApReport {
    /// AP at OKS threshold `0.50`.
    pub ap50: f32,
    /// AP at OKS threshold `0.75`.
    pub ap75: f32,
    /// Mean AP over OKS thresholds `0.50:0.05:0.95`.
    pub ap50_95: f32,
    /// AP at each of the 10 OKS thresholds (`0.50, 0.55, ..., 0.95`).
    pub ap_per_threshold: Vec<f32>,
    /// OKS thresholds (`0.50:0.05:0.95`).
    pub thresholds: Vec<f32>,
    /// Total labeled ground-truth instances evaluated (those with at least one `v > 0` keypoint).
    pub num_truths: u64,
    /// Total finite predictions evaluated.
    pub num_preds: u64,
}

/// Compute COCO-style keypoint AP (`AP@50`, `AP@75`, `AP@50:95`) across a slice of `(preds, truths)`
/// images using greedy per-image OKS matching and 101-point COCO recall interpolation.
///
/// Regular labeled truths (`!ignore && any(v > 0)`) are matched first; ignored truths (`ignore == true`
/// or zero labeled keypoints) are sorted afterward so regular matches always take priority, and
/// predictions matching an ignored truth are excluded from both TP and FP counts (matching
/// `pycocotools/cocoeval.py`). For a zero-labeled truth with `bbox: Some(b)`, OKS is computed
/// against the COCO expanded bounding box; zero-labeled truths without a `bbox` cannot be spatially
/// matched and are skipped (predictions overlapping such unboxed people are not ignored).
/// Predictions with non-finite scores are skipped.
pub fn compute_oks_ap<P, T>(
    images: &[(P, T)],
    sigmas: &[f32],
    max_dets_per_image: Option<usize>,
) -> Result<OksApReport>
where
    P: AsRef<[OksPred]>,
    T: AsRef<[OksTruth]>,
{
    validate_sigmas(sigmas, "compute_oks_ap")?;
    let thresholds = iou_thresholds();
    let recall_pts = recall_points();
    let k_len = sigmas.len();

    // `(score, img_idx, pred_idx, tp_bits, ig_bits)`
    let mut records: Vec<(f32, usize, usize, u16, u16)> = Vec::new();
    let mut num_truths = 0u64;

    for (img_idx, (preds_ref, truths_ref)) in images.iter().enumerate() {
        let preds = preds_ref.as_ref();
        let truths = truths_ref.as_ref();

        for (i, p) in preds.iter().enumerate() {
            if p.keypoints.len() != k_len {
                return Err(Error::Config(format!(
                    "compute_oks_ap: image {img_idx} pred[{i}] has {} keypoints, expected {k_len}",
                    p.keypoints.len()
                )));
            }
        }
        // `(truth_idx, has_labeled_kpts)` partitioned into regular truths first, ignored truths second.
        let mut reg_gts: Vec<(usize, bool)> = Vec::new();
        let mut ign_gts: Vec<(usize, bool)> = Vec::new();
        for (i, t) in truths.iter().enumerate() {
            if t.keypoints.len() != k_len {
                return Err(Error::Config(format!(
                    "compute_oks_ap: image {img_idx} truth[{i}] has {} keypoints, expected {k_len}",
                    t.keypoints.len()
                )));
            }
            if !(t.area.is_finite() && t.area >= 0.0) {
                return Err(Error::Config(format!(
                    "compute_oks_ap: image {img_idx} truth[{i}] area must be finite and >= 0, got {}",
                    t.area
                )));
            }
            for (ki, kp) in t.keypoints.iter().enumerate() {
                if kp.visibility > 2 {
                    return Err(Error::Config(format!(
                        "compute_oks_ap: image {img_idx} truth[{i}].keypoints[{ki}] visibility {} > 2",
                        kp.visibility
                    )));
                }
            }
            let has_labeled = t.keypoints.iter().any(|kp| kp.visibility > 0);
            if !t.ignore && has_labeled {
                reg_gts.push((i, true));
            } else if has_labeled || t.bbox.is_some() {
                ign_gts.push((i, has_labeled));
            }
        }
        let n_reg = reg_gts.len();
        num_truths += n_reg as u64;
        let mut ordered_gts = reg_gts;
        ordered_gts.extend(ign_gts);

        let mut pred_order: Vec<usize> = (0..preds.len()).filter(|&i| preds[i].score.is_finite()).collect();
        pred_order
            .sort_unstable_by(|&a, &b| preds[b].score.total_cmp(&preds[a].score).then_with(|| a.cmp(&b)));
        if let Some(cap) = max_dets_per_image {
            pred_order.truncate(cap);
        }

        let np = pred_order.len();
        let ng = ordered_gts.len();
        if np == 0 {
            continue;
        }
        if ng == 0 {
            for &pi in &pred_order {
                records.push((preds[pi].score, img_idx, pi, 0, 0));
            }
            continue;
        }

        let sim_len = np
            .checked_mul(ng)
            .filter(|&n| n.checked_mul(std::mem::size_of::<f64>()).is_some_and(|b| b <= isize::MAX as usize))
            .ok_or_else(|| {
                Error::Config(format!(
                    "compute_oks_ap: image {img_idx} similarity matrix {np} x {ng} overflows"
                ))
            })?;
        let mut sim = vec![0.0f64; sim_len];
        for (di, &pi) in pred_order.iter().enumerate() {
            for (gi, &(ti, has_labeled)) in ordered_gts.iter().enumerate() {
                let t = &truths[ti];
                sim[di * ng + gi] = if has_labeled {
                    oks_f64(&preds[pi].keypoints, &t.keypoints, t.area as f64, sigmas)?
                } else if let Some(ref bb) = t.bbox {
                    oks_unlabeled_bbox_f64(&preds[pi].keypoints, bb, t.area as f64, sigmas)
                } else {
                    0.0
                };
            }
        }

        let mut tp_bits = vec![0u16; np];
        let mut ig_bits = vec![0u16; np];
        for (t_idx, &thr) in thresholds.iter().enumerate() {
            let thr_cap = thr.min(1.0 - 1e-10);
            let mut gt_used = vec![false; ng];
            for di in 0..np {
                let row = &sim[di * ng..(di + 1) * ng];
                let mut best = thr_cap;
                let mut best_g = usize::MAX;
                for (gi, &v) in row.iter().enumerate() {
                    if gt_used[gi] {
                        continue;
                    }
                    // Prefer regular labeled truths over ignored truths (matching pycocotools).
                    if best_g < n_reg && gi >= n_reg {
                        break;
                    }
                    if v < best {
                        continue;
                    }
                    best = v;
                    best_g = gi;
                }
                if best_g != usize::MAX {
                    gt_used[best_g] = true;
                    if best_g < n_reg {
                        tp_bits[di] |= 1 << t_idx;
                    } else {
                        ig_bits[di] |= 1 << t_idx;
                    }
                }
            }
        }

        for (di, &pi) in pred_order.iter().enumerate() {
            records.push((preds[pi].score, img_idx, pi, tp_bits[di], ig_bits[di]));
        }
    }

    let num_preds = records.len() as u64;
    if num_truths == 0 || records.is_empty() {
        return Ok(OksApReport {
            ap50: 0.0,
            ap75: 0.0,
            ap50_95: 0.0,
            ap_per_threshold: vec![0.0; NUM_IOU_THRESHOLDS],
            thresholds: thresholds.iter().map(|&t| t as f32).collect(),
            num_truths,
            num_preds,
        });
    }

    records.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
    let n_gt_f = num_truths as f64;
    let mut tp_c = [0u64; NUM_IOU_THRESHOLDS];
    let mut fp_c = [0u64; NUM_IOU_THRESHOLDS];
    let mut cur_b = [0usize; NUM_IOU_THRESHOLDS];
    let mut bucket_max = [[0.0f64; NUM_RECALL_POINTS]; NUM_IOU_THRESHOLDS];

    for &(_, _, _, bits, ig) in &records {
        for t in 0..NUM_IOU_THRESHOLDS {
            if (bits >> t) & 1 == 1 {
                tp_c[t] += 1;
                let tp_f = tp_c[t] as f64;
                let rc = tp_f / n_gt_f;
                let mut b = cur_b[t];
                while b + 1 < NUM_RECALL_POINTS && rc >= recall_pts[b + 1] {
                    b += 1;
                }
                cur_b[t] = b;
                let p = tp_f / (tp_f + fp_c[t] as f64 + f64::EPSILON);
                if p > bucket_max[t][b] {
                    bucket_max[t][b] = p;
                }
            } else if (ig >> t) & 1 == 0 {
                fp_c[t] += 1;
            }
        }
    }

    let mut ap = [0.0f64; NUM_IOU_THRESHOLDS];
    for (t, ap_t) in ap.iter_mut().enumerate() {
        let bm = &mut bucket_max[t];
        let mut running = 0.0f64;
        for v in bm.iter_mut().rev() {
            if *v > running {
                running = *v;
            } else {
                *v = running;
            }
        }
        *ap_t = bm.iter().sum::<f64>() / NUM_RECALL_POINTS as f64;
    }

    let ap50_95 = ap.iter().sum::<f64>() / NUM_IOU_THRESHOLDS as f64;
    Ok(OksApReport {
        ap50: ap[0] as f32,
        ap75: ap[5] as f32,
        ap50_95: ap50_95 as f32,
        ap_per_threshold: ap.iter().map(|&v| v as f32).collect(),
        thresholds: thresholds.iter().map(|&t| t as f32).collect(),
        num_truths,
        num_preds,
    })
}
