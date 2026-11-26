//! Exact Hungarian / Linear-Sum Assignment solver (`O(min(N, M)² · max(N, M))`) and
//! [`OneToOneMatcher`] for NMS-free detection heads (DETR / RT-DETR / one-to-one YOLO heads).
//!
//! - [`hungarian_assign`] (and [`hungarian_assign_f32`]): solves the exact minimum-cost linear-sum
//!   assignment problem on any rectangular `rows × cols` finite cost matrix via potential-based shortest
//!   augmenting paths (Jonker-Volgenant / Kuhn-Munkres), handling both `rows <= cols` and `rows > cols`,
//!   negative costs, extreme-magnitude costs (via automatic positive rescaling), and deterministic
//!   tie-breaking.
//! - [`OneToOneMatcher`]: builds a pairwise `[Q × G]` matching cost matrix combining classification
//!   (Focal, BCE, or negative probability), box L1 (in `cxcywh` space, optionally normalized by image size),
//!   and box `1 - IoU/GIoU/DIoU/CIoU` costs, then solves globally optimal one-to-one assignment via
//!   [`hungarian_assign`].

use serde::{Deserialize, Serialize};

use crate::assign::Assignment;
use crate::boxes::{IouKind, Xyxy, iou};
use crate::error::{Error, Result};

/// Result of exact linear-sum assignment on a `rows × cols` cost matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HungarianResult {
    /// Matched `(row, col)` pairs sorted by ascending `row` (length `rows.min(cols)`).
    pub pairs: Vec<(usize, usize)>,
    /// Sum of `costs[row * cols + col]` over all matched pairs in `f64`.
    pub total_cost: f64,
}

/// Error-free double-double (`hi + lo`, ~106-bit significand via Knuth's TwoSum) used for dual
/// potentials `u`, `v`, `minv`, and `delta` in [`hungarian_assign`] so mixed-magnitude cost rows
/// spanning up to ~`10^31` never suffer catastrophic cancellation in dual updates.
#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    const INFINITY: Self = Self { hi: f64::INFINITY, lo: 0.0 };

    #[inline]
    const fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (s, e) = Self::two_sum(self.hi, rhs.hi);
        let lo = e + self.lo + rhs.lo;
        let (hi, lo) = Self::two_sum(s, lo);
        Self { hi, lo }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.add(Self { hi: -rhs.hi, lo: -rhs.lo })
    }

    #[inline]
    fn lt(self, rhs: Self) -> bool {
        self.hi < rhs.hi || (self.hi == rhs.hi && self.lo < rhs.lo)
    }
}

/// Solve the exact minimum-cost linear-sum assignment problem for a row-major `rows × cols` cost matrix.
///
/// Matches `k = rows.min(cols)` distinct `(row, col)` pairs minimizing `sum costs[row * cols + col]`
/// in `O(k² · max(rows, cols))` time and `O(rows + cols)` auxiliary memory.
///
/// Ties are broken deterministically (preferring smaller indices under strict `<` comparisons).
///
/// **Numerical precision:** dual potentials and slacks are maintained in double-double (`~106`-bit)
/// arithmetic (preserving ~31 decimal digits of dynamic range across augmenting paths). Costs whose
/// magnitude exceeds `1e150` are divided by a positive normal divisor (`max_abs / 1e100`) to prevent
/// overflow without subnormal underflow, and `total_cost` is always summed from the original unscaled
/// entries.
///
/// # Errors
/// Returns [`Error::Config`] if `rows * cols` overflows, `costs.len() != rows * cols`, or any entry
/// in `costs` is `NaN` or `±Inf`.
pub fn hungarian_assign(costs: &[f64], rows: usize, cols: usize) -> Result<HungarianResult> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::Config(format!("hungarian_assign: rows ({rows}) * cols ({cols}) overflows")))?;
    if costs.len() != expected {
        return Err(Error::Config(format!(
            "hungarian_assign: costs length {} != rows * cols ({rows} * {cols})",
            costs.len()
        )));
    }
    let mut max_abs = 0.0f64;
    for (idx, &c) in costs.iter().enumerate() {
        if !c.is_finite() {
            let r = idx / cols.max(1);
            let col = idx % cols.max(1);
            return Err(Error::Config(format!(
                "hungarian_assign: costs[{r}, {col}] must be finite, got {c}"
            )));
        }
        let a = c.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if rows == 0 || cols == 0 {
        return Ok(HungarianResult { pairs: Vec::new(), total_cost: 0.0 });
    }

    // Always orient so n <= m for O(n² m) complexity and guaranteed unmatched column in every iteration.
    let transposed = rows > cols;
    let (n, m) = if transposed { (cols, rows) } else { (rows, cols) };

    // Positive normal divisor when magnitudes exceed 1e150 prevents `cost - u[i] - v[j]` overflow in `f64`
    // without subnormal underflow (even at `f64::MAX`) and without row-shift absorption of small entries.
    let inv_div = if max_abs > 1e150 { Some(max_abs / 1e100) } else { None };

    let mut u = vec![Dd::ZERO; n + 1];
    let mut v = vec![Dd::ZERO; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    let mut minv = vec![Dd::ZERO; m + 1];
    let mut used = vec![false; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        minv.fill(Dd::INFINITY);
        used.fill(false);

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let r = i0 - 1;
            let u_i0 = u[i0];
            let mut delta = Dd::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=m {
                if !used[j] {
                    let c = j - 1;
                    let raw = if transposed { costs[c * cols + r] } else { costs[r * cols + c] };
                    let scaled = match inv_div {
                        Some(d) => raw / d,
                        None => raw,
                    };
                    let cur = Dd::from_f64(scaled).sub(u_i0).sub(v[j]);
                    if cur.lt(minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j].lt(delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=m {
                if used[j] {
                    u[p[j]] = u[p[j]].add(delta);
                    v[j] = v[j].sub(delta);
                } else {
                    minv[j] = minv[j].sub(delta);
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs = Vec::with_capacity(n);
    for (j, &matched_i) in p.iter().enumerate().skip(1) {
        if matched_i != 0 {
            let (orig_r, orig_c) = if transposed { (j - 1, matched_i - 1) } else { (matched_i - 1, j - 1) };
            pairs.push((orig_r, orig_c));
        }
    }
    pairs.sort_unstable_by_key(|&(r, _)| r);

    // Sum original unscaled costs directly so total_cost never accumulates dual-potential rounding drift.
    let total_cost: f64 = pairs.iter().map(|&(r, c)| costs[r * cols + c]).sum();
    Ok(HungarianResult { pairs, total_cost })
}

/// Convenience wrapper around [`hungarian_assign`] for `f32` row-major cost matrices.
///
/// # Errors
/// Returns [`Error::Config`] on shape mismatch, overflow, or any non-finite (`NaN` / `±Inf`) entry.
pub fn hungarian_assign_f32(costs: &[f32], rows: usize, cols: usize) -> Result<HungarianResult> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        Error::Config(format!("hungarian_assign_f32: rows ({rows}) * cols ({cols}) overflows"))
    })?;
    if costs.len() != expected {
        return Err(Error::Config(format!(
            "hungarian_assign_f32: costs length {} != rows * cols ({rows} * {cols})",
            costs.len()
        )));
    }
    let mut costs_f64 = Vec::with_capacity(costs.len());
    for (idx, &c) in costs.iter().enumerate() {
        if !c.is_finite() {
            let r = idx / cols.max(1);
            let col = idx % cols.max(1);
            return Err(Error::Config(format!(
                "hungarian_assign_f32: costs[{r}, {col}] must be finite, got {c}"
            )));
        }
        costs_f64.push(c as f64);
    }
    hungarian_assign(&costs_f64, rows, cols)
}

/// Classification cost formulation used by [`OneToOneMatcher`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClsCostKind {
    /// Binary Focal matching cost difference `pos_cost - neg_cost` (Deformable-DETR / RT-DETR / DINO):
    /// `alpha * (1 - p)^gamma * softplus(-x) - (1 - alpha) * p^gamma * softplus(x)`.
    Focal {
        /// Positive balancing factor in `(0, 1)` (default `0.25`).
        alpha: f32,
        /// Focusing exponent `>= 0` (default `2.0`).
        gamma: f32,
    },
    /// Exact BCE switching cost `BCE(x, 1) - BCE(x, 0) = softplus(-x) - softplus(x) = -x`.
    Bce,
    /// Negative sigmoid probability `-sigmoid(x)` (classic DETR).
    Prob,
}

impl Default for ClsCostKind {
    fn default() -> Self {
        Self::Focal { alpha: 0.25, gamma: 2.0 }
    }
}

/// Hyper-parameters for [`OneToOneMatcher`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OneToOneConfig {
    /// Weight of the classification cost term (default `2.0`).
    pub cls_weight: f32,
    /// Weight of the `cxcywh` L1 box regression cost term (default `5.0`).
    pub l1_weight: f32,
    /// Weight of the `1 - iou_kind` box overlap cost term (default `2.0`).
    pub iou_weight: f32,
    /// Overlap metric used for the IoU cost term (default [`IouKind::Giou`]).
    pub iou_kind: IouKind,
    /// Classification cost formulation (default [`ClsCostKind::Focal`] with `alpha = 0.25, gamma = 2.0`).
    pub cls_cost: ClsCostKind,
    /// Optional `(width, height)` divisor to normalize `cxcywh` L1 box coordinates into `[0, 1]`
    /// (e.g. `(img_w, img_h)` when boxes are in input pixels; `None` uses box coordinates as-is).
    pub image_size: Option<(f32, f32)>,
}

impl Default for OneToOneConfig {
    fn default() -> Self {
        Self {
            cls_weight: 2.0,
            l1_weight: 5.0,
            iou_weight: 2.0,
            iou_kind: IouKind::Giou,
            cls_cost: ClsCostKind::default(),
            image_size: None,
        }
    }
}

impl OneToOneConfig {
    /// Validate all matcher hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.cls_weight.is_finite() && self.cls_weight >= 0.0) {
            return Err(Error::Config(format!(
                "one_to_one.cls_weight must be finite and >= 0, got {}",
                self.cls_weight
            )));
        }
        if !(self.l1_weight.is_finite() && self.l1_weight >= 0.0) {
            return Err(Error::Config(format!(
                "one_to_one.l1_weight must be finite and >= 0, got {}",
                self.l1_weight
            )));
        }
        if !(self.iou_weight.is_finite() && self.iou_weight >= 0.0) {
            return Err(Error::Config(format!(
                "one_to_one.iou_weight must be finite and >= 0, got {}",
                self.iou_weight
            )));
        }
        if let ClsCostKind::Focal { alpha, gamma } = self.cls_cost {
            if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
                return Err(Error::Config(format!(
                    "one_to_one.cls_cost.alpha must be finite and in (0, 1), got {alpha}"
                )));
            }
            if !(gamma.is_finite() && gamma >= 0.0) {
                return Err(Error::Config(format!(
                    "one_to_one.cls_cost.gamma must be finite and >= 0, got {gamma}"
                )));
            }
        }
        if let Some((w, h)) = self.image_size
            && !(w.is_finite() && w > 0.0 && h.is_finite() && h > 0.0)
        {
            return Err(Error::Config(format!(
                "one_to_one.image_size must be finite and > 0, got ({w}, {h})"
            )));
        }
        Ok(())
    }
}

/// Per-query assignment produced by [`OneToOneMatcher::match_image`].
///
/// Entries where `fg_mask[q]` is `false` are background (`target_gt[q] = 0`,
/// `target_labels[q] = num_classes`, `target_bboxes[q] = Xyxy::default()`, `target_ious[q] = 0.0`,
/// `matched_costs[q] = 0.0`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OneToOneAssignment {
    /// Matched `(query_idx, gt_idx)` pairs sorted by ascending `query_idx`.
    pub pairs: Vec<(usize, usize)>,
    /// Whether each query is matched foreground (`[Q]`).
    pub fg_mask: Vec<bool>,
    /// Assigned GT index per query (`[Q]`).
    pub target_gt: Vec<usize>,
    /// Assigned class id per query (`[Q]`, `num_classes` for background).
    pub target_labels: Vec<u32>,
    /// Assigned GT box per query (`[Q]`).
    pub target_bboxes: Vec<Xyxy>,
    /// Plain IoU between each foreground query's predicted box and its assigned GT (`[Q]`, in `[0, 1]`).
    pub target_ious: Vec<f32>,
    /// Pairwise matching cost for each foreground query (`[Q]`).
    pub matched_costs: Vec<f64>,
    /// Sum of pairwise matching costs across all matched pairs.
    pub total_cost: f64,
    /// Number of matched foreground queries (`pairs.len()`).
    pub num_fg: usize,
    /// `true` when all prediction and GT inputs were finite; `false` returns an empty assignment cleanly.
    pub inputs_finite: bool,
}

impl OneToOneAssignment {
    fn empty(num_queries: usize, num_classes: usize, inputs_finite: bool) -> Self {
        Self {
            pairs: Vec::new(),
            fg_mask: vec![false; num_queries],
            target_gt: vec![0; num_queries],
            target_labels: vec![num_classes as u32; num_queries],
            target_bboxes: vec![Xyxy::default(); num_queries],
            target_ious: vec![0.0; num_queries],
            matched_costs: vec![0.0; num_queries],
            total_cost: 0.0,
            num_fg: 0,
            inputs_finite,
        }
    }

    /// Convert to [`Assignment`], using [`Self::target_ious`] as `target_scores`.
    pub fn to_assignment(&self) -> Assignment {
        Assignment {
            fg_mask: self.fg_mask.clone(),
            target_gt: self.target_gt.clone(),
            target_labels: self.target_labels.clone(),
            target_bboxes: self.target_bboxes.clone(),
            target_scores: self.target_ious.clone(),
            num_fg: self.num_fg,
        }
    }

    /// Build dense `[Q, num_classes]` row-major continuous quality targets (`target_ious[q]` at
    /// `(q, target_labels[q])`, `0.0` elsewhere).
    pub fn dense_quality_targets(&self, num_classes: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.fg_mask.len() * num_classes];
        for (q, &fg) in self.fg_mask.iter().enumerate() {
            if fg {
                out[q * num_classes + self.target_labels[q] as usize] = self.target_ious[q];
            }
        }
        out
    }
}

/// Globally optimal one-to-one Hungarian matcher for NMS-free detection heads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OneToOneMatcher {
    cfg: OneToOneConfig,
}

impl OneToOneMatcher {
    /// Create a validated [`OneToOneMatcher`].
    pub fn new(cfg: OneToOneConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Return the matcher's configuration.
    pub fn config(&self) -> &OneToOneConfig {
        &self.cfg
    }

    /// Match `Q` predicted queries against `G` ground truths for one image via exact [`hungarian_assign`].
    ///
    /// - `pd_cls_logits`: `[Q * num_classes]` row-major raw classification logits.
    /// - `num_classes`: number of foreground classes `C`.
    /// - `pd_bboxes`: `[Q]` predicted boxes (`Xyxy`).
    /// - `gt_bboxes`: `[G]` ground-truth boxes (`Xyxy`). Degenerate GTs (`w <= 0` or `h <= 0`) are
    ///   skipped cleanly without disturbing valid GTs.
    /// - `gt_labels`: `[G]` class ids (`< num_classes`).
    ///
    /// # Errors
    /// Returns [`Error::Config`] on slice length mismatches, overflow, `num_classes == 0` with non-empty
    /// GTs, or out-of-range `gt_labels`.
    pub fn match_image(
        &self,
        pd_cls_logits: &[f32],
        num_classes: usize,
        pd_bboxes: &[Xyxy],
        gt_bboxes: &[Xyxy],
        gt_labels: &[u32],
    ) -> Result<OneToOneAssignment> {
        let n_q = pd_bboxes.len();
        let n_g = gt_bboxes.len();
        let expected_cls = n_q.checked_mul(num_classes).ok_or_else(|| {
            Error::Config(format!("one_to_one: Q ({n_q}) * num_classes ({num_classes}) overflows"))
        })?;
        if pd_cls_logits.len() != expected_cls {
            return Err(Error::Config(format!(
                "one_to_one: pd_cls_logits length {} != Q * num_classes ({n_q} * {num_classes})",
                pd_cls_logits.len()
            )));
        }
        if gt_labels.len() != n_g {
            return Err(Error::Config(format!(
                "one_to_one: gt_labels length {} != G ({n_g})",
                gt_labels.len()
            )));
        }
        if n_g > 0 && num_classes == 0 {
            return Err(Error::Config(
                "one_to_one: num_classes must be >= 1 when gt_bboxes is non-empty".into(),
            ));
        }
        for (g, &l) in gt_labels.iter().enumerate() {
            if (l as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "one_to_one: gt_labels[{g}] = {l} >= num_classes ({num_classes})"
                )));
            }
        }

        let all_finite = pd_cls_logits.iter().all(|v| v.is_finite())
            && pd_bboxes.iter().all(box_is_finite)
            && gt_bboxes.iter().all(box_is_finite);
        if !all_finite {
            return Ok(OneToOneAssignment::empty(n_q, num_classes, false));
        }
        if n_q == 0 || n_g == 0 {
            return Ok(OneToOneAssignment::empty(n_q, num_classes, true));
        }

        // Filter out degenerate GT boxes (width <= 0 or height <= 0) while preserving original GT indices.
        let valid_gts: Vec<usize> = (0..n_g)
            .filter(|&g| {
                let b = &gt_bboxes[g];
                (b.x2 as f64 - b.x1 as f64) > 0.0 && (b.y2 as f64 - b.y1 as f64) > 0.0
            })
            .collect();
        let n_valid = valid_gts.len();
        if n_valid == 0 {
            return Ok(OneToOneAssignment::empty(n_q, num_classes, true));
        }

        let (inv_w, inv_h) = match self.cfg.image_size {
            Some((w, h)) => (1.0 / w as f64, 1.0 / h as f64),
            None => (1.0, 1.0),
        };
        let cls_w = self.cfg.cls_weight as f64;
        let l1_w = self.cfg.l1_weight as f64;
        let iou_w = self.cfg.iou_weight as f64;

        // Precompute cxcywh in f64 for queries and valid GTs to avoid redundant conversions in the Q x G loop.
        let pd_cxcywh: Vec<[f64; 4]> = pd_bboxes
            .iter()
            .map(|b| {
                let x1 = b.x1 as f64;
                let y1 = b.y1 as f64;
                let x2 = b.x2 as f64;
                let y2 = b.y2 as f64;
                [0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1]
            })
            .collect();
        let gt_cxcywh: Vec<[f64; 4]> = valid_gts
            .iter()
            .map(|&g| {
                let b = &gt_bboxes[g];
                let x1 = b.x1 as f64;
                let y1 = b.y1 as f64;
                let x2 = b.x2 as f64;
                let y2 = b.y2 as f64;
                [0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1]
            })
            .collect();

        let matrix_len = n_q.checked_mul(n_valid).ok_or_else(|| {
            Error::Config(format!("one_to_one: Q ({n_q}) * valid_gts ({n_valid}) overflows"))
        })?;
        let max_elems = (isize::MAX as usize) / std::mem::size_of::<f64>();
        if matrix_len > max_elems {
            return Err(Error::Config(format!(
                "one_to_one: cost matrix of size {n_q} x {n_valid} exceeds allocation limit"
            )));
        }
        let mut cost_matrix = vec![0.0f64; matrix_len];
        for q in 0..n_q {
            let p_box = &pd_bboxes[q];
            let p_c = pd_cxcywh[q];
            let logit_row = &pd_cls_logits[q * num_classes..(q + 1) * num_classes];
            for (vg_idx, &g) in valid_gts.iter().enumerate() {
                let c_cls = if cls_w == 0.0 {
                    0.0
                } else {
                    let logit = logit_row[gt_labels[g] as usize] as f64;
                    cls_cost_f64(logit, self.cfg.cls_cost)
                };
                let c_l1 = if l1_w == 0.0 {
                    0.0
                } else {
                    let g_c = gt_cxcywh[vg_idx];
                    (p_c[0] - g_c[0]).abs() * inv_w
                        + (p_c[1] - g_c[1]).abs() * inv_h
                        + (p_c[2] - g_c[2]).abs() * inv_w
                        + (p_c[3] - g_c[3]).abs() * inv_h
                };
                let c_iou = if iou_w == 0.0 {
                    0.0
                } else {
                    1.0 - iou(p_box, &gt_bboxes[g], self.cfg.iou_kind) as f64
                };
                cost_matrix[q * n_valid + vg_idx] = cls_w * c_cls + l1_w * c_l1 + iou_w * c_iou;
            }
        }

        let solved = hungarian_assign(&cost_matrix, n_q, n_valid)?;
        let mut out = OneToOneAssignment::empty(n_q, num_classes, true);
        out.pairs.reserve(solved.pairs.len());
        for &(q, vg_idx) in &solved.pairs {
            let g = valid_gts[vg_idx];
            let pair_cost = cost_matrix[q * n_valid + vg_idx];
            out.pairs.push((q, g));
            out.fg_mask[q] = true;
            out.target_gt[q] = g;
            out.target_labels[q] = gt_labels[g];
            out.target_bboxes[q] = gt_bboxes[g];
            out.target_ious[q] = iou(&pd_bboxes[q], &gt_bboxes[g], IouKind::Iou);
            out.matched_costs[q] = pair_cost;
        }
        out.num_fg = out.pairs.len();
        out.total_cost = solved.total_cost;
        Ok(out)
    }
}

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Numerically stable `softplus(x) = max(x, 0) + ln(1 + exp(-|x|))` in `f64`.
#[inline]
fn softplus_f64(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Cancellation-free `(sigmoid(x), sigmoid(-x))` in `f64`.
#[inline]
fn sigmoid_pair_f64(x: f64) -> (f64, f64) {
    if x >= 0.0 {
        let e = (-x).exp();
        let inv = 1.0 / (1.0 + e);
        (inv, e * inv)
    } else {
        let e = x.exp();
        let inv = 1.0 / (1.0 + e);
        (e * inv, inv)
    }
}

#[inline]
fn cls_cost_f64(logit: f64, kind: ClsCostKind) -> f64 {
    match kind {
        ClsCostKind::Focal { alpha, gamma } => {
            let (p, om_p) = sigmoid_pair_f64(logit);
            let a = alpha as f64;
            let g = gamma as f64;
            let pos = a * om_p.powf(g) * softplus_f64(-logit);
            let neg = (1.0 - a) * p.powf(g) * softplus_f64(logit);
            pos - neg
        }
        ClsCostKind::Bce => -logit,
        ClsCostKind::Prob => -sigmoid_pair_f64(logit).0,
    }
}
