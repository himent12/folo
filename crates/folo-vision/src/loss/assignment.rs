//! The detached part of the loss on the host (§7.1, §7.3): candidate-only DFL decode + sigmoid and grid-indexed
//! task-aligned assignment per image, giving the batch's foreground anchors ([`assign_batch`]).

use folo_core::cpu::fast_exp;
use folo_core::{Result, Tensor};
use rayon::prelude::*;

use super::{BatchAssignment, ImageTargets, LossConfig, check_inputs, host_f32, invalid};
use crate::assign::{
    GridLevel, LevelRect, Pick, PowOp, TalConfig, candidate_box, candidate_metric, keep_topk,
    level_candidate_rect, nwd_lambda, resolve_picks, validate_grid,
};
use crate::boxes::Xyxy;
use crate::decode::{Anchors, dist2bbox, sigmoid};

/// Vectorisation-friendly finiteness check over a slice (exact IEEE-754 semantics).
///
/// For any finite `f32` `v`, `v * 0.0` is `±0.0` (never overflows); for `±Inf` or `NaN`, `v * 0.0` is `NaN`
/// which poisons its accumulator lane without branching inside the inner loop.
#[inline]
fn all_finite(s: &[f32]) -> bool {
    let mut acc = [0.0f32; 8];
    let (chunks, rest) = s.as_chunks::<8>();
    for ch in chunks {
        for (a, &v) in acc.iter_mut().zip(ch) {
            *a += v * 0.0;
        }
    }
    acc.iter().all(|&v| v == 0.0) && rest.iter().all(|v| v.is_finite())
}

/// DFL decode of one image's channel-major box logits `[4 · reg_max, A]` (§6.4) into `out` (covering anchor
/// indices `base..base + out.len()`), decoding **only** anchors where `mask[a - base] != 0`.
///
/// Iterates side-outer, bin-outer over maximal contiguous active runs so each channel-major bin row is streamed
/// linearly once (cache- and prefetcher-friendly) while computing `fast_exp` only on active candidates, giving
/// bit-identical boxes to a full-grid decode.
fn decode_masked_channel_major(
    reg: &[f32],
    anchors: &Anchors,
    reg_max: usize,
    mask: &[u8],
    base: usize,
    out: &mut [Xyxy],
) {
    let n_a = anchors.len();
    let len = mask.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut n_cand = 0usize;
    let mut pos = 0usize;
    while pos < len {
        while pos < len && mask[pos] == 0 {
            pos += 1;
        }
        if pos >= len {
            break;
        }
        let start = pos;
        while pos < len && mask[pos] != 0 {
            pos += 1;
        }
        runs.push((start, pos));
        n_cand += pos - start;
    }
    if n_cand == 0 {
        return;
    }

    let mut m = vec![0f32; n_cand];
    let mut z = vec![0f32; n_cand];
    let mut e = vec![0f32; n_cand];
    let mut ltrb = vec![[0f32; 4]; n_cand];

    for side in 0..4 {
        let side_base = side * reg_max * n_a;
        m.fill(f32::NEG_INFINITY);
        for bin in 0..reg_max {
            let row = &reg[side_base + bin * n_a..side_base + (bin + 1) * n_a];
            let mut off = 0usize;
            for &(lo, hi) in &runs {
                let k = hi - lo;
                for (mv, &x) in m[off..off + k].iter_mut().zip(&row[base + lo..base + hi]) {
                    *mv = mv.max(x);
                }
                off += k;
            }
        }
        z.fill(0.0);
        e.fill(0.0);
        for bin in 0..reg_max {
            let fi = bin as f32;
            let row = &reg[side_base + bin * n_a..side_base + (bin + 1) * n_a];
            let mut off = 0usize;
            for &(lo, hi) in &runs {
                let k = hi - lo;
                for (((zv, ev), &mv), &x) in z[off..off + k]
                    .iter_mut()
                    .zip(&mut e[off..off + k])
                    .zip(&m[off..off + k])
                    .zip(&row[base + lo..base + hi])
                {
                    let p = fast_exp(x - mv);
                    *zv += p;
                    *ev += p * fi;
                }
                off += k;
            }
        }
        for ((d, &zv), &ev) in ltrb.iter_mut().zip(&z).zip(&e) {
            d[side] = ev / zv;
        }
    }

    let mut off = 0usize;
    for &(lo, hi) in &runs {
        for (i, slot) in out[lo..hi].iter_mut().enumerate() {
            let a = base + lo + i;
            let s = anchors.strides[a];
            *slot = dist2bbox(ltrb[off], anchors.points[a]).scale(s, s);
            off += 1;
        }
    }
}

/// One image's foreground entries.
#[derive(Default)]
struct ImageFg {
    anchor: Vec<u32>,
    label: Vec<u32>,
    score: Vec<f32>,
    target: Vec<Xyxy>,
}

/// Assign one image using O(1)-per-level grid rectangles and candidate-only lazy DFL decode + sigmoid.
fn assign_image_grid(
    reg: &[f32],
    logits: &[f32],
    anchors: &Anchors,
    grid: &[GridLevel],
    reg_max: usize,
    gt: &ImageTargets,
    tal: &TalConfig,
) -> ImageFg {
    let n_a = anchors.len();
    let n_g = gt.boxes.len();
    let n_l = grid.len();

    // 1. O(1) candidate rectangles per (gt, level) and their union mask.
    let mut rects: Vec<LevelRect> = Vec::with_capacity(n_g * n_l);
    let mut cand_mask = vec![0u8; n_a];
    let mut min_a = n_a;
    let mut max_a = 0usize;
    for b in &gt.boxes {
        let cb = candidate_box(b, tal.min_box_size);
        for lvl in grid {
            let rect = level_candidate_rect(&cb, tal.eps, lvl);
            if !rect.is_empty() {
                let first = rect.start + rect.y0 * rect.width + rect.x0;
                let last = rect.start + (rect.y1 - 1) * rect.width + rect.x1;
                min_a = min_a.min(first);
                max_a = max_a.max(last);
                for y in rect.y0..rect.y1 {
                    let row = rect.start + y * rect.width;
                    cand_mask[row + rect.x0..row + rect.x1].fill(1);
                }
            }
            rects.push(rect);
        }
    }
    if min_a >= max_a {
        return ImageFg::default();
    }

    // 2. DFL-decode ONLY anchors in the union candidate mask.
    let mut pd_px = vec![Xyxy::default(); max_a - min_a];
    decode_masked_channel_major(reg, anchors, reg_max, &cand_mask[min_a..max_a], min_a, &mut pd_px);

    // 3. Score only each gt's rectangle candidates, evaluating `sigmoid` lazily on the gt's class row.
    let alpha_op = PowOp::from_exp(tal.alpha);
    let beta_op = PowOp::from_exp(tal.beta);
    let mut selected: Vec<Vec<Pick>> = Vec::with_capacity(n_g);
    for g in 0..n_g {
        let gt_box = &gt.boxes[g];
        let cls_row = &logits[gt.labels[g] as usize * n_a..(gt.labels[g] as usize + 1) * n_a];
        let lam = nwd_lambda(gt_box, tal.nwd_weight);
        let gt_rects = &rects[g * n_l..(g + 1) * n_l];
        let cap: usize = gt_rects.iter().map(|r| r.y1.saturating_sub(r.y0) * r.x1.saturating_sub(r.x0)).sum();
        let mut cands = Vec::with_capacity(cap);
        for r in gt_rects {
            if r.is_empty() {
                continue;
            }
            for y in r.y0..r.y1 {
                let row = r.start + y * r.width;
                for x in r.x0..r.x1 {
                    let a = row + x;
                    let (m, u) = candidate_metric(
                        gt_box,
                        &pd_px[a - min_a],
                        sigmoid(cls_row[a]),
                        lam,
                        tal.nwd_weight,
                        alpha_op,
                        beta_op,
                    );
                    cands.push(Pick { anchor: a as u32, m, u });
                }
            }
        }
        keep_topk(&mut cands, tal.topk);
        selected.push(cands);
    }

    collect_image_fg(gt, &selected, tal)
}

/// Fallback for non-regular / custom [`Anchors`] or non-finite GT/config parameters: scans anchor points with
/// exact `contains_point`, still decoding DFL and evaluating `sigmoid` only on candidate anchors.
fn assign_image_general(
    reg: &[f32],
    logits: &[f32],
    anchors: &Anchors,
    reg_max: usize,
    gt: &ImageTargets,
    tal: &TalConfig,
) -> ImageFg {
    let n_a = anchors.len();
    let n_g = gt.boxes.len();
    let points_px = anchors.points_px();

    let mut per_gt_cands: Vec<Vec<u32>> = Vec::with_capacity(n_g);
    let mut cand_mask = vec![0u8; n_a];
    let mut min_a = n_a;
    let mut max_a = 0usize;
    for b in &gt.boxes {
        let cb = candidate_box(b, tal.min_box_size);
        let mut list = Vec::new();
        for (a, p) in points_px.iter().enumerate() {
            if cb.contains_point(p[0], p[1], tal.eps) {
                cand_mask[a] = 1;
                min_a = min_a.min(a);
                max_a = max_a.max(a + 1);
                list.push(a as u32);
            }
        }
        per_gt_cands.push(list);
    }
    if min_a >= max_a {
        return ImageFg::default();
    }

    let mut pd_px = vec![Xyxy::default(); max_a - min_a];
    decode_masked_channel_major(reg, anchors, reg_max, &cand_mask[min_a..max_a], min_a, &mut pd_px);

    let alpha_op = PowOp::from_exp(tal.alpha);
    let beta_op = PowOp::from_exp(tal.beta);
    let mut selected: Vec<Vec<Pick>> = Vec::with_capacity(n_g);
    for g in 0..n_g {
        let gt_box = &gt.boxes[g];
        let cls_row = &logits[gt.labels[g] as usize * n_a..(gt.labels[g] as usize + 1) * n_a];
        let lam = nwd_lambda(gt_box, tal.nwd_weight);
        let mut cands = Vec::with_capacity(per_gt_cands[g].len());
        for &a_u32 in &per_gt_cands[g] {
            let a = a_u32 as usize;
            let (m, u) = candidate_metric(
                gt_box,
                &pd_px[a - min_a],
                sigmoid(cls_row[a]),
                lam,
                tal.nwd_weight,
                alpha_op,
                beta_op,
            );
            cands.push(Pick { anchor: a_u32, m, u });
        }
        keep_topk(&mut cands, tal.topk);
        selected.push(cands);
    }

    collect_image_fg(gt, &selected, tal)
}

fn collect_image_fg(gt: &ImageTargets, selected: &[Vec<Pick>], tal: &TalConfig) -> ImageFg {
    let resolved = resolve_picks(gt.boxes.len(), selected, tal);
    let mut fg = ImageFg {
        anchor: Vec::with_capacity(resolved.len()),
        label: Vec::with_capacity(resolved.len()),
        score: Vec::with_capacity(resolved.len()),
        target: Vec::with_capacity(resolved.len()),
    };
    for pos in resolved {
        fg.anchor.push(pos.anchor);
        fg.label.push(gt.labels[pos.gt]);
        fg.score.push(pos.score);
        fg.target.push(gt.boxes[pos.gt]);
    }
    fg
}

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// The detached part of the loss (§7.1, §7.3): decode candidate predictions on the host and run the task-aligned
/// assigner on every image.
///
/// Candidate anchors are located in O(1) per FPN level from `anchors.levels`, and DFL decoding plus class
/// sigmoids are computed only on candidate anchors rather than all `A` anchors. Errors on mismatched shapes,
/// `targets.len() != B`, or a gt label `>= nc`.
pub fn assign_batch(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    cfg: &LossConfig,
) -> Result<BatchAssignment> {
    let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg)?;
    if targets.len() != b {
        return Err(invalid(format!("{} target images for a batch of {b}", targets.len())));
    }
    for (i, t) in targets.iter().enumerate() {
        if t.labels.len() != t.boxes.len() {
            return Err(invalid(format!("image {i}: {} boxes but {} labels", t.boxes.len(), t.labels.len())));
        }
        if let Some(&l) = t.labels.iter().find(|&&l| l as usize >= nc) {
            return Err(invalid(format!("image {i}: class {l} >= nc {nc}")));
        }
    }
    let reg_max = cfg.reg_max;
    let grid = validate_grid(anchors);
    let tal_finite = cfg.tal.eps.is_finite() && cfg.tal.min_box_size.is_finite();
    let (box_len, cls_len) = (4 * reg_max * n_a, nc * n_a);

    let per_image: Vec<(bool, ImageFg)> = host_f32(box_raw, |boxes| {
        host_f32(cls_raw, |cls| {
            (0..b)
                .into_par_iter()
                .map(|i| {
                    let reg = &boxes[i * box_len..(i + 1) * box_len];
                    let logits = &cls[i * cls_len..(i + 1) * cls_len];
                    let finite = all_finite(reg) && all_finite(logits);
                    let gt = &targets[i];
                    if !finite || gt.is_empty() || n_a == 0 || cfg.tal.topk == 0 {
                        return (finite, ImageFg::default());
                    }
                    let fg = match &grid {
                        Some(levels) if tal_finite && gt.boxes.iter().all(box_is_finite) => {
                            assign_image_grid(reg, logits, anchors, levels, reg_max, gt, &cfg.tal)
                        }
                        _ => assign_image_general(reg, logits, anchors, reg_max, gt, &cfg.tal),
                    };
                    (true, fg)
                })
                .collect()
        })
    })??;

    let mut out = BatchAssignment {
        batch: b,
        num_anchors: n_a,
        num_classes: nc,
        inputs_finite: per_image.iter().all(|(f, _)| *f),
        ..Default::default()
    };
    if out.inputs_finite {
        for (i, (_, fg)) in per_image.into_iter().enumerate() {
            out.image.extend(std::iter::repeat_n(i as u32, fg.anchor.len()));
            out.anchor.extend(fg.anchor);
            out.label.extend(fg.label);
            out.score.extend(fg.score);
            out.target.extend(fg.target);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assign::assign;
    use crate::decode::make_anchors;
    use folo_core::rng::Rng;

    /// Brute-force full-grid reference using the exact same `fast_exp` DFL decode, full-grid `sigmoid`,
    /// and linear-scan `assign`, for bit-exact parity checks against the grid-indexed lazy implementation.
    fn brute_force_assign_batch(
        box_raw: &Tensor,
        cls_raw: &Tensor,
        anchors: &Anchors,
        targets: &[ImageTargets],
        cfg: &LossConfig,
    ) -> BatchAssignment {
        let (b, nc, n_a) = check_inputs(box_raw, cls_raw, anchors, cfg).unwrap();
        let reg_max = cfg.reg_max;
        let points_px = anchors.points_px();
        let boxes = box_raw.to_vec::<f32>().unwrap();
        let cls = cls_raw.to_vec::<f32>().unwrap();
        let (box_len, cls_len) = (4 * reg_max * n_a, nc * n_a);

        let mut out = BatchAssignment {
            batch: b,
            num_anchors: n_a,
            num_classes: nc,
            inputs_finite: boxes.iter().chain(&cls).all(|v| v.is_finite()),
            ..Default::default()
        };
        if !out.inputs_finite {
            return out;
        }
        for (i, gt) in targets.iter().enumerate() {
            if gt.is_empty() {
                continue;
            }
            let reg = &boxes[i * box_len..(i + 1) * box_len];
            let logits = &cls[i * cls_len..(i + 1) * cls_len];
            // Full-grid channel-major DFL decode.
            let mut pd_px = vec![Xyxy::default(); n_a];
            let all_on = vec![1u8; n_a];
            decode_masked_channel_major(reg, anchors, reg_max, &all_on, 0, &mut pd_px);
            // Full-grid anchor-major scores.
            let mut scores = vec![0f32; n_a * nc];
            for c in 0..nc {
                for a in 0..n_a {
                    scores[a * nc + c] = sigmoid(logits[c * n_a + a]);
                }
            }
            let asg = assign(&scores, nc, &pd_px, &points_px, &gt.boxes, &gt.labels, &cfg.tal);
            for a in (0..n_a).filter(|&a| asg.fg_mask[a]) {
                out.image.push(i as u32);
                out.anchor.push(a as u32);
                out.label.push(asg.target_labels[a]);
                out.score.push(asg.target_scores[a]);
                out.target.push(asg.target_bboxes[a]);
            }
        }
        out
    }

    fn assert_bit_exact(got: &BatchAssignment, want: &BatchAssignment, label: &str) {
        assert_eq!(got.inputs_finite, want.inputs_finite, "{label}: inputs_finite");
        assert_eq!(got.image, want.image, "{label}: image");
        assert_eq!(got.anchor, want.anchor, "{label}: anchor");
        assert_eq!(got.label, want.label, "{label}: label");
        assert_eq!(got.target, want.target, "{label}: target");
        assert_eq!(got.score.len(), want.score.len(), "{label}: score len");
        for (k, (&g, &w)) in got.score.iter().zip(&want.score).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{label}: score[{k}] {g} != {w}");
        }
    }

    #[test]
    fn bit_exact_parity_across_tal_modes_and_edge_cases() {
        let anchors = make_anchors(128, 160, &[8, 16, 32], 0.5);
        let a = anchors.len();
        let (b, nc, reg_max) = (5usize, 6usize, 16usize);
        let mut rng = Rng::seed(0xC0DE_1055);
        let box_raw = Tensor::rand_uniform([b, 4 * reg_max, a], -2.0, 4.0, &mut rng);
        let cls_raw = Tensor::rand_uniform([b, nc, a], -6.0, 3.0, &mut rng);

        // Cover: empty image, edge-clamped/out-of-bounds/exact-boundary boxes, tiny boxes, overlapping boxes.
        let targets = vec![
            ImageTargets::default(),
            ImageTargets {
                boxes: vec![
                    Xyxy::new(-30.0, -20.0, 12.0, 10.0),   // clipped at top-left
                    Xyxy::new(148.0, 118.0, 200.0, 180.0), // clipped at bottom-right
                    Xyxy::new(-80.0, -80.0, -20.0, -20.0), // completely outside
                    Xyxy::new(4.0, 4.0, 20.0, 20.0),       // P3 centre (4, 4) sits exactly on edge
                    Xyxy::new(3.9999998, 4.0, 20.0, 20.0), // near-eps boundary
                ],
                labels: vec![0, 5, 1, 2, 3],
            },
            ImageTargets {
                boxes: vec![
                    Xyxy::new(28.5, 28.5, 34.5, 34.5), // tiny 6x6
                    Xyxy::new(60.0, 60.0, 61.0, 61.0), // tiny 1x1
                    Xyxy::new(0.0, 0.0, 160.0, 128.0), // full-image box
                ],
                labels: vec![4, 4, 1],
            },
            ImageTargets {
                boxes: vec![
                    Xyxy::new(20.0, 20.0, 80.0, 80.0),
                    Xyxy::new(32.0, 24.0, 92.0, 84.0),
                    Xyxy::new(20.0, 20.0, 80.0, 80.0), // identical duplicate GT (conflict tie)
                ],
                labels: vec![2, 2, 3],
            },
            ImageTargets {
                boxes: vec![Xyxy::new(10.0, 10.0, 50.0, 15.0), Xyxy::new(100.0, 4.0, 104.0, 110.0)],
                labels: vec![0, 5],
            },
        ];

        let configs = [
            ("default", LossConfig::default()),
            ("o2o", LossConfig::o2o()),
            (
                "nwd_weight",
                LossConfig {
                    tal: TalConfig { nwd_weight: 0.5, ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
            (
                "eps_free",
                LossConfig {
                    tal: TalConfig { eps_free: true, ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
            (
                "nwd_eps_free_o2o",
                LossConfig {
                    tal: TalConfig { nwd_weight: 0.75, eps_free: true, ..TalConfig::o2o() },
                    ..LossConfig::default()
                },
            ),
            (
                "no_min_box_size",
                LossConfig {
                    tal: TalConfig { min_box_size: 0.0, ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
            (
                "custom_exponents_int",
                LossConfig {
                    tal: TalConfig { alpha: 1.0, beta: 4.0, ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
            (
                "custom_exponents_general",
                LossConfig {
                    tal: TalConfig { alpha: 0.35, beta: 5.5, ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
            (
                "zero_exponents",
                LossConfig {
                    tal: TalConfig { alpha: 0.0, beta: 0.0, ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
            (
                "topk_zero",
                LossConfig { tal: TalConfig { topk: 0, ..TalConfig::default() }, ..LossConfig::default() },
            ),
            (
                "topk2_zero",
                LossConfig {
                    tal: TalConfig { topk2: Some(0), ..TalConfig::default() },
                    ..LossConfig::default()
                },
            ),
        ];

        for (name, cfg) in &configs {
            let got = assign_batch(&box_raw, &cls_raw, &anchors, &targets, cfg).unwrap();
            let want = brute_force_assign_batch(&box_raw, &cls_raw, &anchors, &targets, cfg);
            assert_bit_exact(&got, &want, name);
        }

        // Non-0.5 anchor offset and fallback (empty `levels`) both match bit-for-bit.
        let anchors_off0 = make_anchors(96, 96, &[8, 16], 0.0);
        let a0 = anchors_off0.len();
        let b0 = Tensor::rand_uniform([1, 4 * reg_max, a0], -1.0, 2.0, &mut rng);
        let c0 = Tensor::rand_uniform([1, nc, a0], -4.0, 2.0, &mut rng);
        let t0 = vec![ImageTargets {
            boxes: vec![Xyxy::new(0.0, 0.0, 48.0, 48.0), Xyxy::new(40.0, 40.0, 96.0, 96.0)],
            labels: vec![1, 3],
        }];
        let cfg = LossConfig::default();
        assert_bit_exact(
            &assign_batch(&b0, &c0, &anchors_off0, &t0, &cfg).unwrap(),
            &brute_force_assign_batch(&b0, &c0, &anchors_off0, &t0, &cfg),
            "offset_0",
        );

        let mut irregular = anchors_off0.clone();
        irregular.levels.clear();
        assert_bit_exact(
            &assign_batch(&b0, &c0, &irregular, &t0, &cfg).unwrap(),
            &brute_force_assign_batch(&b0, &c0, &irregular, &t0, &cfg),
            "irregular_fallback",
        );

        // Non-finite input on a non-candidate anchor still marks `inputs_finite = false`.
        let mut cls_nan = c0.to_vec::<f32>().unwrap();
        cls_nan[a0 - 1] = f32::NAN;
        let c_nan = Tensor::from_vec(cls_nan, [1, nc, a0]).unwrap();
        let res = assign_batch(&b0, &c_nan, &anchors_off0, &t0, &cfg).unwrap();
        assert!(!res.inputs_finite && res.num_fg() == 0);
    }

    #[test]
    fn all_finite_handles_extremes_and_non_finite_lanes() {
        let base = [
            0.0f32,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            1.0,
            -1.0,
            42.5,
            1e-30,
        ];
        assert!(all_finite(&[]));
        assert!(all_finite(&base));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for pos in 0..base.len() {
                let mut v = base;
                v[pos] = bad;
                assert!(!all_finite(&v), "missed {bad} at lane {pos}");
            }
        }
    }
}
