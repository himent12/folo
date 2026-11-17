//! The rank term against an f64 reference (value and closed-form gradient), finite differences, its invariance to a
//! per-class increasing linear map of the student's logits, the `rank_min` rule, and its place in `distill_loss`
//! (off: bit-identical to the two-term loss).

use folo_core::Tensor;
use folo_core::rng::Rng;
use folo_core::testing::{GradCheck, grad_check_with};

use super::*;
use crate::loss::distill::{DistillLoss, DistillTargets, TeacherView, distill_loss};

fn cfg(gain: f32, power: f32, min: usize) -> DistillConfig {
    DistillConfig { rank_gain: gain, rank_power: power, rank_min: min, ..DistillConfig::new("t.folo") }
}

fn uniform(seed: u64, dims: [usize; 3], lo: f32, hi: f32) -> Vec<f32> {
    Tensor::rand_uniform(dims, lo, hi, &mut Rng::seed(seed)).to_vec::<f32>().unwrap()
}

fn t3(v: &[f32], dims: [usize; 3]) -> Tensor {
    Tensor::from_vec(v.to_vec(), dims).unwrap()
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// The f64 reference of the module docs: `(L_rank, dL/dx, classes used)` for logits `x`, `y` (`y` not yet
/// floored) of shape `[b, nc, a]`, with the weights taken at `x` (constants).
fn reference(x: &[f32], y: &[f32], dims: [usize; 3], c: &DistillConfig) -> (f64, Vec<f64>, usize) {
    let [b, nc, a] = dims;
    let tau = c.rank_floor as f64;
    let l = (tau / (1.0 - tau)).ln();
    let eps = RANK_EPS as f64;
    let idx = |i: usize, k: usize, j: usize| (i * nc + k) * a + j;
    let mut w = vec![0f64; x.len()];
    for (i, wi) in w.iter_mut().enumerate() {
        let m = sigmoid(x[i] as f64).max(sigmoid(y[i] as f64));
        *wi = if m > tau { m.powf(c.rank_power as f64) } else { 0.0 };
    }
    let xf = |i: usize| (x[i] as f64).max(l);
    let yf = |i: usize| (y[i] as f64).max(l);
    let mut stats = Vec::new();
    for k in 0..nc {
        let cells: Vec<usize> = (0..b).flat_map(|i| (0..a).map(move |j| idx(i, k, j))).collect();
        let ws: f64 = cells.iter().map(|&i| w[i]).sum();
        let n = cells.iter().filter(|&&i| w[i] > 0.0).count();
        let wd = ws.max(f64::MIN_POSITIVE);
        let mx = cells.iter().map(|&i| w[i] * xf(i)).sum::<f64>() / wd;
        let my = cells.iter().map(|&i| w[i] * yf(i)).sum::<f64>() / wd;
        let vx = cells.iter().map(|&i| w[i] * (xf(i) - mx).powi(2)).sum::<f64>() / wd;
        let vy = cells.iter().map(|&i| w[i] * (yf(i) - my).powi(2)).sum::<f64>() / wd;
        let cxy = cells.iter().map(|&i| w[i] * (xf(i) - mx) * (yf(i) - my)).sum::<f64>() / wd;
        let (sx, sy) = ((vx + eps).sqrt(), (vy + eps).sqrt());
        let used = n >= c.rank_min && vy > eps;
        stats.push((cells, ws, mx, my, sx, sy, cxy / (sx * sy), used));
    }
    let total_w: f64 = stats.iter().filter(|s| s.7).map(|s| s.1).sum();
    let mut loss = 0.0;
    let mut grad = vec![0f64; x.len()];
    for (cells, ws, mx, my, sx, sy, rho, used) in &stats {
        if !used {
            continue;
        }
        loss += ws * (1.0 - rho) / total_w;
        for &i in cells {
            if x[i] as f64 >= l {
                grad[i] = -(w[i] / total_w) * ((yf(i) - my) / (sx * sy) - rho * (xf(i) - mx) / (sx * sx));
            }
        }
    }
    (loss, grad, stats.iter().filter(|s| s.7).count())
}

/// The tensor implementation: `(L_rank, dL/dx, classes used)`.
fn implementation(x: &[f32], y: &[f32], dims: [usize; 3], c: &DistillConfig) -> (f32, Vec<f32>, f32) {
    let xt = t3(x, dims).requires_grad_();
    let yt = t3(y, dims);
    let floored = yt.maximum_scalar(floor_logit(c.rank_floor)).unwrap();
    let out = rank_loss(&xt, &floored, &yt.sigmoid().unwrap(), c).unwrap();
    let g = out.loss.backward().unwrap().get(&xt).unwrap().to_vec::<f32>().unwrap();
    (out.loss.to_vec::<f32>().unwrap()[0], g, out.classes.to_vec::<f32>().unwrap()[0])
}

/// A batch like a detector's: most elements far below the floor in both models, a tail of candidates. Class 2 of
/// the second image is shifted so that the classes differ in mean and spread.
fn batch(seed: u64, dims: [usize; 3]) -> (Vec<f32>, Vec<f32>) {
    let t = uniform(seed, dims, -9.0, 4.0);
    let noise = uniform(seed + 1, dims, -2.5, 2.5);
    let s: Vec<f32> = t.iter().zip(&noise).map(|(t, n)| 0.6 * t - 1.0 + n).collect();
    (s, t)
}

fn max_rel(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0 && got.iter().all(|v| v.is_finite()), "the case has content and is finite");
    got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs() / scale).fold(0.0, f64::max)
}

/// Value, closed-form gradient and classes used against the f64 reference, at gamma 0.5, 1 and 0, on a 16-image
/// batch with 80 classes; and at a `rank_min` of the median class's candidate count, where some classes drop out.
#[test]
fn matches_the_f64_reference() {
    let dims = [16, 80, 84];
    let (x, y) = batch(1, dims);
    let tau = DistillConfig::default().rank_floor as f64;
    let mut counts: Vec<usize> = (0..80)
        .map(|k| {
            let cand = |i: usize| sigmoid(x[i] as f64).max(sigmoid(y[i] as f64)) > tau;
            (0..16).flat_map(|b| (0..84).map(move |j| (b * 80 + k) * 84 + j)).filter(|&i| cand(i)).count()
        })
        .collect();
    counts.sort();
    let median = counts[40];
    for (power, min) in [(0.5, 8), (1.0, 8), (0.0, 8), (0.5, median)] {
        let c = cfg(1.0, power, min);
        let (lw, gw, nw) = reference(&x, &y, dims, &c);
        let (lg, gg, ng) = implementation(&x, &y, dims, &c);
        let what = format!("gamma {power} min {min}");
        assert!(lw > 0.05 && lw < 1.5, "{what}: the case has content: {lw}");
        assert!(((lg as f64 - lw) / lw).abs() < 1e-5, "{what}: {lg} vs {lw}");
        assert!(max_rel(&gg, &gw) < 1e-5, "{what}: gradient {}", max_rel(&gg, &gw));
        assert_eq!(ng as usize, nw, "{what}: classes used");
        if min == median {
            assert!(nw > 5 && nw < 75, "{what}: some classes drop out: {nw}");
        }
    }
}

/// Central finite differences of the f64 reference, the weights held at the base point (they are constants of the
/// term), against the tensor gradient.
#[test]
fn gradient_matches_finite_differences() {
    let dims = [2, 3, 20];
    let (x, y) = batch(3, dims);
    let c = cfg(1.0, 0.5, 4);
    let (_, g, _) = implementation(&x, &y, dims, &c);
    // The reference with the base point's weights: the student's probability enters only through them.
    let tau = c.rank_floor as f64;
    let l = (tau / (1.0 - tau)).ln() as f32;
    let h = 1e-3f64;
    let mut checked = 0;
    for i in 0..x.len() {
        // Away from the floor's kink.
        if (x[i] - l).abs() < 0.01 {
            continue;
        }
        let at = |d: f64| {
            let mut xp: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            xp[i] += d;
            loss_frozen(&xp, &x, &y, dims, &c)
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        assert!((g[i] as f64 - fd).abs() < 1e-5 + 1e-3 * fd.abs(), "[{i}]: {} vs {fd}", g[i]);
        checked += 1;
    }
    assert!(checked > 100 && g.iter().any(|v| v.abs() > 1e-3), "the case has content");
}

/// `L_rank` in f64 at logits `xp`, with the weights of the base point `x0`.
fn loss_frozen(xp: &[f64], x0: &[f32], y: &[f32], dims: [usize; 3], c: &DistillConfig) -> f64 {
    let [b, nc, a] = dims;
    let tau = c.rank_floor as f64;
    let l = (tau / (1.0 - tau)).ln();
    let eps = RANK_EPS as f64;
    let w: Vec<f64> = x0
        .iter()
        .zip(y)
        .map(|(&x, &y)| {
            let m = sigmoid(x as f64).max(sigmoid(y as f64));
            if m > tau { m.powf(c.rank_power as f64) } else { 0.0 }
        })
        .collect();
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..nc {
        let cells: Vec<usize> = (0..b).flat_map(|i| (0..a).map(move |j| (i * nc + k) * a + j)).collect();
        let ws: f64 = cells.iter().map(|&i| w[i]).sum();
        let n = cells.iter().filter(|&&i| w[i] > 0.0).count();
        let xf = |i: usize| xp[i].max(l);
        let yf = |i: usize| (y[i] as f64).max(l);
        let mx = cells.iter().map(|&i| w[i] * xf(i)).sum::<f64>() / ws;
        let my = cells.iter().map(|&i| w[i] * yf(i)).sum::<f64>() / ws;
        let vx = cells.iter().map(|&i| w[i] * (xf(i) - mx).powi(2)).sum::<f64>() / ws;
        let vy = cells.iter().map(|&i| w[i] * (yf(i) - my).powi(2)).sum::<f64>() / ws;
        let cxy = cells.iter().map(|&i| w[i] * (xf(i) - mx) * (yf(i) - my)).sum::<f64>() / ws;
        if n >= c.rank_min && vy > eps {
            num += ws * (1.0 - cxy / ((vx + eps).sqrt() * (vy + eps).sqrt()));
            den += ws;
        }
    }
    num / den
}

/// Where the weights do not depend on the student (the teacher above it everywhere), the term is a plain function
/// of `x`: folo-core's finite-difference check of the whole tensor graph.
#[test]
fn tensor_graph_passes_grad_check() {
    let dims = [2, 3, 12];
    let y = uniform(5, dims, -2.0, 3.0);
    let x = uniform(6, dims, -4.5, -2.5);
    let c = cfg(1.0, 0.5, 4);
    let yt = t3(&y, dims);
    let (floored, p) = (yt.maximum_scalar(floor_logit(c.rank_floor)).unwrap(), yt.sigmoid().unwrap());
    let f = |v: &[Tensor]| Ok(rank_loss(&v[0], &floored, &p, &c)?.loss);
    let opts = GradCheck { eps: 1e-2, atol: 1e-4, rtol: 1e-2, max_checks: 0, ..GradCheck::default() };
    grad_check_with(f, &[t3(&x, dims)], &opts).unwrap();
}

/// Per class, an increasing linear map of the student's logits (all above the floor, gamma 0 so the weights stay
/// at one) leaves the term unchanged; a student that is such a map of the teacher has a term of about 0 and no
/// gradient; the negated teacher has the largest term.
#[test]
fn invariant_to_per_class_increasing_linear_maps() {
    let dims = [3, 4, 30];
    let [b, nc, a] = dims;
    // Every logit and its maps stay above the floor (-5.29), and gamma 0 keeps every weight at 1.
    let y = uniform(7, dims, -2.0, 4.0);
    let x = uniform(8, dims, -1.5, 2.0);
    let c = cfg(1.0, 0.0, 8);
    // Invariance is exact up to `eps` in the standard deviations: scales near 1 keep that within the tolerance.
    let (scale, shift) = ([0.5f32, 1.0, 2.5, 1.5], [1.0f32, -0.5, 0.3, 2.0]);
    let map = |v: &[f32]| -> Vec<f32> {
        (0..v.len()).map(|i| scale[(i / a) % nc] * v[i] + shift[(i / a) % nc]).collect()
    };
    let (l0, _, n0) = implementation(&x, &y, dims, &c);
    assert_eq!(n0 as usize, nc);
    let (l1, _, _) = implementation(&map(&x), &y, dims, &c);
    assert!(l0 > 0.3 && (l1 - l0).abs() < 1e-5 * l0, "{l1} vs {l0}");
    let (l2, g2, _) = implementation(&map(&y), &y, dims, &c);
    assert!(l2.abs() < 1e-4 && g2.iter().all(|v| v.abs() < 1e-4), "{l2}");
    let flipped: Vec<f32> = y.iter().map(|v| -v).collect();
    let (l3, _, _) = implementation(&flipped, &y, dims, &c);
    assert!((l3 - 2.0).abs() < 1e-3, "{l3}");
    assert_eq!(b * nc * a, x.len());
}

/// With no logit under the floor, a class's gradient sums to 0 (the mean logit does not move) and its product with
/// `x' − mx` is `−(W_c / Σ W) · rho_c · eps / (vx + eps)` (the spread moves only by the `eps` share).
#[test]
fn moves_the_mean_not_and_the_spread_only_by_eps() {
    let dims = [2, 3, 40];
    let [b, nc, a] = dims;
    // Every logit above the floor and every element a candidate (p >= 0.047 > tau), so every class is used.
    let y = uniform(9, dims, -3.0, 4.0);
    let x = uniform(10, dims, -3.0, 2.0);
    let c = cfg(1.0, 0.5, 8);
    let (_, g, _) = implementation(&x, &y, dims, &c);
    let w: Vec<f64> =
        x.iter().zip(&y).map(|(&x, &y)| sigmoid(x as f64).max(sigmoid(y as f64)).sqrt()).collect();
    let w_all: f64 = w.iter().sum();
    let eps = RANK_EPS as f64;
    for k in 0..nc {
        let cells: Vec<usize> = (0..b).flat_map(|i| (0..a).map(move |j| (i * nc + k) * a + j)).collect();
        let ws: f64 = cells.iter().map(|&i| w[i]).sum();
        let mean = |v: &[f32]| cells.iter().map(|&i| w[i] * v[i] as f64).sum::<f64>() / ws;
        let (mx, my) = (mean(&x), mean(&y));
        let moment = |f: &dyn Fn(usize) -> f64| cells.iter().map(|&i| w[i] * f(i)).sum::<f64>() / ws;
        let vx = moment(&|i| (x[i] as f64 - mx).powi(2));
        let vy = moment(&|i| (y[i] as f64 - my).powi(2));
        let cxy = moment(&|i| (x[i] as f64 - mx) * (y[i] as f64 - my));
        let rho = cxy / ((vx + eps).sqrt() * (vy + eps).sqrt());
        let want_dot = -(ws / w_all) * rho * eps / (vx + eps);
        let sum: f64 = cells.iter().map(|&i| g[i] as f64).sum();
        let dot: f64 = cells.iter().map(|&i| g[i] as f64 * (x[i] as f64 - mx)).sum();
        let norm: f64 = cells.iter().map(|&i| (g[i] as f64).abs()).sum();
        assert!(norm > 1e-2 && sum.abs() < 1e-6, "class {k}: {sum} {norm}");
        assert!(want_dot.abs() > 1e-7 && (dot - want_dot).abs() < 1e-7, "class {k}: {dot} vs {want_dot}");
    }
}

/// A class with fewer than `rank_min` candidates contributes nothing: no gradient, not counted, and the term equals
/// that of the batch without it.
#[test]
fn a_class_under_rank_min_contributes_nothing() {
    let dims = [2, 3, 30];
    let [b, nc, a] = dims;
    let (mut x, mut y) = batch(11, dims);
    // Class 1: far below the floor in both models but at 5 elements.
    for i in 0..b {
        for j in 0..a {
            let k = (i * nc + 1) * a + j;
            let lit = i == 0 && j < 5;
            // The lit five are neither correlated nor anti-correlated (at rho = ±1 the gradient is 0).
            let lit_y = ((j * 7) % 5) as f32 - 2.0;
            (x[k], y[k]) = if lit { (j as f32 - 2.0, lit_y) } else { (-9.0, -8.0 - j as f32 * 0.01) };
        }
    }
    let c = cfg(1.0, 0.5, 8);
    let (l, g, n) = implementation(&x, &y, dims, &c);
    assert_eq!(n, 2.0);
    for i in 0..b {
        assert!(g[(i * nc + 1) * a..(i * nc + 2) * a].iter().all(|v| *v == 0.0));
    }
    let drop = |v: &[f32]| -> Vec<f32> { (0..v.len()).filter(|i| (i / a) % nc != 1).map(|i| v[i]).collect() };
    let (l2, _, n2) = implementation(&drop(&x), &drop(&y), [b, nc - 1, a], &c);
    assert!(n2 == 2.0 && (l - l2).abs() < 1e-6 * l2.max(1e-3), "{l} vs {l2}");
    // With rank_min 5 it counts.
    let (_, g5, n5) = implementation(&x, &y, dims, &cfg(1.0, 0.5, 5));
    assert_eq!(n5, 3.0);
    assert!(g5[a..a + 5].iter().any(|v| v.abs() > 1e-4));
}

/// A class whose teacher does not vary (`vy <= eps`: every teacher logit under the floor, so `y' = l`) is not used,
/// even with many student candidates: no gradient, not counted, and the term equals that of the batch without it
/// (counting it would add `W · 1` with no gradient and dilute the other classes').
#[test]
fn a_class_whose_teacher_does_not_vary_is_not_used() {
    let dims = [2, 3, 30];
    let [b, nc, a] = dims;
    let (mut x, mut y) = batch(13, dims);
    for i in 0..b {
        for j in 0..a {
            let k = (i * nc + 1) * a + j;
            // The student fires on all 60 elements, in varied order; the teacher is far below the floor.
            (x[k], y[k]) = (((j * 7) % 11) as f32 * 0.3 - 1.0, -9.0 - j as f32 * 0.01);
        }
    }
    let c = cfg(1.0, 0.5, 8);
    let (l, g, n) = implementation(&x, &y, dims, &c);
    assert_eq!(n, 2.0);
    for i in 0..b {
        assert!(g[(i * nc + 1) * a..(i * nc + 2) * a].iter().all(|v| *v == 0.0));
    }
    let drop = |v: &[f32]| -> Vec<f32> { (0..v.len()).filter(|i| (i / a) % nc != 1).map(|i| v[i]).collect() };
    let (l2, _, n2) = implementation(&drop(&x), &drop(&y), [b, nc - 1, a], &c);
    assert!(n2 == 2.0 && (l - l2).abs() < 1e-6 * l2.max(1e-3), "{l} vs {l2}");
}

/// No class used (every class under `rank_min`): the term is 0 with a zero gradient, not NaN.
#[test]
fn no_class_used_is_zero() {
    let dims = [1, 2, 6];
    let (x, y) = batch(12, dims);
    let (l, g, n) = implementation(&x, &y, dims, &cfg(1.0, 0.5, 100));
    assert!(l == 0.0 && n == 0.0 && g.iter().all(|v| *v == 0.0), "{l} {n}");
}

fn view(seed: u64, dims: [usize; 3]) -> (Tensor, Tensor, TeacherView) {
    let [b, _, a] = dims;
    let mut rng = Rng::seed(seed);
    let (tb, tc) =
        (Tensor::randn([b, 64, a], 0.0, 2.0, &mut rng), Tensor::rand_uniform(dims, -8.0, 3.0, &mut rng));
    let (sb, sc) =
        (Tensor::randn([b, 64, a], 0.0, 2.0, &mut rng), Tensor::rand_uniform(dims, -8.0, 3.0, &mut rng));
    (sb, sc, TeacherView::new(&tb, &tc, 16, 4.0).unwrap())
}

fn run(sb: &Tensor, sc: &Tensor, t: &DistillTargets, c: &DistillConfig) -> (DistillLoss, Vec<f32>, Vec<f32>) {
    let (z, x) = (sb.detach().requires_grad_(), sc.detach().requires_grad_());
    let out = distill_loss(&z, &x, t, 2.5, c).unwrap();
    let g = out.total.backward().unwrap();
    let (gx, gz) = (g.get(&x).unwrap().to_vec::<f32>().unwrap(), g.get(&z).unwrap().to_vec::<f32>().unwrap());
    (out, gx, gz)
}

/// With `rank_gain` 0 the rank keys change nothing, bit for bit, and the targets carry no rank logits; with it on
/// the total gains exactly `B · rank_gain · L_rank` and its gradient, the box gradient is unchanged, and the
/// reported terms are the two-term loss's plus `rank`.
#[test]
fn in_distill_loss_off_is_bit_identical_and_on_adds_the_term() {
    let dims = [2, 5, 84];
    let (sb, sc, v) = view(13, dims);
    let off = DistillConfig::new("t");
    let t_off = DistillTargets::for_config(&v, &off).unwrap();
    assert!(t_off.rank_logit.is_none());
    let (a, gxa, gza) = run(&sb, &sc, &t_off, &off);
    let odd = DistillConfig { rank_floor: 0.2, rank_power: 3.0, rank_min: 2, ..off.clone() };
    let (b2, gxb, gzb) = run(&sb, &sc, &DistillTargets::new(&v).unwrap(), &odd);
    let bits = |t: &[f32]| t.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(a.total.to_vec::<f32>().unwrap()[0].to_bits(), b2.total.to_vec::<f32>().unwrap()[0].to_bits());
    assert_eq!((bits(&gxa), bits(&gza)), (bits(&gxb), bits(&gzb)));
    assert_eq!(
        (a.cls.to_bits(), a.dfl.to_bits(), a.rank, a.rank_classes),
        (b2.cls.to_bits(), b2.dfl.to_bits(), 0.0, 0.0)
    );

    let on = DistillConfig { rank_gain: 2.0, ..off.clone() };
    let t_on = DistillTargets::for_config(&v, &on).unwrap();
    let (r, gxr, gzr) = run(&sb, &sc, &t_on, &on);
    assert_eq!((r.cls.to_bits(), r.dfl.to_bits()), (a.cls.to_bits(), a.dfl.to_bits()));
    assert_eq!(bits(&gzr), bits(&gza), "the rank term does not touch the boxes");
    let x = sc.detach().requires_grad_();
    let alone = rank_loss(&x, t_on.rank_logit.as_ref().unwrap(), &t_on.cls_prob, &on).unwrap();
    let l_rank = alone.loss.to_vec::<f32>().unwrap()[0];
    let g_rank = alone.loss.backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap();
    assert!(l_rank > 0.05 && (r.rank - 2.0 * l_rank).abs() < 1e-6, "{} vs {l_rank}", r.rank);
    assert_eq!(r.rank_classes, alone.classes.to_vec::<f32>().unwrap()[0]);
    let want = 2.0 * (a.cls + a.dfl + r.rank);
    assert!((r.total.to_vec::<f32>().unwrap()[0] - want).abs() < 1e-5 * want);
    for i in 0..gxr.len() {
        let w = gxa[i] + 2.0 * 2.0 * g_rank[i];
        assert!((gxr[i] - w).abs() < 1e-6 + 1e-5 * w.abs(), "[{i}] {} vs {w}", gxr[i]);
    }
    // The rank term needs its targets.
    let e = distill_loss(&sb, &sc, &t_off, 2.5, &on).unwrap_err();
    assert!(e.to_string().contains("rank"), "{e}");
}

#[test]
fn rank_settings_are_validated_and_default_when_absent() {
    let ok = DistillConfig { rank_gain: 2.0, ..DistillConfig::new("t") };
    assert!(ok.validate().is_ok() && ok.rank_on() && !DistillConfig::new("t").rank_on());
    for (bad, key) in [
        (DistillConfig { rank_gain: -1.0, ..ok.clone() }, "rank_gain"),
        (DistillConfig { rank_gain: f32::NAN, ..ok.clone() }, "rank_gain"),
        (DistillConfig { rank_power: f32::INFINITY, ..ok.clone() }, "rank_power"),
        (DistillConfig { rank_floor: 0.0, ..ok.clone() }, "rank_floor"),
        (DistillConfig { rank_floor: 1.0, ..ok.clone() }, "rank_floor"),
        (DistillConfig { rank_floor: f32::NAN, ..ok.clone() }, "rank_floor"),
        (DistillConfig { rank_min: 1, ..ok.clone() }, "rank_min"),
    ] {
        let e = bad.validate().unwrap_err();
        assert!(e.contains(key), "{e}");
    }
    let c: DistillConfig = serde_json::from_str(r#"{"teacher":"t.folo","cls_gain":0.25}"#).unwrap();
    assert_eq!((c.rank_gain, c.rank_floor, c.rank_power, c.rank_min), (0.0, 0.005, 0.5, 8));
    assert!((floor_logit(0.005) - (-5.293_305)).abs() < 1e-5);
}

/// The gradient probe's norms: the class term's is `B · cls_gain · ‖sigmoid(x) − p~‖ / N`, the rank term's
/// `B · rank_gain · ‖dL_rank/dx‖`, the supervised one that of the supervised loss's own backward; with the rank
/// term off its norm is 0.
#[test]
fn the_gradient_probe_reports_each_terms_norm() {
    use crate::boxes::Xyxy;
    use crate::decode::make_anchors;
    use crate::loss::distill::class_gradient_norms;
    use crate::loss::{ImageTargets, LossConfig, detection_loss};
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let dims = [2, 3, anchors.len()];
    let (sb, sc, v) = view(14, dims);
    let gt = vec![
        ImageTargets { boxes: vec![Xyxy::new(4.0, 6.0, 30.0, 40.0)], labels: vec![1] },
        ImageTargets { boxes: vec![Xyxy::new(20.0, 10.0, 60.0, 50.0)], labels: vec![2] },
    ];
    let lc = LossConfig::default();
    let on = DistillConfig { rank_gain: 2.0, ..DistillConfig::new("t") };
    let t = DistillTargets::for_config(&v, &on).unwrap();
    let [s, c, r] = class_gradient_norms(&sb, &sc, &anchors, &gt, &lc, &t, &on).unwrap();
    let l2 = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32;
    let x = sc.detach().requires_grad_();
    let sup = detection_loss(&sb, &x, &anchors, &gt, &lc).unwrap();
    let n = sup.target_score_sum.max(1.0);
    let want_s = l2(&sup.total.backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap());
    let (p, q) = (sc.sigmoid().unwrap().to_vec::<f32>().unwrap(), t.cls_prob.to_vec::<f32>().unwrap());
    let diff: Vec<f32> = p.iter().zip(&q).map(|(a, b)| 2.0 * 0.5 * (a - b) / n).collect();
    let x = sc.detach().requires_grad_();
    let rl = rank_loss(&x, t.rank_logit.as_ref().unwrap(), &t.cls_prob, &on).unwrap();
    let want_r = 2.0 * 2.0 * l2(&rl.loss.backward().unwrap().get(&x).unwrap().to_vec::<f32>().unwrap());
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-5 * b.abs() && b > 1e-3;
    assert!(
        close(s, want_s) && close(c, l2(&diff)) && close(r, want_r),
        "{s} {c} {r} vs {want_s} {} {want_r}",
        l2(&diff)
    );
    let off = DistillConfig::new("t");
    let [s0, c0, r0] =
        class_gradient_norms(&sb, &sc, &anchors, &gt, &lc, &DistillTargets::new(&v).unwrap(), &off).unwrap();
    assert!(s0 == s && c0 == c && r0 == 0.0);
}
