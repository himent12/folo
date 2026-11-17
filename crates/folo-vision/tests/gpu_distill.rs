//! Distillation (docs/MODEL.md §8.9) on the GPU against the CPU reference: the teacher's targets, the term and its
//! gradients (also under autocast), and a mixed-precision run with an f16 teacher.
//!
//! Tolerances: the term and its gradients are f32 compositions of softplus, sigmoid, log, log-softmax and row sums
//! whose GPU kernels round differently from the CPU's, so values agree to a relative 1e-5 and gradients to 1e-6 plus
//! 1e-4 of the largest one. The fused kernel against the tensor path: values to a relative 1e-5, gradients to 1e-5
//! of the largest one (`same_term`).
//!
//! This test binary asks for tensor cores before the shared GPU context exists; it skips (with a note) without a
//! hardware adapter, and the mixed-precision run without tensor cores.

use std::sync::{Arc, Mutex};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_nn::Module;
use folo_test_support::TestDir;
use folo_vision::decode::make_anchors;
use folo_vision::loss::distill::{distill_loss_fused, fused_supported};
use folo_vision::loss::{DistillConfig, DistillTargets, TeacherView, distill_loss};
use folo_vision::train::OptimizerChoice;
use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};

const R: usize = 16;

/// The GPU (tensor cores on when it has them) and whether it has them, or `None` without a hardware adapter.
fn gpu() -> Option<(Device, bool)> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    folo_gpu::request_tensor_cores();
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    let dev = folo_gpu::init().ok()?;
    let tc = ctx.caps().coop_f16;
    if tc {
        ctx.set_tensor_cores(true);
    }
    Some((dev, tc))
}

macro_rules! gpu {
    () => {
        match gpu() {
            Some(d) => d,
            None => return,
        }
    };
}

fn raw(seed: u64, b: usize, nc: usize, a: usize) -> (Tensor, Tensor) {
    let mut rng = Rng::seed(seed);
    (Tensor::randn([b, 4 * R, a], 0.0, 2.0, &mut rng), Tensor::rand_uniform([b, nc, a], -6.0, 3.0, &mut rng))
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6 + 1e-4 * scale, "{what} [{i}]: {g} vs {w} (largest {scale})");
    }
}

/// The term's values and both gradients, plus the targets, on `dev` (under autocast when `amp`).
fn term_on(dev: Device, amp: bool, temp: f32) -> (f32, f32, Vec<Vec<f32>>) {
    let anchors = make_anchors(64, 96, &[8, 16, 32], 0.5);
    let (sb, sc) = raw(1, 2, 5, anchors.len());
    let (tb, tc) = raw(2, 2, 5, anchors.len());
    let on = |t: &Tensor| t.to_device(dev).unwrap();
    let (z, x) = (on(&sb).requires_grad_(), on(&sc).requires_grad_());
    let _a = amp.then(folo_core::autocast);
    let view = TeacherView::new(&on(&tb), &on(&tc), R, temp).unwrap();
    let t = DistillTargets::new(&view).unwrap();
    let cfg = DistillConfig { temperature: temp, ..DistillConfig::new("t") };
    let out = distill_loss(&z, &x, &t, 3.5, &cfg).unwrap();
    let g = out.total.backward().unwrap();
    let grads = [&x, &z].map(|v| g.get(v).unwrap().to_vec::<f32>().unwrap()).to_vec();
    let mut all = grads;
    all.push(t.cls_entropy.to_vec::<f32>().unwrap());
    all.push(t.box_weight.to_vec::<f32>().unwrap());
    (out.cls, out.dfl, all)
}

/// The term, its gradients and the targets on the GPU agree with the CPU, at T = 1 and 2, and under autocast (the
/// term stays f32).
#[test]
fn term_and_gradients_match_the_cpu() {
    let (dev, _) = gpu!();
    for temp in [1.0, 2.0] {
        let (c0, d0, g0) = term_on(Device::Cpu, false, temp);
        assert!(c0 > 1e-2 && d0 > 1e-2, "the case has content: {c0} {d0}");
        for amp in [false, true] {
            let (c, d, g) = term_on(dev, amp, temp);
            let what = format!("T {temp} amp {amp}");
            assert!(
                (c - c0).abs() <= 1e-5 * c0 && (d - d0).abs() <= 1e-5 * d0,
                "{what}: {c} {d} vs {c0} {d0}"
            );
            for (k, name) in ["cls grad", "box grad", "cls entropy", "box weight"].iter().enumerate() {
                close(&g[k], &g0[k], &format!("{what} {name}"));
            }
        }
    }
}

/// Teacher outputs in f16 (the f16 teacher of a mixed-precision run) give the targets the CPU computes from the
/// same f16 values.
#[test]
fn f16_teacher_outputs_match_the_cpu() {
    let (dev, _) = gpu!();
    let (tb, tc) = raw(3, 1, 3, 126);
    let half = |t: &Tensor| t.to_dtype(DType::F16).unwrap();
    let cpu = TeacherView::new(&half(&tb), &half(&tc), R, 1.0).unwrap();
    let on = |t: &Tensor| half(t).to_device(dev).unwrap();
    let gpu = TeacherView::new(&on(&tb), &on(&tc), R, 1.0).unwrap();
    assert_eq!(gpu.cls_prob.dtype(), DType::F32);
    close(&gpu.cls_prob.to_vec::<f32>().unwrap(), &cpu.cls_prob.to_vec::<f32>().unwrap(), "cls");
    close(&gpu.box_prob.to_vec::<f32>().unwrap(), &cpu.box_prob.to_vec::<f32>().unwrap(), "box");
}

/// `|got − want| <= rel · max |want|` per element: agreement relative to the tensor's scale.
fn close_rel(got: &[f32], want: &[f32], rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    // `f32::max` skips NaN: both sides must be finite before the largest values mean anything.
    assert!(got.iter().chain(want).all(|v| v.is_finite()), "{what}: finite");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "{what}: content");
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    assert!(worst <= rel * scale, "{what}: largest difference {worst} vs largest value {scale}");
}

/// The reported class and DFL values and the total's value, and both gradients of the
/// term (scaled by [`UPSTREAM`]).
type Term = ([f32; 3], Vec<f32>, Vec<f32>);

/// The upstream gradient of the total in [`term`] (not 1, so a backward that ignores it fails the comparison).
const UPSTREAM: f32 = 0.37;

/// The term on `dev` for the student's `s` (f32) and the teacher's `t` (any float dtype), host tensors: fused from
/// the teacher's raw outputs (`fused`), or the tensor path on its targets. The values may be non-finite; the
/// gradients (of `UPSTREAM · total`) are only taken when the term is finite.
fn term(dev: Device, fused: bool, s: [&Tensor; 2], t: [&Tensor; 2], cfg: &DistillConfig) -> Term {
    let on = |t: &Tensor| t.to_device(dev).unwrap();
    let (z, x) = (on(s[0]).detach().requires_grad_(), on(s[1]).detach().requires_grad_());
    let (tb, tc) = (on(t[0]), on(t[1]));
    let out = if fused {
        distill_loss_fused(&z, &x, &tb, &tc, 7.5, cfg).unwrap()
    } else {
        let view = TeacherView::new(&tb, &tc, R, cfg.temperature).unwrap();
        distill_loss(&z, &x, &DistillTargets::new(&view).unwrap(), 7.5, cfg).unwrap()
    };
    let total = out.total.to_vec::<f32>().unwrap()[0];
    let v = [out.cls, out.dfl, total];
    if !out.is_finite() {
        return (v, vec![], vec![]);
    }
    let g = out.total.mul_scalar(UPSTREAM).unwrap().backward().unwrap();
    let [gx, gz] = [&x, &z].map(|v| g.get(v).unwrap().to_vec::<f32>().unwrap());
    (v, gx, gz)
}

/// [`term`] on 4 random images of 80 classes over 2100 anchors (three chunks of the class kernel per row).
fn fused_or_not(dev: Device, fused: bool, half: bool, cfg: &DistillConfig, seed: u64) -> Term {
    let anchors = make_anchors(320, 320, &[8, 16, 32], 0.5);
    let (b, nc, a) = (4, 80, anchors.len());
    let (sb, sc) = raw(seed, b, nc, a);
    let (tb, tc) = raw(seed + 1, b, nc, a);
    let teacher = |t: &Tensor| if half { t.to_dtype(DType::F16).unwrap() } else { t.clone() };
    term(dev, fused, [&sb, &sc], [&teacher(&tb), &teacher(&tc)], cfg)
}

/// The reported terms and the total within a relative `rel`, both gradients within `rel` of their largest value.
fn same_term(got: &Term, want: &Term, rel: f32, what: &str) {
    let ((v, gx, gz), (v0, gx0, gz0)) = (got, want);
    for k in 0..3 {
        assert!((v[k] - v0[k]).abs() <= rel * v0[k].abs(), "{what}: {v:?} vs {v0:?}");
    }
    close_rel(gx, gx0, rel, &format!("{what}: cls grad"));
    close_rel(gz, gz0, rel, &format!("{what}: box grad"));
}

/// The fused kernel (`folo_gpu::distill`) against the tensor path on the same GPU and on the CPU: both reported
/// terms within a relative 1e-5, both gradients within 1e-5 of their largest value; f32 and f16 teacher outputs,
/// T = 1 and 4, 4 images of 80 classes over 2100 anchors.
#[test]
fn fused_term_matches_the_tensor_path() {
    let (dev, _) = gpu!();
    let (_, c) = raw(1, 1, 3, 10);
    let on_gpu = c.to_device(dev).unwrap();
    assert!(fused_supported(&on_gpu, R) && !fused_supported(&on_gpu, 1) && !fused_supported(&c, R));
    assert!(!folo_gpu::distill::supports(1usize << (usize::BITS - 2), 4, 1, 2), "sizes that overflow usize");
    for (temp, half) in [(4.0, false), (1.0, false), (4.0, true), (1.0, true)] {
        let cfg = DistillConfig { temperature: temp, ..DistillConfig::new("t") };
        let what = format!("T {temp} f16 {half}");
        let fused = fused_or_not(dev, true, half, &cfg, 11);
        for (dev0, name) in [(dev, "gpu"), (Device::Cpu, "cpu")] {
            let want = fused_or_not(dev0, false, half, &cfg, 11);
            assert!(want.0[0] > 1e-2 && want.0[1] > 1e-2, "{what}: the case has content: {:?}", want.0);
            same_term(&fused, &want, 1e-5, &format!("{what} vs {name}"));
        }
    }
}

/// Teacher box logits sharing a large offset (2046 to 2048 in f16, 2^24 − 2 to 2^24 in f32, where neighbouring
/// floats are 1 or 2 apart): the softmax ignores the offset, so the term must equal the tensor path's on the same
/// pattern without it (0 to −2, exact) within 1e-5. With the offset the tensor path itself rounds `z / T` to a
/// multiple of 0.5 for `T` = 3 (and so is not the reference); for `T` = 1 it agrees with both.
#[test]
fn fused_term_keeps_precision_at_large_teacher_logits() {
    let (dev, _) = gpu!();
    let (b, nc, a) = (2, 3, 300);
    let (sb, sc) = raw(31, b, nc, a);
    let (_, tc) = raw(32, b, nc, a);
    let cases = [
        (2048.0, DType::F16, 1.0),
        (2048.0, DType::F16, 4.0),
        (16_777_216.0, DType::F32, 1.0),
        (16_777_216.0, DType::F32, 3.0),
    ];
    for (offset, dtype, temp) in cases {
        let pattern = |offset: f32| {
            let n = b * 4 * R * a;
            let tb: Vec<f32> = (0..n).map(|i| offset - ((i / a + i) % 3) as f32).collect();
            Tensor::from_vec(tb, [b, 4 * R, a]).unwrap().to_dtype(dtype).unwrap()
        };
        let tc = tc.to_dtype(dtype).unwrap();
        let cfg = DistillConfig { temperature: temp, ..DistillConfig::new("t") };
        let what = format!("offset {offset} {dtype} T {temp}");
        let want = term(Device::Cpu, false, [&sb, &sc], [&pattern(0.0), &tc], &cfg);
        let got = term(dev, true, [&sb, &sc], [&pattern(offset), &tc], &cfg);
        same_term(&got, &want, 1e-5, &what);
        if temp == 1.0 {
            let tensor = term(Device::Cpu, false, [&sb, &sc], [&pattern(offset), &tc], &cfg);
            same_term(&got, &tensor, 1e-5, &format!("{what}, tensor path with the offset"));
        }
    }
}

/// Non-finite inputs: the fused term is non-finite (NaN, so the trainer skips the batch) exactly when the tensor
/// path's is, and otherwise agrees with it (a teacher class logit of ±inf, a teacher box bin of −inf).
#[test]
fn fused_term_is_non_finite_exactly_when_the_tensor_path_is() {
    let (dev, _) = gpu!();
    let cfg = DistillConfig::new("t");
    let (b, nc, a) = (2, 3, 200);
    let (sb, sc) = raw(21, b, nc, a);
    let (tb, tc) = raw(22, b, nc, a);
    let side0: Vec<usize> = (0..R).map(|r| r * a + 17).collect();
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    // (input: 0 student box, 1 student cls, 2 teacher box, 3 teacher cls; elements; value; tensor path finite)
    let cases: Vec<(usize, Vec<usize>, f32, bool)> = vec![
        (0, vec![17], nan, false),
        (0, vec![17], inf, false),
        (0, vec![17], -inf, false),
        (1, vec![17], nan, false),
        (1, vec![17], inf, false),
        (1, vec![17], -inf, false),
        (2, vec![17], nan, false),
        (2, vec![17], inf, false),
        (2, vec![17], -inf, true),
        (2, side0, -inf, false),
        (3, vec![17], nan, false),
        (3, vec![17], inf, true),
        (3, vec![17], -inf, true),
    ];
    for half in [false, true] {
        for (which, at, value, finite) in &cases {
            let mut v = [&sb, &sc, &tb, &tc].map(|t| t.to_vec::<f32>().unwrap());
            for &i in at {
                v[*which][i] = *value;
            }
            let t: Vec<Tensor> = v
                .into_iter()
                .zip([&sb, &sc, &tb, &tc])
                .map(|(v, t)| Tensor::from_vec(v, t.dims()).unwrap())
                .collect();
            let teacher = |t: &Tensor| if half { t.to_dtype(DType::F16).unwrap() } else { t.clone() };
            let (t0, t1) = (teacher(&t[2]), teacher(&t[3]));
            let what = format!("input {which} at {} ({} elements) = {value}, f16 {half}", at[0], at.len());
            let want = term(dev, false, [&t[0], &t[1]], [&t0, &t1], &cfg);
            let got = term(dev, true, [&t[0], &t[1]], [&t0, &t1], &cfg);
            let is_finite = |v: &[f32; 3]| v[0].is_finite() && v[1].is_finite();
            assert_eq!(is_finite(&want.0), *finite, "{what}: tensor path {:?}", want.0);
            assert_eq!(is_finite(&got.0), *finite, "{what}: fused {:?}", got.0);
            if *finite {
                same_term(&got, &want, 1e-5, &what);
            } else {
                assert!(got.0[0].is_nan() && got.0[1].is_nan() && got.0[2].is_nan(), "{what}: fused NaN");
            }
        }
    }
}

/// Box logits at the edge of f32 (one anchor, side 0 unless said): finite or not as the tensor path, and equal to it
/// where finite.
/// - A teacher side at 2e38 overflows `z / T` for `T` = 0.5 (NaN).
/// - A teacher side of −3.2e38 is a finite side whose maximum is below −3e38 (finite).
/// - A teacher bin of −2e38 at `T` 0.5 has `z / T` = −inf and counts as `s~` = 0 (finite).
/// - Student bins of 2e38 and −2e38 overflow `z − max z` but not `z / T` for `T` = 4 (finite, the term about 1e21
///   with a teacher bin of −160 under the −2e38). At `T` = 1 they overflow `z / T − max (z / T)`: the tensor path's
///   element is then `0 · inf` (a teacher bin of −inf) or inf (a finite one), so the term is non-finite.
/// - All four sides with student bins of 0 and −2e38 and teacher bins of −200 and 0, and teacher class logits of −50
///   (`q` about 2e-22), at `T` = 1: the elements are about 1e37 before `q` and finite after it (the sum of the four
///   unweighted sides overflows, so `q` must weigh each element before the sum).
/// - Teacher bins of 2e38 and −2e38 at `T` = 8e37 (`dfl_gain` 1.2e-38): `z − max z` overflows while `z / T` spans
///   only 5. Compared at 1e-3: the other anchors' bins are all about 0 there, and the two paths' different
///   logarithms of 1/16 add up (the first version was 6% off).
///
/// Not covered: logits that make the term itself approach f32::MAX (the same student side with a teacher that
/// weighs the −2e38 bin), where either path can overflow in its own order of operations.
#[test]
fn fused_term_follows_the_tensor_path_at_the_edge_of_f32() {
    let (dev, _) = gpu!();
    let (b, nc, a) = (1, 3, 64);
    let (sb, sc) = raw(61, b, nc, a);
    let (tb, tc) = raw(62, b, nc, a);
    type Edits = Vec<(usize, f32)>;
    let side0 = |k: usize, f: &dyn Fn(usize) -> f32| -> Edits { (0..R).map(|r| (r * a + k, f(r))).collect() };
    let sides =
        |k: usize, f: &dyn Fn(usize) -> f32| -> Edits { (0..4 * R).map(|j| (j * a + k, f(j % R))).collect() };
    let inf = f32::INFINITY;
    // (T, dfl_gain, student box, teacher box, teacher class (index, value)s, finite, rel)
    type Case = (f32, f32, Edits, Edits, Edits, bool, f32);
    let cases: Vec<Case> = vec![
        (0.5, 1.5, vec![], side0(5, &|_| 2e38), vec![], false, 1e-5),
        (4.0, 1.5, vec![], side0(5, &|_| -3.2e38), vec![], true, 1e-5),
        (4.0, 1.5, vec![], side0(5, &|r| if r == 0 { 0.0 } else { -3.2e38 }), vec![], true, 1e-5),
        (0.5, 1.5, vec![], side0(5, &|r| if r == 0 { 0.0 } else { -2e38 }), vec![], true, 1e-5),
        (4.0, 1.5, vec![(5, 2e38), (a + 5, -2e38)], vec![(5, 0.0), (a + 5, -160.0)], vec![], true, 1e-5),
        (1.0, 1.5, vec![(5, 2e38), (a + 5, -2e38)], vec![(5, 0.0), (a + 5, -inf)], vec![], false, 1e-5),
        (1.0, 1.5, vec![(5, 2e38), (a + 5, -2e38)], vec![(5, 0.0), (a + 5, 0.0)], vec![], false, 1e-5),
        (
            1.0,
            1.5,
            sides(5, &|r| if r == 0 { 0.0 } else { -2e38 }),
            sides(5, &|r| if r == 0 { -200.0 } else { 0.0 }),
            (0..nc).map(|c| (c * a + 5, -50.0)).collect(),
            true,
            1e-5,
        ),
        (8e37, 1.2e-38, vec![], vec![(5, 2e38), (a + 5, -2e38)], vec![], true, 1e-3),
    ];
    for (temp, dfl_gain, s_edits, t_edits, c_edits, finite, rel) in cases {
        let edit = |t: &Tensor, e: &Edits| {
            let mut v = t.to_vec::<f32>().unwrap();
            for &(i, x) in e {
                v[i] = x;
            }
            Tensor::from_vec(v, t.dims()).unwrap()
        };
        let (sb, tb, tc) = (edit(&sb, &s_edits), edit(&tb, &t_edits), edit(&tc, &c_edits));
        let cfg = DistillConfig { temperature: temp, dfl_gain, ..DistillConfig::new("t") };
        let what = format!("T {temp}, student {:?}, teacher {:?}", s_edits.get(1), t_edits.get(1));
        let want = term(dev, false, [&sb, &sc], [&tb, &tc], &cfg);
        let got = term(dev, true, [&sb, &sc], [&tb, &tc], &cfg);
        let is_finite = |v: &[f32; 3]| v[0].is_finite() && v[1].is_finite();
        assert_eq!(is_finite(&want.0), finite, "{what}: tensor path {:?}", want.0);
        assert_eq!(is_finite(&got.0), finite, "{what}: fused {:?}", got.0);
        if finite {
            same_term(&got, &want, rel, &what);
        }
    }
}

/// The backward's scales: `B · cls_gain` overflows f32 for 4 images at `cls_gain` 1e38 while `B · cls_gain / N`
/// and the term do not (student class logits 0, the teacher's 1); the gradients stay finite and equal the tensor
/// path's (which divides first).
#[test]
fn fused_backward_scales_do_not_overflow_before_dividing() {
    let (dev, _) = gpu!();
    let (b, nc, a) = (4, 1, 8);
    let (sb, _) = raw(71, b, nc, a);
    let (tb, _) = raw(72, b, nc, a);
    let sc = Tensor::from_vec(vec![0.0f32; b * nc * a], [b, nc, a]).unwrap();
    let tc = Tensor::from_vec(vec![1.0f32; b * nc * a], [b, nc, a]).unwrap();
    let cfg = DistillConfig { cls_gain: 1e38, ..DistillConfig::new("t") };
    let want = term(dev, false, [&sb, &sc], [&tb, &tc], &cfg);
    let got = term(dev, true, [&sb, &sc], [&tb, &tc], &cfg);
    assert!(want.0.iter().all(|v| v.is_finite()) && !want.1.is_empty(), "tensor path {:?}", want.0);
    same_term(&got, &want, 1e-5, "cls_gain 1e38");
}

/// The backward: the gradients scale with the upstream gradient (f32 or f16: −2.5 times those of a backward from
/// 1); a teacher output overwritten in place after the forward makes the backward an error (it reads the teacher's
/// outputs again).
#[test]
fn fused_backward_takes_an_f16_seed_and_refuses_a_modified_teacher() {
    let (dev, _) = gpu!();
    let cfg = DistillConfig::new("t");
    let (sb, sc) = raw(41, 2, 3, 200);
    let (tb, tc) = raw(42, 2, 3, 200);
    let on = |t: &Tensor| t.to_device(dev).unwrap();
    let (z, x) = (on(&sb).detach().requires_grad_(), on(&sc).detach().requires_grad_());
    let (tb, tc) = (on(&tb), on(&tc));
    let out = distill_loss_fused(&z, &x, &tb, &tc, 7.5, &cfg).unwrap();
    let seed = |dtype| Tensor::full_with([0usize; 0], -2.5, dtype, dev).unwrap();
    let g1 = out.total.backward().unwrap();
    let g32 = out.total.backward_with(&seed(DType::F32)).unwrap();
    let g16 = out.total.backward_with(&seed(DType::F16)).unwrap();
    for v in [&x, &z] {
        let get = |g: &folo_core::autograd::Grads| g.get(v).unwrap().to_vec::<f32>().unwrap();
        let want: Vec<f32> = get(&g1).iter().map(|g| -2.5 * g).collect();
        close_rel(&get(&g32), &want, 1e-6, "f32 seed");
        close_rel(&get(&g16), &want, 1e-6, "f16 seed");
    }
    tc.copy_from(&tc.add_scalar(1.0).unwrap()).unwrap();
    assert!(out.total.backward().is_err(), "a backward after the teacher's class logits changed");
    let out = distill_loss_fused(&z, &x, &tb, &tc, 7.5, &cfg).unwrap();
    tb.copy_from(&tb.add_scalar(1.0).unwrap()).unwrap();
    assert!(out.total.backward().is_err(), "a backward after the teacher's box logits changed");
}

/// More image·class rows than one dispatch dimension takes (256 · 256 = 65536): the class kernel's grid is folded.
#[test]
fn fused_term_takes_more_rows_than_one_dispatch_dimension() {
    let (dev, _) = gpu!();
    let cfg = DistillConfig::new("t");
    let (b, nc, a) = (256, 256, 84);
    let (sb, sc) = raw(51, b, nc, a);
    let (tb, tc) = raw(52, b, nc, a);
    assert!(fused_supported(&sc.to_device(dev).unwrap(), R));
    let want = term(dev, false, [&sb, &sc], [&tb, &tc], &cfg);
    let got = term(dev, true, [&sb, &sc], [&tb, &tc], &cfg);
    same_term(&got, &want, 1e-5, "256 x 256 x 84");
}

/// A shape mismatch, a CPU tensor, an f16 student, a teacher with one f16 and one f32 head and a `reg_max` whose
/// `4 · reg_max` overflows are errors.
#[test]
fn fused_term_rejects_bad_inputs() {
    let (dev, _) = gpu!();
    let cfg = DistillConfig::new("t");
    let (sb, sc) = raw(21, 2, 3, 200);
    let (tb, tc) = raw(22, 2, 3, 200);
    let on = |t: &Tensor| t.to_device(dev).unwrap();
    assert!(distill_loss_fused(&on(&sb), &on(&sc), &on(&tb), &on(&tc), 2.0, &cfg).unwrap().is_finite());
    let (tb2, _) = raw(22, 2, 3, 199);
    assert!(distill_loss_fused(&on(&sb), &on(&sc), &on(&tb2), &on(&tc), 2.0, &cfg).is_err());
    assert!(distill_loss_fused(&sb, &sc, &tb, &tc, 2.0, &cfg).is_err());
    let half = |t: &Tensor| on(&t.to_dtype(DType::F16).unwrap());
    assert!(distill_loss_fused(&on(&sb), &on(&sc), &half(&tb), &on(&tc), 2.0, &cfg).is_err());
    assert!(distill_loss_fused(&on(&sb), &half(&sc), &on(&tb), &on(&tc), 2.0, &cfg).is_err());
    assert!(distill_loss_fused(&on(&sb), &on(&sc), &half(&tb), &half(&tc), 2.0, &cfg).is_ok());
    let s = folo_gpu::distill::KdSettings {
        reg_max: usize::MAX,
        cls_gain: 0.5,
        dfl_gain: 0.5,
        temperature: 2.0,
        normaliser: 1.0,
    };
    assert!(
        folo_gpu::distill::distill_loss(&on(&sb), &on(&sc), &on(&tb), &on(&tc), &s).is_err(),
        "reg_max overflow"
    );
}

fn data() -> TrainData {
    let cfg = SyntheticConfig {
        images: 20,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 128,
        seed: 5,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 4.0 / 20.0,
        ..SyntheticConfig::default()
    };
    let split = |s| Arc::new(SyntheticDataset::new(cfg.clone(), s).unwrap());
    TrainData { train: split(Split::Train), val: split(Split::Val), classes: cfg.classes.clone() }
}

/// A mixed-precision run on the GPU with an f16 teacher: it loads the teacher in f16, trains with
/// finite losses and weights, and reports both terms.
#[test]
fn amp_run_with_an_f16_teacher() {
    let (_, tc) = gpu!();
    if !tc {
        return eprintln!("note: skipping the mixed-precision run: no tensor cores");
    }
    let dir = TestDir::new("gpu-distill");
    let config = folo_nn::io::ModelConfig::new("n", 2).unwrap();
    let mut m = folo_vision::FoloModel::with_rng(config, &mut Rng::seed(7)).unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    let teacher = dir.join("teacher.folo");
    m.save(&teacher).unwrap();
    // gpu_amp.rs's tiny run (16 images at 128 px, batch 4), one epoch.
    let cfg = TrainConfig {
        epochs: 1,
        batch: 4,
        nbs: 4,
        imgsz: 128,
        device: "gpu:0".into(),
        amp: true,
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        distill: Some(DistillConfig::new(teacher.to_string_lossy())),
        ..TrainConfig::default()
    };
    let log: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = log.clone();
    let opts = TrainOptions {
        events: Some(Box::new(move |e| {
            if let folo_api::RunEvent::Log(l) = e {
                sink.lock().unwrap().push(l.message.clone());
            }
        })),
        ..TrainOptions::default()
    };
    let mut t = Trainer::new(cfg, data(), opts).unwrap();
    let s = t.run().unwrap();
    assert!(s.epochs.iter().all(|e| e.train_loss.total.is_finite()));
    let w: Vec<Vec<f32>> =
        t.model().named_tensors().iter().filter_map(|n| n.tensor.to_vec::<f32>().ok()).collect();
    let log = log.lock().unwrap().clone();
    assert!(log.iter().any(|l| l.contains("distilling from") && l.contains("f16")), "{log:?}");
    assert!(log.iter().any(|l| l.contains("distillation cls")), "{log:?}");
    assert!(!log.iter().any(|l| l.contains("non-finite")), "{log:?}");
    assert!(w.iter().flatten().all(|v| v.is_finite()));
}
