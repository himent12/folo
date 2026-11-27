//! Unit tests for [`BatchNorm2d`], its fused CPU training kernel, and [`RunningStats`].

use folo_core::rng::Rng;
use folo_core::{DType, Tensor, no_grad};

use super::cpu::{ACT_NONE, ACT_RELU, ACT_SILU, FORCE_FALLBACK, train_cpu};
use super::{BN_EPS, BatchNorm2d, RunningStats};
use crate::layers::container::Sequential;
use crate::layers::simple::Activation;
use crate::module::{Forward, Module, TensorKind};

/// The fused CPU kernel and the composite path agree (forward, statistics and gradients).
#[test]
fn fused_matches_composite() {
    let mut rng = Rng::seed(9);
    let x = Tensor::randn([3, 4, 5, 6], 0.7, 2.0, &mut rng).requires_grad_();
    let g = Tensor::rand_uniform([4], 0.5, 1.5, &mut rng).requires_grad_();
    let b = Tensor::rand_uniform([4], -0.5, 0.5, &mut rng).requires_grad_();
    let w = Tensor::randn([3, 4, 5, 6], 0.0, 1.0, &mut rng);
    let (y1, m1, v1) = train_cpu::<ACT_NONE>(&x, &g, &b, BN_EPS).unwrap();
    let mean = x.mean([0, 2, 3], true).unwrap();
    let diff = x.sub(&mean).unwrap();
    let var = diff.square().unwrap().mean([0, 2, 3], true).unwrap();
    let inv = var.add_scalar(BN_EPS as f32).unwrap().rsqrt().unwrap();
    let scale = inv.mul(&g.reshape([1, 4, 1, 1]).unwrap()).unwrap();
    let y2 = diff.mul(&scale).unwrap().add(&b.reshape([1, 4, 1, 1]).unwrap()).unwrap();
    let near = |a: &Tensor, b: &Tensor, tol: f32| {
        let (a, b) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst <= tol, "max diff {worst}");
    };
    near(&y1, &y2, 1e-5);
    near(&m1, &mean.reshape([4]).unwrap(), 1e-5);
    near(&v1, &var.reshape([4]).unwrap(), 1e-4);
    let g1 = y1.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    let g2 = y2.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
    for t in [&x, &g, &b] {
        near(g1.get(t).unwrap(), g2.get(t).unwrap(), 1e-4);
    }
}

/// With SiLU or ReLU fused in, the CPU kernel is bit for bit the BatchNorm followed by the activation op:
/// output, running statistics and the gradients of x, gamma and beta.
#[test]
fn fused_activation_is_bitwise_bn_then_act() {
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for act in [Activation::Silu, Activation::Relu] {
        for dims in [[3, 4, 5, 6], [2, 3, 1, 13], [4, 2, 8, 8], [2, 3, 33, 17], [2, 128, 19, 17]] {
            let mut rng = Rng::seed(11);
            let x = Tensor::randn(dims, 0.3, 1.5, &mut rng).requires_grad_();
            let w = Tensor::randn(dims, 0.0, 1.0, &mut rng);
            let run = |fused: bool| {
                let bn = BatchNorm2d::new(dims[1]);
                let mut rng = Rng::seed(5);
                let _g = no_grad();
                bn.weight.copy_from(&Tensor::rand_uniform([dims[1]], 0.5, 1.5, &mut rng)).unwrap();
                bn.bias.copy_from(&Tensor::rand_uniform([dims[1]], -0.5, 0.5, &mut rng)).unwrap();
                drop(_g);
                let y = if fused {
                    bn.forward_act(&x, act).unwrap()
                } else {
                    act.forward(&bn.forward(&x).unwrap()).unwrap()
                };
                let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
                let grads = [&x, &bn.weight, &bn.bias].map(|t| bits(gr.get(t).unwrap()));
                (bits(&y), grads, bits(&bn.running_mean), bits(&bn.running_var))
            };
            assert_eq!(run(true), run(false), "{act:?} {dims:?}");
        }
    }
}

/// The one-read statistics: the mean is bit for bit the old two-pass code's (8-lane `f64` sums per plane,
/// planes added in image order, `/ n`), and the variance matches a two-pass `f64` reference to f32
/// precision, also where `Σv² − n·mean²` cancels: a mean 10⁴ or 10⁷ times the standard deviation, a
/// constant channel, single-element planes, one outlier per plane, an extreme first value, a sum that
/// needs every term. An infinity or a NaN gives a NaN variance, as before.
#[test]
fn one_pass_statistics_match_two_pass_reference() {
    type Case = ([usize; 4], fn(usize) -> f32);
    let cases: [Case; 10] = [
        ([3, 2, 7, 9], |i| 1.0e4 + ((i * 7919) % 101) as f32 * 1e-2),
        // f32 has 1 ulp at 10⁷: integers around 10⁷, spread under 1.
        ([2, 2, 16, 20], |i| 1.0e7 + ((i * 7919) % 3) as f32),
        ([2, 3, 4, 4], |i| if i % 48 < 16 { 3.25 } else { ((i * 31) % 17) as f32 - 8.0 }),
        ([5, 2, 1, 1], |i| (i as f32 * 0.37).sin() * 100.0),
        ([2, 2, 30, 30], |i| if i % 900 == 0 { 1.0e6 } else { (i % 13) as f32 * 0.1 }),
        ([1, 1, 200, 200], |i| if i == 0 { 1.0e6 } else { 0.1 }),
        ([1, 1, 2, 2], |i| [1.0e16, -1.0e16, 1.0, 1.0][i]),
        ([1, 1, 1, 2], |i| [0.0, f32::INFINITY][i]),
        ([1, 1, 1, 2], |i| [0.0, f32::NAN][i]),
        ([2, 1, 3, 100], |i| if i == 150 { f32::INFINITY } else { i as f32 }),
    ];
    for (dims, f) in cases {
        let [nb, c, h, w] = dims;
        let hw = h * w;
        let data: Vec<f32> = (0..nb * c * hw).map(f).collect();
        let x = Tensor::from_vec(data.clone(), dims).unwrap();
        let (g, b) = (Tensor::ones([c]), Tensor::zeros([c]));
        let (_, mean, var) = train_cpu::<ACT_NONE>(&x, &g, &b, BN_EPS).unwrap();
        let (mean, var) = (mean.to_vec::<f32>().unwrap(), var.to_vec::<f32>().unwrap());
        for ch in 0..c {
            let planes = || (0..nb).map(|i| &data[(i * c + ch) * hw..(i * c + ch + 1) * hw]);
            // The old mean, as origin/main computed it.
            let old_sum = |p: &[f32]| {
                let mut acc = [0f64; 8];
                let (chunks, rest) = p.as_chunks::<8>();
                for chunk in chunks {
                    for i in 0..8 {
                        acc[i] += chunk[i] as f64;
                    }
                }
                let tail: f64 = rest.iter().map(|&v| v as f64).sum();
                acc.iter().sum::<f64>() + tail
            };
            let n = (nb * hw) as f64;
            let old_mean = (planes().map(old_sum).fold(0.0, |a, s| a + s) / n) as f32;
            assert_eq!(
                mean[ch].to_bits(),
                old_mean.to_bits(),
                "{dims:?} ch {ch}: mean {} vs {old_mean}",
                mean[ch]
            );
            // Two-pass f64 reference of the variance.
            let vals: Vec<f64> = planes().flatten().map(|&v| v as f64).collect();
            let m = vals.iter().sum::<f64>() / n;
            let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
            let close =
                |got: f32, want: f64| (got as f64 - want).abs() <= 2.0 * f32::EPSILON as f64 * want.abs();
            assert!(
                close(var[ch], v) || (v == 0.0 && var[ch] == 0.0) || (v.is_nan() && var[ch].is_nan()),
                "{dims:?} ch {ch}: var {} vs {v}",
                var[ch]
            );
            if dims == [2, 2, 16, 20] {
                // The case bites: the naive formula is far off here.
                let naive = vals.iter().map(|x| x * x).sum::<f64>() / n - m * m;
                assert!(!close(naive as f32, v), "naive {naive} vs {v}");
            }
        }
    }
    // The values the review pinned: an extreme first value, and a mean that needs the small terms.
    let stats = |dims: [usize; 4], data: Vec<f32>| {
        let x = Tensor::from_vec(data, dims).unwrap();
        let (_, m, v) = train_cpu::<ACT_NONE>(&x, &Tensor::ones([1]), &Tensor::zeros([1]), BN_EPS).unwrap();
        (m.to_vec::<f32>().unwrap()[0], v.to_vec::<f32>().unwrap()[0])
    };
    // At this size the first draft (sums shifted by the plane's first value) was 132 f32 ulp off.
    let mut data = vec![0.1f32; 1000 * 1000];
    data[0] = 1.0e6;
    let (n, k, r) = (1.0e6f64, 1.0e6f64, 0.1f32 as f64);
    let want = (k - r) * (k - r) * (n - 1.0) / (n * n);
    let (_, v) = stats([1, 1, 1000, 1000], data);
    assert!((v as f64 - want).abs() <= f32::EPSILON as f64 * want, "var {v} vs {want}");
    assert_eq!(stats([1, 1, 2, 2], vec![1.0e16, -1.0e16, 1.0, 1.0]).0, 0.5);
    let (m, v) = stats([1, 1, 1, 2], vec![0.0, f32::INFINITY]);
    assert!(m == f32::INFINITY && v.is_nan(), "mean {m} var {v}");
    let (m, v) = stats([1, 1, 1, 2], vec![0.0, f32::NAN]);
    assert!(m.is_nan() && v.is_nan(), "mean {m} var {v}");
}

/// The AVX2 build of the per-plane loops and the plain one give the same bits: output, statistics and
/// gradients of x, gamma and beta, with SiLU fused in. Planes of one value, of whole lanes, and of
/// several statistics blocks with a ragged tail.
#[test]
fn avx2_and_fallback_are_bitwise_equal() {
    use std::sync::atomic::Ordering::SeqCst;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_FALLBACK.store(false, SeqCst);
        }
    }
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for dims in [[3, 2, 1, 1], [2, 3, 5, 7], [2, 2, 16, 16], [1, 2, 17, 19], [2, 3, 33, 17], [2, 128, 19, 17]]
    {
        let mut rng = Rng::seed(13);
        let x = Tensor::randn(dims, 0.3, 1.5, &mut rng).requires_grad_();
        let gamma = Tensor::rand_uniform([dims[1]], 0.5, 1.5, &mut rng).requires_grad_();
        let beta = Tensor::rand_uniform([dims[1]], -0.5, 0.5, &mut rng).requires_grad_();
        let w = Tensor::randn(dims, 0.0, 1.0, &mut rng);
        let run = |fallback: bool| {
            let _reset = Reset;
            FORCE_FALLBACK.store(fallback, SeqCst);
            let (y, m, v) = train_cpu::<ACT_SILU>(&x, &gamma, &beta, BN_EPS).unwrap();
            let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
            let grads = [&x, &gamma, &beta].map(|t| bits(gr.get(t).unwrap()));
            (bits(&y), bits(&m), bits(&v), grads)
        };
        assert_eq!(run(false), run(true), "{dims:?}");
    }
}

/// Channel-parallel (`c >= 4 * threads`) and plane-parallel (`c < 4 * threads`) match bit for bit on
/// forward output, batch stats, and all three gradients across all fused activations, `nb == 1`, and
/// ragged multi-tile planes (`hw > BWD_BLOCK`, `hw % 4 != 0`).
#[test]
fn channel_par_matches_plane_par_bitwise() {
    let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let p8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    for dims in [[2, 16, 19, 17], [1, 16, 17, 15]] {
        let mut rng = Rng::seed(29);
        let x = Tensor::randn(dims, 0.2, 1.4, &mut rng).requires_grad_();
        let gamma = Tensor::rand_uniform([dims[1]], 0.5, 1.5, &mut rng).requires_grad_();
        let beta = Tensor::rand_uniform([dims[1]], -0.5, 0.5, &mut rng).requires_grad_();
        let w = Tensor::randn(dims, 0.0, 1.0, &mut rng);
        let eval = |pool: &rayon::ThreadPool, act: u8| {
            pool.install(|| {
                let (y, m, v) = match act {
                    ACT_SILU => train_cpu::<ACT_SILU>(&x, &gamma, &beta, BN_EPS).unwrap(),
                    ACT_RELU => train_cpu::<ACT_RELU>(&x, &gamma, &beta, BN_EPS).unwrap(),
                    _ => train_cpu::<ACT_NONE>(&x, &gamma, &beta, BN_EPS).unwrap(),
                };
                let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
                let grads = [&x, &gamma, &beta].map(|t| bits(gr.get(t).unwrap()));
                (bits(&y), bits(&m), bits(&v), grads)
            })
        };
        for act in [ACT_NONE, ACT_SILU, ACT_RELU] {
            assert_eq!(eval(&p1, act), eval(&p8, act), "{dims:?} act={act}");
        }
    }
}

/// Reusable `RunningStats::recapture`, `capture_training`, and `all_finite_live`: skips eval-mode BNs,
/// refreshes snapshots in place, catches NaN/+Inf/-Inf, and handles dtype changes cleanly.
#[test]
fn running_stats_recapture_and_capture_training() {
    let mut rng = Rng::seed(7);
    let bn_train = BatchNorm2d::new(4);
    let bn_eval = BatchNorm2d::new(4);
    bn_eval.eval();
    let model = Sequential::new().push_named("train_bn", bn_train).push_named("eval_bn", bn_eval);

    let full = RunningStats::capture(&model).unwrap();
    assert_eq!(full.len(), 4);
    let mut active = RunningStats::capture_training(&model).unwrap();
    assert_eq!(active.len(), 2);
    assert!(!active.is_empty());
    assert!(active.all_finite_live().unwrap());
    assert!(matches!(active.prepare_live_finite_check().unwrap(), super::LiveFiniteCheck::Ready(true)));

    // Step 1: finite forward updates only train_bn.
    let x1 = Tensor::randn([2, 4, 4, 4], 0.5, 1.5, &mut rng);
    model.forward(&x1).unwrap();
    assert!(active.all_finite_live().unwrap());
    assert!(matches!(active.prepare_live_finite_check().unwrap(), super::LiveFiniteCheck::Ready(true)));
    let after_step1: Vec<Vec<f32>> =
        model.named_buffers().iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();

    // Refresh snapshot in place, then run a non-finite batch (NaN, +Inf, -Inf) and restore.
    active.recapture().unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x_bad = Tensor::full([2, 4, 4, 4], bad);
        model.forward(&x_bad).unwrap();
        assert!(!active.all_finite_live().unwrap(), "missed {bad}");
        assert!(matches!(active.prepare_live_finite_check().unwrap(), super::LiveFiniteCheck::Ready(false)));
        assert!(!RunningStats::all_finite(&model).unwrap(), "missed {bad}");
        active.restore().unwrap();
        assert!(active.all_finite_live().unwrap());
        assert!(matches!(active.prepare_live_finite_check().unwrap(), super::LiveFiniteCheck::Ready(true)));
        let restored: Vec<Vec<f32>> =
            model.named_buffers().iter().map(|(_, t)| t.to_vec::<f32>().unwrap()).collect();
        assert_eq!(restored, after_step1, "restore after {bad}");
    }

    // When all BNs are frozen, `capture_training` is empty and all operations are no-ops.
    model.eval();
    assert!(RunningStats::all_finite_eval(&model).unwrap());
    let mut frozen = RunningStats::capture_training(&model).unwrap();
    assert!(frozen.is_empty());
    assert_eq!(frozen.len(), 0);
    frozen.recapture().unwrap();
    frozen.restore().unwrap();
    assert!(frozen.all_finite_live().unwrap());
    // A pre-existing +Inf in a frozen/eval BN buffer is caught by `all_finite_eval`.
    {
        let (_, buf) = model.named_buffers().into_iter().nth(3).unwrap();
        let _g = no_grad();
        buf.fill_assign(f32::INFINITY).unwrap();
    }
    assert!(!RunningStats::all_finite_eval(&model).unwrap());
    {
        let (_, buf) = model.named_buffers().into_iter().nth(3).unwrap();
        let _g = no_grad();
        buf.fill_assign(1.0).unwrap();
    }
    assert!(RunningStats::all_finite_eval(&model).unwrap());

    // Live buffer changing dtype between capture and recapture reallocates its slot cleanly.
    struct BufOnly(Tensor);
    impl Module for BufOnly {
        fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            f("buf", &self.0, TensorKind::Buffer);
        }
    }
    let m = BufOnly(Tensor::full([3], 1.5));
    let mut snap = RunningStats::capture(&m).unwrap();
    {
        let _g = no_grad();
        m.0.to_dtype_(DType::F16).unwrap();
        m.0.fill_assign(2.5).unwrap();
    }
    snap.recapture().unwrap();
    {
        let _g = no_grad();
        m.0.fill_assign(9.0).unwrap();
    }
    snap.restore().unwrap();
    assert_eq!(m.0.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(), [2.5; 3]);
}
