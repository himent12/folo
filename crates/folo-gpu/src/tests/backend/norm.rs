//! Fused GPU `LayerNorm`, `RMSNorm`, and `softmax_backward` parity tests against CPU (forward + backward,
//! subgroup + portable fallback, `vec4` + scalar, cached + uncached, split-K param grads, and extreme values).

use folo_core::{DType, Tensor};

use super::fixtures::{assert_close_t, check, dev, randn};
use crate::kernels::FORCE_NO_SUBGROUPS;

struct ForceGuard(bool);
impl ForceGuard {
    fn set(v: bool) -> Self {
        Self(FORCE_NO_SUBGROUPS.replace(v))
    }
}
impl Drop for ForceGuard {
    fn drop(&mut self) {
        FORCE_NO_SUBGROUPS.set(self.0);
    }
}

#[test]
fn layer_norm_and_rms_norm_match_cpu() {
    let dev = dev!();
    // Covers: d=1, odd tails (3, 17, 1025), vec4 cached (4, 64, 768), vec4 uncached (4100),
    // multi-dim [2, 3, 19], and split-K dweight/dbias (outer=300 >= 256).
    let shapes: &[(&[usize], &str)] = &[
        (&[3, 1], "d=1"),
        (&[4, 3], "d=3 scalar"),
        (&[3, 4], "d=4 vec4"),
        (&[5, 17], "d=17 scalar"),
        (&[2, 3, 19], "3d scalar"),
        (&[2, 4, 8, 16], "4d vec4"),
        (&[3, 768], "d=768 vec4 cached"),
        (&[2, 1025], "d=1025 scalar uncached"),
        (&[2, 4100], "d=4100 vec4 uncached"),
        (&[300, 64], "outer=300 split-k vec4"),
        (&[300, 17], "outer=300 split-k scalar"),
    ];
    for (idx, &(shape, label)) in shapes.iter().enumerate() {
        let d = *shape.last().unwrap();
        let x = randn(shape, 300 + idx as u64 * 3);
        let w = randn(&[d], 301 + idx as u64 * 3).add_scalar(1.0).unwrap();
        let b = randn(&[d], 302 + idx as u64 * 3);
        let tol = if d > 1000 || shape[0] >= 256 { 2e-3 } else { 2e-4 };
        for portable in [false, true] {
            let _g = ForceGuard::set(portable);
            let tag = format!("{label} portable={portable}");
            check(dev, &[x.clone(), w.clone(), b.clone()], tol, 2e-4, &format!("ln wb {tag}"), |v| {
                v[0].layer_norm(Some(&v[1]), Some(&v[2]), 1e-5)
            });
            check(dev, &[x.clone(), w.clone()], tol, 2e-4, &format!("rms w {tag}"), |v| {
                v[0].rms_norm(Some(&v[1]), 1e-5)
            });
        }
    }

    // Optional weight / bias combinations and non-contiguous transposed inputs.
    let x = randn(&[5, 12], 401);
    let w = randn(&[12], 402).add_scalar(1.0).unwrap();
    let b = randn(&[12], 403);
    check(dev, std::slice::from_ref(&x), 1e-4, 1e-4, "ln none", |v| v[0].layer_norm(None, None, 1e-5));
    check(dev, &[x.clone(), w.clone()], 1e-4, 1e-4, "ln w-only", |v| {
        v[0].layer_norm(Some(&v[1]), None, 1e-5)
    });
    check(dev, &[x.clone(), b.clone()], 1e-4, 1e-4, "ln b-only", |v| {
        v[0].layer_norm(None, Some(&v[1]), 1e-5)
    });
    check(dev, std::slice::from_ref(&x), 1e-4, 1e-4, "rms none", |v| v[0].rms_norm(None, 1e-5));
    let xt = randn(&[7, 12], 404);
    let wt = randn(&[7], 405).add_scalar(1.0).unwrap();
    let bt = randn(&[7], 406);
    check(dev, &[xt.clone(), wt.clone(), bt.clone()], 1e-4, 1e-4, "ln transposed", |v| {
        v[0].t()?.layer_norm(Some(&v[1]), Some(&v[2]), 1e-5)
    });
    check(dev, &[xt, wt], 1e-4, 1e-4, "rms transposed", |v| v[0].t()?.rms_norm(Some(&v[1]), 1e-5));

    // Selective requires_grad: only bias, only weight, only x (exercises single-param `norm_bwd_param` paths).
    for (need_w, need_b) in [(true, false), (false, true), (false, false)] {
        for rows in [4usize, 300usize] {
            let x = randn(&[rows, 16], 410 + rows as u64);
            let w = randn(&[16], 411);
            let b = randn(&[16], 412);
            let g_out = randn(&[rows, 16], 413);
            let run = |d: folo_core::Device| {
                let xd = x.to_device(d).unwrap().detach().requires_grad_();
                let mut wd = w.to_device(d).unwrap().detach();
                let mut bd = b.to_device(d).unwrap().detach();
                if need_w {
                    wd = wd.requires_grad_();
                }
                if need_b {
                    bd = bd.requires_grad_();
                }
                let y = xd.layer_norm(Some(&wd), Some(&bd), 1e-5).unwrap();
                let gr = y.backward_with(&g_out.to_device(d).unwrap()).unwrap();
                (
                    gr.get(&xd).unwrap().clone(),
                    need_w.then(|| gr.get(&wd).unwrap().clone()),
                    need_b.then(|| gr.get(&bd).unwrap().clone()),
                )
            };
            let (dxc, dwc, dbc) = run(folo_core::Device::Cpu);
            let (dxg, dwg, dbg) = run(dev);
            assert_close_t(&dxg, &dxc, 5e-4, 2e-4, &format!("ln sel dx rows={rows} {need_w}/{need_b}"));
            if let (Some(a), Some(b)) = (dwg, dwc) {
                assert_close_t(&a, &b, 1e-3, 2e-4, &format!("ln sel dw rows={rows}"));
            }
            if let (Some(a), Some(b)) = (dbg, dbc) {
                assert_close_t(&a, &b, 1e-3, 2e-4, &format!("ln sel db rows={rows}"));
            }
        }
    }
}

#[test]
fn layer_norm_and_rms_norm_edge_cases() {
    let dev = dev!();
    // 1. Empty outer batch: zero-length y/dx and zero dweight/dbias.
    for d in [6usize, 8usize] {
        let x = Tensor::zeros([0, d]).to_device(dev).unwrap().detach().requires_grad_();
        let w = Tensor::ones([d]).to_device(dev).unwrap().detach().requires_grad_();
        let b = Tensor::zeros([d]).to_device(dev).unwrap().detach().requires_grad_();
        let y = x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap();
        assert_eq!(y.dims(), &[0, d]);
        let gr = y.backward_with(&Tensor::zeros([0, d]).to_device(dev).unwrap()).unwrap();
        assert_eq!(gr.get(&w).unwrap().to_vec::<f32>().unwrap(), vec![0.0; d]);
        assert_eq!(gr.get(&b).unwrap().to_vec::<f32>().unwrap(), vec![0.0; d]);

        let yr = x.rms_norm(Some(&w), 1e-5).unwrap();
        let gr2 = yr.backward_with(&Tensor::zeros([0, d]).to_device(dev).unwrap()).unwrap();
        assert_eq!(gr2.get(&w).unwrap().to_vec::<f32>().unwrap(), vec![0.0; d]);
    }

    // 2. Large-offset translation invariance (`sum(dx) == 0` even when `mean` rounds in f32).
    let big = Tensor::from_slice(&[100_000_000.0f32, 100_000_008.0], [2]).unwrap();
    let big_g = big.to_device(dev).unwrap().detach().requires_grad_();
    let g_ln = big_g.layer_norm(None, None, 0.0).unwrap().sum_all().unwrap().backward().unwrap();
    let dx = g_ln.get(&big_g).unwrap().to_vec::<f32>().unwrap();
    assert!((dx[0] + dx[1]).abs() < 1e-6, "GPU LN translation invariance broken: {dx:?}");
    assert!((dx[0] - 0.088_388_35).abs() < 1e-5, "{dx:?}");

    // 3. Extreme finite magnitudes (`1e20`, `1e-23`, `eps = 0`) and huge constant (`1e38`) stay finite.
    for portable in [false, true] {
        let _g = ForceGuard::set(portable);
        for vals in [vec![1e20f32, -1e20], vec![1e-23f32, -1e-23], vec![1e20, -1e20, 1e20, -1e20]] {
            let len = vals.len();
            let t = Tensor::from_vec(vals, [1, len]).unwrap().to_device(dev).unwrap();
            let y_ln = t.layer_norm(None, None, 0.0).unwrap().to_vec::<f32>().unwrap();
            let y_rms = t.rms_norm(None, 0.0).unwrap().to_vec::<f32>().unwrap();
            for (i, (&a, &b)) in y_ln.iter().zip(&y_rms).enumerate() {
                let want = if i % 2 == 0 { 1.0 } else { -1.0 };
                assert!((a - want).abs() < 1e-5, "portable={portable} ln[{i}]={a} want={want}");
                assert!((b - want).abs() < 1e-5, "portable={portable} rms[{i}]={b} want={want}");
            }
        }
        // 1e38 finite sum overflow guard in LayerNorm mean.
        for val in [1e37f32, 1e38f32, -1e38f32] {
            let huge = Tensor::from_vec(vec![val; 8], [1, 8]).unwrap().to_device(dev).unwrap();
            let yh = huge.layer_norm(None, None, 1e-5).unwrap().to_vec::<f32>().unwrap();
            assert!(
                yh.iter().all(|&v| v.is_finite() && v.abs() < 1e-5),
                "portable={portable} val={val:e} huge const LN: {yh:?}"
            );
        }
    }

    // 4. NaN and +Inf propagation match CPU.
    let special =
        Tensor::from_slice(&[1.0f32, f32::NAN, 2.0, 3.0, f32::INFINITY, 1.0, 2.0, 3.0], [2, 4]).unwrap();
    let ln_c = special.layer_norm(None, None, 1e-5).unwrap();
    let ln_g = special.to_device(dev).unwrap().layer_norm(None, None, 1e-5).unwrap();
    assert_close_t(&ln_g, &ln_c, 0.0, 0.0, "layer_norm NaN/Inf");
    let rms_c = special.rms_norm(None, 1e-5).unwrap();
    let rms_g = special.to_device(dev).unwrap().rms_norm(None, 1e-5).unwrap();
    assert_close_t(&rms_g, &rms_c, 0.0, 0.0, "rms_norm NaN/Inf");
}

#[test]
fn softmax_backward_fused_paths_match_cpu() {
    let dev = dev!();
    let cases: &[(&[usize], usize, &str)] = &[
        (&[3, 1], 1, "row d=1"),
        (&[4, 3], 1, "row scalar d=3"),
        (&[4, 16], 1, "row vec4 cached d=16"),
        (&[2, 768], 1, "row vec4 cached d=768"),
        (&[2, 1025], 1, "row scalar uncached d=1025"),
        (&[2, 4100], 1, "row vec4 uncached d=4100"),
        (&[2, 16, 48], 1, "col vec4 d=16 inner=48"),
        (&[2, 16, 17], 1, "col scalar d=16 inner=17"),
        (&[2, 64, 12], 1, "col vec4 d=64 inner=12"),
        (&[2, 5, 7], 0, "col dim0 scalar"),
    ];
    for (i, &(shape, dim, label)) in cases.iter().enumerate() {
        let x = randn(shape, 500 + i as u64);
        for portable in [false, true] {
            let _g = ForceGuard::set(portable);
            for log in [false, true] {
                let f = move |v: &[Tensor]| if log { v[0].log_softmax(dim) } else { v[0].softmax(dim) };
                check(
                    dev,
                    std::slice::from_ref(&x),
                    1e-5,
                    1e-4,
                    &format!("sm_bwd {label} log={log} portable={portable}"),
                    f,
                );
            }
        }
    }

    // F16 softmax_backward kernel parity (row + col, vec4 + scalar, softmax + log_softmax).
    if folo_core::backend::computes_f16(dev) {
        let ctx = crate::GpuContext::shared().unwrap();
        for &(dims, label) in &[
            ((3usize, 16usize, 1usize), "f16 row vec4"),
            ((3, 13, 1), "f16 row scalar"),
            ((2, 16, 12), "f16 col vec4"),
            ((2, 16, 9), "f16 col scalar"),
        ] {
            let (o, d, inn) = dims;
            let n = o * d * inn;
            let x = randn(&[o, d, inn], 600 + n as u64)
                .to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let g = randn(&[o, d, inn], 601 + n as u64)
                .to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            for log in [false, true] {
                let xl = x.detach().requires_grad_();
                let y = if log { xl.log_softmax(1).unwrap() } else { xl.softmax(1).unwrap() };
                let y_f16_vals = y.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
                let ref_dx = if log {
                    g.sub(&y_f16_vals.exp().unwrap().mul(&g.sum(1, true).unwrap()).unwrap()).unwrap()
                } else {
                    y_f16_vals
                        .mul(&g.sub(&g.mul(&y_f16_vals).unwrap().sum(1, true).unwrap()).unwrap())
                        .unwrap()
                };
                let to_h = |t: &Tensor| -> Vec<half::f16> {
                    t.to_vec::<f32>().unwrap().into_iter().map(half::f16::from_f32).collect()
                };
                let gy = crate::GpuTensor::from_f16(&ctx, &to_h(&y_f16_vals), &[n]).unwrap();
                let gg = crate::GpuTensor::from_f16(&ctx, &to_h(&g), &[n]).unwrap();
                let gdx = crate::GpuTensor::empty(&ctx, &[n], crate::DType::F16).unwrap();
                crate::kernels::softmax_bwd::softmax_backward_dim(
                    &ctx,
                    gy.raw(),
                    gg.raw(),
                    gdx.raw(),
                    dims,
                    log,
                    crate::DType::F16,
                )
                .unwrap();
                let got = Tensor::from_vec(gdx.to_vec().unwrap(), [o, d, inn]).unwrap();
                assert_close_t(&got, &ref_dx, 5e-3, 5e-3, &format!("{label} log={log}"));
            }
        }
    }

    let ctx = crate::GpuContext::shared().unwrap();
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
