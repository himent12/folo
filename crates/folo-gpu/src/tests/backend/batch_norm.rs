//! Train-mode `BatchNorm2d` + fused activation parity tests across all dispatch tiers, zero-copy channel-slice
//! views, subgroup vs portable fallbacks, `F32`/`F16`, and central finite-difference gradient checks.

use folo_core::backend::{Backend, UnaryOp, View};
use folo_core::{DType, Layout, Shape, Tensor, no_grad};

use super::fixtures::{assert_close_t, dev, randn};
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

struct BnRefOut {
    y: Vec<f32>,
    mean: Vec<f32>,
    inv: Vec<f32>,
    dx: Vec<f32>,
    dw: Vec<f32>,
    db: Vec<f32>,
}

/// CPU f64 reference for train-mode BatchNorm2d + optional activation (forward and backward).
fn bn_cpu_ref(
    x: &[f32],
    g: &[f32],
    w: &[f32],
    b: &[f32],
    dims: [usize; 3],
    eps: f64,
    act: Option<UnaryOp>,
) -> BnRefOut {
    let [n, c, hw] = dims;
    let m_f = (n * hw) as f64;
    let total = n * c * hw;
    let mut mean_out = vec![0.0f32; c];
    let mut inv_out = vec![0.0f32; c];
    let mut y_out = vec![0.0f32; total];
    let mut dx_out = vec![0.0f32; total];
    let mut dw_out = vec![0.0f32; c];
    let mut db_out = vec![0.0f32; c];
    for ch in 0..c {
        let mut sum = 0.0f64;
        for bi in 0..n {
            let off = (bi * c + ch) * hw;
            for i in 0..hw {
                sum += x[off + i] as f64;
            }
        }
        let mean = sum / m_f;
        let mut sq = 0.0f64;
        for bi in 0..n {
            let off = (bi * c + ch) * hw;
            for i in 0..hw {
                let d = x[off + i] as f64 - mean;
                sq += d * d;
            }
        }
        let inv = 1.0 / (sq / m_f + eps).sqrt();
        mean_out[ch] = mean as f32;
        inv_out[ch] = inv as f32;
        let (wc, bc) = (w[ch] as f64, b[ch] as f64);
        let (mut db, mut dw) = (0.0f64, 0.0f64);
        for bi in 0..n {
            let off = (bi * c + ch) * hw;
            for i in 0..hw {
                let xh = (x[off + i] as f64 - mean) * inv;
                let z = xh * wc + bc;
                let (y, da) = match act {
                    Some(UnaryOp::Silu) => {
                        let s = 1.0 / (1.0 + (-z).exp());
                        (z * s, s * (1.0 + z * (1.0 - s)))
                    }
                    Some(UnaryOp::Relu) => (z.max(0.0), if z > 0.0 { 1.0 } else { 0.0 }),
                    _ => (z, 1.0),
                };
                y_out[off + i] = y as f32;
                let dz = g[off + i] as f64 * da;
                db += dz;
                dw += dz * xh;
            }
        }
        db_out[ch] = db as f32;
        dw_out[ch] = dw as f32;
        let (db_n, dw_n, sc) = (db / m_f, dw / m_f, wc * inv);
        for bi in 0..n {
            let off = (bi * c + ch) * hw;
            for i in 0..hw {
                let xh = (x[off + i] as f64 - mean) * inv;
                let z = xh * wc + bc;
                let da = match act {
                    Some(UnaryOp::Silu) => {
                        let s = 1.0 / (1.0 + (-z).exp());
                        s * (1.0 + z * (1.0 - s))
                    }
                    Some(UnaryOp::Relu) => {
                        if z > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 1.0,
                };
                let dz = g[off + i] as f64 * da;
                dx_out[off + i] = (sc * (dz - db_n - xh * dw_n)) as f32;
            }
        }
    }
    BnRefOut { y: y_out, mean: mean_out, inv: inv_out, dx: dx_out, dw: dw_out, db: db_out }
}

#[test]
fn batch_norm_tiers_strided_and_finite_diff() {
    let dev = dev!();
    let has_f16 = folo_core::backend::computes_f16(dev);
    let _ng = no_grad();
    let eps = 1e-3f64;

    // 1. Dense + channel-slice (`narrow(1, ...)`) parity across every dispatch tier:
    //    - WG=256 REGS=4 (small VEC4 & odd spatial)
    //    - WG=512 REGS=4 bwd / WG=256 REGS=8 fwd (`1024 < m_u <= 2048, C <= 256`)
    //    - WG=256 REGS=8 both (`1024 < m_u <= 2048, C > 256`)
    //    - WG=512 REGS=0 streaming medium (`C = 32`, `16x32x40x40`) and large (`C = 64`, `16x64x80x80`)
    //    - 3-dispatch `(C, chunks)` + 2x/thread `GRID3D` (`C < 32`, both 4-aligned and odd `hw` tail)
    struct BnCase {
        shape: [usize; 4],
        pad_c: usize,
        act: Option<UnaryOp>,
        test_portable: bool,
        label: &'static str,
    }
    let cases = [
        BnCase {
            shape: [3, 17, 7, 11],
            pad_c: 5,
            act: Some(UnaryOp::Silu),
            test_portable: false,
            label: "odd small WG256 REGS4",
        },
        BnCase {
            shape: [16, 64, 20, 20],
            pad_c: 16,
            act: Some(UnaryOp::Silu),
            test_portable: true,
            label: "WG512 REGS4 bwd",
        },
        BnCase {
            shape: [8, 288, 24, 24],
            pad_c: 32,
            act: Some(UnaryOp::Relu),
            test_portable: false,
            label: "C>256 WG256 REGS8",
        },
        BnCase {
            shape: [16, 32, 40, 40],
            pad_c: 16,
            act: Some(UnaryOp::Silu),
            test_portable: true,
            label: "streaming med C=32",
        },
        BnCase {
            shape: [16, 64, 80, 80],
            pad_c: 0,
            act: Some(UnaryOp::Silu),
            test_portable: false,
            label: "streaming large C=64",
        },
        BnCase {
            shape: [16, 16, 40, 40],
            pad_c: 8,
            act: Some(UnaryOp::Silu),
            test_portable: true,
            label: "3-dispatch GRID3D vec4",
        },
        BnCase {
            shape: [4, 16, 25, 29],
            pad_c: 4,
            act: None,
            test_portable: true,
            label: "3-dispatch GRID3D odd tail",
        },
    ];
    for (idx, case) in cases.iter().enumerate() {
        let BnCase { shape, pad_c, act, test_portable, label } = *case;
        let [n, c, h, w_dim] = shape;
        let hw = h * w_dim;
        let c_parent = c + pad_c * 2;
        let x_parent = randn(&[n, c_parent, h, w_dim], 900 + idx as u64 * 4);
        let g_parent = randn(&[n, c_parent, h, w_dim], 901 + idx as u64 * 4);
        let w_cpu = randn(&[c], 902 + idx as u64 * 4).add_scalar(1.0).unwrap();
        let b_cpu = randn(&[c], 903 + idx as u64 * 4);
        let x_slice_cpu = if pad_c > 0 {
            x_parent.narrow(1, pad_c, c).unwrap().contiguous().unwrap()
        } else {
            x_parent.clone()
        };
        let g_slice_cpu = if pad_c > 0 {
            g_parent.narrow(1, pad_c, c).unwrap().contiguous().unwrap()
        } else {
            g_parent.clone()
        };
        let BnRefOut { y: ref_y, mean: ref_m, inv: ref_inv, dx: ref_dx, dw: ref_dw, db: ref_db } = bn_cpu_ref(
            &x_slice_cpu.to_vec::<f32>().unwrap(),
            &g_slice_cpu.to_vec::<f32>().unwrap(),
            &w_cpu.to_vec::<f32>().unwrap(),
            &b_cpu.to_vec::<f32>().unwrap(),
            [n, c, hw],
            eps,
            act,
        );
        let w_g = w_cpu.to_device(dev).unwrap();
        let b_g = b_cpu.to_device(dev).unwrap();
        let x_p_g = x_parent.to_device(dev).unwrap();
        let g_p_g = g_parent.to_device(dev).unwrap();
        let x_g = if pad_c > 0 { x_p_g.narrow(1, pad_c, c).unwrap() } else { x_p_g.clone() };
        let g_g = if pad_c > 0 { g_p_g.narrow(1, pad_c, c).unwrap() } else { g_p_g.clone() };

        let modes: &[bool] = if test_portable { &[false, true] } else { &[false] };
        for &portable in modes {
            let _fg = ForceGuard::set(portable);
            let out = x_g.batch_norm_act_train(&w_g, &b_g, eps, act, None).unwrap().unwrap();
            let (dx, dw, db) = x_g
                .batch_norm_act_train_backward(&g_g, &w_g, &b_g, &out.mean, &out.invstd, act)
                .unwrap()
                .unwrap();
            let tag = format!("bn {label} strided={} portable={portable}", pad_c > 0);
            let stat_tol = if n * hw > 50_000 { 5e-3 } else { 2e-3 };
            assert_close_t(
                &out.mean,
                &Tensor::from_vec(ref_m.clone(), [c]).unwrap(),
                2e-4,
                2e-4,
                &format!("{tag} m"),
            );
            assert_close_t(
                &out.invstd,
                &Tensor::from_vec(ref_inv.clone(), [c]).unwrap(),
                2e-4,
                2e-4,
                &format!("{tag} inv"),
            );
            assert_close_t(
                &out.y,
                &Tensor::from_vec(ref_y.clone(), shape).unwrap(),
                3e-4,
                3e-4,
                &format!("{tag} y"),
            );
            assert_close_t(
                &dx,
                &Tensor::from_vec(ref_dx.clone(), shape).unwrap(),
                5e-4,
                5e-4,
                &format!("{tag} dx"),
            );
            assert_close_t(
                &dw,
                &Tensor::from_vec(ref_dw.clone(), [c]).unwrap(),
                stat_tol,
                stat_tol,
                &format!("{tag} dw"),
            );
            assert_close_t(
                &db,
                &Tensor::from_vec(ref_db.clone(), [c]).unwrap(),
                stat_tol,
                stat_tol,
                &format!("{tag} db"),
            );

            if has_f16 && !portable && label != "C>256 WG256 REGS8" {
                let x16_cpu = x_slice_cpu.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
                let g16_cpu = g_slice_cpu.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
                let ref16 = bn_cpu_ref(
                    &x16_cpu.to_vec::<f32>().unwrap(),
                    &g16_cpu.to_vec::<f32>().unwrap(),
                    &w_cpu.to_vec::<f32>().unwrap(),
                    &b_cpu.to_vec::<f32>().unwrap(),
                    [n, c, hw],
                    eps,
                    act,
                );
                let x16 = x_p_g.to_dtype(DType::F16).unwrap();
                let g16 = g_p_g.to_dtype(DType::F16).unwrap();
                let x16_v = if pad_c > 0 { x16.narrow(1, pad_c, c).unwrap() } else { x16 };
                let g16_v = if pad_c > 0 { g16.narrow(1, pad_c, c).unwrap() } else { g16 };
                let o16 = x16_v.batch_norm_act_train(&w_g, &b_g, eps, act, None).unwrap().unwrap();
                let (dx16, dw16, db16) = x16_v
                    .batch_norm_act_train_backward(&g16_v, &w_g, &b_g, &o16.mean, &o16.invstd, act)
                    .unwrap()
                    .unwrap();
                assert_eq!(o16.y.dtype(), DType::F16);
                assert_eq!(dx16.dtype(), DType::F16);
                assert_close_t(
                    &o16.mean,
                    &Tensor::from_vec(ref16.mean, [c]).unwrap(),
                    5e-4,
                    5e-4,
                    &format!("{tag} f16 m"),
                );
                assert_close_t(
                    &o16.invstd,
                    &Tensor::from_vec(ref16.inv, [c]).unwrap(),
                    5e-4,
                    5e-4,
                    &format!("{tag} f16 inv"),
                );
                assert_close_t(
                    &o16.y.to_dtype(DType::F32).unwrap(),
                    &Tensor::from_vec(ref16.y, shape).unwrap(),
                    4e-3,
                    4e-3,
                    &format!("{tag} f16 y"),
                );
                assert_close_t(
                    &dx16.to_dtype(DType::F32).unwrap(),
                    &Tensor::from_vec(ref16.dx, shape).unwrap(),
                    5e-3,
                    5e-3,
                    &format!("{tag} f16 dx"),
                );
                assert_close_t(
                    &dw16,
                    &Tensor::from_vec(ref16.dw, [c]).unwrap(),
                    stat_tol,
                    stat_tol,
                    &format!("{tag} f16 dw"),
                );
                assert_close_t(
                    &db16,
                    &Tensor::from_vec(ref16.db, [c]).unwrap(),
                    stat_tol,
                    stat_tol,
                    &format!("{tag} f16 db"),
                );
            }
        }
    }

    // 2. Trace check on a fresh context: channel-slice (`narrow(1, ...)`) views across fused register-cached,
    //    streaming, and 3-dispatch tiers must run zero-copy with NO `move_vec4` or `gather_strided` pre-copy.
    let fresh = crate::GpuContext::new().unwrap();
    let backend = crate::GpuBackend::new(fresh.clone(), dev);
    let fused_bwd = if fresh.caps().max_storage_buffers >= 9 {
        vec!["bn_fused_bwd"]
    } else {
        vec!["bn_bwd_reduce", "bn_bwd_finalize", "bn_bwd_dx"]
    };
    for ([n, c_parent, c_off, c, h, w_dim], exp_fwd, exp_bwd) in [
        ([8usize, 64, 16, 32, 8, 8], vec!["bn_fused_fwd"], fused_bwd.clone()),
        ([16, 64, 16, 32, 40, 40], vec!["bn_fused_fwd"], fused_bwd),
        (
            [16, 32, 8, 16, 40, 40],
            vec!["bn_stats", "bn_finalize", "bn_apply"],
            vec!["bn_bwd_reduce", "bn_bwd_finalize", "bn_bwd_dx"],
        ),
    ] {
        let hw = h * w_dim;
        let parent = backend.full(dev, DType::F32, n * c_parent * hw, 1.0).unwrap();
        let l_slice =
            Layout::new(Shape::new(&[n, c, h, w_dim]), vec![c_parent * hw, hw, w_dim, 1].into(), c_off * hw);
        let lc = Layout::contiguous(Shape::new(&[c]));
        let w = backend.full(dev, DType::F32, c, 1.0).unwrap();
        let b = backend.full(dev, DType::F32, c, 0.0).unwrap();
        fresh.trace_begin();
        let (_, mean, inv) = backend
            .batch_norm_act(
                View::new(&parent, &l_slice),
                View::new(&w, &lc),
                View::new(&b, &lc),
                1e-3,
                Some(UnaryOp::Silu),
                None,
            )
            .unwrap();
        let fwd_ops: Vec<&str> = fresh.trace_end().iter().map(|o| o.name).collect();
        assert_eq!(fwd_ops, exp_fwd, "strided bn_fwd must avoid copy pre-pass");
        fresh.trace_begin();
        let _ = backend
            .batch_norm_act_backward(
                View::new(&parent, &l_slice),
                View::new(&parent, &l_slice),
                View::new(&w, &lc),
                View::new(&b, &lc),
                View::new(&mean, &lc),
                View::new(&inv, &lc),
                Some(UnaryOp::Silu),
            )
            .unwrap();
        let bwd_ops: Vec<&str> = fresh.trace_end().iter().map(|o| o.name).collect();
        assert_eq!(bwd_ops, exp_bwd, "strided bn_bwd must avoid copy pre-pass");
    }
    assert!(fresh.kernels_without_registry().is_empty());

    // 3. Central finite-difference gradient check on GPU (SiLU, ReLU, Identity; dense & channel-sliced views).
    let fd_h = 1e-3f32;
    let eval_loss = |x_v: &[f32], w_v: &[f32], b_v: &[f32], g_v: &[f32], shape: [usize; 4], act| -> f32 {
        let xt = Tensor::from_vec(x_v.to_vec(), shape).unwrap().to_device(dev).unwrap();
        let wt = Tensor::from_vec(w_v.to_vec(), [shape[1]]).unwrap().to_device(dev).unwrap();
        let bt = Tensor::from_vec(b_v.to_vec(), [shape[1]]).unwrap().to_device(dev).unwrap();
        let out = xt.batch_norm_act_train(&wt, &bt, eps, act, None).unwrap().unwrap();
        let y = out.y.to_vec::<f32>().unwrap();
        y.iter().zip(g_v).map(|(&yi, &gi)| yi * gi).sum()
    };
    for (act_idx, act) in [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu)].into_iter().enumerate() {
        let shape = [2usize, 3, 2, 3];
        let [n, c, h, w_dim] = shape;
        // Shift inputs away from ReLU's 0 kink so central finite differences stay smooth.
        let x_vec: Vec<f32> = randn(&shape, 950 + act_idx as u64)
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| if v.abs() < 0.25 { v.signum() * 0.5 + 0.3 } else { v })
            .collect();
        let g_vec = randn(&shape, 960 + act_idx as u64).to_vec::<f32>().unwrap();
        let w_vec = randn(&[c], 970 + act_idx as u64).add_scalar(1.2).unwrap().to_vec::<f32>().unwrap();
        let b_vec = randn(&[c], 980 + act_idx as u64).add_scalar(0.4).unwrap().to_vec::<f32>().unwrap();

        // Embed in a wider `[n, c + 2, h, w_dim]` parent and narrow to exercise strided backward in finite-diff too.
        let mut x_pad = vec![99.0f32; n * (c + 2) * h * w_dim];
        let mut g_pad = vec![-77.0f32; n * (c + 2) * h * w_dim];
        for bi in 0..n {
            for ch in 0..c {
                for i in 0..(h * w_dim) {
                    x_pad[(bi * (c + 2) + ch + 1) * h * w_dim + i] = x_vec[(bi * c + ch) * h * w_dim + i];
                    g_pad[(bi * (c + 2) + ch + 1) * h * w_dim + i] = g_vec[(bi * c + ch) * h * w_dim + i];
                }
            }
        }
        let xp_g = Tensor::from_vec(x_pad, [n, c + 2, h, w_dim]).unwrap().to_device(dev).unwrap();
        let gp_g = Tensor::from_vec(g_pad, [n, c + 2, h, w_dim]).unwrap().to_device(dev).unwrap();
        let x_g = xp_g.narrow(1, 1, c).unwrap();
        let g_g = gp_g.narrow(1, 1, c).unwrap();
        let w_g = Tensor::from_vec(w_vec.clone(), [c]).unwrap().to_device(dev).unwrap();
        let b_g = Tensor::from_vec(b_vec.clone(), [c]).unwrap().to_device(dev).unwrap();
        let out = x_g.batch_norm_act_train(&w_g, &b_g, eps, act, None).unwrap().unwrap();
        let (dx_t, dw_t, db_t) = x_g
            .batch_norm_act_train_backward(&g_g, &w_g, &b_g, &out.mean, &out.invstd, act)
            .unwrap()
            .unwrap();
        let dx = dx_t.to_vec::<f32>().unwrap();
        let dw = dw_t.to_vec::<f32>().unwrap();
        let db = db_t.to_vec::<f32>().unwrap();

        let mut x_mut = x_vec.clone();
        for i in 0..x_mut.len() {
            let orig = x_mut[i];
            x_mut[i] = orig + fd_h;
            let lp = eval_loss(&x_mut, &w_vec, &b_vec, &g_vec, shape, act);
            x_mut[i] = orig - fd_h;
            let lm = eval_loss(&x_mut, &w_vec, &b_vec, &g_vec, shape, act);
            x_mut[i] = orig;
            let fd = (lp - lm) / (2.0 * fd_h);
            assert!(
                (dx[i] - fd).abs() < 3e-3 * fd.abs().max(1.0),
                "finite-diff dx[{i}] act={act:?}: got {} vs fd {fd}",
                dx[i]
            );
        }
        let mut w_mut = w_vec.clone();
        let mut b_mut = b_vec.clone();
        for ch in 0..c {
            let ow = w_mut[ch];
            w_mut[ch] = ow + fd_h;
            let lp = eval_loss(&x_vec, &w_mut, &b_vec, &g_vec, shape, act);
            w_mut[ch] = ow - fd_h;
            let lm = eval_loss(&x_vec, &w_mut, &b_vec, &g_vec, shape, act);
            w_mut[ch] = ow;
            let fd_w = (lp - lm) / (2.0 * fd_h);
            assert!(
                (dw[ch] - fd_w).abs() < 3e-3 * fd_w.abs().max(1.0),
                "finite-diff dw[{ch}] act={act:?}: got {} vs fd {fd_w}",
                dw[ch]
            );

            let ob = b_mut[ch];
            b_mut[ch] = ob + fd_h;
            let lp = eval_loss(&x_vec, &w_vec, &b_mut, &g_vec, shape, act);
            b_mut[ch] = ob - fd_h;
            let lm = eval_loss(&x_vec, &w_vec, &b_mut, &g_vec, shape, act);
            b_mut[ch] = ob;
            let fd_b = (lp - lm) / (2.0 * fd_h);
            assert!(
                (db[ch] - fd_b).abs() < 3e-3 * fd_b.abs().max(1.0),
                "finite-diff db[{ch}] act={act:?}: got {} vs fd {fd_b}",
                db[ch]
            );
        }
    }
    let ctx = crate::GpuContext::shared().unwrap();
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
