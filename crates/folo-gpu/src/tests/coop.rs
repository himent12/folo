//! Tensor-core (cooperative-matrix) kernels against the CPU reference.
//!
//! The tests run on their own context created with tensor cores requested, and skip (with a note) when the
//! adapter has none. The reference is the f64 CPU loop on the same f16-rounded inputs, so only the GPU's
//! arithmetic is under test. **Tolerance**, per element: f16 × f16 products are exact in f32 (11 + 11
//! significant bits), so the only accumulation error is the f32 summation, at most `K · u · Σ|a·b|` with
//! `u = 2^-23` (one ulp per addition, which also covers tensor cores that truncate instead of rounding when they
//! align addends). The activation adds a few f32 ulps of the result (SiLU's slope is below 1.1), and an f16 output
//! one rounding, `2^-11 · |y|` plus `2^-24` for subnormals. The bound is applied with a factor of 2 of slack; a
//! wrong tile, a dropped `K` step or a bad index is off by `O(|a·b|)`, orders of magnitude more.

use std::sync::{Mutex, MutexGuard, OnceLock};

use super::{Rng, f16_round, to_f16};
use crate::kernels::conv::{self, Conv2dDesc};
use crate::kernels::coop::{self, CoopConfig};
use crate::kernels::gemm::{self, BatchStrides, GemmBias, GemmDesc};
use crate::kernels::pooling::Window2d;
use crate::tests::gemm::cpu_gemm;
use crate::{DType, GpuContext, GpuOptions, GpuTensor, UnaryOp};

/// The tensor-core context plus a lock: tests flip the run-time switch, so they run one at a time. `None` (a skip)
/// only when the adapter does not qualify; on one that does, failing to get tensor cores is a test failure.
fn coop_ctx() -> Option<(GpuContext, MutexGuard<'static, ()>)> {
    static CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    static LOCK: Mutex<()> = Mutex::new(());
    let ctx = CTX
        .get_or_init(|| {
            let base = super::ctx()?;
            if let Some(why) = coop::adapter_unavailable_reason(base.adapter()) {
                eprintln!("note: skipping tensor-core tests: {why}");
                return None;
            }
            if !GpuOptions::from_env().enable_f16 {
                eprintln!("note: skipping tensor-core tests: FOLO_GPU_NO_F16 is set");
                return None;
            }
            let opts = GpuOptions { tensor_cores: true, ..GpuOptions::from_env() };
            let ctx = GpuContext::with_options(&opts).expect("a context on a qualifying adapter");
            assert!(
                ctx.tensor_cores(),
                "{} qualifies but: {}",
                base.summary().name,
                ctx.tensor_cores_status()
            );
            Some(ctx)
        })
        .clone()?;
    let guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    ctx.set_tensor_cores(true);
    Some((ctx, guard))
}

macro_rules! coop {
    () => {
        match coop_ctx() {
            Some(c) => c,
            None => return,
        }
    };
}

const U32: f64 = 1.0 / (1u64 << 23) as f64;

/// Per-element error bound (see the module docs) for an output `y` whose terms have absolute sum `s` over `k`
/// products, plus the bias magnitude `bias`.
fn bound(y: f32, s: f64, k: usize, bias: f32, out: DType) -> f64 {
    let acc = k.max(1) as f64 * U32 * s + U32 * bias.abs() as f64;
    let act = 1.1 * acc + 8.0 * U32 * y.abs() as f64;
    let round = if out == DType::F16 { y.abs() as f64 / 2048.0 + 1.0 / (1u64 << 24) as f64 } else { 0.0 };
    2.0 * (act + round) + 1e-7
}

#[track_caller]
fn assert_within(got: &[f32], want: &[f32], tol: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for i in 0..got.len() {
        let err = (got[i] as f64 - want[i] as f64).abs();
        assert!(
            err <= tol[i],
            "{what}[{i}]: got {}, want {} (error {err:e} > bound {:e})",
            got[i],
            want[i],
            tol[i]
        );
    }
}

fn upload(ctx: &GpuContext, v: &[f32], dtype: DType) -> GpuTensor {
    match dtype {
        DType::F32 => GpuTensor::from_slice(ctx, v, &[v.len()]).unwrap(),
        DType::F16 => GpuTensor::from_f16(ctx, &to_f16(v), &[v.len()]).unwrap(),
    }
}

fn operand_len(s: &BatchStrides, batch: (usize, usize), rows: usize, cols: usize, ld: usize) -> usize {
    s.offset + s.outer * (batch.0 - 1) + s.inner * (batch.1 - 1) + (rows - 1) * ld + cols
}

/// Runs `d` (f16 inputs) on tensor cores with output `out`, explicitly (`cfg`) or through `gemm::gemm`'s routing
/// (`route`), against the CPU.
fn run_gemm(ctx: &GpuContext, d: &GemmDesc, out: DType, cfg: Option<CoopConfig>, route: bool, seed: u64) {
    assert_eq!(d.dtype, DType::F16);
    let mut rng = Rng::new(seed);
    let (ar, ac) = if d.trans_a { (d.k, d.m) } else { (d.m, d.k) };
    let (br, bc) = if d.trans_b { (d.n, d.k) } else { (d.k, d.n) };
    let batches = d.batch.0 * d.batch.1;
    let a = f16_round(&rng.vec(operand_len(&d.a, d.batch, ar, ac, d.lda), -1.0, 1.0));
    let b = f16_round(&rng.vec(operand_len(&d.b, d.batch, br, bc, d.ldb), -1.0, 1.0));
    let bias_len = match d.bias {
        GemmBias::None => 1,
        GemmBias::Row => d.bias_strides.offset + d.m * batches,
        GemmBias::Col => d.bias_strides.offset + d.n * batches,
    };
    let mut bias = rng.vec(bias_len, -1.0, 1.0);
    if out == DType::F16 {
        bias = f16_round(&bias);
    }
    let want = cpu_gemm(d, &a, &b, Some(&bias));
    // Σ|a·b| per element: the same GEMM on absolute values, no bias or activation.
    let mut dabs = d.clone();
    dabs.bias = GemmBias::None;
    dabs.act = UnaryOp::Identity;
    dabs.alpha = d.alpha.abs();
    let abs = |v: &[f32]| v.iter().map(|x| x.abs()).collect::<Vec<_>>();
    let sums = cpu_gemm(&dabs, &abs(&a), &abs(&b), None);
    let bmax = bias.iter().fold(0f32, |m, x| m.max(x.abs()));
    let tol: Vec<f64> = want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, d.k, bmax, out)).collect();

    let (ga, gb, gbias) = (upload(ctx, &a, DType::F16), upload(ctx, &b, DType::F16), upload(ctx, &bias, out));
    let gc = GpuTensor::zeros(ctx, &[want.len()], out).unwrap();
    let bias_buf = (d.bias != GemmBias::None).then(|| gbias.raw());
    if route {
        assert_eq!(out, DType::F16, "gemm::gemm writes the input dtype");
        gemm::gemm(ctx, d, ga.raw(), gb.raw(), bias_buf, gc.raw()).unwrap();
        assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
    } else {
        coop::gemm(ctx, d, out, ga.raw(), gb.raw(), bias_buf, gc.raw(), cfg).unwrap();
    }
    let got = gc.to_vec().unwrap();
    // The untouched part of C (padding between batches / leading-dimension slack) must stay zero.
    let mut touched = vec![false; want.len()];
    for b1 in 0..d.batch.0 {
        for b2 in 0..d.batch.1 {
            let oc = d.c.offset + b1 * d.c.outer + b2 * d.c.inner;
            for i in 0..d.m {
                for j in 0..d.n {
                    touched[oc + i * d.ldc + j] = true;
                }
            }
        }
    }
    let want: Vec<f32> = want.iter().zip(&touched).map(|(&w, &t)| if t { w } else { 0.0 }).collect();
    assert_within(&got, &want, &tol, &format!("coop gemm {d:?} out {out:?} cfg {cfg:?} route {route}"));
}

fn f16_desc(m: usize, n: usize, k: usize, ta: bool, tb: bool, batch: usize) -> GemmDesc {
    let mut d = GemmDesc::new(m, n, k, ta, tb, batch);
    d.dtype = DType::F16;
    d
}

const CONFIGS: [CoopConfig; 5] =
    [CoopConfig::C128X128, CoopConfig::C64X128, CoopConfig::C64X64, CoopConfig::C32X128, CoopConfig::C16X256];

#[test]
fn self_test_passes_and_reports() {
    let (ctx, _g) = coop!();
    assert!(ctx.caps().coop_f16 && ctx.experimental_features());
    // The raw device of an experimental context is only reachable unsafely; a plain context hands it out.
    assert!(ctx.device().is_none());
    let plain = super::ctx().unwrap();
    if !plain.experimental_features() {
        assert!(plain.device().is_some());
    }
    assert_eq!(ctx.tensor_cores_status(), "on");
    assert!(!ctx.set_tensor_cores(false) && !ctx.tensor_cores());
    assert!(ctx.tensor_cores_status().contains("switched off"));
    assert!(ctx.set_tensor_cores(true));
    assert!(CONFIGS.iter().all(|c| c.valid() && c.workgroup_bytes() <= ctx.caps().max_workgroup_storage));
    assert!(!CoopConfig { bm: 48, bn: 128, bk: 32, wgm: 2, wgn: 4 }.valid());
    assert!(!CoopConfig { bm: 64, bn: 64, bk: 24, wgm: 4, wgn: 2 }.valid());
}

#[test]
fn coop_gemm_every_config_and_transpose() {
    let (ctx, _g) = coop!();
    // Ragged in every dimension (scalar loads), then multiples of 4 that are not tile multiples (vec4 loads).
    for (seed, (m, n, k)) in [(67, 45, 33), (132, 260, 52)].into_iter().enumerate() {
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            for cfg in CONFIGS {
                run_gemm(&ctx, &f16_desc(m, n, k, ta, tb, 1), DType::F16, Some(cfg), false, seed as u64);
            }
        }
    }
    // Exact tile multiples: no bounds checks compiled in.
    for cfg in CONFIGS {
        let (m, n) = (cfg.bm as usize * 2, cfg.bn as usize);
        run_gemm(&ctx, &f16_desc(m, n, 64, false, false, 1), DType::F16, Some(cfg), false, 3);
    }
}

#[test]
fn coop_gemm_odd_shapes_f16_and_f32_out() {
    let (ctx, _g) = coop!();
    let shapes =
        [(1, 1, 1), (3, 5, 7), (17, 33, 9), (31, 200, 27), (16, 300, 40), (130, 129, 17), (200, 70, 300)];
    for (seed, (m, n, k)) in shapes.into_iter().enumerate() {
        for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
            for out in [DType::F16, DType::F32] {
                run_gemm(&ctx, &f16_desc(m, n, k, ta, tb, 1), out, None, false, 10 + seed as u64);
            }
        }
    }
    // A long reduction (accumulation error grows with K; f32 output shows it unrounded).
    run_gemm(&ctx, &f16_desc(64, 96, 4608, false, false, 1), DType::F32, None, false, 20);
}

#[test]
fn coop_gemm_epilogue_bias_act_alpha_batches() {
    let (ctx, _g) = coop!();
    for (i, act) in
        [UnaryOp::Silu, UnaryOp::Relu, UnaryOp::Identity, UnaryOp::Sigmoid].into_iter().enumerate()
    {
        for (j, bias) in [GemmBias::Row, GemmBias::Col, GemmBias::None].into_iter().enumerate() {
            for out in [DType::F16, DType::F32] {
                let mut d = f16_desc(70, 90, 48, false, false, 1);
                d.act = act;
                d.bias = bias;
                d.alpha = 0.75;
                run_gemm(&ctx, &d, out, None, false, (i * 3 + j) as u64);
            }
        }
    }
    // Two-level batch, a shared A (stride 0), padded leading dimensions, offsets and a strided bias.
    let mut d = f16_desc(24, 40, 20, false, false, 2);
    d.batch = (2, 3);
    d.a = BatchStrides { offset: 4, outer: 0, inner: 24 * 20 };
    d.ldb = 44;
    d.b = BatchStrides { offset: 0, outer: 3 * 20 * 44, inner: 20 * 44 };
    d.ldc = 48;
    d.c = BatchStrides { offset: 8, outer: 3 * 24 * 48, inner: 24 * 48 };
    d.bias = GemmBias::Row;
    d.bias_strides = BatchStrides { offset: 2, outer: 3 * 24, inner: 24 };
    d.act = UnaryOp::Silu;
    for out in [DType::F16, DType::F32] {
        run_gemm(&ctx, &d, out, None, false, 30);
    }
    // Unaligned offsets force scalar loads and stores.
    d.a.offset = 3;
    d.c.offset = 5;
    run_gemm(&ctx, &d, DType::F16, None, false, 31);
    // Attention-shaped: batched heads, transposed operands, a softmax scale as alpha.
    let mut d = f16_desc(400, 400, 64, true, false, 2);
    d.alpha = 0.125;
    run_gemm(&ctx, &d, DType::F16, None, false, 32);
    run_gemm(&ctx, &f16_desc(64, 400, 400, false, true, 2), DType::F16, None, false, 33);
}

#[test]
fn gemm_routes_f16_to_tensor_cores() {
    let (ctx, _g) = coop!();
    let mut d = f16_desc(70, 90, 75, false, true, 3);
    d.bias = GemmBias::Col;
    d.act = UnaryOp::Silu;
    run_gemm(&ctx, &d, DType::F16, None, true, 40);
    // The routed result is the tensor-core kernel's (bit for bit), not the regular kernel's.
    let mut rng = Rng::new(41);
    let a = upload(&ctx, &f16_round(&rng.vec(70 * 75 * 3, -1.0, 1.0)), DType::F16);
    let b = upload(&ctx, &f16_round(&rng.vec(90 * 75 * 3, -1.0, 1.0)), DType::F16);
    let bias = upload(&ctx, &f16_round(&rng.vec(90 * 3, -1.0, 1.0)), DType::F16);
    let outs: Vec<Vec<f32>> = [Some(true), None, Some(false)]
        .into_iter()
        .map(|mode| {
            let c = GpuTensor::zeros(&ctx, &[70 * 90 * 3], DType::F16).unwrap();
            match mode {
                Some(on) => {
                    ctx.set_tensor_cores(on);
                    gemm::gemm(&ctx, &d, a.raw(), b.raw(), Some(bias.raw()), c.raw()).unwrap();
                }
                None => coop::gemm(&ctx, &d, DType::F16, a.raw(), b.raw(), Some(bias.raw()), c.raw(), None)
                    .unwrap(),
            }
            c.to_vec().unwrap()
        })
        .collect();
    ctx.set_tensor_cores(true);
    assert_eq!(outs[0], outs[1], "routed f16 GEMM is the tensor-core kernel");
    assert_ne!(outs[0], outs[2], "switched off, the regular kernel runs (different rounding)");
    // f32 GEMMs never take the tensor-core path.
    let d32 = GemmDesc::new(8, 8, 8, false, false, 1);
    let x = GpuTensor::from_slice(&ctx, &[1.0; 64], &[64]).unwrap();
    let y = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    gemm::gemm(&ctx, &d32, x.raw(), x.raw(), None, y.raw()).unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![8.0; 64]);
}

#[test]
fn coop_rejects_misuse() {
    let (ctx, _g) = coop!();
    let buf = GpuTensor::zeros(&ctx, &[64], DType::F32).unwrap();
    let d32 = GemmDesc::new(8, 8, 8, false, false, 1);
    assert!(coop::gemm(&ctx, &d32, DType::F32, buf.raw(), buf.raw(), None, buf.raw(), None).is_err());
    // Buffers too small for the problem (the pool rounds up to 256 bytes, so use a 64³ GEMM).
    let small = GpuTensor::zeros(&ctx, &[4], DType::F16).unwrap();
    let big = f16_desc(64, 64, 64, false, false, 1);
    assert!(coop::gemm(&ctx, &big, DType::F16, small.raw(), small.raw(), None, small.raw(), None).is_err());
    let d = f16_desc(8, 8, 8, false, false, 1);
    let bad = CoopConfig { bm: 48, bn: 128, bk: 32, wgm: 2, wgn: 4 };
    let ok = GpuTensor::zeros(&ctx, &[64], DType::F16).unwrap();
    assert!(coop::gemm(&ctx, &d, DType::F16, ok.raw(), ok.raw(), None, ok.raw(), Some(bad)).is_err());
    // A context without tensor cores refuses the explicit entry points.
    let plain = super::ctx().unwrap();
    if !plain.caps().coop_f16 {
        let ok = GpuTensor::zeros(&plain, &[64], DType::F16).unwrap();
        assert!(matches!(
            coop::gemm(&plain, &d, DType::F16, ok.raw(), ok.raw(), None, ok.raw(), None),
            Err(crate::GpuError::Unsupported(_))
        ));
    }
    // Depthwise convs have no tensor-core path.
    let dw = Conv2dDesc {
        n: 1,
        c_in: 4,
        h: 4,
        w: 4,
        c_out: 4,
        groups: 4,
        win: Window2d::square(3, 1, 1),
        act: UnaryOp::Identity,
        dtype: DType::F16,
    };
    let t = GpuTensor::zeros(&ctx, &[64], DType::F16).unwrap();
    assert!(coop::conv2d(&ctx, &dw, DType::F16, t.raw(), t.raw(), None, t.raw(), None).is_err());
}

// ----- conv ---------------------------------------------------------------------------------------------------

fn cpu_conv(d: &Conv2dDesc, x: &[f32], w: &[f32], bias: Option<&[f32]>, abs: bool) -> Vec<f32> {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let mut y = vec![0f32; d.n * d.c_out * oh * ow];
    for n in 0..d.n {
        for co in 0..d.c_out {
            let g = co / cout_g;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = 0f64;
                    for ci in 0..cin_g {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * d.win.stride.0 + ky * d.win.dilation.0) as isize
                                    - d.win.padding.0 as isize;
                                let ix = (ox * d.win.stride.1 + kx * d.win.dilation.1) as isize
                                    - d.win.padding.1 as isize;
                                if iy < 0 || ix < 0 || iy >= d.h as isize || ix >= d.w as isize {
                                    continue;
                                }
                                let xi =
                                    ((n * d.c_in + g * cin_g + ci) * d.h + iy as usize) * d.w + ix as usize;
                                let p = x[xi] as f64 * w[((co * cin_g + ci) * kh + ky) * kw + kx] as f64;
                                s += if abs { p.abs() } else { p };
                            }
                        }
                    }
                    let i = ((n * d.c_out + co) * oh + oy) * ow + ox;
                    y[i] = if abs { s as f32 } else { d.act.apply(s as f32 + bias.map_or(0.0, |b| b[co])) };
                }
            }
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn cdesc(
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    groups: usize,
    win: Window2d,
) -> Conv2dDesc {
    Conv2dDesc { n, c_in, h, w, c_out, groups, win, act: UnaryOp::Silu, dtype: DType::F16 }
}

/// Runs `d` on tensor cores: through `conv::conv2d`'s routing (`out == F16`, `cfg == None`) and explicitly.
fn run_conv(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    with_bias: bool,
    out: DType,
    cfg: Option<CoopConfig>,
    seed: u64,
) {
    let mut rng = Rng::new(seed);
    let (kh, kw) = d.win.kernel;
    let x = f16_round(&rng.vec(d.n * d.c_in * d.h * d.w, -1.0, 1.0));
    let w = f16_round(&rng.vec(d.c_out * (d.c_in / d.groups) * kh * kw, -1.0, 1.0));
    let mut b = rng.vec(d.c_out, -1.0, 1.0);
    if out == DType::F16 {
        b = f16_round(&b);
    }
    let want = cpu_conv(d, &x, &w, with_bias.then_some(b.as_slice()), false);
    let sums = cpu_conv(d, &x, &w, None, true);
    let k = d.c_in / d.groups * kh * kw;
    let bmax = if with_bias { b.iter().fold(0f32, |m, v| m.max(v.abs())) } else { 0.0 };
    let tol: Vec<f64> = want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, k, bmax, out)).collect();
    let (gx, gw, gb) = (upload(ctx, &x, DType::F16), upload(ctx, &w, DType::F16), upload(ctx, &b, out));
    let gy = GpuTensor::empty(ctx, &[want.len()], out).unwrap();
    let bias = with_bias.then(|| gb.raw());
    if out == DType::F16 && cfg.is_none() {
        conv::conv2d(ctx, d, gx.raw(), gw.raw(), bias, gy.raw()).unwrap();
        assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
        let routed = gy.to_vec().unwrap();
        assert_within(&routed, &want, &tol, &format!("routed conv {d:?} bias={with_bias}"));
    }
    if !d.is_depthwise() {
        coop::conv2d(ctx, d, out, gx.raw(), gw.raw(), bias, gy.raw(), cfg).unwrap();
        let got = gy.to_vec().unwrap();
        assert_within(
            &got,
            &want,
            &tol,
            &format!("coop conv {d:?} bias={with_bias} out {out:?} cfg {cfg:?}"),
        );
    }
}

#[test]
fn coop_conv_variants() {
    let (ctx, _g) = coop!();
    let s1 = Window2d::square(3, 1, 1);
    let s2 = Window2d::square(3, 2, 1);
    for out in [DType::F16, DType::F32] {
        // folo-n / s: 3×3 s1 p1 and 3×3 s2 p1 (C3k2 / downsampling), the 3-channel stem, 1×1 (pointwise GEMM).
        run_conv(&ctx, &cdesc(2, 16, 20, 20, 32, 1, s1), true, out, None, 1);
        run_conv(&ctx, &cdesc(2, 16, 13, 17, 32, 1, s2), true, out, None, 2);
        run_conv(&ctx, &cdesc(1, 3, 33, 31, 16, 1, s2), true, out, None, 3);
        run_conv(&ctx, &cdesc(2, 24, 7, 9, 40, 1, Window2d::square(1, 1, 0)), true, out, None, 4);
        run_conv(&ctx, &cdesc(1, 64, 20, 20, 64, 1, Window2d::square(1, 1, 0)), true, out, None, 5);
        // Ragged channel counts (partial M tiles), batches, no bias.
        run_conv(&ctx, &cdesc(3, 20, 11, 10, 70, 1, s1), false, out, None, 6);
        run_conv(&ctx, &cdesc(1, 32, 9, 9, 130, 1, s1), true, out, None, 7);
        // Everything else the implicit GEMM covers: 1×1 stride 2, dilation, rectangular kernels, groups.
        run_conv(&ctx, &cdesc(2, 6, 9, 9, 4, 1, Window2d::square(1, 2, 0)), true, out, None, 8);
        let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
        run_conv(&ctx, &cdesc(1, 8, 12, 9, 24, 1, dil), true, out, None, 9);
        let rect = Window2d { kernel: (1, 3), stride: (2, 1), padding: (0, 1), dilation: (1, 1) };
        run_conv(&ctx, &cdesc(2, 5, 7, 8, 12, 1, rect), true, out, None, 10);
        run_conv(&ctx, &cdesc(2, 16, 10, 10, 32, 2, s1), true, out, None, 11);
    }
    // Depthwise (the head's and the attention's DWConv) keeps its direct kernel with tensor cores on.
    run_conv(&ctx, &cdesc(2, 32, 10, 10, 32, 32, s1), true, DType::F16, None, 12);
}

#[test]
fn coop_conv_every_config() {
    let (ctx, _g) = coop!();
    for cfg in CONFIGS {
        run_conv(
            &ctx,
            &cdesc(2, 16, 12, 12, 40, 1, Window2d::square(3, 1, 1)),
            true,
            DType::F16,
            Some(cfg),
            20,
        );
        run_conv(
            &ctx,
            &cdesc(1, 8, 16, 16, 128, 1, Window2d::square(3, 2, 1)),
            false,
            DType::F32,
            Some(cfg),
            21,
        );
        run_conv(&ctx, &cdesc(2, 12, 6, 6, 9, 1, Window2d::square(1, 1, 0)), true, DType::F16, Some(cfg), 22);
    }
}

/// The routed conv (Conv + bias + SiLU in one kernel, as folo-nn's fused `ConvBnAct` runs it) agrees with the
/// regular f16 kernel, and switching tensor cores off gives the regular kernel's result again.
#[test]
fn conv_routing_matches_regular_kernel() {
    let (ctx, _g) = coop!();
    let d = cdesc(2, 32, 20, 20, 64, 1, Window2d::square(3, 1, 1));
    let mut rng = Rng::new(50);
    let x = upload(&ctx, &f16_round(&rng.vec(2 * 32 * 400, -1.0, 1.0)), DType::F16);
    let w = upload(&ctx, &f16_round(&rng.vec(64 * 32 * 9, -0.2, 0.2)), DType::F16);
    let b = upload(&ctx, &f16_round(&rng.vec(64, -0.5, 0.5)), DType::F16);
    let run = |on: bool| {
        ctx.set_tensor_cores(on);
        let y = GpuTensor::empty(&ctx, &[2 * 64 * 400], DType::F16).unwrap();
        conv::conv2d(&ctx, &d, x.raw(), w.raw(), Some(b.raw()), y.raw()).unwrap();
        y.to_vec().unwrap()
    };
    let (tc, plain, plain2) = (run(true), run(false), run(false));
    ctx.set_tensor_cores(true);
    assert_eq!(plain, plain2);
    // Both are f16 roundings of nearly the same f32 sum: at most a couple of f16 ulps apart.
    for (i, (&p, &t)) in plain.iter().zip(&tc).enumerate() {
        assert!((p - t).abs() <= 2e-3 * (1.0 + p.abs()), "[{i}] regular {p} vs tensor cores {t}");
    }
}

// ----- conv gradients (mixed-precision training) ----------------------------------------------------------------

/// f64 reference gradients of the conv `d` for the output gradient `g`: `(gx, gw)`, or their absolute-term sums
/// with `abs` (for the error bounds).
fn cpu_conv_grads(d: &Conv2dDesc, x: &[f32], w: &[f32], g: &[f32], abs: bool) -> (Vec<f32>, Vec<f32>) {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let mut gx = vec![0f64; x.len()];
    let mut gw = vec![0f64; w.len()];
    let f = |p: f64| if abs { p.abs() } else { p };
    for n in 0..d.n {
        for co in 0..d.c_out {
            let grp = co / cout_g;
            for oy in 0..oh {
                for ox in 0..ow {
                    let gv = g[((n * d.c_out + co) * oh + oy) * ow + ox] as f64;
                    for ci in 0..cin_g {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * d.win.stride.0 + ky * d.win.dilation.0) as isize
                                    - d.win.padding.0 as isize;
                                let ix = (ox * d.win.stride.1 + kx * d.win.dilation.1) as isize
                                    - d.win.padding.1 as isize;
                                if iy < 0 || ix < 0 || iy >= d.h as isize || ix >= d.w as isize {
                                    continue;
                                }
                                let xi =
                                    ((n * d.c_in + grp * cin_g + ci) * d.h + iy as usize) * d.w + ix as usize;
                                let wi = ((co * cin_g + ci) * kh + ky) * kw + kx;
                                gx[xi] += f(gv * w[wi] as f64);
                                gw[wi] += f(gv * x[xi] as f64);
                            }
                        }
                    }
                }
            }
        }
    }
    let cast = |v: Vec<f64>| v.into_iter().map(|s| s as f32).collect();
    (cast(gx), cast(gw))
}

/// Both gradients of the f16 conv `d` (input gradient f16; weight gradient as `w_out`) against the f64 reference,
/// on tensor cores (`tc`) or the regular kernels.
fn run_conv_grads(ctx: &GpuContext, d: &Conv2dDesc, w_out: DType, tc: bool, seed: u64) {
    ctx.set_tensor_cores(tc);
    let mut rng = Rng::new(seed);
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let x = f16_round(&rng.vec(d.n * d.c_in * d.h * d.w, -1.0, 1.0));
    let w = f16_round(&rng.vec(d.c_out * (d.c_in / d.groups) * kh * kw, -1.0, 1.0));
    let g = f16_round(&rng.vec(d.n * d.c_out * oh * ow, -1.0, 1.0));
    let (want_x, want_w) = cpu_conv_grads(d, &x, &w, &g, false);
    let (sum_x, sum_w) = cpu_conv_grads(d, &x, &w, &g, true);
    let (gx, gw, gg) =
        (upload(ctx, &x, DType::F16), upload(ctx, &w, DType::F16), upload(ctx, &g, DType::F16));
    let dx = GpuTensor::empty(ctx, &[x.len()], DType::F16).unwrap();
    let dw = GpuTensor::empty(ctx, &[w.len()], w_out).unwrap();
    conv::conv2d_backward_input(ctx, d, gg.raw(), gw.raw(), dx.raw()).unwrap();
    conv::conv2d_backward_weight_as(ctx, d, gx.raw(), gg.raw(), dw.raw(), w_out).unwrap();
    assert_eq!(ctx.tensor_cores(), tc, "{}", ctx.tensor_cores_status());
    let ident = |d: &Conv2dDesc| Conv2dDesc { act: UnaryOp::Identity, ..d.clone() };
    let kx = d.c_out / d.groups * kh * kw;
    let tol_x: Vec<f64> =
        want_x.iter().zip(&sum_x).map(|(&y, &s)| bound(y, s as f64, kx, 0.0, DType::F16)).collect();
    let kw_ = d.n * oh * ow;
    let tol_w: Vec<f64> =
        want_w.iter().zip(&sum_w).map(|(&y, &s)| bound(y, s as f64, kw_, 0.0, w_out)).collect();
    let what = format!("{:?} tensor cores {tc}", ident(d));
    assert_within(&dx.to_vec().unwrap(), &want_x, &tol_x, &format!("input gradient of {what}"));
    assert_within(&dw.to_vec().unwrap(), &want_w, &tol_w, &format!("weight gradient ({w_out:?}) of {what}"));
}

#[test]
fn coop_conv_gradients() {
    let (ctx, _g) = coop!();
    let s1 = Window2d::square(3, 1, 1);
    let s2 = Window2d::square(3, 2, 1);
    let p1 = Window2d::square(1, 1, 0);
    let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
    let rect = Window2d { kernel: (1, 3), stride: (2, 1), padding: (0, 1), dilation: (1, 1) };
    let cases = [
        // folo-n / s: 3×3 s1 p1 and s2 p1, the 3-channel stem, 1×1 (input gradient as a GEMM), ragged channels.
        cdesc(2, 16, 20, 20, 32, 1, s1),
        cdesc(2, 16, 13, 17, 32, 1, s2),
        cdesc(1, 3, 33, 31, 16, 1, s2),
        cdesc(2, 24, 7, 9, 40, 1, p1),
        cdesc(3, 20, 11, 10, 70, 1, s1),
        cdesc(1, 130, 6, 6, 33, 1, p1),
        // Dilation, rectangular kernels, a strided 1×1, groups, depthwise (direct kernels).
        cdesc(1, 8, 12, 9, 24, 1, dil),
        cdesc(2, 5, 7, 8, 12, 1, rect),
        cdesc(2, 6, 9, 9, 4, 1, Window2d::square(1, 2, 0)),
        cdesc(2, 16, 10, 10, 32, 2, s1),
        cdesc(2, 32, 10, 10, 32, 32, s1),
        // A long weight-gradient reduction (split-K: N·OH·OW = 6400) with a wide M.
        cdesc(4, 16, 40, 40, 128, 1, s1),
    ];
    for (i, d) in cases.iter().enumerate() {
        for tc in [true, false] {
            for w_out in [DType::F32, DType::F16] {
                run_conv_grads(&ctx, d, w_out, tc, 100 + i as u64);
            }
        }
    }
    ctx.set_tensor_cores(true);
}

/// The weight gradient's f32 output takes f16 gradients whose products sum past the f16 range (an f16 result
/// would overflow) and keeps them finite.
#[test]
fn coop_weight_gradient_f32_output_does_not_overflow() {
    let (ctx, _g) = coop!();
    let d = cdesc(4, 8, 32, 32, 16, 1, Window2d::square(3, 1, 1));
    let x = vec![60000f32; d.n * d.c_in * d.h * d.w];
    let g = vec![60000f32; d.n * d.c_out * d.h * d.w];
    let (gx, gg) = (upload(&ctx, &x, DType::F16), upload(&ctx, &g, DType::F16));
    let dw = GpuTensor::empty(&ctx, &[d.c_out * d.c_in * 9], DType::F32).unwrap();
    conv::conv2d_backward_weight_as(&ctx, &d, gx.raw(), gg.raw(), dw.raw(), DType::F32).unwrap();
    let got = dw.to_vec().unwrap();
    // The centre tap sees every pixel: 4 · 32 · 32 · 60000² = 1.47e13.
    let centre = got[4];
    assert!((centre / 1.47456e13 - 1.0).abs() < 1e-3, "{centre}");
    assert!(got.iter().all(|v| v.is_finite()));
    // An f32 conv's weight gradient cannot be written as f16.
    let f32_conv = Conv2dDesc { dtype: DType::F32, ..d.clone() };
    let x32 = upload(&ctx, &x, DType::F32);
    let g32 = upload(&ctx, &g, DType::F32);
    assert!(
        conv::conv2d_backward_weight_as(&ctx, &f32_conv, x32.raw(), g32.raw(), dw.raw(), DType::F16).is_err()
    );
}
