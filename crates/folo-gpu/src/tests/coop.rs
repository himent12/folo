//! Tensor-core (cooperative-matrix) kernels against the CPU reference.
//!
//! The tests run on their own context created with tensor cores requested, and skip (with a note) when the
//! adapter has none. The reference is the f64 CPU loop on the same f16-rounded inputs, so only the GPU's
//! arithmetic is under test. **Tolerance**, per element: f16 × f16 products are exact in f32 (11 + 11
//! significant bits), so the only accumulation error is the f32 summation, at most `K · u · Σ|a·b|` with
//! `u = 2^-23` (one ulp per addition, which also covers tensor cores that truncate instead of rounding when they
//! align addends). The activation adds a few f32 ulps of the result (SiLU's slope is below 1.1), and an f16 output
//! one rounding, `2^-11 · |y|` plus `2^-24` for subnormals. The bound is applied with a factor of 2 of slack; a
//! wrong tile, a dropped `K` step or a bad index is off by `O(|a·b|)`, orders of magnitude more. The conv cases
//! are in [`super::coop_conv`].

use std::sync::{Mutex, MutexGuard, OnceLock};

use super::{Rng, f16_round, to_f16};
use crate::kernels::conv::Conv2dDesc;
use crate::kernels::coop::{self, CoopConfig};
use crate::kernels::gemm::{self, BatchStrides, GemmBias, GemmDesc};
use crate::kernels::pooling::Window2d;
use crate::tests::gemm::cpu_gemm;
use crate::{DType, GpuContext, GpuOptions, GpuTensor, UnaryOp};

/// The tensor-core context plus a lock: tests flip the run-time switch, so they run one at a time. `None` (a skip)
/// only when the adapter does not qualify; on one that does, failing to get tensor cores is a test failure.
pub(super) fn coop_ctx() -> Option<(GpuContext, MutexGuard<'static, ()>)> {
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
pub(super) fn bound(y: f32, s: f64, k: usize, bias: f32, out: DType) -> f64 {
    let acc = k.max(1) as f64 * U32 * s + U32 * bias.abs() as f64;
    let act = 1.1 * acc + 8.0 * U32 * y.abs() as f64;
    let round = if out == DType::F16 { y.abs() as f64 / 2048.0 + 1.0 / (1u64 << 24) as f64 } else { 0.0 };
    2.0 * (act + round) + 1e-7
}

#[track_caller]
pub(super) fn assert_within(got: &[f32], want: &[f32], tol: &[f64], what: &str) {
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

pub(super) fn upload(ctx: &GpuContext, v: &[f32], dtype: DType) -> GpuTensor {
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

pub(super) const CONFIGS: [CoopConfig; 6] = [
    CoopConfig::C128X128,
    CoopConfig::C64X128,
    CoopConfig::C64X64,
    CoopConfig::C32X128,
    CoopConfig::C32X32,
    CoopConfig::C16X256,
];

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
