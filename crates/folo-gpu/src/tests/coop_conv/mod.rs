//! Tensor-core conv2d tests: shared fixtures and CPU forward reference, split into [`forward`] and [`backward`].

mod backward;
mod forward;

use super::coop::{assert_within, bound, upload};
use super::{Rng, f16_round, traced_splits};
use crate::kernels::conv::{self, Conv2dDesc, split};
use crate::kernels::coop::{self as coop_k, CoopConfig};
use crate::kernels::pooling::Window2d;
use crate::{DType, GpuContext, GpuTensor, UnaryOp};

/// `let (ctx, _guard) = coop!();` — the tensor-core context (as [`super::coop::coop_ctx`]) or an early return.
macro_rules! coop {
    () => {
        match $crate::tests::coop::coop_ctx() {
            Some(c) => c,
            None => return,
        }
    };
}
pub(super) use coop;

pub(super) fn cpu_conv(d: &Conv2dDesc, x: &[f32], w: &[f32], bias: Option<&[f32]>, abs: bool) -> Vec<f32> {
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
pub(super) fn cdesc(
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
pub(super) fn run_conv(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    with_bias: bool,
    out: DType,
    cfg: Option<CoopConfig>,
    seed: u64,
) {
    run_conv_split(ctx, d, with_bias, out, cfg, seed, None);
}

/// [`run_conv`] with the routed conv split into `splits` K chunks (`None`: picked from the shape). Returns the
/// chunks the routed and the explicit conv ran as (0: not run), after checking them against [`split::plan`].
pub(super) fn run_conv_split(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    with_bias: bool,
    out: DType,
    cfg: Option<CoopConfig>,
    seed: u64,
    splits: Option<usize>,
) -> [usize; 2] {
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
    let mut chunks = [0; 2];
    if out == DType::F16 && cfg.is_none() {
        let plan = conv::ConvPlan { tiles: None, splits };
        chunks[0] = traced_splits(ctx, || {
            conv::conv2d_with_plan(ctx, d, gx.raw(), gw.raw(), bias, gy.raw(), plan).unwrap();
        });
        assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
        assert_eq!(
            chunks[0],
            planned_splits(d, None, splits),
            "chunks of routed conv {d:?} splits {splits:?}"
        );
        let routed = gy.to_vec().unwrap();
        assert_within(&routed, &want, &tol, &format!("routed conv {d:?} bias={with_bias} splits {splits:?}"));
    }
    if !d.is_depthwise() {
        chunks[1] = traced_splits(ctx, || {
            coop_k::conv2d(ctx, d, out, gx.raw(), gw.raw(), bias, gy.raw(), cfg).unwrap();
        });
        assert_eq!(chunks[1], planned_splits(d, cfg, None), "chunks of coop conv {d:?} cfg {cfg:?}");
        let got = gy.to_vec().unwrap();
        assert_within(
            &got,
            &want,
            &tol,
            &format!("coop conv {d:?} bias={with_bias} out {out:?} cfg {cfg:?}"),
        );
    }
    chunks
}

/// The `K` chunks [`split::plan`] gives the tensor-core implicit GEMM of `d` with `cfg` (picked when `None`) and
/// `want` chunks requested (1 for the other paths).
pub(super) fn planned_splits(d: &Conv2dDesc, cfg: Option<CoopConfig>, want: Option<usize>) -> usize {
    if d.is_depthwise() || d.is_pointwise() {
        return 1;
    }
    let (oh, ow) = d.output().unwrap();
    let (m, n, k, batch) = gemm_dims(d);
    let cfg = cfg.unwrap_or_else(|| picked_config(d));
    let tiles = m.div_ceil(cfg.bm as usize) * n.div_ceil(cfg.bn as usize);
    split::plan(cfg.bk as usize, k, tiles, batch, d.n * d.c_out * oh * ow, want).0
}

/// `(M, N, K, batch)` of the implicit GEMM of `d`.
pub(super) fn gemm_dims(d: &Conv2dDesc) -> (usize, usize, usize, usize) {
    let (oh, ow) = d.output().unwrap();
    let k = d.c_in / d.groups * d.win.kernel.0 * d.win.kernel.1;
    (d.c_out / d.groups, oh * ow, k, d.n * d.groups)
}

/// The tile config the tensor-core forward conv picks for `d`.
pub(super) fn picked_config(d: &Conv2dDesc) -> CoopConfig {
    let (m, n, _, batch) = gemm_dims(d);
    coop_k::pick_config_conv_batched(m, n, batch)
}

/// `x` (`[n, c, hw]`) placed at channel `at` of a larger `[n, big, hw]` buffer of noise: a channel slice.
pub(super) fn embed(
    x: &[f32],
    n: usize,
    c: usize,
    hw: usize,
    big: usize,
    at: usize,
    rng: &mut Rng,
) -> Vec<f32> {
    let mut out = rng.vec(n * big * hw, -1.0, 1.0);
    for i in 0..n {
        let (src, dst) = (i * c * hw, (i * big + at) * hw);
        out[dst..dst + c * hw].copy_from_slice(&x[src..src + c * hw]);
    }
    out
}

/// `(BM, BN)` of the one tensor-core GEMM that `f` encodes on `ctx`, read from a trace.
#[track_caller]
pub(super) fn traced_tile(ctx: &GpuContext, f: impl FnOnce()) -> (u32, u32) {
    ctx.trace_begin();
    f();
    let ops = ctx.trace_end();
    let gemms: Vec<_> = ops.iter().filter(|op| op.name.starts_with("coop_gemm")).collect();
    let names: Vec<_> = ops.iter().map(|op| op.name).collect();
    assert_eq!(gemms.len(), 1, "one tensor-core GEMM, ran {names:?}");
    let get = |key: &str| -> u32 {
        let v = gemms[0].spec.split(' ').find_map(|p| p.strip_prefix(key));
        v.and_then(|v| v.parse().ok()).unwrap_or_else(|| panic!("no {key} in {:?}", gemms[0].spec))
    };
    (get("BM="), get("BN="))
}
