//! The weight gradient ([`conv2d_backward_weight_as`]): implicit GEMM (`CONV_MODE` 3) over `K = N·OH·OW`,
//! split along `K` into f32 partial slabs when the weight has too few tiles, and the direct depthwise kernel.

use super::depthwise::DEPTHWISE_GRAD;
use super::{Conv2dDesc, check_grad_desc, conv_spec, coop_grad};
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::kernels::gemm::{self, BatchStrides, GEMM, GemmDesc, TileConfig, base_spec, gemm_params};
use crate::kernels::{check_buf, coop, prod};
use crate::pipeline::{Params, Spec};

/// Tile configuration of a regular (not tensor-core) weight gradient with `m = C_out/groups` rows and
/// `n = C_in/groups·KH·KW` columns, by the shape alone: split-K supplies the workgroups, so the tile is picked for
/// arithmetic intensity (as [`coop`]'s weight gradients) rather than by [`gemm::pick_config`]'s occupancy rule,
/// which always ended at the 64×64 tile with 4×4 outputs per thread here.
pub(crate) fn pick_config_weight_grad(ctx: &GpuContext, m: usize, n: usize) -> TileConfig {
    let cfg = match m {
        0..=16 if n <= 32 => STEM,
        _ if n < 128 => TileConfig::SMALL,
        // Weight gradients with at most 32 output channels.
        0..=32 => TileConfig::ROWS32,
        _ if m.div_ceil(128) * 128 * 4 <= m * 5 => TileConfig::LARGE,
        _ => TileConfig::MEDIUM,
    };
    if cfg.workgroup_bytes() <= ctx.caps().max_workgroup_storage { cfg } else { TileConfig::SMALL }
}

/// 16×32 tiles, depth 32, 4×4 per thread, 32 threads: the stem (3→16, `N = 27`), whose 1.6 M-deep reduction
/// split-K spreads over thousands of workgroups; a wider tile would be mostly padding.
const STEM: TileConfig = TileConfig { bm: 16, bn: 32, bk: 32, tm: 4, tn: 4 };

/// Whether the weight gradient of an `m × n` weight takes the regular kernels' [`STEM`] tile even on tensor
/// cores (their narrowest tile is 16×256, 89 % padding at `N = 27`).
pub(crate) fn stem_shaped(m: usize, n: usize) -> bool {
    m <= 16 && n <= 32
}

/// Largest `KH · KW` the direct depthwise weight-gradient kernel unrolls; bigger kernels use the GEMM path.
const MAX_DEPTHWISE_TAPS: usize = 49;
/// Upper bound for split-K partial slabs, in f32 elements (64 MiB).
const MAX_PARTIAL_ELEMS: usize = 16 << 20;

/// The weight gradient's spatial sizes as specialisation constants (`SP_IH`, `SP_IW`, `SP_OH`, `SP_OW`): its B
/// gather divides every `k = (n, oh, ow)` by `OH·OW` and `OW`, which a constant divisor turns into a multiply.
pub(crate) fn weight_grad_spec(spec: Spec, d: &Conv2dDesc, oh: usize, ow: usize) -> Spec {
    spec.u32("SP_IH", d.h as u32).u32("SP_IW", d.w as u32).u32("SP_OH", oh as u32).u32("SP_OW", ow as u32)
}

/// The weight-gradient generators' `IH`, `IW`, `OW` and `OHOW`: the constants of [`weight_grad_spec`] when the
/// spec has them, else the uniforms.
pub(crate) fn weight_grad_dims(spec: &Spec) -> String {
    if spec.get_u32("SP_OW", 0) > 0 {
        "  let IH = SP_IH; let IW = SP_IW; let OW = SP_OW; let OHOW = SP_OH * SP_OW;\n".into()
    } else {
        "  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u); let OHOW = pu(31u);\n".into()
    }
}

/// One element of an invocation's share of a weight gradient's im2col B tile (`bk × bn`, `k`-major rows): row
/// `kq · run + wl`, column `jj`, both WGSL `u32` expressions of `wl = tid % run` and `wq = tid / run` (`run`
/// of the [`WgradPlan`]).
pub(crate) struct WgradUnit {
    /// The row's run index (a constant, or an expression of `wq`).
    pub kq: String,
    /// The column inside the tile.
    pub jj: String,
}

/// An invocation's share of a weight gradient's B tile ([`wgrad_b_units`]).
pub(crate) struct WgradPlan {
    /// Consecutive pixels (`k`) of one column that neighbouring invocations load.
    pub run: u32,
    /// The elements, in register order.
    pub units: Vec<WgradUnit>,
}

/// The weight gradient's B-tile plan for `threads` invocations: neighbouring invocations take `run`
/// neighbouring pixels of one column, then the next columns. Along `k = (n, oh, ow)` neighbours are neighbouring
/// pixels of one image row (a coalesced gather); along `j = (ci, kh, kw)` they would sit a whole image plane
/// apart, so a warp's load would touch one memory sector per invocation. Mixing a short run of pixels with
/// several columns per warp, rather than a whole warp along `k`, keeps the stores into the k-major workgroup
/// tile free of bank conflicts (see the generators' row padding).
pub(crate) fn wgrad_b_units(threads: u32, bk: u32, bn: u32, run: u32) -> WgradPlan {
    let r = run;
    let lanes = threads / r;
    assert!(threads.is_multiple_of(r) && bk.is_multiple_of(r), "weight-gradient B plan: {threads} / {bk}");
    assert!((bk * bn).is_multiple_of(threads), "weight-gradient B plan: {bk}×{bn} over {threads}");
    // Element `e = tid + u · threads` is (row = (e / r / bn) · r + e % r, column = (e / r) % bn), and
    // `e / r = wq + u · lanes`.
    let units = (0..bk * bn / threads)
        .map(|u| {
            if bn.is_multiple_of(lanes) {
                let per = bn / lanes;
                WgradUnit { kq: format!("{}u", u / per), jj: format!("(wq + {}u)", lanes * (u % per)) }
            } else {
                assert!(lanes.is_multiple_of(bn), "weight-gradient B plan: {lanes} lanes, {bn} columns");
                WgradUnit { kq: format!("(wq / {bn}u + {}u)", lanes / bn * u), jj: format!("(wq % {bn}u)") }
            }
        })
        .collect();
    WgradPlan { run, units }
}

/// The WGSL of a weight gradient's B-tile load (from `buf`, converted by `cast`): each distinct run of `units` gets its
/// pixel `(wn, wr)` decoded once, then each unit gathers the image element that pixel read through tap
/// `j = (ci, kh, kw)` into `reg(u)` (zero outside the image or past `K` / `N`).
pub(crate) fn wgrad_b_load(
    plan: &WgradPlan,
    buf: &str,
    reg: impl Fn(usize) -> String,
    cast: &str,
    zero: &str,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let mut runs: Vec<&str> = Vec::new();
    let run = plan.run;
    for (u, w) in plan.units.iter().enumerate() {
        let q = match runs.iter().position(|k| *k == w.kq) {
            Some(q) => q,
            None => {
                runs.push(&w.kq);
                let q = runs.len() - 1;
                let _ = writeln!(
                    s,
                    "    let wk{q} = k0 + {} * {run}u + wl; let wkok{q} = wk{q} < K; let wn{q} = wk{q} / OHOW;\n\
                     \x20   let wr{q} = wk{q} % OHOW; let wb{q} = offB + wn{q} * SBN;\n\
                     \x20   let wih{q} = i32((wr{q} / OW) * SH) - PH; let wiw{q} = i32((wr{q} % OW) * SW) - PW;",
                    w.kq
                );
                q
            }
        };
        let _ = writeln!(
            s,
            "    {{ let j = n0 + {}; let ci = j / (KH * KW); let r = j % (KH * KW);\n\
             \x20     let ih = wih{q} + i32((r / KW) * DH); let iw = wiw{q} + i32((r % KW) * DW);\n\
             \x20     if (wkok{q} && j < N && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
             \x20       {} = {cast}({buf}[wb{q} + (ci * IH + u32(ih)) * IW + u32(iw)]);\n\
             \x20     }} else {{ {} = {zero}; }} }}",
            w.jj,
            reg(u),
            reg(u)
        );
    }
    s
}

/// Gradient of [`conv2d`](crate::kernels::conv::conv2d) with respect to its weight, written as `d.dtype`: [`conv2d_backward_weight_as`].
pub fn conv2d_backward_weight(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gw: &wgpu::Buffer,
) -> Result<()> {
    conv2d_backward_weight_as(ctx, d, x, grad, gw, d.dtype)
}

/// Gradient of [`conv2d`](crate::kernels::conv::conv2d) with respect to its weight. `d` describes the forward problem. `x`:
/// `[N, C_in, H, W]` and `grad`: `[N, C_out, OH, OW]` are stored as `d.dtype` (f32, or f16 with f32
/// accumulation, on tensor cores when the context has them on); `gw`: `[C_out, C_in/groups, KH, KW]` (fully
/// written) as `out`, which is f32 or `d.dtype` (mixed precision keeps f32 master weights, so an f16 conv's
/// weight gradient can be written straight to f32 without an f16 rounding or overflow in between). Split-K
/// partial sums are f32.
pub fn conv2d_backward_weight_as(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gw: &wgpu::Buffer,
    out: DType,
) -> Result<()> {
    let (oh, ow) = check_grad_desc(ctx, d)?;
    if out != DType::F32 && out != d.dtype {
        return Err(crate::error::GpuError::DType { op: "conv2d_backward_weight output", dtype: out });
    }
    let dt = d.dtype;
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let khkw = kh * kw;
    let wn = prod(&[d.c_out, cin_g, khkw])?;
    check_buf(x, prod(&[d.n, d.c_in, d.h, d.w])?, dt, "conv2d_backward_weight input")?;
    check_buf(grad, prod(&[d.n, d.c_out, oh, ow])?, dt, "conv2d_backward_weight grad")?;
    check_buf(gw, wn, out, "conv2d_backward_weight output")?;
    if wn == 0 {
        return Ok(());
    }
    let (ohow, np) = (oh * ow, prod(&[d.n, oh, ow])?);
    if np == 0 {
        ctx.clear(gw, out.bytes(wn));
        return Ok(());
    }
    // f32 partials (several chunks, or an f16 result) end in a column reduction or a cast into `gw`.
    let finish = |tmp: &crate::GpuBuffer, parts: usize| -> Result<()> {
        if parts > 1 {
            let op = crate::kernels::reduce::ReduceOp::Sum;
            crate::kernels::reduce::reduce_as(ctx, op, (tmp.raw(), DType::F32), (gw, out), 1, parts, wn)
        } else {
            crate::kernels::layout::cast(ctx, tmp.raw(), gw, wn, true)
        }
    };
    if d.is_depthwise() && khkw <= MAX_DEPTHWISE_TAPS {
        // Enough workgroups to fill the GPU, at least 4096 pixels each.
        let chunks =
            1024usize.div_ceil(d.c_out).clamp(1, np.div_ceil(4096)).min(MAX_PARTIAL_ELEMS / wn).max(1);
        let ps = np.div_ceil(chunks);
        let chunks = np.div_ceil(ps);
        let tmp = if chunks > 1 || out != DType::F32 {
            Some(ctx.alloc(DType::F32.bytes(prod(&[chunks, wn])?))?)
        } else {
            None
        };
        let target = tmp.as_ref().map_or(gw, |t| t.raw());
        let spec = d.win.spec().u32("MULT", (d.c_out / d.c_in) as u32).u32("MODE", 1).f16(dt == DType::F16);
        let k = ctx.kernel(&DEPTHWISE_GRAD, spec)?;
        let p = Params::new().uz(d.c_out)?.uz(chunks)?.uz(ps)?.uz(np)?.uz(ohow)?.uz(ow)?.uz(d.h)?.uz(d.w)?;
        ctx.dispatch(&k, &[grad, x, target], &p, ctx.grid_1d((d.c_out * chunks) as u64 * 256, 256)?)?;
        if let Some(t) = &tmp {
            finish(t, chunks)?;
        }
        return Ok(());
    }
    let ncols = cin_g * khkw;
    let mut g = GemmDesc::new(cout_g, ncols, np, false, false, 1);
    g.batch = (1, d.groups);
    g.lda = ohow;
    g.ldb = ncols;
    g.ldc = ncols;
    g.a = BatchStrides { offset: 0, outer: 0, inner: cout_g * ohow };
    g.b = BatchStrides { offset: 0, outer: 0, inner: cin_g * d.h * d.w };
    g.c = BatchStrides { offset: 0, outer: 0, inner: cout_g * ncols };
    g.dtype = dt;
    if coop_grad(ctx, d)
        && !stem_shaped(cout_g, ncols)
        && coop::encode_conv_backward_weight(ctx, d, &g, out, grad, x, gw)?
    {
        return Ok(());
    }
    let cfg = pick_config_weight_grad(ctx, cout_g, ncols);
    gemm::check_config(ctx, cfg)?;
    let bk = cfg.bk as usize;
    // Split K when the weight alone has too few tiles: aim for ~512 workgroups, each at least 16 K tiles deep.
    let tiles = cout_g.div_ceil(cfg.bm as usize) * ncols.div_ceil(cfg.bn as usize) * d.groups;
    // (per 256 invocations: narrower workgroups get proportionally more)
    let want = (512 * 256 / cfg.threads() as usize)
        .min(1024)
        .div_ceil(tiles)
        .min(np.div_ceil(16 * bk))
        .min(MAX_PARTIAL_ELEMS / wn)
        .min(65535 / d.groups);
    let ks = np.div_ceil(want.max(1)).next_multiple_of(bk);
    let splits = np.div_ceil(ks);
    // The kernel writes `C` as f32 for partial slabs and for an f32 result of f16 inputs.
    let c_f32 = splits > 1 || out == DType::F32;
    let tmp = if splits > 1 { Some(ctx.alloc(DType::F32.bytes(prod(&[splits, wn])?))?) } else { None };
    let target = tmp.as_ref().map_or(gw, |t| t.raw());
    let check = !cout_g.is_multiple_of(cfg.bm as usize)
        || !ncols.is_multiple_of(cfg.bn as usize)
        || !np.is_multiple_of(bk);
    let spec = base_spec(&g, cfg)
        .bool("VEC_A", ohow.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", ncols.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 3)
        .bool("SPLIT", splits > 1)
        .bool("OUT_F32", c_f32 && dt == DType::F16);
    let kernel = ctx.kernel(&GEMM, weight_grad_spec(conv_spec(spec, d), d, oh, ow))?;
    let p = gemm_params(&g)?
        .at(24)
        .uz(d.h)?
        .uz(d.w)?
        .uz(oh)?
        .uz(ow)?
        .uz(splits)?
        .uz(ks)?
        .uz(wn)?
        .uz(ohow)?
        .uz(d.c_out * ohow)?
        .uz(d.c_in * d.h * d.w)?;
    let groups = [
        ncols.div_ceil(cfg.bn as usize) as u32,
        cout_g.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.groups * splits).unwrap_or(u32::MAX),
    ];
    ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, groups)?;
    if let Some(t) = &tmp {
        finish(t, splits)?;
    }
    Ok(())
}
