//! The weight gradient ([`conv2d_backward_weight_as`]): implicit GEMM (`CONV_MODE` 3) over `K = N·OH·OW`,
//! split along `K` into f32 partial slabs when the weight has too few tiles, and the direct depthwise kernel.

use super::depthwise::DEPTHWISE_GRAD;
use super::{Conv2dDesc, check_grad_desc, conv_spec, coop_grad};
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::kernels::gemm::{self, BatchStrides, GEMM, GemmDesc, TileConfig, base_spec, gemm_params};
use crate::kernels::{check_buf, coop, prod};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Sums `splits` f32 partial weight-gradient slabs (`[splits, wn / 4]` of `vec4<f32>`) into `GW` (`vec4<T>`,
/// f32 or f16): `@workgroup_size(8, 32)` reduces 8 contiguous `vec4` columns (32 f32s = 128 B coalesced
/// sector) per workgroup with 32 lanes streaming along `splits`. Params: 0 = `wn / 4`, 1 = `splits`.
fn gen_wgrad_reduce(_: &Spec) -> String {
    "@group(0) @binding(1) var<storage, read> PART: array<vec4<f32>>;\n\
     @group(0) @binding(2) var<storage, read_write> GW: array<vec4<T>>;\n\
     var<workgroup> smem: array<vec4<f32>, 256>;\n\
     @compute @workgroup_size(8, 32)\n\
     fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
     \x20       @builtin(local_invocation_id) lid: vec3<u32>) {\n\
     \x20 let wn4 = pu(0u); let splits = pu(1u);\n\
     \x20 let col = (wid.y * nwg.x + wid.x) * 8u + lid.x;\n\
     \x20 var acc = vec4<f32>(0.0);\n\
     \x20 if (col < wn4) {\n\
     \x20   for (var s = lid.y; s < splits; s += 32u) { acc += PART[s * wn4 + col]; }\n\
     \x20 }\n\
     \x20 let slot = lid.y * 8u + lid.x;\n\
     \x20 smem[slot] = acc;\n\
     \x20 workgroupBarrier();\n\
     \x20 if (lid.y < 16u) { smem[slot] += smem[slot + 128u]; }\n\
     \x20 workgroupBarrier();\n\
     \x20 if (lid.y < 8u) { smem[slot] += smem[slot + 64u]; }\n\
     \x20 workgroupBarrier();\n\
     \x20 if (lid.y < 4u) { smem[slot] += smem[slot + 32u]; }\n\
     \x20 workgroupBarrier();\n\
     \x20 if (lid.y < 2u) { smem[slot] += smem[slot + 16u]; }\n\
     \x20 workgroupBarrier();\n\
     \x20 if (lid.y == 0u && col < wn4) { GW[col] = vec4<T>(smem[lid.x] + smem[lid.x + 8u]); }\n}\n"
        .into()
}

pub(super) static WGRAD_REDUCE: KernelDef = KernelDef {
    name: "conv2d_wgrad_reduce",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_wgrad_reduce),
};

/// Sums `splits` f32 partial slabs (`[splits, wn]`) into `gw` stored as `out`, using the `vec4` weight-gradient
/// reducer when `wn` is a multiple of 4 (falling back to `reduce::reduce_as` for odd `wn`, or a cast when
/// `splits == 1`).
pub(crate) fn finish_wgrad(
    ctx: &GpuContext,
    tmp: &wgpu::Buffer,
    splits: usize,
    wn: usize,
    gw: &wgpu::Buffer,
    out: DType,
) -> Result<()> {
    if splits > 1 {
        if wn.is_multiple_of(4) {
            let wn4 = wn / 4;
            let k = ctx.kernel(&WGRAD_REDUCE, Spec::new().f16(out == DType::F16))?;
            let p = Params::new().uz(wn4)?.uz(splits)?;
            let groups = ctx.grid_1d(wn4.div_ceil(8) as u64 * 256, 256)?;
            ctx.dispatch(&k, &[tmp, gw], &p, groups)
        } else {
            let op = crate::kernels::reduce::ReduceOp::Sum;
            crate::kernels::reduce::reduce_as(ctx, op, (tmp, DType::F32), (gw, out), 1, splits, wn)
        }
    } else {
        crate::kernels::layout::cast(ctx, tmp, gw, wn, true)
    }
}

/// Whether a 1×1 stride-1 unpadded weight gradient can run as a division-free per-image split-K `A · Bᵀ` GEMM
/// (`K = H·W`, contiguous `vec4` loads on both `dY` and `X`).
pub(crate) fn can_pointwise_wgrad(d: &Conv2dDesc, bk: usize, wn: usize) -> bool {
    let hw = d.h * d.w;
    d.is_pointwise()
        && !d.is_depthwise()
        && hw >= bk
        && d.n.saturating_mul(d.groups) <= 65535
        && d.n.saturating_mul(wn) <= MAX_PARTIAL_ELEMS
}

/// `(sp, chunk_p)` for per-image spatial split-K weight gradients: `sp` chunks of `chunk_p` output pixels per
/// image (`total_slabs = d.n * sp`).
pub(crate) fn plan_wgrad_per_img(
    d: &Conv2dDesc,
    ohow: usize,
    tiles: usize,
    bk: usize,
    threads: usize,
    wn: usize,
    min_chunk_tiles: usize,
) -> (usize, usize) {
    let target = (512 * 256 / threads).min(1024);
    let base_groups = tiles.saturating_mul(d.n);
    let max_sp = ohow
        .div_ceil(min_chunk_tiles * bk)
        .min((MAX_PARTIAL_ELEMS / d.n.saturating_mul(wn).max(1)).max(1))
        .min(65535 / d.n.saturating_mul(d.groups).max(1))
        .max(1);
    let want = target.div_ceil(base_groups.max(1)).min(max_sp).max(1);
    if want <= 1 {
        return (1, ohow);
    }
    let chunk_p = ohow.div_ceil(want).next_multiple_of(bk);
    (ohow.div_ceil(chunk_p), chunk_p)
}

/// `(sp, chunk_p)` for the pointwise per-image split-K weight gradient.
pub(crate) fn plan_pointwise_wgrad(
    d: &Conv2dDesc,
    bm: usize,
    bn: usize,
    bk: usize,
    threads: usize,
    wn: usize,
) -> (usize, usize) {
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let tiles = cout_g.div_ceil(bm) * cin_g.div_ceil(bn) * d.groups;
    plan_wgrad_per_img(d, d.h * d.w, tiles, bk, threads, wn, 8)
}

/// 32×32 tiles, depth 16, 4×4 per thread, 64 threads (8 KiB): narrow 1×1 weight gradients.
const PW32: TileConfig = TileConfig { bm: 32, bn: 32, bk: 16, tm: 4, tn: 4 };
/// 32×64 tiles, depth 16, 4×4 per thread, 128 threads (12 KiB).
const PW32X64: TileConfig = TileConfig { bm: 32, bn: 64, bk: 16, tm: 4, tn: 4 };
/// 64×32 tiles, depth 16, 4×4 per thread, 128 threads (12 KiB).
const PW64X32: TileConfig = TileConfig { bm: 64, bn: 32, bk: 16, tm: 4, tn: 4 };

fn pick_config_pointwise_wgrad(ctx: &GpuContext, m: usize, n: usize) -> TileConfig {
    let cfg = match (m, n) {
        (0..=32, 0..=32) => PW32,
        (0..=32, 33..=64) => PW32X64,
        (33..=64, 0..=32) => PW64X32,
        _ => return pick_config_weight_grad(ctx, m, n),
    };
    if cfg.workgroup_bytes() <= ctx.caps().max_workgroup_storage { cfg } else { TileConfig::SMALL }
}

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

/// 16×32 tiles, depth 32, 4×4 per thread, 32 threads (13.3 KiB padded): the stem (3→16, `N = 27`), whose
/// 1.6 M-deep reduction split-K spreads over a thousand workgroups.
const STEM: TileConfig = TileConfig { bm: 16, bn: 32, bk: 32, tm: 4, tn: 4 };

/// Whether the weight gradient of an `m × n` weight takes the regular kernels' [`STEM`] tile even on tensor
/// cores (their narrowest tile is 16×256, 89 % padding at `N = 27`).
pub(crate) fn stem_shaped(m: usize, n: usize) -> bool {
    m <= 16 && n <= 32
}

/// Whether the regular weight-gradient kernel with tile `cfg` pads its `B` tile rows by one vec4 (`WG_PAD`, against
/// bank conflicts in its stores): only when the padded tile still fits `limit` bytes of workgroup memory. The
/// tile choice and [`gemm::check_config`] count the unpadded tile, which stays the fallback.
pub(crate) fn wgrad_pad(cfg: TileConfig, limit: u32) -> bool {
    cfg.workgroup_bytes() + wgrad_pad_bytes(cfg) <= limit
}

/// Workgroup memory the `WG_PAD` row padding adds: one vec4 of f32 per row of both `B` tiles.
fn wgrad_pad_bytes(cfg: TileConfig) -> u32 {
    2 * cfg.bk * 16
}

/// A GEMM weight gradient's route, for tests: tensor cores, the tile's `(BM, BN)` and the split-K chunk count.
#[cfg(test)]
pub(crate) type WgradRoute = Option<(bool, (u32, u32), usize)>;

#[cfg(test)]
thread_local! {
    /// The route of this thread's last GEMM weight gradient.
    pub(crate) static LAST_ROUTE: std::cell::Cell<WgradRoute> = const { std::cell::Cell::new(None) };
}

/// Largest `KH · KW` the direct depthwise weight-gradient kernel unrolls; bigger kernels use the GEMM path.
pub(super) const MAX_DEPTHWISE_TAPS: usize = 49;
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

/// Distinct `w.jj` columns of `plan`, in order of appearance.
fn wgrad_cols(plan: &WgradPlan) -> Vec<&str> {
    let mut cols: Vec<&str> = Vec::new();
    for w in &plan.units {
        if !cols.iter().any(|c| *c == w.jj) {
            cols.push(&w.jj);
        }
    }
    cols
}

/// Loop-invariant column setup hoisted before the K loop of `CONV_MODE` 3 when a thread owns at most 4 distinct
/// columns (`C64X128`, `C64X64`, `C32X128`, `C32X32`, `LARGE`, `SMALL`): decodes `j = (ci, kh, kw)` once instead
/// of repeating `/ (KH * KW)` and `% (KH * KW)` on every K tile.
pub(crate) fn wgrad_b_setup(plan: &WgradPlan, per_img: bool) -> String {
    use std::fmt::Write as _;
    let cols = wgrad_cols(plan);
    if cols.len() > 4 {
        return String::new();
    }
    let prefix = if per_img { "offB + " } else { "" };
    let mut s = String::new();
    for (c, jj) in cols.iter().enumerate() {
        let _ = writeln!(
            s,
            "  let wj{c} = n0 + {jj}; let wjok{c} = wj{c} < N;\n\
             \x20 let wci{c} = wj{c} / (KH * KW); let wtap{c} = wj{c} - wci{c} * (KH * KW);\n\
             \x20 let wdh{c} = i32((wtap{c} / KW) * DH); let wdw{c} = i32((wtap{c} % KW) * DW);\n\
             \x20 let wcb{c} = {prefix}wci{c} * (IH * IW);"
        );
    }
    s
}

/// The WGSL of a weight gradient's B-tile load (from `buf`, converted by `cast`): each distinct run of `units` gets
/// its output pixel decoded once (`per_img` drops `/ OHOW` and `% OHOW` since `offB` already targets image `b1`),
/// then each unit gathers through tap `j = (ci, kh, kw)` into `reg(u)` with unsigned bounds checks.
pub(crate) fn wgrad_b_load(
    plan: &WgradPlan,
    buf: &str,
    reg: impl Fn(usize) -> String,
    cast: &str,
    zero: &str,
    per_img: bool,
    check: bool,
) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let mut runs: Vec<&str> = Vec::new();
    let cols = wgrad_cols(plan);
    let hoisted = cols.len() <= 4;
    let run = plan.run;
    for (u, w) in plan.units.iter().enumerate() {
        let q = match runs.iter().position(|k| *k == w.kq) {
            Some(q) => q,
            None => {
                runs.push(&w.kq);
                let q = runs.len() - 1;
                let kok = if check { format!(" let wkok{q} = wk{q} < K;") } else { String::new() };
                if per_img {
                    let _ = writeln!(
                        s,
                        "    let wk{q} = k0 + {} * {run}u + wl;{kok}\n\
                         \x20   let wih{q} = i32((wk{q} / OW) * SH) - PH; let wiw{q} = i32((wk{q} % OW) * SW) - PW;",
                        w.kq
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "    let wk{q} = k0 + {} * {run}u + wl;{kok} let wn{q} = wk{q} / OHOW;\n\
                         \x20   let wr{q} = wk{q} - wn{q} * OHOW; let wb{q} = offB + wn{q} * SBN;\n\
                         \x20   let wih{q} = i32((wr{q} / OW) * SH) - PH; let wiw{q} = i32((wr{q} % OW) * SW) - PW;",
                        w.kq
                    );
                }
                q
            }
        };
        let k_cond = if check { format!("wkok{q} && ") } else { String::new() };
        if hoisted {
            let c = cols.iter().position(|jj| *jj == w.jj).unwrap();
            let j_cond = if check { format!("wjok{c} && ") } else { String::new() };
            let wb_pre = if per_img { String::new() } else { format!("wb{q} + ") };
            let _ = writeln!(
                s,
                "    {{ let ih = wih{q} + wdh{c}; let iw = wiw{q} + wdw{c};\n\
                 \x20     if ({k_cond}{j_cond}u32(ih) < IH && u32(iw) < IW) {{\n\
                 \x20       {} = {cast}({buf}[{wb_pre}wcb{c} + u32(ih) * IW + u32(iw)]);\n\
                 \x20     }} else {{ {} = {zero}; }} }}",
                reg(u),
                reg(u)
            );
        } else {
            let j_cond = if check { "j < N && " } else { "" };
            let wb_var = if per_img { "offB".to_string() } else { format!("wb{q}") };
            let _ = writeln!(
                s,
                "    {{ let j = n0 + {}; let ci = j / (KH * KW); let r = j - ci * (KH * KW);\n\
                 \x20     let ih = wih{q} + i32((r / KW) * DH); let iw = wiw{q} + i32((r % KW) * DW);\n\
                 \x20     if ({k_cond}{j_cond}u32(ih) < IH && u32(iw) < IW) {{\n\
                 \x20       {} = {cast}({buf}[{wb_var} + (ci * IH + u32(ih)) * IW + u32(iw)]);\n\
                 \x20     }} else {{ {} = {zero}; }} }}",
                w.jj,
                reg(u),
                reg(u)
            );
        }
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
        finish_wgrad(ctx, tmp.raw(), parts, wn, gw, out)
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
    // Formula 2: division-free contiguous `vec4` batched/split-K `A · Bᵀ` GEMM for 1×1 s1 p0 weight gradients.
    let pw_cfg = pick_config_pointwise_wgrad(ctx, cout_g, cin_g);
    if can_pointwise_wgrad(d, pw_cfg.bk as usize, wn) {
        gemm::check_config(ctx, pw_cfg)?;
        let hw = ohow;
        let (sp, chunk_p) = plan_pointwise_wgrad(
            d,
            pw_cfg.bm as usize,
            pw_cfg.bn as usize,
            pw_cfg.bk as usize,
            pw_cfg.threads() as usize,
            wn,
        );
        let total_slabs = d.n * sp;
        #[cfg(test)]
        LAST_ROUTE.set(Some((false, (pw_cfg.bm, pw_cfg.bn), total_slabs)));
        let mut gp = GemmDesc::new(cout_g, cin_g, hw, false, true, d.n);
        gp.batch = (d.n, d.groups);
        gp.lda = hw;
        gp.ldb = hw;
        gp.ldc = cin_g;
        gp.a = BatchStrides { offset: 0, outer: d.c_out * hw, inner: cout_g * hw };
        gp.b = BatchStrides { offset: 0, outer: d.c_in * hw, inner: cin_g * hw };
        gp.c = BatchStrides { offset: 0, outer: wn, inner: cout_g * cin_g };
        gp.dtype = dt;
        let c_f32 = total_slabs > 1 || out == DType::F32;
        let tmp = if total_slabs > 1 {
            Some(ctx.alloc(DType::F32.bytes(prod(&[total_slabs, wn])?))?)
        } else {
            None
        };
        let target = tmp.as_ref().map_or(gw, |t| t.raw());
        let check = !cout_g.is_multiple_of(pw_cfg.bm as usize)
            || !cin_g.is_multiple_of(pw_cfg.bn as usize)
            || !hw.is_multiple_of(pw_cfg.bk as usize);
        let spec = base_spec(&gp, pw_cfg)
            .bool("VEC_A", hw.is_multiple_of(4))
            .bool("VEC_B", hw.is_multiple_of(4))
            .bool("VEC_C", cin_g.is_multiple_of(4))
            .bool("TRANS_A", false)
            .bool("TRANS_B", true)
            .bool("CHECK", check)
            .u32("CONV_MODE", 0)
            .bool("SPLIT", sp > 1)
            .bool("OUT_F32", c_f32 && dt == DType::F16);
        let kernel = ctx.kernel(&GEMM, spec)?;
        let p = if sp > 1 {
            gemm_params(&gp)?.at(28).uz(sp)?.uz(chunk_p)?.uz(d.n * wn)?
        } else {
            gemm_params(&gp)?
        };
        let groups = [
            cin_g.div_ceil(pw_cfg.bn as usize) as u32,
            cout_g.div_ceil(pw_cfg.bm as usize) as u32,
            u32::try_from(d.n * d.groups * sp).unwrap_or(u32::MAX),
        ];
        ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, groups)?;
        if let Some(t) = &tmp {
            finish(t, total_slabs)?;
        }
        return Ok(());
    }
    let cfg = pick_config_weight_grad(ctx, cout_g, ncols);
    gemm::check_config(ctx, cfg)?;
    let pad = wgrad_pad(cfg, ctx.caps().max_workgroup_storage);
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
    let per_img = ohow >= bk
        && d.n.saturating_mul(d.groups) <= 65535
        && d.n.saturating_mul(wn) <= MAX_PARTIAL_ELEMS
        && want >= d.n;
    if per_img {
        let (sp, chunk_p) = plan_wgrad_per_img(d, ohow, tiles, bk, cfg.threads() as usize, wn, 16);
        let total_slabs = d.n * sp;
        #[cfg(test)]
        LAST_ROUTE.set(Some((false, (cfg.bm, cfg.bn), total_slabs)));
        let mut gp = GemmDesc::new(cout_g, ncols, ohow, false, false, d.n);
        gp.batch = (d.n, d.groups);
        gp.lda = ohow;
        gp.ldb = ncols;
        gp.ldc = ncols;
        gp.a = BatchStrides { offset: 0, outer: d.c_out * ohow, inner: cout_g * ohow };
        gp.b = BatchStrides { offset: 0, outer: d.c_in * d.h * d.w, inner: cin_g * d.h * d.w };
        gp.c = BatchStrides { offset: 0, outer: wn, inner: cout_g * ncols };
        gp.dtype = dt;
        let c_f32 = total_slabs > 1 || out == DType::F32;
        let tmp = if total_slabs > 1 {
            Some(ctx.alloc(DType::F32.bytes(prod(&[total_slabs, wn])?))?)
        } else {
            None
        };
        let target = tmp.as_ref().map_or(gw, |t| t.raw());
        let check = !cout_g.is_multiple_of(cfg.bm as usize)
            || !ncols.is_multiple_of(cfg.bn as usize)
            || !ohow.is_multiple_of(bk);
        let spec = base_spec(&gp, cfg)
            .bool("VEC_A", ohow.is_multiple_of(4))
            .bool("VEC_B", false)
            .bool("VEC_C", ncols.is_multiple_of(4))
            .bool("TRANS_A", false)
            .bool("TRANS_B", false)
            .bool("CHECK", check)
            .u32("CONV_MODE", 3)
            .bool("PER_IMG", true)
            .bool("WG_PAD", pad)
            .bool("SPLIT", sp > 1)
            .bool("OUT_F32", c_f32 && dt == DType::F16);
        let kernel = ctx.kernel(&GEMM, weight_grad_spec(conv_spec(spec, d), d, oh, ow))?;
        let p = gemm_params(&gp)?
            .at(24)
            .uz(d.h)?
            .uz(d.w)?
            .uz(oh)?
            .uz(ow)?
            .uz(sp)?
            .uz(chunk_p)?
            .uz(d.n * wn)?
            .uz(ohow)?;
        let groups = [
            ncols.div_ceil(cfg.bn as usize) as u32,
            cout_g.div_ceil(cfg.bm as usize) as u32,
            u32::try_from(d.n * d.groups * sp).unwrap_or(u32::MAX),
        ];
        ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, groups)?;
        if let Some(t) = &tmp {
            finish(t, total_slabs)?;
        }
        return Ok(());
    }
    let ks = np.div_ceil(want.max(1)).next_multiple_of(bk);
    let splits = np.div_ceil(ks);
    #[cfg(test)]
    LAST_ROUTE.set(Some((false, (cfg.bm, cfg.bn), splits)));
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
        .bool("WG_PAD", pad)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::gemm::gen_gemm;

    /// Workgroup bytes a generated GEMM shader declares (its `var<workgroup>` arrays of arrays of `vec4<f32>`).
    fn declared_workgroup_bytes(wgsl: &str) -> u32 {
        wgsl.lines()
            .filter_map(|l| l.strip_prefix("var<workgroup> "))
            .map(|l| {
                let dims = l.split_once("vec4<f32>, ").expect("vec4 tile").1;
                let (inner, rest) = dims.split_once(">, ").expect("inner length");
                let outer = rest.split_once('>').expect("outer length").0;
                16 * inner.parse::<u32>().unwrap() * outer.parse::<u32>().unwrap()
            })
            .sum()
    }

    /// The mode-3 shader never declares more workgroup memory than the limit its tile was validated against:
    /// `WG_PAD` is on only where the padded tile fits, and the generator adds exactly [`wgrad_pad_bytes`].
    #[test]
    fn weight_grad_pad_fits_the_validated_limit() {
        let tiles = [TileConfig::LARGE, TileConfig::MEDIUM, TileConfig::SMALL, TileConfig::ROWS32, STEM];
        for limit in [16384, 32768, 49152] {
            for cfg in tiles.into_iter().filter(|c| c.workgroup_bytes() <= limit) {
                let pad = wgrad_pad(cfg, limit);
                let g = GemmDesc::new(cfg.bm as usize, cfg.bn as usize, 4096, false, false, 1);
                let spec = base_spec(&g, cfg).u32("CONV_MODE", 3).bool("SPLIT", true).bool("WG_PAD", pad);
                let bytes = declared_workgroup_bytes(&gen_gemm(&spec));
                assert_eq!(bytes, cfg.workgroup_bytes() + u32::from(pad) * wgrad_pad_bytes(cfg), "{cfg:?}");
                assert!(bytes <= limit, "{cfg:?}: {bytes} B of workgroup memory over {limit}");
            }
        }
        // f32 x [1, 8, 8, 8], w [32, 8, 3, 3] on a 16 KiB device: SMALL is exactly 16 KiB unpadded.
        assert_eq!(TileConfig::SMALL.workgroup_bytes(), 16384);
        assert!(!wgrad_pad(TileConfig::SMALL, 16384));
        assert!(wgrad_pad(TileConfig::SMALL, 32768));
    }
}
