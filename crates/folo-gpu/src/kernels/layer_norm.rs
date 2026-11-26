//! Fused WGSL LayerNorm and RMSNorm forward and backward kernels over the last dimension (`[outer, d]`).
//!
//! Forward (`norm_fwd`): one 256-thread workgroup per row, `vec4<f32>` loads/stores when `d % 4 == 0`,
//! per-thread register caching up to 4 units (`d <= 4096` with `VEC4`, `d <= 1024` scalar), 1-barrier
//! 32-wide subgroup reductions with a portable 8-barrier shared-memory tree fallback, and scale-normalized
//! variance/RMS fallback for extreme finite magnitudes (`1e20`, `1e-23`, `eps = 0`).
//!
//! Backward:
//! 1. `norm_bwd_dx`: one 256-thread workgroup per row computes `dx` (tracking `sum(x_hat)` in LayerNorm to
//!    preserve large-offset translation invariance `sum(dx) == 0`), register-cached when `units <= 1024`.
//! 2. `norm_bwd_dwb` / `norm_bwd_single`: coalesced `(32, 8)` column reduction over `outer` for `dweight` and
//!    `dbias`, with split-K when `outer` is large and column tiles are few, finishing via `reduce::reduce_as`.

use super::{check_buf, prod, reduce, use_subgroups, vec_prelude};
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Max per-row units (`d / 4` when `VEC4`, else `d`) cached in per-thread registers (4 units × 256 threads).
const NORM_ROW_CACHE_MAX: usize = 1024;
/// Minimum `outer` rows before splitting `dweight` / `dbias` column reduction along `outer`.
const DWB_SPLIT_MIN_OUTER: usize = 256;
/// Minimum rows per split chunk for `dweight` / `dbias`.
const DWB_MIN_CHUNK: usize = 32;

const REDUCE_HELPERS_SUBGROUPS: &str = r#"
var<workgroup> sh_a: array<f32, 8>;
var<workgroup> sh_b: array<f32, 8>;
var<workgroup> sh_c: array<f32, 8>;

fn reduce_sum_max(lid: u32, sgid: u32, lane: u32, s: f32, mx: f32) -> vec2<f32> {
    let rs = subgroupAdd(s);
    let rm = subgroupMax(mx);
    let sg = min(sgid, 7u);
    if (lane == 0u) {
        sh_a[sg] = rs;
        sh_b[sg] = rm;
    }
    workgroupBarrier();
    var vs = sh_a[lane & 7u];
    var vm = sh_b[lane & 7u];
    vs += subgroupShuffleXor(vs, 4u);
    vm = max(vm, subgroupShuffleXor(vm, 4u));
    vs += subgroupShuffleXor(vs, 2u);
    vm = max(vm, subgroupShuffleXor(vm, 2u));
    vs += subgroupShuffleXor(vs, 1u);
    vm = max(vm, subgroupShuffleXor(vm, 1u));
    workgroupBarrier();
    return vec2<f32>(vs, vm);
}

fn reduce_sum1(lid: u32, sgid: u32, lane: u32, s: f32) -> f32 {
    let rs = subgroupAdd(s);
    let sg = min(sgid, 7u);
    if (lane == 0u) { sh_a[sg] = rs; }
    workgroupBarrier();
    var vs = sh_a[lane & 7u];
    vs += subgroupShuffleXor(vs, 4u);
    vs += subgroupShuffleXor(vs, 2u);
    vs += subgroupShuffleXor(vs, 1u);
    workgroupBarrier();
    return vs;
}

fn reduce_sum3(lid: u32, sgid: u32, lane: u32, a: f32, b: f32, c: f32) -> vec3<f32> {
    let ra = subgroupAdd(a);
    let rb = subgroupAdd(b);
    let rc = subgroupAdd(c);
    let sg = min(sgid, 7u);
    if (lane == 0u) {
        sh_a[sg] = ra;
        sh_b[sg] = rb;
        sh_c[sg] = rc;
    }
    workgroupBarrier();
    var va = sh_a[lane & 7u];
    var vb = sh_b[lane & 7u];
    var vc = sh_c[lane & 7u];
    va += subgroupShuffleXor(va, 4u);
    vb += subgroupShuffleXor(vb, 4u);
    vc += subgroupShuffleXor(vc, 4u);
    va += subgroupShuffleXor(va, 2u);
    vb += subgroupShuffleXor(vb, 2u);
    vc += subgroupShuffleXor(vc, 2u);
    va += subgroupShuffleXor(va, 1u);
    vb += subgroupShuffleXor(vb, 1u);
    vc += subgroupShuffleXor(vc, 1u);
    return vec3<f32>(va, vb, vc);
}
"#;

const REDUCE_HELPERS_PORTABLE: &str = r#"
var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_b: array<f32, 256>;
var<workgroup> sh_c: array<f32, 256>;

fn reduce_sum_max(lid: u32, sgid: u32, lane: u32, s: f32, mx: f32) -> vec2<f32> {
    sh_a[lid] = s;
    sh_b[lid] = mx;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step >>= 1u) {
        if (lid < step) {
            sh_a[lid] += sh_a[lid + step];
            sh_b[lid] = max(sh_b[lid], sh_b[lid + step]);
        }
        workgroupBarrier();
    }
    let res = vec2<f32>(sh_a[0], sh_b[0]);
    workgroupBarrier();
    return res;
}

fn reduce_sum1(lid: u32, sgid: u32, lane: u32, s: f32) -> f32 {
    sh_a[lid] = s;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step >>= 1u) {
        if (lid < step) { sh_a[lid] += sh_a[lid + step]; }
        workgroupBarrier();
    }
    let res = sh_a[0];
    workgroupBarrier();
    return res;
}

fn reduce_sum3(lid: u32, sgid: u32, lane: u32, a: f32, b: f32, c: f32) -> vec3<f32> {
    sh_a[lid] = a;
    sh_b[lid] = b;
    sh_c[lid] = c;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step >>= 1u) {
        if (lid < step) {
            sh_a[lid] += sh_a[lid + step];
            sh_b[lid] += sh_b[lid + step];
            sh_c[lid] += sh_c[lid + step];
        }
        workgroupBarrier();
    }
    let res = vec3<f32>(sh_a[0], sh_b[0], sh_c[0]);
    workgroupBarrier();
    return res;
}
"#;

const LANE_HELPERS: &str = r#"
fn lane_sum(v: vec4<f32>) -> f32 {
    if (VEC) { return (v.x + v.y) + (v.z + v.w); }
    return v.x;
}
fn lane_amax(v: vec4<f32>) -> f32 {
    let a = abs(v);
    if (VEC) { return max(max(a.x, a.y), max(a.z, a.w)); }
    return a.x;
}
fn is_finite1(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u;
}
fn is_inf1(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u;
}
"#;

const NORM_FWD_BODY: &str = include_str!("../shaders/layer_norm_fwd.wgsl");

const NORM_FWD_ENTRY_SG: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    norm_fwd_run(wid, nwg, lid, sgid, lane);
}
"#;

const NORM_FWD_ENTRY_PORTABLE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    norm_fwd_run(wid, nwg, lid, 0u, 0u);
}
"#;

const NORM_BWD_DX_BODY: &str = include_str!("../shaders/layer_norm_bwd_dx.wgsl");

const NORM_BWD_DX_ENTRY_SG: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    norm_bwd_dx_run(wid, nwg, lid, sgid, lane);
}
"#;

const NORM_BWD_DX_ENTRY_PORTABLE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    norm_bwd_dx_run(wid, nwg, lid, 0u, 0u);
}
"#;

/// Coalesced `(32, 8)` column reduction for `dweight` and/or `dbias`.
/// Params: 0 = outer, 1 = units_d (`d / 4` when VEC4 else `d`), 2 = col_groups (`ceil(units_d / 32)`),
/// 3 = splits, 4 = chunk_rows (`ceil(outer / splits)`).
fn gen_bwd_dwb(spec: &Spec, two_out: bool) -> String {
    let vec = spec.get_bool("VEC4");
    let pre = vec_prelude(vec);
    let body = if two_out {
        r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> RSTD: array<f32>;
@group(0) @binding(5) var<storage, read_write> DW: array<E>;
@group(0) @binding(6) var<storage, read_write> DB: array<E>;

var<workgroup> sh_w: array<vec4<f32>, 256>;
var<workgroup> sh_b: array<vec4<f32>, 256>;

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.y * nwg.x + wid.x;
    let outer = pu(0u);
    let units = pu(1u);
    let col_groups = pu(2u);
    let splits = pu(3u);
    let chunk = pu(4u);
    let sidx = g / col_groups;
    let col = (g % col_groups) * 32u + lid.x;
    var aw = vec4<f32>(0.0);
    var ab = vec4<f32>(0.0);
    let valid = sidx < splits && col < units;
    if (valid) {
        let start = sidx * chunk;
        let end = min(start + chunk, outer);
        for (var o = start + lid.y; o < end; o += 8u) {
            let idx = o * units + col;
            let gv = V4(G[idx]);
            var xv = V4(X[idx]);
            if (!RMS) { xv -= vec4<f32>(MEAN[o]); }
            aw += gv * (xv * RSTD[o]);
            ab += gv;
        }
    }
    let slot = lid.y * 32u + lid.x;
    sh_w[slot] = aw;
    sh_b[slot] = ab;
    workgroupBarrier();
    if (lid.y == 0u && valid) {
        let sw = ((sh_w[lid.x] + sh_w[32u + lid.x]) + (sh_w[64u + lid.x] + sh_w[96u + lid.x]))
               + ((sh_w[128u + lid.x] + sh_w[160u + lid.x]) + (sh_w[192u + lid.x] + sh_w[224u + lid.x]));
        let sb = ((sh_b[lid.x] + sh_b[32u + lid.x]) + (sh_b[64u + lid.x] + sh_b[96u + lid.x]))
               + ((sh_b[128u + lid.x] + sh_b[160u + lid.x]) + (sh_b[192u + lid.x] + sh_b[224u + lid.x]));
        DW[sidx * units + col] = OUT(sw);
        DB[sidx * units + col] = OUT(sb);
    }
}
"#
    } else {
        // MODE 0 = dweight, MODE 1 = dbias.
        r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> RSTD: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUTB: array<E>;

var<workgroup> sh_v: array<vec4<f32>, 256>;

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.y * nwg.x + wid.x;
    let outer = pu(0u);
    let units = pu(1u);
    let col_groups = pu(2u);
    let splits = pu(3u);
    let chunk = pu(4u);
    let sidx = g / col_groups;
    let col = (g % col_groups) * 32u + lid.x;
    var acc = vec4<f32>(0.0);
    let valid = sidx < splits && col < units;
    if (valid) {
        let start = sidx * chunk;
        let end = min(start + chunk, outer);
        for (var o = start + lid.y; o < end; o += 8u) {
            let idx = o * units + col;
            let gv = V4(G[idx]);
            if (MODE == 0u) {
                var xv = V4(X[idx]);
                if (!RMS) { xv -= vec4<f32>(MEAN[o]); }
                acc += gv * (xv * RSTD[o]);
            } else {
                acc += gv;
            }
        }
    }
    let slot = lid.y * 32u + lid.x;
    sh_v[slot] = acc;
    workgroupBarrier();
    if (lid.y == 0u && valid) {
        let sv = ((sh_v[lid.x] + sh_v[32u + lid.x]) + (sh_v[64u + lid.x] + sh_v[96u + lid.x]))
               + ((sh_v[128u + lid.x] + sh_v[160u + lid.x]) + (sh_v[192u + lid.x] + sh_v[224u + lid.x]));
        OUTB[sidx * units + col] = OUT(sv);
    }
}
"#
    };
    format!("{pre}{body}")
}

fn gen_fwd(s: &Spec) -> String {
    let pre = vec_prelude(s.get_bool("VEC4"));
    let cache = s.get_bool("CACHE");
    let (red, entry) = if s.is_subgroups() {
        (REDUCE_HELPERS_SUBGROUPS, NORM_FWD_ENTRY_SG)
    } else {
        (REDUCE_HELPERS_PORTABLE, NORM_FWD_ENTRY_PORTABLE)
    };
    format!("{pre}const CACHE_ROW: bool = {cache};\n{LANE_HELPERS}{red}{NORM_FWD_BODY}{entry}")
}

fn gen_bwd_dx(s: &Spec) -> String {
    let pre = vec_prelude(s.get_bool("VEC4"));
    let cache = s.get_bool("CACHE");
    let (red, entry) = if s.is_subgroups() {
        (REDUCE_HELPERS_SUBGROUPS, NORM_BWD_DX_ENTRY_SG)
    } else {
        (REDUCE_HELPERS_PORTABLE, NORM_BWD_DX_ENTRY_PORTABLE)
    };
    format!("{pre}const CACHE_ROW: bool = {cache};\n{LANE_HELPERS}{red}{NORM_BWD_DX_BODY}{entry}")
}

use Access::{Read as R, ReadWrite as RW};

static NORM_FWD: KernelDef =
    KernelDef { name: "norm_fwd", storage: &[R, R, R, RW, RW, RW], source: Source::Generated(gen_fwd) };
static NORM_BWD_DX: KernelDef =
    KernelDef { name: "norm_bwd_dx", storage: &[R, R, R, R, R, RW], source: Source::Generated(gen_bwd_dx) };
static NORM_BWD_DWB: KernelDef = KernelDef {
    name: "norm_bwd_dwb",
    storage: &[R, R, R, R, RW, RW],
    source: Source::Generated(|s| gen_bwd_dwb(s, true)),
};
static NORM_BWD_PARAM: KernelDef = KernelDef {
    name: "norm_bwd_param",
    storage: &[R, R, R, R, RW],
    source: Source::Generated(|s| gen_bwd_dwb(s, false)),
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&NORM_FWD, &NORM_BWD_DX, &NORM_BWD_DWB, &NORM_BWD_PARAM];

/// Split plan `(splits, chunk_rows)` for column-reducing `[outer, d]` parameter gradients.
fn dwb_split_plan(outer: usize, units_d: usize) -> (usize, usize) {
    let col_groups = units_d.div_ceil(32);
    if outer < DWB_SPLIT_MIN_OUTER || col_groups >= 256 {
        return (1, outer);
    }
    let want = 512usize.div_ceil(col_groups);
    let max_s = (outer / DWB_MIN_CHUNK).max(1);
    let s0 = want.min(max_s);
    if s0 <= 1 {
        return (1, outer);
    }
    let chunk = outer.div_ceil(s0);
    let splits = outer.div_ceil(chunk);
    (splits, chunk)
}

/// Fused LayerNorm forward over dense f32 `[outer, d]`, writing `y [outer, d]`, `mean [outer]`, `rstd [outer]`.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    outer: usize,
    d: usize,
    eps: f32,
) -> Result<()> {
    let total = prod(&[outer, d])?;
    check_buf(x, total, DType::F32, "layer_norm input")?;
    check_buf(y, total, DType::F32, "layer_norm output")?;
    check_buf(mean, outer, DType::F32, "layer_norm mean")?;
    check_buf(rstd, outer, DType::F32, "layer_norm rstd")?;
    if let Some(w) = weight {
        check_buf(w, d, DType::F32, "layer_norm weight")?;
    }
    if let Some(b) = bias {
        check_buf(b, d, DType::F32, "layer_norm bias")?;
    }
    if total == 0 {
        return Ok(());
    }
    let vec4 = d.is_multiple_of(4);
    let units = if vec4 { d / 4 } else { d };
    let k = ctx.kernel(
        &NORM_FWD,
        Spec::new()
            .bool("RMS", false)
            .bool("HAS_W", weight.is_some())
            .bool("HAS_B", bias.is_some())
            .bool("VEC4", vec4)
            .bool("CACHE", units <= NORM_ROW_CACHE_MAX)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new().uz(outer)?.uz(units)?.f((1.0 / d as f64) as f32).f(eps).f(eps.sqrt());
    ctx.dispatch(
        &k,
        &[x, weight.unwrap_or(ctx.dummy()), bias.unwrap_or(ctx.dummy()), y, mean, rstd],
        &p,
        ctx.grid_1d(outer as u64 * 256, 256)?,
    )
}

/// Fused RMSNorm forward over dense f32 `[outer, d]`, writing `y [outer, d]` and `rstd [outer]`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_forward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    outer: usize,
    d: usize,
    eps: f32,
) -> Result<()> {
    let total = prod(&[outer, d])?;
    check_buf(x, total, DType::F32, "rms_norm input")?;
    check_buf(y, total, DType::F32, "rms_norm output")?;
    check_buf(rstd, outer, DType::F32, "rms_norm rstd")?;
    if let Some(w) = weight {
        check_buf(w, d, DType::F32, "rms_norm weight")?;
    }
    if total == 0 {
        return Ok(());
    }
    let vec4 = d.is_multiple_of(4);
    let units = if vec4 { d / 4 } else { d };
    let k = ctx.kernel(
        &NORM_FWD,
        Spec::new()
            .bool("RMS", true)
            .bool("HAS_W", weight.is_some())
            .bool("HAS_B", false)
            .bool("VEC4", vec4)
            .bool("CACHE", units <= NORM_ROW_CACHE_MAX)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new().uz(outer)?.uz(units)?.f((1.0 / d as f64) as f32).f(eps).f(eps.sqrt());
    // MEAN binding is never written when `RMS == true`; allocate a tiny scratch dummy so it never aliases `rstd`.
    let dummy_mean = ctx.alloc(4)?;
    ctx.dispatch(
        &k,
        &[x, weight.unwrap_or(ctx.dummy()), ctx.dummy(), y, dummy_mean.raw(), rstd],
        &p,
        ctx.grid_1d(outer as u64 * 256, 256)?,
    )
}

/// Shared implementation for LayerNorm (`!rms`) and RMSNorm (`rms`) backward.
#[allow(clippy::too_many_arguments)]
fn norm_backward_impl(
    ctx: &GpuContext,
    rms: bool,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    mean: Option<&wgpu::Buffer>,
    rstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    dbias: Option<&wgpu::Buffer>,
    outer: usize,
    d: usize,
) -> Result<()> {
    let op = if rms { "rms_norm_backward" } else { "layer_norm_backward" };
    let total = prod(&[outer, d])?;
    check_buf(x, total, DType::F32, op)?;
    check_buf(grad, total, DType::F32, op)?;
    check_buf(dx, total, DType::F32, op)?;
    check_buf(rstd, outer, DType::F32, op)?;
    if let Some(m) = mean {
        check_buf(m, outer, DType::F32, op)?;
    }
    if let Some(w) = weight {
        check_buf(w, d, DType::F32, op)?;
    }
    if let Some(dw) = dweight {
        check_buf(dw, d, DType::F32, op)?;
    }
    if let Some(db) = dbias {
        check_buf(db, d, DType::F32, op)?;
    }
    if outer == 0 || d == 0 {
        if let Some(dw) = dweight {
            ctx.clear(dw, DType::F32.bytes(d));
        }
        if let Some(db) = dbias {
            ctx.clear(db, DType::F32.bytes(d));
        }
        return Ok(());
    }
    let vec4 = d.is_multiple_of(4);
    let units = if vec4 { d / 4 } else { d };
    let mean_buf = mean.unwrap_or(ctx.dummy());
    let k_dx = ctx.kernel(
        &NORM_BWD_DX,
        Spec::new()
            .bool("RMS", rms)
            .bool("HAS_W", weight.is_some())
            .bool("VEC4", vec4)
            .bool("CACHE", units <= NORM_ROW_CACHE_MAX)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p_dx = Params::new().uz(outer)?.uz(units)?.f((1.0 / d as f64) as f32);
    ctx.dispatch(
        &k_dx,
        &[x, grad, weight.unwrap_or(ctx.dummy()), mean_buf, rstd, dx],
        &p_dx,
        ctx.grid_1d(outer as u64 * 256, 256)?,
    )?;

    if dweight.is_none() && dbias.is_none() {
        return Ok(());
    }
    let col_groups = units.div_ceil(32);
    let (splits, chunk) = dwb_split_plan(outer, units);
    let groups = prod(&[splits, col_groups])?;
    let p_dwb = Params::new().uz(outer)?.uz(units)?.uz(col_groups)?.uz(splits)?.uz(chunk)?;
    let grid = ctx.grid_1d(groups as u64 * 256, 256)?;

    match (dweight, dbias) {
        (Some(dw), Some(db)) => {
            let k = ctx.kernel(&NORM_BWD_DWB, Spec::new().bool("RMS", rms).bool("VEC4", vec4))?;
            if splits == 1 {
                ctx.dispatch(&k, &[x, grad, mean_buf, rstd, dw, db], &p_dwb, grid)?;
            } else {
                let part_bytes = DType::F32.bytes(prod(&[splits, d])?);
                let pw = ctx.alloc(part_bytes)?;
                let pb = ctx.alloc(part_bytes)?;
                ctx.dispatch(&k, &[x, grad, mean_buf, rstd, pw.raw(), pb.raw()], &p_dwb, grid)?;
                reduce::reduce_as(
                    ctx,
                    reduce::ReduceOp::Sum,
                    (pw.raw(), DType::F32),
                    (dw, DType::F32),
                    1,
                    splits,
                    d,
                )?;
                reduce::reduce_as(
                    ctx,
                    reduce::ReduceOp::Sum,
                    (pb.raw(), DType::F32),
                    (db, DType::F32),
                    1,
                    splits,
                    d,
                )?;
            }
        }
        (Some(dw), None) => {
            let k =
                ctx.kernel(&NORM_BWD_PARAM, Spec::new().bool("RMS", rms).bool("VEC4", vec4).u32("MODE", 0))?;
            if splits == 1 {
                ctx.dispatch(&k, &[x, grad, mean_buf, rstd, dw], &p_dwb, grid)?;
            } else {
                let pw = ctx.alloc(DType::F32.bytes(prod(&[splits, d])?))?;
                ctx.dispatch(&k, &[x, grad, mean_buf, rstd, pw.raw()], &p_dwb, grid)?;
                reduce::reduce_as(
                    ctx,
                    reduce::ReduceOp::Sum,
                    (pw.raw(), DType::F32),
                    (dw, DType::F32),
                    1,
                    splits,
                    d,
                )?;
            }
        }
        (None, Some(db)) => {
            let k =
                ctx.kernel(&NORM_BWD_PARAM, Spec::new().bool("RMS", rms).bool("VEC4", vec4).u32("MODE", 1))?;
            if splits == 1 {
                ctx.dispatch(&k, &[x, grad, mean_buf, rstd, db], &p_dwb, grid)?;
            } else {
                let pb = ctx.alloc(DType::F32.bytes(prod(&[splits, d])?))?;
                ctx.dispatch(&k, &[x, grad, mean_buf, rstd, pb.raw()], &p_dwb, grid)?;
                reduce::reduce_as(
                    ctx,
                    reduce::ReduceOp::Sum,
                    (pb.raw(), DType::F32),
                    (db, DType::F32),
                    1,
                    splits,
                    d,
                )?;
            }
        }
        (None, None) => unreachable!(),
    }
    Ok(())
}

/// Fused LayerNorm backward over dense f32 `[outer, d]`, writing `dx` and optional `dweight` / `dbias`.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    dbias: Option<&wgpu::Buffer>,
    outer: usize,
    d: usize,
) -> Result<()> {
    norm_backward_impl(ctx, false, x, grad, weight, Some(mean), rstd, dx, dweight, dbias, outer, d)
}

/// Fused RMSNorm backward over dense f32 `[outer, d]`, writing `dx` and optional `dweight`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_backward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    rstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    outer: usize,
    d: usize,
) -> Result<()> {
    norm_backward_impl(ctx, true, x, grad, weight, None, rstd, dx, dweight, None, outer, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_portable_emits_no_subgroup_tokens() {
        for (def, spec) in [
            (
                &NORM_FWD,
                Spec::new()
                    .bool("RMS", false)
                    .bool("HAS_W", true)
                    .bool("HAS_B", true)
                    .bool("VEC4", true)
                    .bool("CACHE", true)
                    .subgroups(false),
            ),
            (
                &NORM_BWD_DX,
                Spec::new()
                    .bool("RMS", true)
                    .bool("HAS_W", true)
                    .bool("VEC4", false)
                    .bool("CACHE", false)
                    .subgroups(false),
            ),
        ] {
            let wgsl = crate::pipeline::render(def, &spec);
            assert!(!wgsl.contains("subgroup"), "portable {} emitted subgroup token:\n{wgsl}", def.name);
        }
    }
}
