//! Fused WGSL `residual_add + LayerNorm` and `residual_add + RMSNorm` forward and backward kernels (`F32` and `F16`).
//!
//! Eliminates the separate elementwise residual-add pass (one full read + write of `[outer, d]`):
//! - Forward (`res_norm_fwd`): loads `x` and `residual` (`vec4` when `d % 4 == 0`), computes `h = x + residual` in
//!   `f32`, writes `h_out` (same dtype as `x`), keeps `h` cached in registers up to 1,024 units (`d <= 4,096` under
//!   `VEC4`), runs 1-barrier 32-wide subgroup reductions (or portable shared-memory tree fallback) with extreme-scale
//!   guards (`1e20`, `1e-23`, `1e38`, `eps = 0`), and writes `y`, `mean` (LayerNorm), and `rstd`.
//! - Backward (`res_norm_bwd_dx`, `res_norm_bwd_dwb`): fuses norm input gradient `dx` with optional upstream
//!   residual skip gradient `grad_res` (`dh = dx_norm + grad_res`) in one pass, with `mu_xh` re-centering and
//!   `ldexp` extreme-scale guards, plus split-K coalesced `(32, 8)` column reductions for `F32` `dweight` / `dbias`.

use super::{check_buf, prod, reduce, use_subgroups, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Max per-row units (`d / 4` when `VEC4`, else `d`) cached in registers (4 × 256).
const ROW_CACHE_MAX: usize = 1024;
/// Minimum `outer` rows before splitting `dweight` / `dbias` across workgroups.
const DWB_SPLIT_MIN_OUTER: usize = 256;
/// Minimum rows per split chunk for `dweight` / `dbias`.
const DWB_MIN_CHUNK: usize = 32;

fn prelude(spec: &Spec) -> &'static str {
    let vec = spec.get_bool("VEC4");
    if spec.is_f16() { vec_prelude_f16(vec) } else { vec_prelude(vec) }
}

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
fn norm_xh(x: vec4<f32>, mean: f32, rstd: f32) -> vec4<f32> {
    if (abs(mean) > 1e30 || (rstd > 0.0 && rstd < 1e-30)) {
        let xs = ldexp(x, vec4<i32>(-96i));
        let ms = select(vec4<f32>(ldexp(mean, -96i)), vec4<f32>(0.0), RMS);
        return (xs - ms) * ldexp(rstd, 96i);
    }
    let xc = select(x - vec4<f32>(mean), x, RMS);
    return xc * rstd;
}
"#;

const REDUCE_SUBGROUPS: &str = r#"
var<workgroup> sh_a: array<f32, 8>;
var<workgroup> sh_b: array<f32, 8>;
var<workgroup> sh_c: array<f32, 8>;
var<workgroup> sh_u: vec2<f32>;

fn reduce_sum_max(lid: u32, sgid: u32, lane: u32, s: f32, mx: f32) -> vec2<f32> {
    let rs = subgroupAdd(s);
    let rm = subgroupMax(mx);
    let sg = min(sgid, 7u);
    if (lane == 0u) { sh_a[sg] = rs; sh_b[sg] = rm; }
    workgroupBarrier();
    if (lid == 0u) {
        let vs = ((sh_a[0] + sh_a[1]) + (sh_a[2] + sh_a[3]))
               + ((sh_a[4] + sh_a[5]) + (sh_a[6] + sh_a[7]));
        let vm = max(
            max(max(sh_b[0], sh_b[1]), max(sh_b[2], sh_b[3])),
            max(max(sh_b[4], sh_b[5]), max(sh_b[6], sh_b[7])),
        );
        sh_u = vec2<f32>(vs, vm);
    }
    return workgroupUniformLoad(&sh_u);
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
    if (lane == 0u) { sh_a[sg] = ra; sh_b[sg] = rb; sh_c[sg] = rc; }
    workgroupBarrier();
    var va = sh_a[lane & 7u];
    var vb = sh_b[lane & 7u];
    var vc = sh_c[lane & 7u];
    va += subgroupShuffleXor(va, 4u); vb += subgroupShuffleXor(vb, 4u); vc += subgroupShuffleXor(vc, 4u);
    va += subgroupShuffleXor(va, 2u); vb += subgroupShuffleXor(vb, 2u); vc += subgroupShuffleXor(vc, 2u);
    va += subgroupShuffleXor(va, 1u); vb += subgroupShuffleXor(vb, 1u); vc += subgroupShuffleXor(vc, 1u);
    return vec3<f32>(va, vb, vc);
}
"#;

const REDUCE_PORTABLE: &str = r#"
var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_b: array<f32, 256>;
var<workgroup> sh_c: array<f32, 256>;
var<workgroup> sh_u: vec2<f32>;

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
    if (lid == 0u) { sh_u = vec2<f32>(sh_a[0], sh_b[0]); }
    return workgroupUniformLoad(&sh_u);
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

const RES_NORM_FWD_BODY: &str = include_str!("../shaders/res_norm_fwd.wgsl");

const RES_NORM_BWD_DX_BODY: &str = include_str!("../shaders/res_norm_bwd_dx.wgsl");

/// Coalesced `(32, 8)` column reduction for `F32` `dweight` and/or `dbias` over `H` (`F32` or `F16`).
/// Params: 0 = `outer`, 1 = `units`, 2 = `col_groups`, 3 = `splits`, 4 = `chunk`.
const RES_NORM_BWD_DWB_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> H: array<E>;
@group(0) @binding(2) var<storage, read> GY: array<E>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> RSTD: array<f32>;
@group(0) @binding(5) var<storage, read> MU_XH: array<f32>;
@group(0) @binding(6) var<storage, read_write> DW: array<PF>;
@group(0) @binding(7) var<storage, read_write> DB: array<PF>;

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
            let gv = V4(GY[idx]);
            if (NEED_W) {
                var mean = 0.0;
                if (!RMS) { mean = MEAN[o]; }
                var xh = norm_xh(V4(H[idx]), mean, RSTD[o]);
                if (!RMS) { xh -= vec4<f32>(MU_XH[o]); }
                aw += gv * xh;
            }
            if (NEED_B) { ab += gv; }
        }
    }
    let slot = lid.y * 32u + lid.x;
    sh_w[slot] = aw;
    sh_b[slot] = ab;
    workgroupBarrier();
    if (lid.y == 0u && valid) {
        if (NEED_W) {
            let sw = ((sh_w[lid.x] + sh_w[32u + lid.x]) + (sh_w[64u + lid.x] + sh_w[96u + lid.x]))
                   + ((sh_w[128u + lid.x] + sh_w[160u + lid.x]) + (sh_w[192u + lid.x] + sh_w[224u + lid.x]));
            DW[sidx * units + col] = PFOUT(sw);
        }
        if (NEED_B) {
            let sb = ((sh_b[lid.x] + sh_b[32u + lid.x]) + (sh_b[64u + lid.x] + sh_b[96u + lid.x]))
                   + ((sh_b[128u + lid.x] + sh_b[160u + lid.x]) + (sh_b[192u + lid.x] + sh_b[224u + lid.x]));
            DB[sidx * units + col] = PFOUT(sb);
        }
    }
}
"#;

fn param_f32_prelude(vec: bool) -> &'static str {
    if vec {
        "alias PF = vec4<f32>;\nfn PF4(x: PF) -> vec4<f32> { return x; }\nfn PFOUT(v: vec4<f32>) -> PF { return v; }\n"
    } else {
        "alias PF = f32;\nfn PF4(x: PF) -> vec4<f32> { return vec4<f32>(x); }\nfn PFOUT(v: vec4<f32>) -> PF { return v.x; }\n"
    }
}

fn entry_for(fn_name: &str, sg: bool) -> String {
    if sg {
        format!(
            "@compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {{\n\
             \x20   {fn_name}(wid, nwg, lid, sgid, lane);\n\
             }}\n"
        )
    } else {
        format!(
            "@compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {{\n\
             \x20   {fn_name}(wid, nwg, lid, 0u, 0u);\n\
             }}\n"
        )
    }
}

fn gen_fwd(s: &Spec) -> String {
    let vec = s.get_bool("VEC4");
    let cache = s.get_bool("CACHE");
    let red = if s.is_subgroups() { REDUCE_SUBGROUPS } else { REDUCE_PORTABLE };
    let entry = entry_for("res_norm_fwd_run", s.is_subgroups());
    format!(
        "{}{}const CACHE_ROW: bool = {cache};\n{LANE_HELPERS}{red}{RES_NORM_FWD_BODY}{entry}",
        prelude(s),
        param_f32_prelude(vec),
    )
}

fn gen_bwd_dx(s: &Spec) -> String {
    let vec = s.get_bool("VEC4");
    let cache = s.get_bool("CACHE");
    let red = if s.is_subgroups() { REDUCE_SUBGROUPS } else { REDUCE_PORTABLE };
    let entry = entry_for("res_norm_bwd_dx_run", s.is_subgroups());
    format!(
        "{}{}const CACHE_ROW: bool = {cache};\n{LANE_HELPERS}{red}{RES_NORM_BWD_DX_BODY}{entry}",
        prelude(s),
        param_f32_prelude(vec),
    )
}

fn gen_bwd_dwb(s: &Spec) -> String {
    let vec = s.get_bool("VEC4");
    format!("{}{}{LANE_HELPERS}{RES_NORM_BWD_DWB_BODY}", prelude(s), param_f32_prelude(vec))
}

use Access::{Read as R, ReadWrite as RW};

static RES_NORM_FWD: KernelDef = KernelDef {
    name: "res_norm_fwd",
    storage: &[R, R, R, R, RW, RW, RW, RW],
    source: Source::Generated(gen_fwd),
};
static RES_NORM_BWD_DX: KernelDef = KernelDef {
    name: "res_norm_bwd_dx",
    storage: &[R, R, R, R, R, R, RW, RW],
    source: Source::Generated(gen_bwd_dx),
};
static RES_NORM_BWD_DWB: KernelDef = KernelDef {
    name: "res_norm_bwd_dwb",
    storage: &[R, R, R, R, R, RW, RW],
    source: Source::Generated(gen_bwd_dwb),
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&RES_NORM_FWD, &RES_NORM_BWD_DX, &RES_NORM_BWD_DWB];

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

#[allow(clippy::too_many_arguments)]
fn res_norm_fwd_impl(
    ctx: &GpuContext,
    rms: bool,
    x: &wgpu::Buffer,
    residual: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    bias: Option<&wgpu::Buffer>,
    h_out: &wgpu::Buffer,
    y: &wgpu::Buffer,
    mean: Option<&wgpu::Buffer>,
    rstd: &wgpu::Buffer,
    outer: usize,
    d: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    let op = if rms { "residual_add_rms_norm_forward" } else { "residual_add_layer_norm_forward" };
    check_dtype(ctx, dtype, op)?;
    if !eps.is_finite() || eps < 0.0 {
        return Err(shape_err(format!("{op}: invalid eps {eps}")));
    }
    if d == 0 {
        return Err(shape_err(format!("{op}: last dimension d must be non-zero")));
    }
    let total = prod(&[outer, d])?;
    if total > u32::MAX as usize {
        return Err(GpuError::Limit(format!("{op}: tensor exceeds u32 indexing")));
    }
    check_buf(x, total, dtype, op)?;
    check_buf(residual, total, dtype, op)?;
    check_buf(h_out, total, dtype, op)?;
    check_buf(y, total, dtype, op)?;
    check_buf(rstd, outer, DType::F32, op)?;
    if let Some(m) = mean {
        check_buf(m, outer, DType::F32, op)?;
    }
    if let Some(w) = weight {
        check_buf(w, d, DType::F32, op)?;
    }
    if let Some(b) = bias {
        check_buf(b, d, DType::F32, op)?;
    }
    if total == 0 {
        return Ok(());
    }
    let vec4 = d.is_multiple_of(4);
    let units = if vec4 { d / 4 } else { d };
    let k = ctx.kernel(
        &RES_NORM_FWD,
        Spec::new()
            .bool("RMS", rms)
            .bool("HAS_W", weight.is_some())
            .bool("HAS_B", bias.is_some())
            .bool("VEC4", vec4)
            .bool("CACHE", units <= ROW_CACHE_MAX)
            .f16(dtype == DType::F16)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new().uz(outer)?.uz(units)?.f((1.0 / d as f64) as f32).f(eps).f(eps.sqrt());
    let dummy_mean = mean.is_none().then(|| ctx.alloc(4)).transpose()?;
    let mean_buf = mean.unwrap_or_else(|| dummy_mean.as_ref().unwrap().raw());
    ctx.dispatch(
        &k,
        &[x, residual, weight.unwrap_or(ctx.dummy()), bias.unwrap_or(ctx.dummy()), h_out, y, mean_buf, rstd],
        &p,
        ctx.grid_1d(outer as u64 * 256, 256)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn res_norm_bwd_impl(
    ctx: &GpuContext,
    rms: bool,
    h: &wgpu::Buffer,
    grad_y: &wgpu::Buffer,
    grad_res: Option<&wgpu::Buffer>,
    weight: Option<&wgpu::Buffer>,
    mean: Option<&wgpu::Buffer>,
    rstd: &wgpu::Buffer,
    dh: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    dbias: Option<&wgpu::Buffer>,
    outer: usize,
    d: usize,
    dtype: DType,
) -> Result<()> {
    let op = if rms { "residual_add_rms_norm_backward" } else { "residual_add_layer_norm_backward" };
    check_dtype(ctx, dtype, op)?;
    if d == 0 {
        return Err(shape_err(format!("{op}: last dimension d must be non-zero")));
    }
    let total = prod(&[outer, d])?;
    if total > u32::MAX as usize {
        return Err(GpuError::Limit(format!("{op}: tensor exceeds u32 indexing")));
    }
    check_buf(h, total, dtype, op)?;
    check_buf(grad_y, total, dtype, op)?;
    check_buf(dh, total, dtype, op)?;
    check_buf(rstd, outer, DType::F32, op)?;
    if let Some(gr) = grad_res {
        check_buf(gr, total, dtype, op)?;
    }
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
    if outer == 0 {
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
    let half = dtype == DType::F16;
    let mean_buf = mean.unwrap_or(ctx.dummy());
    let mu_xh_buf = ctx.alloc(DType::F32.bytes(outer))?;

    let k_dx = ctx.kernel(
        &RES_NORM_BWD_DX,
        Spec::new()
            .bool("RMS", rms)
            .bool("HAS_W", weight.is_some())
            .bool("HAS_GRES", grad_res.is_some())
            .bool("VEC4", vec4)
            .bool("CACHE", units <= ROW_CACHE_MAX)
            .f16(half)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p_dx = Params::new().uz(outer)?.uz(units)?.f((1.0 / d as f64) as f32);
    ctx.dispatch(
        &k_dx,
        &[
            h,
            grad_y,
            grad_res.unwrap_or(ctx.dummy()),
            weight.unwrap_or(ctx.dummy()),
            mean_buf,
            rstd,
            dh,
            mu_xh_buf.raw(),
        ],
        &p_dx,
        ctx.grid_1d(outer as u64 * 256, 256)?,
    )?;

    if dweight.is_none() && dbias.is_none() {
        return Ok(());
    }
    let col_groups = units.div_ceil(32);
    let (splits, chunk) = dwb_split_plan(outer, units);
    let groups = prod(&[splits, col_groups])?;
    let k_dwb = ctx.kernel(
        &RES_NORM_BWD_DWB,
        Spec::new()
            .bool("RMS", rms)
            .bool("NEED_W", dweight.is_some())
            .bool("NEED_B", dbias.is_some())
            .bool("VEC4", vec4)
            .f16(half),
    )?;
    let p_dwb = Params::new().uz(outer)?.uz(units)?.uz(col_groups)?.uz(splits)?.uz(chunk)?;
    let grid = ctx.grid_1d(groups as u64 * 256, 256)?;

    if splits == 1 {
        let sw = dweight.is_none().then(|| ctx.alloc(16)).transpose()?;
        let sb = dbias.is_none().then(|| ctx.alloc(16)).transpose()?;
        let dw_b = dweight.unwrap_or_else(|| sw.as_ref().unwrap().raw());
        let db_b = dbias.unwrap_or_else(|| sb.as_ref().unwrap().raw());
        ctx.dispatch(&k_dwb, &[h, grad_y, mean_buf, rstd, mu_xh_buf.raw(), dw_b, db_b], &p_dwb, grid)?;
    } else {
        let part_bytes = DType::F32.bytes(prod(&[splits, d])?);
        let pw = ctx.alloc(if dweight.is_some() { part_bytes } else { 16 })?;
        let pb = ctx.alloc(if dbias.is_some() { part_bytes } else { 16 })?;
        ctx.dispatch(
            &k_dwb,
            &[h, grad_y, mean_buf, rstd, mu_xh_buf.raw(), pw.raw(), pb.raw()],
            &p_dwb,
            grid,
        )?;
        if let Some(dw) = dweight {
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
        if let Some(db) = dbias {
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
    Ok(())
}

/// Fused `h = x + residual; y = LayerNorm(h, weight, bias, eps)` over dense `[outer, d]` (`F32` or `F16`).
#[allow(clippy::too_many_arguments)]
pub fn residual_add_layer_norm_forward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    residual: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    bias: Option<&wgpu::Buffer>,
    h_out: &wgpu::Buffer,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    outer: usize,
    d: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    res_norm_fwd_impl(ctx, false, x, residual, weight, bias, h_out, y, Some(mean), rstd, outer, d, eps, dtype)
}

/// Fused LayerNorm backward + residual skip gradient accumulation: writes `dh = dx_norm + grad_res` and
/// optional `F32` `dweight [d]`, `dbias [d]`.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_layer_norm_backward(
    ctx: &GpuContext,
    h: &wgpu::Buffer,
    grad_y: &wgpu::Buffer,
    grad_res: Option<&wgpu::Buffer>,
    weight: Option<&wgpu::Buffer>,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    dh: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    dbias: Option<&wgpu::Buffer>,
    outer: usize,
    d: usize,
    dtype: DType,
) -> Result<()> {
    res_norm_bwd_impl(
        ctx,
        false,
        h,
        grad_y,
        grad_res,
        weight,
        Some(mean),
        rstd,
        dh,
        dweight,
        dbias,
        outer,
        d,
        dtype,
    )
}

/// Fused `h = x + residual; y = RMSNorm(h, weight, eps)` over dense `[outer, d]` (`F32` or `F16`).
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm_forward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    residual: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    h_out: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    outer: usize,
    d: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    res_norm_fwd_impl(ctx, true, x, residual, weight, None, h_out, y, None, rstd, outer, d, eps, dtype)
}

/// Fused RMSNorm backward + residual skip gradient accumulation: writes `dh = dx_norm + grad_res` and
/// optional `F32` `dweight [d]`.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm_backward(
    ctx: &GpuContext,
    h: &wgpu::Buffer,
    grad_y: &wgpu::Buffer,
    grad_res: Option<&wgpu::Buffer>,
    weight: Option<&wgpu::Buffer>,
    rstd: &wgpu::Buffer,
    dh: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    outer: usize,
    d: usize,
    dtype: DType,
) -> Result<()> {
    res_norm_bwd_impl(ctx, true, h, grad_y, grad_res, weight, None, rstd, dh, dweight, None, outer, d, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn res_norm_portable_emits_no_subgroup_tokens() {
        for (def, spec) in [
            (
                &RES_NORM_FWD,
                Spec::new()
                    .bool("RMS", false)
                    .bool("HAS_W", true)
                    .bool("HAS_B", true)
                    .bool("VEC4", true)
                    .bool("CACHE", true)
                    .subgroups(false),
            ),
            (
                &RES_NORM_BWD_DX,
                Spec::new()
                    .bool("RMS", true)
                    .bool("HAS_W", true)
                    .bool("HAS_GRES", true)
                    .bool("VEC4", false)
                    .bool("CACHE", false)
                    .subgroups(false),
            ),
            (
                &RES_NORM_BWD_DWB,
                Spec::new().bool("RMS", false).bool("NEED_W", true).bool("NEED_B", true).bool("VEC4", true),
            ),
        ] {
            let wgsl = crate::pipeline::render(def, &spec);
            assert!(!wgsl.contains("subgroup"), "portable {} emitted subgroup token:\n{wgsl}", def.name);
        }
    }
}
