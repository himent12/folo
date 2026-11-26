//! Fused WGSL `softmax` and `log_softmax` backward kernels over any dimension (`[outer, d, inner]`), f32 and f16.
//!
//! - `inner == 1` (`softmax_bwd_rows`): one 256-thread workgroup per row, `vec4` loads/stores when `d % 4 == 0`,
//!   per-thread register caching up to 4 units (`d <= 4096` with `VEC4`, `d <= 1024` scalar), and 1-barrier
//!   32-wide subgroup reduction with a portable 8-barrier shared-memory tree fallback.
//! - `inner > 1` (`softmax_bwd_cols`): coalesced column kernel (`vec4` when `inner % 4 == 0`), one thread per
//!   `(outer, col_unit)` walking `d` in two passes.

use super::{check_buf, prod, use_subgroups, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Max per-row units cached in per-thread registers (4 units × 256 threads).
const SOFTMAX_BWD_CACHE_MAX: usize = 1024;

const ROWS_REDUCE_SG: &str = r#"
var<workgroup> sh: array<f32, 8>;

fn reduce_sum(lid: u32, sgid: u32, lane: u32, s: f32) -> f32 {
    let rs = subgroupAdd(s);
    let sg = min(sgid, 7u);
    if (lane == 0u) { sh[sg] = rs; }
    workgroupBarrier();
    var v = sh[lane & 7u];
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 1u);
    return v;
}
"#;

const ROWS_REDUCE_PORTABLE: &str = r#"
var<workgroup> sh: array<f32, 256>;

fn reduce_sum(lid: u32, sgid: u32, lane: u32, s: f32) -> f32 {
    sh[lid] = s;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step >>= 1u) {
        if (lid < step) { sh[lid] += sh[lid + step]; }
        workgroupBarrier();
    }
    return sh[0];
}
"#;

const ROWS_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> Y: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read_write> DX: array<E>;

fn lane_sum(v: vec4<f32>) -> f32 {
    if (VEC) { return (v.x + v.y) + (v.z + v.w); }
    return v.x;
}

fn softmax_bwd_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let units = pu(1u);
    let valid = row < rows;
    let base = row * units;

    if (CACHE_ROW) {
        var y0 = vec4<f32>(0.0); var g0 = vec4<f32>(0.0);
        var y1 = vec4<f32>(0.0); var g1 = vec4<f32>(0.0);
        var y2 = vec4<f32>(0.0); var g2 = vec4<f32>(0.0);
        var y3 = vec4<f32>(0.0); var g3 = vec4<f32>(0.0);
        let i0 = lid;
        let i1 = lid + 256u;
        let i2 = lid + 512u;
        let i3 = lid + 768u;
        let h0 = valid && i0 < units;
        let h1 = valid && i1 < units;
        let h2 = valid && i2 < units;
        let h3 = valid && i3 < units;
        var acc = 0.0;
        if (h0) {
            y0 = V4(Y[base + i0]); g0 = V4(G[base + i0]);
            acc += lane_sum(select(g0 * y0, g0, LOG));
        }
        if (h1) {
            y1 = V4(Y[base + i1]); g1 = V4(G[base + i1]);
            acc += lane_sum(select(g1 * y1, g1, LOG));
        }
        if (h2) {
            y2 = V4(Y[base + i2]); g2 = V4(G[base + i2]);
            acc += lane_sum(select(g2 * y2, g2, LOG));
        }
        if (h3) {
            y3 = V4(Y[base + i3]); g3 = V4(G[base + i3]);
            acc += lane_sum(select(g3 * y3, g3, LOG));
        }
        let red = reduce_sum(lid, sgid, lane, acc);
        if (!valid) { return; }
        let rv = vec4<f32>(red);
        if (LOG) {
            if (h0) { DX[base + i0] = OUT(g0 - exp(y0) * rv); }
            if (h1) { DX[base + i1] = OUT(g1 - exp(y1) * rv); }
            if (h2) { DX[base + i2] = OUT(g2 - exp(y2) * rv); }
            if (h3) { DX[base + i3] = OUT(g3 - exp(y3) * rv); }
        } else {
            if (h0) { DX[base + i0] = OUT(y0 * (g0 - rv)); }
            if (h1) { DX[base + i1] = OUT(y1 * (g1 - rv)); }
            if (h2) { DX[base + i2] = OUT(y2 * (g2 - rv)); }
            if (h3) { DX[base + i3] = OUT(y3 * (g3 - rv)); }
        }
        return;
    }

    var acc = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            let gv = V4(G[base + i]);
            if (LOG) {
                acc += lane_sum(gv);
            } else {
                acc += lane_sum(gv * V4(Y[base + i]));
            }
        }
    }
    let red = reduce_sum(lid, sgid, lane, acc);
    if (!valid) { return; }
    let rv = vec4<f32>(red);
    if (LOG) {
        for (var i = lid; i < units; i += 256u) {
            let yv = V4(Y[base + i]);
            let gv = V4(G[base + i]);
            DX[base + i] = OUT(gv - exp(yv) * rv);
        }
    } else {
        for (var i = lid; i < units; i += 256u) {
            let yv = V4(Y[base + i]);
            let gv = V4(G[base + i]);
            DX[base + i] = OUT(yv * (gv - rv));
        }
    }
}
"#;

const ROWS_ENTRY_SG: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    softmax_bwd_run(wid, nwg, lid, sgid, lane);
}
"#;

const ROWS_ENTRY_PORTABLE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    softmax_bwd_run(wid, nwg, lid, 0u, 0u);
}
"#;

const COLS_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> Y: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read_write> DX: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = linear_id(wid, nwg, lid, 256u);
    if (t >= pu(0u)) { return; }
    let d = pu(1u);
    let inner = pu(2u);
    let base = (t / inner) * d * inner + t % inner;
    var acc = vec4<f32>(0.0);
    if (LOG) {
        for (var i = 0u; i < d; i++) {
            acc += V4(G[base + i * inner]);
        }
        for (var i = 0u; i < d; i++) {
            let o = base + i * inner;
            DX[o] = OUT(V4(G[o]) - exp(V4(Y[o])) * acc);
        }
    } else {
        for (var i = 0u; i < d; i++) {
            let o = base + i * inner;
            acc += V4(G[o]) * V4(Y[o]);
        }
        for (var i = 0u; i < d; i++) {
            let o = base + i * inner;
            let yv = V4(Y[o]);
            DX[o] = OUT(yv * (V4(G[o]) - acc));
        }
    }
}
"#;

fn prelude(spec: &Spec) -> &'static str {
    let vec = spec.get_bool("VEC4");
    if spec.is_f16() { vec_prelude_f16(vec) } else { vec_prelude(vec) }
}

fn gen_rows(s: &Spec) -> String {
    let pre = prelude(s);
    let cache = s.get_bool("CACHE");
    let (red, entry) = if s.is_subgroups() {
        (ROWS_REDUCE_SG, ROWS_ENTRY_SG)
    } else {
        (ROWS_REDUCE_PORTABLE, ROWS_ENTRY_PORTABLE)
    };
    format!("{pre}const CACHE_ROW: bool = {cache};\n{red}{ROWS_BODY}{entry}")
}

fn gen_cols(s: &Spec) -> String {
    format!("{}{COLS_BODY}", prelude(s))
}

static SOFTMAX_BWD_ROWS: KernelDef = KernelDef {
    name: "softmax_bwd_rows",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_rows),
};
static SOFTMAX_BWD_COLS: KernelDef = KernelDef {
    name: "softmax_bwd_cols",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_cols),
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&SOFTMAX_BWD_ROWS, &SOFTMAX_BWD_COLS];

/// Fused backward of `softmax` (`!log`) or `log_softmax` (`log`) over `[outer, d, inner]` (`dtype`: f32 or f16).
#[allow(clippy::too_many_arguments)]
pub fn softmax_backward_dim(
    ctx: &GpuContext,
    y: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    log: bool,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "softmax_backward")?;
    let total = prod(&[outer, d, inner])?;
    check_buf(y, total, dtype, "softmax_backward y")?;
    check_buf(grad, total, dtype, "softmax_backward grad")?;
    check_buf(dx, total, dtype, "softmax_backward dx")?;
    if total == 0 {
        return Ok(());
    }
    let half = dtype == DType::F16;
    if inner == 1 {
        let vec4 = d.is_multiple_of(4);
        let units = if vec4 { d / 4 } else { d };
        let k = ctx.kernel(
            &SOFTMAX_BWD_ROWS,
            Spec::new()
                .bool("LOG", log)
                .bool("VEC4", vec4)
                .bool("CACHE", units <= SOFTMAX_BWD_CACHE_MAX)
                .f16(half)
                .subgroups(use_subgroups(ctx)),
        )?;
        let p = Params::new().uz(outer)?.uz(units)?;
        ctx.dispatch(&k, &[y, grad, dx], &p, ctx.grid_1d(outer as u64 * 256, 256)?)
    } else {
        let vec4 = inner.is_multiple_of(4);
        let units_inner = if vec4 { inner / 4 } else { inner };
        let cols = prod(&[outer, units_inner])?;
        let k = ctx.kernel(&SOFTMAX_BWD_COLS, Spec::new().bool("LOG", log).bool("VEC4", vec4).f16(half))?;
        let p = Params::new().uz(cols)?.uz(d)?.uz(units_inner)?;
        ctx.dispatch(&k, &[y, grad, dx], &p, ctx.grid_1d(cols as u64, 256)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_bwd_portable_emits_no_subgroup_tokens() {
        let spec = Spec::new().bool("LOG", false).bool("VEC4", true).bool("CACHE", true).subgroups(false);
        let wgsl = crate::pipeline::render(&SOFTMAX_BWD_ROWS, &spec);
        assert!(!wgsl.contains("subgroup"), "portable softmax_bwd_rows emitted subgroup token:\n{wgsl}");
    }
}
