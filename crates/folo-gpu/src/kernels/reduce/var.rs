//! Fused single-pass variance reduction (`reduce_var`) over dense f32 `[outer, r, inner]`.
//!
//! - Contiguous rows (`inner == 1`):
//!   - `units <= 1024` (`r <= 4096` with `VEC4`, `r <= 1024` scalar): single-pass register-cached 256-thread
//!     workgroup per row (1 global read of `X`, 1-barrier 32-wide subgroup reduction or portable shared-memory
//!     tree fallback).
//!   - `units > 1024` with few rows (`split_plan` gives `splits > 1`): single-read split-K parallel Chan/Welford
//!     variance (pass 1 produces `[outer, splits, 3]` `(count, mean, M2)` partials with non-finite propagation;
//!     pass 2 merges `splits` partials via exact pairwise Chan merge).
//!   - `units > 1024` with many rows (`splits == 1`): single-dispatch two-pass row kernel.
//! - Strided columns (`inner > 1`):
//!   - Small `r` (`r <= 16` for `VEC4`, `r <= 32` for scalar): single-pass register-cached column kernel.
//!   - General `r`: coalesced `(64, 4)` workgroup two-pass column kernel (`VEC4` when `inner % 4 == 0`).

use super::split_plan;
use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::kernels::{check_buf, prod, strided, use_subgroups, vec_prelude};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Max per-row units cached in per-thread registers (4 × 256).
const VAR_ROW_CACHE_MAX: usize = 1024;
/// Max `r` cached per column thread under `VEC4` (`16 * vec4<f32>` = 64 floats).
const VAR_COL_SMALL_VEC4: usize = 16;
/// Max `r` cached per column thread under scalar (`32` floats).
const VAR_COL_SMALL_SCALAR: usize = 32;

/// Bounds check for precompiling `"reduce_var_rows"` from a kernel list (`kernels/bounds.rs`).
pub(crate) fn var_rows_fits(spec: &Spec) -> bool {
    spec.get_u32("MODE", 0) <= 2
}

const ROW_HELPERS: &str = r#"
fn lane_sum(v: vec4<f32>) -> f32 {
    if (VEC) { return (v.x + v.y) + (v.z + v.w); }
    return v.x;
}
fn is_nonfinite1(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u;
}
fn chan_merge(na: f32, ma: f32, qa: f32, nb: f32, mb: f32, qb: f32) -> vec3<f32> {
    if (na == 0.0) { return vec3<f32>(nb, mb, qb); }
    if (nb == 0.0) { return vec3<f32>(na, ma, qa); }
    let n = na + nb;
    let d = mb - ma;
    let delta = d * (nb / n);
    return vec3<f32>(n, ma + delta, qa + qb + d * (delta * na));
}
"#;

const ROW_SUM_SG: &str = r#"
var<workgroup> sh_s: array<f32, 8>;

fn reduce_sum(lid: u32, sgid: u32, lane: u32, s: f32) -> f32 {
    let rs = subgroupAdd(s);
    let sg = min(sgid, 7u);
    if (lane == 0u) { sh_s[sg] = rs; }
    workgroupBarrier();
    var vs = sh_s[lane & 7u];
    vs += subgroupShuffleXor(vs, 4u);
    vs += subgroupShuffleXor(vs, 2u);
    vs += subgroupShuffleXor(vs, 1u);
    workgroupBarrier();
    return vs;
}
"#;

const ROW_SUM_PORTABLE: &str = r#"
var<workgroup> sh_s: array<f32, 256>;

fn reduce_sum(lid: u32, sgid: u32, lane: u32, s: f32) -> f32 {
    sh_s[lid] = s;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step >>= 1u) {
        if (lid < step) { sh_s[lid] += sh_s[lid + step]; }
        workgroupBarrier();
    }
    let res = sh_s[0];
    workgroupBarrier();
    return res;
}
"#;

const ROW_CHAN_SG: &str = r#"
var<workgroup> wn: array<f32, 8>;
var<workgroup> wm: array<f32, 8>;
var<workgroup> wq: array<f32, 8>;

fn reduce_chan(lid: u32, sgid: u32, lane: u32, t: vec3<f32>) -> vec3<f32> {
    var st = t;
    for (var off = 16u; off > 0u; off = off >> 1u) {
        let nb = subgroupShuffleDown(st.x, off);
        let mb = subgroupShuffleDown(st.y, off);
        let qb = subgroupShuffleDown(st.z, off);
        if (lane < off) {
            st = chan_merge(st.x, st.y, st.z, nb, mb, qb);
        }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) { wn[sg] = st.x; wm[sg] = st.y; wq[sg] = st.z; }
    workgroupBarrier();
    if (lid == 0u) {
        var acc = vec3<f32>(wn[0], wm[0], wq[0]);
        for (var i = 1u; i < 8u; i += 1u) {
            acc = chan_merge(acc.x, acc.y, acc.z, wn[i], wm[i], wq[i]);
        }
        return acc;
    }
    return vec3<f32>(0.0);
}
"#;

const ROW_CHAN_PORTABLE: &str = r#"
var<workgroup> wn: array<f32, 256>;
var<workgroup> wm: array<f32, 256>;
var<workgroup> wq: array<f32, 256>;

fn reduce_chan(lid: u32, sgid: u32, lane: u32, t: vec3<f32>) -> vec3<f32> {
    wn[lid] = t.x;
    wm[lid] = t.y;
    wq[lid] = t.z;
    workgroupBarrier();
    for (var step = 128u; step > 0u; step >>= 1u) {
        if (lid < step) {
            let m = chan_merge(wn[lid], wm[lid], wq[lid], wn[lid + step], wm[lid + step], wq[lid + step]);
            wn[lid] = m.x;
            wm[lid] = m.y;
            wq[lid] = m.z;
        }
        workgroupBarrier();
    }
    return vec3<f32>(wn[0], wm[0], wq[0]);
}
"#;

/// MODE 0: single-workgroup cached (`CACHE`) or 2-pass row variance.
/// Params: 0 = outer, 1 = units_r, 2 = inv_r, 3 = inv_denom.
const VAR_ROWS_MODE0: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

fn run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let units = pu(1u);
    let inv_r = pf(2u);
    let inv_denom = pf(3u);
    let valid = row < rows;
    let base = row * units;

    if (CACHE_ROW) {
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        let i0 = lid;
        let i1 = lid + 256u;
        let i2 = lid + 512u;
        let i3 = lid + 768u;
        let h0 = valid && i0 < units;
        let h1 = valid && i1 < units;
        let h2 = valid && i2 < units;
        let h3 = valid && i3 < units;
        var s = 0.0;
        if (h0) { v0 = V4(X[base + i0]); s += lane_sum(v0); }
        if (h1) { v1 = V4(X[base + i1]); s += lane_sum(v1); }
        if (h2) { v2 = V4(X[base + i2]); s += lane_sum(v2); }
        if (h3) { v3 = V4(X[base + i3]); s += lane_sum(v3); }
        let mean = reduce_sum(lid, sgid, lane, s) * inv_r;
        let mv = vec4<f32>(mean);
        var sq = 0.0;
        if (h0) { let d = v0 - mv; sq += lane_sum(d * d); }
        if (h1) { let d = v1 - mv; sq += lane_sum(d * d); }
        if (h2) { let d = v2 - mv; sq += lane_sum(d * d); }
        if (h3) { let d = v3 - mv; sq += lane_sum(d * d); }
        let tot = reduce_sum(lid, sgid, lane, sq);
        if (lid == 0u && valid) {
            Y[row] = tot * inv_denom;
        }
        return;
    }

    var s = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            s += lane_sum(V4(X[base + i]));
        }
    }
    let mean = reduce_sum(lid, sgid, lane, s) * inv_r;
    let mv = vec4<f32>(mean);
    var sq = 0.0;
    if (valid) {
        for (var i = lid; i < units; i += 256u) {
            let d = V4(X[base + i]) - mv;
            sq += lane_sum(d * d);
        }
    }
    let tot = reduce_sum(lid, sgid, lane, sq);
    if (lid == 0u && valid) {
        Y[row] = tot * inv_denom;
    }
}
"#;

/// MODE 1: split-K single-read Chan/Welford partial pass.
/// Params: 0 = `outer_s`, 1 = `units_r`, 2 = `splits`, 3 = `units_chunk`, 4 = `nan`.
const VAR_ROWS_MODE1: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;

fn run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let outer_s = pu(0u);
    let units = pu(1u);
    let splits = pu(2u);
    let chunk = pu(3u);
    let nan_v = pf(4u);
    let valid = row < outer_s;
    var nt = 0.0;
    var mt = 0.0;
    var qt = 0.0;
    var nf = false;
    if (valid) {
        let o = row / splits;
        let k = row % splits;
        let start = k * chunk;
        let end = min(start + chunk, units);
        let base = o * units;
        if (VEC) {
            var n4 = 0.0;
            var m4 = vec4<f32>(0.0);
            var q4 = vec4<f32>(0.0);
            for (var i = start + lid; i < end; i += 256u) {
                let v = V4(X[base + i]);
                if (is_nonfinite1(v.x) || is_nonfinite1(v.y) || is_nonfinite1(v.z) || is_nonfinite1(v.w)) {
                    nf = true;
                }
                n4 += 1.0;
                let d = v - m4;
                m4 += d / n4;
                q4 += d * (v - m4);
            }
            if (n4 > 0.0) {
                let ab = chan_merge(n4, m4.x, q4.x, n4, m4.y, q4.y);
                let cd = chan_merge(n4, m4.z, q4.z, n4, m4.w, q4.w);
                let all = chan_merge(ab.x, ab.y, ab.z, cd.x, cd.y, cd.z);
                nt = all.x;
                mt = all.y;
                qt = select(all.z, nan_v, nf);
            }
        } else {
            for (var i = start + lid; i < end; i += 256u) {
                let v = V4(X[base + i]).x;
                if (is_nonfinite1(v)) { nf = true; }
                nt += 1.0;
                let d = v - mt;
                mt += d / nt;
                qt += d * (v - mt);
            }
            if (nf) { qt = nan_v; }
        }
    }
    let res = reduce_chan(lid, sgid, lane, vec3<f32>(nt, mt, qt));
    if (lid == 0u && valid) {
        let off = row * 3u;
        PART[off] = res.x;
        PART[off + 1u] = res.y;
        PART[off + 2u] = res.z;
    }
}
"#;

/// MODE 2: split-K Chan merge finalize pass.
/// Params: 0 = `outer`, 1 = `splits`, 2 = `inv_denom`.
const VAR_ROWS_MODE2: &str = r#"
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;

fn run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let outer = pu(0u);
    let splits = pu(1u);
    let inv_denom = pf(2u);
    let valid = row < outer;
    var acc = vec3<f32>(0.0);
    if (valid) {
        let base = row * splits;
        for (var i = lid; i < splits; i += 256u) {
            let off = (base + i) * 3u;
            acc = chan_merge(acc.x, acc.y, acc.z, PART[off], PART[off + 1u], PART[off + 2u]);
        }
    }
    let res = reduce_chan(lid, sgid, lane, acc);
    if (lid == 0u && valid) {
        Y[row] = res.z * inv_denom;
    }
}
"#;

fn gen_var_rows(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 0);
    let vec = mode < 2 && spec.get_bool("VEC4");
    let cache = spec.get_bool("CACHE");
    let sg = spec.is_subgroups();
    let pre = vec_prelude(vec);
    let red = match (mode, sg) {
        (0, true) => ROW_SUM_SG,
        (0, false) => ROW_SUM_PORTABLE,
        (_, true) => ROW_CHAN_SG,
        (_, false) => ROW_CHAN_PORTABLE,
    };
    let body = match mode {
        0 => VAR_ROWS_MODE0,
        1 => VAR_ROWS_MODE1,
        _ => VAR_ROWS_MODE2,
    };
    let entry = if sg {
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32,\n\
         \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
         \x20   run(wid, nwg, lid, sgid, lane);\n\
         }\n"
    } else {
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20   run(wid, nwg, lid, 0u, 0u);\n\
         }\n"
    };
    format!("{pre}const CACHE_ROW: bool = {cache};\n{ROW_HELPERS}{red}{body}{entry}")
}

/// Column variance kernel (`inner > 1`), `VEC4` when `inner % 4 == 0`.
/// - `SMALL`: 1 thread per `(outer, col_unit)` caches up to 16 (`VEC4`) or 32 (scalar) values in a local array.
/// - General: coalesced `(64, 4)` workgroup walking `r` in two passes with shared-memory reduction across `lid.y`.
///
/// Params:
/// - `SMALL`: 0 = `cols` (`outer * units_inner`), 1 = `r`, 2 = `units_inner`, 3 = `inv_r`, 4 = `inv_denom`.
/// - General: 0 = `outer`, 1 = `r`, 2 = `units_inner`, 3 = `chunks` (`ceil(units_inner / 64)`),
///   4 = `inv_r`, 5 = `inv_denom`.
fn gen_var_cols(spec: &Spec) -> String {
    let vec = spec.get_bool("VEC4");
    let small = spec.get_bool("SMALL");
    let pre = vec_prelude(vec);
    if small {
        let cap = if vec { VAR_COL_SMALL_VEC4 } else { VAR_COL_SMALL_SCALAR };
        format!(
            "{pre}\
             @group(0) @binding(1) var<storage, read> X: array<E>;\n\
             @group(0) @binding(2) var<storage, read_write> Y: array<E>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {{\n\
             \x20 let t = linear_id(wid, nwg, lid, 256u);\n\
             \x20 if (t >= pu(0u)) {{ return; }}\n\
             \x20 let r = pu(1u);\n\
             \x20 let inner = pu(2u);\n\
             \x20 let inv_r = pf(3u);\n\
             \x20 let inv_denom = pf(4u);\n\
             \x20 let base = (t / inner) * r * inner + (t % inner);\n\
             \x20 var vals: array<E, {cap}>;\n\
             \x20 var sum = vec4<f32>(0.0);\n\
             \x20 for (var i = 0u; i < r; i += 1u) {{\n\
             \x20   let v = X[base + i * inner];\n\
             \x20   vals[i] = v;\n\
             \x20   sum += V4(v);\n\
             \x20 }}\n\
             \x20 let mean = sum * inv_r;\n\
             \x20 var sq = vec4<f32>(0.0);\n\
             \x20 for (var i = 0u; i < r; i += 1u) {{\n\
             \x20   let d = V4(vals[i]) - mean;\n\
             \x20   sq += d * d;\n\
             \x20 }}\n\
             \x20 Y[t] = OUT(sq * inv_denom);\n\
             }}\n"
        )
    } else {
        format!(
            "{pre}\
             @group(0) @binding(1) var<storage, read> X: array<E>;\n\
             @group(0) @binding(2) var<storage, read_write> Y: array<E>;\n\
             var<workgroup> sh: array<E, 256>;\n\
             @compute @workgroup_size(64, 4)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
             \x20 let g = wid.y * nwg.x + wid.x;\n\
             \x20 let outer = pu(0u);\n\
             \x20 let r = pu(1u);\n\
             \x20 let inner = pu(2u);\n\
             \x20 let chunks = pu(3u);\n\
             \x20 let inv_r = pf(4u);\n\
             \x20 let inv_denom = pf(5u);\n\
             \x20 let o = g / chunks;\n\
             \x20 let col = (g % chunks) * 64u + lid.x;\n\
             \x20 let valid = o < outer && col < inner;\n\
             \x20 var acc = vec4<f32>(0.0);\n\
             \x20 if (valid) {{\n\
             \x20   let base = o * r * inner + col;\n\
             \x20   for (var i = lid.y; i < r; i += 4u) {{\n\
             \x20     acc += V4(X[base + i * inner]);\n\
             \x20   }}\n\
             \x20 }}\n\
             \x20 let slot = lid.y * 64u + lid.x;\n\
             \x20 sh[slot] = OUT(acc);\n\
             \x20 workgroupBarrier();\n\
             \x20 let mean = ((V4(sh[lid.x]) + V4(sh[64u + lid.x])) + (V4(sh[128u + lid.x]) + V4(sh[192u + lid.x]))) * inv_r;\n\
             \x20 workgroupBarrier();\n\
             \x20 var sq = vec4<f32>(0.0);\n\
             \x20 if (valid) {{\n\
             \x20   let base = o * r * inner + col;\n\
             \x20   for (var i = lid.y; i < r; i += 4u) {{\n\
             \x20     let d = V4(X[base + i * inner]) - mean;\n\
             \x20     sq += d * d;\n\
             \x20   }}\n\
             \x20 }}\n\
             \x20 sh[slot] = OUT(sq);\n\
             \x20 workgroupBarrier();\n\
             \x20 if (lid.y == 0u && valid) {{\n\
             \x20   let tot = (V4(sh[lid.x]) + V4(sh[64u + lid.x])) + (V4(sh[128u + lid.x]) + V4(sh[192u + lid.x]));\n\
             \x20   Y[o * inner + col] = OUT(tot * inv_denom);\n\
             \x20 }}\n\
             }}\n"
        )
    }
}

pub(super) static VAR_ROWS: KernelDef = KernelDef {
    name: "reduce_var_rows",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_var_rows),
};
pub(super) static VAR_COLS: KernelDef = KernelDef {
    name: "reduce_var_cols",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_var_cols),
};

/// Fused variance over the middle dimension of dense f32 `[outer, r, inner]` (biased or Bessel-corrected).
pub fn reduce_var(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (outer, r, inner): (usize, usize, usize),
    unbiased: bool,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, DType::F32, "reduce_var input")?;
    let out_elems = prod(&[outer, inner])?;
    check_buf(y, out_elems, DType::F32, "reduce_var output")?;
    if out_elems == 0 {
        return Ok(());
    }
    let denom = if unbiased { r.saturating_sub(1) } else { r };
    if denom == 0 {
        let nan_bits = f32::NAN.to_bits();
        return strided::fill_words(ctx, y, out_elems, nan_bits, nan_bits);
    }
    let inv_r = (1.0 / r as f64) as f32;
    let inv_denom = (1.0 / denom as f64) as f32;

    if inner == 1 {
        let sg = use_subgroups(ctx);
        let vec4 = r.is_multiple_of(4);
        let units = if vec4 { r / 4 } else { r };
        if units > VAR_ROW_CACHE_MAX {
            let (splits, chunk) = split_plan(outer, r, 1);
            if splits > 1 {
                let outer_s = prod(&[outer, splits])?;
                let part = ctx.alloc(DType::F32.bytes(prod(&[outer_s, 3])?))?;
                let vec4_s = vec4 && chunk.is_multiple_of(4);
                let (r_u, chunk_u) = if vec4_s { (r / 4, chunk / 4) } else { (r, chunk) };
                let k1 =
                    ctx.kernel(&VAR_ROWS, Spec::new().u32("MODE", 1).bool("VEC4", vec4_s).subgroups(sg))?;
                let p1 = Params::new().uz(outer_s)?.uz(r_u)?.uz(splits)?.uz(chunk_u)?.f(f32::NAN);
                ctx.dispatch(&k1, &[x, part.raw()], &p1, ctx.grid_1d(outer_s as u64 * 256, 256)?)?;
                let k2 = ctx.kernel(&VAR_ROWS, Spec::new().u32("MODE", 2).subgroups(sg))?;
                let p2 = Params::new().uz(outer)?.uz(splits)?.f(inv_denom);
                return ctx.dispatch(&k2, &[part.raw(), y], &p2, ctx.grid_1d(outer as u64 * 256, 256)?);
            }
        }
        let k = ctx.kernel(
            &VAR_ROWS,
            Spec::new()
                .u32("MODE", 0)
                .bool("VEC4", vec4)
                .bool("CACHE", units <= VAR_ROW_CACHE_MAX)
                .subgroups(sg),
        )?;
        let p = Params::new().uz(outer)?.uz(units)?.f(inv_r).f(inv_denom);
        return ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(outer as u64 * 256, 256)?);
    }

    let vec4 = inner.is_multiple_of(4);
    let units_inner = if vec4 { inner / 4 } else { inner };
    let small_lim = if vec4 { VAR_COL_SMALL_VEC4 } else { VAR_COL_SMALL_SCALAR };
    if r <= small_lim {
        let cols = prod(&[outer, units_inner])?;
        let k = ctx.kernel(&VAR_COLS, Spec::new().bool("VEC4", vec4).bool("SMALL", true))?;
        let p = Params::new().uz(cols)?.uz(r)?.uz(units_inner)?.f(inv_r).f(inv_denom);
        return ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(cols as u64, 256)?);
    }
    let chunks = units_inner.div_ceil(64);
    let k = ctx.kernel(&VAR_COLS, Spec::new().bool("VEC4", vec4).bool("SMALL", false))?;
    let p = Params::new().uz(outer)?.uz(r)?.uz(units_inner)?.uz(chunks)?.f(inv_r).f(inv_denom);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d((outer * chunks) as u64 * 256, 256)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_rows_portable_emits_no_subgroup_tokens() {
        for (mode, vec4, cache) in
            [(0u32, true, true), (0, false, false), (1, true, false), (2, false, false)]
        {
            let spec = Spec::new().u32("MODE", mode).bool("VEC4", vec4).bool("CACHE", cache).subgroups(false);
            let wgsl = crate::pipeline::render(&VAR_ROWS, &spec);
            assert!(
                !wgsl.contains("subgroup"),
                "portable reduce_var_rows mode={mode} emitted subgroup token:\n{wgsl}"
            );
        }
    }
}
