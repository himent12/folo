//! Fused single-pass WGSL GroupNorm and InstanceNorm forward and backward kernels over NCHW `[N, C, H, W]`
//! (viewed as `[N, C, HW]`, grouped into `G` groups of `C / G` channels each; InstanceNorm is `G == C`).
//!
//! - Forward (`group_norm_fwd`): one 256-thread workgroup per `(n, g)` group (each group is a contiguous
//!   `(C / G) * HW` slice), `vec4` loads/stores when `HW % 4 == 0` (every `vec4` stays inside one channel),
//!   register-cached single-pass up to 1,024 units (`4,096` elements/group under `VEC4`), 1-barrier 32-wide
//!   subgroup reduction with a portable shared-memory tree fallback, and extreme-scale normalization guards
//!   (`1e20`, `1e-23`, `1e38`, `eps = 0`). Supports `F32` and `F16` data with `F32` affine params and stats.
//! - Backward `dx` (`group_norm_bwd_dx`): one 256-thread workgroup per `(n, g)` group, `vec4`, register-cached
//!   up to 1,024 units, 1-barrier subgroup / portable tree 3-way reduction tracking `sum(x_hat)` to preserve
//!   large-offset translation invariance (`sum(dx) == 0`).
//! - Backward `dweight` / `dbias` (`group_norm_bwd_dwb`): coalesced per-channel reduction over `N * HW` with
//!   `vec4`, 1-barrier subgroup / portable tree reduction, and split-K when `C` is small and `N * HW` is large.

use super::{check_buf, prod, reduce, use_subgroups, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Max per-group units (`(C/G) * HW / 4` when `VEC4`, else `(C/G) * HW`) cached in registers (4 × 256).
const GN_CACHE_MAX: usize = 1024;
/// Minimum per-channel units before splitting `dweight` / `dbias` across workgroups.
const DWB_SPLIT_MIN_UNITS: usize = 1024;
/// Minimum units per split chunk for `dweight` / `dbias`.
const DWB_MIN_CHUNK: usize = 512;

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
    if (lane == 0u) {
        sh_a[sg] = rs;
        sh_b[sg] = rm;
    }
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
    if (lid == 0u) {
        sh_u = vec2<f32>(sh_a[0], sh_b[0]);
    }
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

/// Forward GroupNorm body.
/// Params: 0 = rows (`n * groups`), 1 = `m_u` (`cpg * hw_u`), 2 = `hw_u`, 3 = `groups`, 4 = `cpg`,
/// 5 = `inv_m` (`1 / (cpg * hw)`), 6 = `eps`, 7 = `sqrt(eps)`.
const GN_FWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<E>;
@group(0) @binding(5) var<storage, read_write> MEAN: array<f32>;
@group(0) @binding(6) var<storage, read_write> RSTD: array<f32>;

fn apply_affine(xh: vec4<f32>, c: u32) -> E {
    var v = xh;
    if (HAS_W) { v *= W[c]; }
    if (HAS_B) { v += vec4<f32>(B[c]); }
    return OUT(v);
}

fn gn_fwd_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let m_u = pu(1u);
    let hw_u = pu(2u);
    let groups = pu(3u);
    let cpg = pu(4u);
    let inv_m = pf(5u);
    let eps = pf(6u);
    let eps_sqrt = pf(7u);
    let valid = row < rows;
    let base = row * m_u;
    let g_base = (row % groups) * cpg;

    if (CACHE_ROW) {
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        let i0 = lid;
        let i1 = lid + 256u;
        let i2 = lid + 512u;
        let i3 = lid + 768u;
        let h0 = valid && i0 < m_u;
        let h1 = valid && i1 < m_u;
        let h2 = valid && i2 < m_u;
        let h3 = valid && i3 < m_u;
        if (h0) { v0 = V4(X[base + i0]); }
        if (h1) { v1 = V4(X[base + i1]); }
        if (h2) { v2 = V4(X[base + i2]); }
        if (h3) { v3 = V4(X[base + i3]); }

        var s = 0.0;
        var am = 0.0;
        if (h0) { s += lane_sum(v0); am = max(am, lane_amax(v0)); }
        if (h1) { s += lane_sum(v1); am = max(am, lane_amax(v1)); }
        if (h2) { s += lane_sum(v2); am = max(am, lane_amax(v2)); }
        if (h3) { s += lane_sum(v3); am = max(am, lane_amax(v3)); }
        let sm = reduce_sum_max(lid, sgid, lane, s, am);
        var mean = sm.x * inv_m;
        let raw_nonfinite = !is_finite1(sm.x) && sm.y > 0.0 && is_finite1(sm.y);
        if (raw_nonfinite) {
            var ss = 0.0;
            if (h0) { ss += lane_sum(ldexp(v0, vec4<i32>(-32))); }
            if (h1) { ss += lane_sum(ldexp(v1, vec4<i32>(-32))); }
            if (h2) { ss += lane_sum(ldexp(v2, vec4<i32>(-32))); }
            if (h3) { ss += lane_sum(ldexp(v3, vec4<i32>(-32))); }
            mean = ldexp(reduce_sum1(lid, sgid, lane, ss) * inv_m, 32i);
        }
        if (h0) { v0 -= vec4<f32>(mean); }
        if (h1) { v1 -= vec4<f32>(mean); }
        if (h2) { v2 -= vec4<f32>(mean); }
        if (h3) { v3 -= vec4<f32>(mean); }

        var sq = 0.0;
        var dam = 0.0;
        if (h0) { sq += lane_sum(v0 * v0); dam = max(dam, lane_amax(v0)); }
        if (h1) { sq += lane_sum(v1 * v1); dam = max(dam, lane_amax(v1)); }
        if (h2) { sq += lane_sum(v2 * v2); dam = max(dam, lane_amax(v2)); }
        if (h3) { sq += lane_sum(v3 * v3); dam = max(dam, lane_amax(v3)); }
        let sq_m = reduce_sum_max(lid, sgid, lane, sq, dam);
        let ms = sq_m.x * inv_m;
        let sq_nan = sq_m.x != sq_m.x;
        let need_scale = !sq_nan && (is_inf1(sq_m.x) || (ms < 1e-30 && eps < 1e-30))
            && sq_m.y > 0.0 && is_finite1(sq_m.y);
        var rstd = inverseSqrt(ms + eps);
        var shift_s = 0i;
        var inv_adj = 1.0;
        var inv_rms_s = rstd;
        if (need_scale) {
            let scale = max(sq_m.y, eps_sqrt);
            shift_s = select(select(0i, 64i, scale < 1e-18), -64i, scale > 1e18);
            inv_adj = 1.0 / ldexp(scale, shift_s);
            let sv = vec4<i32>(shift_s);
            var usq = 0.0;
            if (h0) { let u = ldexp(v0, sv) * inv_adj; usq += lane_sum(u * u); }
            if (h1) { let u = ldexp(v1, sv) * inv_adj; usq += lane_sum(u * u); }
            if (h2) { let u = ldexp(v2, sv) * inv_adj; usq += lane_sum(u * u); }
            if (h3) { let u = ldexp(v3, sv) * inv_adj; usq += lane_sum(u * u); }
            let usq_tot = reduce_sum1(lid, sgid, lane, usq);
            let es = ldexp(eps_sqrt, shift_s) * inv_adj;
            inv_rms_s = inverseSqrt(usq_tot * inv_m + es * es);
            rstd = ldexp(inv_rms_s * inv_adj, shift_s);
        }
        if (!valid) { return; }
        if (lid == 0u) {
            MEAN[row] = mean;
            RSTD[row] = rstd;
        }
        if (need_scale) {
            let sv = vec4<i32>(shift_s);
            if (h0) { Y[base + i0] = apply_affine((ldexp(v0, sv) * inv_adj) * inv_rms_s, g_base + i0 / hw_u); }
            if (h1) { Y[base + i1] = apply_affine((ldexp(v1, sv) * inv_adj) * inv_rms_s, g_base + i1 / hw_u); }
            if (h2) { Y[base + i2] = apply_affine((ldexp(v2, sv) * inv_adj) * inv_rms_s, g_base + i2 / hw_u); }
            if (h3) { Y[base + i3] = apply_affine((ldexp(v3, sv) * inv_adj) * inv_rms_s, g_base + i3 / hw_u); }
        } else {
            if (h0) { Y[base + i0] = apply_affine(v0 * rstd, g_base + i0 / hw_u); }
            if (h1) { Y[base + i1] = apply_affine(v1 * rstd, g_base + i1 / hw_u); }
            if (h2) { Y[base + i2] = apply_affine(v2 * rstd, g_base + i2 / hw_u); }
            if (h3) { Y[base + i3] = apply_affine(v3 * rstd, g_base + i3 / hw_u); }
        }
        return;
    }

    var s = 0.0;
    var am = 0.0;
    if (valid) {
        for (var i = lid; i < m_u; i += 256u) {
            let v = V4(X[base + i]);
            s += lane_sum(v);
            am = max(am, lane_amax(v));
        }
    }
    let sm = reduce_sum_max(lid, sgid, lane, s, am);
    var mean = sm.x * inv_m;
    let raw_nonfinite = !is_finite1(sm.x) && sm.y > 0.0 && is_finite1(sm.y);
    if (raw_nonfinite) {
        var ss = 0.0;
        if (valid) {
            for (var i = lid; i < m_u; i += 256u) {
                ss += lane_sum(ldexp(V4(X[base + i]), vec4<i32>(-32)));
            }
        }
        mean = ldexp(reduce_sum1(lid, sgid, lane, ss) * inv_m, 32i);
    }

    var sq = 0.0;
    var dam = 0.0;
    if (valid) {
        for (var i = lid; i < m_u; i += 256u) {
            let v = V4(X[base + i]) - vec4<f32>(mean);
            sq += lane_sum(v * v);
            dam = max(dam, lane_amax(v));
        }
    }
    let sq_m = reduce_sum_max(lid, sgid, lane, sq, dam);
    let ms = sq_m.x * inv_m;
    let sq_nan = sq_m.x != sq_m.x;
    let need_scale = !sq_nan && (is_inf1(sq_m.x) || (ms < 1e-30 && eps < 1e-30))
        && sq_m.y > 0.0 && is_finite1(sq_m.y);
    var rstd = inverseSqrt(ms + eps);
    var shift_s = 0i;
    var inv_adj = 1.0;
    var inv_rms_s = rstd;
    if (need_scale) {
        let scale = max(sq_m.y, eps_sqrt);
        shift_s = select(select(0i, 64i, scale < 1e-18), -64i, scale > 1e18);
        inv_adj = 1.0 / ldexp(scale, shift_s);
        let sv = vec4<i32>(shift_s);
        var usq = 0.0;
        if (valid) {
            for (var i = lid; i < m_u; i += 256u) {
                let v = V4(X[base + i]) - vec4<f32>(mean);
                let u = ldexp(v, sv) * inv_adj;
                usq += lane_sum(u * u);
            }
        }
        let usq_tot = reduce_sum1(lid, sgid, lane, usq);
        let es = ldexp(eps_sqrt, shift_s) * inv_adj;
        inv_rms_s = inverseSqrt(usq_tot * inv_m + es * es);
        rstd = ldexp(inv_rms_s * inv_adj, shift_s);
    }
    if (!valid) { return; }
    if (lid == 0u) {
        MEAN[row] = mean;
        RSTD[row] = rstd;
    }
    let sv = vec4<i32>(shift_s);
    for (var i = lid; i < m_u; i += 256u) {
        let v = V4(X[base + i]) - vec4<f32>(mean);
        let xh = select(v * rstd, (ldexp(v, sv) * inv_adj) * inv_rms_s, need_scale);
        Y[base + i] = apply_affine(xh, g_base + i / hw_u);
    }
}
"#;

/// Backward `dx` GroupNorm body.
/// Params: 0 = rows (`n * groups`), 1 = `m_u` (`cpg * hw_u`), 2 = `hw_u`, 3 = `groups`, 4 = `cpg`, 5 = `inv_m`.
const GN_BWD_DX_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read> MEAN: array<f32>;
@group(0) @binding(5) var<storage, read> RSTD: array<f32>;
@group(0) @binding(6) var<storage, read_write> DX: array<E>;

fn gn_bwd_dx_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let m_u = pu(1u);
    let hw_u = pu(2u);
    let groups = pu(3u);
    let cpg = pu(4u);
    let inv_m = pf(5u);
    let valid = row < rows;
    let base = row * m_u;
    let g_base = (row % groups) * cpg;
    var mean = 0.0;
    var rstd = 0.0;
    if (valid) {
        mean = MEAN[row];
        rstd = RSTD[row];
    }

    if (CACHE_ROW) {
        var xh0 = vec4<f32>(0.0); var gw0 = vec4<f32>(0.0);
        var xh1 = vec4<f32>(0.0); var gw1 = vec4<f32>(0.0);
        var xh2 = vec4<f32>(0.0); var gw2 = vec4<f32>(0.0);
        var xh3 = vec4<f32>(0.0); var gw3 = vec4<f32>(0.0);
        let i0 = lid;
        let i1 = lid + 256u;
        let i2 = lid + 512u;
        let i3 = lid + 768u;
        let h0 = valid && i0 < m_u;
        let h1 = valid && i1 < m_u;
        let h2 = valid && i2 < m_u;
        let h3 = valid && i3 < m_u;
        var s1 = 0.0;
        var s2 = 0.0;
        var s3 = 0.0;
        if (h0) {
            xh0 = (V4(X[base + i0]) - vec4<f32>(mean)) * rstd;
            gw0 = V4(G[base + i0]);
            if (HAS_W) { gw0 *= W[g_base + i0 / hw_u]; }
            s1 += lane_sum(gw0);
            s2 += lane_sum(gw0 * xh0);
            s3 += lane_sum(xh0);
        }
        if (h1) {
            xh1 = (V4(X[base + i1]) - vec4<f32>(mean)) * rstd;
            gw1 = V4(G[base + i1]);
            if (HAS_W) { gw1 *= W[g_base + i1 / hw_u]; }
            s1 += lane_sum(gw1);
            s2 += lane_sum(gw1 * xh1);
            s3 += lane_sum(xh1);
        }
        if (h2) {
            xh2 = (V4(X[base + i2]) - vec4<f32>(mean)) * rstd;
            gw2 = V4(G[base + i2]);
            if (HAS_W) { gw2 *= W[g_base + i2 / hw_u]; }
            s1 += lane_sum(gw2);
            s2 += lane_sum(gw2 * xh2);
            s3 += lane_sum(xh2);
        }
        if (h3) {
            xh3 = (V4(X[base + i3]) - vec4<f32>(mean)) * rstd;
            gw3 = V4(G[base + i3]);
            if (HAS_W) { gw3 *= W[g_base + i3 / hw_u]; }
            s1 += lane_sum(gw3);
            s2 += lane_sum(gw3 * xh3);
            s3 += lane_sum(xh3);
        }
        let sums = reduce_sum3(lid, sgid, lane, s1, s2, s3);
        if (!valid) { return; }
        let c2_d = sums.y * inv_m;
        let c1_adj = sums.x * inv_m - (sums.z * inv_m) * c2_d;
        if (h0) { DX[base + i0] = OUT(rstd * (gw0 - (vec4<f32>(c1_adj) + xh0 * c2_d))); }
        if (h1) { DX[base + i1] = OUT(rstd * (gw1 - (vec4<f32>(c1_adj) + xh1 * c2_d))); }
        if (h2) { DX[base + i2] = OUT(rstd * (gw2 - (vec4<f32>(c1_adj) + xh2 * c2_d))); }
        if (h3) { DX[base + i3] = OUT(rstd * (gw3 - (vec4<f32>(c1_adj) + xh3 * c2_d))); }
        return;
    }

    var s1 = 0.0;
    var s2 = 0.0;
    var s3 = 0.0;
    if (valid) {
        for (var i = lid; i < m_u; i += 256u) {
            let xh = (V4(X[base + i]) - vec4<f32>(mean)) * rstd;
            var gw = V4(G[base + i]);
            if (HAS_W) { gw *= W[g_base + i / hw_u]; }
            s1 += lane_sum(gw);
            s2 += lane_sum(gw * xh);
            s3 += lane_sum(xh);
        }
    }
    let sums = reduce_sum3(lid, sgid, lane, s1, s2, s3);
    if (!valid) { return; }
    let c2_d = sums.y * inv_m;
    let c1_adj = sums.x * inv_m - (sums.z * inv_m) * c2_d;
    for (var i = lid; i < m_u; i += 256u) {
        let xh = (V4(X[base + i]) - vec4<f32>(mean)) * rstd;
        var gw = V4(G[base + i]);
        if (HAS_W) { gw *= W[g_base + i / hw_u]; }
        DX[base + i] = OUT(rstd * (gw - (vec4<f32>(c1_adj) + xh * c2_d)));
    }
}
"#;

/// Per-channel `dweight` / `dbias` reduction across `N * hw_u` (with optional split-K).
/// Params: 0 = `total_wgs` (`c * splits`), 1 = `c`, 2 = `groups`, 3 = `cpg`, 4 = `hw_u`,
/// 5 = `m_c` (`n * hw_u`), 6 = `splits`, 7 = `chunk`.
const GN_BWD_DWB_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> RSTD: array<f32>;
@group(0) @binding(5) var<storage, read_write> DW: array<f32>;
@group(0) @binding(6) var<storage, read_write> DB: array<f32>;

fn gn_bwd_dwb_run(wid: vec3<u32>, nwg: vec3<u32>, lid: u32, sgid: u32, lane: u32) {
    let wg = wid.y * nwg.x + wid.x;
    let total_wgs = pu(0u);
    let c = pu(1u);
    let groups = pu(2u);
    let cpg = pu(3u);
    let hw_u = pu(4u);
    let m_c = pu(5u);
    let splits = pu(6u);
    let chunk = pu(7u);
    let valid = wg < total_wgs;
    var aw = 0.0;
    var ab = 0.0;
    if (valid) {
        let ch = wg / splits;
        let sidx = wg % splits;
        let g_idx = ch / cpg;
        let start = sidx * chunk;
        let end = min(start + chunk, m_c);
        for (var e = start + lid; e < end; e += 256u) {
            let ni = e / hw_u;
            let hi = e % hw_u;
            let idx = (ni * c + ch) * hw_u + hi;
            let gv = V4(G[idx]);
            if (NEED_W) {
                let st = ni * groups + g_idx;
                let xh = (V4(X[idx]) - vec4<f32>(MEAN[st])) * RSTD[st];
                aw += lane_sum(gv * xh);
            }
            if (NEED_B) {
                ab += lane_sum(gv);
            }
        }
    }
    let sums = reduce_sum3(lid, sgid, lane, aw, ab, 0.0);
    if (lid == 0u && valid) {
        if (NEED_W) { DW[wg] = sums.x; }
        if (NEED_B) { DB[wg] = sums.y; }
    }
}
"#;

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
    let pre = prelude(s);
    let cache = s.get_bool("CACHE");
    let red = if s.is_subgroups() { REDUCE_SUBGROUPS } else { REDUCE_PORTABLE };
    let entry = entry_for("gn_fwd_run", s.is_subgroups());
    format!("{pre}const CACHE_ROW: bool = {cache};\n{LANE_HELPERS}{red}{GN_FWD_BODY}{entry}")
}

fn gen_bwd_dx(s: &Spec) -> String {
    let pre = prelude(s);
    let cache = s.get_bool("CACHE");
    let red = if s.is_subgroups() { REDUCE_SUBGROUPS } else { REDUCE_PORTABLE };
    let entry = entry_for("gn_bwd_dx_run", s.is_subgroups());
    format!("{pre}const CACHE_ROW: bool = {cache};\n{LANE_HELPERS}{red}{GN_BWD_DX_BODY}{entry}")
}

fn gen_bwd_dwb(s: &Spec) -> String {
    let pre = prelude(s);
    let red = if s.is_subgroups() { REDUCE_SUBGROUPS } else { REDUCE_PORTABLE };
    let entry = entry_for("gn_bwd_dwb_run", s.is_subgroups());
    format!("{pre}{LANE_HELPERS}{red}{GN_BWD_DWB_BODY}{entry}")
}

use Access::{Read as R, ReadWrite as RW};

static GN_FWD: KernelDef =
    KernelDef { name: "group_norm_fwd", storage: &[R, R, R, RW, RW, RW], source: Source::Generated(gen_fwd) };
static GN_BWD_DX: KernelDef = KernelDef {
    name: "group_norm_bwd_dx",
    storage: &[R, R, R, R, R, RW],
    source: Source::Generated(gen_bwd_dx),
};
static GN_BWD_DWB: KernelDef = KernelDef {
    name: "group_norm_bwd_dwb",
    storage: &[R, R, R, R, RW, RW],
    source: Source::Generated(gen_bwd_dwb),
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&GN_FWD, &GN_BWD_DX, &GN_BWD_DWB];

fn check_group_dims(
    n: usize,
    c: usize,
    hw: usize,
    groups: usize,
    op: &'static str,
) -> Result<(usize, usize, usize)> {
    if groups == 0 || !c.is_multiple_of(groups) {
        return Err(shape_err(format!("{op}: channels {c} must be divisible by non-zero groups {groups}")));
    }
    let cpg = c / groups;
    let total = prod(&[n, c, hw])?;
    let rows = prod(&[n, groups])?;
    if n > 0 && (cpg == 0 || hw == 0) {
        return Err(shape_err(format!(
            "{op}: cannot normalize empty group ({n}, {c}, {hw}) with G={groups}"
        )));
    }
    if total > u32::MAX as usize {
        return Err(GpuError::Limit(format!("{op}: tensor exceeds u32 indexing")));
    }
    Ok((cpg, total, rows))
}

fn dwb_split_plan(c: usize, m_c: usize) -> (usize, usize) {
    if c >= 128 || m_c < DWB_SPLIT_MIN_UNITS {
        return (1, m_c);
    }
    let want = 256usize.div_ceil(c.max(1));
    let max_s = (m_c / DWB_MIN_CHUNK).max(1);
    let s0 = want.min(max_s);
    if s0 <= 1 {
        return (1, m_c);
    }
    let chunk = m_c.div_ceil(s0);
    let splits = m_c.div_ceil(chunk);
    (splits, chunk)
}

/// Fused single-pass GroupNorm forward over dense `[n, c, hw]` (`dtype`: `F32` or `F16` with `shader-f16`).
///
/// Writes normalized output `y [n, c, hw]` (same dtype as `x`), plus per-group `mean [n, groups]` and
/// `rstd [n, groups]` in `F32`. Optional per-channel `weight [c]` and `bias [c]` are `F32`.
#[allow(clippy::too_many_arguments)]
pub fn group_norm_forward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    (n, c, hw): (usize, usize, usize),
    groups: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "group_norm_forward")?;
    if !eps.is_finite() || eps < 0.0 {
        return Err(shape_err(format!("group_norm_forward: invalid eps {eps}")));
    }
    let (cpg, total, rows) = check_group_dims(n, c, hw, groups, "group_norm_forward")?;
    check_buf(x, total, dtype, "group_norm input")?;
    check_buf(y, total, dtype, "group_norm output")?;
    check_buf(mean, rows, DType::F32, "group_norm mean")?;
    check_buf(rstd, rows, DType::F32, "group_norm rstd")?;
    if let Some(w) = weight {
        check_buf(w, c, DType::F32, "group_norm weight")?;
    }
    if let Some(b) = bias {
        check_buf(b, c, DType::F32, "group_norm bias")?;
    }
    if total == 0 {
        return Ok(());
    }
    let vec4 = hw.is_multiple_of(4);
    let hw_u = if vec4 { hw / 4 } else { hw };
    let m_u = cpg * hw_u;
    let m_elems = cpg * hw;
    let k = ctx.kernel(
        &GN_FWD,
        Spec::new()
            .bool("HAS_W", weight.is_some())
            .bool("HAS_B", bias.is_some())
            .bool("VEC4", vec4)
            .bool("CACHE", m_u <= GN_CACHE_MAX)
            .f16(dtype == DType::F16)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new()
        .uz(rows)?
        .uz(m_u)?
        .uz(hw_u)?
        .uz(groups)?
        .uz(cpg)?
        .f((1.0 / m_elems as f64) as f32)
        .f(eps)
        .f(eps.sqrt());
    ctx.dispatch(
        &k,
        &[x, weight.unwrap_or(ctx.dummy()), bias.unwrap_or(ctx.dummy()), y, mean, rstd],
        &p,
        ctx.grid_1d(rows as u64 * 256, 256)?,
    )
}

/// Fused GroupNorm backward over dense `[n, c, hw]` (`dtype`: `F32` or `F16` with `shader-f16`).
///
/// Writes `dx [n, c, hw]` (same dtype as `x` and `grad`) and optional `F32` parameter gradients
/// `dweight [c]` and `dbias [c]`.
#[allow(clippy::too_many_arguments)]
pub fn group_norm_backward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    dbias: Option<&wgpu::Buffer>,
    (n, c, hw): (usize, usize, usize),
    groups: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "group_norm_backward")?;
    let (cpg, total, rows) = check_group_dims(n, c, hw, groups, "group_norm_backward")?;
    check_buf(x, total, dtype, "group_norm_backward x")?;
    check_buf(grad, total, dtype, "group_norm_backward grad")?;
    check_buf(dx, total, dtype, "group_norm_backward dx")?;
    check_buf(mean, rows, DType::F32, "group_norm_backward mean")?;
    check_buf(rstd, rows, DType::F32, "group_norm_backward rstd")?;
    if let Some(w) = weight {
        check_buf(w, c, DType::F32, "group_norm_backward weight")?;
    }
    if let Some(dw) = dweight {
        check_buf(dw, c, DType::F32, "group_norm_backward dweight")?;
    }
    if let Some(db) = dbias {
        check_buf(db, c, DType::F32, "group_norm_backward dbias")?;
    }
    if total == 0 {
        if let Some(dw) = dweight {
            ctx.clear(dw, DType::F32.bytes(c));
        }
        if let Some(db) = dbias {
            ctx.clear(db, DType::F32.bytes(c));
        }
        return Ok(());
    }
    let vec4 = hw.is_multiple_of(4);
    let hw_u = if vec4 { hw / 4 } else { hw };
    let m_u = cpg * hw_u;
    let m_elems = cpg * hw;
    let sg = use_subgroups(ctx);
    let half = dtype == DType::F16;

    let k_dx = ctx.kernel(
        &GN_BWD_DX,
        Spec::new()
            .bool("HAS_W", weight.is_some())
            .bool("VEC4", vec4)
            .bool("CACHE", m_u <= GN_CACHE_MAX)
            .f16(half)
            .subgroups(sg),
    )?;
    let p_dx =
        Params::new().uz(rows)?.uz(m_u)?.uz(hw_u)?.uz(groups)?.uz(cpg)?.f((1.0 / m_elems as f64) as f32);
    ctx.dispatch(
        &k_dx,
        &[x, grad, weight.unwrap_or(ctx.dummy()), mean, rstd, dx],
        &p_dx,
        ctx.grid_1d(rows as u64 * 256, 256)?,
    )?;

    if dweight.is_none() && dbias.is_none() {
        return Ok(());
    }
    let m_c = prod(&[n, hw_u])?;
    let (splits, chunk) = dwb_split_plan(c, m_c);
    let total_wgs = prod(&[c, splits])?;
    let k_dwb = ctx.kernel(
        &GN_BWD_DWB,
        Spec::new()
            .bool("NEED_W", dweight.is_some())
            .bool("NEED_B", dbias.is_some())
            .bool("VEC4", vec4)
            .f16(half)
            .subgroups(sg),
    )?;
    let p_dwb =
        Params::new().uz(total_wgs)?.uz(c)?.uz(groups)?.uz(cpg)?.uz(hw_u)?.uz(m_c)?.uz(splits)?.uz(chunk)?;
    let grid = ctx.grid_1d(total_wgs as u64 * 256, 256)?;

    if splits == 1 {
        let scratch_w = dweight.is_none().then(|| ctx.alloc(4)).transpose()?;
        let scratch_b = dbias.is_none().then(|| ctx.alloc(4)).transpose()?;
        let dw_b = dweight.unwrap_or_else(|| scratch_w.as_ref().unwrap().raw());
        let db_b = dbias.unwrap_or_else(|| scratch_b.as_ref().unwrap().raw());
        ctx.dispatch(&k_dwb, &[x, grad, mean, rstd, dw_b, db_b], &p_dwb, grid)?;
    } else {
        let part_bytes = DType::F32.bytes(total_wgs);
        let pw = ctx.alloc(if dweight.is_some() { part_bytes } else { 4 })?;
        let pb = ctx.alloc(if dbias.is_some() { part_bytes } else { 4 })?;
        ctx.dispatch(&k_dwb, &[x, grad, mean, rstd, pw.raw(), pb.raw()], &p_dwb, grid)?;
        if let Some(dw) = dweight {
            reduce::reduce_as(
                ctx,
                reduce::ReduceOp::Sum,
                (pw.raw(), DType::F32),
                (dw, DType::F32),
                c,
                splits,
                1,
            )?;
        }
        if let Some(db) = dbias {
            reduce::reduce_as(
                ctx,
                reduce::ReduceOp::Sum,
                (pb.raw(), DType::F32),
                (db, DType::F32),
                c,
                splits,
                1,
            )?;
        }
    }
    Ok(())
}

/// Fused single-pass InstanceNorm forward over dense `[n, c, hw]` (GroupNorm with `groups == c`).
#[allow(clippy::too_many_arguments)]
pub fn instance_norm_forward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    (n, c, hw): (usize, usize, usize),
    eps: f32,
    dtype: DType,
) -> Result<()> {
    group_norm_forward(ctx, x, weight, bias, y, mean, rstd, (n, c, hw), c, eps, dtype)
}

/// Fused InstanceNorm backward over dense `[n, c, hw]` (GroupNorm backward with `groups == c`).
#[allow(clippy::too_many_arguments)]
pub fn instance_norm_backward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    weight: Option<&wgpu::Buffer>,
    mean: &wgpu::Buffer,
    rstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dweight: Option<&wgpu::Buffer>,
    dbias: Option<&wgpu::Buffer>,
    (n, c, hw): (usize, usize, usize),
    dtype: DType,
) -> Result<()> {
    group_norm_backward(ctx, x, grad, weight, mean, rstd, dx, dweight, dbias, (n, c, hw), c, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_norm_portable_emits_no_subgroup_tokens() {
        for (def, spec) in [
            (
                &GN_FWD,
                Spec::new()
                    .bool("HAS_W", true)
                    .bool("HAS_B", true)
                    .bool("VEC4", true)
                    .bool("CACHE", true)
                    .subgroups(false),
            ),
            (
                &GN_BWD_DX,
                Spec::new().bool("HAS_W", true).bool("VEC4", false).bool("CACHE", false).subgroups(false),
            ),
            (
                &GN_BWD_DWB,
                Spec::new().bool("NEED_W", true).bool("NEED_B", true).bool("VEC4", true).subgroups(false),
            ),
        ] {
            let wgsl = crate::pipeline::render(def, &spec);
            assert!(!wgsl.contains("subgroup"), "portable {} emitted subgroup token:\n{wgsl}", def.name);
            if def.name == "group_norm_fwd" {
                assert!(
                    wgsl.contains("workgroupUniformLoad"),
                    "group_norm_fwd portable must return uniform reduction via workgroupUniformLoad"
                );
            }
        }
    }
}
