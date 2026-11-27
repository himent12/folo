//! Single-dispatch train-mode BatchNorm forward and backward kernels:
//! - Register-cached (`REGS = 4` with `WG = 256 | 512` or `REGS = 8` with `WG = 256`, up to 2,048 units =
//!   8,192 elements/ch under `VEC4`): one workgroup per channel holds its slice in registers and reads `X` (and
//!   `G` in backward) once. Using `WG = 512, REGS = 4` for `1024 < m_u <= 2048` halves per-thread register
//!   pressure in backward (`xh0..3, dz0..3` instead of `xh0..7, dz0..7`) and doubles active warps on narrower `C`.
//! - Streaming two-pass (`REGS = 0`, `WG = 256 | 512`, up to [`FUSED_STREAM_MED_UNITS`] = 6,400 units with
//!   `C >= 32` and [`FUSED_STREAM_MAX_UNITS`] = 25,600 units = 102,400 elements/ch under `VEC4` with `C >= 64`):
//!   keeps L2-resident medium and 80×80 layers in a single dispatch with zero intermediate `PART`/`COEF` buffers,
//!   multiply-free pointer stepping, and 2× unrolled `vec4` loads in backward.
//! - Optional `STRIDED` specialization reads channel-slice NCHW views (`offset`, `n_stride`) directly in place,
//!   avoiding pre-copying non-contiguous `narrow`/`chunk` gradients.

use std::fmt::Write as _;

use super::{COMMON, prelude};
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// Minimum storage buffers needed by [`FUSED_BWD`] (9: X, G, MEAN, INVSTD, W, B, DX, DW, DB).
pub(super) const FUSED_BWD_BUFFERS: u32 = 9;
/// Max units (`N · hw_units`) handled by the register-cached fused BatchNorm kernels
/// (`WG = 256, REGS = 4` up to 1,024 units; `WG = 512, REGS = 4` or `WG = 256, REGS = 8` up to 2,048 units).
pub(super) const FUSED_MAX_UNITS: usize = 256 * 8;
/// Medium streaming unit cap (6,400 units = 25,600 elems/ch under `VEC4`, e.g. `16×40×40`) eligible at `C >= 32`.
pub(super) const FUSED_STREAM_MED_UNITS: usize = 256 * 25;
/// Minimum channels for medium streaming single-dispatch BatchNorm (`m_u <= 6,400`).
pub(super) const FUSED_STREAM_MIN_CHANNELS_MED: usize = 32;
/// Large streaming unit cap (25,600 units = 102,400 elems/ch under `VEC4`, covering `16×80×80` and `32×40×40`
/// when `C >= 64` so each channel's slice stays L2-warm between Pass 1 and Pass 2).
pub(super) const FUSED_STREAM_MAX_UNITS: usize = 256 * 100;
/// Minimum channels for large streaming single-dispatch BatchNorm (`6,400 < m_u <= 25,600`).
pub(super) const FUSED_STREAM_MIN_CHANNELS_LARGE: usize = 64;

/// Bounds check for precompiling `"bn_fused_fwd"` / `"bn_fused_bwd"` from a kernel list (`kernels::bounds`).
pub(crate) fn fused_fits(spec: &Spec) -> bool {
    let regs = spec.get_u32("REGS", 4);
    let wg = spec.get_u32("WG", 256);
    matches!(wg, 256 | 512) && matches!(regs, 0 | 4 | 8) && !(wg == 512 && regs == 8)
}

fn push_welford_tree(s: &mut String, wg: u32) {
    let half = wg / 2;
    let _ = write!(
        s,
        "var<workgroup> wn: array<f32, {wg}>;\n\
         var<workgroup> wm: array<f32, {wg}>;\n\
         var<workgroup> wq: array<f32, {wg}>;\n\
         fn bn_tree_merge(lid: u32) {{\n\
         \x20 for (var s = {half}u; s > 0u; s >>= 1u) {{\n\
         \x20   if (lid < s) {{\n\
         \x20     let na = wn[lid]; let nb = wn[lid + s];\n\
         \x20     if (na == 0.0) {{ wn[lid] = nb; wm[lid] = wm[lid + s]; wq[lid] = wq[lid + s]; }}\n\
         \x20     else if (nb > 0.0) {{\n\
         \x20       let n = na + nb; let d = wm[lid + s] - wm[lid]; let delta = d * (nb / n);\n\
         \x20       wm[lid] += delta; wq[lid] += wq[lid + s] + d * (delta * na); wn[lid] = n;\n\
         \x20     }}\n\
         \x20   }}\n\
         \x20   workgroupBarrier();\n\
         \x20 }}\n\
         }}\n"
    );
}

/// Streaming two-pass single-dispatch train-mode BatchNorm forward (`REGS = 0`, `WG = 256 | 512`).
/// Params: 0 = C, 1 = hw_u, 2 = m_u, 3 = x_off_u, 4 = x_n_stride_u, 5 = WG / hw_u, 6 = WG % hw_u,
/// 7 = eps, 8 = 1 - momentum, 9 = momentum, 10 = momentum · n / (n - 1).
fn gen_fused_fwd_streaming(spec: &Spec) -> String {
    let wg = if spec.get_u32("WG", 256) == 512 { 512u32 } else { 256u32 };
    let n_sg = wg / 32;
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let strided = spec.get_bool("STRIDED");
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    if sg {
        let _ = write!(
            s,
            "var<workgroup> wn: array<f32, {n_sg}>;\n\
             var<workgroup> wm: array<f32, {n_sg}>;\n\
             var<workgroup> wq: array<f32, {n_sg}>;\n\
             var<workgroup> wmean: f32;\n\
             var<workgroup> winv: f32;\n"
        );
    } else {
        push_welford_tree(&mut s, wg);
        s.push_str("var<workgroup> wmean: f32;\nvar<workgroup> winv: f32;\n");
    }
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> B: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<E>;\n\
         @group(0) @binding(5) var<storage, read_write> MEAN: array<f32>;\n\
         @group(0) @binding(6) var<storage, read_write> INVSTD: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> RM: array<f32>;\n\
         @group(0) @binding(8) var<storage, read_write> RV: array<f32>;\n\
         @compute @workgroup_size({wg})\n"
    );
    if sg {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n",
        );
    } else {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {\n",
        );
    }
    let x_off = if strided { "pu(3u)" } else { "0u" };
    let x_ns = if strided { "pu(4u)" } else { "channels * hw" };
    let _ = write!(
        s,
        "  let c = wid.x;\n\
         \x20 let channels = pu(0u);\n\
         \x20 let hw = pu(1u);\n\
         \x20 let m_u = pu(2u);\n\
         \x20 let ch_off = c * hw;\n\
         \x20 let x_off = {x_off};\n\
         \x20 let x_ns = {x_ns};\n\
         \x20 let step_n = pu(5u);\n\
         \x20 let step_i = pu(6u);\n\
         \x20 let x_step = step_n * x_ns + step_i;\n\
         \x20 let x_gap = x_ns - hw;\n\
         \x20 let x0 = V4(X[x_off + ch_off]).x;\n\
         \x20 let base = select(0.0, x0, abs(x0) < 1e37);\n\
         \x20 var k = 0.0;\n\
         \x20 var m4 = vec4<f32>(0.0);\n\
         \x20 var q4 = vec4<f32>(0.0);\n\
         \x20 var e = lid;\n\
         \x20 let n0 = e / hw;\n\
         \x20 var wi = e - n0 * hw;\n\
         \x20 var x_idx = x_off + n0 * x_ns + ch_off + wi;\n\
         \x20 loop {{\n\
         \x20   if (e >= m_u) {{ break; }}\n\
         \x20   let x = V4(X[x_idx]) - base;\n\
         \x20   k += 1.0;\n\
         \x20   let inv_k = 1.0 / k;\n"
    );
    if vec4 {
        let _ = write!(
            s,
            "    let d = x - m4;\n\
             \x20   m4 += d * inv_k;\n\
             \x20   q4 += d * (x - m4);\n\
             \x20   e += {wg}u; wi += step_i; x_idx += x_step;\n\
             \x20   if (wi >= hw) {{ wi -= hw; x_idx += x_gap; }}\n\
             \x20 }}\n\
             \x20 var na = 0.0; var ma = 0.0; var qa = 0.0;\n\
             \x20 if (k > 0.0) {{\n\
             \x20   let d01 = m4.y - m4.x; let m01 = m4.x + 0.5 * d01;\n\
             \x20   let q01 = q4.x + q4.y + (0.5 * k * d01) * d01;\n\
             \x20   let d23 = m4.w - m4.z; let m23 = m4.z + 0.5 * d23;\n\
             \x20   let q23 = q4.z + q4.w + (0.5 * k * d23) * d23;\n\
             \x20   let d_all = m23 - m01;\n\
             \x20   na = 4.0 * k; ma = m01 + 0.5 * d_all; qa = q01 + q23 + (k * d_all) * d_all;\n\
             \x20 }}\n"
        );
    } else {
        let _ = write!(
            s,
            "    let d = x.x - m4.x;\n\
             \x20   m4.x += d * inv_k;\n\
             \x20   q4.x += d * (x.x - m4.x);\n\
             \x20   e += {wg}u; wi += step_i; x_idx += x_step;\n\
             \x20   if (wi >= hw) {{ wi -= hw; x_idx += x_gap; }}\n\
             \x20 }}\n\
             \x20 var na = k; var ma = m4.x; var qa = q4.x;\n"
        );
    }
    if sg {
        let sg_last = n_sg - 1;
        let _ = write!(
            s,
            "  for (var off = 16u; off > 0u; off = off >> 1u) {{\n\
             \x20   let nb = subgroupShuffleDown(na, off);\n\
             \x20   let mb = subgroupShuffleDown(ma, off);\n\
             \x20   let qb = subgroupShuffleDown(qa, off);\n\
             \x20   if (lane < off) {{\n\
             \x20     if (na == 0.0) {{ na = nb; ma = mb; qa = qb; }}\n\
             \x20     else if (nb > 0.0) {{\n\
             \x20       let n = na + nb; let d = mb - ma; let delta = d * (nb / n);\n\
             \x20       ma += delta; qa += qb + d * (delta * na); na = n;\n\
             \x20     }}\n\
             \x20   }}\n\
             \x20 }}\n\
             \x20 let sg = min(sgid, {sg_last}u);\n\
             \x20 if (lane == 0u) {{ wn[sg] = na; wm[sg] = ma; wq[sg] = qa; }}\n\
             \x20 workgroupBarrier();\n\
             \x20 if (lid == 0u) {{\n\
             \x20   var fn_a = wn[0]; var fm_a = wm[0]; var fq_a = wq[0];\n\
             \x20   for (var i = 1u; i < {n_sg}u; i += 1u) {{\n\
             \x20     let nb = wn[i];\n\
             \x20     if (fn_a == 0.0) {{ fn_a = nb; fm_a = wm[i]; fq_a = wq[i]; }}\n\
             \x20     else if (nb > 0.0) {{\n\
             \x20       let n = fn_a + nb; let d = wm[i] - fm_a; let delta = d * (nb / n);\n\
             \x20       fm_a += delta; fq_a += wq[i] + d * (delta * fn_a); fn_a = n;\n\
             \x20     }}\n\
             \x20   }}\n\
             \x20   let var_raw = fq_a / fn_a;\n\
             \x20   let var_b = select(var_raw, 0.0, var_raw < 0.0);\n\
             \x20   let inv_c = 1.0 / sqrt(var_b + pf(7u));\n\
             \x20   let mean_c = base + fm_a;\n\
             \x20   MEAN[c] = mean_c; INVSTD[c] = inv_c;\n\
             \x20   if (RUNNING) {{ RM[c] = RM[c] * pf(8u) + pf(9u) * mean_c; RV[c] = RV[c] * pf(8u) + pf(10u) * var_b; }}\n\
             \x20   wmean = fm_a; winv = inv_c;\n\
             \x20 }}\n\
             \x20 workgroupBarrier();\n"
        );
    } else {
        s.push_str(
            "  wn[lid] = na; wm[lid] = ma; wq[lid] = qa;\n\
             \x20 workgroupBarrier();\n\
             \x20 bn_tree_merge(lid);\n\
             \x20 if (lid == 0u) {\n\
             \x20   let fm_a = wm[0];\n\
             \x20   let var_raw = wq[0] / wn[0];\n\
             \x20   let var_b = select(var_raw, 0.0, var_raw < 0.0);\n\
             \x20   let inv_c = 1.0 / sqrt(var_b + pf(7u));\n\
             \x20   let mean_c = base + fm_a;\n\
             \x20   MEAN[c] = mean_c; INVSTD[c] = inv_c;\n\
             \x20   if (RUNNING) { RM[c] = RM[c] * pf(8u) + pf(9u) * mean_c; RV[c] = RV[c] * pf(8u) + pf(10u) * var_b; }\n\
             \x20   wmean = fm_a; winv = inv_c;\n\
             \x20 }\n\
             \x20 workgroupBarrier();\n",
        );
    }
    s.push_str(
        "  let md_c = wmean;\n\
         \x20 let inv_c = winv;\n\
         \x20 let w_c = W[c];\n\
         \x20 let b_c = B[c];\n\
         \x20 e = lid;\n\
         \x20 wi = e - n0 * hw;\n\
         \x20 x_idx = x_off + n0 * x_ns + ch_off + wi;\n",
    );
    if strided {
        s.push_str(
            "  let y_ns = channels * hw;\n\
             \x20 let y_step = step_n * y_ns + step_i;\n\
             \x20 let y_gap = y_ns - hw;\n\
             \x20 var y_idx = n0 * y_ns + ch_off + wi;\n",
        );
    }
    let y_target = if strided { "y_idx" } else { "x_idx" };
    let y_step = if strided { " y_idx += y_step;" } else { "" };
    let y_gap = if strided { " y_idx += y_gap;" } else { "" };
    s.push_str("  loop {\n    if (e >= m_u) { break; }\n");
    if vec4 {
        let _ = write!(
            s,
            "    let xh = ((V4(X[x_idx]) - base) - md_c) * inv_c;\n\
             \x20   Y[{y_target}] = OUT(bn_act(xh * w_c + b_c));\n"
        );
    } else {
        let _ = write!(
            s,
            "    let xh = ((V4(X[x_idx]).x - base) - md_c) * inv_c;\n\
             \x20   Y[{y_target}] = OUT(vec4<f32>(bn_act_f32(xh * w_c + b_c)));\n"
        );
    }
    let _ = write!(
        s,
        "    e += {wg}u; wi += step_i; x_idx += x_step;{y_step}\n\
         \x20   if (wi >= hw) {{ wi -= hw; x_idx += x_gap;{y_gap} }}\n\
         \x20 }}\n}}\n"
    );
    s
}

/// Single-dispatch train-mode BatchNorm forward: register-cached (`REGS = 4 | 8`) or streaming two-pass
/// (`REGS = 0`), with `WG = 256 | 512` and optional `STRIDED` input view.
/// Grid: `(C, 1, 1)`. Params: 0 = C, 1 = hw_u, 2 = m_u, 3 = x_off_u, 4 = x_n_stride_u,
/// 5 = WG / hw_u, 6 = WG % hw_u, 7 = eps, 8 = 1 - momentum, 9 = momentum, 10 = momentum · n / (n - 1).
fn gen_fused_fwd(spec: &Spec) -> String {
    let raw_regs = spec.get_u32("REGS", 4);
    if raw_regs == 0 {
        return gen_fused_fwd_streaming(spec);
    }
    let wg = if spec.get_u32("WG", 256) == 512 { 512u32 } else { 256u32 };
    let n_sg = wg / 32;
    let sg_last = n_sg - 1;
    let regs = if wg == 512 { raw_regs.clamp(1, 4) } else { raw_regs.clamp(1, 8) };
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let half = spec.is_f16();
    let strided = spec.get_bool("STRIDED");
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    if sg {
        let _ = write!(
            s,
            "var<workgroup> wbase: f32;\n\
             var<workgroup> ws: array<f32, {n_sg}>;\n\
             var<workgroup> wq: array<f32, {n_sg}>;\n"
        );
    } else {
        let _ = write!(
            s,
            "var<workgroup> wbase: f32;\n\
             var<workgroup> wmd: f32;\n\
             var<workgroup> ws: array<f32, {wg}>;\n"
        );
    }
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> B: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<E>;\n\
         @group(0) @binding(5) var<storage, read_write> MEAN: array<f32>;\n\
         @group(0) @binding(6) var<storage, read_write> INVSTD: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> RM: array<f32>;\n\
         @group(0) @binding(8) var<storage, read_write> RV: array<f32>;\n\
         @compute @workgroup_size({wg})\n"
    );
    if sg {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n",
        );
    } else {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {\n",
        );
    }
    let mf_expr = if vec4 { "f32(m_u) * 4.0" } else { "f32(m_u)" };
    let x_off = if strided { "pu(3u)" } else { "0u" };
    let x_ns = if strided { "pu(4u)" } else { "channels * hw" };
    let _ = write!(
        s,
        "  let c = wid.x;\n\
         \x20 let channels = pu(0u);\n\
         \x20 let hw = pu(1u);\n\
         \x20 let m_u = pu(2u);\n\
         \x20 let m_f = {mf_expr};\n\
         \x20 let ch_off = c * hw;\n\
         \x20 let x_off = {x_off};\n\
         \x20 let x_ns = {x_ns};\n\
         \x20 let step_n = pu(5u);\n\
         \x20 let step_i = pu(6u);\n\
         \x20 let x_step = step_n * x_ns + step_i;\n\
         \x20 let x_gap = x_ns - hw;\n\
         \x20 let n0 = lid / hw;\n\
         \x20 var wi = lid - n0 * hw;\n\
         \x20 var x_idx = x_off + n0 * x_ns + ch_off + wi;\n"
    );
    for r in 0..regs {
        let _ = writeln!(s, "  var v{r}: E;");
    }
    for r in 0..regs {
        let off = r * wg;
        let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ v{r} = X[x_idx]; }}");
        if r + 1 < regs {
            let _ = writeln!(
                s,
                "  wi += step_i; x_idx += x_step; let wx{r} = wi >= hw; wi = select(wi, wi - hw, wx{r}); x_idx += select(0u, x_gap, wx{r});"
            );
        }
    }
    s.push_str(
        "  if (lid == 0u) {\n\
         \x20   let x0 = V4(v0).x;\n\
         \x20   wbase = select(0.0, x0, abs(x0) <= 3.4028234e38);\n\
         \x20 }\n\
         \x20 workgroupBarrier();\n\
         \x20 let base = wbase;\n",
    );
    if vec4 {
        s.push_str("  var s4 = vec4<f32>(0.0);\n");
        for r in 0..regs {
            let off = r * wg;
            if half {
                let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ s4 += V4(v{r}) - base; }}");
            } else {
                let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ v{r} = v{r} - base; s4 += v{r}; }}");
            }
        }
        s.push_str("  let s_t = (s4.x + s4.y) + (s4.z + s4.w);\n");
    } else {
        s.push_str("  var s_t = 0.0;\n");
        for r in 0..regs {
            let off = r * wg;
            if half {
                let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ s_t += V4(v{r}).x - base; }}");
            } else {
                let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ v{r} = v{r} - base; s_t += v{r}; }}");
            }
        }
    }
    if sg {
        let _ = write!(
            s,
            "  let rs = subgroupAdd(s_t);\n\
             \x20 let sg = min(sgid, {sg_last}u);\n\
             \x20 if (lane == 0u) {{ ws[sg] = rs; }}\n\
             \x20 workgroupBarrier();\n\
             \x20 let sum_s = subgroupAdd(select(0.0, ws[min(lane, {sg_last}u)], lane < {n_sg}u));\n\
             \x20 let mean_delta = sum_s / m_f;\n"
        );
    } else {
        let _ = write!(
            s,
            "  ws[lid] = s_t;\n\
             \x20 workgroupBarrier();\n\
             \x20 for (var step = {}u; step > 0u; step >>= 1u) {{\n\
             \x20   if (lid < step) {{ ws[lid] += ws[lid + step]; }}\n\
             \x20   workgroupBarrier();\n\
             \x20 }}\n\
             \x20 if (lid == 0u) {{ wmd = ws[0] / m_f; }}\n\
             \x20 workgroupBarrier();\n\
             \x20 let mean_delta = wmd;\n",
            wg / 2
        );
    }
    if vec4 {
        s.push_str("  var q4 = vec4<f32>(0.0);\n");
        for r in 0..regs {
            let off = r * wg;
            if half {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{ let d = (V4(v{r}) - base) - mean_delta; q4 = fma(d, d, q4); }}"
                );
            } else {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{ let d = v{r} - mean_delta; q4 = fma(d, d, q4); }}"
                );
            }
        }
        s.push_str("  let q_t = (q4.x + q4.y) + (q4.z + q4.w);\n");
    } else {
        s.push_str("  var q_t = 0.0;\n");
        for r in 0..regs {
            let off = r * wg;
            if half {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{ let d = (V4(v{r}).x - base) - mean_delta; q_t = fma(d, d, q_t); }}"
                );
            } else {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{ let d = v{r} - mean_delta; q_t = fma(d, d, q_t); }}"
                );
            }
        }
    }
    if sg {
        let _ = write!(
            s,
            "  let rq = subgroupAdd(q_t);\n\
             \x20 if (lane == 0u) {{ wq[sg] = rq; }}\n\
             \x20 workgroupBarrier();\n\
             \x20 let sum_q = subgroupAdd(select(0.0, wq[min(lane, {sg_last}u)], lane < {n_sg}u));\n"
        );
    } else {
        let _ = write!(
            s,
            "  ws[lid] = q_t;\n\
             \x20 workgroupBarrier();\n\
             \x20 for (var step = {}u; step > 0u; step >>= 1u) {{\n\
             \x20   if (lid < step) {{ ws[lid] += ws[lid + step]; }}\n\
             \x20   workgroupBarrier();\n\
             \x20 }}\n\
             \x20 let sum_q = ws[0];\n",
            wg / 2
        );
    }
    s.push_str(
        "  let mean_c = base + mean_delta;\n\
         \x20 let var_raw = sum_q / m_f;\n\
         \x20 let var_b = select(var_raw, 0.0, var_raw < 0.0);\n\
         \x20 let inv_c = 1.0 / sqrt(var_b + pf(7u));\n\
         \x20 if (lid == 0u) {\n\
         \x20   MEAN[c] = mean_c; INVSTD[c] = inv_c;\n\
         \x20   if (RUNNING) { RM[c] = RM[c] * pf(8u) + pf(9u) * mean_c; RV[c] = RV[c] * pf(8u) + pf(10u) * var_b; }\n\
         \x20 }\n\
         \x20 let w_c = W[c]; let b_c = B[c];\n\
         \x20 let y_ns = channels * hw;\n\
         \x20 let y_step = step_n * y_ns + step_i;\n\
         \x20 let y_gap = y_ns - hw;\n\
         \x20 wi = lid - n0 * hw;\n\
         \x20 var y_idx = n0 * y_ns + ch_off + wi;\n",
    );
    for r in 0..regs {
        let off = r * wg;
        if vec4 {
            let xh = if half {
                format!("(V4(v{r}) - mean_c) * inv_c")
            } else {
                format!("(v{r} - mean_delta) * inv_c")
            };
            let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ Y[y_idx] = OUT(bn_act({xh} * w_c + b_c)); }}");
        } else {
            let xh = if half {
                format!("(V4(v{r}).x - mean_c) * inv_c")
            } else {
                format!("(v{r} - mean_delta) * inv_c")
            };
            let _ = writeln!(
                s,
                "  if (lid + {off}u < m_u) {{ Y[y_idx] = OUT(vec4<f32>(bn_act_f32({xh} * w_c + b_c))); }}"
            );
        }
        if r + 1 < regs {
            let _ = writeln!(
                s,
                "  wi += step_i; y_idx += y_step; let wy{r} = wi >= hw; wi = select(wi, wi - hw, wy{r}); y_idx += select(0u, y_gap, wy{r});"
            );
        }
    }
    s.push_str("}\n");
    s
}

/// Single-dispatch train-mode BatchNorm backward: register-cached (`REGS = 4 | 8`) or streaming two-pass
/// (`REGS = 0`, 2× unrolled under `VEC4`), with `WG = 256 | 512` and optional `STRIDED` views for `X` and `G`.
/// Grid: `(C, 1, 1)`. Params: 0 = C, 1 = hw_u, 2 = m_u, 3 = 1 / (N·hw), 5 = WG / hw_u, 6 = WG % hw_u,
/// 7 = x_off_u, 8 = x_n_stride_u, 9 = g_off_u, 10 = g_n_stride_u.
fn gen_fused_bwd(spec: &Spec) -> String {
    let raw_regs = spec.get_u32("REGS", 4);
    let wg = if spec.get_u32("WG", 256) == 512 { 512u32 } else { 256u32 };
    let n_sg = wg / 32;
    let sg_last = n_sg - 1;
    let streaming = raw_regs == 0;
    let regs = if wg == 512 { raw_regs.clamp(1, 4) } else { raw_regs.clamp(1, 8) };
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let strided = spec.get_bool("STRIDED");
    let reg_ty = if vec4 { "vec4<f32>" } else { "f32" };
    let smem_len = if sg { n_sg } else { wg };
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    let _ =
        writeln!(s, "var<workgroup> s1: array<f32, {smem_len}>;\nvar<workgroup> s2: array<f32, {smem_len}>;");
    let _ = write!(
        s,
        "@group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> G: array<E>;\n\
         @group(0) @binding(3) var<storage, read> MEAN: array<f32>;\n\
         @group(0) @binding(4) var<storage, read> INVSTD: array<f32>;\n\
         @group(0) @binding(5) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(6) var<storage, read> B: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> DX: array<E>;\n\
         @group(0) @binding(8) var<storage, read_write> DW: array<f32>;\n\
         @group(0) @binding(9) var<storage, read_write> DB: array<f32>;\n\
         @compute @workgroup_size({wg})\n"
    );
    if sg {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n",
        );
    } else {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {\n",
        );
    }
    s.push_str(
        "  let c = wid.x;\n\
         \x20 let channels = pu(0u);\n\
         \x20 let hw = pu(1u);\n\
         \x20 let m_u = pu(2u);\n\
         \x20 let m = MEAN[c];\n\
         \x20 let inv = INVSTD[c];\n\
         \x20 let wc = W[c];\n\
         \x20 let bc = B[c];\n\
         \x20 let ch_off = c * hw;\n\
         \x20 let step_n = pu(5u);\n\
         \x20 let step_i = pu(6u);\n\
         \x20 let dx_ns = channels * hw;\n\
         \x20 let dx_step = step_n * dx_ns + step_i;\n\
         \x20 let dx_gap = dx_ns - hw;\n\
         \x20 var a1 = vec4<f32>(0.0);\n\
         \x20 var a2 = vec4<f32>(0.0);\n\
         \x20 var e = lid;\n\
         \x20 let n0 = e / hw;\n\
         \x20 var wi = e - n0 * hw;\n",
    );
    if strided {
        s.push_str(
            "  let x_off = pu(7u); let x_ns = pu(8u); let x_step = step_n * x_ns + step_i; let x_gap = x_ns - hw;\n\
             \x20 let g_off = pu(9u); let g_ns = pu(10u); let g_step = step_n * g_ns + step_i; let g_gap = g_ns - hw;\n\
             \x20 var x_idx = x_off + n0 * x_ns + ch_off + wi;\n\
             \x20 var g_idx = g_off + n0 * g_ns + ch_off + wi;\n",
        );
    } else {
        s.push_str("  var x_idx = n0 * dx_ns + ch_off + wi;\n");
    }
    let g_ref = if strided { "g_idx" } else { "x_idx" };
    let adv_p1 = if strided {
        format!(
            "e += {wg}u; wi += step_i; x_idx += x_step; g_idx += g_step; if (wi >= hw) {{ wi -= hw; x_idx += x_gap; g_idx += g_gap; }}"
        )
    } else {
        format!("e += {wg}u; wi += step_i; x_idx += dx_step; if (wi >= hw) {{ wi -= hw; x_idx += dx_gap; }}")
    };
    if streaming {
        if vec4 {
            let g0 = if strided { "gi0" } else { "xi0" };
            let g1 = if strided { "gi1" } else { "xi1" };
            let save_g0 = if strided { " let gi0 = g_idx;" } else { "" };
            let save_g1 = if strided { " let gi1 = g_idx;" } else { "" };
            let _ = write!(
                s,
                "  loop {{\n\
                 \x20   if (e + {wg}u >= m_u) {{ break; }}\n\
                 \x20   let xi0 = x_idx;{save_g0}\n\
                 \x20   {adv_p1}\n\
                 \x20   let xi1 = x_idx;{save_g1}\n\
                 \x20   {adv_p1}\n\
                 \x20   let xv0 = V4(X[xi0]); let gv0 = V4(G[{g0}]);\n\
                 \x20   let xv1 = V4(X[xi1]); let gv1 = V4(G[{g1}]);\n\
                 \x20   let xh0 = (xv0 - vec4<f32>(m)) * inv;\n\
                 \x20   var dz0 = gv0;\n\
                 \x20   if (ACT != 0u) {{ dz0 *= bn_act_grad(xh0 * wc + vec4<f32>(bc)); }}\n\
                 \x20   let xh1 = (xv1 - vec4<f32>(m)) * inv;\n\
                 \x20   var dz1 = gv1;\n\
                 \x20   if (ACT != 0u) {{ dz1 *= bn_act_grad(xh1 * wc + vec4<f32>(bc)); }}\n\
                 \x20   a1 += dz0 + dz1;\n\
                 \x20   a2 += dz0 * xh0 + dz1 * xh1;\n\
                 \x20 }}\n\
                 \x20 if (e < m_u) {{\n\
                 \x20   let xh = (V4(X[x_idx]) - vec4<f32>(m)) * inv;\n\
                 \x20   var dz = V4(G[{g_ref}]);\n\
                 \x20   if (ACT != 0u) {{ dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }}\n\
                 \x20   a1 += dz; a2 += dz * xh;\n\
                 \x20 }}\n"
            );
        } else {
            let _ = write!(
                s,
                "  loop {{\n\
                 \x20   if (e >= m_u) {{ break; }}\n\
                 \x20   let xh = (V4(X[x_idx]).x - m) * inv;\n\
                 \x20   var dz = V4(G[{g_ref}]).x;\n\
                 \x20   if (ACT != 0u) {{ dz *= bn_act_grad_f32(xh * wc + bc); }}\n\
                 \x20   a1.x += dz; a2.x += dz * xh;\n\
                 \x20   {adv_p1}\n\
                 \x20 }}\n"
            );
        }
    } else {
        for r in 0..regs {
            let _ = writeln!(s, "  var xh{r}: {reg_ty}; var dz{r}: {reg_ty};");
        }
        for r in 0..regs {
            let off = r * wg;
            if vec4 {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{\n\
                     \x20   let xh = (V4(X[x_idx]) - vec4<f32>(m)) * inv;\n\
                     \x20   var dz = V4(G[{g_ref}]);\n\
                     \x20   if (ACT != 0u) {{ dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }}\n\
                     \x20   xh{r} = xh; dz{r} = dz;\n\
                     \x20   a1 += dz; a2 += dz * xh;\n\
                     \x20 }}"
                );
            } else {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{\n\
                     \x20   let xh = (V4(X[x_idx]).x - m) * inv;\n\
                     \x20   var dz = V4(G[{g_ref}]).x;\n\
                     \x20   if (ACT != 0u) {{ dz *= bn_act_grad_f32(xh * wc + bc); }}\n\
                     \x20   xh{r} = xh; dz{r} = dz;\n\
                     \x20   a1.x += dz; a2.x += dz * xh;\n\
                     \x20 }}"
                );
            }
            if r + 1 < regs {
                if strided {
                    let _ = writeln!(
                        s,
                        "  wi += step_i; x_idx += x_step; g_idx += g_step; let wx{r} = wi >= hw; wi = select(wi, wi - hw, wx{r}); x_idx += select(0u, x_gap, wx{r}); g_idx += select(0u, g_gap, wx{r});"
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "  wi += step_i; x_idx += dx_step; let wx{r} = wi >= hw; wi = select(wi, wi - hw, wx{r}); x_idx += select(0u, dx_gap, wx{r});"
                    );
                }
            }
        }
    }
    let (v1, v2) = if vec4 {
        ("(a1.x + a1.y) + (a1.z + a1.w)", "(a2.x + a2.y) + (a2.z + a2.w)")
    } else {
        ("a1.x", "a2.x")
    };
    if sg {
        let _ = write!(
            s,
            "  let r1 = subgroupAdd({v1});\n\
             \x20 let r2 = subgroupAdd({v2});\n\
             \x20 let sg = min(sgid, {sg_last}u);\n\
             \x20 if (lane == 0u) {{ s1[sg] = r1; s2[sg] = r2; }}\n\
             \x20 workgroupBarrier();\n\
             \x20 let db = subgroupAdd(select(0.0, s1[min(lane, {sg_last}u)], lane < {n_sg}u));\n\
             \x20 let dw = subgroupAdd(select(0.0, s2[min(lane, {sg_last}u)], lane < {n_sg}u));\n\
             \x20 if (lid == 0u) {{ DB[c] = db; DW[c] = dw; }}\n\
             \x20 let db_n = db * pf(3u);\n\
             \x20 let dw_n = dw * pf(3u);\n"
        );
    } else {
        let _ = write!(
            s,
            "  s1[lid] = {v1}; s2[lid] = {v2};\n\
             \x20 workgroupBarrier();\n\
             \x20 for (var step = {}u; step > 0u; step >>= 1u) {{\n\
             \x20   if (lid < step) {{ s1[lid] += s1[lid + step]; s2[lid] += s2[lid + step]; }}\n\
             \x20   workgroupBarrier();\n\
             \x20 }}\n\
             \x20 if (lid == 0u) {{\n\
             \x20   let db = s1[0]; let dw = s2[0];\n\
             \x20   DB[c] = db; DW[c] = dw;\n\
             \x20   s1[0] = db * pf(3u); s2[0] = dw * pf(3u);\n\
             \x20 }}\n\
             \x20 workgroupBarrier();\n\
             \x20 let db_n = s1[0];\n\
             \x20 let dw_n = s2[0];\n",
            wg / 2
        );
    }
    s.push_str(
        "  let scale = wc * inv;\n\
         \x20 e = lid;\n\
         \x20 wi = e - n0 * hw;\n",
    );
    if streaming {
        if strided {
            s.push_str(
                "  x_idx = x_off + n0 * x_ns + ch_off + wi;\n\
                 \x20 g_idx = g_off + n0 * g_ns + ch_off + wi;\n\
                 \x20 var dx_idx = n0 * dx_ns + ch_off + wi;\n",
            );
        } else {
            s.push_str("  x_idx = n0 * dx_ns + ch_off + wi;\n");
        }
        let dx_ref = if strided { "dx_idx" } else { "x_idx" };
        let adv_p2 = if strided {
            format!(
                "e += {wg}u; wi += step_i; x_idx += x_step; g_idx += g_step; dx_idx += dx_step; \
                 if (wi >= hw) {{ wi -= hw; x_idx += x_gap; g_idx += g_gap; dx_idx += dx_gap; }}"
            )
        } else {
            format!(
                "e += {wg}u; wi += step_i; x_idx += dx_step; if (wi >= hw) {{ wi -= hw; x_idx += dx_gap; }}"
            )
        };
        if vec4 {
            let g0 = if strided { "gi0" } else { "xi0" };
            let g1 = if strided { "gi1" } else { "xi1" };
            let d0 = if strided { "di0" } else { "xi0" };
            let d1 = if strided { "di1" } else { "xi1" };
            let save0 = if strided { " let gi0 = g_idx; let di0 = dx_idx;" } else { "" };
            let save1 = if strided { " let gi1 = g_idx; let di1 = dx_idx;" } else { "" };
            let _ = write!(
                s,
                "  loop {{\n\
                 \x20   if (e + {wg}u >= m_u) {{ break; }}\n\
                 \x20   let xi0 = x_idx;{save0}\n\
                 \x20   {adv_p2}\n\
                 \x20   let xi1 = x_idx;{save1}\n\
                 \x20   {adv_p2}\n\
                 \x20   let xv0 = V4(X[xi0]); let gv0 = V4(G[{g0}]);\n\
                 \x20   let xv1 = V4(X[xi1]); let gv1 = V4(G[{g1}]);\n\
                 \x20   let xh0 = (xv0 - vec4<f32>(m)) * inv;\n\
                 \x20   var dz0 = gv0;\n\
                 \x20   if (ACT != 0u) {{ dz0 *= bn_act_grad(xh0 * wc + vec4<f32>(bc)); }}\n\
                 \x20   let xh1 = (xv1 - vec4<f32>(m)) * inv;\n\
                 \x20   var dz1 = gv1;\n\
                 \x20   if (ACT != 0u) {{ dz1 *= bn_act_grad(xh1 * wc + vec4<f32>(bc)); }}\n\
                 \x20   DX[{d0}] = OUT(scale * (dz0 - db_n - xh0 * dw_n));\n\
                 \x20   DX[{d1}] = OUT(scale * (dz1 - db_n - xh1 * dw_n));\n\
                 \x20 }}\n\
                 \x20 if (e < m_u) {{\n\
                 \x20   let xh = (V4(X[x_idx]) - vec4<f32>(m)) * inv;\n\
                 \x20   var dz = V4(G[{g_ref}]);\n\
                 \x20   if (ACT != 0u) {{ dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }}\n\
                 \x20   DX[{dx_ref}] = OUT(scale * (dz - db_n - xh * dw_n));\n\
                 \x20 }}\n"
            );
        } else {
            let _ = write!(
                s,
                "  loop {{\n\
                 \x20   if (e >= m_u) {{ break; }}\n\
                 \x20   let xh = (V4(X[x_idx]).x - m) * inv;\n\
                 \x20   var dz = V4(G[{g_ref}]).x;\n\
                 \x20   if (ACT != 0u) {{ dz *= bn_act_grad_f32(xh * wc + bc); }}\n\
                 \x20   DX[{dx_ref}] = OUT(vec4<f32>(scale * (dz - db_n - xh * dw_n)));\n\
                 \x20   {adv_p2}\n\
                 \x20 }}\n"
            );
        }
    } else {
        s.push_str("  var dx_idx = n0 * dx_ns + ch_off + wi;\n");
        for r in 0..regs {
            let off = r * wg;
            if vec4 {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{ DX[dx_idx] = OUT(scale * (dz{r} - db_n - xh{r} * dw_n)); }}"
                );
            } else {
                let _ = writeln!(
                    s,
                    "  if (lid + {off}u < m_u) {{ DX[dx_idx] = OUT(vec4<f32>(scale * (dz{r} - db_n - xh{r} * dw_n))); }}"
                );
            }
            if r + 1 < regs {
                let _ = writeln!(
                    s,
                    "  wi += step_i; dx_idx += dx_step; let wd{r} = wi >= hw; wi = select(wi, wi - hw, wd{r}); dx_idx += select(0u, dx_gap, wd{r});"
                );
            }
        }
    }
    s.push_str("}\n");
    s
}

use Access::{Read as R, ReadWrite as RW};

pub(super) static FUSED_FWD: KernelDef = KernelDef {
    name: "bn_fused_fwd",
    storage: &[R, R, R, RW, RW, RW, RW, RW],
    source: Source::Generated(gen_fused_fwd),
};
pub(super) static FUSED_BWD: KernelDef = KernelDef {
    name: "bn_fused_bwd",
    storage: &[R, R, R, R, R, R, RW, RW, RW],
    source: Source::Generated(gen_fused_bwd),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bn_fused_portable_emits_no_subgroup_tokens() {
        for (def, spec) in [
            (
                &FUSED_FWD,
                Spec::new()
                    .u32("ACT", 1)
                    .bool("VEC4", true)
                    .bool("RUNNING", true)
                    .u32("REGS", 4)
                    .u32("WG", 256)
                    .subgroups(false),
            ),
            (
                &FUSED_FWD,
                Spec::new()
                    .u32("ACT", 1)
                    .bool("VEC4", true)
                    .bool("RUNNING", true)
                    .u32("REGS", 4)
                    .u32("WG", 512)
                    .bool("STRIDED", true)
                    .subgroups(false),
            ),
            (
                &FUSED_FWD,
                Spec::new()
                    .u32("ACT", 1)
                    .bool("VEC4", true)
                    .bool("RUNNING", true)
                    .u32("REGS", 0)
                    .u32("WG", 512)
                    .subgroups(false),
            ),
            (
                &FUSED_FWD,
                Spec::new()
                    .u32("ACT", 0)
                    .bool("VEC4", false)
                    .bool("RUNNING", false)
                    .u32("REGS", 0)
                    .u32("WG", 256)
                    .subgroups(false),
            ),
            (
                &FUSED_BWD,
                Spec::new().u32("ACT", 1).bool("VEC4", true).u32("REGS", 8).u32("WG", 256).subgroups(false),
            ),
            (
                &FUSED_BWD,
                Spec::new()
                    .u32("ACT", 1)
                    .bool("VEC4", true)
                    .u32("REGS", 4)
                    .u32("WG", 512)
                    .bool("STRIDED", true)
                    .subgroups(false),
            ),
            (
                &FUSED_BWD,
                Spec::new()
                    .u32("ACT", 1)
                    .bool("VEC4", true)
                    .u32("REGS", 0)
                    .u32("WG", 512)
                    .bool("STRIDED", true)
                    .subgroups(false),
            ),
            (
                &FUSED_BWD,
                Spec::new().u32("ACT", 2).bool("VEC4", false).u32("REGS", 0).u32("WG", 256).subgroups(false),
            ),
        ] {
            assert!(fused_fits(&spec), "expected fused_fits for {} {:?}", def.name, spec);
            let wgsl = crate::pipeline::render(def, &spec);
            assert!(!wgsl.contains("subgroup"), "portable {} emitted subgroup token:\n{wgsl}", def.name);
        }
        assert!(!fused_fits(&Spec::new().u32("REGS", 1)));
        assert!(!fused_fits(&Spec::new().u32("REGS", 16)));
        assert!(!fused_fits(&Spec::new().u32("WG", 128).u32("REGS", 4)));
        assert!(!fused_fits(&Spec::new().u32("WG", 512).u32("REGS", 8)));
    }
}
