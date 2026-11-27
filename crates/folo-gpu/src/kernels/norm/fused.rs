//! Single-dispatch train-mode BatchNorm forward and backward kernels:
//! - Register-cached (`REGS = 4 | 8`, up to 2,048 units = 8,192 elements/ch under `VEC4`): each channel's
//!   256-thread workgroup reads `X` (and `G` in backward) once into explicit register slots, reduces statistics /
//!   parameter gradients across the workgroup (32-wide subgroups or portable shared-memory tree fallback), and
//!   writes `Y` / `DX` directly from registers without intermediate `PART` / `COEF` buffers.
//! - Streaming two-pass (`REGS = 0`, above 2,048 up to [`FUSED_STREAM_MAX_UNITS`] = 6,400 units = 25,600
//!   elements/ch under `VEC4` when `C >= 64`): keeps L2-resident medium layers (e.g. `16×40×40` at 64–256 ch)
//!   in a single dispatch with zero intermediate `PART` / `COEF` allocations and zero finalize dispatches, running
//!   vectorized Welford/Chan in Pass 1 and normalizing/activating from L2-warm `X` (and `G`) in Pass 2.

use std::fmt::Write as _;

use super::{COMMON, WELFORD, prelude};
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// Minimum storage buffers needed by [`FUSED_BWD`] (9: X, G, MEAN, INVSTD, W, B, DX, DW, DB).
pub(super) const FUSED_BWD_BUFFERS: u32 = 9;
/// Max units (`N · hw_units`) handled by the register-cached fused BatchNorm kernels
/// (`REGS = 4` up to 1,024 units, `REGS = 8` up to 2,048 units = 8,192 elements per channel under `VEC4`).
pub(super) const FUSED_MAX_UNITS: usize = 256 * 8;
/// Max units (`N · hw_units`) handled by the streaming two-pass single-dispatch fused BatchNorm kernels
/// (`REGS = 0`, up to 6,400 units = 25,600 elements per channel under `VEC4`, covering `16×40×40` and
/// `8×80×80` when `C >= 64` so all SMs stay busy and `X`/`G` remain L2-hot between passes).
pub(super) const FUSED_STREAM_MAX_UNITS: usize = 256 * 25;
/// Minimum channels required for streaming single-dispatch BatchNorm (`REGS = 0`) so one workgroup per channel
/// maintains high SM occupancy.
pub(super) const FUSED_STREAM_MIN_CHANNELS: usize = 64;

/// Bounds check for precompiling `"bn_fused_fwd"` / `"bn_fused_bwd"` from a kernel list (`kernels::bounds`).
pub(crate) fn fused_fits(spec: &Spec) -> bool {
    matches!(spec.get_u32("REGS", 4), 0 | 4 | 8)
}

/// Streaming two-pass single-dispatch train-mode BatchNorm forward (`REGS = 0`): Pass 1 accumulates 4-lane
/// vectorized Welford `(count, mean, M2)` per thread and merges across the workgroup via Chan's pairwise formula
/// (1-barrier 32-wide `subgroupShuffleDown` or portable `bn_tree_merge`), writes `MEAN`, `INVSTD`, and optional
/// running stats, then Pass 2 reads L2-warm `X` and writes `Y`.
fn gen_fused_fwd_streaming(spec: &Spec) -> String {
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    if sg {
        s.push_str(
            "var<workgroup> wn: array<f32, 8>;\n\
             var<workgroup> wm: array<f32, 8>;\n\
             var<workgroup> wq: array<f32, 8>;\n\
             var<workgroup> wmean: f32;\n\
             var<workgroup> winv: f32;\n",
        );
    } else {
        s.push_str(WELFORD);
        s.push_str("var<workgroup> wmean: f32;\nvar<workgroup> winv: f32;\n");
    }
    s.push_str(
        "@group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> B: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<E>;\n\
         @group(0) @binding(5) var<storage, read_write> MEAN: array<f32>;\n\
         @group(0) @binding(6) var<storage, read_write> INVSTD: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> RM: array<f32>;\n\
         @group(0) @binding(8) var<storage, read_write> RV: array<f32>;\n\
         @compute @workgroup_size(256)\n",
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
         \x20 var k = 0.0;\n\
         \x20 var m4 = vec4<f32>(0.0);\n\
         \x20 var q4 = vec4<f32>(0.0);\n\
         \x20 var w = bn_walk_start(0u, lid, hw);\n\
         \x20 loop {\n\
         \x20   if (w.e >= m_u) { break; }\n\
         \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
         \x20   let x = V4(X[idx]);\n\
         \x20   k += 1.0;\n\
         \x20   let inv_k = 1.0 / k;\n",
    );
    if vec4 {
        s.push_str(
            "    let d = x - m4;\n\
             \x20   m4 += d * inv_k;\n\
             \x20   q4 += d * (x - m4);\n\
             \x20   bn_walk_next(&w, hw);\n\
             \x20 }\n\
             \x20 var na = 0.0;\n\
             \x20 var ma = 0.0;\n\
             \x20 var qa = 0.0;\n\
             \x20 if (k > 0.0) {\n\
             \x20   let d01 = m4.y - m4.x;\n\
             \x20   let m01 = m4.x + 0.5 * d01;\n\
             \x20   let q01 = q4.x + q4.y + (0.5 * k * d01) * d01;\n\
             \x20   let d23 = m4.w - m4.z;\n\
             \x20   let m23 = m4.z + 0.5 * d23;\n\
             \x20   let q23 = q4.z + q4.w + (0.5 * k * d23) * d23;\n\
             \x20   let d_all = m23 - m01;\n\
             \x20   na = 4.0 * k;\n\
             \x20   ma = m01 + 0.5 * d_all;\n\
             \x20   qa = q01 + q23 + (k * d_all) * d_all;\n\
             \x20 }\n",
        );
    } else {
        s.push_str(
            "    let d = x.x - m4.x;\n\
             \x20   m4.x += d * inv_k;\n\
             \x20   q4.x += d * (x.x - m4.x);\n\
             \x20   bn_walk_next(&w, hw);\n\
             \x20 }\n\
             \x20 var na = k;\n\
             \x20 var ma = m4.x;\n\
             \x20 var qa = q4.x;\n",
        );
    }
    if sg {
        s.push_str(
            "  for (var off = 16u; off > 0u; off = off >> 1u) {\n\
             \x20   let nb = subgroupShuffleDown(na, off);\n\
             \x20   let mb = subgroupShuffleDown(ma, off);\n\
             \x20   let qb = subgroupShuffleDown(qa, off);\n\
             \x20   if (lane < off) {\n\
             \x20     if (na == 0.0) {\n\
             \x20       na = nb; ma = mb; qa = qb;\n\
             \x20     } else if (nb > 0.0) {\n\
             \x20       let n = na + nb;\n\
             \x20       let d = mb - ma;\n\
             \x20       let delta = d * (nb / n);\n\
             \x20       ma += delta;\n\
             \x20       qa += qb + d * (delta * na);\n\
             \x20       na = n;\n\
             \x20     }\n\
             \x20   }\n\
             \x20 }\n\
             \x20 let sg = min(sgid, 7u);\n\
             \x20 if (lane == 0u) { wn[sg] = na; wm[sg] = ma; wq[sg] = qa; }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (lid == 0u) {\n\
             \x20   var fn_a = wn[0]; var fm_a = wm[0]; var fq_a = wq[0];\n\
             \x20   for (var i = 1u; i < 8u; i += 1u) {\n\
             \x20     let nb = wn[i];\n\
             \x20     if (fn_a == 0.0) {\n\
             \x20       fn_a = nb; fm_a = wm[i]; fq_a = wq[i];\n\
             \x20     } else if (nb > 0.0) {\n\
             \x20       let n = fn_a + nb;\n\
             \x20       let d = wm[i] - fm_a;\n\
             \x20       let delta = d * (nb / n);\n\
             \x20       fm_a += delta;\n\
             \x20       fq_a += wq[i] + d * (delta * fn_a);\n\
             \x20       fn_a = n;\n\
             \x20     }\n\
             \x20   }\n\
             \x20   let var_raw = fq_a / fn_a;\n\
             \x20   let var_b = select(var_raw, 0.0, var_raw < 0.0);\n\
             \x20   let inv_c = 1.0 / sqrt(var_b + pf(7u));\n\
             \x20   MEAN[c] = fm_a;\n\
             \x20   INVSTD[c] = inv_c;\n\
             \x20   if (RUNNING) {\n\
             \x20     RM[c] = RM[c] * pf(8u) + pf(9u) * fm_a;\n\
             \x20     RV[c] = RV[c] * pf(8u) + pf(10u) * var_b;\n\
             \x20   }\n\
             \x20   wmean = fm_a;\n\
             \x20   winv = inv_c;\n\
             \x20 }\n\
             \x20 workgroupBarrier();\n",
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
             \x20   MEAN[c] = fm_a;\n\
             \x20   INVSTD[c] = inv_c;\n\
             \x20   if (RUNNING) {\n\
             \x20     RM[c] = RM[c] * pf(8u) + pf(9u) * fm_a;\n\
             \x20     RV[c] = RV[c] * pf(8u) + pf(10u) * var_b;\n\
             \x20   }\n\
             \x20   wmean = fm_a;\n\
             \x20   winv = inv_c;\n\
             \x20 }\n\
             \x20 workgroupBarrier();\n",
        );
    }
    s.push_str(
        "  let mean_c = wmean;\n\
         \x20 let inv_c = winv;\n\
         \x20 let w_c = W[c];\n\
         \x20 let b_c = B[c];\n\
         \x20 w = bn_walk_start(0u, lid, hw);\n\
         \x20 loop {\n\
         \x20   if (w.e >= m_u) { break; }\n\
         \x20   let idx = (w.n * channels + c) * hw + w.i;\n",
    );
    if vec4 {
        s.push_str(
            "    let xh = (V4(X[idx]) - mean_c) * inv_c;\n\
             \x20   Y[idx] = OUT(bn_act(xh * w_c + b_c));\n",
        );
    } else {
        s.push_str(
            "    let xh = (V4(X[idx]).x - mean_c) * inv_c;\n\
             \x20   Y[idx] = OUT(vec4<f32>(bn_act_f32(xh * w_c + b_c)));\n",
        );
    }
    s.push_str("    bn_walk_next(&w, hw);\n  }\n}\n");
    s
}

/// Single-dispatch train-mode BatchNorm forward: register-cached (`REGS = 4 | 8`) or streaming two-pass
/// (`REGS = 0`).
/// Grid: `(C, 1, 1)`. Params: 0 = C, 1 = hw_units, 2 = m_units, 5 = 256 / hw_units, 6 = 256 % hw_units,
/// 7 = eps, 8 = 1 - momentum, 9 = momentum, 10 = momentum · n / (n - 1).
fn gen_fused_fwd(spec: &Spec) -> String {
    let raw_regs = spec.get_u32("REGS", 4);
    if raw_regs == 0 {
        return gen_fused_fwd_streaming(spec);
    }
    let regs = raw_regs.clamp(1, 8);
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let half = spec.is_f16();
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    if sg {
        s.push_str(
            "var<workgroup> wbase: f32;\n\
             var<workgroup> ws: array<f32, 8>;\n\
             var<workgroup> wq: array<f32, 8>;\n",
        );
    } else {
        s.push_str(
            "var<workgroup> wbase: f32;\n\
             var<workgroup> wmd: f32;\n\
             var<workgroup> ws: array<f32, 256>;\n",
        );
    }
    s.push_str(
        "@group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> B: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<E>;\n\
         @group(0) @binding(5) var<storage, read_write> MEAN: array<f32>;\n\
         @group(0) @binding(6) var<storage, read_write> INVSTD: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> RM: array<f32>;\n\
         @group(0) @binding(8) var<storage, read_write> RV: array<f32>;\n\
         @compute @workgroup_size(256)\n",
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
    let _ = write!(
        s,
        "  let c = wid.x;\n\
         \x20 let channels = pu(0u);\n\
         \x20 let hw = pu(1u);\n\
         \x20 let m_u = pu(2u);\n\
         \x20 let m_f = {mf_expr};\n\
         \x20 var w = bn_walk_start(0u, lid, hw);\n"
    );
    for r in 0..regs {
        let _ = writeln!(s, "  var v{r}: E;");
    }
    for r in 0..regs {
        let _ = writeln!(s, "  if (w.e < m_u) {{ v{r} = X[(w.n * channels + c) * hw + w.i]; }}");
        if r + 1 < regs {
            s.push_str("  bn_walk_next(&w, hw);\n");
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
    // For F32, subtract `base` in place into `v{r}` before the reduction barrier so Pass 2 sees the already-
    // shifted values (zero extra registers, avoids driver fast-math reassociation of `(x - base) - mean_delta`
    // into `x - (base + mean_delta)`). F16 inputs fit in 11 mantissa bits (<= 65504), so `f32` already has
    // 13 guard bits and we keep `v{r}` packed in `vec4<f16>`/`f16` registers.
    if vec4 {
        s.push_str("  var s4 = vec4<f32>(0.0);\n");
        for r in 0..regs {
            let off = r * 256;
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
            let off = r * 256;
            if half {
                let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ s_t += V4(v{r}).x - base; }}");
            } else {
                let _ = writeln!(s, "  if (lid + {off}u < m_u) {{ v{r} = v{r} - base; s_t += v{r}; }}");
            }
        }
    }
    if sg {
        s.push_str(
            "  let rs = subgroupAdd(s_t);\n\
             \x20 let sg = min(sgid, 7u);\n\
             \x20 if (lane == 0u) { ws[sg] = rs; }\n\
             \x20 workgroupBarrier();\n\
             \x20 let sum_s = ((ws[0] + ws[1]) + (ws[2] + ws[3])) + ((ws[4] + ws[5]) + (ws[6] + ws[7]));\n\
             \x20 let mean_delta = sum_s / m_f;\n",
        );
    } else {
        s.push_str(
            "  ws[lid] = s_t;\n\
             \x20 workgroupBarrier();\n\
             \x20 for (var step = 128u; step > 0u; step >>= 1u) {\n\
             \x20   if (lid < step) { ws[lid] += ws[lid + step]; }\n\
             \x20   workgroupBarrier();\n\
             \x20 }\n\
             \x20 if (lid == 0u) { wmd = ws[0] / m_f; }\n\
             \x20 workgroupBarrier();\n\
             \x20 let mean_delta = wmd;\n",
        );
    }
    if vec4 {
        s.push_str("  var q4 = vec4<f32>(0.0);\n");
        for r in 0..regs {
            let off = r * 256;
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
            let off = r * 256;
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
        s.push_str(
            "  let rq = subgroupAdd(q_t);\n\
             \x20 if (lane == 0u) { wq[sg] = rq; }\n\
             \x20 workgroupBarrier();\n\
             \x20 let sum_q = ((wq[0] + wq[1]) + (wq[2] + wq[3])) + ((wq[4] + wq[5]) + (wq[6] + wq[7]));\n",
        );
    } else {
        s.push_str(
            "  ws[lid] = q_t;\n\
             \x20 workgroupBarrier();\n\
             \x20 for (var step = 128u; step > 0u; step >>= 1u) {\n\
             \x20   if (lid < step) { ws[lid] += ws[lid + step]; }\n\
             \x20   workgroupBarrier();\n\
             \x20 }\n\
             \x20 let sum_q = ws[0];\n",
        );
    }
    s.push_str(
        "  let mean_c = base + mean_delta;\n\
         \x20 let var_raw = sum_q / m_f;\n\
         \x20 let var_b = select(var_raw, 0.0, var_raw < 0.0);\n\
         \x20 let inv_c = 1.0 / sqrt(var_b + pf(7u));\n\
         \x20 if (lid == 0u) {\n\
         \x20   MEAN[c] = mean_c;\n\
         \x20   INVSTD[c] = inv_c;\n\
         \x20   if (RUNNING) {\n\
         \x20     RM[c] = RM[c] * pf(8u) + pf(9u) * mean_c;\n\
         \x20     RV[c] = RV[c] * pf(8u) + pf(10u) * var_b;\n\
         \x20   }\n\
         \x20 }\n\
         \x20 let w_c = W[c];\n\
         \x20 let b_c = B[c];\n\
         \x20 w = bn_walk_start(0u, lid, hw);\n",
    );
    for r in 0..regs {
        if vec4 {
            let xh = if half {
                format!("(V4(v{r}) - mean_c) * inv_c")
            } else {
                format!("(v{r} - mean_delta) * inv_c")
            };
            let _ = writeln!(
                s,
                "  if (w.e < m_u) {{\n\
                 \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                 \x20   let xh = {xh};\n\
                 \x20   Y[idx] = OUT(bn_act(xh * w_c + b_c));\n\
                 \x20 }}"
            );
        } else {
            let xh = if half {
                format!("(V4(v{r}).x - mean_c) * inv_c")
            } else {
                format!("(v{r} - mean_delta) * inv_c")
            };
            let _ = writeln!(
                s,
                "  if (w.e < m_u) {{\n\
                 \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                 \x20   let xh = {xh};\n\
                 \x20   Y[idx] = OUT(vec4<f32>(bn_act_f32(xh * w_c + b_c)));\n\
                 \x20 }}"
            );
        }
        if r + 1 < regs {
            s.push_str("  bn_walk_next(&w, hw);\n");
        }
    }
    s.push_str("}\n");
    s
}

/// Single-dispatch train-mode BatchNorm backward: register-cached (`REGS = 4 | 8`, loads `X` and `G` once and
/// reuses `xh` / `dz` from registers) or streaming two-pass (`REGS = 0`, reads L2-warm `X` and `G` in Pass 2).
/// Grid: `(C, 1, 1)`. Params: 0 = C, 1 = hw_units, 2 = m_units, 3 = 1 / (N·hw), 5 = 256 / hw_units,
/// 6 = 256 % hw_units.
fn gen_fused_bwd(spec: &Spec) -> String {
    let raw_regs = spec.get_u32("REGS", 4);
    let streaming = raw_regs == 0;
    let regs = raw_regs.clamp(1, 8);
    let sg = spec.is_subgroups();
    let vec4 = spec.get_bool("VEC4");
    let reg_ty = if vec4 { "vec4<f32>" } else { "f32" };
    let smem_len = if sg { 8 } else { 256 };
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    let _ =
        writeln!(s, "var<workgroup> s1: array<f32, {smem_len}>;\nvar<workgroup> s2: array<f32, {smem_len}>;");
    s.push_str(
        "@group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> G: array<E>;\n\
         @group(0) @binding(3) var<storage, read> MEAN: array<f32>;\n\
         @group(0) @binding(4) var<storage, read> INVSTD: array<f32>;\n\
         @group(0) @binding(5) var<storage, read> W: array<f32>;\n\
         @group(0) @binding(6) var<storage, read> B: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> DX: array<E>;\n\
         @group(0) @binding(8) var<storage, read_write> DW: array<f32>;\n\
         @group(0) @binding(9) var<storage, read_write> DB: array<f32>;\n\
         @compute @workgroup_size(256)\n",
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
         \x20 var a1 = vec4<f32>(0.0);\n\
         \x20 var a2 = vec4<f32>(0.0);\n\
         \x20 var w = bn_walk_start(0u, lid, hw);\n",
    );
    if streaming {
        if vec4 {
            s.push_str(
                "  loop {\n\
                 \x20   if (w.e >= m_u) { break; }\n\
                 \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                 \x20   let xh = (V4(X[idx]) - vec4<f32>(m)) * inv;\n\
                 \x20   var dz = V4(G[idx]);\n\
                 \x20   if (ACT != 0u) { dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }\n\
                 \x20   a1 += dz; a2 += dz * xh;\n\
                 \x20   bn_walk_next(&w, hw);\n\
                 \x20 }\n",
            );
        } else {
            s.push_str(
                "  loop {\n\
                 \x20   if (w.e >= m_u) { break; }\n\
                 \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                 \x20   let xh = (V4(X[idx]).x - m) * inv;\n\
                 \x20   var dz = V4(G[idx]).x;\n\
                 \x20   if (ACT != 0u) { dz *= bn_act_grad_f32(xh * wc + bc); }\n\
                 \x20   a1.x += dz; a2.x += dz * xh;\n\
                 \x20   bn_walk_next(&w, hw);\n\
                 \x20 }\n",
            );
        }
    } else {
        for r in 0..regs {
            let _ = writeln!(s, "  var xh{r}: {reg_ty}; var dz{r}: {reg_ty};");
        }
        for r in 0..regs {
            if vec4 {
                let _ = writeln!(
                    s,
                    "  if (w.e < m_u) {{\n\
                     \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                     \x20   let xh = (V4(X[idx]) - vec4<f32>(m)) * inv;\n\
                     \x20   var dz = V4(G[idx]);\n\
                     \x20   if (ACT != 0u) {{ dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }}\n\
                     \x20   xh{r} = xh; dz{r} = dz;\n\
                     \x20   a1 += dz; a2 += dz * xh;\n\
                     \x20 }}"
                );
            } else {
                let _ = writeln!(
                    s,
                    "  if (w.e < m_u) {{\n\
                     \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                     \x20   let xh = (V4(X[idx]).x - m) * inv;\n\
                     \x20   var dz = V4(G[idx]).x;\n\
                     \x20   if (ACT != 0u) {{ dz *= bn_act_grad_f32(xh * wc + bc); }}\n\
                     \x20   xh{r} = xh; dz{r} = dz;\n\
                     \x20   a1.x += dz; a2.x += dz * xh;\n\
                     \x20 }}"
                );
            }
            if r + 1 < regs {
                s.push_str("  bn_walk_next(&w, hw);\n");
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
             \x20 let sg = min(sgid, 7u);\n\
             \x20 if (lane == 0u) {{ s1[sg] = r1; s2[sg] = r2; }}\n\
             \x20 workgroupBarrier();\n\
             \x20 let db = ((s1[0] + s1[1]) + (s1[2] + s1[3])) + ((s1[4] + s1[5]) + (s1[6] + s1[7]));\n\
             \x20 let dw = ((s2[0] + s2[1]) + (s2[2] + s2[3])) + ((s2[4] + s2[5]) + (s2[6] + s2[7]));\n\
             \x20 if (lid == 0u) {{ DB[c] = db; DW[c] = dw; }}\n\
             \x20 let db_n = db * pf(3u);\n\
             \x20 let dw_n = dw * pf(3u);\n"
        );
    } else {
        let _ = write!(
            s,
            "  s1[lid] = {v1}; s2[lid] = {v2};\n\
             \x20 workgroupBarrier();\n\
             \x20 for (var step = 128u; step > 0u; step >>= 1u) {{\n\
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
             \x20 let dw_n = s2[0];\n"
        );
    }
    s.push_str(
        "  let scale = wc * inv;\n\
         \x20 w = bn_walk_start(0u, lid, hw);\n",
    );
    if streaming {
        if vec4 {
            s.push_str(
                "  loop {\n\
                 \x20   if (w.e >= m_u) { break; }\n\
                 \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                 \x20   let xh = (V4(X[idx]) - vec4<f32>(m)) * inv;\n\
                 \x20   var dz = V4(G[idx]);\n\
                 \x20   if (ACT != 0u) { dz *= bn_act_grad(xh * wc + vec4<f32>(bc)); }\n\
                 \x20   DX[idx] = OUT(scale * (dz - db_n - xh * dw_n));\n\
                 \x20   bn_walk_next(&w, hw);\n\
                 \x20 }\n",
            );
        } else {
            s.push_str(
                "  loop {\n\
                 \x20   if (w.e >= m_u) { break; }\n\
                 \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                 \x20   let xh = (V4(X[idx]).x - m) * inv;\n\
                 \x20   var dz = V4(G[idx]).x;\n\
                 \x20   if (ACT != 0u) { dz *= bn_act_grad_f32(xh * wc + bc); }\n\
                 \x20   DX[idx] = OUT(vec4<f32>(scale * (dz - db_n - xh * dw_n)));\n\
                 \x20   bn_walk_next(&w, hw);\n\
                 \x20 }\n",
            );
        }
    } else {
        for r in 0..regs {
            if vec4 {
                let _ = writeln!(
                    s,
                    "  if (w.e < m_u) {{\n\
                     \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                     \x20   DX[idx] = OUT(scale * (dz{r} - db_n - xh{r} * dw_n));\n\
                     \x20 }}"
                );
            } else {
                let _ = writeln!(
                    s,
                    "  if (w.e < m_u) {{\n\
                     \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
                     \x20   DX[idx] = OUT(vec4<f32>(scale * (dz{r} - db_n - xh{r} * dw_n)));\n\
                     \x20 }}"
                );
            }
            if r + 1 < regs {
                s.push_str("  bn_walk_next(&w, hw);\n");
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
                    .subgroups(false),
            ),
            (
                &FUSED_FWD,
                Spec::new()
                    .u32("ACT", 1)
                    .bool("VEC4", true)
                    .bool("RUNNING", true)
                    .u32("REGS", 0)
                    .subgroups(false),
            ),
            (
                &FUSED_FWD,
                Spec::new()
                    .u32("ACT", 0)
                    .bool("VEC4", false)
                    .bool("RUNNING", false)
                    .u32("REGS", 0)
                    .subgroups(false),
            ),
            (&FUSED_BWD, Spec::new().u32("ACT", 1).bool("VEC4", true).u32("REGS", 8).subgroups(false)),
            (&FUSED_BWD, Spec::new().u32("ACT", 1).bool("VEC4", true).u32("REGS", 0).subgroups(false)),
            (&FUSED_BWD, Spec::new().u32("ACT", 2).bool("VEC4", false).u32("REGS", 0).subgroups(false)),
        ] {
            assert!(fused_fits(&spec), "expected fused_fits for {} {:?}", def.name, spec);
            let wgsl = crate::pipeline::render(def, &spec);
            assert!(!wgsl.contains("subgroup"), "portable {} emitted subgroup token:\n{wgsl}", def.name);
        }
        assert!(!fused_fits(&Spec::new().u32("REGS", 1)));
        assert!(!fused_fits(&Spec::new().u32("REGS", 16)));
    }
}
