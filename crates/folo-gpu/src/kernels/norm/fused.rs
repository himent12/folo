//! Single-dispatch register-cached train-mode BatchNorm forward and backward kernels: each channel's
//! 256-thread workgroup reads `X` (and `G` in backward) once into explicit register slots (`REGS` = 4 or 8),
//! reduces statistics / parameter gradients across the workgroup (1-barrier 32-wide subgroup reduction or
//! portable shared-memory tree fallback), and writes `Y` / `DX` directly from registers without intermediate
//! `PART` / `K` buffers.

use std::fmt::Write as _;

use super::{COMMON, WELFORD, prelude};
use crate::pipeline::{Access, KernelDef, Source, Spec};

/// Minimum storage buffers needed by [`FUSED_BWD`] (9: X, G, MEAN, INVSTD, W, B, DX, DW, DB).
pub(super) const FUSED_BWD_BUFFERS: u32 = 9;
/// Max units (`N · hw_units`) handled by the single-dispatch register-cached fused BatchNorm kernels
/// (`REGS = 4` up to 1,024 units, `REGS = 8` up to 2,048 units = 8,192 elements per channel under `VEC4`).
pub(super) const FUSED_MAX_UNITS: usize = 256 * 8;

/// Bounds check for precompiling `"bn_fused_fwd"` / `"bn_fused_bwd"` from a kernel list (`kernels::bounds`).
pub(crate) fn fused_fits(spec: &Spec) -> bool {
    matches!(spec.get_u32("REGS", 4), 4 | 8)
}

/// Single-dispatch register-cached train-mode BatchNorm forward: loads each channel's `X` once into explicit
/// scalar/vec4 register slots (`REGS` = 4 or 8), accumulates 4-lane Welford stats, merges across the 256-thread
/// workgroup (1-barrier 32-wide Chan `subgroupShuffleDown` or portable `bn_tree_merge`), writes `MEAN`,
/// `INVSTD`, optional running stats, and normalizes + activates directly from registers into `Y`.
/// Grid: `(C, 1, 1)`. Params: 0 = C, 1 = hw_units, 2 = m_units, 5 = 256 / hw_units, 6 = 256 % hw_units,
/// 7 = eps, 8 = 1 - momentum, 9 = momentum, 10 = momentum · n / (n - 1).
fn gen_fused_fwd(spec: &Spec) -> String {
    let regs = spec.get_u32("REGS", 4).clamp(1, 8);
    let sg = spec.is_subgroups();
    let mut s = String::new();
    s.push_str(prelude(spec));
    s.push_str(COMMON);
    if !sg {
        s.push_str(WELFORD);
    } else {
        s.push_str(
            "var<workgroup> wn: array<f32, 8>;\n\
             var<workgroup> wm: array<f32, 8>;\n\
             var<workgroup> wq: array<f32, 8>;\n",
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
    s.push_str(
        "  let c = wid.x;\n\
         \x20 let channels = pu(0u);\n\
         \x20 let hw = pu(1u);\n\
         \x20 let m_u = pu(2u);\n\
         \x20 var k = 0.0;\n\
         \x20 var m4 = vec4<f32>(0.0);\n\
         \x20 var q4 = vec4<f32>(0.0);\n\
         \x20 var w = bn_walk_start(0u, lid, hw);\n",
    );
    for r in 0..regs {
        let _ = writeln!(s, "  var v{r}: E;");
    }
    for r in 0..regs {
        let _ = writeln!(
            s,
            "  if (w.e < m_u) {{\n\
             \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
             \x20   v{r} = X[idx];\n\
             \x20   let xv = V4(v{r});\n\
             \x20   k += 1.0;\n\
             \x20   let inv_k = 1.0 / k;\n\
             \x20   if (VEC) {{\n\
             \x20     let d = xv - m4;\n\
             \x20     m4 += d * inv_k;\n\
             \x20     q4 += d * (xv - m4);\n\
             \x20   }} else {{\n\
             \x20     let d = xv.x - m4.x;\n\
             \x20     m4.x += d * inv_k;\n\
             \x20     q4.x += d * (xv.x - m4.x);\n\
             \x20   }}\n\
             \x20 }}"
        );
        if r + 1 < regs {
            s.push_str("  bn_walk_next(&w, hw);\n");
        }
    }
    s.push_str(
        "  var na = 0.0;\n\
         \x20 var ma = 0.0;\n\
         \x20 var qa = 0.0;\n\
         \x20 if (k > 0.0) {\n\
         \x20   if (!VEC) {\n\
         \x20     na = k; ma = m4.x; qa = q4.x;\n\
         \x20   } else {\n\
         \x20     let d01 = m4.y - m4.x;\n\
         \x20     let m01 = m4.x + 0.5 * d01;\n\
         \x20     let q01 = q4.x + q4.y + (0.5 * k * d01) * d01;\n\
         \x20     let d23 = m4.w - m4.z;\n\
         \x20     let m23 = m4.z + 0.5 * d23;\n\
         \x20     let q23 = q4.z + q4.w + (0.5 * k * d23) * d23;\n\
         \x20     let d_all = m23 - m01;\n\
         \x20     na = 4.0 * k;\n\
         \x20     ma = m01 + 0.5 * d_all;\n\
         \x20     qa = q01 + q23 + (k * d_all) * d_all;\n\
         \x20   }\n\
         \x20 }\n",
    );
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
             \x20   var fa = wn[0]; var fm = wm[0]; var fq = wq[0];\n\
             \x20   for (var i = 1u; i < 8u; i += 1u) {\n\
             \x20     let nb = wn[i];\n\
             \x20     if (fa == 0.0) {\n\
             \x20       fa = nb; fm = wm[i]; fq = wq[i];\n\
             \x20     } else if (nb > 0.0) {\n\
             \x20       let n = fa + nb;\n\
             \x20       let d = wm[i] - fm;\n\
             \x20       let delta = d * (nb / n);\n\
             \x20       fm += delta;\n\
             \x20       fq += wq[i] + d * (delta * fa);\n\
             \x20       fa = n;\n\
             \x20     }\n\
             \x20   }\n\
             \x20   let var_b = max(fq / fa, 0.0);\n\
             \x20   let inv = 1.0 / sqrt(var_b + pf(7u));\n\
             \x20   MEAN[c] = fm;\n\
             \x20   INVSTD[c] = inv;\n\
             \x20   if (RUNNING) {\n\
             \x20     RM[c] = RM[c] * pf(8u) + pf(9u) * fm;\n\
             \x20     RV[c] = RV[c] * pf(8u) + pf(10u) * var_b;\n\
             \x20   }\n\
             \x20   wm[0] = fm;\n\
             \x20   wq[0] = inv;\n\
             \x20 }\n\
             \x20 workgroupBarrier();\n",
        );
    } else {
        s.push_str(
            "  wn[lid] = na; wm[lid] = ma; wq[lid] = qa;\n\
             \x20 workgroupBarrier();\n\
             \x20 bn_tree_merge(lid);\n\
             \x20 if (lid == 0u) {\n\
             \x20   let fm = wm[0];\n\
             \x20   let var_b = max(wq[0] / wn[0], 0.0);\n\
             \x20   let inv = 1.0 / sqrt(var_b + pf(7u));\n\
             \x20   MEAN[c] = fm;\n\
             \x20   INVSTD[c] = inv;\n\
             \x20   if (RUNNING) {\n\
             \x20     RM[c] = RM[c] * pf(8u) + pf(9u) * fm;\n\
             \x20     RV[c] = RV[c] * pf(8u) + pf(10u) * var_b;\n\
             \x20   }\n\
             \x20   wq[0] = inv;\n\
             \x20 }\n\
             \x20 workgroupBarrier();\n",
        );
    }
    s.push_str(
        "  let mean_c = wm[0];\n\
         \x20 let inv_c = wq[0];\n\
         \x20 let w_c = W[c];\n\
         \x20 let b_c = B[c];\n\
         \x20 w = bn_walk_start(0u, lid, hw);\n",
    );
    for r in 0..regs {
        let _ = writeln!(
            s,
            "  if (w.e < m_u) {{\n\
             \x20   let idx = (w.n * channels + c) * hw + w.i;\n\
             \x20   if (VEC) {{\n\
             \x20     let xh = (V4(v{r}) - mean_c) * inv_c;\n\
             \x20     Y[idx] = OUT(bn_act(xh * w_c + b_c));\n\
             \x20   }} else {{\n\
             \x20     let xh = (V4(v{r}).x - mean_c) * inv_c;\n\
             \x20     Y[idx] = OUT(vec4<f32>(bn_act_f32(xh * w_c + b_c)));\n\
             \x20   }}\n\
             \x20 }}"
        );
        if r + 1 < regs {
            s.push_str("  bn_walk_next(&w, hw);\n");
        }
    }
    s.push_str("}\n");
    s
}

/// Single-dispatch register-cached train-mode BatchNorm backward: loads `X` and `G` once per channel, computes
/// `xh` and `dz` once (halving activation-derivative `exp()` work), reduces `(db, dw)` across the workgroup
/// (1-barrier 32-wide `subgroupAdd` or portable tree), writes `DB` and `DW`, and writes `DX` directly from
/// register-cached `xh` and `dz`.
/// Grid: `(C, 1, 1)`. Params: 0 = C, 1 = hw_units, 2 = m_units, 3 = 1 / (N·hw), 5 = 256 / hw_units,
/// 6 = 256 % hw_units.
fn gen_fused_bwd(spec: &Spec) -> String {
    let regs = spec.get_u32("REGS", 4).clamp(1, 8);
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
             \x20 if (lid == 0u) {{\n\
             \x20   let db = ((s1[0] + s1[1]) + (s1[2] + s1[3])) + ((s1[4] + s1[5]) + (s1[6] + s1[7]));\n\
             \x20   let dw = ((s2[0] + s2[1]) + (s2[2] + s2[3])) + ((s2[4] + s2[5]) + (s2[6] + s2[7]));\n\
             \x20   DB[c] = db; DW[c] = dw;\n\
             \x20   s1[0] = db * pf(3u); s2[0] = dw * pf(3u);\n\
             \x20 }}\n\
             \x20 workgroupBarrier();\n"
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
             \x20 workgroupBarrier();\n"
        );
    }
    s.push_str(
        "  let db_n = s1[0];\n\
         \x20 let dw_n = s2[0];\n\
         \x20 let scale = wc * inv;\n\
         \x20 w = bn_walk_start(0u, lid, hw);\n",
    );
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
            (&FUSED_BWD, Spec::new().u32("ACT", 1).bool("VEC4", true).u32("REGS", 8).subgroups(false)),
        ] {
            let wgsl = crate::pipeline::render(def, &spec);
            assert!(!wgsl.contains("subgroup"), "portable {} emitted subgroup token:\n{wgsl}", def.name);
        }
    }
}
