//! Tiled fused Scaled Dot-Product Attention (`O = softmax(scale * Q @ K^T [+ causal]) @ V`) in WGSL.
//!
//! Computes attention in a single dispatch without materializing the `[B, Sq, Sk]` attention matrix in global
//! memory (FlashAttention-style online Milakov-Gimelshein softmax):
//! - Workgroup size: 1D `256` threads = `8` rows of `32` lanes (`ty = sgid`, `tx = lane` on 32-wide subgroup
//!   adapters, guaranteeing every row maps cleanly onto one hardware subgroup without relying on 2D workgroup
//!   swizzle order; `ty = lid >> 5`, `tx = lid & 31` on the portable shared-memory fallback).
//! - Each workgroup computes `BR = 16` query rows (2 rows per 32-lane row group) and up to `128` output
//!   head-dim columns (`D_CHUNKS` in `1..=4` register slots per row), tiling keys/values in `BC = 32` blocks
//!   with `33`-padded shared memory (`<= 11.5 KB`, zero bank conflicts, safe under WebGPU's 16 KB limit).
//! - For `D <= 64`, `Q` is loaded into shared memory once before the KV loop (`CACHE_Q = true`); larger `D`
//!   streams `Q` in 32-wide chunks across arbitrary `D` (including `D > 128` via `d_tiles`).
//! - When `CAUSAL` is true, KV tiles strictly past the workgroup's maximum query row are skipped (uniform
//!   early exit) and masked positions are isolated from future `NaN`/`Inf` values in `V`.

use std::fmt::Write as _;

use super::{check_buf, prod, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Query rows per workgroup.
const BR: usize = 16;
/// Max output head-dim columns per workgroup (`4 * 32`).
const BD_TILE: usize = 128;
/// Max head dim where `Q` stays resident in shared memory across all KV tiles.
const CACHE_Q_MAX_D: usize = 64;

/// Bounds check for precompiling `"sdpa_fwd"` from a kernel list (`kernels/bounds.rs`).
pub(crate) fn sdpa_fits(spec: &Spec) -> bool {
    (1..=4).contains(&spec.get_u32("D_CHUNKS", 1))
}

/// Generates the tiled online-softmax SDPA forward shader.
/// Spec: `CAUSAL` (bool), `CACHE_Q` (bool), `D_CHUNKS` (`1..=4`), `f16` (bool), `subgroups` (bool).
/// Params: 0 = `total_blocks`, 1 = `sq`, 2 = `sk`, 3 = `d`, 4 = `q_tiles`, 5 = `d_tiles`,
/// 6 = `scale` (f32), 7 = `neg_inf` (f32).
fn gen_sdpa_fwd(spec: &Spec) -> String {
    let causal = spec.get_bool("CAUSAL");
    let cache_q = spec.get_bool("CACHE_Q");
    let d_chunks = spec.get_u32("D_CHUNKS", 1).clamp(1, 4);
    let sg = spec.is_subgroups();
    let q_stride = if cache_q { 65u32 } else { 33u32 };
    let q_elems = 16u32 * q_stride;

    let mut s = String::with_capacity(6144);
    let _ = writeln!(
        s,
        "@group(0) @binding(1) var<storage, read> Q: array<T>;\n\
         @group(0) @binding(2) var<storage, read> K: array<T>;\n\
         @group(0) @binding(3) var<storage, read> V: array<T>;\n\
         @group(0) @binding(4) var<storage, read_write> O: array<T>;\n\
         var<workgroup> sh_q: array<f32, {q_elems}>;\n\
         var<workgroup> sh_kv: array<f32, 1056>;\n\
         var<workgroup> sh_p: array<f32, 512>;"
    );
    if !sg {
        s.push_str("var<workgroup> sh_red: array<f32, 256>;\n");
    }
    s.push_str(
        "fn isnan1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }\n\
         fn max_nan(a: f32, b: f32) -> f32 {\n\
         \x20 return select(select(select(a, b, b > a), b, isnan1(b)), a, isnan1(a));\n\
         }\n",
    );
    if sg {
        s.push_str(
            "fn row_max32(ty: u32, tx: u32, v: f32, neg_inf: f32) -> f32 {\n\
             \x20 let is_n = isnan1(v);\n\
             \x20 let mask = subgroupBallot(is_n).x;\n\
             \x20 let nan_lane = select(0u, firstTrailingBit(mask), mask != 0u);\n\
             \x20 return select(subgroupMax(select(v, neg_inf, is_n)), subgroupShuffle(v, nan_lane), mask != 0u);\n\
             }\n\
             fn row_sum32(ty: u32, tx: u32, v: f32) -> f32 {\n\
             \x20 return subgroupAdd(v);\n\
             }\n",
        );
    } else {
        s.push_str(
            "fn row_max32(ty: u32, tx: u32, v: f32, neg_inf: f32) -> f32 {\n\
             \x20 let base = ty * 32u;\n\
             \x20 sh_red[base + tx] = v;\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 16u) { sh_red[base + tx] = max_nan(sh_red[base + tx], sh_red[base + tx + 16u]); }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 8u) { sh_red[base + tx] = max_nan(sh_red[base + tx], sh_red[base + tx + 8u]); }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 4u) { sh_red[base + tx] = max_nan(sh_red[base + tx], sh_red[base + tx + 4u]); }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 2u) { sh_red[base + tx] = max_nan(sh_red[base + tx], sh_red[base + tx + 2u]); }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx == 0u) { sh_red[base] = max_nan(sh_red[base], sh_red[base + 1u]); }\n\
             \x20 workgroupBarrier();\n\
             \x20 let res = sh_red[base];\n\
             \x20 workgroupBarrier();\n\
             \x20 return res;\n\
             }\n\
             fn row_sum32(ty: u32, tx: u32, v: f32) -> f32 {\n\
             \x20 let base = ty * 32u;\n\
             \x20 sh_red[base + tx] = v;\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 16u) { sh_red[base + tx] += sh_red[base + tx + 16u]; }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 8u) { sh_red[base + tx] += sh_red[base + tx + 8u]; }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 4u) { sh_red[base + tx] += sh_red[base + tx + 4u]; }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx < 2u) { sh_red[base + tx] += sh_red[base + tx + 2u]; }\n\
             \x20 workgroupBarrier();\n\
             \x20 if (tx == 0u) { sh_red[base] += sh_red[base + 1u]; }\n\
             \x20 workgroupBarrier();\n\
             \x20 let res = sh_red[base];\n\
             \x20 workgroupBarrier();\n\
             \x20 return res;\n\
             }\n",
        );
    }

    s.push_str("@compute @workgroup_size(256)\n");
    if sg {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32,\n\
             \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
             \x20 let ty = min(sgid, 7u);\n\
             \x20 let tx = lane;\n",
        );
    } else {
        s.push_str(
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
             \x20       @builtin(local_invocation_index) lid: u32) {\n\
             \x20 let ty = lid >> 5u;\n\
             \x20 let tx = lid & 31u;\n",
        );
    }
    s.push_str(
        "  let block_id = wid.y * nwg.x + wid.x;\n\
         \x20 if (block_id >= pu(0u)) { return; }\n\
         \x20 let sq = pu(1u);\n\
         \x20 let sk = pu(2u);\n\
         \x20 let d = pu(3u);\n\
         \x20 let q_tiles = pu(4u);\n\
         \x20 let d_tiles = pu(5u);\n\
         \x20 let scale = pf(6u);\n\
         \x20 let neg_inf = pf(7u);\n\
         \x20 let q_tile = block_id % q_tiles;\n\
         \x20 let tmp = block_id / q_tiles;\n\
         \x20 let d_tile = tmp % d_tiles;\n\
         \x20 let b = tmp / d_tiles;\n\
         \x20 let q_base = b * sq * d;\n\
         \x20 let kv_base = b * sk * d;\n\
         \x20 let q0_tile = q_tile * 16u;\n\
         \x20 let d_out_base = d_tile * 128u;\n\
         \x20 let r0 = ty;\n\
         \x20 let r1 = ty + 8u;\n\
         \x20 let q_row0 = q0_tile + r0;\n\
         \x20 let q_row1 = q0_tile + r1;\n",
    );

    if cache_q {
        // Load [16, D] (D <= 64) once into sh_q[16 * 65]: 256 threads load 16 * 64 = 1024 floats (4 per thread).
        s.push_str(
            "  for (var i = lid; i < 1024u; i += 256u) {\n\
             \x20   let qr = i >> 6u;\n\
             \x20   let qc = i & 63u;\n\
             \x20   let gr = q0_tile + qr;\n\
             \x20   var qv = 0.0;\n\
             \x20   if (gr < sq && qc < d) { qv = f32(Q[q_base + gr * d + qc]); }\n\
             \x20   sh_q[qr * 65u + qc] = qv;\n\
             \x20 }\n\
             \x20 workgroupBarrier();\n",
        );
    }

    s.push_str("  var m0 = neg_inf; var l0 = 0.0;\n  var m1 = neg_inf; var l1 = 0.0;\n");
    for c in 0..d_chunks {
        let _ = writeln!(s, "  var o0_{c} = 0.0; var o1_{c} = 0.0;");
    }

    if causal {
        s.push_str("  let kv_end = min(sk, q0_tile + 16u);\n");
    } else {
        s.push_str("  let kv_end = sk;\n");
    }

    s.push_str("  for (var k0 = 0u; k0 < kv_end; k0 += 32u) {\n");
    s.push_str("    var s0 = 0.0;\n    var s1 = 0.0;\n");
    s.push_str("    for (var d0 = 0u; d0 < d; d0 += 32u) {\n");
    if !cache_q {
        s.push_str(
            "      for (var i = lid; i < 512u; i += 256u) {\n\
             \x20       let qr = i >> 5u;\n\
             \x20       let qc = i & 31u;\n\
             \x20       let gr = q0_tile + qr;\n\
             \x20       let gc = d0 + qc;\n\
             \x20       var qv = 0.0;\n\
             \x20       if (gr < sq && gc < d) { qv = f32(Q[q_base + gr * d + gc]); }\n\
             \x20       sh_q[qr * 33u + qc] = qv;\n\
             \x20     }\n",
        );
    }
    s.push_str(
        "      for (var i = lid; i < 1024u; i += 256u) {\n\
         \x20       let kr = i >> 5u;\n\
         \x20       let kc = i & 31u;\n\
         \x20       let gr = k0 + kr;\n\
         \x20       let gc = d0 + kc;\n\
         \x20       var kv = 0.0;\n\
         \x20       if (gr < sk && gc < d) { kv = f32(K[kv_base + gr * d + gc]); }\n\
         \x20       sh_kv[kr * 33u + kc] = kv;\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20     let dk_end = min(32u, d - d0);\n\
         \x20     let k_off = tx * 33u;\n",
    );
    if cache_q {
        s.push_str(
            "      let q0_off = r0 * 65u + d0;\n\
             \x20     let q1_off = r1 * 65u + d0;\n",
        );
    } else {
        s.push_str(
            "      let q0_off = r0 * 33u;\n\
             \x20     let q1_off = r1 * 33u;\n",
        );
    }
    s.push_str(
        "      for (var dk = 0u; dk < dk_end; dk += 1u) {\n\
         \x20       let kv = sh_kv[k_off + dk];\n\
         \x20       s0 = fma(sh_q[q0_off + dk], kv, s0);\n\
         \x20       s1 = fma(sh_q[q1_off + dk], kv, s1);\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20   }\n",
    );

    // Mask and online softmax update for r0 and r1.
    s.push_str("    let k_col = k0 + tx;\n");
    if causal {
        s.push_str(
            "    let valid_k0 = q_row0 < sq && k_col < sk && k_col <= q_row0;\n\
             \x20   let valid_k1 = q_row1 < sq && k_col < sk && k_col <= q_row1;\n",
        );
    } else {
        s.push_str(
            "    let valid_k0 = q_row0 < sq && k_col < sk;\n\
             \x20   let valid_k1 = q_row1 < sq && k_col < sk;\n",
        );
    }
    s.push_str(
        "    s0 = select(neg_inf, s0 * scale, valid_k0);\n\
         \x20   s1 = select(neg_inf, s1 * scale, valid_k1);\n",
    );

    for (r_idx, (r_var, s_var, vk_var, m_var, l_var)) in
        [(0, ("r0", "s0", "valid_k0", "m0", "l0")), (1, ("r1", "s1", "valid_k1", "m1", "l1"))]
    {
        let _ = writeln!(
            s,
            "    {{\n\
             \x20     let mb = row_max32(ty, tx, {s_var}, neg_inf);\n\
             \x20     var alpha = 1.0;\n\
             \x20     var pv = 0.0;\n\
             \x20     if (isnan1(mb) || mb != neg_inf) {{\n\
             \x20       let mn = max_nan({m_var}, mb);\n\
             \x20       let is_inf_mn = (bitcast<u32>(mn) & 0x7fffffffu) == 0x7f800000u;\n\
             \x20       if ({l_var} == 0.0) {{\n\
             \x20         alpha = 0.0;\n\
             \x20       }} else if (is_inf_mn) {{\n\
             \x20         alpha = select(0.0, 1.0, {m_var} == mn);\n\
             \x20       }} else {{\n\
             \x20         alpha = exp({m_var} - mn);\n\
             \x20       }}\n\
             \x20       pv = select(0.0, exp({s_var} - mn), {vk_var});\n\
             \x20       {m_var} = mn;\n\
             \x20     }}\n\
             \x20     let lb = row_sum32(ty, tx, pv);\n\
             \x20     let l_old = {l_var};\n\
             \x20     let l_new = l_old * alpha + lb;\n\
             \x20     let l_ok = l_new > 0.0 && (bitcast<u32>(l_new) & 0x7fffffffu) < 0x7f800000u;\n\
             \x20     let inv_ln = select(1.0, 1.0 / l_new, l_ok);\n\
             \x20     let o_scale = select(alpha, (l_old * alpha) * inv_ln, l_ok);\n\
             \x20     {l_var} = l_new;\n\
             \x20     sh_p[{r_var} * 32u + tx] = pv * inv_ln;"
        );
        for c in 0..d_chunks {
            let _ = writeln!(s, "      o{r_idx}_{c} *= o_scale;");
        }
        s.push_str("    }\n");
    }
    s.push_str("    workgroupBarrier();\n");
    s.push_str("    let k_len = min(32u, sk - k0);\n");

    // Accumulate O += P_norm @ V for each active output head-dim chunk.
    for c in 0..d_chunks {
        let off = c * 32;
        let _ = write!(
            s,
            "    {{\n\
             \x20     let d_col0 = d_out_base + {off}u;\n\
             \x20     if (d_col0 < d) {{\n\
             \x20       for (var i = lid; i < 1024u; i += 256u) {{\n\
             \x20         let vr = i >> 5u;\n\
             \x20         let vc = i & 31u;\n\
             \x20         let gr = k0 + vr;\n\
             \x20         let gc = d_col0 + vc;\n\
             \x20         var vv0 = 0.0;\n\
             \x20         if (gr < sk && gc < d) {{ vv0 = f32(V[kv_base + gr * d + gc]); }}\n\
             \x20         sh_kv[vr * 33u + vc] = vv0;\n\
             \x20       }}\n\
             \x20       workgroupBarrier();\n\
             \x20       for (var kc = 0u; kc < k_len; kc += 1u) {{\n\
             \x20         let vv = sh_kv[kc * 33u + tx];\n"
        );
        if causal {
            let _ = write!(
                s,
                "         if (k0 + kc <= q_row0) {{ o0_{c} = fma(sh_p[r0 * 32u + kc], vv, o0_{c}); }}\n\
                 \x20         if (k0 + kc <= q_row1) {{ o1_{c} = fma(sh_p[r1 * 32u + kc], vv, o1_{c}); }}\n"
            );
        } else {
            let _ = write!(
                s,
                "         o0_{c} = fma(sh_p[r0 * 32u + kc], vv, o0_{c});\n\
                 \x20         o1_{c} = fma(sh_p[r1 * 32u + kc], vv, o1_{c});\n"
            );
        }
        s.push_str("        }\n        workgroupBarrier();\n      }\n    }\n");
    }
    s.push_str("  }\n");

    // `o` is already normalized for valid finite `l > 0`; propagate NaN when `l == 0` or non-finite.
    s.push_str(
        "  let ok0 = l0 > 0.0 && (bitcast<u32>(l0) & 0x7fffffffu) < 0x7f800000u;\n\
         \x20 let ok1 = l1 > 0.0 && (bitcast<u32>(l1) & 0x7fffffffu) < 0x7f800000u;\n\
         \x20 let fin0 = select(1.0 / l0, 1.0, ok0);\n\
         \x20 let fin1 = select(1.0 / l1, 1.0, ok1);\n",
    );
    for c in 0..d_chunks {
        let off = c * 32;
        let _ = write!(
            s,
            "  {{\n\
             \x20   let d_col = d_out_base + {off}u + tx;\n\
             \x20   if (d_col < d) {{\n\
             \x20     if (q_row0 < sq) {{ O[q_base + q_row0 * d + d_col] = T(o0_{c} * fin0); }}\n\
             \x20     if (q_row1 < sq) {{ O[q_base + q_row1 * d + d_col] = T(o1_{c} * fin1); }}\n\
             \x20   }}\n\
             \x20 }}\n"
        );
    }
    s.push_str("}\n");
    s
}

use Access::{Read as R, ReadWrite as RW};

static SDPA_FWD: KernelDef =
    KernelDef { name: "sdpa_fwd", storage: &[R, R, R, RW], source: Source::Generated(gen_sdpa_fwd) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&SDPA_FWD];

/// Tiled fused Scaled Dot-Product Attention forward: `O = softmax(scale * Q @ K^T [+ causal]) @ V`.
///
/// Shapes (dense row-major, `b` folds `batch * num_heads`):
/// - `q`, `out`: `[b, sq, d]`
/// - `k`, `v`: `[b, sk, d]`
///
/// `dtype` may be `F32` or (with `shader-f16`) `F16`; attention scores, softmax, and output accumulation run in
/// `f32` and round once on store.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward(
    ctx: &GpuContext,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    out: &wgpu::Buffer,
    (b, sq, sk, d): (usize, usize, usize, usize),
    scale: f32,
    causal: bool,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "sdpa_forward")?;
    let q_elems = prod(&[b, sq, d])?;
    let kv_elems = prod(&[b, sk, d])?;
    check_buf(q, q_elems, dtype, "sdpa q")?;
    check_buf(k, kv_elems, dtype, "sdpa k")?;
    check_buf(v, kv_elems, dtype, "sdpa v")?;
    check_buf(out, q_elems, dtype, "sdpa out")?;
    if q_elems == 0 {
        return Ok(());
    }
    if sk == 0 {
        return Err(shape_err("sdpa_forward: empty key/value sequence (sk == 0)"));
    }
    if q_elems > u32::MAX as usize || kv_elems > u32::MAX as usize {
        return Err(GpuError::Limit("sdpa_forward: tensor exceeds u32 indexing".into()));
    }
    let q_tiles = sq.div_ceil(BR);
    let d_tiles = d.div_ceil(BD_TILE);
    let total_blocks = prod(&[b, d_tiles, q_tiles])?;
    if total_blocks > u32::MAX as usize {
        return Err(GpuError::Limit("sdpa_forward: dispatch exceeds u32 block count".into()));
    }
    let d_chunks = d.min(BD_TILE).div_ceil(32) as u32;
    let cache_q = d <= CACHE_Q_MAX_D;
    let kern = ctx.kernel(
        &SDPA_FWD,
        Spec::new()
            .bool("CAUSAL", causal)
            .bool("CACHE_Q", cache_q)
            .u32("D_CHUNKS", d_chunks)
            .f16(dtype == DType::F16)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new()
        .uz(total_blocks)?
        .uz(sq)?
        .uz(sk)?
        .uz(d)?
        .uz(q_tiles)?
        .uz(d_tiles)?
        .f(scale)
        .f(f32::NEG_INFINITY);
    ctx.dispatch(&kern, &[q, k, v, out], &p, ctx.grid_1d(total_blocks as u64 * 256, 256)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdpa_portable_emits_no_subgroup_tokens() {
        for (causal, cache_q, chunks) in [(false, true, 1), (true, true, 2), (true, false, 4)] {
            let spec = Spec::new()
                .bool("CAUSAL", causal)
                .bool("CACHE_Q", cache_q)
                .u32("D_CHUNKS", chunks)
                .subgroups(false);
            let wgsl = crate::pipeline::render(&SDPA_FWD, &spec);
            assert!(!wgsl.contains("subgroup"), "portable sdpa emitted subgroup token:\n{wgsl}");
        }
    }
}
