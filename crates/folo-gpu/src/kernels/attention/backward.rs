//! FlashAttention-2 style tiled WGSL backward pass for Scaled Dot-Product Attention (`dQ`, `dK`, `dV`).
//!
//! Three zero-atomic dispatches, each staying within `<= 8` storage bindings and `<= 14.5 KB` workgroup memory
//! (safe under WebGPU's minimum limits):
//! 1. `sdpa_bwd_delta`: packs `STATS[b, q] = vec2(lse[b, q], sum_j dO[b, q, j] * O[b, q, j])` in `F32` (8 query
//!    rows per 256-thread workgroup, 32-lane subgroup or portable row reduction).
//! 2. `sdpa_bwd_dkdv`: tiles `K`/`V` in `BC = 16` rows × up to `128` head-dim cols (`D_CHUNKS` in `1..=4`,
//!    arbitrary `D` via `d_tiles`), streaming `Q`/`dO` in `BR = 32` blocks (with causal early-start `q0 >= k0`).
//! 3. `sdpa_bwd_dq`: tiles `Q`/`dO` in `BR = 16` rows × up to `128` head-dim cols (`CACHE_Q` caches both `Q` and
//!    `dO` in shared memory for `D <= 64`), streaming `K`/`V` in `BC = 32` blocks (with causal early-exit).

use std::fmt::Write as _;

use super::{BD_TILE, BR, CACHE_Q_MAX_D, SdpaMask, check_mask, push_main_header, push_row_helpers};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::kernels::{check_buf, prod, use_subgroups};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Key/value rows per workgroup in `sdpa_bwd_dkdv`.
const BC_BWD: usize = 16;

/// Generates `sdpa_bwd_delta`: packs `STATS[row] = vec2(LSE[row], dot(dO[row], O[row]))`.
/// Params: 0 = `total_rows` (`b * sq`), 1 = `d`.
fn gen_sdpa_bwd_delta(spec: &Spec) -> String {
    let sg = spec.is_subgroups();
    let mut s = String::with_capacity(2048);
    s.push_str(
        "@group(0) @binding(1) var<storage, read> O: array<T>;\n\
         @group(0) @binding(2) var<storage, read> DO: array<T>;\n\
         @group(0) @binding(3) var<storage, read> LSE: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> STATS: array<vec2<f32>>;\n",
    );
    if !sg {
        s.push_str("var<workgroup> sh_red: array<f32, 256>;\n");
    }
    push_row_helpers(&mut s, sg, false);
    push_main_header(&mut s, sg);
    s.push_str(
        "  let block_id = wid.y * nwg.x + wid.x;\n\
         \x20 let total_rows = pu(0u);\n\
         \x20 let d = pu(1u);\n\
         \x20 let row = block_id * 8u + ty;\n\
         \x20 var acc = 0.0;\n\
         \x20 if (row < total_rows) {\n\
         \x20   let base = row * d;\n\
         \x20   for (var j = tx; j < d; j += 32u) {\n\
         \x20     acc = fma(f32(DO[base + j]), f32(O[base + j]), acc);\n\
         \x20   }\n\
         \x20 }\n\
         \x20 let delta = row_sum32(ty, tx, acc);\n\
         \x20 if (row < total_rows && tx == 0u) {\n\
         \x20   STATS[row] = vec2<f32>(LSE[row], delta);\n\
         \x20 }\n\
         }\n",
    );
    s
}

/// Generates `sdpa_bwd_dkdv`: tiled `dK` and `dV` kernel (`BC_BWD = 16` KV rows, `BR = 32` Q streaming blocks).
/// Spec: `CAUSAL`, `HAS_MASK`, `D_CHUNKS` (`1..=4`), `f16`, `subgroups`.
/// Params: 0 = `total_blocks`, 1 = `sq`, 2 = `sk`, 3 = `d`, 4 = `k_tiles`, 5 = `d_tiles`,
/// 6 = `scale` (f32), 7 = `neg_inf` (f32), 8 = `mask_b_stride`.
fn gen_sdpa_bwd_dkdv(spec: &Spec) -> String {
    let causal = spec.get_bool("CAUSAL");
    let has_mask = spec.get_bool("HAS_MASK");
    let d_chunks = spec.get_u32("D_CHUNKS", 1).clamp(1, 4);
    let sg = spec.is_subgroups();

    let mut s = String::with_capacity(8192);
    s.push_str(
        "@group(0) @binding(1) var<storage, read> Q: array<T>;\n\
         @group(0) @binding(2) var<storage, read> K: array<T>;\n\
         @group(0) @binding(3) var<storage, read> V: array<T>;\n\
         @group(0) @binding(4) var<storage, read> DO: array<T>;\n\
         @group(0) @binding(5) var<storage, read> STATS: array<vec2<f32>>;\n\
         @group(0) @binding(6) var<storage, read> MASK: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> DK: array<T>;\n\
         @group(0) @binding(8) var<storage, read_write> DV: array<T>;\n\
         var<workgroup> sh_kv: array<f32, 528>;\n\
         var<workgroup> sh_q: array<f32, 1056>;\n\
         var<workgroup> sh_p: array<f32, 512>;\n\
         var<workgroup> sh_ds: array<f32, 512>;\n",
    );
    if has_mask {
        s.push_str("var<workgroup> sh_valid: array<u32, 512>;\n");
    }
    push_main_header(&mut s, sg);

    s.push_str(
        "  let block_id = wid.y * nwg.x + wid.x;\n\
         \x20 if (block_id >= pu(0u)) { return; }\n\
         \x20 let sq = pu(1u);\n\
         \x20 let sk = pu(2u);\n\
         \x20 let d = pu(3u);\n\
         \x20 let k_tiles = pu(4u);\n\
         \x20 let d_tiles = pu(5u);\n\
         \x20 let scale = pf(6u);\n\
         \x20 let neg_inf = pf(7u);\n\
         \x20 let mask_b_stride = pu(8u);\n\
         \x20 let k_tile = block_id % k_tiles;\n\
         \x20 let tmp = block_id / k_tiles;\n\
         \x20 let d_tile = tmp % d_tiles;\n\
         \x20 let b = tmp / d_tiles;\n\
         \x20 let q_base = b * sq * d;\n\
         \x20 let kv_base = b * sk * d;\n\
         \x20 let stats_base = b * sq;\n\
         \x20 let mask_base = b * mask_b_stride;\n\
         \x20 let k0_tile = k_tile * 16u;\n\
         \x20 let d_out_base = d_tile * 128u;\n\
         \x20 let r0 = ty;\n\
         \x20 let r1 = ty + 8u;\n\
         \x20 let k_row0 = k0_tile + r0;\n\
         \x20 let k_row1 = k0_tile + r1;\n",
    );

    for c in 0..d_chunks {
        let _ =
            writeln!(s, "  var dk0_{c} = 0.0; var dk1_{c} = 0.0;\n  var dv0_{c} = 0.0; var dv1_{c} = 0.0;");
    }

    if causal {
        // Every k_row in this tile has k_row >= k0_tile, so queries q_row < k0_tile never attend here.
        s.push_str("  let q_start = k0_tile;\n");
    } else {
        s.push_str("  let q_start = 0u;\n");
    }

    s.push_str(
        "  for (var q0 = q_start; q0 < sq; q0 += 32u) {\n\
         \x20   var s0 = 0.0; var s1 = 0.0;\n\
         \x20   for (var d0 = 0u; d0 < d; d0 += 32u) {\n\
         \x20     for (var i = lid; i < 512u; i += 256u) {\n\
         \x20       let kr = i >> 5u; let kc = i & 31u;\n\
         \x20       let gr = k0_tile + kr; let gc = d0 + kc;\n\
         \x20       var kv = 0.0;\n\
         \x20       if (gr < sk && gc < d) { kv = f32(K[kv_base + gr * d + gc]); }\n\
         \x20       sh_kv[kr * 33u + kc] = kv;\n\
         \x20     }\n\
         \x20     for (var i = lid; i < 1024u; i += 256u) {\n\
         \x20       let qr = i >> 5u; let qc = i & 31u;\n\
         \x20       let gr = q0 + qr; let gc = d0 + qc;\n\
         \x20       var qv = 0.0;\n\
         \x20       if (gr < sq && gc < d) { qv = f32(Q[q_base + gr * d + gc]); }\n\
         \x20       sh_q[qr * 33u + qc] = qv;\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20     let dk_end = min(32u, d - d0);\n\
         \x20     let q_off = tx * 33u;\n\
         \x20     let k0_off = r0 * 33u;\n\
         \x20     let k1_off = r1 * 33u;\n\
         \x20     for (var dk = 0u; dk < dk_end; dk += 1u) {\n\
         \x20       let qv = sh_q[q_off + dk];\n\
         \x20       s0 = fma(sh_kv[k0_off + dk], qv, s0);\n\
         \x20       s1 = fma(sh_kv[k1_off + dk], qv, s1);\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20   }\n\
         \x20   var dp0 = 0.0; var dp1 = 0.0;\n\
         \x20   for (var d0 = 0u; d0 < d; d0 += 32u) {\n\
         \x20     for (var i = lid; i < 512u; i += 256u) {\n\
         \x20       let vr = i >> 5u; let vc = i & 31u;\n\
         \x20       let gr = k0_tile + vr; let gc = d0 + vc;\n\
         \x20       var vv = 0.0;\n\
         \x20       if (gr < sk && gc < d) { vv = f32(V[kv_base + gr * d + gc]); }\n\
         \x20       sh_kv[vr * 33u + vc] = vv;\n\
         \x20     }\n\
         \x20     for (var i = lid; i < 1024u; i += 256u) {\n\
         \x20       let qr = i >> 5u; let qc = i & 31u;\n\
         \x20       let gr = q0 + qr; let gc = d0 + qc;\n\
         \x20       var dov = 0.0;\n\
         \x20       if (gr < sq && gc < d) { dov = f32(DO[q_base + gr * d + gc]); }\n\
         \x20       sh_q[qr * 33u + qc] = dov;\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20     let dk_end = min(32u, d - d0);\n\
         \x20     let do_off = tx * 33u;\n\
         \x20     let v0_off = r0 * 33u;\n\
         \x20     let v1_off = r1 * 33u;\n\
         \x20     for (var dk = 0u; dk < dk_end; dk += 1u) {\n\
         \x20       let dov = sh_q[do_off + dk];\n\
         \x20       dp0 = fma(sh_kv[v0_off + dk], dov, dp0);\n\
         \x20       dp1 = fma(sh_kv[v1_off + dk], dov, dp1);\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20   }\n\
         \x20   let q_row = q0 + tx;\n\
         \x20   var st = vec2<f32>(neg_inf, 0.0);\n\
         \x20   if (q_row < sq) { st = STATS[stats_base + q_row]; }\n\
         \x20   let lse_q = st.x;\n\
         \x20   let delta_q = st.y;\n",
    );

    if causal {
        s.push_str(
            "    var v0 = k_row0 < sk && q_row < sq && k_row0 <= q_row;\n\
             \x20   var v1 = k_row1 < sk && q_row < sq && k_row1 <= q_row;\n",
        );
    } else {
        s.push_str(
            "    var v0 = k_row0 < sk && q_row < sq;\n\
             \x20   var v1 = k_row1 < sk && q_row < sq;\n",
        );
    }
    for (r_var, k_var, v_var, s_var, dp_var) in
        [("r0", "k_row0", "v0", "s0", "dp0"), ("r1", "k_row1", "v1", "s1", "dp1")]
    {
        let _ = write!(
            s,
            "    {{\n\
             \x20     var p = 0.0;\n\
             \x20     var ds = 0.0;\n\
             \x20     if ({v_var}) {{\n"
        );
        if has_mask {
            let _ = write!(
                s,
                "       let mv = MASK[mask_base + q_row * sk + {k_var}];\n\
                 \x20       if (mv == neg_inf) {{\n\
                 \x20         {v_var} = false;\n\
                 \x20       }} else if (lse_q != neg_inf) {{\n\
                 \x20         p = exp({s_var} * scale + mv - lse_q);\n\
                 \x20         ds = p * ({dp_var} - delta_q) * scale;\n\
                 \x20       }}\n"
            );
        } else {
            let _ = write!(
                s,
                "       if (lse_q != neg_inf) {{\n\
                 \x20         p = exp({s_var} * scale - lse_q);\n\
                 \x20         ds = p * ({dp_var} - delta_q) * scale;\n\
                 \x20       }}\n"
            );
        }
        let _ = write!(
            s,
            "     }}\n\
             \x20     sh_p[{r_var} * 32u + tx] = p;\n\
             \x20     sh_ds[{r_var} * 32u + tx] = ds;\n"
        );
        if has_mask {
            let _ = writeln!(s, "      sh_valid[{r_var} * 32u + tx] = select(0u, 1u, {v_var});");
        }
        s.push_str("    }\n");
    }

    s.push_str("    workgroupBarrier();\n    let q_len = min(32u, sq - q0);\n");

    for c in 0..d_chunks {
        let off = c * 32;
        let _ = write!(
            s,
            "    {{\n\
             \x20     let d_col0 = d_out_base + {off}u;\n\
             \x20     if (d_col0 < d) {{\n\
             \x20       for (var i = lid; i < 1024u; i += 256u) {{\n\
             \x20         let qr = i >> 5u; let qc = i & 31u;\n\
             \x20         let gr = q0 + qr; let gc = d_col0 + qc;\n\
             \x20         var dov = 0.0;\n\
             \x20         if (gr < sq && gc < d) {{ dov = f32(DO[q_base + gr * d + gc]); }}\n\
             \x20         sh_q[qr * 33u + qc] = dov;\n\
             \x20       }}\n\
             \x20       workgroupBarrier();\n\
             \x20       for (var qc = 0u; qc < q_len; qc += 1u) {{\n\
             \x20         let dov = sh_q[qc * 33u + tx];\n"
        );
        if has_mask {
            let _ = write!(
                s,
                "         if (sh_valid[r0 * 32u + qc] != 0u) {{ dv0_{c} = fma(sh_p[r0 * 32u + qc], dov, dv0_{c}); }}\n\
                 \x20         if (sh_valid[r1 * 32u + qc] != 0u) {{ dv1_{c} = fma(sh_p[r1 * 32u + qc], dov, dv1_{c}); }}\n"
            );
        } else if causal {
            let _ = write!(
                s,
                "         if (k_row0 <= q0 + qc) {{ dv0_{c} = fma(sh_p[r0 * 32u + qc], dov, dv0_{c}); }}\n\
                 \x20         if (k_row1 <= q0 + qc) {{ dv1_{c} = fma(sh_p[r1 * 32u + qc], dov, dv1_{c}); }}\n"
            );
        } else {
            let _ = write!(
                s,
                "         dv0_{c} = fma(sh_p[r0 * 32u + qc], dov, dv0_{c});\n\
                 \x20         dv1_{c} = fma(sh_p[r1 * 32u + qc], dov, dv1_{c});\n"
            );
        }
        let _ = write!(
            s,
            "       }}\n\
             \x20       workgroupBarrier();\n\
             \x20       for (var i = lid; i < 1024u; i += 256u) {{\n\
             \x20         let qr = i >> 5u; let qc = i & 31u;\n\
             \x20         let gr = q0 + qr; let gc = d_col0 + qc;\n\
             \x20         var qv = 0.0;\n\
             \x20         if (gr < sq && gc < d) {{ qv = f32(Q[q_base + gr * d + gc]); }}\n\
             \x20         sh_q[qr * 33u + qc] = qv;\n\
             \x20       }}\n\
             \x20       workgroupBarrier();\n\
             \x20       for (var qc = 0u; qc < q_len; qc += 1u) {{\n\
             \x20         let qv = sh_q[qc * 33u + tx];\n"
        );
        if has_mask {
            let _ = write!(
                s,
                "         if (sh_valid[r0 * 32u + qc] != 0u) {{ dk0_{c} = fma(sh_ds[r0 * 32u + qc], qv, dk0_{c}); }}\n\
                 \x20         if (sh_valid[r1 * 32u + qc] != 0u) {{ dk1_{c} = fma(sh_ds[r1 * 32u + qc], qv, dk1_{c}); }}\n"
            );
        } else if causal {
            let _ = write!(
                s,
                "         if (k_row0 <= q0 + qc) {{ dk0_{c} = fma(sh_ds[r0 * 32u + qc], qv, dk0_{c}); }}\n\
                 \x20         if (k_row1 <= q0 + qc) {{ dk1_{c} = fma(sh_ds[r1 * 32u + qc], qv, dk1_{c}); }}\n"
            );
        } else {
            let _ = write!(
                s,
                "         dk0_{c} = fma(sh_ds[r0 * 32u + qc], qv, dk0_{c});\n\
                 \x20         dk1_{c} = fma(sh_ds[r1 * 32u + qc], qv, dk1_{c});\n"
            );
        }
        s.push_str("        }\n        workgroupBarrier();\n      }\n    }\n");
    }
    s.push_str("  }\n");

    for c in 0..d_chunks {
        let off = c * 32;
        let _ = write!(
            s,
            "  {{\n\
             \x20   let d_col = d_out_base + {off}u + tx;\n\
             \x20   if (d_col < d) {{\n\
             \x20     if (k_row0 < sk) {{\n\
             \x20       DK[kv_base + k_row0 * d + d_col] = T(dk0_{c});\n\
             \x20       DV[kv_base + k_row0 * d + d_col] = T(dv0_{c});\n\
             \x20     }}\n\
             \x20     if (k_row1 < sk) {{\n\
             \x20       DK[kv_base + k_row1 * d + d_col] = T(dk1_{c});\n\
             \x20       DV[kv_base + k_row1 * d + d_col] = T(dv1_{c});\n\
             \x20     }}\n\
             \x20   }}\n\
             \x20 }}\n"
        );
    }
    s.push_str("}\n");
    s
}

/// Generates `sdpa_bwd_dq`: tiled `dQ` kernel (`BR = 16` Q rows, `BC = 32` KV streaming blocks).
/// Spec: `CAUSAL`, `HAS_MASK`, `CACHE_Q`, `D_CHUNKS` (`1..=4`), `f16`, `subgroups`.
/// Params: 0 = `total_blocks`, 1 = `sq`, 2 = `sk`, 3 = `d`, 4 = `q_tiles`, 5 = `d_tiles`,
/// 6 = `scale` (f32), 7 = `neg_inf` (f32), 8 = `mask_b_stride`.
fn gen_sdpa_bwd_dq(spec: &Spec) -> String {
    let causal = spec.get_bool("CAUSAL");
    let has_mask = spec.get_bool("HAS_MASK");
    let cache_q = spec.get_bool("CACHE_Q");
    let d_chunks = spec.get_u32("D_CHUNKS", 1).clamp(1, 4);
    let sg = spec.is_subgroups();
    // When `HAS_MASK` is true and `CACHE_Q` is true, keep shared memory <= 16 KB (`sh_valid` adds 2 KB,
    // so only cache `sh_do` separately when `sh_q + sh_do + sh_kv + sh_ds + sh_valid <= 16 KB`; `sh_q` + `sh_do`
    // is 8320 B, `sh_kv` 4224 B, `sh_ds` 2048 B = 14592 B, + `sh_valid` 2048 B = 16640 B > 16384 B).
    // Therefore pack `sh_valid` as a 16-word bitmask (or skip `sh_do` cache when `has_mask`)! Even simpler:
    // use `cache_do = cache_q && !has_mask`, keeping total shared memory strictly <= 14.5 KB in EVERY config!
    let cache_do = cache_q && !has_mask;
    let q_stride = if cache_q { 65u32 } else { 33u32 };
    let q_elems = 16u32 * q_stride;

    let mut s = String::with_capacity(8192);
    let _ = writeln!(
        s,
        "@group(0) @binding(1) var<storage, read> Q: array<T>;\n\
         @group(0) @binding(2) var<storage, read> K: array<T>;\n\
         @group(0) @binding(3) var<storage, read> V: array<T>;\n\
         @group(0) @binding(4) var<storage, read> DO: array<T>;\n\
         @group(0) @binding(5) var<storage, read> STATS: array<vec2<f32>>;\n\
         @group(0) @binding(6) var<storage, read> MASK: array<f32>;\n\
         @group(0) @binding(7) var<storage, read_write> DQ: array<T>;\n\
         var<workgroup> sh_q: array<f32, {q_elems}>;\n\
         var<workgroup> sh_kv: array<f32, 1056>;\n\
         var<workgroup> sh_ds: array<f32, 512>;"
    );
    if cache_do {
        s.push_str("var<workgroup> sh_do: array<f32, 1040>;\n");
    } else if cache_q {
        // Streaming `dO` uses a small 16 x 33 tile (2112 B) so `sh_q` stays cached!
        s.push_str("var<workgroup> sh_do: array<f32, 528>;\n");
    }
    if has_mask {
        s.push_str("var<workgroup> sh_valid: array<u32, 512>;\n");
    }
    push_main_header(&mut s, sg);

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
         \x20 let mask_b_stride = pu(8u);\n\
         \x20 let q_tile = block_id % q_tiles;\n\
         \x20 let tmp = block_id / q_tiles;\n\
         \x20 let d_tile = tmp % d_tiles;\n\
         \x20 let b = tmp / d_tiles;\n\
         \x20 let q_base = b * sq * d;\n\
         \x20 let kv_base = b * sk * d;\n\
         \x20 let stats_base = b * sq;\n\
         \x20 let mask_base = b * mask_b_stride;\n\
         \x20 let q0_tile = q_tile * 16u;\n\
         \x20 let d_out_base = d_tile * 128u;\n\
         \x20 let r0 = ty;\n\
         \x20 let r1 = ty + 8u;\n\
         \x20 let q_row0 = q0_tile + r0;\n\
         \x20 let q_row1 = q0_tile + r1;\n\
         \x20 var st0 = vec2<f32>(neg_inf, 0.0);\n\
         \x20 var st1 = vec2<f32>(neg_inf, 0.0);\n\
         \x20 if (q_row0 < sq) { st0 = STATS[stats_base + q_row0]; }\n\
         \x20 if (q_row1 < sq) { st1 = STATS[stats_base + q_row1]; }\n\
         \x20 let lse0 = st0.x; let delta0 = st0.y;\n\
         \x20 let lse1 = st1.x; let delta1 = st1.y;\n",
    );

    if cache_q {
        s.push_str(
            "  for (var i = lid; i < 1024u; i += 256u) {\n\
             \x20   let qr = i >> 6u; let qc = i & 63u;\n\
             \x20   let gr = q0_tile + qr;\n\
             \x20   var qv = 0.0;\n\
             \x20   if (gr < sq && qc < d) { qv = f32(Q[q_base + gr * d + qc]); }\n\
             \x20   sh_q[qr * 65u + qc] = qv;\n",
        );
        if cache_do {
            s.push_str(
                "    var dov = 0.0;\n\
                 \x20   if (gr < sq && qc < d) { dov = f32(DO[q_base + gr * d + qc]); }\n\
                 \x20   sh_do[qr * 65u + qc] = dov;\n",
            );
        }
        s.push_str("  }\n  workgroupBarrier();\n");
    }

    for c in 0..d_chunks {
        let _ = writeln!(s, "  var dq0_{c} = 0.0; var dq1_{c} = 0.0;");
    }

    if causal {
        s.push_str("  let kv_end = min(sk, q0_tile + 16u);\n");
    } else {
        s.push_str("  let kv_end = sk;\n");
    }

    s.push_str(
        "  for (var k0 = 0u; k0 < kv_end; k0 += 32u) {\n\
         \x20   var s0 = 0.0; var s1 = 0.0;\n\
         \x20   for (var d0 = 0u; d0 < d; d0 += 32u) {\n",
    );
    if !cache_q {
        s.push_str(
            "      for (var i = lid; i < 512u; i += 256u) {\n\
             \x20       let qr = i >> 5u; let qc = i & 31u;\n\
             \x20       let gr = q0_tile + qr; let gc = d0 + qc;\n\
             \x20       var qv = 0.0;\n\
             \x20       if (gr < sq && gc < d) { qv = f32(Q[q_base + gr * d + gc]); }\n\
             \x20       sh_q[qr * 33u + qc] = qv;\n\
             \x20     }\n",
        );
    }
    s.push_str(
        "      for (var i = lid; i < 1024u; i += 256u) {\n\
         \x20       let kr = i >> 5u; let kc = i & 31u;\n\
         \x20       let gr = k0 + kr; let gc = d0 + kc;\n\
         \x20       var kv = 0.0;\n\
         \x20       if (gr < sk && gc < d) { kv = f32(K[kv_base + gr * d + gc]); }\n\
         \x20       sh_kv[kr * 33u + kc] = kv;\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20     let dk_end = min(32u, d - d0);\n\
         \x20     let k_off = tx * 33u;\n",
    );
    if cache_q {
        s.push_str("      let q0_off = r0 * 65u + d0; let q1_off = r1 * 65u + d0;\n");
    } else {
        s.push_str("      let q0_off = r0 * 33u; let q1_off = r1 * 33u;\n");
    }
    s.push_str(
        "      for (var dk = 0u; dk < dk_end; dk += 1u) {\n\
         \x20       let kv = sh_kv[k_off + dk];\n\
         \x20       s0 = fma(sh_q[q0_off + dk], kv, s0);\n\
         \x20       s1 = fma(sh_q[q1_off + dk], kv, s1);\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20   }\n\
         \x20   var dp0 = 0.0; var dp1 = 0.0;\n\
         \x20   for (var d0 = 0u; d0 < d; d0 += 32u) {\n",
    );
    if !cache_do {
        let do_arr = if cache_q { "sh_do" } else { "sh_q" };
        let _ = write!(
            s,
            "      for (var i = lid; i < 512u; i += 256u) {{\n\
             \x20       let qr = i >> 5u; let qc = i & 31u;\n\
             \x20       let gr = q0_tile + qr; let gc = d0 + qc;\n\
             \x20       var dov = 0.0;\n\
             \x20       if (gr < sq && gc < d) {{ dov = f32(DO[q_base + gr * d + gc]); }}\n\
             \x20       {do_arr}[qr * 33u + qc] = dov;\n\
             \x20     }}\n"
        );
    }
    s.push_str(
        "      for (var i = lid; i < 1024u; i += 256u) {\n\
         \x20       let vr = i >> 5u; let vc = i & 31u;\n\
         \x20       let gr = k0 + vr; let gc = d0 + vc;\n\
         \x20       var vv = 0.0;\n\
         \x20       if (gr < sk && gc < d) { vv = f32(V[kv_base + gr * d + gc]); }\n\
         \x20       sh_kv[vr * 33u + vc] = vv;\n\
         \x20     }\n\
         \x20     workgroupBarrier();\n\
         \x20     let dk_end = min(32u, d - d0);\n\
         \x20     let v_off = tx * 33u;\n",
    );
    let do_arr = if cache_q { "sh_do" } else { "sh_q" };
    if cache_do {
        s.push_str("      let do0_off = r0 * 65u + d0; let do1_off = r1 * 65u + d0;\n");
    } else {
        s.push_str("      let do0_off = r0 * 33u; let do1_off = r1 * 33u;\n");
    }
    let _ = write!(
        s,
        "      for (var dk = 0u; dk < dk_end; dk += 1u) {{\n\
         \x20       let vv = sh_kv[v_off + dk];\n\
         \x20       dp0 = fma({do_arr}[do0_off + dk], vv, dp0);\n\
         \x20       dp1 = fma({do_arr}[do1_off + dk], vv, dp1);\n\
         \x20     }}\n\
         \x20     workgroupBarrier();\n\
         \x20   }}\n\
         \x20   let k_col = k0 + tx;\n"
    );

    if causal {
        s.push_str(
            "    var v0 = q_row0 < sq && k_col < sk && k_col <= q_row0;\n\
             \x20   var v1 = q_row1 < sq && k_col < sk && k_col <= q_row1;\n",
        );
    } else {
        s.push_str(
            "    var v0 = q_row0 < sq && k_col < sk;\n\
             \x20   var v1 = q_row1 < sq && k_col < sk;\n",
        );
    }
    for (r_var, q_var, v_var, s_var, dp_var, lse_var, del_var) in [
        ("r0", "q_row0", "v0", "s0", "dp0", "lse0", "delta0"),
        ("r1", "q_row1", "v1", "s1", "dp1", "lse1", "delta1"),
    ] {
        let _ = write!(
            s,
            "    {{\n\
             \x20     var ds = 0.0;\n\
             \x20     if ({v_var}) {{\n"
        );
        if has_mask {
            let _ = write!(
                s,
                "       let mv = MASK[mask_base + {q_var} * sk + k_col];\n\
                 \x20       if (mv == neg_inf) {{\n\
                 \x20         {v_var} = false;\n\
                 \x20       }} else if ({lse_var} != neg_inf) {{\n\
                 \x20         let p = exp({s_var} * scale + mv - {lse_var});\n\
                 \x20         ds = p * ({dp_var} - {del_var}) * scale;\n\
                 \x20       }}\n"
            );
        } else {
            let _ = write!(
                s,
                "       if ({lse_var} != neg_inf) {{\n\
                 \x20         let p = exp({s_var} * scale - {lse_var});\n\
                 \x20         ds = p * ({dp_var} - {del_var}) * scale;\n\
                 \x20       }}\n"
            );
        }
        let _ = writeln!(s, "      }}\n      sh_ds[{r_var} * 32u + tx] = ds;");
        if has_mask {
            let _ = writeln!(s, "      sh_valid[{r_var} * 32u + tx] = select(0u, 1u, {v_var});");
        }
        s.push_str("    }\n");
    }

    s.push_str("    workgroupBarrier();\n    let k_len = min(32u, sk - k0);\n");

    for c in 0..d_chunks {
        let off = c * 32;
        let _ = write!(
            s,
            "    {{\n\
             \x20     let d_col0 = d_out_base + {off}u;\n\
             \x20     if (d_col0 < d) {{\n\
             \x20       for (var i = lid; i < 1024u; i += 256u) {{\n\
             \x20         let kr = i >> 5u; let kc = i & 31u;\n\
             \x20         let gr = k0 + kr; let gc = d_col0 + kc;\n\
             \x20         var kv = 0.0;\n\
             \x20         if (gr < sk && gc < d) {{ kv = f32(K[kv_base + gr * d + gc]); }}\n\
             \x20         sh_kv[kr * 33u + kc] = kv;\n\
             \x20       }}\n\
             \x20       workgroupBarrier();\n\
             \x20       for (var kc = 0u; kc < k_len; kc += 1u) {{\n\
             \x20         let kv = sh_kv[kc * 33u + tx];\n"
        );
        if has_mask {
            let _ = write!(
                s,
                "         if (sh_valid[r0 * 32u + kc] != 0u) {{ dq0_{c} = fma(sh_ds[r0 * 32u + kc], kv, dq0_{c}); }}\n\
                 \x20         if (sh_valid[r1 * 32u + kc] != 0u) {{ dq1_{c} = fma(sh_ds[r1 * 32u + kc], kv, dq1_{c}); }}\n"
            );
        } else if causal {
            let _ = write!(
                s,
                "         if (k0 + kc <= q_row0) {{ dq0_{c} = fma(sh_ds[r0 * 32u + kc], kv, dq0_{c}); }}\n\
                 \x20         if (k0 + kc <= q_row1) {{ dq1_{c} = fma(sh_ds[r1 * 32u + kc], kv, dq1_{c}); }}\n"
            );
        } else {
            let _ = write!(
                s,
                "         dq0_{c} = fma(sh_ds[r0 * 32u + kc], kv, dq0_{c});\n\
                 \x20         dq1_{c} = fma(sh_ds[r1 * 32u + kc], kv, dq1_{c});\n"
            );
        }
        s.push_str("        }\n        workgroupBarrier();\n      }\n    }\n");
    }
    s.push_str("  }\n");

    for c in 0..d_chunks {
        let off = c * 32;
        let _ = write!(
            s,
            "  {{\n\
             \x20   let d_col = d_out_base + {off}u + tx;\n\
             \x20   if (d_col < d) {{\n\
             \x20     if (q_row0 < sq) {{ DQ[q_base + q_row0 * d + d_col] = T(dq0_{c}); }}\n\
             \x20     if (q_row1 < sq) {{ DQ[q_base + q_row1 * d + d_col] = T(dq1_{c}); }}\n\
             \x20   }}\n\
             \x20 }}\n"
        );
    }
    s.push_str("}\n");
    s
}

use Access::{Read as R, ReadWrite as RW};

pub(super) static SDPA_BWD_DELTA: KernelDef = KernelDef {
    name: "sdpa_bwd_delta",
    storage: &[R, R, R, RW],
    source: Source::Generated(gen_sdpa_bwd_delta),
};
pub(super) static SDPA_BWD_DKDV: KernelDef = KernelDef {
    name: "sdpa_bwd_dkdv",
    storage: &[R, R, R, R, R, R, RW, RW],
    source: Source::Generated(gen_sdpa_bwd_dkdv),
};
pub(super) static SDPA_BWD_DQ: KernelDef = KernelDef {
    name: "sdpa_bwd_dq",
    storage: &[R, R, R, R, R, R, RW],
    source: Source::Generated(gen_sdpa_bwd_dq),
};

/// Tiled fused Scaled Dot-Product Attention backward: computes `dQ [b, sq, d]`, `dK [b, sk, d]`, and `dV [b, sk, d]`
/// matching [`super::sdpa_forward_masked`].
///
/// Requires the saved forward output `o [b, sq, d]` and per-query log-sum-exp `lse [b, sq]` (`F32`), plus upstream
/// gradient `do_grad [b, sq, d]`. Optional additive `mask` and `causal` must match the forward pass.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_backward(
    ctx: &GpuContext,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    o: &wgpu::Buffer,
    lse: &wgpu::Buffer,
    do_grad: &wgpu::Buffer,
    mask: Option<SdpaMask<'_>>,
    dq: &wgpu::Buffer,
    dk: &wgpu::Buffer,
    dv: &wgpu::Buffer,
    (b, sq, sk, d): (usize, usize, usize, usize),
    scale: f32,
    causal: bool,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "sdpa_backward")?;
    let q_elems = prod(&[b, sq, d])?;
    let kv_elems = prod(&[b, sk, d])?;
    let q_rows = prod(&[b, sq])?;
    check_buf(q, q_elems, dtype, "sdpa_backward q")?;
    check_buf(k, kv_elems, dtype, "sdpa_backward k")?;
    check_buf(v, kv_elems, dtype, "sdpa_backward v")?;
    check_buf(o, q_elems, dtype, "sdpa_backward o")?;
    check_buf(lse, q_rows, DType::F32, "sdpa_backward lse")?;
    check_buf(do_grad, q_elems, dtype, "sdpa_backward do")?;
    check_buf(dq, q_elems, dtype, "sdpa_backward dq")?;
    check_buf(dk, kv_elems, dtype, "sdpa_backward dk")?;
    check_buf(dv, kv_elems, dtype, "sdpa_backward dv")?;
    let (mask_buf, mask_b_stride) = check_mask(ctx, mask, (b, sq, sk), "sdpa_backward")?;

    if b == 0 || d == 0 {
        return Ok(());
    }
    if sq == 0 {
        if kv_elems > 0 {
            ctx.clear(dk, dtype.bytes(kv_elems));
            ctx.clear(dv, dtype.bytes(kv_elems));
        }
        return Ok(());
    }
    if sk == 0 {
        return Err(shape_err("sdpa_backward: empty key/value sequence (sk == 0)"));
    }
    if q_elems > u32::MAX as usize || kv_elems > u32::MAX as usize {
        return Err(GpuError::Limit("sdpa_backward: tensor exceeds u32 indexing".into()));
    }

    let sg = use_subgroups(ctx);
    let half = dtype == DType::F16;
    let d_tiles = d.div_ceil(BD_TILE);
    let d_chunks = d.min(BD_TILE).div_ceil(32) as u32;
    let q_tiles = sq.div_ceil(BR);
    let k_tiles = sk.div_ceil(BC_BWD);
    let q_blocks = prod(&[b, d_tiles, q_tiles])?;
    let k_blocks = prod(&[b, d_tiles, k_tiles])?;
    let delta_blocks = q_rows.div_ceil(8);
    if q_blocks > u32::MAX as usize || k_blocks > u32::MAX as usize || delta_blocks > u32::MAX as usize {
        return Err(GpuError::Limit("sdpa_backward: dispatch exceeds u32 block count".into()));
    }

    // 1. Precompute packed `STATS[b, sq] = vec2(lse, dot(dO, O))`.
    let stats_bytes = prod(&[q_rows, 8])? as u64;
    let stats_buf = ctx.alloc(stats_bytes)?;
    let k_delta = ctx.kernel(&SDPA_BWD_DELTA, Spec::new().f16(half).subgroups(sg))?;
    let p_delta = Params::new().uz(q_rows)?.uz(d)?;
    ctx.dispatch(
        &k_delta,
        &[o, do_grad, lse, stats_buf.raw()],
        &p_delta,
        ctx.grid_1d(delta_blocks as u64 * 256, 256)?,
    )?;

    // 2. Tiled `dK` and `dV` pass.
    let k_dkdv = ctx.kernel(
        &SDPA_BWD_DKDV,
        Spec::new()
            .bool("CAUSAL", causal)
            .bool("HAS_MASK", mask.is_some())
            .u32("D_CHUNKS", d_chunks)
            .f16(half)
            .subgroups(sg),
    )?;
    let p_dkdv = Params::new()
        .uz(k_blocks)?
        .uz(sq)?
        .uz(sk)?
        .uz(d)?
        .uz(k_tiles)?
        .uz(d_tiles)?
        .f(scale)
        .f(f32::NEG_INFINITY)
        .uz(mask_b_stride)?;
    ctx.dispatch(
        &k_dkdv,
        &[q, k, v, do_grad, stats_buf.raw(), mask_buf, dk, dv],
        &p_dkdv,
        ctx.grid_1d(k_blocks as u64 * 256, 256)?,
    )?;

    // 3. Tiled `dQ` pass.
    let k_dq = ctx.kernel(
        &SDPA_BWD_DQ,
        Spec::new()
            .bool("CAUSAL", causal)
            .bool("HAS_MASK", mask.is_some())
            .bool("CACHE_Q", d <= CACHE_Q_MAX_D)
            .u32("D_CHUNKS", d_chunks)
            .f16(half)
            .subgroups(sg),
    )?;
    let p_dq = Params::new()
        .uz(q_blocks)?
        .uz(sq)?
        .uz(sk)?
        .uz(d)?
        .uz(q_tiles)?
        .uz(d_tiles)?
        .f(scale)
        .f(f32::NEG_INFINITY)
        .uz(mask_b_stride)?;
    ctx.dispatch(
        &k_dq,
        &[q, k, v, do_grad, stats_buf.raw(), mask_buf, dq],
        &p_dq,
        ctx.grid_1d(q_blocks as u64 * 256, 256)?,
    )
}
