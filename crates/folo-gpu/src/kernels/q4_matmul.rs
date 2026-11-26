//! WGSL symmetric 4-bit block-dequantizing matmul (`W4A32` / `W4A16`) matching `folo_core::cpu::q4::Q4Block`.
//!
//! Each `Q4Block` stores 32 signed 4-bit weights `q ∈ [-8, 7]` in 5 packed `u32` words (20 bytes):
//! - word `0`: `f32` block scale (`bitcast<f32>(w0)`)
//! - words `1..=4`: 16 packed bytes `qs[0..16]` in little-endian order, where byte `qs[j]` holds `(q[j] + 8)` in
//!   its low nibble (bits `0..4`) and `(q[j + 16] + 8)` in its high nibble (bits `4..8`).
//!
//! Computes `Y[b, m, n] = act(sum_{k=0..K-1} A[b, m, k] * dequant(W_q4[n, k]) + bias[n])` for arbitrary
//! `(batch, M, N, K)` including non-tile-aligned tails:
//! - `q4_gemv` (`M == 1`): 8 output channels per 256-thread workgroup (32 lanes per channel) with 1-instruction
//!   `subgroupAdd` on 32-wide subgroup hardware and a portable shared-memory tree fallback.
//! - `q4_matmul` (`M > 1`): `64 × 64 × 32` shared-memory tiled GEMM (16 KiB workgroup storage, `4 × 4` register
//!   tile per thread) where each `BK = 32` tile maps 1-to-1 to a single `Q4Block` per output row, unpacking 8
//!   weights per thread per tile and reusing them across all 64 rows of `A`.

use folo_core::cpu::q4::{BLOCK_SIZE as Q4_BLOCK_SIZE, Q4Block, Q4Matrix};

use super::{UNARY_FN, check_buf, prod, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::elementwise::UnaryOp;
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Number of `u32` words stored per [`Q4Block`] (`1` scale word + `4` nibble words = 20 bytes).
pub const Q4_BLOCK_WORDS: usize = 5;

/// Descriptor for a `Q4Block` weight-dequantized linear / matmul dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4MatmulDesc {
    /// Outer batch size on `A` and `Y` (`W_q4` and `bias` are shared across batches).
    pub batch: usize,
    /// Rows of `A` and `Y` per batch.
    pub m: usize,
    /// Output features (`W_q4` rows / `Y` columns).
    pub n: usize,
    /// Inner reduction dimension (need not be a multiple of 32).
    pub k: usize,
    /// Fused activation applied after bias addition.
    pub act: UnaryOp,
    /// Storage dtype of `A`, optional `bias`, and `Y` (`F32` or `F16`; block scales and accumulation are `F32`).
    pub dtype: DType,
}

impl Q4MatmulDesc {
    /// Matmul `[batch, m, k] @ dequant(W_q4[n, k])^T` with identity activation.
    pub fn new(batch: usize, m: usize, n: usize, k: usize, dtype: DType) -> Self {
        Self { batch, m, n, k, act: UnaryOp::Identity, dtype }
    }

    /// Number of [`Q4Block`]s stored per row of `W_q4` (`ceil(K / 32)`).
    pub fn blocks_per_row(&self) -> usize {
        self.k.div_ceil(Q4_BLOCK_SIZE)
    }

    /// Total `u32` words per row of `W_q4` (`blocks_per_row * 5`).
    pub fn words_per_row(&self) -> usize {
        self.blocks_per_row() * Q4_BLOCK_WORDS
    }
}

/// Packs a slice of [`Q4Block`]s into `[blocks.len() * 5]` `u32` words for [`q4_matmul`].
pub fn pack_q4_blocks(blocks: &[Q4Block]) -> Vec<u32> {
    let mut out = Vec::with_capacity(blocks.len() * Q4_BLOCK_WORDS);
    for blk in blocks {
        out.push(blk.scale.to_bits());
        for &chunk in blk.qs.as_chunks::<4>().0 {
            out.push(u32::from_le_bytes(chunk));
        }
    }
    out
}

/// Packs a [`Q4Matrix`] into `[mat.rows * mat.blocks_per_row * 5]` `u32` words for [`q4_matmul`].
pub fn pack_q4_matrix(mat: &Q4Matrix) -> Vec<u32> {
    pack_q4_blocks(&mat.blocks)
}

/// Shared WGSL biased-nibble unpacking helpers matching `Q4Block`'s split-half layout.
const Q4_UNPACK: &str = r#"
fn unpack_q4_lo(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(w & 0xfu) - 8),
        f32(i32((w >> 8u) & 0xfu) - 8),
        f32(i32((w >> 16u) & 0xfu) - 8),
        f32(i32((w >> 24u) & 0xfu) - 8),
    );
}
fn unpack_q4_hi(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32((w >> 4u) & 0xfu) - 8),
        f32(i32((w >> 12u) & 0xfu) - 8),
        f32(i32((w >> 20u) & 0xfu) - 8),
        f32(i32((w >> 28u) & 0xfu) - 8),
    );
}
"#;

/// `64 × 64 × 32` shared-memory tiled Q4-weight matmul (`16 × 16 = 256` threads, `4 × 4` register tile per
/// thread, 16 KiB workgroup storage). Each `BK = 32` tile matches one [`Q4Block`] per output channel.
/// Spec: `ACT`, `HAS_BIAS`, `f16`.
/// Params: 0 = `M`, 1 = `N`, 2 = `K`, 3 = `blocks_per_row`, 4 = `act_alpha` (f32).
fn gen_q4_matmul(_spec: &Spec) -> String {
    format!(
        "{UNARY_FN}{Q4_UNPACK}\n\
         @group(0) @binding(1) var<storage, read> A: array<T>;\n\
         @group(0) @binding(2) var<storage, read> W: array<u32>;\n\
         @group(0) @binding(3) var<storage, read> BIAS: array<T>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<T>;\n\
         var<workgroup> As: array<array<f32, 64>, 32>;\n\
         var<workgroup> Bs: array<array<f32, 64>, 32>;\n\
         @compute @workgroup_size(16, 16, 1)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
             let M = pu(0u); let N = pu(1u); let K = pu(2u);\n\
             let bpr = pu(3u); let act_alpha = pf(4u);\n\
             let tx = lid.x; let ty = lid.y; let tid = ty * 16u + tx;\n\
             let m0 = wid.y * 64u; let n0 = wid.x * 64u; let batch = wid.z;\n\
             let offA = batch * M * K;\n\
             let offY = batch * M * N;\n\
             var acc0 = vec4<f32>(0.0);\n\
             var acc1 = vec4<f32>(0.0);\n\
             var acc2 = vec4<f32>(0.0);\n\
             var acc3 = vec4<f32>(0.0);\n\
             let row_idx = tid >> 2u;\n\
             let w_idx = tid & 3u;\n\
             for (var blk = 0u; blk < bpr; blk += 1u) {{\n\
                 let k0 = blk * 32u;\n\
                 // Load 64 × 32 tile of A (2 steps × 4 elements per thread).\n\
                 let am_g = m0 + row_idx;\n\
                 for (var step = 0u; step < 2u; step += 1u) {{\n\
                     let c4 = w_idx + step * 4u;\n\
                     let col0 = k0 + c4 * 4u;\n\
                     var v0 = 0.0; var v1 = 0.0; var v2 = 0.0; var v3 = 0.0;\n\
                     if (am_g < M) {{\n\
                         let base_a = offA + am_g * K + col0;\n\
                         if (col0 < K) {{ v0 = f32(A[base_a]); }}\n\
                         if (col0 + 1u < K) {{ v1 = f32(A[base_a + 1u]); }}\n\
                         if (col0 + 2u < K) {{ v2 = f32(A[base_a + 2u]); }}\n\
                         if (col0 + 3u < K) {{ v3 = f32(A[base_a + 3u]); }}\n\
                     }}\n\
                     As[c4 * 4u][row_idx] = v0;\n\
                     As[c4 * 4u + 1u][row_idx] = v1;\n\
                     As[c4 * 4u + 2u][row_idx] = v2;\n\
                     As[c4 * 4u + 3u][row_idx] = v3;\n\
                 }}\n\
                 // Load 64 × 32 tile of Q4 weights: 4 threads per row unpack 8 weights each (4 low + 4 high).\n\
                 let bn_g = n0 + row_idx;\n\
                 var lo = vec4<f32>(0.0);\n\
                 var hi = vec4<f32>(0.0);\n\
                 if (bn_g < N) {{\n\
                     let blk_base = (bn_g * bpr + blk) * 5u;\n\
                     let sc = bitcast<f32>(W[blk_base]);\n\
                     let qw = W[blk_base + 1u + w_idx];\n\
                     lo = unpack_q4_lo(qw) * sc;\n\
                     hi = unpack_q4_hi(qw) * sc;\n\
                     let klo = k0 + w_idx * 4u;\n\
                     if (klo >= K) {{ lo.x = 0.0; }}\n\
                     if (klo + 1u >= K) {{ lo.y = 0.0; }}\n\
                     if (klo + 2u >= K) {{ lo.z = 0.0; }}\n\
                     if (klo + 3u >= K) {{ lo.w = 0.0; }}\n\
                     let khi = klo + 16u;\n\
                     if (khi >= K) {{ hi.x = 0.0; }}\n\
                     if (khi + 1u >= K) {{ hi.y = 0.0; }}\n\
                     if (khi + 2u >= K) {{ hi.z = 0.0; }}\n\
                     if (khi + 3u >= K) {{ hi.w = 0.0; }}\n\
                 }}\n\
                 let lo_off = w_idx * 4u;\n\
                 let hi_off = 16u + lo_off;\n\
                 Bs[lo_off][row_idx] = lo.x;\n\
                 Bs[lo_off + 1u][row_idx] = lo.y;\n\
                 Bs[lo_off + 2u][row_idx] = lo.z;\n\
                 Bs[lo_off + 3u][row_idx] = lo.w;\n\
                 Bs[hi_off][row_idx] = hi.x;\n\
                 Bs[hi_off + 1u][row_idx] = hi.y;\n\
                 Bs[hi_off + 2u][row_idx] = hi.z;\n\
                 Bs[hi_off + 3u][row_idx] = hi.w;\n\
                 workgroupBarrier();\n\
                 let rm = ty * 4u;\n\
                 let cn = tx * 4u;\n\
                 for (var kk = 0u; kk < 32u; kk += 1u) {{\n\
                     let bv = vec4<f32>(Bs[kk][cn], Bs[kk][cn + 1u], Bs[kk][cn + 2u], Bs[kk][cn + 3u]);\n\
                     acc0 = fma(vec4<f32>(As[kk][rm]), bv, acc0);\n\
                     acc1 = fma(vec4<f32>(As[kk][rm + 1u]), bv, acc1);\n\
                     acc2 = fma(vec4<f32>(As[kk][rm + 2u]), bv, acc2);\n\
                     acc3 = fma(vec4<f32>(As[kk][rm + 3u]), bv, acc3);\n\
                 }}\n\
                 workgroupBarrier();\n\
             }}\n\
             let col0 = n0 + tx * 4u;\n\
             var bv_bias = vec4<f32>(0.0);\n\
             if (HAS_BIAS && N > 0u) {{\n\
                 bv_bias = vec4<f32>(\n\
                     f32(BIAS[min(col0, N - 1u)]),\n\
                     f32(BIAS[min(col0 + 1u, N - 1u)]),\n\
                     f32(BIAS[min(col0 + 2u, N - 1u)]),\n\
                     f32(BIAS[min(col0 + 3u, N - 1u)]),\n\
                 );\n\
             }}\n\
             let r0 = m0 + ty * 4u;\n\
             let v0 = unary4(ACT, acc0 + bv_bias, act_alpha);\n\
             let v1 = unary4(ACT, acc1 + bv_bias, act_alpha);\n\
             let v2 = unary4(ACT, acc2 + bv_bias, act_alpha);\n\
             let v3 = unary4(ACT, acc3 + bv_bias, act_alpha);\n\
             if (r0 < M) {{\n\
                 let base = offY + r0 * N + col0;\n\
                 if (col0 < N) {{ Y[base] = T(v0.x); }}\n\
                 if (col0 + 1u < N) {{ Y[base + 1u] = T(v0.y); }}\n\
                 if (col0 + 2u < N) {{ Y[base + 2u] = T(v0.z); }}\n\
                 if (col0 + 3u < N) {{ Y[base + 3u] = T(v0.w); }}\n\
             }}\n\
             if (r0 + 1u < M) {{\n\
                 let base = offY + (r0 + 1u) * N + col0;\n\
                 if (col0 < N) {{ Y[base] = T(v1.x); }}\n\
                 if (col0 + 1u < N) {{ Y[base + 1u] = T(v1.y); }}\n\
                 if (col0 + 2u < N) {{ Y[base + 2u] = T(v1.z); }}\n\
                 if (col0 + 3u < N) {{ Y[base + 3u] = T(v1.w); }}\n\
             }}\n\
             if (r0 + 2u < M) {{\n\
                 let base = offY + (r0 + 2u) * N + col0;\n\
                 if (col0 < N) {{ Y[base] = T(v2.x); }}\n\
                 if (col0 + 1u < N) {{ Y[base + 1u] = T(v2.y); }}\n\
                 if (col0 + 2u < N) {{ Y[base + 2u] = T(v2.z); }}\n\
                 if (col0 + 3u < N) {{ Y[base + 3u] = T(v2.w); }}\n\
             }}\n\
             if (r0 + 3u < M) {{\n\
                 let base = offY + (r0 + 3u) * N + col0;\n\
                 if (col0 < N) {{ Y[base] = T(v3.x); }}\n\
                 if (col0 + 1u < N) {{ Y[base + 1u] = T(v3.y); }}\n\
                 if (col0 + 2u < N) {{ Y[base + 2u] = T(v3.z); }}\n\
                 if (col0 + 3u < N) {{ Y[base + 3u] = T(v3.w); }}\n\
             }}\n\
         }}\n"
    )
}

/// Vector-matrix Q4 dequant GEMV (`M == 1`): 8 output channels per 256-thread workgroup (32 lanes per channel),
/// with 1-instruction `subgroupAdd` on 32-wide subgroup hardware and a portable shared-memory tree fallback.
/// Spec: `ACT`, `HAS_BIAS`, `f16`, `subgroups`.
/// Params: 0 = `N`, 1 = `K`, 2 = `blocks_per_row`, 3 = `act_alpha` (f32).
fn gen_q4_gemv(spec: &Spec) -> String {
    let sg = spec.is_subgroups();
    let (entry_sig, lane_setup) = if sg {
        (
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_id) row_in_wg: u32, @builtin(subgroup_invocation_id) lane: u32)",
            "",
        )
    } else {
        (
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32)",
            "    let lane = lid & 31u;\n    let row_in_wg = lid >> 5u;\n",
        )
    };
    let reduce_code = if sg {
        "let total = subgroupAdd(acc);\n    if (lane == 0u && valid) {\n"
    } else {
        "sh[lid] = acc;\n    workgroupBarrier();\n    \
         if (lane < 16u) { sh[lid] += sh[lid + 16u]; } workgroupBarrier();\n    \
         if (lane < 8u) { sh[lid] += sh[lid + 8u]; } workgroupBarrier();\n    \
         if (lane < 4u) { sh[lid] += sh[lid + 4u]; } workgroupBarrier();\n    \
         if (lane < 2u) { sh[lid] += sh[lid + 2u]; } workgroupBarrier();\n    \
         if (lane == 0u && valid) {\n        let total = sh[lid] + sh[lid + 1u];\n"
    };
    let sh_decl = if sg { "" } else { "var<workgroup> sh: array<f32, 256>;\n" };
    format!(
        "{UNARY_FN}{Q4_UNPACK}\n\
         @group(0) @binding(1) var<storage, read> A: array<T>;\n\
         @group(0) @binding(2) var<storage, read> W: array<u32>;\n\
         @group(0) @binding(3) var<storage, read> BIAS: array<T>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<T>;\n\
         {sh_decl}\
         @compute @workgroup_size(256)\n\
         {entry_sig} {{\n\
             let N = pu(0u); let K = pu(1u); let bpr = pu(2u); let act_alpha = pf(3u);\n\
             let batch = wid.z;\n\
         {lane_setup}\
             let out_n = wid.x * 8u + row_in_wg;\n\
             let valid = out_n < N;\n\
             var acc = 0.0;\n\
             if (valid) {{\n\
                 let offA = batch * K;\n\
                 let wrow = out_n * bpr;\n\
                 for (var blk = lane; blk < bpr; blk += 32u) {{\n\
                     let blk_base = (wrow + blk) * 5u;\n\
                     let sc = bitcast<f32>(W[blk_base]);\n\
                     let k0 = blk * 32u;\n\
                     var bsum = 0.0;\n\
                     for (var j = 0u; j < 4u; j += 1u) {{\n\
                         let qw = W[blk_base + 1u + j];\n\
                         let qlo = unpack_q4_lo(qw);\n\
                         let qhi = unpack_q4_hi(qw);\n\
                         let klo = k0 + j * 4u;\n\
                         var alo = vec4<f32>(0.0);\n\
                         if (klo < K) {{ alo.x = f32(A[offA + klo]); }}\n\
                         if (klo + 1u < K) {{ alo.y = f32(A[offA + klo + 1u]); }}\n\
                         if (klo + 2u < K) {{ alo.z = f32(A[offA + klo + 2u]); }}\n\
                         if (klo + 3u < K) {{ alo.w = f32(A[offA + klo + 3u]); }}\n\
                         let khi = klo + 16u;\n\
                         var ahi = vec4<f32>(0.0);\n\
                         if (khi < K) {{ ahi.x = f32(A[offA + khi]); }}\n\
                         if (khi + 1u < K) {{ ahi.y = f32(A[offA + khi + 1u]); }}\n\
                         if (khi + 2u < K) {{ ahi.z = f32(A[offA + khi + 2u]); }}\n\
                         if (khi + 3u < K) {{ ahi.w = f32(A[offA + khi + 3u]); }}\n\
                         bsum += dot(alo, qlo) + dot(ahi, qhi);\n\
                     }}\n\
                     acc += bsum * sc;\n\
                 }}\n\
             }}\n\
             {reduce_code}\
                 var v = total;\n\
                 if (HAS_BIAS) {{ v += f32(BIAS[out_n]); }}\n\
                 Y[batch * N + out_n] = T(unary1(ACT, v, act_alpha));\n\
             }}\n\
         }}\n"
    )
}

use Access::{Read as R, ReadWrite as RW};

static Q4_MATMUL: KernelDef =
    KernelDef { name: "q4_matmul", storage: &[R, R, R, RW], source: Source::Generated(gen_q4_matmul) };
static Q4_GEMV: KernelDef =
    KernelDef { name: "q4_gemv", storage: &[R, R, R, RW], source: Source::Generated(gen_q4_gemv) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&Q4_MATMUL, &Q4_GEMV];

/// Dispatches `Q4Block` weight-dequantized matmul `Y = act(A @ dequant(W_q4)^T + bias)`.
///
/// `w_q4` holds `d.n * d.blocks_per_row() * 5` `u32` words produced by [`pack_q4_blocks`] / [`pack_q4_matrix`].
/// `y` must not alias `a`, `w_q4`, or `bias`.
pub fn q4_matmul(
    ctx: &GpuContext,
    d: &Q4MatmulDesc,
    a: &wgpu::Buffer,
    w_q4: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "q4_matmul")?;
    let a_elems = prod(&[d.batch, d.m, d.k])?;
    let y_elems = prod(&[d.batch, d.m, d.n])?;
    let bpr = d.blocks_per_row();
    let w_words = prod(&[d.n, bpr, Q4_BLOCK_WORDS])?;
    check_buf(a, a_elems, d.dtype, "q4_matmul A")?;
    check_buf(w_q4, w_words, DType::F32, "q4_matmul W_q4")?;
    if let Some(b) = bias {
        check_buf(b, d.n, d.dtype, "q4_matmul bias")?;
    }
    check_buf(y, y_elems, d.dtype, "q4_matmul Y")?;
    if y_elems == 0 {
        return Ok(());
    }
    if std::iter::once(a).chain(std::iter::once(w_q4)).chain(bias).any(|buf| buf == y) {
        return Err(shape_err("q4_matmul: output buffer must not alias A, W_q4, or bias"));
    }
    let max_wg = ctx.caps().max_workgroups_per_dim as usize;
    if a_elems > u32::MAX as usize
        || y_elems > u32::MAX as usize
        || w_words > u32::MAX as usize
        || d.batch > max_wg
    {
        return Err(GpuError::Limit("q4_matmul: dimensions exceed dispatch limits".into()));
    }
    let bias_buf = bias.unwrap_or(ctx.dummy());

    if d.m == 1 {
        let gx = d.n.div_ceil(8);
        if gx > max_wg {
            return Err(GpuError::Limit("q4_matmul: N exceeds dispatch limits".into()));
        }
        let k = ctx.kernel(
            &Q4_GEMV,
            Spec::new()
                .u32("ACT", d.act.code())
                .bool("HAS_BIAS", bias.is_some())
                .f16(d.dtype == DType::F16)
                .subgroups(use_subgroups(ctx)),
        )?;
        let p = Params::new().uz(d.n)?.uz(d.k)?.uz(bpr)?.f(d.act.alpha());
        return ctx.dispatch(&k, &[a, w_q4, bias_buf, y], &p, [gx as u32, 1, d.batch as u32]);
    }

    let gx = d.n.div_ceil(64);
    let gy = d.m.div_ceil(64);
    if gx > max_wg || gy > max_wg {
        return Err(GpuError::Limit("q4_matmul: M or N exceeds dispatch limits".into()));
    }
    let k = ctx.kernel(
        &Q4_MATMUL,
        Spec::new().u32("ACT", d.act.code()).bool("HAS_BIAS", bias.is_some()).f16(d.dtype == DType::F16),
    )?;
    let p = Params::new().uz(d.m)?.uz(d.n)?.uz(d.k)?.uz(bpr)?.f(d.act.alpha());
    ctx.dispatch(&k, &[a, w_q4, bias_buf, y], &p, [gx as u32, gy as u32, d.batch as u32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_gemv_portable_emits_no_subgroup_tokens() {
        let spec = Spec::new().u32("ACT", 8).bool("HAS_BIAS", true).subgroups(false);
        let wgsl = crate::pipeline::render(&Q4_GEMV, &spec);
        assert!(!wgsl.contains("subgroup"), "portable q4_gemv emitted subgroup token:\n{wgsl}");
    }
}
