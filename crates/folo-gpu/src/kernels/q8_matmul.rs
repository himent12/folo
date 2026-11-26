//! WGSL packed int8 weight-dequantized matmul (`W8A32` / `W8A16`) with fused bias + activation epilogue.
//!
//! Computes `Y[b, m, n] = act(sum_{k=0..K-1} A[b, m, k] * (i8_weight[n, k] * scale[n, k / group_size]) + bias[n])`:
//! - `A`: dense `[batch, M, K]` (`F32` or `F16`)
//! - `W_q8`: `[N, ceil(K / 4)]` `u32` words packing 4 little-endian signed `i8` weights per word (trailing padding
//!   bytes beyond `K` are masked/zeroed), reducing weight memory traffic 4× vs `F32` (2× vs `F16`).
//! - `scales`: `F32` per-channel `[N]` (`group_size == 0`) or per-group `[N, ceil(K / group_size)]`
//!   (`group_size > 0`, multiple of 4).
//! - Optional `bias`: `[N]` (`dtype`) and fused activation (`UnaryOp`, e.g. `Identity`, `Silu`, `Gelu`, `Relu`).
//!
//! Dispatches two specialized WGSL paths:
//! 1. `q8_gemv`: for `M == 1` (vector-matrix decode), 8 output channels per 256-thread workgroup (32 lanes per
//!    channel) with 1-instruction `subgroupAdd` (or portable shared-memory tree fallback).
//! 2. `q8_matmul`: for `M > 1`, `64 × 64 × 32` shared-memory tiled GEMM (16 KiB workgroup storage, 4×4 register
//!    accumulators per thread) that dequantizes each packed `u32` weight word once per `BK` tile and reuses it
//!    across all 64 rows of `A`.

use super::{UNARY_FN, check_buf, prod, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::elementwise::UnaryOp;
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Descriptor for a packed int8 weight-dequantized linear / matmul dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8MatmulDesc {
    /// Outer batch size on `A` and `Y` (`W_q8`, `scales`, and `bias` are shared across batches).
    pub batch: usize,
    /// Rows of `A` and `Y` per batch.
    pub m: usize,
    /// Output features (`W_q8` rows / `Y` columns).
    pub n: usize,
    /// Inner reduction dimension.
    pub k: usize,
    /// Quantization group size along `K`: `0` for per-channel `[N]` scales, or a positive multiple of 4 for
    /// per-group `[N, ceil(K / group_size)]` scales.
    pub group_size: usize,
    /// Fused activation applied after bias addition.
    pub act: UnaryOp,
    /// Storage dtype of `A`, optional `bias`, and `Y` (`F32` or `F16`; `scales` and accumulation are `F32`).
    pub dtype: DType,
}

impl Q8MatmulDesc {
    /// Per-channel symmetric int8 matmul `[batch, m, k] @ [n, k]^T`.
    pub fn per_channel(batch: usize, m: usize, n: usize, k: usize, dtype: DType) -> Self {
        Self { batch, m, n, k, group_size: 0, act: UnaryOp::Identity, dtype }
    }

    /// Number of packed `u32` words per row of `W_q8` (`ceil(K / 4)`).
    pub fn k_packed(&self) -> usize {
        self.k.div_ceil(4)
    }

    /// Number of scale entries per output channel (`1` when `group_size == 0`, else `ceil(K / group_size)`).
    pub fn num_groups(&self) -> usize {
        if self.group_size == 0 { 1 } else { self.k.div_ceil(self.group_size).max(1) }
    }
}

/// Packs row-major `[n, k]` signed `i8` weights into `[n, ceil(k / 4)]` little-endian `u32` words.
/// Trailing padding lanes in the final word of each row are set to `0`.
pub fn pack_i8_weights(weights: &[i8], n: usize, k: usize) -> Vec<u32> {
    assert_eq!(weights.len(), n * k, "pack_i8_weights: length mismatch");
    let kp = k.div_ceil(4);
    let mut out = vec![0u32; n * kp];
    for row in 0..n {
        let src = &weights[row * k..(row + 1) * k];
        let dst = &mut out[row * kp..(row + 1) * kp];
        for (i, &w) in src.iter().enumerate() {
            dst[i / 4] |= (w as u8 as u32) << ((i % 4) * 8);
        }
    }
    out
}

/// Shared WGSL signed-byte unpacking helper (exact arithmetic shift sign-extension on all backends).
const S8_UNPACK: &str = r#"
fn unpack_s8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(bitcast<i32>(w << 24u) >> 24u),
        f32(bitcast<i32>(w << 16u) >> 24u),
        f32(bitcast<i32>(w << 8u) >> 24u),
        f32(bitcast<i32>(w) >> 24u),
    );
}
"#;

/// `64 × 64 × 32` shared-memory tiled int8-weight matmul (`16 × 16 = 256` threads, `4 × 4` per thread,
/// 16 KiB workgroup storage).
/// Spec: `ACT`, `HAS_BIAS`, `GROUPED`, `f16`.
/// Params: 0 = `M`, 1 = `N`, 2 = `K`, 3 = `K_packed`, 4 = `num_groups`, 5 = `group_words` (`group_size / 4`),
/// 6 = `act_alpha` (f32).
fn gen_q8_matmul(spec: &Spec) -> String {
    let grouped = spec.get_bool("GROUPED");
    let scale_expr =
        if grouped { "SCALES[bn_g * num_groups + (k_word / group_words)]" } else { "SCALES[bn_g]" };
    format!(
        "{UNARY_FN}{S8_UNPACK}\n\
         @group(0) @binding(1) var<storage, read> A: array<T>;\n\
         @group(0) @binding(2) var<storage, read> W: array<u32>;\n\
         @group(0) @binding(3) var<storage, read> SCALES: array<f32>;\n\
         @group(0) @binding(4) var<storage, read> BIAS: array<T>;\n\
         @group(0) @binding(5) var<storage, read_write> Y: array<T>;\n\
         var<workgroup> As: array<array<f32, 64>, 32>;\n\
         var<workgroup> Bs: array<array<f32, 64>, 32>;\n\
         @compute @workgroup_size(16, 16, 1)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
             let M = pu(0u); let N = pu(1u); let K = pu(2u);\n\
             let kp = pu(3u); let num_groups = pu(4u); let group_words = pu(5u);\n\
             let act_alpha = pf(6u);\n\
             let tx = lid.x; let ty = lid.y; let tid = ty * 16u + tx;\n\
             let m0 = wid.y * 64u; let n0 = wid.x * 64u; let batch = wid.z;\n\
             let offA = batch * M * K;\n\
             let offY = batch * M * N;\n\
             var acc0 = vec4<f32>(0.0);\n\
             var acc1 = vec4<f32>(0.0);\n\
             var acc2 = vec4<f32>(0.0);\n\
             var acc3 = vec4<f32>(0.0);\n\
             let a_row = tid >> 2u;\n\
             let a_w = tid & 3u;\n\
             let b_row = tid >> 2u;\n\
             let b_w = tid & 3u;\n\
             let n_tiles = (K + 31u) / 32u;\n\
             for (var t = 0u; t < n_tiles; t += 1u) {{\n\
                 let k0 = t * 32u;\n\
                 let kw0 = t * 8u;\n\
                 for (var step = 0u; step < 2u; step += 1u) {{\n\
                     let c4 = a_w + step * 4u;\n\
                     let col0 = k0 + c4 * 4u;\n\
                     let am_g = m0 + a_row;\n\
                     var v0 = 0.0; var v1 = 0.0; var v2 = 0.0; var v3 = 0.0;\n\
                     if (am_g < M) {{\n\
                         let base_a = offA + am_g * K + col0;\n\
                         if (col0 < K) {{ v0 = f32(A[base_a]); }}\n\
                         if (col0 + 1u < K) {{ v1 = f32(A[base_a + 1u]); }}\n\
                         if (col0 + 2u < K) {{ v2 = f32(A[base_a + 2u]); }}\n\
                         if (col0 + 3u < K) {{ v3 = f32(A[base_a + 3u]); }}\n\
                     }}\n\
                     As[c4 * 4u][a_row] = v0;\n\
                     As[c4 * 4u + 1u][a_row] = v1;\n\
                     As[c4 * 4u + 2u][a_row] = v2;\n\
                     As[c4 * 4u + 3u][a_row] = v3;\n\
                     let k_word = kw0 + b_w + step * 4u;\n\
                     let bn_g = n0 + b_row;\n\
                     var dq = vec4<f32>(0.0);\n\
                     if (bn_g < N && k_word < kp) {{\n\
                         let sc = {scale_expr};\n\
                         dq = unpack_s8x4(W[bn_g * kp + k_word]) * sc;\n\
                         let k_base = k_word * 4u;\n\
                         if (k_base + 1u >= K) {{ dq.y = 0.0; }}\n\
                         if (k_base + 2u >= K) {{ dq.z = 0.0; }}\n\
                         if (k_base + 3u >= K) {{ dq.w = 0.0; }}\n\
                     }}\n\
                     let bk_off = (b_w + step * 4u) * 4u;\n\
                     Bs[bk_off][b_row] = dq.x;\n\
                     Bs[bk_off + 1u][b_row] = dq.y;\n\
                     Bs[bk_off + 2u][b_row] = dq.z;\n\
                     Bs[bk_off + 3u][b_row] = dq.w;\n\
                 }}\n\
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
                 if (t + 1u < n_tiles) {{ workgroupBarrier(); }}\n\
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

/// Vector-matrix int8 dequant GEMV (`M == 1`): 8 output channels per 256-thread workgroup (32 lanes per channel),
/// with 1-instruction `subgroupAdd` on 32-wide subgroup hardware and a portable shared-memory tree fallback.
/// Spec: `ACT`, `HAS_BIAS`, `GROUPED`, `f16`, `subgroups`.
/// Params: 0 = `N`, 1 = `K`, 2 = `K_packed`, 3 = `num_groups`, 4 = `group_words`, 5 = `act_alpha` (f32).
fn gen_q8_gemv(spec: &Spec) -> String {
    let grouped = spec.get_bool("GROUPED");
    let sg = spec.is_subgroups();
    let scale_expr =
        if grouped { "SCALES[out_n * num_groups + (kw / group_words)]" } else { "SCALES[out_n]" };
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
        "{UNARY_FN}{S8_UNPACK}\n\
         @group(0) @binding(1) var<storage, read> A: array<T>;\n\
         @group(0) @binding(2) var<storage, read> W: array<u32>;\n\
         @group(0) @binding(3) var<storage, read> SCALES: array<f32>;\n\
         @group(0) @binding(4) var<storage, read> BIAS: array<T>;\n\
         @group(0) @binding(5) var<storage, read_write> Y: array<T>;\n\
         {sh_decl}\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{\n\
             let N = pu(0u); let K = pu(1u); let kp = pu(2u);\n\
             let num_groups = pu(3u); let group_words = pu(4u);\n\
             let act_alpha = pf(5u);\n\
             let batch = wid.z;\n\
             let lane = lid & 31u;\n\
             let row_in_wg = lid >> 5u;\n\
             let out_n = wid.x * 8u + row_in_wg;\n\
             let valid = out_n < N;\n\
             var acc = 0.0;\n\
             if (valid) {{\n\
                 let offA = batch * K;\n\
                 let wbase = out_n * kp;\n\
                 for (var kw = lane; kw < kp; kw += 32u) {{\n\
                     let k0 = kw * 4u;\n\
                     var av = vec4<f32>(0.0);\n\
                     if (k0 < K) {{ av.x = f32(A[offA + k0]); }}\n\
                     if (k0 + 1u < K) {{ av.y = f32(A[offA + k0 + 1u]); }}\n\
                     if (k0 + 2u < K) {{ av.z = f32(A[offA + k0 + 2u]); }}\n\
                     if (k0 + 3u < K) {{ av.w = f32(A[offA + k0 + 3u]); }}\n\
                     let dq = unpack_s8x4(W[wbase + kw]) * {scale_expr};\n\
                     acc += dot(av, dq);\n\
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

static Q8_MATMUL: KernelDef =
    KernelDef { name: "q8_matmul", storage: &[R, R, R, R, RW], source: Source::Generated(gen_q8_matmul) };
static Q8_GEMV: KernelDef =
    KernelDef { name: "q8_gemv", storage: &[R, R, R, R, RW], source: Source::Generated(gen_q8_gemv) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&Q8_MATMUL, &Q8_GEMV];

/// Dispatches packed int8 weight-dequantized matmul `Y = act(A @ dequant(W_q8, scales)^T + bias)`.
#[allow(clippy::too_many_arguments)]
pub fn q8_matmul(
    ctx: &GpuContext,
    d: &Q8MatmulDesc,
    a: &wgpu::Buffer,
    w_q8: &wgpu::Buffer,
    scales: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "q8_matmul")?;
    if d.group_size != 0 && !d.group_size.is_multiple_of(4) {
        return Err(shape_err(format!(
            "q8_matmul: group_size ({}) must be 0 (per-channel) or a multiple of 4",
            d.group_size
        )));
    }
    let a_elems = prod(&[d.batch, d.m, d.k])?;
    let y_elems = prod(&[d.batch, d.m, d.n])?;
    let kp = d.k_packed();
    let num_groups = d.num_groups();
    let w_words = prod(&[d.n, kp])?;
    let scale_elems = prod(&[d.n, num_groups])?;
    check_buf(a, a_elems, d.dtype, "q8_matmul A")?;
    check_buf(w_q8, w_words, DType::F32, "q8_matmul W_q8")?;
    check_buf(scales, scale_elems, DType::F32, "q8_matmul scales")?;
    if let Some(b) = bias {
        check_buf(b, d.n, d.dtype, "q8_matmul bias")?;
    }
    check_buf(y, y_elems, d.dtype, "q8_matmul Y")?;
    if y_elems == 0 {
        return Ok(());
    }
    if a_elems > u32::MAX as usize
        || y_elems > u32::MAX as usize
        || w_words > u32::MAX as usize
        || scale_elems > u32::MAX as usize
        || d.batch > ctx.caps().max_workgroups_per_dim as usize
    {
        return Err(GpuError::Limit("q8_matmul: dimensions exceed dispatch limits".into()));
    }
    let grouped = d.group_size > 0;
    let group_words = if grouped { d.group_size / 4 } else { 1 };
    let bias_buf = bias.unwrap_or(ctx.dummy());

    if d.m == 1 {
        let gx = d.n.div_ceil(8);
        if gx > ctx.caps().max_workgroups_per_dim as usize {
            return Err(GpuError::Limit("q8_matmul: N exceeds dispatch limits".into()));
        }
        let k = ctx.kernel(
            &Q8_GEMV,
            Spec::new()
                .u32("ACT", d.act.code())
                .bool("HAS_BIAS", bias.is_some())
                .bool("GROUPED", grouped)
                .f16(d.dtype == DType::F16)
                .subgroups(use_subgroups(ctx)),
        )?;
        let p = Params::new().uz(d.n)?.uz(d.k)?.uz(kp)?.uz(num_groups)?.uz(group_words)?.f(d.act.alpha());
        return ctx.dispatch(&k, &[a, w_q8, scales, bias_buf, y], &p, [gx as u32, 1, d.batch as u32]);
    }

    let gx = d.n.div_ceil(64);
    let gy = d.m.div_ceil(64);
    let max_wg = ctx.caps().max_workgroups_per_dim as usize;
    if gx > max_wg || gy > max_wg {
        return Err(GpuError::Limit("q8_matmul: M or N exceeds dispatch limits".into()));
    }
    let k = ctx.kernel(
        &Q8_MATMUL,
        Spec::new()
            .u32("ACT", d.act.code())
            .bool("HAS_BIAS", bias.is_some())
            .bool("GROUPED", grouped)
            .f16(d.dtype == DType::F16),
    )?;
    let p =
        Params::new().uz(d.m)?.uz(d.n)?.uz(d.k)?.uz(kp)?.uz(num_groups)?.uz(group_words)?.f(d.act.alpha());
    ctx.dispatch(&k, &[a, w_q8, scales, bias_buf, y], &p, [gx as u32, gy as u32, d.batch as u32])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_gemv_portable_emits_no_subgroup_tokens() {
        let spec = Spec::new().u32("ACT", 8).bool("HAS_BIAS", true).bool("GROUPED", true).subgroups(false);
        let wgsl = crate::pipeline::render(&Q8_GEMV, &spec);
        assert!(!wgsl.contains("subgroup"), "portable q8_gemv emitted subgroup token:\n{wgsl}");
    }
}
