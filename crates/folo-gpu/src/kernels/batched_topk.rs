//! WGSL batched arbitrary-axis `topk` (`F32` and `F16`) with deterministic CPU-matching tie-breaking.
//!
//! Views a dense tensor along axis `dim` as `[outer, d, inner]` and selects the `k` largest (or smallest) values
//! and their `i64` indices along the middle axis into `[outer, k, inner]` outputs, ordered by exact IEEE total
//! order (`-NaN < -inf < ... < -0 == +0 < ... < inf < NaN`, descending when `largest`, ascending otherwise;
//! ties broken by lower index — identical to `folo-core` CPU `topk`):
//! - **Contiguous `d <= 512` (`inner == 1`)**: single-pass 256-thread workgroup Batcher bitonic sort in shared
//!   memory.
//! - **Contiguous 2-stage parallel block merge (`inner == 1`, `d > 512`, `k <= 64`, `tiles * k <= 512`)**:
//!   each 512-element tile extracts its local top-`k` into a single interleaved `(key, orig_idx)` candidate buffer
//!   (keeping every pipeline within 4 storage bindings), then a second workgroup-per-row bitonic merge produces
//!   the final global top-`k` with a single full read of `X`.
//! - **Contiguous multi-pass selection (`inner == 1`, large `d` / large `k`)**: workgroup-parallel `k`-pass
//!   selection with 32-wide subgroup reduction (or portable shared-memory tree fallback).
//! - **Strided axis (`inner > 1`)**: coalesced per-column dispatch across `inner` with single-pass register-sorted
//!   top-`k` for `k <= 32` and `k`-pass scan above.

use super::{check_buf, prod, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Maximum row length handled by single-pass workgroup bitonic sort.
const BITONIC_MAX: usize = 512;
/// Maximum `k` for the 2-stage parallel block-merge path.
const STAGE_MAX_K: usize = 64;
/// Maximum `k` for the single-pass register-sorted strided column path.
const COL_REG_MAX_K: usize = 32;

/// Descriptor for a batched arbitrary-axis `topk` dispatch over dense `[outer, dim_len, inner]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedTopkDesc {
    /// Product of dimensions before `dim`.
    pub outer: usize,
    /// Length of the reduced axis `dim`.
    pub dim_len: usize,
    /// Product of dimensions after `dim` (`1` when `dim` is the last axis).
    pub inner: usize,
    /// Number of top elements to select (`0 <= k <= dim_len`).
    pub k: usize,
    /// `true` for largest `k` (descending), `false` for smallest `k` (ascending).
    pub largest: bool,
    /// Storage dtype of input and output values (`F32` or `F16`). Indices are written as `i64`.
    pub dtype: DType,
}

/// Shared WGSL monotone total-order key and tie-breaking comparator matching `folo-core` CPU `topk`.
const TOPK_COMMON: &str = r#"
const TOPK_EMPTY: u32 = 0xffffffffu;

fn key_f32_bits(bits: u32) -> u32 {
    let b = select(bits, 0u, bits == 0x80000000u);
    return select(b ^ 0x80000000u, ~b, (b >> 31u) == 1u);
}

fn key_f16_bits(hbits: u32) -> u32 {
    let b = select(hbits, 0u, hbits == 0x8000u);
    return select(b ^ 0x8000u, (~b) & 0xffffu, (b >> 15u) == 1u);
}

fn before(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    let val_lt = select((ka < kb), (ka > kb), LARGEST);
    return select((ia < ib), val_lt, (ka != kb));
}

fn better_slot(ka: u32, ia: u32, kb: u32, ib: u32) -> bool {
    let real_cmp = before(ka, ia, kb, ib);
    let b_empty = ib == TOPK_EMPTY;
    return select(select(real_cmp, true, b_empty), false, (ia == TOPK_EMPTY));
}
"#;

/// Storage helpers: both F32 and F16 read `X` as raw `array<u32>` so IEEE NaN payloads and signed zeros are
/// keyed bit-exact without driver `f16 -> f32` NaN canonicalization.
fn value_io(is_f16: bool) -> &'static str {
    if is_f16 {
        "alias V_ST = f16;\n\
         fn read_x_key(idx: u32) -> u32 {\n\
             let word = X[idx >> 1u];\n\
             let hbits = (word >> ((idx & 1u) * 16u)) & 0xffffu;\n\
             return key_f16_bits(hbits);\n\
         }\n\
         fn read_x_val(idx: u32) -> f16 {\n\
             let word = X[idx >> 1u];\n\
             let pair = unpack2x16float(word);\n\
             return f16(select(pair.x, pair.y, (idx & 1u) == 1u));\n\
         }\n"
    } else {
        "alias V_ST = u32;\n\
         fn read_x_key(idx: u32) -> u32 { return key_f32_bits(X[idx]); }\n\
         fn read_x_val(idx: u32) -> u32 { return X[idx]; }\n"
    }
}

/// Shared bitonic sort loop body operating on `sh_k[512]` and `sh_i[512]`.
const BITONIC_SORT_LOOP: &str = r#"
    let half = n_pad >> 1u;
    for (var size = 2u; size <= n_pad; size = size << 1u) {
        for (var stride = size >> 1u; stride > 0u; stride = stride >> 1u) {
            if (lid < half) {
                let lo = lid & (stride - 1u);
                let p = ((lid - lo) << 1u) | lo;
                let q = p | stride;
                let kp = sh_k[p];
                let ip = sh_i[p];
                let kq = sh_k[q];
                let iq = sh_i[q];
                let best_first = (p & size) == 0u;
                let swap = select(better_slot(kp, ip, kq, iq), better_slot(kq, iq, kp, ip), best_first);
                if (swap) {
                    sh_k[p] = kq; sh_i[p] = iq;
                    sh_k[q] = kp; sh_i[q] = ip;
                }
            }
            workgroupBarrier();
        }
    }
"#;

/// Single-pass contiguous bitonic top-k (`d <= 512`, `F32` or `F16`).
/// Params: 0 = `outer`, 1 = `d`, 2 = `k`.
fn gen_bitonic(spec: &Spec) -> String {
    let io = value_io(spec.is_f16());
    format!(
        "{TOPK_COMMON}\n\
         @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         {io}\
         @group(0) @binding(2) var<storage, read_write> V: array<V_ST>;\n\
         @group(0) @binding(3) var<storage, read_write> I: array<u32>;\n\
         var<workgroup> sh_k: array<u32, 512>;\n\
         var<workgroup> sh_i: array<u32, 512>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let row = wid.y * nwg.x + wid.x;\n\
             let rows = pu(0u); let d = pu(1u); let k = pu(2u);\n\
             let valid = row < rows;\n\
             let base = row * d;\n\
             var n_pad = 2u;\n\
             while (n_pad < d) {{ n_pad = n_pad << 1u; }}\n\
             let i0 = lid;\n\
             if (i0 < n_pad) {{\n\
                 if (valid && i0 < d) {{ sh_k[i0] = read_x_key(base + i0); sh_i[i0] = i0; }}\n\
                 else {{ sh_k[i0] = 0u; sh_i[i0] = TOPK_EMPTY; }}\n\
             }}\n\
             let i1 = lid + 256u;\n\
             if (i1 < n_pad) {{\n\
                 if (valid && i1 < d) {{ sh_k[i1] = read_x_key(base + i1); sh_i[i1] = i1; }}\n\
                 else {{ sh_k[i1] = 0u; sh_i[i1] = TOPK_EMPTY; }}\n\
             }}\n\
             workgroupBarrier();\n\
             {BITONIC_SORT_LOOP}\
             if (!valid) {{ return; }}\n\
             for (var j = lid; j < k; j += 256u) {{\n\
                 let idx = sh_i[j];\n\
                 V[row * k + j] = read_x_val(base + idx);\n\
                 I[2u * (row * k + j)] = idx;\n\
                 I[2u * (row * k + j) + 1u] = 0u;\n\
             }}\n\
         }}\n"
    )
}

/// Stage 1 of 2-stage parallel block top-k: each 512-element tile extracts its local top-`k` into interleaved
/// `CAND[2*slot] = key`, `CAND[2*slot + 1] = orig_idx` (2 storage bindings total).
/// Params: 0 = `outer`, 1 = `d`, 2 = `k`, 3 = `tiles`.
fn gen_stage1(spec: &Spec) -> String {
    let io = value_io(spec.is_f16());
    format!(
        "{TOPK_COMMON}\n\
         @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         {io}\
         @group(0) @binding(2) var<storage, read_write> CAND: array<u32>;\n\
         var<workgroup> sh_k: array<u32, 512>;\n\
         var<workgroup> sh_i: array<u32, 512>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let g = wid.y * nwg.x + wid.x;\n\
             let rows = pu(0u); let d = pu(1u); let k = pu(2u); let tiles = pu(3u);\n\
             let valid = g < rows * tiles;\n\
             let row = g / tiles;\n\
             let tile = g % tiles;\n\
             let base = row * d;\n\
             let start = tile * 512u;\n\
             let n_pad = 512u;\n\
             let g0 = start + lid;\n\
             if (valid && g0 < d) {{ sh_k[lid] = read_x_key(base + g0); sh_i[lid] = g0; }}\n\
             else {{ sh_k[lid] = 0u; sh_i[lid] = TOPK_EMPTY; }}\n\
             let g1 = start + lid + 256u;\n\
             if (valid && g1 < d) {{ sh_k[lid + 256u] = read_x_key(base + g1); sh_i[lid + 256u] = g1; }}\n\
             else {{ sh_k[lid + 256u] = 0u; sh_i[lid + 256u] = TOPK_EMPTY; }}\n\
             workgroupBarrier();\n\
             {BITONIC_SORT_LOOP}\
             if (!valid) {{ return; }}\n\
             let out_base = g * k;\n\
             if (lid < k) {{\n\
                 let slot = out_base + lid;\n\
                 CAND[2u * slot] = sh_k[lid];\n\
                 CAND[2u * slot + 1u] = sh_i[lid];\n\
             }}\n\
         }}\n"
    )
}

/// Stage 2 of 2-stage parallel block top-k: merges `tiles * k <= 512` interleaved candidates per row and gathers
/// final `V` & `I` (4 storage bindings total: `X`, `CAND`, `V`, `I`).
/// Params: 0 = `outer`, 1 = `d`, 2 = `k`, 3 = `n_cand` (`tiles * k`).
fn gen_stage2(spec: &Spec) -> String {
    let io = value_io(spec.is_f16());
    format!(
        "{TOPK_COMMON}\n\
         @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
         {io}\
         @group(0) @binding(2) var<storage, read> CAND: array<u32>;\n\
         @group(0) @binding(3) var<storage, read_write> V: array<V_ST>;\n\
         @group(0) @binding(4) var<storage, read_write> I: array<u32>;\n\
         var<workgroup> sh_k: array<u32, 512>;\n\
         var<workgroup> sh_i: array<u32, 512>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let row = wid.y * nwg.x + wid.x;\n\
             let rows = pu(0u); let d = pu(1u); let k = pu(2u); let n_cand = pu(3u);\n\
             let valid = row < rows;\n\
             let cbase = row * n_cand;\n\
             var n_pad = 2u;\n\
             while (n_pad < n_cand) {{ n_pad = n_pad << 1u; }}\n\
             let i0 = lid;\n\
             if (i0 < n_pad) {{\n\
                 if (valid && i0 < n_cand) {{\n\
                     let s0 = cbase + i0;\n\
                     sh_k[i0] = CAND[2u * s0];\n\
                     sh_i[i0] = CAND[2u * s0 + 1u];\n\
                 }} else {{\n\
                     sh_k[i0] = 0u; sh_i[i0] = TOPK_EMPTY;\n\
                 }}\n\
             }}\n\
             let i1 = lid + 256u;\n\
             if (i1 < n_pad) {{\n\
                 if (valid && i1 < n_cand) {{\n\
                     let s1 = cbase + i1;\n\
                     sh_k[i1] = CAND[2u * s1];\n\
                     sh_i[i1] = CAND[2u * s1 + 1u];\n\
                 }} else {{\n\
                     sh_k[i1] = 0u; sh_i[i1] = TOPK_EMPTY;\n\
                 }}\n\
             }}\n\
             workgroupBarrier();\n\
             {BITONIC_SORT_LOOP}\
             if (!valid) {{ return; }}\n\
             let xbase = row * d;\n\
             if (lid < k) {{\n\
                 let idx = sh_i[lid];\n\
                 V[row * k + lid] = read_x_val(xbase + idx);\n\
                 I[2u * (row * k + lid)] = idx;\n\
                 I[2u * (row * k + lid) + 1u] = 0u;\n\
             }}\n\
         }}\n"
    )
}

/// Multi-pass contiguous selection fallback (`F32` and `F16`, subgroups or portable tree).
/// Params: 0 = `outer`, 1 = `d`, 2 = `k`.
fn gen_select(spec: &Spec) -> String {
    let io = value_io(spec.is_f16());
    let sg = spec.is_subgroups();
    if sg {
        format!(
            "{TOPK_COMMON}\n\
             @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
             {io}\
             @group(0) @binding(2) var<storage, read_write> V: array<V_ST>;\n\
             @group(0) @binding(3) var<storage, read_write> I: array<u32>;\n\
             var<workgroup> sh_k: array<u32, 8>;\n\
             var<workgroup> sh_i: array<u32, 8>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                     @builtin(local_invocation_index) lid: u32,\n\
                     @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {{\n\
                 let row = wid.y * nwg.x + wid.x;\n\
                 let rows = pu(0u); let d = pu(1u); let k = pu(2u);\n\
                 let valid = row < rows;\n\
                 let base = row * d;\n\
                 let n_chunks = (d + 255u) / 256u;\n\
                 let sg = min(sgid, 7u);\n\
                 var pk = 0u; var pi = TOPK_EMPTY;\n\
                 for (var j = 0u; j < k; j++) {{\n\
                     var bk = 0u; var bi = TOPK_EMPTY;\n\
                     if (valid) {{\n\
                         for (var c = 0u; c < n_chunks; c += 1u) {{\n\
                             let i = c * 256u + lid;\n\
                             if (i < d) {{\n\
                                 let ki = read_x_key(base + i);\n\
                                 if (j == 0u || before(pk, pi, ki, i)) {{\n\
                                     if (bi == TOPK_EMPTY || before(ki, i, bk, bi)) {{ bk = ki; bi = i; }}\n\
                                 }}\n\
                             }}\n\
                         }}\n\
                     }}\n\
                     for (var off = 16u; off > 0u; off = off >> 1u) {{\n\
                         let ok = subgroupShuffleDown(bk, off);\n\
                         let oi = subgroupShuffleDown(bi, off);\n\
                         if (lane < off && better_slot(ok, oi, bk, bi)) {{ bk = ok; bi = oi; }}\n\
                     }}\n\
                     if (lane == 0u) {{ sh_k[sg] = bk; sh_i[sg] = bi; }}\n\
                     workgroupBarrier();\n\
                     var rk = sh_k[lane & 7u]; var ri = sh_i[lane & 7u];\n\
                     for (var off = 4u; off > 0u; off = off >> 1u) {{\n\
                         let ok = subgroupShuffleXor(rk, off);\n\
                         let oi = subgroupShuffleXor(ri, off);\n\
                         if (better_slot(ok, oi, rk, ri)) {{ rk = ok; ri = oi; }}\n\
                     }}\n\
                     pk = rk; pi = ri;\n\
                     if (lid == 0u && valid) {{\n\
                         V[row * k + j] = read_x_val(base + pi);\n\
                         I[2u * (row * k + j)] = pi;\n\
                         I[2u * (row * k + j) + 1u] = 0u;\n\
                     }}\n\
                     workgroupBarrier();\n\
                 }}\n\
             }}\n"
        )
    } else {
        format!(
            "{TOPK_COMMON}\n\
             @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
             {io}\
             @group(0) @binding(2) var<storage, read_write> V: array<V_ST>;\n\
             @group(0) @binding(3) var<storage, read_write> I: array<u32>;\n\
             var<workgroup> sh_k: array<u32, 256>;\n\
             var<workgroup> sh_i: array<u32, 256>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                     @builtin(local_invocation_index) lid: u32) {{\n\
                 let row = wid.y * nwg.x + wid.x;\n\
                 let rows = pu(0u); let d = pu(1u); let k = pu(2u);\n\
                 let valid = row < rows;\n\
                 let base = row * d;\n\
                 let n_chunks = (d + 255u) / 256u;\n\
                 var pk = 0u; var pi = TOPK_EMPTY;\n\
                 for (var j = 0u; j < k; j++) {{\n\
                     var bk = 0u; var bi = TOPK_EMPTY;\n\
                     if (valid) {{\n\
                         for (var c = 0u; c < n_chunks; c += 1u) {{\n\
                             let i = c * 256u + lid;\n\
                             if (i < d) {{\n\
                                 let ki = read_x_key(base + i);\n\
                                 if (j == 0u || before(pk, pi, ki, i)) {{\n\
                                     if (bi == TOPK_EMPTY || before(ki, i, bk, bi)) {{ bk = ki; bi = i; }}\n\
                                 }}\n\
                             }}\n\
                         }}\n\
                     }}\n\
                     sh_k[lid] = bk; sh_i[lid] = bi;\n\
                     workgroupBarrier();\n\
                     for (var s = 128u; s > 0u; s = s >> 1u) {{\n\
                         if (lid < s) {{\n\
                             let ok = sh_k[lid + s]; let oi = sh_i[lid + s];\n\
                             if (better_slot(ok, oi, sh_k[lid], sh_i[lid])) {{\n\
                                 sh_k[lid] = ok; sh_i[lid] = oi;\n\
                             }}\n\
                         }}\n\
                         workgroupBarrier();\n\
                     }}\n\
                     pk = sh_k[0]; pi = sh_i[0];\n\
                     if (lid == 0u && valid) {{\n\
                         V[row * k + j] = read_x_val(base + pi);\n\
                         I[2u * (row * k + j)] = pi;\n\
                         I[2u * (row * k + j) + 1u] = 0u;\n\
                     }}\n\
                     workgroupBarrier();\n\
                 }}\n\
             }}\n"
        )
    }
}

/// Coalesced strided-axis top-k for `inner > 1` (`[outer, d, inner]` -> `[outer, k, inner]`).
/// When `SMALL_K` (`k <= 32`), keeps a sorted register buffer in a single pass over `d`; otherwise runs `k` passes.
/// Params: 0 = `outer`, 1 = `d`, 2 = `inner`, 3 = `k`.
fn gen_cols(spec: &Spec) -> String {
    let io = value_io(spec.is_f16());
    let small_k = spec.get_bool("SMALL_K");
    if small_k {
        format!(
            "{TOPK_COMMON}\n\
             @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
             {io}\
             @group(0) @binding(2) var<storage, read_write> V: array<V_ST>;\n\
             @group(0) @binding(3) var<storage, read_write> I: array<u32>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                     @builtin(local_invocation_index) lid: u32) {{\n\
                 let col = linear_id(wid, nwg, lid, 256u);\n\
                 let outer = pu(0u); let d = pu(1u); let inner = pu(2u); let k = pu(3u);\n\
                 if (col >= outer * inner) {{ return; }}\n\
                 let o = col / inner;\n\
                 let ii = col % inner;\n\
                 let xbase = o * d * inner + ii;\n\
                 var tk: array<u32, {COL_REG_MAX_K}>;\n\
                 var ti: array<u32, {COL_REG_MAX_K}>;\n\
                 for (var j = 0u; j < k; j++) {{ tk[j] = 0u; ti[j] = TOPK_EMPTY; }}\n\
                 for (var r = 0u; r < d; r++) {{\n\
                     let kr = read_x_key(xbase + r * inner);\n\
                     if (better_slot(kr, r, tk[k - 1u], ti[k - 1u])) {{\n\
                         var pos = k - 1u;\n\
                         while (pos > 0u && better_slot(kr, r, tk[pos - 1u], ti[pos - 1u])) {{\n\
                             tk[pos] = tk[pos - 1u];\n\
                             ti[pos] = ti[pos - 1u];\n\
                             pos -= 1u;\n\
                         }}\n\
                         tk[pos] = kr;\n\
                         ti[pos] = r;\n\
                     }}\n\
                 }}\n\
                 let obase = o * k * inner + ii;\n\
                 for (var j = 0u; j < k; j++) {{\n\
                     let idx = ti[j];\n\
                     let dst = obase + j * inner;\n\
                     V[dst] = read_x_val(xbase + idx * inner);\n\
                     I[2u * dst] = idx;\n\
                     I[2u * dst + 1u] = 0u;\n\
                 }}\n\
             }}\n"
        )
    } else {
        format!(
            "{TOPK_COMMON}\n\
             @group(0) @binding(1) var<storage, read> X: array<u32>;\n\
             {io}\
             @group(0) @binding(2) var<storage, read_write> V: array<V_ST>;\n\
             @group(0) @binding(3) var<storage, read_write> I: array<u32>;\n\
             @compute @workgroup_size(256)\n\
             fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                     @builtin(local_invocation_index) lid: u32) {{\n\
                 let col = linear_id(wid, nwg, lid, 256u);\n\
                 let outer = pu(0u); let d = pu(1u); let inner = pu(2u); let k = pu(3u);\n\
                 if (col >= outer * inner) {{ return; }}\n\
                 let o = col / inner;\n\
                 let ii = col % inner;\n\
                 let xbase = o * d * inner + ii;\n\
                 let obase = o * k * inner + ii;\n\
                 var pk = 0u; var pi = TOPK_EMPTY;\n\
                 for (var j = 0u; j < k; j++) {{\n\
                     var bk = 0u; var bi = TOPK_EMPTY;\n\
                     for (var r = 0u; r < d; r++) {{\n\
                         let kr = read_x_key(xbase + r * inner);\n\
                         if (j > 0u && !before(pk, pi, kr, r)) {{ continue; }}\n\
                         if (bi == TOPK_EMPTY || before(kr, r, bk, bi)) {{ bk = kr; bi = r; }}\n\
                     }}\n\
                     pk = bk; pi = bi;\n\
                     let dst = obase + j * inner;\n\
                     V[dst] = read_x_val(xbase + pi * inner);\n\
                     I[2u * dst] = pi;\n\
                     I[2u * dst + 1u] = 0u;\n\
                 }}\n\
             }}\n"
        )
    }
}

use Access::{Read as R, ReadWrite as RW};

static BTOPK_BITONIC: KernelDef =
    KernelDef { name: "btopk_bitonic", storage: &[R, RW, RW], source: Source::Generated(gen_bitonic) };
static BTOPK_STAGE1: KernelDef =
    KernelDef { name: "btopk_stage1", storage: &[R, RW], source: Source::Generated(gen_stage1) };
static BTOPK_STAGE2: KernelDef =
    KernelDef { name: "btopk_stage2", storage: &[R, R, RW, RW], source: Source::Generated(gen_stage2) };
static BTOPK_SELECT: KernelDef =
    KernelDef { name: "btopk_select", storage: &[R, RW, RW], source: Source::Generated(gen_select) };
static BTOPK_COLS: KernelDef =
    KernelDef { name: "btopk_cols", storage: &[R, RW, RW], source: Source::Generated(gen_cols) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] =
    &[&BTOPK_BITONIC, &BTOPK_STAGE1, &BTOPK_STAGE2, &BTOPK_SELECT, &BTOPK_COLS];

/// Dispatches batched arbitrary-axis `topk` over `[outer, dim_len, inner]` into `values` (`dtype`) and `indices`
/// (`i64`, two `u32` words per element), with deterministic IEEE total ordering and ascending-index ties.
pub fn batched_topk(
    ctx: &GpuContext,
    d: &BatchedTopkDesc,
    x: &wgpu::Buffer,
    values: &wgpu::Buffer,
    indices: &wgpu::Buffer,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "batched_topk")?;
    if d.k > d.dim_len {
        return Err(shape_err(format!("batched_topk: k = {} exceeds dim_len {}", d.k, d.dim_len)));
    }
    let in_elems = prod(&[d.outer, d.dim_len, d.inner])?;
    let out_elems = prod(&[d.outer, d.k, d.inner])?;
    let x_words = if d.dtype == DType::F16 { in_elems.div_ceil(2) } else { in_elems };
    check_buf(x, x_words, DType::F32, "batched_topk input")?;
    check_buf(values, out_elems, d.dtype, "batched_topk values")?;
    check_buf(indices, prod(&[out_elems, 2])?, DType::F32, "batched_topk indices")?;
    if out_elems == 0 {
        return Ok(());
    }
    if x == values || x == indices || values == indices {
        return Err(shape_err("batched_topk: input, values, and indices buffers must be distinct"));
    }
    if in_elems > u32::MAX as usize || prod(&[out_elems, 2])? > u32::MAX as usize {
        return Err(GpuError::Limit("batched_topk: tensor exceeds u32 indexing".into()));
    }
    let f16 = d.dtype == DType::F16;

    if d.inner == 1 {
        if d.dim_len <= BITONIC_MAX {
            let k = ctx.kernel(&BTOPK_BITONIC, Spec::new().bool("LARGEST", d.largest).f16(f16))?;
            let p = Params::new().uz(d.outer)?.uz(d.dim_len)?.uz(d.k)?;
            return ctx.dispatch(&k, &[x, values, indices], &p, ctx.grid_1d(d.outer as u64 * 256, 256)?);
        }
        let tiles = d.dim_len.div_ceil(BITONIC_MAX);
        let n_cand = tiles.saturating_mul(d.k);
        if d.k <= STAGE_MAX_K && n_cand <= BITONIC_MAX {
            let total_tiles = prod(&[d.outer, tiles])?;
            let cand_words = prod(&[d.outer, n_cand, 2])?;
            if cand_words <= u32::MAX as usize {
                let cand = ctx.alloc(DType::F32.bytes(cand_words))?;

                let k1 = ctx.kernel(&BTOPK_STAGE1, Spec::new().bool("LARGEST", d.largest).f16(f16))?;
                let p1 = Params::new().uz(d.outer)?.uz(d.dim_len)?.uz(d.k)?.uz(tiles)?;
                ctx.dispatch(&k1, &[x, cand.raw()], &p1, ctx.grid_1d(total_tiles as u64 * 256, 256)?)?;

                let k2 = ctx.kernel(&BTOPK_STAGE2, Spec::new().bool("LARGEST", d.largest).f16(f16))?;
                let p2 = Params::new().uz(d.outer)?.uz(d.dim_len)?.uz(d.k)?.uz(n_cand)?;
                return ctx.dispatch(
                    &k2,
                    &[x, cand.raw(), values, indices],
                    &p2,
                    ctx.grid_1d(d.outer as u64 * 256, 256)?,
                );
            }
        }
        let k = ctx.kernel(
            &BTOPK_SELECT,
            Spec::new().bool("LARGEST", d.largest).f16(f16).subgroups(use_subgroups(ctx)),
        )?;
        let p = Params::new().uz(d.outer)?.uz(d.dim_len)?.uz(d.k)?;
        ctx.dispatch(&k, &[x, values, indices], &p, ctx.grid_1d(d.outer as u64 * 256, 256)?)
    } else {
        let cols = prod(&[d.outer, d.inner])?;
        let k = ctx.kernel(
            &BTOPK_COLS,
            Spec::new().bool("LARGEST", d.largest).bool("SMALL_K", d.k <= COL_REG_MAX_K).f16(f16),
        )?;
        let p = Params::new().uz(d.outer)?.uz(d.dim_len)?.uz(d.inner)?.uz(d.k)?;
        ctx.dispatch(&k, &[x, values, indices], &p, ctx.grid_1d(cols as u64, 256)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btopk_select_portable_emits_no_subgroup_tokens() {
        let spec = Spec::new().bool("LARGEST", true).f16(false).subgroups(false);
        let wgsl = crate::pipeline::render(&BTOPK_SELECT, &spec);
        assert!(!wgsl.contains("subgroup"), "portable btopk_select emitted subgroup token:\n{wgsl}");
    }
}
