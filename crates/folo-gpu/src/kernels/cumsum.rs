//! WGSL work-efficient parallel prefix-sum (`cumsum`) along an arbitrary axis (`F32` and `F16`).
//!
//! Views a dense tensor along axis `dim` as `[outer, d, inner]`:
//! - **Contiguous axis (`inner == 1`)**: Blelloch parallel scan in workgroup memory (`512` elements per 256-thread
//!   workgroup, `O(log 512)` steps). Rows with `d <= 512` complete in a single pass; longer rows use a
//!   hierarchical multi-block scan (`cumsum_rows_block` -> recursive exclusive `cumsum` on `f32` block totals ->
//!   `cumsum_rows_add`) so `F16` long scans retain full `f32` accumulation before final rounding.
//! - **Strided axis (`inner > 1`)**: coalesced per-column scan across `inner` (`cumsum_cols`), switching to a
//!   hierarchical block-parallel scan (`cumsum_cols_block` -> recursive exclusive scan -> `cumsum_cols_add`) when
//!   `d > 512` and `outer * inner < 256` so long scans over few columns still saturate the GPU.

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Elements scanned per 256-thread workgroup on the contiguous Blelloch path.
const ROW_BLOCK: usize = 512;
/// Elements scanned per chunk on the hierarchical strided-column path.
const COL_CHUNK: usize = 256;

/// Single-block Blelloch parallel scan (`512` elements per 256-thread workgroup).
/// Spec: `EXCLUSIVE`, `WRITE_SUMS`, `Y_F32` (writes `f32` output when multi-block F16), `f16` (`X`, and `Y` unless `Y_F32`).
/// Params: 0 = `outer`, 1 = `d`, 2 = `blocks`.
fn gen_cumsum_rows_block(spec: &Spec) -> String {
    let exclusive = spec.get_bool("EXCLUSIVE");
    let write_sums = spec.get_bool("WRITE_SUMS");
    let y_f32 = spec.get_bool("Y_F32");
    let y_ty = if y_f32 { "f32" } else { "T" };
    let y_cast = |expr: &str| if y_f32 { expr.to_string() } else { format!("T({expr})") };
    let out0 = if exclusive { y_cast("sh[i0]") } else { y_cast("sh[i0] + v0") };
    let out1 = if exclusive { y_cast("sh[i1]") } else { y_cast("sh[i1] + v1") };
    let sum_write = if write_sums { "        if (valid) { SUMS[g] = sh[511u]; }\n" } else { "" };
    format!(
        "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<{y_ty}>;\n\
         @group(0) @binding(3) var<storage, read_write> SUMS: array<f32>;\n\
         var<workgroup> sh: array<f32, 512>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let g = wid.y * nwg.x + wid.x;\n\
             let outer = pu(0u); let d = pu(1u); let blocks = pu(2u);\n\
             let total_wg = outer * blocks;\n\
             let valid = g < total_wg;\n\
             let row = g / blocks;\n\
             let blk = g % blocks;\n\
             let base = row * d + blk * 512u;\n\
             let col0 = blk * 512u + lid;\n\
             let col1 = col0 + 256u;\n\
             var v0 = 0.0;\n\
             var v1 = 0.0;\n\
             if (valid && col0 < d) {{ v0 = f32(X[base + lid]); }}\n\
             if (valid && col1 < d) {{ v1 = f32(X[base + lid + 256u]); }}\n\
             let i0 = lid;\n\
             let i1 = lid + 256u;\n\
             sh[i0] = v0;\n\
             sh[i1] = v1;\n\
             workgroupBarrier();\n\
             // Blelloch upsweep.\n\
             var offset = 1u;\n\
             for (var stride = 256u; stride > 0u; stride = stride >> 1u) {{\n\
                 if (lid < stride) {{\n\
                     let ai = offset * (2u * lid + 1u) - 1u;\n\
                     let bi = offset * (2u * lid + 2u) - 1u;\n\
                     sh[bi] += sh[ai];\n\
                 }}\n\
                 offset = offset << 1u;\n\
                 workgroupBarrier();\n\
             }}\n\
             if (lid == 0u) {{\n\
         {sum_write}\
                 sh[511u] = 0.0;\n\
             }}\n\
             workgroupBarrier();\n\
             // Blelloch downsweep.\n\
             for (var stride = 1u; stride < 512u; stride = stride << 1u) {{\n\
                 offset = offset >> 1u;\n\
                 if (lid < stride) {{\n\
                     let ai = offset * (2u * lid + 1u) - 1u;\n\
                     let bi = offset * (2u * lid + 2u) - 1u;\n\
                     let tmp = sh[ai];\n\
                     sh[ai] = sh[bi];\n\
                     sh[bi] += tmp;\n\
                 }}\n\
                 workgroupBarrier();\n\
             }}\n\
             if (valid && col0 < d) {{ Y[base + lid] = {out0}; }}\n\
             if (valid && col1 < d) {{ Y[base + lid + 256u] = {out1}; }}\n\
         }}\n"
    )
}

/// Adds per-block exclusive prefix offsets to local `f32` scans and writes `Y` in `T`.
/// Params: 0 = `outer`, 1 = `d`, 2 = `blocks`.
const CUMSUM_ROWS_ADD_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> LOCAL: array<f32>;
@group(0) @binding(2) var<storage, read> OFFSETS: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let idx = linear_id(wid, nwg, lid, 256u);
    let outer = pu(0u); let d = pu(1u); let blocks = pu(2u);
    let total = outer * d;
    if (idx >= total) { return; }
    let row = idx / d;
    let col = idx % d;
    let blk = col / 512u;
    Y[idx] = T(LOCAL[idx] + OFFSETS[row * blocks + blk]);
}
"#;

/// Coalesced single-pass column scan for `inner > 1` (`[outer, d, inner]`).
/// Spec: `EXCLUSIVE`, `f16`.
/// Params: 0 = `outer`, 1 = `d`, 2 = `inner`.
fn gen_cumsum_cols(spec: &Spec) -> String {
    let exclusive = spec.get_bool("EXCLUSIVE");
    let step = if exclusive {
        "        let v = f32(X[idx]);\n        Y[idx] = T(acc);\n        acc += v;\n"
    } else {
        "        acc += f32(X[idx]);\n        Y[idx] = T(acc);\n"
    };
    format!(
        "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
         @group(0) @binding(2) var<storage, read_write> Y: array<T>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let col_id = linear_id(wid, nwg, lid, 256u);\n\
             let outer = pu(0u); let d = pu(1u); let inner = pu(2u);\n\
             let n = outer * inner;\n\
             if (col_id >= n) {{ return; }}\n\
             let o = col_id / inner;\n\
             let i = col_id % inner;\n\
             let base = o * d * inner + i;\n\
             var acc = 0.0;\n\
             for (var k = 0u; k < d; k += 1u) {{\n\
                 let idx = base + k * inner;\n\
             {step}\
             }}\n\
         }}\n"
    )
}

/// Chunk-parallel coalesced column scan for long strided scans (`[outer, d, inner]`, `COL_CHUNK = 256`).
/// Writes local `f32` scan to `LOCAL` and per-chunk totals `[outer, blocks, inner]` to `SUMS`.
/// Spec: `EXCLUSIVE`, `f16`.
/// Params: 0 = `outer`, 1 = `d`, 2 = `inner`, 3 = `blocks`.
fn gen_cumsum_cols_block(spec: &Spec) -> String {
    let exclusive = spec.get_bool("EXCLUSIVE");
    let step = if exclusive {
        "        let v = f32(X[idx]);\n        LOCAL[idx] = acc;\n        acc += v;\n"
    } else {
        "        acc += f32(X[idx]);\n        LOCAL[idx] = acc;\n"
    };
    format!(
        "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
         @group(0) @binding(2) var<storage, read_write> LOCAL: array<f32>;\n\
         @group(0) @binding(3) var<storage, read_write> SUMS: array<f32>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let tid = linear_id(wid, nwg, lid, 256u);\n\
             let outer = pu(0u); let d = pu(1u); let inner = pu(2u); let blocks = pu(3u);\n\
             let total_jobs = outer * blocks * inner;\n\
             if (tid >= total_jobs) {{ return; }}\n\
             let i = tid % inner;\n\
             let tmp = tid / inner;\n\
             let blk = tmp % blocks;\n\
             let o = tmp / blocks;\n\
             let k0 = blk * {COL_CHUNK}u;\n\
             let k_end = min(k0 + {COL_CHUNK}u, d);\n\
             let base = o * d * inner + i;\n\
             var acc = 0.0;\n\
             for (var k = k0; k < k_end; k += 1u) {{\n\
                 let idx = base + k * inner;\n\
             {step}\
             }}\n\
             SUMS[(o * blocks + blk) * inner + i] = acc;\n\
         }}\n"
    )
}

/// Adds per-chunk exclusive offsets `[outer, blocks, inner]` to `LOCAL` and writes `Y` in `T`.
/// Params: 0 = `outer`, 1 = `d`, 2 = `inner`, 3 = `blocks`.
const CUMSUM_COLS_ADD_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> LOCAL: array<f32>;
@group(0) @binding(2) var<storage, read> OFFSETS: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let idx = linear_id(wid, nwg, lid, 256u);
    let outer = pu(0u); let d = pu(1u); let inner = pu(2u); let blocks = pu(3u);
    let total = outer * d * inner;
    if (idx >= total) { return; }
    let i = idx % inner;
    let tmp = idx / inner;
    let k = tmp % d;
    let o = tmp / d;
    let blk = k / 256u;
    Y[idx] = T(LOCAL[idx] + OFFSETS[(o * blocks + blk) * inner + i]);
}
"#;

use Access::{Read as R, ReadWrite as RW};

static CUMSUM_ROWS_BLOCK: KernelDef = KernelDef {
    name: "cumsum_rows_block",
    storage: &[R, RW, RW],
    source: Source::Generated(gen_cumsum_rows_block),
};
static CUMSUM_ROWS_ADD: KernelDef =
    KernelDef { name: "cumsum_rows_add", storage: &[R, R, RW], source: Source::Static(CUMSUM_ROWS_ADD_SRC) };
static CUMSUM_COLS: KernelDef =
    KernelDef { name: "cumsum_cols", storage: &[R, RW], source: Source::Generated(gen_cumsum_cols) };
static CUMSUM_COLS_BLOCK: KernelDef = KernelDef {
    name: "cumsum_cols_block",
    storage: &[R, RW, RW],
    source: Source::Generated(gen_cumsum_cols_block),
};
static CUMSUM_COLS_ADD: KernelDef =
    KernelDef { name: "cumsum_cols_add", storage: &[R, R, RW], source: Source::Static(CUMSUM_COLS_ADD_SRC) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] =
    &[&CUMSUM_ROWS_BLOCK, &CUMSUM_ROWS_ADD, &CUMSUM_COLS, &CUMSUM_COLS_BLOCK, &CUMSUM_COLS_ADD];

/// Inclusive prefix sum (`cumsum`) over the middle axis of dense `[outer, d, inner]` (`F32` or `F16`).
///
/// `y` must not alias `x`.
pub fn cumsum(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    dims: (usize, usize, usize),
    dtype: DType,
) -> Result<()> {
    cumsum_ex(ctx, x, y, dims, false, dtype)
}

/// Inclusive (`exclusive == false`) or exclusive (`exclusive == true`) prefix sum over the middle axis of dense
/// `[outer, d, inner]` (`F32` or `F16`, always accumulated in `f32`).
///
/// `y` must not alias `x`.
pub fn cumsum_ex(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    exclusive: bool,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "cumsum")?;
    let total = prod(&[outer, d, inner])?;
    check_buf(x, total, dtype, "cumsum input")?;
    check_buf(y, total, dtype, "cumsum output")?;
    if total == 0 {
        return Ok(());
    }
    if x == y {
        return Err(shape_err("cumsum: output buffer must not alias input"));
    }
    if total > u32::MAX as usize {
        return Err(GpuError::Limit("cumsum: tensor length exceeds u32 indexing".into()));
    }

    if inner == 1 {
        let blocks = d.div_ceil(ROW_BLOCK);
        if blocks == 1 {
            let k = ctx.kernel(
                &CUMSUM_ROWS_BLOCK,
                Spec::new()
                    .bool("EXCLUSIVE", exclusive)
                    .bool("WRITE_SUMS", false)
                    .bool("Y_F32", false)
                    .f16(dtype == DType::F16),
            )?;
            let p = Params::new().uz(outer)?.uz(d)?.uz(1)?;
            return ctx.dispatch(&k, &[x, y, ctx.dummy()], &p, ctx.grid_1d(outer as u64 * 256, 256)?);
        }
        let total_wg = prod(&[outer, blocks])?;
        if total_wg > u32::MAX as usize {
            return Err(GpuError::Limit("cumsum: workgroup count exceeds u32 indexing".into()));
        }
        let local = ctx.alloc(DType::F32.bytes(total))?;
        let block_sums = ctx.alloc(DType::F32.bytes(total_wg))?;
        let block_offsets = ctx.alloc(DType::F32.bytes(total_wg))?;

        let k1 = ctx.kernel(
            &CUMSUM_ROWS_BLOCK,
            Spec::new()
                .bool("EXCLUSIVE", exclusive)
                .bool("WRITE_SUMS", true)
                .bool("Y_F32", true)
                .f16(dtype == DType::F16),
        )?;
        let p1 = Params::new().uz(outer)?.uz(d)?.uz(blocks)?;
        ctx.dispatch(
            &k1,
            &[x, local.raw(), block_sums.raw()],
            &p1,
            ctx.grid_1d(total_wg as u64 * 256, 256)?,
        )?;

        cumsum_ex(ctx, block_sums.raw(), block_offsets.raw(), (outer, blocks, 1), true, DType::F32)?;

        let k3 = ctx.kernel(&CUMSUM_ROWS_ADD, Spec::new().f16(dtype == DType::F16))?;
        ctx.dispatch(&k3, &[local.raw(), block_offsets.raw(), y], &p1, ctx.grid_1d(total as u64, 256)?)
    } else {
        let cols = prod(&[outer, inner])?;
        if d <= 512 || cols >= 256 {
            let k =
                ctx.kernel(&CUMSUM_COLS, Spec::new().bool("EXCLUSIVE", exclusive).f16(dtype == DType::F16))?;
            let p = Params::new().uz(outer)?.uz(d)?.uz(inner)?;
            return ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(cols as u64, 256)?);
        }
        let blocks = d.div_ceil(COL_CHUNK);
        let jobs = prod(&[outer, blocks, inner])?;
        if jobs > u32::MAX as usize {
            return Err(GpuError::Limit("cumsum: job count exceeds u32 indexing".into()));
        }
        let local = ctx.alloc(DType::F32.bytes(total))?;
        let sums = ctx.alloc(DType::F32.bytes(jobs))?;
        let offsets = ctx.alloc(DType::F32.bytes(jobs))?;

        let k1 = ctx
            .kernel(&CUMSUM_COLS_BLOCK, Spec::new().bool("EXCLUSIVE", exclusive).f16(dtype == DType::F16))?;
        let p = Params::new().uz(outer)?.uz(d)?.uz(inner)?.uz(blocks)?;
        ctx.dispatch(&k1, &[x, local.raw(), sums.raw()], &p, ctx.grid_1d(jobs as u64, 256)?)?;

        cumsum_ex(ctx, sums.raw(), offsets.raw(), (outer, blocks, inner), true, DType::F32)?;

        let k3 = ctx.kernel(&CUMSUM_COLS_ADD, Spec::new().f16(dtype == DType::F16))?;
        ctx.dispatch(&k3, &[local.raw(), offsets.raw(), y], &p, ctx.grid_1d(total as u64, 256)?)
    }
}
