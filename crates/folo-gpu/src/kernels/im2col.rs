//! WGSL shared-memory tiled `im2col` and atomic-free tiled gather `col2im` (`F32` and `F16`).
//!
//! Layout matches NCHW `folo-core` / `folo-gpu` convolutions:
//! - Image `X`: `[N, C, H, W]`
//! - Unfolded column matrix `COL`: `[N, C * KH * KW, OH * OW]`, where row `(c * KH + ky) * KW + kx` and column
//!   `oy * OW + ox` holds `X[n, c, oy * SH - PH + ky * DH, ox * SW - PW + kx * DW]` (zero-padded out of bounds).
//!
//! Two complementary tiled kernels:
//! - **`im2col_tiled`**: each `(16, 16)` workgroup stages one `TILE_IH × TILE_IW` input tile per `(n, c)` plane into
//!   workgroup memory (up to 8 KiB) with coalesced reads, then emits all `KH × KW` kernel taps with coalesced writes
//!   across `ox`. Falls back to `im2col_generic` when an extreme kernel/stride/dilation exceeds shared memory.
//! - **`col2im_tiled`**: 100% atomic-free deterministic gather. Each `(16, 16)` workgroup owns a `16 × 16` input
//!   pixel tile `(iy0..iy0+16, ix0..ix0+16)`. For each `(ky, kx)` tap, computes the workgroup-uniform contributing
//!   `[oy_min..=oy_max] × [ox_min..=ox_max]` patch (at most `16 × 16` = 1 KiB for any stride/dilation/padding),
//!   stages it cooperatively into shared memory with coalesced reads, and accumulates into `f32` registers before
//!   writing `X` once.

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::kernels::pooling::Window2d;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Maximum shared-memory floats (`8 KiB`) for the tiled `im2col` input patch.
const MAX_IM2COL_TILE_ELEMS: usize = 2048;

/// Descriptor for `im2col` and `col2im` on NCHW `[n, c, h, w]` tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Im2ColDesc {
    /// Batch size `N`.
    pub n: usize,
    /// Channels `C`.
    pub c: usize,
    /// Input height `H`.
    pub h: usize,
    /// Input width `W`.
    pub w: usize,
    /// Window geometry (kernel, stride, padding, dilation).
    pub win: Window2d,
    /// Storage dtype (`F32` or `F16`; `col2im` accumulates in `f32`).
    pub dtype: DType,
}

impl Im2ColDesc {
    /// Output spatial size `(OH, OW)`.
    pub fn output_hw(&self) -> Result<(usize, usize)> {
        self.win.output(self.h, self.w)
    }

    /// Number of rows per batch in `COL` (`C * KH * KW`).
    pub fn col_rows(&self) -> Result<usize> {
        prod(&[self.c, self.win.kernel.0, self.win.kernel.1])
    }

    /// Number of columns per batch in `COL` (`OH * OW`).
    pub fn col_cols(&self) -> Result<usize> {
        let (oh, ow) = self.output_hw()?;
        prod(&[oh, ow])
    }

    fn tile_in_hw(&self) -> Option<(usize, usize)> {
        let th = 15usize
            .checked_mul(self.win.stride.0)?
            .checked_add((self.win.kernel.0 - 1).checked_mul(self.win.dilation.0)?)?
            .checked_add(1)?;
        let tw = 15usize
            .checked_mul(self.win.stride.1)?
            .checked_add((self.win.kernel.1 - 1).checked_mul(self.win.dilation.1)?)?
            .checked_add(1)?;
        let elems = th.checked_mul(tw)?;
        (elems <= MAX_IM2COL_TILE_ELEMS).then_some((th, tw))
    }
}

/// Bounds check for precompiling `"im2col_tiled"` (`kernels/bounds.rs`).
pub(crate) fn im2col_tiled_fits(spec: &Spec) -> bool {
    let kh = spec.get_u32("KH", 0) as usize;
    let kw = spec.get_u32("KW", 0) as usize;
    let sh = spec.get_u32("SH", 0) as usize;
    let sw = spec.get_u32("SW", 0) as usize;
    let dh = spec.get_u32("DH", 0) as usize;
    let dw = spec.get_u32("DW", 0) as usize;
    if kh == 0 || kw == 0 || sh == 0 || sw == 0 || dh == 0 || dw == 0 {
        return false;
    }
    let Some(th) = 15usize
        .checked_mul(sh)
        .and_then(|v| v.checked_add((kh - 1).checked_mul(dh)?))
        .and_then(|v| v.checked_add(1))
    else {
        return false;
    };
    let Some(tw) = 15usize
        .checked_mul(sw)
        .and_then(|v| v.checked_add((kw - 1).checked_mul(dw)?))
        .and_then(|v| v.checked_add(1))
    else {
        return false;
    };
    th.checked_mul(tw).is_some_and(|e| e <= MAX_IM2COL_TILE_ELEMS)
}

/// Shared-memory tiled `im2col` (`16 × 16` output spatial tile per workgroup).
/// Spec: `KH`, `KW`, `SH`, `SW`, `DH`, `DW`, `f16`.
/// Params: 0 = `total_tiles`, 1 = `C`, 2 = `H`, 3 = `W`, 4 = `OH`, 5 = `OW`, 6 = `PH` (i32), 7 = `PW` (i32),
/// 8 = `tiles_x`, 9 = `tiles_y`.
fn gen_im2col_tiled(spec: &Spec) -> String {
    let kh = spec.get_u32("KH", 3);
    let kw = spec.get_u32("KW", 3);
    let sh = spec.get_u32("SH", 1);
    let sw = spec.get_u32("SW", 1);
    let dh = spec.get_u32("DH", 1);
    let dw = spec.get_u32("DW", 1);
    let tile_ih = 15 * sh + (kh - 1) * dh + 1;
    let tile_iw = 15 * sw + (kw - 1) * dw + 1;
    let tile_n = tile_ih * tile_iw;
    format!(
        "@group(0) @binding(1) var<storage, read> X: array<T>;\n\
         @group(0) @binding(2) var<storage, read_write> COL: array<T>;\n\
         var<workgroup> sh_in: array<f32, {tile_n}>;\n\
         @compute @workgroup_size(16, 16)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {{\n\
             let tile_id = wid.y * nwg.x + wid.x;\n\
             let valid_wg = tile_id < pu(0u);\n\
             let C = pu(1u); let H = pu(2u); let W = pu(3u);\n\
             let OH = pu(4u); let OW = pu(5u);\n\
             let ph = pi(6u); let pw = pi(7u);\n\
             let tiles_x = pu(8u); let tiles_y = pu(9u);\n\
             let tx = tile_id % tiles_x;\n\
             let tmp = tile_id / tiles_x;\n\
             let ty = tmp % tiles_y;\n\
             let nc = tmp / tiles_y;\n\
             let oy0 = ty * 16u;\n\
             let ox0 = tx * 16u;\n\
             let iy0 = i32(oy0 * {sh}u) - ph;\n\
             let ix0 = i32(ox0 * {sw}u) - pw;\n\
             let x_base = nc * H * W;\n\
             let tid = lid.y * 16u + lid.x;\n\
             for (var idx = tid; idx < {tile_n}u; idx += 256u) {{\n\
                 var v = 0.0;\n\
                 if (valid_wg) {{\n\
                     let ly = idx / {tile_iw}u;\n\
                     let lx = idx % {tile_iw}u;\n\
                     let iy = iy0 + i32(ly);\n\
                     let ix = ix0 + i32(lx);\n\
                     if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {{\n\
                         v = f32(X[x_base + u32(iy) * W + u32(ix)]);\n\
                     }}\n\
                 }}\n\
                 sh_in[idx] = v;\n\
             }}\n\
             workgroupBarrier();\n\
             if (!valid_wg) {{ return; }}\n\
             let oy = oy0 + lid.y;\n\
             let ox = ox0 + lid.x;\n\
             if (oy >= OH || ox >= OW) {{ return; }}\n\
             let o_idx = oy * OW + ox;\n\
             let hw_out = OH * OW;\n\
             let n_idx = nc / C;\n\
             let c_idx = nc % C;\n\
             let col_batch_base = n_idx * C * {kh}u * {kw}u * hw_out;\n\
             for (var ky = 0u; ky < {kh}u; ky += 1u) {{\n\
                 let sy = lid.y * {sh}u + ky * {dh}u;\n\
                 for (var kx = 0u; kx < {kw}u; kx += 1u) {{\n\
                     let sx = lid.x * {sw}u + kx * {dw}u;\n\
                     let row = (c_idx * {kh}u + ky) * {kw}u + kx;\n\
                     COL[col_batch_base + row * hw_out + o_idx] = T(sh_in[sy * {tile_iw}u + sx]);\n\
                 }}\n\
             }}\n\
         }}\n"
    )
}

/// Generic fallback `im2col` for extreme windows that exceed the shared-memory tile budget.
/// Params: 0 = `total`, 1 = `C`, 2 = `H`, 3 = `W`, 4 = `OH`, 5 = `OW`, 6 = `KH`, 7 = `KW`,
/// 8 = `SH`, 9 = `SW`, 10 = `PH` (i32), 11 = `PW` (i32), 12 = `DH`, 13 = `DW`.
const IM2COL_GENERIC_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<T>;
@group(0) @binding(2) var<storage, read_write> COL: array<T>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let idx = linear_id(wid, nwg, lid, 256u);
    let total = pu(0u);
    if (idx >= total) { return; }
    let C = pu(1u); let H = pu(2u); let W = pu(3u);
    let OH = pu(4u); let OW = pu(5u);
    let KH = pu(6u); let KW = pu(7u);
    let SH = pu(8u); let SW = pu(9u);
    let PH = pi(10u); let PW = pi(11u);
    let DH = pu(12u); let DW = pu(13u);
    let hw_out = OH * OW;
    let o_idx = idx % hw_out;
    let tmp0 = idx / hw_out;
    let row = tmp0 % (C * KH * KW);
    let n = tmp0 / (C * KH * KW);
    let oy = o_idx / OW;
    let ox = o_idx % OW;
    let kx = row % KW;
    let tmp1 = row / KW;
    let ky = tmp1 % KH;
    let c = tmp1 / KH;
    let iy = i32(oy * SH + ky * DH) - PH;
    let ix = i32(ox * SW + kx * DW) - PW;
    var v = 0.0;
    if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
        v = f32(X[((n * C + c) * H + u32(iy)) * W + u32(ix)]);
    }
    COL[idx] = T(v);
}
"#;

/// Atomic-free shared-memory tiled gather `col2im` (`16 × 16` input pixel tile per workgroup, 1 KiB shared mem).
/// Params: 0 = `total_tiles`, 1 = `C`, 2 = `H`, 3 = `W`, 4 = `OH`, 5 = `OW`, 6 = `KH`, 7 = `KW`,
/// 8 = `SH`, 9 = `SW`, 10 = `PH` (i32), 11 = `PW` (i32), 12 = `DH`, 13 = `DW`, 14 = `tiles_x`, 15 = `tiles_y`.
const COL2IM_TILED_SRC: &str = r#"
@group(0) @binding(1) var<storage, read> COL: array<T>;
@group(0) @binding(2) var<storage, read_write> X: array<T>;

var<workgroup> sh_col: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_id = wid.y * nwg.x + wid.x;
    let valid_wg = tile_id < pu(0u);
    let C = pu(1u); let H = pu(2u); let W = pu(3u);
    let OH = pu(4u); let OW = pu(5u);
    let KH = pu(6u); let KW = pu(7u);
    let SH = i32(pu(8u)); let SW = i32(pu(9u));
    let PH = pi(10u); let PW = pi(11u);
    let DH = i32(pu(12u)); let DW = i32(pu(13u));
    let tiles_x = pu(14u); let tiles_y = pu(15u);
    let tx = tile_id % tiles_x;
    let tmp = tile_id / tiles_x;
    let ty = tmp % tiles_y;
    let nc = tmp / tiles_y;
    let iy0 = ty * 16u;
    let ix0 = tx * 16u;
    let iy = iy0 + lid.y;
    let ix = ix0 + lid.x;
    let n_idx = nc / C;
    let c_idx = nc % C;
    let hw_out = OH * OW;
    let col_batch_base = n_idx * C * KH * KW * hw_out;
    let oh_i = i32(OH);
    let ow_i = i32(OW);

    var acc = 0.0;
    for (var ky = 0u; ky < KH; ky += 1u) {
        let num_y0 = i32(iy0) + PH - i32(ky) * DH;
        let num_y1 = num_y0 + 15;
        let oy_min = select(0, (num_y0 + SH - 1) / SH, num_y0 > 0);
        let oy_max_raw = select(-1, num_y1 / SH, num_y1 >= 0);
        let oy_max = min(oy_max_raw, oh_i - 1);
        if (oy_min > oy_max) { continue; }
        let oy_cnt = u32(oy_max - oy_min + 1);
        let ty_cur = i32(iy) + PH - i32(ky) * DH;
        let y_ok = ty_cur >= 0 && (ty_cur % SH == 0);
        let oy_cur = ty_cur / SH;

        for (var kx = 0u; kx < KW; kx += 1u) {
            let num_x0 = i32(ix0) + PW - i32(kx) * DW;
            let num_x1 = num_x0 + 15;
            let ox_min = select(0, (num_x0 + SW - 1) / SW, num_x0 > 0);
            let ox_max_raw = select(-1, num_x1 / SW, num_x1 >= 0);
            let ox_max = min(ox_max_raw, ow_i - 1);
            if (ox_min > ox_max) { continue; }
            let ox_cnt = u32(ox_max - ox_min + 1);
            let row = (c_idx * KH + ky) * KW + kx;
            let col_row_base = col_batch_base + row * hw_out;

            if (valid_wg && lid.y < oy_cnt && lid.x < ox_cnt) {
                let goy = u32(oy_min + i32(lid.y));
                let gox = u32(ox_min + i32(lid.x));
                sh_col[lid.y * 16u + lid.x] = f32(COL[col_row_base + goy * OW + gox]);
            }
            workgroupBarrier();

            if (valid_wg && y_ok && oy_cur >= oy_min && oy_cur <= oy_max) {
                let tx_cur = i32(ix) + PW - i32(kx) * DW;
                if (tx_cur >= 0 && (tx_cur % SW == 0)) {
                    let ox_cur = tx_cur / SW;
                    if (ox_cur >= ox_min && ox_cur <= ox_max) {
                        acc += sh_col[u32(oy_cur - oy_min) * 16u + u32(ox_cur - ox_min)];
                    }
                }
            }
            workgroupBarrier();
        }
    }

    if (valid_wg && iy < H && ix < W) {
        X[nc * H * W + iy * W + ix] = T(acc);
    }
}
"#;

use Access::{Read as R, ReadWrite as RW};

static IM2COL_TILED: KernelDef =
    KernelDef { name: "im2col_tiled", storage: &[R, RW], source: Source::Generated(gen_im2col_tiled) };
static IM2COL_GENERIC: KernelDef =
    KernelDef { name: "im2col_generic", storage: &[R, RW], source: Source::Static(IM2COL_GENERIC_SRC) };
static COL2IM_TILED: KernelDef =
    KernelDef { name: "col2im_tiled", storage: &[R, RW], source: Source::Static(COL2IM_TILED_SRC) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&IM2COL_TILED, &IM2COL_GENERIC, &COL2IM_TILED];

fn validate_im2col(
    ctx: &GpuContext,
    d: &Im2ColDesc,
    x: &wgpu::Buffer,
    col: &wgpu::Buffer,
    op: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    check_dtype(ctx, d.dtype, op)?;
    d.win.check_limits()?;
    let (oh, ow) = d.output_hw()?;
    let x_elems = prod(&[d.n, d.c, d.h, d.w])?;
    let col_elems = prod(&[d.n, d.c, d.win.kernel.0, d.win.kernel.1, oh, ow])?;
    check_buf(x, x_elems, d.dtype, op)?;
    check_buf(col, col_elems, d.dtype, op)?;
    if x_elems > u32::MAX as usize || col_elems > u32::MAX as usize {
        return Err(GpuError::Limit(format!("{op}: tensor exceeds u32 indexing")));
    }
    Ok((oh, ow, x_elems, col_elems))
}

/// Dispatches tiled `im2col`: unfolds NCHW `x` `[N, C, H, W]` into `col` `[N, C * KH * KW, OH * OW]`.
pub fn im2col(ctx: &GpuContext, d: &Im2ColDesc, x: &wgpu::Buffer, col: &wgpu::Buffer) -> Result<()> {
    let (oh, ow, _, col_elems) = validate_im2col(ctx, d, x, col, "im2col")?;
    if col_elems == 0 {
        return Ok(());
    }
    if x == col {
        return Err(shape_err("im2col: input and output buffers must not alias"));
    }
    if d.tile_in_hw().is_some() {
        let tiles_x = ow.div_ceil(16);
        let tiles_y = oh.div_ceil(16);
        let total_tiles = prod(&[d.n, d.c, tiles_y, tiles_x])?;
        if total_tiles <= u32::MAX as usize {
            let k = ctx.kernel(
                &IM2COL_TILED,
                Spec::new()
                    .u32("KH", d.win.kernel.0 as u32)
                    .u32("KW", d.win.kernel.1 as u32)
                    .u32("SH", d.win.stride.0 as u32)
                    .u32("SW", d.win.stride.1 as u32)
                    .u32("DH", d.win.dilation.0 as u32)
                    .u32("DW", d.win.dilation.1 as u32)
                    .f16(d.dtype == DType::F16),
            )?;
            let p = Params::new()
                .uz(total_tiles)?
                .uz(d.c)?
                .uz(d.h)?
                .uz(d.w)?
                .uz(oh)?
                .uz(ow)?
                .uz(d.win.padding.0)?
                .uz(d.win.padding.1)?
                .uz(tiles_x)?
                .uz(tiles_y)?;
            return ctx.dispatch(&k, &[x, col], &p, ctx.grid_1d(total_tiles as u64 * 256, 256)?);
        }
    }
    let k = ctx.kernel(&IM2COL_GENERIC, Spec::new().f16(d.dtype == DType::F16))?;
    let p = Params::new()
        .uz(col_elems)?
        .uz(d.c)?
        .uz(d.h)?
        .uz(d.w)?
        .uz(oh)?
        .uz(ow)?
        .uz(d.win.kernel.0)?
        .uz(d.win.kernel.1)?
        .uz(d.win.stride.0)?
        .uz(d.win.stride.1)?
        .uz(d.win.padding.0)?
        .uz(d.win.padding.1)?
        .uz(d.win.dilation.0)?
        .uz(d.win.dilation.1)?;
    ctx.dispatch(&k, &[x, col], &p, ctx.grid_1d(col_elems as u64, 256)?)
}

/// Dispatches atomic-free tiled gather `col2im`: folds `col` `[N, C * KH * KW, OH * OW]` back into NCHW `x`
/// `[N, C, H, W]` (overwriting `x` deterministically without atomics).
pub fn col2im(ctx: &GpuContext, d: &Im2ColDesc, col: &wgpu::Buffer, x: &wgpu::Buffer) -> Result<()> {
    let (oh, ow, x_elems, _) = validate_im2col(ctx, d, x, col, "col2im")?;
    if x_elems == 0 {
        return Ok(());
    }
    if x == col {
        return Err(shape_err("col2im: input and output buffers must not alias"));
    }
    let tiles_x = d.w.div_ceil(16);
    let tiles_y = d.h.div_ceil(16);
    let total_tiles = prod(&[d.n, d.c, tiles_y, tiles_x])?;
    if total_tiles > u32::MAX as usize {
        return Err(GpuError::Limit("col2im: tile count exceeds u32 indexing".into()));
    }
    let k = ctx.kernel(&COL2IM_TILED, Spec::new().f16(d.dtype == DType::F16))?;
    let p = Params::new()
        .uz(total_tiles)?
        .uz(d.c)?
        .uz(d.h)?
        .uz(d.w)?
        .uz(oh)?
        .uz(ow)?
        .uz(d.win.kernel.0)?
        .uz(d.win.kernel.1)?
        .uz(d.win.stride.0)?
        .uz(d.win.stride.1)?
        .uz(d.win.padding.0)?
        .uz(d.win.padding.1)?
        .uz(d.win.dilation.0)?
        .uz(d.win.dilation.1)?
        .uz(tiles_x)?
        .uz(tiles_y)?;
    ctx.dispatch(&k, &[col, x], &p, ctx.grid_1d(total_tiles as u64 * 256, 256)?)
}
