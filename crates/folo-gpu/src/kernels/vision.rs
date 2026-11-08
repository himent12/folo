//! Detector pre- and post-processing: letterboxing `u8` RGB images into the network input (f32, or f16 for
//! half-precision inference), and decoding the detection head (f32 or f16, decoded in f32) into a compact list
//! of above-threshold candidates, so that a prediction uploads raw pixels and downloads only what NMS needs.

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

static LETTERBOX: KernelDef = KernelDef {
    name: "letterbox",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/letterbox.wgsl")),
};
static DETECT: KernelDef = KernelDef {
    name: "detect",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/detect.wgsl")),
};

/// One image of [`letterbox_rgb8`]: where it goes on the canvas and how it is sampled (folo-data's
/// `ResizePlan`: box prefilter `prefilter`, then the bilinear taps at word `taps` of the taps buffer: `x0, x1,
/// wx` with `resized[0]` entries each, then `y0, y1, wy` with `resized[1]` each, weights in 1/2048).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterboxDesc {
    /// Canvas `[width, height]`.
    pub out: [usize; 2],
    /// Source `[width, height]`; the source buffer is `height * width * 3` bytes, row-major RGB.
    pub src: [usize; 2],
    /// Top-left corner `[x, y]` of the resized image on the canvas.
    pub pad: [usize; 2],
    /// Resized `[width, height]`.
    pub resized: [usize; 2],
    /// Box-prefilter factors `[fx, fy]`.
    pub prefilter: [usize; 2],
    /// Grey level of the padding.
    pub pad_value: u8,
    /// Factor applied to every output value (`1 / 255` for network input).
    pub scale: f32,
    /// Element offset of this image's `[3, OH, OW]` block in the output.
    pub out_offset: usize,
    /// Word offset of this image's taps.
    pub taps: usize,
}

/// Letterbox one image (see [`LetterboxDesc`]) into `y` of `dtype`. In f32 bit-exact with folo-data's CPU
/// resize followed by `value as f32 * scale`; in f16 that value rounded once.
pub fn letterbox_rgb8(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    taps: &wgpu::Buffer,
    y: &wgpu::Buffer,
    d: &LetterboxDesc,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "letterbox")?;
    let [ow, oh] = d.out;
    let [sw, sh] = d.src;
    let [rw, rh] = d.resized;
    let [fx, fy] = d.prefilter;
    if [ow, oh, sw, sh, rw, rh, fx, fy].contains(&0) || d.pad[0] + rw > ow || d.pad[1] + rh > oh {
        return Err(shape_err(format!("letterbox: invalid geometry {d:?}")));
    }
    // The prefilter sums a block (plus the rounding term) in u32; the plain bilinear sums stay below 2^31.
    if !prefilter_fits(fx, fy) {
        return Err(GpuError::Limit(format!("letterbox: a {fx}x{fy} prefilter block is too large")));
    }
    // Prefiltered taps must index inside the prefiltered image (checked on the host when built).
    check_buf(src, (sw * sh * 3).div_ceil(4), DType::F32, "letterbox source")?;
    check_buf(taps, d.taps + 3 * (rw + rh), DType::F32, "letterbox taps")?;
    let n = ow.checked_mul(oh).ok_or_else(|| shape_err("letterbox: canvas too large"))?;
    check_buf(y, d.out_offset + 3 * n, dtype, "letterbox output")?;
    let k = ctx.kernel(&LETTERBOX, Spec::new().f16(dtype == DType::F16))?;
    let p = Params::new()
        .uz(ow)?
        .uz(oh)?
        .uz(sw)?
        .uz(sh)?
        .uz(d.pad[0])?
        .uz(d.pad[1])?
        .uz(rw)?
        .uz(rh)?
        .uz(fx)?
        .uz(fy)?
        .u(d.pad_value as u32)
        .f(d.scale)
        .uz(d.out_offset)?
        .uz(d.taps)?;
    ctx.dispatch(&k, &[src, taps, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Whether the shader can sum and round an `fx x fy` prefilter block in u32: `255 · n + n / 2` for
/// `n = fx · fy` (the CPU sums in u64).
pub fn prefilter_fits(fx: usize, fy: usize) -> bool {
    (fx as u64)
        .checked_mul(fy as u64)
        .and_then(|n| n.checked_mul(255).and_then(|s| s.checked_add(n / 2)))
        .is_some_and(|v| v <= u32::MAX as u64)
}

/// Words per candidate record of [`detect_candidates`].
pub const RECORD_WORDS: usize = 7;

/// Decode `box_raw` (`[B, 4·rm, A]`), `cls_raw` (`[B, nc, A]`, both of `dtype`: f32 or f16, decoded in f32)
/// and `anchors` (f32 `[3, A]`: x, y in grid units, stride) into `out` (u32 `[B, 1 + 7·A]`): per image a
/// count, then that many records `(anchor, class, score, x1, y1, x2, y2)` in arrival order, one per anchor
/// whose best class score is `> conf`. `out` is cleared first.
#[allow(clippy::too_many_arguments)]
pub fn detect_candidates(
    ctx: &GpuContext,
    box_raw: &wgpu::Buffer,
    cls_raw: &wgpu::Buffer,
    anchors: &wgpu::Buffer,
    out: &wgpu::Buffer,
    batch: usize,
    anchors_n: usize,
    nc: usize,
    rm: usize,
    conf: f32,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "detect_candidates")?;
    if anchors_n == 0 || nc == 0 || rm == 0 {
        return Err(shape_err("detect_candidates: empty head"));
    }
    let ba = batch.checked_mul(anchors_n).ok_or_else(|| shape_err("detect_candidates: too many anchors"))?;
    check_buf(box_raw, ba * 4 * rm, dtype, "detect box_raw")?;
    check_buf(cls_raw, ba * nc, dtype, "detect cls_raw")?;
    check_buf(anchors, 3 * anchors_n, DType::F32, "detect anchors")?;
    let words = batch * (1 + RECORD_WORDS * anchors_n);
    check_buf(out, words, DType::F32, "detect output")?;
    // Every index the shader forms must fit in u32.
    if (words as u64).max((ba * nc.max(4 * rm)) as u64) > u32::MAX as u64 {
        return Err(GpuError::Limit("detect_candidates: head too large".into()));
    }
    if batch == 0 {
        return Ok(());
    }
    ctx.clear(out, 4 * words as u64);
    let k = ctx.kernel(&DETECT, Spec::new().f16(dtype == DType::F16))?;
    let p = Params::new().uz(ba)?.uz(anchors_n)?.uz(nc)?.uz(rm)?.f(conf);
    ctx.dispatch(&k, &[box_raw, cls_raw, anchors, out], &p, ctx.grid_1d(ba as u64, 256)?)
}

#[cfg(test)]
mod tests {
    use super::prefilter_fits;

    #[test]
    fn prefilter_guard_counts_the_rounding_term() {
        assert!(prefilter_fits(1, 1));
        // 4100² · 255 + 4100² / 2 = 4 294 955 000 fits; 4101² · 255 fits but adding the rounding term wraps.
        assert!(prefilter_fits(4100, 4100));
        assert!((4101u64 * 4101 * 255) <= u32::MAX as u64);
        assert!(!prefilter_fits(4101, 4101));
        assert!(!prefilter_fits(usize::MAX, 2));
    }
}
