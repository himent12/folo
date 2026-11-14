//! Candidate-first detection on the GPU (docs/MODEL.md §9.5): a level's class pass lists its candidates and the
//! pixels their box branch reads ([`class_sites`]), the box convs run at those sites only
//! ([`super::conv::conv2d_sites`], with [`scatter_sites`] writing the first conv's values back into a map for the
//! second), and [`decode_sites`] decodes the candidates' boxes into one compact list per image, the same records
//! [`super::vision::detect_candidates`] makes. Every list has a fixed capacity, so the dispatches do not depend on
//! how many candidates an image has; a level whose candidates did not fit is flagged in the output.

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::kernels::prod;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

static CLASS: KernelDef = KernelDef {
    name: "sites_class",
    storage: &[Access::Read, Access::ReadWrite, Access::ReadWrite, Access::ReadWrite, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/sites_class.wgsl")),
};
static SCATTER: KernelDef = KernelDef {
    name: "sites_scatter",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/sites_scatter.wgsl")),
};
static DECODE: KernelDef = KernelDef {
    name: "sites_decode",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/sites_decode.wgsl")),
};

/// Words per record of [`decode_sites`] (as [`super::vision::RECORD_WORDS`]).
pub const RECORD_WORDS: usize = 7;
/// Words before an image's records in [`decode_sites`]' output: the record count and the overflow bits.
pub const HEADER_WORDS: usize = 2;

/// One level's map and lists for [`class_sites`]: `b` images of `h × w` pixels, `nc` classes, candidate lists of
/// `cap` slots and field lists of `fcap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SiteLists {
    /// Images.
    pub b: usize,
    /// Map height.
    pub h: usize,
    /// Map width.
    pub w: usize,
    /// Classes.
    pub nc: usize,
    /// Candidate slots per image.
    pub cap: usize,
    /// Field slots per image (`9 · cap` or `h · w`, whichever is smaller, never overflows).
    pub fcap: usize,
}

/// The class pass of one level: from `cls` (`[B, nc, h, w]` logits, `dtype`) every pixel whose best class score
/// (first maximum of the sigmoids) is `> conf`, into `sites` (u32 `[B, 1 + cap]`: the count of all candidates,
/// then the pixels of the first `cap`, in arrival order) and `recs` (u32 `[B, 2·cap]`: class and score bits per
/// slot), and the 3×3 windows of the slotted candidates, each pixel once, into `field` (u32 `[B, 1 + fcap]`).
/// `mark` (u32 `[B, h·w]`) is scratch. Clears `sites`, `field` and `mark` first.
#[allow(clippy::too_many_arguments)]
pub fn class_sites(
    ctx: &GpuContext,
    cls: &wgpu::Buffer,
    l: &SiteLists,
    conf: f32,
    dtype: DType,
    (sites, recs, field, mark): (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
) -> Result<()> {
    check_dtype(ctx, dtype, "class_sites")?;
    let hw = prod(&[l.h, l.w])?;
    if hw == 0 || l.nc == 0 || l.cap == 0 || l.fcap == 0 {
        return Err(shape_err("class_sites: empty level"));
    }
    let bhw = prod(&[l.b, hw])?;
    check_buf(cls, prod(&[bhw, l.nc])?, dtype, "class_sites classes")?;
    let (sites_n, fields_n) = (prod(&[l.b, l.cap + 1])?, prod(&[l.b, l.fcap + 1])?);
    check_buf(sites, sites_n, DType::F32, "class_sites sites")?;
    check_buf(recs, prod(&[l.b, 2 * l.cap])?, DType::F32, "class_sites records")?;
    check_buf(field, fields_n, DType::F32, "class_sites field")?;
    check_buf(mark, bhw, DType::F32, "class_sites mark")?;
    if (bhw * l.nc).max(sites_n).max(fields_n) > u32::MAX as usize {
        return Err(GpuError::Limit("class_sites: level too large".into()));
    }
    if l.b == 0 {
        return Ok(());
    }
    ctx.clear(sites, 4 * sites_n as u64);
    ctx.clear(field, 4 * fields_n as u64);
    ctx.clear(mark, 4 * bhw as u64);
    let k = ctx.kernel(&CLASS, Spec::new().f16(dtype == DType::F16))?;
    let p = Params::new().uz(bhw)?.uz(hw)?.uz(l.nc)?.f(conf).uz(l.w)?.uz(l.h)?.uz(l.cap)?.uz(l.fcap)?;
    ctx.dispatch(&k, &[cls, sites, recs, field, mark], &p, ctx.grid_1d(bhw as u64, 256)?)
}

/// `dst[b][c][pixel] = src[b][c][s]` (`src` `[B, C, cap]`, `dst` `[B, C, hw]`, both `dtype`) for the live slots `s` of
/// `sites` (u32 `[B, 1 + cap]`, as [`class_sites`] makes); the rest of `dst` is left as it is.
#[allow(clippy::too_many_arguments)]
pub fn scatter_sites(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    sites: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    (b, c, cap, hw): (usize, usize, usize, usize),
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "scatter_sites")?;
    let n = prod(&[b, c, cap])?;
    check_buf(src, n, dtype, "scatter_sites source")?;
    check_buf(sites, prod(&[b, cap + 1])?, DType::F32, "scatter_sites sites")?;
    check_buf(dst, prod(&[b, c, hw])?, dtype, "scatter_sites target")?;
    if n == 0 {
        return Ok(());
    }
    if prod(&[b, c, hw])?.max(n) > u32::MAX as usize {
        return Err(GpuError::Limit("scatter_sites: map too large".into()));
    }
    let k = ctx.kernel(&SCATTER, Spec::new().f16(dtype == DType::F16))?;
    let p = Params::new().uz(n)?.uz(cap)?.uz(c)?.uz(hw)?;
    ctx.dispatch(&k, &[src, sites, dst], &p, ctx.grid_1d(n as u64, 256)?)
}

/// One level of [`decode_sites`].
#[derive(Clone, Copy, Debug)]
pub struct DecodeLevel<'a> {
    /// The box logits at the slots, `[B, 4·reg_max, cap]` of the decode's `dtype`.
    pub raw: &'a wgpu::Buffer,
    /// The level's site list, u32 `[B, 1 + cap]` ([`class_sites`]).
    pub sites: &'a wgpu::Buffer,
    /// The level's records, u32 `[B, 2·cap]` ([`class_sites`]).
    pub recs: &'a wgpu::Buffer,
    /// Slots per image.
    pub cap: usize,
    /// Map width.
    pub w: usize,
    /// Stride.
    pub stride: f32,
    /// The level's first anchor.
    pub start: usize,
}

/// Decodes the candidates of every level into `out` (u32 `[B, HEADER_WORDS + 7·Σ cap]`): per image the record
/// count, a word with bit `l` set when level `l` had more candidates than slots, then the records `(anchor,
/// class, score, x1, y1, x2, y2)` (floats as their bits) in arrival order, bit for bit those of
/// [`super::vision::detect_candidates`] on the dense head. Clears `out` first.
pub fn decode_sites(
    ctx: &GpuContext,
    levels: &[DecodeLevel<'_>],
    b: usize,
    rm: usize,
    dtype: DType,
    out: &wgpu::Buffer,
) -> Result<()> {
    check_dtype(ctx, dtype, "decode_sites")?;
    if levels.is_empty() || levels.len() > 32 || rm == 0 {
        return Err(shape_err("decode_sites: 1 to 32 levels and reg_max > 0"));
    }
    let slots: usize = levels.iter().map(|l| l.cap).sum();
    let words = HEADER_WORDS + RECORD_WORDS * slots;
    let total = prod(&[b, words])?;
    check_buf(out, total, DType::F32, "decode_sites output")?;
    for l in levels {
        check_buf(l.raw, prod(&[b, 4 * rm, l.cap])?, dtype, "decode_sites logits")?;
        check_buf(l.sites, prod(&[b, l.cap + 1])?, DType::F32, "decode_sites sites")?;
        check_buf(l.recs, prod(&[b, 2 * l.cap])?, DType::F32, "decode_sites records")?;
        if l.cap == 0 || l.w == 0 {
            return Err(shape_err("decode_sites: empty level"));
        }
    }
    if total.max(prod(&[b, 4 * rm, slots])?) > u32::MAX as usize {
        return Err(GpuError::Limit("decode_sites: output too large".into()));
    }
    if b == 0 {
        return Ok(());
    }
    ctx.clear(out, 4 * total as u64);
    let k = ctx.kernel(&DECODE, Spec::new().f16(dtype == DType::F16))?;
    for (i, l) in levels.iter().enumerate() {
        let n = b * l.cap;
        let p =
            Params::new().uz(n)?.uz(l.cap)?.uz(rm)?.uz(l.w)?.f(l.stride).uz(l.start)?.uz(words)?.u(1 << i);
        ctx.dispatch(&k, &[l.raw, l.sites, l.recs, out], &p, ctx.grid_1d(n as u64, 256)?)?;
    }
    Ok(())
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&CLASS, &SCATTER, &DECODE];
