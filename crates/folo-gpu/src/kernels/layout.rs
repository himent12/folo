//! Data movement: permute / transpose (copy), concat, nearest upsampling and f32 ↔ f16 casts.

use super::{check_buf, prod, shape_numel, vec_prelude};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::elementwise::MAX_RANK;
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

const RW: &[Access] = &[Access::Read, Access::ReadWrite];

static PERMUTE: KernelDef = KernelDef {
    name: "permute",
    storage: RW,
    source: Source::Static(include_str!("../shaders/permute.wgsl")),
};
static TRANSPOSE: KernelDef = KernelDef {
    name: "transpose",
    storage: RW,
    source: Source::Static(include_str!("../shaders/transpose.wgsl")),
};
static UPSAMPLE: KernelDef = KernelDef {
    name: "upsample",
    storage: RW,
    source: Source::Static(include_str!("../shaders/upsample.wgsl")),
};
static CAST: KernelDef =
    KernelDef { name: "cast", storage: RW, source: Source::Static(include_str!("../shaders/cast.wgsl")) };

fn gen_copy_rows(spec: &Spec) -> String {
    let vec = spec.get_bool("VEC4");
    // The copy is type-agnostic: move f32 words (or vec4 of them); f16 data is moved in u32 pairs by the
    // caller only when row lengths are even.
    format!("{}{}", vec_prelude(vec), include_str!("../shaders/copy_rows.wgsl"))
}

static COPY_ROWS: KernelDef =
    KernelDef { name: "copy_rows", storage: RW, source: Source::Generated(gen_copy_rows) };

/// Coalesces a permutation: returns `(out dims, input stride per out dim)` with size-1 dims dropped and
/// neighbouring output dims merged when they are also neighbours in the input.
pub fn coalesce_permute(shape: &[usize], perm: &[usize]) -> Result<(Vec<usize>, Vec<usize>)> {
    let r = shape.len();
    let mut seen = vec![false; r];
    if perm.len() != r || perm.iter().any(|&p| p >= r || std::mem::replace(&mut seen[p], true)) {
        return Err(shape_err(format!("{perm:?} is not a permutation of {r} dims")));
    }
    let mut strides = vec![1usize; r];
    for i in (0..r.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut dims: Vec<usize> = Vec::new();
    let mut st: Vec<usize> = Vec::new();
    for &p in perm {
        let (d, s) = (shape[p], strides[p]);
        if d == 1 {
            continue;
        }
        if let (Some(ld), Some(&ls)) = (dims.last_mut(), st.last())
            && ls == s * d
        {
            *ld *= d;
            *st.last_mut().expect("non-empty") = s;
            continue;
        }
        dims.push(d);
        st.push(s);
    }
    Ok((dims, st))
}

/// `y = x.permute(perm)` materialised densely, for a dense `x` of `shape`.
pub fn permute(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    shape: &[usize],
    perm: &[usize],
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "permute")?;
    let n = shape_numel(shape)?;
    check_buf(x, n, dtype, "permute input")?;
    check_buf(y, n, dtype, "permute output")?;
    if n == 0 {
        return Ok(());
    }
    let (dims, st) = coalesce_permute(shape, perm)?;
    let f16 = dtype == DType::F16;
    // Identity after coalescing: a plain copy.
    if dims.len() <= 1 {
        ctx.copy(x, 0, y, 0, dtype.bytes(n));
        return Ok(());
    }
    // [B, R, C] -> [B, C, R]: tiled transpose with coalesced reads and writes.
    let batched_2d = match dims.len() {
        2 => st[0] == 1 && st[1] == dims[0],
        3 => st[0] == dims[1] * dims[2] && st[1] == 1 && st[2] == dims[1],
        _ => false,
    };
    if batched_2d {
        let (b, c, r) = if dims.len() == 2 { (1, dims[0], dims[1]) } else { (dims[0], dims[1], dims[2]) };
        let k = ctx.kernel(&TRANSPOSE, Spec::new().f16(f16))?;
        let p = Params::new().uz(r)?.uz(c)?;
        let groups = [c.div_ceil(32) as u32, r.div_ceil(32) as u32, u32::try_from(b).unwrap_or(u32::MAX)];
        return ctx.dispatch(&k, &[x, y], &p, groups);
    }
    if dims.len() > MAX_RANK {
        return Err(shape_err(format!("permute: rank {} after coalescing exceeds {MAX_RANK}", dims.len())));
    }
    let k = ctx.kernel(&PERMUTE, Spec::new().u32("RANK", dims.len() as u32).f16(f16))?;
    let mut p = Params::new().uz(n)?.at(4);
    for i in 0..MAX_RANK {
        p = p.uz(dims.get(i).copied().unwrap_or(1))?;
    }
    for i in 0..MAX_RANK {
        p = p.uz(st.get(i).copied().unwrap_or(0))?;
    }
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Copies `outer` rows of `row` elements from dense `x` into `y` at `y[o * dst_row + dst_off ..]` (concat).
#[allow(clippy::too_many_arguments)]
pub fn copy_rows(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    row: usize,
    dst_row: usize,
    dst_off: usize,
    dtype: DType,
) -> Result<()> {
    if dst_off.checked_add(row).is_none_or(|end| end > dst_row) {
        return Err(shape_err("copy_rows: row overruns the destination row"));
    }
    check_buf(x, prod(&[outer, row])?, dtype, "concat input")?;
    check_buf(y, prod(&[outer, dst_row])?, dtype, "concat output")?;
    if outer * row == 0 {
        return Ok(());
    }
    // Work in 32-bit words: f16 rows must then be even-length and start on even offsets.
    let (row, dst_row, dst_off) = match dtype {
        DType::F32 => (row, dst_row, dst_off),
        DType::F16 if row.is_multiple_of(2) && dst_row.is_multiple_of(2) && dst_off.is_multiple_of(2) => {
            (row / 2, dst_row / 2, dst_off / 2)
        }
        DType::F16 => return Err(GpuError::Unsupported("concat of f16 blocks with an odd length or offset")),
    };
    if outer == 1 {
        // One contiguous block: a buffer copy is fastest.
        ctx.copy(x, 0, y, (dst_off * 4) as u64, (row * 4) as u64);
        return Ok(());
    }
    let vec = row % 4 == 0 && dst_row % 4 == 0 && dst_off % 4 == 0;
    let q = if vec { 4 } else { 1 };
    let k = ctx.kernel(&COPY_ROWS, Spec::new().bool("VEC4", vec))?;
    let units = outer * row / q;
    let p = Params::new().uz(units)?.uz(row / q)?.uz(dst_row / q)?.uz(dst_off / q)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Nearest-neighbour upsampling of `planes` NCHW planes from `h × w` to `oh × ow`.
#[allow(clippy::too_many_arguments)]
pub fn upsample_nearest2d(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    planes: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "upsample_nearest2d")?;
    check_buf(x, prod(&[planes, h, w])?, dtype, "upsample input")?;
    check_buf(y, prod(&[planes, oh, ow])?, dtype, "upsample output")?;
    let n = planes * oh * ow;
    if n == 0 {
        return Ok(());
    }
    if h == 0 || w == 0 {
        return Err(shape_err("upsample: empty input plane"));
    }
    // The shader computes `dst * in` in u32.
    if (oh as u64) * (h as u64) > u32::MAX as u64 || (ow as u64) * (w as u64) > u32::MAX as u64 {
        return Err(shape_err(format!("upsample {h}x{w} -> {oh}x{ow} is too large")));
    }
    let k = ctx.kernel(&UPSAMPLE, Spec::new().f16(dtype == DType::F16))?;
    let p = Params::new().uz(n)?.uz(h)?.uz(w)?.uz(oh)?.uz(ow)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Converts `n` elements between f32 and f16 (`to_f16` picks the direction). Needs no `shader-f16`.
pub fn cast(ctx: &GpuContext, x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize, to_f16: bool) -> Result<()> {
    let (src, dst) = if to_f16 { (DType::F32, DType::F16) } else { (DType::F16, DType::F32) };
    check_buf(x, n, src, "cast input")?;
    check_buf(y, n, dst, "cast output")?;
    if n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(&CAST, Spec::new().bool("TO_F16", to_f16))?;
    let p = Params::new().uz(n)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n.div_ceil(2) as u64, 256)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&PERMUTE, &TRANSPOSE, &UPSAMPLE, &CAST, &COPY_ROWS];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_permutes() {
        // NCHW -> NHWC: [N, C, HW] -> [N, HW, C].
        let (d, s) = coalesce_permute(&[2, 3, 4, 5], &[0, 2, 3, 1]).unwrap();
        assert_eq!(d, vec![2, 20, 3]);
        assert_eq!(s, vec![60, 1, 20]);
        let (d, s) = coalesce_permute(&[2, 3, 4], &[0, 1, 2]).unwrap();
        assert_eq!((d, s), (vec![24], vec![1]));
        assert!(coalesce_permute(&[2, 3], &[0, 0]).is_err());
    }
}
