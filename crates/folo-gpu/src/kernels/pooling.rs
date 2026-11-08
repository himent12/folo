//! 2-D max pooling (NCHW).

use super::{check_buf, prod};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Window geometry shared by pooling and convolution: kernel, stride, padding and dilation per axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window2d {
    /// Kernel size `(h, w)`.
    pub kernel: (usize, usize),
    /// Stride `(h, w)`.
    pub stride: (usize, usize),
    /// Zero padding `(h, w)` on each side.
    pub padding: (usize, usize),
    /// Dilation `(h, w)`.
    pub dilation: (usize, usize),
}

impl Window2d {
    /// Square window with the given kernel, stride and padding, dilation 1.
    pub fn square(kernel: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel: (kernel, kernel),
            stride: (stride, stride),
            padding: (padding, padding),
            dilation: (1, 1),
        }
    }

    /// Output size for an `h × w` input (floor mode, as PyTorch with `ceil_mode = false`).
    pub fn output(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let axis = |n: usize, k: usize, s: usize, p: usize, d: usize| -> Result<usize> {
            if k == 0 || s == 0 || d == 0 {
                return Err(shape_err(format!(
                    "window {self:?}: kernel, stride and dilation must be non-zero"
                )));
            }
            let span = d.checked_mul(k - 1).and_then(|v| v.checked_add(1));
            let padded = p.checked_mul(2).and_then(|v| v.checked_add(n));
            let (Some(span), Some(padded)) = (span, padded) else {
                return Err(shape_err(format!("window {self:?} overflows for a {h}x{w} input")));
            };
            if padded < span {
                return Err(shape_err(format!("window {self:?} does not fit a {h}x{w} input")));
            }
            Ok((padded - span) / s + 1)
        };
        Ok((
            axis(h, self.kernel.0, self.stride.0, self.padding.0, self.dilation.0)?,
            axis(w, self.kernel.1, self.stride.1, self.padding.1, self.dilation.1)?,
        ))
    }

    pub(crate) fn spec(&self) -> Spec {
        Spec::new()
            .u32("KH", self.kernel.0 as u32)
            .u32("KW", self.kernel.1 as u32)
            .u32("SH", self.stride.0 as u32)
            .u32("SW", self.stride.1 as u32)
            .i32("PH", self.padding.0 as i32)
            .i32("PW", self.padding.1 as i32)
            .u32("DH", self.dilation.0 as u32)
            .u32("DW", self.dilation.1 as u32)
    }

    pub(crate) fn check_limits(&self) -> Result<()> {
        let all = [self.kernel, self.stride, self.padding, self.dilation];
        if all.iter().any(|&(a, b)| a > 1 << 16 || b > 1 << 16) {
            return Err(shape_err(format!("window {self:?} is implausibly large")));
        }
        Ok(())
    }
}

static MAXPOOL: KernelDef = KernelDef {
    name: "max_pool2d",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/maxpool.wgsl")),
};

/// `max_pool2d` over `planes = N·C` planes of `h × w`; padded positions are ignored (−∞).
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    planes: usize,
    h: usize,
    w: usize,
    win: &Window2d,
    dtype: DType,
) -> Result<(usize, usize)> {
    check_dtype(ctx, dtype, "max_pool2d")?;
    win.check_limits()?;
    if win.padding.0 * 2 > win.kernel.0 || win.padding.1 * 2 > win.kernel.1 {
        return Err(shape_err(format!("max_pool2d: padding must be at most half the kernel, got {win:?}")));
    }
    let (oh, ow) = win.output(h, w)?;
    check_buf(x, prod(&[planes, h, w])?, dtype, "max_pool2d input")?;
    check_buf(y, prod(&[planes, oh, ow])?, dtype, "max_pool2d output")?;
    let n = planes * oh * ow;
    if n == 0 {
        return Ok((oh, ow));
    }
    let k = ctx.kernel(&MAXPOOL, win.spec().f16(dtype == DType::F16))?;
    let p = Params::new().uz(n)?.uz(h)?.uz(w)?.uz(oh)?.uz(ow)?.f(f32::NEG_INFINITY);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)?;
    Ok((oh, ow))
}
