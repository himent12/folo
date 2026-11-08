//! Elementwise kernels: unary ops, broadcasting binary ops and fused bias + activation.

use super::{UNARY_FN, check_buf, vec_prelude};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Elementwise unary operations (also used as fused activations).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    /// `x`
    Identity,
    /// `e^x`
    Exp,
    /// `ln x`
    Log,
    /// `√x`
    Sqrt,
    /// `1/√x`
    Rsqrt,
    /// `-x`
    Neg,
    /// `|x|`
    Abs,
    /// `1 / (1 + e^-x)`
    Sigmoid,
    /// `x · sigmoid(x)`
    Silu,
    /// `max(x, 0)`
    Relu,
    /// `tanh x`
    Tanh,
    /// GELU, tanh approximation.
    Gelu,
    /// `x` if `x ≥ 0` else `alpha · x`
    LeakyRelu(f32),
    /// `x · relu6(x + 3) / 6`
    Hardswish,
    /// `relu6(x + 3) / 6`
    Hardsigmoid,
    /// `x²`
    Square,
    /// `1/x`
    Recip,
    /// `sin x`
    Sin,
    /// `cos x`
    Cos,
    /// `⌊x⌋`
    Floor,
    /// `⌈x⌉`
    Ceil,
    /// `min(max(x, 0), 6)`
    Relu6,
}

impl UnaryOp {
    /// Op code in `shaders/unary_fn.wgsl`.
    pub(crate) fn code(self) -> u32 {
        match self {
            UnaryOp::Identity => 0,
            UnaryOp::Exp => 1,
            UnaryOp::Log => 2,
            UnaryOp::Sqrt => 3,
            UnaryOp::Rsqrt => 4,
            UnaryOp::Neg => 5,
            UnaryOp::Abs => 6,
            UnaryOp::Sigmoid => 7,
            UnaryOp::Silu => 8,
            UnaryOp::Relu => 9,
            UnaryOp::Tanh => 10,
            UnaryOp::Gelu => 11,
            UnaryOp::LeakyRelu(_) => 12,
            UnaryOp::Hardswish => 13,
            UnaryOp::Hardsigmoid => 14,
            UnaryOp::Square => 15,
            UnaryOp::Recip => 16,
            UnaryOp::Sin => 17,
            UnaryOp::Cos => 18,
            UnaryOp::Floor => 19,
            UnaryOp::Ceil => 20,
            UnaryOp::Relu6 => 21,
        }
    }

    /// The op's scalar parameter (only `LeakyRelu` has one).
    pub(crate) fn alpha(self) -> f32 {
        match self {
            UnaryOp::LeakyRelu(a) => a,
            _ => 0.0,
        }
    }

    /// CPU reference, used by tests and as documentation of the exact semantics.
    pub fn apply(self, x: f32) -> f32 {
        let tanh = |v: f32| if v.abs() >= 9.0 { v.signum() } else { v.tanh() };
        match self {
            UnaryOp::Identity => x,
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Rsqrt => 1.0 / x.sqrt(),
            UnaryOp::Neg => -x,
            UnaryOp::Abs => x.abs(),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Silu => x / (1.0 + (-x).exp()),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Tanh => tanh(x),
            UnaryOp::Gelu => 0.5 * x * (1.0 + tanh(0.797_884_6 * (x + 0.044715 * x * x * x))),
            UnaryOp::LeakyRelu(a) => {
                if x >= 0.0 {
                    x
                } else {
                    a * x
                }
            }
            UnaryOp::Hardswish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            UnaryOp::Hardsigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
            UnaryOp::Square => x * x,
            UnaryOp::Recip => 1.0 / x,
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Floor => x.floor(),
            UnaryOp::Ceil => x.ceil(),
            UnaryOp::Relu6 => x.clamp(0.0, 6.0),
        }
    }
}

/// Elementwise binary operations with NumPy broadcasting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `max(a, b)`
    Max,
    /// `min(a, b)`
    Min,
    /// `a ^ b` (C `powf` semantics: negative bases with integer exponents are allowed)
    Pow,
}

impl BinaryOp {
    fn code(self) -> u32 {
        self as u32
    }

    /// CPU reference.
    pub fn apply(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Max => a.max(b),
            BinaryOp::Min => a.min(b),
            BinaryOp::Pow => a.powf(b),
        }
    }
}

fn gen_unary(spec: &Spec) -> String {
    format!("{}{}{}", vec_prelude(spec.get_bool("VEC4")), UNARY_FN, include_str!("../shaders/unary.wgsl"))
}

fn gen_binary(spec: &Spec) -> String {
    format!("{}{}", vec_prelude(spec.get_bool("VEC4")), include_str!("../shaders/binary.wgsl"))
}

fn gen_bias_act(spec: &Spec) -> String {
    format!("{}{}{}", vec_prelude(spec.get_bool("VEC4")), UNARY_FN, include_str!("../shaders/bias_act.wgsl"))
}

static UNARY: KernelDef = KernelDef {
    name: "unary",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_unary),
};
static BINARY: KernelDef = KernelDef {
    name: "binary",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_binary),
};
static BIAS_ACT: KernelDef = KernelDef {
    name: "bias_act",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_bias_act),
};

/// Maximum rank after broadcasting dims are coalesced.
pub const MAX_RANK: usize = 8;

/// `y[i] = op(x[i])` for `n` f32 elements.
pub fn unary(ctx: &GpuContext, op: UnaryOp, x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize) -> Result<()> {
    check_buf(x, n, DType::F32, "unary input")?;
    check_buf(y, n, DType::F32, "unary output")?;
    let vec = n.is_multiple_of(4);
    let units = if vec { n / 4 } else { n };
    let k = ctx.kernel(&UNARY, Spec::new().u32("OP", op.code()).bool("VEC4", vec))?;
    let p = Params::new().uz(units)?.f(op.alpha());
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// `y = act(x + bias[c])` where `c = (i / inner) % channels`; `bias = None` applies only the activation.
///
/// For NCHW activations `channels = C` and `inner = H·W`; for `[rows, features]` use `inner = 1`.
#[allow(clippy::too_many_arguments)]
pub fn bias_act(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    n: usize,
    channels: usize,
    inner: usize,
    act: UnaryOp,
) -> Result<()> {
    check_buf(x, n, DType::F32, "bias_act input")?;
    check_buf(y, n, DType::F32, "bias_act output")?;
    if let Some(b) = bias {
        if channels == 0 || inner == 0 {
            return Err(shape_err("bias_act: channels and inner must be non-zero"));
        }
        check_buf(b, channels, DType::F32, "bias_act bias")?;
    }
    let vec = n.is_multiple_of(4);
    let units = if vec { n / 4 } else { n };
    let k = ctx.kernel(
        &BIAS_ACT,
        Spec::new().u32("OP", act.code()).bool("HAS_BIAS", bias.is_some()).bool("VEC4", vec),
    )?;
    let p = Params::new().uz(units)?.f(act.alpha()).uz(channels.max(1))?.uz(inner.max(1))?;
    ctx.dispatch(&k, &[x, bias.unwrap_or(ctx.dummy()), y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// NumPy broadcast of two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let r = a.len().max(b.len());
    let mut out = vec![0; r];
    for i in 0..r {
        let da = if i + a.len() >= r { a[i + a.len() - r] } else { 1 };
        let db = if i + b.len() >= r { b[i + b.len() - r] } else { 1 };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(shape_err(format!("shapes {a:?} and {b:?} do not broadcast"))),
        };
    }
    Ok(out)
}

/// Contiguous strides of `shape` aligned to `out` (rank `r`), with 0 for broadcast dims.
fn bcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let r = out.len();
    let mut st = vec![0; r];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        let oi = i + r - shape.len();
        st[oi] = if shape[i] == 1 { 0 } else { acc };
        acc *= shape[i];
    }
    st
}

/// Coalesces `(dims, a strides, b strides)`: drops size-1 dims and merges neighbours that are contiguous in
/// both operands. Returns innermost-last vectors.
fn coalesce(out: &[usize], sa: &[usize], sb: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut d: Vec<usize> = Vec::new();
    let mut a: Vec<usize> = Vec::new();
    let mut b: Vec<usize> = Vec::new();
    for i in 0..out.len() {
        if out[i] == 1 {
            continue;
        }
        if let (Some(&ld), Some(&la), Some(&lb)) = (d.last(), a.last(), b.last())
            && la == sa[i] * out[i]
            && lb == sb[i] * out[i]
        {
            let n = d.len() - 1;
            d[n] = ld * out[i];
            a[n] = sa[i];
            b[n] = sb[i];
            continue;
        }
        d.push(out[i]);
        a.push(sa[i]);
        b.push(sb[i]);
    }
    (d, a, b)
}

/// `out = a (op) b` with NumPy broadcasting; `out` must hold `broadcast_shape(a, b)` elements.
pub fn binary(
    ctx: &GpuContext,
    op: BinaryOp,
    a: &wgpu::Buffer,
    a_shape: &[usize],
    b: &wgpu::Buffer,
    b_shape: &[usize],
    out: &wgpu::Buffer,
) -> Result<Vec<usize>> {
    let oshape = broadcast_shape(a_shape, b_shape)?;
    let n: usize = oshape.iter().product();
    check_buf(a, a_shape.iter().product(), DType::F32, "binary lhs")?;
    check_buf(b, b_shape.iter().product(), DType::F32, "binary rhs")?;
    check_buf(out, n, DType::F32, "binary output")?;
    if n == 0 {
        return Ok(oshape);
    }
    let (dims, sa, sb) =
        coalesce(&oshape, &bcast_strides(a_shape, &oshape), &bcast_strides(b_shape, &oshape));
    let same = dims.len() <= 1 && sa.iter().chain(&sb).all(|&s| s == 1);
    if dims.len() > MAX_RANK {
        return Err(shape_err(format!("binary: rank {} after coalescing exceeds {MAX_RANK}", dims.len())));
    }
    let vec = same && n.is_multiple_of(4);
    let units = if vec { n / 4 } else { n };
    let rank = if same { 0 } else { dims.len() as u32 };
    let k = ctx.kernel(&BINARY, Spec::new().u32("OP", op.code()).u32("RANK", rank).bool("VEC4", vec))?;
    let mut p = Params::new().uz(units)?.at(4);
    for i in 0..MAX_RANK {
        p = p.uz(dims.get(i).copied().unwrap_or(1))?;
    }
    for i in 0..MAX_RANK {
        p = p.uz(sa.get(i).copied().unwrap_or(0))?;
    }
    for i in 0..MAX_RANK {
        p = p.uz(sb.get(i).copied().unwrap_or(0))?;
    }
    let p = p.f(f32::NAN).f(f32::INFINITY);
    ctx.dispatch(&k, &[a, b, out], &p, ctx.grid_1d(units as u64, 256)?)?;
    Ok(oshape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_bias_pattern() {
        // [N, C, H, W] + [1, C, 1, 1] → [N, C, HW] with b strides [0, 1, 0].
        let out = [2, 3, 4, 5];
        let (d, a, b) = coalesce(&out, &bcast_strides(&out, &out), &bcast_strides(&[1, 3, 1, 1], &out));
        assert_eq!(d, vec![2, 3, 20]);
        assert_eq!(a, vec![60, 20, 1]);
        assert_eq!(b, vec![0, 1, 0]);
        let (d, a, b) = coalesce(&out, &bcast_strides(&out, &out), &bcast_strides(&out, &out));
        assert_eq!((d, a, b), (vec![120], vec![1], vec![1]));
    }

    #[test]
    fn broadcast_shapes() {
        assert_eq!(broadcast_shape(&[2, 1, 4], &[3, 1]).unwrap(), vec![2, 3, 4]);
        assert!(broadcast_shape(&[2, 3], &[4]).is_err());
        assert_eq!(broadcast_shape(&[], &[5]).unwrap(), vec![5]);
    }
}
