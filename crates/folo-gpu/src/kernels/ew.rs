//! Strided elementwise kernels with folo-core's op set: unary ops and their input gradients, binary ops and
//! comparisons, and `where`.
//!
//! Every input is a strided view (offset + strides, stride 0 broadcasts) of the output's shape, so broadcast
//! and transposed operands are read in place instead of being copied first. The shape is coalesced across all
//! operands; when every input is contiguous the kernel indexes linearly (with `vec4` loads when sizes and
//! offsets allow), otherwise it walks up to [`MAX_RANK`] dims. Semantics follow folo-core's CPU kernels: stable
//! sigmoid / softplus, NaN-ignoring `max` / `min` (Rust's `f32::max`), C `powf`, rounding half away from zero.
//!
//! Values are stored as f32 or (with `shader-f16`) f16: every value operand and the output share one storage
//! type, the arithmetic is f32 either way (an f16 input is widened on load and the result rounded once on the
//! store). A `where` condition is f32 or `u8` whatever the values are.

use std::fmt::Write as _;

use folo_core::backend::{BinaryOp, UnaryOp};

use super::shape_numel;
use super::strided::{Strided, coalesce};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::elementwise::MAX_RANK;
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// What an elementwise dispatch computes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EwOp {
    /// `op(x)`.
    Unary(UnaryOp),
    /// `grad * op'(x)`; inputs `[x, grad]`.
    UnaryBackward(UnaryOp),
    /// `a op b`; inputs `[a, b]`.
    Binary(BinaryOp),
    /// `cond != 0 ? a : b`; inputs `[cond, a, b]`, `cond` is f32 or (when `cond_u8`) bytes.
    Where {
        /// The condition is a `u8` tensor.
        cond_u8: bool,
    },
}

/// One input operand: a buffer read through a strided view of the output's shape.
#[derive(Clone, Copy, Debug)]
pub struct EwInput<'a> {
    /// The buffer.
    pub buf: &'a wgpu::Buffer,
    /// Element offset and strides (dims are the output's).
    pub view: &'a Strided,
}

/// Op code and scalar parameters `(a, b)` of a unary op in `ew_fns`.
fn unary_code(op: UnaryOp) -> Result<(u32, f32, f32)> {
    use UnaryOp::*;
    Ok(match op {
        Neg => (0, 0.0, 0.0),
        Abs => (1, 0.0, 0.0),
        Exp => (2, 0.0, 0.0),
        Log => (3, 0.0, 0.0),
        Sqrt => (4, 0.0, 0.0),
        Rsqrt => (5, 0.0, 0.0),
        Recip => (6, 0.0, 0.0),
        Square => (7, 0.0, 0.0),
        Sin => (8, 0.0, 0.0),
        Cos => (9, 0.0, 0.0),
        Tanh => (10, 0.0, 0.0),
        Sigmoid => (11, 0.0, 0.0),
        Silu => (12, 0.0, 0.0),
        Relu => (13, 0.0, 0.0),
        LeakyRelu(a) => (14, a, 0.0),
        Hardswish => (15, 0.0, 0.0),
        Softplus => (16, 0.0, 0.0),
        Clamp(lo, hi) => (17, lo, hi),
        PowScalar(p) => (18, p, 0.0),
        AddScalar(c) => (19, c, 0.0),
        MulScalar(c) => (20, c, 0.0),
        RSubScalar(c) => (21, c, 0.0),
        RDivScalar(c) => (22, c, 0.0),
        DivScalar(c) => (23, c, 0.0),
        MaxScalar(c) => (24, c, 0.0),
        MinScalar(c) => (25, c, 0.0),
        Floor => (26, 0.0, 0.0),
        Ceil => (27, 0.0, 0.0),
        Round => (28, 0.0, 0.0),
        Sign => (29, 0.0, 0.0),
        _ => return Err(GpuError::Unsupported("unary op")),
    })
}

fn binary_code(op: BinaryOp) -> Result<u32> {
    use BinaryOp::*;
    Ok(match op {
        Add => 0,
        Sub => 1,
        Mul => 2,
        Div => 3,
        Pow => 4,
        Maximum => 5,
        Minimum => 6,
        Eq => 7,
        Ne => 8,
        Gt => 9,
        Ge => 10,
        Lt => 11,
        Le => 12,
        _ => return Err(GpuError::Unsupported("binary op")),
    })
}

/// Op functions over `V` (`f32` or `vec4<f32>`); `a` / `b` are the op's scalar parameters.
const EW_FNS: &str = r#"
fn isnanv(x: V) -> BV { return (bitcast<UV>(x) & UV(0x7fffffffu)) > UV(0x7f800000u); }
// Rust f32::max / f32::min: a NaN operand loses.
fn maxn(a: V, b: V) -> V { return select(select(max(a, b), a, isnanv(b)), b, isnanv(a)); }
fn minn(a: V, b: V) -> V { return select(select(min(a, b), a, isnanv(b)), b, isnanv(a)); }
fn sigm(x: V) -> V { let e = exp(-abs(x)); let r = V(1.0) / (V(1.0) + e); return select(e * r, r, x >= V(0.0)); }
fn log1pv(e: V) -> V { let u = V(1.0) + e; return select(log(u) * (e / (u - V(1.0))), e, u == V(1.0)); }
fn tanhv(x: V) -> V {
    return select(select(tanh(clamp(x, V(-9.0), V(9.0))), sign(x), abs(x) >= V(9.0)), x, isnanv(x));
}
fn sgn(x: V) -> V { return select(select(V(0.0), V(-1.0), x < V(0.0)), V(1.0), x > V(0.0)); }
fn roundv(x: V) -> V { let t = trunc(x); return select(t, t + sgn(x), abs(x - t) >= V(0.5)); }
fn mask(c: BV) -> V { return select(V(0.0), V(1.0), c); }
fn ordered(x: V, y: V) -> BV { return !(isnanv(x) | isnanv(y)); }

// C `powf` for one lane (see binary.wgsl); NaN and +inf come from params 62 / 63.
fn pow1(x: f32, y: f32) -> f32 {
    let bx = bitcast<u32>(x);
    let ax = bx & 0x7fffffffu;
    let ay = bitcast<u32>(y) & 0x7fffffffu;
    let nan = pf(62u);
    let inf = pf(63u);
    if (ay == 0u || bx == 0x3f800000u) { return 1.0; }
    if (ax > 0x7f800000u || ay > 0x7f800000u) { return nan; }
    if (ay == 0x7f800000u) {
        if (ax == 0x3f800000u) { return 1.0; }
        return select(0.0, inf, (ax > 0x3f800000u) == (y > 0.0));
    }
    let yi = round(y);
    let is_int = yi == y;
    let odd = is_int && abs(yi - 2.0 * floor(yi * 0.5)) == 1.0;
    let negate = (bx >> 31u) == 1u && odd;
    if (ax == 0u || ax == 0x7f800000u) {
        let r = select(0.0, inf, (ax == 0u) == (y < 0.0));
        return select(r, -r, negate);
    }
    if ((bx >> 31u) == 1u && !is_int) { return nan; }
    let r = exp2(y * log2(bitcast<f32>(ax)));
    return select(r, -r, negate);
}

fn un(x: V, a: f32, b: f32) -> V {
    switch OP {
        case 0u: { return -x; }
        case 1u: { return abs(x); }
        case 2u: { return exp(x); }
        case 3u: { return log(x); }
        case 4u: { return sqrt(x); }
        case 5u: { return V(1.0) / sqrt(x); }
        case 6u: { return V(1.0) / x; }
        case 7u: { return x * x; }
        case 8u: { return sin(x); }
        case 9u: { return cos(x); }
        case 10u: { return tanhv(x); }
        case 11u: { return sigm(x); }
        case 12u: { return x * sigm(x); }
        case 13u: { return maxn(x, V(0.0)); }
        case 14u: { return select(V(a) * x, x, x > V(0.0)); }
        case 15u: { return x * clamp(x + V(3.0), V(0.0), V(6.0)) * V(1.0 / 6.0); }
        case 16u: { return max(x, V(0.0)) + log1pv(exp(-abs(x))); }
        case 17u: { return minn(maxn(x, V(a)), V(b)); }
        case 18u: {
            if (a == 2.0) { return x * x; }
            if (a == 0.5) { return sqrt(x); }
            return powv(x, V(a));
        }
        case 19u: { return x + V(a); }
        case 20u: { return x * V(a); }
        case 21u: { return V(a) - x; }
        case 22u: { return V(a) / x; }
        case 23u: { return x / V(a); }
        case 24u: { return maxn(x, V(a)); }
        case 25u: { return minn(x, V(a)); }
        case 26u: { return floor(x); }
        case 27u: { return ceil(x); }
        case 28u: { return roundv(x); }
        default: { return sgn(x); }
    }
}

fn unb(x: V, g: V, a: f32, b: f32) -> V {
    switch OP {
        case 0u: { return -g; }
        case 1u: { return g * sgn(x); }
        case 2u: { return g * exp(x); }
        case 3u: { return g / x; }
        case 4u: { return g * V(0.5) / sqrt(x); }
        case 5u: { return V(-0.5) * g / sqrt(x) / x; }
        case 6u: { return -(g / x) / x; }
        case 7u: { return V(2.0) * x * g; }
        case 8u: { return g * cos(x); }
        case 9u: { return -g * sin(x); }
        case 10u: { let t = tanhv(x); return g * (V(1.0) - t * t); }
        case 11u: { let s = sigm(x); return g * s * (V(1.0) - s); }
        case 12u: { let s = sigm(x); return g * (s + x * s * (V(1.0) - s)); }
        case 13u: { return select(V(0.0), g, x > V(0.0)); }
        case 14u: { return select(V(a) * g, g, x > V(0.0)); }
        case 15u: {
            let mid = g * (V(2.0) * x + V(3.0)) * V(1.0 / 6.0);
            return select(select(mid, g, x > V(3.0)), V(0.0), x < V(-3.0));
        }
        case 16u: { return g * sigm(x); }
        case 17u: { return select(V(0.0), g, (x >= V(a)) & (x <= V(b))); }
        case 18u: {
            if (a == 0.0) { return V(0.0); }
            return g * V(a) * powv(x, V(a - 1.0));
        }
        case 19u: { return g; }
        case 20u: { return g * V(a); }
        case 21u: { return -g; }
        case 22u: { return -(g * (V(a) / x)) / x; }
        case 23u: { return g / V(a); }
        case 24u: { return select(V(0.0), g, x >= V(a)); }
        case 25u: { return select(V(0.0), g, x <= V(a)); }
        default: { return V(0.0); }
    }
}

fn bin(x: V, y: V) -> V {
    switch OP {
        case 0u: { return x + y; }
        case 1u: { return x - y; }
        case 2u: { return x * y; }
        case 3u: { return x / y; }
        case 4u: { return powv(x, y); }
        case 5u: { return maxn(x, y); }
        case 6u: { return minn(x, y); }
        // Comparisons with a NaN operand are false (true for `ne`), whatever the driver assumes about NaN.
        case 7u: { return mask((x == y) & ordered(x, y)); }
        case 8u: { return mask(!((x == y) & ordered(x, y))); }
        case 9u: { return mask((x > y) & ordered(x, y)); }
        case 10u: { return mask((x >= y) & ordered(x, y)); }
        case 11u: { return mask((x < y) & ordered(x, y)); }
        default: { return mask((x <= y) & ordered(x, y)); }
    }
}
"#;

const POW_SCALAR: &str = "fn powv(x: f32, y: f32) -> f32 { return pow1(x, y); }\n";
const POW_VEC: &str = "fn powv(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {\n\
    \x20 return vec4<f32>(pow1(x.x, y.x), pow1(x.y, y.y), pow1(x.z, y.z), pow1(x.w, y.w));\n}\n";

/// Spec: KIND (1 unary, 2 unary backward, 3 binary, 4 where), OP, RANK (0 = linear), VEC4, COND_U8, and the
/// f16 switch (value storage `S` is f16 / `vec4<f16>`, the arithmetic type `V` stays f32 / `vec4<f32>`).
/// Params: 0 = output units, 2 = a, 3 = b, 4.. dims, 12.. / 20.. / 28.. strides of inputs 0 / 1 / 2,
/// 36 / 37 / 38 = input offsets (in units), 62 = NaN, 63 = +inf.
fn gen_ew(spec: &Spec) -> String {
    let kind = spec.get_u32("KIND", 1);
    let rank = spec.get_u32("RANK", 0);
    let vec = spec.get_bool("VEC4");
    let cond_u8 = spec.get_bool("COND_U8");
    let half = spec.is_f16();
    let nin = match kind {
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    };
    let mut s = String::new();
    if vec {
        s.push_str("alias V = vec4<f32>;\nalias UV = vec4<u32>;\nalias BV = vec4<bool>;\n");
    } else {
        s.push_str("alias V = f32;\nalias UV = u32;\nalias BV = bool;\n");
    }
    let _ = writeln!(
        s,
        "alias S = {};",
        match (half, vec) {
            (true, true) => "vec4<f16>",
            (true, false) => "f16",
            (false, true) => "vec4<f32>",
            (false, false) => "f32",
        }
    );
    s.push_str(EW_FNS);
    s.push_str(if vec { POW_VEC } else { POW_SCALAR });
    for i in 0..nin {
        let ty = match (i == 0 && kind == 4, cond_u8) {
            (true, true) => "u32",
            (true, false) => "V",
            _ => "S",
        };
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> X{i}: array<{ty}>;", i + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> Y: array<S>;", nin + 1);
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let a = pf(2u); let b = pf(3u);\n",
    );
    for k in 0..nin {
        if rank == 0 {
            let _ = writeln!(s, "  let o{k} = pu({}u) + i;", 36 + k);
        } else {
            let _ = writeln!(s, "  var o{k} = pu({}u);", 36 + k);
        }
    }
    if rank > 0 {
        s.push_str("  var rem = i;\n");
        for d in 0..rank {
            let k = rank - 1 - d;
            if k == 0 {
                s.push_str("  { let c = rem;\n");
            } else {
                let _ = writeln!(s, "  {{ let dim = pu({}u); let c = rem % dim; rem = rem / dim;", 4 + k);
            }
            for j in 0..nin {
                let _ = writeln!(s, "    o{j} += c * pu({}u);", 12 + 8 * j + k);
            }
            s.push_str("  }\n");
        }
    }
    match kind {
        1 => s.push_str("  Y[i] = S(un(V(X0[o0]), a, b));\n"),
        2 => s.push_str("  Y[i] = S(unb(V(X0[o0]), V(X1[o1]), a, b));\n"),
        3 => s.push_str("  Y[i] = S(bin(V(X0[o0]), V(X1[o1])));\n"),
        _ => {
            if cond_u8 {
                s.push_str("  let c = ((X0[o0 >> 2u] >> ((o0 & 3u) * 8u)) & 0xffu) != 0u;\n");
            } else {
                // Nonzero by bit pattern, so NaN counts as true (as on the CPU) whatever the driver assumes.
                s.push_str("  let c = (bitcast<UV>(X0[o0]) & UV(0x7fffffffu)) != UV(0u);\n");
            }
            s.push_str("  Y[i] = select(X2[o2], X1[o1], c);\n");
        }
    }
    s.push_str("}\n");
    s
}

static EW1: KernelDef =
    KernelDef { name: "ew1", storage: &[Access::Read, Access::ReadWrite], source: Source::Generated(gen_ew) };
static EW2: KernelDef = KernelDef {
    name: "ew2",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_ew),
};
static EW3: KernelDef = KernelDef {
    name: "ew3",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_ew),
};

/// Computes `op` over `dims` into the dense `out`. `inputs` are the op's operands (see [`EwOp`]), each a
/// strided view with the output's dims; the value operands and `out` are stored as `dtype` (f32 or f16).
pub fn elementwise(
    ctx: &GpuContext,
    op: EwOp,
    dims: &[usize],
    inputs: &[EwInput<'_>],
    out: &wgpu::Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "elementwise")?;
    let (kind, code, a, b, nin, cond_u8) = match op {
        EwOp::Unary(u) => {
            let (c, a, b) = unary_code(u)?;
            (1, c, a, b, 1, false)
        }
        EwOp::UnaryBackward(u) => {
            let (c, a, b) = unary_code(u)?;
            (2, c, a, b, 2, false)
        }
        EwOp::Binary(o) => (3, binary_code(o)?, 0.0, 0.0, 2, false),
        EwOp::Where { cond_u8 } => (4, 0, 0.0, 0.0, 3, cond_u8),
    };
    if inputs.len() != nin {
        return Err(shape_err(format!("elementwise: {op:?} takes {nin} inputs, got {}", inputs.len())));
    }
    let n = shape_numel(dims)?;
    super::check_buf(out, n, dtype, "elementwise output")?;
    for (i, inp) in inputs.iter().enumerate() {
        if inp.view.dims != dims {
            return Err(shape_err(format!(
                "elementwise: input {i} has dims {:?}, want {dims:?}",
                inp.view.dims
            )));
        }
        let elems = inp.view.extent()?;
        if i == 0 && cond_u8 {
            if inp.buf.size() < (elems as u64).next_multiple_of(4) {
                return Err(shape_err("elementwise: condition buffer too small"));
            }
        } else {
            let dt = if i == 0 && kind == 4 { DType::F32 } else { dtype };
            super::check_buf(inp.buf, elems, dt, "elementwise input")?;
        }
    }
    if n == 0 {
        return Ok(());
    }
    let strides: Vec<&[usize]> = inputs.iter().map(|i| i.view.strides.as_slice()).collect();
    let (cd, cs) = coalesce(dims, &strides);
    let linear = cd.len() <= 1 && cs.iter().all(|s| s.first().is_none_or(|&v| v == 1));
    let offsets: Vec<usize> = inputs.iter().map(|i| i.view.offset).collect();
    let vec = linear && !cond_u8 && n.is_multiple_of(4) && offsets.iter().all(|o| o.is_multiple_of(4));
    if !linear && cd.len() > MAX_RANK {
        return Err(shape_err(format!("elementwise: rank {} after coalescing exceeds {MAX_RANK}", cd.len())));
    }
    let q = if vec { 4 } else { 1 };
    let units = n / q;
    let spec = Spec::new()
        .u32("KIND", kind)
        .u32("OP", code)
        .u32("RANK", if linear { 0 } else { cd.len() as u32 })
        .bool("VEC4", vec)
        .bool("COND_U8", cond_u8)
        .f16(dtype == DType::F16);
    let def = match nin {
        1 => &EW1,
        2 => &EW2,
        _ => &EW3,
    };
    let k = ctx.kernel(def, spec)?;
    let mut p = Params::new().uz(units)?.u(0).f(a).f(b).at(4);
    for i in 0..MAX_RANK {
        p = p.uz(if linear { 1 } else { cd.get(i).copied().unwrap_or(1) })?;
    }
    for j in 0..3 {
        for i in 0..MAX_RANK {
            let v = if linear { 0 } else { cs.get(j).and_then(|s| s.get(i)).copied().unwrap_or(0) };
            p = p.uz(v)?;
        }
    }
    for j in 0..3 {
        p = p.uz(offsets.get(j).copied().unwrap_or(0) / q)?;
    }
    let p = p.at(62).f(f32::NAN).f(f32::INFINITY);
    let mut bufs: Vec<&wgpu::Buffer> = inputs.iter().map(|i| i.buf).collect();
    bufs.push(out);
    ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(units as u64, 256)?)
}
