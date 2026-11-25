//! Strided elementwise kernels with folo-core's op set: unary ops and their input gradients, binary ops and
//! comparisons, and `where`.
//!
//! Every input is a strided view (offset + strides, stride 0 broadcasts) of the output's shape, so broadcast
//! and transposed operands are read in place instead of being copied first. The shape is coalesced across all
//! operands; when the innermost coalesced dim is a multiple of 4 and every operand is either contiguous along
//! it (with 4-aligned offset and outer strides) or broadcast along it (stride 0, no alignment required), the
//! kernel emits `vec4` loads/stores (splatting broadcast operands and unpacking 4 `u8` condition bytes per
//! `u32` word). Otherwise it falls back to scalar indexing over up to [`MAX_RANK`] dims. Each generated shader
//! emits only the active `(KIND, OP)` body and the helper functions it references. Semantics follow
//! folo-core's CPU kernels: stable sigmoid / softplus, NaN-ignoring `max` / `min` (Rust's `f32::max`), C
//! `powf`, rounding half away from zero.
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

const FN_ISNANV: &str =
    "fn isnanv(x: V) -> BV { return (bitcast<UV>(x) & UV(0x7fffffffu)) > UV(0x7f800000u); }\n";
// Rust f32::max / f32::min: a NaN operand loses.
const FN_MAXN: &str =
    "fn maxn(a: V, b: V) -> V { return select(select(max(a, b), a, isnanv(b)), b, isnanv(a)); }\n";
const FN_MINN: &str =
    "fn minn(a: V, b: V) -> V { return select(select(min(a, b), a, isnanv(b)), b, isnanv(a)); }\n";
const FN_SIGM: &str = "fn sigm(x: V) -> V { let e = exp(-abs(x)); let r = V(1.0) / (V(1.0) + e); return select(e * r, r, x >= V(0.0)); }\n";
const FN_LOG1PV: &str = "fn log1pv(e: V) -> V { let u = V(1.0) + e; return select(log(u) * (e / (u - V(1.0))), e, u == V(1.0)); }\n";
const FN_TANHV: &str = "fn tanhv(x: V) -> V {\n    return select(select(tanh(clamp(x, V(-9.0), V(9.0))), sign(x), abs(x) >= V(9.0)), x, isnanv(x));\n}\n";
const FN_SGN: &str =
    "fn sgn(x: V) -> V { return select(select(V(0.0), V(-1.0), x < V(0.0)), V(1.0), x > V(0.0)); }\n";
const FN_ROUNDV: &str =
    "fn roundv(x: V) -> V { let t = trunc(x); return select(t, t + sgn(x), abs(x - t) >= V(0.5)); }\n";
const FN_MASK: &str = "fn mask(c: BV) -> V { return select(V(0.0), V(1.0), c); }\n";
const FN_ORDERED: &str = "fn ordered(x: V, y: V) -> BV { return !(isnanv(x) | isnanv(y)); }\n";

// C `powf` for one lane (see binary.wgsl); NaN and +inf come from params 62 / 63.
const FN_POW1: &str = r#"fn pow1(x: f32, y: f32) -> f32 {
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
"#;

const POW_SCALAR: &str = "fn powv(x: f32, y: f32) -> f32 { return pow1(x, y); }\n";
const POW_VEC: &str = "fn powv(x: vec4<f32>, y: vec4<f32>) -> vec4<f32> {\n\
    \x20 return vec4<f32>(pow1(x.x, y.x), pow1(x.y, y.y), pow1(x.z, y.z), pow1(x.w, y.w));\n}\n";

/// Appends only the helper functions and active op function (`un`, `unb`, or `bin`) needed by `(kind, op)`.
fn push_ew_op(s: &mut String, kind: u32, op: u32, vec: bool) {
    let mut need_isnan = false;
    let mut need_maxn = false;
    let mut need_minn = false;
    let mut need_sigm = false;
    let mut need_log1p = false;
    let mut need_tanh = false;
    let mut need_sgn = false;
    let mut need_round = false;
    let mut need_mask = false;
    let mut need_ordered = false;
    let mut need_pow = false;

    let body = match kind {
        1 => match op {
            0 => "return -x;",
            1 => "return abs(x);",
            2 => "return exp(x);",
            3 => "return log(x);",
            4 => "return sqrt(x);",
            5 => "return V(1.0) / sqrt(x);",
            6 => "return V(1.0) / x;",
            7 => "return x * x;",
            8 => "return sin(x);",
            9 => "return cos(x);",
            10 => {
                need_isnan = true;
                need_tanh = true;
                "return tanhv(x);"
            }
            11 => {
                need_sigm = true;
                "return sigm(x);"
            }
            12 => {
                need_sigm = true;
                "return x * sigm(x);"
            }
            13 => {
                need_isnan = true;
                need_maxn = true;
                "return maxn(x, V(0.0));"
            }
            14 => "return select(V(a) * x, x, x > V(0.0));",
            15 => "return x * clamp(x + V(3.0), V(0.0), V(6.0)) * V(1.0 / 6.0);",
            16 => {
                need_log1p = true;
                "return max(x, V(0.0)) + log1pv(exp(-abs(x)));"
            }
            17 => {
                need_isnan = true;
                need_maxn = true;
                need_minn = true;
                "return minn(maxn(x, V(a)), V(b));"
            }
            18 => {
                need_pow = true;
                "if (a == 2.0) { return x * x; }\n    if (a == 0.5) { return sqrt(x); }\n    return powv(x, V(a));"
            }
            19 => "return x + V(a);",
            20 => "return x * V(a);",
            21 => "return V(a) - x;",
            22 => "return V(a) / x;",
            23 => "return x / V(a);",
            24 => {
                need_isnan = true;
                need_maxn = true;
                "return maxn(x, V(a));"
            }
            25 => {
                need_isnan = true;
                need_minn = true;
                "return minn(x, V(a));"
            }
            26 => "return floor(x);",
            27 => "return ceil(x);",
            28 => {
                need_sgn = true;
                need_round = true;
                "return roundv(x);"
            }
            _ => {
                need_sgn = true;
                "return sgn(x);"
            }
        },
        2 => match op {
            0 => "return -g;",
            1 => {
                need_sgn = true;
                "return g * sgn(x);"
            }
            2 => "return g * exp(x);",
            3 => "return g / x;",
            4 => "return g * V(0.5) / sqrt(x);",
            5 => "return V(-0.5) * g / sqrt(x) / x;",
            6 => "return -(g / x) / x;",
            7 => "return V(2.0) * x * g;",
            8 => "return g * cos(x);",
            9 => "return -g * sin(x);",
            10 => {
                need_isnan = true;
                need_tanh = true;
                "let t = tanhv(x); return g * (V(1.0) - t * t);"
            }
            11 => {
                need_sigm = true;
                "let s = sigm(x); return g * s * (V(1.0) - s);"
            }
            12 => {
                need_sigm = true;
                "let s = sigm(x); return g * (s + x * s * (V(1.0) - s));"
            }
            13 => "return select(V(0.0), g, x > V(0.0));",
            14 => "return select(V(a) * g, g, x > V(0.0));",
            15 => {
                "let mid = g * (V(2.0) * x + V(3.0)) * V(1.0 / 6.0);\n    \
                 return select(select(mid, g, x > V(3.0)), V(0.0), x < V(-3.0));"
            }
            16 => {
                need_sigm = true;
                "return g * sigm(x);"
            }
            17 => "return select(V(0.0), g, (x >= V(a)) & (x <= V(b)));",
            18 => {
                need_pow = true;
                "if (a == 0.0) { return V(0.0); }\n    return g * V(a) * powv(x, V(a - 1.0));"
            }
            19 => "return g;",
            20 => "return g * V(a);",
            21 => "return -g;",
            22 => "return -(g * (V(a) / x)) / x;",
            23 => "return g / V(a);",
            24 => "return select(V(0.0), g, x >= V(a));",
            25 => "return select(V(0.0), g, x <= V(a));",
            _ => "return V(0.0);",
        },
        3 => match op {
            0 => "return x + y;",
            1 => "return x - y;",
            2 => "return x * y;",
            3 => "return x / y;",
            4 => {
                need_pow = true;
                "return powv(x, y);"
            }
            5 => {
                need_isnan = true;
                need_maxn = true;
                "return maxn(x, y);"
            }
            6 => {
                need_isnan = true;
                need_minn = true;
                "return minn(x, y);"
            }
            // Comparisons with a NaN operand are false (true for `ne`), whatever the driver assumes about NaN.
            7 => {
                need_isnan = true;
                need_ordered = true;
                need_mask = true;
                "return mask((x == y) & ordered(x, y));"
            }
            8 => {
                need_isnan = true;
                need_ordered = true;
                need_mask = true;
                "return mask(!((x == y) & ordered(x, y)));"
            }
            9 => {
                need_isnan = true;
                need_ordered = true;
                need_mask = true;
                "return mask((x > y) & ordered(x, y));"
            }
            10 => {
                need_isnan = true;
                need_ordered = true;
                need_mask = true;
                "return mask((x >= y) & ordered(x, y));"
            }
            11 => {
                need_isnan = true;
                need_ordered = true;
                need_mask = true;
                "return mask((x < y) & ordered(x, y));"
            }
            _ => {
                need_isnan = true;
                need_ordered = true;
                need_mask = true;
                "return mask((x <= y) & ordered(x, y));"
            }
        },
        _ => return,
    };

    if need_isnan {
        s.push_str(FN_ISNANV);
    }
    if need_maxn {
        s.push_str(FN_MAXN);
    }
    if need_minn {
        s.push_str(FN_MINN);
    }
    if need_sigm {
        s.push_str(FN_SIGM);
    }
    if need_log1p {
        s.push_str(FN_LOG1PV);
    }
    if need_tanh {
        s.push_str(FN_TANHV);
    }
    if need_sgn {
        s.push_str(FN_SGN);
    }
    if need_round {
        s.push_str(FN_ROUNDV);
    }
    if need_mask {
        s.push_str(FN_MASK);
    }
    if need_ordered {
        s.push_str(FN_ORDERED);
    }
    if need_pow {
        s.push_str(FN_POW1);
        s.push_str(if vec { POW_VEC } else { POW_SCALAR });
    }
    match kind {
        1 => {
            let _ = writeln!(s, "fn un(x: V, a: f32, b: f32) -> V {{\n    {body}\n}}");
        }
        2 => {
            let _ = writeln!(s, "fn unb(x: V, g: V, a: f32, b: f32) -> V {{\n    {body}\n}}");
        }
        3 => {
            let _ = writeln!(s, "fn bin(x: V, y: V) -> V {{\n    {body}\n}}");
        }
        _ => {}
    }
}

/// Spec: KIND (1 unary, 2 unary backward, 3 binary, 4 where), OP, RANK (0 = linear), VEC4, BCAST (3-bit mask of
/// inputs whose innermost coalesced dim is broadcast under VEC4, read as scalars and splatted), COND_U8, and the
/// f16 switch (value storage `S` is f16 / `vec4<f16>`, scalar broadcast storage `S1` is f16 / f32, and the
/// arithmetic type `V` stays f32 / `vec4<f32>`).
/// Params: 0 = output units, 2 = a, 3 = b, 4.. dims, 12.. / 20.. / 28.. strides of inputs 0 / 1 / 2,
/// 36 / 37 / 38 = input offsets (in `vec4` / `u32` units for vector inputs, in elements for scalar/broadcast
/// inputs), 62 = NaN, 63 = +inf.
fn gen_ew(spec: &Spec) -> String {
    let kind = spec.get_u32("KIND", 1);
    let op = spec.get_u32("OP", 0);
    let rank = spec.get_u32("RANK", 0);
    let vec = spec.get_bool("VEC4");
    let bcast = if vec { spec.get_u32("BCAST", 0) } else { 0 };
    let cond_u8 = spec.get_bool("COND_U8");
    let half = spec.is_f16();
    let nin: u32 = match kind {
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    };
    let is_bcast = |k: u32| vec && ((bcast & (1u32 << k)) != 0);
    let mut s = String::new();
    if vec {
        s.push_str("alias V = vec4<f32>;\nalias UV = vec4<u32>;\nalias BV = vec4<bool>;\n");
    } else {
        s.push_str("alias V = f32;\nalias UV = u32;\nalias BV = bool;\n");
    }
    let _ = writeln!(
        s,
        "alias S = {};\nalias S1 = {};",
        match (half, vec) {
            (true, true) => "vec4<f16>",
            (true, false) => "f16",
            (false, true) => "vec4<f32>",
            (false, false) => "f32",
        },
        if half { "f16" } else { "f32" },
    );
    push_ew_op(&mut s, kind, op, vec);
    for i in 0..nin {
        let ty = match (i == 0 && kind == 4, cond_u8, is_bcast(i)) {
            (true, true, _) => "u32",
            (true, false, true) => "f32",
            (true, false, false) => "V",
            (false, _, true) => "S1",
            (false, _, false) => "S",
        };
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> X{i}: array<{ty}>;", i + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> Y: array<S>;", nin + 1);
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n",
    );
    if kind <= 2 {
        s.push_str("  let a = pf(2u); let b = pf(3u);\n");
    }
    for k in 0..nin {
        if rank == 0 {
            if is_bcast(k) {
                let _ = writeln!(s, "  let o{k} = pu({}u);", 36 + k);
            } else {
                let _ = writeln!(s, "  let o{k} = pu({}u) + i;", 36 + k);
            }
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
    let load_v = |k: u32| {
        if is_bcast(k) { format!("V(f32(X{k}[o{k}]))") } else { format!("V(X{k}[o{k}])") }
    };
    let load_s = |k: u32| {
        if is_bcast(k) { format!("S(X{k}[o{k}])") } else { format!("X{k}[o{k}]") }
    };
    match kind {
        1 => {
            let _ = writeln!(s, "  Y[i] = S(un({}, a, b));", load_v(0));
        }
        2 => {
            let _ = writeln!(s, "  Y[i] = S(unb({}, {}, a, b));", load_v(0), load_v(1));
        }
        3 => {
            let _ = writeln!(s, "  Y[i] = S(bin({}, {}));", load_v(0), load_v(1));
        }
        _ => {
            if cond_u8 {
                if !vec {
                    s.push_str("  let c = ((X0[o0 >> 2u] >> ((o0 & 3u) * 8u)) & 0xffu) != 0u;\n");
                } else if is_bcast(0) {
                    s.push_str("  let c = vec4<bool>(((X0[o0 >> 2u] >> ((o0 & 3u) * 8u)) & 0xffu) != 0u);\n");
                } else {
                    s.push_str(
                        "  let w = X0[o0];\n  \
                         let c = (vec4<u32>(w) & vec4<u32>(0xffu, 0xff00u, 0xff0000u, 0xff000000u)) != vec4<u32>(0u);\n",
                    );
                }
            } else if is_bcast(0) {
                // Nonzero by bit pattern, so NaN counts as true (as on the CPU) whatever the driver assumes.
                s.push_str("  let c = vec4<bool>((bitcast<u32>(X0[o0]) & 0x7fffffffu) != 0u);\n");
            } else {
                s.push_str("  let c = (bitcast<UV>(X0[o0]) & UV(0x7fffffffu)) != UV(0u);\n");
            }
            let _ = writeln!(s, "  Y[i] = select({}, {}, c);", load_s(2), load_s(1));
        }
    }
    s.push_str("}\n");
    s
}

struct EwPlan {
    dims: Vec<usize>,
    strides: Vec<Vec<usize>>,
    offsets: Vec<usize>,
    vec: bool,
    bcast: u32,
    linear: bool,
}

/// Coalesces `dims`/`strides` and detects whether the innermost coalesced dim can be vectorized by 4 (with
/// broadcast operands splatted from scalar reads).
fn plan_ew(dims: &[usize], strides: &[&[usize]], mut offsets: Vec<usize>) -> EwPlan {
    let (mut cd, mut cs) = coalesce(dims, strides);
    let mut bcast_mask = 0u32;
    let vec = !cd.is_empty() && {
        let last = cd.len() - 1;
        cd[last].is_multiple_of(4)
            && cs.iter().zip(&offsets).enumerate().all(|(j, (s, &off))| {
                if s[last] == 0 {
                    bcast_mask |= 1u32 << j;
                    true
                } else {
                    s[last] == 1 && off.is_multiple_of(4) && s[..last].iter().all(|v| v.is_multiple_of(4))
                }
            })
    };
    let bcast = if vec { bcast_mask } else { 0 };
    if vec {
        let last = cd.len() - 1;
        cd[last] /= 4;
        for (j, (s, off)) in cs.iter_mut().zip(offsets.iter_mut()).enumerate() {
            if (bcast & (1u32 << j)) == 0 {
                *off /= 4;
                for v in &mut s[..last] {
                    *v /= 4;
                }
            }
        }
        // Dropping a trailing dim that shrank to 1 can merge adjacent outer dims (e.g. `[C, 4] + [C, 1]`).
        let s_refs: Vec<&[usize]> = cs.iter().map(Vec::as_slice).collect();
        (cd, cs) = coalesce(&cd, &s_refs);
    }
    let linear = cd.is_empty()
        || (cd.len() == 1
            && cs.iter().enumerate().all(|(j, s)| s[0] == usize::from((bcast & (1u32 << j)) == 0)));
    EwPlan { dims: cd, strides: cs, offsets, vec, bcast, linear }
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
    let offsets: Vec<usize> = inputs.iter().map(|i| i.view.offset).collect();
    let plan = plan_ew(dims, &strides, offsets);
    if !plan.linear && plan.dims.len() > MAX_RANK {
        return Err(shape_err(format!(
            "elementwise: rank {} after coalescing exceeds {MAX_RANK}",
            plan.dims.len()
        )));
    }
    let q = if plan.vec { 4 } else { 1 };
    let units = n / q;
    let mut spec = Spec::new()
        .u32("KIND", kind)
        .u32("OP", code)
        .u32("RANK", if plan.linear { 0 } else { plan.dims.len() as u32 })
        .bool("VEC4", plan.vec)
        .bool("COND_U8", cond_u8)
        .f16(dtype == DType::F16);
    if plan.bcast != 0 {
        spec = spec.u32("BCAST", plan.bcast);
    }
    let def = match nin {
        1 => &EW1,
        2 => &EW2,
        _ => &EW3,
    };
    let k = ctx.kernel(def, spec)?;
    let mut p = Params::new().uz(units)?.u(0).f(a).f(b).at(4);
    for i in 0..MAX_RANK {
        p = p.uz(if plan.linear { 1 } else { plan.dims.get(i).copied().unwrap_or(1) })?;
    }
    for j in 0..3 {
        for i in 0..MAX_RANK {
            let v = if plan.linear {
                0
            } else {
                plan.strides.get(j).and_then(|s| s.get(i)).copied().unwrap_or(0)
            };
            p = p.uz(v)?;
        }
    }
    for j in 0..3 {
        p = p.uz(plan.offsets.get(j).copied().unwrap_or(0))?;
    }
    let p = p.at(62).f(f32::NAN).f(f32::INFINITY);
    let mut bufs: Vec<&wgpu::Buffer> = inputs.iter().map(|i| i.buf).collect();
    bufs.push(out);
    ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(units as u64, 256)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&EW1, &EW2, &EW3];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_ew_prunes_unused_ops_and_helpers() {
        let add_src = gen_ew(&Spec::new().u32("KIND", 3).u32("OP", 0).u32("RANK", 0).bool("VEC4", true));
        assert!(add_src.contains("fn bin(") && add_src.contains("return x + y;"));
        for dead in ["fn un(", "fn unb(", "fn pow1(", "fn tanhv(", "fn sigm(", "fn isnanv(", "fn roundv("] {
            assert!(!add_src.contains(dead), "add shader unexpectedly contains {dead}");
        }

        let silu_src = gen_ew(&Spec::new().u32("KIND", 1).u32("OP", 12).u32("RANK", 0).bool("VEC4", true));
        assert!(silu_src.contains("fn sigm(") && silu_src.contains("fn un("));
        for dead in ["fn unb(", "fn bin(", "fn pow1(", "fn tanhv(", "fn isnanv("] {
            assert!(!silu_src.contains(dead), "silu shader unexpectedly contains {dead}");
        }

        let where_src = gen_ew(
            &Spec::new()
                .u32("KIND", 4)
                .u32("OP", 0)
                .u32("RANK", 2)
                .bool("VEC4", true)
                .u32("BCAST", 0b101)
                .bool("COND_U8", true),
        );
        for dead in ["fn un(", "fn unb(", "fn bin(", "fn pow1(", "fn sigm("] {
            assert!(!where_src.contains(dead), "where shader unexpectedly contains {dead}");
        }
        assert!(where_src.contains("X0: array<u32>"));
        assert!(where_src.contains("X1: array<S>"));
        assert!(where_src.contains("X2: array<S1>"));
    }

    #[test]
    fn plan_ew_vectorizes_broadcast_and_rejects_unaligned() {
        // [2, 3, 4, 5] + [1, 3, 1, 1] -> [2, 3, 20] -> vec4 [2, 3, 5] with bcast bit 1 (odd offset allowed).
        let p = plan_ew(&[2, 3, 4, 5], &[&[60, 20, 5, 1], &[0, 1, 0, 0]], vec![8, 3]);
        assert!(p.vec && !p.linear && p.bcast == 0b10);
        assert_eq!(p.dims, vec![2, 3, 5]);
        assert_eq!(p.strides, vec![vec![15, 5, 1], vec![0, 1, 0]]);
        assert_eq!(p.offsets, vec![2, 3]);

        // [3, 4] + [3, 1] -> inner dim 4 collapses to 1 and drops cleanly to 1D rank 1.
        let p = plan_ew(&[3, 4], &[&[4, 1], &[1, 0]], vec![0, 1]);
        assert!(p.vec && !p.linear && p.bcast == 0b10);
        assert_eq!(p.dims, vec![3]);
        assert_eq!(p.strides, vec![vec![1], vec![1]]);
        assert_eq!(p.offsets, vec![0, 1]);

        // 1D scalar broadcast [1024] + [1] stays linear (RANK = 0).
        let p = plan_ew(&[1024], &[&[1], &[0]], vec![4, 7]);
        assert!(p.vec && p.linear && p.bcast == 0b10 && p.dims == vec![256]);

        // Unaligned vector offset rejects vec4.
        let p = plan_ew(&[1024], &[&[1], &[0]], vec![2, 0]);
        assert!(!p.vec && p.bcast == 0 && p.offsets == vec![2, 0]);
    }
}
