//! Fused `vec4` WGSL activation backward and epilogue kernels (`F32` and `F16`):
//!
//! - [`silu_backward`]: `dx = grad * silu'(x)` with numerically stable sigmoid (`exp(-|x|)`).
//! - [`gelu_backward`]: `dx = grad * gelu_tanh'(x)` matching [`super::elementwise::UnaryOp::Gelu`], guarded against
//!   cubic/hyperbolic overflow for large `|x|`.
//! - [`swiglu_forward`] / [`swiglu_backward`]: fused single-pass `y = silu(gate) * up` and `(dgate, dup)` without
//!   materializing intermediate tensors.
//! - [`geglu_forward`] / [`geglu_backward`]: fused single-pass `y = gelu(gate) * up` and `(dgate, dup)` without
//!   materializing intermediate tensors.
//! - [`bias_add_silu`]: layout-specialized `vec4` `y = silu(x + bias[(i / inner) % channels])` with a single scalar
//!   splat per `vec4` when `inner % 4 == 0` (NCHW conv outputs), a coalesced `vec4` bias load when
//!   `inner == 1 && channels % 4 == 0` (linear/MLP layers), and a scalar fallback otherwise.

use super::{check_buf, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::{ResidualMode, check_dtype};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Activation supported by [`bias_residual_act`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedAct {
    /// `x`
    Identity,
    /// `x * sigmoid(x)`
    Silu,
    /// Tanh-approximation GELU matching [`super::elementwise::UnaryOp::Gelu`].
    Gelu,
    /// `max(x, 0)` (NaN-ignoring, matching Rust `f32::max`).
    Relu,
}

impl FusedAct {
    pub(crate) fn code(self) -> u32 {
        match self {
            FusedAct::Identity => 0,
            FusedAct::Silu => 1,
            FusedAct::Gelu => 2,
            FusedAct::Relu => 3,
        }
    }
}

fn prelude(spec: &Spec) -> &'static str {
    let vec = spec.get_bool("VEC4");
    if spec.is_f16() { vec_prelude_f16(vec) } else { vec_prelude(vec) }
}

/// Shared WGSL numerically stable sigmoid and GELU helpers operating on `vec4<f32>`.
const ACT_HELPERS: &str = r#"
fn isnan4(x: vec4<f32>) -> vec4<bool> {
    return (bitcast<vec4<u32>>(x) & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
}
fn sigm4(x: vec4<f32>) -> vec4<f32> {
    let e = exp(-abs(x));
    let r = vec4<f32>(1.0) / (vec4<f32>(1.0) + e);
    return select(e * r, r, x >= vec4<f32>(0.0));
}
fn silu4(x: vec4<f32>) -> vec4<f32> {
    let s = sigm4(x);
    let big_neg = (x < vec4<f32>(-88.0)) & !isnan4(x);
    return select(x * s, vec4<f32>(0.0), big_neg);
}
fn silu_deriv4(x: vec4<f32>) -> vec4<f32> {
    let s = sigm4(x);
    let big_pos = (x > vec4<f32>(88.0)) & !isnan4(x);
    let big_neg = (x < vec4<f32>(-88.0)) & !isnan4(x);
    let mid = s + x * s * (vec4<f32>(1.0) - s);
    return select(select(mid, vec4<f32>(0.0), big_neg), vec4<f32>(1.0), big_pos);
}
fn gelu4(x: vec4<f32>) -> vec4<f32> {
    // For |x| >= 11, 2|u| >= 112.5 so exp(-2|u|) == 0 in IEEE f32 (even subnormals); saturate to avoid x^3 overflow.
    let sat = (abs(x) >= vec4<f32>(11.0)) & !isnan4(x);
    let xc = select(x, vec4<f32>(0.0), sat);
    let k = 0.7978845608028654;
    let c = 0.044715;
    let u = k * (xc + c * xc * xc * xc);
    let e = exp(-2.0 * abs(u));
    let inv = vec4<f32>(1.0) / (vec4<f32>(1.0) + e);
    let one_plus_t = select(2.0 * e * inv, 2.0 * inv, u >= vec4<f32>(0.0));
    let mid = 0.5 * xc * one_plus_t;
    return select(mid, select(vec4<f32>(0.0), x, x > vec4<f32>(0.0)), sat);
}
fn relu4(x: vec4<f32>) -> vec4<f32> {
    return select(max(x, vec4<f32>(0.0)), vec4<f32>(0.0), isnan4(x));
}
fn apply_fused_act4(act: u32, x: vec4<f32>) -> vec4<f32> {
    switch act {
        case 1u: { return silu4(x); }
        case 2u: { return gelu4(x); }
        case 3u: { return relu4(x); }
        default: { return x; }
    }
}
fn gelu_deriv4(x: vec4<f32>) -> vec4<f32> {
    // For |x| >= 11, 2|u| >= 112.5 so exp(-2|u|) underflows below f32 min subnormal; saturate to avoid x^3 overflow.
    // Using e = exp(-2|u|) avoids catastrophic cancellation in both (1 + tanh(u)) and (1 - tanh^2(u)) for |u| >= 9.
    let sat = (abs(x) >= vec4<f32>(11.0)) & !isnan4(x);
    let xc = select(x, vec4<f32>(0.0), sat);
    let k = 0.7978845608028654;
    let c = 0.044715;
    let x2 = xc * xc;
    let u = k * (xc + c * x2 * xc);
    let e = exp(-2.0 * abs(u));
    let inv = vec4<f32>(1.0) / (vec4<f32>(1.0) + e);
    let one_plus_t = select(2.0 * e * inv, 2.0 * inv, u >= vec4<f32>(0.0));
    let sech2 = 4.0 * e * inv * inv;
    let du = k * (vec4<f32>(1.0) + (3.0 * c) * x2);
    let mid = 0.5 * (one_plus_t + xc * sech2 * du);
    return select(mid, select(vec4<f32>(0.0), vec4<f32>(1.0), x > vec4<f32>(0.0)), sat);
}
"#;

const SILU_BWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read_write> DX: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    DX[i] = OUT(V4(G[i]) * silu_deriv4(V4(X[i])));
}
"#;

const GELU_BWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read_write> DX: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    DX[i] = OUT(V4(G[i]) * gelu_deriv4(V4(X[i])));
}
"#;

const SWIGLU_FWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> GATE: array<E>;
@group(0) @binding(2) var<storage, read> UP: array<E>;
@group(0) @binding(3) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    Y[i] = OUT(silu4(V4(GATE[i])) * V4(UP[i]));
}
"#;

const SWIGLU_BWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> GATE: array<E>;
@group(0) @binding(2) var<storage, read> UP: array<E>;
@group(0) @binding(3) var<storage, read> G: array<E>;
@group(0) @binding(4) var<storage, read_write> DGATE: array<E>;
@group(0) @binding(5) var<storage, read_write> DUP: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let gate = V4(GATE[i]);
    let up = V4(UP[i]);
    let grad = V4(G[i]);
    let d_silu = silu_deriv4(gate);
    DUP[i] = OUT(grad * silu4(gate));
    // When |d_silu| < 1, apply it to the larger factor first so `grad * up` cannot spuriously overflow.
    let grad_bigger = abs(grad) >= abs(up);
    let p_shrink = select(grad * (up * d_silu), (grad * d_silu) * up, grad_bigger);
    let p_direct = (grad * up) * d_silu;
    DGATE[i] = OUT(select(p_direct, p_shrink, abs(d_silu) < vec4<f32>(1.0)));
}
"#;

const GEGLU_FWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> GATE: array<E>;
@group(0) @binding(2) var<storage, read> UP: array<E>;
@group(0) @binding(3) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    Y[i] = OUT(gelu4(V4(GATE[i])) * V4(UP[i]));
}
"#;

const GEGLU_BWD_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> GATE: array<E>;
@group(0) @binding(2) var<storage, read> UP: array<E>;
@group(0) @binding(3) var<storage, read> G: array<E>;
@group(0) @binding(4) var<storage, read_write> DGATE: array<E>;
@group(0) @binding(5) var<storage, read_write> DUP: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let gate = V4(GATE[i]);
    let up = V4(UP[i]);
    let grad = V4(G[i]);
    let d_gelu = gelu_deriv4(gate);
    DUP[i] = OUT(grad * gelu4(gate));
    // When |d_gelu| < 1, apply it to the larger factor first so `grad * up` cannot spuriously overflow.
    let grad_bigger = abs(grad) >= abs(up);
    let p_shrink = select(grad * (up * d_gelu), (grad * d_gelu) * up, grad_bigger);
    let p_direct = (grad * up) * d_gelu;
    DGATE[i] = OUT(select(p_direct, p_shrink, abs(d_gelu) < vec4<f32>(1.0)));
}
"#;

const GEGLU_BWD_DGATE_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> GATE: array<E>;
@group(0) @binding(2) var<storage, read> UP: array<E>;
@group(0) @binding(3) var<storage, read> G: array<E>;
@group(0) @binding(4) var<storage, read_write> DGATE: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let gate = V4(GATE[i]);
    let up = V4(UP[i]);
    let grad = V4(G[i]);
    let d_gelu = gelu_deriv4(gate);
    let grad_bigger = abs(grad) >= abs(up);
    let p_shrink = select(grad * (up * d_gelu), (grad * d_gelu) * up, grad_bigger);
    let p_direct = (grad * up) * d_gelu;
    DGATE[i] = OUT(select(p_direct, p_shrink, abs(d_gelu) < vec4<f32>(1.0)));
}
"#;

const GEGLU_BWD_DUP_BODY: &str = r#"
@group(0) @binding(1) var<storage, read> GATE: array<E>;
@group(0) @binding(2) var<storage, read> G: array<E>;
@group(0) @binding(3) var<storage, read_write> DUP: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    DUP[i] = OUT(V4(G[i]) * gelu4(V4(GATE[i])));
}
"#;

/// Layout-specialized bias + SiLU:
/// - `MODE == 0`: `VEC4` with `inner % 4 == 0` (NCHW) — 1 scalar bias splat per `vec4` (`BIAS[(i / inner_u) % c]`).
/// - `MODE == 1`: `VEC4` with `inner == 1 && channels % 4 == 0` (`[rows, C]`) — coalesced `vec4` bias load (`BIAS[i % ch_u]`).
/// - `MODE == 2`: scalar fallback (`BIAS[(i / inner) % c]`).
///
/// Params: 0 = `units`, 1 = `channels` (or `ch_u` in MODE 1), 2 = `inner_u` (in MODE 0) / `inner` (in MODE 2).
fn gen_bias_add_silu(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 2);
    let has_bias = spec.get_bool("HAS_BIAS");
    let vec = mode != 2;
    let half = spec.is_f16();
    let pre = if half { vec_prelude_f16(vec) } else { vec_prelude(vec) };
    let bias_ty = match (mode == 1, half) {
        (true, true) => "vec4<f16>",
        (true, false) => "vec4<f32>",
        (false, true) => "f16",
        (false, false) => "f32",
    };
    let bias_add = if !has_bias {
        ""
    } else if mode == 0 {
        "    v += vec4<f32>(f32(BIAS[(i / pu(2u)) % pu(1u)]));\n"
    } else if mode == 1 {
        "    v += vec4<f32>(BIAS[i % pu(1u)]);\n"
    } else {
        "    v += vec4<f32>(f32(BIAS[(i / pu(2u)) % pu(1u)]));\n"
    };
    format!(
        "{pre}{ACT_HELPERS}\n\
         @group(0) @binding(1) var<storage, read> X: array<E>;\n\
         @group(0) @binding(2) var<storage, read> BIAS: array<{bias_ty}>;\n\
         @group(0) @binding(3) var<storage, read_write> Y: array<E>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let i = linear_id(wid, nwg, lid, 256u);\n\
             if (i >= pu(0u)) {{ return; }}\n\
             var v = V4(X[i]);\n\
         {bias_add}\
             Y[i] = OUT(silu4(v));\n\
         }}\n"
    )
}

/// Bounds check for precompiling `"bias_add_silu"` (`kernels/bounds.rs`).
pub(crate) fn bias_add_silu_fits(spec: &Spec) -> bool {
    spec.get_u32("MODE", 2) <= 2
}

/// Layout-specialized fused bias + optional residual + activation (`Identity`, `Silu`, `Gelu`, `Relu`).
/// Params: 0 = `units`, 1 = `channels` (or `ch_u` in MODE 1), 2 = `inner_u` / `inner`, 3 = `beta`.
fn gen_bias_residual_act(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 2);
    let has_bias = spec.get_bool("HAS_BIAS");
    let res_mode = spec.get_u32("RES_MODE", 0);
    let x_f32 = spec.get_bool("X_F32");
    let vec = mode != 2;
    let half = spec.is_f16();
    let pre = if half { vec_prelude_f16(vec) } else { vec_prelude(vec) };
    let x_ty = match (vec, half && !x_f32) {
        (true, true) => "vec4<f16>",
        (true, false) => "vec4<f32>",
        (false, true) => "f16",
        (false, false) => "f32",
    };
    let x_load = if x_f32 && !vec {
        "vec4<f32>(X[i])"
    } else if x_f32 {
        "X[i]"
    } else {
        "V4(X[i])"
    };
    let bias_ty = match (mode == 1, half) {
        (true, true) => "vec4<f16>",
        (true, false) => "vec4<f32>",
        (false, true) => "f16",
        (false, false) => "f32",
    };
    let bias_add = if !has_bias {
        ""
    } else if mode == 1 {
        "    v += vec4<f32>(BIAS[i % pu(1u)]);\n"
    } else {
        "    v += vec4<f32>(f32(BIAS[(i / pu(2u)) % pu(1u)]));\n"
    };
    let pre_res = if res_mode == 1 { "    v = fma(V4(RES[i]), vec4<f32>(pf(3u)), v);\n" } else { "" };
    let post_res = if res_mode == 2 { "    v = fma(V4(RES[i]), vec4<f32>(pf(3u)), v);\n" } else { "" };
    format!(
        "{pre}{ACT_HELPERS}\n\
         @group(0) @binding(1) var<storage, read> X: array<{x_ty}>;\n\
         @group(0) @binding(2) var<storage, read> BIAS: array<{bias_ty}>;\n\
         @group(0) @binding(3) var<storage, read> RES: array<E>;\n\
         @group(0) @binding(4) var<storage, read_write> Y: array<E>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let i = linear_id(wid, nwg, lid, 256u);\n\
             if (i >= pu(0u)) {{ return; }}\n\
             var v = {x_load};\n\
         {bias_add}\
         {pre_res}\
             v = apply_fused_act4(ACT, v);\n\
         {post_res}\
             Y[i] = OUT(v);\n\
         }}\n"
    )
}

/// Bounds check for precompiling `"bias_residual_act"` (`kernels/bounds.rs`).
pub(crate) fn bias_residual_act_fits(spec: &Spec) -> bool {
    spec.get_u32("MODE", 2) <= 2 && spec.get_u32("RES_MODE", 0) <= 2 && spec.get_u32("ACT", 0) <= 3
}

use Access::{Read as R, ReadWrite as RW};

static SILU_BWD: KernelDef = KernelDef {
    name: "silu_bwd",
    storage: &[R, R, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{SILU_BWD_BODY}", prelude(s))),
};
static GELU_BWD: KernelDef = KernelDef {
    name: "gelu_bwd",
    storage: &[R, R, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{GELU_BWD_BODY}", prelude(s))),
};
static SWIGLU_FWD: KernelDef = KernelDef {
    name: "swiglu_fwd",
    storage: &[R, R, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{SWIGLU_FWD_BODY}", prelude(s))),
};
static SWIGLU_BWD: KernelDef = KernelDef {
    name: "swiglu_bwd",
    storage: &[R, R, R, RW, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{SWIGLU_BWD_BODY}", prelude(s))),
};
static GEGLU_FWD: KernelDef = KernelDef {
    name: "geglu_fwd",
    storage: &[R, R, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{GEGLU_FWD_BODY}", prelude(s))),
};
static GEGLU_BWD: KernelDef = KernelDef {
    name: "geglu_bwd",
    storage: &[R, R, R, RW, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{GEGLU_BWD_BODY}", prelude(s))),
};
static GEGLU_BWD_DGATE: KernelDef = KernelDef {
    name: "geglu_bwd_dgate",
    storage: &[R, R, R, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{GEGLU_BWD_DGATE_BODY}", prelude(s))),
};
static GEGLU_BWD_DUP: KernelDef = KernelDef {
    name: "geglu_bwd_dup",
    storage: &[R, R, RW],
    source: Source::Generated(|s| format!("{}{ACT_HELPERS}{GEGLU_BWD_DUP_BODY}", prelude(s))),
};
static BIAS_ADD_SILU: KernelDef =
    KernelDef { name: "bias_add_silu", storage: &[R, R, RW], source: Source::Generated(gen_bias_add_silu) };
static BIAS_RESIDUAL_ACT: KernelDef = KernelDef {
    name: "bias_residual_act",
    storage: &[R, R, R, RW],
    source: Source::Generated(gen_bias_residual_act),
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[
    &SILU_BWD,
    &GELU_BWD,
    &SWIGLU_FWD,
    &SWIGLU_BWD,
    &GEGLU_FWD,
    &GEGLU_BWD,
    &GEGLU_BWD_DGATE,
    &GEGLU_BWD_DUP,
    &BIAS_ADD_SILU,
    &BIAS_RESIDUAL_ACT,
];

fn check_len(n: usize, op: &'static str) -> Result<(bool, usize)> {
    let vec = n.is_multiple_of(4);
    let units = if vec { n / 4 } else { n };
    if units > u32::MAX as usize {
        return Err(GpuError::Limit(format!("{op}: length exceeds u32 indexing")));
    }
    Ok((vec, units))
}

/// Fused `vec4` SiLU backward: `dx[i] = grad[i] * silu'(x[i])` for `n` elements (`F32` or `F16`).
pub fn silu_backward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "silu_backward")?;
    check_buf(x, n, dtype, "silu_backward x")?;
    check_buf(grad, n, dtype, "silu_backward grad")?;
    check_buf(dx, n, dtype, "silu_backward dx")?;
    if n == 0 {
        return Ok(());
    }
    let (vec, units) = check_len(n, "silu_backward")?;
    let k = ctx.kernel(&SILU_BWD, Spec::new().bool("VEC4", vec).f16(dtype == DType::F16))?;
    let p = Params::new().uz(units)?;
    ctx.dispatch(&k, &[x, grad, dx], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Fused `vec4` tanh-approx GELU backward: `dx[i] = grad[i] * gelu'(x[i])` for `n` elements (`F32` or `F16`).
pub fn gelu_backward(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "gelu_backward")?;
    check_buf(x, n, dtype, "gelu_backward x")?;
    check_buf(grad, n, dtype, "gelu_backward grad")?;
    check_buf(dx, n, dtype, "gelu_backward dx")?;
    if n == 0 {
        return Ok(());
    }
    let (vec, units) = check_len(n, "gelu_backward")?;
    let k = ctx.kernel(&GELU_BWD, Spec::new().bool("VEC4", vec).f16(dtype == DType::F16))?;
    let p = Params::new().uz(units)?;
    ctx.dispatch(&k, &[x, grad, dx], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Fused `vec4` SwiGLU forward: `y[i] = silu(gate[i]) * up[i]` for `n` elements (`F32` or `F16`).
pub fn swiglu_forward(
    ctx: &GpuContext,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "swiglu_forward")?;
    check_buf(gate, n, dtype, "swiglu_forward gate")?;
    check_buf(up, n, dtype, "swiglu_forward up")?;
    check_buf(y, n, dtype, "swiglu_forward y")?;
    if n == 0 {
        return Ok(());
    }
    let (vec, units) = check_len(n, "swiglu_forward")?;
    let k = ctx.kernel(&SWIGLU_FWD, Spec::new().bool("VEC4", vec).f16(dtype == DType::F16))?;
    let p = Params::new().uz(units)?;
    ctx.dispatch(&k, &[gate, up, y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Fused `vec4` SwiGLU backward: writes both `dgate` and `dup` in one pass for `n` elements (`F32` or `F16`).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_backward(
    ctx: &GpuContext,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    dgate: &wgpu::Buffer,
    dup: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "swiglu_backward")?;
    check_buf(gate, n, dtype, "swiglu_backward gate")?;
    check_buf(up, n, dtype, "swiglu_backward up")?;
    check_buf(grad, n, dtype, "swiglu_backward grad")?;
    check_buf(dgate, n, dtype, "swiglu_backward dgate")?;
    check_buf(dup, n, dtype, "swiglu_backward dup")?;
    if n == 0 {
        return Ok(());
    }
    let (vec, units) = check_len(n, "swiglu_backward")?;
    let k = ctx.kernel(&SWIGLU_BWD, Spec::new().bool("VEC4", vec).f16(dtype == DType::F16))?;
    let p = Params::new().uz(units)?;
    ctx.dispatch(&k, &[gate, up, grad, dgate, dup], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Fused `vec4` GeGLU forward: `y[i] = gelu(gate[i]) * up[i]` for `n` elements (`F32` or `F16`).
pub fn geglu_forward(
    ctx: &GpuContext,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "geglu_forward")?;
    check_buf(gate, n, dtype, "geglu_forward gate")?;
    check_buf(up, n, dtype, "geglu_forward up")?;
    check_buf(y, n, dtype, "geglu_forward y")?;
    if n == 0 {
        return Ok(());
    }
    if y == gate || y == up {
        return Err(shape_err("geglu_forward: output buffer must not alias gate or up"));
    }
    let (vec, units) = check_len(n, "geglu_forward")?;
    let k = ctx.kernel(&GEGLU_FWD, Spec::new().bool("VEC4", vec).f16(dtype == DType::F16))?;
    let p = Params::new().uz(units)?;
    ctx.dispatch(&k, &[gate, up, y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Fused `vec4` GeGLU backward: writes both `dgate` and `dup` in one pass for `n` elements (`F32` or `F16`).
#[allow(clippy::too_many_arguments)]
pub fn geglu_backward(
    ctx: &GpuContext,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    dgate: &wgpu::Buffer,
    dup: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "geglu_backward")?;
    check_buf(gate, n, dtype, "geglu_backward gate")?;
    check_buf(up, n, dtype, "geglu_backward up")?;
    check_buf(grad, n, dtype, "geglu_backward grad")?;
    check_buf(dgate, n, dtype, "geglu_backward dgate")?;
    check_buf(dup, n, dtype, "geglu_backward dup")?;
    if n == 0 {
        return Ok(());
    }
    if dgate == dup || [gate, up, grad].iter().any(|&src| src == dgate || src == dup) {
        return Err(shape_err("geglu_backward: output buffers must be distinct from inputs and each other"));
    }
    let (vec, units) = check_len(n, "geglu_backward")?;
    let spec = Spec::new().bool("VEC4", vec).f16(dtype == DType::F16);
    let p = Params::new().uz(units)?;
    let grid = ctx.grid_1d(units as u64, 256)?;
    if ctx.device().is_some_and(|dev| dev.limits().max_storage_buffers_per_shader_stage < 5) {
        let k_dg = ctx.kernel(&GEGLU_BWD_DGATE, spec.clone())?;
        ctx.dispatch(&k_dg, &[gate, up, grad, dgate], &p, grid)?;
        let k_du = ctx.kernel(&GEGLU_BWD_DUP, spec)?;
        return ctx.dispatch(&k_du, &[gate, grad, dup], &p, grid);
    }
    let k = ctx.kernel(&GEGLU_BWD, spec)?;
    ctx.dispatch(&k, &[gate, up, grad, dgate, dup], &p, grid)
}

/// Fused `vec4` bias-add + SiLU: `y[i] = silu(x[i] + bias[(i / inner) % channels])` (`F32` or `F16`).
///
/// Selects a layout-specialized `vec4` kernel when `inner % 4 == 0` (NCHW) or `inner == 1 && channels % 4 == 0`
/// (`[rows, C]`), or when `bias` is `None` and `n % 4 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn bias_add_silu(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    n: usize,
    channels: usize,
    inner: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "bias_add_silu")?;
    check_buf(x, n, dtype, "bias_add_silu x")?;
    check_buf(y, n, dtype, "bias_add_silu y")?;
    if let Some(b) = bias {
        if channels == 0 || inner == 0 {
            return Err(shape_err("bias_add_silu: channels and inner must be non-zero"));
        }
        check_buf(b, channels, dtype, "bias_add_silu bias")?;
    }
    if n == 0 {
        return Ok(());
    }
    let (mode, units, p1, p2) = match bias {
        None if n.is_multiple_of(4) => (0u32, n / 4, 1usize, 1usize),
        None => (2u32, n, 1usize, 1usize),
        Some(_) if n.is_multiple_of(4) && inner.is_multiple_of(4) => (0u32, n / 4, channels, inner / 4),
        Some(_) if n.is_multiple_of(4) && inner == 1 && channels.is_multiple_of(4) => {
            (1u32, n / 4, channels / 4, 1usize)
        }
        Some(_) => (2u32, n, channels, inner),
    };
    if units > u32::MAX as usize {
        return Err(GpuError::Limit("bias_add_silu: length exceeds u32 indexing".into()));
    }
    let k = ctx.kernel(
        &BIAS_ADD_SILU,
        Spec::new().u32("MODE", mode).bool("HAS_BIAS", bias.is_some()).f16(dtype == DType::F16),
    )?;
    let p = Params::new().uz(units)?.uz(p1)?.uz(p2)?;
    ctx.dispatch(&k, &[x, bias.unwrap_or(ctx.dummy()), y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// Fused `vec4` bias-add + optional residual + activation (`Identity`, `Silu`, `Gelu`, `Relu`) for `F32` / `F16`.
///
/// Computes `y[i] = act(x[i] + bias[c] + beta * res[i])` when `ResidualMode::PreAct`,
/// `y[i] = act(x[i] + bias[c]) + beta * res[i]` when `ResidualMode::PostAct`, or `y[i] = act(x[i] + bias[c])`
/// when `residual` is `None` (where `c = (i / inner) % channels`). `y` must not alias `x`, `bias`, or `residual`.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_act(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    residual: Option<(&wgpu::Buffer, f32, ResidualMode)>,
    y: &wgpu::Buffer,
    n: usize,
    channels: usize,
    inner: usize,
    act: FusedAct,
    dtype: DType,
) -> Result<()> {
    bias_residual_act_impl(ctx, x, bias, residual, y, n, channels, inner, act, dtype, false)
}

/// [`bias_residual_act`] reading `x` as `F32` while `bias`, `residual`, and `y` use `dtype` (keeps full
/// `F32` intermediate precision for `F16` depthwise conv epilogues).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bias_residual_act_from_f32(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    residual: Option<(&wgpu::Buffer, f32, ResidualMode)>,
    y: &wgpu::Buffer,
    n: usize,
    channels: usize,
    inner: usize,
    act: FusedAct,
    dtype: DType,
) -> Result<()> {
    bias_residual_act_impl(ctx, x, bias, residual, y, n, channels, inner, act, dtype, dtype == DType::F16)
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_act_impl(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    residual: Option<(&wgpu::Buffer, f32, ResidualMode)>,
    y: &wgpu::Buffer,
    n: usize,
    channels: usize,
    inner: usize,
    act: FusedAct,
    dtype: DType,
    x_f32: bool,
) -> Result<()> {
    check_dtype(ctx, dtype, "bias_residual_act")?;
    let x_dtype = if x_f32 { DType::F32 } else { dtype };
    check_buf(x, n, x_dtype, "bias_residual_act x")?;
    check_buf(y, n, dtype, "bias_residual_act y")?;
    if let Some(b) = bias {
        if channels == 0 || inner == 0 {
            return Err(shape_err("bias_residual_act: channels and inner must be non-zero"));
        }
        check_buf(b, channels, dtype, "bias_residual_act bias")?;
    }
    let active_res = residual.filter(|(_, _, m)| *m != ResidualMode::None);
    let (res_buf, beta, res_mode) = match active_res {
        Some((rb, b, m)) => {
            check_buf(rb, n, dtype, "bias_residual_act residual")?;
            (rb, b, m)
        }
        None => (ctx.dummy(), 0.0, ResidualMode::None),
    };
    if n == 0 {
        return Ok(());
    }
    if std::iter::once(x).chain(bias).chain(active_res.map(|(rb, _, _)| rb)).any(|buf| buf == y) {
        return Err(shape_err("bias_residual_act: output buffer must not alias x, bias, or residual"));
    }
    let (mode, units, p1, p2) = match bias {
        None if n.is_multiple_of(4) => (0u32, n / 4, 1usize, 1usize),
        None => (2u32, n, 1usize, 1usize),
        Some(_) if n.is_multiple_of(4) && inner.is_multiple_of(4) => (0u32, n / 4, channels, inner / 4),
        Some(_) if n.is_multiple_of(4) && inner == 1 && channels.is_multiple_of(4) => {
            (1u32, n / 4, channels / 4, 1usize)
        }
        Some(_) => (2u32, n, channels, inner),
    };
    if units > u32::MAX as usize {
        return Err(GpuError::Limit("bias_residual_act: length exceeds u32 indexing".into()));
    }
    let k = ctx.kernel(
        &BIAS_RESIDUAL_ACT,
        Spec::new()
            .u32("MODE", mode)
            .u32("ACT", act.code())
            .u32("RES_MODE", res_mode.code())
            .bool("HAS_BIAS", bias.is_some())
            .bool("X_F32", x_f32)
            .f16(dtype == DType::F16),
    )?;
    let p = Params::new().uz(units)?.uz(p1)?.uz(p2)?.f(beta);
    ctx.dispatch(&k, &[x, bias.unwrap_or(ctx.dummy()), res_buf, y], &p, ctx.grid_1d(units as u64, 256)?)
}
