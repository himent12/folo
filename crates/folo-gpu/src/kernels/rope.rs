//! WGSL 2D Rotary Positional Embedding (`RoPE2d`) forward and backward kernels (`F32` and `F16`).
//!
//! Operates on dense `[outer, height, width, head_dim]` tensors (e.g. `outer = B * H_heads`), applying axial 2D
//! rotations to the first `dim_y + dim_x <= head_dim` channels:
//! - Channels `[0 .. dim_y)` rotate using the grid `y` coordinate (`0..height`).
//! - Channels `[dim_y .. dim_y + dim_x)` rotate using the grid `x` coordinate (`0..width`).
//! - Trailing channels `[dim_y + dim_x .. head_dim)` pass through unchanged.
//!
//! Supports both [`RopeLayout::HalfSplit`] (NeoX pairs `(i, i + half)`) and [`RopeLayout::Interleaved`] (adjacent
//! pairs `(2i, 2i + 1)`), with a `vec4` 2-pair-per-thread fast path when interleaved and 4-aligned. Provides both
//! precomputed `F32` `(cos, sin)` tables ([`rope2d_forward`] / [`rope2d_backward`]) and zero-table on-the-fly trig
//! ([`rope2d_trig_forward`] / [`rope2d_trig_backward`]).

use super::{check_buf, prod, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Channel-pairing convention for rotary embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// NeoX / Vision-RoPE half-split: pairs `(i, i + dim / 2)` within each axis slice.
    HalfSplit,
    /// Interleaved adjacent pairs: `(2*i, 2*i + 1)` within each axis slice.
    Interleaved,
}

/// Shape and configuration for a 2D RoPE pass over dense `[outer, height, width, head_dim]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rope2dDesc {
    /// Leading dimensions product (e.g. `batch * num_heads`).
    pub outer: usize,
    /// 2D spatial grid height (`H`).
    pub height: usize,
    /// 2D spatial grid width (`W`).
    pub width: usize,
    /// Head dimension (`D`).
    pub head_dim: usize,
    /// Channels rotated by `y` coordinate (must be even).
    pub dim_y: usize,
    /// Channels rotated by `x` coordinate (must be even; `dim_y + dim_x <= head_dim`).
    pub dim_x: usize,
    /// Pair layout (`HalfSplit` or `Interleaved`).
    pub layout: RopeLayout,
    /// Activation storage dtype (`F32` or `F16`; trig tables and rotation math are always `F32`).
    pub dtype: DType,
}

impl Rope2dDesc {
    /// Standard full-dimension axial 2D RoPE splitting `head_dim` evenly (`dim_y = dim_x = head_dim / 2`).
    pub fn split_even(
        outer: usize,
        height: usize,
        width: usize,
        head_dim: usize,
        layout: RopeLayout,
        dtype: DType,
    ) -> Self {
        let half = head_dim / 2;
        Self { outer, height, width, head_dim, dim_y: half, dim_x: half, layout, dtype }
    }

    fn validate(&self, op: &'static str) -> Result<usize> {
        if !self.dim_y.is_multiple_of(2) || !self.dim_x.is_multiple_of(2) {
            return Err(shape_err(format!(
                "{op}: dim_y ({}) and dim_x ({}) must be even",
                self.dim_y, self.dim_x
            )));
        }
        let rot =
            self.dim_y.checked_add(self.dim_x).ok_or_else(|| shape_err(format!("{op}: dim overflow")))?;
        if rot > self.head_dim {
            return Err(shape_err(format!(
                "{op}: dim_y ({}) + dim_x ({}) exceeds head_dim ({})",
                self.dim_y, self.dim_x, self.head_dim
            )));
        }
        let total = prod(&[self.outer, self.height, self.width, self.head_dim])?;
        if total > u32::MAX as usize {
            return Err(GpuError::Limit(format!("{op}: tensor exceeds u32 indexing")));
        }
        Ok(total)
    }

    fn can_vec4(&self) -> bool {
        self.layout == RopeLayout::Interleaved
            && self.head_dim.is_multiple_of(4)
            && self.dim_y.is_multiple_of(4)
            && self.dim_x.is_multiple_of(4)
    }
}

fn prelude(spec: &Spec) -> &'static str {
    let vec = spec.get_bool("VEC4");
    if spec.is_f16() { vec_prelude_f16(vec) } else { vec_prelude(vec) }
}

/// Table-driven 2D RoPE shader.
/// Spec: `INTERLEAVED`, `BACKWARD`, `VEC4`, `f16`.
/// Params: 0 = units (`total / 4` if VEC4 else `total`), 1 = `height`, 2 = `width`,
/// 3 = `head_dim_u` (`head_dim / 4` if VEC4 else `head_dim`), 4 = `dim_y`, 5 = `dim_x`.
fn gen_rope2d_table(spec: &Spec) -> String {
    let pre = prelude(spec);
    let vec = spec.get_bool("VEC4");
    let body = if vec {
        r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> COS_Y: array<f32>;
@group(0) @binding(3) var<storage, read> SIN_Y: array<f32>;
@group(0) @binding(4) var<storage, read> COS_X: array<f32>;
@group(0) @binding(5) var<storage, read> SIN_X: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = pu(1u);
    let w = pu(2u);
    let du = pu(3u);
    let dim_y = pu(4u);
    let dim_x = pu(5u);
    let d4 = i % du;
    let tok = i / du;
    let wx = tok % w;
    let hy = (tok / w) % h;
    let d = d4 * 4u;
    let v = V4(X[i]);
    if (d < dim_y) {
        let half_y = dim_y >> 1u;
        let p = hy * half_y + (d >> 1u);
        let c0 = COS_Y[p];
        let c1 = COS_Y[p + 1u];
        let s0 = select(SIN_Y[p], -SIN_Y[p], BACKWARD);
        let s1 = select(SIN_Y[p + 1u], -SIN_Y[p + 1u], BACKWARD);
        Y[i] = OUT(vec4<f32>(
            v.x * c0 - v.y * s0,
            v.x * s0 + v.y * c0,
            v.z * c1 - v.w * s1,
            v.z * s1 + v.w * c1,
        ));
    } else if (d < dim_y + dim_x) {
        let half_x = dim_x >> 1u;
        let p = wx * half_x + ((d - dim_y) >> 1u);
        let c0 = COS_X[p];
        let c1 = COS_X[p + 1u];
        let s0 = select(SIN_X[p], -SIN_X[p], BACKWARD);
        let s1 = select(SIN_X[p + 1u], -SIN_X[p + 1u], BACKWARD);
        Y[i] = OUT(vec4<f32>(
            v.x * c0 - v.y * s0,
            v.x * s0 + v.y * c0,
            v.z * c1 - v.w * s1,
            v.z * s1 + v.w * c1,
        ));
    } else {
        Y[i] = OUT(v);
    }
}
"#
    } else {
        r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read> COS_Y: array<f32>;
@group(0) @binding(3) var<storage, read> SIN_Y: array<f32>;
@group(0) @binding(4) var<storage, read> COS_X: array<f32>;
@group(0) @binding(5) var<storage, read> SIN_X: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = pu(1u);
    let w = pu(2u);
    let d_all = pu(3u);
    let dim_y = pu(4u);
    let dim_x = pu(5u);
    let d = i % d_all;
    let tok = i / d_all;
    let base = tok * d_all;
    let xv = V4(X[i]).x;
    if (d < dim_y) {
        let half_y = dim_y >> 1u;
        let hy = (tok / w) % h;
        var pair = 0u;
        var is_second = false;
        var partner = 0u;
        if (INTERLEAVED) {
            pair = d >> 1u;
            is_second = (d & 1u) != 0u;
            partner = select(d + 1u, d - 1u, is_second);
        } else {
            is_second = d >= half_y;
            pair = select(d, d - half_y, is_second);
            partner = select(d + half_y, d - half_y, is_second);
        }
        let pv = V4(X[base + partner]).x;
        let ti = hy * half_y + pair;
        let c = COS_Y[ti];
        let s = select(SIN_Y[ti], -SIN_Y[ti], BACKWARD);
        let out_v = select(xv * c - pv * s, pv * s + xv * c, is_second);
        Y[i] = OUT(vec4<f32>(out_v));
    } else if (d < dim_y + dim_x) {
        let dr = d - dim_y;
        let half_x = dim_x >> 1u;
        let wx = tok % w;
        var pair = 0u;
        var is_second = false;
        var partner = 0u;
        if (INTERLEAVED) {
            pair = dr >> 1u;
            is_second = (dr & 1u) != 0u;
            partner = dim_y + select(dr + 1u, dr - 1u, is_second);
        } else {
            is_second = dr >= half_x;
            pair = select(dr, dr - half_x, is_second);
            partner = dim_y + select(dr + half_x, dr - half_x, is_second);
        }
        let pv = V4(X[base + partner]).x;
        let ti = wx * half_x + pair;
        let c = COS_X[ti];
        let s = select(SIN_X[ti], -SIN_X[ti], BACKWARD);
        let out_v = select(xv * c - pv * s, pv * s + xv * c, is_second);
        Y[i] = OUT(vec4<f32>(out_v));
    } else {
        Y[i] = OUT(vec4<f32>(xv));
    }
}
"#
    };
    format!("{pre}{body}")
}

/// On-the-fly trigonometric 2D RoPE shader (zero frequency-table reads).
/// Spec: `INTERLEAVED`, `BACKWARD`, `VEC4`, `f16`.
/// Params: 0 = units, 1 = `height`, 2 = `width`, 3 = `head_dim_u`, 4 = `dim_y`, 5 = `dim_x`,
/// 6 = `-2.0 * ln(base_y) / dim_y` (f32), 7 = `-2.0 * ln(base_x) / dim_x` (f32).
fn gen_rope2d_trig(spec: &Spec) -> String {
    let pre = prelude(spec);
    let vec = spec.get_bool("VEC4");
    let body = if vec {
        r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = pu(1u);
    let w = pu(2u);
    let du = pu(3u);
    let dim_y = pu(4u);
    let dim_x = pu(5u);
    let d4 = i % du;
    let tok = i / du;
    let wx = tok % w;
    let hy = (tok / w) % h;
    let d = d4 * 4u;
    let v = V4(X[i]);
    if (d < dim_y) {
        let p = d >> 1u;
        let k = pf(6u);
        let a0 = f32(hy) * exp(f32(p) * k);
        let a1 = f32(hy) * exp(f32(p + 1u) * k);
        let c0 = cos(a0); let c1 = cos(a1);
        let s0 = select(sin(a0), -sin(a0), BACKWARD);
        let s1 = select(sin(a1), -sin(a1), BACKWARD);
        Y[i] = OUT(vec4<f32>(
            v.x * c0 - v.y * s0,
            v.x * s0 + v.y * c0,
            v.z * c1 - v.w * s1,
            v.z * s1 + v.w * c1,
        ));
    } else if (d < dim_y + dim_x) {
        let p = (d - dim_y) >> 1u;
        let k = pf(7u);
        let a0 = f32(wx) * exp(f32(p) * k);
        let a1 = f32(wx) * exp(f32(p + 1u) * k);
        let c0 = cos(a0); let c1 = cos(a1);
        let s0 = select(sin(a0), -sin(a0), BACKWARD);
        let s1 = select(sin(a1), -sin(a1), BACKWARD);
        Y[i] = OUT(vec4<f32>(
            v.x * c0 - v.y * s0,
            v.x * s0 + v.y * c0,
            v.z * c1 - v.w * s1,
            v.z * s1 + v.w * c1,
        ));
    } else {
        Y[i] = OUT(v);
    }
}
"#
    } else {
        r#"
@group(0) @binding(1) var<storage, read> X: array<E>;
@group(0) @binding(2) var<storage, read_write> Y: array<E>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_id(wid, nwg, lid, 256u);
    if (i >= pu(0u)) { return; }
    let h = pu(1u);
    let w = pu(2u);
    let d_all = pu(3u);
    let dim_y = pu(4u);
    let dim_x = pu(5u);
    let d = i % d_all;
    let tok = i / d_all;
    let base = tok * d_all;
    let xv = V4(X[i]).x;
    if (d < dim_y) {
        let half_y = dim_y >> 1u;
        let hy = (tok / w) % h;
        var pair = 0u;
        var is_second = false;
        var partner = 0u;
        if (INTERLEAVED) {
            pair = d >> 1u;
            is_second = (d & 1u) != 0u;
            partner = select(d + 1u, d - 1u, is_second);
        } else {
            is_second = d >= half_y;
            pair = select(d, d - half_y, is_second);
            partner = select(d + half_y, d - half_y, is_second);
        }
        let pv = V4(X[base + partner]).x;
        let angle = f32(hy) * exp(f32(pair) * pf(6u));
        let c = cos(angle);
        let s = select(sin(angle), -sin(angle), BACKWARD);
        let out_v = select(xv * c - pv * s, pv * s + xv * c, is_second);
        Y[i] = OUT(vec4<f32>(out_v));
    } else if (d < dim_y + dim_x) {
        let dr = d - dim_y;
        let half_x = dim_x >> 1u;
        let wx = tok % w;
        var pair = 0u;
        var is_second = false;
        var partner = 0u;
        if (INTERLEAVED) {
            pair = dr >> 1u;
            is_second = (dr & 1u) != 0u;
            partner = dim_y + select(dr + 1u, dr - 1u, is_second);
        } else {
            is_second = dr >= half_x;
            pair = select(dr, dr - half_x, is_second);
            partner = dim_y + select(dr + half_x, dr - half_x, is_second);
        }
        let pv = V4(X[base + partner]).x;
        let angle = f32(wx) * exp(f32(pair) * pf(7u));
        let c = cos(angle);
        let s = select(sin(angle), -sin(angle), BACKWARD);
        let out_v = select(xv * c - pv * s, pv * s + xv * c, is_second);
        Y[i] = OUT(vec4<f32>(out_v));
    } else {
        Y[i] = OUT(vec4<f32>(xv));
    }
}
"#
    };
    format!("{pre}{body}")
}

use Access::{Read as R, ReadWrite as RW};

static ROPE2D_TABLE: KernelDef = KernelDef {
    name: "rope2d_table",
    storage: &[R, R, R, R, R, RW],
    source: Source::Generated(gen_rope2d_table),
};
static ROPE2D_TRIG: KernelDef =
    KernelDef { name: "rope2d_trig", storage: &[R, RW], source: Source::Generated(gen_rope2d_trig) };

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&ROPE2D_TABLE, &ROPE2D_TRIG];

#[allow(clippy::too_many_arguments)]
fn rope2d_table_impl(
    ctx: &GpuContext,
    desc: &Rope2dDesc,
    x: &wgpu::Buffer,
    cos_y: &wgpu::Buffer,
    sin_y: &wgpu::Buffer,
    cos_x: &wgpu::Buffer,
    sin_x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    backward: bool,
) -> Result<()> {
    let op = if backward { "rope2d_backward" } else { "rope2d_forward" };
    check_dtype(ctx, desc.dtype, op)?;
    let total = desc.validate(op)?;
    check_buf(x, total, desc.dtype, op)?;
    check_buf(y, total, desc.dtype, op)?;
    let ny = prod(&[desc.height, desc.dim_y / 2])?;
    let nx = prod(&[desc.width, desc.dim_x / 2])?;
    if ny > 0 {
        check_buf(cos_y, ny, DType::F32, op)?;
        check_buf(sin_y, ny, DType::F32, op)?;
    }
    if nx > 0 {
        check_buf(cos_x, nx, DType::F32, op)?;
        check_buf(sin_x, nx, DType::F32, op)?;
    }
    if total == 0 {
        return Ok(());
    }
    let vec4 = desc.can_vec4();
    let units = if vec4 { total / 4 } else { total };
    let du = if vec4 { desc.head_dim / 4 } else { desc.head_dim };
    let k = ctx.kernel(
        &ROPE2D_TABLE,
        Spec::new()
            .bool("INTERLEAVED", desc.layout == RopeLayout::Interleaved)
            .bool("BACKWARD", backward)
            .bool("VEC4", vec4)
            .f16(desc.dtype == DType::F16),
    )?;
    let p =
        Params::new().uz(units)?.uz(desc.height)?.uz(desc.width)?.uz(du)?.uz(desc.dim_y)?.uz(desc.dim_x)?;
    ctx.dispatch(
        &k,
        &[
            x,
            if ny > 0 { cos_y } else { ctx.dummy() },
            if ny > 0 { sin_y } else { ctx.dummy() },
            if nx > 0 { cos_x } else { ctx.dummy() },
            if nx > 0 { sin_x } else { ctx.dummy() },
            y,
        ],
        &p,
        ctx.grid_1d(units as u64, 256)?,
    )
}

/// Table-driven 2D RoPE forward: applies axial `(cos_y, sin_y)` and `(cos_x, sin_x)` rotations to `x` -> `y`.
#[allow(clippy::too_many_arguments)]
pub fn rope2d_forward(
    ctx: &GpuContext,
    desc: &Rope2dDesc,
    x: &wgpu::Buffer,
    cos_y: &wgpu::Buffer,
    sin_y: &wgpu::Buffer,
    cos_x: &wgpu::Buffer,
    sin_x: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Result<()> {
    rope2d_table_impl(ctx, desc, x, cos_y, sin_y, cos_x, sin_x, y, false)
}

/// Table-driven 2D RoPE backward (inverse rotation by `-theta`): computes `dx` from upstream `grad_y`.
#[allow(clippy::too_many_arguments)]
pub fn rope2d_backward(
    ctx: &GpuContext,
    desc: &Rope2dDesc,
    grad_y: &wgpu::Buffer,
    cos_y: &wgpu::Buffer,
    sin_y: &wgpu::Buffer,
    cos_x: &wgpu::Buffer,
    sin_x: &wgpu::Buffer,
    dx: &wgpu::Buffer,
) -> Result<()> {
    rope2d_table_impl(ctx, desc, grad_y, cos_y, sin_y, cos_x, sin_x, dx, true)
}

fn rope2d_trig_impl(
    ctx: &GpuContext,
    desc: &Rope2dDesc,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    base_y: f32,
    base_x: f32,
    backward: bool,
) -> Result<()> {
    let op = if backward { "rope2d_trig_backward" } else { "rope2d_trig_forward" };
    check_dtype(ctx, desc.dtype, op)?;
    if (desc.dim_y > 0 && (!base_y.is_finite() || base_y <= 0.0))
        || (desc.dim_x > 0 && (!base_x.is_finite() || base_x <= 0.0))
    {
        return Err(shape_err(format!("{op}: base frequencies must be finite and > 0")));
    }
    let total = desc.validate(op)?;
    check_buf(x, total, desc.dtype, op)?;
    check_buf(y, total, desc.dtype, op)?;
    if total == 0 {
        return Ok(());
    }
    let vec4 = desc.can_vec4();
    let units = if vec4 { total / 4 } else { total };
    let du = if vec4 { desc.head_dim / 4 } else { desc.head_dim };
    let ky = if desc.dim_y > 0 { (-2.0 * (base_y as f64).ln() / desc.dim_y as f64) as f32 } else { 0.0 };
    let kx = if desc.dim_x > 0 { (-2.0 * (base_x as f64).ln() / desc.dim_x as f64) as f32 } else { 0.0 };
    let k = ctx.kernel(
        &ROPE2D_TRIG,
        Spec::new()
            .bool("INTERLEAVED", desc.layout == RopeLayout::Interleaved)
            .bool("BACKWARD", backward)
            .bool("VEC4", vec4)
            .f16(desc.dtype == DType::F16),
    )?;
    let p = Params::new()
        .uz(units)?
        .uz(desc.height)?
        .uz(desc.width)?
        .uz(du)?
        .uz(desc.dim_y)?
        .uz(desc.dim_x)?
        .f(ky)
        .f(kx);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(units as u64, 256)?)
}

/// On-the-fly trigonometric 2D RoPE forward with frequency bases `(base_y, base_x)` (e.g. `10_000.0`).
pub fn rope2d_trig_forward(
    ctx: &GpuContext,
    desc: &Rope2dDesc,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    base_y: f32,
    base_x: f32,
) -> Result<()> {
    rope2d_trig_impl(ctx, desc, x, y, base_y, base_x, false)
}

/// On-the-fly trigonometric 2D RoPE backward (inverse rotation) with frequency bases `(base_y, base_x)`.
pub fn rope2d_trig_backward(
    ctx: &GpuContext,
    desc: &Rope2dDesc,
    grad_y: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    base_y: f32,
    base_x: f32,
) -> Result<()> {
    rope2d_trig_impl(ctx, desc, grad_y, dx, base_y, base_x, true)
}
