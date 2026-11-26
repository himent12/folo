//! Standalone `vec4`-vectorized WGSL `where_select` kernel with safe non-multiple-of-4 tail handling,
//! broadcast operands, `F32` / `U8` conditions, and `F32` / `F16` values.
//!
//! For any length `n >= 4` whose operands are `vec4`-aligned (`Dense`, `Scalar`, or `RepeatInner`/`TileInner`
//! with `inner % 4 == 0`), dispatches a 128-bit `vec4` kernel over the aligned prefix `[0 .. 4 * (n / 4))` and
//! finishes the `1..=3` tail elements with a single-workgroup scalar tail pass (zero buffer over-reads/writes).

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Addressing pattern of one operand in [`where_select`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectBroadcast {
    /// Dense buffer of `n` elements (`buf[i]`).
    Dense,
    /// Single-element broadcast (`buf[0]`).
    Scalar,
    /// Outer/channel broadcast: `n / inner` elements repeated every `inner` (`buf[i / inner]`).
    RepeatInner {
        /// Inner repeat count (must be non-zero and divide `n`).
        inner: usize,
    },
    /// Inner/feature broadcast: `inner` elements tiled across `n / inner` rows (`buf[i % inner]`).
    TileInner {
        /// Inner tile length (must be non-zero and divide `n`).
        inner: usize,
    },
}

impl SelectBroadcast {
    fn mode(self) -> u32 {
        match self {
            SelectBroadcast::Dense => 0,
            SelectBroadcast::Scalar => 1,
            SelectBroadcast::RepeatInner { .. } => 2,
            SelectBroadcast::TileInner { .. } => 3,
        }
    }

    fn vec4_ok(self) -> bool {
        match self {
            SelectBroadcast::Dense | SelectBroadcast::Scalar => true,
            SelectBroadcast::RepeatInner { inner } | SelectBroadcast::TileInner { inner } => {
                inner.is_multiple_of(4)
            }
        }
    }

    fn inner_or_one(self) -> usize {
        match self {
            SelectBroadcast::Dense | SelectBroadcast::Scalar => 1,
            SelectBroadcast::RepeatInner { inner } | SelectBroadcast::TileInner { inner } => inner,
        }
    }

    fn required_elems(self, n: usize, what: &'static str) -> Result<usize> {
        match self {
            SelectBroadcast::Dense => Ok(n),
            SelectBroadcast::Scalar => Ok(if n == 0 { 0 } else { 1 }),
            SelectBroadcast::RepeatInner { inner } => {
                if inner == 0 || !n.is_multiple_of(inner) {
                    return Err(shape_err(format!(
                        "{what}: inner {inner} must be non-zero and divide n={n}"
                    )));
                }
                Ok(n / inner)
            }
            SelectBroadcast::TileInner { inner } => {
                if inner == 0 || !n.is_multiple_of(inner) {
                    return Err(shape_err(format!(
                        "{what}: inner {inner} must be non-zero and divide n={n}"
                    )));
                }
                Ok(if n == 0 { 0 } else { inner })
            }
        }
    }
}

/// One operand of [`where_select`].
#[derive(Clone, Copy, Debug)]
pub struct SelectOperand<'a> {
    /// Storage buffer.
    pub buf: &'a wgpu::Buffer,
    /// Broadcast pattern over `[0 .. n)`.
    pub broadcast: SelectBroadcast,
}

impl<'a> SelectOperand<'a> {
    /// Dense `n`-element operand.
    pub fn dense(buf: &'a wgpu::Buffer) -> Self {
        Self { buf, broadcast: SelectBroadcast::Dense }
    }

    /// 1-element scalar broadcast operand.
    pub fn scalar(buf: &'a wgpu::Buffer) -> Self {
        Self { buf, broadcast: SelectBroadcast::Scalar }
    }
}

/// Vectorized `where_select` shader (`vec4` per thread over `n / 4` aligned units).
/// Spec: `MC`, `MA`, `MB` (`0` dense, `1` scalar, `2` repeat_inner, `3` tile_inner), `COND_U8`, `f16`.
/// Params: 0 = `units` (`n / 4`), 1 = `ic4`, 2 = `ia4`, 3 = `ib4`.
fn gen_where_select_vec4(spec: &Spec) -> String {
    let mc = spec.get_u32("MC", 0);
    let ma = spec.get_u32("MA", 0);
    let mb = spec.get_u32("MB", 0);
    let cond_u8 = spec.get_bool("COND_U8");
    let half = spec.is_f16();
    let v_ty = if half { "vec4<f16>" } else { "vec4<f32>" };
    let s_ty = if half { "f16" } else { "f32" };
    let cond_ty = if cond_u8 {
        "u32"
    } else if mc == 0 || mc == 3 {
        "vec4<f32>"
    } else {
        "f32"
    };
    let a_ty = if ma == 0 || ma == 3 { v_ty } else { s_ty };
    let b_ty = if mb == 0 || mb == 3 { v_ty } else { s_ty };

    let cond_expr = if cond_u8 {
        match mc {
            0 => "let cw = COND[t];\n    let c = (vec4<u32>(cw) & vec4<u32>(0xffu, 0xff00u, 0xff0000u, 0xff000000u)) != vec4<u32>(0u);".to_string(),
            1 => "let c = vec4<bool>((COND[0] & 0xffu) != 0u);".to_string(),
            2 => "let ci = t / pu(1u);\n    let c = vec4<bool>(((COND[ci >> 2u] >> ((ci & 3u) * 8u)) & 0xffu) != 0u);".to_string(),
            _ => "let cw = COND[t % pu(1u)];\n    let c = (vec4<u32>(cw) & vec4<u32>(0xffu, 0xff00u, 0xff0000u, 0xff000000u)) != vec4<u32>(0u);".to_string(),
        }
    } else {
        match mc {
            0 => {
                "let c = (bitcast<vec4<u32>>(COND[t]) & vec4<u32>(0x7fffffffu)) != vec4<u32>(0u);".to_string()
            }
            1 => "let c = vec4<bool>((bitcast<u32>(COND[0]) & 0x7fffffffu) != 0u);".to_string(),
            2 => "let c = vec4<bool>((bitcast<u32>(COND[t / pu(1u)]) & 0x7fffffffu) != 0u);".to_string(),
            _ => "let c = (bitcast<vec4<u32>>(COND[t % pu(1u)]) & vec4<u32>(0x7fffffffu)) != vec4<u32>(0u);"
                .to_string(),
        }
    };
    let val_expr = |name: &str, mode: u32, p_idx: u32| -> String {
        match mode {
            0 => format!("{name}[t]"),
            1 => format!("{v_ty}({name}[0])"),
            2 => format!("{v_ty}({name}[t / pu({p_idx}u)])"),
            _ => format!("{name}[t % pu({p_idx}u)]"),
        }
    };
    let va = val_expr("A", ma, 2);
    let vb = val_expr("B", mb, 3);
    format!(
        "@group(0) @binding(1) var<storage, read> COND: array<{cond_ty}>;\n\
         @group(0) @binding(2) var<storage, read> A: array<{a_ty}>;\n\
         @group(0) @binding(3) var<storage, read> B: array<{b_ty}>;\n\
         @group(0) @binding(4) var<storage, read_write> OUT: array<{v_ty}>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let t = linear_id(wid, nwg, lid, 256u);\n\
             if (t >= pu(0u)) {{ return; }}\n\
             {cond_expr}\n\
             OUT[t] = select({vb}, {va}, c);\n\
         }}\n"
    )
}

/// Scalar `where_select` shader supporting an arbitrary starting element offset (handles odd shapes and
/// non-multiple-of-4 tails).
/// Spec: `MC`, `MA`, `MB`, `COND_U8`, `f16`.
/// Params: 0 = `count`, 1 = `offset`, 2 = `ic`, 3 = `ia`, 4 = `ib`.
fn gen_where_select_scalar(spec: &Spec) -> String {
    let mc = spec.get_u32("MC", 0);
    let ma = spec.get_u32("MA", 0);
    let mb = spec.get_u32("MB", 0);
    let cond_u8 = spec.get_bool("COND_U8");
    let half = spec.is_f16();
    let s_ty = if half { "f16" } else { "f32" };
    let cond_ty = if cond_u8 { "u32" } else { "f32" };
    let idx_expr = |mode: u32, p_idx: u32| -> String {
        match mode {
            0 => "i".to_string(),
            1 => "0u".to_string(),
            2 => format!("i / pu({p_idx}u)"),
            _ => format!("i % pu({p_idx}u)"),
        }
    };
    let ci = idx_expr(mc, 2);
    let ai = idx_expr(ma, 3);
    let bi = idx_expr(mb, 4);
    let cond_line = if cond_u8 {
        format!("let ci = {ci};\n    let c = ((COND[ci >> 2u] >> ((ci & 3u) * 8u)) & 0xffu) != 0u;")
    } else {
        format!("let c = (bitcast<u32>(COND[{ci}]) & 0x7fffffffu) != 0u;")
    };
    format!(
        "@group(0) @binding(1) var<storage, read> COND: array<{cond_ty}>;\n\
         @group(0) @binding(2) var<storage, read> A: array<{s_ty}>;\n\
         @group(0) @binding(3) var<storage, read> B: array<{s_ty}>;\n\
         @group(0) @binding(4) var<storage, read_write> OUT: array<{s_ty}>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
                 @builtin(local_invocation_index) lid: u32) {{\n\
             let t = linear_id(wid, nwg, lid, 256u);\n\
             if (t >= pu(0u)) {{ return; }}\n\
             let i = pu(1u) + t;\n\
             {cond_line}\n\
             OUT[i] = select(B[{bi}], A[{ai}], c);\n\
         }}\n"
    )
}

/// Bounds check for precompiling `"where_select_vec4"` / `"where_select_scalar"` (`kernels/bounds.rs`).
pub(crate) fn where_select_fits(spec: &Spec) -> bool {
    spec.get_u32("MC", 0) <= 3 && spec.get_u32("MA", 0) <= 3 && spec.get_u32("MB", 0) <= 3
}

use Access::{Read as R, ReadWrite as RW};

static WHERE_SELECT_VEC4: KernelDef = KernelDef {
    name: "where_select_vec4",
    storage: &[R, R, R, RW],
    source: Source::Generated(gen_where_select_vec4),
};
static WHERE_SELECT_SCALAR: KernelDef = KernelDef {
    name: "where_select_scalar",
    storage: &[R, R, R, RW],
    source: Source::Generated(gen_where_select_scalar),
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&WHERE_SELECT_VEC4, &WHERE_SELECT_SCALAR];

/// `out[i] = cond[i] != 0 ? a[i] : b[i]` over `n` elements (`F32` or `F16` values; `F32` or `U8` condition).
///
/// Uses a 128-bit `vec4` kernel across the full `4 * (n / 4)` prefix whenever operand broadcasts are 4-aligned
/// (finishing any `1..=3` tail elements safely in a scalar pass), falling back to scalar otherwise.
#[allow(clippy::too_many_arguments)]
pub fn where_select(
    ctx: &GpuContext,
    cond: SelectOperand<'_>,
    cond_u8: bool,
    a: SelectOperand<'_>,
    b: SelectOperand<'_>,
    out: &wgpu::Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype(ctx, dtype, "where_select")?;
    let c_elems = cond.broadcast.required_elems(n, "where_select cond")?;
    let a_elems = a.broadcast.required_elems(n, "where_select a")?;
    let b_elems = b.broadcast.required_elems(n, "where_select b")?;
    if cond_u8 {
        let need = (c_elems as u64).next_multiple_of(4);
        if cond.buf.size() < need {
            return Err(shape_err(format!(
                "where_select cond: buffer of {} bytes is smaller than {need} bytes",
                cond.buf.size()
            )));
        }
    } else {
        check_buf(cond.buf, c_elems, DType::F32, "where_select cond")?;
    }
    check_buf(a.buf, a_elems, dtype, "where_select a")?;
    check_buf(b.buf, b_elems, dtype, "where_select b")?;
    check_buf(out, n, dtype, "where_select out")?;
    if n == 0 {
        return Ok(());
    }
    if n > u32::MAX as usize {
        return Err(GpuError::Limit("where_select: length exceeds u32 indexing".into()));
    }

    let spec = Spec::new()
        .u32("MC", cond.broadcast.mode())
        .u32("MA", a.broadcast.mode())
        .u32("MB", b.broadcast.mode())
        .bool("COND_U8", cond_u8)
        .f16(dtype == DType::F16);

    let can_vec4 = n >= 4 && cond.broadcast.vec4_ok() && a.broadcast.vec4_ok() && b.broadcast.vec4_ok();

    if can_vec4 {
        let full = n / 4;
        let k4 = ctx.kernel(&WHERE_SELECT_VEC4, spec.clone())?;
        let p4 = Params::new()
            .uz(full)?
            .uz(cond.broadcast.inner_or_one() / 4)?
            .uz(a.broadcast.inner_or_one() / 4)?
            .uz(b.broadcast.inner_or_one() / 4)?;
        ctx.dispatch(&k4, &[cond.buf, a.buf, b.buf, out], &p4, ctx.grid_1d(full as u64, 256)?)?;

        let rem = n % 4;
        if rem > 0 {
            let ks = ctx.kernel(&WHERE_SELECT_SCALAR, spec)?;
            let ps = Params::new()
                .uz(rem)?
                .uz(full * 4)?
                .uz(cond.broadcast.inner_or_one())?
                .uz(a.broadcast.inner_or_one())?
                .uz(b.broadcast.inner_or_one())?;
            ctx.dispatch(&ks, &[cond.buf, a.buf, b.buf, out], &ps, [1, 1, 1])?;
        }
        return Ok(());
    }

    let ks = ctx.kernel(&WHERE_SELECT_SCALAR, spec)?;
    let ps = Params::new()
        .uz(n)?
        .uz(0)?
        .uz(cond.broadcast.inner_or_one())?
        .uz(a.broadcast.inner_or_one())?
        .uz(b.broadcast.inner_or_one())?;
    ctx.dispatch(&ks, &[cond.buf, a.buf, b.buf, out], &ps, ctx.grid_1d(n as u64, 256)?)
}
