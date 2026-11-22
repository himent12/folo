//! Distillation kernels: the fused class and DFL terms of docs/MODEL.md §8.9 on raw buffers, forward (values,
//! reduced in a fixed order) and backward (the closed-form gradients). [`crate::distill`] wraps them for folo-core
//! tensors (with the backward as a custom op).
//!
//! Buffers (all dense from byte 0, `B` images, `A` anchors, `NC` classes, `R = reg_max`):
//! - the student's `box_raw [B, 4R, A]` and `cls_raw [B, NC, A]`, f32;
//! - the teacher's `box_raw` and `cls_raw` of the same shapes, f32 or f16 ([`KdDesc::teacher_f16`]: two values per
//!   u32 word, decoded bit by bit (`f16_bits_to_f32`), so no `shader-f16` is needed and infinities survive on every
//!   backend);
//! - `part_cls` and `part_dfl`: the partial sums of each term ([`KdDesc::cls_partials`], [`KdDesc::dfl_partials`]);
//!   `misc [4]` u32 (bit 0 of word 0: an input that makes a term non-finite); `stats [4]` and `total [1]` f32 (`shaders/kd_combine.wgsl`).

use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

const COMMON: &str = include_str!("../shaders/detect_common.wgsl");
const KD_COMMON: &str = include_str!("../shaders/kd_common.wgsl");

/// The teacher's element type for the generator: `TW` and the shared helpers.
fn prelude(spec: &Spec) -> String {
    let tw = if spec.get_bool("TF16") { "u32" } else { "f32" };
    // `detect_common.wgsl` needs CIoU's `V_K` (unused here), declared as the detection kernels declare it.
    let vk = super::detect::v_k();
    // Supply defaults for optional specialisation constants absent from older precompile lists or combine.
    let tf16 = if spec.get("TF16").is_none() { "const TF16: bool = false;\n" } else { "" };
    let has_gt = if spec.get("HAS_GT").is_none() { "const HAS_GT: bool = false;\n" } else { "" };
    let use_ver_w = if spec.get("USE_VER_W").is_none() { "const USE_VER_W: bool = false;\n" } else { "" };
    format!("alias TW = {tw};\n{tf16}{has_gt}{use_ver_w}const V_K: f32 = {vk:?}f;\n{COMMON}{KD_COMMON}")
}

macro_rules! kd_kernel {
    ($id:ident, $name:literal, $file:literal, [$($acc:ident),*]) => {
        static $id: KernelDef = KernelDef {
            name: $name,
            storage: &[$(Access::$acc),*],
            source: Source::Generated(|spec| {
                format!("{}{}", prelude(spec), include_str!(concat!("../shaders/", $file)))
            }),
        };
    };
}

// The `det_` prefix puts these kernels under the detection kernels' precompile bounds (`kernels/bounds.rs`: `R` at
// most `MAX_REG_MAX`).
kd_kernel!(CLS_FWD, "det_kd_cls_fwd", "kd_cls_fwd.wgsl", [Read, Read, ReadWrite, ReadWrite, Read]);
kd_kernel!(CLS_BWD, "det_kd_cls_bwd", "kd_cls_bwd.wgsl", [Read, Read, Read, ReadWrite, Read]);
kd_kernel!(DFL, "det_kd_dfl", "kd_dfl.wgsl", [Read, Read, Read, ReadWrite, ReadWrite, Read, Read]);
kd_kernel!(GT_VERIFY, "det_kd_gt_verify", "kd_gt_verify.wgsl", [Read, Read, Read, Read, Read, ReadWrite]);
kd_kernel!(COMBINE, "det_kd_combine", "kd_combine.wgsl", [Read, Read, Read, ReadWrite, ReadWrite]);

/// Anchors per class-term workgroup (4 per thread).
const CLS_CHUNK: usize = 1024;

/// Anchors per DFL workgroup (1 per thread).
const DFL_WG: u32 = 256;

/// Problem sizes and settings of one distillation call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdDesc {
    /// Images.
    pub b: usize,
    /// Anchors per image.
    pub a: usize,
    /// Classes.
    pub nc: usize,
    /// DFL bins per side (at most [`super::detect::MAX_REG_MAX`]).
    pub reg_max: usize,
    /// Whether the teacher's outputs are f16 (else f32).
    pub teacher_f16: bool,
    /// BCKD focal disagreement exponent (`0.0` = plain Bernoulli KL).
    pub bckd_gamma: f32,
}

impl KdDesc {
    /// Class partial sums: one per (image·class row, chunk of 1024 anchors), row-major.
    pub fn cls_partials(&self) -> usize {
        self.b * self.nc * self.a.div_ceil(CLS_CHUNK)
    }

    /// The DFL kernel's grid (one thread per (image, anchor), folded into x / y).
    fn dfl_grid(&self, ctx: &GpuContext) -> Result<[u32; 3]> {
        ctx.grid_1d((self.b * self.a) as u64, DFL_WG)
    }

    /// DFL partial sums: one per workgroup of the DFL grid.
    pub fn dfl_partials(&self, ctx: &GpuContext) -> Result<usize> {
        let g = self.dfl_grid(ctx)?;
        Ok(g[0] as usize * g[1] as usize)
    }

    /// Errors on sizes the kernels do not take.
    pub fn validate(&self) -> Result<()> {
        use super::detect::MAX_REG_MAX;
        if !(2..=MAX_REG_MAX).contains(&self.reg_max) {
            return Err(shape_err(format!("distill: reg_max {} must be in 2..={MAX_REG_MAX}", self.reg_max)));
        }
        if self.nc == 0 || self.a == 0 || self.b == 0 {
            return Err(shape_err("distill: empty batch, anchors or classes"));
        }
        let elems = |c: usize| self.b.checked_mul(c)?.checked_mul(self.a);
        let words = elems(4 * self.reg_max).zip(elems(self.nc)).map(|(x, y)| x.max(y));
        if words.is_none_or(|w| w >= u32::MAX as usize / 4) {
            return Err(shape_err("distill: problem too large for 32-bit indexing"));
        }
        Ok(())
    }

    fn spec(&self) -> Spec {
        Spec::new().bool("TF16", self.teacher_f16)
    }
}

/// The forward: the class and DFL partial sums into `part_cls` and `part_dfl`, the non-finite flag into `misc` (zeroed by the
/// caller), then the final reduction into `stats` and `total`. `temp` is `T`; `scales` are `cls_gain / N`,
/// `dfl_gain · T² / (4 N)` and `B`.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &GpuContext,
    d: &KdDesc,
    student: [&wgpu::Buffer; 2],
    teacher: [&wgpu::Buffer; 2],
    gt_dense: Option<&wgpu::Buffer>,
    ver_w: Option<&wgpu::Buffer>,
    temp: f32,
    scales: [f32; 3],
    part_cls: &wgpu::Buffer,
    part_dfl: &wgpu::Buffer,
    misc: &wgpu::Buffer,
    stats: &wgpu::Buffer,
    total: &wgpu::Buffer,
) -> Result<()> {
    d.validate()?;
    let [s_box, s_cls] = student;
    let [t_box, t_cls] = teacher;
    let dummy = ctx.dummy().clone();
    let chunks = d.a.div_ceil(CLS_CHUNK);
    let has_gt = gt_dense.is_some();
    let k = ctx.kernel(&CLS_FWD, d.spec().bool("HAS_GT", has_gt))?;
    let groups = d.cls_partials();
    let p = Params::new().uz(d.a)?.uz(chunks)?.uz(groups)?.uz(d.nc)?.uz(d.b * d.a)?.f(d.bckd_gamma);
    // One workgroup per partial, folded into x / y like a 1-D grid of `groups` workgroups.
    let grid = ctx.grid_1d(groups as u64 * 256, 256)?;
    ctx.dispatch(&k, &[s_cls, t_cls, part_cls, misc, gt_dense.unwrap_or(&dummy)], &p, grid)?;
    let use_ver_w = ver_w.is_some();
    let k =
        ctx.kernel(&DFL, d.spec().u32("R", d.reg_max as u32).u32("MODE", 0).bool("USE_VER_W", use_ver_w))?;
    let p = Params::new().uz(d.a)?.uz(d.nc)?.uz(d.b * d.a)?.f(temp).f(0.0);
    ctx.dispatch(
        &k,
        &[s_box, t_box, t_cls, part_dfl, misc, &dummy, ver_w.unwrap_or(&dummy)],
        &p,
        d.dfl_grid(ctx)?,
    )?;
    let k = ctx.kernel(&COMBINE, Spec::new())?;
    let p =
        Params::new().uz(d.cls_partials())?.uz(d.dfl_partials(ctx)?)?.f(scales[0]).f(scales[1]).f(scales[2]);
    ctx.dispatch(&k, &[part_cls, part_dfl, misc, stats, total], &p, [1, 1, 1])
}

/// Positive-only GT-verified DFL KD weights (`ver_w [B·A]`, zeroed by caller).
#[allow(clippy::too_many_arguments)]
pub fn gt_verify_weights(
    ctx: &GpuContext,
    asg_desc: &super::detect::DetectDesc,
    teacher_f16: bool,
    s_box: &wgpu::Buffer,
    t_box: &wgpu::Buffer,
    meta: &wgpu::Buffer,
    pick: &wgpu::Buffer,
    slot: &wgpu::Buffer,
    ver_w: &wgpu::Buffer,
) -> Result<()> {
    if asg_desc.slots() == 0 {
        return Ok(());
    }
    let spec =
        Spec::new().bool("TF16", teacher_f16).u32("R", asg_desc.reg_max as u32).u32("K", asg_desc.k as u32);
    let k = ctx.kernel(&GT_VERIFY, spec)?;
    let p =
        Params::new().uz(asg_desc.slots())?.uz(asg_desc.a)?.uz(asg_desc.g)?.uz(asg_desc.gt_box_offset())?;
    ctx.dispatch(&k, &[s_box, t_box, meta, pick, slot, ver_w], &p, ctx.grid_1d(asg_desc.slots() as u64, 64)?)
}

/// The backward: the gradients with respect to the student's `box_raw` and `cls_raw` into `dbox` and `dcls`
/// (every element written), scaled by the upstream gradient `grad` (`[1]`). `scales` are `B · cls_gain / N` and
/// `B · dfl_gain · T / (4 N)`.
#[allow(clippy::too_many_arguments)]
pub fn backward(
    ctx: &GpuContext,
    d: &KdDesc,
    student: [&wgpu::Buffer; 2],
    teacher: [&wgpu::Buffer; 2],
    gt_dense: Option<&wgpu::Buffer>,
    ver_w: Option<&wgpu::Buffer>,
    temp: f32,
    scales: [f32; 2],
    grad: &wgpu::Buffer,
    dbox: &wgpu::Buffer,
    dcls: &wgpu::Buffer,
) -> Result<()> {
    d.validate()?;
    let [s_box, s_cls] = student;
    let [t_box, t_cls] = teacher;
    let dummy = ctx.dummy().clone();
    let n = d.b * d.nc * d.a;
    let has_gt = gt_dense.is_some();
    let k = ctx.kernel(&CLS_BWD, d.spec().bool("HAS_GT", has_gt))?;
    let p = Params::new().uz(n)?.f(scales[0]).uz(d.a)?.uz(d.nc)?.uz(d.b * d.a)?.f(d.bckd_gamma);
    ctx.dispatch(
        &k,
        &[s_cls, t_cls, grad, dcls, gt_dense.unwrap_or(&dummy)],
        &p,
        ctx.grid_1d(n as u64, 256)?,
    )?;
    let use_ver_w = ver_w.is_some();
    let k =
        ctx.kernel(&DFL, d.spec().u32("R", d.reg_max as u32).u32("MODE", 1).bool("USE_VER_W", use_ver_w))?;
    let p = Params::new().uz(d.a)?.uz(d.nc)?.uz(d.b * d.a)?.f(temp).f(scales[1]);
    ctx.dispatch(&k, &[s_box, t_box, t_cls, dbox, &dummy, grad, ver_w.unwrap_or(grad)], &p, d.dfl_grid(ctx)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&CLS_FWD, &CLS_BWD, &DFL, &GT_VERIFY, &COMBINE];
