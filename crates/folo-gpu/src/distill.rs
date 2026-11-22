//! The distillation term of docs/MODEL.md §8.9 (class KL and DFL KL against a frozen teacher) on folo-core GPU
//! tensors, fused: two kernels read the student's and the teacher's raw head outputs (the teacher's class head
//! twice: for the class term and for the DFL weight `q`) and reduce the values in a fixed order, a third combines
//! the partial sums, and the backward is two kernels that write the closed-form gradients. No probability tensor of
//! the teacher is formed and nothing waits for the GPU: the caller reads `stats` once.
//!
//! The math is folo-vision's `distill_loss` (`crates/folo-vision/src/loss/distill/`), element by element, with the
//! transcendental functions of `shaders/detect_common.wgsl`; the teacher's entropy and log-probabilities are taken
//! from its logits (softplus, log-sum-exp) rather than as logarithms of rounded probabilities, so values agree with
//! `distill_loss` to float rounding, not bit for bit.

use std::sync::Arc;

use folo_core::{DType, Device, Error, Result, Tensor};

use crate::backend::{context_for, dense_buffer};
use crate::context::GpuContext;
use crate::detect::{Buffers, dense, gpu};
use crate::kernels::distill::{self as k, KdDesc};

/// Settings of [`distill_loss`] (folo-vision's `DistillConfig` and the supervised normaliser).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdSettings {
    /// DFL bins per side.
    pub reg_max: usize,
    /// `λc`, the class term's weight.
    pub cls_gain: f32,
    /// `λd`, the DFL term's weight.
    pub dfl_gain: f32,
    /// The DFL temperature `T` (> 0).
    pub temperature: f32,
    /// `N`, the supervised one-to-many loss's normaliser (`max(target score sum, 1)`).
    pub normaliser: f32,
    /// BCKD focal disagreement exponent (`0.0` = plain Bernoulli KL).
    pub bckd_gamma: f32,
}

/// What [`distill_loss`] returns.
pub struct DeviceKd {
    /// `B · (λc · L_kc + λd · L_kd)`, rank 0, differentiable with respect to the student's `box_raw` and `cls_raw`.
    /// NaN for the inputs that make the tensor path's term non-finite (docs/MODEL.md §8.9: a student value, or a
    /// student box value over `T`, that is not finite, a NaN teacher value, a teacher box value over `T` of +inf, a
    /// side whose teacher box values over `T` are all −inf); a teacher logit of ±inf otherwise gives the limit.
    pub total: Tensor,
    /// `[4]` f32: `λc · L_kc`, `λd · L_kd` (without `B`), the non-finite flag (1 or 0), 0. The backward does not
    /// read it.
    pub stats: Tensor,
}

fn err(msg: impl Into<String>) -> Error {
    Error::InvalidArgument { op: "folo_gpu::distill", msg: msg.into() }
}

fn check(student: [&Tensor; 2], teacher: [&Tensor; 2], s: &KdSettings) -> Result<KdDesc> {
    let [s_box, s_cls] = student;
    let [t_box, t_cls] = teacher;
    for t in student {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch {
                op: "folo_gpu::distill",
                expected: DType::F32,
                got: t.dtype(),
            });
        }
    }
    let tf = t_cls.dtype();
    if !matches!(tf, DType::F32 | DType::F16) || t_box.dtype() != tf {
        return Err(err(format!(
            "teacher outputs must both be f32 or f16, got {} and {}",
            t_box.dtype(),
            tf
        )));
    }
    let device = s_cls.device();
    for t in [s_box, t_box, t_cls] {
        if t.device() != device {
            return Err(Error::DeviceMismatch { op: "folo_gpu::distill", lhs: device, rhs: t.device() });
        }
    }
    let ((b, ch, a), (b2, nc, a2)) = match (s_box.dims3(), s_cls.dims3()) {
        (Ok(x), Ok(y)) => (x, y),
        _ => return Err(err("box_raw and cls_raw must be rank 3")),
    };
    if s.reg_max.checked_mul(4) != Some(ch)
        || (b2, a2) != (b, a)
        || t_box.dims() != s_box.dims()
        || t_cls.dims() != s_cls.dims()
    {
        return Err(err(format!(
            "student {:?} / {:?} and teacher {:?} / {:?} must be [B, {}, A] and [B, nc, A]",
            s_box.dims(),
            s_cls.dims(),
            t_box.dims(),
            t_cls.dims(),
            s.reg_max.saturating_mul(4)
        )));
    }
    if !(s.temperature.is_finite() && s.temperature > 0.0 && s.normaliser >= 1.0) {
        return Err(err(format!("temperature {} must be > 0 and N {} >= 1", s.temperature, s.normaliser)));
    }
    let d = KdDesc { b, a, nc, reg_max: s.reg_max, teacher_f16: tf == DType::F16, bckd_gamma: s.bckd_gamma };
    d.validate().map_err(|e| err(e.to_string()))?;
    Ok(d)
}

/// Whether [`distill_loss`] takes a student head of `b` images, `nc` classes, `a` anchors and `reg_max` bins per
/// side (on a folo-gpu device).
pub fn supports(b: usize, nc: usize, a: usize, reg_max: usize) -> bool {
    KdDesc { b, a, nc, reg_max, teacher_f16: false, bckd_gamma: 0.0 }.validate().is_ok()
}

/// Everything the backward reads, kept alive by the custom op.
struct Saved {
    ctx: GpuContext,
    device: Device,
    desc: KdDesc,
    inputs: [Tensor; 4],
    gt_dense: Option<Tensor>,
    ver_w: Option<Tensor>,
    temp: f32,
    scales: [f32; 2],
}

impl Saved {
    fn backward(&self, g: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let d = &self.desc;
        if g.device() != self.device {
            return Err(Error::DeviceMismatch { op: "folo_gpu::distill", lhs: self.device, rhs: g.device() });
        }
        // The kernels read the upstream gradient as f32 (a seed may be f16).
        let g = if g.dtype() == DType::F32 { g.clone() } else { g.to_dtype(DType::F32)? };
        let (_g, gbuf) = dense(&g)?;
        let buf = |t: &Tensor| dense_buffer(t).ok_or_else(|| err("saved tensor is not dense"));
        let [s_box, s_cls, t_box, t_cls] = &self.inputs;
        let bufs = Buffers { ctx: self.ctx.clone(), device: self.device };
        let (dbox, dbox_b) = bufs.alloc(DType::F32, &[d.b, 4 * d.reg_max, d.a], false)?;
        let (dcls, dcls_b) = bufs.alloc(DType::F32, &[d.b, d.nc, d.a], false)?;
        let gt_b = self.gt_dense.as_ref().map(&buf).transpose()?;
        let vw_b = self.ver_w.as_ref().map(&buf).transpose()?;
        k::backward(
            &self.ctx,
            d,
            [&buf(s_box)?, &buf(s_cls)?],
            [&buf(t_box)?, &buf(t_cls)?],
            gt_b.as_ref(),
            vw_b.as_ref(),
            self.temp,
            self.scales,
            &gbuf,
            &dbox_b,
            &dcls_b,
        )
        .map_err(gpu)?;
        // No gradient for the teacher's outputs (inputs only so that autograd checks their storage versions).
        Ok(vec![Some(dbox), Some(dcls), None, None])
    }
}

/// The backward's scales `B · λc / N` and `B · λd · T / (4 N)`, rounded once from f64.
fn backward_scales(s: &KdSettings, b: usize) -> [f32; 2] {
    let (n, t, b) = (f64::from(s.normaliser), f64::from(s.temperature), b as f64);
    [(b * f64::from(s.cls_gain) / n) as f32, (b * f64::from(s.dfl_gain) * t / (4.0 * n)) as f32]
}

/// The class and DFL distillation terms of docs/MODEL.md §8.9 for the student's raw one-to-many outputs `box_raw:
/// [B, 4 · reg_max, A]` and `cls_raw: [B, nc, A]` (f32) and the teacher's raw outputs of the same shapes (both f32
/// or both f16), all on one folo-gpu device. See [`DeviceKd`]; read `stats` for the values.
///
/// Gradients: `dL/dx = B · λc · (sigmoid(x) − p~) / N` and `dL/dz = B · λd · T · q_a · (softmax(z / T) − s~) /
/// (4 N)`, times the upstream gradient (any float dtype). None for the teacher's outputs; a backward after one of
/// the four inputs was modified in place is an error.
pub fn distill_loss(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    teacher_box: &Tensor,
    teacher_cls: &Tensor,
    s: &KdSettings,
) -> Result<DeviceKd> {
    distill_loss_with_context(box_raw, cls_raw, teacher_box, teacher_cls, s, None, false)
}

/// [`distill_loss`] with optional supervised [`crate::detect::Assignment`] context for conflict-free BCKD
/// (`s.bckd_gamma > 0` clamps positive teacher targets to `max(p_t, t_gt)`) and positive-only GT-verified DFL KD
/// (`gt_verify = true`).
pub fn distill_loss_with_context(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    teacher_box: &Tensor,
    teacher_cls: &Tensor,
    s: &KdSettings,
    gt_asg: Option<&crate::detect::Assignment>,
    gt_verify: bool,
) -> Result<DeviceKd> {
    let d = check([box_raw, cls_raw], [teacher_box, teacher_cls], s)?;
    let device = box_raw.device();
    let ctx = context_for(device).ok_or_else(|| err(format!("{device} is not a folo-gpu device")))?;
    let bufs = Buffers { ctx: ctx.clone(), device };
    let (sb, sb_b) = dense(box_raw)?;
    let (sc, sc_b) = dense(cls_raw)?;
    let (tb, tb_b) = dense(teacher_box)?;
    let (tc, tc_b) = dense(teacher_cls)?;
    let (gt_dense_t, gt_dense_b) = match gt_asg.filter(|_| s.bckd_gamma > 0.0) {
        Some(asg) => {
            let b = dense_buffer(&asg.dense).ok_or_else(|| err("assignment dense is not on GPU"))?;
            (Some(asg.dense.clone()), Some(b))
        }
        None => (None, None),
    };
    let (ver_w_t, ver_w_b) = match gt_asg.filter(|_| gt_verify) {
        Some(asg) => {
            let (vw_t, vw_b) = bufs.alloc(DType::F32, &[d.b, d.a], true)?;
            let meta_b = dense_buffer(&asg.meta).ok_or_else(|| err("assignment meta is not on GPU"))?;
            let pick_b = dense_buffer(&asg.picks).ok_or_else(|| err("assignment picks is not on GPU"))?;
            let slot_b = dense_buffer(&asg.slots).ok_or_else(|| err("assignment slots is not on GPU"))?;
            k::gt_verify_weights(
                &ctx,
                &asg.desc,
                d.teacher_f16,
                &sb_b,
                &tb_b,
                &meta_b,
                &pick_b,
                &slot_b,
                &vw_b,
            )
            .map_err(gpu)?;
            (Some(vw_t), Some(vw_b))
        }
        None => (None, None),
    };
    let (_pc, pc) = bufs.alloc(DType::F32, &[d.cls_partials()], false)?;
    let (_pd, pd) = bufs.alloc(DType::F32, &[d.dfl_partials(&ctx).map_err(gpu)?], false)?;
    let (_misc, misc) = bufs.alloc(DType::U32, &[4], true)?;
    let (stats, stats_b) = bufs.alloc(DType::F32, &[4], false)?;
    let (total, total_b) = bufs.alloc(DType::F32, &[], false)?;
    let (n, t, nb) = (s.normaliser, s.temperature, d.b as f32);
    let fwd_scales = [s.cls_gain / n, s.dfl_gain * t * t / (4.0 * n), nb];
    k::forward(
        &ctx,
        &d,
        [&sb_b, &sc_b],
        [&tb_b, &tc_b],
        gt_dense_b.as_ref(),
        ver_w_b.as_ref(),
        t,
        fwd_scales,
        &pc,
        &pd,
        &misc,
        &stats_b,
        &total_b,
    )
    .map_err(gpu)?;
    let saved = Arc::new(Saved {
        ctx: ctx.clone(),
        device,
        desc: d,
        inputs: [sb, sc, tb, tc],
        gt_dense: gt_dense_t,
        ver_w: ver_w_t,
        temp: t,
        // In f64, so `B · λ` cannot overflow before the division by `N` (the tensor path's backward divides first).
        scales: backward_scales(s, d.b),
    });
    // The teacher's outputs are inputs too, without gradient: the backward reads them, so autograd must refuse a
    // backward after they were overwritten in place (the saved dense handles may share their storage).
    let inputs = [box_raw, cls_raw, teacher_box, teacher_cls];
    let total = Tensor::custom_op("distill_loss_gpu", &inputs, total, move |g| saved.backward(g))?;
    Ok(DeviceKd { total, stats })
}
