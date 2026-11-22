//! `LazyAlignHead`: FOLO v2 decoupled detection head.
//!
//! - **DW-free dense `ClsBranchV2`**: `[Conv1×1 align] + WS-QARepConv3x3(c_cls → c_cls) + Conv1×1(nc)`.
//! - **Aligned compact `BoxBranchV2`**: `2 × WS-QARepConv3x3(c_box) + Conv1×1(4·reg_max = 64)` (`% 16 == 0`),
//!   plus an opt-in **separate `iou_out: Conv1×1(c_box → 1)`** localization-quality head (runs lazily on
//!   surviving candidates only, keeping `box.out` strictly 64-channel aligned).
//! - **`PeakSuppress` `o2o` spatial competition gate**: exact local `3×3` peak preservation with content-
//!   adaptive lateral inhibition (`Z_o2o <= Z`), sharing `BoxBranchV2` between `o2m` and `o2o`.

use folo_core::{Tensor, is_grad_enabled, with_no_grad};
use folo_nn::{Conv2d, Conv2dConfig, ConvBnAct, Forward, Module, Result, RunningStats, TensorKind, init};
use rayon::prelude::*;

use crate::model::blocks::{Fuse, Init};
use crate::model::head::LEVELS;

use super::blocks::WsQARepConv3x3;
use super::config::MAL_CLS_BIAS_PRIOR;

/// `ln(e^1 - 1) = 0.54132485` so `softplus(ALPHA_INIT) == 1.0`.
const PEAK_SUPPRESS_ALPHA_INIT: f32 = 0.541_324_85;

/// Compact aligned box regression branch with opt-in separate `iou_out` head.
///
/// Tensor names: `conv0.*`, `conv1.*`, `out.{weight, bias}`, optional `iou_out.{weight, bias}`.
pub struct BoxBranchV2 {
    /// `WS-QARepConv3x3(C_L → c_box)`.
    pub conv0: WsQARepConv3x3,
    /// `WS-QARepConv3x3(c_box → c_box)`.
    pub conv1: WsQARepConv3x3,
    /// `Conv2d(c_box → 4·reg_max, 1×1)` with bias (strictly `64` channels for `reg_max = 16`).
    pub out: Conv2d,
    /// Opt-in separate localization-quality projection `Conv2d(c_box → 1, 1×1)` with bias.
    pub iou_out: Option<Conv2d>,
}

impl BoxBranchV2 {
    /// Build a v2 box branch for level input width `c_in`.
    pub fn new(c_in: usize, c_box: usize, reg_max: usize, iou_head: bool, init_ctx: &mut Init) -> Self {
        let conv0 = WsQARepConv3x3::new(c_in, c_box, init_ctx);
        let conv1 = WsQARepConv3x3::new(c_box, c_box, init_ctx);
        let out = Conv2d::new(Conv2dConfig::new(c_box, 4 * reg_max, 1), init_ctx.rng);
        let iou_out = iou_head.then(|| Conv2d::new(Conv2dConfig::new(c_box, 1, 1), init_ctx.rng));
        BoxBranchV2 { conv0, conv1, out, iou_out }
    }

    /// Forward returning `(box_logits [B, 4·reg_max, H, W], iou_logit Option<[B, 1, H, W]>)`.
    pub fn forward_box_iou(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let feat = self.conv1.forward(&self.conv0.forward(x)?)?;
        let box_raw = self.out.forward(&feat)?;
        let iou_raw = self.iou_out.as_ref().map(|m| m.forward(&feat)).transpose()?;
        Ok((box_raw, iou_raw))
    }

    /// Forward returning only `box_logits [B, 4·reg_max, H, W]` without computing `iou_out`.
    pub fn forward_box_only(&self, x: &Tensor) -> Result<Tensor> {
        let feat = self.conv1.forward(&self.conv0.forward(x)?)?;
        self.out.forward(&feat)
    }
}

impl Module for BoxBranchV2 {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("conv0", &self.conv0);
        f("conv1", &self.conv1);
        f("out", &self.out);
        if let Some(iou) = &self.iou_out {
            f("iou_out", iou);
        }
    }
}

impl Forward for BoxBranchV2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_box_only(x)
    }
}

impl Fuse for BoxBranchV2 {
    fn fuse_(&mut self) -> Result<()> {
        self.conv0.fuse_()?;
        self.conv1.fuse_()
    }
}

/// Depthwise-free dense classification branch (`[1×1 align] + WS-QARepConv3x3 + 1×1 out`).
///
/// Tensor names: optional `align.*`, `conv0.*`, `out.{weight, bias}`.
pub struct ClsBranchV2 {
    /// Optional `Conv(C_L → c_cls, 1×1)` (omitted when `C_L == c_cls`, e.g. P3 on `folo2-n`).
    pub align: Option<ConvBnAct>,
    /// Dense `WS-QARepConv3x3(c_cls → c_cls)`.
    pub conv0: WsQARepConv3x3,
    /// `Conv2d(c_cls → nc, 1×1)` with bias.
    pub out: Conv2d,
}

impl ClsBranchV2 {
    /// Build a DW-free v2 classification branch for level input width `c_in`.
    pub fn new(c_in: usize, c_cls: usize, num_classes: usize, init_ctx: &mut Init) -> Self {
        let align = (c_in != c_cls).then(|| init_ctx.conv(c_in, c_cls, 1, 1));
        let conv0 = WsQARepConv3x3::new(c_cls, c_cls, init_ctx);
        let out = Conv2d::new(Conv2dConfig::new(c_cls, num_classes, 1), init_ctx.rng);
        ClsBranchV2 { align, conv0, out }
    }

    /// Forward returning `(cls_logits [B, nc, H, W], pre_out_feat [B, c_cls, H, W])`.
    pub fn forward_with_feat(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = match &self.align {
            Some(a) => a.forward(x)?,
            None => x.clone(),
        };
        let feat = self.conv0.forward(&h)?;
        let logits = self.out.forward(&feat)?;
        Ok((logits, feat))
    }
}

impl Module for ClsBranchV2 {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        if let Some(a) = &self.align {
            f("align", a);
        }
        f("conv0", &self.conv0);
        f("out", &self.out);
    }
}

impl Forward for ClsBranchV2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_feat(x)?.0)
    }
}

impl Fuse for ClsBranchV2 {
    fn fuse_(&mut self) -> Result<()> {
        if let Some(a) = &mut self.align {
            a.fuse_()?;
        }
        self.conv0.fuse_()
    }
}

/// Content-adaptive lateral competition gate for NMS-free `o2o` classification logits:
///
/// ```text
/// g = sigmoid(Conv1x1(feat_cls, c_cls -> 1))
/// Z_o2o = Z - softplus(alpha_c) * g * (MaxPool3x3_s1(Z) - Z)
/// ```
///
/// Local `3×3` maxima satisfy `MaxPool3x3(Z) - Z == 0`, preserving true peaks identically while
/// suppressing non-peak duplicates (`Z_o2o <= Z` everywhere).
pub struct PeakSuppress {
    /// Spatial gate projection `Conv2d(c_cls → 1, 1×1)`.
    pub gate: Conv2d,
    /// Per-class unconstrained suppression strength `[nc]` (mapped through `softplus`).
    pub alpha_raw: Tensor,
    /// Number of classes.
    pub num_classes: usize,
}

impl PeakSuppress {
    /// Build a `PeakSuppress` gate for `c_cls` feature channels and `num_classes` classes.
    pub fn new(c_cls: usize, num_classes: usize, init_ctx: &mut Init) -> Self {
        let gate = Conv2d::new(Conv2dConfig::new(c_cls, 1, 1), init_ctx.rng);
        let alpha_raw = Tensor::full([num_classes], PEAK_SUPPRESS_ALPHA_INIT).requires_grad_();
        PeakSuppress { gate, alpha_raw, num_classes }
    }

    /// Apply content-adaptive peak-preserving suppression to `logits [B, nc, H, W]` using `feat [B, c_cls, H, W]`.
    pub fn apply(&self, logits: &Tensor, feat: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(feat)?.sigmoid()?;
        let g = if g.dtype() != logits.dtype() && folo_core::autocast_dtype().is_some() {
            g.to_dtype(logits.dtype())?
        } else {
            g
        };
        let alpha =
            self.alpha_raw.softplus()?.to_dtype(logits.dtype())?.reshape([1, self.num_classes, 1, 1])?;
        let z_max = logits.max_pool2d(3, 1, 1)?;
        let penalty = z_max.sub(logits)?.mul(&g)?.mul(&alpha)?;
        Ok(logits.sub(&penalty)?)
    }
}

impl Module for PeakSuppress {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("alpha_raw", &self.alpha_raw, TensorKind::Parameter);
    }

    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("gate", &self.gate);
    }
}

/// One level of the one-to-one (`o2o`) classification branch with [`PeakSuppress`].
/// Box regression is shared with [`HeadLevelV2::box_branch`] (cutting `o2o` head VRAM).
pub struct O2oClsLevel {
    /// DW-free classification branch producing pre-suppression `o2o` logits and features.
    pub cls_branch: ClsBranchV2,
    /// Content-adaptive `3×3` peak competition gate.
    pub suppress: PeakSuppress,
}

impl O2oClsLevel {
    fn new(c_in: usize, c_cls: usize, num_classes: usize, init_ctx: &mut Init) -> Self {
        O2oClsLevel {
            cls_branch: ClsBranchV2::new(c_in, c_cls, num_classes, init_ctx),
            suppress: PeakSuppress::new(c_cls, num_classes, init_ctx),
        }
    }

    fn forward_o2o_cls(&self, x: &Tensor) -> Result<Tensor> {
        let (logits, feat) = self.cls_branch.forward_with_feat(x)?;
        self.suppress.apply(&logits, &feat)
    }

    fn fuse_(&mut self) -> Result<()> {
        self.cls_branch.fuse_()
    }
}

impl Module for O2oClsLevel {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cls", &self.cls_branch);
        f("suppress", &self.suppress);
    }
}

/// One level of [`LazyAlignHead`].
pub struct HeadLevelV2 {
    /// Compact aligned box branch (`head.box.L.*`).
    pub box_branch: BoxBranchV2,
    /// DW-free dense classification branch (`head.cls.L.*`).
    pub cls_branch: ClsBranchV2,
    /// Spatial stride (`8`, `16`, or `32`).
    pub stride: u32,
}

impl HeadLevelV2 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        c_in: usize,
        stride: u32,
        c_box: usize,
        c_cls: usize,
        nc: usize,
        reg_max: usize,
        iou_head: bool,
        init_ctx: &mut Init,
    ) -> Self {
        HeadLevelV2 {
            box_branch: BoxBranchV2::new(c_in, c_box, reg_max, iou_head, init_ctx),
            cls_branch: ClsBranchV2::new(c_in, c_cls, nc, init_ctx),
            stride,
        }
    }

    fn fuse_(&mut self) -> Result<()> {
        self.box_branch.fuse_()?;
        self.cls_branch.fuse_()
    }
}

/// FOLO v2 decoupled detection head (`LazyAlignHead`).
pub struct LazyAlignHead {
    /// Per-level `o2m` box and class branches (`p3`, `p4`, `p5`).
    pub levels: Vec<HeadLevelV2>,
    /// Optional `o2o` classification branches with [`PeakSuppress`] (sharing `box_branch` with `o2m`).
    pub o2o_cls: Option<Vec<O2oClsLevel>>,
    /// Box branch hidden width (`48` on `n`/`s`).
    pub c_box: usize,
    /// Class branch hidden width (`64` on `n`/`s`).
    pub c_cls: usize,
    /// DFL bins per side (`16`).
    pub reg_max: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Whether the opt-in separate `iou_out` branch is present.
    pub has_iou_head: bool,
    /// Whether `o2o` detaches its feature inputs during training (default `true`; progressive un-detach sets `false`).
    pub o2o_detach: bool,
    /// Whether `cls.out.bias` uses the MAL prior (`-4.595`) instead of the plain-BCE prior.
    pub mal_bias_prior: bool,
}

impl LazyAlignHead {
    /// Construct a `LazyAlignHead` for P3/P4/P5 input channels `in_channels`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: [usize; 3],
        strides: [u32; 3],
        c_box: usize,
        c_cls: usize,
        num_classes: usize,
        reg_max: usize,
        o2o: bool,
        iou_head: bool,
        mal_bias_prior: bool,
        init_ctx: &mut Init,
    ) -> Result<Self> {
        let levels = in_channels
            .iter()
            .zip(&strides)
            .map(|(&c, &s)| HeadLevelV2::new(c, s, c_box, c_cls, num_classes, reg_max, iou_head, init_ctx))
            .collect();
        let o2o_cls = o2o.then(|| {
            in_channels.iter().map(|&c| O2oClsLevel::new(c, c_cls, num_classes, init_ctx)).collect()
        });
        let head = LazyAlignHead {
            levels,
            o2o_cls,
            c_box,
            c_cls,
            reg_max,
            num_classes,
            has_iou_head: iou_head,
            o2o_detach: true,
            mal_bias_prior,
        };
        head.reset_priors()?;
        Ok(head)
    }

    /// Write output bias priors: `box.out.bias = 2.0`, `iou_out.bias = 0.0`, and either
    /// [`MAL_CLS_BIAS_PRIOR`] (when `mal_bias_prior`) or FOLO v1's per-level `cls_bias_prior`.
    pub fn reset_priors(&self) -> Result<()> {
        for l in &self.levels {
            if let Some(b) = &l.box_branch.out.bias {
                init::constant_(b, init::BOX_BIAS_PRIOR)?;
            }
            if let Some(iou) = &l.box_branch.iou_out
                && let Some(b) = &iou.bias
            {
                init::constant_(b, 0.0)?;
            }
            if let Some(b) = &l.cls_branch.out.bias {
                let prior = if self.mal_bias_prior {
                    MAL_CLS_BIAS_PRIOR
                } else {
                    init::cls_bias_prior(self.num_classes, l.stride)
                };
                init::constant_(b, prior)?;
            }
        }
        if let Some(o2o) = &self.o2o_cls {
            for (ol, l) in o2o.iter().zip(&self.levels) {
                if let Some(b) = &ol.cls_branch.out.bias {
                    let prior = if self.mal_bias_prior {
                        MAL_CLS_BIAS_PRIOR
                    } else {
                        init::cls_bias_prior(self.num_classes, l.stride)
                    };
                    init::constant_(b, prior)?;
                }
            }
        }
        Ok(())
    }

    /// One-to-many raw outputs: `(box_raw [B, 4·reg_max, A], cls_raw [B, nc, A], iou_raw Option<[B, 1, A]>)`.
    pub fn forward(&self, feats: [&Tensor; 3]) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let run_box = |i: usize| -> Result<(Tensor, Option<Tensor>)> {
            let (l, x) = (&self.levels[i], feats[i]);
            let (b, _, h, w) = x.dims4()?;
            let (bx, iou) = l.box_branch.forward_box_iou(x)?;
            let bx = bx.reshape([b, 4 * self.reg_max, h * w])?;
            let iou = iou.map(|t| t.reshape([b, 1, h * w])).transpose()?;
            Ok((bx, iou))
        };
        let run_cls = |i: usize| -> Result<Tensor> {
            let (l, x) = (&self.levels[i], feats[i]);
            let (b, _, h, w) = x.dims4()?;
            Ok(l.cls_branch.forward(x)?.reshape([b, self.num_classes, h * w])?)
        };
        let cpu_eval = feats.iter().all(|x| x.device().is_cpu()) && !is_grad_enabled();
        let (box_outs, cls_outs) = if cpu_eval {
            rayon::join(
                || (0..3).into_par_iter().map(|i| with_no_grad(|| run_box(i))).collect::<Result<Vec<_>>>(),
                || (0..3).into_par_iter().map(|i| with_no_grad(|| run_cls(i))).collect::<Result<Vec<_>>>(),
            )
        } else {
            ((0..3).map(run_box).collect::<Result<Vec<_>>>(), (0..3).map(run_cls).collect::<Result<Vec<_>>>())
        };
        let box_outs = box_outs?;
        let cls_outs = cls_outs?;
        let boxes: Vec<Tensor> = box_outs.iter().map(|(b, _)| b.clone()).collect();
        let box_raw = Tensor::cat(&boxes, 2)?;
        let cls_raw = Tensor::cat(&cls_outs, 2)?;
        let iou_raw = if self.has_iou_head {
            let ious: Vec<Tensor> = box_outs.into_iter().map(|(_, t)| t.expect("iou_out")).collect();
            Some(Tensor::cat(&ious, 2)?)
        } else {
            None
        };
        Ok((box_raw, cls_raw, iou_raw))
    }

    /// One-to-one raw outputs `(box_raw, cls_raw_o2o, iou_raw)` sharing `box_branch`.
    ///
    /// When `o2o_detach` is true and autograd is enabled, gradients stop strictly at the backbone/neck feature
    /// inputs `feats` while `box_branch` weights still receive `o2o` box+DFL gradients (running `box_branch` on
    /// the detached features). When `o2o_detach` is false or autograd is disabled, `shared_box` is reused directly
    /// with zero redundant compute.
    pub fn forward_o2o(
        &self,
        feats: [&Tensor; 3],
        shared_box: Option<&Tensor>,
    ) -> Result<Option<(Tensor, Tensor, Option<Tensor>)>> {
        let Some(o2o_levels) = &self.o2o_cls else { return Ok(None) };
        let detach_feats = self.o2o_detach && is_grad_enabled();
        let detached;
        let cls_feats = if detach_feats {
            detached = feats.map(Tensor::detach);
            [&detached[0], &detached[1], &detached[2]]
        } else {
            feats
        };
        let cls_parts: Vec<Tensor> = (0..3)
            .map(|i| {
                let x = cls_feats[i];
                let (b, _, h, w) = x.dims4()?;
                Ok(o2o_levels[i].forward_o2o_cls(x)?.reshape([b, self.num_classes, h * w])?)
            })
            .collect::<Result<_>>()?;
        let cls_raw = Tensor::cat(&cls_parts, 2)?;
        let (box_raw, iou_raw) = match (shared_box, detach_feats) {
            (Some(b), false) => (b.clone(), None),
            (Some(_), true) => {
                // Recompute on detached `feats` so `o2o` box+DFL losses train `box_branch` weights without
                // pushing gradients into the backbone/neck, while keeping training-mode `box_branch` BN running
                // stats updated only once per batch (from the `o2m` pass) and skipping `iou_out`.
                // Only snapshot/restore BNs that are actively in training mode: eval-mode (frozen) BNs read
                // `running_mean`/`running_var` inside the autograd graph, so touching them would trigger
                // `ModifiedInPlace` during `backward()`.
                let mut boxes = Vec::with_capacity(3);
                for i in 0..3 {
                    let x = cls_feats[i];
                    let (b, _, h, w) = x.dims4()?;
                    let bb = &self.levels[i].box_branch;
                    let snaps: Vec<RunningStats> = [&bb.conv0.bn, &bb.conv1.bn]
                        .into_iter()
                        .flatten()
                        .filter(|bn| bn.is_training())
                        .map(RunningStats::capture)
                        .collect::<Result<_>>()?;
                    let res = bb.forward_box_only(x);
                    for snap in snaps {
                        snap.restore()?;
                    }
                    let bx = res?;
                    boxes.push(bx.reshape([b, 4 * self.reg_max, h * w])?);
                }
                (Tensor::cat(&boxes, 2)?, None)
            }
            (None, _) => {
                let mut boxes = Vec::with_capacity(3);
                let mut ious = Vec::with_capacity(3);
                for i in 0..3 {
                    let x = cls_feats[i];
                    let (b, _, h, w) = x.dims4()?;
                    let (bx, iou) = self.levels[i].box_branch.forward_box_iou(x)?;
                    boxes.push(bx.reshape([b, 4 * self.reg_max, h * w])?);
                    if let Some(t) = iou {
                        ious.push(t.reshape([b, 1, h * w])?);
                    }
                }
                let iou_raw = if self.has_iou_head { Some(Tensor::cat(&ious, 2)?) } else { None };
                (Tensor::cat(&boxes, 2)?, iou_raw)
            }
        };
        Ok(Some((box_raw, cls_raw, iou_raw)))
    }
}

impl Module for LazyAlignHead {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        for (l, name) in self.levels.iter().zip(LEVELS) {
            f(&format!("box.{name}"), &l.box_branch);
            f(&format!("cls.{name}"), &l.cls_branch);
        }
        if let Some(o2o) = &self.o2o_cls {
            for (ol, name) in o2o.iter().zip(LEVELS) {
                f(&format!("o2o.{name}"), ol);
            }
        }
    }
}

impl Fuse for LazyAlignHead {
    fn fuse_(&mut self) -> Result<()> {
        for l in &mut self.levels {
            l.fuse_()?;
        }
        if let Some(o2o) = &mut self.o2o_cls {
            for ol in o2o {
                ol.fuse_()?;
            }
        }
        Ok(())
    }
}
