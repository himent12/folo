//! The detect head of docs/MODEL.md §6: per level a box branch (DFL logits) and a class branch, flattened and
//! concatenated along the anchor axis.

use folo_core::{Tensor, is_grad_enabled, with_no_grad};
use folo_nn::{Activation, Conv2d, Conv2dConfig, ConvBnAct, Forward, Module, Result, init};

use rayon::prelude::*;

use super::blocks::{Fuse, Init};

/// The level names of the head, `p3 p4 p5` (strides 8, 16, 32).
pub const LEVELS: [&str; 3] = ["p3", "p4", "p5"];

/// `box_L` of MODEL.md §6.1: `Conv2d(c_box → 4·reg_max)(Conv(c_box→c_box, 3)(Conv(C_L→c_box, 3)(x)))`.
/// Tensor names: `conv0.*`, `conv1.*`, `out.{weight, bias}`.
pub struct BoxBranch {
    /// `Conv(C_L→c_box, 3, 1)`.
    pub conv0: ConvBnAct,
    /// `Conv(c_box→c_box, 3, 1)`.
    pub conv1: ConvBnAct,
    /// `Conv2d(c_box → 4·reg_max, 1×1)` with bias.
    pub out: Conv2d,
}

impl Module for BoxBranch {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("conv0", &self.conv0);
        f("conv1", &self.conv1);
        f("out", &self.out);
    }
}

impl Forward for BoxBranch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.conv1.forward(&self.conv0.forward(x)?)?)
    }
}

/// `cls_L` of MODEL.md §6.1: `DWConv(C_L, 3)`, `Conv(C_L→c_cls, 1)`, `DWConv(c_cls, 3)`, `Conv(c_cls→c_cls, 1)`,
/// `Conv2d(c_cls → nc, 1×1)`. Tensor names: `dw0.*`, `pw0.*`, `dw1.*`, `pw1.*`, `out.{weight, bias}`.
pub struct ClsBranch {
    /// `DWConv(C_L, 3)`.
    pub dw0: ConvBnAct,
    /// `Conv(C_L→c_cls, 1, 1)`.
    pub pw0: ConvBnAct,
    /// `DWConv(c_cls, 3)`.
    pub dw1: ConvBnAct,
    /// `Conv(c_cls→c_cls, 1, 1)`.
    pub pw1: ConvBnAct,
    /// `Conv2d(c_cls → nc, 1×1)` with bias: the class-output layer FORMAT.md's partial loading skips.
    pub out: Conv2d,
}

impl Module for ClsBranch {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("dw0", &self.dw0);
        f("pw0", &self.pw0);
        f("dw1", &self.dw1);
        f("pw1", &self.pw1);
        f("out", &self.out);
    }
}

impl Forward for ClsBranch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.pw0.forward(&self.dw0.forward(x)?)?;
        let y = self.pw1.forward(&self.dw1.forward(&y)?)?;
        self.out.forward(&y)
    }
}

/// One head level: its box and class branches.
pub struct HeadLevel {
    /// The box branch, `head.box.L.*`.
    pub box_branch: BoxBranch,
    /// The class branch, `head.cls.L.*`.
    pub cls_branch: ClsBranch,
    /// The level's stride.
    pub stride: u32,
}

/// The detect head (MODEL.md §6, `head = "o2m"`).
///
/// Canonical order (MODEL.md §5): for each level `p3, p4, p5` in turn, `box.conv0, box.conv1, box.out,
/// cls.dw0, cls.pw0, cls.dw1, cls.pw1, cls.out`, which is also the order the weights are drawn in.
pub struct Detect {
    /// The levels `p3, p4, p5`.
    pub levels: Vec<HeadLevel>,
    /// Box branch width, `max(16, C_P3 / 4, 4·reg_max)`.
    pub c_box: usize,
    /// Class branch width, `max(C_P3, min(nc, 100))`.
    pub c_cls: usize,
    /// DFL bins.
    pub reg_max: usize,
    /// Number of classes.
    pub num_classes: usize,
}

impl Detect {
    /// A head over level inputs of `in_channels` channels (P3, P4, P5) with `strides`, initialised per §2.9 and
    /// with the bias priors of §6.5 written over the output biases.
    pub fn new(
        in_channels: [usize; 3],
        strides: [u32; 3],
        num_classes: usize,
        reg_max: usize,
        init: &mut Init,
    ) -> Result<Self> {
        let c_p3 = in_channels[0];
        let c_box = 16.max(c_p3 / 4).max(4 * reg_max);
        let c_cls = c_p3.max(num_classes.min(100));
        let mut levels = Vec::with_capacity(3);
        for (&c, &stride) in in_channels.iter().zip(&strides) {
            let box_branch = BoxBranch {
                conv0: init.conv(c, c_box, 3, 1),
                conv1: init.conv(c_box, c_box, 3, 1),
                out: Conv2d::new(Conv2dConfig::new(c_box, 4 * reg_max, 1), init.rng),
            };
            let cls_branch = ClsBranch {
                dw0: init.conv_with(c, c, 3, 1, c, Activation::Silu),
                pw0: init.conv(c, c_cls, 1, 1),
                dw1: init.conv_with(c_cls, c_cls, 3, 1, c_cls, Activation::Silu),
                pw1: init.conv(c_cls, c_cls, 1, 1),
                out: Conv2d::new(Conv2dConfig::new(c_cls, num_classes, 1), init.rng),
            };
            levels.push(HeadLevel { box_branch, cls_branch, stride });
        }
        let head = Detect { levels, c_box, c_cls, reg_max, num_classes };
        head.reset_priors()?;
        Ok(head)
    }

    /// Write MODEL.md §6.5's bias priors: `head.box.L.out.bias = 2.0`, `head.cls.L.out.bias =
    /// ln(5 / nc / (640 / s_L)²)`.
    pub fn reset_priors(&self) -> Result<()> {
        for l in &self.levels {
            if let Some(b) = &l.box_branch.out.bias {
                init::constant_(b, init::BOX_BIAS_PRIOR)?;
            }
            if let Some(b) = &l.cls_branch.out.bias {
                init::constant_(b, init::cls_bias_prior(self.num_classes, l.stride))?;
            }
        }
        Ok(())
    }

    /// Raw outputs `(box_raw [B, 4·reg_max, A], cls_raw [B, nc, A])` (MODEL.md §6.3) from the level features.
    ///
    /// On the CPU with nothing recorded (inference), the six branch chains (box and class, three levels) run
    /// side by side on the rayon pool: the P4 / P5 convs are too small to occupy every thread on their own.
    /// Each chain is the same sequence of deterministic ops either way, so the result does not change.
    pub fn forward(&self, feats: [&Tensor; 3]) -> Result<(Tensor, Tensor)> {
        let branch = |j: usize| -> Result<Tensor> {
            let (l, x) = (&self.levels[j / 2], feats[j / 2]);
            let (b, _, h, w) = x.dims4()?;
            if j.is_multiple_of(2) {
                Ok(l.box_branch.forward(x)?.reshape([b, 4 * self.reg_max, h * w])?)
            } else {
                Ok(l.cls_branch.forward(x)?.reshape([b, self.num_classes, h * w])?)
            }
        };
        let n = 2 * self.levels.len().min(feats.len());
        let outs: Vec<Tensor> = if feats.iter().all(|x| x.device().is_cpu()) && !is_grad_enabled() {
            // Grad mode is per thread: the workers must not record either.
            (0..n).into_par_iter().map(|j| with_no_grad(|| branch(j))).collect::<Result<_>>()?
        } else {
            (0..n).map(branch).collect::<Result<_>>()?
        };
        let (boxes, classes): (Vec<_>, Vec<_>) =
            outs.into_iter().enumerate().partition(|(j, _)| j.is_multiple_of(2));
        let strip = |v: Vec<(usize, Tensor)>| v.into_iter().map(|(_, t)| t).collect::<Vec<_>>();
        Ok((Tensor::cat(&strip(boxes), 2)?, Tensor::cat(&strip(classes), 2)?))
    }
}

impl Module for Detect {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        for (l, name) in self.levels.iter().zip(LEVELS) {
            f(&format!("box.{name}"), &l.box_branch);
            f(&format!("cls.{name}"), &l.cls_branch);
        }
    }
}

impl Fuse for Detect {
    fn fuse_(&mut self) -> Result<()> {
        for l in &mut self.levels {
            let (b, c) = (&mut l.box_branch, &mut l.cls_branch);
            b.conv0.fuse_()?;
            b.conv1.fuse_()?;
            c.dw0.fuse_()?;
            c.pw0.fuse_()?;
            c.dw1.fuse_()?;
            c.pw1.fuse_()?;
        }
        Ok(())
    }
}
