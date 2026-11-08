//! The FOLO detector family `folo-n/s/m/l/x` (docs/MODEL.md §2–§6, §9).
//!
//! - [`FoloModel`]: backbone (Conv stem, C3k2 stages, SPPF, C2PSA), PAN-FPN neck and the detect head, built
//!   from a [`ModelConfig`] (`folo.config`) for any scale and class count. [`FoloModel::forward`] returns the
//!   raw training outputs of MODEL.md §6.3 as a [`HeadOutput`], with the anchors of the input size.
//! - [`blocks`] and [`head`]: the blocks of MODEL.md §2 and §6.1, with MODEL.md §5's tensor names.
//! - Inference: [`FoloModel::fuse`] folds every BatchNorm into its conv; [`FoloModel::predict_raw`] runs the
//!   forward pass and decodes + NMSes CPU copies of the outputs with [`crate::decode`] and [`crate::nms`].
//! - Weights: [`FoloModel::save`], [`FoloModel::export`], [`FoloModel::from_file`] /
//!   [`FoloModel::open`] and [`FoloModel::transfer_from`] (partial loading for a new class list) go through
//!   folo-nn's `.folo` format (docs/FORMAT.md).
//!
//! ```
//! use folo_core::{no_grad, Tensor};
//! use folo_nn::Module;
//! use folo_vision::model::{FoloModel, ModelConfig};
//!
//! # fn main() -> folo_nn::Result<()> {
//! let model = FoloModel::new(ModelConfig::new("n", 4)?)?;
//! assert_eq!(model.num_parameters(), 2_590_604);
//! model.eval();
//! let _g = no_grad();
//! let out = model.forward(&Tensor::zeros([1, 3, 64, 96]))?;
//! assert_eq!(out.box_raw.dims(), [1, 64, 8 * 12 + 4 * 6 + 2 * 3]);
//! assert_eq!(out.cls_raw.dims(), [1, 4, 126]);
//! # Ok(())
//! # }
//! ```

pub mod blocks;
pub mod head;
mod io;
mod predict;

use folo_core::rng::Rng;
use folo_core::{Device, Tensor};
use folo_nn::{ConvBnAct, Error, Forward, Mode, Module, Result};

use crate::decode::{Anchors, make_anchors};
use blocks::{C2Psa, C3k2, Fuse, Init, Sppf};
use head::Detect;

pub use folo_nn::io::ModelConfig;
pub use predict::{ImagePrediction, RawPrediction};

/// The layer prefixes of MODEL.md §4, indexed by layer number (`""` for the parameter-free Upsample and Concat
/// layers). [`FoloModel::forward_traced`] reports outputs by the same index.
pub const LAYER_NAMES: [&str; 24] = [
    "backbone.stem",
    "backbone.down2",
    "backbone.stage2",
    "backbone.down3",
    "backbone.stage3",
    "backbone.down4",
    "backbone.stage4",
    "backbone.down5",
    "backbone.stage5",
    "backbone.sppf",
    "backbone.psa",
    "",
    "",
    "neck.td4",
    "",
    "",
    "neck.out3",
    "neck.down3",
    "",
    "neck.out4",
    "neck.down4",
    "",
    "neck.out5",
    "head",
];

/// The scale arithmetic of MODEL.md §3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scaling {
    /// Depth multiplier `d`.
    pub depth: f64,
    /// Width multiplier `w`.
    pub width: f64,
    /// Channel cap `M`, applied before the width multiplier.
    pub max_channels: usize,
    /// True for `m`, `l` and `x`: every C3k2 nests C3k blocks.
    pub force_c3k: bool,
}

impl Scaling {
    /// The scaling of a validated config.
    pub fn of(config: &ModelConfig) -> Self {
        Scaling {
            depth: config.depth,
            width: config.width,
            max_channels: config.max_channels as usize,
            force_c3k: matches!(config.scale.as_str(), "m" | "l" | "x"),
        }
    }

    /// `ch(c) = ceil(min(c, M) · w / 8) · 8`.
    pub fn ch(&self, c: usize) -> usize {
        (c.min(self.max_channels) as f64 * self.width / 8.0).ceil() as usize * 8
    }

    /// `rep(n) = max(round(n · d), 1)` if `n > 1`, else `n`.
    pub fn rep(&self, n: usize) -> usize {
        if n > 1 { ((n as f64 * self.depth).round() as usize).max(1) } else { n }
    }

    /// The C3k2 `c3k` flag: the base flag, forced on for `m`, `l`, `x`.
    pub fn c3k(&self, base: bool) -> bool {
        base || self.force_c3k
    }
}

/// The backbone, layers 0–10 of MODEL.md §4 (`backbone.*`).
pub struct Backbone {
    /// 0: `Conv(64, k3, s2)`, P1/2.
    pub stem: ConvBnAct,
    /// 1: `Conv(128, k3, s2)`, P2/4.
    pub down2: ConvBnAct,
    /// 2: `C3k2(256, n2, e=0.25)`.
    pub stage2: C3k2,
    /// 3: `Conv(256, k3, s2)`, P3/8.
    pub down3: ConvBnAct,
    /// 4: `C3k2(512, n2, e=0.25)`.
    pub stage3: C3k2,
    /// 5: `Conv(512, k3, s2)`, P4/16.
    pub down4: ConvBnAct,
    /// 6: `C3k2(512, n2, c3k)`.
    pub stage4: C3k2,
    /// 7: `Conv(1024, k3, s2)`, P5/32.
    pub down5: ConvBnAct,
    /// 8: `C3k2(1024, n2, c3k)`.
    pub stage5: C3k2,
    /// 9: `SPPF(1024, k5)`.
    pub sppf: Sppf,
    /// 10: `C2PSA(1024, n2)`.
    pub psa: C2Psa,
}

impl Module for Backbone {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("stem", &self.stem);
        f("down2", &self.down2);
        f("stage2", &self.stage2);
        f("down3", &self.down3);
        f("stage3", &self.stage3);
        f("down4", &self.down4);
        f("stage4", &self.stage4);
        f("down5", &self.down5);
        f("stage5", &self.stage5);
        f("sppf", &self.sppf);
        f("psa", &self.psa);
    }
}

impl Fuse for Backbone {
    fn fuse_(&mut self) -> Result<()> {
        self.stem.fuse_()?;
        self.down2.fuse_()?;
        self.stage2.fuse_()?;
        self.down3.fuse_()?;
        self.stage3.fuse_()?;
        self.down4.fuse_()?;
        self.stage4.fuse_()?;
        self.down5.fuse_()?;
        self.stage5.fuse_()?;
        self.sppf.fuse_()?;
        self.psa.fuse_()
    }
}

/// The PAN-FPN neck, layers 11–22 of MODEL.md §4 (`neck.*`).
pub struct Neck {
    /// 13: top-down `C3k2(512, n2)` on `cat(up(P5), P4)`.
    pub td4: C3k2,
    /// 16: `C3k2(256, n2)` on `cat(up(td4), P3)`, the P3 output.
    pub out3: C3k2,
    /// 17: `Conv(256, k3, s2)`.
    pub down3: ConvBnAct,
    /// 19: `C3k2(512, n2)` on `cat(down3, td4)`, the P4 output.
    pub out4: C3k2,
    /// 20: `Conv(512, k3, s2)`.
    pub down4: ConvBnAct,
    /// 22: `C3k2(1024, n2, c3k)` on `cat(down4, psa)`, the P5 output.
    pub out5: C3k2,
}

impl Module for Neck {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("td4", &self.td4);
        f("out3", &self.out3);
        f("down3", &self.down3);
        f("out4", &self.out4);
        f("down4", &self.down4);
        f("out5", &self.out5);
    }
}

impl Fuse for Neck {
    fn fuse_(&mut self) -> Result<()> {
        self.td4.fuse_()?;
        self.out3.fuse_()?;
        self.down3.fuse_()?;
        self.out4.fuse_()?;
        self.down4.fuse_()?;
        self.out5.fuse_()
    }
}

/// The training output of the model (MODEL.md §6.3), the loss's input.
#[derive(Clone)]
pub struct HeadOutput {
    /// `[B, 4·reg_max, A]`: DFL logits, channel `side · reg_max + bin`, sides `(left, top, right, bottom)`.
    pub box_raw: Tensor,
    /// `[B, nc, A]`: class logits.
    pub cls_raw: Tensor,
    /// The anchor points (grid units, cell centres), per-anchor strides and level shapes for this input size
    /// (MODEL.md §6.2: level-major, row-major within a level).
    pub anchors: Anchors,
    /// DFL bins per side.
    pub reg_max: usize,
    /// Number of classes.
    pub num_classes: usize,
}

impl HeadOutput {
    /// Images in the batch.
    pub fn batch(&self) -> usize {
        self.box_raw.dims()[0]
    }

    /// Anchors per image, `A`.
    pub fn num_anchors(&self) -> usize {
        self.anchors.len()
    }
}

/// A FOLO detector (docs/MODEL.md): `folo-<scale>` with `num_classes` classes.
///
/// Tensor names and their canonical order are MODEL.md §5's (`backbone.stem.conv.weight` first,
/// `head.cls.p5.out.bias` last); initialisation draws every tensor in that order from one seeded stream (§2.9)
/// and then writes the head bias priors (§6.5). The model starts in training mode; call `eval()` (from
/// [`Module`]) for inference, or [`FoloModel::fuse`].
pub struct FoloModel {
    config: ModelConfig,
    classes: Vec<String>,
    imgsz: u32,
    mode: Mode,
    fused: bool,
    /// Layers 0–10.
    pub backbone: Backbone,
    /// Layers 11–22.
    pub neck: Neck,
    /// Layer 23.
    pub head: Detect,
}

impl FoloModel {
    /// Build the model of `config`, initialised from seed 0. Class names default to `class0`, `class1`, …,
    /// and the training image size to 640.
    pub fn new(config: ModelConfig) -> Result<Self> {
        Self::with_rng(config, &mut Rng::seed(0))
    }

    /// Build the model of `config`, drawing the initial weights from `rng` in canonical order.
    pub fn with_rng(config: ModelConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let s = Scaling::of(&config);
        let mut init = Init { rng, bn_eps: config.bn_eps, bn_momentum: config.bn_momentum };
        let i = &mut init;
        let (ch, rep) = (|c| s.ch(c), |n| s.rep(n));
        let c_in = config.in_channels;

        let (c0, c1, c2, c3, c4) = (ch(64), ch(128), ch(256), ch(256), ch(512));
        let (c5, c6, c7, c8, c9) = (ch(512), ch(512), ch(1024), ch(1024), ch(1024));
        let backbone = Backbone {
            stem: i.conv(c_in, c0, 3, 2),
            down2: i.conv(c0, c1, 3, 2),
            stage2: C3k2::new(c1, c2, rep(2), s.c3k(false), 0.25, i),
            down3: i.conv(c2, c3, 3, 2),
            stage3: C3k2::new(c3, c4, rep(2), s.c3k(false), 0.25, i),
            down4: i.conv(c4, c5, 3, 2),
            stage4: C3k2::new(c5, c6, rep(2), s.c3k(true), 0.5, i),
            down5: i.conv(c6, c7, 3, 2),
            stage5: C3k2::new(c7, c8, rep(2), s.c3k(true), 0.5, i),
            sppf: Sppf::new(c8, c9, 5, i),
            psa: C2Psa::new(c9, rep(2), i)?,
        };
        let c10 = c9;
        let (c13, c16, c17, c19, c20, c22) = (ch(512), ch(256), ch(256), ch(512), ch(512), ch(1024));
        let neck = Neck {
            td4: C3k2::new(c10 + c6, c13, rep(2), s.c3k(false), 0.5, i),
            out3: C3k2::new(c13 + c4, c16, rep(2), s.c3k(false), 0.5, i),
            down3: i.conv(c16, c17, 3, 2),
            out4: C3k2::new(c17 + c13, c19, rep(2), s.c3k(false), 0.5, i),
            down4: i.conv(c19, c20, 3, 2),
            out5: C3k2::new(c20 + c10, c22, rep(2), s.c3k(true), 0.5, i),
        };
        let strides = [config.strides[0], config.strides[1], config.strides[2]];
        let head = Detect::new([c16, c19, c22], strides, config.num_classes, config.reg_max as usize, i)?;
        let classes = (0..config.num_classes).map(|c| format!("class{c}")).collect();
        Ok(FoloModel { config, classes, imgsz: 640, mode: Mode::new(), fused: false, backbone, neck, head })
    }

    /// The model config (`folo.config`).
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// `folo-<scale>` (`folo.arch`).
    pub fn arch(&self) -> String {
        format!("folo-{}", self.config.scale)
    }

    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        self.config.num_classes
    }

    /// The class names, index = class id (`folo.classes`).
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Set the class names; there must be exactly `num_classes` of them.
    pub fn set_classes(&mut self, classes: Vec<String>) -> Result<()> {
        if classes.len() != self.config.num_classes {
            return Err(Error::Invalid(format!(
                "the model has {} classes, got {} names",
                self.config.num_classes,
                classes.len()
            )));
        }
        self.classes = classes;
        Ok(())
    }

    /// The training image size recorded in weight files (`folo.imgsz`).
    pub fn imgsz(&self) -> u32 {
        self.imgsz
    }

    /// Set the training image size recorded in weight files.
    pub fn set_imgsz(&mut self, imgsz: u32) {
        self.imgsz = imgsz;
    }

    /// The head strides, `[8, 16, 32]`.
    pub fn strides(&self) -> [u32; 3] {
        [self.config.strides[0], self.config.strides[1], self.config.strides[2]]
    }

    /// The anchors of an `h × w` input (MODEL.md §6.2).
    pub fn anchors(&self, h: usize, w: usize) -> Anchors {
        make_anchors(h as u32, w as u32, &self.config.strides, 0.5)
    }

    /// True once [`FoloModel::fuse`] has run (or the model was loaded from a fused file).
    pub fn is_fused(&self) -> bool {
        self.fused
    }

    /// Fold every BatchNorm into its conv (MODEL.md §9.1, in `F32`) and switch to eval mode. The head
    /// `Conv2d`s are unchanged. The fused model has the tensor set of a `folo.fused = "true"` file and is for
    /// inference only. A no-op if already fused.
    pub fn fuse(&mut self) -> Result<()> {
        self.eval();
        if self.fused {
            return Ok(());
        }
        self.backbone.fuse_()?;
        self.neck.fuse_()?;
        self.head.fuse_()?;
        self.fused = true;
        Ok(())
    }

    /// The raw outputs (MODEL.md §6.3) of an `[B, 3, H, W]` batch with `H` and `W` positive multiples of 32.
    /// In training mode BatchNorm uses (and updates) batch statistics; in eval mode its running statistics.
    pub fn forward(&self, x: &Tensor) -> Result<HeadOutput> {
        self.forward_traced(x, &mut |_, _| {})
    }

    /// [`FoloModel::forward`], calling `trace(layer, output)` with the output of every layer 0–22 of
    /// MODEL.md §4 (see [`LAYER_NAMES`]) as it is computed.
    pub fn forward_traced(&self, x: &Tensor, trace: &mut dyn FnMut(usize, &Tensor)) -> Result<HeadOutput> {
        let (h, w) = self.check_input(x)?;
        let b = &self.backbone;
        let n = &self.neck;
        let mut t = |i: usize, y: Tensor| -> Tensor {
            trace(i, &y);
            y
        };
        // Only the skip connections (x4, x6, x10, x13) and the head inputs outlive the next layer. Every other
        // intermediate lives in `y`, whose old value is dropped on each reassignment (the skips right after their
        // concat), so inference can recycle buffers as it goes.
        let mut y = t(0, b.stem.forward(x)?);
        y = t(1, b.down2.forward(&y)?);
        y = t(2, b.stage2.forward(&y)?);
        y = t(3, b.down3.forward(&y)?);
        let x4 = t(4, b.stage3.forward(&y)?);
        y = t(5, b.down4.forward(&x4)?);
        let x6 = t(6, b.stage4.forward(&y)?);
        y = t(7, b.down5.forward(&x6)?);
        y = t(8, b.stage5.forward(&y)?);
        y = t(9, b.sppf.forward(&y)?);
        let x10 = t(10, b.psa.forward(&y)?);
        y = t(11, x10.upsample_nearest2d(2)?);
        y = t(12, Tensor::cat(&[&y, &x6], 1)?);
        drop(x6);
        let x13 = t(13, n.td4.forward(&y)?);
        y = t(14, x13.upsample_nearest2d(2)?);
        y = t(15, Tensor::cat(&[&y, &x4], 1)?);
        drop(x4);
        let x16 = t(16, n.out3.forward(&y)?);
        y = t(17, n.down3.forward(&x16)?);
        y = t(18, Tensor::cat(&[&y, &x13], 1)?);
        drop(x13);
        let x19 = t(19, n.out4.forward(&y)?);
        y = t(20, n.down4.forward(&x19)?);
        y = t(21, Tensor::cat(&[&y, &x10], 1)?);
        drop(x10);
        let x22 = t(22, n.out5.forward(&y)?);
        drop(y);
        let (box_raw, cls_raw) = self.head.forward([&x16, &x19, &x22])?;
        Ok(HeadOutput {
            box_raw,
            cls_raw,
            anchors: self.anchors(h, w),
            reg_max: self.head.reg_max,
            num_classes: self.head.num_classes,
        })
    }

    fn check_input(&self, x: &Tensor) -> Result<(usize, usize)> {
        let bad = |why: &str| Error::Invalid(format!("FoloModel::forward: input {:?} {why}", x.dims()));
        let (b, c, h, w) = x.dims4().map_err(|_| bad("must be [B, 3, H, W]"))?;
        if c != self.config.in_channels {
            return Err(bad(&format!("must have {} channels", self.config.in_channels)));
        }
        let s = *self.config.strides.last().expect("validated") as usize;
        if b == 0 || h == 0 || w == 0 || !h.is_multiple_of(s) || !w.is_multiple_of(s) {
            return Err(bad(&format!("must be non-empty with H and W multiples of {s}")));
        }
        Ok((h, w))
    }

    /// The device of the weights.
    pub fn device(&self) -> Device {
        self.backbone.stem.conv.weight.device()
    }
}

impl Module for FoloModel {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("backbone", &self.backbone);
        f("neck", &self.neck);
        f("head", &self.head);
    }

    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_rules() {
        let row = |s: &str| Scaling::of(&ModelConfig::new(s, 80).unwrap());
        let n = row("n");
        assert_eq!([n.ch(64), n.ch(256), n.ch(1024)], [16, 64, 256]);
        assert_eq!([n.rep(1), n.rep(2), n.rep(3)], [1, 1, 2]);
        assert!(!n.c3k(false) && n.c3k(true));
        let m = row("m");
        assert_eq!([m.ch(1024), m.ch(256)], [512, 256]);
        assert!(m.c3k(false));
        let x = row("x");
        assert_eq!([x.ch(64), x.ch(1024), x.rep(2)], [96, 768, 2]);
    }
}
