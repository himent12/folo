//! The FOLO detector family `folo-n/s/m/l/x` (docs/MODEL.md §2–§6, §9).
//!
//! - [`FoloModel`]: backbone (Conv stem, C3k2 stages, SPPF, C2PSA), PAN-FPN neck and the detect head, built
//!   from a [`ModelConfig`] (`folo.config`) for any scale and class count. [`FoloModel::forward`] returns the
//!   raw training outputs of MODEL.md §6.3 as a [`HeadOutput`], with the anchors of the input size.
//! - [`blocks`] and [`head`]: the blocks of MODEL.md §2 and §6.1, with MODEL.md §5's tensor names.
//! - Inference: [`FoloModel::fuse`] folds every BatchNorm into its conv; [`FoloModel::predict_raw`] runs the
//!   forward pass, decodes the outputs (on their device) and runs [`crate::nms`] on the host.
//! - Devices: [`FoloModel::to_device`] moves the whole model in place; [`FoloModel::open_on`] loads onto a device.
//! - Weights: [`FoloModel::save`], [`FoloModel::export`], [`FoloModel::from_file`] /
//!   [`FoloModel::open`] and [`FoloModel::transfer_from`] (partial loading for a new class list) go through
//!   folo-nn's `.folo` format (docs/FORMAT.md).
//!
//! Files: [`blocks`] (§2), [`head`] (§6), `io.rs` (`.folo` weight IO), `predict.rs` (decode and the raw prediction
//! path, §6.4 and §9.3), `lazy.rs` and `lazy_gpu.rs` (candidate-first prediction on the CPU and on a GPU, §9.5),
//! [`quant`] (int8 inference, §9.6), and [`v2`] (FOLO v2-Iso clean-sheet model family).
//! This file holds [`FoloModel`] itself: construction for a scale and class count, the forward pass, fusion and
//! device moves.
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
pub mod detector;
pub mod head;
mod io;
mod lazy;
mod lazy_gpu;
mod predict;
pub mod quant;
pub mod v2;

pub use detector::{Detector, DetectorOutput};

use folo_core::rng::Rng;
use std::sync::Arc;

use folo_core::{Device, Tensor};
use folo_nn::{ConvBnAct, Error, Forward, Mode, Module, Result};

use crate::decode::{Anchors, make_anchors};
use blocks::{C2Psa, C3k2, Fuse, Init, Sppf};
use head::Detect;
pub use head::HeadMode;

pub use folo_nn::io::ModelConfig;
pub use lazy::ImageCandidates;
pub(crate) use predict::{DflAux, DflAuxOutputs};
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
    /// `[B, 4·reg_max, A]`: DFL logits, channel `side · reg_max + bin`, sides `(left, top, right, bottom)`, of
    /// the branch [`HeadOutput::mode`] names.
    pub box_raw: Tensor,
    /// `[B, nc, A]`: class logits of the same branch.
    pub cls_raw: Tensor,
    /// The branch `box_raw` and `cls_raw` come from. It also picks the post-processing of
    /// [`HeadOutput::detections`]: NMS for `o2m`, the top-k selection of MODEL.md §6.6 for `o2o`.
    pub mode: HeadMode,
    /// In training mode, the one-to-one branch's `(box_raw, cls_raw)` next to the one-to-many outputs (a
    /// `head = "o2o"` model, MODEL.md §6.6: the second term of the dual loss); `None` otherwise.
    pub o2o: Option<(Tensor, Tensor)>,
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

/// The callback of [`FoloModel::forward_traced`]: `(layer, output)`.
type LayerTrace<'a> = dyn FnMut(usize, &Tensor) + 'a;

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
    head_mode: HeadMode,
    /// Recorded GPU forwards of the fused model ([`FoloModel::set_forward_plans`]; off until turned on).
    plans: folo_gpu::ForwardPlans,
    /// The int8 plan of the backbone and neck on the CPU ([`FoloModel::set_int8`]; off until set).
    int8: Option<Arc<quant::Int8Plan>>,
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
        if config.is_v2() {
            return Err(Error::Metadata(format!(
                "FoloModel requires family `folo`, got `{}` (use FoloV2Model or Detector)",
                config.family
            )));
        }
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
        let (nc, reg_max, o2o) = (config.num_classes, config.reg_max as usize, config.has_o2o());
        let head = Detect::new([c16, c19, c22], strides, nc, reg_max, o2o, i)?;
        let classes = (0..config.num_classes).map(|c| format!("class{c}")).collect();
        let head_mode = if o2o { HeadMode::O2o } else { HeadMode::O2m };
        let (mode, fused) = (Mode::new(), false);
        let plans = folo_gpu::ForwardPlans::with_enabled(false);
        let int8 = None;
        Ok(FoloModel {
            config,
            classes,
            imgsz: 640,
            mode,
            fused,
            head_mode,
            plans,
            int8,
            backbone,
            neck,
            head,
        })
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

    /// Whether the model has the one-to-one (NMS-free) branch (`head = "o2o"`, MODEL.md §6.6).
    pub fn has_o2o(&self) -> bool {
        self.head.o2o.is_some()
    }

    /// The branch an eval-mode forward computes and predicts with: the config's (`o2o` for a `head = "o2o"`
    /// model, else `o2m`) unless [`FoloModel::set_head_mode`] changed it.
    pub fn head_mode(&self) -> HeadMode {
        self.head_mode
    }

    /// Predict with the one-to-many branch and NMS (`O2m`, any model) or with the one-to-one branch and no NMS
    /// (`O2o`, only a model that has it). Training is unaffected: a training-mode forward computes both.
    pub fn set_head_mode(&mut self, mode: HeadMode) -> Result<()> {
        if mode == HeadMode::O2o && !self.has_o2o() {
            return Err(Error::Invalid(format!(
                "{} has no one-to-one (NMS-free) head (folo.config head is `{}`); train with head o2o for one",
                self.arch(),
                self.config.head
            )));
        }
        self.head_mode = mode;
        Ok(())
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

    /// Return an independent eval-mode clone of `self` with every `ConvBnAct` BatchNorm pre-folded into its
    /// convolution (MODEL.md §9.1), leaving `self` (its weights, fused state and train/eval mode) untouched.
    ///
    /// Reconstructs the layer graph from `self.config` and copies every tensor, `classes`, `imgsz`,
    /// `head_mode`, and any active `int8` calibration plan. Used by the trainer to pre-fold EMA or live
    /// weights once per validation pass instead of re-folding on every validation batch, and to enable fused
    /// kernels / GPU [`folo_gpu::ForwardPlans`] during validation.
    pub fn fused_eval_clone(&self) -> Result<Self> {
        let mut clone = Self::new(self.config.clone())?;
        clone.set_classes(self.classes.clone())?;
        clone.set_imgsz(self.imgsz);
        clone.set_head_mode(self.head_mode)?;
        if self.fused {
            clone.fuse()?;
        }
        let dev = self.device();
        if dev != clone.device() {
            clone.to_device(dev)?;
        }
        let dtype = self.dtype();
        if dtype != clone.dtype() {
            Module::to_dtype(&mut clone, dtype)?;
        }
        {
            let _g = folo_core::no_grad();
            let src = self.named_tensors();
            let dst = clone.named_tensors();
            if src.len() != dst.len() {
                return Err(Error::Invalid(format!(
                    "fused_eval_clone: tensor count mismatch ({} vs {})",
                    src.len(),
                    dst.len()
                )));
            }
            for (s, d) in src.iter().zip(&dst) {
                if s.name != d.name {
                    return Err(Error::Invalid(format!(
                        "fused_eval_clone: tensor name mismatch (`{}` vs `{}`)",
                        s.name, d.name
                    )));
                }
                if d.tensor.device() != s.tensor.device() {
                    d.tensor.copy_from(&s.tensor.to_device(d.tensor.device())?)?;
                } else {
                    d.tensor.copy_from(&s.tensor)?;
                }
            }
        }
        if !self.fused {
            clone.fuse()?;
        }
        clone.int8 = self.int8.clone();
        clone.eval();
        for p in clone.parameters() {
            p.set_requires_grad(false);
        }
        Ok(clone)
    }

    /// Run the backbone and neck (layers 0–22) in int8 on the CPU with the ranges of `cal` (docs/MODEL.md §9.6),
    /// or in `f32` again with `None`. Needs a fused `f32` model, the one `cal` was measured on when it records
    /// its [`quant::fingerprint`], on the CPU (where every range is checked); the head stays `f32`. Applies to
    /// eval-mode CPU forwards and candidate-first prediction; GPU forwards (after a move) and training are
    /// unchanged.
    pub fn set_int8(&mut self, cal: Option<quant::Calibration>) -> Result<()> {
        if cal.is_some() && (!self.fused || self.dtype() != folo_core::DType::F32) {
            return Err(Error::Invalid("int8 needs a fused F32 model (FoloModel::fuse)".into()));
        }
        if cal.is_some() && !self.device().is_cpu() {
            return Err(Error::Invalid("int8 runs on the CPU: move the model there first".into()));
        }
        if let Some(want) = cal.as_ref().and_then(|c| c.weights.as_deref())
            && want != quant::fingerprint(self)?
        {
            return Err(Error::Invalid(
                "the int8 calibration was measured on other weights: run `folo quantize` on these".into(),
            ));
        }
        if let Some(c) = &cal {
            // Every boundary the graph reads must have a usable range: one small f32 pass of the fake-quant
            // reference names the first gap.
            let x = folo_core::Tensor::zeros([1, 3, 64, 64]);
            let _g = folo_core::no_grad();
            quant::FakeQuant::new(c).features(self, &x).map_err(|e| Error::Invalid(e.to_string()))?;
        }
        self.int8 = cal.map(|c| Arc::new(quant::Int8Plan::new(c)));
        Ok(())
    }

    /// The calibration of the int8 plan, if [`FoloModel::set_int8`] turned it on.
    pub fn int8(&self) -> Option<&quant::Calibration> {
        self.int8.as_deref().map(quant::Int8Plan::calibration)
    }

    /// The raw outputs (MODEL.md §6.3) of an `[B, 3, H, W]` batch with `H` and `W` positive multiples of 32.
    /// In training mode BatchNorm uses (and updates) batch statistics, and the output holds the one-to-many
    /// branch plus, in a `head = "o2o"` model, the one-to-one branch ([`HeadOutput::o2o`]). In eval mode
    /// BatchNorm uses its running statistics and only the [`FoloModel::head_mode`] branch runs.
    pub fn forward(&self, x: &Tensor) -> Result<HeadOutput> {
        self.forward_in_pool(x, self.head_mode)
    }

    /// [`FoloModel::forward`] in eval mode with the `mode` branch, whatever [`FoloModel::head_mode`] says
    /// (`o2o` needs a model that has it). In training mode it is [`FoloModel::forward`].
    pub fn forward_head(&self, x: &Tensor, mode: HeadMode) -> Result<HeadOutput> {
        if mode == HeadMode::O2o && !self.has_o2o() {
            return Err(Error::Invalid(format!(
                "{} has no one-to-one (NMS-free) head (folo.config head is `{}`)",
                self.arch(),
                self.config.head
            )));
        }
        self.forward_in_pool(x, mode)
    }

    /// [`FoloModel::forward_impl`] without a trace, on a CPU from inside the rayon pool ([`folo_core::cpu::in_pool`]):
    /// its hundred-odd parallel kernels then start on a worker instead of each waking the pool from outside. A fused
    /// eval-mode model on a GPU goes through its recorded plans ([`FoloModel::forward_planned`]).
    fn forward_in_pool(&self, x: &Tensor, mode: HeadMode) -> Result<HeadOutput> {
        if x.device().is_cpu() {
            folo_core::cpu::in_pool(|| self.forward_impl(x, None, mode))
        } else if self.fused && !self.is_training() {
            self.forward_planned(x, mode)
        } else {
            self.forward_impl(x, None, mode)
        }
    }

    /// The eval forward through [`folo_gpu::ForwardPlans`]: with plans on ([`FoloModel::set_forward_plans`]) and
    /// under `no_grad`, its second call for an input shape and dtype records the forward's dispatches and later calls
    /// replay them (bit for bit the same outputs) as long as no tensor of the model changed.
    fn forward_planned(&self, x: &Tensor, mode: HeadMode) -> Result<HeadOutput> {
        let (h, w) = self.check_input(x)?;
        let o2o = mode == HeadMode::O2o && self.has_o2o();
        let tensors = |f: &mut dyn FnMut(&Tensor)| each_tensor(self, f);
        let mut out = self.plans.run(&[o2o as u64], x, &tensors, |x| {
            let o = self.forward_impl(x, None, mode)?;
            Ok::<_, Error>(vec![o.box_raw, o.cls_raw])
        })?;
        let (Some(cls_raw), Some(box_raw)) = (out.pop(), out.pop()) else {
            unreachable!("two outputs");
        };
        Ok(HeadOutput {
            box_raw,
            cls_raw,
            mode: if o2o { HeadMode::O2o } else { HeadMode::O2m },
            o2o: None,
            anchors: self.anchors(h, w),
            reg_max: self.head.reg_max,
            num_classes: self.head.num_classes,
        })
    }

    /// Turns the recorded GPU forwards of this model on or off (off drops them; [`folo_gpu::ForwardPlans`]). They are
    /// off for a model on its own and on inside a [`crate::predict::Predictor`], which never lends its model out
    /// mutably. With plans on, a fused eval-mode forward on a GPU under `no_grad` is recorded at its second call per
    /// input shape and head mode and replayed from the third, bit for bit, until a tensor of the model is written or
    /// replaced. What a plan does not see is the layers' own settings (an activation, a stride, a block's `residual`,
    /// SPPF's `k`): turn plans off before changing one. `FOLO_GPU_NO_PLAN=1` or
    /// [`folo_gpu::GpuContext::set_forward_plans`] turns them off for a whole context.
    pub fn set_forward_plans(&self, on: bool) {
        self.plans.set_enabled(on);
    }

    /// Forwards of this model answered from a recorded plan so far.
    pub fn plan_replays(&self) -> u64 {
        self.plans.replays()
    }

    /// Why the last recording of this model's forward did not become a plan ([`folo_gpu::ForwardPlans::refusal`]).
    pub fn plan_refusal(&self) -> Option<String> {
        self.plans.refusal()
    }

    /// [`FoloModel::forward`], calling `trace(layer, output)` with the output of every layer 0–22 of
    /// MODEL.md §4 (see [`LAYER_NAMES`]) as it is computed.
    pub fn forward_traced(&self, x: &Tensor, trace: &mut dyn FnMut(usize, &Tensor)) -> Result<HeadOutput> {
        self.forward_impl(x, Some(trace), self.head_mode)
    }

    /// Whether CPU int8 inference with the direct `u8` stem is active for this model.
    pub(crate) fn int8_u8_stem_active(&self) -> bool {
        self.int8
            .as_ref()
            .is_some_and(|p| p.calibration().u8_stem && !p.calibration().is_f32("backbone.stem"))
            && self.device().is_cpu()
            && !self.is_training()
            && self.config.in_channels == 3
    }

    /// Forward pass reading a pre-packed `[b, h, w]` `u32` `[R, G, B, 0]` canvas straight into the direct `u8`
    /// int8 stem (skipping `f32` canvas materialisation).
    pub(crate) fn forward_rgb0_u8(&self, rgb0: &[u32], b: usize, h: usize, w: usize) -> Result<HeadOutput> {
        self.check_hw(b, h, w)?;
        let plan = self
            .int8
            .as_ref()
            .ok_or_else(|| Error::Invalid("forward_rgb0_u8 requires an int8 plan".into()))?;
        let feats = quant::exec::features_rgb0_u8(self, plan, rgb0, b, h, w)?;
        self.head_from_features(feats, h, w, self.head_mode)
    }

    /// The forward pass: [`FoloModel::features`], then the head.
    fn forward_impl(
        &self,
        x: &Tensor,
        trace: Option<&mut LayerTrace<'_>>,
        head_mode: HeadMode,
    ) -> Result<HeadOutput> {
        let (h, w) = self.check_input(x)?;
        let feats = self.features(x, trace)?;
        self.head_from_features(feats, h, w, head_mode)
    }

    fn head_from_features(
        &self,
        [x16, x19, x22]: [Tensor; 3],
        h: usize,
        w: usize,
        head_mode: HeadMode,
    ) -> Result<HeadOutput> {
        let feats = [&x16, &x19, &x22];
        let (mode, (box_raw, cls_raw), o2o) = if self.is_training() {
            (HeadMode::O2m, self.head.forward(feats)?, self.head.forward_o2o(feats)?)
        } else if head_mode == HeadMode::O2o
            && let Some(raw) = self.head.forward_o2o(feats)?
        {
            (HeadMode::O2o, raw, None)
        } else {
            (HeadMode::O2m, self.head.forward(feats)?, None)
        };
        Ok(HeadOutput {
            box_raw,
            cls_raw,
            mode,
            o2o,
            anchors: self.anchors(h, w),
            reg_max: self.head.reg_max,
            num_classes: self.head.num_classes,
        })
    }

    /// Layers 0–22 of a checked input: the head's inputs (the P3, P4 and P5 features, layers 16, 19 and 22).
    /// Without a trace, the neck's Concat layers (12, 15, 18, 21) are never written: the C3k2 after each reads
    /// both parts in place ([`C3k2::forward_cat`]).
    fn features(&self, x: &Tensor, mut trace: Option<&mut LayerTrace<'_>>) -> Result<[Tensor; 3]> {
        if let Some(plan) = &self.int8
            && trace.is_none()
            && x.device().is_cpu()
            && !self.is_training()
        {
            return quant::exec::features(self, plan, x);
        }
        let b = &self.backbone;
        let n = &self.neck;
        let traced = trace.is_some();
        let mut t = |i: usize, y: Tensor| -> Tensor {
            if let Some(f) = trace.as_mut() {
                f(i, &y);
            }
            y
        };
        // Layer `l` (a Concat) then the C3k2 `blk` after it, as layer `l + 1`.
        let cat_c3k2 =
            |t: &mut dyn FnMut(usize, Tensor) -> Tensor, l: usize, blk: &C3k2, a: &Tensor, c: &Tensor| {
                if traced {
                    let y = t(l, Tensor::cat(&[a, c], 1)?);
                    Ok::<_, Error>(t(l + 1, blk.forward(&y)?))
                } else {
                    Ok(t(l + 1, blk.forward_cat(&[a, c])?))
                }
            };
        // Only the skip connections (x4, x6, x10, x13) and the head inputs outlive the next layer. Every other
        // intermediate lives in `y`, whose old value is dropped on each reassignment (the skips right after the
        // C3k2 that reads them), so inference can recycle buffers as it goes.
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
        let x13 = cat_c3k2(&mut t, 12, &n.td4, &y, &x6)?;
        drop(x6);
        y = t(14, x13.upsample_nearest2d(2)?);
        let x16 = cat_c3k2(&mut t, 15, &n.out3, &y, &x4)?;
        drop(x4);
        y = t(17, n.down3.forward(&x16)?);
        let x19 = cat_c3k2(&mut t, 18, &n.out4, &y, &x13)?;
        drop(x13);
        y = t(20, n.down4.forward(&x19)?);
        let x22 = cat_c3k2(&mut t, 21, &n.out5, &y, &x10)?;
        Ok([x16, x19, x22])
    }

    fn check_hw(&self, b: usize, h: usize, w: usize) -> Result<()> {
        let s = *self.config.strides.last().expect("validated") as usize;
        if b == 0 || h == 0 || w == 0 || !h.is_multiple_of(s) || !w.is_multiple_of(s) {
            return Err(Error::Invalid(format!(
                "FoloModel::forward: input [{b}, {}, {h}, {w}] must be non-empty with H and W multiples of {s}",
                self.config.in_channels
            )));
        }
        Ok(())
    }

    fn check_input(&self, x: &Tensor) -> Result<(usize, usize)> {
        let bad = |why: &str| Error::Invalid(format!("FoloModel::forward: input {:?} {why}", x.dims()));
        let (b, c, h, w) = x.dims4().map_err(|_| bad("must be [B, 3, H, W]"))?;
        if c != self.config.in_channels {
            return Err(bad(&format!("must have {} channels", self.config.in_channels)));
        }
        self.check_hw(b, h, w)?;
        Ok((h, w))
    }

    /// The device of the weights.
    pub fn device(&self) -> Device {
        self.backbone.stem.conv.weight.device()
    }

    /// The floating dtype of the weights: `F32`, or `F16` after [`Module::to_dtype`] for half-precision
    /// inference (the forward then takes and returns `F16` tensors).
    pub fn dtype(&self) -> folo_core::DType {
        self.backbone.stem.conv.weight.dtype()
    }

    /// Move every weight and BatchNorm buffer to `device` in place (folo-nn's [`Module::to_device`]): tensor
    /// identities, names, `requires_grad` and the train/eval and fused state are kept, so an optimizer or
    /// EMA built over this model stays valid (move their own state with `Optimizer::to_device` /
    /// `ModelEma::to_device`). Inputs must be on the same device; [`FoloModel::forward`] then runs entirely
    /// there, and the outputs ([`HeadOutput`]) live there too.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use folo_vision::model::{FoloModel, ModelConfig};
    /// let dev = folo_gpu::init()?; // Device::Gpu(0)
    /// let mut model = FoloModel::new(ModelConfig::new("n", 80)?)?;
    /// model.to_device(dev)?;
    /// model.fuse()?; // fused Conv + bias + SiLU kernels on the GPU
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        // The move writes every weight, which drops the int8 plan's packed weights (kept with them).
        Module::to_device(self, device)
    }
}

/// Calls `f` with every tensor of `m` and its descendants, in canonical order, without allocating.
fn each_tensor(m: &dyn Module, f: &mut dyn FnMut(&Tensor)) {
    m.visit_tensors(&mut |_, t, _| f(t));
    m.visit_children(&mut |_, c| each_tensor(c, f));
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

    #[test]
    fn fused_eval_clone_matches_unfused_and_leaves_source_untouched() {
        let model = FoloModel::new(ModelConfig::new("n", 3).unwrap()).unwrap();
        model.train();
        let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(17));
        // Run one train step so BN running stats are non-trivial, then compare eval outputs.
        let _ = model.forward(&x).unwrap();
        model.eval();
        let ref_out = {
            let _g = folo_core::no_grad();
            model.forward(&x).unwrap()
        };
        model.train();
        assert!(!model.is_fused());
        assert!(model.is_training());

        let fused = model.fused_eval_clone().unwrap();
        assert!(fused.is_fused());
        assert!(!fused.is_training());
        // Source model remains unfused and in training mode.
        assert!(!model.is_fused());
        assert!(model.is_training());

        let fused_out = {
            let _g = folo_core::no_grad();
            fused.forward(&x).unwrap()
        };
        for (a, b) in
            ref_out.box_raw.to_vec::<f32>().unwrap().iter().zip(fused_out.box_raw.to_vec::<f32>().unwrap())
        {
            assert!((a - b).abs() < 1e-4, "box_raw mismatch: {a} vs {b}");
        }
        for (a, b) in
            ref_out.cls_raw.to_vec::<f32>().unwrap().iter().zip(fused_out.cls_raw.to_vec::<f32>().unwrap())
        {
            assert!((a - b).abs() < 1e-4, "cls_raw mismatch: {a} vs {b}");
        }

        // Cloning an already-fused model also produces matching outputs.
        let fused2 = fused.fused_eval_clone().unwrap();
        assert!(fused2.is_fused());
        let fused2_out = {
            let _g = folo_core::no_grad();
            fused2.forward(&x).unwrap()
        };
        assert_eq!(fused_out.box_raw.to_vec::<f32>().unwrap(), fused2_out.box_raw.to_vec::<f32>().unwrap());
        assert_eq!(fused_out.cls_raw.to_vec::<f32>().unwrap(), fused2_out.cls_raw.to_vec::<f32>().unwrap());
    }
}
