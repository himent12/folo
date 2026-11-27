//! Clean-sheet **FOLO v2** detector family (`folo2-n/s/m/l/x`).
//!
//! Co-designed for FOLO's custom Rust/WGSL/AVX-VNNI kernels while keeping 100% standard CPU+GPU autograd
//! and ONNX opset-17 compatibility:
//! - [`config`]: [`FoloV2Config`] and `ceil16`-aligned [`ScalingV2`] (shared `c_box=48, c_cls=64` on `n/s`).
//! - [`blocks`]: [`StemV2`], [`WsQARepConv3x3`], [`FoloBlock`], [`space_to_depth_2x2`], [`HyperContext`],
//!   and relocated [`FullPadGate`].
//! - [`head`]: [`LazyAlignHead`] with DW-free dense `cls`, aligned `box.out = 64`, separate opt-in
//!   `iou_out`, and `PeakSuppress` `o2o` spatial competition.

pub mod blocks;
pub mod config;
pub mod head;
mod io;
pub mod iou;
mod lazy;

use std::sync::Arc;

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::{ConvBnAct, Error, Forward, Mode, Module, ModuleList, Result};
use rayon::prelude::*;

use crate::boxes::Xyxy;
use crate::decode::{Anchors, make_anchors};
use crate::model::blocks::{Fuse, Init, Sppf};
use crate::model::{HeadMode, HeadOutput, quant};
use crate::nms::{Detection, NmsConfig, postprocess_with_inputs};

pub use blocks::{
    FoloBlock, FullPadGate, HyperContext, QARepBranches, StemV2, WsQARepConv3x3, space_to_depth_2x2,
};
pub use config::{FoloV2Config, MAL_CLS_BIAS_PRIOR, ScalingV2};
pub use head::{BoxBranchV2, ClsBranchV2, HeadLevelV2, LazyAlignHead, O2oClsLevel, PeakSuppress};
pub use iou::{DEFAULT_IOU_FUSION_ALPHA, fuse_cls_iou_scores, iou_quality_loss};

/// FOLO v2 backbone: `StemV2`, wide `QARep` `stage2` (P2 tap), `FoloBlock` stages 3–5, `SPPF`, and
/// `HyperContext` (`HyperACE-Lite` at P5).
pub struct BackboneV2 {
    /// Layer 0: `SPD(2×2) + Conv1×1(12 → c0)` (or `Conv3×3 s=2` when `spd_stem == false`), P1/2.
    pub stem: StemV2,
    /// Layer 1: `Conv(c0 → c1, 3×3, s=2)`, P2/4.
    pub down2: ConvBnAct,
    /// Layer 2: `stage2_m` wide `WS-QARepConv3x3(c1 → c1)` layers (P2 tap for `SPD-P2→P3`).
    pub stage2: ModuleList<WsQARepConv3x3>,
    /// Layer 3: `Conv(c1 → c3, 3×3, s=2)`, P3/8.
    pub down3: ConvBnAct,
    /// Layer 4: `FoloBlock(c3 → c4, bb_m, c_mid = c3 / 2)` (P3 tap).
    pub stage3: FoloBlock,
    /// Layer 5: `Conv(c4 → c5, 3×3, s=2)`, P4/16.
    pub down4: ConvBnAct,
    /// Layer 6: `FoloBlock(c5 → c6, bb_m, c_mid = c6 / 2)` (P4 tap).
    pub stage4: FoloBlock,
    /// Layer 7: `Conv(c6 → c7, 3×3, s=2)`, P5/32.
    pub down5: ConvBnAct,
    /// Layer 8: `FoloBlock(c7 → c8, bb_m, c_mid = c8 / 2)`.
    pub stage5: FoloBlock,
    /// Layer 9: `SPPF(c8 → c9, k=5)`.
    pub sppf: Sppf,
    /// Layer 10: `HyperContext(c9, M)` at P5, returning `(P5, g_scene)`.
    pub hyper: HyperContext,
}

impl Module for BackboneV2 {
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
        f("hyper", &self.hyper);
    }
}

impl Fuse for BackboneV2 {
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
        self.hyper.fuse_()
    }
}

/// FOLO v2 SPD-PAN neck with lossless `SPD-P2→P3` cross-scale injection and relocated `FullPAD-Lite`
/// FiLM gates applied strictly to `f32` head inputs (`P3_head`, `P4_head`) AFTER `down3`/`down4` taps.
pub struct NeckV2 {
    /// Top-down P4 fusion: `FoloBlock(cat[up(P5), P4] → c13)`.
    pub td4: FoloBlock,
    /// Lossless `SPD(2×2)(P2)` channel projection `Conv(4·c1 → c_p2, 1×1)`.
    pub p2_proj: ConvBnAct,
    /// P3 output fusion: `FoloBlock(cat[up(td4), P3, p2_spd] → c16)` (3 zero-copy parts, 8,400 anchors kept).
    pub out3: FoloBlock,
    /// Bottom-up stride-2 `Conv(c16 → c16, 3×3, s=2)` reading un-modulated `out3` (contiguous int8!).
    pub down3: ConvBnAct,
    /// P4 output fusion: `FoloBlock(cat[down3, td4] → c19)`.
    pub out4: FoloBlock,
    /// Bottom-up stride-2 `Conv(c19 → c19, 3×3, s=2)` reading un-modulated `out4` (contiguous int8!).
    pub down4: ConvBnAct,
    /// P5 output fusion: `FoloBlock(cat[down4, P5] → c22)`.
    pub out5: FoloBlock,
    /// Relocated `FullPAD-Lite` FiLM gate applied to `P3` head input only (after `down3` tap!).
    pub pad_p3: Option<FullPadGate>,
    /// Relocated `FullPAD-Lite` FiLM gate applied to `P4` head input only (after `down4` tap!).
    pub pad_p4: Option<FullPadGate>,
}

impl Module for NeckV2 {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("td4", &self.td4);
        f("p2_proj", &self.p2_proj);
        f("out3", &self.out3);
        f("down3", &self.down3);
        f("out4", &self.out4);
        f("down4", &self.down4);
        f("out5", &self.out5);
        if let Some(g) = &self.pad_p3 {
            f("pad_p3", g);
        }
        if let Some(g) = &self.pad_p4 {
            f("pad_p4", g);
        }
    }
}

impl Fuse for NeckV2 {
    fn fuse_(&mut self) -> Result<()> {
        self.td4.fuse_()?;
        self.p2_proj.fuse_()?;
        self.out3.fuse_()?;
        self.down3.fuse_()?;
        self.out4.fuse_()?;
        self.down4.fuse_()?;
        self.out5.fuse_()
    }
}

/// Raw forward output of [`FoloV2Model`].
///
/// Mirrors [`HeadOutput`] with the opt-in [`HeadOutputV2::iou_raw`] localization-quality logit tensor
/// (`[B, 1, A]`), and converts cleanly into [`HeadOutput`] via [`HeadOutputV2::into_head_output`].
#[derive(Clone)]
pub struct HeadOutputV2 {
    /// `[B, 4·reg_max, A]`: DFL logits (strictly `64` channels for `reg_max = 16`).
    pub box_raw: Tensor,
    /// `[B, nc, A]`: class logits of [`HeadOutputV2::mode`].
    pub cls_raw: Tensor,
    /// Optional `[B, 1, A]` geometric localization-quality logits (when `config.iou_head == true`).
    pub iou_raw: Option<Tensor>,
    /// Branch producing `box_raw` and `cls_raw`.
    pub mode: HeadMode,
    /// In training mode on an `o2o` model, the one-to-one `(box_raw, cls_raw)` outputs.
    pub o2o: Option<(Tensor, Tensor)>,
    /// Anchors for this input spatial size (8,400 at 640×640).
    pub anchors: Anchors,
    /// DFL bins per side.
    pub reg_max: usize,
    /// Number of classes.
    pub num_classes: usize,
}

impl HeadOutputV2 {
    /// Batch size `B`.
    pub fn batch(&self) -> usize {
        self.box_raw.dims()[0]
    }

    /// Anchors per image `A`.
    pub fn num_anchors(&self) -> usize {
        self.anchors.len()
    }

    /// Convert into v1's [`HeadOutput`] for compatibility with existing decode/NMS/loss utilities.
    pub fn into_head_output(self) -> HeadOutput {
        HeadOutput {
            box_raw: self.box_raw,
            cls_raw: self.cls_raw,
            mode: self.mode,
            o2o: self.o2o,
            anchors: self.anchors,
            reg_max: self.reg_max,
            num_classes: self.num_classes,
        }
    }

    /// Validate `box_raw`, `cls_raw`, `anchors`, and optional `iou_raw` shapes before decoding.
    fn check_decode_shapes(&self, check_iou: bool) -> Result<(usize, usize, usize)> {
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if nc == 0 || rm < 2 {
            return Err(Error::Invalid(format!("HeadOutputV2: invalid num_classes={nc} or reg_max={rm}")));
        }
        if self.anchors.strides.len() != a
            || self.box_raw.dims() != [b, 4 * rm, a]
            || self.cls_raw.dims() != [b, nc, a]
        {
            return Err(Error::Invalid(format!(
                "HeadOutputV2: box_raw {:?} / cls_raw {:?} do not match {a} anchors (nc={nc}, reg_max={rm})",
                self.box_raw.dims(),
                self.cls_raw.dims()
            )));
        }
        if check_iou
            && let Some(iou_raw) = &self.iou_raw
            && iou_raw.dims() != [b, 1, a]
        {
            return Err(Error::Invalid(format!(
                "HeadOutputV2: iou_raw {:?} does not match [{b}, 1, {a}]",
                iou_raw.dims()
            )));
        }
        Ok((b, a, nc))
    }

    /// Decode every image's `(boxes [A], scores [A, nc])` in input pixels, fusing `iou_raw` with exponent
    /// `iou_alpha` in `0..=1` (`p_cls^(1 - alpha) * sigmoid(iou_raw)^alpha`) when `iou_raw` is present and
    /// `iou_alpha > 0`.
    pub fn decode_images(&self, iou_alpha: f32) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        if !iou_alpha.is_finite() || !(0.0..=1.0).contains(&iou_alpha) {
            return Err(Error::Invalid(format!(
                "HeadOutputV2::decode_images: iou_alpha must be in 0..=1, got {iou_alpha}"
            )));
        }
        let (b, a, nc) = self.check_decode_shapes(iou_alpha > 0.0)?;
        if a == 0 {
            return Ok(vec![(Vec::new(), Vec::new()); b]);
        }
        let head = self.clone().into_head_output();
        let mut decoded = head.decode_images()?;
        if iou_alpha == 0.0 {
            return Ok(decoded);
        }
        let Some(iou_raw) = &self.iou_raw else {
            return Ok(decoded);
        };
        let _g = no_grad();
        let iou_host = if iou_raw.device().is_cpu() {
            iou_raw.to_dtype(DType::F32)?
        } else {
            iou_raw.to_device(Device::Cpu)?.to_dtype(DType::F32)?
        };
        let iou_vals = iou_host.to_vec::<f32>()?;
        for ((_, scores), iou_img) in decoded.iter_mut().zip(iou_vals.chunks_exact(a)) {
            fuse_cls_iou_scores(scores, iou_img, nc, iou_alpha);
        }
        Ok(decoded)
    }

    /// Decode and post-process every image into detections, fusing `iou_raw` with exponent `iou_alpha` in
    /// `0..=1` when `iou_raw` is present and `iou_alpha > 0`. When `iou_raw` is `None` or `iou_alpha == 0.0`,
    /// delegates straight to [`HeadOutput::detections`] (preserving every fast path).
    pub fn detections(&self, nms: &NmsConfig, iou_alpha: f32) -> Result<Vec<Vec<Detection>>> {
        if !iou_alpha.is_finite() || !(0.0..=1.0).contains(&iou_alpha) {
            return Err(Error::Invalid(format!(
                "HeadOutputV2::detections: iou_alpha must be in 0..=1, got {iou_alpha}"
            )));
        }
        let (b, a, nc) = self.check_decode_shapes(iou_alpha > 0.0)?;
        if a == 0 {
            return Ok(vec![Vec::new(); b]);
        }
        let head = self.clone().into_head_output();
        if self.iou_raw.is_none() || iou_alpha == 0.0 {
            return head.detections(nms);
        }
        let mode = self.mode;
        let aux = head.dfl_aux_for(nms)?;
        let decoded = self.decode_images(iou_alpha)?;
        Ok(decoded
            .into_par_iter()
            .enumerate()
            .map(|(i, (b, s))| postprocess_with_inputs(&b, &s, aux.inputs(i), nc, nms, mode))
            .collect())
    }
}

/// A FOLO v2 detector (`folo2-<scale>`).
pub struct FoloV2Model {
    config: FoloV2Config,
    classes: Vec<String>,
    imgsz: u32,
    mode: Mode,
    fused: bool,
    head_mode: HeadMode,
    /// The int8 plan of the backbone and neck on the CPU ([`FoloV2Model::set_int8`]; off until set).
    int8: Option<Arc<quant::Int8Plan>>,
    /// Backbone (`backbone.*`).
    pub backbone: BackboneV2,
    /// SPD-PAN neck + relocated head-input FiLM gates (`neck.*`).
    pub neck: NeckV2,
    /// Decoupled `LazyAlignHead` (`head.*`).
    pub head: LazyAlignHead,
}

impl FoloV2Model {
    /// Build a FOLO v2 model from `config` using seed `0`.
    pub fn new(config: FoloV2Config) -> Result<Self> {
        Self::with_rng(config, &mut Rng::seed(0))
    }

    /// Build a FOLO v2 model from `config`, drawing initial weights deterministically from `rng`.
    pub fn with_rng(config: FoloV2Config, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let s = ScalingV2::of(&config);
        let mut init_ctx = Init { rng, bn_eps: config.bn_eps, bn_momentum: config.bn_momentum };
        let i = &mut init_ctx;

        let (c0, c1, c3, c4) = (s.ch(64), s.ch(128), s.ch(256), s.ch(512));
        let (c5, c6, c7, c8) = (s.ch(512), s.ch(512), s.ch(1024), s.ch(1024));
        let c9 = c8;
        let (c13, c16, c19, c22) = (s.ch(512), s.ch(256), s.ch(512), s.ch(1024));
        let c_p2 = c1;
        let c_g = c9 / 2;

        let stage2_layers = (0..s.stage2_m).map(|_| WsQARepConv3x3::new(c1, c1, i)).collect();
        let backbone = BackboneV2 {
            stem: StemV2::new(config.in_channels, c0, config.spd_stem, i),
            down2: i.conv(c0, c1, 3, 2),
            stage2: ModuleList::with_prefix("rep", stage2_layers),
            down3: i.conv(c1, c3, 3, 2),
            stage3: FoloBlock::new(c3, c4, s.bb_m, c3 / 2, i),
            down4: i.conv(c4, c5, 3, 2),
            stage4: FoloBlock::new(c5, c6, s.bb_m, c6 / 2, i),
            down5: i.conv(c6, c7, 3, 2),
            stage5: FoloBlock::new(c7, c8, s.bb_m, c8 / 2, i),
            sppf: Sppf::new(c8, c9, 5, i),
            hyper: HyperContext::new(c9, s.hyper_edges, i),
        };

        let pad_p3 = config.fullpad.then(|| FullPadGate::new(c_g, c16, i)).transpose()?;
        let pad_p4 = config.fullpad.then(|| FullPadGate::new(c_g, c19, i)).transpose()?;
        let neck = NeckV2 {
            td4: FoloBlock::new(c9 + c6, c13, s.neck_m, c13 / 2, i),
            p2_proj: i.conv(4 * c1, c_p2, 1, 1),
            out3: FoloBlock::new(c13 + c4 + c_p2, c16, s.out3_m, c16 / 2, i),
            down3: i.conv(c16, c16, 3, 2),
            out4: FoloBlock::new(c16 + c13, c19, s.neck_m, c19 / 2, i),
            down4: i.conv(c19, c19, 3, 2),
            out5: FoloBlock::new(c19 + c9, c22, s.out5_m, c22 / 2, i),
            pad_p3,
            pad_p4,
        };

        let head = LazyAlignHead::new(
            [c16, c19, c22],
            config.strides,
            s.c_box,
            s.c_cls,
            config.num_classes,
            config.reg_max,
            config.has_o2o(),
            config.iou_head,
            config.mal_bias_prior,
            i,
        )?;
        let head_mode = config.head;
        let classes = (0..config.num_classes).map(|k| format!("class{k}")).collect();
        Ok(FoloV2Model {
            config,
            classes,
            imgsz: 640,
            mode: Mode::new(),
            fused: false,
            head_mode,
            int8: None,
            backbone,
            neck,
            head,
        })
    }

    /// The model configuration.
    pub fn config(&self) -> &FoloV2Config {
        &self.config
    }

    /// Class names (`folo.classes`), length `num_classes`.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Set class names (`folo.classes`); must match `num_classes`.
    pub fn set_classes(&mut self, classes: Vec<String>) -> Result<()> {
        if classes.len() != self.config.num_classes {
            return Err(Error::Invalid(format!(
                "FoloV2Model::set_classes: expected {} class names, got {}",
                self.config.num_classes,
                classes.len()
            )));
        }
        self.classes = classes;
        Ok(())
    }

    /// Default training / inference image size (`folo.imgsz`).
    pub fn imgsz(&self) -> u32 {
        self.imgsz
    }

    /// Set default training / inference image size (`folo.imgsz`).
    pub fn set_imgsz(&mut self, imgsz: u32) {
        self.imgsz = imgsz;
    }

    /// Architecture identifier, e.g. `"folo2-n"`.
    pub fn arch(&self) -> String {
        format!("folo2-{}", self.config.scale)
    }

    /// Number of detection classes.
    pub fn num_classes(&self) -> usize {
        self.config.num_classes
    }

    /// Detection head strides `[8, 16, 32]`.
    pub fn strides(&self) -> [u32; 3] {
        self.config.strides
    }

    /// Anchors for an `h × w` input (8,400 for 640×640).
    pub fn anchors(&self, h: usize, w: usize) -> Anchors {
        make_anchors(h as u32, w as u32, &self.config.strides, 0.5)
    }

    /// Whether the model has the one-to-one (`PeakSuppress`) head branch.
    pub fn has_o2o(&self) -> bool {
        self.head.o2o_cls.is_some()
    }

    /// Active evaluation head mode (`O2m` or `O2o`).
    pub fn head_mode(&self) -> HeadMode {
        self.head_mode
    }

    /// Select evaluation head mode (`O2o` requires a model built with `HeadMode::O2o`).
    pub fn set_head_mode(&mut self, mode: HeadMode) -> Result<()> {
        if mode == HeadMode::O2o && !self.has_o2o() {
            return Err(Error::Invalid(format!("{} has no one-to-one head branch", self.arch())));
        }
        self.head_mode = mode;
        Ok(())
    }

    /// True once [`FoloV2Model::fuse`] has folded all `WS-QARepConv3x3` branches and BatchNorms.
    pub fn is_fused(&self) -> bool {
        self.fused
    }

    /// Collapse every `WS-QARepConv3x3` multi-branch kernel and BatchNorm into fused dense convs and
    /// switch to eval mode. Idempotent.
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

    /// Return a detached, eval-mode, fused clone of this model on the same device and dtype without mutating
    /// `self` (used by `Trainer::validate_epoch` and `FoloV2Model::export` so `WS-QARepConv3x3` multi-branch
    /// convolutions and BatchNorms are folded cleanly in `F32` before inference or export).
    pub fn fused_eval_clone(&self) -> Result<Self> {
        let mut clone = FoloV2Model::new(self.config.clone())?;
        clone.classes = self.classes.clone();
        clone.imgsz = self.imgsz;
        clone.head_mode = self.head_mode;
        clone.head.o2o_detach = self.head.o2o_detach;
        if self.fused {
            clone.fuse()?;
        }
        let dev = self.device();
        let dtype = self.dtype();
        clone.to_dtype(dtype)?;
        clone.to_device(dev)?;
        {
            let _g = no_grad();
            let src = self.named_tensors();
            let dst = clone.named_tensors();
            if src.len() != dst.len() {
                return Err(Error::Invalid(format!(
                    "FoloV2Model::fused_eval_clone: tensor count mismatch ({} vs {})",
                    src.len(),
                    dst.len()
                )));
            }
            for (s, d) in src.iter().zip(dst.iter()) {
                if s.name != d.name {
                    return Err(Error::Invalid(format!(
                        "FoloV2Model::fused_eval_clone: tensor name mismatch (`{}` vs `{}`)",
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
            if dtype != DType::F32 {
                clone.to_dtype(DType::F32)?;
            }
            clone.fuse()?;
            if dtype != DType::F32 {
                clone.to_dtype(dtype)?;
            }
        }
        clone.int8 = self.active_int8().map(|p| Arc::new(quant::Int8Plan::new(p.calibration().clone())));
        clone.eval();
        for p in clone.parameters() {
            p.set_requires_grad(false);
        }
        Ok(clone)
    }

    /// Return a detached, eval-mode, fused CPU `F32` clone without mutating `self`.
    ///
    /// Used by [`FoloV2Model::export`] so unfused `F16` models fold directly in `F32` without rounding back to
    /// `F16` and GPU models download each tensor once instead of round-tripping through device memory.
    pub(crate) fn fused_cpu_f32_clone(&self) -> Result<Self> {
        let mut clone = FoloV2Model::new(self.config.clone())?;
        clone.classes = self.classes.clone();
        clone.imgsz = self.imgsz;
        clone.head_mode = self.head_mode;
        clone.head.o2o_detach = self.head.o2o_detach;
        if self.fused {
            clone.fuse()?;
        }
        {
            let _g = no_grad();
            let src = self.named_tensors();
            let dst = clone.named_tensors();
            if src.len() != dst.len() {
                return Err(Error::Invalid(format!(
                    "FoloV2Model::fused_cpu_f32_clone: tensor count mismatch ({} vs {})",
                    src.len(),
                    dst.len()
                )));
            }
            for (s, d) in src.iter().zip(dst.iter()) {
                if s.name != d.name {
                    return Err(Error::Invalid(format!(
                        "FoloV2Model::fused_cpu_f32_clone: tensor name mismatch (`{}` vs `{}`)",
                        s.name, d.name
                    )));
                }
                let cpu_t = if s.tensor.device().is_cpu() {
                    s.tensor.clone()
                } else {
                    s.tensor.to_device(Device::Cpu)?
                };
                let f32_t = if cpu_t.dtype() == DType::F32 { cpu_t } else { cpu_t.to_dtype(DType::F32)? };
                d.tensor.copy_from(&f32_t)?;
            }
        }
        if !self.fused {
            clone.fuse()?;
        }
        clone.int8 = self.active_int8().map(|p| Arc::new(quant::Int8Plan::new(p.calibration().clone())));
        clone.eval();
        for p in clone.parameters() {
            p.set_requires_grad(false);
        }
        Ok(clone)
    }

    /// Device holding the model weights.
    pub fn device(&self) -> Device {
        self.backbone.stem.conv.conv.weight.device()
    }

    /// Floating dtype of the model weights.
    pub fn dtype(&self) -> DType {
        self.backbone.stem.conv.conv.weight.dtype()
    }

    /// Move all weights and buffers to `device` in place.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        Module::to_device(self, device)
    }

    /// Cast all floating weights and buffers to `dtype` in place. Converting away from `F32` (lossy) clears any
    /// attached int8 calibration plan so a subsequent `F16 → F32` round-trip cannot reuse stale ranges.
    pub fn to_dtype(&mut self, dtype: DType) -> Result<()> {
        if dtype != DType::F32 {
            self.int8 = None;
        }
        Module::to_dtype(self, dtype)
    }

    /// Run the backbone and SPD-PAN neck in int8 on the CPU with the ranges of `cal` (docs/MODEL.md §9.6), or
    /// in `f32` again with `None`. Needs a fused `f32` model on the CPU (matching [`quant::fingerprint_v2`] when
    /// recorded); the head and relocated `FullPadGate`s stay `f32`. Applies to eval-mode CPU forwards and
    /// candidate-first prediction; GPU forwards and training are unchanged.
    pub fn set_int8(&mut self, cal: Option<quant::Calibration>) -> Result<()> {
        if cal.is_some() && (!self.fused || self.dtype() != DType::F32) {
            return Err(Error::Invalid("int8 needs a fused F32 model (FoloV2Model::fuse)".into()));
        }
        if cal.is_some() && !self.device().is_cpu() {
            return Err(Error::Invalid("int8 runs on the CPU: move the model there first".into()));
        }
        if let Some(want) = cal.as_ref().and_then(|c| c.weights.as_deref())
            && want != quant::fingerprint_v2(self)?
        {
            return Err(Error::Invalid(
                "the int8 calibration was measured on other weights: re-calibrate on these".into(),
            ));
        }
        if let Some(c) = &cal {
            let x = Tensor::zeros([1, self.config.in_channels, 64, 64]);
            let _g = no_grad();
            quant::FakeQuant::new(c).features_v2(self, &x).map_err(|e| Error::Invalid(e.to_string()))?;
        }
        self.int8 = cal.map(|c| Arc::new(quant::Int8Plan::new(c)));
        Ok(())
    }

    /// Active, non-invalidated int8 plan (requires `F32` weights).
    pub(crate) fn active_int8(&self) -> Option<&Arc<quant::Int8Plan>> {
        (self.dtype() == DType::F32).then(|| self.int8.as_ref().filter(|p| p.is_valid())).flatten()
    }

    /// The calibration of the int8 plan, if [`FoloV2Model::set_int8`] turned it on and it has not been
    /// invalidated by weight replacement or dtype conversion.
    pub fn int8(&self) -> Option<&quant::Calibration> {
        self.active_int8().map(|p| p.calibration())
    }

    /// Whether CPU int8 inference with the direct `u8` stem (`SPD(2×2) + 1×1` or `3×3 s=2`) is active.
    pub(crate) fn int8_u8_stem_active(&self) -> bool {
        self.active_int8()
            .is_some_and(|p| p.calibration().u8_stem && !p.calibration().is_f32("backbone.stem.conv"))
            && self.device().is_cpu()
            && !self.is_training()
            && self.config.in_channels == 3
    }

    /// Forward pass reading a pre-packed `[b, h, w]` `u32` `[R, G, B, 0]` canvas straight into the direct `u8`
    /// int8 stem (skipping `f32` canvas materialisation).
    pub(crate) fn forward_rgb0_u8(&self, rgb0: &[u32], b: usize, h: usize, w: usize) -> Result<HeadOutputV2> {
        self.check_hw(b, h, w)?;
        let plan = self
            .active_int8()
            .ok_or_else(|| Error::Invalid("forward_rgb0_u8 requires an int8 plan".into()))?;
        folo_core::cpu::in_pool(|| {
            let feats = quant::exec::features_v2_rgb0_u8(self, plan, rgb0, b, h, w)?;
            self.head_from_features(feats, h, w, self.head_mode)
        })
    }

    /// Compute the three head-input feature maps `[P3_head, P4_head, P5_head]`.
    ///
    /// Note that `neck.down3` and `neck.down4` read the raw `out3`/`out4` tensors *before* relocated
    /// `FullPAD-Lite` FiLM gates modulate `P3_head`/`P4_head`, keeping the bottom-up neck free of
    /// float barriers in CPU int8 mode.
    pub fn forward_features(&self, x: &Tensor) -> Result<[Tensor; 3]> {
        self.check_input(x)?;
        if let Some(plan) = self.active_int8()
            && x.device().is_cpu()
            && !self.is_training()
        {
            return quant::exec::features_v2(self, plan, x);
        }
        let b = &self.backbone;
        let n = &self.neck;

        let y = b.stem.forward(x)?;
        let y = b.down2.forward(&y)?;
        let p2 = b.stage2.forward(&y)?;
        let y = b.down3.forward(&p2)?;
        let p3 = b.stage3.forward(&y)?;
        let y = b.down4.forward(&p3)?;
        let p4 = b.stage4.forward(&y)?;
        let y = b.down5.forward(&p4)?;
        let y = b.stage5.forward(&y)?;
        let y = b.sppf.forward(&y)?;
        let (p5, g_scene) = b.hyper.forward_with_scene(&y)?;

        let up11 = p5.upsample_nearest2d(2)?;
        let td4 = n.td4.forward_cat(&[&up11, &p4])?;
        drop(p4);

        let up14 = td4.upsample_nearest2d(2)?;
        let p2_spd = n.p2_proj.forward(&space_to_depth_2x2(&p2)?)?;
        drop(p2);
        let out3_raw = n.out3.forward_cat(&[&up14, &p3, &p2_spd])?;
        drop(p3);
        drop(p2_spd);

        // Bottom-up down3/down4 tap raw neck outputs BEFORE head-only FullPAD FiLM modulation.
        let nd3 = n.down3.forward(&out3_raw)?;
        let out4_raw = n.out4.forward_cat(&[&nd3, &td4])?;
        drop(td4);
        let nd4 = n.down4.forward(&out4_raw)?;
        let out5 = n.out5.forward_cat(&[&nd4, &p5])?;
        drop(p5);

        let p3_head = match &n.pad_p3 {
            Some(gate) => gate.forward(&out3_raw, &g_scene)?,
            None => out3_raw,
        };
        let p4_head = match &n.pad_p4 {
            Some(gate) => gate.forward(&out4_raw, &g_scene)?,
            None => out4_raw,
        };
        Ok([p3_head, p4_head, out5])
    }

    /// Run the full detector forward pass on `[B, in_channels, H, W]` (`H, W` positive multiples of 32).
    pub fn forward(&self, x: &Tensor) -> Result<HeadOutputV2> {
        self.forward_head(x, self.head_mode)
    }

    /// Run the detector forward pass selecting `mode` in eval mode (in training mode, `o2m` and optional
    /// `o2o` both run).
    pub fn forward_head(&self, x: &Tensor, mode: HeadMode) -> Result<HeadOutputV2> {
        if mode == HeadMode::O2o && !self.has_o2o() {
            return Err(Error::Invalid(format!("{} has no one-to-one head branch", self.arch())));
        }
        if x.device().is_cpu() {
            folo_core::cpu::in_pool(|| {
                let (h, w) = self.check_input(x)?;
                let feats = self.forward_features(x)?;
                self.head_from_features(feats, h, w, mode)
            })
        } else {
            let (h, w) = self.check_input(x)?;
            let feats = self.forward_features(x)?;
            self.head_from_features(feats, h, w, mode)
        }
    }

    fn head_from_features(
        &self,
        [p3, p4, p5]: [Tensor; 3],
        h: usize,
        w: usize,
        mode: HeadMode,
    ) -> Result<HeadOutputV2> {
        let feats = [&p3, &p4, &p5];
        let (active_mode, box_raw, cls_raw, iou_raw, o2o) = if self.is_training() {
            let (b_raw, c_raw, i_raw) = self.head.forward(feats)?;
            let o2o = self.head.forward_o2o(feats, Some(&b_raw))?.map(|(b, c, _)| (b, c));
            (HeadMode::O2m, b_raw, c_raw, i_raw, o2o)
        } else if mode == HeadMode::O2o
            && let Some((b_raw, c_raw, i_raw)) = self.head.forward_o2o(feats, None)?
        {
            (HeadMode::O2o, b_raw, c_raw, i_raw, None)
        } else {
            let (b_raw, c_raw, i_raw) = self.head.forward(feats)?;
            (HeadMode::O2m, b_raw, c_raw, i_raw, None)
        };
        Ok(HeadOutputV2 {
            box_raw,
            cls_raw,
            iou_raw,
            mode: active_mode,
            o2o,
            anchors: self.anchors(h, w),
            reg_max: self.head.reg_max,
            num_classes: self.head.num_classes,
        })
    }

    fn check_hw(&self, b: usize, h: usize, w: usize) -> Result<()> {
        let s = self.config.strides[2] as usize;
        if b == 0 || h == 0 || w == 0 || !h.is_multiple_of(s) || !w.is_multiple_of(s) {
            return Err(Error::Invalid(format!(
                "FoloV2Model::forward: input [{b}, {}, {h}, {w}] must be non-empty with H and W multiples of {s}",
                self.config.in_channels
            )));
        }
        Ok(())
    }

    fn check_input(&self, x: &Tensor) -> Result<(usize, usize)> {
        let bad = |why: &str| Error::Invalid(format!("FoloV2Model::forward: input {:?} {why}", x.dims()));
        let (b, c, h, w) = x.dims4().map_err(|_| bad("must be [B, C, H, W]"))?;
        if c != self.config.in_channels {
            return Err(bad(&format!("must have {} channels", self.config.in_channels)));
        }
        self.check_hw(b, h, w)?;
        Ok((h, w))
    }
}

impl Module for FoloV2Model {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("backbone", &self.backbone);
        f("neck", &self.neck);
        f("head", &self.head);
    }

    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}
