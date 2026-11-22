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

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor};
use folo_nn::{ConvBnAct, Error, Forward, Mode, Module, ModuleList, Result};

use crate::decode::{Anchors, make_anchors};
use crate::model::blocks::{Fuse, Init, Sppf};
use crate::model::{HeadMode, HeadOutput};

pub use blocks::{
    FoloBlock, FullPadGate, HyperContext, QARepBranches, StemV2, WsQARepConv3x3, space_to_depth_2x2,
};
pub use config::{FoloV2Config, MAL_CLS_BIAS_PRIOR, ScalingV2};
pub use head::{BoxBranchV2, ClsBranchV2, HeadLevelV2, LazyAlignHead, O2oClsLevel, PeakSuppress};

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
}

/// A FOLO v2 detector (`folo2-<scale>`).
pub struct FoloV2Model {
    config: FoloV2Config,
    mode: Mode,
    fused: bool,
    head_mode: HeadMode,
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
        Ok(FoloV2Model { config, mode: Mode::new(), fused: false, head_mode, backbone, neck, head })
    }

    /// The model configuration.
    pub fn config(&self) -> &FoloV2Config {
        &self.config
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

    /// Cast all floating weights and buffers to `dtype` in place.
    pub fn to_dtype(&mut self, dtype: DType) -> Result<()> {
        Module::to_dtype(self, dtype)
    }

    /// Compute the three head-input feature maps `[P3_head, P4_head, P5_head]`.
    ///
    /// Note that `neck.down3` and `neck.down4` read the raw `out3`/`out4` tensors *before* relocated
    /// `FullPAD-Lite` FiLM gates modulate `P3_head`/`P4_head`, keeping the bottom-up neck free of
    /// float barriers in CPU int8 mode.
    pub fn forward_features(&self, x: &Tensor) -> Result<[Tensor; 3]> {
        self.check_input(x)?;
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
        let (h, w) = self.check_input(x)?;
        let [p3, p4, p5] = self.forward_features(x)?;
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

    fn check_input(&self, x: &Tensor) -> Result<(usize, usize)> {
        let bad = |why: &str| Error::Invalid(format!("FoloV2Model::forward: input {:?} {why}", x.dims()));
        let (b, c, h, w) = x.dims4().map_err(|_| bad("must be [B, C, H, W]"))?;
        if c != self.config.in_channels {
            return Err(bad(&format!("must have {} channels", self.config.in_channels)));
        }
        let s = self.config.strides[2] as usize;
        if b == 0 || h == 0 || w == 0 || !h.is_multiple_of(s) || !w.is_multiple_of(s) {
            return Err(bad(&format!("must be non-empty with H and W multiples of {s}")));
        }
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
