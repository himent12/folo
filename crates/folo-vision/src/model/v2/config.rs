//! Configuration and `ceil16`-aligned channel scaling for the FOLO v2 detector family (`folo2-n/s/m/l/x`).

use folo_nn::{BN_EPS, BN_MOMENTUM, Error, Result};
use serde::{Deserialize, Serialize};

use crate::model::HeadMode;

/// Focal classification bias prior `ln(0.01 / 0.99) = -4.59512`, coupled strictly to MAL-trained runs.
pub const MAL_CLS_BIAS_PRIOR: f32 = -4.595_12;

/// Configuration of a [`super::FoloV2Model`].
///
/// Defaults build the locked `FOLO v2-Iso` architecture:
/// - `ceil16`-aligned widths across every scale (`n`, `s`, `m`, `l`, `x`),
/// - shared `c_box = 48, c_cls = 64` across `n` and `s` for zero-adapter Box + Cls CrossKD,
/// - lossless `SPD-P2→P3` cross-scale neck injection (keeping 8,400 anchors at 640×640),
/// - `HyperContext` (`HyperACE-Lite`) at P5 + relocated `FullPAD-Lite` FiLM on `f32` head inputs only,
/// - `WS-QARepConv3x3` inside `FoloBlock` and `LazyAlignHead`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoloV2Config {
    /// Scale name: `"n"`, `"s"`, `"m"`, `"l"`, or `"x"`.
    pub scale: String,
    /// Number of detection classes.
    pub num_classes: usize,
    /// Input image channels (default `3`).
    pub in_channels: usize,
    /// DFL bins per box side (default `16`, giving `4 * reg_max = 64` box channels).
    pub reg_max: usize,
    /// Detection head strides (default `[8, 16, 32]`).
    pub strides: [u32; 3],
    /// Detection head mode (`O2m` or dual `O2m + O2o` with `PeakSuppress`).
    pub head: HeadMode,
    /// Enable the opt-in separate `iou_out: Conv1×1(c_box → 1)` localization-quality head branch.
    /// Keeps `box.out` strictly `4 * reg_max = 64` channels (`% 16 == 0`).
    pub iou_head: bool,
    /// Use lossless `SPD(2×2) + Conv1×1(4·in_channels → c0)` stem (`true`) or standard `Conv3×3 s=2`
    /// stem (`false`, pairing with CPU `u8-stem`).
    pub spd_stem: bool,
    /// High-capacity `v2-Plus` tier (`+1` `WS-QARepConv3x3` in `stage3/4/5` and `neck.out3` for `n`/`s`).
    pub plus: bool,
    /// Enable relocated `FullPAD-Lite` global-context FiLM gates on `P3` and `P4` head inputs only
    /// (applied after `neck.down3`/`neck.down4` taps so the neck stays contiguous int8).
    pub fullpad: bool,
    /// Initialise `cls.out.bias` to [`MAL_CLS_BIAS_PRIOR`] (`-4.595`, `p0 = 0.01`, for MAL runs) instead
    /// of FOLO v1's per-level `cls_bias_prior` (for plain-BCE runs).
    pub mal_bias_prior: bool,
    /// BatchNorm epsilon (default `1e-3`).
    pub bn_eps: f64,
    /// BatchNorm momentum (default `0.03`).
    pub bn_momentum: f64,
}

impl FoloV2Config {
    /// Build a default `FOLO v2-Iso` config for `scale` (`"n" | "s" | "m" | "l" | "x"`) and `num_classes`.
    pub fn new(scale: &str, num_classes: usize) -> Result<Self> {
        let cfg = FoloV2Config {
            scale: scale.to_ascii_lowercase(),
            num_classes,
            in_channels: 3,
            reg_max: 16,
            strides: [8, 16, 32],
            head: HeadMode::O2m,
            iou_head: false,
            spd_stem: true,
            plus: false,
            fullpad: true,
            mal_bias_prior: false,
            bn_eps: BN_EPS,
            bn_momentum: BN_MOMENTUM,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Enable or disable the one-to-one (`PeakSuppress`) head branch.
    pub fn with_head(mut self, head: HeadMode) -> Self {
        self.head = head;
        self
    }

    /// Enable or disable the separate `iou_out: Conv1×1(c_box → 1)` branch.
    pub fn with_iou_head(mut self, on: bool) -> Self {
        self.iou_head = on;
        self
    }

    /// Select `SPD-Stem` (`true`) or standard `Conv3×3 s=2` stem (`false`).
    pub fn with_spd_stem(mut self, on: bool) -> Self {
        self.spd_stem = on;
        self
    }

    /// Select `v2-Plus` (`true`) or `v2-Iso` (`false`).
    pub fn with_plus(mut self, on: bool) -> Self {
        self.plus = on;
        self
    }

    /// Enable or disable relocated `FullPAD-Lite` FiLM gates on `P3`/`P4` head inputs.
    pub fn with_fullpad(mut self, on: bool) -> Self {
        self.fullpad = on;
        self
    }

    /// Use MAL focal bias prior (`-4.595`) instead of the plain-BCE prior.
    pub fn with_mal_bias_prior(mut self, on: bool) -> Self {
        self.mal_bias_prior = on;
        self
    }

    /// Whether the config builds the dual `o2m + o2o` head.
    pub fn has_o2o(&self) -> bool {
        self.head == HeadMode::O2o
    }

    /// Validate all fields before constructing a model.
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.scale.as_str(), "n" | "s" | "m" | "l" | "x") {
            return Err(Error::Invalid(format!(
                "FoloV2Config: unknown scale `{}` (expected n, s, m, l or x)",
                self.scale
            )));
        }
        if self.num_classes == 0 {
            return Err(Error::Invalid("FoloV2Config: num_classes must be positive".into()));
        }
        if self.in_channels == 0 {
            return Err(Error::Invalid("FoloV2Config: in_channels must be positive".into()));
        }
        if self.reg_max == 0 {
            return Err(Error::Invalid("FoloV2Config: reg_max must be positive".into()));
        }
        if self.strides != [8, 16, 32] {
            return Err(Error::Invalid(format!(
                "FoloV2Config: unsupported strides {:?} (expected [8, 16, 32])",
                self.strides
            )));
        }
        Ok(())
    }
}

/// Resolved `ceil16`-aligned widths and depths for a [`FoloV2Config`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingV2 {
    /// Width multiplier.
    pub width: f64,
    /// Pre-multiplier channel cap.
    pub max_channels: usize,
    /// Number of `WS-QARepConv3x3` layers in `backbone.stage2` (P2/4).
    pub stage2_m: usize,
    /// Number of `WS-QARepConv3x3` layers in `backbone.stage3/4/5`.
    pub bb_m: usize,
    /// Number of `WS-QARepConv3x3` layers in `neck.td4` and `neck.out4`.
    pub neck_m: usize,
    /// Number of `WS-QARepConv3x3` layers in `neck.out3`.
    pub out3_m: usize,
    /// Number of `WS-QARepConv3x3` layers in `neck.out5`.
    pub out5_m: usize,
    /// Hyperedges `M` in `HyperContext` at P5.
    pub hyper_edges: usize,
    /// Box branch hidden width (`% 16 == 0`; locked to `48` on both `n` and `s`).
    pub c_box: usize,
    /// Class branch hidden width (`% 16 == 0`; locked to `64` on both `n` and `s`).
    pub c_cls: usize,
}

impl ScalingV2 {
    /// Resolve scale parameters for a validated [`FoloV2Config`].
    pub fn of(cfg: &FoloV2Config) -> Self {
        let plus = usize::from(cfg.plus);
        match cfg.scale.as_str() {
            "n" => ScalingV2 {
                width: 0.25,
                max_channels: 1024,
                stage2_m: 1,
                bb_m: 2 + plus,
                neck_m: 1,
                out3_m: 1 + plus,
                out5_m: 2,
                hyper_edges: 16,
                c_box: 48,
                c_cls: 64,
            },
            "s" => ScalingV2 {
                width: 0.50,
                max_channels: 1024,
                stage2_m: 1,
                bb_m: 2 + plus,
                neck_m: 1,
                out3_m: 1 + plus,
                out5_m: 2,
                hyper_edges: 32,
                c_box: 48,
                c_cls: 64,
            },
            "m" => ScalingV2 {
                width: 0.75,
                max_channels: 768,
                stage2_m: 2,
                bb_m: 3 + plus,
                neck_m: 2,
                out3_m: 2 + plus,
                out5_m: 3,
                hyper_edges: 32,
                c_box: 64,
                c_cls: 128,
            },
            "l" => ScalingV2 {
                width: 1.00,
                max_channels: 768,
                stage2_m: 2,
                bb_m: 4 + plus,
                neck_m: 2,
                out3_m: 2 + plus,
                out5_m: 3,
                hyper_edges: 48,
                c_box: 64,
                c_cls: 160,
            },
            "x" => ScalingV2 {
                width: 1.25,
                max_channels: 1024,
                stage2_m: 2,
                bb_m: 4 + plus,
                neck_m: 2,
                out3_m: 2 + plus,
                out5_m: 4,
                hyper_edges: 64,
                c_box: 80,
                c_cls: 192,
            },
            other => panic!("FoloV2Config::validate must be called first (got `{other}`)"),
        }
    }

    /// `ceil16(min(base, max_channels) * width)` — strictly 16-channel aligned for `16×16×16` tensor cores.
    pub fn ch(&self, base: usize) -> usize {
        ((base.min(self.max_channels) as f64 * self.width / 16.0).ceil() as usize).max(1) * 16
    }
}
