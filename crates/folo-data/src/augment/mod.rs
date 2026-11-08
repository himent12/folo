//! Box-aware augmentations driven by an explicit, seeded [`Rng`](crate::Rng).
//!
//! Every function here keeps boxes consistent with pixels: geometric ops transform boxes by their corners, clip
//! them to the output and drop the ones that became too small or too thin ([`BoxFilter`]). [`TrainTransform`]
//! chains them the way YOLO training does: mosaic → random affine → mixup → HSV → flips.

mod geometry;
mod hsv;
mod mosaic;
mod pipeline;

pub use geometry::{
    BoxFilter, affine_sample, flip_lr, flip_ud, random_affine, random_affine_matrix, resize_long_side,
    transform_boxes,
};
pub use hsv::{apply_hsv, random_hsv};
pub use mosaic::{mixup, mosaic4};
pub use pipeline::{TrainTransform, Transformed};

use serde::{Deserialize, Serialize};

/// Augmentation parameters. The JSON shape matches `TrainConfig.augment` in `docs/API.md`; fields missing from the
/// JSON take the defaults below (the YOLO defaults).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AugmentConfig {
    /// Probability of building a 4-image mosaic (default 1.0).
    pub mosaic: f32,
    /// Probability of mixing in a second (augmented) sample (default 0.0).
    pub mixup: f32,
    /// HSV gains `[hue, saturation, value]` (default `[0.015, 0.7, 0.4]`). Hue is a fraction of a full turn.
    pub hsv: [f32; 3],
    /// Probability of a horizontal flip (default 0.5).
    pub fliplr: f32,
    /// Probability of a vertical flip (default 0.0).
    pub flipud: f32,
    /// Random scale range `1 ± scale` (default 0.5).
    pub scale: f32,
    /// Random translation as a fraction of the output size, `± translate` (default 0.1).
    pub translate: f32,
    /// Random rotation range in degrees, `± degrees` (default 0.0).
    pub degrees: f32,
    /// Random shear range in degrees, `± shear` (default 0.0).
    pub shear: f32,
    /// Turn mosaic (and mixup) off for the last `close_mosaic` epochs (default 10).
    pub close_mosaic: u32,
    /// Drop boxes narrower or shorter than this many output pixels (default 2.0).
    pub min_box_px: f32,
    /// Drop boxes that kept less than this fraction of their area after clipping (default 0.1).
    pub min_area_ratio: f32,
    /// Drop boxes with a larger aspect ratio than this (default 100).
    pub max_aspect: f32,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        AugmentConfig {
            mosaic: 1.0,
            mixup: 0.0,
            hsv: [0.015, 0.7, 0.4],
            fliplr: 0.5,
            flipud: 0.0,
            scale: 0.5,
            translate: 0.1,
            degrees: 0.0,
            shear: 0.0,
            close_mosaic: 10,
            min_box_px: 2.0,
            min_area_ratio: 0.1,
            max_aspect: 100.0,
        }
    }
}

impl AugmentConfig {
    /// No augmentation at all (images are only letterboxed).
    pub fn none() -> Self {
        AugmentConfig {
            mosaic: 0.0,
            mixup: 0.0,
            hsv: [0.0; 3],
            fliplr: 0.0,
            flipud: 0.0,
            scale: 0.0,
            translate: 0.0,
            degrees: 0.0,
            shear: 0.0,
            close_mosaic: 0,
            ..AugmentConfig::default()
        }
    }

    /// Whether mosaic is still on in `epoch` (0-based) of a run of `epochs` epochs. With `epochs == 0` (unknown
    /// length) it is always on.
    pub fn mosaic_active(&self, epoch: u32, epochs: u32) -> bool {
        epochs == 0 || epoch.saturating_add(self.close_mosaic) < epochs
    }

    /// The box filter these parameters describe.
    pub fn box_filter(&self) -> BoxFilter {
        BoxFilter {
            min_px: self.min_box_px,
            min_area_ratio: self.min_area_ratio,
            max_aspect: self.max_aspect,
        }
    }

    /// Checks ranges; returns a message for the first bad value.
    pub fn validate(&self) -> Result<(), String> {
        let probs = [
            ("mosaic", self.mosaic),
            ("mixup", self.mixup),
            ("fliplr", self.fliplr),
            ("flipud", self.flipud),
        ];
        for (name, p) in probs {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("augment.{name} must be a probability in [0, 1], got {p}"));
            }
        }
        let ranges = [
            ("hsv[0]", self.hsv[0]),
            ("hsv[1]", self.hsv[1]),
            ("hsv[2]", self.hsv[2]),
            ("translate", self.translate),
            ("degrees", self.degrees),
            ("shear", self.shear),
            ("min_box_px", self.min_box_px),
            ("min_area_ratio", self.min_area_ratio),
        ];
        for (name, v) in ranges {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("augment.{name} must be finite and >= 0, got {v}"));
            }
        }
        if !(self.scale.is_finite() && (0.0..1.0).contains(&self.scale)) {
            return Err(format!("augment.scale must be in [0, 1), got {}", self.scale));
        }
        if self.max_aspect.is_nan() || self.max_aspect <= 1.0 {
            return Err(format!("augment.max_aspect must be > 1, got {}", self.max_aspect));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_json_shape() {
        let json = r#"{ "mosaic": 1.0, "mixup": 0.0, "hsv": [0.015, 0.7, 0.4], "fliplr": 0.5, "scale": 0.5,
                        "translate": 0.1, "close_mosaic": 10 }"#;
        let cfg: AugmentConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg, AugmentConfig::default());
        assert!(cfg.validate().is_ok());
        assert!(cfg.mosaic_active(89, 100));
        assert!(!cfg.mosaic_active(90, 100));
        assert!(cfg.mosaic_active(1000, 0));
        assert!(AugmentConfig { mosaic: 2.0, ..cfg }.validate().is_err());
    }
}
