//! Shared numeric validation bounds for CLI arguments.
//!
//! Runtime argument validation (`infer`, `quantize`, `dataset`, `label`) and `folo describe` both read from this
//! module so accepted numeric ranges never drift between validation errors, `--help`, and `folo describe --json`.

use serde_json::{Map, Value, json};

/// A numeric bound value (unsigned integer or finite float) serialized without spurious `.0` on integers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundValue {
    /// An unsigned integer bound.
    Int(u64),
    /// A floating-point bound.
    Float(f64),
}

impl BoundValue {
    fn to_json(self) -> Value {
        match self {
            Self::Int(v) => json!(v),
            Self::Float(v) => json!(v),
        }
    }
}

/// Validated numeric bounds for a CLI argument, shared between runtime checks and `folo describe`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericBounds {
    /// Lower bound, if any.
    pub min: Option<BoundValue>,
    /// Upper bound, if any.
    pub max: Option<BoundValue>,
    /// Whether `min` is inclusive (`>=`) rather than strictly exclusive (`>`).
    pub min_inclusive: bool,
    /// Whether `max` is inclusive (`<=`) rather than strictly exclusive (`<`).
    pub max_inclusive: bool,
    /// Required integer divisor, if any (e.g. `32` for network stride).
    pub multiple_of: Option<u64>,
}

impl NumericBounds {
    /// Inclusive integer range `[min..=max]`.
    pub const fn int_range(min: u64, max: u64) -> Self {
        Self {
            min: Some(BoundValue::Int(min)),
            max: Some(BoundValue::Int(max)),
            min_inclusive: true,
            max_inclusive: true,
            multiple_of: None,
        }
    }

    /// Lower-bounded integer `[>= min]`.
    pub const fn int_min(min: u64) -> Self {
        Self {
            min: Some(BoundValue::Int(min)),
            max: None,
            min_inclusive: true,
            max_inclusive: true,
            multiple_of: None,
        }
    }

    /// Inclusive integer range `[min..=max]` restricted to multiples of `step`.
    pub const fn int_range_multiple(min: u64, max: u64, step: u64) -> Self {
        Self {
            min: Some(BoundValue::Int(min)),
            max: Some(BoundValue::Int(max)),
            min_inclusive: true,
            max_inclusive: true,
            multiple_of: Some(step),
        }
    }

    /// Lower-bounded integer `[>= min]` restricted to multiples of `step`.
    pub const fn int_min_multiple(min: u64, step: u64) -> Self {
        Self {
            min: Some(BoundValue::Int(min)),
            max: None,
            min_inclusive: true,
            max_inclusive: true,
            multiple_of: Some(step),
        }
    }

    /// Floating-point range with configurable endpoint inclusivity.
    pub const fn float_range(min: f64, min_inclusive: bool, max: f64, max_inclusive: bool) -> Self {
        Self {
            min: Some(BoundValue::Float(min)),
            max: Some(BoundValue::Float(max)),
            min_inclusive,
            max_inclusive,
            multiple_of: None,
        }
    }

    /// Lower-bounded finite float `[>= min]`.
    pub const fn float_min(min: f64, min_inclusive: bool) -> Self {
        Self {
            min: Some(BoundValue::Float(min)),
            max: None,
            min_inclusive,
            max_inclusive: true,
            multiple_of: None,
        }
    }

    /// Whether an unsigned integer value satisfies these bounds.
    pub fn contains_u64(self, v: u64) -> bool {
        if let Some(BoundValue::Int(min)) = self.min {
            let ok = if self.min_inclusive { v >= min } else { v > min };
            if !ok {
                return false;
            }
        }
        if let Some(BoundValue::Int(max)) = self.max {
            let ok = if self.max_inclusive { v <= max } else { v < max };
            if !ok {
                return false;
            }
        }
        if let Some(step) = self.multiple_of
            && (step == 0 || !v.is_multiple_of(step))
        {
            return false;
        }
        true
    }

    /// Whether a finite floating-point value satisfies these bounds (`NaN` and infinities always fail).
    pub fn contains_f64(self, v: f64) -> bool {
        if !v.is_finite() {
            return false;
        }
        if let Some(min) = self.min {
            let lo = match min {
                BoundValue::Int(n) => n as f64,
                BoundValue::Float(f) => f,
            };
            let ok = if self.min_inclusive { v >= lo } else { v > lo };
            if !ok {
                return false;
            }
        }
        if let Some(max) = self.max {
            let hi = match max {
                BoundValue::Int(n) => n as f64,
                BoundValue::Float(f) => f,
            };
            let ok = if self.max_inclusive { v <= hi } else { v < hi };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Serializes the active bounds as a JSON object (`minimum` / `exclusive_minimum`,
    /// `maximum` / `exclusive_maximum`, `multiple_of`).
    pub fn to_json(self) -> Value {
        let mut map = Map::new();
        if let Some(min) = self.min {
            let key = if self.min_inclusive { "minimum" } else { "exclusive_minimum" };
            map.insert(key.into(), min.to_json());
        }
        if let Some(max) = self.max {
            let key = if self.max_inclusive { "maximum" } else { "exclusive_maximum" };
            map.insert(key.into(), max.to_json());
        }
        if let Some(step) = self.multiple_of {
            map.insert("multiple_of".into(), json!(step));
        }
        Value::Object(map)
    }
}

/// Unit interval `[0..=1]` (confidence thresholds, IoU thresholds, probabilities).
pub const UNIT_INTERVAL: NumericBounds = NumericBounds::float_range(0.0, true, 1.0, true);
/// Positive integer `[>= 1]` (batch sizes, max detections, image counts, 1-based offsets).
pub const POSITIVE_INT: NumericBounds = NumericBounds::int_min(1);
/// Non-negative finite float `[>= 0]`.
pub const NON_NEGATIVE_FLOAT: NumericBounds = NumericBounds::float_min(0.0, true);

/// Network stride divisor for square input sizes.
pub const IMGSZ_STRIDE: u64 = 32;
/// Maximum square input size for inference, ONNX export, quantization, and synthetic generation.
pub const INFER_IMGSZ_MAX: u32 = 8192;

/// Square input size for `predict`, `export`, and `quantize`: `32..=8192`, multiple of 32.
pub const INFER_IMGSZ: NumericBounds =
    NumericBounds::int_range_multiple(IMGSZ_STRIDE, INFER_IMGSZ_MAX as u64, IMGSZ_STRIDE);
/// Square input size for `val`: `>= 32`, multiple of 32 (no 8192 upper cap, matching `val::check_config`).
pub const VAL_IMGSZ: NumericBounds = NumericBounds::int_min_multiple(IMGSZ_STRIDE, IMGSZ_STRIDE);

/// `folo train` bounds (matching `folo_vision::TrainConfig::validate` and `folo_data::AugmentConfig::validate`).
pub const TRAIN_EPOCHS: NumericBounds = NumericBounds::int_range(1, 10_000);
/// `folo train --batch`.
pub const TRAIN_BATCH: NumericBounds = NumericBounds::int_range(1, 1024);
/// `folo train --imgsz`.
pub const TRAIN_IMGSZ: NumericBounds = NumericBounds::int_range_multiple(64, 4096, IMGSZ_STRIDE);
/// `folo train --lr0`.
pub const TRAIN_LR0: NumericBounds = NumericBounds::float_range(0.0, false, 10.0, false);
/// `folo train --lrf`.
pub const TRAIN_LRF: NumericBounds = NumericBounds::float_range(0.0, false, 1.0, true);
/// `folo train --scale`.
pub const TRAIN_SCALE: NumericBounds = NumericBounds::float_range(0.0, true, 1.0, false);

/// `folo quantize --tail` (open interval `(0, 0.5)`).
pub const QUANTIZE_TAIL: NumericBounds = NumericBounds::float_range(0.0, false, 0.5, false);

/// `folo dataset synth --images`.
pub const SYNTH_IMAGES: NumericBounds = NumericBounds::int_range(1, 1_000_000);
/// `folo dataset synth --imgsz`.
pub const SYNTH_IMGSZ: NumericBounds = NumericBounds::int_range(32, INFER_IMGSZ_MAX as u64);

/// `folo label start --batch`.
pub const LABEL_BATCH: NumericBounds = NumericBounds::int_range(1, 16);
/// `folo label start --parallel`.
pub const LABEL_PARALLEL: NumericBounds = NumericBounds::int_range(1, 8);
/// `folo label status --log`.
pub const LABEL_LOG: NumericBounds = NumericBounds::int_range(0, 50);
/// `folo label images --limit`.
pub const LABEL_LIMIT: NumericBounds = NumericBounds::int_range(1, 1000);
/// `folo label view --size`.
pub const LABEL_VIEW_SIZE: NumericBounds = NumericBounds::int_range(256, 1568);
/// `folo label export --val-fraction`.
pub const LABEL_VAL_FRACTION: NumericBounds = NumericBounds::float_range(0.0, true, 0.5, true);

/// `folo bench create --max-images`.
pub const BENCH_MAX_IMAGES: NumericBounds = NumericBounds::int_range(1, 5000);

/// Validated numeric bounds for CLI arguments, keyed by `(command_path, arg_id)`.
pub const ARG_BOUNDS: &[(&str, &str, NumericBounds)] = &[
    // `folo train`
    ("folo train", "epochs", TRAIN_EPOCHS),
    ("folo train", "batch", TRAIN_BATCH),
    ("folo train", "imgsz", TRAIN_IMGSZ),
    ("folo train", "lr0", TRAIN_LR0),
    ("folo train", "lrf", TRAIN_LRF),
    ("folo train", "warmup_epochs", NON_NEGATIVE_FLOAT),
    ("folo train", "mosaic", UNIT_INTERVAL),
    ("folo train", "mixup", UNIT_INTERVAL),
    ("folo train", "hsv", NON_NEGATIVE_FLOAT),
    ("folo train", "fliplr", UNIT_INTERVAL),
    ("folo train", "scale", TRAIN_SCALE),
    ("folo train", "translate", NON_NEGATIVE_FLOAT),
    // `folo val`
    ("folo val", "batch", POSITIVE_INT),
    ("folo val", "imgsz", VAL_IMGSZ),
    ("folo val", "conf", UNIT_INTERVAL),
    ("folo val", "iou", UNIT_INTERVAL),
    ("folo val", "max_det", POSITIVE_INT),
    ("folo val", "fusion_iou", UNIT_INTERVAL),
    // `folo predict`
    ("folo predict", "conf", UNIT_INTERVAL),
    ("folo predict", "iou", UNIT_INTERVAL),
    ("folo predict", "max_det", POSITIVE_INT),
    ("folo predict", "fusion_iou", UNIT_INTERVAL),
    ("folo predict", "imgsz", INFER_IMGSZ),
    // `folo export`
    ("folo export", "imgsz", INFER_IMGSZ),
    ("folo export", "batch", POSITIVE_INT),
    ("folo export", "max_det", POSITIVE_INT),
    // `folo quantize`
    ("folo quantize", "images", POSITIVE_INT),
    ("folo quantize", "tail", QUANTIZE_TAIL),
    ("folo quantize", "imgsz", INFER_IMGSZ),
    ("folo quantize", "batch", POSITIVE_INT),
    // `folo dataset synth`
    ("folo dataset synth", "images", SYNTH_IMAGES),
    ("folo dataset synth", "imgsz", SYNTH_IMGSZ),
    ("folo dataset synth", "val_frac", UNIT_INTERVAL),
    // `folo label`
    ("folo label start", "batch", LABEL_BATCH),
    ("folo label start", "parallel", LABEL_PARALLEL),
    ("folo label status", "log", LABEL_LOG),
    ("folo label images", "limit", LABEL_LIMIT),
    ("folo label view", "size", LABEL_VIEW_SIZE),
    ("folo label view", "from", POSITIVE_INT),
    ("folo label export", "val_fraction", LABEL_VAL_FRACTION),
    // `folo bench`
    ("folo bench create", "max_images", BENCH_MAX_IMAGES),
    ("folo bench run", "max_images", POSITIVE_INT),
];

/// Looks up the validated numeric bounds for `(cmd_path, arg_id)`, if any.
pub fn for_arg(cmd_path: &str, arg_id: &str) -> Option<NumericBounds> {
    ARG_BOUNDS.iter().find(|&&(p, a, _)| p == cmd_path && a == arg_id).map(|&(_, _, b)| b)
}
