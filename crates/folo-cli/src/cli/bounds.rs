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
        if let Some(min) = self.min {
            let ok = match min {
                BoundValue::Int(lo) => {
                    if self.min_inclusive {
                        v >= lo
                    } else {
                        v > lo
                    }
                }
                BoundValue::Float(lo) => {
                    if !lo.is_finite() {
                        false
                    } else if lo < 0.0 {
                        true
                    } else if lo == 0.0 {
                        self.min_inclusive || v > 0
                    } else {
                        let floor = lo.floor();
                        if floor >= (u64::MAX as f64) {
                            false
                        } else {
                            let k = floor as u64;
                            if lo == floor && self.min_inclusive { v >= k } else { v > k }
                        }
                    }
                }
            };
            if !ok {
                return false;
            }
        }
        if let Some(max) = self.max {
            let ok = match max {
                BoundValue::Int(hi) => {
                    if self.max_inclusive {
                        v <= hi
                    } else {
                        v < hi
                    }
                }
                BoundValue::Float(hi) => {
                    if !hi.is_finite() || hi < 0.0 {
                        false
                    } else if hi == 0.0 {
                        self.max_inclusive && v == 0
                    } else {
                        let floor = hi.floor();
                        if floor >= (u64::MAX as f64) {
                            true
                        } else {
                            let k = floor as u64;
                            if hi == floor && !self.max_inclusive { v < k } else { v <= k }
                        }
                    }
                }
            };
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
            if !lo.is_finite() {
                return false;
            }
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
            if !hi.is_finite() {
                return false;
            }
            let ok = if self.max_inclusive { v <= hi } else { v < hi };
            if !ok {
                return false;
            }
        }
        if let Some(step) = self.multiple_of
            && (step == 0 || (v % (step as f64)) != 0.0)
        {
            return false;
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
/// Strictly positive finite float `(> 0)` (e.g. TTA scale multipliers).
pub const POSITIVE_FLOAT: NumericBounds = NumericBounds::float_min(0.0, false);
/// Half-open unit interval `[0, 1)` (SAHI tile overlap ratio).
pub const TILE_OVERLAP: NumericBounds = NumericBounds::float_range(0.0, true, 1.0, false);

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
/// `folo train --auto-batch-target`.
pub const TRAIN_AUTO_BATCH_TARGET: NumericBounds = NumericBounds::float_range(
    folo_vision::train::MIN_AUTO_BATCH_UTILIZATION,
    true,
    folo_vision::train::MAX_AUTO_BATCH_UTILIZATION,
    true,
);
/// `folo train --imgsz`.
pub const TRAIN_IMGSZ: NumericBounds = NumericBounds::int_range_multiple(64, 4096, IMGSZ_STRIDE);
/// `folo train --lr0`.
pub const TRAIN_LR0: NumericBounds = NumericBounds::float_range(0.0, false, 10.0, false);
/// `folo train --lrf`.
pub const TRAIN_LRF: NumericBounds = NumericBounds::float_range(0.0, false, 1.0, true);
/// `folo train --scale`.
pub const TRAIN_SCALE: NumericBounds = NumericBounds::float_range(0.0, true, 1.0, false);
/// `folo train --cloud --max-hours`.
pub const CLOUD_MAX_HOURS: NumericBounds = NumericBounds::float_range(0.25, true, 720.0, true);
/// `folo train --cloud --disk-gb`.
pub const CLOUD_DISK_GB: NumericBounds = NumericBounds::int_range(10, 4096);
/// `folo train --cloud --max-preempt-retries`.
pub const CLOUD_MAX_PREEMPT_RETRIES: NumericBounds =
    NumericBounds::int_range(0, folo_server::store::MAX_PREEMPT_RETRIES as u64);

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

/// `folo serve --port` (`0..=65535`; `0` picks a free port).
pub const SERVE_PORT: NumericBounds = NumericBounds::int_range(0, u16::MAX as u64);

/// `folo dataset check --dhash-threshold` (64-bit Hamming distance `0..=64`).
pub const DHASH_THRESHOLD: NumericBounds = NumericBounds::int_range(0, 64);

/// `folo bench create --max-images` / `--images`.
pub const BENCH_MAX_IMAGES: NumericBounds = NumericBounds::int_range(1, 5000);
/// `folo bench create --generated --imgsz`.
pub const BENCH_GENERATED_IMGSZ: NumericBounds = NumericBounds::int_range(
    folo_server::bench::GENERATED_MIN_IMGSZ as u64,
    folo_server::bench::GENERATED_MAX_IMGSZ as u64,
);
/// `folo bench create --seed` (exact JS safe integer limit `2^53 - 1`, matching `folo_server::bench::MAX_SEED`).
pub const BENCH_SEED: NumericBounds = NumericBounds::int_range(0, (1u64 << 53) - 1);

/// Validated numeric bounds for CLI arguments, keyed by `(command_path, arg_id)`.
pub const ARG_BOUNDS: &[(&str, &str, NumericBounds)] = &[
    // `folo serve`
    ("folo serve", "port", SERVE_PORT),
    // `folo train`
    ("folo train", "epochs", TRAIN_EPOCHS),
    ("folo train", "auto_batch_target", TRAIN_AUTO_BATCH_TARGET),
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
    ("folo train", "max_hours", CLOUD_MAX_HOURS),
    ("folo train", "disk_gb", CLOUD_DISK_GB),
    ("folo train", "max_preempt_retries", CLOUD_MAX_PREEMPT_RETRIES),
    // `folo val`
    ("folo val", "batch", POSITIVE_INT),
    ("folo val", "imgsz", VAL_IMGSZ),
    ("folo val", "conf", UNIT_INTERVAL),
    ("folo val", "iou", UNIT_INTERVAL),
    ("folo val", "max_det", POSITIVE_INT),
    ("folo val", "fusion_iou", UNIT_INTERVAL),
    ("folo val", "worst", POSITIVE_INT),
    ("folo val", "diag_conf", UNIT_INTERVAL),
    ("folo val", "diag_tp_iou", UNIT_INTERVAL),
    ("folo val", "diag_bg_iou", UNIT_INTERVAL),
    // `folo predict`
    ("folo predict", "conf", UNIT_INTERVAL),
    ("folo predict", "iou", UNIT_INTERVAL),
    ("folo predict", "max_det", POSITIVE_INT),
    ("folo predict", "fusion_iou", UNIT_INTERVAL),
    ("folo predict", "imgsz", INFER_IMGSZ),
    ("folo predict", "tile", POSITIVE_INT),
    ("folo predict", "tile_overlap", TILE_OVERLAP),
    ("folo predict", "tile_batch", POSITIVE_INT),
    ("folo predict", "tta_scales", POSITIVE_FLOAT),
    ("folo predict", "track_high_thresh", UNIT_INTERVAL),
    ("folo predict", "track_low_thresh", UNIT_INTERVAL),
    ("folo predict", "track_new_thresh", UNIT_INTERVAL),
    ("folo predict", "track_match_iou", UNIT_INTERVAL),
    ("folo predict", "track_min_hits", POSITIVE_INT),
    // `folo export`
    ("folo export", "imgsz", INFER_IMGSZ),
    ("folo export", "batch", POSITIVE_INT),
    ("folo export", "max_det", POSITIVE_INT),
    // `folo quantize`
    ("folo quantize", "images", POSITIVE_INT),
    ("folo quantize", "tail", QUANTIZE_TAIL),
    ("folo quantize", "imgsz", INFER_IMGSZ),
    ("folo quantize", "batch", POSITIVE_INT),
    ("folo quantize", "max_f32_macs", UNIT_INTERVAL),
    // `folo dataset check`
    ("folo dataset check", "dhash_threshold", DHASH_THRESHOLD),
    // `folo dataset synth`
    ("folo dataset synth", "images", SYNTH_IMAGES),
    ("folo dataset synth", "imgsz", SYNTH_IMGSZ),
    ("folo dataset synth", "val_frac", UNIT_INTERVAL),
    // `folo dataset split`
    ("folo dataset split", "val", UNIT_INTERVAL),
    ("folo dataset split", "test", UNIT_INTERVAL),
    // `folo dataset merge`
    ("folo dataset merge", "val", UNIT_INTERVAL),
    ("folo dataset merge", "test", UNIT_INTERVAL),
    // `folo dataset filter`
    ("folo dataset filter", "min_box_area", UNIT_INTERVAL),
    ("folo dataset filter", "max_box_area", UNIT_INTERVAL),
    ("folo dataset filter", "min_box_width", NON_NEGATIVE_FLOAT),
    ("folo dataset filter", "max_box_width", NON_NEGATIVE_FLOAT),
    ("folo dataset filter", "min_box_height", NON_NEGATIVE_FLOAT),
    ("folo dataset filter", "max_box_height", NON_NEGATIVE_FLOAT),
    ("folo dataset filter", "min_image_width", POSITIVE_INT),
    ("folo dataset filter", "max_image_width", POSITIVE_INT),
    ("folo dataset filter", "min_image_height", POSITIVE_INT),
    ("folo dataset filter", "max_image_height", POSITIVE_INT),
    // `folo label`
    ("folo label start", "batch", LABEL_BATCH),
    ("folo label start", "parallel", LABEL_PARALLEL),
    ("folo label status", "log", LABEL_LOG),
    ("folo label images", "limit", LABEL_LIMIT),
    ("folo label view", "size", LABEL_VIEW_SIZE),
    ("folo label view", "from", POSITIVE_INT),
    ("folo label set", "remove", POSITIVE_INT),
    ("folo label export", "val_fraction", LABEL_VAL_FRACTION),
    // `folo bench`
    ("folo bench create", "images", BENCH_MAX_IMAGES),
    ("folo bench create", "imgsz", BENCH_GENERATED_IMGSZ),
    ("folo bench create", "max_images", BENCH_MAX_IMAGES),
    ("folo bench create", "seed", BENCH_SEED),
    ("folo bench run", "max_images", POSITIVE_INT),
];

/// Looks up the validated numeric bounds for `(cmd_path, arg_id)`, if any.
pub fn for_arg(cmd_path: &str, arg_id: &str) -> Option<NumericBounds> {
    ARG_BOUNDS.iter().find(|&&(p, a, _)| p == cmd_path && a == arg_id).map(|&(_, _, b)| b)
}

/// Peer arguments unconditionally required whenever an argument is supplied, keyed by `(command_path, arg_id)`.
///
/// `clap` enforces these via `#[arg(requires = ...)]`, but `clap_builder` does not expose a public `requires`
/// getter for reflection, so `folo describe` reads this table and unit tests verify every entry against the real
/// clap parser.
pub const ARG_REQUIRES: &[(&str, &str, &[&str])] = &[
    ("folo serve", "tls_cert", &["tls_key"]),
    ("folo serve", "tls_key", &["tls_cert"]),
    ("folo predict", "tile_overlap", &["tile"]),
    ("folo predict", "no_tile_full_image", &["tile"]),
    ("folo predict", "tile_batch", &["tile"]),
    ("folo predict", "tile_merge", &["tile"]),
    ("folo predict", "tta_scales", &["tta"]),
    ("folo predict", "no_tta_hflip", &["tta"]),
    ("folo predict", "track_high_thresh", &["track"]),
    ("folo predict", "track_low_thresh", &["track"]),
    ("folo predict", "track_new_thresh", &["track"]),
    ("folo predict", "track_match_iou", &["track"]),
    ("folo predict", "track_buffer", &["track"]),
    ("folo predict", "track_min_hits", &["track"]),
    ("folo predict", "track_class_agnostic", &["track"]),
    ("folo bench create", "images", &["generated"]),
    ("folo bench create", "classes", &["generated"]),
    ("folo bench create", "imgsz", &["generated"]),
    ("folo bench create", "vary_aspect", &["generated"]),
    ("folo bench compare", "min_delta", &["fail_on_regression"]),
    ("folo train", "offer", &["cloud"]),
    ("folo train", "max_hours", &["cloud"]),
    ("folo train", "disk_gb", &["cloud"]),
    ("folo train", "max_preempt_retries", &["cloud"]),
];

/// Peer arguments required when an env-backed argument is passed explicitly on the command line (enforced at
/// runtime rather than by clap so ambient environment variables like `FOLO_DATA_DIR` / `FOLO_SERVER` do not break
/// unrelated local runs).
pub const ARG_CLI_REQUIRES: &[(&str, &str, &[&str])] = &[
    ("folo train", "data_dir", &["cloud"]),
    ("folo train", "server", &["cloud"]),
    ("folo train", "token", &["cloud"]),
    ("folo train", "tls_ca", &["cloud"]),
];

/// Effective default values for conditional `Option<T>` arguments that cannot carry clap `default_value_t`
/// without tripping `requires = ...` or losing presence detection, keyed by `(command_path, arg_id)`.
pub const ARG_EFFECTIVE_DEFAULTS: &[(&str, &str, &str)] = &[
    ("folo train", "auto_batch_target", "0.80"),
    ("folo train", "max_hours", "12"),
    ("folo train", "max_preempt_retries", "3"),
    ("folo bench compare", "min_delta", "0.0"),
];

/// Looks up the peer argument ids required whenever `(cmd_path, arg_id)` is supplied on the command line.
pub fn required_args(cmd_path: &str, arg_id: &str) -> &'static [&'static str] {
    ARG_REQUIRES
        .iter()
        .chain(ARG_CLI_REQUIRES.iter())
        .find(|&&(p, a, _)| p == cmd_path && a == arg_id)
        .map_or(&[], |&(_, _, reqs)| reqs)
}

/// Looks up the effective default value for a conditional argument, if one is declared.
pub fn effective_default(cmd_path: &str, arg_id: &str) -> Option<&'static str> {
    ARG_EFFECTIVE_DEFAULTS.iter().find(|&&(p, a, _)| p == cmd_path && a == arg_id).map(|&(_, _, val)| val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_f64_handles_inclusive_exclusive_nonfinite_and_multiple_of() {
        // Inclusive [0..=1]
        assert!(UNIT_INTERVAL.contains_f64(0.0));
        assert!(UNIT_INTERVAL.contains_f64(0.5));
        assert!(UNIT_INTERVAL.contains_f64(1.0));
        assert!(!UNIT_INTERVAL.contains_f64(-0.001));
        assert!(!UNIT_INTERVAL.contains_f64(1.001));
        assert!(!UNIT_INTERVAL.contains_f64(f64::NAN));
        assert!(!UNIT_INTERVAL.contains_f64(f64::INFINITY));
        assert!(!UNIT_INTERVAL.contains_f64(f64::NEG_INFINITY));

        // Open (0, 0.5) and (0, 10)
        assert!(!QUANTIZE_TAIL.contains_f64(0.0));
        assert!(QUANTIZE_TAIL.contains_f64(1e-4));
        assert!(!QUANTIZE_TAIL.contains_f64(0.5));
        assert!(!QUANTIZE_TAIL.contains_f64(f64::NAN));
        assert!(!TRAIN_LR0.contains_f64(0.0));
        assert!(TRAIN_LR0.contains_f64(0.01));
        assert!(!TRAIN_LR0.contains_f64(10.0));

        // Half-open (0, 1] and [0, 1)
        assert!(!TRAIN_LRF.contains_f64(0.0));
        assert!(TRAIN_LRF.contains_f64(1.0));
        assert!(TRAIN_SCALE.contains_f64(0.0));
        assert!(!TRAIN_SCALE.contains_f64(1.0));

        // Lower-bounded >= 0
        assert!(NON_NEGATIVE_FLOAT.contains_f64(0.0));
        assert!(NON_NEGATIVE_FLOAT.contains_f64(100.0));
        assert!(!NON_NEGATIVE_FLOAT.contains_f64(-1e-9));
        assert!(!NON_NEGATIVE_FLOAT.contains_f64(f64::INFINITY));
        assert!(!NON_NEGATIVE_FLOAT.contains_f64(f64::NAN));

        // Integer multiple_of enforced for floats too (and step 0 rejected)
        assert!(INFER_IMGSZ.contains_f64(32.0));
        assert!(INFER_IMGSZ.contains_f64(640.0));
        assert!(!INFER_IMGSZ.contains_f64(33.0));
        assert!(!INFER_IMGSZ.contains_f64(32.5));
        assert!(!NumericBounds::int_min_multiple(1, 0).contains_f64(32.0));
    }

    #[test]
    fn contains_u64_enforces_int_float_and_multiple_of_bounds() {
        assert!(!POSITIVE_INT.contains_u64(0));
        assert!(POSITIVE_INT.contains_u64(1));
        assert!(INFER_IMGSZ.contains_u64(32));
        assert!(INFER_IMGSZ.contains_u64(8192));
        assert!(!INFER_IMGSZ.contains_u64(0));
        assert!(!INFER_IMGSZ.contains_u64(33));
        assert!(!INFER_IMGSZ.contains_u64(8224));
        assert!(!NumericBounds::int_min_multiple(1, 0).contains_u64(32));

        assert!(BENCH_SEED.contains_u64(0));
        assert!(BENCH_SEED.contains_u64((1u64 << 53) - 1));
        assert!(!BENCH_SEED.contains_u64(1u64 << 53));

        // Float bounds are enforced on u64 rather than silently bypassed
        assert!(UNIT_INTERVAL.contains_u64(0));
        assert!(UNIT_INTERVAL.contains_u64(1));
        assert!(!UNIT_INTERVAL.contains_u64(2));
        assert!(!UNIT_INTERVAL.contains_u64(42));
        assert!(!QUANTIZE_TAIL.contains_u64(0));
        assert!(!QUANTIZE_TAIL.contains_u64(1));
        assert!(TRAIN_SCALE.contains_u64(0));
        assert!(!TRAIN_SCALE.contains_u64(1));
        assert!(!TRAIN_LR0.contains_u64(0));
        assert!(TRAIN_LR0.contains_u64(1));
        assert!(TRAIN_LR0.contains_u64(9));
        assert!(!TRAIN_LR0.contains_u64(10));
        assert!(!NumericBounds::float_range(f64::NAN, true, 1.0, true).contains_u64(0));
    }

    #[test]
    fn to_json_serializes_active_keys_only() {
        assert_eq!(UNIT_INTERVAL.to_json(), json!({ "minimum": 0.0, "maximum": 1.0 }));
        assert_eq!(POSITIVE_INT.to_json(), json!({ "minimum": 1 }));
        assert_eq!(INFER_IMGSZ.to_json(), json!({ "minimum": 32, "maximum": 8192, "multiple_of": 32 }));
        assert_eq!(VAL_IMGSZ.to_json(), json!({ "minimum": 32, "multiple_of": 32 }));
        assert_eq!(TRAIN_LR0.to_json(), json!({ "exclusive_minimum": 0.0, "exclusive_maximum": 10.0 }));
        assert_eq!(TRAIN_LRF.to_json(), json!({ "exclusive_minimum": 0.0, "maximum": 1.0 }));
        assert_eq!(TRAIN_SCALE.to_json(), json!({ "minimum": 0.0, "exclusive_maximum": 1.0 }));
    }
}
