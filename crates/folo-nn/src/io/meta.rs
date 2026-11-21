//! `.folo` metadata (`__metadata__`, docs/FORMAT.md) and the v1 model config (docs/MODEL.md §3).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// `folo.format`.
pub const FORMAT_VERSION: &str = "1";

/// Keys only checkpoints carry. They are dropped when a checkpoint's metadata is reused for a weights file.
pub(crate) const CHECKPOINT_KEYS: [&str; 9] = [
    "folo.train",
    "folo.optim",
    "folo.sched",
    "folo.ema",
    "folo.rng",
    "folo.epoch",
    "folo.best",
    "folo.amp",
    "folo.nonfinite_streak",
];

/// `folo.kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    /// Inference / transfer weights: the model namespace only.
    Weights,
    /// Resumable training state.
    Checkpoint,
}

impl FileKind {
    /// `"weights"` or `"checkpoint"`.
    pub fn as_str(self) -> &'static str {
        match self {
            FileKind::Weights => "weights",
            FileKind::Checkpoint => "checkpoint",
        }
    }
}

/// The floating dtype of a file's parameters and buffers (`folo.dtype`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDType {
    /// 32-bit (training, default).
    F32,
    /// 16-bit (exported inference files).
    F16,
}

impl FloatDType {
    /// `"F32"` or `"F16"`.
    pub fn as_str(self) -> &'static str {
        match self {
            FloatDType::F32 => "F32",
            FloatDType::F16 => "F16",
        }
    }

    /// The core dtype.
    pub fn core(self) -> folo_core::DType {
        match self {
            FloatDType::F32 => folo_core::DType::F32,
            FloatDType::F16 => folo_core::DType::F16,
        }
    }
}

/// `(depth, width, max_channels)` of a scale (docs/MODEL.md §3).
pub fn scale_row(scale: &str) -> Option<(f64, f64, u32)> {
    Some(match scale {
        "n" => (0.50, 0.25, 1024),
        "s" => (0.50, 0.50, 1024),
        "m" => (0.50, 1.00, 512),
        "l" => (1.00, 1.00, 512),
        "x" => (1.00, 1.50, 512),
        _ => return None,
    })
}

fn default_bn_eps() -> f64 {
    crate::BN_EPS
}

fn default_bn_momentum() -> f64 {
    crate::BN_MOMENTUM
}

/// The model config, `folo.config` (docs/MODEL.md §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// `"folo"`.
    pub family: String,
    /// Spec version, 1.
    pub spec: u32,
    /// `n s m l x`.
    pub scale: String,
    /// Depth multiplier.
    pub depth: f64,
    /// Width multiplier.
    pub width: f64,
    /// Channel cap before the width multiplier.
    pub max_channels: u32,
    /// Number of classes (`len(folo.classes)`).
    pub num_classes: usize,
    /// Input channels, 3.
    pub in_channels: usize,
    /// Head strides, `[8, 16, 32]`.
    pub strides: Vec<u32>,
    /// DFL bins, 16.
    pub reg_max: u32,
    /// Head type: `"o2m"` (one-to-many + NMS, v1's), or `"o2o"`: both branches (docs/MODEL.md §6.6, trained with
    /// dual assignment), predicting with the one-to-one branch and no NMS. [`ModelConfig::has_o2o`].
    pub head: String,
    /// BatchNorm ε.
    #[serde(default = "default_bn_eps")]
    pub bn_eps: f64,
    /// BatchNorm momentum.
    #[serde(default = "default_bn_momentum")]
    pub bn_momentum: f64,
}

impl ModelConfig {
    /// The v1 config of a scale with `num_classes` classes.
    pub fn new(scale: &str, num_classes: usize) -> Result<Self> {
        let (depth, width, max_channels) = scale_row(scale)
            .ok_or_else(|| Error::Metadata(format!("unknown scale `{scale}` (expected n, s, m, l or x)")))?;
        let c = ModelConfig {
            family: "folo".into(),
            spec: 1,
            scale: scale.into(),
            depth,
            width,
            max_channels,
            num_classes,
            in_channels: 3,
            strides: vec![8, 16, 32],
            reg_max: 16,
            head: "o2m".into(),
            bn_eps: crate::BN_EPS,
            bn_momentum: crate::BN_MOMENTUM,
        };
        c.validate()?;
        Ok(c)
    }

    /// Whether the model has the one-to-one (NMS-free) branch, `head = "o2o"` (docs/MODEL.md §6.6).
    pub fn has_o2o(&self) -> bool {
        self.head == "o2o"
    }

    /// The v1 rules of docs/MODEL.md §3.
    pub fn validate(&self) -> Result<()> {
        let mut bad = Vec::new();
        if self.family != "folo" {
            bad.push(format!("family is `{}`, expected `folo`", self.family));
        }
        if self.spec != 1 {
            bad.push(format!("spec {} is not supported (this reader implements spec 1)", self.spec));
        }
        match scale_row(&self.scale) {
            None => bad.push(format!("unknown scale `{}`", self.scale)),
            Some((d, w, m)) => {
                if self.depth != d || self.width != w || self.max_channels != m {
                    bad.push(format!(
                        "scale `{}` requires depth {d}, width {w}, max_channels {m}; got {}, {}, {}",
                        self.scale, self.depth, self.width, self.max_channels
                    ));
                }
            }
        }
        if self.num_classes == 0 {
            bad.push("num_classes must be at least 1".into());
        }
        if self.in_channels != 3 {
            bad.push(format!("in_channels {} (v1 requires 3)", self.in_channels));
        }
        if self.strides != [8, 16, 32] {
            bad.push(format!("strides {:?} (v1 requires [8, 16, 32])", self.strides));
        }
        if self.reg_max != 16 {
            bad.push(format!("reg_max {} (v1 requires 16)", self.reg_max));
        }
        match self.head.as_str() {
            "o2m" | "o2o" => {}
            h => bad.push(format!("unknown head `{h}` (expected `o2m` or `o2o`)")),
        }
        if !(self.bn_eps > 0.0 && self.bn_eps.is_finite())
            || !(self.bn_momentum > 0.0 && self.bn_momentum < 1.0)
        {
            bad.push(format!("invalid BatchNorm eps {} / momentum {}", self.bn_eps, self.bn_momentum));
        }
        if bad.is_empty() { Ok(()) } else { Err(Error::Metadata(format!("folo.config: {}", bad.join("; ")))) }
    }
}

/// `folo.metrics`: best validation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// mAP@0.5.
    pub map50: f64,
    /// mAP@0.5:0.95.
    pub map50_95: f64,
    /// Precision at `best_f1_conf`.
    pub precision: f64,
    /// Recall at `best_f1_conf`.
    pub recall: f64,
    /// The confidence threshold precision and recall are taken at (the best class-mean F1, docs/METRICS.md).
    /// Absent in files written before it; never written as `null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_f1_conf: Option<f64>,
}

/// The metadata every `.folo` file has, typed. Keys this struct does not model (checkpoint contracts, unknown
/// `folo.*` keys, foreign keys) are kept verbatim in `extra`, so a rewrite preserves them.
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    /// `folo.kind`.
    pub kind: FileKind,
    /// `folo.arch`, informational (e.g. `folo-n`).
    pub arch: String,
    /// `folo.config`, validated.
    pub config: ModelConfig,
    /// `folo.classes`: class names, index = class id.
    pub classes: Vec<String>,
    /// `folo.imgsz`.
    pub imgsz: u32,
    /// `folo.fused`.
    pub fused: bool,
    /// `folo.dtype`.
    pub dtype: FloatDType,
    /// `folo.metrics`.
    pub metrics: Option<Metrics>,
    /// `folo.created` (RFC 3339).
    pub created: Option<String>,
    /// `folo.version`.
    pub version: Option<String>,
    /// Every other key.
    pub extra: BTreeMap<String, String>,
}

impl Metadata {
    /// Metadata for a file of `kind` with `folo.arch = "folo-<scale>"`, `F32`, unfused, and this crate's
    /// version. Checks `num_classes == classes.len()`.
    pub fn new(kind: FileKind, config: ModelConfig, classes: Vec<String>, imgsz: u32) -> Result<Self> {
        let m = Metadata {
            kind,
            arch: format!("folo-{}", config.scale),
            config,
            classes,
            imgsz,
            fused: false,
            dtype: FloatDType::F32,
            metrics: None,
            created: None,
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            extra: BTreeMap::new(),
        };
        m.validate()?;
        Ok(m)
    }

    fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.config.num_classes != self.classes.len() {
            return Err(Error::Metadata(format!(
                "folo.config.num_classes is {} but folo.classes has {} names",
                self.config.num_classes,
                self.classes.len()
            )));
        }
        if self.imgsz == 0 {
            return Err(Error::Metadata("folo.imgsz must be positive".into()));
        }
        Ok(())
    }

    /// Parse and validate a `__metadata__` map. A map without `folo.format` is not a `.folo` file: the error
    /// points to the import path.
    pub fn from_map(map: &BTreeMap<String, String>) -> Result<Self> {
        let bad = |m: String| Error::Metadata(m);
        let get = |k: &str| map.get(k).ok_or_else(|| bad(format!("missing required key `{k}`")));
        let format = map.get("folo.format").ok_or_else(|| {
            bad("no `folo.format`: this is a plain safetensors file; load it with ModelFile::import \
                 (`folo import --config <json> --classes <json>`), supplying the config and class names"
                .into())
        })?;
        if format.split('.').next() != Some(FORMAT_VERSION) {
            return Err(bad(format!(
                "folo.format `{format}` is not supported (this reader reads version {FORMAT_VERSION})"
            )));
        }
        let kind = match get("folo.kind")?.as_str() {
            "weights" => FileKind::Weights,
            "checkpoint" => FileKind::Checkpoint,
            k => return Err(bad(format!("unknown folo.kind `{k}`"))),
        };
        let config: ModelConfig =
            serde_json::from_str(get("folo.config")?).map_err(|e| bad(format!("folo.config: {e}")))?;
        let classes: Vec<String> =
            serde_json::from_str(get("folo.classes")?).map_err(|e| bad(format!("folo.classes: {e}")))?;
        let imgsz =
            get("folo.imgsz")?.parse().map_err(|_| bad("folo.imgsz is not a positive integer".into()))?;
        let fused = match map.get("folo.fused").map(String::as_str) {
            None | Some("false") => false,
            Some("true") => true,
            Some(v) => return Err(bad(format!("folo.fused must be \"true\" or \"false\", got `{v}`"))),
        };
        let dtype = match map.get("folo.dtype").map(String::as_str) {
            None | Some("F32") => FloatDType::F32,
            Some("F16") => FloatDType::F16,
            Some(v) => return Err(bad(format!("folo.dtype must be F32 or F16, got `{v}`"))),
        };
        let metrics = map
            .get("folo.metrics")
            .map(|s| serde_json::from_str(s).map_err(|e| bad(format!("folo.metrics: {e}"))))
            .transpose()?;
        const TYPED: [&str; 10] = [
            "folo.format",
            "folo.kind",
            "folo.arch",
            "folo.config",
            "folo.classes",
            "folo.imgsz",
            "folo.fused",
            "folo.dtype",
            "folo.metrics",
            "folo.created",
        ];
        let extra = map
            .iter()
            .filter(|(k, _)| !TYPED.contains(&k.as_str()) && k.as_str() != "folo.version")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let m = Metadata {
            kind,
            arch: get("folo.arch")?.clone(),
            config,
            classes,
            imgsz,
            fused,
            dtype,
            metrics,
            created: map.get("folo.created").cloned(),
            version: map.get("folo.version").cloned(),
            extra,
        };
        m.validate()?;
        Ok(m)
    }

    /// The `__metadata__` map to write.
    pub fn to_map(&self) -> BTreeMap<String, String> {
        let mut m = self.extra.clone();
        m.insert("folo.format".into(), FORMAT_VERSION.into());
        m.insert("folo.kind".into(), self.kind.as_str().into());
        m.insert("folo.arch".into(), self.arch.clone());
        m.insert("folo.config".into(), json(&self.config));
        m.insert("folo.classes".into(), json(&self.classes));
        m.insert("folo.imgsz".into(), self.imgsz.to_string());
        if self.fused {
            m.insert("folo.fused".into(), "true".into());
        }
        m.insert("folo.dtype".into(), self.dtype.as_str().into());
        if let Some(x) = &self.metrics {
            m.insert("folo.metrics".into(), json(x));
        }
        if let Some(c) = &self.created {
            m.insert("folo.created".into(), c.clone());
        }
        if let Some(v) = &self.version {
            m.insert("folo.version".into(), v.clone());
        }
        m
    }

    /// This metadata as a `weights` file's: kind `weights`, `folo.version` stamped with this writer's version,
    /// checkpoint-only keys dropped, unknown keys kept.
    pub fn for_weights(&self) -> Metadata {
        let mut m = self.clone();
        m.kind = FileKind::Weights;
        m.version = Some(env!("CARGO_PKG_VERSION").to_string());
        m.extra.retain(|k, _| !CHECKPOINT_KEYS.contains(&k.as_str()));
        m
    }
}

fn json<T: Serialize + ?Sized>(v: &T) -> String {
    serde_json::to_string(v).expect("metadata serialises")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rules() {
        let c = ModelConfig::new("n", 80).unwrap();
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<ModelConfig>(&s).unwrap(), c);
        let mut bad = c.clone();
        bad.width = 0.5;
        assert!(bad.validate().is_err());
        bad = c.clone();
        bad.head = "dual".into();
        assert!(bad.validate().unwrap_err().to_string().contains("unknown head `dual`"));
        bad.head = "o2o".into();
        assert!(bad.validate().is_ok() && bad.has_o2o() && !c.has_o2o());
        assert!(serde_json::from_str::<ModelConfig>(&s.replace("\"family\"", "\"x\":1,\"family\"")).is_err());
        assert!(Metadata::new(FileKind::Weights, c.clone(), vec!["a".into()], 640).is_err());
        let m = Metadata::new(
            FileKind::Weights,
            ModelConfig::new("s", 2).unwrap(),
            vec!["a".into(), "b".into()],
            320,
        )
        .unwrap();
        let mut map = m.to_map();
        map.insert("folo.future".into(), "x".into());
        let back = Metadata::from_map(&map).unwrap();
        assert_eq!(back.extra.get("folo.future").map(String::as_str), Some("x"));
        assert_eq!(back.to_map(), map);
        map.insert("folo.format".into(), "2".into());
        assert!(Metadata::from_map(&map).unwrap_err().to_string().contains("not supported"));
        assert!(Metadata::from_map(&BTreeMap::new()).unwrap_err().to_string().contains("import"));
    }

    #[test]
    fn for_weights_drops_all_checkpoint_keys_and_keeps_unknown() {
        let mut m = Metadata::new(
            FileKind::Checkpoint,
            ModelConfig::new("n", 2).unwrap(),
            vec!["a".into(), "b".into()],
            64,
        )
        .unwrap();
        m.created = Some("2026-11-21T00:00:00Z".into());
        m.version = Some("0.0.1".into());
        for k in [
            "folo.train",
            "folo.optim",
            "folo.sched",
            "folo.ema",
            "folo.rng",
            "folo.epoch",
            "folo.best",
            "folo.amp",
            "folo.nonfinite_streak",
        ] {
            m.extra.insert(k.into(), "1".into());
        }
        m.extra.insert("folo.custom".into(), "keep".into());

        let w = m.for_weights();
        assert_eq!(w.kind, FileKind::Weights);
        assert_eq!(w.version.as_deref(), Some(env!("CARGO_PKG_VERSION")));
        assert_eq!(w.created.as_deref(), Some("2026-11-21T00:00:00Z"));
        assert_eq!(w.extra.get("folo.custom").map(String::as_str), Some("keep"));
        for k in ["folo.amp", "folo.nonfinite_streak", "folo.train", "folo.optim", "folo.best"] {
            assert!(!w.extra.contains_key(k), "checkpoint key `{k}` leaked into weights metadata");
        }
    }
}
