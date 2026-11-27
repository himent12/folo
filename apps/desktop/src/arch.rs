//! Canonical model architectures across FOLO v1 (`folo-n..x`) and FOLO v2 (`folo2-n..x`).
//!
//! The wire `folo_api::Arch` enum only encodes the 5 scale buckets (`N`, `S`, `M`, `L`, `X`),
//! while `folo-server` advertises `folo2_arch` in `/api/system`, accepts `"arch": "folo2-*"` on
//! `POST /api/runs` and `POST /api/runs/estimate`, emits `x-folo-arch: folo2-*` on
//! `GET /api/models/{id}`, and emits `--model folo2-*` in `GET /api/runs/{id}/cli`.
//! This module pairs the wire scale with its [`ArchFamily`] so every Desktop screen, filter,
//! estimate, and CLI builder treats `folo-*` and `folo2-*` as first-class citizens.

use crate::api::Arch;

/// Feature flag advertised by `folo-server` in `/api/system` when `folo2-*` training & inference
/// are supported (`crates/folo-server/src/routes/system.rs`).
pub const FEATURE_FOLO2_ARCH: &str = "folo2_arch";

/// Model architecture family: FOLO v1 (`folo-*`) or FOLO v2-Iso (`folo2-*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ArchFamily {
    #[default]
    Folo,
    Folo2,
}

impl ArchFamily {
    /// Long display label (`"FOLO v1"` / `"FOLO v2"`).
    pub const fn label(self) -> &'static str {
        match self {
            ArchFamily::Folo => "FOLO v1",
            ArchFamily::Folo2 => "FOLO v2",
        }
    }

    /// Compact badge / pill label (`"v1"` / `"v2"`).
    pub const fn short_label(self) -> &'static str {
        match self {
            ArchFamily::Folo => "v1",
            ArchFamily::Folo2 => "v2",
        }
    }

    pub const fn is_v2(self) -> bool {
        matches!(self, ArchFamily::Folo2)
    }
}

/// Full architecture identity (`folo-n..x` or `folo2-n..x`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelArch {
    pub family: ArchFamily,
    pub scale: Arch,
}

impl Default for ModelArch {
    fn default() -> Self {
        Self::v1(Arch::default())
    }
}

impl ModelArch {
    #[cfg(test)]
    pub const ALL_V1: [ModelArch; 5] =
        [Self::v1(Arch::N), Self::v1(Arch::S), Self::v1(Arch::M), Self::v1(Arch::L), Self::v1(Arch::X)];

    #[cfg(test)]
    pub const ALL_V2: [ModelArch; 5] =
        [Self::v2(Arch::N), Self::v2(Arch::S), Self::v2(Arch::M), Self::v2(Arch::L), Self::v2(Arch::X)];

    pub const fn new(family: ArchFamily, scale: Arch) -> Self {
        Self { family, scale }
    }

    pub const fn v1(scale: Arch) -> Self {
        Self { family: ArchFamily::Folo, scale }
    }

    #[cfg(test)]
    pub const fn v2(scale: Arch) -> Self {
        Self { family: ArchFamily::Folo2, scale }
    }

    pub const fn is_v2(self) -> bool {
        self.family.is_v2()
    }

    /// Canonical wire & display string (`"folo-n"`..`"folo-x"` or `"folo2-n"`..`"folo2-x"`).
    pub const fn as_str(self) -> &'static str {
        match (self.family, self.scale) {
            (ArchFamily::Folo, Arch::N) => "folo-n",
            (ArchFamily::Folo, Arch::S) => "folo-s",
            (ArchFamily::Folo, Arch::M) => "folo-m",
            (ArchFamily::Folo, Arch::L) => "folo-l",
            (ArchFamily::Folo, Arch::X) => "folo-x",
            (ArchFamily::Folo2, Arch::N) => "folo2-n",
            (ArchFamily::Folo2, Arch::S) => "folo2-s",
            (ArchFamily::Folo2, Arch::M) => "folo2-m",
            (ArchFamily::Folo2, Arch::L) => "folo2-l",
            (ArchFamily::Folo2, Arch::X) => "folo2-x",
        }
    }

    /// Parse `"folo-n"`..`"folo-x"` or `"folo2-n"`..`"folo2-x"` (case-insensitive, trimmed).
    pub fn parse(raw: &str) -> Option<Self> {
        let s = raw.trim().to_ascii_lowercase();
        let (family, rest) = if let Some(r) = s.strip_prefix("folo2-") {
            (ArchFamily::Folo2, r)
        } else {
            let r = s.strip_prefix("folo-")?;
            (ArchFamily::Folo, r)
        };
        let scale = match rest {
            "n" => Arch::N,
            "s" => Arch::S,
            "m" => Arch::M,
            "l" => Arch::L,
            "x" => Arch::X,
            _ => return None,
        };
        Some(Self { family, scale })
    }

    /// Suffix used in suggested run names (`"s"` for `folo-s`, `"folo2-s"` for `folo2-s`),
    /// matching `folo-server` and FOLO Studio.
    pub const fn name_suffix(self) -> &'static str {
        match (self.family, self.scale) {
            (ArchFamily::Folo, Arch::N) => "n",
            (ArchFamily::Folo, Arch::S) => "s",
            (ArchFamily::Folo, Arch::M) => "m",
            (ArchFamily::Folo, Arch::L) => "l",
            (ArchFamily::Folo, Arch::X) => "x",
            (ArchFamily::Folo2, _) => self.as_str(),
        }
    }

    /// JSON `"arch"` override needed for `POST /api/runs` and `POST /api/runs/estimate`.
    /// `None` for v1 (already carried by `folo_api::TrainConfig::arch`); `Some("folo2-*")` for v2.
    pub const fn wire_override(self) -> Option<&'static str> {
        match self.family {
            ArchFamily::Folo => None,
            ArchFamily::Folo2 => Some(self.as_str()),
        }
    }

    /// Approximate parameter count (for tests & VRAM estimation).
    #[cfg(test)]
    pub const fn params_f64(self) -> f64 {
        match (self.family, self.scale) {
            (ArchFamily::Folo, Arch::N) => 2.62e6,
            (ArchFamily::Folo, Arch::S) => 9.46e6,
            (ArchFamily::Folo, Arch::M) => 20.1e6,
            (ArchFamily::Folo, Arch::L) => 25.4e6,
            (ArchFamily::Folo, Arch::X) => 57.0e6,
            (ArchFamily::Folo2, Arch::N) => 2.85e6,
            (ArchFamily::Folo2, Arch::S) => 9.85e6,
            (ArchFamily::Folo2, Arch::M) => 20.8e6,
            (ArchFamily::Folo2, Arch::L) => 26.2e6,
            (ArchFamily::Folo2, Arch::X) => 58.8e6,
        }
    }

    /// Approximate GFLOPs at 640×640 input (scales quadratically with `imgsz`).
    pub const fn gflops_640(self) -> f64 {
        match (self.family, self.scale) {
            (ArchFamily::Folo, Arch::N) => 6.54,
            (ArchFamily::Folo, Arch::S) => 21.59,
            (ArchFamily::Folo, Arch::M) => 68.10,
            (ArchFamily::Folo, Arch::L) => 87.15,
            (ArchFamily::Folo, Arch::X) => 195.27,
            (ArchFamily::Folo2, Arch::N) => 6.9,
            (ArchFamily::Folo2, Arch::S) => 22.4,
            (ArchFamily::Folo2, Arch::M) => 70.5,
            (ArchFamily::Folo2, Arch::L) => 90.0,
            (ArchFamily::Folo2, Arch::X) => 201.5,
        }
    }

    /// Approximate per-image activation memory (MiB) at 640×640 for VRAM estimation.
    pub const fn act_mib_640(self) -> f64 {
        match (self.family, self.scale) {
            (ArchFamily::Folo, Arch::N) => 210.0,
            (ArchFamily::Folo, Arch::S) => 390.0,
            (ArchFamily::Folo, Arch::M) => 740.0,
            (ArchFamily::Folo, Arch::L) => 960.0,
            (ArchFamily::Folo, Arch::X) => 1650.0,
            (ArchFamily::Folo2, Arch::N) => 228.0,
            (ArchFamily::Folo2, Arch::S) => 420.0,
            (ArchFamily::Folo2, Arch::M) => 790.0,
            (ArchFamily::Folo2, Arch::L) => 1030.0,
            (ArchFamily::Folo2, Arch::X) => 1750.0,
        }
    }

    /// Extract `--model <arch>` or `--arch <arch>` (and `=` forms) from a `folo train` CLI argv slice.
    pub fn from_cli_argv(argv: &[String]) -> Option<Self> {
        let mut iter = argv.iter();
        while let Some(arg) = iter.next() {
            if arg == "--model" || arg == "--arch" {
                return iter.next().and_then(|v| Self::parse(v));
            }
            if let Some(rest) = arg.strip_prefix("--model=").or_else(|| arg.strip_prefix("--arch=")) {
                return Self::parse(rest);
            }
        }
        None
    }

    /// Parse an optional raw override string, falling back to `folo-<scale>`.
    pub fn from_raw_or(raw: Option<&str>, fallback: Arch) -> Self {
        raw.and_then(Self::parse).unwrap_or_else(|| Self::v1(fallback))
    }

    /// Whether `query` (already lowercased) matches this architecture or its family label.
    pub fn matches_query(self, q_lower: &str) -> bool {
        self.as_str().contains(q_lower)
            || self.family.short_label() == q_lower
            || self.family.label().to_ascii_lowercase().contains(q_lower)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format_round_trip_for_v1_and_v2() {
        for arch in ModelArch::ALL_V1.into_iter().chain(ModelArch::ALL_V2) {
            assert_eq!(ModelArch::parse(arch.as_str()), Some(arch));
            assert_eq!(ModelArch::parse(&arch.as_str().to_uppercase()), Some(arch));
            assert_eq!(arch.is_v2(), arch.as_str().starts_with("folo2-"));
            assert_eq!(arch.wire_override().is_some(), arch.is_v2());
        }
        assert_eq!(ModelArch::parse("folo3-n"), None);
        assert_eq!(ModelArch::parse("yolo26n"), None);
        assert_eq!(ModelArch::v1(Arch::S).name_suffix(), "s");
        assert_eq!(ModelArch::v2(Arch::S).name_suffix(), "folo2-s");
    }

    #[test]
    fn from_cli_argv_reads_both_flag_forms() {
        let split: Vec<String> =
            ["folo", "train", "--model", "folo2-m", "--epochs", "50"].into_iter().map(String::from).collect();
        assert_eq!(ModelArch::from_cli_argv(&split), Some(ModelArch::v2(Arch::M)));

        let eq: Vec<String> = ["folo", "train", "--model=folo2-x"].into_iter().map(String::from).collect();
        assert_eq!(ModelArch::from_cli_argv(&eq), Some(ModelArch::v2(Arch::X)));

        let none: Vec<String> =
            ["folo", "train", "--weights", "a.folo"].into_iter().map(String::from).collect();
        assert_eq!(ModelArch::from_cli_argv(&none), None);
    }

    #[test]
    fn v2_specs_scale_monotonically_and_exceed_v1_baseline() {
        for (v1, v2) in ModelArch::ALL_V1.into_iter().zip(ModelArch::ALL_V2) {
            assert!(v2.params_f64() > v1.params_f64(), "{v2:?} > {v1:?}");
            assert!(v2.gflops_640() > v1.gflops_640(), "{v2:?} > {v1:?}");
            assert!(v2.act_mib_640() > v1.act_mib_640(), "{v2:?} > {v1:?}");
        }
    }
}
