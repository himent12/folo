//! Vision benchmarks (docs/BENCH.md): the engine behind `/api/bench`.
//!
//! A benchmark is a frozen set of labeled images, a plan of stress conditions and a seed; an entry is one
//! contestant's detections on it. This crate is everything about that which is not storage or scheduling:
//!
//! - [`conditions`]: the twelve stress conditions ([`Condition`]), each a deterministic change to an image that
//!   keeps its size, at severities 1 to [`folo_api::BENCH_SEVERITIES`]; [`conditions::render`] turns an image
//!   file into the canonical JPEG every contestant is graded on.
//! - [`plan`]: the catalog, the default plan and plan validation.
//! - [`score`]: every number of docs/BENCH.md's *Definitions*: accuracy with its bootstrap interval, robustness,
//!   self-labeling, calibration, localization and the total, returned as the [`folo_api`] wire types.
//! - `pixels.rs` (private): the pixel kernels behind the conditions (sRGB tables, seeded noise, convolution,
//!   resampling).
//!
//! It is a pure library: no files, no threads of its own (rayon data parallelism only), no server types. AP comes
//! from [`folo_vision::Evaluator`] and [`folo_vision::cocoeval::CocoEval`]; there is no second AP here. Every
//! result is a function of its inputs alone: the same bytes and numbers on every run, machine and thread count.

pub mod conditions;
mod pixels;
pub mod plan;
pub mod score;

pub use conditions::{Condition, Rendered};

/// The version of what this crate computes: the pixels of every condition and every definition of the score.
/// Bump it when a condition's pixels or a definition of the score changes: it goes into every benchmark's
/// fingerprint, so entries graded by different engines never compare as equals. `tests/golden.rs` pins the
/// pixels, the canonical JPEG and a score, and fails on any change that is not marked by a bump here.
pub const ENGINE_VERSION: u32 = 1;
pub use plan::{catalog, check_plan, default_plan, variants};
pub use score::{Evaluation, Map, Outcome, Sample};

/// What can go wrong when making a variant of an image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A severity outside `1..=BENCH_SEVERITIES`.
    Severity(u8),
    /// The image file could not be decoded (the decoder's message).
    Decode(String),
    /// The image could not be encoded as JPEG (the encoder's message; e.g. a side over 65,535 px).
    Encode(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Severity(s) => {
                write!(f, "severity {s} is outside 1..={}", folo_api::BENCH_SEVERITIES)
            }
            Error::Decode(m) => write!(f, "cannot decode the image: {m}"),
            Error::Encode(m) => write!(f, "cannot encode the image: {m}"),
        }
    }
}

impl std::error::Error for Error {}
