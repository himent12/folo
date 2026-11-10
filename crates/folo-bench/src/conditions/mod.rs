//! The stress conditions of a benchmark (docs/BENCH.md, *Conditions*).
//!
//! A [`Condition`] is a deterministic change to an image that keeps its size, at severities 1 (barely visible) to
//! [`BENCH_SEVERITIES`] (severe, and a person can still do the job). [`apply`] makes a variant from decoded
//! pixels; [`render`] goes from an image file to the canonical JPEG of the variant ([`encode_jpeg`], quality 95),
//! the bytes every contestant is graded on, with the variant's true boxes.
//!
//! **Resolution.** Everything measured in pixels (blur sizes, motion length, how much detail low resolution keeps)
//! is given for an image whose long side is 640 px and scales with the image's long side, so a severity means the
//! same on a 320 px thumbnail and on a 4K frame. The parameters are the `const` tables of this module;
//! [`markdown_table`] renders them for docs/BENCH.md, and a test keeps the two in step.
//!
//! **Determinism.** A variant is a function of the pixels, the condition, the severity and the seed alone: noise is
//! drawn per row from the seed, patches per box, and no value depends on the platform's libm or on the thread
//! count (see the `pixels` module). [`variant_seed`] derives the seed from the benchmark's seed, the image and the
//! variant, so every image and every variant has its own noise.

mod catalog;
mod encoding;
mod transform;

pub use catalog::markdown_table;
pub use encoding::{Rendered, decode, encode_jpeg, render, variant_seed};
pub use transform::{apply, moved_boxes};

/// The JPEG quality of the canonical encoding of a variant ([`encode_jpeg`]).
pub const CANONICAL_QUALITY: u8 = 95;

/// The long side, in pixels, that the pixel sizes of the tables below are given for.
pub const REFERENCE_SIDE: f32 = 640.0;

// ---------------------------------------------------------------------------------------------------------------
// Parameters, per severity 1..=5
// ---------------------------------------------------------------------------------------------------------------

/// `low_light`: exposure gain in linear light.
const LOW_LIGHT_GAIN: [f32; 5] = [0.6, 0.4, 0.25, 0.15, 0.08];
/// `low_light`: noise level `n`; the noise is Gaussian in linear light with `σ = n · √(signal + 0.004)` (shot
/// noise plus a read-noise floor).
const LOW_LIGHT_NOISE: [f32; 5] = [0.02, 0.03, 0.04, 0.05, 0.06];
/// `overexposure`: exposure gain in linear light, clipped at white.
const OVEREXPOSURE_GAIN: [f32; 5] = [1.3, 1.6, 2.0, 2.5, 3.2];
/// `low_contrast`: the share of each value's deviation from the image's mean grey that is kept.
const LOW_CONTRAST_KEEP: [f32; 5] = [0.75, 0.58, 0.45, 0.34, 0.25];
/// `defocus`: Gaussian σ in px at 640.
const DEFOCUS_SIGMA: [f32; 5] = [1.0, 2.0, 3.0, 4.5, 6.0];
/// `motion_blur`: length of the linear blur in px at 640.
const MOTION_LENGTH: [f32; 5] = [4.0, 8.0, 13.0, 19.0, 26.0];
/// `sensor_noise`: Gaussian σ per pixel and channel, in 8-bit levels.
const SENSOR_NOISE_SIGMA: [f32; 5] = [5.0, 12.0, 22.0, 34.0, 50.0];
/// `haze`: transmission, the share of the scene's own light that reaches the camera.
const HAZE_TRANSMISSION: [f32; 5] = [0.82, 0.66, 0.52, 0.4, 0.3];
/// `haze`: Gaussian σ of the scattering blur in px at 640.
const HAZE_SIGMA: [f32; 5] = [0.4, 0.8, 1.2, 1.6, 2.0];
/// `haze`: the airlight the scene is blended toward (8-bit sRGB, a cool white).
const HAZE_AIRLIGHT: [f32; 3] = [222.0, 228.0, 234.0];
/// `color_cast`: the share of red kept, in linear light.
const CAST_RED: [f32; 5] = [0.8, 0.6, 0.45, 0.33, 0.22];
/// `color_cast`: green keeps `1 − CAST_GREEN_SHARE · (1 − red kept)` (water absorbs green a little too).
const CAST_GREEN_SHARE: f32 = 0.25;
/// `color_cast`: the scene is mixed toward this blue-green (linear light) by `CAST_TINT_SHARE · (1 − red kept)`.
const CAST_TINT: [f32; 3] = [0.0, 0.16, 0.2];
/// See [`CAST_TINT`].
const CAST_TINT_SHARE: f32 = 0.3;
/// `jpeg`: the quality of the re-encoding.
const JPEG_QUALITY: [u8; 5] = [40, 20, 12, 8, 5];
/// `low_resolution`: downscale factor at 640 (the image keeps the detail of a `640 / factor` px long side).
const LOW_RESOLUTION_FACTOR: [f32; 5] = [1.5, 2.0, 3.0, 4.0, 6.0];
/// `distance`: the scale of the scene on the canvas.
const DISTANCE_SCALE: [f32; 5] = [0.8, 0.65, 0.5, 0.375, 0.25];
/// `occlusion`: the share of each box's area covered.
const OCCLUSION_AREA: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.5];

/// Streams of a variant's seed (so the motion direction and the patches do not reuse the rows' noise).
const MOTION_STREAM: u64 = 0x4D07_10B1;
const OCCLUSION_STREAM: u64 = 0x0CC1_0510;

// ---------------------------------------------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------------------------------------------

/// A stress condition. [`Condition::ALL`] lists them in display order; [`Condition::id`] is the stable id a plan
/// names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Condition {
    /// Darker, as a short exposure, with its noise.
    LowLight,
    /// Brighter, highlights clipped.
    Overexposure,
    /// Values pulled toward the mean grey.
    LowContrast,
    /// Gaussian blur.
    Defocus,
    /// Linear blur along a seeded direction.
    MotionBlur,
    /// Gaussian noise per pixel and channel.
    SensorNoise,
    /// Blend toward a bright airlight, softened.
    Haze,
    /// Red absorbed, a blue-green tint.
    ColorCast,
    /// A low-quality JPEG round trip.
    Jpeg,
    /// Downscaled and scaled back up.
    LowResolution,
    /// The scene shrunk onto a same-size canvas; boxes move with it.
    Distance,
    /// A flat patch over part of every box; boxes stay.
    Occlusion,
}

#[cfg(test)]
mod tests;
