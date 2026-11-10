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

use folo_api::{BBox, BENCH_SEVERITIES, BenchCondition, BenchConditionGroup};
use folo_core::rng::Rng;
use image::RgbImage;

use crate::Error;
use crate::pixels::{
    Plane, area_taps, bilinear_taps, convolve_separable, convolve_taps, for_rows, gauss, gaussian_kernel,
    mean_rgb, mix, resample, row_rng, srgb,
};

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

impl Condition {
    /// Every condition, in display order.
    pub const ALL: [Condition; 12] = [
        Condition::LowLight,
        Condition::Overexposure,
        Condition::LowContrast,
        Condition::Defocus,
        Condition::MotionBlur,
        Condition::SensorNoise,
        Condition::Haze,
        Condition::ColorCast,
        Condition::Jpeg,
        Condition::LowResolution,
        Condition::Distance,
        Condition::Occlusion,
    ];

    /// The stable id, e.g. `low_light` ([`folo_api::BenchCondition::id`]).
    pub fn id(self) -> &'static str {
        match self {
            Condition::LowLight => "low_light",
            Condition::Overexposure => "overexposure",
            Condition::LowContrast => "low_contrast",
            Condition::Defocus => "defocus",
            Condition::MotionBlur => "motion_blur",
            Condition::SensorNoise => "sensor_noise",
            Condition::Haze => "haze",
            Condition::ColorCast => "color_cast",
            Condition::Jpeg => "jpeg",
            Condition::LowResolution => "low_resolution",
            Condition::Distance => "distance",
            Condition::Occlusion => "occlusion",
        }
    }

    /// The condition with this id (exact, case-sensitive); `None` for anything else.
    pub fn parse(id: &str) -> Option<Condition> {
        Condition::ALL.into_iter().find(|c| c.id() == id)
    }

    /// The display name, e.g. `Low light`.
    pub fn name(self) -> &'static str {
        match self {
            Condition::LowLight => "Low light",
            Condition::Overexposure => "Overexposure",
            Condition::LowContrast => "Low contrast",
            Condition::Defocus => "Defocus",
            Condition::MotionBlur => "Motion blur",
            Condition::SensorNoise => "Sensor noise",
            Condition::Haze => "Haze",
            Condition::ColorCast => "Colour cast",
            Condition::Jpeg => "JPEG compression",
            Condition::LowResolution => "Low resolution",
            Condition::Distance => "Distance",
            Condition::Occlusion => "Occlusion",
        }
    }

    /// What kind of trouble it is.
    pub fn group(self) -> BenchConditionGroup {
        use BenchConditionGroup as G;
        match self {
            Condition::LowLight | Condition::Overexposure | Condition::LowContrast => G::Light,
            Condition::Defocus | Condition::MotionBlur => G::Blur,
            Condition::SensorNoise => G::Noise,
            Condition::Haze | Condition::ColorCast => G::Medium,
            Condition::Jpeg | Condition::LowResolution => G::Digital,
            Condition::Distance | Condition::Occlusion => G::Geometry,
        }
    }

    /// What it does to an image and what it stands for, in a sentence, for users.
    pub fn description(self) -> &'static str {
        match self {
            Condition::LowLight => {
                "Darkens the scene as a short exposure does and adds the sensor noise that comes with it: dusk, \
                 indoors, night."
            }
            Condition::Overexposure => {
                "Brightens the scene as a long exposure does, clipping highlights to white: glare, a bright sky, a \
                 flash too close."
            }
            Condition::LowContrast => {
                "Pulls every pixel toward the image's mean grey: flat light, an overcast day, a washed-out screen."
            }
            Condition::Defocus => "Blurs the image as a lens out of focus does.",
            Condition::MotionBlur => {
                "Smears the image along one direction, as a moving camera or subject does during the exposure."
            }
            Condition::SensorNoise => "Adds grain to every pixel and channel, as a small or hot sensor does.",
            Condition::Haze => {
                "Scatters light: blends the scene toward a bright veil and softens detail, as fog, smoke or murky \
                 water do."
            }
            Condition::ColorCast => {
                "Absorbs red and tints the scene blue-green, as water does with depth (or tinted light)."
            }
            Condition::Jpeg => {
                "Re-encodes the image as a low-quality JPEG, with the blocks and smeared colour of strong \
                 compression."
            }
            Condition::LowResolution => {
                "Downscales the image and scales it back up, so fine detail is lost: a cheap camera, a digital \
                 zoom."
            }
            Condition::Distance => {
                "Shrinks the whole scene onto a canvas of the same size, so every object is smaller, as if seen \
                 from farther away; the true boxes move with it."
            }
            Condition::Occlusion => {
                "Covers part of every object with a flat patch, as something in front of it would; the true boxes \
                 stay."
            }
        }
    }

    /// The condition as the catalog lists it.
    pub fn info(self) -> BenchCondition {
        BenchCondition {
            id: self.id().to_string(),
            name: self.name().to_string(),
            group: self.group(),
            description: self.description().to_string(),
        }
    }

    /// Whether the condition moves the true boxes (only [`Condition::Distance`] does).
    pub fn moves_boxes(self) -> bool {
        self == Condition::Distance
    }

    /// The parameter the tables set, as docs/BENCH.md names it, and its value at each severity.
    fn parameters(self) -> (&'static str, [String; 5]) {
        let f = |t: [f32; 5], fmt: &dyn Fn(f32) -> String| -> [String; 5] { t.map(fmt) };
        let plain = |v: f32| format!("{v}");
        match self {
            Condition::LowLight => (
                "gain in linear light, noise `n` (σ = n·√(signal + 0.004))",
                std::array::from_fn(|i| format!("×{} n {}", LOW_LIGHT_GAIN[i], LOW_LIGHT_NOISE[i])),
            ),
            Condition::Overexposure => {
                ("gain in linear light, clipped", f(OVEREXPOSURE_GAIN, &|v| format!("×{v}")))
            }
            Condition::LowContrast => {
                ("share of the deviation from the mean kept", f(LOW_CONTRAST_KEEP, &plain))
            }
            Condition::Defocus => ("Gaussian σ, px at 640", f(DEFOCUS_SIGMA, &plain)),
            Condition::MotionBlur => ("length, px at 640", f(MOTION_LENGTH, &plain)),
            Condition::SensorNoise => ("σ, of 255", f(SENSOR_NOISE_SIGMA, &plain)),
            Condition::Haze => (
                "transmission `t`, blur σ px at 640",
                std::array::from_fn(|i| format!("t {} σ {}", HAZE_TRANSMISSION[i], HAZE_SIGMA[i])),
            ),
            Condition::ColorCast => ("red kept in linear light", f(CAST_RED, &|v| format!("×{v}"))),
            Condition::Jpeg => ("quality", JPEG_QUALITY.map(|q| q.to_string())),
            Condition::LowResolution => {
                ("downscale factor at 640 (at least 1)", f(LOW_RESOLUTION_FACTOR, &|v| format!("{v}×")))
            }
            Condition::Distance => ("scale of the scene", f(DISTANCE_SCALE, &|v| format!("×{v}"))),
            Condition::Occlusion => {
                ("share of each box covered", f(OCCLUSION_AREA, &|v| format!("{}%", (v * 100.0).round())))
            }
        }
    }
}

/// The conditions table of docs/BENCH.md (between its `conditions:begin` / `conditions:end` markers), rendered
/// from the parameter tables.
pub fn markdown_table() -> String {
    let mut s = String::from(
        "| id | Name | Group | What it does | Parameter | 1 | 2 | 3 | 4 | 5 |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    for c in Condition::ALL {
        let (param, values) = c.parameters();
        let group = serde_group(c.group());
        s.push_str(&format!("| `{}` | {} | {group} | {} | {param} |", c.id(), c.name(), c.description()));
        for v in values {
            s.push_str(&format!(" {v} |"));
        }
        s.push('\n');
    }
    s
}

fn serde_group(g: BenchConditionGroup) -> &'static str {
    match g {
        BenchConditionGroup::Light => "light",
        BenchConditionGroup::Blur => "blur",
        BenchConditionGroup::Noise => "noise",
        BenchConditionGroup::Medium => "medium",
        BenchConditionGroup::Digital => "digital",
        BenchConditionGroup::Geometry => "geometry",
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Seeds and encoding
// ---------------------------------------------------------------------------------------------------------------

/// FNV-1a of a string: a stable 64-bit key.
fn key(s: &str) -> u64 {
    s.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01B3))
}

/// The seed of one variant: a function of the benchmark's seed, a stable key of the image (its id within the
/// benchmark), the condition's id and the severity. Equal inputs give equal seeds on every platform and version
/// of this crate.
pub fn variant_seed(bench_seed: u64, image_key: &str, condition: Condition, severity: u8) -> u64 {
    mix(mix(mix(bench_seed, key(image_key)), key(condition.id())), severity as u64)
}

/// The canonical encoding of a variant: a baseline JPEG at quality [`CANONICAL_QUALITY`]. Fails for an empty
/// image or one with a side over 65,535 px.
pub fn encode_jpeg(image: &RgbImage) -> Result<Vec<u8>, Error> {
    encode_jpeg_quality(image, CANONICAL_QUALITY)
}

fn encode_jpeg_quality(image: &RgbImage, quality: u8) -> Result<Vec<u8>, Error> {
    use image::ImageEncoder;
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::Encode("the image is empty".into()));
    }
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
        .write_image(image.as_raw(), image.width(), image.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| Error::Encode(e.to_string()))?;
    Ok(out)
}

/// Decodes an image file (JPEG, PNG, BMP, WebP) to upright RGB exactly as datasets are read
/// ([`folo_data::decode`]: the JPEG EXIF orientation is applied), so the pixels line up with the truth boxes.
pub fn decode(bytes: &[u8]) -> Result<RgbImage, Error> {
    let img = folo_data::decode(bytes).map_err(|e| Error::Decode(e.to_string()))?;
    Ok(RgbImage::from_raw(img.width, img.height, img.data).expect("folo-data images are width x height x 3"))
}

/// A variant as it is served: the canonical JPEG and the true boxes in its coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    /// The JPEG bytes ([`encode_jpeg`]).
    pub jpeg: Vec<u8>,
    /// Width in pixels (the original's, after EXIF orientation).
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The true boxes under the condition ([`apply`]).
    pub boxes: Vec<BBox>,
}

/// An image file in, the variant out: [`decode`], [`apply`], [`encode_jpeg`]. `boxes` are the image's true boxes
/// in pixels of the upright image; `seed` is the variant's ([`variant_seed`]).
pub fn render(
    bytes: &[u8],
    condition: Condition,
    severity: u8,
    boxes: &[BBox],
    seed: u64,
) -> Result<Rendered, Error> {
    level(severity)?;
    let img = decode(bytes)?;
    let (img, boxes) = apply(condition, severity, &img, boxes, seed)?;
    Ok(Rendered { jpeg: encode_jpeg(&img)?, width: img.width(), height: img.height(), boxes })
}

// ---------------------------------------------------------------------------------------------------------------
// Applying a condition
// ---------------------------------------------------------------------------------------------------------------

/// The table index of a severity, or [`Error::Severity`] outside `1..=BENCH_SEVERITIES`.
fn level(severity: u8) -> Result<usize, Error> {
    if (1..=BENCH_SEVERITIES).contains(&severity) {
        Ok(severity as usize - 1)
    } else {
        Err(Error::Severity(severity))
    }
}

/// `image` under `condition` at `severity` (`1..=5`), and the true boxes in the variant's coordinates.
///
/// The output has the input's size. The boxes come back unchanged except for [`Condition::Distance`], which
/// scales and shifts them exactly as it scales and shifts the pixels (a box with non-finite coordinates stays
/// non-finite). [`Condition::Occlusion`] covers part of each box that has a finite, positive area inside the image;
/// other boxes are left alone. Everything random comes from `seed` ([`variant_seed`]). An empty image comes back
/// as it is, and so does one JPEG cannot hold (a side over 65,535 px) under [`Condition::Jpeg`]. A severity
/// outside `1..=5` is [`Error::Severity`]; there is no other error.
pub fn apply(
    condition: Condition,
    severity: u8,
    image: &RgbImage,
    boxes: &[BBox],
    seed: u64,
) -> Result<(RgbImage, Vec<BBox>), Error> {
    let i = level(severity)?;
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w == 0 || h == 0 {
        return Ok((image.clone(), boxes.to_vec()));
    }
    // Pixel sizes of the tables are for a 640 px long side.
    let px = w.max(h) as f32 / REFERENCE_SIDE;
    let raw = image.as_raw();
    let mut boxes = boxes.to_vec();
    let data = match condition {
        Condition::LowLight => low_light(raw, w, LOW_LIGHT_GAIN[i], LOW_LIGHT_NOISE[i], seed),
        Condition::Overexposure => {
            let (t, g) = (srgb(), OVEREXPOSURE_GAIN[i]);
            map_values(raw, |_, v| t.enc(t.lin(v) * g))
        }
        Condition::LowContrast => {
            let m = mean_rgb(raw).iter().map(|&v| v as f32).sum::<f32>() / 3.0;
            let k = LOW_CONTRAST_KEEP[i];
            map_values(raw, |_, v| (m + k * (v as f32 - m) + 0.5).clamp(0.0, 255.0) as u8)
        }
        Condition::Defocus => {
            let k = gaussian_kernel(DEFOCUS_SIGMA[i] * px);
            convolve_separable(&Plane::from_u8(raw, w, h), &k, &k).to_u8()
        }
        Condition::MotionBlur => {
            let taps = motion_kernel(MOTION_LENGTH[i] * px, seed);
            convolve_taps(&Plane::from_u8(raw, w, h), &taps).to_u8()
        }
        Condition::SensorNoise => {
            let sigma = SENSOR_NOISE_SIGMA[i];
            let mut out = raw.to_vec();
            for_rows(&mut out, w * 3, |y, row| {
                let mut rng = row_rng(seed, y);
                for v in row {
                    *v = (*v as f32 + sigma * gauss(&mut rng) + 0.5).clamp(0.0, 255.0) as u8;
                }
            });
            out
        }
        Condition::Haze => {
            let k = gaussian_kernel(HAZE_SIGMA[i] * px);
            let mut p = convolve_separable(&Plane::from_u8(raw, w, h), &k, &k);
            let t = HAZE_TRANSMISSION[i];
            let veil = HAZE_AIRLIGHT.map(|a| a * (1.0 - t));
            for q in p.data.as_chunks_mut::<3>().0 {
                for (v, a) in q.iter_mut().zip(veil) {
                    *v = *v * t + a;
                }
            }
            p.to_u8()
        }
        Condition::ColorCast => {
            let (t, r) = (srgb(), CAST_RED[i]);
            let keep = [r, 1.0 - CAST_GREEN_SHARE * (1.0 - r), 1.0];
            let m = CAST_TINT_SHARE * (1.0 - r);
            let luts: [[u8; 256]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|v| t.enc(t.lin(v as u8) * keep[c] * (1.0 - m) + CAST_TINT[c] * m))
            });
            map_values(raw, |c, v| luts[c][v as usize])
        }
        Condition::Jpeg => {
            let bytes = encode_jpeg_quality(image, JPEG_QUALITY[i]).ok();
            let back =
                bytes.and_then(|b| image::load_from_memory_with_format(&b, image::ImageFormat::Jpeg).ok());
            // A JPEG round trip of an encodable image cannot fail; an image JPEG cannot hold stays as it is.
            match back.map(|d| d.into_rgb8()) {
                Some(d) if d.dimensions() == image.dimensions() => d.into_raw(),
                _ => raw.to_vec(),
            }
        }
        Condition::LowResolution => {
            let f = (LOW_RESOLUTION_FACTOR[i] * px).max(1.0);
            let (w2, h2) = (shrink(w, 1.0 / f), shrink(h, 1.0 / f));
            if (w2, h2) == (w, h) {
                raw.to_vec()
            } else {
                let p = Plane::from_u8(raw, w, h);
                let small = resample(&p, w2, h2, &area_taps(w, w2), &area_taps(h, h2));
                resample(&small, w, h, &bilinear_taps(w2, w), &bilinear_taps(h2, h)).to_u8()
            }
        }
        Condition::Distance => {
            let s = DISTANCE_SCALE[i];
            let (w2, h2) = (shrink(w, s), shrink(h, s));
            let (ox, oy) = ((w - w2) / 2, (h - h2) / 2);
            let small =
                resample(&Plane::from_u8(raw, w, h), w2, h2, &area_taps(w, w2), &area_taps(h, h2)).to_u8();
            let fill = mean_rgb(raw);
            let mut out = vec![0u8; raw.len()];
            for_rows(&mut out, w * 3, |y, row| {
                for p in row.as_chunks_mut::<3>().0 {
                    *p = fill;
                }
                if (oy..oy + h2).contains(&y) {
                    let src = &small[(y - oy) * w2 * 3..(y - oy + 1) * w2 * 3];
                    row[ox * 3..(ox + w2) * 3].copy_from_slice(src);
                }
            });
            distance_boxes(&mut boxes, w, h, s);
            out
        }
        Condition::Occlusion => occlusion(raw, w, h, &boxes, OCCLUSION_AREA[i], seed),
    };
    Ok((RgbImage::from_raw(w as u32, h as u32, data).expect("conditions keep the size"), boxes))
}

/// The true boxes of a `width` x `height` image under `condition` at `severity`, without the pixels: exactly the
/// boxes [`apply`] returns (only [`Condition::Distance`] moves them), for listing a variant's truth without
/// rendering it. A severity outside `1..=5` is [`Error::Severity`].
pub fn moved_boxes(
    condition: Condition,
    severity: u8,
    width: u32,
    height: u32,
    boxes: &[BBox],
) -> Result<Vec<BBox>, Error> {
    let i = level(severity)?;
    let mut boxes = boxes.to_vec();
    if condition == Condition::Distance && width > 0 && height > 0 {
        distance_boxes(&mut boxes, width as usize, height as usize, DISTANCE_SCALE[i]);
    }
    Ok(boxes)
}

/// [`Condition::Distance`]'s box mapping: input coordinate x lands on `ox + x · w2 / w` (`area_taps`), as the
/// pixels do.
fn distance_boxes(boxes: &mut [BBox], w: usize, h: usize, s: f32) {
    let (w2, h2) = (shrink(w, s), shrink(h, s));
    let (ox, oy) = ((w - w2) / 2, (h - h2) / 2);
    let (sx, sy) = (w2 as f64 / w as f64, h2 as f64 / h as f64);
    let (fx, fy) = (|x: f32| (ox as f64 + x as f64 * sx) as f32, |y: f32| (oy as f64 + y as f64 * sy) as f32);
    for b in boxes {
        (b.x1, b.y1, b.x2, b.y2) = (fx(b.x1), fy(b.y1), fx(b.x2), fy(b.y2));
    }
}

/// `n · s` rounded, at least 1 and at most `n`.
fn shrink(n: usize, s: f32) -> usize {
    ((n as f64 * s as f64).round() as usize).clamp(1, n)
}

/// Every value through `f(channel, value)`, by a table per channel.
fn map_values(raw: &[u8], f: impl Fn(usize, u8) -> u8) -> Vec<u8> {
    let luts: [[u8; 256]; 3] = std::array::from_fn(|c| std::array::from_fn(|v| f(c, v as u8)));
    let mut out = raw.to_vec();
    for p in out.as_chunks_mut::<3>().0 {
        for (c, v) in p.iter_mut().enumerate() {
            *v = luts[c][*v as usize];
        }
    }
    out
}

/// Gain in linear light, then shot and read noise in linear light (per pixel and channel).
fn low_light(raw: &[u8], w: usize, gain: f32, n: f32, seed: u64) -> Vec<u8> {
    let t = srgb();
    let mut out = raw.to_vec();
    for_rows(&mut out, w * 3, |y, row| {
        let mut rng = row_rng(seed, y);
        for v in row {
            let s = t.lin(*v) * gain;
            let sigma = n * (s + 0.004).sqrt();
            *v = t.enc(s + sigma * gauss(&mut rng));
        }
    });
    out
}

/// The taps of a linear blur of `length` px along a direction drawn from `seed`: the segment, centred on the
/// pixel, sampled every half pixel and splatted bilinearly onto the pixel grid, normalised to sum 1.
fn motion_kernel(length: f32, seed: u64) -> Vec<(isize, isize, f32)> {
    // A uniform direction without trigonometry: a point of the unit disc, normalised.
    let mut rng = Rng::seed(mix(seed, MOTION_STREAM));
    let (ux, uy) = loop {
        let (u, v) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
        let r2 = u * u + v * v;
        if r2 > 0.01 && r2 <= 1.0 {
            let r = r2.sqrt();
            break (u / r, v / r);
        }
    };
    let length = if length.is_finite() { length.max(0.0) } else { 0.0 };
    let r = (length / 2.0).ceil() as isize + 1;
    let side = (2 * r + 1) as usize;
    let mut grid = vec![0.0f64; side * side];
    let n = (2.0 * length).ceil() as usize + 1;
    for j in 0..n {
        let t = if n == 1 { 0.0 } else { -length / 2.0 + length * j as f32 / (n - 1) as f32 };
        let (x, y) = (t * ux, t * uy);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f64, (y - y0) as f64);
        for (dx, dy, wt) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let gx = (x0 as isize + dx + r) as usize;
            let gy = (y0 as isize + dy + r) as usize;
            grid[gy * side + gx] += wt;
        }
    }
    let total: f64 = grid.iter().sum();
    let mut taps = Vec::new();
    for (k, &v) in grid.iter().enumerate() {
        if v > 1e-9 {
            taps.push(((k % side) as isize - r, (k / side) as isize - r, (v / total) as f32));
        }
    }
    taps
}

/// A flat grey patch over `area` of every box (a seeded sub-rectangle: aspect ratio within 1:2 to 2:1 of the
/// box's, position uniform inside it). Boxes are clipped to the image; one without a finite, positive area there
/// gets no patch.
fn occlusion(raw: &[u8], w: usize, h: usize, boxes: &[BBox], area: f32, seed: u64) -> Vec<u8> {
    let mut out = raw.to_vec();
    for (bi, b) in boxes.iter().enumerate() {
        let mut rng = Rng::seed(mix(seed ^ OCCLUSION_STREAM, bi as u64));
        if ![b.x1, b.y1, b.x2, b.y2].iter().all(|v| v.is_finite()) {
            continue;
        }
        let (x1, y1) = (b.x1.max(0.0), b.y1.max(0.0));
        let (x2, y2) = (b.x2.min(w as f32), b.y2.min(h as f32));
        let (bw, bh) = (x2 - x1, y2 - y1);
        if !(bw.is_finite() && bh.is_finite() && bw > 0.0 && bh > 0.0) {
            continue;
        }
        // Aspect ratio a in [1/2, 2], symmetric in log scale: the patch is sqrt(area·a) by sqrt(area/a) of the box.
        let u = rng.uniform(1.0, 2.0);
        let a = if rng.next_u32() & 1 == 0 { u } else { 1.0 / u };
        let (mut fw, mut fh) = ((area * a).sqrt(), (area / a).sqrt());
        if fw > 1.0 {
            (fw, fh) = (1.0, area);
        } else if fh > 1.0 {
            (fw, fh) = (area, 1.0);
        }
        let (pw, ph) = (fw * bw, fh * bh);
        let px0 = x1 + rng.next_f32() * (bw - pw);
        let py0 = y1 + rng.next_f32() * (bh - ph);
        let grey = 40 + rng.below(176) as u8;
        let cx = |v: f32| (v.round().max(0.0) as usize).min(w);
        let cy = |v: f32| (v.round().max(0.0) as usize).min(h);
        let (xa, xb, ya, yb) = (cx(px0), cx(px0 + pw), cy(py0), cy(py0 + ph));
        for y in ya..yb {
            out[(y * w + xa) * 3..(y * w + xb) * 3].fill(grey);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic test scene: a smooth gradient, a checkerboard, fine stripes and two flat rectangles.
    pub(crate) fn scene(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32 / w.max(1) as f32, y as f32 / h.max(1) as f32);
            let check = ((x / 16 + y / 16) % 2) as f32;
            let stripe = (x % 4 < 2) as u8 as f32;
            let mut p = [
                (40.0 + 150.0 * fx + 40.0 * check) as u8,
                (60.0 + 120.0 * fy + 30.0 * stripe) as u8,
                (90.0 + 80.0 * (1.0 - fx) + 40.0 * check) as u8,
            ];
            if (w / 5..w / 3).contains(&x) && (h / 4..h / 2).contains(&y) {
                p = [230, 40, 30];
            }
            if (w / 2..w * 4 / 5).contains(&x) && (h / 2..h * 5 / 6).contains(&y) {
                p = [20, 30, 200];
            }
            image::Rgb(p)
        })
    }

    fn bx(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { class: 0, x1, y1, x2, y2 }
    }

    fn mean(img: &RgbImage) -> f64 {
        img.as_raw().iter().map(|&v| v as f64).sum::<f64>() / img.as_raw().len() as f64
    }

    fn std(img: &RgbImage) -> f64 {
        let m = mean(img);
        (img.as_raw().iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / img.as_raw().len() as f64).sqrt()
    }

    /// Mean absolute difference between horizontal and vertical neighbours: high-frequency energy.
    fn hf(img: &RgbImage) -> f64 {
        let (w, h) = img.dimensions();
        let mut s = 0.0;
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                let (p, r, d) = (img.get_pixel(x, y), img.get_pixel(x + 1, y), img.get_pixel(x, y + 1));
                for c in 0..3 {
                    s += (p[c] as f64 - r[c] as f64).abs() + (p[c] as f64 - d[c] as f64).abs();
                }
            }
        }
        s / (w as f64 * h as f64)
    }

    fn diff(a: &RgbImage, b: &RgbImage) -> f64 {
        a.as_raw().iter().zip(b.as_raw()).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>()
            / a.as_raw().len() as f64
    }

    fn run(c: Condition, s: u8, img: &RgbImage) -> RgbImage {
        apply(c, s, img, &[bx(10.0, 10.0, 60.0, 50.0)], 7).unwrap().0
    }

    #[test]
    fn ids_parse_exactly() {
        for c in Condition::ALL {
            assert_eq!(Condition::parse(c.id()), Some(c));
        }
        assert_eq!(Condition::parse("Haze"), None);
        assert_eq!(Condition::parse(""), None);
    }

    #[test]
    fn every_condition_keeps_the_size_even_for_tiny_and_odd_images() {
        for (w, h) in [(1, 1), (1, 9), (7, 5), (33, 17), (0, 4)] {
            let img = scene(w, h);
            for c in Condition::ALL {
                for s in 1..=BENCH_SEVERITIES {
                    let (out, boxes) = apply(c, s, &img, &[bx(0.0, 0.0, 3.0, 3.0)], 1).unwrap();
                    assert_eq!(out.dimensions(), (w, h), "{} {s} {w}x{h}", c.id());
                    assert_eq!(boxes.len(), 1);
                }
            }
        }
    }

    #[test]
    fn severity_outside_the_range_is_an_error() {
        let img = scene(8, 8);
        for s in [0, 6, 255] {
            assert_eq!(apply(Condition::Haze, s, &img, &[], 0), Err(Error::Severity(s)));
        }
        assert!(render(&[], Condition::Haze, 0, &[], 0).is_err());
    }

    #[test]
    fn same_seed_same_bytes_and_seeds_move_the_noise() {
        let img = scene(96, 64);
        let boxes = [bx(5.0, 5.0, 60.0, 40.0), bx(30.0, 20.0, 90.0, 60.0)];
        for c in Condition::ALL {
            let a = apply(c, 3, &img, &boxes, 11).unwrap();
            assert!(a == apply(c, 3, &img, &boxes, 11).unwrap(), "{}", c.id());
            let b = apply(c, 3, &img, &boxes, 12).unwrap();
            let seeded = matches!(
                c,
                Condition::LowLight | Condition::SensorNoise | Condition::MotionBlur | Condition::Occlusion
            );
            assert_eq!(a.0 != b.0, seeded, "{}: does the seed matter", c.id());
        }
        let s = variant_seed(0, "img_1", Condition::Haze, 3);
        assert_eq!(s, variant_seed(0, "img_1", Condition::Haze, 3));
        let others = [
            variant_seed(1, "img_1", Condition::Haze, 3),
            variant_seed(0, "img_2", Condition::Haze, 3),
            variant_seed(0, "img_1", Condition::Defocus, 3),
            variant_seed(0, "img_1", Condition::Haze, 4),
        ];
        assert!(others.iter().all(|&o| o != s));
    }

    #[test]
    fn boxes_stay_except_under_distance() {
        let img = scene(64, 48);
        let boxes = [bx(5.0, 5.0, 30.0, 20.0), bx(f32::NAN, 0.0, 3.0, 3.0)];
        for c in Condition::ALL {
            let (_, out) = apply(c, 5, &img, &boxes, 3).unwrap();
            if c.moves_boxes() {
                assert_ne!(out[0], boxes[0]);
                assert!(out[1].x1.is_nan());
            } else {
                assert_eq!(out[0], boxes[0], "{}", c.id());
                assert!(out[1].x1.is_nan());
            }
        }
    }

    #[test]
    fn effects_grow_with_severity() {
        // A long side of 480 px: low_resolution's mildest factor (1.5 at 640) still downscales (1.125).
        let img = scene(480, 360);
        let at = |c: Condition| -> Vec<RgbImage> { (1..=5).map(|s| run(c, s, &img)).collect() };
        let falling = |name: &str, v: Vec<f64>, base: f64| {
            assert!(v[0] < base, "{name}: {base} -> {v:?}");
            assert!(v.windows(2).all(|p| p[1] < p[0]), "{name}: {v:?}");
        };
        let rising = |name: &str, v: Vec<f64>, base: f64| {
            assert!(v[0] > base, "{name}: {base} -> {v:?}");
            assert!(v.windows(2).all(|p| p[1] > p[0]), "{name}: {v:?}");
        };
        falling("low_light brightness", at(Condition::LowLight).iter().map(mean).collect(), mean(&img));
        rising("overexposure brightness", at(Condition::Overexposure).iter().map(mean).collect(), mean(&img));
        falling("low_contrast spread", at(Condition::LowContrast).iter().map(std).collect(), std(&img));
        falling("defocus detail", at(Condition::Defocus).iter().map(hf).collect(), hf(&img));
        falling("motion detail", at(Condition::MotionBlur).iter().map(hf).collect(), hf(&img));
        rising("noise", at(Condition::SensorNoise).iter().map(|v| diff(v, &img)).collect(), 0.0);
        falling("haze spread", at(Condition::Haze).iter().map(std).collect(), std(&img));
        let red = |i: &RgbImage| i.pixels().map(|p| p[0] as f64).sum::<f64>();
        falling("cast red", at(Condition::ColorCast).iter().map(red).collect(), red(&img));
        // JPEG's error is not strictly monotonic in quality on synthetic stripes; it is from 1 to 3 to 5.
        let jpeg: Vec<f64> = at(Condition::Jpeg).iter().map(|v| diff(v, &img)).collect();
        rising("jpeg error", vec![jpeg[0], jpeg[2], jpeg[4]], 0.0);
        let sizes: Vec<f64> =
            JPEG_QUALITY.iter().map(|&q| encode_jpeg_quality(&img, q).unwrap().len() as f64).collect();
        falling("jpeg bytes", sizes, encode_jpeg(&img).unwrap().len() as f64);
        falling("low_resolution detail", at(Condition::LowResolution).iter().map(hf).collect(), hf(&img));
        let changed =
            |i: &RgbImage| i.as_raw().iter().zip(img.as_raw()).filter(|(a, b)| a != b).count() as f64;
        rising("occluded pixels", at(Condition::Occlusion).iter().map(changed).collect(), 0.0);
        let width = |s: u8| {
            let (_, b) = apply(Condition::Distance, s, &img, &[bx(10.0, 10.0, 110.0, 90.0)], 0).unwrap();
            (b[0].x2 - b[0].x1) as f64
        };
        falling("distance box width", (1..=5).map(width).collect(), 100.0);
    }

    #[test]
    fn distance_moves_boxes_exactly_with_the_pixels() {
        // A white rectangle on black; after the condition, the white pixels are exactly the moved box.
        for (w, h, s) in [(640u32, 480u32, 3u8), (320, 200, 5), (300, 300, 1)] {
            let (x1, y1, x2, y2) = (100u32, 40u32, 220u32, 160u32);
            let img = RgbImage::from_fn(w, h, |x, y| {
                image::Rgb(if (x1..x2).contains(&x) && (y1..y2).contains(&y) { [255; 3] } else { [0; 3] })
            });
            let (out, b) =
                apply(Condition::Distance, s, &img, &[bx(x1 as f32, y1 as f32, x2 as f32, y2 as f32)], 0)
                    .unwrap();
            // Pixels at least half covered by the rectangle are at least half white.
            let (mut lo, mut hi) = ((u32::MAX, u32::MAX), (0, 0));
            for (x, y, p) in out.enumerate_pixels() {
                if p[0] >= 128 {
                    lo = (lo.0.min(x), lo.1.min(y));
                    hi = (hi.0.max(x + 1), hi.1.max(y + 1));
                }
            }
            let b = b[0];
            for (got, want) in [(lo.0, b.x1), (lo.1, b.y1), (hi.0, b.x2), (hi.1, b.y2)] {
                assert!((got as f32 - want).abs() <= 0.5, "{w}x{h} s{s}: pixels {lo:?}..{hi:?}, box {b:?}");
            }
        }
        // Exactly: 640 x 480 at scale 0.5 is 320 x 240 at (160, 120), so x 100 lands on 160 + 50.
        let img = scene(640, 480);
        let (_, b) = apply(Condition::Distance, 3, &img, &[bx(100.0, 40.0, 220.0, 160.0)], 0).unwrap();
        assert_eq!(b[0], bx(210.0, 140.0, 270.0, 200.0));
    }

    #[test]
    fn moved_boxes_are_the_boxes_apply_returns() {
        let boxes = [bx(100.0, 40.0, 220.0, 160.0), bx(0.0, 0.0, 33.5, 17.25)];
        for (w, h) in [(640u32, 480u32), (333, 101)] {
            let img = scene(w, h);
            for c in Condition::ALL {
                for s in 1..=5 {
                    let (_, want) = apply(c, s, &img, &boxes, 3).unwrap();
                    assert_eq!(moved_boxes(c, s, w, h, &boxes).unwrap(), want, "{} {s} {w}x{h}", c.id());
                }
            }
        }
        assert_eq!(moved_boxes(Condition::Distance, 0, 10, 10, &boxes), Err(Error::Severity(0)));
    }

    #[test]
    fn pixel_sizes_scale_with_the_long_side() {
        // A vertical step edge under defocus, motion blur and low resolution: the width of the transition,
        // Σ min(v, 255 − v) / 255 along a row (proportional to the blur's size), doubles with the image.
        let spread = |c: Condition, w: u32| {
            let img =
                RgbImage::from_fn(w, w / 2, |x, _| image::Rgb(if x < w / 2 { [0; 3] } else { [255; 3] }));
            let out = run(c, 3, &img);
            (0..w).map(|x| out.get_pixel(x, w / 4)[0]).map(|v| v.min(255 - v) as f64 / 255.0).sum::<f64>()
        };
        for c in [Condition::Defocus, Condition::MotionBlur, Condition::LowResolution] {
            let (a, b) = (spread(c, 640), spread(c, 1280));
            assert!(a > 0.5 && (b / a - 2.0).abs() < 0.2, "{}: {a} {b}", c.id());
        }
    }

    #[test]
    fn occlusion_covers_the_share_of_the_box() {
        let img = RgbImage::from_pixel(200, 200, image::Rgb([0, 255, 0]));
        let boxes = [bx(20.0, 20.0, 120.0, 100.0)];
        for s in 1..=5u8 {
            let (out, _) = apply(Condition::Occlusion, s, &img, &boxes, 9).unwrap();
            let covered = out.pixels().filter(|p| p[1] != 255).count() as f32;
            let want = OCCLUSION_AREA[s as usize - 1] * 100.0 * 80.0;
            assert!((covered - want).abs() <= 0.06 * want + 120.0, "s{s}: {covered} vs {want}");
            // Only inside the box.
            for (x, y, p) in out.enumerate_pixels() {
                if p[1] != 255 {
                    assert!((20..120).contains(&x) && (20..100).contains(&y));
                }
            }
        }
        // Boxes outside the image or degenerate get no patch and do not panic.
        let odd = [bx(-50.0, -50.0, -10.0, -10.0), bx(f32::NAN, 0.0, 10.0, 10.0), bx(5.0, 5.0, 5.0, 50.0)];
        let (out, _) = apply(Condition::Occlusion, 5, &img, &odd, 9).unwrap();
        assert!(out == img, "a degenerate box got a patch");
    }

    #[test]
    fn render_decodes_applies_and_encodes() {
        let img = scene(120, 80);
        let png = {
            let mut v = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png).unwrap();
            v
        };
        let boxes = [bx(10.0, 10.0, 50.0, 40.0)];
        let r = render(&png, Condition::Distance, 3, &boxes, 5).unwrap();
        assert_eq!((r.width, r.height), (120, 80));
        let (want, want_boxes) = apply(Condition::Distance, 3, &img, &boxes, 5).unwrap();
        assert_eq!(r.boxes, want_boxes);
        assert_eq!(r.jpeg, encode_jpeg(&want).unwrap());
        assert_eq!(decode(&r.jpeg).unwrap().dimensions(), (120, 80));
        assert!(matches!(render(b"not an image", Condition::Haze, 1, &[], 0), Err(Error::Decode(_))));
        assert!(matches!(encode_jpeg(&RgbImage::new(0, 3)), Err(Error::Encode(_))));
    }

    #[test]
    fn docs_table_is_in_step() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/BENCH.md");
        let doc = std::fs::read_to_string(path).expect("docs/BENCH.md").replace("\r\n", "\n");
        let begin = doc.find("<!-- conditions:begin").expect("begin marker");
        let begin = begin + doc[begin..].find('\n').expect("end of the begin marker") + 1;
        let end = doc.find("<!-- conditions:end -->").expect("end marker");
        let want = markdown_table();
        if std::env::var_os("FOLO_BLESS").is_some() {
            std::fs::write(path, format!("{}{want}{}", &doc[..begin], &doc[end..])).unwrap();
            return;
        }
        assert!(
            doc[begin..end] == want,
            "docs/BENCH.md's conditions table differs from crates/folo-bench/src/conditions.rs; regenerate it with \
             `FOLO_BLESS=1 cargo test -p folo-bench docs_table_is_in_step`"
        );
    }
}
