//! What each condition is: its id, name, group, description and parameters, and the docs table of them.

use folo_api::{BenchCondition, BenchConditionGroup};

use super::{
    CAST_RED, Condition, DEFOCUS_SIGMA, DISTANCE_SCALE, HAZE_SIGMA, HAZE_TRANSMISSION, JPEG_QUALITY,
    LOW_CONTRAST_KEEP, LOW_LIGHT_GAIN, LOW_LIGHT_NOISE, LOW_RESOLUTION_FACTOR, MOTION_LENGTH, OCCLUSION_AREA,
    OVEREXPOSURE_GAIN, SENSOR_NOISE_SIGMA,
};

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
