// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/directx_renderer.rs (get_font_info,
// get_gamma_ratios), crates/gpui/src/platform/windows/alpha_correction.hlsl
//! The glyph contrast and gamma correction GPUI derives from the system's DirectWrite rendering params,
//! and the shader's correction formula on the CPU.

use crate::GlyphBlend;
use anyhow::Result;
use windows::Win32::Graphics::DirectWrite::{IDWriteFactory5, IDWriteRenderingParams1};
use windows::core::Interface;

/// GPUI's `get_font_info`: the default rendering params' gamma (as ratios) and grayscale enhanced
/// contrast.
pub(super) fn system_glyph_blend(factory: &IDWriteFactory5) -> Result<GlyphBlend> {
    // SAFETY: plain DirectWrite calls.
    unsafe {
        let render_params: IDWriteRenderingParams1 = factory.CreateRenderingParams()?.cast()?;
        Ok(GlyphBlend::new(
            gamma_ratios(render_params.GetGamma()),
            render_params.GetGrayscaleEnhancedContrast(),
        ))
    }
}

/// Gamma ratios for brightening / darkening edges (GPUI's `get_gamma_ratios`, after Windows Terminal's
/// `dwrite.cpp`): the gamma rounded to a tenth in 1.0..=2.2, normalized.
pub fn gamma_ratios(gamma: f32) -> [f32; 4] {
    const GAMMA_INCORRECT_TARGET_RATIOS: [[f32; 4]; 13] = [
        [0.0000 / 4.0, 0.0000 / 4.0, 0.0000 / 4.0, 0.0000 / 4.0], // gamma = 1.0
        [0.0166 / 4.0, -0.0807 / 4.0, 0.2227 / 4.0, -0.0751 / 4.0], // gamma = 1.1
        [0.0350 / 4.0, -0.1760 / 4.0, 0.4325 / 4.0, -0.1370 / 4.0], // gamma = 1.2
        [0.0543 / 4.0, -0.2821 / 4.0, 0.6302 / 4.0, -0.1876 / 4.0], // gamma = 1.3
        [0.0739 / 4.0, -0.3963 / 4.0, 0.8167 / 4.0, -0.2287 / 4.0], // gamma = 1.4
        [0.0933 / 4.0, -0.5161 / 4.0, 0.9926 / 4.0, -0.2616 / 4.0], // gamma = 1.5
        [0.1121 / 4.0, -0.6395 / 4.0, 1.1588 / 4.0, -0.2877 / 4.0], // gamma = 1.6
        [0.1300 / 4.0, -0.7649 / 4.0, 1.3159 / 4.0, -0.3080 / 4.0], // gamma = 1.7
        [0.1469 / 4.0, -0.8911 / 4.0, 1.4644 / 4.0, -0.3234 / 4.0], // gamma = 1.8
        [0.1627 / 4.0, -1.0170 / 4.0, 1.6051 / 4.0, -0.3347 / 4.0], // gamma = 1.9
        [0.1773 / 4.0, -1.1420 / 4.0, 1.7385 / 4.0, -0.3426 / 4.0], // gamma = 2.0
        [0.1908 / 4.0, -1.2652 / 4.0, 1.8650 / 4.0, -0.3476 / 4.0], // gamma = 2.1
        [0.2031 / 4.0, -1.3864 / 4.0, 1.9851 / 4.0, -0.3501 / 4.0], // gamma = 2.2
    ];

    const NORM13: f32 = ((0x10000 as f64) / (255.0 * 255.0) * 4.0) as f32;
    const NORM24: f32 = ((0x100 as f64) / (255.0) * 4.0) as f32;

    let index = ((gamma * 10.0).round() as usize).clamp(10, 22) - 10;
    let ratios = GAMMA_INCORRECT_TARGET_RATIOS[index];

    [ratios[0] * NORM13, ratios[1] * NORM24, ratios[2] * NORM13, ratios[3] * NORM24]
}

/// REC. 601 luminance.
fn color_brightness(color: [f32; 3]) -> f32 {
    color[0] * 0.30 + color[1] * 0.59 + color[2] * 0.11
}

fn light_on_dark_contrast(enhanced_contrast: f32, color: [f32; 3]) -> f32 {
    let brightness = color_brightness(color);
    let multiplier = (4.0 * (0.75 - brightness)).clamp(0.0, 1.0);
    enhanced_contrast * multiplier
}

fn enhance_contrast(alpha: f32, k: f32) -> f32 {
    alpha * (k + 1.0) / (alpha * k + 1.0)
}

fn apply_alpha_correction(a: f32, b: f32, g: [f32; 4]) -> f32 {
    let brightness_adjustment = g[0] * b + g[1];
    let correction = brightness_adjustment * a + (g[2] * b + g[3]);
    a + a * (1.0 - a) * correction
}

/// The shaders' `apply_contrast_and_gamma_correction`: a coverage `sample` (0..=1) of text in `color`
/// (straight RGB) made darker or lighter by the enhanced contrast and gamma ratios.
pub fn apply_contrast_and_gamma_correction(
    sample: f32,
    color: [f32; 3],
    enhanced_contrast_factor: f32,
    gamma_ratios: [f32; 4],
) -> f32 {
    let enhanced_contrast = light_on_dark_contrast(enhanced_contrast_factor, color);
    let brightness = color_brightness(color);
    let contrasted = enhance_contrast(sample, enhanced_contrast);
    apply_alpha_correction(contrasted, brightness, gamma_ratios)
}

#[cfg(test)]
mod tests {
    use super::{apply_contrast_and_gamma_correction, gamma_ratios};

    #[test]
    fn gamma_ratios_pick_the_rounded_row() {
        assert_eq!(gamma_ratios(1.0), [0.; 4]);
        assert_eq!(gamma_ratios(0.5), gamma_ratios(1.0), "clamped below");
        assert_eq!(gamma_ratios(3.0), gamma_ratios(2.2), "clamped above");
        let g = gamma_ratios(1.8);
        let norm13 = (65536.0 / (255.0 * 255.0) * 4.0) as f32;
        assert!((g[0] - 0.1469 / 4.0 * norm13).abs() < 1e-6);
    }

    #[test]
    fn correction_keeps_the_ends_and_darkens_dark_text_edges() {
        let ratios = gamma_ratios(1.8);
        for color in [[0., 0., 0.], [1., 1., 1.]] {
            assert_eq!(apply_contrast_and_gamma_correction(0.0, color, 1.0, ratios), 0.0);
            assert!((apply_contrast_and_gamma_correction(1.0, color, 1.0, ratios) - 1.0).abs() < 1e-6);
        }
        // Black text, half coverage, worked by hand from the HLSL: contrast k = 1 gives 2/3, then
        // a + a (1 - a) (g1 a + g3) with g1 = -0.8911 / 4 * 1024 / 255, g3 = -0.3234 / 4 * 1024 / 255.
        let (g1, g3) = (-0.8911 / 4.0 * 1024.0 / 255.0, -0.3234 / 4.0 * 1024.0 / 255.0);
        let a: f64 = 2.0 / 3.0;
        let want = a + a * (1.0 - a) * (g1 * a + g3);
        let got = apply_contrast_and_gamma_correction(0.5, [0., 0., 0.], 1.0, ratios) as f64;
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
}
