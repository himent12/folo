//! Applying a condition to an image's pixels and its true boxes.

use folo_api::{BBox, BENCH_SEVERITIES};
use folo_core::rng::Rng;
use image::RgbImage;

use crate::Error;
use crate::pixels::{
    Plane, area_taps, bilinear_taps, convolve_separable, convolve_taps, for_rows, gauss, gaussian_kernel,
    mean_rgb, mix, resample, row_rng, srgb,
};

use super::encoding::encode_jpeg_quality;
use super::{
    CAST_GREEN_SHARE, CAST_RED, CAST_TINT, CAST_TINT_SHARE, Condition, DEFOCUS_SIGMA, DISTANCE_SCALE,
    HAZE_AIRLIGHT, HAZE_SIGMA, HAZE_TRANSMISSION, JPEG_QUALITY, LOW_CONTRAST_KEEP, LOW_LIGHT_GAIN,
    LOW_LIGHT_NOISE, LOW_RESOLUTION_FACTOR, MOTION_LENGTH, MOTION_STREAM, OCCLUSION_AREA, OCCLUSION_STREAM,
    OVEREXPOSURE_GAIN, REFERENCE_SIDE, SENSOR_NOISE_SIGMA,
};

// ---------------------------------------------------------------------------------------------------------------
// Applying a condition
// ---------------------------------------------------------------------------------------------------------------

/// The table index of a severity, or [`Error::Severity`] outside `1..=BENCH_SEVERITIES`.
pub(super) fn level(severity: u8) -> Result<usize, Error> {
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
/// other boxes are left alone. Everything random comes from `seed`
/// ([`variant_seed`](crate::conditions::encoding::variant_seed)). An empty image comes back as it is, and so
/// does one JPEG cannot hold (a side over 65,535 px) under [`Condition::Jpeg`]. A severity outside `1..=5` is
/// [`Error::Severity`]; there is no other error.
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
