//! Range calibration (docs/MODEL.md §9.6): the `f32` model over sample images, a histogram of every quantised
//! boundary, and a range per boundary chosen from it.
//!
//! Two passes over the same images: the first records each boundary's minimum and maximum, the second a
//! histogram of [`BINS`] bins between them. [`Calibrator::finish`] then picks each range by [`Method`].

use std::collections::BTreeMap;

use folo_core::Tensor;
use folo_nn::{ConvBnAct, Forward, Result};
use serde::{Deserialize, Serialize};

use super::QParam;
use super::walk::{self, Exec};
use crate::model::FoloModel;

/// Histogram bins per boundary.
pub const BINS: usize = 2048;

/// How a boundary's range is chosen from its histogram.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Method {
    /// The observed minimum and maximum.
    MinMax,
    /// The `p` and `1 − p` quantiles (`p = 1e-4`: the 99.99% range).
    Percentile {
        /// Mass cut from each tail.
        p: f64,
    },
    /// The range that minimises the expected squared error of rounding (`Δ² / 12` inside) plus clipping
    /// (outside), over histogram-edge candidates.
    Mse,
}

/// The stored result of calibration: a `[lo, hi]` range per boundary name ([`walk::Exec`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// How the ranges were chosen.
    pub method: Method,
    /// Images the ranges were measured on.
    pub images: usize,
    /// Boundary name to `[lo, hi]`.
    pub ranges: BTreeMap<String, [f32; 2]>,
    /// Convs that keep `f32` weights (mixed precision): their inputs are dequantised and their outputs
    /// quantised again.
    #[serde(default)]
    pub f32_convs: Vec<String>,
    /// The [`super::fingerprint`] of the weights the ranges were measured on (`None`: not checked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
}

impl Calibration {
    /// The grid of boundary `name`.
    pub fn param(&self, name: &str) -> Result<QParam> {
        let [lo, hi] = self.ranges.get(name).copied().ok_or_else(|| {
            folo_nn::Error::Invalid(format!(
                "int8 calibration has no range for `{name}`: re-run `folo quantize`"
            ))
        })?;
        Ok(QParam::from_range(lo, hi))
    }

    /// Whether conv `name` keeps `f32` weights.
    pub fn is_f32(&self, name: &str) -> bool {
        self.f32_convs.iter().any(|c| c == name)
    }
}

#[derive(Default)]
struct Stats {
    min: f32,
    max: f32,
    seen: bool,
    hist: Vec<u64>,
}

impl Stats {
    fn add(&mut self, v: &[f32], pass: u8) {
        if pass == 0 {
            for &x in v {
                if !self.seen {
                    (self.min, self.max, self.seen) = (x, x, true);
                }
                self.min = self.min.min(x);
                self.max = self.max.max(x);
            }
            return;
        }
        if self.hist.is_empty() {
            self.hist = vec![0; BINS];
        }
        let (lo, w) = (self.min, (self.max - self.min).max(1e-12) / BINS as f32);
        for &x in v {
            let b = (((x - lo) / w) as isize).clamp(0, BINS as isize - 1) as usize;
            self.hist[b] += 1;
        }
    }

    /// `[lo, hi]` by `method`.
    fn range(&self, method: Method) -> [f32; 2] {
        if self.hist.is_empty() || method == Method::MinMax {
            return [self.min, self.max];
        }
        let w = (self.max - self.min) as f64 / BINS as f64;
        let edge = |i: usize| self.min as f64 + w * i as f64;
        let total: u64 = self.hist.iter().sum();
        match method {
            Method::MinMax => unreachable!(),
            Method::Percentile { p } => {
                let cut = (p * total as f64) as u64;
                let (mut lo, mut acc) = (0usize, 0u64);
                while lo < BINS && acc + self.hist[lo] <= cut {
                    acc += self.hist[lo];
                    lo += 1;
                }
                let (mut hi, mut acc) = (BINS, 0u64);
                while hi > 0 && acc + self.hist[hi - 1] <= cut {
                    acc += self.hist[hi - 1];
                    hi -= 1;
                }
                [edge(lo) as f32, edge(hi.max(lo + 1)) as f32]
            }
            Method::Mse => self.mse_range(w),
        }
    }

    /// Search `lo` over the lowest and `hi` over the highest edges with prefix sums: the clipping error of the
    /// bins outside is `Σ n (c − edge)²`, the rounding error inside `n_in · Δ² / 12`.
    fn mse_range(&self, w: f64) -> [f32; 2] {
        let n = BINS;
        let c = |i: usize| self.min as f64 + w * (i as f64 + 0.5);
        let (mut s0, mut s1, mut s2) = (vec![0f64; n + 1], vec![0f64; n + 1], vec![0f64; n + 1]);
        for i in 0..n {
            let (h, x) = (self.hist[i] as f64, c(i));
            s0[i + 1] = s0[i] + h;
            s1[i + 1] = s1[i] + h * x;
            s2[i + 1] = s2[i] + h * x * x;
        }
        let edge = |i: usize| self.min as f64 + w * i as f64;
        // The edges that keep 0 inside, coarse enough to stay cheap.
        let zero = (((0.0 - self.min as f64) / w).floor().max(0.0) as usize).min(n);
        let step = (n / 512).max(1);
        let los: Vec<usize> = (0..=zero).step_by(step).chain([zero]).collect();
        let his: Vec<usize> = (zero.max(1)..=n).rev().step_by(step).chain([zero.max(1)]).collect();
        let mut best = (f64::INFINITY, 0, n);
        for &l in &los {
            for &h in &his {
                if h <= l {
                    continue;
                }
                let (el, eh) = (edge(l).min(0.0), edge(h).max(0.0));
                let d = (eh - el) / 255.0;
                let below = s2[l] - 2.0 * el * s1[l] + el * el * s0[l];
                let above = (s2[n] - s2[h]) - 2.0 * eh * (s1[n] - s1[h]) + eh * eh * (s0[n] - s0[h]);
                let inside = (s0[h] - s0[l]) * d * d / 12.0;
                let e = below + above + inside;
                if e < best.0 {
                    best = (e, l, h);
                }
            }
        }
        [edge(best.1) as f32, edge(best.2) as f32]
    }
}

/// Collects the boundary statistics of calibration images: [`Calibrator::observe`] every batch, call
/// [`Calibrator::next_pass`], observe the same batches again, then [`Calibrator::finish`].
#[derive(Default)]
pub struct Calibrator {
    stats: BTreeMap<String, Stats>,
    pass: u8,
    /// Images observed in each pass.
    images: [usize; 2],
}

impl Calibrator {
    /// An empty calibrator in its first pass.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the `f32` model on `x` (`[B, 3, H, W]`, `f32`, on the CPU) and record every boundary.
    pub fn observe(&mut self, model: &FoloModel, x: &Tensor) -> Result<()> {
        let _g = folo_core::no_grad();
        self.images[self.pass as usize] += x.dims()[0];
        let mut e = Observe { cal: self };
        folo_core::cpu::in_pool(|| walk::features(model, &mut e, x))?;
        Ok(())
    }

    /// Start the histogram pass.
    pub fn next_pass(&mut self) {
        self.pass = 1;
    }

    /// The ranges by `method`. Refused unless both passes observed the same images, at least one.
    pub fn finish(&self, method: Method) -> Result<Calibration> {
        let [a, b] = self.images;
        if a == 0 || self.pass != 1 || a != b {
            return Err(folo_nn::Error::Invalid(format!(
                "int8 calibration needs two passes over the same images, at least one (saw {a}, then {b})"
            )));
        }
        let ranges = self.stats.iter().map(|(k, s)| (k.clone(), s.range(method))).collect();
        Ok(Calibration { method, images: a, ranges, f32_convs: Vec::new(), weights: None })
    }

    fn record(&mut self, name: &str, t: &Tensor) -> Result<()> {
        let v = t.to_vec::<f32>()?;
        let pass = self.pass;
        self.stats.entry(name.to_string()).or_default().add(&v, pass);
        Ok(())
    }
}

/// The `f32` model, recording each boundary.
struct Observe<'a> {
    cal: &'a mut Calibrator,
}

/// `act(conv(cat(parts))) (+ res)` in `f32`.
pub(super) fn conv_f32(conv: &ConvBnAct, parts: &[&Tensor], res: Option<&Tensor>) -> Result<Tensor> {
    let y = conv.forward_cat(parts)?;
    Ok(match res {
        Some(r) => r.add(&y)?,
        None => y,
    })
}

impl Exec for Observe<'_> {
    type A = Tensor;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Tensor> {
        let y = conv.forward(x)?;
        self.cal.record(name, &y)?;
        Ok(y)
    }

    fn conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&Tensor],
        res: Option<&Tensor>,
    ) -> Result<Tensor> {
        let y = conv_f32(conv, parts, res)?;
        self.cal.record(name, &y)?;
        Ok(y)
    }

    fn conv_f32(&mut self, _name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<Tensor> {
        conv.forward(x)
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<Tensor> {
        self.cal.record(name, x)?;
        Ok(x.clone())
    }

    fn dequantize(&mut self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn narrow(&mut self, x: &Tensor, start: usize, len: usize) -> Result<Tensor> {
        Ok(x.narrow(1, start, len)?)
    }

    fn max_pool(&mut self, x: &Tensor, k: usize) -> Result<Tensor> {
        Ok(x.max_pool2d(k, 1, k / 2)?)
    }

    fn upsample2(&mut self, x: &Tensor) -> Result<Tensor> {
        Ok(x.upsample_nearest2d(2)?)
    }

    fn dims(&self, x: &Tensor) -> [usize; 4] {
        let d = x.dims();
        [d[0], d[1], d[2], d[3]]
    }
}

/// Calibrate `model` (fused, eval mode, `f32` on the CPU) on `batches`, which is called twice and must yield
/// the same images both times.
pub fn calibrate<I>(model: &FoloModel, mut batches: impl FnMut() -> I, method: Method) -> Result<Calibration>
where
    I: Iterator<Item = Result<Tensor>>,
{
    let mut c = Calibrator::new();
    for x in batches() {
        c.observe(model, &x?)?;
    }
    c.next_pass();
    for x in batches() {
        c.observe(model, &x?)?;
    }
    let mut cal = c.finish(method)?;
    cal.weights = Some(super::fingerprint(model)?);
    Ok(cal)
}

/// [`calibrate`] on `images` images of `part` drawn at random by `seed` (all of them when it has fewer), letterboxed
/// to `imgsz` as validation does, in batches of `batch` read by `workers` threads (`folo quantize`). Returns the
/// calibration and the number of images.
pub fn calibrate_sample(
    model: &FoloModel,
    part: &folo_data::FileDataset,
    images: usize,
    seed: u64,
    (imgsz, batch, workers): (u32, usize, usize),
    method: Method,
) -> Result<(Calibration, usize)> {
    use folo_data::{DataLoader, Dataset, FileDataset, LoaderConfig};
    let mut entries = part.entries().to_vec();
    folo_core::rng::Rng::seed(seed).shuffle(&mut entries);
    entries.truncate(images);
    let n = entries.len();
    let sample = FileDataset::new(part.split(), part.class_names().to_vec(), entries);
    let data = |e: folo_data::DataError| folo_nn::Error::Invalid(format!("calibration images: {e}"));
    let loader = DataLoader::new(
        std::sync::Arc::new(sample),
        LoaderConfig { workers, ..LoaderConfig::val(batch, imgsz) },
    )
    .map_err(data)?;
    let batches = || {
        loader.epoch(0).map(|b| {
            let mut b = b.map_err(data)?;
            Ok(Tensor::from_vec(std::mem::take(&mut b.images), b.shape)?)
        })
    };
    Ok((calibrate(model, batches, method)?, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(v: &[f32]) -> Stats {
        let mut s = Stats::default();
        s.add(v, 0);
        s.add(v, 1);
        s
    }

    #[test]
    fn ranges_have_their_known_answers() {
        // 0..20480 step 1: about 10 values a bin.
        let v: Vec<f32> = (0..20480).map(|i| i as f32).collect();
        let s = stats(&v);
        assert_eq!(s.range(Method::MinMax), [0.0, 20479.0]);
        assert_eq!(s.range(Method::Percentile { p: 0.0 }), [0.0, 20479.0]);
        // 1% off each end, to a bin (~10 wide).
        let [lo, hi] = s.range(Method::Percentile { p: 0.01 });
        assert!((lo - 204.8).abs() < 12.0 && (hi - (20479.0 - 204.8)).abs() < 12.0, "[{lo}, {hi}]");
        // Uniform: clipping costs more than rounding, so MSE keeps (about) the whole range.
        let [lo, hi] = s.range(Method::Mse);
        assert!(lo == 0.0 && hi > 20479.0 * 0.98, "[{lo}, {hi}]");
        // A million values in [0, 100) and one at 10 000: MSE trades the outlier's clipping, (10⁴ − h)², against
        // the rounding of the rest, 10⁶ · (h / 255)² / 12, which is least at h ≈ 4 383; min/max keeps 10 000.
        let mut v: Vec<f32> = (0..1_000_000).map(|i| (i % 100) as f32).collect();
        v.push(10_000.0);
        let s = stats(&v);
        assert_eq!(s.range(Method::MinMax), [0.0, 10_000.0]);
        let [lo, hi] = s.range(Method::Mse);
        assert!(lo == 0.0 && (3_900.0..4_900.0).contains(&hi), "[{lo}, {hi}]");
    }
}
