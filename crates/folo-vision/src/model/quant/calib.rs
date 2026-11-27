//! Range calibration (docs/MODEL.md §9.6): the `f32` model over sample images, per-tensor and per-channel
//! histograms of every quantised boundary, and a range per boundary (and per channel in `version: 2`) chosen
//! from them.
//!
//! Two passes over the same images: the first records each boundary's minimum and maximum, the second a
//! histogram of [`BINS`] (per-tensor) and [`CH_BINS`] (per-channel) bins between them. [`Calibrator::finish`]
//! then picks each range by [`Method`].

use std::collections::BTreeMap;

use folo_core::Tensor;
use folo_nn::{ConvBnAct, Forward, Result};
use serde::{Deserialize, Serialize};

use super::QParam;
use super::walk::{self, Exec};
use crate::model::FoloModel;
use crate::model::blocks::{C3k2, C3k2Block};

/// Per-tensor histogram bins per boundary.
pub const BINS: usize = 2048;
/// Per-channel histogram bins per boundary (`version: 2`).
pub const CH_BINS: usize = 256;

/// Schema version 1: per-tensor `ranges` only, `f32` stem.
pub const VERSION_V1: u32 = 1;
/// Schema version 2: per-channel C4 `channel_ranges`, per-part concat scale decoupling, and direct `u8` stem.
pub const VERSION_V2: u32 = 2;

fn default_version() -> u32 {
    VERSION_V1
}

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

/// The stored result of calibration: a `[lo, hi]` range per boundary name ([`walk::Exec`]), plus optional
/// per-channel ranges and direct `u8` stem opt-in (`version: 2`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    /// Schema version (`1`: per-tensor; `2`: per-channel C4 grids + direct `u8` stem).
    #[serde(default = "default_version")]
    pub version: u32,
    /// How the ranges were chosen.
    pub method: Method,
    /// Images the ranges were measured on.
    pub images: usize,
    /// Boundary name to per-tensor `[lo, hi]`.
    pub ranges: BTreeMap<String, [f32; 2]>,
    /// Per-channel `[lo, hi]` ranges per boundary name (`version >= 2`; empty in `version: 1`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub channel_ranges: BTreeMap<String, Vec<[f32; 2]>>,
    /// Run the 3-channel stem directly on `u8` pixels (`scale = 1/255, zero = 0`) via AVX-VNNI.
    #[serde(default)]
    pub u8_stem: bool,
    /// Convs that keep `f32` weights (mixed precision): their inputs are dequantised and their outputs
    /// quantised again.
    #[serde(default)]
    pub f32_convs: Vec<String>,
    /// The [`super::fingerprint`] of the weights the ranges were measured on (`None`: not checked).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
}

fn range_to_param(name: &str, [lo, hi]: [f32; 2]) -> Result<QParam> {
    let g = (lo.is_finite() && hi.is_finite() && lo <= hi).then(|| QParam::from_range(lo, hi));
    match g {
        Some(g) if g.scale.is_finite() && g.inv().is_finite() => Ok(g),
        _ => Err(folo_nn::Error::Invalid(format!(
            "int8 calibration range of `{name}` is [{lo}, {hi}]: re-run `folo quantize`"
        ))),
    }
}

impl Calibration {
    /// Validates the calibration schema version (`1..=2`).
    pub fn check_version(&self) -> Result<()> {
        if (VERSION_V1..=VERSION_V2).contains(&self.version) {
            Ok(())
        } else {
            Err(folo_nn::Error::Metadata(format!(
                "unsupported `{}` calibration version {} (supported: {VERSION_V1}..={VERSION_V2})",
                super::META_KEY,
                self.version
            )))
        }
    }

    /// The per-tensor grid of boundary `name`: its range must be finite with `lo <= hi`, and its width too (so
    /// the grid's step and its inverse are finite; a zero-width range gets a tiny step).
    pub fn param(&self, name: &str) -> Result<QParam> {
        self.check_version()?;
        let r = self.ranges.get(name).copied().ok_or_else(|| {
            folo_nn::Error::Invalid(format!(
                "int8 calibration has no range for `{name}`: re-run `folo quantize`"
            ))
        })?;
        range_to_param(name, r)
    }

    /// Per-channel grids of length `c` for boundary `name`. Falls back cleanly to `[self.param(name)?; c]` on
    /// `version: 1` calibrations (or when `channel_ranges` has no entry for `name`).
    pub fn channel_params(&self, name: &str, c: usize) -> Result<Vec<QParam>> {
        let fallback = self.param(name)?;
        let Some(crs) = self.channel_ranges.get(name) else {
            return Ok(vec![fallback; c]);
        };
        if crs.len() != c {
            return Err(folo_nn::Error::Invalid(format!(
                "int8 calibration channel_ranges of `{name}` has {} channels, expected {c}",
                crs.len()
            )));
        }
        crs.iter().map(|&r| range_to_param(name, r)).collect()
    }

    /// Whether conv `name` keeps `f32` weights.
    pub fn is_f32(&self, name: &str) -> bool {
        self.f32_convs.iter().any(|c| c == name)
    }

    /// Adds the sensitivity-default P5 tail (`neck.out5.*`) conv names to `f32_convs` (deduplicated).
    pub fn with_p5_tail_f32(mut self, model: &FoloModel) -> Self {
        for name in default_sensitivity_f32_convs(model) {
            if !self.is_f32(&name) {
                self.f32_convs.push(name);
            }
        }
        self
    }

    /// Selects and appends sensitivity-ranked `f32_convs` under `policy` (folded/unfolded weight SQNR vs MAC
    /// budget with residual bottleneck pairing, deduplicated).
    pub fn with_sensitivity_f32(
        mut self,
        model: &FoloModel,
        policy: super::SensitivityPolicy,
    ) -> Result<Self> {
        for name in super::select_f32_convs(model, &self, policy)? {
            if !self.is_f32(&name) {
                self.f32_convs.push(name);
            }
        }
        Ok(self)
    }
}

/// Sensitivity + cost-model default `f32_convs`: the P5 neck tail (`neck.out5.*`), which both has the lowest
/// weight SQNR in `folo-n`/`folo-s` (36.5–37.7 dB) and sits at 20×20 directly feeding the `f32` head (zero extra
/// requantisation cost).
pub fn default_sensitivity_f32_convs(model: &FoloModel) -> Vec<String> {
    let mut out = Vec::new();
    collect_c3k2_names("neck.out5", &model.neck.out5, &mut out);
    out
}

fn collect_c3k2_names(prefix: &str, b: &C3k2, out: &mut Vec<String>) {
    out.push(format!("{prefix}.cv1"));
    for (j, blk) in b.blocks.iter().enumerate() {
        let bp = format!("{prefix}.block{j}");
        match blk {
            C3k2Block::Bottleneck(_) => {
                out.push(format!("{bp}.cv1"));
                out.push(format!("{bp}.cv2"));
            }
            C3k2Block::C3k(ck) => {
                out.push(format!("{bp}.cv1"));
                for k in 0..ck.blocks.len() {
                    out.push(format!("{bp}.block{k}.cv1"));
                    out.push(format!("{bp}.block{k}.cv2"));
                }
                out.push(format!("{bp}.cv2"));
                out.push(format!("{bp}.cv3"));
            }
        }
    }
    out.push(format!("{prefix}.cv2"));
}

#[derive(Default)]
struct Stats {
    min: f32,
    max: f32,
    seen: bool,
    hist: Vec<u64>,
}

impl Stats {
    fn add(&mut self, v: &[f32], pass: u8, bins: usize) {
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
            self.hist = vec![0; bins];
        }
        let span = (self.max as f64 - self.min as f64).max(1e-12);
        let (lo, w) = (self.min as f64, span / bins as f64);
        for &x in v {
            let b = (((x as f64 - lo) / w) as isize).clamp(0, bins as isize - 1) as usize;
            self.hist[b] += 1;
        }
    }

    /// `[lo, hi]` by `method`.
    fn range(&self, method: Method) -> [f32; 2] {
        let bins = self.hist.len();
        if bins == 0 || method == Method::MinMax || self.max <= self.min {
            return [self.min, self.max];
        }
        let w = (self.max as f64 - self.min as f64) / bins as f64;
        let edge = |i: usize| self.min as f64 + w * i as f64;
        let total: u64 = self.hist.iter().sum();
        match method {
            Method::MinMax => unreachable!(),
            Method::Percentile { p } => {
                let cut = (p.clamp(0.0, 0.5) * total as f64) as u64;
                let (mut lo, mut acc) = (0usize, 0u64);
                while lo < bins && acc + self.hist[lo] <= cut {
                    acc += self.hist[lo];
                    lo += 1;
                }
                let (mut hi, mut acc) = (bins, 0u64);
                while hi > 0 && acc + self.hist[hi - 1] <= cut {
                    acc += self.hist[hi - 1];
                    hi -= 1;
                }
                let lo = lo.min(bins - 1);
                let hi = hi.clamp(lo + 1, bins);
                [edge(lo) as f32, edge(hi) as f32]
            }
            Method::Mse => self.mse_range(w, bins),
        }
    }

    /// Search `lo` over the lowest and `hi` over the highest edges with prefix sums.
    fn mse_range(&self, w: f64, n: usize) -> [f32; 2] {
        let c = |i: usize| self.min as f64 + w * (i as f64 + 0.5);
        let (mut s0, mut s1, mut s2) = (vec![0f64; n + 1], vec![0f64; n + 1], vec![0f64; n + 1]);
        for i in 0..n {
            let (h, x) = (self.hist[i] as f64, c(i));
            s0[i + 1] = s0[i] + h;
            s1[i + 1] = s1[i] + h * x;
            s2[i + 1] = s2[i] + h * x * x;
        }
        let edge = |i: usize| self.min as f64 + w * i as f64;
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

#[derive(Default)]
struct BoundaryStats {
    tensor: Stats,
    channels: Vec<Stats>,
}

/// Collects the boundary statistics of calibration images: [`Calibrator::observe`] every batch, call
/// [`Calibrator::next_pass`], observe the same batches again, then [`Calibrator::finish`].
#[derive(Default)]
pub struct Calibrator {
    stats: BTreeMap<String, BoundaryStats>,
    pass: u8,
    /// Images observed in each pass.
    images: [usize; 2],
    /// An observation failed part way (its statistics are partial): the calibration is refused.
    failed: bool,
}

fn validate_method(method: Method) -> Result<()> {
    if let Method::Percentile { p } = method
        && (!p.is_finite() || !(0.0..0.5).contains(&p))
    {
        return Err(folo_nn::Error::Invalid(format!(
            "int8 calibration percentile p must be finite and in [0, 0.5), got {p}"
        )));
    }
    Ok(())
}

impl Calibrator {
    /// An empty calibrator in its first pass.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the `f32` model on `x` (`[B, 3, H, W]`, `f32`, on the CPU) and record every boundary.
    pub fn observe(&mut self, model: &FoloModel, x: &Tensor) -> Result<()> {
        let _g = folo_core::no_grad();
        let mut e = Observe { cal: self };
        let r = folo_core::cpu::in_pool(|| walk::features(model, &mut e, x));
        match r {
            Ok(_) => self.images[self.pass as usize] += x.dims()[0],
            Err(_) => self.failed = true,
        }
        r.map(|_| ())
    }

    /// Start the histogram pass.
    pub fn next_pass(&mut self) {
        self.pass = 1;
    }

    /// The ranges by `method` (`version: 2`, with per-channel C4 ranges and `u8_stem: true`). Refused unless
    /// both passes observed the same images, at least one, and no observation failed.
    pub fn finish(&self, method: Method) -> Result<Calibration> {
        let [a, b] = self.images;
        if self.failed {
            return Err(folo_nn::Error::Invalid("int8 calibration: an observation failed".into()));
        }
        if a == 0 || self.pass != 1 || a != b {
            return Err(folo_nn::Error::Invalid(format!(
                "int8 calibration needs two passes over the same images, at least one (saw {a}, then {b})"
            )));
        }
        validate_method(method)?;
        let ranges = self.stats.iter().map(|(k, s)| (k.clone(), s.tensor.range(method))).collect();
        let channel_ranges = self
            .stats
            .iter()
            .map(|(k, s)| (k.clone(), s.channels.iter().map(|cs| cs.range(method)).collect()))
            .collect();
        Ok(Calibration {
            version: VERSION_V2,
            method,
            images: a,
            ranges,
            channel_ranges,
            u8_stem: true,
            f32_convs: Vec::new(),
            weights: None,
        })
    }

    fn record(&mut self, name: &str, t: &Tensor) -> Result<()> {
        let (n, c, h, w) = t.dims4()?;
        let hw = h * w;
        let v = t.to_vec::<f32>()?;
        if let Some(&bad) = v.iter().find(|x| !x.is_finite()) {
            return Err(folo_nn::Error::Invalid(format!(
                "int8 calibration: non-finite activation ({bad}) at `{name}`"
            )));
        }
        let pass = self.pass;
        let entry = self.stats.entry(name.to_string()).or_default();
        entry.tensor.add(&v, pass, BINS);
        if entry.channels.len() != c {
            entry.channels = (0..c).map(|_| Stats::default()).collect();
        }
        for img in 0..n {
            for (ch, ch_stats) in entry.channels.iter_mut().enumerate().take(c) {
                let base = (img * c + ch) * hw;
                ch_stats.add(&v[base..base + hw], pass, CH_BINS);
            }
        }
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
    validate_method(method)?;
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
/// to `imgsz` as validation does, in batches of `batch` read by `workers` threads (`folo quantize`). Applies the
/// sensitivity-aware selective-`f32` policy ([`Calibration::with_sensitivity_f32`] with
/// [`super::SensitivityPolicy::default`]). Returns the calibration and the number of images.
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
    let cal = calibrate(model, batches, method)?
        .with_sensitivity_f32(model, super::SensitivityPolicy::default())?;
    Ok((cal, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(v: &[f32]) -> Stats {
        let mut s = Stats::default();
        s.add(v, 0, BINS);
        s.add(v, 1, BINS);
        s
    }

    #[test]
    fn ranges_have_their_known_answers() {
        let v: Vec<f32> = (0..20480).map(|i| i as f32).collect();
        let s = stats(&v);
        assert_eq!(s.range(Method::MinMax), [0.0, 20479.0]);
        assert_eq!(s.range(Method::Percentile { p: 0.0 }), [0.0, 20479.0]);
        let [lo, hi] = s.range(Method::Percentile { p: 0.01 });
        assert!((lo - 204.8).abs() < 12.0 && (hi - (20479.0 - 204.8)).abs() < 12.0, "[{lo}, {hi}]");
        let [lo, hi] = s.range(Method::Mse);
        assert!(lo == 0.0 && hi > 20479.0 * 0.98, "[{lo}, {hi}]");
        let mut v: Vec<f32> = (0..1_000_000).map(|i| (i % 100) as f32).collect();
        v.push(10_000.0);
        let s = stats(&v);
        assert_eq!(s.range(Method::MinMax), [0.0, 10_000.0]);
        let [lo, hi] = s.range(Method::Mse);
        assert!(lo == 0.0 && (3_900.0..4_900.0).contains(&hi), "[{lo}, {hi}]");
    }

    #[test]
    fn stats_handles_constant_channels_wide_spans_and_clamped_percentile() {
        // Constant (e.g. dead ReLU) channel: w == 0 must not produce NaN in Percentile or Mse.
        let c = stats(&[0.0; 64]);
        assert_eq!(c.range(Method::MinMax), [0.0, 0.0]);
        assert_eq!(c.range(Method::Percentile { p: 0.01 }), [0.0, 0.0]);
        assert_eq!(c.range(Method::Mse), [0.0, 0.0]);

        let c_pos = stats(&[3.5; 64]);
        assert_eq!(c_pos.range(Method::Percentile { p: 0.01 }), [3.5, 3.5]);
        assert_eq!(c_pos.range(Method::Mse), [3.5, 3.5]);

        // Wide span exceeding f32::MAX must not overflow to Inf/NaN in histogramming or range search.
        let wide = stats(&[-2.0e38, -1.0e38, 0.0, 1.0e38, 2.0e38]);
        for m in [Method::MinMax, Method::Percentile { p: 0.01 }, Method::Mse] {
            let [lo, hi] = wide.range(m);
            assert!(lo.is_finite() && hi.is_finite() && lo <= hi, "{m:?} -> [{lo}, {hi}]");
            assert!(lo >= -2.0e38 && hi <= 2.0e38, "{m:?} -> [{lo}, {hi}]");
        }

        // Clamped internal percentile never walks past `bins` or emits `hi > max`.
        let unit = stats(&[0.0, 1.0, 2.0, 3.0]);
        let [lo, hi] = unit.range(Method::Percentile { p: 0.5 });
        assert!(lo.is_finite() && hi.is_finite() && lo < hi && lo >= 0.0 && hi <= 3.0, "[{lo}, {hi}]");
    }

    #[test]
    fn calibrator_rejects_invalid_percentile_and_nonfinite_activations() {
        let mut cal = Calibrator::new();
        let ok = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], [1, 1, 2, 2]).unwrap();
        cal.record("stem", &ok).unwrap();
        cal.images[0] = 1;
        cal.next_pass();
        cal.record("stem", &ok).unwrap();
        cal.images[1] = 1;

        for bad_p in [-0.01, 0.5, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = cal.finish(Method::Percentile { p: bad_p }).unwrap_err();
            assert!(
                err.to_string().contains("percentile p"),
                "expected percentile validation error for {bad_p}, got: {err}"
            );
        }
        assert!(cal.finish(Method::Percentile { p: 0.0 }).is_ok());
        assert!(cal.finish(Method::Percentile { p: 0.49 }).is_ok());

        let mut bad_cal = Calibrator::new();
        let nan_t = Tensor::from_vec(vec![1.0f32, f32::NAN, 2.0, 3.0], [1, 1, 2, 2]).unwrap();
        let err = bad_cal.record("stem", &nan_t).unwrap_err();
        assert!(err.to_string().contains("non-finite"), "{err}");

        let inf_t = Tensor::from_vec(vec![1.0f32, f32::INFINITY, 2.0, 3.0], [1, 1, 2, 2]).unwrap();
        let err = bad_cal.record("stem", &inf_t).unwrap_err();
        assert!(err.to_string().contains("non-finite"), "{err}");
    }
}
