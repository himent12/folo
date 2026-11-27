//! Sensitivity-aware selective `f32` policy for int8 calibration (docs/MODEL.md §9.6).
//!
//! Computes each eligible conv's real quantisation distortion (folded-weight and scale-unfolded weight SQNR using
//! the model's fused weights and calibrated input grids, honouring `version >= 2` per-part concat decoupling)
//! alongside its relative MAC share at a reference `640×640` input. [`select_f32_convs`] greedily promotes the
//! highest-sensitivity convs within a tight MAC budget — pairing residual bottleneck convs so no extra `u8 <-> f32`
//! round-trip is introduced inside a bottleneck, and optionally keeping the head-facing `neck.out5.cv2` in `f32`
//! (zero extra requantisation cost because it writes directly into the `f32` head).

use std::collections::HashSet;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, Grid, quantize_weights, should_decouple_parts};
use folo_nn::{ConvBnAct, Result};

use super::walk::{self, Exec};
use super::{Calibration, VERSION_V2, int8_able};
use crate::model::FoloModel;
use crate::model::blocks::Attention;

/// Reference input resolution (`640×640`) used to compute relative per-conv MAC fractions during sensitivity
/// analysis.
const REF_IMGSZ: usize = 640;

/// Per-conv quantisation sensitivity and compute cost measured by [`analyze_sensitivity`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConvSensitivity {
    /// Conv prefix in the `.folo` graph (e.g. `"backbone.stage2.block0.cv1"`, `"neck.out5.cv2"`).
    pub name: String,
    /// Quantisation SQNR of the input-scale-folded `s8` weights (`10 · log10(||w'||² / ||w' − q(w')||²)`), dB.
    pub folded_weight_sqnr_db: f64,
    /// Quantisation SQNR after dividing folded `s8` weights back by each input channel's activation scale, dB.
    /// Diagnostic metric capturing per-channel dynamic-range spread across input channels.
    pub unfolded_weight_sqnr_db: f64,
    /// Multiply-accumulate operations of this conv at a single `640×640` image.
    pub macs: u64,
    /// Fraction of total eligible backbone+neck MACs (`macs / total_macs`).
    pub mac_fraction: f64,
    /// Output spatial size `(h, w)` at a `640×640` input.
    pub out_hw: (usize, usize),
    /// Companion conv in the same residual bottleneck (`cv1 <-> cv2`), if any, so both can be promoted together
    /// without an internal `u8 <-> f32` round-trip.
    pub residual_partner: Option<String>,
    /// Composite priority score (higher = stronger candidate for `f32_convs`).
    pub score: f64,
}

/// Configuration for [`select_f32_convs`] and [`Calibration::with_sensitivity_f32`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensitivityPolicy {
    /// Maximum fraction of total eligible backbone+neck MACs allowed in `f32_convs` (default `0.05`, i.e. 5%).
    pub max_f32_mac_fraction: f64,
    /// Only promote convs whose [`ConvSensitivity::folded_weight_sqnr_db`] is below this threshold in dB
    /// (default `35.0`: promotes genuinely outlier-distorted layers without needlessly pulling well-quantised
    /// `36–39 dB` layers out of int8).
    pub sqnr_threshold_db: f64,
    /// Always include the head-facing P5 output conv `neck.out5.cv2` when within budget (zero extra requantisation
    /// cost because it feeds only the `f32` head).
    pub include_p5_head_out: bool,
    /// When promoting a residual bottleneck's `cv1` or `cv2`, treat the pair atomically (promote both together
    /// only when both fit in the MAC budget, avoiding an extra `u8 <-> f32` round-trip inside the bottleneck).
    pub pair_residual_bottlenecks: bool,
}

impl Default for SensitivityPolicy {
    fn default() -> Self {
        Self {
            max_f32_mac_fraction: 0.05,
            sqnr_threshold_db: 35.0,
            include_p5_head_out: true,
            pair_residual_bottlenecks: true,
        }
    }
}

impl SensitivityPolicy {
    /// Validates policy bounds (`max_f32_mac_fraction` finite and in `0.0..=1.0`, `sqnr_threshold_db` not `NaN`).
    pub fn validate(&self) -> Result<()> {
        if !self.max_f32_mac_fraction.is_finite() || !(0.0..=1.0).contains(&self.max_f32_mac_fraction) {
            return Err(folo_nn::Error::Invalid(format!(
                "SensitivityPolicy::max_f32_mac_fraction must be finite and in 0.0..=1.0, got {}",
                self.max_f32_mac_fraction
            )));
        }
        if self.sqnr_threshold_db.is_nan() {
            return Err(folo_nn::Error::Invalid(
                "SensitivityPolicy::sqnr_threshold_db must not be NaN".into(),
            ));
        }
        Ok(())
    }
}

/// Symbolic activation passed through [`walk::features`] during sensitivity analysis (zero tensor compute).
#[derive(Clone)]
struct SymAct {
    dims: [usize; 4],
    grids: Vec<Grid>,
}

struct Analyzer<'a> {
    cal: &'a Calibration,
    input_hw: (usize, usize),
    pending_attn_dims: Option<[usize; 4]>,
    entries: Vec<ConvSensitivity>,
}

impl Analyzer<'_> {
    fn inspect_conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&SymAct],
        res: Option<&SymAct>,
    ) -> Result<Option<(usize, usize)>> {
        let [n, _, h, w] = parts[0].dims;
        let c: usize = parts.iter().map(|a| a.dims[1]).sum();
        let cfg = conv.conv.config();
        let o = cfg.out_channels;
        let able = parts
            .iter()
            .all(|a| a.dims[1].is_multiple_of(4))
            .then(|| int8_able(conv, [n, c, h, w], res.map(|r| r.dims)))
            .flatten();
        let Some((oh, ow)) = able else {
            let wd = conv.conv.weight.dims();
            let (kh, kw) = if wd.len() == 4 { (wd[2], wd[3]) } else { (cfg.kernel, cfg.kernel) };
            let oh = (h + 2 * cfg.padding)
                .checked_sub(cfg.dilation * kh.saturating_sub(1) + 1)
                .map(|v| v / cfg.stride.max(1) + 1);
            let ow = (w + 2 * cfg.padding)
                .checked_sub(cfg.dilation * kw.saturating_sub(1) + 1)
                .map(|v| v / cfg.stride.max(1) + 1);
            return Ok(oh.zip(ow));
        };

        let taps = cfg.kernel * cfg.kernel;
        let macs = (c as u64)
            .saturating_mul(o as u64)
            .saturating_mul(taps as u64)
            .saturating_mul(oh as u64)
            .saturating_mul(ow as u64);

        let orig_w = conv.conv.weight.to_dtype(folo_core::DType::F32)?.to_vec::<f32>()?;
        let part_grids: Vec<&[Grid]> = parts.iter().map(|a| a.grids.as_slice()).collect();
        let decouple = self.cal.version >= VERSION_V2 && should_decouple_parts(&part_grids);
        let (folded_sqnr, unfolded_sqnr) = measure_weight_sqnr(&orig_w, o, c, taps, &part_grids, decouple);

        // Identify residual partner for residual bottlenecks (`<prefix>.cv1` <-> `<prefix>.cv2`).
        let residual_partner = if res.is_some()
            && let Some(prefix) = name.strip_suffix(".cv2")
            && prefix.contains(".block")
            && !prefix.contains(".psa.")
        {
            let partner = format!("{prefix}.cv1");
            if let Some(prev) = self.entries.iter_mut().rev().find(|e| e.name == partner) {
                prev.residual_partner = Some(name.to_string());
            }
            Some(partner)
        } else {
            None
        };

        self.entries.push(ConvSensitivity {
            name: name.to_string(),
            folded_weight_sqnr_db: folded_sqnr,
            unfolded_weight_sqnr_db: unfolded_sqnr,
            macs,
            mac_fraction: 0.0,
            out_hw: (oh, ow),
            residual_partner,
            score: 0.0,
        });
        Ok(Some((oh, ow)))
    }
}

/// Measure both folded (`w' = w * s_in`) and scale-unfolded (`q(w') / s_in`) `s8` weight SQNR in dB, mirroring
/// `QConv::new_parts`'s exact per-part quantisation.
fn measure_weight_sqnr(
    orig: &[f32],
    o: usize,
    c: usize,
    taps: usize,
    part_grids: &[&[Grid]],
    decouple: bool,
) -> (f64, f64) {
    let mut f_sig = 0.0f64;
    let mut f_err = 0.0f64;
    let mut u_sig = 0.0f64;
    let mut u_err = 0.0f64;

    let mut eval_slice = |ch0: usize, pg: &[Grid]| {
        let pc = pg.len();
        let mut folded = Vec::with_capacity(o * pc * taps);
        for oc in 0..o {
            let base = (oc * c + ch0) * taps;
            for (ic, g) in pg.iter().enumerate() {
                let s = g.scale;
                folded.extend(orig[base + ic * taps..base + (ic + 1) * taps].iter().map(|&v| v * s));
            }
        }
        let (q, scales) = quantize_weights(&folded, o);
        let k = pc * taps;
        for (oc, &sc) in scales.iter().enumerate().take(o) {
            let s_o = f64::from(sc);
            let orig_base = (oc * c + ch0) * taps;
            for (ic, g) in pg.iter().enumerate() {
                let s_i = f64::from(g.scale).max(1e-12);
                for t in 0..taps {
                    let fi = oc * k + ic * taps + t;
                    let fv = f64::from(folded[fi]);
                    let fq = f64::from(q[fi]) * s_o;
                    let fd = fv - fq;
                    f_sig += fv * fv;
                    f_err += fd * fd;

                    let ov = f64::from(orig[orig_base + ic * taps + t]);
                    let rv = fq / s_i;
                    let ud = ov - rv;
                    u_sig += ov * ov;
                    u_err += ud * ud;
                }
            }
        }
    };

    if decouple && part_grids.len() > 1 {
        let mut ch0 = 0;
        for pg in part_grids {
            eval_slice(ch0, pg);
            ch0 += pg.len();
        }
    } else {
        let all: Vec<Grid> = part_grids.iter().flat_map(|pg| pg.iter().copied()).collect();
        eval_slice(0, &all);
    }

    let to_db = |sig: f64, err: f64| {
        if sig == 0.0 || err == 0.0 { 100.0 } else { 10.0 * (sig / err).log10() }
    };
    (to_db(f_sig, f_err), to_db(u_sig, u_err))
}

/// Boundary priority multiplier: `neck.out5.cv2` writes straight into the `f32` head with zero extra
/// requantisation cost; all other convs start at `1.0` and are penalised in [`analyze_sensitivity_at`] by both
/// MAC share and output spatial area (high-resolution `u8 <-> f32` conversion bandwidth).
fn boundary_priority(name: &str) -> f64 {
    if name == "neck.out5.cv2" { 1.3 } else { 1.0 }
}

impl Exec for Analyzer<'_> {
    type A = SymAct;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<SymAct> {
        let n = x.dims()[0];
        let (h, w) = self.input_hw;
        let cfg = conv.conv.config();
        let (oh, ow) = q8::out_size(h, w, cfg.kernel, cfg.stride, cfg.padding).unwrap_or((h / 2, w / 2));
        let grids = self.cal.channel_params(name, cfg.out_channels)?;
        Ok(SymAct { dims: [n, cfg.out_channels, oh, ow], grids })
    }

    fn conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&SymAct],
        res: Option<&SymAct>,
    ) -> Result<SymAct> {
        let [n, _, h, w] = parts[0].dims;
        let cfg = conv.conv.config();
        let (oh, ow) = self.inspect_conv(name, conv, parts, res)?.unwrap_or((h, w));
        // Mirror `f32` residual broadcasting (e.g. a 1×1 conv output added to a 2×2 residual produces 2×2).
        let (oh, ow) = match res {
            Some(r) => (oh.max(r.dims[2]), ow.max(r.dims[3])),
            None => (oh, ow),
        };
        let grids = self.cal.channel_params(name, cfg.out_channels)?;
        Ok(SymAct { dims: [n, cfg.out_channels, oh, ow], grids })
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &SymAct) -> Result<Tensor> {
        let _ = self.inspect_conv(name, conv, &[x], None)?;
        Ok(Tensor::zeros([0]))
    }

    fn attend_and_pe(
        &mut self,
        a: &Attention,
        _qkv: &Tensor,
        b: usize,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        // Shape-only bypass: never invokes `a.pe.forward`, so unfused training-mode BatchNorm stats stay
        // untouched and GPU/non-F32 models do not run tensor ops during sensitivity analysis.
        self.pending_attn_dims = Some([b, a.proj.conv.config().in_channels, h, w]);
        Ok(Tensor::zeros([0]))
    }

    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&SymAct]) -> Result<Tensor> {
        let cfg = conv.conv.config();
        let _ = self.inspect_conv(name, conv, parts, None)?;
        let _ = self.cal.channel_params(name, cfg.out_channels)?;
        Ok(Tensor::zeros([0]))
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<SymAct> {
        let [n, c, h, w] = match self.pending_attn_dims.take() {
            Some(d) => d,
            None => {
                let (n, c, h, w) = x.dims4()?;
                [n, c, h, w]
            }
        };
        let grids = self.cal.channel_params(name, c)?;
        Ok(SymAct { dims: [n, c, h, w], grids })
    }

    fn dequantize(&mut self, _x: &SymAct) -> Result<Tensor> {
        Ok(Tensor::zeros([0]))
    }

    fn narrow(&mut self, x: &SymAct, start: usize, len: usize) -> Result<SymAct> {
        let [n, _, h, w] = x.dims;
        Ok(SymAct { dims: [n, len, h, w], grids: x.grids[start..start + len].to_vec() })
    }

    fn max_pool(&mut self, x: &SymAct, _k: usize) -> Result<SymAct> {
        Ok(x.clone())
    }

    fn upsample2(&mut self, x: &SymAct) -> Result<SymAct> {
        let [n, c, h, w] = x.dims;
        Ok(SymAct { dims: [n, c, 2 * h, 2 * w], grids: x.grids.clone() })
    }

    fn dims(&self, x: &SymAct) -> [usize; 4] {
        x.dims
    }
}

/// Analyse per-conv quantisation sensitivity across `model`'s int8-eligible backbone and neck convolutions using
/// `cal`'s calibrated activation grids at the `640×640` reference resolution. Results are sorted descending by
/// [`ConvSensitivity::score`].
///
/// Purely symbolic (zero convolutions, zero feature-map allocations, and no mutation of unfused BatchNorm stats).
pub fn analyze_sensitivity(model: &FoloModel, cal: &Calibration) -> Result<Vec<ConvSensitivity>> {
    analyze_sensitivity_at(model, cal, (REF_IMGSZ, REF_IMGSZ))
}

/// Like [`analyze_sensitivity`], evaluating exact per-conv spatial geometry and MACs at `input_hw = (h, w)`.
pub fn analyze_sensitivity_at(
    model: &FoloModel,
    cal: &Calibration,
    input_hw: (usize, usize),
) -> Result<Vec<ConvSensitivity>> {
    cal.check_version()?;
    if input_hw.0 == 0 || input_hw.1 == 0 {
        return Err(folo_nn::Error::Invalid(format!(
            "sensitivity analysis input_hw must be positive, got {}x{}",
            input_hw.0, input_hw.1
        )));
    }
    let _g = folo_core::no_grad();
    let x = Tensor::zeros([1, model.config().in_channels, 1, 1]);
    let mut analyzer = Analyzer { cal, input_hw, pending_attn_dims: None, entries: Vec::new() };
    let _ = walk::features(model, &mut analyzer, &x)?;

    let mut total_macs = 0u64;
    for e in &analyzer.entries {
        total_macs = total_macs.saturating_add(e.macs);
    }
    let total_f = (total_macs.max(1)) as f64;
    let input_area = (input_hw.0.saturating_mul(input_hw.1)).max(1) as f64;
    for e in &mut analyzer.entries {
        e.mac_fraction = e.macs as f64 / total_f;
        let distortion = (42.0 - e.folded_weight_sqnr_db).max(0.0);
        let spatial_cost = (e.out_hw.0.saturating_mul(e.out_hw.1)) as f64 / input_area;
        e.score = distortion * boundary_priority(&e.name) / (e.mac_fraction + spatial_cost + 0.015);
    }
    analyzer.entries.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.name.cmp(&b.name)));
    Ok(analyzer.entries)
}

/// Select `f32_convs` for `model` and `cal` under `policy` (sensitivity-ranked, MAC-budgeted, with atomic residual
/// bottleneck pairing to avoid redundant internal `u8 <-> f32` round-trips).
pub fn select_f32_convs(
    model: &FoloModel,
    cal: &Calibration,
    policy: SensitivityPolicy,
) -> Result<Vec<String>> {
    select_f32_convs_with_existing(model, cal, policy, &[])
}

/// Like [`select_f32_convs`], accounting for `existing` `f32_convs` against `policy.max_f32_mac_fraction` and
/// returning only the newly selected conv names that still fit within the remaining budget.
pub(crate) fn select_f32_convs_with_existing(
    model: &FoloModel,
    cal: &Calibration,
    policy: SensitivityPolicy,
    existing: &[String],
) -> Result<Vec<String>> {
    policy.validate()?;
    cal.check_version()?;
    let budget = policy.max_f32_mac_fraction;
    if budget <= 0.0 {
        return Ok(Vec::new());
    }
    let ranked = analyze_sensitivity(model, cal)?;
    let by_name: std::collections::HashMap<&str, &ConvSensitivity> =
        ranked.iter().map(|e| (e.name.as_str(), e)).collect();

    let mut selected: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut used_frac = 0.0f64;

    for name in existing {
        if seen.insert(name.clone())
            && let Some(entry) = by_name.get(name.as_str())
        {
            used_frac += entry.mac_fraction;
        }
    }
    if used_frac >= budget {
        return Ok(Vec::new());
    }

    let try_add = |names: &[&str], used: &mut f64, out: &mut Vec<String>, seen: &mut HashSet<String>| {
        let mut add_frac = 0.0f64;
        for &n in names {
            if !seen.contains(n)
                && let Some(entry) = by_name.get(n)
            {
                add_frac += entry.mac_fraction;
            }
        }
        if *used + add_frac > budget + 1e-9 {
            return false;
        }
        for &n in names {
            if seen.insert(n.to_string()) {
                out.push(n.to_string());
            }
        }
        *used += add_frac;
        true
    };

    // 1. Head-facing P5 output conv (`neck.out5.cv2`): zero extra requantisation cost (writes f32 straight to head).
    if policy.include_p5_head_out && !seen.contains("neck.out5.cv2") && by_name.contains_key("neck.out5.cv2")
    {
        try_add(&["neck.out5.cv2"], &mut used_frac, &mut selected, &mut seen);
    }

    // 2. Greedily promote the highest-sensitivity convs whose folded-weight SQNR falls below `sqnr_threshold_db`.
    //    Residual bottleneck pairs `(cv1, cv2)` are promoted atomically when `pair_residual_bottlenecks` is set so
    //    we never introduce a redundant internal `u8 <-> f32` conversion between `cv1` and `cv2`.
    for entry in &ranked {
        if entry.folded_weight_sqnr_db.is_nan() || entry.folded_weight_sqnr_db >= policy.sqnr_threshold_db {
            continue;
        }
        if seen.contains(&entry.name) {
            continue;
        }
        if policy.pair_residual_bottlenecks
            && let Some(partner) = entry.residual_partner.as_deref()
            && by_name.contains_key(partner)
        {
            try_add(&[entry.name.as_str(), partner], &mut used_frac, &mut selected, &mut seen);
        } else {
            try_add(&[entry.name.as_str()], &mut used_frac, &mut selected, &mut seen);
        }
    }

    selected.sort();
    Ok(selected)
}
