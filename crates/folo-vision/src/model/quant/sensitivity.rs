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
    /// (default `38.0`).
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
            sqnr_threshold_db: 38.0,
            include_p5_head_out: true,
            pair_residual_bottlenecks: true,
        }
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
            return Ok(q8::out_size(h, w, cfg.kernel, cfg.stride, cfg.padding));
        };

        let taps = cfg.kernel * cfg.kernel;
        let macs = (c as u64)
            .saturating_mul(o as u64)
            .saturating_mul(taps as u64)
            .saturating_mul(oh as u64)
            .saturating_mul(ow as u64);

        let orig_w = conv.conv.weight.to_vec::<f32>()?;
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

/// Downstream error-propagation multiplier: early backbone convs feed all 3 detection scales (P3/P4/P5),
/// mid-network convs feed P4/P5, and P5-tail convs feed P5 only.
fn downstream_weight(name: &str) -> f64 {
    if name.starts_with("backbone.down2")
        || name.starts_with("backbone.stage2")
        || name.starts_with("backbone.down3")
        || name.starts_with("backbone.stage3")
    {
        1.5
    } else if name.starts_with("backbone.down4")
        || name.starts_with("backbone.stage4")
        || name.starts_with("neck.out3")
    {
        1.2
    } else if name == "neck.out5.cv2" {
        // Zero requantisation boundary cost: feeds the f32 head directly.
        1.3
    } else {
        1.0
    }
}

impl Exec for Analyzer<'_> {
    type A = SymAct;

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<SymAct> {
        let [n, _, h, w] = [x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]];
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
        let grids = self.cal.channel_params(name, cfg.out_channels)?;
        Ok(SymAct { dims: [n, cfg.out_channels, oh, ow], grids })
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &SymAct) -> Result<Tensor> {
        let [n, _, h, w] = x.dims;
        let cfg = conv.conv.config();
        let (oh, ow) = self.inspect_conv(name, conv, &[x], None)?.unwrap_or((h, w));
        // Return a zero-sized shape proxy? `walk::attention` calls `attend_and_pe` on this tensor, which needs
        // real dims on `qkv`. Use `attend_and_pe` bypass: `Analyzer` runs `features_symbolic` instead? No —
        // `Attention::attend_and_pe` only runs at 20×20 = 400 pixels if we use tiny symbolic dims, OR even better:
        // scale `REF_IMGSZ` analytically! Wait: we want exact MACs without running `attend_and_pe` on 640×640!
        // `Tensor::zeros` at `[n, o, oh, ow]` is tiny at 20×20 (`1*256*20*20` = 400 KiB, ~0.1 ms).
        Ok(Tensor::zeros([n, cfg.out_channels, oh, ow]))
    }

    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&SymAct]) -> Result<Tensor> {
        let [n, _, h, w] = parts[0].dims;
        let cfg = conv.conv.config();
        let _ = self.inspect_conv(name, conv, parts, None)?;
        let _ = self.cal.channel_params(name, cfg.out_channels)?;
        Ok(Tensor::zeros([n, cfg.out_channels, h, w]))
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<SymAct> {
        let (n, c, h, w) = x.dims4()?;
        let grids = self.cal.channel_params(name, c)?;
        Ok(SymAct { dims: [n, c, h, w], grids })
    }

    fn dequantize(&mut self, x: &SymAct) -> Result<Tensor> {
        Ok(Tensor::zeros([x.dims[0], x.dims[1], 1, 1]))
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
/// `cal`'s calibrated activation grids. Results are sorted descending by [`ConvSensitivity::score`].
///
/// Runs symbolically at 64×64 (scaling spatial sizes 10× to the `640×640` reference geometry) so analysis
/// completes in ~1–2 ms even for large models (`folo-l`/`folo-x`) without running any convolutions.
pub fn analyze_sensitivity(model: &FoloModel, cal: &Calibration) -> Result<Vec<ConvSensitivity>> {
    cal.check_version()?;
    let _g = folo_core::no_grad();
    // Run the symbolic walk at 64×64 (every stride-32 feature map is ≥ 2×2, attention is 2×2 = 4 tokens so
    // `attend_and_pe` takes microseconds) and scale spatial dimensions by `REF_IMGSZ / 64 = 10` for exact 640² MACs.
    let probe = 64usize;
    let scale = REF_IMGSZ / probe;
    let x = Tensor::zeros([1, model.config().in_channels, probe, probe]);
    let mut analyzer = Analyzer { cal, entries: Vec::new() };
    let _ = walk::features(model, &mut analyzer, &x)?;

    let mut total_macs = 0u64;
    for e in &mut analyzer.entries {
        e.out_hw = (e.out_hw.0 * scale, e.out_hw.1 * scale);
        e.macs = e.macs.saturating_mul((scale * scale) as u64);
        total_macs = total_macs.saturating_add(e.macs);
    }
    let total_f = (total_macs.max(1)) as f64;
    for e in &mut analyzer.entries {
        e.mac_fraction = e.macs as f64 / total_f;
        let distortion = (42.0 - e.folded_weight_sqnr_db).max(0.0);
        let dw = downstream_weight(&e.name);
        e.score = distortion * dw / (e.mac_fraction + 0.015);
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
    let budget = policy.max_f32_mac_fraction.clamp(0.0, 1.0);
    if budget <= 0.0 {
        return Ok(Vec::new());
    }
    let ranked = analyze_sensitivity(model, cal)?;
    let by_name: std::collections::HashMap<&str, &ConvSensitivity> =
        ranked.iter().map(|e| (e.name.as_str(), e)).collect();

    let mut selected: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut used_frac = 0.0f64;

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
    if policy.include_p5_head_out && by_name.contains_key("neck.out5.cv2") {
        try_add(&["neck.out5.cv2"], &mut used_frac, &mut selected, &mut seen);
    }

    // 2. Greedily promote the highest-sensitivity convs whose folded-weight SQNR falls below `sqnr_threshold_db`.
    //    Residual bottleneck pairs `(cv1, cv2)` are promoted atomically when `pair_residual_bottlenecks` is set so
    //    we never introduce a redundant internal `u8 <-> f32` conversion between `cv1` and `cv2`.
    for entry in &ranked {
        if entry.folded_weight_sqnr_db >= policy.sqnr_threshold_db {
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
