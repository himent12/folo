//! Gated Linear Unit feed-forward modules (**SwiGLU** and **GeGLU**; Shazeer 2020) and numerically stable
//! tanh-approximate **GELU** ([`gelu_tanh`]).
//!
//! ```text
//! SwiGLU(gate, value) = SiLU(gate) * value
//! GeGLU(gate, value)  = GELU_tanh(gate) * value
//! GatedFfn(x)         = down_proj(GLU(gate_proj(x), up_proj(x)))
//! ```
//!
//! [`gelu_tanh`] evaluates `0.5 * x * (1 + tanh(u))` via the cancellation-free identity
//! `x * sigmoid(2 * u)` with `u = sqrt(2 / pi) * (x + 0.044715 * x^3)` in `f64`, saturating cleanly for
//! `|x| > 25` so large finite inputs (up to `±f32::MAX`) never overflow `x^3` or produce `0 * Inf = NaN`
//! in backward.

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use super::Linear;
use crate::module::{Forward, Module};
use crate::{Error, Result};

/// `sqrt(2 / pi)` in `f64`.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// Cubic coefficient in Hendrycks & Gimpel (2016) tanh-GELU.
const GELU_CUBIC: f64 = 0.044_715;
/// Above `|x| = 25`, `2 * u > 1150`, where `sigmoid(-2u)` underflows to exact `0.0` in `f64`.
const GELU_SAT_X: f64 = 25.0;

#[inline]
fn sigmoid_f64(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Exact single-element tanh-GELU `(y, dy/dx)` in `f64`, cancellation-free and overflow-safe up to `±f32::MAX`.
///
/// Non-finite inputs (`NaN`, `±Inf`) return `(NaN, NaN)`.
#[inline]
pub(crate) fn gelu_tanh_elem_f64(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if x > GELU_SAT_X {
        return (x, 1.0);
    }
    if x < -GELU_SAT_X {
        return (0.0, 0.0);
    }
    let x2 = x * x;
    let two_u = 2.0 * SQRT_2_OVER_PI * x * (1.0 + GELU_CUBIC * x2);
    let s = sigmoid_f64(two_u);
    let one_m_s = sigmoid_f64(-two_u);
    let d_two_u = 2.0 * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x2);
    let y = x * s;
    let dy = s + (x * d_two_u) * (s * one_m_s);
    (y, dy)
}

/// Numerically stable tanh-approximate GELU activation (Hendrycks & Gimpel 2016) with exact `f64`
/// analytical backward gradients:
///
/// ```text
/// GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
///              = x * sigmoid(2 * sqrt(2 / pi) * (x + 0.044715 * x^3))
/// ```
///
/// Supports `F32` and `F16` tensors of any shape on any device, preserves autograd through empty tensors,
/// stays finite for all finite inputs up to `±f32::MAX`, and propagates `NaN` for non-finite inputs.
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    if x.dtype() == DType::F16 {
        return Ok(gelu_tanh(&x.to_dtype(DType::F32)?)?.to_dtype(DType::F16)?);
    }
    if x.dtype() != DType::F32 {
        return Err(Error::Core(folo_core::Error::DTypeMismatch {
            op: "gelu_tanh",
            expected: DType::F32,
            got: x.dtype(),
        }));
    }
    let dev = x.device();
    let shape = x.shape().clone();
    let n = shape.numel();
    if n == 0 {
        let out_t = Tensor::zeros_with(&shape, DType::F32, dev)?;
        return Ok(Tensor::custom_op("gelu_tanh", &[x], out_t, move |_g| {
            Ok(vec![Some(Tensor::zeros_with(&shape, DType::F32, dev)?)])
        })?);
    }
    let xv = x.to_vec::<f32>()?;
    let mut out = vec![0.0f32; n];
    let mut dydx = vec![0.0f64; n];
    for i in 0..n {
        let (yv, gv) = gelu_tanh_elem_f64(xv[i] as f64);
        out[i] = yv as f32;
        dydx[i] = gv;
    }
    let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
    Ok(Tensor::custom_op("gelu_tanh", &[x], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let gi = gv[i] as f64;
            if gi == 0.0 && dydx[i].is_finite() {
                continue;
            }
            grad[i] = (gi * dydx[i]) as f32;
        }
        Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
    })?)
}

fn check_glu_pair(gate: &Tensor, value: &Tensor, op: &'static str) -> Result<()> {
    if gate.device() != value.device() {
        return Err(Error::Core(folo_core::Error::DeviceMismatch {
            op,
            lhs: gate.device(),
            rhs: value.device(),
        }));
    }
    if gate.dtype() != value.dtype() {
        return Err(Error::Core(folo_core::Error::DTypeMismatch {
            op,
            expected: gate.dtype(),
            got: value.dtype(),
        }));
    }
    if gate.shape() != value.shape() {
        return Err(Error::Invalid(format!(
            "{op}: gate shape {} != value shape {}",
            gate.shape(),
            value.shape()
        )));
    }
    Ok(())
}

/// Element-wise SwiGLU gating: `SiLU(gate) * value` (Shazeer 2020).
pub fn swiglu(gate: &Tensor, value: &Tensor) -> Result<Tensor> {
    check_glu_pair(gate, value, "swiglu")?;
    Ok(gate.silu()?.mul(value)?)
}

/// Element-wise GeGLU gating: `GELU_tanh(gate) * value` (Shazeer 2020).
pub fn geglu(gate: &Tensor, value: &Tensor) -> Result<Tensor> {
    check_glu_pair(gate, value, "geglu")?;
    Ok(gelu_tanh(gate)?.mul(value)?)
}

/// Gating activation for [`GatedFfn`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GluKind {
    /// SwiGLU: `SiLU(gate_proj(x)) * up_proj(x)` (LLaMA / Qwen / Mistral).
    #[default]
    SwiGlu,
    /// GeGLU: `GELU_tanh(gate_proj(x)) * up_proj(x)` (Gemma / PaLM / T5v1.1).
    GeGlu,
}

/// Gated Linear Unit Feed-Forward Network (`SwiGLU` or `GeGLU`; Shazeer 2020):
///
/// ```text
/// y = down_proj(act(gate_proj(x)) * up_proj(x))
/// ```
///
/// Child module names: `gate_proj`, `up_proj`, `down_proj` (giving dotted parameter names
/// `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`, plus optional `.bias`).
/// Operates over the last dimension of any rank `>= 1` tensor (`[D]`, `[B, D]`, `[B, L, D]`, `[B, H, W, D]`).
pub struct GatedFfn {
    /// Gating projection `[hidden_features, in_features]`.
    pub gate_proj: Linear,
    /// Value projection `[hidden_features, in_features]`.
    pub up_proj: Linear,
    /// Output projection `[out_features, hidden_features]`.
    pub down_proj: Linear,
    kind: GluKind,
}

impl GatedFfn {
    /// Construct a [`GatedFfn`] with explicit [`GluKind`] and seeded `Linear` projections.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        kind: GluKind,
        bias: bool,
        rng: &mut Rng,
    ) -> Result<Self> {
        if in_features == 0 || hidden_features == 0 || out_features == 0 {
            return Err(Error::Invalid(format!(
                "GatedFfn: dimensions must be >= 1, got in={in_features}, hidden={hidden_features}, out={out_features}"
            )));
        }
        let gate_proj = Linear::new(in_features, hidden_features, bias, rng);
        let up_proj = Linear::new(in_features, hidden_features, bias, rng);
        let down_proj = Linear::new(hidden_features, out_features, bias, rng);
        Ok(Self { gate_proj, up_proj, down_proj, kind })
    }

    /// Construct a SwiGLU feed-forward module (`SiLU(gate_proj(x)) * up_proj(x)`).
    pub fn swiglu(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        bias: bool,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::new(in_features, hidden_features, out_features, GluKind::SwiGlu, bias, rng)
    }

    /// Construct a GeGLU feed-forward module (`GELU_tanh(gate_proj(x)) * up_proj(x)`).
    pub fn geglu(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        bias: bool,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::new(in_features, hidden_features, out_features, GluKind::GeGlu, bias, rng)
    }

    /// Assemble a [`GatedFfn`] from existing [`Linear`] layers, validating that projection shapes match.
    pub fn from_layers(gate_proj: Linear, up_proj: Linear, down_proj: Linear, kind: GluKind) -> Result<Self> {
        let (gh, gi) = gate_proj.weight.dims2()?;
        let (uh, ui) = up_proj.weight.dims2()?;
        let (do_, di) = down_proj.weight.dims2()?;
        if gh == 0 || gi == 0 || do_ == 0 {
            return Err(Error::Invalid("GatedFfn: projection dimensions must be >= 1".into()));
        }
        if (uh, ui) != (gh, gi) {
            return Err(Error::Invalid(format!(
                "GatedFfn: up_proj [{uh}, {ui}] does not match gate_proj [{gh}, {gi}]"
            )));
        }
        if di != gh {
            return Err(Error::Invalid(format!(
                "GatedFfn: down_proj input dim {di} does not match hidden dim {gh}"
            )));
        }
        Ok(Self { gate_proj, up_proj, down_proj, kind })
    }

    /// Parameter-matched hidden width `round_up(floor(8 * dim / 3), multiple_of)` (Shazeer 2020 / LLaMA)
    /// so the 3-matrix GLU FFN matches the parameter count of a standard `4 * dim` 2-matrix MLP.
    pub fn recommended_hidden(dim: usize, multiple_of: usize) -> usize {
        let m = multiple_of.max(1);
        let raw = (dim.saturating_mul(8) / 3).max(1);
        raw.div_ceil(m).saturating_mul(m)
    }

    /// The gating variant (`SwiGlu` or `GeGlu`).
    pub fn kind(&self) -> GluKind {
        self.kind
    }
}

impl Module for GatedFfn {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("gate_proj", &self.gate_proj);
        f("up_proj", &self.up_proj);
        f("down_proj", &self.down_proj);
    }
}

impl Forward for GatedFfn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let hidden = match self.kind {
            GluKind::SwiGlu => swiglu(&gate, &up)?,
            GluKind::GeGlu => geglu(&gate, &up)?,
        };
        self.down_proj.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::StandardGroups;
    use folo_core::testing::{GradCheck, grad_check_with};

    /// `f64` central finite-difference gradient check for `gelu_tanh_elem_f64` across standard,
    /// near-zero, saturation-boundary, and extreme (`±50`, `±f32::MAX`) inputs.
    #[test]
    fn gelu_tanh_f64_central_fd_and_extreme_inputs() {
        let xs = [-50.0f64, -24.9, -10.0, -3.5, -1.0, -0.2, 0.0, 0.3, 1.2, 4.0, 10.0, 24.9, 50.0];
        for &x in &xs {
            let (y, dy) = gelu_tanh_elem_f64(x);
            assert!(y.is_finite() && dy.is_finite(), "x={x}: y={y} dy={dy}");
            let h = 1e-6f64;
            let fd = (gelu_tanh_elem_f64(x + h).0 - gelu_tanh_elem_f64(x - h).0) / (2.0 * h);
            let tol = 1e-8 + 1e-5 * fd.abs().max(dy.abs());
            assert!((dy - fd).abs() <= tol, "gelu_tanh x={x}: ana={dy:e} fd={fd:e}");
        }

        // Exact saturation at ±f32::MAX (no x^3 overflow or 0 * Inf = NaN).
        let ext = Tensor::from_slice(&[-f32::MAX, -500.0, -50.0, 0.0, 50.0, 500.0, f32::MAX], [7])
            .unwrap()
            .requires_grad_();
        let y = gelu_tanh(&ext).unwrap();
        let yv = y.to_vec::<f32>().unwrap();
        assert_eq!(yv[0], 0.0);
        assert_eq!(yv[1], 0.0);
        assert_eq!(yv[3], 0.0);
        assert_eq!(yv[5], 500.0);
        assert_eq!(yv[6], f32::MAX);
        assert!(yv.iter().all(|v| v.is_finite()));
        let gv = y.sum_all().unwrap().backward().unwrap().get(&ext).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(gv[0], 0.0);
        assert_eq!(gv[1], 0.0);
        assert!((gv[3] - 0.5).abs() < 1e-6);
        assert_eq!(gv[5], 1.0);
        assert_eq!(gv[6], 1.0);
        assert!(gv.iter().all(|v| v.is_finite()));

        // Non-finite inputs poison both forward and backward.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bx = Tensor::from_slice(&[bad, 1.0], [2]).unwrap().requires_grad_();
            let by = gelu_tanh(&bx).unwrap();
            let byv = by.to_vec::<f32>().unwrap();
            assert!(byv[0].is_nan() && byv[1].is_finite(), "bad {bad}: {byv:?}");
            let bgv = by.sum_all().unwrap().backward().unwrap().get(&bx).unwrap().to_vec::<f32>().unwrap();
            assert!(bgv[0].is_nan() && bgv[1].is_finite(), "bad {bad} grad: {bgv:?}");
        }

        // Empty tensor keeps autograd wired.
        let empty = Tensor::zeros([0, 4]).requires_grad_();
        let ey = gelu_tanh(&empty).unwrap();
        assert_eq!(ey.dims(), &[0, 4]);
        let eg = ey.sum_all().unwrap().backward().unwrap();
        assert_eq!(eg.get(&empty).unwrap().dims(), &[0, 4]);
    }

    /// Full finite-difference gradient checks for `swiglu`, `geglu`, and `GatedFfn` (`SwiGlu` & `GeGlu`)
    /// w.r.t. inputs AND all projection weights & biases.
    #[test]
    fn swiglu_geglu_and_gated_ffn_grad_checks() {
        let mut rng = Rng::seed(2026);
        let gate = Tensor::randn([2, 3, 4], 0.0, 1.0, &mut rng);
        let value = Tensor::randn([2, 3, 4], 0.0, 1.0, &mut rng);
        let gc = GradCheck { eps: 2e-3, atol: 2e-3, rtol: 2e-2, max_checks: 16, seed: 99 };

        grad_check_with(
            |inps| swiglu(&inps[0], &inps[1]).unwrap().square()?.sum_all(),
            &[gate.clone(), value.clone()],
            &gc,
        )
        .unwrap();
        grad_check_with(|inps| geglu(&inps[0], &inps[1]).unwrap().square()?.sum_all(), &[gate, value], &gc)
            .unwrap();

        for kind in [GluKind::SwiGlu, GluKind::GeGlu] {
            let ffn = GatedFfn::new(4, 6, 3, kind, true, &mut rng).unwrap();
            let x = Tensor::randn([2, 3, 4], 0.0, 0.8, &mut rng);
            let out = ffn.forward(&x).unwrap();
            assert_eq!(out.dims(), &[2, 3, 3]);

            // Verify dotted parameter names and StandardGroups split.
            let names: Vec<String> = ffn.named_parameters().into_iter().map(|(n, _)| n).collect();
            assert_eq!(
                names,
                vec![
                    "gate_proj.weight",
                    "gate_proj.bias",
                    "up_proj.weight",
                    "up_proj.bias",
                    "down_proj.weight",
                    "down_proj.bias",
                ]
            );
            let sg = StandardGroups::from_module(&ffn);
            assert_eq!((sg.decay.len(), sg.bias.len(), sg.bn.len()), (3, 3, 0));

            // Gradient check w.r.t. input x and every projection weight & bias simultaneously.
            let inps = vec![
                x,
                ffn.gate_proj.weight.detach(),
                ffn.gate_proj.bias.as_ref().unwrap().detach(),
                ffn.up_proj.weight.detach(),
                ffn.up_proj.bias.as_ref().unwrap().detach(),
                ffn.down_proj.weight.detach(),
                ffn.down_proj.bias.as_ref().unwrap().detach(),
            ];
            grad_check_with(
                |t| {
                    let g_l = Linear::from_tensors(t[1].clone(), Some(t[2].clone())).unwrap();
                    let u_l = Linear::from_tensors(t[3].clone(), Some(t[4].clone())).unwrap();
                    let d_l = Linear::from_tensors(t[5].clone(), Some(t[6].clone())).unwrap();
                    let m = GatedFfn::from_layers(g_l, u_l, d_l, kind).unwrap();
                    m.forward(&t[0]).unwrap().square()?.sum_all()
                },
                &inps,
                &GradCheck { eps: 2e-3, atol: 3e-3, rtol: 2e-2, max_checks: 12, seed: 13 },
            )
            .unwrap();
        }

        // recommended_hidden helper & validation rejections.
        assert_eq!(GatedFfn::recommended_hidden(64, 16), 176); // ceil(170 / 16) * 16 = 176
        assert_eq!(GatedFfn::recommended_hidden(128, 64), 384); // ceil(341 / 64) * 64 = 384
        assert!(GatedFfn::swiglu(0, 8, 4, false, &mut rng).is_err());
        assert!(GatedFfn::geglu(4, 0, 4, false, &mut rng).is_err());
        assert!(GatedFfn::swiglu(4, 8, 0, false, &mut rng).is_err());
        assert!(swiglu(&Tensor::zeros([2, 3]), &Tensor::zeros([2, 4])).is_err());
        assert!(geglu(&Tensor::zeros([2, 3]), &Tensor::zeros([2, 4])).is_err());
        let l1 = Linear::new(4, 6, false, &mut rng);
        let l2 = Linear::new(4, 5, false, &mut rng);
        let l3 = Linear::new(6, 4, false, &mut rng);
        assert!(GatedFfn::from_layers(l1, l2, l3, GluKind::SwiGlu).is_err());
    }
}
