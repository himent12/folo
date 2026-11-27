//! FOLO v2 hardware-co-designed building blocks:
//! - [`space_to_depth_2x2`]: lossless 2×2 spatial-to-channel folding (ONNX opset-17 `SpaceToDepth`).
//! - [`WsQARepConv3x3`]: weight-space quantization-aware reparameterized 3×3 convolution with
//!   `3×3 + 1×3 + 3×1 + 1×1 + γ_id·I` branches folded into a single `W_eff` before one Post-BN during
//!   training and algebraically collapsed into one dense `3×3` conv at inference.
//! - [`FoloBlock`]: `ceil16`-aligned QARep-CSP / Rep-ELAN block with `≤ 3` zero-copy `forward_cat` parts.
//! - [`HyperContext`]: linear `O(N·M)` P5 hypergraph global context (`HyperACE-Lite`).
//! - [`FullPadGate`]: zero-init residual FiLM gate injecting P5 scene context into `f32` head inputs.

use std::sync::RwLock;

use folo_core::cpu::gemm::{Strided, gemm};
use folo_core::{Conv2dParams, DType, Device, Tensor, is_grad_enabled, no_grad};
use folo_nn::{
    Activation, BatchNorm2d, Conv2d, Conv2dConfig, ConvBnAct, Error, Forward, Linear, Module, ModuleList,
    Result, TensorKind, fuse_conv_bn, init,
};
use rayon::prelude::*;

use crate::model::blocks::{Fuse, Init};

const CONV3X3_S1_P1: Conv2dParams =
    Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };

/// Lossless `2×2` space-to-depth transform matching ONNX opset-17 `SpaceToDepth(blocksize=2)`:
/// maps `[B, C, H, W]` (with even `H, W`) to `[B, 4·C, H/2, W/2]` by concatenating the four sub-pixel
/// phases `(0,0), (0,1), (1,0), (1,1)` along the channel axis.
pub fn space_to_depth_2x2(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x
        .dims4()
        .map_err(|_| Error::Invalid(format!("space_to_depth_2x2: expected NCHW, got {:?}", x.dims())))?;
    if !h.is_multiple_of(2) || !w.is_multiple_of(2) {
        return Err(Error::Invalid(format!("space_to_depth_2x2: spatial dims ({h}, {w}) must be even")));
    }
    // Single-pass contiguous CPU fast path under `no_grad` (avoids 4 strided slice views + `Tensor::cat`).
    if !is_grad_enabled()
        && x.device().is_cpu()
        && x.dtype() == DType::F32
        && b > 0
        && c > 0
        && h > 0
        && w > 0
    {
        let (oh, ow) = (h / 2, w / 2);
        let (hw, ohw) = (h * w, oh * ow);
        let mut out = folo_core::cpu::take_buffer(b * 4 * c * ohw);
        x.with_f32_slice(|src| {
            let run_img = |(bi, dst_img): (usize, &mut [f32])| {
                let src_img = &src[bi * c * hw..(bi + 1) * c * hw];
                let (p0, rest) = dst_img.split_at_mut(c * ohw);
                let (p1, rest) = rest.split_at_mut(c * ohw);
                let (p2, p3) = rest.split_at_mut(c * ohw);
                for ch in 0..c {
                    let sp = &src_img[ch * hw..(ch + 1) * hw];
                    let d0 = &mut p0[ch * ohw..(ch + 1) * ohw];
                    let d1 = &mut p1[ch * ohw..(ch + 1) * ohw];
                    let d2 = &mut p2[ch * ohw..(ch + 1) * ohw];
                    let d3 = &mut p3[ch * ohw..(ch + 1) * ohw];
                    for oy in 0..oh {
                        let r_even = &sp[(2 * oy) * w..(2 * oy + 1) * w];
                        let r_odd = &sp[(2 * oy + 1) * w..(2 * oy + 2) * w];
                        let o0 = &mut d0[oy * ow..(oy + 1) * ow];
                        let o1 = &mut d1[oy * ow..(oy + 1) * ow];
                        let o2 = &mut d2[oy * ow..(oy + 1) * ow];
                        let o3 = &mut d3[oy * ow..(oy + 1) * ow];
                        for ox in 0..ow {
                            o0[ox] = r_even[2 * ox];
                            o1[ox] = r_even[2 * ox + 1];
                            o2[ox] = r_odd[2 * ox];
                            o3[ox] = r_odd[2 * ox + 1];
                        }
                    }
                }
            };
            if b > 1 && b * c * hw >= 32_768 {
                out.par_chunks_exact_mut(4 * c * ohw).enumerate().for_each(run_img);
            } else {
                out.chunks_exact_mut(4 * c * ohw).enumerate().for_each(run_img);
            }
        })?;
        return Ok(Tensor::from_vec(out, [b, 4 * c, oh, ow])?);
    }
    let tl = x.slice(2, 0, h, 2)?.slice(3, 0, w, 2)?;
    let tr = x.slice(2, 0, h, 2)?.slice(3, 1, w, 2)?;
    let bl = x.slice(2, 1, h, 2)?.slice(3, 0, w, 2)?;
    let br = x.slice(2, 1, h, 2)?.slice(3, 1, w, 2)?;
    Ok(Tensor::cat(&[&tl, &tr, &bl, &br], 1)?)
}

/// Lossless `SPD(2×2)` + `Conv1×1(4·c_in → c_out)` stem (or standard `Conv3×3 s=2` when `spd == false`).
pub struct StemV2 {
    /// The convolution after optional `SPD(2×2)`.
    pub conv: ConvBnAct,
    /// Whether `space_to_depth_2x2` runs before `conv`.
    pub spd: bool,
}

impl StemV2 {
    /// Build a v2 stem (`SPD + 1×1` when `spd`, else standard `3×3 s=2`).
    pub fn new(c_in: usize, c_out: usize, spd: bool, init: &mut Init) -> Self {
        let conv = if spd { init.conv(4 * c_in, c_out, 1, 1) } else { init.conv(c_in, c_out, 3, 2) };
        StemV2 { conv, spd }
    }
}

impl Module for StemV2 {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("conv", &self.conv);
    }
}

impl Forward for StemV2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.spd {
            let s = space_to_depth_2x2(x)?;
            self.conv.forward(&s)
        } else {
            self.conv.forward(x)
        }
    }
}

impl Fuse for StemV2 {
    fn fuse_(&mut self) -> Result<()> {
        self.conv.fuse_()
    }
}

/// Training-time branches of [`WsQARepConv3x3`].
pub struct QARepBranches {
    /// Isotropic `3×3` branch weight, `[Cout, Cin, 3, 3]`.
    pub w3x3: Tensor,
    /// Horizontal strip `1×3` branch weight, `[Cout, Cin, 1, 3]`.
    pub w1x3: Tensor,
    /// Vertical strip `3×1` branch weight, `[Cout, Cin, 3, 1]`.
    pub w3x1: Tensor,
    /// Pointwise `1×1` branch weight, `[Cout, Cin, 1, 1]`.
    pub w1x1: Tensor,
    /// Per-channel learnable identity scale `[Cout]` (present iff `Cin == Cout`, initialized to `1.0`).
    pub gamma_id: Option<Tensor>,
    /// Cached `[C, C]` identity matrix on the active `(device, dtype)` (never a saved buffer).
    eye_cache: Option<RwLock<Tensor>>,
}

impl QARepBranches {
    fn eye_on(&self, dev: Device, dt: DType, c: usize) -> Result<Tensor> {
        let lock = self.eye_cache.as_ref().expect("eye_cache present iff gamma_id is Some");
        {
            let guard = lock.read().expect("eye_cache lock");
            if guard.device() == dev && guard.dtype() == dt {
                return Ok(guard.clone());
            }
        }
        let fresh = no_grad();
        let eye = Tensor::eye(c).to_device(dev)?.to_dtype(dt)?;
        drop(fresh);
        let mut guard = lock.write().expect("eye_cache lock");
        *guard = eye.clone();
        Ok(eye)
    }
}

/// Weight-Space Quantization-Aware Reparameterized `3×3` convolution (`WS-QARepConv3x3`).
///
/// During training, the five linear branches (`3×3`, `1×3`, `3×1`, `1×1`, and per-channel identity
/// `γ_id · I` when `c_in == c_out`) sum **in weight space** into a single effective `3×3` kernel `W_eff`
/// before one shared post-sum BatchNorm + SiLU:
///
/// ```text
/// W_eff = W_3x3 + pad(W_1x3) + pad(W_3x1) + pad(W_1x1) + diag_3x3(γ_id)
/// y = act(BN_post(conv3x3(x, W_eff)))
/// ```
///
/// Forward outputs and gradients are algebraically identical to running 5 separate branches in activation
/// space and summing before `BN_post`, at 1× activation compute/VRAM. Calling [`Fuse::fuse_`] folds
/// `W_eff` and `BN_post` into one standard dense `Conv2d(3×3)` + bias + activation (zero residual `Add`s).
pub struct WsQARepConv3x3 {
    /// Active multi-branch training parameters (`None` after [`Fuse::fuse_`]).
    pub branches: Option<QARepBranches>,
    /// Single post-sum BatchNorm (`None` after [`Fuse::fuse_`]).
    pub bn: Option<BatchNorm2d>,
    /// Fused inference dense `3×3` conv (`Some` after [`Fuse::fuse_`]).
    pub fused: Option<Conv2d>,
    /// Post-BN activation (SiLU by default).
    pub act: Activation,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
}

impl WsQARepConv3x3 {
    /// Construct a `WS-QARepConv3x3(c_in → c_out)` with SiLU activation.
    ///
    /// Each of the four conv branches is Kaiming-uniform initialised with variance scaled by `1/4`
    /// (`bound = 0.5 / sqrt(fan_in_branch)`) so the summed `W_eff` starts at unit variance alongside `γ_id = 1`.
    pub fn new(c_in: usize, c_out: usize, init_ctx: &mut Init) -> Self {
        Self::with_act(c_in, c_out, Activation::Silu, init_ctx)
    }

    /// Construct a `WS-QARepConv3x3(c_in → c_out)` with an explicit activation.
    pub fn with_act(c_in: usize, c_out: usize, act: Activation, init_ctx: &mut Init) -> Self {
        let scale = 0.5;
        let b33 = scale * init::default_conv_bound(c_in * 9);
        let b13 = scale * init::default_conv_bound(c_in * 3);
        let b31 = scale * init::default_conv_bound(c_in * 3);
        let b11 = scale * init::default_conv_bound(c_in);
        let w3x3 = init::uniform([c_out, c_in, 3, 3], b33, init_ctx.rng).requires_grad_();
        let w1x3 = init::uniform([c_out, c_in, 1, 3], b13, init_ctx.rng).requires_grad_();
        let w3x1 = init::uniform([c_out, c_in, 3, 1], b31, init_ctx.rng).requires_grad_();
        let w1x1 = init::uniform([c_out, c_in, 1, 1], b11, init_ctx.rng).requires_grad_();
        let (gamma_id, eye_cache) = if c_in == c_out {
            (Some(Tensor::ones([c_out]).requires_grad_()), Some(RwLock::new(Tensor::eye(c_out))))
        } else {
            (None, None)
        };
        let bn = BatchNorm2d::with_options(c_out, init_ctx.bn_eps, init_ctx.bn_momentum);
        WsQARepConv3x3 {
            branches: Some(QARepBranches { w3x3, w1x3, w3x1, w1x1, gamma_id, eye_cache }),
            bn: Some(bn),
            fused: None,
            act,
            in_channels: c_in,
            out_channels: c_out,
        }
    }

    /// True once [`Fuse::fuse_`] has collapsed all branches and `BN_post` into a single `Conv2d`.
    pub fn is_fused(&self) -> bool {
        self.fused.is_some()
    }

    /// Assemble the effective `3×3` weight tensor `W_eff` in weight space (differentiable during training).
    pub fn effective_weight(&self) -> Result<Tensor> {
        if let Some(conv) = &self.fused {
            return Ok(conv.weight.clone());
        }
        let b = self
            .branches
            .as_ref()
            .ok_or_else(|| Error::Invalid("WsQARepConv3x3: missing branches and fused conv".into()))?;
        let mut w = b.w3x3.clone();
        w = w.add(&b.w1x3.pad(&[(1, 1), (0, 0)], 0.0)?)?;
        w = w.add(&b.w3x1.pad(&[(0, 0), (1, 1)], 0.0)?)?;
        w = w.add(&b.w1x1.pad(&[(1, 1), (1, 1)], 0.0)?)?;
        if let Some(gamma_id) = &b.gamma_id {
            let c = self.out_channels;
            let eye = b.eye_on(gamma_id.device(), gamma_id.dtype(), c)?;
            let id_1x1 = gamma_id.reshape([c, 1])?.mul(&eye)?.reshape([c, c, 1, 1])?;
            w = w.add(&id_1x1.pad(&[(1, 1), (1, 1)], 0.0)?)?;
        }
        Ok(w)
    }

    /// Explicit 5-branch activation-space reference forward (for algebraic equivalence verification in tests).
    pub fn forward_explicit_branches(&self, x: &Tensor) -> Result<Tensor> {
        let b = self.branches.as_ref().ok_or_else(|| {
            Error::Invalid("WsQARepConv3x3::forward_explicit_branches: already fused".into())
        })?;
        let bn = self
            .bn
            .as_ref()
            .ok_or_else(|| Error::Invalid("WsQARepConv3x3::forward_explicit_branches: missing bn".into()))?;
        let p13 = Conv2dParams { stride: [1, 1], padding: [0, 1], dilation: [1, 1], groups: 1 };
        let p31 = Conv2dParams { stride: [1, 1], padding: [1, 0], dilation: [1, 1], groups: 1 };
        let p11 = Conv2dParams { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 };
        let mut z = x.conv2d_with(&b.w3x3, None, &CONV3X3_S1_P1)?;
        z = z.add(&x.conv2d_with(&b.w1x3, None, &p13)?)?;
        z = z.add(&x.conv2d_with(&b.w3x1, None, &p31)?)?;
        z = z.add(&x.conv2d_with(&b.w1x1, None, &p11)?)?;
        if let Some(gamma_id) = &b.gamma_id {
            let scale = gamma_id.reshape([1, self.out_channels, 1, 1])?;
            z = z.add(&x.mul(&scale)?)?;
        }
        bn.forward_act(&z, self.act)
    }
}

impl Module for WsQARepConv3x3 {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        if let Some(b) = &self.branches {
            f("w3x3", &b.w3x3, TensorKind::Parameter);
            f("w1x3", &b.w1x3, TensorKind::Parameter);
            f("w3x1", &b.w3x1, TensorKind::Parameter);
            f("w1x1", &b.w1x1, TensorKind::Parameter);
            if let Some(g) = &b.gamma_id {
                f("gamma_id", g, TensorKind::Parameter);
            }
        }
    }

    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        if let Some(bn) = &self.bn {
            f("bn", bn);
        }
        if let Some(conv) = &self.fused {
            f("conv", conv);
        }
    }
}

impl Forward for WsQARepConv3x3 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let act = self.act.unary_op();
        if let Some(conv) = &self.fused {
            return Ok(x.conv2d_act(&conv.weight, conv.bias.as_ref(), &CONV3X3_S1_P1, act)?);
        }
        let bn = self.bn.as_ref().expect("unfused WsQARepConv3x3 has bn");
        let w_eff = self.effective_weight()?;
        if bn.is_training() {
            let z = x.conv2d_with(&w_eff, None, &CONV3X3_S1_P1)?;
            bn.forward_act(&z, self.act)
        } else if !x.device().is_cpu() || !is_grad_enabled() {
            let (scale, shift) = bn.eval_scale_shift()?;
            let w_f = w_eff.mul(&scale.reshape([self.out_channels, 1, 1, 1])?)?;
            Ok(x.conv2d_act(&w_f, Some(&shift), &CONV3X3_S1_P1, act)?)
        } else {
            let z = x.conv2d_with(&w_eff, None, &CONV3X3_S1_P1)?;
            self.act.forward(&bn.forward(&z)?)
        }
    }
}

impl Fuse for WsQARepConv3x3 {
    fn fuse_(&mut self) -> Result<()> {
        if self.fused.is_some() {
            return Ok(());
        }
        let w_eff = {
            let _g = no_grad();
            self.effective_weight()?
        };
        let bn = self.bn.as_ref().expect("unfused WsQARepConv3x3 has bn");
        let dtype = w_eff.dtype();
        let (w, b) =
            fuse_conv_bn(&w_eff, None, &bn.weight, &bn.bias, &bn.running_mean, &bn.running_var, bn.eps)?;
        let (w, b) = (w.to_dtype(dtype)?, b.to_dtype(dtype)?);
        let cfg = Conv2dConfig::new(self.in_channels, self.out_channels, 3).stride(1).padding(1);
        let fused = Conv2d::from_tensors(cfg, w, Some(b))?;
        self.fused = Some(fused);
        self.bn = None;
        self.branches = None;
        Ok(())
    }
}

/// Hardware-aligned QARep-CSP / Rep-ELAN block (`FoloBlock`).
///
/// Enforces two hardware invariants:
/// 1. Every `c_mid` is a multiple of 16 (`≥ 32` on `n`, `≥ 64` on `s`) — zero narrow `< 32` convs.
/// 2. `cv2` reads at most 3 concat parts (`[cv1_out, (y_mid), y_last]`), staying inside `MAX_CAT_PARTS = 4`
///    for zero-copy `forward_cat`.
pub struct FoloBlock {
    /// `Conv(c1 → 2·c_mid, 1×1)` with SiLU.
    pub cv1: ConvBnAct,
    /// `m` sequential [`WsQARepConv3x3`] layers (`c_mid → c_mid`).
    pub reps: ModuleList<WsQARepConv3x3>,
    /// `Conv((2 + taps)·c_mid → c2, 1×1)` with SiLU (`taps = 1` for `m <= 2`, else `2`).
    pub cv2: ConvBnAct,
    /// Mid-branch width (`% 16 == 0`).
    pub c_mid: usize,
}

impl FoloBlock {
    /// Build `FoloBlock(c1 → c2, m, c_mid)`.
    pub fn new(c1: usize, c2: usize, m: usize, c_mid: usize, init: &mut Init) -> Self {
        assert!(m >= 1, "FoloBlock requires m >= 1");
        let taps = if m <= 2 { 1 } else { 2 };
        let cv1 = init.conv(c1, 2 * c_mid, 1, 1);
        let reps = (0..m).map(|_| WsQARepConv3x3::new(c_mid, c_mid, init)).collect();
        let cv2 = init.conv((2 + taps) * c_mid, c2, 1, 1);
        FoloBlock { cv1, reps: ModuleList::with_prefix("rep", reps), cv2, c_mid }
    }

    /// Forward reading concatenated inputs `xs` without materialising `cat(xs)` when fused.
    pub fn forward_cat(&self, xs: &[&Tensor]) -> Result<Tensor> {
        let y = self.cv1.forward_cat(xs)?;
        let m = self.reps.len();
        let mid_idx = if m >= 3 { Some(m / 2 - 1) } else { None };
        let mut cur = y.narrow(1, self.c_mid, self.c_mid)?;
        let mut mid_tap = None;
        for (j, rep) in self.reps.iter().enumerate() {
            cur = rep.forward(&cur)?;
            if mid_idx == Some(j) {
                mid_tap = Some(cur.clone());
            }
        }
        match &mid_tap {
            Some(mid) => self.cv2.forward_cat(&[&y, mid, &cur]),
            None => self.cv2.forward_cat(&[&y, &cur]),
        }
    }
}

impl Module for FoloBlock {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cv1", &self.cv1);
        self.reps.visit_children(f);
        f("cv2", &self.cv2);
    }
}

impl Forward for FoloBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_cat(&[x])
    }
}

impl Fuse for FoloBlock {
    fn fuse_(&mut self) -> Result<()> {
        self.cv1.fuse_()?;
        self.reps.fuse_()?;
        self.cv2.fuse_()
    }
}

/// Linear `O(N·M)` hypergraph global context at P5 (`HyperACE-Lite`), replacing `C2PSA`.
///
/// Aggregates `N = H·W` spatial tokens into `M` semantic hyperedges (`M = 16/32`), refines the `M`
/// hyperedges with a compact FFN, broadcasts back to `P5` with a residual skip, and returns the
/// pooled scene summary `g_scene ∈ [B, C_g]` for relocated [`FullPadGate`] head-input modulation.
pub struct HyperContext {
    /// Fused `Conv(c5 → m_edges + c_g, 1×1, act=none)`.
    pub av_proj: ConvBnAct,
    /// First hyperedge FFN projection (`c_g → c_g`).
    pub ffn1: Linear,
    /// Second hyperedge FFN projection (`c_g → c_g`).
    pub ffn2: Linear,
    /// Output projection `Conv(c_g → c5, 1×1, act=none)`.
    pub out_proj: ConvBnAct,
    /// Number of hyperedges `M`.
    pub m_edges: usize,
    /// Hyperedge channel width `c_g = c5 / 2`.
    pub c_g: usize,
}

impl HyperContext {
    /// Build `HyperContext(c5, m_edges)`.
    pub fn new(c5: usize, m_edges: usize, init_ctx: &mut Init) -> Self {
        let c_g = c5 / 2;
        let av_proj = init_ctx.conv_with(c5, m_edges + c_g, 1, 1, 1, Activation::Identity);
        let ffn1 = Linear::new(c_g, c_g, true, init_ctx.rng);
        let ffn2 = Linear::new(c_g, c_g, true, init_ctx.rng);
        let out_proj = init_ctx.conv_with(c_g, c5, 1, 1, 1, Activation::Identity);
        HyperContext { av_proj, ffn1, ffn2, out_proj, m_edges, c_g }
    }

    /// Run the hypergraph gather, hyperedge FFN, scene pooling, and broadcast from `av = av_proj(x)`
    /// (`[b, m_edges + c_g, h, w]`), returning `(y_ctx [b, c_g, h, w], g_scene [b, c_g])`.
    ///
    /// Shared by [`HyperContext::forward_with_scene`] and the quantised graph walk (`model::quant::walk`).
    pub(crate) fn attend_and_scene(
        &self,
        av: &Tensor,
        b: usize,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(pair) = self.attend_and_scene_cpu_fast(av, b, h, w)? {
            return Ok(pair);
        }
        let n = h * w;
        let to_f32 = |t: Tensor| -> folo_core::Result<Tensor> {
            if t.dtype() == folo_core::DType::F32 { Ok(t) } else { t.to_dtype(folo_core::DType::F32) }
        };
        let a_logits = to_f32(av.narrow(1, 0, self.m_edges)?.reshape([b, self.m_edges, n])?)?;
        let v = to_f32(av.narrow(1, self.m_edges, self.c_g)?.reshape([b, self.c_g, n])?)?;
        let a_gather = a_logits.softmax(-1)?; // [B, M, N] in F32: gather over spatial tokens
        let a_bcast = a_logits.softmax(1)?; // [B, M, N] in F32: convex combination over hyperedges
        let e = v.matmul(&a_gather.transpose(1, 2)?)?; // [B, C_g, M]
        let e_t_f32 = to_f32(e.transpose(1, 2)?)?; // [B, M, C_g] in F32
        let ffn_in = if e_t_f32.dtype() != self.ffn1.weight.dtype() && folo_core::autocast_dtype().is_none() {
            e_t_f32.to_dtype(self.ffn1.weight.dtype())?
        } else {
            e_t_f32.clone()
        };
        let h1 = to_f32(self.ffn1.forward(&ffn_in)?)?.silu()?;
        let h1 = if h1.dtype() != self.ffn2.weight.dtype() && folo_core::autocast_dtype().is_none() {
            h1.to_dtype(self.ffn2.weight.dtype())?
        } else {
            h1
        };
        let e_ffn_f32 = to_f32(self.ffn2.forward(&h1)?)?;
        let e_prime_t = e_t_f32.add(&e_ffn_f32)?; // [B, M, C_g] in F32
        let g_scene = e_prime_t.mean(1, false)?; // [B, C_g] in F32
        let e_prime = e_prime_t.transpose(1, 2)?; // [B, C_g, M] in F32
        let y_ctx = e_prime.matmul(&a_bcast)?.reshape([b, self.c_g, h, w])?;
        Ok((y_ctx, g_scene))
    }

    fn attend_and_scene_cpu_fast(
        &self,
        av: &Tensor,
        b: usize,
        h: usize,
        w: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let (m, cg) = (self.m_edges, self.c_g);
        let Some(n) = h.checked_mul(w) else { return Ok(None) };
        let eligible = !is_grad_enabled()
            && b > 0
            && m > 0
            && cg > 0
            && n > 0
            && av.device().is_cpu()
            && av.dtype() == DType::F32
            && av.dims() == [b, m + cg, h, w]
            && self.ffn1.weight.device().is_cpu()
            && self.ffn1.weight.dtype() == DType::F32
            && self.ffn1.weight.dims() == [cg, cg]
            && self.ffn2.weight.device().is_cpu()
            && self.ffn2.weight.dtype() == DType::F32
            && self.ffn2.weight.dims() == [cg, cg]
            && self
                .ffn1
                .bias
                .as_ref()
                .is_none_or(|t| t.device().is_cpu() && t.dtype() == DType::F32 && t.dims() == [cg])
            && self
                .ffn2
                .bias
                .as_ref()
                .is_none_or(|t| t.device().is_cpu() && t.dtype() == DType::F32 && t.dims() == [cg]);
        if !eligible {
            return Ok(None);
        }
        // Snapshot small FFN weights/biases once before locking `av` if any storage aliases.
        let w1 = self.ffn1.weight.to_vec::<f32>()?;
        let b1 = self.ffn1.bias.as_ref().map(|t| t.to_vec::<f32>()).transpose()?;
        let w2 = self.ffn2.weight.to_vec::<f32>()?;
        let b2 = self.ffn2.bias.as_ref().map(|t| t.to_vec::<f32>()).transpose()?;

        let mut y_ctx = folo_core::cpu::take_buffer(b * cg * n);
        let mut g_scene = folo_core::cpu::take_buffer(b * cg);
        let mut a_gather = vec![0.0f32; m * n];
        let mut a_bcast = vec![0.0f32; m * n];
        let mut e_t = vec![0.0f32; m * cg];
        let mut h1 = vec![0.0f32; m * cg];
        let mut e_ffn = vec![0.0f32; m * cg];
        let av_stride = (m + cg) * n;
        let inv_m = 1.0 / m as f32;

        av.with_f32_slice(|av_s| {
            for bi in 0..b {
                let a_off = bi * av_stride;
                let v_off = a_off + m * n;
                let a_slice = &av_s[a_off..a_off + m * n];

                // 1. a_gather = softmax(a_logits, dim=-1) over n for each hyperedge row.
                for r in 0..m {
                    let src = &a_slice[r * n..(r + 1) * n];
                    let dst = &mut a_gather[r * n..(r + 1) * n];
                    let mut mx = f32::NEG_INFINITY;
                    let mut non_finite = false;
                    for &v in src {
                        if !v.is_finite() {
                            non_finite = true;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                    if non_finite {
                        let mx = src.iter().copied().fold(f32::NEG_INFINITY, |a, b| {
                            if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
                        });
                        let mut sum = 0.0f32;
                        for (d, &v) in dst.iter_mut().zip(src) {
                            let e = (v - mx).exp();
                            *d = e;
                            sum += e;
                        }
                        for d in dst.iter_mut() {
                            *d /= sum;
                        }
                    } else {
                        let mut sum = 0.0f32;
                        for (d, &v) in dst.iter_mut().zip(src) {
                            let e = folo_core::cpu::fast_exp(v - mx);
                            *d = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        for d in dst.iter_mut() {
                            *d *= inv;
                        }
                    }
                }

                // 2. a_bcast = softmax(a_logits, dim=1) over m for each spatial token column.
                for j in 0..n {
                    let mut mx = f32::NEG_INFINITY;
                    let mut non_finite = false;
                    for r in 0..m {
                        let v = a_slice[r * n + j];
                        if !v.is_finite() {
                            non_finite = true;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                    if non_finite {
                        let mut mx = f32::NEG_INFINITY;
                        for r in 0..m {
                            let v = a_slice[r * n + j];
                            mx = if mx.is_nan() || v.is_nan() { f32::NAN } else { mx.max(v) };
                        }
                        let mut sum = 0.0f32;
                        for r in 0..m {
                            let e = (a_slice[r * n + j] - mx).exp();
                            a_bcast[r * n + j] = e;
                            sum += e;
                        }
                        for r in 0..m {
                            a_bcast[r * n + j] /= sum;
                        }
                    } else {
                        let mut sum = 0.0f32;
                        for r in 0..m {
                            let e = folo_core::cpu::fast_exp(a_slice[r * n + j] - mx);
                            a_bcast[r * n + j] = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        for r in 0..m {
                            a_bcast[r * n + j] *= inv;
                        }
                    }
                }

                // 3. e_t[m, cg] = a_gather[m, n] @ v[cg, n]^T (direct without transpose copy!).
                let ga = Strided::lhs(&a_gather, 0, n, 1);
                let vb = Strided::rhs(av_s, v_off, 1, n);
                gemm(m, cg, n, &ga, &vb, &mut e_t, cg, false, true);

                // 4. h1[m, cg] = silu(e_t @ w1^T + b1)
                let et_a = Strided::lhs(&e_t, 0, cg, 1);
                let w1_b = Strided::rhs(&w1, 0, 1, cg);
                gemm(m, cg, cg, &et_a, &w1_b, &mut h1, cg, false, true);
                if let Some(bs) = b1.as_deref() {
                    for row in h1.chunks_exact_mut(cg) {
                        for (x, &bv) in row.iter_mut().zip(bs) {
                            *x = folo_core::cpu::silu(*x + bv);
                        }
                    }
                } else {
                    for x in &mut h1 {
                        *x = folo_core::cpu::silu(*x);
                    }
                }

                // 5. e_ffn[m, cg] = h1 @ w2^T + b2; e_prime_t = e_t + e_ffn; g_scene = mean_m(e_prime_t)
                let h1_a = Strided::lhs(&h1, 0, cg, 1);
                let w2_b = Strided::rhs(&w2, 0, 1, cg);
                gemm(m, cg, cg, &h1_a, &w2_b, &mut e_ffn, cg, false, true);
                let gs_out = &mut g_scene[bi * cg..(bi + 1) * cg];
                gs_out.fill(0.0);
                for r in 0..m {
                    let row_et = &mut e_t[r * cg..(r + 1) * cg];
                    let row_ffn = &e_ffn[r * cg..(r + 1) * cg];
                    match b2.as_deref() {
                        Some(bs) => {
                            for c in 0..cg {
                                let ep = row_et[c] + row_ffn[c] + bs[c];
                                row_et[c] = ep;
                                gs_out[c] += ep;
                            }
                        }
                        None => {
                            for c in 0..cg {
                                let ep = row_et[c] + row_ffn[c];
                                row_et[c] = ep;
                                gs_out[c] += ep;
                            }
                        }
                    }
                }
                for g in gs_out.iter_mut() {
                    *g *= inv_m;
                }

                // 6. y_ctx[cg, n] = e_prime_t^T[cg, m] @ a_bcast[m, n]
                let ep_a = Strided::lhs(&e_t, 0, 1, cg);
                let ab_b = Strided::rhs(&a_bcast, 0, n, 1);
                let dst_y = &mut y_ctx[bi * cg * n..(bi + 1) * cg * n];
                gemm(cg, n, m, &ep_a, &ab_b, dst_y, n, false, true);
            }
        })?;

        Ok(Some((Tensor::from_vec(y_ctx, [b, cg, h, w])?, Tensor::from_vec(g_scene, [b, cg])?)))
    }

    /// Run P5 hypergraph context; returns `(y_p5 [B, C5, H, W], g_scene [B, C_g])`.
    ///
    /// Softmax attention weights, FFN `silu`, hyperedge residual addition, and global scene pooling (`g_scene`)
    /// always execute in `F32` (both under AMP `autocast()` and in explicit `F16` inference) so high-magnitude
    /// logits and reductions over `M` hyperedges never overflow or quantize in `F16`.
    pub fn forward_with_scene(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, _, h, w) = x.dims4()?;
        let av = self.av_proj.forward(x)?;
        let (y_ctx, g_scene) = self.attend_and_scene(&av, b, h, w)?;
        let y_ctx =
            if y_ctx.dtype() != self.out_proj.conv.weight.dtype() && folo_core::autocast_dtype().is_none() {
                y_ctx.to_dtype(self.out_proj.conv.weight.dtype())?
            } else {
                y_ctx
            };
        let proj = self.out_proj.forward(&y_ctx)?;
        let proj = if proj.dtype() != x.dtype() { proj.to_dtype(x.dtype())? } else { proj };
        let out = x.add(&proj)?;
        Ok((out, g_scene))
    }
}

impl Module for HyperContext {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("av_proj", &self.av_proj);
        f("ffn1", &self.ffn1);
        f("ffn2", &self.ffn2);
        f("out_proj", &self.out_proj);
    }
}

impl Forward for HyperContext {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_scene(x)?.0)
    }
}

impl Fuse for HyperContext {
    fn fuse_(&mut self) -> Result<()> {
        self.av_proj.fuse_()?;
        self.out_proj.fuse_()
    }
}

/// Zero-initialised global-context FiLM gate (`FullPAD-Lite`) applied strictly at `f32` head inputs
/// (after `neck.down3`/`neck.down4` taps so the CPU int8 neck has zero float barriers).
///
/// Computes `feat * (1 + tanh(proj(g_scene)))`. With `proj` zero-initialised, step 0 is exact identity.
pub struct FullPadGate {
    /// Linear projection `c_g → c_tgt`, zero-initialised for exact identity start.
    pub proj: Linear,
    /// Target feature channel count.
    pub c_tgt: usize,
}

impl FullPadGate {
    /// Build a zero-initialised `FullPadGate(c_g → c_tgt)`.
    pub fn new(c_g: usize, c_tgt: usize, init_ctx: &mut Init) -> Result<Self> {
        let proj = Linear::new(c_g, c_tgt, true, init_ctx.rng);
        init::constant_(&proj.weight, 0.0)?;
        if let Some(b) = &proj.bias {
            init::constant_(b, 0.0)?;
        }
        Ok(FullPadGate { proj, c_tgt })
    }

    /// Modulate `feat [B, C_tgt, H, W]` with `g_scene [B, C_g]`.
    ///
    /// Computes `feat * (1 + tanh(proj(g_scene)))` in `F32` before casting the modulated result back to
    /// `feat.dtype()`, so small gate updates near zero initialisation are not rounded away before multiplication
    /// under `F16` AMP or half-precision inference.
    pub fn forward(&self, feat: &Tensor, g_scene: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = feat.dims4()?;
        // Fused single-pass CPU F32 path under `no_grad` (avoids tanh/add_scalar/reshape tensor allocs).
        if !is_grad_enabled()
            && b > 0
            && c > 0
            && h > 0
            && w > 0
            && feat.device().is_cpu()
            && feat.dtype() == DType::F32
            && g_scene.device().is_cpu()
            && g_scene.dtype() == DType::F32
            && g_scene.dims().len() == 2
            && g_scene.dims()[0] == b
            && self.proj.weight.device().is_cpu()
            && self.proj.weight.dtype() == DType::F32
            && self.proj.weight.dims()[0] == c
        {
            let raw = self.proj.forward(g_scene)?;
            let hw = h * w;
            let mut out = folo_core::cpu::take_buffer(b * c * hw);
            let gate_vals: Vec<f32> = raw.to_vec::<f32>()?.into_iter().map(|v| 1.0 + v.tanh()).collect();
            feat.with_f32_slice(|fs| {
                let scale_plane = |(idx, dst): (usize, &mut [f32])| {
                    let g = gate_vals[idx];
                    let src = &fs[idx * hw..(idx + 1) * hw];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = s * g;
                    }
                };
                if out.len() >= 32_768 {
                    out.par_chunks_exact_mut(hw).enumerate().for_each(scale_plane);
                } else {
                    out.chunks_exact_mut(hw).enumerate().for_each(scale_plane);
                }
            })?;
            return Ok(Tensor::from_vec(out, [b, c, h, w])?);
        }
        let g_in = if g_scene.dtype() != self.proj.weight.dtype() && folo_core::autocast_dtype().is_none() {
            g_scene.to_dtype(self.proj.weight.dtype())?
        } else {
            g_scene.clone()
        };
        let raw = self.proj.forward(&g_in)?;
        let raw_f32 =
            if raw.dtype() == folo_core::DType::F32 { raw } else { raw.to_dtype(folo_core::DType::F32)? };
        let gate = raw_f32.tanh()?.add_scalar(1.0)?.reshape([b, c, 1, 1])?;
        if feat.dtype() == folo_core::DType::F32 {
            Ok(feat.mul(&gate)?)
        } else {
            Ok(feat.to_dtype(folo_core::DType::F32)?.mul(&gate)?.to_dtype(feat.dtype())?)
        }
    }
}

impl Module for FullPadGate {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("proj", &self.proj);
    }
}
