//! FOLO v2 hardware-co-designed building blocks:
//! - [`space_to_depth_2x2`]: lossless 2×2 spatial-to-channel folding (ONNX opset-17 `SpaceToDepth`).
//! - [`WsQARepConv3x3`]: weight-space quantization-aware reparameterized 3×3 convolution with
//!   `3×3 + 1×3 + 3×1 + 1×1 + γ_id·I` branches folded into a single `W_eff` before one Post-BN during
//!   training and algebraically collapsed into one dense `3×3` conv at inference.
//! - [`FoloBlock`]: `ceil16`-aligned QARep-CSP / Rep-ELAN block with `≤ 3` zero-copy `forward_cat` parts.
//! - [`HyperContext`]: linear `O(N·M)` P5 hypergraph global context (`HyperACE-Lite`).
//! - [`FullPadGate`]: zero-init residual FiLM gate injecting P5 scene context into `f32` head inputs.

use std::sync::RwLock;

use folo_core::{Conv2dParams, DType, Device, Tensor, is_grad_enabled, no_grad};
use folo_nn::{
    Activation, BatchNorm2d, Conv2d, Conv2dConfig, ConvBnAct, Error, Forward, Linear, Module, ModuleList,
    Result, TensorKind, fuse_conv_bn, init,
};

use crate::model::blocks::{Fuse, Init};

const CONV3X3_S1_P1: Conv2dParams =
    Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };

/// Lossless `2×2` space-to-depth transform matching ONNX opset-17 `SpaceToDepth(blocksize=2)`:
/// maps `[B, C, H, W]` (with even `H, W`) to `[B, 4·C, H/2, W/2]` by concatenating the four sub-pixel
/// phases `(0,0), (0,1), (1,0), (1,1)` along the channel axis.
pub fn space_to_depth_2x2(x: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = x
        .dims4()
        .map_err(|_| Error::Invalid(format!("space_to_depth_2x2: expected NCHW, got {:?}", x.dims())))?;
    if !h.is_multiple_of(2) || !w.is_multiple_of(2) {
        return Err(Error::Invalid(format!("space_to_depth_2x2: spatial dims ({h}, {w}) must be even")));
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

    /// Run P5 hypergraph context; returns `(y_p5 [B, C5, H, W], g_scene [B, C_g])`.
    pub fn forward_with_scene(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, _, h, w) = x.dims4()?;
        let n = h * w;
        let av = self.av_proj.forward(x)?;
        let a_logits = av.narrow(1, 0, self.m_edges)?.reshape([b, self.m_edges, n])?;
        let v = av.narrow(1, self.m_edges, self.c_g)?.reshape([b, self.c_g, n])?;
        let a_gather = a_logits.softmax(-1)?; // [B, M, N]: gather over spatial tokens
        let a_bcast = a_logits.softmax(1)?; // [B, M, N]: convex combination over hyperedges (H*W-invariant)
        let e = v.matmul(&a_gather.transpose(1, 2)?)?; // [B, C_g, M]
        let e_t = e.transpose(1, 2)?; // [B, M, C_g]
        let e_ffn = self.ffn2.forward(&self.ffn1.forward(&e_t)?.silu()?)?;
        let e_prime_t = e_t.add(&e_ffn)?; // [B, M, C_g]
        let g_scene = e_prime_t.mean(1, false)?; // [B, C_g]
        let e_prime = e_prime_t.transpose(1, 2)?; // [B, C_g, M]
        let y_ctx = e_prime.matmul(&a_bcast)?.reshape([b, self.c_g, h, w])?;
        let proj = self.out_proj.forward(&y_ctx)?;
        let x_res = if x.dtype() != proj.dtype() && folo_core::autocast_dtype().is_some() {
            x.to_dtype(proj.dtype())?
        } else {
            x.clone()
        };
        let out = x_res.add(&proj)?;
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
    pub fn forward(&self, feat: &Tensor, g_scene: &Tensor) -> Result<Tensor> {
        let (b, c, _, _) = feat.dims4()?;
        let gate = self.proj.forward(g_scene)?.tanh()?.add_scalar(1.0)?.reshape([b, c, 1, 1])?;
        let gate = if gate.dtype() != feat.dtype() && folo_core::autocast_dtype().is_some() {
            gate.to_dtype(feat.dtype())?
        } else {
            gate
        };
        Ok(feat.mul(&gate)?)
    }
}

impl Module for FullPadGate {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("proj", &self.proj);
    }
}
