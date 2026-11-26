//! Multi-Head Cross-Attention ([`MultiHeadCrossAttention`]) with optional QK-RMSNorm, additive attention
//! masking, and 2D Rotary Positional Embedding ([`RoPE2d`]) integration.
//!
//! For query `Q: [B, L_q, embed_dim]`, key `K: [B, L_kv, kv_dim]`, and value `V: [B, L_kv, kv_dim]`:
//! ```text
//! q = split_heads(q_proj(Q))                        // [B, H, L_q, D_h]
//! k = split_heads(k_proj(K))                        // [B, H, L_kv, D_h]
//! v = split_heads(v_proj(V))                        // [B, H, L_kv, D_h]
//! scores = (q @ k^T) / sqrt(D_h) [+ attn_mask]      // [B, H, L_q, L_kv]
//! out = out_proj(merge_heads(softmax(scores) @ v))  // [B, L_q, embed_dim]
//! ```
//!
//! Tensor names: `q_proj.{weight,bias}`, `k_proj.{weight,bias}`, `v_proj.{weight,bias}`,
//! `out_proj.{weight,bias}`, plus optional `q_norm.weight`, `k_norm.weight` when `qk_norm = true`.

use folo_core::rng::Rng;
use folo_core::{DType, Tensor};

use super::rmsnorm::{RMSNORM_DEFAULT_EPS, RmsNorm};
use super::rope2d::RoPE2d;
use super::simple::Linear;
use crate::module::{Forward, Module};
use crate::{Error, Result};

type RopeSpec<'a> = (&'a RoPE2d, (usize, usize), (usize, usize));

/// Hyper-parameters for [`MultiHeadCrossAttention`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiHeadCrossAttentionConfig {
    /// Query and output embedding dimension (`>= 1`, divisible by `num_heads`).
    pub embed_dim: usize,
    /// Key / value input dimension (`None` defaults to `embed_dim`).
    pub kv_dim: Option<usize>,
    /// Number of attention heads (`>= 1`, divides `embed_dim`).
    pub num_heads: usize,
    /// Include bias terms in Q / K / V / output projections (default `true`).
    pub bias: bool,
    /// Apply per-head [`RmsNorm`] to Q and K before scaled dot-product attention (default `false`).
    pub qk_norm: bool,
    /// Epsilon for optional QK-RMSNorm (default [`RMSNORM_DEFAULT_EPS`]).
    pub qk_norm_eps: f32,
}

impl MultiHeadCrossAttentionConfig {
    /// Default cross-attention with `kv_dim = embed_dim`, biases enabled, and QK-norm disabled.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embed_dim,
            kv_dim: None,
            num_heads,
            bias: true,
            qk_norm: false,
            qk_norm_eps: RMSNORM_DEFAULT_EPS,
        }
    }

    /// Set a distinct key/value input dimension `kv_dim`.
    pub fn kv_dim(mut self, kv_dim: usize) -> Self {
        self.kv_dim = Some(kv_dim);
        self
    }

    /// Enable or disable projection biases.
    pub fn bias(mut self, on: bool) -> Self {
        self.bias = on;
        self
    }

    /// Enable or disable per-head QK-RMSNorm.
    pub fn qk_norm(mut self, on: bool) -> Self {
        self.qk_norm = on;
        self
    }

    /// Set the epsilon for optional per-head QK-RMSNorm.
    pub fn qk_norm_eps(mut self, eps: f32) -> Self {
        self.qk_norm_eps = eps;
        self
    }

    /// Effective key/value input dimension.
    #[inline]
    pub fn resolved_kv_dim(&self) -> usize {
        self.kv_dim.unwrap_or(self.embed_dim)
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.embed_dim == 0 {
            return Err(Error::Invalid("MultiHeadCrossAttention: embed_dim must be >= 1, got 0".into()));
        }
        if self.num_heads == 0 {
            return Err(Error::Invalid("MultiHeadCrossAttention: num_heads must be >= 1, got 0".into()));
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(Error::Invalid(format!(
                "MultiHeadCrossAttention: embed_dim ({}) must be divisible by num_heads ({})",
                self.embed_dim, self.num_heads
            )));
        }
        let kv = self.resolved_kv_dim();
        if kv == 0 {
            return Err(Error::Invalid("MultiHeadCrossAttention: kv_dim must be >= 1, got 0".into()));
        }
        if !(self.qk_norm_eps.is_finite() && self.qk_norm_eps > 0.0) {
            return Err(Error::Invalid(format!(
                "MultiHeadCrossAttention: qk_norm_eps must be finite and > 0, got {}",
                self.qk_norm_eps
            )));
        }
        Ok(())
    }
}

/// Multi-Head Cross-Attention layer supporting distinct query and key/value sequences, additive attention
/// masks, optional QK-RMSNorm, and 2D RoPE ([`RoPE2d`]).
pub struct MultiHeadCrossAttention {
    /// Query linear projection (`embed_dim -> embed_dim`).
    pub q_proj: Linear,
    /// Key linear projection (`kv_dim -> embed_dim`).
    pub k_proj: Linear,
    /// Value linear projection (`kv_dim -> embed_dim`).
    pub v_proj: Linear,
    /// Output linear projection (`embed_dim -> embed_dim`).
    pub out_proj: Linear,
    /// Optional per-head RMSNorm on projected queries (`head_dim`).
    pub q_norm: Option<RmsNorm>,
    /// Optional per-head RMSNorm on projected keys (`head_dim`).
    pub k_norm: Option<RmsNorm>,
    config: MultiHeadCrossAttentionConfig,
}

impl MultiHeadCrossAttention {
    /// Construct a [`MultiHeadCrossAttention`] module initialized from `rng`.
    pub fn new(config: MultiHeadCrossAttentionConfig, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let d = config.embed_dim;
        let kv = config.resolved_kv_dim();
        let head_dim = d / config.num_heads;
        let q_proj = Linear::new(d, d, config.bias, rng);
        let k_proj = Linear::new(kv, d, config.bias, rng);
        let v_proj = Linear::new(kv, d, config.bias, rng);
        let out_proj = Linear::new(d, d, config.bias, rng);
        let q_norm = config.qk_norm.then(|| RmsNorm::new(head_dim, config.qk_norm_eps)).transpose()?;
        let k_norm = config.qk_norm.then(|| RmsNorm::new(head_dim, config.qk_norm_eps)).transpose()?;
        Ok(Self { q_proj, k_proj, v_proj, out_proj, q_norm, k_norm, config })
    }

    /// Layer configuration.
    #[inline]
    pub fn config(&self) -> MultiHeadCrossAttentionConfig {
        self.config
    }

    /// Embedding dimension `embed_dim`.
    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Key/value input dimension.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.config.resolved_kv_dim()
    }

    /// Number of heads.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    /// Dimension per head (`embed_dim / num_heads`).
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.config.embed_dim / self.config.num_heads
    }

    /// Cross-attention where key and value share the same `key_value` sequence `[B, L_kv, kv_dim]`.
    pub fn forward_cross(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_cross_qkv(query, key_value, key_value, attn_mask)
    }

    /// Cross-attention with separate `query: [B, L_q, embed_dim]`, `key: [B, L_kv, kv_dim]`, and
    /// `value: [B, L_kv, kv_dim]`, plus an optional additive `attn_mask` broadcastable to
    /// `[B, num_heads, L_q, L_kv]`.
    pub fn forward_cross_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_impl(query, key, value, None, attn_mask)
    }

    /// Cross-attention applying 2D Rotary Positional Embeddings ([`RoPE2d`]) to projected per-head `Q`
    /// (with spatial grid `q_hw = (H_q, W_q)`) and `K` (with spatial grid `kv_hw = (H_kv, W_kv)`).
    pub fn forward_cross_with_rope2d(
        &self,
        query: &Tensor,
        q_hw: (usize, usize),
        key_value: &Tensor,
        kv_hw: (usize, usize),
        rope: &RoPE2d,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_impl(query, key_value, key_value, Some((rope, q_hw, kv_hw)), attn_mask)
    }

    fn forward_impl(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        rope: Option<RopeSpec<'_>>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dev = self.q_proj.weight.device();
        for (label, t) in [("query", query), ("key", key), ("value", value)] {
            if !matches!(t.dtype(), DType::F32 | DType::F16) {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: {label} dtype must be f32 or f16, got {}",
                    t.dtype()
                )));
            }
            if t.device() != dev {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: {label} on {:?} does not match weights on {dev:?}",
                    t.device()
                )));
            }
        }
        if query.dtype() != key.dtype() || query.dtype() != value.dtype() {
            return Err(Error::Invalid(format!(
                "MultiHeadCrossAttention: query/key/value dtypes must match (got {}, {}, {})",
                query.dtype(),
                key.dtype(),
                value.dtype()
            )));
        }
        let (bq, lq, dq) = query.dims3().map_err(|_| {
            Error::Invalid(format!(
                "MultiHeadCrossAttention: query must be [B, L_q, D], got {:?}",
                query.dims()
            ))
        })?;
        let (bk, lk, dk) = key.dims3().map_err(|_| {
            Error::Invalid(format!(
                "MultiHeadCrossAttention: key must be [B, L_kv, D_kv], got {:?}",
                key.dims()
            ))
        })?;
        let (bv, lv, dv) = value.dims3().map_err(|_| {
            Error::Invalid(format!(
                "MultiHeadCrossAttention: value must be [B, L_kv, D_kv], got {:?}",
                value.dims()
            ))
        })?;
        let d = self.config.embed_dim;
        let kv_d = self.config.resolved_kv_dim();
        if dq != d || dk != kv_d || dv != kv_d {
            return Err(Error::Invalid(format!(
                "MultiHeadCrossAttention: dims mismatch (query {dq} vs {d}, key {dk} / value {dv} vs {kv_d})"
            )));
        }
        if bq != bk || bq != bv || lk != lv {
            return Err(Error::Invalid(format!(
                "MultiHeadCrossAttention: batch/length mismatch (Q [{bq},{lq}], K [{bk},{lk}], V [{bv},{lv}])"
            )));
        }
        let h = self.config.num_heads;
        let hd = d / h;
        if let Some(mask) = attn_mask {
            if mask.device() != dev {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: attn_mask on {:?} does not match weights on {dev:?}",
                    mask.device()
                )));
            }
            if !matches!(mask.dtype(), DType::F32 | DType::F16) {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: attn_mask dtype must be f32 or f16, got {}",
                    mask.dtype()
                )));
            }
            if !broadcasts_to(mask.dims(), &[bq, h, lq, lk]) {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: attn_mask shape {:?} cannot broadcast to [{bq}, {h}, {lq}, {lk}]",
                    mask.dims()
                )));
            }
        }
        if let Some((rope2d, (qh, qw), (kvh, kvw))) = rope {
            if rope2d.head_dim() != hd {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: RoPE2d head_dim {} != attention head_dim {hd}",
                    rope2d.head_dim()
                )));
            }
            if rope2d.inv_freq.device() != dev {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: RoPE2d on {:?} does not match weights on {dev:?}",
                    rope2d.inv_freq.device()
                )));
            }
            let q_cells = qh.checked_mul(qw).ok_or_else(|| {
                Error::Invalid(format!("MultiHeadCrossAttention: q_hw ({qh}, {qw}) overflows"))
            })?;
            let kv_cells = kvh.checked_mul(kvw).ok_or_else(|| {
                Error::Invalid(format!("MultiHeadCrossAttention: kv_hw ({kvh}, {kvw}) overflows"))
            })?;
            if q_cells != lq || kv_cells != lk {
                return Err(Error::Invalid(format!(
                    "MultiHeadCrossAttention: RoPE2d grid sizes ({qh}*{qw}={q_cells}, {kvh}*{kvw}={kv_cells}) do not match sequence lengths (L_q={lq}, L_kv={lk})"
                )));
            }
        }
        if lq > 0 && lk == 0 {
            return Err(Error::Invalid(
                "MultiHeadCrossAttention: key/value sequence length L_kv must be >= 1 when L_q > 0".into(),
            ));
        }
        if bq == 0 || lq == 0 {
            // Preserve autograd linkage through every participating projection, norm, and input.
            let mut q_empty = self.q_proj.forward(query)?;
            let mut k_empty = self.k_proj.forward(key)?;
            let v_empty = self.v_proj.forward(value)?;
            if let Some(qn) = &self.q_norm {
                q_empty = qn.forward(&q_empty.reshape([bq, lq, h, hd])?)?.reshape([bq, lq, d])?;
            }
            if let Some(kn) = &self.k_norm {
                k_empty = kn.forward(&k_empty.reshape([bq, lk, h, hd])?)?.reshape([bq, lk, d])?;
            }
            let mut anchor = k_empty.sum_all()?.mul_scalar(0.0)?.add(&v_empty.sum_all()?.mul_scalar(0.0)?)?;
            if let Some(mask) = attn_mask
                && mask.requires_grad()
            {
                let m_sum = mask.to_dtype(anchor.dtype())?.sum_all()?.mul_scalar(0.0)?;
                anchor = anchor.add(&m_sum)?;
            }
            let anchor =
                if anchor.dtype() != q_empty.dtype() { anchor.to_dtype(q_empty.dtype())? } else { anchor };
            return self.out_proj.forward(&q_empty.add(&anchor)?);
        }

        let mut q = self.q_proj.forward(query)?.reshape([bq, lq, h, hd])?.transpose(1, 2)?.contiguous()?;
        let mut k = self.k_proj.forward(key)?.reshape([bq, lk, h, hd])?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(value)?.reshape([bq, lk, h, hd])?.transpose(1, 2)?.contiguous()?;

        if let Some(qn) = &self.q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(kn) = &self.k_norm {
            k = kn.forward(&k)?;
        }
        if let Some((rope2d, (qh, qw), (kvh, kvw))) = rope {
            q = rope2d.apply(&q, qh, qw)?;
            k = rope2d.apply(&k, kvh, kvw)?;
        }

        let scale = (hd as f64).sqrt().recip() as f32;
        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.mul_scalar(scale)?;
        let weights = if let Some(mask) = attn_mask {
            let m =
                if mask.dtype() != scores.dtype() { mask.to_dtype(scores.dtype())? } else { mask.clone() };
            scores = scores.add(&m)?;
            // Guard fully-masked rows (every key logit == -inf) so softmax yields exact 0 weights
            // and 0 gradients instead of NaN, while letting genuine NaN logits propagate unchanged.
            let is_neg_inf = scores.eq_scalar(f32::NEG_INFINITY)?;
            let all_neg_inf = is_neg_inf
                .to_dtype(DType::F32)?
                .sum(3, true)?
                .eq_scalar(lk as f32)?
                .broadcast_to(scores.shape())?;
            let zeros = Tensor::zeros_with(scores.shape(), scores.dtype(), scores.device())?;
            let safe_scores = all_neg_inf.where_cond(&zeros, &scores)?;
            let raw_weights = safe_scores.softmax(3)?;
            let w_zeros = Tensor::zeros_with(raw_weights.shape(), raw_weights.dtype(), raw_weights.device())?;
            all_neg_inf.where_cond(&w_zeros, &raw_weights)?
        } else {
            scores.softmax(3)?
        };
        let weights = if weights.dtype() != v.dtype() { weights.to_dtype(v.dtype())? } else { weights };
        let ctx = weights.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape([bq, lq, d])?;
        self.out_proj.forward(&ctx)
    }
}

fn broadcasts_to(src: &[usize], dst: &[usize]) -> bool {
    if src.len() > dst.len() {
        return false;
    }
    let offset = dst.len() - src.len();
    src.iter().enumerate().all(|(i, &s)| s == 1 || s == dst[offset + i])
}

impl Module for MultiHeadCrossAttention {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("q_proj", &self.q_proj);
        f("k_proj", &self.k_proj);
        f("v_proj", &self.v_proj);
        f("out_proj", &self.out_proj);
        if let Some(qn) = &self.q_norm {
            f("q_norm", qn);
        }
        if let Some(kn) = &self.k_norm {
            f("k_norm", kn);
        }
    }
}

impl Forward for MultiHeadCrossAttention {
    /// Self-attention shorthand (`forward_cross(x, x, None)`), valid when `kv_dim == embed_dim`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_cross(x, x, None)
    }
}
