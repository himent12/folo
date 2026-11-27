//! The building blocks of docs/MODEL.md §2: Bottleneck, C3k, C3k2, SPPF, Attention, PSABlock and C2PSA, on top
//! of folo-nn's `ConvBnAct` (MODEL.md's `Conv`).
//!
//! Every block reports its children in canonical order (MODEL.md §5) and draws its initial weights in that
//! same order, so `named_parameters()` gives the documented `.folo` names and one seed gives one model.

use folo_core::cpu::gemm::{Strided, gemm};
use folo_core::rng::Rng;
use folo_core::{DType, Tensor};
use folo_nn::{
    Activation, BN_EPS, BN_MOMENTUM, Conv2dConfig, ConvBnAct, Error, Forward, Module, ModuleList, Result,
};
use rayon::prelude::*;

/// Construction context: the model's RNG stream and the BatchNorm options of its config.
pub struct Init<'a> {
    /// The seeded stream every conv weight draws from, in canonical order.
    pub rng: &'a mut Rng,
    /// BatchNorm ε (`folo.config.bn_eps`).
    pub bn_eps: f64,
    /// BatchNorm momentum (`folo.config.bn_momentum`).
    pub bn_momentum: f64,
}

impl<'a> Init<'a> {
    /// docs/MODEL.md's BatchNorm defaults (`eps = 1e-3`, `momentum = 0.03`).
    pub fn new(rng: &'a mut Rng) -> Self {
        Init { rng, bn_eps: BN_EPS, bn_momentum: BN_MOMENTUM }
    }

    /// `Conv(c1→c2, k, s, g, act)` (MODEL.md §2.1).
    pub fn conv_with(
        &mut self,
        c1: usize,
        c2: usize,
        k: usize,
        s: usize,
        g: usize,
        act: Activation,
    ) -> ConvBnAct {
        let cfg = Conv2dConfig::new(c1, c2, k).stride(s).groups(g);
        ConvBnAct::with_config(cfg, act, self.bn_eps, self.bn_momentum, self.rng)
    }

    /// `Conv(c1→c2, k, s)` with SiLU.
    pub fn conv(&mut self, c1: usize, c2: usize, k: usize, s: usize) -> ConvBnAct {
        self.conv_with(c1, c2, k, s, 1, Activation::Silu)
    }
}

/// Fold every BatchNorm of a block into its conv, in place (docs/MODEL.md §9.1).
pub trait Fuse {
    /// Fuse in place; a no-op for parts already fused.
    fn fuse_(&mut self) -> Result<()>;
}

impl Fuse for ConvBnAct {
    fn fuse_(&mut self) -> Result<()> {
        ConvBnAct::fuse_(self)
    }
}

impl<M: Fuse> Fuse for ModuleList<M> {
    fn fuse_(&mut self) -> Result<()> {
        self.items_mut().iter_mut().try_for_each(Fuse::fuse_)
    }
}

/// `int(c · e)` of MODEL.md (truncation).
fn hidden(c: usize, e: f64) -> usize {
    (c as f64 * e) as usize
}

/// docs/MODEL.md §2.2: `Bottleneck(c1→c2, k=(3,3), e, shortcut=true)`. Residual when `c1 == c2`.
/// Tensor names: `cv1.*`, `cv2.*`.
pub struct Bottleneck {
    /// `Conv(c1→h, 3, 1)`.
    pub cv1: ConvBnAct,
    /// `Conv(h→c2, 3, 1)`.
    pub cv2: ConvBnAct,
    /// Whether the input is added to the output.
    pub residual: bool,
}

impl Bottleneck {
    /// `Bottleneck(c1→c2, k=(3,3), e)` with the shortcut on (it applies only when `c1 == c2`).
    pub fn new(c1: usize, c2: usize, e: f64, init: &mut Init) -> Self {
        let h = hidden(c2, e);
        let cv1 = init.conv(c1, h, 3, 1);
        let cv2 = init.conv(h, c2, 3, 1);
        Bottleneck { cv1, cv2, residual: c1 == c2 }
    }
}

impl Module for Bottleneck {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cv1", &self.cv1);
        f("cv2", &self.cv2);
    }
}

impl Forward for Bottleneck {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv2.forward(&self.cv1.forward(x)?)?;
        Ok(if self.residual { x.add(&y)? } else { y })
    }
}

impl Fuse for Bottleneck {
    fn fuse_(&mut self) -> Result<()> {
        self.cv1.fuse_()?;
        self.cv2.fuse_()
    }
}

/// docs/MODEL.md §2.3: `C3k(c1→c2, n=2, e=0.5)`. Tensor names: `cv1.*`, `cv2.*`, `cv3.*`, `block{j}.*`.
pub struct C3k {
    /// `Conv(c1→h, 1, 1)`, the bottleneck path.
    pub cv1: ConvBnAct,
    /// `Conv(c1→h, 1, 1)`, the bypass.
    pub cv2: ConvBnAct,
    /// `Conv(2h→c2, 1, 1)`.
    pub cv3: ConvBnAct,
    /// `n` bottlenecks `h→h`, `e = 1.0`.
    pub blocks: ModuleList<Bottleneck>,
}

impl C3k {
    /// `C3k(c1→c2, n, e)`.
    pub fn new(c1: usize, c2: usize, n: usize, e: f64, init: &mut Init) -> Self {
        let h = hidden(c2, e);
        let cv1 = init.conv(c1, h, 1, 1);
        let cv2 = init.conv(c1, h, 1, 1);
        let cv3 = init.conv(2 * h, c2, 1, 1);
        let blocks = (0..n).map(|_| Bottleneck::new(h, h, 1.0, init)).collect();
        C3k { cv1, cv2, cv3, blocks: ModuleList::with_prefix("block", blocks) }
    }
}

impl Module for C3k {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cv1", &self.cv1);
        f("cv2", &self.cv2);
        f("cv3", &self.cv3);
        self.blocks.visit_children(f);
    }
}

impl Forward for C3k {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.blocks.forward(&self.cv1.forward(x)?)?;
        let b = self.cv2.forward(x)?;
        self.cv3.forward_cat(&[&a, &b])
    }
}

impl Fuse for C3k {
    fn fuse_(&mut self) -> Result<()> {
        self.cv1.fuse_()?;
        self.cv2.fuse_()?;
        self.cv3.fuse_()?;
        self.blocks.fuse_()
    }
}

/// An inner block of a [`C3k2`]: a [`C3k`] when `c3k`, else a [`Bottleneck`] (`e = 0.5`).
pub enum C3k2Block {
    /// `Bottleneck(c→c, k=(3,3), e=0.5)`.
    Bottleneck(Bottleneck),
    /// `C3k(c→c, n=2, e=0.5)`.
    C3k(C3k),
}

impl Module for C3k2Block {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        match self {
            C3k2Block::Bottleneck(b) => b.visit_children(f),
            C3k2Block::C3k(b) => b.visit_children(f),
        }
    }
}

impl Forward for C3k2Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            C3k2Block::Bottleneck(b) => b.forward(x),
            C3k2Block::C3k(b) => b.forward(x),
        }
    }
}

impl Fuse for C3k2Block {
    fn fuse_(&mut self) -> Result<()> {
        match self {
            C3k2Block::Bottleneck(b) => b.fuse_(),
            C3k2Block::C3k(b) => b.fuse_(),
        }
    }
}

/// docs/MODEL.md §2.4: `C3k2(c1→c2, n, c3k, e)`. Tensor names: `cv1.*`, `cv2.*`, `block{i}.*`.
pub struct C3k2 {
    /// `Conv(c1→2c, 1, 1)`, split into `[c, c]`.
    pub cv1: ConvBnAct,
    /// `Conv((2+n)·c → c2, 1, 1)`.
    pub cv2: ConvBnAct,
    /// The `n` inner blocks, applied in a chain to the second half of `cv1`.
    pub blocks: ModuleList<C3k2Block>,
    c: usize,
}

impl C3k2 {
    /// `C3k2(c1→c2, n, c3k, e)`.
    pub fn new(c1: usize, c2: usize, n: usize, c3k: bool, e: f64, init: &mut Init) -> Self {
        let c = hidden(c2, e);
        let cv1 = init.conv(c1, 2 * c, 1, 1);
        let cv2 = init.conv((2 + n) * c, c2, 1, 1);
        let blocks = (0..n)
            .map(|_| {
                if c3k {
                    C3k2Block::C3k(C3k::new(c, c, 2, 0.5, init))
                } else {
                    C3k2Block::Bottleneck(Bottleneck::new(c, c, 0.5, init))
                }
            })
            .collect();
        C3k2 { cv1, cv2, blocks: ModuleList::with_prefix("block", blocks), c }
    }
}

impl Module for C3k2 {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cv1", &self.cv1);
        f("cv2", &self.cv2);
        self.blocks.visit_children(f);
    }
}

impl C3k2 {
    /// `forward(&Tensor::cat(xs, 1))` without writing the concatenation when the convs run fused
    /// ([`ConvBnAct::forward_cat`]).
    pub fn forward_cat(&self, xs: &[&Tensor]) -> Result<Tensor> {
        // `cat(split(y), b_1, ..)` is `cat(y, b_1, ..)`: cv2 reads y whole, and the chain starts on its
        // second half (a channel slice, read in place by the CPU conv).
        let y = self.cv1.forward_cat(xs)?;
        let mut outs: Vec<Tensor> = Vec::with_capacity(self.blocks.len());
        for b in self.blocks.iter() {
            let next = match outs.last() {
                Some(prev) => b.forward(prev)?,
                None => b.forward(&y.narrow(1, self.c, self.c)?)?,
            };
            outs.push(next);
        }
        let mut parts = Vec::with_capacity(1 + outs.len());
        parts.push(&y);
        parts.extend(&outs);
        self.cv2.forward_cat(&parts)
    }
}

impl Forward for C3k2 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_cat(&[x])
    }
}

impl Fuse for C3k2 {
    fn fuse_(&mut self) -> Result<()> {
        self.cv1.fuse_()?;
        self.cv2.fuse_()?;
        self.blocks.fuse_()
    }
}

/// docs/MODEL.md §2.5: `SPPF(c1→c2, k=5)`, three chained `k × k` max pools (stride 1, padding `k / 2`).
/// Tensor names: `cv1.*`, `cv2.*`.
pub struct Sppf {
    /// `Conv(c1→c1/2, 1, 1)` with SiLU.
    pub cv1: ConvBnAct,
    /// `Conv(4·(c1/2)→c2, 1, 1)`.
    pub cv2: ConvBnAct,
    /// Pool window.
    pub k: usize,
}

impl Sppf {
    /// `SPPF(c1→c2, k)`.
    pub fn new(c1: usize, c2: usize, k: usize, init: &mut Init) -> Self {
        let h = c1 / 2;
        let cv1 = init.conv(c1, h, 1, 1);
        let cv2 = init.conv(4 * h, c2, 1, 1);
        Sppf { cv1, cv2, k }
    }
}

impl Module for Sppf {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cv1", &self.cv1);
        f("cv2", &self.cv2);
    }
}

impl Forward for Sppf {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y0 = self.cv1.forward(x)?;
        let pool = |t: &Tensor| t.max_pool2d(self.k, 1, self.k / 2);
        let y1 = pool(&y0)?;
        let y2 = pool(&y1)?;
        let y3 = pool(&y2)?;
        self.cv2.forward_cat(&[&y0, &y1, &y2, &y3])
    }
}

impl Fuse for Sppf {
    fn fuse_(&mut self) -> Result<()> {
        self.cv1.fuse_()?;
        self.cv2.fuse_()
    }
}

/// docs/MODEL.md §2.6: multi-head self-attention over the `H·W` positions. Tensor names: `qkv.*`, `proj.*`,
/// `pe.*`.
pub struct Attention {
    /// `Conv(C → C + 2·kd·heads, 1, 1, act=none)`.
    pub qkv: ConvBnAct,
    /// `Conv(C→C, 1, 1, act=none)`.
    pub proj: ConvBnAct,
    /// Positional encoding, depthwise `Conv(C→C, 3, 1, g=C, act=none)` on `v`.
    pub pe: ConvBnAct,
    /// Number of heads.
    pub heads: usize,
    /// Query / key width per head, `hd / 2`.
    pub key_dim: usize,
    /// Value width per head, `C / heads`.
    pub head_dim: usize,
}

impl Attention {
    /// `Attention(C, heads)`. `C` must be divisible by `heads`.
    pub fn new(c: usize, heads: usize, init: &mut Init) -> Result<Self> {
        if heads == 0 || !c.is_multiple_of(heads) || c / heads < 2 {
            return Err(Error::Invalid(format!(
                "Attention: {c} channels cannot be split into {heads} heads"
            )));
        }
        let head_dim = c / heads;
        let key_dim = head_dim / 2;
        let qkv = init.conv_with(c, c + 2 * key_dim * heads, 1, 1, 1, Activation::Identity);
        let proj = init.conv_with(c, c, 1, 1, 1, Activation::Identity);
        let pe = init.conv_with(c, c, 3, 1, c, Activation::Identity);
        Ok(Attention { qkv, proj, pe, heads, key_dim, head_dim })
    }

    /// Computes `o + pe(v)` from `qkv` (`[b, heads * (2 * kd + hd), h, w]`), shared by [`Attention::forward`]
    /// and the quantised graph walk (`model::quant::walk`).
    ///
    /// On CPU `F32` under `no_grad` with a standard fused 3×3 depthwise `pe`, runs directly on `qkv`'s slices
    /// via strided GEMMs, in-place scaled softmax, and fused `o += pe(v)` with zero intermediate Q/K/V/attn
    /// copies or `pe` buffer allocations; falls back to generic tensor ops otherwise.
    pub(crate) fn attend_and_pe(&self, qkv: &Tensor, b: usize, h: usize, w: usize) -> Result<Tensor> {
        let (heads, kd, hd) = (self.heads, self.key_dim, self.head_dim);
        let c = heads * hd;
        let n = h * w;
        if let Some(out) = self.attend_and_pe_cpu_fast(qkv, b, h, w)? {
            return Ok(out);
        }
        let qkv = qkv.reshape([b, heads, 2 * kd + hd, n])?;
        let q = qkv.narrow(2, 0, kd)?;
        let k = qkv.narrow(2, kd, kd)?;
        let v = qkv.narrow(2, 2 * kd, hd)?;
        // A[b, h, i, j] = softmax_j(scale · Σ_d q[d, i] · k[d, j]).
        let scale = (kd as f64).powf(-0.5) as f32;
        let attn = q.transpose(2, 3)?.matmul(&k)?.mul_scalar(scale)?.softmax(-1)?;
        // o[b, h, d, i] = Σ_j v[d, j] · A[i, j].
        let o = v.matmul(&attn.transpose(2, 3)?)?.reshape([b, c, h, w])?;
        let v = v.reshape([b, c, h, w])?;
        Ok(o.add(&self.pe.forward(&v)?)?)
    }

    fn attend_and_pe_cpu_fast(&self, qkv: &Tensor, b: usize, h: usize, w: usize) -> Result<Option<Tensor>> {
        let (heads, kd, hd) = (self.heads, self.key_dim, self.head_dim);
        let Some(c) = heads.checked_mul(hd) else { return Ok(None) };
        let Some(n) = h.checked_mul(w) else { return Ok(None) };
        let Some(qkv_c) =
            kd.checked_mul(2).and_then(|v| v.checked_add(hd)).and_then(|v| v.checked_mul(heads))
        else {
            return Ok(None);
        };
        if n.checked_mul(n).is_none() || b.checked_mul(c).and_then(|v| v.checked_mul(n)).is_none() {
            return Ok(None);
        }
        let pe_cfg = self.pe.conv.config();
        let eligible = !folo_core::is_grad_enabled()
            && b > 0
            && h > 0
            && w > 0
            && heads > 0
            && kd > 0
            && hd > 0
            && qkv.device().is_cpu()
            && qkv.dtype() == DType::F32
            && qkv.dims() == [b, qkv_c, h, w]
            && self.pe.bn.is_none()
            && self.pe.act == Activation::Identity
            && pe_cfg.in_channels == c
            && pe_cfg.out_channels == c
            && pe_cfg.kernel == 3
            && pe_cfg.stride == 1
            && pe_cfg.padding == 1
            && pe_cfg.dilation == 1
            && pe_cfg.groups == c
            && self.pe.conv.weight.device().is_cpu()
            && self.pe.conv.weight.dtype() == DType::F32
            && self.pe.conv.weight.dims() == [c, 1, 3, 3]
            && self.pe.conv.bias.as_ref().is_none_or(|bias| {
                bias.device().is_cpu() && bias.dtype() == DType::F32 && bias.dims() == [c]
            });
        if !eligible {
            return Ok(None);
        }

        // Snapshot small depthwise PE weights/bias before locking `qkv` storage (avoids any aliasing lock issue).
        let pe_w = self.pe.conv.weight.to_vec::<f32>()?;
        let pe_b = self.pe.conv.bias.as_ref().map(|bias| bias.to_vec::<f32>()).transpose()?;
        let scale = (kd as f64).powf(-0.5) as f32;
        let head_stride = (2 * kd + hd) * n;
        let batch_stride = heads * head_stride;

        let mut out = folo_core::cpu::take_buffer(b * c * n);
        let mut scores = folo_core::cpu::take_buffer(n * n);

        qkv.with_f32_slice(|qkv_s| {
            for bi in 0..b {
                for hi in 0..heads {
                    let base = bi * batch_stride + hi * head_stride;
                    let q_off = base;
                    let k_off = base + kd * n;
                    let v_off = base + 2 * kd * n;

                    // 1. scores[i, j] = Σ_d q[d, i] · k[d, j] (strided in place, zero copies).
                    let qa = Strided::lhs(qkv_s, q_off, 1, n);
                    let kb = Strided::rhs(qkv_s, k_off, n, 1);
                    gemm(n, n, kd, &qa, &kb, &mut scores, n, false, true);

                    // 2. In-place scaled softmax over rows of `scores[n, n]`.
                    let softmax_row = |row: &mut [f32]| {
                        let mut max = f32::NEG_INFINITY;
                        let mut non_finite = false;
                        for x in row.iter_mut() {
                            let v = *x * scale;
                            *x = v;
                            if !v.is_finite() {
                                non_finite = true;
                            }
                            if v > max {
                                max = v;
                            }
                        }
                        if non_finite {
                            let mut sum = 0.0f32;
                            let m = row.iter().copied().fold(f32::NEG_INFINITY, |a, b| {
                                if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
                            });
                            for x in row.iter_mut() {
                                let e = (*x - m).exp();
                                *x = e;
                                sum += e;
                            }
                            for x in row.iter_mut() {
                                *x /= sum;
                            }
                            return;
                        }
                        let mut sum = 0.0f32;
                        for x in row.iter_mut() {
                            let e = folo_core::cpu::fast_exp(*x - max);
                            *x = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        for x in row.iter_mut() {
                            *x *= inv;
                        }
                    };
                    if scores.len() >= 16_384 {
                        scores.par_chunks_mut(n).for_each(softmax_row);
                    } else {
                        scores.chunks_mut(n).for_each(softmax_row);
                    }

                    // 3. o[d, i] = Σ_j v[d, j] · scores[i, j] (v read in place, scores^T strided).
                    let va = Strided::lhs(qkv_s, v_off, n, 1);
                    let ab = Strided::rhs(&scores, 0, 1, n);
                    let out_head = &mut out[(bi * c + hi * hd) * n..(bi * c + (hi + 1) * hd) * n];
                    gemm(hd, n, n, &va, &ab, out_head, n, false, true);
                }
            }

            // 4. Fused depthwise 3×3 `pe(v)` accumulated directly into `out` (zero v-copy or pe-buffer alloc).
            let add_pe_plane = |(idx, out_plane): (usize, &mut [f32])| {
                let bi = idx / c;
                let ch = idx % c;
                let hi = ch / hd;
                let d = ch % hd;
                let v_off = bi * batch_stride + hi * head_stride + (2 * kd + d) * n;
                let v_plane = &qkv_s[v_off..v_off + n];
                let w9 = &pe_w[ch * 9..ch * 9 + 9];
                let bias = pe_b.as_ref().map_or(0.0, |bs| bs[ch]);
                let (w00, w01, w02) = (w9[0], w9[1], w9[2]);
                let (w10, w11, w12) = (w9[3], w9[4], w9[5]);
                let (w20, w21, w22) = (w9[6], w9[7], w9[8]);

                // Interior rows & columns (branchless, auto-vectorises with AVX2 FMA).
                if h >= 3 && w >= 3 {
                    for y in 1..h - 1 {
                        let r0 = &v_plane[(y - 1) * w..y * w];
                        let r1 = &v_plane[y * w..(y + 1) * w];
                        let r2 = &v_plane[(y + 1) * w..(y + 2) * w];
                        let dst = &mut out_plane[y * w..(y + 1) * w];
                        for x in 1..w - 1 {
                            let taps = w00 * r0[x - 1]
                                + w01 * r0[x]
                                + w02 * r0[x + 1]
                                + w10 * r1[x - 1]
                                + w11 * r1[x]
                                + w12 * r1[x + 1]
                                + w20 * r2[x - 1]
                                + w21 * r2[x]
                                + w22 * r2[x + 1];
                            dst[x] += taps + bias;
                        }
                    }
                }
                // Border pixels (top row, bottom row, left/right edges of interior rows).
                let border_pixel = |y: usize, x: usize| -> f32 {
                    let mut taps = 0.0f32;
                    for ky in 0..3usize {
                        let iy = y + ky;
                        if iy == 0 || iy > h {
                            continue;
                        }
                        let row = &v_plane[(iy - 1) * w..iy * w];
                        for kx in 0..3usize {
                            let ix = x + kx;
                            if ix == 0 || ix > w {
                                continue;
                            }
                            taps += w9[ky * 3 + kx] * row[ix - 1];
                        }
                    }
                    taps + bias
                };
                for x in 0..w {
                    out_plane[x] += border_pixel(0, x);
                    if h > 1 {
                        out_plane[(h - 1) * w + x] += border_pixel(h - 1, x);
                    }
                }
                for y in 1..h.saturating_sub(1) {
                    out_plane[y * w] += border_pixel(y, 0);
                    if w > 1 {
                        out_plane[y * w + (w - 1)] += border_pixel(y, w - 1);
                    }
                }
            };
            if out.len() >= 16_384 {
                out.par_chunks_mut(n).enumerate().for_each(add_pe_plane);
            } else {
                out.chunks_mut(n).enumerate().for_each(add_pe_plane);
            }
        })?;

        // Return `scores` to folo-core's CPU buffer cache.
        let _ = Tensor::from_vec(scores, [n * n]);
        Ok(Some(Tensor::from_vec(out, [b, c, h, w])?))
    }
}

impl Module for Attention {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("qkv", &self.qkv);
        f("proj", &self.proj);
        f("pe", &self.pe);
    }
}

impl Forward for Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = x.dims4()?;
        let qkv = self.qkv.forward(x)?;
        let pin = self.attend_and_pe(&qkv, b, h, w)?;
        self.proj.forward(&pin)
    }
}

impl Fuse for Attention {
    fn fuse_(&mut self) -> Result<()> {
        self.qkv.fuse_()?;
        self.proj.fuse_()?;
        self.pe.fuse_()
    }
}

/// docs/MODEL.md §2.7: `PSABlock(c)`, attention then a feed-forward, both residual. Tensor names: `attn.*`,
/// `ffn1.*`, `ffn2.*`.
pub struct PsaBlock {
    /// `Attention(c, heads = max(c / 64, 1))`.
    pub attn: Attention,
    /// `Conv(c→2c, 1, 1)` with SiLU.
    pub ffn1: ConvBnAct,
    /// `Conv(2c→c, 1, 1, act=none)`.
    pub ffn2: ConvBnAct,
}

impl PsaBlock {
    /// `PSABlock(c)`.
    pub fn new(c: usize, init: &mut Init) -> Result<Self> {
        let attn = Attention::new(c, (c / 64).max(1), init)?;
        let ffn1 = init.conv(c, 2 * c, 1, 1);
        let ffn2 = init.conv_with(2 * c, c, 1, 1, 1, Activation::Identity);
        Ok(PsaBlock { attn, ffn1, ffn2 })
    }
}

impl Module for PsaBlock {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("attn", &self.attn);
        f("ffn1", &self.ffn1);
        f("ffn2", &self.ffn2);
    }
}

impl Forward for PsaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.attn.forward(x)?)?;
        let y = self.ffn2.forward(&self.ffn1.forward(&x)?)?;
        Ok(x.add(&y)?)
    }
}

impl Fuse for PsaBlock {
    fn fuse_(&mut self) -> Result<()> {
        self.attn.fuse_()?;
        self.ffn1.fuse_()?;
        self.ffn2.fuse_()
    }
}

/// docs/MODEL.md §2.7: `C2PSA(c1→c1, n)`. Tensor names: `cv1.*`, `cv2.*`, `block{i}.*`.
pub struct C2Psa {
    /// `Conv(c1→2c, 1, 1)`, split into `[c, c]` with `c = c1 / 2`.
    pub cv1: ConvBnAct,
    /// `Conv(2c→c1, 1, 1)`.
    pub cv2: ConvBnAct,
    /// `n` PSA blocks applied in sequence to the second half.
    pub blocks: ModuleList<PsaBlock>,
    c: usize,
}

impl C2Psa {
    /// `C2PSA(c1→c1, n)`.
    pub fn new(c1: usize, n: usize, init: &mut Init) -> Result<Self> {
        let c = c1 / 2;
        let cv1 = init.conv(c1, 2 * c, 1, 1);
        let cv2 = init.conv(2 * c, c1, 1, 1);
        let blocks = (0..n).map(|_| PsaBlock::new(c, init)).collect::<Result<Vec<_>>>()?;
        Ok(C2Psa { cv1, cv2, blocks: ModuleList::with_prefix("block", blocks), c })
    }

    /// Heads of each attention block, `max(c1 / 2 / 64, 1)`.
    pub fn heads(&self) -> usize {
        self.blocks.items().first().map_or(0, |b| b.attn.heads)
    }
}

impl Module for C2Psa {
    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        f("cv1", &self.cv1);
        f("cv2", &self.cv2);
        self.blocks.visit_children(f);
    }
}

impl Forward for C2Psa {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let ab = y.split_sizes(&[self.c, self.c], 1)?;
        let b = self.blocks.forward(&ab[1])?;
        self.cv2.forward_cat(&[&ab[0], &b])
    }
}

impl Fuse for C2Psa {
    fn fuse_(&mut self) -> Result<()> {
        self.cv1.fuse_()?;
        self.cv2.fuse_()?;
        self.blocks.fuse_()
    }
}
