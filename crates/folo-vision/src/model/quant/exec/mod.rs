//! The int8 executor (docs/MODEL.md §9.6): the quantised v1/v2 graphs ([`walk`]) on folo-core's int8 kernels
//! ([`folo_core::cpu::q8`]).
//!
//! Activations live in reusable C4 `u32` scratch buffers ([`super::pool::ScratchPool`]) checked out once per
//! forward pass from [`Int8Plan`], eliminating per-layer `Arc` heap allocations and small-buffer `malloc`/`free`
//! churn. Per-channel boundary grids (`version: 2`, or lazily cached uniform grids for `version: 1`) are stored in
//! a single FNV-1a table with precomputed half-split hashes and per-part max scales; each conv's packed weights
//! (and direct `u8` stem) are kept with its weight tensor ([`Tensor::derived_f32`]) until written.

mod act;
mod plan;
mod v2;

pub use act::QAct;
pub use plan::Int8Plan;
pub use v2::{features_v2, features_v2_rgb0_u8};

use std::cell::{Cell, OnceCell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut, QStem};
use folo_nn::{ConvBnAct, Error, Result};
use rayon::prelude::*;

use self::act::{QActF32, QActRepr, QActU8};
use self::plan::GridOwner;
use super::cache::{ConvLiveness, PartKeyRef, StemLiveness, load_qconv_view, load_qstem};
use super::calib::conv_f32;
use super::pool::{ScratchPool, SlotHandle};
use super::walk::{self, Exec};
use super::{ConvView, VERSION_V2, int8_able, static_int8_able, static_int8_able_rep, stem_int8_able};
use crate::model::blocks::Attention;
use crate::model::v2::{FullPadGate, HyperContext, StemV2, WsQARepConv3x3};
use crate::model::{FoloModel, attn_simd};

/// The int8 executor of one forward pass.
struct Int8<'a> {
    plan: &'a Int8Plan,
    /// Optional pre-packed `[n, h, w]` `[R, G, B, 0]` input from `Letterbox::apply_rgb0_u8`.
    rgb0_input: Option<(&'a [u32], [usize; 3])>,
    /// Per-forward reusable C4 scratch buffer pool and per-step packed-weight cache.
    pool: Rc<RefCell<ScratchPool>>,
    conv_step: Cell<usize>,
    boundary_step: Cell<usize>,
}

impl<'a> Int8<'a> {
    /// Cached per-step boundary lookup: hits `plan.boundary_list[slot]` in O(1) without FNV hashing on steady-state
    /// forwards, falling back to `plan.boundary_for_slot` on miss.
    fn boundary_for(&self, name: &str, c: usize) -> Result<(GridOwner<'a>, bool)> {
        let step = self.boundary_step.get();
        self.boundary_step.set(step + 1);
        if let Some(slot) = self.pool.borrow().boundary_slot(step)
            && let Some(hit) = self.plan.boundary_at_slot(slot, name, c)
        {
            return Ok(hit);
        }
        let (bg, is_f32, slot) = self.plan.boundary_for_slot(name, c)?;
        if let Some(slot) = slot {
            self.pool.borrow_mut().store_boundary_slot(step, slot);
        }
        Ok((bg, is_f32))
    }
    /// `conv`'s packed weights for inputs `parts` (cached on `conv.conv.weight`).
    fn qconv_view(
        &self,
        conv: ConvView<'_>,
        parts: &[&QActU8<'a>],
    ) -> Result<(Arc<QConv>, Option<ConvLiveness>)> {
        fn to_ref<'b, 'a>(a: &'b QActU8<'a>) -> PartKeyRef<'b> {
            PartKeyRef {
                grids_arc: &a.bg.as_ref().grids,
                grid_offset: a.grid_offset,
                channels: a.dims[1],
                grids_hash: a.grids_hash,
                max_scale: a.max_scale,
            }
        }
        let allow_decouple = self.plan.cal.version >= VERSION_V2;
        if parts.len() <= 8 {
            let first = to_ref(parts[0]);
            let mut arr = [first; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                arr[i] = to_ref(a);
            }
            load_qconv_view(conv, &arr[..parts.len()], allow_decouple)
        } else {
            let pk: Vec<PartKeyRef<'_>> = parts.iter().map(|a| to_ref(a)).collect();
            load_qconv_view(conv, &pk, allow_decouple)
        }
    }

    #[cfg(test)]
    fn qconv(&self, conv: &ConvBnAct, parts: &[&QActU8<'a>]) -> Result<Arc<QConv>> {
        Ok(self.qconv_view(ConvView::from_cba(conv), parts)?.0)
    }

    /// The stem's packed `u8` weights (cached on `conv.conv.weight`).
    fn qstem(&self, conv: &ConvBnAct) -> Result<(Arc<QStem>, Option<StemLiveness>)> {
        load_qstem(conv)
    }

    /// Run `conv_f32` over `parts` without heap `Vec` allocations for `parts.len() <= 8`.
    fn run_conv_f32_parts(conv: &ConvBnAct, parts: &[&QAct<'a>], res: Option<&QAct<'a>>) -> Result<Tensor> {
        let r = res.map(QAct::to_f32).transpose()?;
        if parts.len() <= 8 {
            let mut ts: [Option<Tensor>; 8] = [const { None }; 8];
            for (i, a) in parts.iter().enumerate() {
                ts[i] = Some(a.to_f32()?);
            }
            let first = ts[0].as_ref().expect("non-empty parts");
            let mut refs = [first; 8];
            for i in 1..parts.len() {
                refs[i] = ts[i].as_ref().expect("initialized part");
            }
            conv_f32(conv, &refs[..parts.len()], r.as_ref())
        } else {
            let ts = parts.iter().map(|a| a.to_f32()).collect::<Result<Vec<_>>>()?;
            let refs: Vec<&Tensor> = ts.iter().collect();
            conv_f32(conv, &refs, r.as_ref())
        }
    }

    fn max_pool_u8(&mut self, u: &QActU8<'a>, k: usize) -> QAct<'a> {
        let [n, c, h, w] = u.dims;
        let mut out = self.pool.borrow_mut().take_buf(n, c, h * w);
        {
            let pool = self.pool.borrow();
            q8::max_pool(u.view(&pool), n, h, w, k, &mut out);
        }
        QAct {
            repr: QActRepr::U8(QActU8 {
                buf: SlotHandle::new(&self.pool, out),
                offset: 0,
                img_stride: c / 4 * h * w,
                dims: u.dims,
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        }
    }

    /// Fused 3-stage `SPPF` C4 max pooling (`y0 -> y1 -> y2 -> y3`) running all 3 stages per C4 group in L1
    /// cache across Rayon workers, keeping worker threads warm for `sppf.cv2` and staying bit-exact with 3×
    /// `q8::max_pool`.
    fn sppf_pools_u8(&mut self, u: &QActU8<'a>, k: usize) -> [QAct<'a>; 3] {
        let [n, c, h, w] = u.dims;
        let hw = h * w;
        let groups = c / 4;
        let total = n * groups * hw;
        let r = k / 2;
        let (mut out1, mut out2, mut out3) = {
            let mut pool = self.pool.borrow_mut();
            (pool.take_buf(n, c, hw), pool.take_buf(n, c, hw), pool.take_buf(n, c, hw))
        };
        if total > 0 {
            let pool = self.pool.borrow();
            let src_data = pool.slice(u.buf.slot());
            let u_offset = u.offset;
            let u_img_stride = u.img_stride;
            type SppfOutItem<'s> = (usize, ((&'s mut [u32], &'s mut [u32]), &'s mut [u32]));
            let run_group = |tmp: &mut Vec<u32>, (ig, ((o1, o2), o3)): SppfOutItem<'_>| {
                if tmp.len() < hw {
                    tmp.resize(hw, 0);
                }
                let (img, g) = (ig / groups, ig % groups);
                let src = &src_data[u_offset + img * u_img_stride + g * hw..][..hw];
                attn_simd::max_pool_c4_plane(src, h, w, r, &mut tmp[..hw], o1);
                attn_simd::max_pool_c4_plane(o1, h, w, r, &mut tmp[..hw], o2);
                attn_simd::max_pool_c4_plane(o2, h, w, r, &mut tmp[..hw], o3);
            };
            if n * groups >= 8 && hw > 0 {
                if hw <= 1024 {
                    out1[..total]
                        .par_chunks_exact_mut(hw)
                        .zip(out2[..total].par_chunks_exact_mut(hw))
                        .zip(out3[..total].par_chunks_exact_mut(hw))
                        .enumerate()
                        .for_each(|(ig, ((o1, o2), o3))| {
                            let mut stack_tmp = [0u32; 1024];
                            let tmp = &mut stack_tmp[..hw];
                            let (img, g) = (ig / groups, ig % groups);
                            let src = &src_data[u_offset + img * u_img_stride + g * hw..][..hw];
                            attn_simd::max_pool_c4_plane(src, h, w, r, tmp, o1);
                            attn_simd::max_pool_c4_plane(o1, h, w, r, tmp, o2);
                            attn_simd::max_pool_c4_plane(o2, h, w, r, tmp, o3);
                        });
                } else {
                    out1[..total]
                        .par_chunks_exact_mut(hw)
                        .zip(out2[..total].par_chunks_exact_mut(hw))
                        .zip(out3[..total].par_chunks_exact_mut(hw))
                        .enumerate()
                        .for_each_init(|| Vec::with_capacity(hw), run_group);
                }
            } else {
                let mut tmp = vec![0u32; hw];
                out1[..total]
                    .chunks_exact_mut(hw)
                    .zip(out2[..total].chunks_exact_mut(hw))
                    .zip(out3[..total].chunks_exact_mut(hw))
                    .enumerate()
                    .for_each(|item| run_group(&mut tmp, item));
            }
        }
        let wrap = |buf: Vec<u32>| QAct {
            repr: QActRepr::U8(QActU8 {
                buf: SlotHandle::new(&self.pool, buf),
                offset: 0,
                img_stride: groups * hw,
                dims: u.dims,
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        };
        [wrap(out1), wrap(out2), wrap(out3)]
    }

    /// Execute the int8 conv into `out`, borrowing all input C4 views and the cached [`QConv`] from `self.pool`
    /// simultaneously with zero mutex locks or `Arc` refcount traffic on steady-state cache hit.
    fn run_view(
        &self,
        name: &str,
        conv: ConvView<'_>,
        parts: &[&QActU8<'a>],
        res: Option<&QActU8<'a>>,
        out: QOut<'_>,
    ) -> Result<[usize; 4]> {
        let [n, _, h, w] = parts[0].dims;
        if parts.iter().any(|a| a.dims[0] != n || a.dims[2] != h || a.dims[3] != w) {
            return Err(Error::Invalid(format!("int8 {name}: parts of different shapes")));
        }
        let allow_decouple = self.plan.cal.version >= VERSION_V2;
        let step = self.conv_step.get();
        self.conv_step.set(step + 1);
        let ensure_cached = |pkeys: &[(&Arc<[Grid]>, usize, usize)]| -> Result<Option<Arc<QConv>>> {
            if self.pool.borrow().conv_hit(step, conv, pkeys, allow_decouple) {
                return Ok(None);
            }
            let (qc, liveness) = self.qconv_view(conv, parts)?;
            if let Some(live) = liveness {
                self.pool.borrow_mut().store_conv(step, conv, pkeys, allow_decouple, live, qc);
                Ok(None)
            } else {
                Ok(Some(qc))
            }
        };
        let uncached = if parts.len() <= 8 {
            let p0 = (&parts[0].bg.as_ref().grids, parts[0].grid_offset, parts[0].dims[1]);
            let mut pk = [p0; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                pk[i] = (&a.bg.as_ref().grids, a.grid_offset, a.dims[1]);
            }
            ensure_cached(&pk[..parts.len()])?
        } else {
            let pk: Vec<_> = parts.iter().map(|a| (&a.bg.as_ref().grids, a.grid_offset, a.dims[1])).collect();
            ensure_cached(&pk)?
        };
        let pool = self.pool.borrow();
        let qc: &QConv = match &uncached {
            Some(arc) => arc,
            None => pool.conv_at(step),
        };
        let (oh, ow) = qc.out_hw(h, w);
        let res = res.map(|r| (r.view(&pool), r.grids()));
        if parts.len() <= 8 {
            let mut v_arr = [parts[0].view(&pool); 8];
            let mut z_arr: [&[u8]; 8] = [&[]; 8];
            for (i, a) in parts.iter().enumerate() {
                v_arr[i] = a.view(&pool);
                z_arr[i] = a.zeros();
            }
            q8::conv_per_channel(&v_arr[..parts.len()], &z_arr[..parts.len()], n, h, w, qc, res, out);
        } else {
            let views: Vec<C4<'_>> = parts.iter().map(|a| a.view(&pool)).collect();
            let zeros: Vec<&[u8]> = parts.iter().map(|a| a.zeros()).collect();
            q8::conv_per_channel(&views, &zeros, n, h, w, qc, res, out);
        }
        Ok([n, qc.out_channels(), oh, ow])
    }

    pub(super) fn run_parts_into(
        &self,
        name: &str,
        conv: ConvView<'_>,
        parts: &[&QAct<'a>],
        res: Option<&QAct<'a>>,
        out: QOut<'_>,
    ) -> Result<[usize; 4]> {
        let u8_res = res.map(|r| r.as_u8(&self.pool)).transpose()?;
        if parts.len() <= 8 {
            let first = parts[0].as_u8(&self.pool)?;
            let mut arr = [first; 8];
            for (i, a) in parts.iter().enumerate().skip(1) {
                arr[i] = a.as_u8(&self.pool)?;
            }
            self.run_view(name, conv, &arr[..parts.len()], u8_res, out)
        } else {
            let u8_parts: Vec<&QActU8<'a>> =
                parts.iter().map(|a| a.as_u8(&self.pool)).collect::<Result<_>>()?;
            self.run_view(name, conv, &u8_parts, u8_res, out)
        }
    }

    fn upsample2_u8(&mut self, u: &QActU8<'a>) -> QAct<'a> {
        let [n, c, h, w] = u.dims;
        let mut out = self.pool.borrow_mut().take_buf(n, c, 4 * h * w);
        {
            let pool = self.pool.borrow();
            q8::upsample2(u.view(&pool), n, h, w, &mut out);
        }
        QAct {
            repr: QActRepr::U8(QActU8 {
                buf: SlotHandle::new(&self.pool, out),
                offset: 0,
                img_stride: c / 4 * 4 * h * w,
                dims: [n, c, 2 * h, 2 * w],
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        }
    }

    pub(super) fn rgb0_to_f32_tensor(rgb0: &[u32], [n, h, w]: [usize; 3]) -> Result<Tensor> {
        let hw = h * w;
        let mut planar = folo_core::cpu::take_buffer(n * 3 * hw);
        const PIXEL_SCALE: f32 = 1.0 / 255.0;
        for img in 0..n {
            let src = &rgb0[img * hw..(img + 1) * hw];
            let base = img * 3 * hw;
            for (p, &word) in src.iter().enumerate() {
                let [r, g, b, _] = word.to_le_bytes();
                planar[base + p] = f32::from(r) * PIXEL_SCALE;
                planar[base + hw + p] = f32::from(g) * PIXEL_SCALE;
                planar[base + 2 * hw + p] = f32::from(b) * PIXEL_SCALE;
            }
        }
        Ok(Tensor::from_vec(planar, vec![n, 3, h, w])?)
    }
}

impl<'a> Exec for Int8<'a> {
    type A = QAct<'a>;

    fn wants_f32(&self, name: &str, conv: &ConvBnAct) -> bool {
        self.plan.is_f32(name) || !static_int8_able(conv)
    }

    fn wants_f32_rep(&self, name: &str, rep: &WsQARepConv3x3) -> bool {
        self.plan.is_f32(name) || !static_int8_able_rep(rep)
    }

    fn stem(&mut self, name: &str, conv: &ConvBnAct, x: &Tensor) -> Result<QAct<'a>> {
        use folo_nn::Forward;
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.boundary_for(name, o)?;
        let (n, c, h, w) = match self.rgb0_input {
            Some((_, [n, h, w])) => (n, 3, h, w),
            None => x.dims4()?,
        };
        if self.plan.cal.u8_stem
            && !is_f32
            && let Some((oh, ow)) = stem_int8_able(conv, [n, c, h, w])
        {
            let uncached_stem = if self.pool.borrow().stem_hit(conv) {
                None
            } else {
                let (qs, liveness) = self.qstem(conv)?;
                if let Some(live) = liveness {
                    self.pool.borrow_mut().store_stem(conv, live, qs);
                    None
                } else {
                    Some(qs)
                }
            };
            let mut out = self.pool.borrow_mut().take_buf(n, o, oh * ow);
            if let Some((rgb0, _)) = self.rgb0_input {
                let pool = self.pool.borrow();
                let qs: &QStem = match &uncached_stem {
                    Some(arc) => arc,
                    None => pool.stem(),
                };
                q8::stem_u8(
                    rgb0,
                    n,
                    h,
                    w,
                    qs,
                    QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
                );
            } else {
                let mut rgb0 = self.pool.borrow_mut().take_buf(n, 4, h * w);
                let res = x.with_f32_slice(|v| q8::pack_f32_to_rgb0_u8(v, n, h, w, &mut rgb0));
                if let Err(e) = res {
                    let mut p = self.pool.borrow_mut();
                    p.return_buf(rgb0);
                    p.return_buf(out);
                    return Err(e.into());
                }
                {
                    let pool = self.pool.borrow();
                    let qs: &QStem = match &uncached_stem {
                        Some(arc) => arc,
                        None => pool.stem(),
                    };
                    q8::stem_u8(
                        &rgb0,
                        n,
                        h,
                        w,
                        qs,
                        QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
                    );
                }
                self.pool.borrow_mut().return_buf(rgb0);
            }
            return Ok(QAct::new(&self.pool, out, [n, o, oh, ow], bg));
        }
        let y = if let Some((rgb0, dims)) = self.rgb0_input {
            let x_f32 = Self::rgb0_to_f32_tensor(rgb0, dims)?;
            conv.forward(&x_f32)?
        } else {
            conv.forward(x)?
        };
        QAct::from_f32_lazy(y, bg)
    }

    fn stem_v2(&mut self, name: &str, stem: &StemV2, x: &Tensor) -> Result<QAct<'a>> {
        self.stem_v2_impl(name, stem, x)
    }

    fn conv(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&QAct<'a>],
        res: Option<&QAct<'a>>,
    ) -> Result<QAct<'a>> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                parts
                    .iter()
                    .all(|a| a.dims()[1].is_multiple_of(4))
                    .then(|| int8_able(conv, [n, c, h, w], res.map(QAct::dims)))
                    .flatten()
            })
            .flatten();
        let Some((oh, ow)) = able else {
            let y = Self::run_conv_f32_parts(conv, parts, res)?;
            return QAct::from_f32_lazy(y, bg);
        };
        let mut out = self.pool.borrow_mut().take_buf(n, o, oh * ow);
        let dims = match self.run_parts_into(
            name,
            ConvView::from_cba(conv),
            parts,
            res,
            QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
        ) {
            Ok(d) => d,
            Err(e) => {
                self.pool.borrow_mut().return_buf(out);
                return Err(e);
            }
        };
        Ok(QAct::new(&self.pool, out, dims, bg))
    }

    fn conv_f32_out(
        &mut self,
        name: &str,
        conv: &ConvBnAct,
        parts: &[&QAct<'a>],
        res: Option<&QAct<'a>>,
    ) -> Result<QAct<'a>> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                parts
                    .iter()
                    .all(|a| a.dims()[1].is_multiple_of(4))
                    .then(|| int8_able(conv, [n, c, h, w], res.map(QAct::dims)))
                    .flatten()
            })
            .flatten();
        let Some((oh, ow)) = able else {
            let y = Self::run_conv_f32_parts(conv, parts, res)?;
            return QAct::from_f32_lazy(y, bg);
        };
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims =
            self.run_parts_into(name, ConvView::from_cba(conv), parts, res, QOut::F32 { data: &mut out })?;
        let y = Tensor::from_vec(out, dims.to_vec())?;
        QAct::from_f32_lazy(y, bg)
    }

    fn rep_conv(
        &mut self,
        name: &str,
        rep: &WsQARepConv3x3,
        x: &QAct<'a>,
        out_f32: bool,
    ) -> Result<QAct<'a>> {
        self.rep_conv_impl(name, rep, x, out_f32)
    }

    fn conv_f32(&mut self, name: &str, conv: &ConvBnAct, x: &QAct<'a>) -> Result<Tensor> {
        let xd = x.dims();
        let (Some((oh, ow)), false) = (int8_able(conv, xd, None), self.plan.is_f32(name)) else {
            return conv_f32(conv, &[&x.to_f32()?], None);
        };
        let u8_x = x.as_u8(&self.pool)?;
        let (n, o) = (xd[0], conv.conv.config().out_channels);
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims =
            self.run_view(name, ConvView::from_cba(conv), &[u8_x], None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn conv_out_f32(&mut self, name: &str, conv: &ConvBnAct, parts: &[&QAct<'a>]) -> Result<Tensor> {
        let [n, _, h, w] = parts[0].dims();
        let c = parts.iter().map(|a| a.dims()[1]).sum();
        let o = conv.conv.config().out_channels;
        let (_, is_f32) = self.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                parts
                    .iter()
                    .all(|a| a.dims()[1].is_multiple_of(4))
                    .then(|| int8_able(conv, [n, c, h, w], None))
                    .flatten()
            })
            .flatten();
        let Some((oh, ow)) = able else {
            return Self::run_conv_f32_parts(conv, parts, None);
        };
        let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
        let dims =
            self.run_parts_into(name, ConvView::from_cba(conv), parts, None, QOut::F32 { data: &mut out })?;
        Ok(Tensor::from_vec(out, dims.to_vec())?)
    }

    fn attend_and_pe_q(
        &mut self,
        a: &Attention,
        qkv: &Tensor,
        res_dims: [usize; 4],
        pin_name: &str,
        out_f32: bool,
    ) -> Result<QAct<'a>> {
        let [b, _, h, w] = res_dims;
        let c = a.heads * a.head_dim;
        let proj_runs_int8 =
            !out_f32 && c.is_multiple_of(4) && int8_able(&a.proj, [b, c, h, w], Some(res_dims)).is_some();
        if proj_runs_int8 {
            let (bg, _) = self.boundary_for(pin_name, c)?;
            let mut out = self.pool.borrow_mut().take_buf(b, c, h * w);
            match a.attend_and_pe_into_c4(qkv, b, h, w, &bg.as_ref().grids, &mut out) {
                Ok(true) => return Ok(QAct::new(&self.pool, out, [b, c, h, w], bg)),
                Ok(false) => {
                    self.pool.borrow_mut().return_buf(out);
                }
                Err(e) => {
                    self.pool.borrow_mut().return_buf(out);
                    return Err(e);
                }
            }
        }
        let attn_out = self.attend_and_pe(a, qkv, b, h, w)?;
        self.quantize(pin_name, &attn_out)
    }

    fn hyper_attend_and_scene_q(
        &mut self,
        h_ctx: &HyperContext,
        av: &Tensor,
        res_dims: [usize; 4],
        pin_name: &str,
        out_f32: bool,
    ) -> Result<(QAct<'a>, Tensor)> {
        self.hyper_attend_and_scene_q_impl(h_ctx, av, res_dims, pin_name, out_f32)
    }

    fn dequantize_fullpad_gate(
        &mut self,
        x: &QAct<'a>,
        gate: &FullPadGate,
        g_scene: &Tensor,
    ) -> Result<Tensor> {
        self.dequantize_fullpad_gate_impl(x, gate, g_scene)
    }

    fn materialize_f32(&mut self, x: &QAct<'a>) -> Result<Option<QAct<'a>>> {
        match &x.repr {
            QActRepr::U8(u) => {
                let tensor = u.to_f32()?;
                let u8_cache = OnceCell::new();
                let _ = u8_cache.set(u.clone());
                Ok(Some(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor,
                        dims: u.dims,
                        bg: u.bg.clone(),
                        grid_offset: u.grid_offset,
                        grids_hash: u.grids_hash,
                        max_scale: u.max_scale,
                        u8_cache,
                    }),
                }))
            }
            QActRepr::F32(_) => Ok(None),
        }
    }

    fn quantize(&mut self, name: &str, x: &Tensor) -> Result<QAct<'a>> {
        let c = x.dims4()?.1;
        let (bg, _) = self.boundary_for(name, c)?;
        QAct::from_f32_lazy(x.clone(), bg)
    }

    fn dequantize(&mut self, x: &QAct<'a>) -> Result<Tensor> {
        x.to_f32()
    }

    fn ensure_u8(&mut self, x: &QAct<'a>) -> Result<()> {
        if let QActRepr::F32(f) = &x.repr
            && f.dims[1].is_multiple_of(4)
        {
            let _ = x.as_u8(&self.pool)?;
        }
        Ok(())
    }

    fn narrow(&mut self, x: &QAct<'a>, start: usize, len: usize) -> Result<QAct<'a>> {
        match &x.repr {
            QActRepr::U8(u) => Ok(QAct { repr: QActRepr::U8(u.narrow_slice(start, len)?) }),
            QActRepr::F32(f) => {
                let [n, c, h, w] = f.dims;
                if start + len > c {
                    return Err(Error::Invalid(format!(
                        "int8: channels {start}..{} out of bounds for {c}",
                        start + len
                    )));
                }
                let grid_offset = f.grid_offset + start;
                let (grids_hash, max_scale) = f.bg.as_ref().slice_meta(grid_offset, len);
                let u8_cache = OnceCell::new();
                if let Some(u) = f.u8_cache.get()
                    && start.is_multiple_of(4)
                    && len.is_multiple_of(4)
                {
                    let _ = u8_cache.set(u.narrow_slice(start, len)?);
                }
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor: f.tensor.narrow(1, start, len)?,
                        dims: [n, len, h, w],
                        bg: f.bg.clone(),
                        grid_offset,
                        grids_hash,
                        max_scale,
                        u8_cache,
                    }),
                })
            }
        }
    }

    fn max_pool(&mut self, x: &QAct<'a>, k: usize) -> Result<QAct<'a>> {
        match &x.repr {
            QActRepr::U8(u) => Ok(self.max_pool_u8(u, k)),
            QActRepr::F32(f) => {
                if let Some(u) = f.u8_cache.get() {
                    return Ok(self.max_pool_u8(u, k));
                }
                let tensor = f.tensor.max_pool2d(k, 1, k / 2)?;
                let (n, c, oh, ow) = tensor.dims4()?;
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor,
                        dims: [n, c, oh, ow],
                        bg: f.bg.clone(),
                        grid_offset: f.grid_offset,
                        grids_hash: f.grids_hash,
                        max_scale: f.max_scale,
                        u8_cache: OnceCell::new(),
                    }),
                })
            }
        }
    }

    fn sppf_pools(&mut self, y0: &QAct<'a>, k: usize) -> Result<[QAct<'a>; 3]> {
        match &y0.repr {
            QActRepr::U8(u) => {
                let u_clone = u.clone();
                Ok(self.sppf_pools_u8(&u_clone, k))
            }
            QActRepr::F32(f) => {
                if let Some(u) = f.u8_cache.get() {
                    let u_clone = u.clone();
                    return Ok(self.sppf_pools_u8(&u_clone, k));
                }
                let y1 = self.max_pool(y0, k)?;
                let y2 = self.max_pool(&y1, k)?;
                let y3 = self.max_pool(&y2, k)?;
                Ok([y1, y2, y3])
            }
        }
    }

    fn upsample2(&mut self, x: &QAct<'a>) -> Result<QAct<'a>> {
        self.upsample2_for(x, true)
    }

    fn upsample2_for(&mut self, x: &QAct<'a>, consumer_is_f32: bool) -> Result<QAct<'a>> {
        match &x.repr {
            QActRepr::U8(u) => Ok(self.upsample2_u8(u)),
            QActRepr::F32(f) => {
                if !consumer_is_f32 && f.dims[1].is_multiple_of(4) {
                    let u = x.as_u8(&self.pool)?.clone();
                    return Ok(self.upsample2_u8(&u));
                }
                let [n, c, h, w] = f.dims;
                Ok(QAct {
                    repr: QActRepr::F32(QActF32 {
                        tensor: f.tensor.upsample_nearest2d(2)?,
                        dims: [n, c, 2 * h, 2 * w],
                        bg: f.bg.clone(),
                        grid_offset: f.grid_offset,
                        grids_hash: f.grids_hash,
                        max_scale: f.max_scale,
                        u8_cache: OnceCell::new(),
                    }),
                })
            }
        }
    }

    fn space_to_depth2_parts(&mut self, x: &QAct<'a>, consumer_is_f32: bool) -> Result<[QAct<'a>; 4]> {
        self.space_to_depth2_parts_impl(x, consumer_is_f32)
    }

    fn dims(&self, x: &QAct<'a>) -> [usize; 4] {
        x.dims()
    }
}

fn run_walk(
    model: &FoloModel,
    plan: &Int8Plan,
    rgb0_input: Option<(&[u32], [usize; 3])>,
    x: &Tensor,
) -> Result<[Tensor; 3]> {
    let pool = plan.checkout_scratch();
    let prev_stats = pool.stats();
    let pool_rc = Rc::new(RefCell::new(pool));
    let mut exec = Int8 {
        plan,
        rgb0_input,
        pool: Rc::clone(&pool_rc),
        conv_step: Cell::new(0),
        boundary_step: Cell::new(0),
    };
    let res = walk::features(model, &mut exec, x);
    drop(exec);
    if let Ok(cell) = Rc::try_unwrap(pool_rc) {
        plan.checkin_scratch(cell.into_inner(), prev_stats);
    }
    res
}

/// Layers 0–22 of `model` on `x` with the int8 plan: the head's three inputs (`f32`). Call it on the CPU, inside
/// folo-core's pool, under `no_grad`, on a fused eval-mode `f32` model.
pub fn features(model: &FoloModel, plan: &Int8Plan, x: &Tensor) -> Result<[Tensor; 3]> {
    run_walk(model, plan, None, x)
}

/// [`features`] reading a pre-packed `[n, h, w]` `u32` `[R, G, B, 0]` canvas (from `Letterbox::apply_rgb0_u8`)
/// straight into the direct `u8` AVX-VNNI stem with zero `f32` image materialisation.
pub fn features_rgb0_u8(
    model: &FoloModel,
    plan: &Int8Plan,
    rgb0: &[u32],
    n: usize,
    h: usize,
    w: usize,
) -> Result<[Tensor; 3]> {
    let dummy = Tensor::zeros([0]);
    run_walk(model, plan, Some((rgb0, [n, h, w])), &dummy)
}

#[cfg(test)]
mod tests;
