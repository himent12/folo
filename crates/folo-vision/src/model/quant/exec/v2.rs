//! FOLO v2-Iso int8 executor operations: direct `u8` `SPD(2×2)` stem packing, fused `WsQARepConv3x3`
//! dispatch, zero-copy C4 `space_to_depth_2x2` (`P2 → P3`), and `features_v2` / `features_v2_rgb0_u8`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::cell::{Cell, OnceCell, RefCell};
use std::rc::Rc;

use folo_core::Tensor;
use folo_core::cpu::q8::{self, C4, Grid, QOut};
use folo_nn::{Error, Forward, Result};
use rayon::prelude::*;

use super::{GridOwner, Int8, Int8Plan, QAct, QActF32, QActRepr, QActU8};
#[cfg(target_arch = "x86_64")]
use crate::model::attn_simd;
use crate::model::quant::cache::BoundaryGrids;
use crate::model::quant::walk::{self, Exec};
use crate::model::quant::{ConvView, int8_able, int8_able_view, spd_stem_int8_able};
use crate::model::v2::{FoloV2Model, FullPadGate, HyperContext, StemV2, WsQARepConv3x3};

type Spd3Row<'a> = (usize, ((&'a mut [u32], &'a mut [u32]), &'a mut [u32]));
type Spd4Group<'a> = (usize, (((&'a mut [u32], &'a mut [u32]), &'a mut [u32]), &'a mut [u32]));

/// Convert a single `f32` pixel in `[0, 1]` to `u8`, bit-exact with `folo_core::cpu::q8::pack_f32_to_rgb0_u8`.
#[inline(always)]
fn f32_to_pixel_u8(x: f32) -> u8 {
    let v = x * 255.0;
    let v = if v > 0.0 { v } else { 0.0 };
    let v = if v < 255.0 { v } else { 255.0 };
    v.round_ties_even() as i32 as u8
}

/// Pack planar `[n, 3, h, w]` `f32` pixels in `[0, 1]` directly into `[n, 3, h/2, w/2]` C4 words for
/// `SPD(2×2)` (`12` channels = 3 C4 groups), avoiding an intermediate `[n, h, w]` `rgb0` buffer.
fn pack_f32_to_spd12_c4(x: &[f32], n: usize, h: usize, w: usize, out: &mut [u32]) {
    let (oh, ow) = (h / 2, w / 2);
    let (hw, ohw) = (h * w, oh * ow);
    let run_img = |(img, dst): (usize, &mut [u32])| {
        let base = img * 3 * hw;
        let r_plane = &x[base..base + hw];
        let g_plane = &x[base + hw..base + 2 * hw];
        let b_plane = &x[base + 2 * hw..base + 3 * hw];
        let (g0, rest) = dst.split_at_mut(ohw);
        let (g1, g2) = rest.split_at_mut(ohw);
        let run_rows = |(oy, ((o0, o1), o2)): Spd3Row<'_>| {
            let e = (2 * oy) * w;
            let o = (2 * oy + 1) * w;
            let (re, ro) = (&r_plane[e..e + w], &r_plane[o..o + w]);
            let (ge, go) = (&g_plane[e..e + w], &g_plane[o..o + w]);
            let (be, bo) = (&b_plane[e..e + w], &b_plane[o..o + w]);
            for ox in 0..ow {
                let x0 = 2 * ox;
                let x1 = x0 + 1;
                o0[ox] = u32::from_le_bytes([
                    f32_to_pixel_u8(re[x0]),
                    f32_to_pixel_u8(ge[x0]),
                    f32_to_pixel_u8(be[x0]),
                    f32_to_pixel_u8(re[x1]),
                ]);
                o1[ox] = u32::from_le_bytes([
                    f32_to_pixel_u8(ge[x1]),
                    f32_to_pixel_u8(be[x1]),
                    f32_to_pixel_u8(ro[x0]),
                    f32_to_pixel_u8(go[x0]),
                ]);
                o2[ox] = u32::from_le_bytes([
                    f32_to_pixel_u8(bo[x0]),
                    f32_to_pixel_u8(ro[x1]),
                    f32_to_pixel_u8(go[x1]),
                    f32_to_pixel_u8(bo[x1]),
                ]);
            }
        };
        if ohw >= 16_384 {
            g0.par_chunks_exact_mut(ow)
                .zip(g1.par_chunks_exact_mut(ow))
                .zip(g2.par_chunks_exact_mut(ow))
                .enumerate()
                .for_each(run_rows);
        } else {
            g0.chunks_exact_mut(ow)
                .zip(g1.chunks_exact_mut(ow))
                .zip(g2.chunks_exact_mut(ow))
                .enumerate()
                .for_each(run_rows);
        }
    };
    let total = n * 3 * ohw;
    if n > 1 && total >= 32_768 {
        out[..total].par_chunks_exact_mut(3 * ohw).enumerate().for_each(run_img);
    } else {
        out[..total].chunks_exact_mut(3 * ohw).enumerate().for_each(run_img);
    }
}

/// Deinterleave 16 adjacent `u32` words `[a0,b0,a1,b1,...,a7,b7]` into `([a0..a7], [b0..b7])` in 4 AVX2 ops.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn deinterleave_u32x16_avx2(lo: __m256i, hi: __m256i) -> (__m256i, __m256i) {
    let lo_f = _mm256_castsi256_ps(lo);
    let hi_f = _mm256_castsi256_ps(hi);
    let even_shuf = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(lo_f, hi_f));
    let odd_shuf = _mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(lo_f, hi_f));
    (_mm256_permute4x64_epi64::<0xD8>(even_shuf), _mm256_permute4x64_epi64::<0xD8>(odd_shuf))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pack_rgb0_to_spd12_row_avx2(
    r_even: &[u32],
    r_odd: &[u32],
    o0: &mut [u32],
    o1: &mut [u32],
    o2: &mut [u32],
    ow: usize,
) {
    let m24 = _mm256_set1_epi32(0x00FF_FFFFu32 as i32);
    let m16 = _mm256_set1_epi32(0x0000_FFFFu32 as i32);
    let m08 = _mm256_set1_epi32(0x0000_00FFu32 as i32);
    let mut ox = 0usize;
    // SAFETY: `2 * (ox + 8) <= 2 * ow <= r_even.len(), r_odd.len()` and `ox + 8 <= ow <= o{0,1,2}.len()`.
    unsafe {
        while ox + 8 <= ow {
            let e_ptr = r_even.as_ptr().add(2 * ox) as *const __m256i;
            let o_ptr = r_odd.as_ptr().add(2 * ox) as *const __m256i;
            let (p_tl, p_tr) =
                deinterleave_u32x16_avx2(_mm256_loadu_si256(e_ptr), _mm256_loadu_si256(e_ptr.add(1)));
            let (p_bl, p_br) =
                deinterleave_u32x16_avx2(_mm256_loadu_si256(o_ptr), _mm256_loadu_si256(o_ptr.add(1)));
            let v0 = _mm256_or_si256(_mm256_and_si256(p_tl, m24), _mm256_slli_epi32::<24>(p_tr));
            let v1 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi32::<8>(p_tr), m16),
                _mm256_slli_epi32::<16>(p_bl),
            );
            let v2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi32::<16>(p_bl), m08),
                _mm256_slli_epi32::<8>(p_br),
            );
            _mm256_storeu_si256(o0.as_mut_ptr().add(ox) as *mut __m256i, v0);
            _mm256_storeu_si256(o1.as_mut_ptr().add(ox) as *mut __m256i, v1);
            _mm256_storeu_si256(o2.as_mut_ptr().add(ox) as *mut __m256i, v2);
            ox += 8;
        }
    }
    for ox in ox..ow {
        let p_tl = r_even[2 * ox];
        let p_tr = r_even[2 * ox + 1];
        let p_bl = r_odd[2 * ox];
        let p_br = r_odd[2 * ox + 1];
        o0[ox] = (p_tl & 0x00FF_FFFF) | (p_tr << 24);
        o1[ox] = ((p_tr >> 8) & 0x0000_FFFF) | (p_bl << 16);
        o2[ox] = ((p_bl >> 16) & 0x0000_00FF) | (p_br << 8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn space_to_depth2_row_avx2(
    r_even: &[u32],
    r_odd: &[u32],
    t0: &mut [u32],
    t1: &mut [u32],
    t2: &mut [u32],
    t3: &mut [u32],
    ow: usize,
) {
    let mut ox = 0usize;
    // SAFETY: `2 * (ox + 8) <= 2 * ow <= r_even.len(), r_odd.len()` and `ox + 8 <= ow <= t{0,1,2,3}.len()`.
    unsafe {
        while ox + 8 <= ow {
            let e_ptr = r_even.as_ptr().add(2 * ox) as *const __m256i;
            let o_ptr = r_odd.as_ptr().add(2 * ox) as *const __m256i;
            let (e_tl, e_tr) =
                deinterleave_u32x16_avx2(_mm256_loadu_si256(e_ptr), _mm256_loadu_si256(e_ptr.add(1)));
            let (o_bl, o_br) =
                deinterleave_u32x16_avx2(_mm256_loadu_si256(o_ptr), _mm256_loadu_si256(o_ptr.add(1)));
            _mm256_storeu_si256(t0.as_mut_ptr().add(ox) as *mut __m256i, e_tl);
            _mm256_storeu_si256(t1.as_mut_ptr().add(ox) as *mut __m256i, e_tr);
            _mm256_storeu_si256(t2.as_mut_ptr().add(ox) as *mut __m256i, o_bl);
            _mm256_storeu_si256(t3.as_mut_ptr().add(ox) as *mut __m256i, o_br);
            ox += 8;
        }
    }
    for ox in ox..ow {
        t0[ox] = r_even[2 * ox];
        t1[ox] = r_even[2 * ox + 1];
        t2[ox] = r_odd[2 * ox];
        t3[ox] = r_odd[2 * ox + 1];
    }
}

/// Pack `[n, h, w]` `u32` `[R, G, B, 0]` pixels directly into `[n, 3, h/2, w/2]` C4 words representing
/// `space_to_depth_2x2` of the 3-channel RGB image (`12` channels = 3 C4 groups:
/// `[R_tl, G_tl, B_tl, R_tr]`, `[G_tr, B_tr, R_bl, G_bl]`, `[B_bl, R_br, G_br, B_br]`).
fn pack_rgb0_to_spd12_c4(rgb0: &[u32], n: usize, h: usize, w: usize, out: &mut [u32]) {
    let (oh, ow) = (h / 2, w / 2);
    let (hw, ohw) = (h * w, oh * ow);
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = attn_simd::has_avx2_fma();
    let run_img = |(img, dst): (usize, &mut [u32])| {
        let src = &rgb0[img * hw..(img + 1) * hw];
        let (g0, rest) = dst.split_at_mut(ohw);
        let (g1, g2) = rest.split_at_mut(ohw);
        let run_rows = |(oy, ((o0, o1), o2)): Spd3Row<'_>| {
            let r_even = &src[(2 * oy) * w..(2 * oy + 1) * w];
            let r_odd = &src[(2 * oy + 1) * w..(2 * oy + 2) * w];
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                // SAFETY: AVX2 detected above; row slices cover `2 * ow` and `ow` elements.
                unsafe { pack_rgb0_to_spd12_row_avx2(r_even, r_odd, o0, o1, o2, ow) };
                return;
            }
            for ox in 0..ow {
                let p_tl = r_even[2 * ox];
                let p_tr = r_even[2 * ox + 1];
                let p_bl = r_odd[2 * ox];
                let p_br = r_odd[2 * ox + 1];
                o0[ox] = (p_tl & 0x00FF_FFFF) | (p_tr << 24);
                o1[ox] = ((p_tr >> 8) & 0x0000_FFFF) | (p_bl << 16);
                o2[ox] = ((p_bl >> 16) & 0x0000_00FF) | (p_br << 8);
            }
        };
        if ohw >= 16_384 {
            g0.par_chunks_exact_mut(ow)
                .zip(g1.par_chunks_exact_mut(ow))
                .zip(g2.par_chunks_exact_mut(ow))
                .enumerate()
                .for_each(run_rows);
        } else {
            g0.chunks_exact_mut(ow)
                .zip(g1.chunks_exact_mut(ow))
                .zip(g2.chunks_exact_mut(ow))
                .enumerate()
                .for_each(run_rows);
        }
    };
    let total = n * 3 * ohw;
    if n > 1 && total >= 32_768 {
        out[..total].par_chunks_exact_mut(3 * ohw).enumerate().for_each(run_img);
    } else {
        out[..total].chunks_exact_mut(3 * ohw).enumerate().for_each(run_img);
    }
}

/// Lossless `2×2` space-to-depth on C4 `u8` codes: maps `[n, c, h, w]` into 4 contiguous channel slices
/// `[tl, tr, bl, br]` (`[n, c, h/2, w/2]` each) packed in one `ScratchPool` buffer `[n, 4*c, h/2, w/2]`.
fn space_to_depth2_c4(u: &QActU8<'_>, src_data: &[u32], out: &mut [u32]) {
    let [n, c, h, w] = u.dims;
    let (oh, ow) = (h / 2, w / 2);
    let (hw, ohw) = (h * w, oh * ow);
    let groups = c / 4;
    let part_words = groups * ohw;
    let img_out_words = 4 * part_words;
    let u_offset = u.offset;
    let u_img_stride = u.img_stride;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = attn_simd::has_avx2_fma();

    for img in 0..n {
        let dst_img = &mut out[img * img_out_words..(img + 1) * img_out_words];
        let (d0, rest) = dst_img.split_at_mut(part_words);
        let (d1, rest) = rest.split_at_mut(part_words);
        let (d2, d3) = rest.split_at_mut(part_words);
        let run_group = |(g, (((o0, o1), o2), o3)): Spd4Group<'_>| {
            let src = &src_data[u_offset + img * u_img_stride + g * hw..][..hw];
            for oy in 0..oh {
                let r_even = &src[(2 * oy) * w..(2 * oy + 1) * w];
                let r_odd = &src[(2 * oy + 1) * w..(2 * oy + 2) * w];
                let t0 = &mut o0[oy * ow..(oy + 1) * ow];
                let t1 = &mut o1[oy * ow..(oy + 1) * ow];
                let t2 = &mut o2[oy * ow..(oy + 1) * ow];
                let t3 = &mut o3[oy * ow..(oy + 1) * ow];
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    // SAFETY: AVX2 detected above; row slices cover `2 * ow` and `ow` elements.
                    unsafe { space_to_depth2_row_avx2(r_even, r_odd, t0, t1, t2, t3, ow) };
                    continue;
                }
                for ox in 0..ow {
                    t0[ox] = r_even[2 * ox];
                    t1[ox] = r_even[2 * ox + 1];
                    t2[ox] = r_odd[2 * ox];
                    t3[ox] = r_odd[2 * ox + 1];
                }
            }
        };
        if part_words >= 16_384 {
            d0.par_chunks_exact_mut(ohw)
                .zip(d1.par_chunks_exact_mut(ohw))
                .zip(d2.par_chunks_exact_mut(ohw))
                .zip(d3.par_chunks_exact_mut(ohw))
                .enumerate()
                .for_each(run_group);
        } else {
            d0.chunks_exact_mut(ohw)
                .zip(d1.chunks_exact_mut(ohw))
                .zip(d2.chunks_exact_mut(ohw))
                .zip(d3.chunks_exact_mut(ohw))
                .enumerate()
                .for_each(run_group);
        }
    }
}

impl<'a> Int8<'a> {
    pub(super) fn stem_v2_impl(&mut self, name: &str, stem: &StemV2, x: &Tensor) -> Result<QAct<'a>> {
        if !stem.spd {
            return self.stem(name, &stem.conv, x);
        }
        let conv = &stem.conv;
        let o = conv.conv.config().out_channels;
        let (bg, is_f32) = self.boundary_for(name, o)?;
        let (n, c, h, w) = match self.rgb0_input {
            Some((_, [n, h, w])) => (n, 3, h, w),
            None => x.dims4()?,
        };
        if self.plan.cal.u8_stem
            && !is_f32
            && let Some((oh, ow)) = spd_stem_int8_able(conv, [n, c, h, w])
        {
            let mut spd_buf = self.pool.borrow_mut().take_buf(n, 12, (h / 2) * (w / 2));
            if let Some((rgb0, _)) = self.rgb0_input {
                pack_rgb0_to_spd12_c4(rgb0, n, h, w, &mut spd_buf);
            } else {
                let res = x.with_f32_slice(|v| pack_f32_to_spd12_c4(v, n, h, w, &mut spd_buf));
                if let Err(e) = res {
                    self.pool.borrow_mut().return_buf(spd_buf);
                    return Err(e.into());
                }
            }
            let spd_act = QActU8::new(
                &self.pool,
                spd_buf,
                [n, 12, h / 2, w / 2],
                GridOwner::Borrowed(BoundaryGrids::pixel_12()),
            );
            let mut out = self.pool.borrow_mut().take_buf(n, o, oh * ow);
            let dims = match self.run_view(
                name,
                ConvView::from_cba(conv),
                &[&spd_act],
                None,
                QOut::U8PerChannel { data: &mut out, grids: &bg.as_ref().grids },
            ) {
                Ok(d) => d,
                Err(e) => {
                    self.pool.borrow_mut().return_buf(out);
                    return Err(e);
                }
            };
            drop(spd_act);
            return Ok(QAct::new(&self.pool, out, dims, bg));
        }
        let y = if let Some((rgb0, dims)) = self.rgb0_input {
            let x_f32 = Self::rgb0_to_f32_tensor(rgb0, dims)?;
            stem.forward(&x_f32)?
        } else {
            stem.forward(x)?
        };
        QAct::from_f32_lazy(y, bg)
    }

    pub(super) fn rep_conv_impl(
        &mut self,
        name: &str,
        rep: &WsQARepConv3x3,
        x: &QAct<'a>,
        out_f32: bool,
    ) -> Result<QAct<'a>> {
        let xd = x.dims();
        let [n, _, _, _] = xd;
        let o = rep.out_channels;
        let (bg, is_f32) = self.boundary_for(name, o)?;
        let able = (!is_f32)
            .then(|| {
                ConvView::from_rep(rep).and_then(|view| {
                    xd[1]
                        .is_multiple_of(4)
                        .then(|| int8_able_view(view, xd, None))
                        .flatten()
                        .map(|hw| (view, hw))
                })
            })
            .flatten();
        let Some((view, (oh, ow))) = able else {
            let xf = x.to_f32()?;
            let y = rep.forward(&xf)?;
            return QAct::from_f32_lazy(y, bg);
        };
        if out_f32 {
            let mut out = folo_core::cpu::take_buffer(n * o * oh * ow);
            let dims = self.run_parts_into(name, view, &[x], None, QOut::F32 { data: &mut out })?;
            let y = Tensor::from_vec(out, dims.to_vec())?;
            return QAct::from_f32_lazy(y, bg);
        }
        let mut out = self.pool.borrow_mut().take_buf(n, o, oh * ow);
        let dims = match self.run_parts_into(
            name,
            view,
            &[x],
            None,
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

    fn space_to_depth2_u8(&mut self, u: &QActU8<'a>) -> Result<[QAct<'a>; 4]> {
        let [n, c, h, w] = u.dims;
        if h == 0 || w == 0 || !h.is_multiple_of(2) || !w.is_multiple_of(2) {
            return Err(Error::Invalid(format!(
                "int8 space_to_depth2: spatial dims ({h}, {w}) must be positive and even"
            )));
        }
        let (oh, ow) = (h / 2, w / 2);
        let ohw = oh * ow;
        let part_words = (c / 4) * ohw;
        let img_stride = 4 * part_words;
        let mut out = self.pool.borrow_mut().take_buf(n, 4 * c, ohw);
        {
            let pool = self.pool.borrow();
            space_to_depth2_c4(u, pool.slice(u.buf.slot()), &mut out);
        }
        let handle = super::SlotHandle::new(&self.pool, out);
        let mk = |phase: usize| QAct {
            repr: QActRepr::U8(QActU8 {
                buf: handle.clone(),
                offset: phase * part_words,
                img_stride,
                dims: [n, c, oh, ow],
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        };
        let p0 = mk(0);
        let p1 = mk(1);
        let p2 = mk(2);
        let p3 = QAct {
            repr: QActRepr::U8(QActU8 {
                buf: handle,
                offset: 3 * part_words,
                img_stride,
                dims: [n, c, oh, ow],
                bg: u.bg.clone(),
                grid_offset: u.grid_offset,
                grids_hash: u.grids_hash,
                max_scale: u.max_scale,
            }),
        };
        Ok([p0, p1, p2, p3])
    }

    pub(super) fn space_to_depth2_parts_impl(
        &mut self,
        x: &QAct<'a>,
        consumer_is_f32: bool,
    ) -> Result<[QAct<'a>; 4]> {
        match &x.repr {
            QActRepr::U8(u) => {
                let u_clone = u.clone();
                self.space_to_depth2_u8(&u_clone)
            }
            QActRepr::F32(f) => {
                if !consumer_is_f32 && f.dims[1].is_multiple_of(4) {
                    let u = x.as_u8(&self.pool)?.clone();
                    return self.space_to_depth2_u8(&u);
                }
                let [n, c, h, w] = f.dims;
                if h == 0 || w == 0 || !h.is_multiple_of(2) || !w.is_multiple_of(2) {
                    return Err(Error::Invalid(format!(
                        "int8 space_to_depth2: spatial dims ({h}, {w}) must be positive and even"
                    )));
                }
                let (oh, ow) = (h / 2, w / 2);
                let spd = crate::model::v2::space_to_depth_2x2(&f.tensor)?;
                let mk = |phase: usize| -> Result<QAct<'a>> {
                    Ok(QAct {
                        repr: QActRepr::F32(QActF32 {
                            tensor: spd.narrow(1, phase * c, c)?,
                            dims: [n, c, oh, ow],
                            bg: f.bg.clone(),
                            grid_offset: f.grid_offset,
                            grids_hash: f.grids_hash,
                            max_scale: f.max_scale,
                            u8_cache: OnceCell::new(),
                        }),
                    })
                };
                Ok([mk(0)?, mk(1)?, mk(2)?, mk(3)?])
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn hyper_attend_and_scene_q_impl(
        &mut self,
        h_ctx: &HyperContext,
        av: &Tensor,
        res_dims: [usize; 4],
        pin_name: &str,
        out_f32: bool,
    ) -> Result<(QAct<'a>, Tensor)> {
        let [b, _, h, w] = res_dims;
        let cg = h_ctx.c_g;
        let proj_runs_int8 = !out_f32
            && cg.is_multiple_of(4)
            && int8_able(&h_ctx.out_proj, [b, cg, h, w], Some(res_dims)).is_some();
        if proj_runs_int8 {
            let (bg, _) = self.boundary_for(pin_name, cg)?;
            let mut out = self.pool.borrow_mut().take_buf(b, cg, h * w);
            match h_ctx.attend_and_scene_into_c4(av, b, h, w, &bg.as_ref().grids, &mut out) {
                Ok(Some(g_scene)) => return Ok((QAct::new(&self.pool, out, [b, cg, h, w], bg), g_scene)),
                Ok(None) => {
                    self.pool.borrow_mut().return_buf(out);
                }
                Err(e) => {
                    self.pool.borrow_mut().return_buf(out);
                    return Err(e);
                }
            }
        }
        let (y_ctx, g_scene) = self.hyper_attend_and_scene(h_ctx, av, b, h, w)?;
        let pin = self.quantize(pin_name, &y_ctx)?;
        Ok((pin, g_scene))
    }

    /// Fused `dequantize` + relocated `FullPadGate`: folds each channel's `1 + tanh(proj(g_scene))` multiplier
    /// directly into the per-channel dequantisation scale (`scale * gate_val`), writing the modulated `f32` head
    /// input in a single pass with zero intermediate un-modulated `f32` tensor allocation.
    pub(super) fn dequantize_fullpad_gate_impl(
        &mut self,
        x: &QAct<'a>,
        gate: &FullPadGate,
        g_scene: &Tensor,
    ) -> Result<Tensor> {
        if let QActRepr::U8(u) = &x.repr {
            let [n, c, h, w] = u.dims;
            const STACK_GATES: usize = 512;
            let mut stack_gates = [0.0f32; STACK_GATES];
            let mut heap_gates = Vec::new();
            let gates: &mut [f32] = if n * c <= STACK_GATES {
                &mut stack_gates[..n * c]
            } else {
                heap_gates.resize(n * c, 0.0);
                &mut heap_gates
            };
            if gate.gate_values_cpu(g_scene, n, c, gates)? {
                let base_grids = u.grids();
                // Guard against extreme `[0, f32::MAX]` grids or non-finite gates where folding `scale * gate`
                // could alter `+Inf`/`NaN` boundary rounding vs separate dequantize + multiply.
                let can_fold = base_grids.iter().all(|g| g.scale.is_finite() && g.scale.abs() <= 1e35)
                    && gates.iter().all(|g| g.is_finite());
                if can_fold {
                    let hw = h * w;
                    let mut out = folo_core::cpu::take_buffer(n * c * hw);
                    let pool = self.pool.borrow();
                    let view = u.view(&pool);
                    const STACK_GRIDS: usize = 512;
                    let mut stack_grids = [Grid { scale: 0.0, zero: 0 }; STACK_GRIDS];
                    let mut heap_grids = Vec::new();
                    let scaled: &mut [Grid] = if c <= STACK_GRIDS {
                        &mut stack_grids[..c]
                    } else {
                        heap_grids.resize(c, Grid { scale: 0.0, zero: 0 });
                        &mut heap_grids
                    };
                    for img in 0..n {
                        let g_row = &gates[img * c..(img + 1) * c];
                        for ch in 0..c {
                            scaled[ch] =
                                Grid { scale: base_grids[ch].scale * g_row[ch], zero: base_grids[ch].zero };
                        }
                        let img_view = C4 {
                            data: view.data,
                            offset: view.offset + img * view.img_stride,
                            img_stride: view.img_stride,
                            c,
                            zero: view.zero,
                        };
                        q8::dequantize_per_channel(
                            img_view,
                            1,
                            hw,
                            scaled,
                            &mut out[img * c * hw..(img + 1) * c * hw],
                        );
                    }
                    drop(pool);
                    return Ok(Tensor::from_vec(out, [n, c, h, w])?);
                }
            }
        }
        let raw = x.to_f32()?;
        gate.forward(&raw, g_scene)
    }
}

fn run_walk_v2(
    model: &FoloV2Model,
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
    let res = walk::features_v2(model, &mut exec, x);
    drop(exec);
    if let Ok(cell) = Rc::try_unwrap(pool_rc) {
        plan.checkin_scratch(cell.into_inner(), prev_stats);
    }
    res
}

/// Backbone and SPD-PAN neck of a [`FoloV2Model`] on `x` with the int8 plan: the head's three inputs (`f32`).
/// Call it on the CPU, inside folo-core's pool, under `no_grad`, on a fused eval-mode `f32` model.
pub fn features_v2(model: &FoloV2Model, plan: &Int8Plan, x: &Tensor) -> Result<[Tensor; 3]> {
    run_walk_v2(model, plan, None, x)
}

/// [`features_v2`] reading a pre-packed `[n, h, w]` `u32` `[R, G, B, 0]` canvas (from `Letterbox::apply_rgb0_u8`)
/// straight into the direct `u8` stem (`SPD(2×2) + 1×1` or `3×3 s=2`) with zero `f32` image materialisation.
pub fn features_v2_rgb0_u8(
    model: &FoloV2Model,
    plan: &Int8Plan,
    rgb0: &[u32],
    n: usize,
    h: usize,
    w: usize,
) -> Result<[Tensor; 3]> {
    let dummy = Tensor::zeros([0]);
    run_walk_v2(model, plan, Some((rgb0, [n, h, w])), &dummy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::quant::pool::ScratchPool;

    /// AVX2 `pack_rgb0_to_spd12_c4` and `space_to_depth2_c4` match scalar bit-for-bit across small, odd,
    /// SIMD-aligned, and SIMD+tail widths.
    #[test]
    fn spd12_and_space_to_depth2_c4_match_scalar_bit_for_bit() {
        for &(n, h, w) in &[(1, 2, 2), (1, 4, 6), (1, 4, 14), (1, 6, 16), (2, 4, 18), (1, 10, 34), (2, 8, 64)]
        {
            let rgb0: Vec<u32> = (0..n * h * w)
                .map(|i: usize| {
                    let r = (i.wrapping_mul(37) % 251) as u32;
                    let g = (i.wrapping_mul(73) % 251) as u32;
                    let b = (i.wrapping_mul(113) % 251) as u32;
                    r | (g << 8) | (b << 16)
                })
                .collect();
            let (oh, ow) = (h / 2, w / 2);
            let ohw = oh * ow;
            let mut got = vec![0u32; n * 3 * ohw];
            pack_rgb0_to_spd12_c4(&rgb0, n, h, w, &mut got);

            let mut want = vec![0u32; n * 3 * ohw];
            for img in 0..n {
                let src = &rgb0[img * h * w..(img + 1) * h * w];
                let dst = &mut want[img * 3 * ohw..(img + 1) * 3 * ohw];
                let (g0, rest) = dst.split_at_mut(ohw);
                let (g1, g2) = rest.split_at_mut(ohw);
                for oy in 0..oh {
                    for ox in 0..ow {
                        let p_tl = src[(2 * oy) * w + 2 * ox];
                        let p_tr = src[(2 * oy) * w + 2 * ox + 1];
                        let p_bl = src[(2 * oy + 1) * w + 2 * ox];
                        let p_br = src[(2 * oy + 1) * w + 2 * ox + 1];
                        let idx = oy * ow + ox;
                        g0[idx] = (p_tl & 0x00FF_FFFF) | (p_tr << 24);
                        g1[idx] = ((p_tr >> 8) & 0x0000_FFFF) | (p_bl << 16);
                        g2[idx] = ((p_bl >> 16) & 0x0000_00FF) | (p_br << 8);
                    }
                }
            }
            assert_eq!(got, want, "pack_rgb0_to_spd12_c4 mismatch for ({n}, {h}, {w})");

            // Also test `space_to_depth2_c4` for c = 12 (3 C4 groups).
            let c = 12usize;
            let groups = c / 4;
            let hw = h * w;
            let src_c4: Vec<u32> =
                (0..n * groups * hw).map(|i| (i as u32).wrapping_mul(2654435761) ^ 0xA55A_3C17).collect();
            let pool = Rc::new(RefCell::new(ScratchPool::default()));
            let u = QActU8::new(
                &pool,
                src_c4.clone(),
                [n, c, h, w],
                GridOwner::Borrowed(BoundaryGrids::pixel_12()),
            );
            let mut got_s2d = vec![0u32; n * 4 * groups * ohw];
            {
                let p = pool.borrow();
                space_to_depth2_c4(&u, p.slice(u.buf.slot()), &mut got_s2d);
            }
            let mut want_s2d = vec![0u32; n * 4 * groups * ohw];
            let part_words = groups * ohw;
            for img in 0..n {
                let dst_img = &mut want_s2d[img * 4 * part_words..(img + 1) * 4 * part_words];
                for g in 0..groups {
                    let sp = &src_c4[img * groups * hw + g * hw..][..hw];
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let idx = g * ohw + oy * ow + ox;
                            dst_img[idx] = sp[(2 * oy) * w + 2 * ox];
                            dst_img[part_words + idx] = sp[(2 * oy) * w + 2 * ox + 1];
                            dst_img[2 * part_words + idx] = sp[(2 * oy + 1) * w + 2 * ox];
                            dst_img[3 * part_words + idx] = sp[(2 * oy + 1) * w + 2 * ox + 1];
                        }
                    }
                }
            }
            assert_eq!(got_s2d, want_s2d, "space_to_depth2_c4 mismatch for ({n}, {c}, {h}, {w})");
        }
    }
}
