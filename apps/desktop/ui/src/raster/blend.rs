//! Pixel blending on 8-bit gamma-space channels, as GPUI's DX11 renderer does (`B8G8R8A8_UNORM`,
//! `SRC_ALPHA / INV_SRC_ALPHA`), in integer arithmetic. Pixels are `0x00RRGGBB`.
//!
//! The GPU blends `src.rgb * a + dst * (1 - a)` in float and rounds to 8 bits. Here every factor is
//! fixed point with 16 fractional bits, so the result is the float one rounded the same way except
//! on exact ties (within 1/255 always).

use crate::{Hsla, Rgba};

/// One in the 16-bit fixed point alphas use.
pub(crate) const ONE: u32 = 1 << 16;

/// GPUI's colour conversion (`From<Hsla> for Rgba`), as `[r, g, b, a]`.
#[inline]
pub(crate) fn rgba(color: Hsla) -> [f32; 4] {
    let c = Rgba::from(color);
    [c.r, c.g, c.b, c.a]
}

/// `saturate` (NaN becomes 0, as a UNORM conversion does).
#[inline(always)]
pub(crate) fn sat(v: f32) -> f32 {
    if v > 0. { v.min(1.) } else { 0. }
}

/// A float channel as a UNORM render target stores it (the reference the integer paths match).
#[cfg(test)]
#[inline(always)]
pub(crate) fn unorm(v: f32) -> u32 {
    (sat(v) * 255. + 0.5) as u32
}

/// An alpha in `0..=ONE`.
#[inline(always)]
pub(crate) fn alpha16(a: f32) -> u32 {
    (sat(a) * ONE as f32 + 0.5) as u32
}

#[inline(always)]
pub(crate) fn pack(r: u32, g: u32, b: u32) -> u32 {
    r << 16 | g << 8 | b
}

/// A constant source colour: premultiplied channels (`c * a * 255`, 16 fractional bits) and `1 - a`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Src {
    pub r: u32,
    pub g: u32,
    pub b: u32,
    /// `1 - a` in `0..=ONE`.
    pub inv: u32,
    /// The pixel an opaque source writes.
    pub packed: u32,
}

impl Src {
    /// Draws nothing.
    pub const NOOP: Src = Src { r: 0, g: 0, b: 0, inv: ONE, packed: 0 };

    /// From the shader's output: straight colour and alpha.
    #[inline]
    pub fn new(c: [f32; 4]) -> Self {
        let a = sat(c[3]);
        let k = a * 255. * ONE as f32;
        let r = (sat(c[0]) * k + 0.5) as u32;
        let g = (sat(c[1]) * k + 0.5) as u32;
        let b = (sat(c[2]) * k + 0.5) as u32;
        let inv = ONE - alpha16(a);
        let h = ONE / 2;
        Src { r, g, b, inv, packed: pack((r + h) >> 16, (g + h) >> 16, (b + h) >> 16) }
    }

    /// Whether drawing this changes nothing.
    #[inline]
    pub fn is_noop(&self) -> bool {
        self.inv == ONE
    }

    #[inline(always)]
    pub fn over(&self, dst: u32) -> u32 {
        if self.inv == 0 {
            return self.packed;
        }
        let h = ONE / 2;
        let r = (self.r + ((dst >> 16) & 0xff) * self.inv + h) >> 16;
        let g = (self.g + ((dst >> 8) & 0xff) * self.inv + h) >> 16;
        let b = (self.b + (dst & 0xff) * self.inv + h) >> 16;
        pack(r, g, b)
    }
}

/// Draws `src` over every pixel of `span`.
#[inline]
pub(crate) fn fill_span(span: &mut [u32], src: &Src) {
    if src.inv == 0 {
        span.fill(src.packed);
    } else if !src.is_noop() {
        #[cfg(target_arch = "x86_64")]
        if span.len() >= 4
            && src.inv < ONE
            && src.r.max(src.g).max(src.b).saturating_add(255 * src.inv).saturating_add(ONE / 2) < (256 << 16)
        {
            let (chunks, tail) = span.as_chunks_mut::<4>();
            // SAFETY: SSE2 is guaranteed on all x86_64 targets; the guard above guarantees
            // `src.inv <= u16::MAX` and every blended channel stays in `0..=255` so `_mm_packus_epi16`
            // never saturates and matches `Src::over` bit-for-bit.
            unsafe { fill_span_sse2(chunks, src) };
            for p in tail {
                *p = src.over(*p);
            }
            return;
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if span.len() >= 4
            && src.inv < ONE
            && src.r.max(src.g).max(src.b).saturating_add(255 * src.inv).saturating_add(ONE / 2) < (256 << 16)
        {
            let (chunks, tail) = span.as_chunks_mut::<4>();
            // SAFETY: NEON is guaranteed on all aarch64 targets; the guard above guarantees
            // `src.inv <= u16::MAX` and every blended channel stays in `0..=255` so narrowing matches
            // `Src::over` bit-for-bit.
            unsafe { fill_span_neon(chunks, src) };
            for p in tail {
                *p = src.over(*p);
            }
            return;
        }
        for p in span {
            *p = src.over(*p);
        }
    }
}

/// Blends `src` over 4 `0x00RRGGBB` pixels per iteration using baseline little-endian aarch64 NEON.
///
/// Caller must ensure `1 <= src.inv < ONE` and `max(r, g, b) + 255 * inv + ONE / 2 < 256 << 16`.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn fill_span_neon(chunks: &mut [[u32; 4]], src: &Src) {
    use std::arch::aarch64::*;

    let h = ONE / 2;
    let bias_arr = [src.b + h, src.g + h, src.r + h, 0u32];
    // SAFETY: `bias_arr` holds 4 contiguous `u32` lanes.
    let src_bias = unsafe { vld1q_u32(bias_arr.as_ptr()) };
    let inv16 = vdup_n_u16(src.inv as u16);
    let rgb_mask = vreinterpretq_u8_u32(vdupq_n_u32(0x00ff_ffff));

    for chunk in chunks {
        let ptr = chunk.as_mut_ptr().cast::<u8>();
        // SAFETY: `chunk` is `&mut [u32; 4]` (16 bytes), valid for 128-bit load/store.
        let px = vandq_u8(unsafe { vld1q_u8(ptr) }, rgb_mask);

        let d01 = vmovl_u8(vget_low_u8(px));
        let p0 = vshrn_n_u32::<16>(vmlal_u16(src_bias, vget_low_u16(d01), inv16));
        let p1 = vshrn_n_u32::<16>(vmlal_u16(src_bias, vget_high_u16(d01), inv16));
        let out01 = vmovn_u16(vcombine_u16(p0, p1));

        let d23 = vmovl_u8(vget_high_u8(px));
        let p2 = vshrn_n_u32::<16>(vmlal_u16(src_bias, vget_low_u16(d23), inv16));
        let p3 = vshrn_n_u32::<16>(vmlal_u16(src_bias, vget_high_u16(d23), inv16));
        let out23 = vmovn_u16(vcombine_u16(p2, p3));

        // SAFETY: `ptr` points to 16 writable bytes of `chunk`.
        unsafe { vst1q_u8(ptr, vcombine_u8(out01, out23)) };
    }
}

/// Blends `src` over 4 `0x00RRGGBB` pixels per iteration using baseline x86_64 SSE2.
///
/// Caller must ensure `1 <= src.inv < ONE` and `max(r, g, b) + 255 * inv + ONE / 2 < 256 << 16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn fill_span_sse2(chunks: &mut [[u32; 4]], src: &Src) {
    use std::arch::x86_64::*;

    let h = ONE / 2;
    let (rb, gb, bb) = (src.r + h, src.g + h, src.b + h);
    let zero = _mm_setzero_si128();
    let rgb_mask = _mm_set1_epi32(0x00ff_ffff);
    let inv16 = _mm_set1_epi16(src.inv as i16);
    let sign_bit = _mm_set1_epi16(i16::MIN);
    // Little-endian `0x00RRGGBB` unpacks to `[B, G, R, 0]` per pixel.
    let src_lo = _mm_setr_epi16(bb as i16, gb as i16, rb as i16, 0, bb as i16, gb as i16, rb as i16, 0);
    let src_hi = _mm_setr_epi16(
        (bb >> 16) as i16,
        (gb >> 16) as i16,
        (rb >> 16) as i16,
        0,
        (bb >> 16) as i16,
        (gb >> 16) as i16,
        (rb >> 16) as i16,
        0,
    );
    let src_lo_bias = _mm_xor_si128(src_lo, sign_bit);

    for chunk in chunks {
        let ptr = chunk.as_mut_ptr().cast::<__m128i>();
        // SAFETY: `chunk` is `&mut [u32; 4]` (16 bytes), valid for unaligned 128-bit load/store.
        let px = _mm_and_si128(unsafe { _mm_loadu_si128(ptr) }, rgb_mask);

        let d01 = _mm_unpacklo_epi8(px, zero);
        let lo01 = _mm_mullo_epi16(d01, inv16);
        let hi01 = _mm_mulhi_epu16(d01, inv16);
        let sum01 = _mm_add_epi16(lo01, src_lo);
        // Unsigned carry (`sum01 < src_lo`): -1 (0xffff) when carried, 0 otherwise; subtract to add 1.
        let carry01 = _mm_cmplt_epi16(_mm_xor_si128(sum01, sign_bit), src_lo_bias);
        let out01 = _mm_sub_epi16(_mm_add_epi16(hi01, src_hi), carry01);

        let d23 = _mm_unpackhi_epi8(px, zero);
        let lo23 = _mm_mullo_epi16(d23, inv16);
        let hi23 = _mm_mulhi_epu16(d23, inv16);
        let sum23 = _mm_add_epi16(lo23, src_lo);
        let carry23 = _mm_cmplt_epi16(_mm_xor_si128(sum23, sign_bit), src_lo_bias);
        let out23 = _mm_sub_epi16(_mm_add_epi16(hi23, src_hi), carry23);

        // SAFETY: `ptr` points to 16 writable bytes of `chunk`.
        unsafe { _mm_storeu_si128(ptr, _mm_packus_epi16(out01, out23)) };
    }
}

/// A colour whose alpha varies per pixel (masks): `c * 255` with 8 fractional bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Tint {
    pub r: u32,
    pub g: u32,
    pub b: u32,
    /// What a pixel of alpha `ONE` becomes (the same rounding as `over`).
    pub packed: u32,
}

impl Tint {
    #[inline]
    pub fn new(c: [f32; 4]) -> Self {
        let q = |v: f32| (sat(v) * 255. * 256. + 0.5) as u32;
        let (r, g, b) = (q(c[0]), q(c[1]), q(c[2]));
        let h = 1 << 7;
        Tint { r, g, b, packed: pack((r + h) >> 8, (g + h) >> 8, (b + h) >> 8) }
    }

    /// `self` at alpha `a` (`0..=ONE`) over `dst`. Branch-free: alpha 0 gives `dst` and alpha `ONE`
    /// gives `packed` through the same formula.
    #[inline(always)]
    pub fn over(&self, dst: u32, a: u32) -> u32 {
        debug_assert!(a <= ONE);
        // Cq * a + (d << 8) * (ONE - a) <= 65280 * 65536: fits in u32 with the rounding term.
        let ia = ONE - a;
        let h = 1 << 23;
        let r = (self.r * a + (((dst >> 16) & 0xff) << 8) * ia + h) >> 24;
        let g = (self.g * a + (((dst >> 8) & 0xff) << 8) * ia + h) >> 24;
        let b = (self.b * a + ((dst & 0xff) << 8) * ia + h) >> 24;
        pack(r, g, b)
    }
}

/// A premultiplied 8-bit pixel (`a`, `r`, `g`, `b` with `r, g, b <= a`) at extra alpha `m` (`0..=ONE`)
/// over `dst`.
#[inline(always)]
pub(crate) fn over_pm8(dst: u32, a: u32, r: u32, g: u32, b: u32, m: u32) -> u32 {
    if m >= ONE && a == 255 {
        return pack(r, g, b);
    }
    let inv = ONE - (a * m + 127) / 255;
    let h = ONE / 2;
    let or = (r * m + ((dst >> 16) & 0xff) * inv + h) >> 16;
    let og = (g * m + ((dst >> 8) & 0xff) * inv + h) >> 16;
    let ob = (b * m + (dst & 0xff) * inv + h) >> 16;
    pack(or.min(255), og.min(255), ob.min(255))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(c: [f32; 4], dst: u32) -> [u32; 3] {
        let a = sat(c[3]);
        let d = [(dst >> 16) & 0xff, (dst >> 8) & 0xff, dst & 0xff];
        std::array::from_fn(|i| unorm(sat(c[i]) * a + d[i] as f32 / 255. * (1. - a)))
    }

    fn channels(p: u32) -> [u32; 3] {
        [(p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff]
    }

    #[test]
    fn blends_round_like_a_unorm_target() {
        let colours =
            [[1., 0.5, 0.25, 0.5], [0.2, 0.4, 0.6, 0.1], [0.9, 0.9, 0.9, 0.99], [0.3, 0.7, 0.1, 1.]];
        let dsts = [0x000000, 0xffffff, 0x123456, 0x80ff01];
        for c in colours {
            for d in dsts {
                let want = reference(c, d);
                let got = channels(Src::new(c).over(d));
                let tint = channels(Tint::new(c).over(d, alpha16(c[3])));
                for i in 0..3 {
                    assert!(want[i].abs_diff(got[i]) <= 1 && want[i].abs_diff(tint[i]) <= 1, "{c:?} {d:#x}");
                }
            }
        }
        // Hand-worked: 0.5 white over black is 127.5 -> 128; 0.25 of 200 over 100 is 125.
        assert_eq!(Src::new([1., 1., 1., 0.5]).over(0), 0x808080);
        assert_eq!(Src::new([200. / 255., 0., 0., 0.25]).over(pack(100, 0, 0)), pack(125, 0, 0));
        assert_eq!(Src::new([0.2, 0.4, 0.6, 1.]).packed, pack(51, 102, 153));
        assert!(Src::new([1., 1., 1., 0.]).is_noop());
    }

    #[test]
    fn fill_span_matches_scalar_over_bit_for_bit() {
        let mut srcs = vec![
            Src::NOOP,
            Src::new([0., 0., 0., 0.]),
            Src::new([1., 1., 1., 1.]),
            Src::new([0.2, 0.4, 0.6, 1.]),
            Src::new([1., 0.5, 0.25, 0.5]),
            Src::new([0., 1., 0.75, 1. / 65536.]),
            Src::new([1., 1., 1., 65535. / 65536.]),
            Src::new([0.99, 0.01, 0.5, 0.37]),
        ];
        // Synthetic boundary `Src` values stressing 16-bit carry transitions (0x0000, 0x7fff, 0x8000, 0xffff).
        for inv in [1u32, 2, 0x7fff, 0x8000, 0x8001, 0xffff] {
            for lo in [0x0000u16, 0x7fff, 0x8000, 0xffff] {
                let r = (10 << 16) | u32::from(lo);
                let g = (20 << 16) | u32::from(lo.wrapping_sub(0x8000));
                let b = u32::from(lo);
                if r.max(g).max(b) + 255 * inv + (ONE / 2) < (256 << 16) {
                    srcs.push(Src { r, g, b, inv, packed: 0 });
                }
            }
        }
        // Span lengths covering 0, sub-vector tails, aligned multiples of 4, and unaligned tails.
        let dst_pattern = [
            0x0000_0000u32,
            0x00ff_ffff,
            0xff12_3456, // non-zero high byte must be cleared just like `pack(r, g, b)`
            0xab80_7f01,
            0x00fe_017f,
            0x7f00_ff80,
            0x0001_0203,
            0xdead_beef,
        ];
        for src in srcs {
            for len in 0..=19 {
                let initial: Vec<u32> = (0..len)
                    .map(|i| dst_pattern[(i + len) % dst_pattern.len()].wrapping_add(i as u32 * 0x0003_070b))
                    .collect();
                let mut got = initial.clone();
                fill_span(&mut got, &src);
                let mut want = initial;
                if !src.is_noop() {
                    for p in &mut want {
                        *p = src.over(*p);
                    }
                }
                assert_eq!(got, want, "mismatch for src={src:?} len={len}");
            }
        }
    }
}
