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
        for p in span {
            *p = src.over(*p);
        }
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

    /// `self` at alpha `a` (`0..=ONE`) over `dst`.
    #[inline(always)]
    pub fn over(&self, dst: u32, a: u32) -> u32 {
        if a >= ONE {
            return self.packed;
        }
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
        let colours = [[1., 0.5, 0.25, 0.5], [0.2, 0.4, 0.6, 0.1], [0.9, 0.9, 0.9, 0.99], [0.3, 0.7, 0.1, 1.]];
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
}
