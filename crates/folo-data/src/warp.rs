//! Affine transforms and a fast bilinear warp for 8-bit RGB images.
//!
//! Coordinates are continuous with pixel `i` covering `[i, i + 1)`, so the same matrix maps both pixel positions
//! and box corners. The warp inverts the matrix, steps the source position along each destination row in 16.16
//! fixed point and samples with 11-bit bilinear weights; rows run in parallel.

use crate::image::Image;
use crate::par::for_rows;

/// A 2x3 affine matrix `[a, b, c, d, e, f]` mapping `(x, y)` to `(a x + b y + c, d x + e y + f)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f64; 6]);

impl Default for Affine {
    fn default() -> Self {
        Affine::IDENTITY
    }
}

impl Affine {
    /// The identity transform.
    pub const IDENTITY: Affine = Affine([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    /// Translation by `(tx, ty)`.
    pub fn translate(tx: f64, ty: f64) -> Self {
        Affine([1.0, 0.0, tx, 0.0, 1.0, ty])
    }

    /// Scale about the origin.
    pub fn scale(sx: f64, sy: f64) -> Self {
        Affine([sx, 0.0, 0.0, 0.0, sy, 0.0])
    }

    /// Rotation about the origin by `degrees`, counter-clockwise on screen (y points down), as OpenCV does.
    pub fn rotate(degrees: f64) -> Self {
        let (s, c) = degrees.to_radians().sin_cos();
        Affine([c, s, 0.0, -s, c, 0.0])
    }

    /// Shear: `x += tan(sx_deg) * y`, `y += tan(sy_deg) * x`.
    pub fn shear(sx_deg: f64, sy_deg: f64) -> Self {
        Affine([1.0, sx_deg.to_radians().tan(), 0.0, sy_deg.to_radians().tan(), 1.0, 0.0])
    }

    /// `other ∘ self`: apply `self` first, then `other`.
    pub fn then(&self, other: &Affine) -> Affine {
        let [a, b, c, d, e, f] = self.0;
        let [p, q, r, s, t, u] = other.0;
        Affine([
            p * a + q * d,
            p * b + q * e,
            p * c + q * f + r,
            s * a + t * d,
            s * b + t * e,
            s * c + t * f + u,
        ])
    }

    /// Determinant of the linear part (area scale factor).
    pub fn det(&self) -> f64 {
        let [a, b, _, d, e, _] = self.0;
        a * e - b * d
    }

    /// The inverse transform, or `None` when the matrix is singular.
    pub fn inverse(&self) -> Option<Affine> {
        let [a, b, c, d, e, f] = self.0;
        let det = self.det();
        if det.abs() < 1e-12 {
            return None;
        }
        let (ia, ib, id, ie) = (e / det, -b / det, -d / det, a / det);
        Some(Affine([ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)]))
    }

    /// Applies the transform to a point.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d, e, f] = self.0;
        (a * x + b * y + c, d * x + e * y + f)
    }

    /// Whether this is (numerically) the identity.
    pub fn is_identity(&self) -> bool {
        self.0.iter().zip(Affine::IDENTITY.0).all(|(a, b)| (a - b).abs() < 1e-9)
    }
}

const FRAC: u32 = 16;
const WBITS: u32 = 11;
const WONE: u32 = 1 << WBITS;

/// Warps `src` by `m` (source → destination) into a `width x height` image; uncovered pixels get `border`.
pub fn warp_affine(src: &Image, m: &Affine, width: u32, height: u32, border: [u8; 3]) -> Image {
    let mut out = vec![0u8; width as usize * height as usize * 3];
    let Some(inv) = m.inverse() else {
        // Degenerate transform: nothing of the source is visible.
        let mut img = Image { width, height, data: out };
        img.fill_rect(0, 0, width as i64, height as i64, border);
        return img;
    };
    let [a, b, c, d, e, f] = inv.0;
    let (sw, sh) = (src.width as i64, src.height as i64);
    let sstride = src.stride();
    let sdata = &src.data;
    let fone = (1i64 << FRAC) as f64;
    for_rows(&mut out, width as usize * 3, |y, row| {
        // Source position (in pixel-index space, centres at integers) of destination pixel centre (0.5, y + 0.5).
        let yc = y as f64 + 0.5;
        let u0 = a * 0.5 + b * yc + c - 0.5;
        let v0 = d * 0.5 + e * yc + f - 0.5;
        let mut fu = (u0 * fone).round() as i64;
        let mut fv = (v0 * fone).round() as i64;
        let du = (a * fone).round() as i64;
        let dv = (d * fone).round() as i64;
        for px in row.as_chunks_mut::<3>().0.iter_mut() {
            let ix = fu >> FRAC;
            let iy = fv >> FRAC;
            let wx = ((fu >> (FRAC - WBITS)) & (WONE as i64 - 1)) as u32;
            let wy = ((fv >> (FRAC - WBITS)) & (WONE as i64 - 1)) as u32;
            fu += du;
            fv += dv;
            if ix >= 0 && iy >= 0 && ix + 1 < sw && iy + 1 < sh {
                let o = iy as usize * sstride + ix as usize * 3;
                let (r0, r1) = (&sdata[o..o + 6], &sdata[o + sstride..o + sstride + 6]);
                let (iwx, iwy) = (WONE - wx, WONE - wy);
                for ch in 0..3 {
                    let t = r0[ch] as u32 * iwx + r0[ch + 3] as u32 * wx;
                    let u = r1[ch] as u32 * iwx + r1[ch + 3] as u32 * wx;
                    px[ch] = ((t * iwy + u * wy + (1 << (2 * WBITS - 1))) >> (2 * WBITS)) as u8;
                }
            } else if ix < -1 || iy < -1 || ix >= sw || iy >= sh {
                px.copy_from_slice(&border);
            } else {
                // Straddles the edge: missing neighbours take the border colour.
                let fetch = |x: i64, y: i64, ch: usize| -> u32 {
                    if x >= 0 && y >= 0 && x < sw && y < sh {
                        sdata[y as usize * sstride + x as usize * 3 + ch] as u32
                    } else {
                        border[ch] as u32
                    }
                };
                let (iwx, iwy) = (WONE - wx, WONE - wy);
                for (ch, out) in px.iter_mut().enumerate() {
                    let t = fetch(ix, iy, ch) * iwx + fetch(ix + 1, iy, ch) * wx;
                    let u = fetch(ix, iy + 1, ch) * iwx + fetch(ix + 1, iy + 1, ch) * wx;
                    *out = ((t * iwy + u * wy + (1 << (2 * WBITS - 1))) >> (2 * WBITS)) as u8;
                }
            }
        }
    });
    Image { width, height, data: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(w: u32, h: u32) -> Image {
        let mut img = Image::new(w, h, [0, 0, 0]);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 13 % 251) as u8;
        }
        img
    }

    #[test]
    fn matrix_algebra() {
        let m = Affine::rotate(30.0).then(&Affine::scale(2.0, 1.5)).then(&Affine::translate(3.0, -4.0));
        let inv = m.inverse().unwrap();
        let (x, y) = m.apply(5.0, 7.0);
        let (bx, by) = inv.apply(x, y);
        assert!((bx - 5.0).abs() < 1e-9 && (by - 7.0).abs() < 1e-9);
        assert!(m.then(&inv).is_identity());
        assert!((m.det() - 3.0).abs() < 1e-9);
        // OpenCV convention: +90 degrees takes the x axis to -y (up on screen).
        let (rx, ry) = Affine::rotate(90.0).apply(1.0, 0.0);
        assert!(rx.abs() < 1e-12 && (ry + 1.0).abs() < 1e-12);
    }

    #[test]
    fn identity_and_integer_translation_are_exact() {
        let img = pattern(31, 17);
        assert_eq!(warp_affine(&img, &Affine::IDENTITY, 31, 17, [1, 2, 3]), img);
        let t = warp_affine(&img, &Affine::translate(3.0, 2.0), 31, 17, [1, 2, 3]);
        assert_eq!(t.pixel(3, 2), img.pixel(0, 0));
        assert_eq!(t.pixel(30, 16), img.pixel(27, 14));
        assert_eq!(t.pixel(0, 0), [1, 2, 3]);
        assert_eq!(t.pixel(2, 5), [1, 2, 3]);
    }

    #[test]
    fn mirror_by_matrix() {
        let img = pattern(20, 10);
        let flip = Affine([-1.0, 0.0, 20.0, 0.0, 1.0, 0.0]);
        let out = warp_affine(&img, &flip, 20, 10, [0; 3]);
        for x in 0..20 {
            assert_eq!(out.pixel(x, 4), img.pixel(19 - x, 4));
        }
    }

    #[test]
    fn parallel_matches_serial() {
        let img = pattern(400, 300);
        let m = Affine::translate(-200.0, -150.0)
            .then(&Affine::rotate(17.0))
            .then(&Affine::scale(1.3, 1.3))
            .then(&Affine::translate(320.0, 320.0));
        let a = warp_affine(&img, &m, 640, 640, [114; 3]);
        let b = crate::par::serial(|| warp_affine(&img, &m, 640, 640, [114; 3]));
        assert_eq!(a, b);
    }
}
