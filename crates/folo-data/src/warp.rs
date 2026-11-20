//! Affine transforms and a fast bilinear warp for 8-bit RGB images.
//!
//! Coordinates are continuous with pixel `i` covering `[i, i + 1)`, so the same matrix maps both pixel positions
//! and box corners. The warp inverts the matrix, steps the source position along each destination row in 16.16
//! fixed point and samples with 11-bit bilinear weights; rows run in parallel.
//!
//! When the source position does not move vertically along a destination row (no rotation or shear: the default
//! training augmentation), every pixel of a destination row reads the same two source rows, and the warp blends
//! those two rows once into `u32` sums and then interpolates along them, without a bounds check per pixel. The
//! products and their single rounding are those of the general path, so both give the same bytes. The warp reads
//! its source as a [`Plane`]: images placed on a plane of one colour, so a mosaic is warped straight from its four
//! images without painting its canvas first.

use crate::image::Image;
use crate::par::{for_bands, for_rows};

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

/// An image placed on a [`Plane`]: plane pixel `(x, y)` in `[x0, x1) x [y0, y1)` is pixel
/// `(x - x0 + sx, y - y0 + sy)` of `img`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tile<'a> {
    pub img: &'a Image,
    /// The covered rectangle `[x0, y0, x1, y1]` on the plane (end-exclusive), inside `img` once offset.
    pub dst: [i64; 4],
    /// `[sx, sy]`: the pixel of `img` at `(x0, y0)`.
    pub src: [i64; 2],
}

/// A warp's source: tiles on an endless plane of colour `fill`, of which `width x height` from the origin is the
/// source image (its size for the matrix). A plane pixel no tile covers is `fill`, inside that rectangle or not.
/// Tiles do not overlap.
#[derive(Clone, Debug)]
pub(crate) struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub fill: [u8; 3],
    pub tiles: Vec<Tile<'a>>,
}

impl<'a> Plane<'a> {
    /// `img` with `fill` around it.
    pub fn image(img: &'a Image, fill: [u8; 3]) -> Self {
        let tile = Tile { img, dst: [0, 0, img.width as i64, img.height as i64], src: [0, 0] };
        Plane { width: img.width, height: img.height, fill, tiles: vec![tile] }
    }

    /// Writes plane row `y`, columns `x0 .. x0 + out.len() / 3`, into `out` (interleaved RGB).
    fn row(&self, y: i64, x0: i64, out: &mut [u8]) {
        for px in out.as_chunks_mut::<3>().0.iter_mut() {
            *px = self.fill;
        }
        let x1 = x0 + (out.len() / 3) as i64;
        for t in &self.tiles {
            let [tx0, ty0, tx1, ty1] = t.dst;
            let (a, b) = (tx0.max(x0), tx1.min(x1));
            if y < ty0 || y >= ty1 || a >= b {
                continue;
            }
            let img = t.img;
            let (sx, sy) = ((a - tx0 + t.src[0]) as usize, (y - ty0 + t.src[1]) as usize);
            let n = (b - a) as usize * 3;
            out[(a - x0) as usize * 3..][..n].copy_from_slice(&img.data[sy * img.stride() + sx * 3..][..n]);
        }
    }

    /// The `width x height` source image itself.
    pub fn render(&self) -> Image {
        let mut img = Image {
            width: self.width,
            height: self.height,
            data: crate::buffers::scratch(self.width as usize * self.height as usize * 3),
        };
        if img.data.is_empty() {
            return img;
        }
        for px in img.data.as_chunks_mut::<3>().0.iter_mut() {
            *px = self.fill;
        }
        let stride = img.stride();
        for (y, row) in img.data.chunks_exact_mut(stride).enumerate() {
            let y = y as i64;
            if self.tiles.iter().any(|t| (t.dst[1]..t.dst[3]).contains(&y)) {
                self.row(y, 0, row);
            }
        }
        img
    }
}

/// Warps `src` by `m` (source → destination) into a `width x height` image; uncovered pixels get `border`.
pub fn warp_affine(src: &Image, m: &Affine, width: u32, height: u32, border: [u8; 3]) -> Image {
    warp_plane(&Plane::image(src, border), m, width, height)
}

/// Warps `plane` by `m` into a `width x height` image: bit for bit [`warp_affine`] of [`Plane::render`] with the
/// plane's fill as the border, without rendering the plane when the source row does not change along a destination
/// row.
pub(crate) fn warp_plane(plane: &Plane, m: &Affine, width: u32, height: u32) -> Image {
    let mut out = crate::buffers::scratch(width as usize * height as usize * 3);
    let border = plane.fill;
    let Some(inv) = m.inverse() else {
        // Degenerate transform: nothing of the source is visible.
        let mut img = Image { width, height, data: out };
        img.fill_rect(0, 0, width as i64, height as i64, border);
        return img;
    };
    if (inv.0[3] * (1i64 << FRAC) as f64).round() as i64 == 0 {
        warp_rows(plane, &inv, width as usize, &mut out);
    } else {
        match plane.tiles.as_slice() {
            // One tile that is its whole image and the whole plane: warp the image itself. (A tile showing part of
            // a larger image must not: the general path would read the image's pixels outside the plane.)
            [t] if t.dst == [0, 0, plane.width as i64, plane.height as i64]
                && t.src == [0, 0]
                && (t.img.width, t.img.height) == (plane.width, plane.height) =>
            {
                warp_general(t.img, &inv, width as usize, border, &mut out)
            }
            _ => {
                let rendered = plane.render();
                warp_general(&rendered, &inv, width as usize, border, &mut out);
                crate::buffers::recycle(rendered.data);
            }
        }
    }
    Image { width, height, data: out }
}

/// The warp of `src` by the inverse matrix `inv` into `out` (rows of `width` pixels), any matrix: each pixel steps
/// its own source position.
fn warp_general(src: &Image, inv: &Affine, width: usize, border: [u8; 3], out: &mut [u8]) {
    let [a, b, c, d, e, f] = inv.0;
    let (sw, sh) = (src.width as i64, src.height as i64);
    let sstride = src.stride();
    let sdata = &src.data;
    let fone = (1i64 << FRAC) as f64;
    for_rows(out, width * 3, |y, row| {
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
}

/// The warp of `plane` by the inverse matrix `inv` into `out` (rows of `width` pixels) when the source row does not
/// change along a destination row (`inv`'s vertical step per pixel rounds to zero). Per destination row, the two
/// source rows `iy`, `iy + 1` over the columns the row reaches are blended into `v = top·(1 − wy) + bottom·wy`
/// (`u32`), then each pixel is `(v[ix]·(1 − wx) + v[ix + 1]·wx + 2^21) >> 22`: the four products and the one
/// rounding of the general path's `(t·(1 − wy) + u·wy + 2^21) >> 22`, so the same bytes. A pixel whose taps are
/// all off the plane's source rectangle gets the fill colour, as the general path gives it the border.
fn warp_rows(plane: &Plane, inv: &Affine, width: usize, out: &mut [u8]) {
    let [a, b, c, d, e, f] = inv.0;
    let fone = (1i64 << FRAC) as f64;
    let du = (a * fone).round() as i64;
    let (pw, ph) = (plane.width as i64, plane.height as i64);
    let fill = plane.fill;
    for_bands(out, width * 3, |y0, band| {
        // Scratch for columns `-1 ..= pw` at most.
        let cap = (pw as usize + 2) * 3;
        let (mut top, mut bottom, mut v) = (vec![0u8; cap], vec![0u8; cap], vec![0u32; cap]);
        for (i, row) in band.chunks_exact_mut(width * 3).enumerate() {
            let yc = (y0 + i) as f64 + 0.5;
            let u0 = a * 0.5 + b * yc + c - 0.5;
            let v0 = d * 0.5 + e * yc + f - 0.5;
            let fu0 = (u0 * fone).round() as i64;
            let fv = (v0 * fone).round() as i64;
            let iy = fv >> FRAC;
            let wy = ((fv >> (FRAC - WBITS)) & (WONE as i64 - 1)) as u32;
            // Left taps run from `ix_a` to `ix_b`; off columns `-1 ..= pw - 1` both taps are `fill`. The last
            // position in `i128`: the per-pixel steps below stay in `i64` as the general path's do, but the product
            // `du · (width - 1)` alone can overflow where the running sum does not.
            let last_fu = fu0 as i128 + du as i128 * (width as i128 - 1);
            let ix_a = fu0 >> FRAC;
            let ix_b = (last_fu >> FRAC).clamp(i64::MIN as i128 / 2, i64::MAX as i128 / 2) as i64;
            let lo = ix_a.min(ix_b).max(-1);
            let hi = (ix_a.max(ix_b) + 1).min(pw);
            if iy < -1 || iy >= ph || hi <= lo {
                for px in row.as_chunks_mut::<3>().0.iter_mut() {
                    *px = fill;
                }
                continue;
            }
            let n = (hi - lo + 1) as usize * 3;
            plane.row(iy, lo, &mut top[..n]);
            plane.row(iy + 1, lo, &mut bottom[..n]);
            let iwy = WONE - wy;
            for ((v, &t), &u) in v[..n].iter_mut().zip(&top[..n]).zip(&bottom[..n]) {
                *v = t as u32 * iwy + u as u32 * wy;
            }
            // A pixel whose left tap is column `lo + k` reads `v[k]` and `v[k + 1]`, so needs `k < last`.
            let last = (n / 3 - 1) as u64;
            let mut fu = fu0;
            for px in row.as_chunks_mut::<3>().0.iter_mut() {
                let k = ((fu >> FRAC) - lo) as u64;
                let wx = ((fu >> (FRAC - WBITS)) & (WONE as i64 - 1)) as u32;
                fu += du;
                if k < last {
                    let t = &v[k as usize * 3..][..6];
                    let iwx = WONE - wx;
                    for ch in 0..3 {
                        px[ch] =
                            ((t[ch] * iwx + t[ch + 3] * wx + (1 << (2 * WBITS - 1))) >> (2 * WBITS)) as u8;
                    }
                } else {
                    *px = fill;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

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

    /// Scale (mirrored sometimes, each axis its own) and translate, no rotation or shear: no vertical step.
    fn random_upright(rng: &mut Rng, sw: u32, sh: u32) -> Affine {
        let mut scale = || {
            let s = rng.uniform(0.2, 3.0) as f64;
            if rng.chance(0.2) { -s } else { s }
        };
        let (sx, sy) = (scale(), scale());
        let (tx, ty) = (rng.uniform(-40.0, 90.0) as f64, rng.uniform(-40.0, 90.0) as f64);
        Affine::translate(-(sw as f64) / 2.0, -(sh as f64) / 2.0)
            .then(&Affine::scale(sx, sy))
            .then(&Affine::translate(tx, ty))
    }

    #[test]
    fn row_path_matches_the_general_path() {
        let mut rng = Rng::new(5);
        let images: Vec<Image> =
            [(40, 30), (17, 53), (64, 64), (5, 3), (1, 1)].map(|(w, h)| pattern(w, h)).into();
        for case in 0..400 {
            let img = &images[case % images.len()];
            let m = random_upright(&mut rng, img.width, img.height);
            let inv = m.inverse().unwrap();
            assert_eq!((inv.0[3] * (1 << FRAC) as f64).round(), 0.0);
            let (w, h) = (1 + rng.below(70), 1 + rng.below(50));
            let border = [rng.below(256) as u8, 114, 7];
            let (mut rows, mut general) = (vec![0u8; w * h * 3], vec![0u8; w * h * 3]);
            warp_rows(&Plane::image(img, border), &inv, w, &mut rows);
            warp_general(img, &inv, w, border, &mut general);
            assert_eq!(rows, general, "case {case}: {m:?} into {w}x{h}");
        }
    }

    #[test]
    fn plane_warps_as_its_rendered_image() {
        let mut rng = Rng::new(8);
        let images: Vec<Image> = [(30, 20), (25, 41), (12, 12), (50, 9)].map(|(w, h)| pattern(w, h)).into();
        for case in 0..200 {
            // Four tiles in the quadrants around a centre, cropped by the plane's edges, as a mosaic places them.
            let (pw, ph) = (40 + rng.below(30) as i64, 40 + rng.below(30) as i64);
            let (xc, yc) = (rng.below(pw as usize) as i64, rng.below(ph as usize) as i64);
            let mut tiles = Vec::new();
            for (k, img) in images.iter().enumerate() {
                let (w, h) = (img.width as i64, img.height as i64);
                let (x0, x1) = if k % 2 == 0 { ((xc - w).max(0), xc) } else { (xc, (xc + w).min(pw)) };
                let (y0, y1) = if k < 2 { ((yc - h).max(0), yc) } else { (yc, (yc + h).min(ph)) };
                if x1 > x0 && y1 > y0 {
                    let src =
                        [if k % 2 == 0 { w - (x1 - x0) } else { 0 }, if k < 2 { h - (y1 - y0) } else { 0 }];
                    tiles.push(Tile { img, dst: [x0, y0, x1, y1], src });
                }
            }
            let plane = Plane { width: pw as u32, height: ph as u32, fill: [114; 3], tiles };
            let rendered = plane.render();
            let (w, h) = (1 + rng.below(60) as u32, 1 + rng.below(60) as u32);
            let m = if case % 4 == 3 {
                // A rotation takes the general path, through the rendered plane.
                Affine::rotate(rng.uniform(-30.0, 30.0) as f64).then(&Affine::translate(10.0, 5.0))
            } else {
                random_upright(&mut rng, plane.width, plane.height)
            };
            assert_eq!(
                warp_plane(&plane, &m, w, h),
                warp_affine(&rendered, &m, w, h, [114; 3]),
                "case {case}"
            );
            let inv = m.inverse().unwrap();
            let mut general = vec![0u8; (w * h * 3) as usize];
            warp_general(&rendered, &inv, w as usize, [114; 3], &mut general);
            assert_eq!(warp_plane(&plane, &m, w, h).data, general, "case {case}");
        }
    }

    /// A single tile showing part of a larger image (a mosaic's one visible tile, cropped by the canvas), warped
    /// with shear: the warp reads only the shown part, as the rendered plane has it.
    #[test]
    fn a_cropped_tile_shows_only_its_part() {
        let img = pattern(4, 4);
        let tiles = |dst, src| vec![Tile { img: &img, dst, src }];
        for (dst, src) in [([0, 0, 2, 2], [0, 0]), ([0, 0, 2, 2], [2, 2]), ([0, 0, 3, 2], [1, 0])] {
            let plane =
                Plane { width: dst[2] as u32, height: dst[3] as u32, fill: [114; 3], tiles: tiles(dst, src) };
            for m in [
                Affine([1.0, 0.0, -2.0, 1.0, 1.0, -4.0]),
                Affine([1.0, 0.3, 0.0, 0.2, 1.0, 0.0]),
                Affine::rotate(20.0).then(&Affine::scale(2.0, 2.0)),
            ] {
                for (w, h) in [(1, 1), (5, 4)] {
                    assert_eq!(
                        warp_plane(&plane, &m, w, h),
                        warp_affine(&plane.render(), &m, w, h, [114; 3]),
                        "{dst:?} {src:?} {m:?}"
                    );
                }
            }
        }
    }

    /// An empty plane renders as an empty image and warps to the fill.
    #[test]
    fn empty_planes() {
        for (pw, ph) in [(0, 0), (0, 5), (5, 0)] {
            let plane = Plane { width: pw, height: ph, fill: [9, 8, 7], tiles: Vec::new() };
            let r = plane.render();
            assert_eq!((r.width, r.height, r.data.len()), (pw, ph, 0));
            for m in [Affine::translate(1.0, 2.0), Affine::rotate(10.0)] {
                assert_eq!(warp_plane(&plane, &m, 3, 2), Image::new(3, 2, [9, 8, 7]));
            }
        }
    }

    /// A step so large that `du · (width - 1)` overflows `i64` while every running position the general path adds up
    /// fits: the row path gives the general path's bytes, without overflowing.
    #[test]
    fn a_huge_step_does_not_overflow() {
        let src = pattern(1, 1);
        let m = Affine::scale(2f64.powi(-39), 1.0).then(&Affine::translate(128.5, 0.0));
        let inv = m.inverse().unwrap();
        let mut general = vec![0u8; 257 * 3];
        warp_general(&src, &inv, 257, [114; 3], &mut general);
        assert_eq!(warp_affine(&src, &m, 257, 1, [114; 3]).data, general);
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
