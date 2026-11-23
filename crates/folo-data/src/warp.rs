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
use crate::simd::bilinear::{
    vblend_rgb_to_rgba32, vshift_rgb_to_rgba32, warp_general_interior_to_rgb, warp_hinterp_interior_to_rgb,
};
use crate::simd::hsv_fused::{HsvTables, rgb_row_to_planar_fused, warp_interior_to_planar_fused};
use crate::simd::{PlanarDst, PlanarRow, assert_image_data, for_planar_bands};

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
    pub(crate) fn row(&self, y: i64, x0: i64, out: &mut [u8]) {
        fill_rgb(out, self.fill);
        self.overlay_tiles(y, x0, out);
    }

    /// Borrows a contiguous slice straight from a single tile when one covers all of `[x0, x0 + len / 3)` at row `y`.
    #[inline]
    pub(crate) fn row_ref_direct(&self, y: i64, x0: i64, len: usize) -> Option<&'a [u8]> {
        if len == 0 {
            return Some(&[]);
        }
        let x1 = x0 + (len / 3) as i64;
        for t in &self.tiles {
            let [tx0, ty0, tx1, ty1] = t.dst;
            if y >= ty0 && y < ty1 && tx0 <= x0 && tx1 >= x1 {
                let img = t.img;
                let (sx, sy) = ((x0 - tx0 + t.src[0]) as usize, (y - ty0 + t.src[1]) as usize);
                return Some(&img.data[sy * img.stride() + sx * 3..][..len]);
            }
        }
        None
    }

    /// Copies the portions of all tiles intersecting row `y`, columns `x0 .. x0 + out.len() / 3`, into `out`.
    #[inline]
    fn overlay_tiles(&self, y: i64, x0: i64, out: &mut [u8]) {
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
        fill_rgb(&mut img.data, self.fill);
        let stride = img.stride();
        for (y, row) in img.data.chunks_exact_mut(stride).enumerate() {
            let y = y as i64;
            if self.tiles.iter().any(|t| (t.dst[1]..t.dst[3]).contains(&y)) {
                self.overlay_tiles(y, 0, row);
            }
        }
        img
    }
}

/// Fills an interleaved RGB buffer with one colour (`memset` when all three channels match).
#[inline]
fn fill_rgb(out: &mut [u8], fill: [u8; 3]) {
    if fill[0] == fill[1] && fill[1] == fill[2] {
        out.fill(fill[0]);
    } else {
        for px in out.as_chunks_mut::<3>().0.iter_mut() {
            *px = fill;
        }
    }
}

/// Contiguous sub-range `[x0, x1) ⊆ [0, width)` where `lo <= f0 + x * df < hi_excl`.
#[inline]
fn linear_range(f0: i128, df: i128, lo: i128, hi_excl: i128, width: usize) -> (usize, usize) {
    if lo >= hi_excl || width == 0 {
        return (0, 0);
    }
    if df == 0 {
        return if f0 >= lo && f0 < hi_excl { (0, width) } else { (0, 0) };
    }
    let w = width as i128;
    let (x0, x1) = if df > 0 {
        let start = if f0 >= lo { 0 } else { (lo - f0 + df - 1) / df };
        let end = if f0 >= hi_excl { 0 } else { (hi_excl - f0 + df - 1) / df };
        (start, end)
    } else {
        let ad = -df;
        let start = if f0 < hi_excl { 0 } else { (f0 - hi_excl) / ad + 1 };
        let end = if f0 < lo { 0 } else { (f0 - lo) / ad + 1 };
        (start, end)
    };
    let a = x0.clamp(0, w) as usize;
    let b = x1.clamp(a as i128, w) as usize;
    (a, b)
}

/// Warps `src` by `m` (source → destination) into a `width x height` image; uncovered pixels get `border`.
pub fn warp_affine(src: &Image, m: &Affine, width: u32, height: u32, border: [u8; 3]) -> Image {
    warp_plane(&Plane::image(src, border), m, width, height)
}

/// Warps `plane` by `m` into a `width x height` image: bit for bit [`warp_affine`] of [`Plane::render`] with the
/// plane's fill as the border, without rendering the plane when the source row does not change along a destination
/// row.
pub(crate) fn warp_plane(plane: &Plane, m: &Affine, width: u32, height: u32) -> Image {
    for t in &plane.tiles {
        assert_image_data(t.img);
    }
    if width == 0 || height == 0 {
        return Image { width, height, data: Vec::new() };
    }
    let mut out = crate::buffers::scratch(width as usize * height as usize * 3);
    let border = plane.fill;
    let Some(inv) = m.inverse() else {
        // Degenerate transform: nothing of the source is visible.
        let mut img = Image { width, height, data: out };
        img.fill_rect(0, 0, width as i64, height as i64, border);
        return img;
    };
    if (inv.0[3] * (1i64 << FRAC) as f64).round() as i64 == 0
        && rows_fit(&inv, width as usize, height as usize)
    {
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

/// Warps `plane` by `m`, applies optional HSV jitter and flips, and writes straight into planar `dst` (`u8` or
/// `f32 / 255.0`) in a single L1-resident pass without allocating an intermediate HWC image. Bit for bit identical to
/// `warp_plane` followed by `apply_hsv`, `flip_ud`, `flip_lr`, and planar conversion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_plane_fused(
    plane: &Plane,
    m: &Affine,
    width: u32,
    height: u32,
    hsv: Option<&HsvTables>,
    flip_ud: bool,
    flip_lr: bool,
    dst: &mut PlanarDst<'_>,
) {
    for t in &plane.tiles {
        assert_image_data(t.img);
    }
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return;
    }
    let fill_post = hsv.map_or(plane.fill, |t| t.apply_pixel(plane.fill));
    if m.is_identity() && (plane.width, plane.height) == (width, height) {
        for_planar_bands(dst, w, h, |dy0, band| {
            let mut scratch = Vec::new();
            let rows = band.rows(w);
            for bi in 0..rows {
                let src_y = if flip_ud { h - 1 - (dy0 + bi) } else { dy0 + bi } as i64;
                let mut drow = band.row_slices(bi, w);
                if let Some(direct) = plane.row_ref_direct(src_y, 0, w * 3) {
                    rgb_row_to_planar_fused(direct, hsv, flip_lr, &mut drow);
                } else {
                    if scratch.len() < w * 3 {
                        scratch.resize(w * 3, 0);
                    }
                    plane.row(src_y, 0, &mut scratch[..w * 3]);
                    rgb_row_to_planar_fused(&scratch[..w * 3], hsv, flip_lr, &mut drow);
                }
            }
        });
        return;
    }
    let Some(inv) = m.inverse() else {
        for dy in 0..h {
            dst.row_slices(dy, w).fill_range(0, w, fill_post);
        }
        return;
    };
    if (inv.0[3] * (1i64 << FRAC) as f64).round() as i64 == 0 && rows_fit(&inv, w, h) {
        warp_rows_fused(plane, &inv, w, h, fill_post, hsv, flip_ud, flip_lr, dst);
    } else {
        match plane.tiles.as_slice() {
            [t] if t.dst == [0, 0, plane.width as i64, plane.height as i64]
                && t.src == [0, 0]
                && (t.img.width, t.img.height) == (plane.width, plane.height) =>
            {
                warp_general_fused(t.img, &inv, w, h, plane.fill, hsv, flip_ud, flip_lr, dst);
            }
            _ => {
                let rendered = plane.render();
                warp_general_fused(&rendered, &inv, w, h, plane.fill, hsv, flip_ud, flip_lr, dst);
                crate::buffers::recycle(rendered.data);
            }
        }
    }
}

/// Scalar `warp_general` pixel loop on `row[..count * 3]` starting at `(fu, fv)`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn warp_general_scalar_span(
    sdata: &[u8],
    sw: i64,
    sh: i64,
    sstride: usize,
    mut fu: i64,
    mut fv: i64,
    du: i64,
    dv: i64,
    border: [u8; 3],
    row: &mut [u8],
) {
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
}

/// Computes one destination row `y` of `warp_general`, taking the AVX2 interior span when safe.
#[allow(clippy::too_many_arguments)]
fn warp_general_row(
    sdata: &[u8],
    sw: i64,
    sh: i64,
    sstride: usize,
    inv: &Affine,
    y: usize,
    width: usize,
    border: [u8; 3],
    row: &mut [u8],
) {
    let [a, b, c, d, e, f] = inv.0;
    let fone = (1i64 << FRAC) as f64;
    let yc = y as f64 + 0.5;
    let fu0 = ((a * 0.5 + b * yc + c - 0.5) * fone).round() as i64;
    let fv0 = ((d * 0.5 + e * yc + f - 0.5) * fone).round() as i64;
    let du = (a * fone).round() as i64;
    let dv = (d * fone).round() as i64;

    if sw >= 2
        && sh >= 3
        && width >= 6
        && i64::try_from(fu0 as i128 + du as i128 * (width - 1) as i128).is_ok()
        && i64::try_from(fv0 as i128 + dv as i128 * (width - 1) as i128).is_ok()
    {
        let (ux0, ux1) = linear_range(fu0 as i128, du as i128, 0, (sw as i128 - 1) << FRAC, width);
        let (vy0, vy1) = linear_range(fv0 as i128, dv as i128, 0, (sh as i128 - 2) << FRAC, width);
        let x0 = ux0.max(vy0);
        let x1 = ux1.min(vy1);
        if x1 >= x0 + 6 {
            if x0 > 0 {
                warp_general_scalar_span(
                    sdata,
                    sw,
                    sh,
                    sstride,
                    fu0,
                    fv0,
                    du,
                    dv,
                    border,
                    &mut row[..x0 * 3],
                );
            }
            let fu_x0 = (fu0 as i128 + x0 as i128 * du as i128) as i64;
            let fv_x0 = (fv0 as i128 + x0 as i128 * dv as i128) as i64;
            warp_general_interior_to_rgb(
                sdata,
                sw as usize,
                sh as usize,
                sstride,
                fu_x0,
                fv_x0,
                du,
                dv,
                &mut row[x0 * 3..x1 * 3],
            );
            if x1 < width {
                let fu_x1 = (fu0 as i128 + x1 as i128 * du as i128) as i64;
                let fv_x1 = (fv0 as i128 + x1 as i128 * dv as i128) as i64;
                warp_general_scalar_span(
                    sdata,
                    sw,
                    sh,
                    sstride,
                    fu_x1,
                    fv_x1,
                    du,
                    dv,
                    border,
                    &mut row[x1 * 3..],
                );
            }
            return;
        }
    }
    warp_general_scalar_span(sdata, sw, sh, sstride, fu0, fv0, du, dv, border, row);
}

/// The warp of `src` by the inverse matrix `inv` into `out` (rows of `width` pixels), any matrix: each pixel steps
/// its own source position.
fn warp_general(src: &Image, inv: &Affine, width: usize, border: [u8; 3], out: &mut [u8]) {
    assert_image_data(src);
    let (sw, sh) = (src.width as i64, src.height as i64);
    let sstride = src.stride();
    let sdata = &src.data;
    for_rows(out, width * 3, |y, row| {
        warp_general_row(sdata, sw, sh, sstride, inv, y, width, border, row);
    });
}

#[allow(clippy::too_many_arguments)]
fn warp_general_fused(
    src: &Image,
    inv: &Affine,
    width: usize,
    height: usize,
    border: [u8; 3],
    hsv: Option<&HsvTables>,
    flip_ud: bool,
    flip_lr: bool,
    dst: &mut PlanarDst<'_>,
) {
    assert_image_data(src);
    let (sw, sh) = (src.width as i64, src.height as i64);
    let sstride = src.stride();
    let sdata = &src.data;
    for_planar_bands(dst, width, height, |dy0, band| {
        let mut row_rgb = vec![0u8; width * 3];
        let rows = band.rows(width);
        for bi in 0..rows {
            let src_y = if flip_ud { height - 1 - (dy0 + bi) } else { dy0 + bi };
            warp_general_row(sdata, sw, sh, sstride, inv, src_y, width, border, &mut row_rgb);
            let mut drow = band.row_slices(bi, width);
            rgb_row_to_planar_fused(&row_rgb, hsv, flip_lr, &mut drow);
        }
    });
}

/// Reusable per-band scratch for [`warp_rows`] and [`warp_rows_fused`]: caches copied multi-tile rows and the
/// vertically blended `[R, G, B, 0]` `u32` buffer across consecutive destination rows.
#[derive(Default)]
struct PlaneRowCache {
    top: Vec<u8>,
    bot: Vec<u8>,
    v: Vec<u32>,
    top_key: Option<(i64, i64, usize)>,
    bot_key: Option<(i64, i64, usize)>,
    v_key: Option<(i64, u32, i64, usize)>,
}

impl PlaneRowCache {
    #[inline]
    fn prepare(&mut self, plane: &Plane, iy: i64, wy: u32, lo: i64, cols: usize) -> &[u32] {
        let key = (iy, wy, lo, cols);
        if self.v_key == Some(key) {
            return &self.v[..cols * 4];
        }
        let n = cols * 3;
        if self.v.len() < cols * 4 {
            self.v.resize(cols * 4, 0);
        }
        let top_direct = plane.row_ref_direct(iy, lo, n);
        if top_direct.is_none() && self.top_key != Some((iy, lo, n)) {
            if self.bot_key == Some((iy, lo, n)) {
                std::mem::swap(&mut self.top, &mut self.bot);
                std::mem::swap(&mut self.top_key, &mut self.bot_key);
            } else {
                if self.top.len() < n {
                    self.top.resize(n, 0);
                }
                plane.row(iy, lo, &mut self.top[..n]);
                self.top_key = Some((iy, lo, n));
            }
        }
        if wy == 0 {
            let top = top_direct.unwrap_or_else(|| &self.top[..n]);
            vshift_rgb_to_rgba32(top, &mut self.v[..cols * 4]);
            self.v_key = Some(key);
            return &self.v[..cols * 4];
        }
        let bot_direct = plane.row_ref_direct(iy + 1, lo, n);
        if bot_direct.is_none() && self.bot_key != Some((iy + 1, lo, n)) {
            if top_direct.is_some() && self.top_key == Some((iy + 1, lo, n)) {
                std::mem::swap(&mut self.top, &mut self.bot);
                std::mem::swap(&mut self.top_key, &mut self.bot_key);
            } else {
                if self.bot.len() < n {
                    self.bot.resize(n, 0);
                }
                plane.row(iy + 1, lo, &mut self.bot[..n]);
                self.bot_key = Some((iy + 1, lo, n));
            }
        }
        let top = top_direct.unwrap_or_else(|| &self.top[..n]);
        let bot = bot_direct.unwrap_or_else(|| &self.bot[..n]);
        vblend_rgb_to_rgba32(top, bot, wy, &mut self.v[..cols * 4]);
        self.v_key = Some(key);
        &self.v[..cols * 4]
    }
}

/// Whether [`warp_rows`] may run (both paths give the same bytes; this picks where the rows are the faster and safe
/// one): the step is at most 4 source pixels per output pixel (a row blends the source span it crosses, so a much
/// smaller output would blend far more than it samples), and every horizontal position it samples, `fu0 + k·du`
/// for `k < width` in each of `height` rows, fits in `i64`. Then the general path's running sums reach them without
/// wrapping; otherwise the general path runs, whose `i64` sums wrap in a release build exactly as they always did.
fn rows_fit(inv: &Affine, width: usize, height: usize) -> bool {
    let [a, b, c, ..] = inv.0;
    let fone = (1i64 << FRAC) as f64;
    let du = (a * fone).round() as i64 as i128;
    if du.abs() > 4 << FRAC {
        return false;
    }
    (0..height).all(|y| {
        let yc = y as f64 + 0.5;
        let fu0 = ((a * 0.5 + b * yc + c - 0.5) * fone).round() as i64 as i128;
        i64::try_from(fu0 + du * (width.max(1) as i128 - 1)).is_ok()
    })
}

/// The warp of `plane` by the inverse matrix `inv` into `out` (rows of `width` pixels) when the source row does not
/// change along a destination row (`inv`'s vertical step per pixel rounds to zero). Per destination row, the two
/// source rows `iy`, `iy + 1` over the columns the row reaches are blended into `v = top·(1 − wy) + bottom·wy`
/// (`u32`, padded `[R, G, B, 0]` for AVX2), then each interior pixel is
/// `(v[ix]·(1 − wx) + v[ix + 1]·wx + 2^21) >> 22`: the four products and the one rounding of the general path's
/// `(t·(1 − wy) + u·wy + 2^21) >> 22`, so the same bytes.
fn warp_rows(plane: &Plane, inv: &Affine, width: usize, out: &mut [u8]) {
    let [a, b, c, d, e, f] = inv.0;
    let fone = (1i64 << FRAC) as f64;
    let du = (a * fone).round() as i64;
    let (pw, ph) = (plane.width as i64, plane.height as i64);
    let fill = plane.fill;
    for_bands(out, width * 3, |y0, band| {
        let mut cache = PlaneRowCache::default();
        for (i, row) in band.chunks_exact_mut(width * 3).enumerate() {
            let yc = (y0 + i) as f64 + 0.5;
            let fu0 = ((a * 0.5 + b * yc + c - 0.5) * fone).round() as i64;
            let fv = ((d * 0.5 + e * yc + f - 0.5) * fone).round() as i64;
            let iy = fv >> FRAC;
            let wy = ((fv >> (FRAC - WBITS)) & (WONE as i64 - 1)) as u32;
            let last_fu = fu0 as i128 + du as i128 * (width as i128 - 1);
            let (ix_a, ix_b) = (fu0 >> FRAC, (last_fu >> FRAC) as i64);
            let lo = ix_a.min(ix_b).max(-1);
            let hi = (ix_a.max(ix_b) + 1).min(pw);
            if iy < -1 || iy >= ph || hi <= lo {
                fill_rgb(row, fill);
                continue;
            }
            let (x_start, x_end) =
                linear_range(fu0 as i128, du as i128, (lo as i128) << FRAC, (hi as i128) << FRAC, width);
            if x_start >= x_end {
                fill_rgb(row, fill);
                continue;
            }
            let cols = (hi - lo + 1) as usize;
            let v = cache.prepare(plane, iy, wy, lo, cols);
            if x_start > 0 {
                fill_rgb(&mut row[..x_start * 3], fill);
            }
            let rel_fu0 = (fu0 as i128 + du as i128 * x_start as i128 - ((lo as i128) << FRAC)) as i64;
            warp_hinterp_interior_to_rgb(v, rel_fu0, du, &mut row[x_start * 3..x_end * 3]);
            if x_end < width {
                fill_rgb(&mut row[x_end * 3..], fill);
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn warp_rows_fused(
    plane: &Plane,
    inv: &Affine,
    width: usize,
    height: usize,
    fill_post: [u8; 3],
    hsv: Option<&HsvTables>,
    flip_ud: bool,
    flip_lr: bool,
    dst: &mut PlanarDst<'_>,
) {
    let [a, b, c, d, e, f] = inv.0;
    let fone = (1i64 << FRAC) as f64;
    let du = (a * fone).round() as i64;
    let (pw, ph) = (plane.width as i64, plane.height as i64);
    for_planar_bands(dst, width, height, |dy0, band| {
        let mut cache = PlaneRowCache::default();
        let rows = band.rows(width);
        for bi in 0..rows {
            let src_y = if flip_ud { height - 1 - (dy0 + bi) } else { dy0 + bi };
            let yc = src_y as f64 + 0.5;
            let fu0 = ((a * 0.5 + b * yc + c - 0.5) * fone).round() as i64;
            let fv = ((d * 0.5 + e * yc + f - 0.5) * fone).round() as i64;
            let iy = fv >> FRAC;
            let wy = ((fv >> (FRAC - WBITS)) & (WONE as i64 - 1)) as u32;
            let last_fu = fu0 as i128 + du as i128 * (width as i128 - 1);
            let (ix_a, ix_b) = (fu0 >> FRAC, (last_fu >> FRAC) as i64);
            let lo = ix_a.min(ix_b).max(-1);
            let hi = (ix_a.max(ix_b) + 1).min(pw);
            let mut drow: PlanarRow<'_> = band.row_slices(bi, width);
            if iy < -1 || iy >= ph || hi <= lo {
                drow.fill_range(0, width, fill_post);
                continue;
            }
            let (x_start, x_end) =
                linear_range(fu0 as i128, du as i128, (lo as i128) << FRAC, (hi as i128) << FRAC, width);
            if x_start >= x_end {
                drow.fill_range(0, width, fill_post);
                continue;
            }
            let cols = (hi - lo + 1) as usize;
            let v = cache.prepare(plane, iy, wy, lo, cols);
            if flip_lr {
                drow.fill_range(0, width - x_end, fill_post);
                drow.fill_range(width - x_start, width, fill_post);
            } else {
                drow.fill_range(0, x_start, fill_post);
                drow.fill_range(x_end, width, fill_post);
            }
            let rel_fu0 = (fu0 as i128 + du as i128 * x_start as i128 - ((lo as i128) << FRAC)) as i64;
            warp_interior_to_planar_fused(v, rel_fu0, du, width, x_start, x_end, hsv, flip_lr, &mut drow);
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

    /// An empty output is returned at once, whatever the other side's size (no per-row work, no scratch).
    #[test]
    fn empty_outputs() {
        let src = pattern(1, 1);
        for (w, h) in [(0, u32::MAX), (u32::MAX, 0), (0, 0)] {
            let out = warp_affine(&src, &Affine::IDENTITY, w, h, [1; 3]);
            assert_eq!((out.width, out.height, out.data.len()), (w, h, 0));
        }
        let wide = Image { width: 1 << 30, height: 0, data: Vec::new() };
        assert!(warp_affine(&wide, &Affine::IDENTITY, 1, 0, [1; 3]).data.is_empty());
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
    /// fits: the warp gives the general path's bytes, without overflowing.
    #[test]
    fn a_huge_step_does_not_overflow() {
        let src = pattern(1, 1);
        let m = Affine::scale(2f64.powi(-39), 1.0).then(&Affine::translate(128.5, 0.0));
        let inv = m.inverse().unwrap();
        let mut general = vec![0u8; 257 * 3];
        warp_general(&src, &inv, 257, [114; 3], &mut general);
        assert_eq!(warp_affine(&src, &m, 257, 1, [114; 3]).data, general);
    }

    /// A step whose sampled positions leave `i64` takes the general path, whose sums wrap (release build) as before:
    /// here column 511's position wraps round to -0.5, half border, half image.
    #[test]
    fn positions_beyond_i64_take_the_general_path() {
        let m = Affine::scale(2f64.powi(-39), 1.0).then(&Affine::translate(-0.5, 0.0));
        let inv = m.inverse().unwrap();
        assert!(!rows_fit(&inv, 512, 1));
        // Steps of more than 4 source pixels per output pixel take the general path too.
        assert!(!rows_fit(&inv, 1, 1));
        assert!(!rows_fit(&Affine::scale(0.2, 1.0).inverse().unwrap(), 10, 1));
        assert!(rows_fit(&Affine::scale(0.25, 1.0).inverse().unwrap(), 10, 1));
        // The sums overflow, which a build with overflow checks stops at; compare the bytes where they wrap.
        #[cfg(not(debug_assertions))]
        {
            let src = Image::new(1, 1, [0; 3]);
            let mut general = vec![0u8; 512 * 3];
            warp_general(&src, &inv, 512, [114; 3], &mut general);
            assert_eq!(&general[511 * 3..], &[57; 3]);
            assert_eq!(warp_affine(&src, &m, 512, 1, [114; 3]).data, general);
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
