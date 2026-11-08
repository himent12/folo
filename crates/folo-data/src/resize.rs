//! Fast bilinear resize for 8-bit RGB images.
//!
//! Pixel centres are aligned (`src = (dst + 0.5) * scale - 0.5`), so pixel *edges* map linearly and box coordinates
//! scale exactly by `dst / src`. Weights are 11-bit fixed point; rows are processed in parallel. When shrinking by 2x
//! or more an integer box filter runs first so large downscales do not alias.

use crate::image::Image;
use crate::par::for_rows;

const BITS: u32 = 11;
const ONE: u32 = 1 << BITS;

/// Per destination coordinate: the two source indices (already multiplied by the element step) and the weight of
/// the second one.
struct Taps {
    i0: Vec<u32>,
    i1: Vec<u32>,
    w: Vec<u32>,
}

/// Builds taps for `dst` outputs over `src` inputs, where the source has `src_extent` real pixels of size `cell`
/// (`cell > 1` after a box prefilter whose last cell may be partial).
fn taps(dst: u32, src: u32, src_extent: f64, step: u32) -> Taps {
    let scale = src_extent / dst as f64;
    let mut t = Taps { i0: Vec::with_capacity(dst as usize), i1: Vec::new(), w: Vec::new() };
    t.i1.reserve(dst as usize);
    t.w.reserve(dst as usize);
    let max = (src - 1) as f64;
    for d in 0..dst {
        let s = ((d as f64 + 0.5) * scale - 0.5).clamp(0.0, max);
        let i0 = s.floor() as u32;
        let i1 = (i0 + 1).min(src - 1);
        let w = (((s - i0 as f64) * ONE as f64).round() as u32).min(ONE);
        t.i0.push(i0 * step);
        t.i1.push(i1 * step);
        t.w.push(w);
    }
    t
}

/// Resizes `src` to `width x height` with bilinear filtering.
pub fn resize(src: &Image, width: u32, height: u32) -> Image {
    let mut dst = vec![0u8; width as usize * height as usize * 3];
    resize_into(src, &mut dst, width as usize * 3, 0, width, height);
    Image { width, height, data: dst }
}

/// Resizes `src` into a `width x height` window of a larger RGB buffer: row `y` of the window starts at
/// `dst[y * dst_stride + x_offset * 3]`. `dst` must hold at least `height` rows of `dst_stride` bytes.
pub fn resize_into(src: &Image, dst: &mut [u8], dst_stride: usize, x_offset: usize, width: u32, height: u32) {
    assert!(width > 0 && height > 0 && src.width > 0 && src.height > 0, "resize to or from an empty image");
    assert!(x_offset * 3 + width as usize * 3 <= dst_stride, "resize window exceeds the row stride");
    let dst = &mut dst[..dst_stride * height as usize];
    if src.width == width && src.height == height {
        let row = width as usize * 3;
        for_rows(dst, dst_stride, |y, out| {
            out[x_offset * 3..x_offset * 3 + row].copy_from_slice(&src.data[y * row..(y + 1) * row]);
        });
        return;
    }
    // Integer box prefilter for large reductions.
    let fx = (src.width / width).max(1);
    let fy = (src.height / height).max(1);
    let filtered;
    let (s, ext_w, ext_h) = if fx >= 2 || fy >= 2 {
        filtered = box_downsample(src, fx, fy);
        (&filtered, src.width as f64 / fx as f64, src.height as f64 / fy as f64)
    } else {
        (src, src.width as f64, src.height as f64)
    };
    let xt = taps(width, s.width, ext_w, 3);
    let yt = taps(height, s.height, ext_h, 1);
    let sstride = s.stride();
    let sdata = &s.data;
    for_rows(dst, dst_stride, |y, out| {
        let top = &sdata[yt.i0[y] as usize * sstride..][..sstride];
        let bot = &sdata[yt.i1[y] as usize * sstride..][..sstride];
        let wy = yt.w[y];
        let iwy = ONE - wy;
        let out = &mut out[x_offset * 3..x_offset * 3 + width as usize * 3];
        for (x, px) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (a, b, wx) = (xt.i0[x] as usize, xt.i1[x] as usize, xt.w[x]);
            let iwx = ONE - wx;
            for c in 0..3 {
                let t = top[a + c] as u32 * iwx + top[b + c] as u32 * wx;
                let u = bot[a + c] as u32 * iwx + bot[b + c] as u32 * wx;
                px[c] = ((t * iwy + u * wy + (1 << (2 * BITS - 1))) >> (2 * BITS)) as u8;
            }
        }
    });
}

/// Averages `fx x fy` blocks. The last block in each direction may be partial; it averages what is there.
fn box_downsample(src: &Image, fx: u32, fy: u32) -> Image {
    let (w, h) = (src.width as usize, src.height as usize);
    let (fx, fy) = (fx as usize, fy as usize);
    let ow = w.div_ceil(fx);
    let oh = h.div_ceil(fy);
    let mut out = vec![0u8; ow * oh * 3];
    let stride = w * 3;
    for_rows(&mut out, ow * 3, |oy, row| {
        let y0 = oy * fy;
        let y1 = (y0 + fy).min(h);
        let mut acc = vec![0u32; ow * 3];
        for y in y0..y1 {
            let srow = &src.data[y * stride..(y + 1) * stride];
            for ox in 0..ow {
                let x0 = ox * fx;
                let x1 = (x0 + fx).min(w);
                let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
                for p in srow[x0 * 3..x1 * 3].as_chunks::<3>().0.iter() {
                    r += p[0] as u32;
                    g += p[1] as u32;
                    b += p[2] as u32;
                }
                acc[ox * 3] += r;
                acc[ox * 3 + 1] += g;
                acc[ox * 3 + 2] += b;
            }
        }
        for ox in 0..ow {
            let n = (((ox * fx + fx).min(w) - ox * fx) * (y1 - y0)) as u32;
            for c in 0..3 {
                row[ox * 3 + c] = ((acc[ox * 3 + c] + n / 2) / n) as u8;
            }
        }
    });
    Image { width: ow as u32, height: oh as u32, data: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_constant() {
        let mut img = Image::new(7, 5, [10, 20, 30]);
        img.set_pixel(3, 2, [200, 100, 0]);
        assert_eq!(resize(&img, 7, 5), img);
        let flat = Image::new(37, 23, [9, 99, 199]);
        for (w, h) in [(80, 40), (5, 3), (36, 23), (1, 1), (200, 7)] {
            let r = resize(&flat, w, h);
            assert!(r.data.chunks(3).all(|p| p == [9, 99, 199]), "{w}x{h}");
        }
    }

    #[test]
    fn upscale_2x_interpolates() {
        // Two pixels 0 and 200 -> 4 pixels: centres at -0.25, 0.25, 0.75, 1.25 in source space.
        let img = Image::from_raw(2, 1, vec![0, 0, 0, 200, 200, 200]).unwrap();
        let r = resize(&img, 4, 1);
        let v: Vec<u8> = r.data.chunks(3).map(|p| p[0]).collect();
        assert_eq!(v, vec![0, 50, 150, 200]);
    }

    #[test]
    fn large_downscale_averages() {
        // A fine checkerboard averages to mid grey instead of aliasing to black or white.
        let mut img = Image::new(64, 64, [0, 0, 0]);
        for y in 0..64 {
            for x in 0..64 {
                if (x + y) % 2 == 0 {
                    img.set_pixel(x, y, [255, 255, 255]);
                }
            }
        }
        let r = resize(&img, 8, 8);
        assert!(r.data.iter().all(|&v| (120..=135).contains(&v)), "{:?}", &r.data[..12]);
    }

    #[test]
    fn parallel_matches_serial() {
        let mut img = Image::new(301, 211, [0, 0, 0]);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        let a = resize(&img, 640, 480);
        let b = crate::par::serial(|| resize(&img, 640, 480));
        assert_eq!(a, b);
    }
}
