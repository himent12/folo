//! Direct `u8` RGB0-to-C4 AVX-VNNI stem (`QStem` + [`stem_u8`]): input pixels are already exact `u8` codes
//! (`scale = 1/255`, `zero = 0`), so a 3-channel stem runs directly on packed `[R, G, B, 0]` `u32` words in
//! L1-resident row strips (`9 × ow × 4` bytes ≈ 12 KiB for 640² s2), avoiding the `f32` stem write + scalar
//! requantize pass.

use rayon::prelude::*;

use super::conv::{MAX_PAD, MAX_STRIDE, QOut, out_size, quantize_weights};
use super::kernel::{Epi, MR, NR, Out, Tile, kernel};

/// Scale of `u8` image pixels normalised to `[0, 1]`.
pub const PIXEL_SCALE: f32 = 1.0 / 255.0;

/// Packed weights of a 3-channel `u8` stem (`[o, 3, k, k]`, `o` a multiple of 4): channel 3 of every packed
/// `i32` word is 0 (matching the `0` pad byte of `[R, G, B, 0]` pixels), and `1 / 255` is folded into `scale`.
#[derive(Debug, Clone)]
pub struct QStem {
    /// Output channels (a multiple of 4).
    pub(super) o: usize,
    /// Kernel size `k × k`.
    pub(super) k: usize,
    /// Stride.
    pub(super) stride: usize,
    /// Zero padding on each side.
    pub(super) pad: usize,
    /// SiLU in the epilogue (else identity).
    pub(super) silu: bool,
    /// K steps: `k * k`.
    ks: usize,
    /// `[o / 4][ks][MR]` words of `[w_r, w_g, w_b, 0]` s8 weights.
    a: Vec<i32>,
    /// Per output channel: weight scale with `1 / 255` folded in.
    scale: Vec<f32>,
    bias: Vec<f32>,
}

impl QStem {
    /// Whether [`QStem::new`] takes this 3-channel geometry.
    pub fn supports((o, c, k): (usize, usize, usize), stride: usize, pad: usize) -> bool {
        c == 3
            && o > 0
            && o.is_multiple_of(4)
            && (1..=15).contains(&k)
            && (1..=MAX_STRIDE).contains(&stride)
            && pad <= MAX_PAD
    }

    /// Quantise and pack `[o, 3, k, k]` weights `w` with `bias` for `u8` `[R, G, B, 0]` inputs (`scale = 1/255`,
    /// `zero = 0`).
    pub fn new(
        w: &[f32],
        bias: &[f32],
        (o, k): (usize, usize),
        stride: usize,
        pad: usize,
        silu: bool,
    ) -> Self {
        assert!(
            QStem::supports((o, 3, k), stride, pad),
            "QStem: unsupported geometry ({o}, 3, {k}) s{stride} p{pad}"
        );
        let taps = k * k;
        assert_eq!(w.len(), o * 3 * taps, "QStem: weight length");
        assert_eq!(bias.len(), o, "QStem: bias length");
        let folded: Vec<f32> = w.iter().map(|&v| v * PIXEL_SCALE).collect();
        let (q, scale) = quantize_weights(&folded, o);
        let ks = taps;
        let mut a = vec![0i32; o * ks];
        for oc in 0..o {
            let row = &q[oc * 3 * taps..(oc + 1) * 3 * taps];
            for t in 0..taps {
                let bytes = [row[t] as u8, row[taps + t] as u8, row[2 * taps + t] as u8, 0u8];
                a[(oc / 4 * ks + t) * MR + oc % 4] = i32::from_le_bytes(bytes);
            }
        }
        QStem { o, k, stride, pad, silu, ks, a, scale, bias: bias.to_vec() }
    }

    /// Output channels.
    pub fn out_channels(&self) -> usize {
        self.o
    }

    /// Output size of an `h × w` input.
    pub fn out_hw(&self, h: usize, w: usize) -> (usize, usize) {
        out_size(h, w, self.k, self.stride, self.pad)
            .unwrap_or_else(|| panic!("QStem: {h}×{w} input under a {}×{} kernel", self.k, self.k))
    }
}

/// Convert a single `f32` pixel in `[0, 1]` to `u8` (`round_ties_even(clamp(x, 0, 1) * 255)`).
#[inline(always)]
fn f32_to_pixel_u8(x: f32) -> u8 {
    let v = super::silu::minps(super::silu::maxps(x * 255.0, 0.0), 255.0);
    v.round_ties_even() as i32 as u8
}

/// Pack `[n, 3, h, w]` `f32` inputs in `[0, 1]` into `[n, h, w]` `u32` words `[R, G, B, 0]`, bit-exact between
/// scalar and AVX2.
pub fn pack_f32_to_rgb0_u8(x: &[f32], n: usize, h: usize, w: usize, out: &mut [u32]) {
    let hw = h * w;
    assert!(x.len() >= n * 3 * hw && out.len() >= n * hw, "pack_f32_to_rgb0_u8: buffer too short");
    if n * hw == 0 {
        return;
    }
    const BLOCK: usize = 4096;
    out[..n * hw].par_chunks_mut(hw).enumerate().for_each(|(img, dst)| {
        let base = img * 3 * hw;
        let (rp, gp, bp) =
            (&x[base..base + hw], &x[base + hw..base + 2 * hw], &x[base + 2 * hw..base + 3 * hw]);
        dst.par_chunks_mut(BLOCK).enumerate().for_each(|(b, o)| {
            let p0 = b * BLOCK;
            pack_rgb0_slice(&rp[p0..p0 + o.len()], &gp[p0..p0 + o.len()], &bp[p0..p0 + o.len()], o);
        });
    });
}

fn pack_rgb0_slice(r: &[f32], g: &[f32], b: &[f32], out: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    if super::kernel::uses_avx2() {
        // SAFETY: AVX2 and FMA were detected; slices have identical lengths.
        return unsafe { pack_rgb0_slice_avx2(r, g, b, out) };
    }
    for i in 0..out.len() {
        out[i] = u32::from_le_bytes([f32_to_pixel_u8(r[i]), f32_to_pixel_u8(g[i]), f32_to_pixel_u8(b[i]), 0]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_rgb0_slice_avx2(r: &[f32], g: &[f32], b: &[f32], out: &mut [u32]) {
    use std::arch::x86_64::*;
    let len = out.len();
    let mut i = 0;
    let scale = _mm256_set1_ps(255.0);
    let zero = _mm256_setzero_ps();
    let z_i = _mm256_setzero_si256();
    let transpose = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7,
        11, 15,
    );
    // SAFETY: `i + 8 <= len` and all four slices have length `len`.
    unsafe {
        while i + 8 <= len {
            let vr = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(r.as_ptr().add(i)), scale), zero),
                scale,
            ));
            let vg = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(g.as_ptr().add(i)), scale), zero),
                scale,
            ));
            let vb = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(b.as_ptr().add(i)), scale), zero),
                scale,
            ));
            let p01 = _mm256_packs_epi32(vr, vg);
            let p23 = _mm256_packs_epi32(vb, z_i);
            let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), transpose);
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, bytes);
            i += 8;
        }
    }
    for j in i..len {
        out[j] = u32::from_le_bytes([f32_to_pixel_u8(r[j]), f32_to_pixel_u8(g[j]), f32_to_pixel_u8(b[j]), 0]);
    }
}

/// An output pointer shared by tasks writing disjoint rows.
struct OutPtr<T>(*mut T);
unsafe impl<T> Sync for OutPtr<T> {}
unsafe impl<T> Send for OutPtr<T> {}
impl<T> OutPtr<T> {
    fn at(&self, i: usize) -> *mut T {
        // SAFETY: callers stay strictly inside their own disjoint output slice.
        unsafe { self.0.add(i) }
    }
}

/// Direct `u8` stem convolution: `x_rgb0` is `[n, h, w]` packed `u32` words `[R, G, B, 0]`, `stem` is the
/// packed 3-channel kernel, and `out` is C4 u8 (single-grid or per-channel) or `f32`. Runs row bands in
/// parallel with L1-resident tap strips; bit-identical across thread counts and scalar/AVX2/VNNI.
pub fn stem_u8(x_rgb0: &[u32], n: usize, h: usize, w: usize, stem: &QStem, out: QOut<'_>) {
    let hw = h * w;
    assert!(x_rgb0.len() >= n * hw, "q8::stem_u8: input too short");
    let (oh, ow) = stem.out_hw(h, w);
    let ohw = oh * ow;
    let og = stem.o / 4;
    let (qptr, fptr, out_grids): (OutPtr<u32>, OutPtr<f32>, Option<Vec<super::Grid>>) = match out {
        QOut::U8 { data, grid } => {
            assert!(data.len() >= n * og * ohw, "q8::stem_u8: u8 output too short");
            (OutPtr(data.as_mut_ptr()), OutPtr(std::ptr::null_mut()), Some(vec![grid; stem.o]))
        }
        QOut::U8PerChannel { data, grids } => {
            assert!(data.len() >= n * og * ohw, "q8::stem_u8: u8 output too short");
            assert_eq!(grids.len(), stem.o, "q8::stem_u8: output grids length");
            (OutPtr(data.as_mut_ptr()), OutPtr(std::ptr::null_mut()), Some(grids.to_vec()))
        }
        QOut::F32 { data } => {
            assert!(data.len() >= n * stem.o * ohw, "q8::stem_u8: f32 output too short");
            (OutPtr(std::ptr::null_mut()), OutPtr(data.as_mut_ptr()), None)
        }
    };
    if n * ohw == 0 {
        return;
    }
    let kern = kernel();
    let (k, s, p, ks) = (stem.k, stem.stride, stem.pad, stem.ks);
    let strip_w = ow.next_multiple_of(NR);
    let chunks = ow.div_ceil(NR);
    let zero_corr = [0i32; MR];
    let resw = [0u32; NR];
    let group_out: Vec<([f32; MR], [i32; MR])> = match &out_grids {
        Some(gs) => (0..og)
            .map(|g| {
                (
                    [gs[4 * g].inv(), gs[4 * g + 1].inv(), gs[4 * g + 2].inv(), gs[4 * g + 3].inv()],
                    [
                        gs[4 * g].zero as i32,
                        gs[4 * g + 1].zero as i32,
                        gs[4 * g + 2].zero as i32,
                        gs[4 * g + 3].zero as i32,
                    ],
                )
            })
            .collect(),
        None => Vec::new(),
    };
    let threads = rayon::current_num_threads();
    let total_rows = n * oh;
    let band = (total_rows / (3 * threads).max(1)).max(1);
    let bands = total_rows.div_ceil(band);
    (0..bands).into_par_iter().for_each(|bi| {
        let r0 = bi * band;
        let r1 = (r0 + band).min(total_rows);
        let mut strips = vec![0u32; ks * strip_w];
        let mut row_ptrs = vec![std::ptr::null::<u32>(); ks];
        for (t, ptr) in row_ptrs.iter_mut().enumerate().take(ks) {
            // SAFETY: `t * strip_w` is inside `strips` and every strip has `chunks * NR` words.
            *ptr = unsafe { strips.as_ptr().add(t * strip_w) };
        }
        let mut acc: Tile = [[0; NR]; MR];
        let mut qw = [0u32; NR];
        let mut fv = [[0f32; NR]; MR];
        for row_idx in r0..r1 {
            let (img, oy) = (row_idx / oh, row_idx % oh);
            let src = &x_rgb0[img * hw..(img + 1) * hw];
            for ky in 0..k {
                let iy = (oy * s + ky) as isize - p as isize;
                if iy < 0 || iy >= h as isize {
                    for kx in 0..k {
                        let t = ky * k + kx;
                        strips[t * strip_w..t * strip_w + ow].fill(0);
                    }
                    continue;
                }
                let srow = &src[iy as usize * w..(iy as usize + 1) * w];
                for kx in 0..k {
                    let t = ky * k + kx;
                    let dst = &mut strips[t * strip_w..t * strip_w + ow];
                    for (ox, d) in dst.iter_mut().enumerate() {
                        let ix = (ox * s + kx) as isize - p as isize;
                        *d = if ix < 0 || ix >= w as isize { 0 } else { srow[ix as usize] };
                    }
                }
            }
            for g in 0..og {
                let out_kind = if out_grids.is_some() {
                    let (inv, zero) = &group_out[g];
                    Out::U8 { inv, zero }
                } else {
                    Out::F32
                };
                let e = Epi {
                    corr: &zero_corr,
                    scale: stem.scale[4 * g..4 * g + 4].try_into().unwrap(),
                    bias: stem.bias[4 * g..4 * g + 4].try_into().unwrap(),
                    silu: stem.silu,
                    res: None,
                    out: out_kind,
                };
                let panel = stem.a[g * ks * MR..].as_ptr();
                for chunk in 0..chunks {
                    let col = chunk * NR;
                    let cols = NR.min(ow - col);
                    // SAFETY: `panel` has `ks * MR` words; each strip has `strip_w >= col + NR` words.
                    unsafe { kern(ks, panel, row_ptrs.as_ptr(), col, &mut acc) };
                    if cols == NR {
                        e.apply_full(&acc, &resw, &mut qw, &mut fv);
                    } else {
                        e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                    }
                    if out_grids.is_some() {
                        let at = (img * og + g) * ohw + oy * ow + col;
                        // SAFETY: disjoint output row owned by this task.
                        unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), qptr.at(at), cols) };
                    } else {
                        for (rr, row) in fv.iter().enumerate() {
                            let at = (img * stem.o + 4 * g + rr) * ohw + oy * ow + col;
                            // SAFETY: disjoint output row owned by this task.
                            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), fptr.at(at), cols) };
                        }
                    }
                }
            }
        }
    });
}
