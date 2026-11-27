//! Direct `u8` RGB0-to-C4 AVX-VNNI stem (`QStem` + [`stem_u8`]): input pixels are already exact `u8` codes
//! (`scale = 1/255`, `zero = 0`), so a 3-channel stem runs directly on packed `[R, G, B, 0]` `u32` words in
//! L1-resident row strips (`9 × ow × 4` bytes ≈ 12 KiB for 640² s2), avoiding the `f32` stem write + scalar
//! requantize pass.

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
    pub(super) a: Vec<i32>,
    /// Per output channel: weight scale with `1 / 255` folded in.
    pub(super) scale: Vec<f32>,
    pub(super) bias: Vec<f32>,
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
    let hw = h.checked_mul(w).expect("pack_f32_to_rgb0_u8: h * w overflow");
    let n_hw = n.checked_mul(hw).expect("pack_f32_to_rgb0_u8: n * h * w overflow");
    let x_need = n_hw.checked_mul(3).expect("pack_f32_to_rgb0_u8: n * 3 * h * w overflow");
    assert!(x.len() >= x_need && out.len() >= n_hw, "pack_f32_to_rgb0_u8: buffer too short");
    if n_hw == 0 {
        return;
    }
    if n_hw < crate::cpu::PAR_MIN {
        for img in 0..n {
            let base = img * 3 * hw;
            pack_rgb0_slice(
                &x[base..base + hw],
                &x[base + hw..base + 2 * hw],
                &x[base + 2 * hw..base + 3 * hw],
                &mut out[img * hw..(img + 1) * hw],
            );
        }
        return;
    }
    let threads = rayon::current_num_threads();
    let block = (hw / (2 * threads).max(1)).clamp(4096, 16_384);
    let blocks = hw.div_ceil(block);
    let optr = OutPtr(out.as_mut_ptr());
    let optr = &optr;
    crate::cpu::par_for(n * blocks, &|t| {
        let (img, b) = (t / blocks, t % blocks);
        let p0 = b * block;
        let len = block.min(hw - p0);
        let base = img * 3 * hw + p0;
        // SAFETY: each `(img, b)` owns disjoint words `img * hw + p0..+len` inside `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(optr.at(img * hw + p0), len) };
        pack_rgb0_slice(
            &x[base..base + len],
            &x[base + hw..base + hw + len],
            &x[base + 2 * hw..base + 2 * hw + len],
            o,
        );
    });
}

fn pack_rgb0_slice(r: &[f32], g: &[f32], b: &[f32], out: &mut [u32]) {
    let len = out.len();
    assert!(r.len() >= len && g.len() >= len && b.len() >= len, "pack_rgb0_slice: input too short");
    #[cfg(target_arch = "x86_64")]
    if super::kernel::uses_avx2() {
        // SAFETY: AVX2 and FMA were detected; slice lengths asserted above.
        return unsafe { pack_rgb0_slice_avx2(r, g, b, out) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if super::kernel::uses_wasm_simd() {
        return super::wasm_simd::pack_rgb0_slice_wasm(r, g, b, out);
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
    // SAFETY: `i + 16 <= len` (and `i + 8 <= len`) and all four slices have length `len`.
    unsafe {
        while i + 16 <= len {
            let vra = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(r.as_ptr().add(i)), scale), zero),
                scale,
            ));
            let vga = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(g.as_ptr().add(i)), scale), zero),
                scale,
            ));
            let vba = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(b.as_ptr().add(i)), scale), zero),
                scale,
            ));
            let bytes_a = _mm256_shuffle_epi8(
                _mm256_packus_epi16(_mm256_packs_epi32(vra, vga), _mm256_packs_epi32(vba, z_i)),
                transpose,
            );
            let vrb = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(r.as_ptr().add(i + 8)), scale), zero),
                scale,
            ));
            let vgb = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(g.as_ptr().add(i + 8)), scale), zero),
                scale,
            ));
            let vbb = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(b.as_ptr().add(i + 8)), scale), zero),
                scale,
            ));
            let bytes_b = _mm256_shuffle_epi8(
                _mm256_packus_epi16(_mm256_packs_epi32(vrb, vgb), _mm256_packs_epi32(vbb, z_i)),
                transpose,
            );
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, bytes_a);
            _mm256_storeu_si256(out.as_mut_ptr().add(i + 8) as *mut __m256i, bytes_b);
            i += 16;
        }
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
    let hw = h.checked_mul(w).expect("q8::stem_u8: h * w overflow");
    let n_hw = n.checked_mul(hw).expect("q8::stem_u8: n * h * w overflow");
    assert!(x_rgb0.len() >= n_hw, "q8::stem_u8: input too short");
    let (oh, ow) = stem.out_hw(h, w);
    let ohw = oh.checked_mul(ow).expect("q8::stem_u8: oh * ow overflow");
    let og = stem.o / 4;
    let n_ohw = n.checked_mul(ohw).expect("q8::stem_u8: n * oh * ow overflow");
    let u8_need = n_ohw.checked_mul(og).expect("q8::stem_u8: u8 output size overflow");
    let f32_need = n_ohw.checked_mul(stem.o).expect("q8::stem_u8: f32 output size overflow");
    type GroupGrid = ([f32; MR], [i32; MR]);
    const STACK_OG: usize = 64;
    let mut out_stack: [GroupGrid; STACK_OG] = [([0.0; MR], [0; MR]); STACK_OG];
    let out_heap: Vec<GroupGrid>;
    let (qptr, fptr, is_u8, group_out): (OutPtr<u32>, OutPtr<f32>, bool, &[GroupGrid]) = match out {
        QOut::U8 { data, grid } => {
            assert!(data.len() >= u8_need, "q8::stem_u8: u8 output too short");
            let pair = ([grid.inv(); MR], [grid.zero as i32; MR]);
            let slice: &[GroupGrid] = if og <= STACK_OG {
                out_stack[..og].fill(pair);
                &out_stack[..og]
            } else {
                out_heap = vec![pair; og];
                &out_heap
            };
            (OutPtr(data.as_mut_ptr()), OutPtr(std::ptr::null_mut()), true, slice)
        }
        QOut::U8PerChannel { data, grids } => {
            assert!(data.len() >= u8_need, "q8::stem_u8: u8 output too short");
            assert_eq!(grids.len(), stem.o, "q8::stem_u8: output grids length");
            let pair_of = |g: usize| {
                (
                    [
                        grids[4 * g].inv(),
                        grids[4 * g + 1].inv(),
                        grids[4 * g + 2].inv(),
                        grids[4 * g + 3].inv(),
                    ],
                    [
                        grids[4 * g].zero as i32,
                        grids[4 * g + 1].zero as i32,
                        grids[4 * g + 2].zero as i32,
                        grids[4 * g + 3].zero as i32,
                    ],
                )
            };
            let slice: &[GroupGrid] = if og <= STACK_OG {
                for (g, dst) in out_stack[..og].iter_mut().enumerate() {
                    *dst = pair_of(g);
                }
                &out_stack[..og]
            } else {
                out_heap = (0..og).map(pair_of).collect();
                &out_heap
            };
            (OutPtr(data.as_mut_ptr()), OutPtr(std::ptr::null_mut()), true, slice)
        }
        QOut::F32 { data } => {
            assert!(data.len() >= f32_need, "q8::stem_u8: f32 output too short");
            (OutPtr(std::ptr::null_mut()), OutPtr(data.as_mut_ptr()), false, &[])
        }
    };
    if n_ohw == 0 {
        return;
    }
    let kern = kernel();
    #[cfg(target_arch = "x86_64")]
    let fused_isa = super::fused::active_isa();
    let (k, s, p, ks) = (stem.k, stem.stride, stem.pad, stem.ks);
    let strip_w = ow.checked_next_multiple_of(NR).expect("q8::stem_u8: strip_w overflow");
    let strip_len = ks.checked_mul(strip_w).expect("q8::stem_u8: strip buffer overflow");
    let chunks = ow.div_ceil(NR);
    let zero_corr = [0i32; MR];
    let resw = [0u32; NR];
    let threads = rayon::current_num_threads();
    let total_rows = n.checked_mul(oh).expect("q8::stem_u8: n * oh overflow");
    let band = (total_rows / (2 * threads).max(1)).max(4);
    let bands = total_rows.div_ceil(band);
    let fast_k3s2p1 = k == 3 && s == 2 && p == 1;
    let run_band = |bi: usize| {
        let r0 = bi * band;
        let r1 = (r0 + band).min(total_rows);
        let mut acc: Tile = [[0; NR]; MR];
        let mut qw = [0u32; NR];
        let mut fv = [[0f32; NR]; MR];
        if fast_k3s2p1 {
            // 3 rotating row slots: each slot holds `even` (`strip_w`, for kx=1) and `odd` (`strip_w + 1`,
            // with `odd[0] = 0` so `odd[0..]` is kx=0 and `odd[1..]` is kx=2). Row `2*oy + 1` (ky=2) becomes
            // next row's `ky = 0`, cutting strip packing by 4.5× and eliminating inner-loop branches.
            // Stack-backed for `strip_w <= STACK_W` (covers up to 1024px inputs with zero heap allocs).
            const STACK_W: usize = 512;
            let mut ev_stack = [[0u32; STACK_W]; 3];
            let mut od_stack = [[0u32; STACK_W + 1]; 3];
            let mut ev_heap;
            let mut od_heap;
            let (evens, odds): ([&mut [u32]; 3], [&mut [u32]; 3]) = if strip_w <= STACK_W {
                let [e0, e1, e2] = &mut ev_stack;
                let [o0, o1, o2] = &mut od_stack;
                (
                    [&mut e0[..strip_w], &mut e1[..strip_w], &mut e2[..strip_w]],
                    [&mut o0[..strip_w + 1], &mut o1[..strip_w + 1], &mut o2[..strip_w + 1]],
                )
            } else {
                ev_heap = [vec![0u32; strip_w], vec![0u32; strip_w], vec![0u32; strip_w]];
                od_heap = [vec![0u32; strip_w + 1], vec![0u32; strip_w + 1], vec![0u32; strip_w + 1]];
                let [e0, e1, e2] = &mut ev_heap;
                let [o0, o1, o2] = &mut od_heap;
                (
                    [e0.as_mut_slice(), e1.as_mut_slice(), e2.as_mut_slice()],
                    [o0.as_mut_slice(), o1.as_mut_slice(), o2.as_mut_slice()],
                )
            };
            let fill_slot = |ev: &mut [u32], od: &mut [u32], img: usize, iy: isize| {
                od[0] = 0;
                if iy < 0 || iy >= h as isize {
                    ev[..ow].fill(0);
                    od[1..ow + 1].fill(0);
                } else {
                    let srow = &x_rgb0[img * hw + iy as usize * w..][..w];
                    super::deinterleave_even_odd(srow, &mut ev[..ow], &mut od[1..ow + 1], 0);
                }
            };
            let mut row_ptrs = [std::ptr::null::<u32>(); 9];
            let mut prev_img = usize::MAX;
            let mut prev_oy = usize::MAX;
            let mut ord = [0usize, 1, 2];
            for row_idx in r0..r1 {
                let (img, oy) = (row_idx / oh, row_idx % oh);
                if img == prev_img && oy == prev_oy + 1 {
                    ord.rotate_left(2);
                    fill_slot(evens[ord[1]], odds[ord[1]], img, (2 * oy) as isize);
                    fill_slot(evens[ord[2]], odds[ord[2]], img, (2 * oy + 1) as isize);
                } else {
                    ord = [0, 1, 2];
                    fill_slot(evens[0], odds[0], img, (2 * oy) as isize - 1);
                    fill_slot(evens[1], odds[1], img, (2 * oy) as isize);
                    fill_slot(evens[2], odds[2], img, (2 * oy + 1) as isize);
                }
                prev_img = img;
                prev_oy = oy;
                for (ky, &sl) in ord.iter().enumerate() {
                    let od_ptr = odds[sl].as_ptr();
                    let ev_ptr = evens[sl].as_ptr();
                    // SAFETY: `odds[sl]` has `strip_w + 1` words (`od_ptr.add(1)` has `strip_w`); `evens[sl]` has `strip_w`.
                    row_ptrs[3 * ky] = od_ptr;
                    row_ptrs[3 * ky + 1] = ev_ptr;
                    row_ptrs[3 * ky + 2] = unsafe { od_ptr.add(1) };
                }
                #[cfg(target_arch = "x86_64")]
                if let Some(isa) = fused_isa {
                    // SAFETY: `isa` matches `active_isa()`; row `oy` of `img` is disjointly owned by this band.
                    unsafe {
                        super::fused::run_stem_row(
                            isa,
                            &stem.a,
                            &stem.scale,
                            &stem.bias,
                            og,
                            ow,
                            ohw,
                            img,
                            oy,
                            stem.silu,
                            is_u8,
                            &row_ptrs,
                            group_out,
                            qptr.0,
                            fptr.0,
                        );
                    }
                    continue;
                }
                for chunk in 0..chunks {
                    let col = chunk * NR;
                    let cols = NR.min(ow - col);
                    for g in 0..og {
                        let out_kind = if is_u8 {
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
                        // SAFETY: `panel` has `ks * MR` words; each strip has `strip_w >= col + NR` words.
                        unsafe { kern(ks, panel, row_ptrs.as_ptr(), col, &mut acc) };
                        if is_u8 {
                            let at = (img * og + g) * ohw + oy * ow + col;
                            if cols == NR {
                                // SAFETY: disjoint `NR`-word output slice owned by this task.
                                unsafe { e.apply_full_u8_direct(&acc, std::ptr::null(), qptr.at(at)) };
                            } else {
                                e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                                // SAFETY: disjoint `cols`-word output slice owned by this task.
                                unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), qptr.at(at), cols) };
                            }
                        } else {
                            if cols == NR {
                                e.apply_full(&acc, &resw, &mut qw, &mut fv);
                            } else {
                                e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                            }
                            for (rr, row) in fv.iter().enumerate() {
                                let at = (img * stem.o + 4 * g + rr) * ohw + oy * ow + col;
                                // SAFETY: disjoint output row owned by this task.
                                unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), fptr.at(at), cols) };
                            }
                        }
                    }
                }
            }
            return;
        }
        let mut strips = vec![0u32; strip_len];
        let mut row_ptrs = vec![std::ptr::null::<u32>(); ks];
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
            let base = strips.as_ptr();
            for (t, ptr) in row_ptrs.iter_mut().enumerate().take(ks) {
                // SAFETY: `t * strip_w` is inside `strips` and every strip has `strip_w >= chunks * NR` words.
                *ptr = unsafe { base.add(t * strip_w) };
            }
            for chunk in 0..chunks {
                let col = chunk * NR;
                let cols = NR.min(ow - col);
                for g in 0..og {
                    let out_kind = if is_u8 {
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
                    // SAFETY: `panel` has `ks * MR` words; each strip has `strip_w >= col + NR` words.
                    unsafe { kern(ks, panel, row_ptrs.as_ptr(), col, &mut acc) };
                    if is_u8 {
                        let at = (img * og + g) * ohw + oy * ow + col;
                        if cols == NR {
                            // SAFETY: disjoint output row owned by this task.
                            unsafe { e.apply_full_u8_direct(&acc, std::ptr::null(), qptr.at(at)) };
                        } else {
                            e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                            // SAFETY: disjoint output row owned by this task.
                            unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), qptr.at(at), cols) };
                        }
                    } else {
                        if cols == NR {
                            e.apply_full(&acc, &resw, &mut qw, &mut fv);
                        } else {
                            e.apply(&acc, cols, &resw, &mut qw, &mut fv);
                        }
                        for (rr, row) in fv.iter().enumerate() {
                            let at = (img * stem.o + 4 * g + rr) * ohw + oy * ow + col;
                            // SAFETY: disjoint output row owned by this task.
                            unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), fptr.at(at), cols) };
                        }
                    }
                }
            }
        }
    };
    if bands <= 1 || threads <= 1 {
        (0..bands).for_each(run_band);
    } else {
        crate::cpu::par_for(bands, &run_band);
    }
}
