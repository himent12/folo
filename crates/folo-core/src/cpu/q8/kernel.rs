//! The int8 micro-kernel and its epilogue: `acc[r][j] = Σ_p Σ_t a[p][r].byte(t) · b_p[j].byte(t)` over a
//! `MR × NR` tile (4 output channels, one C4 group, × 24 columns), the four bytes of each i32 being four
//! consecutive K entries (u8 activations × s8 weights), accumulated exactly in i32. One AVX-VNNI version
//! (`vpdpbusd`) and a portable one, chosen at run time; both give the same integers.
//!
//! The epilogue turns a finished tile into outputs: [`Epi::apply`] (portable) and its AVX2 twin, bit for bit the
//! same (the SiLU is [`silu_q8`], whose scalar and vector forms run the same operations). The twin runs on any CPU
//! with AVX2 and FMA, VNNI or not; `FOLO_Q8_SCALAR` turns off both vector paths.

use std::sync::OnceLock;

/// Output channels per tile: one C4 group.
pub const MR: usize = 4;
/// Columns (output pixels) per tile: three ymm registers of 8 i32 lanes.
pub const NR: usize = 24;

/// A finished tile of i32 accumulators.
pub type Tile = [[i32; NR]; MR];

/// `acc = A_panel · B` over `ks` K steps: `a` is the `[ks][MR]` panel, B's row `p` is the `NR` words at
/// `rows[p] + col`.
pub(crate) type QKernel = unsafe fn(usize, *const i32, *const *const u32, usize, &mut Tile);

/// The kernel in use: AVX-VNNI when the CPU has it (and `FOLO_Q8_SCALAR` is unset), else the portable one.
pub(crate) fn kernel() -> QKernel {
    static K: OnceLock<QKernel> = OnceLock::new();
    *K.get_or_init(|| if uses_vnni() { kernel_vnni_entry as QKernel } else { kernel_scalar as QKernel })
}

/// True when the int8 kernels run on AVX-VNNI (`vpdpbusd`), false on the portable fallback.
pub fn uses_vnni() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            uses_avx2() && std::is_x86_feature_detected!("avxvnni")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// True when the epilogue runs its AVX2 + FMA form (unless `FOLO_Q8_SCALAR` is set).
#[cfg(target_arch = "x86_64")]
pub(crate) fn uses_avx2() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var_os("FOLO_Q8_SCALAR").is_none()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
    })
}

/// The portable kernel.
///
/// # Safety
/// `a` points to `ks · MR` words; each `rows[p]` (`p < ks`) to at least `col + NR` words.
pub(crate) unsafe fn kernel_scalar(
    ks: usize,
    a: *const i32,
    rows: *const *const u32,
    col: usize,
    acc: &mut Tile,
) {
    *acc = [[0; NR]; MR];
    for p in 0..ks {
        // SAFETY: guaranteed by the caller.
        let (ap, bp) = unsafe { (std::slice::from_raw_parts(a.add(p * MR), MR), (*rows.add(p)).add(col)) };
        // SAFETY: as above.
        let b = unsafe { std::slice::from_raw_parts(bp, NR) };
        for (r, &w) in ap.iter().enumerate() {
            let wb = w.to_le_bytes();
            for (j, &x) in b.iter().enumerate() {
                let xb = x.to_le_bytes();
                acc[r][j] += (0..4).map(|t| xb[t] as i32 * wb[t] as i8 as i32).sum::<i32>();
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn kernel_vnni_entry(ks: usize, a: *const i32, rows: *const *const u32, col: usize, acc: &mut Tile) {
    // SAFETY: selected only after detecting avx2 and avxvnni; the pointer contract is forwarded.
    unsafe { kernel_vnni(ks, a, rows, col, acc) }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn kernel_vnni_entry(ks: usize, a: *const i32, rows: *const *const u32, col: usize, acc: &mut Tile) {
    // SAFETY: forwarded.
    unsafe { kernel_scalar(ks, a, rows, col, acc) }
}

/// The AVX-VNNI kernel: 12 ymm accumulators, 3 B loads and 4 weight broadcasts per K step.
///
/// # Safety
/// As [`kernel_scalar`], and the CPU supports AVX2 and AVX-VNNI.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn kernel_vnni(ks: usize, a: *const i32, rows: *const *const u32, col: usize, acc: &mut Tile) {
    use std::arch::x86_64::*;
    let z = _mm256_setzero_si256();
    let (mut c00, mut c01, mut c02) = (z, z, z);
    let (mut c10, mut c11, mut c12) = (z, z, z);
    let (mut c20, mut c21, mut c22) = (z, z, z);
    let (mut c30, mut c31, mut c32) = (z, z, z);
    // SAFETY: the caller guarantees `ks · MR` words at `a` and `col + NR` words at every row.
    unsafe {
        for p in 0..ks {
            let bp = (*rows.add(p)).add(col) as *const __m256i;
            let b0 = _mm256_loadu_si256(bp);
            let b1 = _mm256_loadu_si256(bp.add(1));
            let b2 = _mm256_loadu_si256(bp.add(2));
            let ap = a.add(p * MR);
            let w0 = _mm256_set1_epi32(*ap);
            c00 = _mm256_dpbusd_avx_epi32(c00, b0, w0);
            c01 = _mm256_dpbusd_avx_epi32(c01, b1, w0);
            c02 = _mm256_dpbusd_avx_epi32(c02, b2, w0);
            let w1 = _mm256_set1_epi32(*ap.add(1));
            c10 = _mm256_dpbusd_avx_epi32(c10, b0, w1);
            c11 = _mm256_dpbusd_avx_epi32(c11, b1, w1);
            c12 = _mm256_dpbusd_avx_epi32(c12, b2, w1);
            let w2 = _mm256_set1_epi32(*ap.add(2));
            c20 = _mm256_dpbusd_avx_epi32(c20, b0, w2);
            c21 = _mm256_dpbusd_avx_epi32(c21, b1, w2);
            c22 = _mm256_dpbusd_avx_epi32(c22, b2, w2);
            let w3 = _mm256_set1_epi32(*ap.add(3));
            c30 = _mm256_dpbusd_avx_epi32(c30, b0, w3);
            c31 = _mm256_dpbusd_avx_epi32(c31, b1, w3);
            c32 = _mm256_dpbusd_avx_epi32(c32, b2, w3);
        }
        let out = acc.as_mut_ptr() as *mut __m256i;
        for (i, v) in [c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32].into_iter().enumerate() {
            _mm256_storeu_si256(out.add(i), v);
        }
    }
}

/// `max(a, b)` as `vmaxps` computes it (`b` when either is NaN).
#[inline(always)]
fn maxps(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `min(a, b)` as `vminps` computes it (`b` when either is NaN).
#[inline(always)]
fn minps(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// The epilogue's SiLU, `x / (1 + e^{−x})` with the exponent clamped to `±87` (so `e` and `2ⁿ` stay normal) and
/// `x` to at least `−88` (below it SiLU is under 1e-36): folo-core's `exp` polynomial (as in its vectorised
/// SiLU, relative error about 1e-7) with one power-of-two scale. The AVX2 epilogue computes the same operations
/// in the same order (FMAs as `mul_add`), so the same bits. NaN stays NaN.
#[inline]
pub fn silu_q8(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    let x = maxps(-88.0, x);
    let t = maxps(minps(-x, 87.0), -87.0);
    let n = (t * LOG2E).round_ties_even();
    let r = (-n).mul_add(0.693_359_4, t);
    let r = (-n).mul_add(-2.121_944_4e-4, r);
    let mut p: f32 = 1.987_569_1e-4;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_5e-1);
    p = p.mul_add(r, 0.5);
    let y = (p * r).mul_add(r, r) + 1.0;
    let e = y * f32::from_bits(((n as i32 + 127) as u32) << 23);
    x / (1.0 + e)
}

/// [`silu_q8`] on 8 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) fn silu_q8_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_max_ps(_mm256_set1_ps(-88.0), x);
    let neg = _mm256_xor_ps(x, _mm256_set1_ps(-0.0));
    let t = _mm256_max_ps(_mm256_min_ps(neg, _mm256_set1_ps(87.0)), _mm256_set1_ps(-87.0));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        t,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), t);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    let y = _mm256_add_ps(_mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r), _mm256_set1_ps(1.0));
    let s = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)));
    let e = _mm256_mul_ps(y, _mm256_castsi256_ps(s));
    _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

/// The output of a tile's epilogue.
#[derive(Clone, Copy)]
pub(crate) enum Out {
    /// u8 codes on the grid `(inv_scale, zero)`, packed four channels to a word (C4).
    U8 { inv: f32, zero: i32 },
    /// `f32` values.
    F32,
}

/// One output channel group's epilogue: `y = (acc − corr) · scale + bias`, SiLU if `silu`, `+ rs · (r − rz)`
/// with a residual, then quantised (`clamp(round(y · inv) + zero, 0, 255)`, ties to even) or kept `f32`.
#[derive(Clone, Copy)]
pub(crate) struct Epi<'a> {
    pub corr: &'a [i32; MR],
    pub scale: &'a [f32; MR],
    pub bias: &'a [f32; MR],
    pub silu: bool,
    /// Residual grid `(scale, zero)`.
    pub res: Option<(f32, i32)>,
    pub out: Out,
}

/// The value of row `r` of word `w` (C4: byte `r`).
#[inline(always)]
fn byte(w: u32, r: usize) -> i32 {
    ((w >> (8 * r)) & 0xFF) as i32
}

impl Epi<'_> {
    #[inline(always)]
    fn value(&self, r: usize, acc: i32, res: u32) -> f32 {
        let mut y = (acc.wrapping_sub(self.corr[r]) as f32).mul_add(self.scale[r], self.bias[r]);
        if self.silu {
            y = silu_q8(y);
        }
        if let Some((rs, rz)) = self.res {
            y = rs.mul_add((byte(res, r) - rz) as f32, y);
        }
        y
    }

    /// The portable epilogue of `acc` for the first `n` columns: C4 words into `q` (with `res` the residual
    /// words), or `f32` rows into `f` (`f[r][j]`).
    pub(crate) fn apply(
        &self,
        acc: &Tile,
        n: usize,
        res: &[u32; NR],
        q: &mut [u32; NR],
        f: &mut [[f32; NR]; MR],
    ) {
        for j in 0..n {
            let mut word = 0u32;
            for r in 0..MR {
                let y = self.value(r, acc[r][j], res[j]);
                match self.out {
                    Out::F32 => f[r][j] = y,
                    Out::U8 { inv, zero } => {
                        let v = minps(maxps(y * inv, -512.0), 512.0).round_ties_even() as i32;
                        word |= ((v + zero).clamp(0, 255) as u32) << (8 * r);
                    }
                }
            }
            q[j] = word;
        }
    }

    /// [`Epi::apply`] for all `NR` columns, vectorised when the CPU has AVX2 and FMA; the same bits.
    pub(crate) fn apply_full(&self, acc: &Tile, res: &[u32; NR], q: &mut [u32; NR], f: &mut [[f32; NR]; MR]) {
        #[cfg(target_arch = "x86_64")]
        if uses_avx2() {
            // SAFETY: AVX2 and FMA were detected.
            return unsafe { self.apply_avx2(acc, res, q, f) };
        }
        self.apply(acc, NR, res, q, f)
    }

    /// The AVX2 epilogue, 8 columns at a time. u8 outputs: the four rows' codes (`|v| ≤ 512`, so exact in i16)
    /// are packed to i16 with signed saturation, the zero point is added, the result is packed to u8 with unsigned
    /// saturation (the clamp to `[0, 255]`), and each 128-bit lane's 4 × 4 bytes are transposed into C4 words.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn apply_avx2(&self, acc: &Tile, res: &[u32; NR], q: &mut [u32; NR], f: &mut [[f32; NR]; MR]) {
        use std::arch::x86_64::*;
        // SAFETY: every pointer is to a local array of the right length.
        unsafe {
            let (lo, hi) = (_mm256_set1_ps(-512.0), _mm256_set1_ps(512.0));
            // Within each 128-bit lane: row-major bytes (row r, pixel i at 4r + i) to pixel-major (4i + r).
            let transpose = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14,
                3, 7, 11, 15,
            );
            for k in 0..NR / 8 {
                let w = _mm256_loadu_si256(res.as_ptr().add(8 * k) as *const __m256i);
                let mut v = [_mm256_setzero_si256(); MR];
                for (r, vr) in v.iter_mut().enumerate() {
                    let a = _mm256_loadu_si256(acc[r].as_ptr().add(8 * k) as *const __m256i);
                    let t = _mm256_cvtepi32_ps(_mm256_sub_epi32(a, _mm256_set1_epi32(self.corr[r])));
                    let mut y =
                        _mm256_fmadd_ps(t, _mm256_set1_ps(self.scale[r]), _mm256_set1_ps(self.bias[r]));
                    if self.silu {
                        y = silu_q8_avx2(y);
                    }
                    if let Some((rs, rz)) = self.res {
                        let b = _mm256_and_si256(
                            _mm256_srlv_epi32(w, _mm256_set1_epi32(8 * r as i32)),
                            _mm256_set1_epi32(0xFF),
                        );
                        let d = _mm256_cvtepi32_ps(_mm256_sub_epi32(b, _mm256_set1_epi32(rz)));
                        y = _mm256_fmadd_ps(_mm256_set1_ps(rs), d, y);
                    }
                    match self.out {
                        Out::F32 => _mm256_storeu_ps(f[r].as_mut_ptr().add(8 * k), y),
                        Out::U8 { inv, .. } => {
                            let y =
                                _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(y, _mm256_set1_ps(inv)), lo), hi);
                            *vr = _mm256_cvtps_epi32(y);
                        }
                    }
                }
                if let Out::U8 { zero, .. } = self.out {
                    let z = _mm256_set1_epi16(zero as i16);
                    let p01 = _mm256_adds_epi16(_mm256_packs_epi32(v[0], v[1]), z);
                    let p23 = _mm256_adds_epi16(_mm256_packs_epi32(v[2], v[3]), z);
                    let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), transpose);
                    _mm256_storeu_si256(q.as_mut_ptr().add(8 * k) as *mut __m256i, bytes);
                }
            }
        }
    }
}
