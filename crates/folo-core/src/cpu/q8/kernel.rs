//! The int8 micro-kernel and its epilogue: `acc[r][j] = Σ_p Σ_t a[p][r].byte(t) · b_p[j].byte(t)` over a
//! `MR × NR` tile (4 output channels, one C4 group, × 24 columns), the four bytes of each i32 being four
//! consecutive K entries (u8 activations × s8 weights), accumulated exactly in i32. One AVX-VNNI version
//! (`vpdpbusd`) and a portable one, chosen at run time; both give the same integers.
//!
//! The epilogue turns a finished tile into outputs: [`Epi::apply`] (portable) and its AVX2 twin, bit for bit the
//! same (the SiLU is [`super::silu::silu_q8`], division-free and bit-exact on both paths). The twin runs on any
//! CPU with AVX2 and FMA, VNNI or not; `FOLO_Q8_SCALAR` turns off both vector paths.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
pub(crate) use super::silu::silu_q8_avx2;
use super::silu::{maxps, minps, silu_q8};

/// Output channels per tile: one C4 group.
pub const MR: usize = 4;
/// Columns (output pixels) per tile: three ymm registers of 8 i32 lanes.
pub const NR: usize = 24;

/// A finished tile of i32 accumulators.
pub type Tile = [[i32; NR]; MR];

/// An L1-resident `f32` tile (`4 × 24 × 4 = 384` bytes) for register-safe sequential per-part concat
/// accumulation (`tile_f32[r][j] += (acc_p[r][j] - corr_p[r]) * scale_p[r]`).
pub type TileF32 = [[f32; NR]; MR];

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

/// Fold one part's finished `i32` tile into `dst`: `dst[r][j] = (acc[r][j] - corr[r]) * scale[r] + dst[r][j]`.
/// Scalar and AVX2 execute the exact same FMA operations in the same order.
pub(crate) fn accumulate_part_f32(acc: &Tile, corr: &[i32; MR], scale: &[f32; MR], dst: &mut TileF32) {
    #[cfg(target_arch = "x86_64")]
    if uses_avx2() {
        // SAFETY: AVX2 and FMA were detected.
        return unsafe { accumulate_part_f32_avx2(acc, corr, scale, dst) };
    }
    for r in 0..MR {
        let (cr, sr) = (corr[r], scale[r]);
        for j in 0..NR {
            dst[r][j] = (acc[r][j].wrapping_sub(cr) as f32).mul_add(sr, dst[r][j]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_part_f32_avx2(acc: &Tile, corr: &[i32; MR], scale: &[f32; MR], dst: &mut TileF32) {
    use std::arch::x86_64::*;
    // SAFETY: local arrays of fixed length `MR × NR` (`NR = 24 = 3 * 8`).
    unsafe {
        for r in 0..MR {
            let vc = _mm256_set1_epi32(corr[r]);
            let vs = _mm256_set1_ps(scale[r]);
            for k in 0..NR / 8 {
                let a = _mm256_loadu_si256(acc[r].as_ptr().add(8 * k) as *const __m256i);
                let t = _mm256_cvtepi32_ps(_mm256_sub_epi32(a, vc));
                let cur = _mm256_loadu_ps(dst[r].as_ptr().add(8 * k));
                let next = _mm256_fmadd_ps(t, vs, cur);
                _mm256_storeu_ps(dst[r].as_mut_ptr().add(8 * k), next);
            }
        }
    }
}

/// The output of a tile's epilogue (supports per-channel C4 grids at zero extra SIMD cost).
#[derive(Clone, Copy)]
pub(crate) enum Out<'a> {
    /// u8 codes on per-channel grids `(inv_scale[r], zero[r])`, packed four channels to a word (C4).
    U8 { inv: &'a [f32; MR], zero: &'a [i32; MR] },
    /// `f32` values.
    F32,
}

/// One output channel group's epilogue: `y = (acc − corr) · scale + bias` (or `tile_f32 + bias`), SiLU if `silu`,
/// `+ rs[r] · (byte(res, r) − rz[r])` with a residual, then quantised per channel (`clamp(round(y · inv[r]) +
/// zero[r], 0, 255)`, ties to even) or kept `f32`.
#[derive(Clone, Copy)]
pub(crate) struct Epi<'a> {
    pub corr: &'a [i32; MR],
    pub scale: &'a [f32; MR],
    pub bias: &'a [f32; MR],
    pub silu: bool,
    /// Per-channel residual grids `(scale[r], zero[r])`.
    pub res: Option<(&'a [f32; MR], &'a [i32; MR])>,
    pub out: Out<'a>,
}

/// The value of row `r` of word `w` (C4: byte `r`).
#[inline(always)]
fn byte(w: u32, r: usize) -> i32 {
    ((w >> (8 * r)) & 0xFF) as i32
}

impl Epi<'_> {
    #[inline(always)]
    fn finish_value(&self, r: usize, mut y: f32, res: u32) -> f32 {
        if self.silu {
            y = silu_q8(y);
        }
        if let Some((rs, rz)) = self.res {
            y = rs[r].mul_add((byte(res, r) - rz[r]) as f32, y);
        }
        y
    }

    #[inline(always)]
    fn value(&self, r: usize, acc: i32, res: u32) -> f32 {
        let y = (acc.wrapping_sub(self.corr[r]) as f32).mul_add(self.scale[r], self.bias[r]);
        self.finish_value(r, y, res)
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
                        let v = minps(maxps(y * inv[r], -512.0), 512.0).round_ties_even() as i32;
                        word |= ((v + zero[r]).clamp(0, 255) as u32) << (8 * r);
                    }
                }
            }
            q[j] = word;
        }
    }

    /// Portable epilogue from a pre-accumulated `TileF32` (adds `bias`, runs SiLU/residual/requant).
    pub(crate) fn apply_f32(
        &self,
        tile: &TileF32,
        n: usize,
        res: &[u32; NR],
        q: &mut [u32; NR],
        f: &mut [[f32; NR]; MR],
    ) {
        for j in 0..n {
            let mut word = 0u32;
            for r in 0..MR {
                let y = self.finish_value(r, tile[r][j] + self.bias[r], res[j]);
                match self.out {
                    Out::F32 => f[r][j] = y,
                    Out::U8 { inv, zero } => {
                        let v = minps(maxps(y * inv[r], -512.0), 512.0).round_ties_even() as i32;
                        word |= ((v + zero[r]).clamp(0, 255) as u32) << (8 * r);
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

    /// [`Epi::apply_f32`] for all `NR` columns, vectorised when the CPU has AVX2 and FMA; the same bits.
    pub(crate) fn apply_f32_full(
        &self,
        tile: &TileF32,
        res: &[u32; NR],
        q: &mut [u32; NR],
        f: &mut [[f32; NR]; MR],
    ) {
        #[cfg(target_arch = "x86_64")]
        if uses_avx2() {
            // SAFETY: AVX2 and FMA were detected.
            return unsafe { self.apply_f32_avx2(tile, res, q, f) };
        }
        self.apply_f32(tile, NR, res, q, f)
    }

    /// The AVX2 epilogue, 8 columns at a time, specialised by `(silu, res, out)` so broadcasts are hoisted
    /// out of the `NR / 8` loop and zero branches execute inside it.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn apply_avx2(
        &self,
        acc: &Tile,
        res: &[u32; NR],
        q: &mut [u32; NR],
        f: &mut [[f32; NR]; MR],
    ) {
        // SAFETY: forwarded; `q` has `NR` writable words.
        unsafe { self.apply_avx2_ptr(acc, res.as_ptr(), q.as_mut_ptr(), f) }
    }

    /// [`apply_full`] writing a full `NR`-word contiguous C4 tile directly to `q_dst` (skipping the `qw` copy).
    ///
    /// # Safety
    /// `q_dst` is writable for `NR` words; `res_ptr` is readable for `NR` words when `self.res.is_some()`.
    pub(crate) unsafe fn apply_full_u8_direct(&self, acc: &Tile, res_ptr: *const u32, q_dst: *mut u32) {
        #[cfg(target_arch = "x86_64")]
        if uses_avx2() {
            let mut dummy = [[0f32; NR]; MR];
            // SAFETY: AVX2 and FMA were detected; pointers guaranteed by caller.
            return unsafe { self.apply_avx2_ptr(acc, res_ptr, q_dst, &mut dummy) };
        }
        let mut resw = [0u32; NR];
        if self.res.is_some() {
            // SAFETY: `res_ptr` is readable for `NR` words.
            unsafe { std::ptr::copy_nonoverlapping(res_ptr, resw.as_mut_ptr(), NR) };
        }
        let mut qw = [0u32; NR];
        let mut dummy = [[0f32; NR]; MR];
        self.apply(acc, NR, &resw, &mut qw, &mut dummy);
        // SAFETY: `q_dst` is writable for `NR` words.
        unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), q_dst, NR) };
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn apply_avx2_ptr(&self, acc: &Tile, res: *const u32, q: *mut u32, f: &mut [[f32; NR]; MR]) {
        // SAFETY: pointer contract forwarded.
        unsafe {
            match (self.silu, self.res.is_some(), matches!(self.out, Out::U8 { .. })) {
                (true, false, true) => self.apply_avx2_spec::<true, false, true>(acc, res, q, f),
                (false, false, true) => self.apply_avx2_spec::<false, false, true>(acc, res, q, f),
                (true, true, true) => self.apply_avx2_spec::<true, true, true>(acc, res, q, f),
                (false, true, true) => self.apply_avx2_spec::<false, true, true>(acc, res, q, f),
                (true, false, false) => self.apply_avx2_spec::<true, false, false>(acc, res, q, f),
                (false, false, false) => self.apply_avx2_spec::<false, false, false>(acc, res, q, f),
                (true, true, false) => self.apply_avx2_spec::<true, true, false>(acc, res, q, f),
                (false, true, false) => self.apply_avx2_spec::<false, true, false>(acc, res, q, f),
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn apply_avx2_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
        &self,
        acc: &Tile,
        res: *const u32,
        q: *mut u32,
        f: &mut [[f32; NR]; MR],
    ) {
        use std::arch::x86_64::*;
        // SAFETY: every pointer is valid for `NR` words / `MR x NR` elements per caller contract.
        unsafe {
            let (lo, hi) = (_mm256_set1_ps(-512.0), _mm256_set1_ps(512.0));
            let transpose = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14,
                3, 7, 11, 15,
            );
            let (inv_arr, z01, z23) = match self.out {
                Out::U8 { inv, zero } if OUT_U8 => {
                    let (z0, z1, z2, z3) = (zero[0] as i16, zero[1] as i16, zero[2] as i16, zero[3] as i16);
                    (
                        *inv,
                        _mm256_setr_epi16(z0, z0, z0, z0, z1, z1, z1, z1, z0, z0, z0, z0, z1, z1, z1, z1),
                        _mm256_setr_epi16(z2, z2, z2, z2, z3, z3, z3, z3, z2, z2, z2, z2, z3, z3, z3, z3),
                    )
                }
                _ => ([0.0; MR], _mm256_setzero_si256(), _mm256_setzero_si256()),
            };
            let (rs_arr, rz_arr) = match self.res {
                Some((rs, rz)) if HAS_RES => (*rs, *rz),
                _ => ([0.0; MR], [0; MR]),
            };
            let mask_ff = _mm256_set1_epi32(0xFF);
            for k in 0..NR / 8 {
                let w = if HAS_RES {
                    _mm256_loadu_si256(res.add(8 * k) as *const __m256i)
                } else {
                    _mm256_setzero_si256()
                };
                let mut v = [_mm256_setzero_si256(); MR];
                for (r, vr) in v.iter_mut().enumerate() {
                    let a = _mm256_loadu_si256(acc[r].as_ptr().add(8 * k) as *const __m256i);
                    let t = _mm256_cvtepi32_ps(_mm256_sub_epi32(a, _mm256_set1_epi32(self.corr[r])));
                    let mut y =
                        _mm256_fmadd_ps(t, _mm256_set1_ps(self.scale[r]), _mm256_set1_ps(self.bias[r]));
                    if SILU {
                        y = silu_q8_avx2(y);
                    }
                    if HAS_RES {
                        let shifted = match r {
                            0 => w,
                            1 => _mm256_srli_epi32::<8>(w),
                            2 => _mm256_srli_epi32::<16>(w),
                            _ => _mm256_srli_epi32::<24>(w),
                        };
                        let b = _mm256_and_si256(shifted, mask_ff);
                        let d = _mm256_cvtepi32_ps(_mm256_sub_epi32(b, _mm256_set1_epi32(rz_arr[r])));
                        y = _mm256_fmadd_ps(_mm256_set1_ps(rs_arr[r]), d, y);
                    }
                    if OUT_U8 {
                        let y = _mm256_min_ps(
                            _mm256_max_ps(_mm256_mul_ps(y, _mm256_set1_ps(inv_arr[r])), lo),
                            hi,
                        );
                        *vr = _mm256_cvtps_epi32(y);
                    } else {
                        _mm256_storeu_ps(f[r].as_mut_ptr().add(8 * k), y);
                    }
                }
                if OUT_U8 {
                    let p01 = _mm256_adds_epi16(_mm256_packs_epi32(v[0], v[1]), z01);
                    let p23 = _mm256_adds_epi16(_mm256_packs_epi32(v[2], v[3]), z23);
                    let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), transpose);
                    _mm256_storeu_si256(q.add(8 * k) as *mut __m256i, bytes);
                }
            }
        }
    }

    /// [`apply_f32_full`] writing a full `NR`-word contiguous C4 tile directly to `q_dst`.
    ///
    /// # Safety
    /// `q_dst` is writable for `NR` words; `res_ptr` is readable for `NR` words when `self.res.is_some()`.
    pub(crate) unsafe fn apply_f32_full_u8_direct(
        &self,
        tile: &TileF32,
        res_ptr: *const u32,
        q_dst: *mut u32,
    ) {
        #[cfg(target_arch = "x86_64")]
        if uses_avx2() {
            let mut dummy = [[0f32; NR]; MR];
            // SAFETY: AVX2 and FMA were detected; pointers guaranteed by caller.
            return unsafe { self.apply_f32_avx2_ptr(tile, res_ptr, q_dst, &mut dummy) };
        }
        let mut resw = [0u32; NR];
        if self.res.is_some() {
            // SAFETY: `res_ptr` is readable for `NR` words.
            unsafe { std::ptr::copy_nonoverlapping(res_ptr, resw.as_mut_ptr(), NR) };
        }
        let mut qw = [0u32; NR];
        let mut dummy = [[0f32; NR]; MR];
        self.apply_f32(tile, NR, &resw, &mut qw, &mut dummy);
        // SAFETY: `q_dst` is writable for `NR` words.
        unsafe { std::ptr::copy_nonoverlapping(qw.as_ptr(), q_dst, NR) };
    }

    /// AVX2 epilogue from a pre-accumulated `TileF32`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn apply_f32_avx2(
        &self,
        tile: &TileF32,
        res: &[u32; NR],
        q: &mut [u32; NR],
        f: &mut [[f32; NR]; MR],
    ) {
        // SAFETY: forwarded.
        unsafe { self.apply_f32_avx2_ptr(tile, res.as_ptr(), q.as_mut_ptr(), f) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn apply_f32_avx2_ptr(
        &self,
        tile: &TileF32,
        res: *const u32,
        q: *mut u32,
        f: &mut [[f32; NR]; MR],
    ) {
        // SAFETY: pointer contract forwarded.
        unsafe {
            match (self.silu, self.res.is_some(), matches!(self.out, Out::U8 { .. })) {
                (true, false, true) => self.apply_f32_avx2_spec::<true, false, true>(tile, res, q, f),
                (false, false, true) => self.apply_f32_avx2_spec::<false, false, true>(tile, res, q, f),
                (true, true, true) => self.apply_f32_avx2_spec::<true, true, true>(tile, res, q, f),
                (false, true, true) => self.apply_f32_avx2_spec::<false, true, true>(tile, res, q, f),
                (true, false, false) => self.apply_f32_avx2_spec::<true, false, false>(tile, res, q, f),
                (false, false, false) => self.apply_f32_avx2_spec::<false, false, false>(tile, res, q, f),
                (true, true, false) => self.apply_f32_avx2_spec::<true, true, false>(tile, res, q, f),
                (false, true, false) => self.apply_f32_avx2_spec::<false, true, false>(tile, res, q, f),
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn apply_f32_avx2_spec<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
        &self,
        tile: &TileF32,
        res: *const u32,
        q: *mut u32,
        f: &mut [[f32; NR]; MR],
    ) {
        use std::arch::x86_64::*;
        // SAFETY: pointers valid for `NR` words / `MR x NR` elements per caller contract.
        unsafe {
            let (lo, hi) = (_mm256_set1_ps(-512.0), _mm256_set1_ps(512.0));
            let transpose = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14,
                3, 7, 11, 15,
            );
            let (inv_arr, z01, z23) = match self.out {
                Out::U8 { inv, zero } if OUT_U8 => {
                    let (z0, z1, z2, z3) = (zero[0] as i16, zero[1] as i16, zero[2] as i16, zero[3] as i16);
                    (
                        *inv,
                        _mm256_setr_epi16(z0, z0, z0, z0, z1, z1, z1, z1, z0, z0, z0, z0, z1, z1, z1, z1),
                        _mm256_setr_epi16(z2, z2, z2, z2, z3, z3, z3, z3, z2, z2, z2, z2, z3, z3, z3, z3),
                    )
                }
                _ => ([0.0; MR], _mm256_setzero_si256(), _mm256_setzero_si256()),
            };
            let (rs_arr, rz_arr) = match self.res {
                Some((rs, rz)) if HAS_RES => (*rs, *rz),
                _ => ([0.0; MR], [0; MR]),
            };
            let mask_ff = _mm256_set1_epi32(0xFF);
            for k in 0..NR / 8 {
                let w = if HAS_RES {
                    _mm256_loadu_si256(res.add(8 * k) as *const __m256i)
                } else {
                    _mm256_setzero_si256()
                };
                let mut v = [_mm256_setzero_si256(); MR];
                for (r, vr) in v.iter_mut().enumerate() {
                    let t = _mm256_loadu_ps(tile[r].as_ptr().add(8 * k));
                    let mut y = _mm256_add_ps(t, _mm256_set1_ps(self.bias[r]));
                    if SILU {
                        y = silu_q8_avx2(y);
                    }
                    if HAS_RES {
                        let shifted = match r {
                            0 => w,
                            1 => _mm256_srli_epi32::<8>(w),
                            2 => _mm256_srli_epi32::<16>(w),
                            _ => _mm256_srli_epi32::<24>(w),
                        };
                        let b = _mm256_and_si256(shifted, mask_ff);
                        let d = _mm256_cvtepi32_ps(_mm256_sub_epi32(b, _mm256_set1_epi32(rz_arr[r])));
                        y = _mm256_fmadd_ps(_mm256_set1_ps(rs_arr[r]), d, y);
                    }
                    if OUT_U8 {
                        let y = _mm256_min_ps(
                            _mm256_max_ps(_mm256_mul_ps(y, _mm256_set1_ps(inv_arr[r])), lo),
                            hi,
                        );
                        *vr = _mm256_cvtps_epi32(y);
                    } else {
                        _mm256_storeu_ps(f[r].as_mut_ptr().add(8 * k), y);
                    }
                }
                if OUT_U8 {
                    let p01 = _mm256_adds_epi16(_mm256_packs_epi32(v[0], v[1]), z01);
                    let p23 = _mm256_adds_epi16(_mm256_packs_epi32(v[2], v[3]), z23);
                    let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), transpose);
                    _mm256_storeu_si256(q.add(8 * k) as *mut __m256i, bytes);
                }
            }
        }
    }
}
