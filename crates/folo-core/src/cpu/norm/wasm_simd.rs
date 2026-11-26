//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) `layer_norm` and `rms_norm` forward and backward kernels,
//! bitwise identical to their scalar twins in `norm/mod.rs` and `norm/rms.rs` on all non-NaN inputs.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::ln_apply_scalar;
use super::ln_bwd_write_scalar;
use super::rms::{rms_apply_scalar, rms_bwd_write_scalar};

// ---------- LayerNorm forward ----------

pub(super) fn ln_apply_wasm(
    x: &[f32],
    y: &mut [f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    mean: f32,
    rstd: f32,
) {
    let n = x.len();
    assert_eq!(y.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    if let Some(b) = b {
        assert_eq!(b.len(), n);
    }
    let vm = f32x4_splat(mean);
    let vr = f32x4_splat(rstd);
    let mut i = 0usize;
    // SAFETY: slice lengths asserted above, loops check `i + 8 <= n` / `i + 4 <= n`.
    unsafe {
        match (w, b) {
            (Some(w), Some(b)) => {
                while i + 8 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    let n1 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i + 4) as *const v128), vm), vr);
                    let y0 = f32x4_add(
                        f32x4_mul(n0, v128_load(w.as_ptr().add(i) as *const v128)),
                        v128_load(b.as_ptr().add(i) as *const v128),
                    );
                    let y1 = f32x4_add(
                        f32x4_mul(n1, v128_load(w.as_ptr().add(i + 4) as *const v128)),
                        v128_load(b.as_ptr().add(i + 4) as *const v128),
                    );
                    v128_store(y.as_mut_ptr().add(i) as *mut v128, y0);
                    v128_store(y.as_mut_ptr().add(i + 4) as *mut v128, y1);
                    i += 8;
                }
                if i + 4 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    let y0 = f32x4_add(
                        f32x4_mul(n0, v128_load(w.as_ptr().add(i) as *const v128)),
                        v128_load(b.as_ptr().add(i) as *const v128),
                    );
                    v128_store(y.as_mut_ptr().add(i) as *mut v128, y0);
                    i += 4;
                }
            }
            (Some(w), None) => {
                while i + 8 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    let n1 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i + 4) as *const v128), vm), vr);
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_mul(n0, v128_load(w.as_ptr().add(i) as *const v128)),
                    );
                    v128_store(
                        y.as_mut_ptr().add(i + 4) as *mut v128,
                        f32x4_mul(n1, v128_load(w.as_ptr().add(i + 4) as *const v128)),
                    );
                    i += 8;
                }
                if i + 4 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_mul(n0, v128_load(w.as_ptr().add(i) as *const v128)),
                    );
                    i += 4;
                }
            }
            (None, Some(b)) => {
                while i + 8 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    let n1 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i + 4) as *const v128), vm), vr);
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_add(n0, v128_load(b.as_ptr().add(i) as *const v128)),
                    );
                    v128_store(
                        y.as_mut_ptr().add(i + 4) as *mut v128,
                        f32x4_add(n1, v128_load(b.as_ptr().add(i + 4) as *const v128)),
                    );
                    i += 8;
                }
                if i + 4 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_add(n0, v128_load(b.as_ptr().add(i) as *const v128)),
                    );
                    i += 4;
                }
            }
            (None, None) => {
                while i + 8 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    let n1 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i + 4) as *const v128), vm), vr);
                    v128_store(y.as_mut_ptr().add(i) as *mut v128, n0);
                    v128_store(y.as_mut_ptr().add(i + 4) as *mut v128, n1);
                    i += 8;
                }
                if i + 4 <= n {
                    let n0 = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(i) as *const v128), vm), vr);
                    v128_store(y.as_mut_ptr().add(i) as *mut v128, n0);
                    i += 4;
                }
            }
        }
    }
    if i < n {
        ln_apply_scalar(&x[i..], &mut y[i..], w.map(|s| &s[i..]), b.map(|s| &s[i..]), mean, rstd);
    }
}

// ---------- LayerNorm backward ----------

/// 16-lane Wasm SIMD128 `(sum(gw), sum(gw * x_hat), sum(x_hat))`, bitwise identical to `ln_bwd_sums_scalar`.
pub(super) fn ln_bwd_sums_wasm(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    mean: f32,
    rstd: f32,
) -> (f32, f32, f32) {
    let n = x.len();
    assert_eq!(g.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    let mut a1 = [0.0f32; 16];
    let mut a2 = [0.0f32; 16];
    let mut a3 = [0.0f32; 16];
    let vm = f32x4_splat(mean);
    let vr = f32x4_splat(rstd);
    let z = f32x4_splat(0.0);
    let mut s1 = [z; 4];
    let mut s2 = [z; 4];
    let mut s3 = [z; 4];
    let mut i = 0usize;
    // SAFETY: slice lengths asserted above, loops check `i + 16 <= n`, `a1..a3` hold 16 floats.
    unsafe {
        while i + 16 <= n {
            for k in 0..4 {
                let off = i + k * 4;
                let xh = f32x4_mul(f32x4_sub(v128_load(x.as_ptr().add(off) as *const v128), vm), vr);
                let gv = v128_load(g.as_ptr().add(off) as *const v128);
                let gw = match w {
                    Some(w) => f32x4_mul(gv, v128_load(w.as_ptr().add(off) as *const v128)),
                    None => gv,
                };
                s1[k] = f32x4_add(s1[k], gw);
                s2[k] = f32x4_add(s2[k], f32x4_mul(gw, xh));
                s3[k] = f32x4_add(s3[k], xh);
            }
            i += 16;
        }
        for k in 0..4 {
            v128_store(a1.as_mut_ptr().add(k * 4) as *mut v128, s1[k]);
            v128_store(a2.as_mut_ptr().add(k * 4) as *mut v128, s2[k]);
            v128_store(a3.as_mut_ptr().add(k * 4) as *mut v128, s3[k]);
        }
    }
    for k in 0..(n - i) {
        let xh = (x[i + k] - mean) * rstd;
        let gw = match w {
            Some(w) => g[i + k] * w[i + k],
            None => g[i + k],
        };
        a1[k] += gw;
        a2[k] += gw * xh;
        a3[k] += xh;
    }
    let (mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..16 {
        c1 += a1[k];
        c2 += a2[k];
        c3 += a3[k];
    }
    (c1, c2, c3)
}

#[allow(clippy::too_many_arguments)]
unsafe fn ln_bwd_write_kernel<const HAS_W: bool, const HAS_DW: bool, const HAS_DB: bool>(
    x: *const f32,
    g: *const f32,
    w: *const f32,
    dx: *mut f32,
    dw: *mut f32,
    db: *mut f32,
    n: usize,
    mean: f32,
    rstd: f32,
    c1_d: f32,
    c2_d: f32,
) -> usize {
    let vm = f32x4_splat(mean);
    let vr = f32x4_splat(rstd);
    let vc1 = f32x4_splat(c1_d);
    let vc2 = f32x4_splat(c2_d);
    let mut i = 0usize;
    // SAFETY: caller guarantees valid non-aliasing pointers of length `n`.
    unsafe {
        while i + 4 <= n {
            let xh = f32x4_mul(f32x4_sub(v128_load(x.add(i) as *const v128), vm), vr);
            let gv = v128_load(g.add(i) as *const v128);
            let gw = if HAS_W { f32x4_mul(gv, v128_load(w.add(i) as *const v128)) } else { gv };
            let adj = f32x4_add(vc1, f32x4_mul(xh, vc2));
            v128_store(dx.add(i) as *mut v128, f32x4_mul(vr, f32x4_sub(gw, adj)));
            if HAS_DW {
                let dwp = dw.add(i) as *mut v128;
                v128_store(dwp, f32x4_add(v128_load(dwp as *const v128), f32x4_mul(gv, xh)));
            }
            if HAS_DB {
                let dbp = db.add(i) as *mut v128;
                v128_store(dbp, f32x4_add(v128_load(dbp as *const v128), gv));
            }
            i += 4;
        }
    }
    i
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ln_bwd_write_wasm(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    mut db: Option<&mut [f32]>,
    mean: f32,
    rstd: f32,
    c1_d: f32,
    c2_d: f32,
) {
    let n = x.len();
    assert_eq!(g.len(), n);
    assert_eq!(dx.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    if let Some(ref dw) = dw {
        assert_eq!(dw.len(), n);
    }
    if let Some(ref db) = db {
        assert_eq!(db.len(), n);
    }
    let (has_w, has_dw, has_db) = (w.is_some(), dw.is_some(), db.is_some());
    let xp = x.as_ptr();
    let gp = g.as_ptr();
    let wp = w.map_or(std::ptr::null(), <[f32]>::as_ptr);
    let dxp = dx.as_mut_ptr();
    let dwp = dw.as_deref_mut().map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr);
    let dbp = db.as_deref_mut().map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr);
    // SAFETY: lengths asserted above and mutable slices are disjoint.
    let i = unsafe {
        match (has_w, has_dw, has_db) {
            (true, true, true) => {
                ln_bwd_write_kernel::<true, true, true>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (true, true, false) => {
                ln_bwd_write_kernel::<true, true, false>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (true, false, true) => {
                ln_bwd_write_kernel::<true, false, true>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (true, false, false) => ln_bwd_write_kernel::<true, false, false>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
            (false, true, true) => {
                ln_bwd_write_kernel::<false, true, true>(xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d)
            }
            (false, true, false) => ln_bwd_write_kernel::<false, true, false>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
            (false, false, true) => ln_bwd_write_kernel::<false, false, true>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
            (false, false, false) => ln_bwd_write_kernel::<false, false, false>(
                xp, gp, wp, dxp, dwp, dbp, n, mean, rstd, c1_d, c2_d,
            ),
        }
    };
    if i < n {
        ln_bwd_write_scalar(
            &x[i..],
            &g[i..],
            w.map(|s| &s[i..]),
            &mut dx[i..],
            dw.map(|s| &mut s[i..]),
            db.map(|s| &mut s[i..]),
            mean,
            rstd,
            c1_d,
            c2_d,
        );
    }
}

// ---------- RmsNorm forward & backward ----------

pub(super) fn rms_apply_wasm(x: &[f32], y: &mut [f32], w: Option<&[f32]>, rstd: f32) {
    let n = x.len();
    assert_eq!(y.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    let vr = f32x4_splat(rstd);
    let mut i = 0usize;
    // SAFETY: slice lengths asserted above, loops check `i + 8 <= n` / `i + 4 <= n`.
    unsafe {
        match w {
            Some(w) => {
                while i + 8 <= n {
                    let n0 = f32x4_mul(v128_load(x.as_ptr().add(i) as *const v128), vr);
                    let n1 = f32x4_mul(v128_load(x.as_ptr().add(i + 4) as *const v128), vr);
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_mul(n0, v128_load(w.as_ptr().add(i) as *const v128)),
                    );
                    v128_store(
                        y.as_mut_ptr().add(i + 4) as *mut v128,
                        f32x4_mul(n1, v128_load(w.as_ptr().add(i + 4) as *const v128)),
                    );
                    i += 8;
                }
                if i + 4 <= n {
                    let n0 = f32x4_mul(v128_load(x.as_ptr().add(i) as *const v128), vr);
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_mul(n0, v128_load(w.as_ptr().add(i) as *const v128)),
                    );
                    i += 4;
                }
            }
            None => {
                while i + 8 <= n {
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_mul(v128_load(x.as_ptr().add(i) as *const v128), vr),
                    );
                    v128_store(
                        y.as_mut_ptr().add(i + 4) as *mut v128,
                        f32x4_mul(v128_load(x.as_ptr().add(i + 4) as *const v128), vr),
                    );
                    i += 8;
                }
                if i + 4 <= n {
                    v128_store(
                        y.as_mut_ptr().add(i) as *mut v128,
                        f32x4_mul(v128_load(x.as_ptr().add(i) as *const v128), vr),
                    );
                    i += 4;
                }
            }
        }
    }
    if i < n {
        rms_apply_scalar(&x[i..], &mut y[i..], w.map(|s| &s[i..]), rstd);
    }
}

/// 16-lane Wasm SIMD128 `sum(gw * x_hat)`, bitwise identical to `rms_bwd_c_scalar`.
pub(super) fn rms_bwd_c_wasm(x: &[f32], g: &[f32], w: Option<&[f32]>, rstd: f32) -> f32 {
    let n = x.len();
    assert_eq!(g.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    let mut acc = [0.0f32; 16];
    let vr = f32x4_splat(rstd);
    let mut s = [f32x4_splat(0.0); 4];
    let mut i = 0usize;
    // SAFETY: slice lengths asserted above, loops check `i + 16 <= n`, `acc` holds 16 floats.
    unsafe {
        while i + 16 <= n {
            for (k, sk) in s.iter_mut().enumerate() {
                let off = i + k * 4;
                let xh = f32x4_mul(v128_load(x.as_ptr().add(off) as *const v128), vr);
                let gv = v128_load(g.as_ptr().add(off) as *const v128);
                let gw = match w {
                    Some(w) => f32x4_mul(gv, v128_load(w.as_ptr().add(off) as *const v128)),
                    None => gv,
                };
                *sk = f32x4_add(*sk, f32x4_mul(gw, xh));
            }
            i += 16;
        }
        for (k, &sk) in s.iter().enumerate() {
            v128_store(acc.as_mut_ptr().add(k * 4) as *mut v128, sk);
        }
    }
    for k in 0..(n - i) {
        let xh = x[i + k] * rstd;
        let gw = match w {
            Some(w) => g[i + k] * w[i + k],
            None => g[i + k],
        };
        acc[k] += gw * xh;
    }
    let mut out = 0.0f32;
    for a in acc {
        out += a;
    }
    out
}

#[allow(clippy::too_many_arguments)]
unsafe fn rms_bwd_write_kernel<const HAS_W: bool, const HAS_DW: bool>(
    x: *const f32,
    g: *const f32,
    w: *const f32,
    dx: *mut f32,
    dw: *mut f32,
    n: usize,
    rstd: f32,
    c_d: f32,
) -> usize {
    let vr = f32x4_splat(rstd);
    let vc = f32x4_splat(c_d);
    let mut i = 0usize;
    // SAFETY: caller guarantees valid non-aliasing pointers of length `n`.
    unsafe {
        while i + 4 <= n {
            let xh = f32x4_mul(v128_load(x.add(i) as *const v128), vr);
            let gv = v128_load(g.add(i) as *const v128);
            let gw = if HAS_W { f32x4_mul(gv, v128_load(w.add(i) as *const v128)) } else { gv };
            v128_store(dx.add(i) as *mut v128, f32x4_mul(vr, f32x4_sub(gw, f32x4_mul(xh, vc))));
            if HAS_DW {
                let dwp = dw.add(i) as *mut v128;
                v128_store(dwp, f32x4_add(v128_load(dwp as *const v128), f32x4_mul(gv, xh)));
            }
            i += 4;
        }
    }
    i
}

pub(super) fn rms_bwd_write_wasm(
    x: &[f32],
    g: &[f32],
    w: Option<&[f32]>,
    dx: &mut [f32],
    mut dw: Option<&mut [f32]>,
    rstd: f32,
    c_d: f32,
) {
    let n = x.len();
    assert_eq!(g.len(), n);
    assert_eq!(dx.len(), n);
    if let Some(w) = w {
        assert_eq!(w.len(), n);
    }
    if let Some(ref dw) = dw {
        assert_eq!(dw.len(), n);
    }
    let (has_w, has_dw) = (w.is_some(), dw.is_some());
    let xp = x.as_ptr();
    let gp = g.as_ptr();
    let wp = w.map_or(std::ptr::null(), <[f32]>::as_ptr);
    let dxp = dx.as_mut_ptr();
    let dwp = dw.as_deref_mut().map_or(std::ptr::null_mut(), <[f32]>::as_mut_ptr);
    // SAFETY: lengths asserted above and mutable slices are disjoint.
    let i = unsafe {
        match (has_w, has_dw) {
            (true, true) => rms_bwd_write_kernel::<true, true>(xp, gp, wp, dxp, dwp, n, rstd, c_d),
            (true, false) => rms_bwd_write_kernel::<true, false>(xp, gp, wp, dxp, dwp, n, rstd, c_d),
            (false, true) => rms_bwd_write_kernel::<false, true>(xp, gp, wp, dxp, dwp, n, rstd, c_d),
            (false, false) => rms_bwd_write_kernel::<false, false>(xp, gp, wp, dxp, dwp, n, rstd, c_d),
        }
    };
    if i < n {
        rms_bwd_write_scalar(
            &x[i..],
            &g[i..],
            w.map(|s| &s[i..]),
            &mut dx[i..],
            dw.map(|s| &mut s[i..]),
            rstd,
            c_d,
        );
    }
}
