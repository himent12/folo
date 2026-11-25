//! Vectorized CPU `f32` kernels (AVX2 + scalar fallback) for optimizers, Muon orthogonalisation,
//! gradient accumulation / clipping, and AMP unscaling.
//!
//! Every AVX2 kernel intentionally omits `fma` so every lane executes separate IEEE-754 multiply and
//! add/subtract instructions, matching the scalar reference bit-for-bit on all non-NaN IEEE-754 inputs
//! (same contract as `crate::ema`).

mod vec_ops;

#[cfg(test)]
mod tests;

pub(crate) use vec_ops::{
    axpby_slice, muon_dw_polar_fused_slice, muon_dw_polar_step_slice, muon_momentum_nesterov_slice,
    ortho_normalize_batch, scale_slice, sum_sq_f32_slice, sum_sq_f64_slice, unscale_check_slice,
    unscale_check_sum_sq_slice,
};

/// Coupled SGD update on contiguous CPU slices:
/// ```text
/// d = g + wd * p               (when wd != 0)
/// b = mu * b + d               (when buf is Some and mu != 0)
/// u = d + mu * b | b           (Nesterov vs plain)
/// p = p + (-lr) * u
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgd_coupled_slice(
    p: &mut [f32],
    g: &[f32],
    buf: Option<&mut [f32]>,
    lr: f32,
    mu: f32,
    wd: f32,
    nesterov: bool,
) {
    assert_eq!(p.len(), g.len());
    let has_wd = wd != 0.0;
    match buf {
        Some(b) if mu != 0.0 => {
            assert_eq!(p.len(), b.len());
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 checked above; slice lengths verified equal.
                unsafe {
                    match (has_wd, nesterov) {
                        (false, false) => sgd_coupled_mom_avx2::<false, false>(p, g, b, lr, mu, wd),
                        (false, true) => sgd_coupled_mom_avx2::<false, true>(p, g, b, lr, mu, wd),
                        (true, false) => sgd_coupled_mom_avx2::<true, false>(p, g, b, lr, mu, wd),
                        (true, true) => sgd_coupled_mom_avx2::<true, true>(p, g, b, lr, mu, wd),
                    }
                }
                return;
            }
            sgd_coupled_mom_scalar(p, g, b, lr, mu, wd, has_wd, nesterov);
        }
        _ => {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 checked above; slice lengths verified equal.
                unsafe {
                    if has_wd {
                        sgd_coupled_nomom_avx2::<true>(p, g, lr, wd);
                    } else {
                        sgd_coupled_nomom_avx2::<false>(p, g, lr, wd);
                    }
                }
                return;
            }
            sgd_coupled_nomom_scalar(p, g, lr, wd, has_wd);
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sgd_coupled_mom_scalar(
    p: &mut [f32],
    g: &[f32],
    b: &mut [f32],
    lr: f32,
    mu: f32,
    wd: f32,
    has_wd: bool,
    nesterov: bool,
) {
    let neg_lr = -lr;
    for ((p, &g), b) in p.iter_mut().zip(g).zip(b) {
        let d = if has_wd { g + wd * *p } else { g };
        *b = mu * *b + d;
        let u = if nesterov { d + mu * *b } else { *b };
        *p += neg_lr * u;
    }
}

#[inline]
pub(super) fn sgd_coupled_nomom_scalar(p: &mut [f32], g: &[f32], lr: f32, wd: f32, has_wd: bool) {
    let neg_lr = -lr;
    for (p, &g) in p.iter_mut().zip(g) {
        let d = if has_wd { g + wd * *p } else { g };
        *p += neg_lr * d;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sgd_coupled_mom_avx2<const HAS_WD: bool, const NESTEROV: bool>(
    p: &mut [f32],
    g: &[f32],
    b: &mut [f32],
    lr: f32,
    mu: f32,
    wd: f32,
) {
    use std::arch::x86_64::*;

    let n = p.len();
    let v_neg_lr = _mm256_set1_ps(-lr);
    let v_mu = _mm256_set1_ps(mu);
    let v_wd = _mm256_set1_ps(wd);
    let p_ptr = p.as_mut_ptr();
    let g_ptr = g.as_ptr();
    let b_ptr = b.as_mut_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let vp = _mm256_loadu_ps(p_ptr.add($off));
            let vg = _mm256_loadu_ps(g_ptr.add($off));
            let vb = _mm256_loadu_ps(b_ptr.add($off));
            let vd = if HAS_WD { _mm256_add_ps(vg, _mm256_mul_ps(v_wd, vp)) } else { vg };
            let nb = _mm256_add_ps(_mm256_mul_ps(v_mu, vb), vd);
            let vu = if NESTEROV { _mm256_add_ps(vd, _mm256_mul_ps(v_mu, nb)) } else { nb };
            let np = _mm256_add_ps(vp, _mm256_mul_ps(v_neg_lr, vu));
            _mm256_storeu_ps(b_ptr.add($off), nb);
            _mm256_storeu_ps(p_ptr.add($off), np);
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `p`, `g`, `b` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `p`, `g`, `b` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    sgd_coupled_mom_scalar(&mut p[i..], &g[i..], &mut b[i..], lr, mu, wd, HAS_WD, NESTEROV);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sgd_coupled_nomom_avx2<const HAS_WD: bool>(p: &mut [f32], g: &[f32], lr: f32, wd: f32) {
    use std::arch::x86_64::*;

    let n = p.len();
    let v_neg_lr = _mm256_set1_ps(-lr);
    let v_wd = _mm256_set1_ps(wd);
    let p_ptr = p.as_mut_ptr();
    let g_ptr = g.as_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let vp = _mm256_loadu_ps(p_ptr.add($off));
            let vg = _mm256_loadu_ps(g_ptr.add($off));
            let vd = if HAS_WD { _mm256_add_ps(vg, _mm256_mul_ps(v_wd, vp)) } else { vg };
            let np = _mm256_add_ps(vp, _mm256_mul_ps(v_neg_lr, vd));
            _mm256_storeu_ps(p_ptr.add($off), np);
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `p`, `g` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `p`, `g` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    sgd_coupled_nomom_scalar(&mut p[i..], &g[i..], lr, wd, HAS_WD);
}

/// Decoupled / cautious SGD update on contiguous CPU slices:
/// ```text
/// b = mu * b + g                              (when buf is Some and mu != 0)
/// u = g + mu * b | b | g
/// decay = wd_step * p if wd_step != 0 && (!cautious || u * p > 0) else 0
/// p = p - lr * u - decay
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgd_decoupled_slice(
    p: &mut [f32],
    g: &[f32],
    buf: Option<&mut [f32]>,
    lr: f32,
    mu: f32,
    wd_step: f32,
    nesterov: bool,
    cautious: bool,
) {
    assert_eq!(p.len(), g.len());
    let has_wd = wd_step != 0.0;
    match buf {
        Some(b) if mu != 0.0 => {
            assert_eq!(p.len(), b.len());
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 checked above; slice lengths verified equal.
                unsafe {
                    match (has_wd, cautious, nesterov) {
                        (false, _, false) => {
                            sgd_decoupled_mom_avx2::<false, false, false>(p, g, b, lr, mu, wd_step)
                        }
                        (false, _, true) => {
                            sgd_decoupled_mom_avx2::<false, false, true>(p, g, b, lr, mu, wd_step)
                        }
                        (true, false, false) => {
                            sgd_decoupled_mom_avx2::<true, false, false>(p, g, b, lr, mu, wd_step)
                        }
                        (true, false, true) => {
                            sgd_decoupled_mom_avx2::<true, false, true>(p, g, b, lr, mu, wd_step)
                        }
                        (true, true, false) => {
                            sgd_decoupled_mom_avx2::<true, true, false>(p, g, b, lr, mu, wd_step)
                        }
                        (true, true, true) => {
                            sgd_decoupled_mom_avx2::<true, true, true>(p, g, b, lr, mu, wd_step)
                        }
                    }
                }
                return;
            }
            sgd_decoupled_mom_scalar(p, g, b, lr, mu, wd_step, has_wd, cautious, nesterov);
        }
        _ => {
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 checked above; slice lengths verified equal.
                unsafe {
                    match (has_wd, cautious) {
                        (false, _) => sgd_decoupled_nomom_avx2::<false, false>(p, g, lr, wd_step),
                        (true, false) => sgd_decoupled_nomom_avx2::<true, false>(p, g, lr, wd_step),
                        (true, true) => sgd_decoupled_nomom_avx2::<true, true>(p, g, lr, wd_step),
                    }
                }
                return;
            }
            sgd_decoupled_nomom_scalar(p, g, lr, wd_step, has_wd, cautious);
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn sgd_decoupled_mom_scalar(
    p: &mut [f32],
    g: &[f32],
    b: &mut [f32],
    lr: f32,
    mu: f32,
    wd_step: f32,
    has_wd: bool,
    cautious: bool,
    nesterov: bool,
) {
    for ((p, &g), b) in p.iter_mut().zip(g).zip(b) {
        *b = mu * *b + g;
        let u = if nesterov { g + mu * *b } else { *b };
        let decay = if has_wd && (!cautious || u * *p > 0.0) { wd_step * *p } else { 0.0 };
        *p = *p - lr * u - decay;
    }
}

#[inline]
pub(super) fn sgd_decoupled_nomom_scalar(
    p: &mut [f32],
    g: &[f32],
    lr: f32,
    wd_step: f32,
    has_wd: bool,
    cautious: bool,
) {
    for (p, &g) in p.iter_mut().zip(g) {
        let decay = if has_wd && (!cautious || g * *p > 0.0) { wd_step * *p } else { 0.0 };
        *p = *p - lr * g - decay;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sgd_decoupled_mom_avx2<const HAS_WD: bool, const CAUTIOUS: bool, const NESTEROV: bool>(
    p: &mut [f32],
    g: &[f32],
    b: &mut [f32],
    lr: f32,
    mu: f32,
    wd_step: f32,
) {
    use std::arch::x86_64::*;

    let n = p.len();
    let v_lr = _mm256_set1_ps(lr);
    let v_mu = _mm256_set1_ps(mu);
    let v_wd = _mm256_set1_ps(wd_step);
    let v_zero = _mm256_setzero_ps();
    let p_ptr = p.as_mut_ptr();
    let g_ptr = g.as_ptr();
    let b_ptr = b.as_mut_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let vp = _mm256_loadu_ps(p_ptr.add($off));
            let vg = _mm256_loadu_ps(g_ptr.add($off));
            let vb = _mm256_loadu_ps(b_ptr.add($off));
            let nb = _mm256_add_ps(_mm256_mul_ps(v_mu, vb), vg);
            let vu = if NESTEROV { _mm256_add_ps(vg, _mm256_mul_ps(v_mu, nb)) } else { nb };
            let sub = _mm256_sub_ps(vp, _mm256_mul_ps(v_lr, vu));
            let np = if HAS_WD {
                let raw = _mm256_mul_ps(v_wd, vp);
                let decay = if CAUTIOUS {
                    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(vu, vp), v_zero);
                    _mm256_and_ps(mask, raw)
                } else {
                    raw
                };
                _mm256_sub_ps(sub, decay)
            } else {
                sub
            };
            _mm256_storeu_ps(b_ptr.add($off), nb);
            _mm256_storeu_ps(p_ptr.add($off), np);
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `p`, `g`, `b` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `p`, `g`, `b` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    sgd_decoupled_mom_scalar(&mut p[i..], &g[i..], &mut b[i..], lr, mu, wd_step, HAS_WD, CAUTIOUS, NESTEROV);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sgd_decoupled_nomom_avx2<const HAS_WD: bool, const CAUTIOUS: bool>(
    p: &mut [f32],
    g: &[f32],
    lr: f32,
    wd_step: f32,
) {
    use std::arch::x86_64::*;

    let n = p.len();
    let v_lr = _mm256_set1_ps(lr);
    let v_wd = _mm256_set1_ps(wd_step);
    let v_zero = _mm256_setzero_ps();
    let p_ptr = p.as_mut_ptr();
    let g_ptr = g.as_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let vp = _mm256_loadu_ps(p_ptr.add($off));
            let vg = _mm256_loadu_ps(g_ptr.add($off));
            let sub = _mm256_sub_ps(vp, _mm256_mul_ps(v_lr, vg));
            let np = if HAS_WD {
                let raw = _mm256_mul_ps(v_wd, vp);
                let decay = if CAUTIOUS {
                    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_mul_ps(vg, vp), v_zero);
                    _mm256_and_ps(mask, raw)
                } else {
                    raw
                };
                _mm256_sub_ps(sub, decay)
            } else {
                sub
            };
            _mm256_storeu_ps(p_ptr.add($off), np);
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `p`, `g` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `p`, `g` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    sgd_decoupled_nomom_scalar(&mut p[i..], &g[i..], lr, wd_step, HAS_WD, CAUTIOUS);
}

/// AdamW update on contiguous CPU slices (PyTorch evaluation order).
#[allow(clippy::too_many_arguments)]
pub(crate) fn adamw_slice(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    decay: Option<f32>,
    b1: f32,
    one_minus_b1: f32,
    b2: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
) {
    assert_eq!(p.len(), g.len());
    assert_eq!(p.len(), m.len());
    assert_eq!(p.len(), v.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 checked above; slice lengths verified equal.
        unsafe {
            if let Some(d) = decay {
                adamw_avx2::<true>(
                    p,
                    g,
                    m,
                    v,
                    d,
                    b1,
                    one_minus_b1,
                    b2,
                    one_minus_b2,
                    step_size,
                    bc2_sqrt,
                    eps,
                );
            } else {
                adamw_avx2::<false>(
                    p,
                    g,
                    m,
                    v,
                    1.0,
                    b1,
                    one_minus_b1,
                    b2,
                    one_minus_b2,
                    step_size,
                    bc2_sqrt,
                    eps,
                );
            }
        }
        return;
    }
    adamw_scalar(p, g, m, v, decay, b1, one_minus_b1, b2, one_minus_b2, step_size, bc2_sqrt, eps);
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn adamw_scalar(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    decay: Option<f32>,
    b1: f32,
    one_minus_b1: f32,
    b2: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
) {
    for (((p, &g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
        if let Some(d) = decay {
            *p *= d;
        }
        *m = b1 * *m + one_minus_b1 * g;
        *v = b2 * *v + one_minus_b2 * (g * g);
        let denom = v.sqrt() / bc2_sqrt + eps;
        *p += -step_size * (*m / denom);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn adamw_avx2<const HAS_DECAY: bool>(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    decay: f32,
    b1: f32,
    one_minus_b1: f32,
    b2: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
) {
    use std::arch::x86_64::*;

    let n = p.len();
    let v_decay = _mm256_set1_ps(decay);
    let v_b1 = _mm256_set1_ps(b1);
    let v_1mb1 = _mm256_set1_ps(one_minus_b1);
    let v_b2 = _mm256_set1_ps(b2);
    let v_1mb2 = _mm256_set1_ps(one_minus_b2);
    let v_neg_step = _mm256_set1_ps(-step_size);
    let v_bc2_sqrt = _mm256_set1_ps(bc2_sqrt);
    let v_eps = _mm256_set1_ps(eps);

    let p_ptr = p.as_mut_ptr();
    let g_ptr = g.as_ptr();
    let m_ptr = m.as_mut_ptr();
    let v_ptr = v.as_mut_ptr();
    let mut i = 0;

    macro_rules! step8 {
        ($off:expr) => {{
            let mut vp = _mm256_loadu_ps(p_ptr.add($off));
            let vg = _mm256_loadu_ps(g_ptr.add($off));
            let vm = _mm256_loadu_ps(m_ptr.add($off));
            let vv = _mm256_loadu_ps(v_ptr.add($off));
            if HAS_DECAY {
                vp = _mm256_mul_ps(vp, v_decay);
            }
            let nm = _mm256_add_ps(_mm256_mul_ps(v_b1, vm), _mm256_mul_ps(v_1mb1, vg));
            let g2 = _mm256_mul_ps(vg, vg);
            let nv = _mm256_add_ps(_mm256_mul_ps(v_b2, vv), _mm256_mul_ps(v_1mb2, g2));
            let denom = _mm256_add_ps(_mm256_div_ps(_mm256_sqrt_ps(nv), v_bc2_sqrt), v_eps);
            let np = _mm256_add_ps(vp, _mm256_mul_ps(v_neg_step, _mm256_div_ps(nm, denom)));
            _mm256_storeu_ps(m_ptr.add($off), nm);
            _mm256_storeu_ps(v_ptr.add($off), nv);
            _mm256_storeu_ps(p_ptr.add($off), np);
        }};
    }

    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` and `p`, `g`, `m`, `v` have length `n`.
        unsafe {
            step8!(i);
            step8!(i + 8);
            step8!(i + 16);
            step8!(i + 24);
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n` and `p`, `g`, `m`, `v` have length `n`.
        unsafe {
            step8!(i);
        }
        i += 8;
    }
    adamw_scalar(
        &mut p[i..],
        &g[i..],
        &mut m[i..],
        &mut v[i..],
        HAS_DECAY.then_some(decay),
        b1,
        one_minus_b1,
        b2,
        one_minus_b2,
        step_size,
        bc2_sqrt,
        eps,
    );
}
