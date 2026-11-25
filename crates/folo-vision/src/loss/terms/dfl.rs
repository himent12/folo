//! 1D Optimal-Transport cumulative tail-barrier DFL term (`OT-DFL`).

use folo_core::{DType, Result, Tensor};

/// Overflow/underflow-safe `exp(a - b)` in `f64` for `a <= b` down to `-700.0`.
#[inline(always)]
pub(super) fn exp_neg_f64(a: f32, b: f32) -> f64 {
    let d = a as f64 - b as f64;
    if d > -700.0 { d.exp() } else { 0.0 }
}

/// Exact `O(R)` shifted `(max, sum_exp)` 1D optimal-transport cumulative tail-barrier DFL term:
/// `(dfl_gain * ot_weight / tss) * Σ_{n, side} side_wt * Σ_{k != j} softplus(log Z_excl(k) - log Z_ret(k))`
/// for `logits: [N, 4, R]`, accumulating retained and excluded masses separately in `f64` so tiny excluded masses
/// (e.g. `[-89, 0, 0, -1000]`), extreme logits (`[-1e38; 5, 0]`, `[-1e8, -1e8, -1e8, 0]`), and multi-positive sums
/// never suffer cancellation or premature overflow.
pub(super) fn ot_dfl_tail(
    logits: &Tensor,
    left_bins: &[usize],
    side_wt: &[f32],
    ot_weight: f32,
    tss: f32,
    dfl_gain: f32,
) -> Result<Tensor> {
    let (n, sides, r) = logits.dims3()?;
    debug_assert_eq!(sides, 4);
    let dev = logits.device();
    if r <= 2 || ot_weight <= 0.0 || n == 0 {
        return Tensor::zeros_with([], DType::F32, dev);
    }
    let softplus_sig = |m_e: f32, a_e: f64, m_r: f32, a_r: f64| -> (f64, f64) {
        let delta = (m_e as f64 - m_r as f64) + (a_e.ln() - a_r.ln());
        if delta >= 0.0 {
            let e = (-delta).exp();
            (delta + e.ln_1p(), 1.0 / (1.0 + e))
        } else if delta > -700.0 {
            let e = delta.exp();
            (e.ln_1p(), e / (1.0 + e))
        } else {
            (0.0, 0.0)
        }
    };
    let data = logits.to_vec::<f32>()?;
    let mut grad_unit = vec![0.0f64; n * 4 * r];
    let mut total = 0.0f64;
    let mut pref_m = vec![0.0f32; r];
    let mut pref_a = vec![0.0f64; r];
    let mut m_ret = vec![0.0f32; r];
    let mut ret_mant = vec![0.0f64; r];
    let mut m_exc = vec![0.0f32; r];
    let mut exc_mant = vec![0.0f64; r];
    for idx in 0..n * 4 {
        let wt = side_wt[idx] as f64;
        if wt == 0.0 {
            continue;
        }
        let j = left_bins[idx];
        let row = &data[idx * r..(idx + 1) * r];
        let mut pm = row[0];
        let mut pa = 1.0f64;
        pref_m[0] = pm;
        pref_a[0] = pa;
        for k in 1..r - 1 {
            let x = row[k];
            if x > pm {
                pa = pa * exp_neg_f64(pm, x) + 1.0;
                pm = x;
            } else {
                pa += exp_neg_f64(x, pm);
            }
            pref_m[k] = pm;
            pref_a[k] = pa;
        }
        let mut l_tail = 0.0f64;
        let mut sm = row[r - 1];
        let mut sa = 1.0f64;
        for step in 0..r - 1 {
            let k = (r - 2) - step;
            if step > 0 {
                let x = row[k + 1];
                if x > sm {
                    sa = sa * exp_neg_f64(sm, x) + 1.0;
                    sm = x;
                } else {
                    sa += exp_neg_f64(x, sm);
                }
            }
            if k < j {
                let (sp, sig) = softplus_sig(pref_m[k], pref_a[k], sm, sa);
                l_tail += sp;
                m_exc[k] = pref_m[k];
                exc_mant[k] = sig / pref_a[k];
                m_ret[k] = sm;
                ret_mant[k] = sig / sa;
            } else if k > j {
                let (sp, sig) = softplus_sig(sm, sa, pref_m[k], pref_a[k]);
                l_tail += sp;
                m_ret[k] = pref_m[k];
                ret_mant[k] = sig / pref_a[k];
                m_exc[k] = sm;
                exc_mant[k] = sig / sa;
            }
        }
        if j > 0 {
            for t in 1..j {
                ret_mant[t] += ret_mant[t - 1] * exp_neg_f64(m_ret[t], m_ret[t - 1]);
            }
            let mut t = j - 1;
            while t > 0 {
                t -= 1;
                exc_mant[t] += exc_mant[t + 1] * exp_neg_f64(m_exc[t], m_exc[t + 1]);
            }
        }
        if j + 2 < r {
            let mut t = r - 2;
            while t > j + 1 {
                t -= 1;
                ret_mant[t] += ret_mant[t + 1] * exp_neg_f64(m_ret[t], m_ret[t + 1]);
            }
            for t in j + 2..r - 1 {
                exc_mant[t] += exc_mant[t - 1] * exp_neg_f64(m_exc[t], m_exc[t - 1]);
            }
        }
        total += wt * l_tail;
        let out_row = &mut grad_unit[idx * r..(idx + 1) * r];
        for (k, out_k) in out_row.iter_mut().enumerate() {
            let xk = row[k];
            let mut d_ot = 0.0f64;
            if j > 0 {
                if k < j {
                    d_ot += exp_neg_f64(xk, m_exc[k]) * exc_mant[k];
                }
                if k > 0 {
                    let ti = (k - 1).min(j - 1);
                    d_ot -= exp_neg_f64(xk, m_ret[ti]) * ret_mant[ti];
                }
            }
            if j + 2 < r {
                if k + 1 < r {
                    let ti = k.max(j + 1);
                    d_ot -= exp_neg_f64(xk, m_ret[ti]) * ret_mant[ti];
                }
                if k > j + 1 {
                    let ti = (k - 1).min(r - 2);
                    d_ot += exp_neg_f64(xk, m_exc[ti]) * exc_mant[ti];
                }
            }
            *out_k = wt * d_ot;
        }
    }
    let out_val = ((total / tss as f64) * ot_weight as f64 * dfl_gain as f64) as f32;
    let out = Tensor::scalar(out_val).to_device(dev)?;
    let shape = [n, 4, r];
    Tensor::custom_op("ot_dfl_tail", &[logits], out, move |g| {
        let gv = g.item::<f32>()? as f64;
        let scale = gv * (dfl_gain as f64 / tss as f64) * (ot_weight as f64);
        let scaled: Vec<f32> = grad_unit.iter().map(|&u| (u * scale) as f32).collect();
        Ok(vec![Some(Tensor::from_vec(scaled, shape)?.to_device(dev)?)])
    })
}

/// Closed-form 1D excess CRPS / squared Energy Distance DFL term (`CRPS-DFL`):
/// `(dfl_gain * crps_weight / tss) * Σ_{n, side} side_wt * Σ_{k=0..R-2} e_k²`
/// on `p = softmax(z)` with target bins `(j, w_l, w_r = 1 - w_l)`, computing cumulative masses from both sides
/// (`F_k = Σ_{i<=k} p_i` for `k <= j`, `S_k = Σ_{i>k} p_i` for `k > j`) so tails never cancel. Works for any `R >= 2`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn crps_dfl(
    logits: &Tensor,
    left_bins: &[usize],
    left_w: &[f32],
    side_wt: &[f32],
    crps_weight: f32,
    tss: f32,
    dfl_gain: f32,
) -> Result<Tensor> {
    let (n, sides, r) = logits.dims3()?;
    debug_assert_eq!(sides, 4);
    let dev = logits.device();
    if r < 2 || crps_weight <= 0.0 || n == 0 {
        return Tensor::zeros_with([], DType::F32, dev);
    }
    let data = logits.to_vec::<f32>()?;
    let mut grad_unit = vec![0.0f64; n * 4 * r];
    let mut total = 0.0f64;
    let mut p_row = vec![0.0f64; r];
    let mut e_cdf = vec![0.0f64; r - 1];
    let mut g_p = vec![0.0f64; r];
    for idx in 0..n * 4 {
        let wt = side_wt[idx] as f64;
        if wt == 0.0 {
            continue;
        }
        let j = left_bins[idx];
        let wl = left_w[idx] as f64;
        let wr = 1.0 - wl;
        let row = &data[idx * r..(idx + 1) * r];
        let m = row.iter().copied().fold(row[0], f32::max);
        let mut z = 0.0f64;
        for (k, &x) in row.iter().enumerate() {
            let e = exp_neg_f64(x, m);
            p_row[k] = e;
            z += e;
        }
        for pk in p_row.iter_mut() {
            *pk /= z;
        }
        let mut f_mass = 0.0f64;
        for k in 0..=j {
            f_mass += p_row[k];
            if k < j {
                e_cdf[k] = f_mass;
            }
        }
        let mut s_mass = 0.0f64;
        for k_rev in (j + 1..r).rev() {
            s_mass += p_row[k_rev];
            let k = k_rev - 1;
            if k > j {
                e_cdf[k] = -s_mass;
            }
        }
        e_cdf[j] = if wl > 0.5 { wr - s_mass } else { f_mass - wl };

        let mut crps_side = 0.0f64;
        for &ek in &e_cdf {
            crps_side += ek * ek;
        }
        total += wt * crps_side;

        g_p[r - 1] = 0.0;
        let mut g_acc = 0.0f64;
        let mut gbar = 0.0f64;
        for k in (0..r - 1).rev() {
            g_acc += 2.0 * e_cdf[k];
            g_p[k] = g_acc;
            gbar += p_row[k] * g_acc;
        }
        let out_row = &mut grad_unit[idx * r..(idx + 1) * r];
        for k in 0..r {
            out_row[k] = wt * p_row[k] * (g_p[k] - gbar);
        }
    }
    let out_val = ((total / tss as f64) * crps_weight as f64 * dfl_gain as f64) as f32;
    let out = Tensor::scalar(out_val).to_device(dev)?;
    let shape = [n, 4, r];
    Tensor::custom_op("crps_dfl", &[logits], out, move |g| {
        let gv = g.item::<f32>()? as f64;
        let scale = gv * (dfl_gain as f64 / tss as f64) * (crps_weight as f64);
        let scaled: Vec<f32> = grad_unit.iter().map(|&u| (u * scale) as f32).collect();
        Ok(vec![Some(Tensor::from_vec(scaled, shape)?.to_device(dev)?)])
    })
}
