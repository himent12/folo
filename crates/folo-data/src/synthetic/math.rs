//! Sine, cosine and powers of two for the synthetic shapes, from IEEE basic operations only (`+ - * /`, `round`).
//!
//! The platform's libm is not correctly rounded, and its last bits differ between systems. Worse, the compiler
//! folds a call whose argument it can see into the correctly rounded value, so whether a vertex came from the
//! compiler or from libm depended on inlining, that is on the build profile: release and debug builds drew a
//! square 1 ulp apart and wrote different images and labels for one seed. These functions give the same bits
//! wherever they run and however they are compiled. The generator uses no other approximation from libm: what it
//! still calls (`sqrt`, `floor`, `round`, the remainder of `rem_euclid`) is exactly specified, the same everywhere.

use std::f64::consts::{FRAC_2_PI, LN_2};

/// `π/2` split in two (fdlibm's `pio2_1` / `pio2_1t`): the first has 33 significant bits, so `k * PIO2_HI` is exact
/// for the `|k| < 2^20` an argument below about 1.6 million gives.
const PIO2_HI: f64 = f64::from_bits(0x3FF9_21FB_5440_0000);
const PIO2_LO: f64 = f64::from_bits(0x3DD0_B461_1A62_6331);

/// `(sin x, cos x)` for finite `x`: for `|x| < 1e6` (the generator's stripes stay below 1000 at 8192 px) each within
/// an ulp of the true value, almost always the correctly rounded one (the work is done in `f64`); beyond, less
/// accurate, still the same bits everywhere.
pub(crate) fn sin_cos(x: f32) -> (f32, f32) {
    debug_assert!(x.is_finite(), "sin_cos({x})");
    let x = x as f64;
    // x = k π/2 + r with |r| <= π/4, then the Taylor series of both on r (the last term is below 1e-24).
    let k = (x * FRAC_2_PI).round();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    let r2 = r * r;
    let (mut s, mut ts) = (r, r);
    let (mut c, mut tc) = (1.0, 1.0);
    for i in 1..=11 {
        let n = (2 * i) as f64;
        ts *= -r2 / (n * (n + 1.0));
        tc *= -r2 / ((n - 1.0) * n);
        s += ts;
        c += tc;
    }
    let (s, c) = match (k as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    };
    (s as f32, c as f32)
}

/// `2^x` for `|x| <= 126`, within an ulp (almost always correctly rounded).
pub(crate) fn exp2(x: f32) -> f32 {
    debug_assert!(x.abs() <= 126.0, "exp2({x})");
    let x = x as f64;
    // 2^x = 2^k e^(f ln 2) with |f| <= 1/2; the series of e^y for |y| <= 0.35 (the last term is below 1e-24).
    let k = x.round();
    let y = (x - k) * LN_2;
    let (mut sum, mut term) = (1.0, 1.0);
    for i in 1..=18 {
        term *= y / i as f64;
        sum += term;
    }
    (sum * f64::from_bits(((1023 + k as i64) as u64) << 52)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correctly rounded `f32` of an `f64` reference, and the distance in ulps of `got` from it.
    fn ulps(got: f32, want: f64) -> u32 {
        let want = want as f32;
        if got == want { 0 } else { (got.to_bits() as i64 - want.to_bits() as i64).unsigned_abs() as u32 }
    }

    #[test]
    fn sine_and_cosine_are_within_an_ulp() {
        let mut exact = 0;
        let n = 20_000;
        for i in 0..n {
            let x = -30.0 + 60.0 * i as f32 / n as f32;
            let (s, c) = sin_cos(x);
            let (ws, wc) = (x as f64).sin_cos();
            assert!(ulps(s, ws) <= 1 && ulps(c, wc) <= 1, "{x}: {s} {c} against {ws} {wc}");
            exact += (ulps(s, ws) == 0 && ulps(c, wc) == 0) as u32;
        }
        assert!(exact > n * 99 / 100, "{exact} of {n} correctly rounded");
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
    }

    #[test]
    fn powers_of_two_are_exact_at_integers_and_close_between() {
        for k in -20..=20 {
            assert_eq!(exp2(k as f32), (k as f64).exp2() as f32);
        }
        for i in 0..=3000 {
            let x = i as f32 / 1000.0;
            assert!(ulps(exp2(x), (x as f64).exp2()) <= 1, "{x}");
        }
    }
}
