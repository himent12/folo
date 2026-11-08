//! A small, fast, seedable RNG (xoshiro256++ seeded through SplitMix64).
//!
//! folo-data does not depend on folo-core, so it carries its own generator. Everything random in the pipeline
//! (shuffling, augmentation, synthetic data) takes an explicit `&mut Rng`, which makes results reproducible.

/// Seedable pseudo random number generator (xoshiro256++).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

/// SplitMix64 step, also used as a strong 64-bit mixer.
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hashes a list of words into one seed. Used to derive independent streams such as `(seed, epoch, index)`.
pub fn mix_seed(parts: &[u64]) -> u64 {
    let mut state = 0x5EED_F010_0000_0000u64;
    let mut out = 0u64;
    for &p in parts {
        state ^= p;
        out = splitmix64(&mut state);
        state = out;
    }
    out
}

impl Rng {
    /// Creates a generator from a seed. Different seeds give independent-looking streams.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Rng { s }
    }

    /// Creates a generator for a derived stream, e.g. `Rng::derive(&[seed, epoch, index])`.
    pub fn derive(parts: &[u64]) -> Self {
        Rng::new(mix_seed(parts))
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Next 32 random bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f32` in `[lo, hi)`.
    #[inline]
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in `[0, n)`; `n` must be non-zero.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        // Lemire's multiply-shift; the bias is below 2^-32 for any realistic n.
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform integer in `[lo, hi]` (inclusive).
    #[inline]
    pub fn range_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(lo <= hi);
        lo + self.below((hi - lo + 1) as usize) as i64
    }

    /// `true` with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f32) -> bool {
        p > 0.0 && self.f32() < p
    }

    /// Standard normal sample (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.f64();
        let u2 = self.f64();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.f32();
            assert_eq!(x, b.f32());
            assert!((0.0..1.0).contains(&x));
            assert!(a.below(10) < 10);
            b.below(10);
        }
        assert_ne!(Rng::derive(&[1, 2]).next_u64(), Rng::derive(&[2, 1]).next_u64());
    }

    #[test]
    fn uniform_mean() {
        let mut r = Rng::new(1);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| r.f32() as f64).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01);
        let nm: f64 = (0..n).map(|_| r.normal() as f64).sum::<f64>() / n as f64;
        assert!(nm.abs() < 0.02);
    }
}
