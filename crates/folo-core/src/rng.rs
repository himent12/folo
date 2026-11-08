//! Seeded, fast, reproducible random numbers (xoshiro256++ seeded through SplitMix64).
//!
//! Every random op in FOLO takes an explicit `&mut Rng`, so a run is reproducible from its seed. Use
//! [`Rng::fork`] to derive independent streams (per worker, per layer) deterministically.

/// A xoshiro256++ generator. Not cryptographically secure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Name of the algorithm, stored next to [`Rng::state`] in checkpoints (`folo.rng` in docs/FORMAT.md).
    pub const ALGO: &'static str = "xoshiro256pp";

    /// The full generator state, for checkpoints. [`Rng::from_state`] resumes the exact stream.
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    /// A generator resuming from [`Rng::state`]. `None` for the all-zero state, which xoshiro cannot leave.
    pub fn from_state(s: [u64; 4]) -> Option<Rng> {
        (s != [0; 4]).then_some(Rng { s })
    }

    /// A generator from a 64-bit seed. Equal seeds give equal streams on every platform.
    pub fn seed(seed: u64) -> Self {
        let mut st = seed;
        let s = [splitmix64(&mut st), splitmix64(&mut st), splitmix64(&mut st), splitmix64(&mut st)];
        Rng { s }
    }

    /// The next 64 random bits.
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

    /// The next 32 random bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in `[0, 1)` with 24 bits of precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform integer in `0..n` (unbiased, Lemire's method). `n` must be > 0.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below(0)");
        loop {
            let m = (self.next_u64() as u128) * (n as u128);
            let lo = m as u64;
            if lo >= n.wrapping_neg() % n {
                return (m >> 64) as u64;
            }
        }
    }

    /// True with probability `p`.
    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// A standard normal sample (Box-Muller, computed in `f64`).
    pub fn normal(&mut self) -> f32 {
        let u1 = 1.0 - self.next_f64(); // (0, 1]
        let u2 = self.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// Fill `out` with normal samples of the given mean and standard deviation (uses both Box-Muller outputs).
    pub fn fill_normal(&mut self, out: &mut [f32], mean: f32, std: f32) {
        let (pairs, rest) = out.as_chunks_mut::<2>();
        for pair in pairs {
            let u1 = 1.0 - self.next_f64();
            let u2 = self.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
            pair[0] = mean + std * (r * c) as f32;
            pair[1] = mean + std * (r * s) as f32;
        }
        for v in rest {
            *v = mean + std * self.normal();
        }
    }

    /// Shuffle a slice in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    /// An independent generator derived from this one (advances `self`).
    pub fn fork(&mut self) -> Rng {
        Rng::seed(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_well_spread() {
        let mut a = Rng::seed(42);
        let mut b = Rng::seed(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::seed(7);
        let n = 100_000;
        let mut v = vec![0f32; n];
        r.fill_normal(&mut v, 1.0, 2.0);
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 1.0).abs() < 0.03 && (var.sqrt() - 2.0).abs() < 0.03, "{mean} {var}");
        let u: f64 = (0..n).map(|_| r.next_f32() as f64).sum::<f64>() / n as f64;
        assert!((u - 0.5).abs() < 0.01);
        let mut counts = [0usize; 5];
        for _ in 0..n {
            counts[r.below(5) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (c as f64 - n as f64 / 5.0).abs() < 1000.0));
    }

    #[test]
    fn state_round_trip() {
        let mut a = Rng::seed(99);
        for _ in 0..10 {
            a.next_u64();
        }
        let mut b = Rng::from_state(a.state()).unwrap();
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert!(Rng::from_state([0; 4]).is_none());
        assert_eq!(Rng::ALGO, "xoshiro256pp");
    }
}
