//! A fast multiply-xor hash for tile signatures and cache keys (not DoS-resistant; keys are ours).

use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x9E37_79B9_7F4A_7C15;

/// Mixes one 64-bit word into `h`.
#[inline(always)]
pub(crate) fn mix(h: u64, word: u64) -> u64 {
    (h ^ word).wrapping_mul(K).rotate_left(29)
}

/// Mixes two `f32`s (their bits) into `h` as one word.
#[inline(always)]
pub(crate) fn mix_f2(h: u64, a: f32, b: f32) -> u64 {
    mix(h, (a.to_bits() as u64) << 32 | b.to_bits() as u64)
}

/// The final avalanche, so nearby inputs land far apart.
#[inline(always)]
pub(crate) fn finish(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h
}

/// A `Hasher` built on `mix`, for the renderer's hash maps.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        finish(self.0)
    }

    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks {
            self.0 = mix(self.0, u64::from_le_bytes(*c));
        }
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.0 = mix(self.0, u64::from_le_bytes(w) ^ (rest.len() as u64) << 56);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.0 = mix(self.0, i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.0 = mix(self.0, i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = mix(self.0, i);
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = mix(self.0, i as u64);
    }
}

/// `BuildHasher` for `FastHasher`.
pub type FastBuild = BuildHasherDefault<FastHasher>;
