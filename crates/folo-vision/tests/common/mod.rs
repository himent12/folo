//! Shared helpers for tests and benches: a tiny seeded RNG and random boxes.
#![allow(dead_code)]

use folo_vision::Xyxy;

/// SplitMix64: small, fast, deterministic.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A random box inside a `size x size` image with sides in `[min_side, max_side)`.
    pub fn bbox(&mut self, size: f32, min_side: f32, max_side: f32) -> Xyxy {
        let w = self.range(min_side, max_side);
        let h = self.range(min_side, max_side);
        let x = self.range(0.0, size - w);
        let y = self.range(0.0, size - h);
        Xyxy::new(x, y, x + w, y + h)
    }

    /// `b` with each edge moved by up to `frac` of the box size.
    pub fn jitter(&mut self, b: &Xyxy, frac: f32) -> Xyxy {
        let (w, h) = (b.width(), b.height());
        Xyxy::new(
            b.x1 + self.range(-frac, frac) * w,
            b.y1 + self.range(-frac, frac) * h,
            b.x2 + self.range(-frac, frac) * w,
            b.y2 + self.range(-frac, frac) * h,
        )
    }
}
