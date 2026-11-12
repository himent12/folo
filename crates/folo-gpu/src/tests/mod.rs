//! GPU kernel tests against straightforward CPU references.
//!
//! Every test asks for the shared context and returns early, printing a note, when there is no hardware
//! adapter (CI runners have none). Software adapters (WARP, llvmpipe) are skipped too unless
//! `FOLO_GPU_TEST_SOFTWARE=1`, because shader compilation there is slow. Data comes from a seeded RNG.

mod amp;
pub(crate) mod backend;
mod backend_net;
mod binds;
mod conv;
mod coop;
mod elementwise;
mod f16;
mod gemm;
mod layout;
mod pipeline_cache;
mod reduce;
mod runtime;

use crate::GpuContext;

/// The shared context, or `None` (with a note on stderr) when no suitable adapter exists.
pub(crate) fn ctx() -> Option<GpuContext> {
    match GpuContext::shared() {
        Ok(c) => {
            let software = c.summary().device_type == "Cpu";
            if software && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
                eprintln!(
                    "note: skipping GPU test on software adapter {} (set FOLO_GPU_TEST_SOFTWARE=1)",
                    c.summary().name
                );
                return None;
            }
            Some(c)
        }
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            None
        }
    }
}

/// `let ctx = gpu!();` — the shared context or an early return.
macro_rules! gpu {
    () => {
        match $crate::tests::ctx() {
            Some(c) => c,
            None => return,
        }
    };
}
pub(crate) use gpu;

/// Small deterministic RNG (splitmix64).
pub(crate) struct Rng(u64);

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

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }

    pub fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

/// Asserts `|got - want| <= atol + rtol·|want|` elementwise (NaNs and infinities must match exactly).
#[track_caller]
pub(crate) fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, f32::NEG_INFINITY);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() || g.is_nan() {
            assert!(w.is_nan() && g.is_nan(), "{what}[{i}]: got {g}, want {w}");
            continue;
        }
        if w.is_infinite() || g.is_infinite() {
            assert_eq!(g, w, "{what}[{i}]");
            continue;
        }
        let err = (g - w).abs() - rtol * w.abs();
        if err > worst.1 {
            worst = (i, err);
        }
    }
    let (i, err) = worst;
    assert!(
        err <= atol,
        "{what}[{i}]: got {}, want {} (error beyond rtol {err}, atol {atol}, rtol {rtol})",
        got[i],
        want[i]
    );
}

/// Rounds through f16 (for f16 kernel references).
pub(crate) fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
}

/// Converts to f16.
pub(crate) fn to_f16(v: &[f32]) -> Vec<half::f16> {
    v.iter().map(|&x| half::f16::from_f32(x)).collect()
}

/// The `K` chunks the forward conv that `f` encodes on `ctx` ran as: the `S` of its split-K finish pass, 1 when it
/// did not split. Read from a trace, so no other test may encode on `ctx` meanwhile.
#[track_caller]
pub(crate) fn traced_splits(ctx: &GpuContext, f: impl FnOnce()) -> usize {
    ctx.trace_begin();
    f();
    let splits: Vec<usize> = ctx
        .trace_end()
        .iter()
        .filter(|op| op.name == "conv_split_finish")
        .map(|op| {
            let s = op.spec.split(' ').find_map(|p| p.strip_prefix("S="));
            s.and_then(|s| s.parse().ok()).unwrap_or_else(|| panic!("no S in {:?}", op.spec))
        })
        .collect();
    assert!(splits.len() <= 1, "one conv, several split-K finish passes: {splits:?}");
    splits.first().copied().unwrap_or(1)
}
