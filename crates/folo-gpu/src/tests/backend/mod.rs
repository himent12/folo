//! The folo-core backend against folo-core's CPU backend: every op the trait dispatches, forward and (through
//! autograd) backward, on seeded random data with strided, broadcast and odd-sized operands.
//!
//! Tolerances: elementwise ops match to `1e-5` relative (GPU `exp` / `tanh` / `pow` are a few ulp from the
//! CPU's); reductions, GEMM and convolutions to `1e-4` relative + a small absolute term scaled to the
//! reduction length (different summation order, FMA); copies, casts and index ops exactly.

mod conv;
mod conv_cat;
mod elementwise;
mod fixtures;
mod index;
mod matmul;
mod pooling;
mod reduce;
mod transfer;

pub(crate) use fixtures::{assert_close_t, dev, device, randn};
