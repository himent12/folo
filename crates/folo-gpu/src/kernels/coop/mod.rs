//! Tensor-core (cooperative-matrix) f16 GEMM and implicit-GEMM conv: f16 inputs, f32 accumulation, f16 or f32
//! output, fused `alpha`, bias and activation.
//!
//! These kernels use wgpu's experimental cooperative matrices (`EXPERIMENTAL_COOPERATIVE_MATRIX`, WGSL
//! `coop_mat16x16<f16, A>` / `coopLoadT` / `coopMultiplyAdd` / `coopStoreT`; `SPV_KHR_cooperative_matrix` on
//! Vulkan, i.e. NVIDIA tensor cores). A context only has them when the user opted in and the adapter qualifies (see
//! [`crate::GpuOptions::tensor_cores`] and [`experimental_opt_in`] for the safety conditions); [`super::gemm::gemm`]
//! and [`super::conv::conv2d`] then route f16 problems here and fall back to the regular kernels on anything
//! unexpected (a kernel that fails to build turns tensor cores off for the context).
//!
//! **Layout.** A workgroup of `32 · WGM · WGN` invocations (`64` for `C16X32`, `128` for `C16X64`/`C32X32`/
//! `C32X64`/`C32X128`/`C64X64`, `256` for `C64X128`/`C128X128`/`C16X256`; 2, 4 or 8 subgroups of 32) computes a
//! `BM × BN` tile of `C`. The subgroups are arranged `WGM × WGN`; each owns `FM × FN` accumulator fragments of
//! 16×16 f32. `K` is walked in steps of `BK`: the next `A` and `B` tiles are fetched into registers with ordinary
//! (bounds-checked) loads while the current ones are consumed from double-buffered f16 tiles in workgroup memory,
//! one barrier per step, exactly like the f32 generator in [`super::gemm`]. Plain operands store row-major tiles
//! (`As[m][k]`, `Bs[k][n]`, read with `coopLoadT`); transposed operands (`TRANS_A` / `TRANS_B`) and `CONV_MODE = 3`
//! weight gradients store column-major tiles (`A_COL`: `As[k][m]`, `B_COL`: `Bs[n][k]`, read with `coopLoad`) when
//! they fit workgroup storage so stores into shared memory remain contiguous without bank-conflicting scatters.
//! Every tile row carries 8 halves of padding (16-byte aligned rows, fewer bank conflicts). `CONV_MODE = 1` gathers
//! the `B` tile straight from the NCHW image as the f32 implicit-GEMM conv does (no im2col buffer).
//!
//! **Epilogue.** Cooperative matrices have no per-element access, so each subgroup stores one fragment at a time
//! into its own 1 KiB slice of an f32 scratch array, and its 32 lanes then apply `alpha`, the bias and the
//! activation to 8 elements each and write them to `C` with bounds checks.
//!
//! Files: `shader.rs` the WGSL generator; `matmul.rs` the GEMM ([`gemm`]); `convolution.rs` the conv forward
//! ([`conv2d`]) and gradients; `support.rs` the device checks and the opt-in.

mod convolution;
mod matmul;
mod shader;
mod support;

use crate::context::GpuContext;
use crate::error::{Result, shape_err};

pub use self::convolution::conv2d;
pub(crate) use self::convolution::{
    encode_conv, encode_conv_backward_input, encode_conv_backward_input_phase,
    encode_conv_backward_input_subpixel, encode_conv_backward_weight, encode_conv_sites, stride2_dx_tile,
};
pub use self::matmul::gemm;
pub(crate) use self::matmul::{encode_gemm, encode_gemm_cat};
pub use self::support::adapter_unavailable_reason;
pub(crate) use self::support::{
    FEATURES, adapter_supports, experimental_opt_in, self_test, unavailable_reason,
};

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&crate::pipeline::KernelDef] =
    &[&shader::COOP, &shader::COOP_CAT, &shader::COOP_SITES, &support::PROBE];

/// Halves of padding at the end of every workgroup-tile row.
const PAD: u32 = 8;
/// f32 elements of one 16×16 fragment.
const FRAG: u32 = 256;

/// Tile configuration of the tensor-core kernels: a `bm × bn` tile of `C` per workgroup, `wgm · wgn` subgroups
/// (2, 4 or 8), each owning `(bm / 16 / wgm) × (bn / 16 / wgn)` accumulator fragments, `bk` deep `K` tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopConfig {
    /// Tile rows.
    pub bm: u32,
    /// Tile columns.
    pub bn: u32,
    /// Tile depth (a multiple of 16).
    pub bk: u32,
    /// Subgroup rows.
    pub wgm: u32,
    /// Subgroup columns (`wgm · wgn ∈ {2, 4, 8}`).
    pub wgn: u32,
}

impl CoopConfig {
    /// 128×128 tiles, 4×2 fragments per subgroup (256 threads).
    pub const C128X128: CoopConfig = CoopConfig { bm: 128, bn: 128, bk: 32, wgm: 2, wgn: 4 };
    /// 64×128 tiles, 2×2 fragments per subgroup (256 threads).
    pub const C64X128: CoopConfig = CoopConfig { bm: 64, bn: 128, bk: 32, wgm: 2, wgn: 4 };
    /// 64×64 tiles, 2×2 fragments per subgroup (128 threads, 4 subgroups, 23–24 KiB workgroup memory).
    pub const C64X64: CoopConfig = CoopConfig { bm: 64, bn: 64, bk: 32, wgm: 2, wgn: 2 };
    /// 32×128 tiles (short `M`, e.g. 32 output channels), 2×2 fragments per subgroup (128 threads, 4 subgroups).
    pub const C32X128: CoopConfig = CoopConfig { bm: 32, bn: 128, bk: 32, wgm: 1, wgn: 4 };
    /// 32×64 tiles (narrow `M <= 32` weight gradients with moderate `N`), 1×2 fragments per subgroup (128 threads).
    pub const C32X64: CoopConfig = CoopConfig { bm: 32, bn: 64, bk: 32, wgm: 2, wgn: 2 };
    /// 32×32 tiles (batch-1 small spatial planes like 20×20), 1×1 fragment per subgroup (128 threads, 14 KiB
    /// workgroup memory): 4× more tiles than `C64X64`, eliminating split-K finish dispatches deterministically.
    pub const C32X32: CoopConfig = CoopConfig { bm: 32, bn: 32, bk: 32, wgm: 2, wgn: 2 };
    /// 16×256 tiles (`M <= 16` wide spatial planes), 1×2 fragments per subgroup (256 threads).
    pub const C16X256: CoopConfig = CoopConfig { bm: 16, bn: 256, bk: 32, wgm: 1, wgn: 8 };
    /// 16×64 tiles (`M <= 16` weight gradients), 1×1 fragment per subgroup (128 threads, 16–17 KiB workgroup memory).
    pub const C16X64: CoopConfig = CoopConfig { bm: 16, bn: 64, bk: 32, wgm: 1, wgn: 4 };
    /// 16×32 tiles (narrow `M <= 16, N <= 32` stem weight gradients), 1×1 fragment per subgroup (64 threads, 10 KiB).
    pub const C16X32: CoopConfig = CoopConfig { bm: 16, bn: 32, bk: 32, wgm: 1, wgn: 2 };

    /// Subgroups per workgroup (`2`, `4` or `8`).
    pub const fn subgroups(&self) -> u32 {
        self.wgm * self.wgn
    }

    /// Invocations per workgroup (`64`, `128` or `256`, always a multiple of 32).
    pub const fn threads(&self) -> u32 {
        self.subgroups() * 32
    }

    fn fm(&self) -> u32 {
        self.bm / 16 / self.wgm
    }

    fn fnn(&self) -> u32 {
        self.bn / 16 / self.wgn
    }

    /// Whether the generator supports this configuration: 2, 4 or 8 subgroups that split the tile into whole
    /// fragments, `bk` a multiple of 16, and tile loads that map evenly onto the workgroup's invocations.
    pub fn valid(&self) -> bool {
        let Self { bm, bn, bk, wgm, wgn } = *self;
        let sgs = wgm.saturating_mul(wgn);
        if [bm, bn, bk, wgm, wgn].contains(&0) || !matches!(sgs, 2 | 4 | 8) || bk > 64 {
            return false;
        }
        let threads = sgs * 32;
        bn <= threads
            && bm.is_multiple_of(16 * wgm)
            && bn.is_multiple_of(16 * wgn)
            && bk.is_multiple_of(16)
            // Every tile-load plan walks `threads / inner` lines per step (see `lines`).
            && [bk, bk / 4, bm, bm / 4, bn, bn / 4].iter().all(|&i| i > 0 && threads.is_multiple_of(i))
            && (bk * bn).is_multiple_of(threads)
    }

    /// Workgroup memory used for row-major tiles: two `A` and two `B` f16 tiles plus the f32 epilogue scratch.
    pub fn workgroup_bytes(&self) -> u32 {
        self.workgroup_bytes_for(false, false)
    }

    /// Workgroup memory used with optional column-major `As` (`a_col`: `BK × (BM + PAD)`) and/or `Bs` (`b_col`:
    /// `BN × (BK + PAD)`).
    pub(crate) fn workgroup_bytes_for(&self, a_col: bool, b_col: bool) -> u32 {
        let a = if a_col { self.bk * (self.bm + PAD) } else { self.bm * (self.bk + PAD) };
        let b = if b_col { self.bn * (self.bk + PAD) } else { self.bk * (self.bn + PAD) };
        2 * (a + b) * 2 + self.subgroups() * FRAG * 4
    }
}

/// Whether column-major workgroup tiles (`A_COL` / `B_COL`, loaded via `coopLoad` so transposed or wgrad stores
/// write contiguously without bank-conflicting scatters) fit `ctx`'s workgroup-storage limit for `cfg`.
pub(crate) fn col_major_tiles(ctx: &GpuContext, cfg: CoopConfig, want_a: bool, want_b: bool) -> (bool, bool) {
    let max = ctx.caps().max_workgroup_storage;
    let a_col = want_a && cfg.workgroup_bytes_for(true, false) <= max;
    let b_col = want_b && cfg.workgroup_bytes_for(a_col, true) <= max;
    (a_col, b_col)
}

/// Picks a tile configuration for an `m × n` output (times `batch`): short `M` gets the flat tiles, otherwise the
/// largest tile that still yields enough workgroups to fill the GPU and wastes at most a quarter of its rows.
pub fn pick_config(m: usize, n: usize, batch: usize) -> CoopConfig {
    if m <= 16 {
        return CoopConfig::C16X256;
    }
    if m <= 32 {
        return CoopConfig::C32X128;
    }
    for cfg in [CoopConfig::C128X128, CoopConfig::C64X128] {
        let (bm, bn) = (cfg.bm as usize, cfg.bn as usize);
        let groups = m.div_ceil(bm).saturating_mul(n.div_ceil(bn)).saturating_mul(batch);
        let waste = m.div_ceil(bm).saturating_mul(bm) as f64 / m as f64;
        if groups >= 128 && waste <= 1.25 {
            return cfg;
        }
    }
    CoopConfig::C64X64
}

/// [`pick_config`] for convs (forward implicit GEMM, 1×1 GEMM, concatenation GEMM, and backward `dX`): short `M`
/// gets the flat tiles as there, then `C64X128` for outputs of at least 128 channels and 6400 pixels (80 px and
/// up) or whenever 64-column tiles would exceed the 65,535 X-workgroup limit, else `C64X64`. Never picks
/// `C128X128` (proven 14–63% slower on Ada due to shared-memory occupancy).
pub fn pick_config_conv(m: usize, n: usize) -> CoopConfig {
    if m <= 32 {
        return pick_config(m, n, 1);
    }
    if (m >= 128 && n >= 6400) || n.div_ceil(64) > 65_535 { CoopConfig::C64X128 } else { CoopConfig::C64X64 }
}

/// Batch-aware conv tile picker: on `batch == 1` with small spatial planes (`n <= 400`, e.g. 20×20) and
/// `17..=256` channels, selects [`CoopConfig::C32X32`] (4× more tiles, eliminating split-K finish dispatches
/// deterministically); otherwise delegates to [`pick_config_conv`].
pub fn pick_config_conv_batched(m: usize, n: usize, batch: usize) -> CoopConfig {
    if batch == 1 && n <= 400 && (17..=256).contains(&m) {
        return CoopConfig::C32X32;
    }
    pick_config_conv(m, n)
}

/// [`pick_config_conv_batched`] when it fits the device, else `None` (the caller's own pick then decides).
pub(crate) fn conv_config_batched(ctx: &GpuContext, m: usize, n: usize, batch: usize) -> Option<CoopConfig> {
    Some(pick_config_conv_batched(m, n, batch)).filter(|&c| fits(ctx, c))
}

fn fits(ctx: &GpuContext, cfg: CoopConfig) -> bool {
    cfg.valid() && cfg.workgroup_bytes() <= ctx.caps().max_workgroup_storage
}

/// The explicit config (an error if it does not fit the device), or the picked one (`None` if it does not fit,
/// so the caller falls back to the regular kernels).
fn resolve(
    ctx: &GpuContext,
    cfg: Option<CoopConfig>,
    pick: impl FnOnce() -> CoopConfig,
) -> Result<Option<CoopConfig>> {
    match cfg {
        Some(c) if fits(ctx, c) => Ok(Some(c)),
        Some(c) => Err(shape_err(format!("invalid tensor-core tile config {c:?} for this device"))),
        None => Ok(Some(pick()).filter(|&c| fits(ctx, c))),
    }
}
