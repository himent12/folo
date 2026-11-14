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
//! **Layout.** A workgroup of 256 invocations (8 subgroups of 32) computes a `BM × BN` tile of `C`. The 8
//! subgroups are arranged `WGM × WGN`; each owns `FM × FN` accumulator fragments of 16×16 f32. `K` is walked in
//! steps of `BK`: the next `A` and `B` tiles are fetched into registers with ordinary (bounds-checked) loads while
//! the current ones are consumed from double-buffered f16 tiles in workgroup memory, one barrier per step, exactly
//! like the f32 generator in [`super::gemm`]. The tiles are row-major with 8 halves of padding per row (16-byte
//! aligned rows, fewer bank conflicts): `As[m][k]` and `Bs[k][n]`, read with `coopLoadT`. `CONV_MODE = 1` gathers
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
    encode_conv, encode_conv_backward_input, encode_conv_backward_input_phase, encode_conv_backward_weight,
    encode_conv_sites,
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

/// Invocations per workgroup.
const THREADS: u32 = 256;
/// Subgroups per workgroup (32-wide subgroups are required, see [`unavailable_reason`]).
const SUBGROUPS: u32 = 8;
/// Halves of padding at the end of every workgroup-tile row.
const PAD: u32 = 8;
/// f32 elements of one 16×16 fragment.
const FRAG: u32 = 256;

/// Tile configuration of the tensor-core kernels: a `bm × bn` tile of `C` per workgroup, 8 subgroups arranged
/// `wgm × wgn`, each owning `(bm / 16 / wgm) × (bn / 16 / wgn)` accumulator fragments, `bk` deep `K` tiles.
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
    /// Subgroup columns (`wgm · wgn == 8`).
    pub wgn: u32,
}

impl CoopConfig {
    /// 128×128 tiles, 4×2 fragments per subgroup.
    pub const C128X128: CoopConfig = CoopConfig { bm: 128, bn: 128, bk: 32, wgm: 2, wgn: 4 };
    /// 64×128 tiles, 2×2 fragments per subgroup.
    pub const C64X128: CoopConfig = CoopConfig { bm: 64, bn: 128, bk: 32, wgm: 2, wgn: 4 };
    /// 64×64 tiles, 1×2 fragments per subgroup.
    pub const C64X64: CoopConfig = CoopConfig { bm: 64, bn: 64, bk: 32, wgm: 4, wgn: 2 };
    /// 32×128 tiles (short `M`, e.g. 32 output channels), 1×2 fragments per subgroup.
    pub const C32X128: CoopConfig = CoopConfig { bm: 32, bn: 128, bk: 32, wgm: 2, wgn: 4 };
    /// 16×256 tiles (`M <= 16`), 1×2 fragments per subgroup.
    pub const C16X256: CoopConfig = CoopConfig { bm: 16, bn: 256, bk: 32, wgm: 1, wgn: 8 };

    fn fm(&self) -> u32 {
        self.bm / 16 / self.wgm
    }

    fn fnn(&self) -> u32 {
        self.bn / 16 / self.wgn
    }

    /// Whether the generator supports this configuration: 8 subgroups that split the tile into whole
    /// fragments, `bk` a multiple of 16, and tile loads that map evenly onto 256 invocations.
    pub fn valid(&self) -> bool {
        let Self { bm, bn, bk, wgm, wgn } = *self;
        if [bm, bn, bk, wgm, wgn].contains(&0) || wgm * wgn != SUBGROUPS || bk > 64 || bn > THREADS {
            return false;
        }
        bm.is_multiple_of(16 * wgm)
            && bn.is_multiple_of(16 * wgn)
            && bk.is_multiple_of(16)
            // Every tile-load plan walks `256 / inner` lines per step (see `lines`).
            && [bk, bk / 4, bm, bm / 4, bn, bn / 4].iter().all(|&i| THREADS.is_multiple_of(i))
            && (bk * bn).is_multiple_of(THREADS)
    }

    /// Workgroup memory used: two `A` and two `B` f16 tiles plus the f32 epilogue scratch.
    pub fn workgroup_bytes(&self) -> u32 {
        2 * (self.bm * (self.bk + PAD) + self.bk * (self.bn + PAD)) * 2 + SUBGROUPS * FRAG * 4
    }
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

/// [`pick_config`] for the forward convs (implicit GEMM, the 1×1 GEMM and the concatenation GEMM): short `M` gets
/// the flat tiles as there, then 64×128 for outputs of at least 128 channels (per group: `m` is `C_out / groups`)
/// and 6400 pixels (80 px and up), else 64×64. Measured on an RTX 4080 SUPER at folo-n / folo-s's shapes, batch 16
/// (`cargo bench -p folo-gpu --bench conv_model`, `FOLO_CONV_TUNE=1 FOLO_CONV_BIG=1`): 128×128 was never the
/// fastest; 64×64 took 14–63% less time than it at 40 and 20 px and on the stride-2 convs, and 64×128 2.8–5.3%
/// less than 64×64 on the three large outputs the sweep has. Not measured: folo-m / l's wider layers on their own,
/// grouped convs, non-640 inputs. Plain GEMMs and the conv input gradients keep [`pick_config`].
pub fn pick_config_conv(m: usize, n: usize) -> CoopConfig {
    if m <= 32 {
        return pick_config(m, n, 1);
    }
    if m >= 128 && n >= 6400 { CoopConfig::C64X128 } else { CoopConfig::C64X64 }
}

/// [`pick_config_conv`] when it fits the device, else `None` (the caller's own pick then decides).
pub(crate) fn conv_config(ctx: &GpuContext, m: usize, n: usize) -> Option<CoopConfig> {
    Some(pick_config_conv(m, n)).filter(|&c| fits(ctx, c))
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
