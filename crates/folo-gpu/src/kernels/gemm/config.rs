//! GEMM descriptors, batch stride layouts, and tile configuration selection.

use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::elementwise::UnaryOp;

/// Bias added in the epilogue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GemmBias {
    /// No bias.
    #[default]
    None,
    /// `bias[row]` (one per output row, e.g. a conv output channel).
    Row,
    /// `bias[col]` (one per output column, e.g. a linear layer's feature).
    Col,
}

/// Element offset plus strides for a two-level batch index `(b1, b2)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchStrides {
    /// Element offset of batch `(0, 0)`.
    pub offset: usize,
    /// Stride (elements) for the outer batch index.
    pub outer: usize,
    /// Stride (elements) for the inner batch index.
    pub inner: usize,
}

impl BatchStrides {
    /// Contiguous matrices of `size` elements, outer batch only.
    pub fn packed(size: usize) -> Self {
        Self { offset: 0, outer: size, inner: 0 }
    }

    pub(crate) fn aligned4(&self) -> bool {
        self.offset.is_multiple_of(4) && self.outer.is_multiple_of(4) && self.inner.is_multiple_of(4)
    }
}

/// A full GEMM description. Matrices are row-major with leading dimensions `lda`, `ldb`, `ldc`; `A` is
/// `M × K` (`K × M` when `trans_a`), `B` is `K × N` (`N × K` when `trans_b`), `C` is `M × N`.
#[derive(Clone, Debug)]
pub struct GemmDesc {
    /// Rows of `C`.
    pub m: usize,
    /// Columns of `C`.
    pub n: usize,
    /// Reduction length.
    pub k: usize,
    /// `A` is stored transposed (`K × M`).
    pub trans_a: bool,
    /// `B` is stored transposed (`N × K`).
    pub trans_b: bool,
    /// Leading dimension of `A` as stored.
    pub lda: usize,
    /// Leading dimension of `B` as stored.
    pub ldb: usize,
    /// Leading dimension of `C`.
    pub ldc: usize,
    /// Batch counts `(outer, inner)`; `(1, 1)` for a single GEMM.
    pub batch: (usize, usize),
    /// Batch addressing of `A`.
    pub a: BatchStrides,
    /// Batch addressing of `B`.
    pub b: BatchStrides,
    /// Batch addressing of `C`.
    pub c: BatchStrides,
    /// Batch addressing of the bias (offset of the first bias element per batch).
    pub bias_strides: BatchStrides,
    /// Scale applied to `A · B`.
    pub alpha: f32,
    /// Bias kind.
    pub bias: GemmBias,
    /// Activation applied last.
    pub act: UnaryOp,
    /// Storage type of `A`, `B`, bias and `C` (accumulation is always f32).
    pub dtype: DType,
}

impl GemmDesc {
    /// A plain packed `M × K · K × N` GEMM (optionally batched with packed matrices).
    pub fn new(m: usize, n: usize, k: usize, trans_a: bool, trans_b: bool, batch: usize) -> Self {
        Self {
            m,
            n,
            k,
            trans_a,
            trans_b,
            lda: if trans_a { m } else { k },
            ldb: if trans_b { k } else { n },
            ldc: n,
            batch: (batch, 1),
            // Saturating: an absurd size then fails the extent check instead of wrapping.
            a: BatchStrides::packed(m.saturating_mul(k)),
            b: BatchStrides::packed(k.saturating_mul(n)),
            c: BatchStrides::packed(m.saturating_mul(n)),
            bias_strides: BatchStrides::default(),
            alpha: 1.0,
            bias: GemmBias::None,
            act: UnaryOp::Identity,
            dtype: DType::F32,
        }
    }

    /// Floating-point operations (2·M·N·K per batch).
    pub fn flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64 * (self.batch.0 * self.batch.1) as f64
    }
}

/// Tile configuration (exposed for benchmarks; [`super::gemm`] picks one automatically).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileConfig {
    /// Tile rows.
    pub bm: u32,
    /// Tile columns.
    pub bn: u32,
    /// Tile depth.
    pub bk: u32,
    /// Rows per thread (multiple of 4).
    pub tm: u32,
    /// Columns per thread (multiple of 4).
    pub tn: u32,
}

impl TileConfig {
    /// 128×128 tiles, depth 8, 8×8 outputs per thread, 256 threads (16 KiB of workgroup memory).
    pub const LARGE: TileConfig = TileConfig { bm: 128, bn: 128, bk: 8, tm: 8, tn: 8 };
    /// 64×128 tiles, depth 16, 4×8 per thread, 256 threads: mid-size problems and `M <= 64`.
    pub const MEDIUM: TileConfig = TileConfig { bm: 64, bn: 128, bk: 16, tm: 4, tn: 8 };
    /// 64×64 tiles, depth 16, 4×4 per thread, 256 threads: small problems that would leave SMs idle.
    pub const SMALL: TileConfig = TileConfig { bm: 64, bn: 64, bk: 16, tm: 4, tn: 4 };
    /// 32×128 tiles, depth 32, 4×4 per thread, 256 threads (40 KiB): `M <= 32` (convs with 17–32 output
    /// channels), where 64-row tiles leave half the rows empty.
    pub const ROWS32: TileConfig = TileConfig { bm: 32, bn: 128, bk: 32, tm: 4, tn: 4 };
    /// 16×128 tiles, depth 32, 4×4 per thread, 128 threads (36 KiB): `M <= 16` (a 16-channel stem).
    pub const ROWS16: TileConfig = TileConfig { bm: 16, bn: 128, bk: 32, tm: 4, tn: 4 };

    pub(crate) fn threads(&self) -> u32 {
        (self.bm / self.tm) * (self.bn / self.tn)
    }

    pub(crate) fn workgroup_bytes(&self) -> u32 {
        2 * self.bk * (self.bm + self.bn) * 4
    }

    /// Whether the generator supports this configuration: every dimension non-zero and at most 256, thread
    /// tiles that divide the workgroup tile exactly (otherwise outputs would be skipped), multiples of 4 for
    /// the vec4 accumulators, at most 256 threads, and tile loads that divide evenly among the threads.
    pub fn valid(&self) -> bool {
        let dims = [self.bm, self.bn, self.bk, self.tm, self.tn];
        if dims.iter().any(|&d| d == 0 || d > 256) {
            return false;
        }
        if !self.bm.is_multiple_of(self.tm)
            || !self.bn.is_multiple_of(self.tn)
            || !self.tm.is_multiple_of(4)
            || !self.tn.is_multiple_of(4)
            || !self.bk.is_multiple_of(4)
        {
            return false;
        }
        let t = self.threads();
        t <= 256
            && t.is_multiple_of(self.bk / 4)
            && t.is_multiple_of(self.bk)
            && t.is_multiple_of(self.bm / 4)
            && t.is_multiple_of(self.bn / 4)
            && t.is_multiple_of(self.bm)
            && t.is_multiple_of(self.bn)
            && (self.bm * self.bk / 4).is_multiple_of(t)
            && (self.bn * self.bk / 4).is_multiple_of(t)
    }
}

/// Fails unless `cfg` is [`TileConfig::valid`] and fits the device's workgroup memory.
pub(crate) fn check_config(ctx: &GpuContext, cfg: TileConfig) -> Result<()> {
    if !cfg.valid() || cfg.workgroup_bytes() > ctx.caps().max_workgroup_storage {
        return Err(shape_err(format!("invalid GEMM tile config {cfg:?} for this device")));
    }
    Ok(())
}

/// Picks a tile config from the problem size and the device's workgroup memory.
///
/// The largest tile wins as long as it still yields enough workgroups to fill the GPU (measured on an RTX
/// 4080 SUPER: 64+ workgroups) and does not waste more than a quarter of its rows on a short `M`.
pub fn pick_config(ctx: &GpuContext, d: &GemmDesc) -> TileConfig {
    let batch = d.batch.0.saturating_mul(d.batch.1);
    for cfg in [TileConfig::LARGE, TileConfig::MEDIUM] {
        let (bm, bn) = (cfg.bm as usize, cfg.bn as usize);
        let groups = d.m.div_ceil(bm).saturating_mul(d.n.div_ceil(bn)).saturating_mul(batch);
        let row_waste = d.m.div_ceil(bm).saturating_mul(bm) as f64 / d.m.max(1) as f64;
        if groups >= 64 && row_waste <= 1.25 && cfg.workgroup_bytes() <= ctx.caps().max_workgroup_storage {
            return cfg;
        }
    }
    TileConfig::SMALL
}

/// [`pick_config`] for the forward convolutions, with the narrow [`TileConfig::ROWS16`] / [`TileConfig::ROWS32`]
/// tiles for `M <= 16` / `M <= 32` (output channels per group) when they still give 64+ workgroups.
pub fn pick_config_conv(ctx: &GpuContext, d: &GemmDesc) -> TileConfig {
    let batch = d.batch.0.saturating_mul(d.batch.1);
    let narrow = match d.m {
        0..=16 => Some(TileConfig::ROWS16),
        17..=32 => Some(TileConfig::ROWS32),
        _ => None,
    };
    match narrow {
        Some(cfg)
            if d.n.div_ceil(cfg.bn as usize).saturating_mul(batch) >= 64
                && cfg.workgroup_bytes() <= ctx.caps().max_workgroup_storage =>
        {
            cfg
        }
        _ => pick_config(ctx, d),
    }
}
