//! The residual operand of the GEMM and conv epilogues (`RESID` in the GEMM and tensor-core generators): a tensor
//! of `C`'s shape added after the activation, so a residual connection costs no dispatch of its own.

use super::check_buf;
use super::gemm::{BatchStrides, GemmDesc, extent};
use crate::error::Result;
use crate::pipeline::Params;

/// First uniform word of a residual's offset and stride per outer batch index ([`Residual`]).
pub(crate) const RESID_WORD: usize = 52;

/// A tensor of `C`'s shape and dtype added to a GEMM's (or forward conv's) result after the activation,
/// `C = act(alpha · A · B + bias) + R` (a residual connection): element `(b1, b2, row, col)` of `R` is at
/// `offset + b1 · outer + b2 · C's inner batch stride + row · ldc + col`.
#[derive(Clone, Copy, Debug)]
pub struct Residual<'a> {
    /// The buffer holding `R`.
    pub buf: &'a wgpu::Buffer,
    /// Element offset of batch `(0, 0)`.
    pub offset: usize,
    /// Elements between outer batches.
    pub outer: usize,
}

impl Residual<'_> {
    /// Checks `R` against the GEMM `d` (`C`'s shape, `ldc`, inner batch stride and dtype).
    pub(crate) fn check(&self, d: &GemmDesc) -> Result<()> {
        if d.m == 0 || d.n == 0 || d.batch.0 == 0 || d.batch.1 == 0 {
            return Ok(());
        }
        let s = BatchStrides { offset: self.offset, outer: self.outer, inner: d.c.inner };
        check_buf(self.buf, extent(&s, d.batch, d.m, d.n, d.ldc)?, d.dtype, "residual")
    }

    /// Whether `vec4` access fits (with a `vec4` `C`).
    pub(crate) fn aligned4(&self) -> bool {
        self.offset.is_multiple_of(4) && self.outer.is_multiple_of(4)
    }

    /// The parameter block `p` with the residual's offset and stride at [`RESID_WORD`].
    pub(crate) fn params(&self, p: Params) -> Result<Params> {
        p.at(RESID_WORD).uz(self.offset)?.uz(self.outer)
    }
}
