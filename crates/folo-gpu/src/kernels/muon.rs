//! MuSGD's Muon half on the GPU ([`muon_step_many`]): the Nesterov mix, the Frobenius normalisation, the
//! odd-quintic Newton–Schulz / Polar-FOLO orthogonalisation of every matrix-shaped weight (with optional
//! zero-extra-GEMM step-0 Schatten-4 spectral rescaling) and its update `p −= lr · scale · O`, in a fixed handful
//! of dispatches per step for all of them.
//!
//! Every matrix `U` (`rows × cols`) is oriented so that it has no more rows than columns (`r ≤ c`, transposed when
//! `rows > cols`) and laid out back to back in one workspace `X` with 4-aligned row strides (`ld_c = align4(c)`,
//! zero-padded in `c..ld_c`; `A` and `Bm` similarly with `ld_r = align4(r)`), sorted longest-`c` first (LPT) so
//! high-`K` matrices start at workgroup 0. Each polynomial step runs three grouped `vec4` GEMMs over all matrices
//! of the group at once (`muon/shader.rs`):
//! - `MODE 0`: `A = X·Xᵀ` (symmetric: upper-triangle tiles only; diagonal tiles load `X` once; off-diagonal tiles
//!   write both `(m, n)` and register-transposed `(n, m)` as `vec4<f32>`, preserving bit-exact symmetry),
//! - `MODE 1`: `Bm = b_k·A + c_k·(A·A)` (same upper-triangle symmetry and diagonal reuse, since `A` is symmetric),
//! - `MODE 2`: `X' = a_k·X + Bm·X` (rectangular `BM × BN` tiles matching wide `r ≤ c` weights).
//!
//! When `schatten4` is set, a single lightweight reduction (`muon_s4`, one workgroup per matrix over `A₀`)
//! computes `1 / (‖A₀‖_F^(1/2) + eps / (‖U‖_F + eps))` into the table's scratch tail, and step 0's `MODE 1` /
//! `MODE 2` fold that factor into `(a₀, b₀, c₀)` with zero extra GEMMs or extra storage bindings.

mod shader;

use std::ops::Range;

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::pipeline::{Params, Spec};

pub(crate) use shader::{DEFS, many_def_for};

/// Matrices one multi-tensor dispatch (the gather, the update) takes.
pub(super) const MANY: usize = super::optim::MANY;
/// `u32` words per matrix in the table.
pub(super) const STRIDE: usize = 8;
/// Maximum polynomial steps a single [`MuonSettings`] holds (mirrors `folo_core::backend::MAX_MUON_STEPS`).
pub const MAX_STEPS: usize = 8;

/// Smallest multiple of 4 at least `n`.
#[inline]
const fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// One matrix of [`muon_step_many`].
#[derive(Clone, Copy, Debug)]
pub struct MuonMatrix<'a> {
    /// The parameter `p` (`rows · cols` f32), updated in place.
    pub p: &'a wgpu::Buffer,
    /// The Muon momentum buffer `m` (`rows · cols` f32, already updated this step).
    pub m: &'a wgpu::Buffer,
    /// The gradient `g` (`rows · cols` f32).
    pub g: &'a wgpu::Buffer,
    /// Rows (the parameter's first dimension).
    pub rows: usize,
    /// Columns (the product of the others).
    pub cols: usize,
    /// Factor of the orthogonalised direction (`folo_nn::optim::muon_scale`).
    pub scale: f32,
    /// Exact zero-GEMM per-channel Polar LMO (depthwise convs).
    pub depthwise: bool,
}

/// The settings of [`muon_step_many`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonSettings {
    /// `μ`.
    pub momentum: f32,
    /// `u = (g · (1 − μ)) · 1 + m · μ` when set, else `u = m`.
    pub nesterov: bool,
    /// Per-step odd-quintic coefficients `(a_k, b_k, c_k)` for `k < steps`.
    pub step_coeffs: [(f32, f32, f32); MAX_STEPS],
    /// Polynomial steps (`<= MAX_STEPS`).
    pub steps: usize,
    /// Zero-extra-GEMM step-0 Schatten-4 spectral rescaling (`α = ‖A₀‖_F^(1/2) + eps / (‖U‖_F + eps)`).
    pub schatten4: bool,
    /// Added to the Frobenius norm before the division.
    pub eps: f32,
    /// The update's learning rate: `p = p · 1 + O · (−lr)`.
    pub lr: f32,
}

impl MuonSettings {
    /// Uniform coefficients `(a, b, c)` repeated for `steps` steps (no Schatten-4 rescaling).
    pub fn uniform(
        momentum: f32,
        nesterov: bool,
        coeffs: (f32, f32, f32),
        steps: usize,
        eps: f32,
        lr: f32,
    ) -> Self {
        MuonSettings {
            momentum,
            nesterov,
            step_coeffs: [coeffs; MAX_STEPS],
            steps,
            schatten4: false,
            eps,
            lr,
        }
    }
}

/// Tile configuration of `muon_gemm` (`BM × BN` output tile, `TM × TN` per thread, K-tile depth `DEPTH`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MuonTile {
    pub bm: u32,
    pub bn: u32,
    pub tm: u32,
    pub tn: u32,
    pub depth: u32,
}

impl MuonTile {
    /// 32×32 square tile, 4×4 per thread (64 threads), K-depth 32 (128-byte cache-line loads, 16 KiB dbuf / 8 KiB
    /// single-buf): used for symmetric `MODE 0` (`A = X·Xᵀ`) and `MODE 1` (`Bm = b·A + c·A·A`).
    const SQ32_D32: Self = Self { bm: 32, bn: 32, tm: 4, tn: 4, depth: 32 };
    /// 64×64 update tile, 4×4 per thread (256 threads), K-depth 16 (16 KiB dbuf / 8 KiB single-buf): used for
    /// `MODE 2` (`X' = a·X + Bm·X`) on smaller models (`max_r <= 256`, e.g. `folo-n`).
    const SQ64_D16: Self = Self { bm: 64, bn: 64, tm: 4, tn: 4, depth: 16 };
    /// 64×128 update tile, 4×8 per thread (256 threads), K-depth 16 (24 KiB dbuf / 12 KiB single-buf, 2× `Bm`
    /// reuse): used for `MODE 2` on larger models (`max_r > 256`, e.g. `folo-s` and above).
    const UP64X128_D16: Self = Self { bm: 64, bn: 128, tm: 4, tn: 8, depth: 16 };

    /// Workgroup threads.
    #[inline]
    pub(crate) const fn threads(self) -> u32 {
        (self.bm / self.tm) * (self.bn / self.tn)
    }

    /// Workgroup storage bytes (`As`, `Bs`, and 16-byte-aligned `prob`).
    #[inline]
    pub(crate) const fn workgroup_bytes(self, dbuf: bool) -> u32 {
        let mult = if dbuf { 2 } else { 1 };
        mult * self.depth * (self.bm + self.bn) * 4 + (13 * 4_u32).next_multiple_of(16)
    }

    /// Whether a kernel list may precompile this tile configuration for `mode`.
    pub(crate) fn valid(self, mode: u32) -> bool {
        match mode {
            0 | 1 => self == Self::SQ32_D32,
            2 => matches!(self, Self::SQ64_D16 | Self::UP64X128_D16),
            _ => false,
        }
    }
}

/// Validates `muon_gemm` specialisation constants for precompile (`bounds.rs`).
pub(crate) fn gemm_fits(spec: &Spec) -> bool {
    let mode = spec.get_u32("MODE", 0);
    let tile = MuonTile {
        bm: spec.get_u32("BM", 64),
        bn: spec.get_u32("BN", 64),
        tm: spec.get_u32("TM", 4),
        tn: spec.get_u32("TN", 4),
        depth: spec.get_u32("DEPTH", 16),
    };
    tile.valid(mode)
}

/// Picks `(square_tile, update_tile)` for one group of matrices.
fn pick_tiles(items: &[&MuonMatrix<'_>]) -> (MuonTile, MuonTile) {
    let max_r = items.iter().map(|it| it.rows.min(it.cols)).max().unwrap_or(0);
    let up = if max_r <= 256 { MuonTile::SQ64_D16 } else { MuonTile::UP64X128_D16 };
    (MuonTile::SQ32_D32, up)
}

/// For every matrix: `U = nesterov ? (g · (1 − μ)) · 1 + m · μ : m`, `X = U / (‖U‖_F + eps)` (transposed so
/// that rows ≤ cols), `steps` times `A = X·Xᵀ; X = a_k·X + (b_k·A + c_k·A·A)·X` (with optional step-0 Schatten-4
/// spectral rescaling), transposed back, `O = X · scale` (when `scale ≠ 1`) and `p = p · 1 + O · (−lr)` in place.
/// The matrices go in groups whose workspaces each fit one storage binding ([`partition`]), about
/// `2 · ⌈n / 16⌉ + 3 · steps + 2` (+ 1 for `schatten4`) dispatches per group of `n`. The parameters must not
/// alias the momenta or gradients.
///
/// Returns `Ok(false)`, having written nothing, when the device cannot run it: fewer than 5 storage bindings
/// per kernel, or a matrix whose workspace alone is past the largest storage binding. The caller then uses
/// tensor ops.
pub fn muon_step_many(ctx: &GpuContext, items: &[MuonMatrix<'_>], s: &MuonSettings) -> Result<bool> {
    let caps = ctx.caps();
    let max_bytes = caps.max_buffer_size.min(caps.max_storage_buffer_binding_size);
    muon_step_within(ctx, items, s, max_bytes, caps.max_storage_buffers as usize)
}

/// Storage bindings the smallest dispatches take: the gather of one matrix (`G`, `M`, `X`, `PART`, `TBL`) and
/// the grouped GEMM (`P`, `Q`, `R`, `C`, `TBL`).
const MIN_BINDINGS: usize = 5;

/// [`muon_step_many`] with workspaces of at most `max_bytes` each and `bindings` storage bindings per kernel
/// (the device's, or smaller ones in tests).
pub(crate) fn muon_step_within(
    ctx: &GpuContext,
    items: &[MuonMatrix<'_>],
    s: &MuonSettings,
    max_bytes: u64,
    bindings: usize,
) -> Result<bool> {
    if s.steps > MAX_STEPS {
        return Err(GpuError::Shape(format!("muon: steps {} exceeds maximum {MAX_STEPS}", s.steps)));
    }
    for it in items {
        for (b, what) in [(it.p, "muon param"), (it.m, "muon momentum"), (it.g, "muon gradient")] {
            check_buf(b, it.rows * it.cols, DType::F32, what)?;
        }
    }
    let items: Vec<&MuonMatrix<'_>> = items.iter().filter(|it| it.rows * it.cols > 0).collect();
    if items.is_empty() {
        return Ok(true);
    }
    if bindings < MIN_BINDINGS {
        return Ok(false);
    }
    let (dw_items, mat_items): (Vec<&MuonMatrix<'_>>, Vec<&MuonMatrix<'_>>) =
        items.into_iter().partition(|it| it.depthwise);
    let shapes: Vec<(usize, usize)> = mat_items.iter().map(|it| (it.rows, it.cols)).collect();
    let Some(groups) = partition(&shapes, max_bytes) else { return Ok(false) };
    if !dw_items.is_empty() {
        run_dw_group(ctx, &dw_items, s, bindings)?;
    }
    for g in groups {
        run_group(ctx, &mat_items[g], s, bindings)?;
    }
    Ok(true)
}

/// Branch A: exact zero-GEMM per-channel block-diagonal Polar LMO for depthwise conv weights.
fn run_dw_group(
    ctx: &GpuContext,
    items: &[&MuonMatrix<'_>],
    s: &MuonSettings,
    bindings: usize,
) -> Result<()> {
    let j_dw = MANY.min((bindings - 1) / 3).max(1);
    let mut table: Vec<u32> = Vec::with_capacity(items.len() * 4);
    for it in items {
        for v in [it.rows as u64, it.cols as u64, u64::from(it.scale.to_bits()), 0u64] {
            table.push(u32::try_from(v).map_err(|_| GpuError::Limit("muon: dw too large".into()))?);
        }
    }
    let tbuf = ctx.upload(bytemuck::cast_slice(&table))?;
    let (mu, omu) = (s.momentum, 1.0 - s.momentum);
    for (ci, chunk) in items.chunks(j_dw).enumerate() {
        let spec = Spec::new().u32("J", chunk.len() as u32).bool("NESTEROV", s.nesterov);
        let k = ctx.kernel(shader::many_def("muon_dw_lmo", chunk.len()), spec)?;
        let mut starts = Vec::with_capacity(chunk.len());
        let mut start = 0u64;
        for it in chunk {
            starts.push(u32::try_from(start).map_err(|_| GpuError::Limit("muon: dw too large".into()))?);
            start += (it.rows as u64).div_ceil(256);
        }
        let total = u32::try_from(start).map_err(|_| GpuError::Limit("muon: dw too large".into()))?;
        let mut p = Params::new().uz(ci * j_dw)?.f(mu).f(omu).u(total).f(s.eps).f(-s.lr).at(16);
        for st in starts {
            p = p.u(st);
        }
        p = p.at(32);
        for it in chunk {
            p = p.uz(it.rows)?;
        }
        let mut bufs: Vec<&wgpu::Buffer> = Vec::with_capacity(3 * chunk.len() + 1);
        for it in chunk {
            bufs.push(it.p);
            bufs.push(it.g);
            bufs.push(it.m);
        }
        bufs.push(tbuf.raw());
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(start * 256, 256)?)?;
    }
    Ok(())
}

/// Splits the matrices (`rows × cols` each, none empty) into consecutive runs whose workspaces (`X` twice:
/// `Σ r · align4(c)` f32; `A` and `Bm`: `Σ r · align4(r)`; one partial per 256 padded elements; the table)
/// each fit in `max_bytes` and index in `u32`. `None` when one matrix's alone does not.
fn partition(shapes: &[(usize, usize)], max_bytes: u64) -> Option<Vec<Range<usize>>> {
    let cap = (max_bytes / 4).min(u64::from(u32::MAX));
    let mut groups = Vec::new();
    let (mut start, mut sum) = (0, [0u64; 4]);
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        let (r, c) = (rows.min(cols), rows.max(cols));
        let ld_r = align4(r) as u64;
        let ld_c = align4(c) as u64;
        let r64 = r as u64;
        let n = r64.checked_mul(ld_c)?;
        let n_sq = r64.checked_mul(ld_r)?;
        let need = [n, n_sq, n.div_ceil(256).checked_mul(2)?, STRIDE as u64 + 2];
        if need.iter().any(|&v| v > cap) {
            return None;
        }
        if sum.iter().zip(&need).any(|(s, v)| s + v > cap) {
            groups.push(start..i);
            (start, sum) = (i, [0; 4]);
        }
        for (s, v) in sum.iter_mut().zip(need) {
            *s += v;
        }
    }
    if start < shapes.len() {
        groups.push(start..shapes.len());
    }
    Some(groups)
}

/// The whole update of one group of [`partition`], all in its own workspace.
fn run_group(ctx: &GpuContext, items: &[&MuonMatrix<'_>], s: &MuonSettings, bindings: usize) -> Result<()> {
    let (j_gather, j_apply) = (MANY.min((bindings - 3) / 2), MANY.min(bindings - 2));
    // Longest-K-first (LPT) ordering so high-K matrices start at workgroup 0 instead of running as tail stragglers.
    let mut ordered: Vec<&MuonMatrix<'_>> = items.to_vec();
    ordered.sort_by(|a, b| {
        let (ra, ca) = (a.rows.min(a.cols), a.rows.max(a.cols));
        let (rb, cb) = (b.rows.min(b.cols), b.rows.max(b.cols));
        cb.cmp(&ca).then_with(|| rb.cmp(&ra))
    });
    let items = ordered.as_slice();
    let (tile_sq, tile_up) = pick_tiles(items);
    let max_wg_bytes = ctx.caps().max_workgroup_storage;
    let dbuf_sq = tile_sq.workgroup_bytes(true) <= max_wg_bytes;
    let dbuf_up = tile_up.workgroup_bytes(true) <= max_wg_bytes;

    // The table: per matrix (r, c, trans, off_x, off_sq, first_wg, first_t_sq, first_t_up), then each matrix's
    // scale bits (`count` words), then `count` scratch words for step-0 Schatten-4 `1 / alpha`.
    let count = items.len();
    let mut table: Vec<u32> = Vec::with_capacity(count * (STRIDE + 2));
    let (mut off, mut off_sq, mut wg, mut t_sq, mut t_up) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for it in items {
        let (r, c) = (it.rows.min(it.cols), it.rows.max(it.cols));
        let (ld_r, ld_c) = (align4(r) as u64, align4(c) as u64);
        let tr_sq = r.div_ceil(tile_sq.bm as usize) as u64;
        let tr_up = r.div_ceil(tile_up.bm as usize) as u64;
        let tc_up = c.div_ceil(tile_up.bn as usize) as u64;
        for v in [r as u64, c as u64, u64::from(it.rows > it.cols), off, off_sq, wg, t_sq, t_up] {
            table.push(u32::try_from(v).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?);
        }
        let n_pad = r as u64 * ld_c;
        off += n_pad;
        off_sq += r as u64 * ld_r;
        wg += n_pad.div_ceil(256);
        t_sq += tr_sq * (tr_sq + 1) / 2;
        t_up += tr_up * tc_up;
    }
    for v in [off, off_sq, wg * 2, t_sq, t_up] {
        u32::try_from(v).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?;
    }
    table.extend(items.iter().map(|it| it.scale.to_bits()));
    table.resize(count * (STRIDE + 2), 1.0f32.to_bits());
    let tbuf = ctx.upload(bytemuck::cast_slice(&table))?;
    let xa = ctx.alloc(DType::F32.bytes(off as usize))?;
    let xb = ctx.alloc(DType::F32.bytes(off as usize))?;
    let am = ctx.alloc(DType::F32.bytes(off_sq as usize))?;
    let bm = ctx.alloc(DType::F32.bytes(off_sq as usize))?;
    let part = ctx.alloc(DType::F32.bytes(wg as usize * 2))?;
    let den = ctx.alloc(DType::F32.bytes(count * 2))?;
    let (x0, x1) = (xa.raw(), xb.raw());

    // 1. u, oriented with zero-padded align4 rows, into `x0`, with its partial sums of squares.
    let (mu, omu) = (s.momentum, 1.0 - s.momentum);
    for (ci, chunk) in items.chunks(j_gather).enumerate() {
        let spec = Spec::new().u32("J", chunk.len() as u32).bool("NESTEROV", s.nesterov);
        let k = ctx.kernel(shader::many_def("muon_gather", chunk.len()), spec)?;
        let (p, groups) =
            chunk_params(Params::new().uz(ci * j_gather)?.f(mu).f(omu), chunk, true, u32::from(s.schatten4))?;
        let mut bufs: Vec<&wgpu::Buffer> = Vec::with_capacity(2 * chunk.len() + 3);
        for it in chunk {
            bufs.push(it.g);
            bufs.push(it.m);
        }
        bufs.extend([x0, part.raw(), tbuf.raw()]);
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
    }
    // 2. Each matrix's `‖U‖_F + eps` from its partials, then `X /= ‖U‖_F + eps` over all of them.
    let k = ctx.kernel(&shader::NORM, Spec::new())?;
    let p = Params::new().uz(count)?.f(s.eps);
    ctx.dispatch(&k, &[part.raw(), den.raw(), tbuf.raw()], &p, ctx.grid_1d(count as u64 * 256, 256)?)?;
    let k = ctx.kernel(&shader::DIVIDE, Spec::new())?;
    ctx.dispatch(
        &k,
        &[x0, den.raw(), tbuf.raw()],
        &Params::new().uz(count)?.u(wg as u32),
        ctx.grid_1d(wg * 256, 256)?,
    )?;
    // 3. The odd-quintic polynomial steps, ping-ponging between `x0` and `x1`.
    let spec_of = |mode: u32, t: MuonTile, dbuf: bool, s4: bool| {
        Spec::new()
            .u32("MODE", mode)
            .u32("BM", t.bm)
            .u32("BN", t.bn)
            .u32("TM", t.tm)
            .u32("TN", t.tn)
            .u32("DEPTH", t.depth)
            .bool("DBUF", dbuf)
            .bool("S4", s4)
    };
    let gram = ctx.kernel(&shader::GROUPED, spec_of(0, tile_sq, dbuf_sq, false))?;
    let poly = ctx.kernel(&shader::GROUPED, spec_of(1, tile_sq, dbuf_sq, false))?;
    let update = ctx.kernel(&shader::GROUPED, spec_of(2, tile_up, dbuf_up, false))?;
    let sq_threads = tile_sq.threads();
    let up_threads = tile_up.threads();
    let sq_grid = ctx.grid_1d(t_sq * u64::from(sq_threads), sq_threads)?;
    let up_grid = ctx.grid_1d(t_up * u64::from(up_threads), up_threads)?;
    let (mut cur, mut next) = (x0, x1);
    for step in 0..s.steps {
        let (a, b, c) = s.step_coeffs[step];
        let p_sq = Params::new().uz(count)?.f(a).f(b).f(c).u(t_sq as u32);
        let p_up = Params::new().uz(count)?.f(a).f(b).f(c).u(t_up as u32);
        ctx.dispatch(&gram, &[cur, cur, cur, am.raw(), tbuf.raw()], &p_sq, sq_grid)?;
        if step == 0 && s.schatten4 {
            let s4_k = ctx.kernel(&shader::S4_NORM, Spec::new())?;
            let ps4 = Params::new().uz(count)?.f(s.eps);
            ctx.dispatch(
                &s4_k,
                &[am.raw(), den.raw(), tbuf.raw()],
                &ps4,
                ctx.grid_1d(count as u64 * 256, 256)?,
            )?;
            let poly_s4 = ctx.kernel(&shader::GROUPED, spec_of(1, tile_sq, dbuf_sq, true))?;
            let update_s4 = ctx.kernel(&shader::GROUPED, spec_of(2, tile_up, dbuf_up, true))?;
            let sq = [am.raw(), am.raw(), am.raw(), bm.raw(), tbuf.raw()];
            ctx.dispatch(&poly_s4, &sq, &p_sq, sq_grid)?;
            ctx.dispatch(&update_s4, &[bm.raw(), cur, cur, next, tbuf.raw()], &p_up, up_grid)?;
        } else {
            let sq = [am.raw(), am.raw(), am.raw(), bm.raw(), tbuf.raw()];
            ctx.dispatch(&poly, &sq, &p_sq, sq_grid)?;
            ctx.dispatch(&update, &[bm.raw(), cur, cur, next, tbuf.raw()], &p_up, up_grid)?;
        }
        std::mem::swap(&mut cur, &mut next);
    }
    // 4. `p = p · 1 + (X · scale) · (−lr)`, read back in each parameter's own layout.
    for (ci, chunk) in items.chunks(j_apply).enumerate() {
        let k = ctx
            .kernel(shader::many_def("muon_apply", chunk.len()), Spec::new().u32("J", chunk.len() as u32))?;
        let (p, groups) = chunk_params(Params::new().uz(ci * j_apply)?.f(-s.lr).uz(count)?, chunk, false, 0)?;
        let mut bufs: Vec<&wgpu::Buffer> = chunk.iter().map(|it| it.p).collect();
        bufs.extend([cur, tbuf.raw()]);
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
    }
    Ok(())
}

/// A multi-tensor dispatch's params: `base` (words 0..3), total logical workgroups at word `3` (for 2D-folded
/// `grid_1d` early exit), `flag` at word `4` (`muon_gather`: `1` enables `2^-80` overflow-safe norm rescaling
/// for Polar-FOLO, `0` keeps legacy `Ns5`), then the first workgroup of matrix `t` of the chunk at `16 + t` and
/// its logical element count `r · c` at `32 + t`. When `padded` is true (gather), workgroups cover the 4-aligned
/// padded row width `r · align4(c)`.
fn chunk_params(base: Params, chunk: &[&MuonMatrix<'_>], padded: bool, flag: u32) -> Result<(Params, u64)> {
    let mut starts = Vec::with_capacity(chunk.len());
    let mut start = 0u64;
    for it in chunk {
        starts.push(u32::try_from(start).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?);
        let elems = if padded {
            let (r, c) = (it.rows.min(it.cols), it.rows.max(it.cols));
            (r * align4(c)) as u64
        } else {
            (it.rows * it.cols) as u64
        };
        start += elems.div_ceil(256);
    }
    let total = u32::try_from(start).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?;
    let mut p = base.at(3).u(total).u(flag).at(16);
    for s in starts {
        p = p.u(s);
    }
    p = p.at(32);
    for it in chunk {
        p = p.uz(it.rows * it.cols)?;
    }
    Ok((p, start))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Workspace words of a run of matrices: `X` (padded), `A` (padded), the partials, the table.
    fn words(shapes: &[(usize, usize)]) -> [u64; 4] {
        let mut sum = [0u64; 4];
        for &(rows, cols) in shapes {
            let (r, c) = (rows.min(cols), rows.max(cols));
            let n = (r * align4(c)) as u64;
            let n_sq = (r * align4(r)) as u64;
            for (s, v) in sum.iter_mut().zip([n, n_sq, n.div_ceil(256) * 2, STRIDE as u64 + 3]) {
                *s += v;
            }
        }
        sum
    }

    /// Every group's workspaces fit the limit and the groups cover the matrices in order; a matrix that alone
    /// does not fit declines the call.
    #[test]
    fn partition_fits_the_limit() {
        // The largest, (200, 63), takes 12,600 words of `X` (50,400 bytes); 32,812 padded words (131,248 bytes).
        let shapes = [(16, 27), (130, 40), (200, 63), (1, 5), (5, 1), (96, 96), (3, 9), (40, 30), (64, 64)];
        let total_bytes = words(&shapes)[0] * 4;
        for max_bytes in [50_400, 50_403, 65_536, 100_000, total_bytes - 4, total_bytes, 1 << 30] {
            let groups = partition(&shapes, max_bytes).unwrap();
            let mut next = 0;
            for g in &groups {
                assert_eq!(g.start, next, "{max_bytes}: {groups:?}");
                assert!(g.end > g.start, "{max_bytes}: {groups:?}");
                let w = words(&shapes[g.clone()]);
                assert!(w.iter().all(|&v| v * 4 <= max_bytes), "{max_bytes}: {g:?} {w:?}");
                next = g.end;
            }
            assert_eq!(next, shapes.len(), "{max_bytes}: {groups:?}");
            let one = words(&shapes).iter().all(|&v| v * 4 <= max_bytes);
            assert_eq!(groups.len() == 1, one, "{max_bytes}: {groups:?}");
        }
        assert!(partition(&shapes, 50_400).unwrap().len() >= 4);
        // One matrix past the limit declines the whole call.
        assert_eq!(partition(&shapes, 50_396), None);
        assert_eq!(partition(&[(1, 1), (160, 1152)], 160 * 1152 * 4 - 4), None);
        assert_eq!(partition(&[(1, 1), (160, 1152)], 160 * 1152 * 4), Some(vec![0..1, 1..2]));
        assert_eq!(partition(&[], 16), Some(vec![]));
    }
}
