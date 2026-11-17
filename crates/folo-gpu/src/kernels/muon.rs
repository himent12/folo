//! MuSGD's Muon half on the GPU ([`muon_step_many`]): the Nesterov mix, the Frobenius normalisation, the quintic
//! Newton–Schulz iteration of every matrix-shaped weight and its update `p −= lr · scale · O`, in a fixed handful
//! of dispatches per step for all of them (the tensor-op formulation took about fifteen per weight).
//!
//! Every matrix `U` (`rows × cols`) is oriented so that it has no more rows than columns (transposed when
//! `rows > cols`) and laid out back to back in one workspace `X` (one per group of matrices when all of them
//! would not fit one storage binding). Each Newton–Schulz step is three grouped GEMMs over all matrices of the
//! group at once, each workgroup finding its matrix and tile in a small table:
//! `A = X·Xᵀ`, `Bm = b·A + c·(A·A)` (`A` is symmetric, bit for bit, so `A·A = A·Aᵀ`), `X' = a·X + Bm·X`. The
//! arithmetic is the tensor ops' (`folo_nn::optim::orthogonalize_many`, then `p · 1 + O · (−lr)`), op for op,
//! except for the summation order of the products and of the norm; zero padding of the reference's row buckets
//! changes nothing and is not needed here, as every matrix is its own problem.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// Matrices one multi-tensor dispatch (the gather, the update) takes.
const MANY: usize = super::optim::MANY;
/// `u32` words per matrix in the table.
const STRIDE: usize = 8;
/// Output tile of the grouped GEMM (square) and its depth; 4×4 outputs per invocation.
const TILE: usize = 64;
const DEPTH: u32 = 16;

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
}

/// The settings of [`muon_step_many`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonSettings {
    /// `μ`.
    pub momentum: f32,
    /// `u = (g · (1 − μ)) · 1 + m · μ` when set, else `u = m`.
    pub nesterov: bool,
    /// Newton–Schulz coefficients `(a, b, c)`.
    pub coeffs: (f32, f32, f32),
    /// Newton–Schulz steps.
    pub steps: usize,
    /// Added to the Frobenius norm before the division.
    pub eps: f32,
    /// The update's learning rate: `p = p · 1 + O · (−lr)`.
    pub lr: f32,
}

/// For every matrix: `U = nesterov ? (g · (1 − μ)) · 1 + m · μ : m`, `X = U / (‖U‖_F + eps)` (transposed so
/// that rows ≤ cols), `steps` times `A = X·Xᵀ; X = a·X + (b·A + c·A·A)·X`, transposed back, `O = X · scale`
/// (when `scale ≠ 1`) and `p = p · 1 + O · (−lr)` in place. The matrices go in groups whose workspaces each fit
/// one storage binding ([`partition`]), about `2 · ⌈n / 16⌉ + 3 · steps + 2` dispatches per group of `n`. The
/// parameters must not alias the momenta or gradients.
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
    let shapes: Vec<(usize, usize)> = items.iter().map(|it| (it.rows, it.cols)).collect();
    let Some(groups) = partition(&shapes, max_bytes) else { return Ok(false) };
    for g in groups {
        run_group(ctx, &items[g], s, bindings)?;
    }
    Ok(true)
}

/// Splits the matrices (`rows × cols` each, none empty) into consecutive runs whose workspaces (`X` twice:
/// `Σ rows · cols` f32; `A` and `Bm`: `Σ r²`, `r` the smaller side; one partial per 256 elements; the table)
/// each fit in `max_bytes` and index in `u32`. `None` when one matrix's alone does not.
fn partition(shapes: &[(usize, usize)], max_bytes: u64) -> Option<Vec<Range<usize>>> {
    let cap = (max_bytes / 4).min(u64::from(u32::MAX));
    let mut groups = Vec::new();
    let (mut start, mut sum) = (0, [0u64; 4]);
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        let (r, c) = (rows.min(cols) as u64, rows.max(cols) as u64);
        let n = r.checked_mul(c)?;
        let need = [n, r * r, n.div_ceil(256), STRIDE as u64 + 1];
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
    // Under a step guard (`GpuContext::set_step_guard`) the parameter update binds it last and skips when it says
    // so; the momenta are not written here, and the workspaces are scratch.
    let guard = ctx.step_guard();
    let (j_gather, j_apply) =
        (MANY.min((bindings - 3) / 2), MANY.min(bindings - 2 - usize::from(guard.is_some())).max(1));
    // The table: per matrix (oriented rows r, cols c, transposed, element offset, offset in the r×r scratch,
    // first workgroup of the elementwise passes, first tile of the r×r products, first tile of the update),
    // then each matrix's scale.
    let mut table: Vec<u32> = Vec::with_capacity(items.len() * (STRIDE + 1));
    let (mut off, mut off_sq, mut wg, mut t_sq, mut t_up) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for it in items {
        let (r, c) = (it.rows.min(it.cols), it.rows.max(it.cols));
        let (tr, tc) = (r.div_ceil(TILE) as u64, c.div_ceil(TILE) as u64);
        for v in [r as u64, c as u64, u64::from(it.rows > it.cols), off, off_sq, wg, t_sq, t_up] {
            table.push(u32::try_from(v).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?);
        }
        let n = (r * c) as u64;
        off += n;
        off_sq += (r * r) as u64;
        wg += n.div_ceil(256);
        t_sq += tr * tr;
        t_up += tr * tc;
    }
    let count = items.len();
    for v in [off, off_sq, wg, t_sq, t_up] {
        u32::try_from(v).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?;
    }
    table.extend(items.iter().map(|it| it.scale.to_bits()));
    let tbuf = ctx.upload(bytemuck::cast_slice(&table))?;
    let xa = ctx.alloc(DType::F32.bytes(off as usize))?;
    let xb = ctx.alloc(DType::F32.bytes(off as usize))?;
    let am = ctx.alloc(DType::F32.bytes(off_sq as usize))?;
    let bm = ctx.alloc(DType::F32.bytes(off_sq as usize))?;
    // One partial sum of squares per 256 elements, then one denominator per matrix.
    let part = ctx.alloc(DType::F32.bytes(wg as usize))?;
    let den = ctx.alloc(DType::F32.bytes(count))?;
    let (x0, x1) = (xa.raw(), xb.raw());

    // 1. u, oriented, into `x0`, with its partial sums of squares.
    let (mu, omu) = (s.momentum, 1.0 - s.momentum);
    for (ci, chunk) in items.chunks(j_gather).enumerate() {
        let spec = Spec::new().u32("J", chunk.len() as u32).bool("NESTEROV", s.nesterov);
        let k = ctx.kernel(many_def("muon_gather", chunk.len()), spec)?;
        let (p, groups) = chunk_params(Params::new().uz(ci * j_gather)?.f(mu).f(omu), chunk)?;
        let mut bufs: Vec<&wgpu::Buffer> = Vec::with_capacity(2 * chunk.len() + 3);
        for it in chunk {
            bufs.push(it.g);
            bufs.push(it.m);
        }
        bufs.extend([x0, part.raw(), tbuf.raw()]);
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
    }
    // 2. Each matrix's `‖U‖_F + eps` from its partials, then `X /= ‖U‖_F + eps` over all of them.
    let k = ctx.kernel(&NORM, Spec::new())?;
    let p = Params::new().uz(count)?.f(s.eps);
    ctx.dispatch(&k, &[part.raw(), den.raw(), tbuf.raw()], &p, ctx.grid_1d(count as u64 * 256, 256)?)?;
    let k = ctx.kernel(&DIVIDE, Spec::new())?;
    ctx.dispatch(&k, &[x0, den.raw(), tbuf.raw()], &Params::new().uz(count)?, ctx.grid_1d(wg * 256, 256)?)?;
    // 3. The Newton–Schulz steps, ping-ponging between `x0` and `x1`.
    let (a, b, c) = s.coeffs;
    let p = Params::new().uz(count)?.f(a).f(b).f(c);
    let gram = ctx.kernel(&GROUPED, Spec::new().u32("MODE", 0))?;
    let poly = ctx.kernel(&GROUPED, Spec::new().u32("MODE", 1))?;
    let update = ctx.kernel(&GROUPED, Spec::new().u32("MODE", 2))?;
    let (mut cur, mut next) = (x0, x1);
    for _ in 0..s.steps {
        ctx.dispatch(&gram, &[cur, cur, cur, am.raw(), tbuf.raw()], &p, ctx.grid_1d(t_sq * 256, 256)?)?;
        let sq = [am.raw(), am.raw(), am.raw(), bm.raw(), tbuf.raw()];
        ctx.dispatch(&poly, &sq, &p, ctx.grid_1d(t_sq * 256, 256)?)?;
        ctx.dispatch(&update, &[bm.raw(), cur, cur, next, tbuf.raw()], &p, ctx.grid_1d(t_up * 256, 256)?)?;
        std::mem::swap(&mut cur, &mut next);
    }
    // 4. `p = p · 1 + (X · scale) · (−lr)`, read back in each parameter's own layout.
    for (ci, chunk) in items.chunks(j_apply).enumerate() {
        let name = if guard.is_some() { "muon_apply_guarded" } else { "muon_apply" };
        let k = ctx.kernel(many_def(name, chunk.len()), Spec::new().u32("J", chunk.len() as u32))?;
        let (p, groups) = chunk_params(Params::new().uz(ci * j_apply)?.f(-s.lr).uz(count)?, chunk)?;
        let mut bufs: Vec<&wgpu::Buffer> = chunk.iter().map(|it| it.p).collect();
        bufs.extend([cur, tbuf.raw()]);
        bufs.extend(guard.as_ref());
        ctx.dispatch(&k, &bufs, &p, ctx.grid_1d(groups * 256, 256)?)?;
    }
    Ok(())
}

/// A multi-tensor dispatch's params: `base` (its words from 0), then the first workgroup of matrix `t` of the
/// chunk at `16 + t` and its element count at `32 + t`. Returns them and the dispatch's workgroups.
fn chunk_params(base: Params, chunk: &[&MuonMatrix<'_>]) -> Result<(Params, u64)> {
    let mut p = base.at(16);
    let mut start = 0u64;
    for it in chunk {
        p = p.u(u32::try_from(start).map_err(|_| GpuError::Limit("muon: matrices too large".into()))?);
        start += ((it.rows * it.cols) as u64).div_ceil(256);
    }
    p = p.at(32);
    for it in chunk {
        p = p.uz(it.rows * it.cols)?;
    }
    Ok((p, start))
}

/// The WGSL that finds this invocation's matrix: the last whose `field` (a running start) is at or before `w`.
fn find_matrix(field: usize) -> String {
    format!(
        "fn find_matrix(w: u32) -> u32 {{\n\
         \x20 var lo = 0u; var hi = pu(0u);\n\
         \x20 loop {{\n\
         \x20   if (lo + 1u >= hi) {{ break; }}\n\
         \x20   let mid = (lo + hi) / 2u;\n\
         \x20   if (TBL[mid * {STRIDE}u + {field}u] <= w) {{ lo = mid; }} else {{ hi = mid; }}\n\
         \x20 }}\n\
         \x20 return lo;\n}}\n"
    )
}

/// The index in the oriented matrix of element `e` of the `rows × cols` original (`trans`: `rows > cols`, the
/// oriented one is `[cols, rows] = [r, c]`).
const ORIENT: &str = "fn oriented(e: u32, r: u32, c: u32, trans: bool) -> u32 {\n\
     \x20 if (trans) { return (e % r) * c + e / r; }\n\
     \x20 return e;\n}\n";

/// The multi-tensor kernels' binding layouts, by name and matrix count, kept for the process: `muon_gather`
/// (`G`, `M` per matrix, then `X`, `PART`, `TBL`), `muon_apply` (`P` per matrix, then `X`, `TBL`) and
/// `muon_apply_guarded` (the same and the step guard).
fn many_def(name: &'static str, j: usize) -> &'static KernelDef {
    static DEFS: OnceLock<Mutex<HashMap<(&'static str, usize), &'static KernelDef>>> = OnceLock::new();
    let mut defs = DEFS.get_or_init(Default::default).lock().expect("muon defs lock");
    defs.entry((name, j)).or_insert_with(|| {
        let (storage, source): (Vec<Access>, fn(&Spec) -> String) = if name == "muon_gather" {
            let s = std::iter::repeat_n(Access::Read, 2 * j).chain([
                Access::ReadWrite,
                Access::ReadWrite,
                Access::Read,
            ]);
            (s.collect(), gen_gather)
        } else if name == "muon_apply" {
            (
                std::iter::repeat_n(Access::ReadWrite, j).chain([Access::Read, Access::Read]).collect(),
                gen_apply,
            )
        } else {
            (
                std::iter::repeat_n(Access::ReadWrite, j)
                    .chain([Access::Read, Access::Read, Access::Read])
                    .collect(),
                gen_apply_guarded,
            )
        };
        let storage = Box::leak(storage.into_boxed_slice());
        Box::leak(Box::new(KernelDef { name, storage, source: Source::Generated(source) }))
    })
}

/// The per-layout kernel recorded as `name` with `spec` (the precompile registry, `kernels::find_def`): the
/// [`many_def`] its call site picked, read back from `J`. `None` for other names and for a `J` no call site sets.
pub(crate) fn many_def_for(name: &str, spec: &Spec) -> Option<&'static KernelDef> {
    let j = spec.get_u32("J", 0) as usize;
    let name = ["muon_gather", "muon_apply", "muon_apply_guarded"].into_iter().find(|n| *n == name)?;
    (1..=MANY).contains(&j).then(|| many_def(name, j))
}

/// The multi-tensor prologue: chunk-local matrix `t` of this workgroup, its element `e` and the table offset `q`
/// of its matrix (`pu(0)` is the chunk's first matrix).
fn many_prologue(s: &mut String, j: usize) {
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let w = linear_id(wid, nwg, 0u, 256u) / 256u;\n\
         \x20 var t = 0u;\n",
    );
    for t in 1..j {
        let _ = writeln!(s, "  if (w >= pu({}u)) {{ t = {t}u; }}", 16 + t);
    }
    let _ = writeln!(
        s,
        "  let lw = w - pu(16u + t); let e = lw * 256u + lid; let ok = e < pu(32u + t);\n\
         \x20 let q = (pu(0u) + t) * {STRIDE}u;"
    );
}

fn gen_gather(spec: &Spec) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> G{t}: array<f32>;", 2 * t + 1);
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> M{t}: array<f32>;", 2 * t + 2);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> X: array<f32>;", 2 * j + 1);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> PART: array<f32>;", 2 * j + 2);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> TBL: array<u32>;", 2 * j + 3);
    s.push_str("var<workgroup> red: array<f32, 256>;\n");
    s.push_str(ORIENT);
    many_prologue(&mut s, j);
    s.push_str("  var g = 0.0; var m = 0.0;\n  if (ok) {\n  switch t {\n");
    for t in 0..j {
        let _ = writeln!(s, "  case {t}u: {{ g = G{t}[e]; m = M{t}[e]; }}");
    }
    // `u = (g · (1 − μ)) · 1 + m · μ`: the tensor ops' `g.mul_scalar(1 − μ)`, then the axpby `p · 1 + m · μ`.
    // Out-of-range invocations add a zero to the partial (no early return: the reduction has barriers).
    s.push_str(
        "  default: {}\n  }\n  }\n\
         \x20 var u = m;\n\
         \x20 if (NESTEROV) { u = (g * pf(2u)) * 1.0 + m * pf(1u); }\n\
         \x20 if (ok) { X[TBL[q + 3u] + oriented(e, TBL[q], TBL[q + 1u], TBL[q + 2u] != 0u)] = u; }\n\
         \x20 red[lid] = select(0.0, u * u, ok);\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var h = 128u; h > 0u; h >>= 1u) {\n\
         \x20   if (lid < h) { red[lid] += red[lid + h]; }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 if (lid == 0u) { PART[TBL[q + 5u] + lw] = red[0]; }\n}\n",
    );
    s
}

fn gen_apply_guarded(spec: &Spec) -> String {
    gen_apply_with(spec, true)
}

fn gen_apply(spec: &Spec) -> String {
    gen_apply_with(spec, false)
}

fn gen_apply_with(spec: &Spec, guard: bool) -> String {
    let j = spec.get_u32("J", 1) as usize;
    let mut s = String::new();
    for t in 0..j {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read_write> P{t}: array<f32>;", t + 1);
    }
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> X: array<f32>;", j + 1);
    let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> TBL: array<u32>;", j + 2);
    if guard {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> SKIP: array<f32>;", j + 3);
        s.push_str(crate::context::SKIP_FN);
    }
    s.push_str(ORIENT);
    many_prologue(&mut s, j);
    if guard {
        s.push_str("  if (skip_step()) { return; }\n");
    }
    // `O = X · scale` (the tensor ops multiply only when the scale is not 1), then the axpby `p · 1 + O · (−lr)`.
    let _ = writeln!(
        s,
        "  if (!ok) {{ return; }}\n\
         \x20 let sc = bitcast<f32>(TBL[pu(2u) * {STRIDE}u + pu(0u) + t]);\n\
         \x20 var o = X[TBL[q + 3u] + oriented(e, TBL[q], TBL[q + 1u], TBL[q + 2u] != 0u)];\n\
         \x20 if (sc != 1.0) {{ o = o * sc; }}\n\
         \x20 switch t {{"
    );
    for t in 0..j {
        let _ = writeln!(s, "  case {t}u: {{ P{t}[e] = P{t}[e] * 1.0 + o * pf(1u); }}");
    }
    s.push_str("  default: {}\n  }\n}\n");
    s
}

/// One workgroup per matrix: `D[i] = sqrt(Σ partials) + eps` over the matrix's partials in a fixed order.
static NORM: KernelDef = KernelDef {
    name: "muon_norm",
    storage: &[Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Static(
        "@group(0) @binding(1) var<storage, read> PART: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> D: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> TBL: array<u32>;\n\
         var<workgroup> red: array<f32, 256>;\n\
         @compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, 0u, 256u) / 256u;\n\
         \x20 let ok = i < pu(0u);\n\
         \x20 let q = min(i, pu(0u) - 1u) * 8u;\n\
         \x20 let first = TBL[q + 5u];\n\
         \x20 let n = select(0u, (TBL[q] * TBL[q + 1u] + 255u) / 256u, ok);\n\
         \x20 var acc = 0.0;\n\
         \x20 for (var e = lid; e < n; e += 256u) { acc += PART[first + e]; }\n\
         \x20 red[lid] = acc;\n\
         \x20 workgroupBarrier();\n\
         \x20 for (var h = 128u; h > 0u; h >>= 1u) {\n\
         \x20   if (lid < h) { red[lid] += red[lid + h]; }\n\
         \x20   workgroupBarrier();\n\
         \x20 }\n\
         \x20 if (ok && lid == 0u) { D[i] = sqrt(red[0]) + pf(1u); }\n}\n",
    ),
};

static DIVIDE: KernelDef = KernelDef {
    name: "muon_divide",
    storage: &[Access::ReadWrite, Access::Read, Access::Read],
    source: Source::Generated(gen_divide),
};

/// `X /= D[i]` over every matrix's elements (256 per workgroup, as the gather's partials).
fn gen_divide(_: &Spec) -> String {
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n\
         @group(0) @binding(2) var<storage, read> D: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> TBL: array<u32>;\n",
    );
    s.push_str(&find_matrix(5));
    let _ = writeln!(
        s,
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {{\n\
         \x20 let w = linear_id(wid, nwg, 0u, 256u) / 256u;\n\
         \x20 let i = find_matrix(w);\n\
         \x20 let q = i * {STRIDE}u;\n\
         \x20 let e = (w - TBL[q + 5u]) * 256u + lid;\n\
         \x20 if (e >= TBL[q] * TBL[q + 1u]) {{ return; }}\n\
         \x20 let at = TBL[q + 3u] + e;\n\
         \x20 X[at] = X[at] / D[i];\n}}"
    );
    s
}

static GROUPED: KernelDef = KernelDef {
    name: "muon_gemm",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite, Access::Read],
    source: Source::Generated(gen_grouped),
};

/// This module's fixed kernels, for the precompile registry (`kernels::find_def`; the gather and the update are
/// built per layout, [`many_def_for`]).
pub(crate) static DEFS: &[&KernelDef] = &[&NORM, &DIVIDE, &GROUPED];

/// The grouped GEMM of one Newton–Schulz stage (`MODE` 0: `A = X·Xᵀ`, 1: `Bm = b·A + c·(A·A)`, 2:
/// `X' = a·X + Bm·X`): bindings `P`, `Q`, `R` (the epilogue's addend), `C`, `TBL`. A workgroup computes one 64×64
/// tile of one matrix's product, 4×4 outputs per invocation, `K` walked 16 deep through k-major vec4 tiles in
/// workgroup memory (the k loop unrolled here: naga's loop guards keep the driver from unrolling it).
fn gen_grouped(spec: &Spec) -> String {
    let mode = spec.get_u32("MODE", 0);
    let t4 = TILE / 4;
    let mut s = String::from(
        "@group(0) @binding(1) var<storage, read> SP: array<f32>;\n\
         @group(0) @binding(2) var<storage, read> SQ: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> SR: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> SC: array<f32>;\n\
         @group(0) @binding(5) var<storage, read> TBL: array<u32>;\n",
    );
    let _ = writeln!(s, "var<workgroup> As: array<array<vec4<f32>, {t4}>, {DEPTH}>;");
    let _ = writeln!(s, "var<workgroup> Bs: array<array<vec4<f32>, {t4}>, {DEPTH}>;");
    // The problem of this workgroup, made workgroup-uniform for the barriers: M, N, K, the P / Q / C offsets and
    // leading dimensions, the tile row and column.
    s.push_str("var<workgroup> prob: array<u32, 12>;\n");
    s.push_str(&find_matrix(if mode == 2 { 7 } else { 6 }));
    s.push_str(
        "@compute @workgroup_size(16, 16, 1)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) tid: u32, @builtin(local_invocation_id) lid3: vec3<u32>) {\n\
         \x20 if (tid == 0u) {\n\
         \x20   let w = linear_id(wid, nwg, 0u, 256u) / 256u;\n\
         \x20   let i = find_matrix(w);\n\
         \x20   let q = i * 8u;\n\
         \x20   let r = TBL[q]; let c = TBL[q + 1u]; let ox = TBL[q + 3u]; let oa = TBL[q + 4u];\n",
    );
    let _ = writeln!(s, "    let tr = (r + {0}u) / {TILE}u; let tc = (c + {0}u) / {TILE}u;", TILE - 1);
    // M, N, K, offP, ldP, offQ, ldQ, offR / C, ldC, tile row, tile column.
    match mode {
        0 => s.push_str(
            "    let t = w - TBL[q + 6u];\n\
             \x20   prob = array<u32, 12>(r, r, c, ox, c, ox, c, oa, r, t / tr, t % tr, 0u);\n",
        ),
        1 => s.push_str(
            "    let t = w - TBL[q + 6u];\n\
             \x20   prob = array<u32, 12>(r, r, r, oa, r, oa, r, oa, r, t / tr, t % tr, 0u);\n",
        ),
        _ => s.push_str(
            "    let t = w - TBL[q + 7u];\n\
             \x20   prob = array<u32, 12>(r, c, r, oa, r, ox, c, ox, c, t / tc, t % tc, 0u);\n",
        ),
    }
    s.push_str(
        "  }\n\
         \x20 let pr = workgroupUniformLoad(&prob);\n\
         \x20 let M = pr[0]; let N = pr[1]; let K = pr[2];\n\
         \x20 let offP = pr[3]; let ldP = pr[4]; let offQ = pr[5]; let ldQ = pr[6]; let offC = pr[7]; let ldC = pr[8];\n",
    );
    let _ = writeln!(s, "  let m0 = pr[9] * {TILE}u; let n0 = pr[10] * {TILE}u;");
    s.push_str("  let tx = lid3.x; let ty = lid3.y;\n");
    for i in 0..4 {
        let _ = writeln!(s, "  var acc{i} = vec4<f32>(0.0);");
    }
    let _ = writeln!(s, "  for (var k0 = 0u; k0 < K; k0 += {DEPTH}u) {{");
    // P[m][k] = SP[offP + m·ldP + k] (contiguous along k): invocations along k, then rows.
    for u in 0..4 {
        let _ = writeln!(
            s,
            "    {{ let kk = tid % {DEPTH}u; let mm = tid / {DEPTH}u + {}u; let m = m0 + mm; let k = k0 + kk;\n\
             \x20     var v = 0.0; if (m < M && k < K) {{ v = SP[offP + m * ldP + k]; }} As[kk][mm / 4u][mm % 4u] = v; }}",
            u * (256 / DEPTH as usize)
        );
    }
    for u in 0..4 {
        if mode == 2 {
            // Q[k][n] = SQ[offQ + k·ldQ + n] (along n).
            let _ = writeln!(
                s,
                "    {{ let nn = tid % {TILE}u; let kk = tid / {TILE}u + {}u; let n = n0 + nn; let k = k0 + kk;\n\
                 \x20     var v = 0.0; if (n < N && k < K) {{ v = SQ[offQ + k * ldQ + n]; }} Bs[kk][nn / 4u][nn % 4u] = v; }}",
                u * (256 / TILE)
            );
        } else {
            // Q[k][n] = SQ[offQ + n·ldQ + k] (Xᵀ, or A itself as A is symmetric; along k).
            let _ = writeln!(
                s,
                "    {{ let kk = tid % {DEPTH}u; let nn = tid / {DEPTH}u + {}u; let n = n0 + nn; let k = k0 + kk;\n\
                 \x20     var v = 0.0; if (n < N && k < K) {{ v = SQ[offQ + n * ldQ + k]; }} Bs[kk][nn / 4u][nn % 4u] = v; }}",
                u * (256 / DEPTH as usize)
            );
        }
    }
    s.push_str("    workgroupBarrier();\n");
    for kk in 0..DEPTH {
        let _ = writeln!(s, "    {{ let a = As[{kk}u][ty]; let b = Bs[{kk}u][tx];");
        for i in 0..4 {
            let _ = writeln!(s, "      acc{i} = fma(vec4<f32>(a[{i}]), b, acc{i});");
        }
        s.push_str("    }\n");
    }
    s.push_str("    workgroupBarrier();\n  }\n");
    s.push_str("  let a = pf(1u); let bb = pf(2u); let cc = pf(3u);\n");
    for i in 0..4 {
        let _ = writeln!(s, "  {{ let m = m0 + ty * 4u + {i}u; if (m < M) {{");
        for jn in 0..4 {
            let _ = writeln!(
                s,
                "    {{ let n = n0 + tx * 4u + {jn}u; if (n < N) {{ let at = offC + m * ldC + n;"
            );
            let v = match mode {
                0 => format!("acc{i}[{jn}]"),
                // `am.mul_scalar(b).add(&am.matmul(&am)?.mul_scalar(c))`.
                1 => format!("SR[at] * bb + acc{i}[{jn}] * cc"),
                // `x.mul_scalar(a).add(&bm.matmul(&x))`.
                _ => format!("SR[at] * a + acc{i}[{jn}]"),
            };
            let _ = writeln!(s, "      SC[at] = {v}; }} }}");
        }
        s.push_str("  } }\n");
    }
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Workspace words of a run of matrices: `X`, `A`, the partials, the table.
    fn words(shapes: &[(usize, usize)]) -> [u64; 4] {
        let mut sum = [0u64; 4];
        for &(rows, cols) in shapes {
            let (r, c) = (rows.min(cols) as u64, rows.max(cols) as u64);
            for (s, v) in sum.iter_mut().zip([r * c, r * r, (r * c).div_ceil(256), STRIDE as u64 + 1]) {
                *s += v;
            }
        }
        sum
    }

    /// Every group's workspaces fit the limit and the groups cover the matrices in order; a matrix that alone
    /// does not fit declines the call.
    #[test]
    fn partition_fits_the_limit() {
        // The largest, (200, 63), takes 12,600 words of `X` (50,400 bytes); 32,781 in all (131,124 bytes).
        let shapes = [(16, 27), (130, 40), (200, 63), (1, 5), (5, 1), (96, 96), (3, 9), (40, 30), (64, 64)];
        for max_bytes in [50_400, 50_403, 65_536, 100_000, 131_120, 131_124, 1 << 30] {
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
