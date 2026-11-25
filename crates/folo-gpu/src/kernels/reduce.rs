//! Reductions along an arbitrary dim, and softmax / log-softmax along the last dim.
//!
//! A reduction over dim `d` of a dense tensor views it as `[outer, R, inner]` (`R` = size of `d`). When
//! `inner == 1` the reduced axis is contiguous and one workgroup reduces each row with a tree in workgroup
//! memory; otherwise each thread owns one output column and walks `R` with coalesced loads across columns.
//!
//! The `*_as` / dtype-taking entry points also read and write f16 storage (with `shader-f16`); the arithmetic
//! is always f32, so an f16 reduction or softmax accumulates in f32 and rounds once on the store.

use super::{check_buf, prod, use_subgroups};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{Result, shape_err};
use crate::kernels::gemm::check_dtype;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

/// `alias XT` / `alias YT` (the storage types of `X` and `Y`) in front of `body`, from the spec's `XH` / `YH`.
fn typed(spec: &Spec, body: &str) -> String {
    let t = |h: bool| if h { "f16" } else { "f32" };
    format!("alias XT = {};\nalias YT = {};\n{body}", t(spec.get_bool("XH")), t(spec.get_bool("YH")))
}

/// The spec switches for an `x` of `xdt` and a `y` of `ydt`.
fn typed_spec(ctx: &GpuContext, xdt: DType, ydt: DType, op: &'static str) -> Result<Spec> {
    check_dtype(ctx, xdt, op)?;
    check_dtype(ctx, ydt, op)?;
    let (xh, yh) = (xdt == DType::F16, ydt == DType::F16);
    Ok(Spec::new().bool("XH", xh).bool("YH", yh).f16(xh || yh))
}

const REDUCE_ROWS_MAIN_SUBGROUPS: &str = r#"
var<workgroup> partial: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let r = pu(1u);
    let acc = reduce_rows_thread(row, rows, r, lid);
    var r_sg: f32;
    switch OP {
        case 1u: {
            let nan_v = isnan1(acc);
            let mask = subgroupBallot(nan_v).x;
            let nan_lane = select(0u, firstTrailingBit(mask), mask != 0u);
            r_sg = select(subgroupMax(select(acc, identity(), nan_v)), subgroupShuffle(acc, nan_lane), mask != 0u);
        }
        case 2u: {
            let nan_v = isnan1(acc);
            let mask = subgroupBallot(nan_v).x;
            let nan_lane = select(0u, firstTrailingBit(mask), mask != 0u);
            r_sg = select(subgroupMin(select(acc, identity(), nan_v)), subgroupShuffle(acc, nan_lane), mask != 0u);
        }
        default: { r_sg = subgroupAdd(acc); }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) { partial[sg] = r_sg; }
    workgroupBarrier();
    if (lid == 0u && row < rows) {
        let total = combine(
            combine(combine(partial[0], partial[1]), combine(partial[2], partial[3])),
            combine(combine(partial[4], partial[5]), combine(partial[6], partial[7])),
        );
        Y[row] = YT(total * pf(2u));
    }
}
"#;

const REDUCE_ROWS_MAIN_PORTABLE: &str = r#"
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let r = pu(1u);
    partial[lid] = reduce_rows_thread(row, rows, r, lid);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            partial[lid] = combine(partial[lid], partial[lid + s]);
        }
        workgroupBarrier();
    }
    if (lid == 0u && row < rows) {
        Y[row] = YT(partial[0] * pf(2u));
    }
}
"#;

const SOFTMAX_MAIN_SUBGROUPS: &str = r#"
var<workgroup> partial: array<vec2<f32>, 8>;

fn reduce_online(lid: u32, sgid: u32, lane: u32, v: vec2<f32>) -> vec2<f32> {
    var st = v;
    for (var off = 16u; off > 0u; off >>= 1u) {
        let om = subgroupShuffleDown(st.x, off);
        let od = subgroupShuffleDown(st.y, off);
        if (lane < off) { st = online_merge(st, vec2<f32>(om, od)); }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) { partial[sg] = st; }
    workgroupBarrier();
    // Butterfly-reduce the 8 subgroup partials across every subgroup so all 256 threads receive the final
    // `(m, d)` with a single workgroupBarrier().
    var r = partial[lane & 7u];
    r = online_merge(r, vec2<f32>(subgroupShuffleXor(r.x, 4u), subgroupShuffleXor(r.y, 4u)));
    r = online_merge(r, vec2<f32>(subgroupShuffleXor(r.x, 2u), subgroupShuffleXor(r.y, 2u)));
    r = online_merge(r, vec2<f32>(subgroupShuffleXor(r.x, 1u), subgroupShuffleXor(r.y, 1u)));
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    softmax_run(wid, nwg, lid, sgid, lane);
}
"#;

const SOFTMAX_MAIN_PORTABLE: &str = r#"
var<workgroup> partial: array<vec2<f32>, 256>;

fn reduce_online(lid: u32, sgid: u32, lane: u32, v: vec2<f32>) -> vec2<f32> {
    partial[lid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { partial[lid] = online_merge(partial[lid], partial[lid + s]); }
        workgroupBarrier();
    }
    return partial[0];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    softmax_run(wid, nwg, lid, 0u, 0u);
}
"#;

const REDUCE_ROWS_THREAD_VEC4: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<vec4<XT>>;

fn reduce_rows_thread(row: u32, rows: u32, r: u32, lid: u32) -> f32 {
    var acc4 = vec4<f32>(identity());
    if (row < rows) {
        if (SPLIT_K) {
            let splits = pu(4u);
            let chunk = pu(5u);
            let o = row / splits;
            let k = row % splits;
            let start = k * chunk;
            let len = min(chunk, r - start);
            let base = o * r + start;
            for (var j = lid; j < len; j += 256u) {
                acc4 = combine4(acc4, vec4<f32>(X[base + j]));
            }
        } else {
            let base = row * r;
            for (var i = lid; i < r; i += 256u) {
                acc4 = combine4(acc4, vec4<f32>(X[base + i]));
            }
        }
    }
    return combine(combine(acc4.x, acc4.y), combine(acc4.z, acc4.w));
}
"#;

const REDUCE_ROWS_THREAD_SCALAR: &str = r#"
@group(0) @binding(1) var<storage, read> X: array<XT>;

fn reduce_rows_thread(row: u32, rows: u32, r: u32, lid: u32) -> f32 {
    var acc = identity();
    if (row < rows) {
        if (SPLIT_K) {
            let splits = pu(4u);
            let chunk = pu(5u);
            let o = row / splits;
            let k = row % splits;
            let start = k * chunk;
            let len = min(chunk, r - start);
            let base = o * r + start;
            for (var j = lid; j < len; j += 256u) {
                acc = combine(acc, f32(X[base + j]));
            }
        } else {
            let base = row * r;
            for (var i = lid; i < r; i += 256u) {
                acc = combine(acc, f32(X[base + i]));
            }
        }
    }
    return acc;
}
"#;

fn gen_rows(s: &Spec) -> String {
    let split = s.get_bool("SPLIT");
    let vec4 = s.get_bool("VEC4");
    let thread = if vec4 { REDUCE_ROWS_THREAD_VEC4 } else { REDUCE_ROWS_THREAD_SCALAR };
    let main = if s.is_subgroups() { REDUCE_ROWS_MAIN_SUBGROUPS } else { REDUCE_ROWS_MAIN_PORTABLE };
    typed(
        s,
        &format!(
            "const SPLIT_K: bool = {split};\n{}{thread}{main}",
            include_str!("../shaders/reduce_rows.wgsl")
        ),
    )
}
fn gen_cols(s: &Spec) -> String {
    let split = s.get_bool("SPLIT");
    typed(s, &format!("const SPLIT_K: bool = {split};\n{}", include_str!("../shaders/reduce_cols.wgsl")))
}
/// Maximum row length cached in per-thread registers (4 elements × 256 threads) for single-pass [`SOFTMAX`].
const SOFTMAX_ROW_CACHE_MAX: usize = 1024;
/// Maximum column length cached in a per-thread local array for single-pass [`SOFTMAX_COLS`] (covers DFL `d = 16`).
const SOFTMAX_COL_SMALL_MAX: usize = 32;

fn gen_softmax(s: &Spec) -> String {
    let cache = s.get_bool("CACHE");
    let main = if s.is_subgroups() { SOFTMAX_MAIN_SUBGROUPS } else { SOFTMAX_MAIN_PORTABLE };
    typed(
        s,
        &format!("const CACHE_ROW: bool = {cache};\n{main}\n{}", include_str!("../shaders/softmax.wgsl")),
    )
}
fn gen_softmax_cols(s: &Spec) -> String {
    let small = s.get_bool("SMALL");
    typed(s, &format!("const SMALL_COL: bool = {small};\n{}", include_str!("../shaders/softmax_cols.wgsl")))
}

/// Reduction kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum.
    Sum,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
    /// Arithmetic mean.
    Mean,
}

impl ReduceOp {
    fn code(self) -> u32 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0,
            ReduceOp::Max => 1,
            ReduceOp::Min => 2,
        }
    }

    fn identity(self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }
}

static ROWS: KernelDef = KernelDef {
    name: "reduce_rows",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_rows),
};
static COLS: KernelDef = KernelDef {
    name: "reduce_cols",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_cols),
};
static SOFTMAX: KernelDef = KernelDef {
    name: "softmax",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_softmax),
};

/// Reduces `x` viewed as `[outer, r, inner]` over the middle axis into `y` (`[outer, inner]`), f32.
pub fn reduce(
    ctx: &GpuContext,
    op: ReduceOp,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    reduce_as(ctx, op, (x, DType::F32), (y, DType::F32), outer, r, inner)
}

/// [`reduce`] with `x` and `y` stored as the given dtypes (f32 or f16; f32 arithmetic either way).
pub fn reduce_as(
    ctx: &GpuContext,
    op: ReduceOp,
    (x, xdt): (&wgpu::Buffer, DType),
    (y, ydt): (&wgpu::Buffer, DType),
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    let scale = if op == ReduceOp::Mean { 1.0 / r as f32 } else { 1.0 };
    reduce_scaled(ctx, op, (x, xdt), (y, ydt), outer, r, inner, scale)
}

#[allow(clippy::too_many_arguments)]
fn reduce_scaled(
    ctx: &GpuContext,
    op: ReduceOp,
    (x, xdt): (&wgpu::Buffer, DType),
    (y, ydt): (&wgpu::Buffer, DType),
    outer: usize,
    r: usize,
    inner: usize,
    scale: f32,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, xdt, "reduce input")?;
    check_buf(y, prod(&[outer, inner])?, ydt, "reduce output")?;
    if outer * inner == 0 {
        return Ok(());
    }
    let spec = typed_spec(ctx, xdt, ydt, "reduce")?.u32("OP", op.code());
    if inner == 1 {
        let vec4 = r > 0 && r.is_multiple_of(4);
        let units_r = if vec4 { r / 4 } else { r };
        let k = ctx.kernel(&ROWS, spec.bool("VEC4", vec4).subgroups(use_subgroups(ctx)))?;
        let p = Params::new().uz(outer)?.uz(units_r)?.f(scale).f(op.identity());
        ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(outer as u64 * 256, 256)?)
    } else {
        let chunks = inner.div_ceil(64);
        let k = ctx.kernel(&COLS, spec)?;
        let p = Params::new().uz(outer)?.uz(r)?.f(scale).uz(inner)?.uz(chunks)?.f(op.identity());
        ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d((outer * chunks) as u64 * 256, 256)?)
    }
}

/// Splits `shape` around `dim` into `(outer, r, inner)`.
pub fn split_dim(shape: &[usize], dim: usize) -> Result<(usize, usize, usize)> {
    if dim >= shape.len() {
        return Err(shape_err(format!("dim {dim} out of range for {shape:?}")));
    }
    Ok((shape[..dim].iter().product(), shape[dim], shape[dim + 1..].iter().product()))
}

/// Softmax (or log-softmax when `log`) over contiguous rows of length `n`, f32.
pub fn softmax(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    n: usize,
    log: bool,
) -> Result<()> {
    softmax_rows(ctx, x, y, rows, n, log, DType::F32)
}

/// [`softmax`] for `x` and `y` of `dtype` (f32 or f16; the max, the sum and the exponentials are f32).
pub fn softmax_rows(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    n: usize,
    log: bool,
    dtype: DType,
) -> Result<()> {
    check_buf(x, prod(&[rows, n])?, dtype, "softmax input")?;
    check_buf(y, prod(&[rows, n])?, dtype, "softmax output")?;
    if rows * n == 0 {
        return Ok(());
    }
    let k = ctx.kernel(
        &SOFTMAX,
        typed_spec(ctx, dtype, dtype, "softmax")?
            .bool("LOG", log)
            .bool("CACHE", n <= SOFTMAX_ROW_CACHE_MAX)
            .subgroups(use_subgroups(ctx)),
    )?;
    let p = Params::new().uz(rows)?.uz(n)?.f(f32::NEG_INFINITY);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(rows as u64 * 256, 256)?)
}

// ----- folo-core backend: arg-reductions, softmax over any dim, long-row splitting, top-k -----------------------

static ARG: KernelDef = KernelDef {
    name: "arg_reduce",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Static(include_str!("../shaders/arg_reduce.wgsl")),
};
static SOFTMAX_COLS: KernelDef = KernelDef {
    name: "softmax_cols",
    storage: &[Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_softmax_cols),
};
/// Maximum row length sorted in-place in shared memory with a single-read bitonic sort for [`TOPK`].
const TOPK_BITONIC_MAX: usize = 512;

const TOPK_MAIN_BITONIC: &str = r#"
var<workgroup> sh_k: array<u32, 512>;
var<workgroup> sh_i: array<u32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let d = pu(1u);
    let k = pu(2u);
    let valid = row < rows;
    let base = row * d;
    var n_pad = 2u;
    while (n_pad < d) { n_pad = n_pad << 1u; }
    let i0 = lid;
    if (i0 < n_pad) {
        if (valid && i0 < d) {
            sh_k[i0] = key(X[base + i0]);
            sh_i[i0] = i0;
        } else {
            sh_k[i0] = 0u;
            sh_i[i0] = TOPK_EMPTY;
        }
    }
    let i1 = lid + 256u;
    if (i1 < n_pad) {
        if (valid && i1 < d) {
            sh_k[i1] = key(X[base + i1]);
            sh_i[i1] = i1;
        } else {
            sh_k[i1] = 0u;
            sh_i[i1] = TOPK_EMPTY;
        }
    }
    workgroupBarrier();
    let half = n_pad >> 1u;
    for (var size = 2u; size <= n_pad; size = size << 1u) {
        for (var stride = size >> 1u; stride > 0u; stride = stride >> 1u) {
            if (lid < half) {
                let lo = lid & (stride - 1u);
                let p = ((lid - lo) << 1u) | lo;
                let q = p | stride;
                let kp = sh_k[p];
                let ip = sh_i[p];
                let kq = sh_k[q];
                let iq = sh_i[q];
                let best_first = (p & size) == 0u;
                let swap = select(better_slot(kp, ip, kq, iq), better_slot(kq, iq, kp, ip), best_first);
                if (swap) {
                    sh_k[p] = kq;
                    sh_i[p] = iq;
                    sh_k[q] = kp;
                    sh_i[q] = ip;
                }
            }
            workgroupBarrier();
        }
    }
    if (!valid) { return; }
    for (var j = lid; j < k; j += 256u) {
        let idx = sh_i[j];
        V[row * k + j] = X[base + idx];
        I[2u * (row * k + j)] = idx;
        I[2u * (row * k + j) + 1u] = 0u;
    }
}
"#;

const TOPK_MAIN_SELECT_SUBGROUPS: &str = r#"
var<workgroup> sh_k: array<u32, 8>;
var<workgroup> sh_i: array<u32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let d = pu(1u);
    let k = pu(2u);
    let valid = row < rows;
    let base = row * d;
    let sg = min(sgid, 7u);
    var pk = 0u;
    var pi = TOPK_EMPTY;
    for (var j = 0u; j < k; j++) {
        var bk = 0u;
        var bi = TOPK_EMPTY;
        if (valid) {
            for (var i = lid; i < d; i += 256u) {
                let ki = key(X[base + i]);
                if (j > 0u && !before(pk, pi, ki, i)) { continue; }
                if (bi == TOPK_EMPTY || before(ki, i, bk, bi)) { bk = ki; bi = i; }
            }
        }
        for (var off = 16u; off > 0u; off = off >> 1u) {
            let ok = subgroupShuffleDown(bk, off);
            let oi = subgroupShuffleDown(bi, off);
            if (lane < off && better_slot(ok, oi, bk, bi)) { bk = ok; bi = oi; }
        }
        if (lane == 0u) {
            sh_k[sg] = bk;
            sh_i[sg] = bi;
        }
        workgroupBarrier();
        var rk = sh_k[lane & 7u];
        var ri = sh_i[lane & 7u];
        for (var off = 4u; off > 0u; off = off >> 1u) {
            let ok = subgroupShuffleXor(rk, off);
            let oi = subgroupShuffleXor(ri, off);
            if (better_slot(ok, oi, rk, ri)) { rk = ok; ri = oi; }
        }
        pk = rk;
        pi = ri;
        if (lid == 0u && valid) {
            V[row * k + j] = X[base + pi];
            I[2u * (row * k + j)] = pi;
            I[2u * (row * k + j) + 1u] = 0u;
        }
        if (j + 1u < k) { workgroupBarrier(); }
    }
}
"#;

const TOPK_MAIN_SELECT_PORTABLE: &str = r#"
var<workgroup> sh_k: array<u32, 256>;
var<workgroup> sh_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = wid.y * nwg.x + wid.x;
    let rows = pu(0u);
    let d = pu(1u);
    let k = pu(2u);
    let valid = row < rows;
    let base = row * d;
    var pk = 0u;
    var pi = TOPK_EMPTY;
    for (var j = 0u; j < k; j++) {
        var bk = 0u;
        var bi = TOPK_EMPTY;
        if (valid) {
            for (var i = lid; i < d; i += 256u) {
                let ki = key(X[base + i]);
                if (j > 0u && !before(pk, pi, ki, i)) { continue; }
                if (bi == TOPK_EMPTY || before(ki, i, bk, bi)) { bk = ki; bi = i; }
            }
        }
        sh_k[lid] = bk;
        sh_i[lid] = bi;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                let ok = sh_k[lid + s];
                let oi = sh_i[lid + s];
                if (better_slot(ok, oi, sh_k[lid], sh_i[lid])) {
                    sh_k[lid] = ok;
                    sh_i[lid] = oi;
                }
            }
            workgroupBarrier();
        }
        pk = sh_k[0];
        pi = sh_i[0];
        if (lid == 0u && valid) {
            V[row * k + j] = X[base + pi];
            I[2u * (row * k + j)] = pi;
            I[2u * (row * k + j) + 1u] = 0u;
        }
        if (j + 1u < k) { workgroupBarrier(); }
    }
}
"#;

fn gen_topk(s: &Spec) -> String {
    let main = if s.get_bool("BITONIC") {
        TOPK_MAIN_BITONIC
    } else if s.is_subgroups() {
        TOPK_MAIN_SELECT_SUBGROUPS
    } else {
        TOPK_MAIN_SELECT_PORTABLE
    };
    format!("{}{main}", include_str!("../shaders/topk.wgsl"))
}

static TOPK: KernelDef = KernelDef {
    name: "topk",
    storage: &[Access::Read, Access::ReadWrite, Access::ReadWrite],
    source: Source::Generated(gen_topk),
};

/// Ceiling-division split plan `(splits, chunk_len)` for a reduction over `[outer, r, inner]`.
///
/// Splits when a single pass would launch fewer than 256 workgroups and `r >= 8192`, targeting ~512 total
/// workgroups with at least `MIN_SPLIT_CHUNK` elements per chunk. Recomputing `splits = r.div_ceil(chunk_len)`
/// guarantees every split chunk is non-empty (`(splits - 1) * chunk_len < r <= splits * chunk_len`), eliminating
/// wasted empty partial tiles for prime/odd `r`.
const MIN_SPLIT_CHUNK: usize = 1024;

fn split_plan(outer: usize, r: usize, inner: usize) -> (usize, usize) {
    if outer == 0 || inner == 0 || r < 8192 {
        return (1, r);
    }
    let groups = if inner == 1 { outer } else { outer.saturating_mul(inner.div_ceil(64)) };
    if groups >= 256 {
        return (1, r);
    }
    let want = 512usize.div_ceil(groups);
    let max_s = r / MIN_SPLIT_CHUNK;
    let s0 = want.min(max_s);
    if s0 <= 1 {
        return (1, r);
    }
    let mut chunk = r.div_ceil(s0);
    if inner == 1 && r.is_multiple_of(4) {
        chunk = chunk.next_multiple_of(4);
    }
    let splits = r.div_ceil(chunk);
    (splits, chunk)
}

/// [`reduce`] (sum / mean / max / min) that splits a long reduction over few outputs into two passes using
/// ceiling-division chunks so the GPU fills up for any `r` (including prime/odd lengths): pass 1 reduces
/// `[outer, r, inner]` into `splits` non-empty chunks `[outer, splits, inner]` in f32, and pass 2 reduces the
/// `splits` partials into `y` (applying `1 / r` on pass 2 for [`ReduceOp::Mean`]). `x` and `y` hold `dtype`
/// (f32 or f16); the partial results between the passes are f32.
#[allow(clippy::too_many_arguments)]
pub fn reduce_split(
    ctx: &GpuContext,
    op: ReduceOp,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
    dtype: DType,
) -> Result<()> {
    let (splits, chunk) = split_plan(outer, r, inner);
    if splits <= 1 {
        return reduce_as(ctx, op, (x, dtype), (y, dtype), outer, r, inner);
    }
    check_buf(x, prod(&[outer, r, inner])?, dtype, "reduce input")?;
    check_buf(y, prod(&[outer, inner])?, dtype, "reduce output")?;
    let outer_s = prod(&[outer, splits])?;
    let tmp = ctx.alloc(DType::F32.bytes(prod(&[outer_s, inner])?))?;
    let pass_op = if op == ReduceOp::Mean { ReduceOp::Sum } else { op };
    let spec = typed_spec(ctx, dtype, DType::F32, "reduce")?.u32("OP", pass_op.code()).bool("SPLIT", true);
    if inner == 1 {
        let vec4 = r > 0 && r.is_multiple_of(4) && chunk.is_multiple_of(4);
        let (units_r, units_chunk) = if vec4 { (r / 4, chunk / 4) } else { (r, chunk) };
        let k = ctx.kernel(&ROWS, spec.bool("VEC4", vec4).subgroups(use_subgroups(ctx)))?;
        let p = Params::new()
            .uz(outer_s)?
            .uz(units_r)?
            .f(1.0)
            .f(pass_op.identity())
            .uz(splits)?
            .uz(units_chunk)?;
        ctx.dispatch(&k, &[x, tmp.raw()], &p, ctx.grid_1d(outer_s as u64 * 256, 256)?)?;
    } else {
        let chunks = inner.div_ceil(64);
        let k = ctx.kernel(&COLS, spec)?;
        let p = Params::new()
            .uz(outer_s)?
            .uz(r)?
            .f(1.0)
            .uz(inner)?
            .uz(chunks)?
            .f(pass_op.identity())
            .uz(splits)?
            .uz(chunk)?;
        ctx.dispatch(&k, &[x, tmp.raw()], &p, ctx.grid_1d((outer_s * chunks) as u64 * 256, 256)?)?;
    }
    let scale = if op == ReduceOp::Mean { 1.0 / r as f32 } else { 1.0 };
    reduce_scaled(ctx, pass_op, (tmp.raw(), DType::F32), (y, dtype), outer, splits, inner, scale)
}

/// Index of the first maximum (`max`) or minimum over the middle axis of dense f32 `[outer, r, inner]`, as
/// `i64` (two words per element) into `y` (`[outer, inner]`). NaN counts as the extreme (the first NaN wins).
pub fn arg_reduce(
    ctx: &GpuContext,
    max: bool,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    outer: usize,
    r: usize,
    inner: usize,
) -> Result<()> {
    check_buf(x, prod(&[outer, r, inner])?, DType::F32, "arg_reduce input")?;
    check_buf(y, prod(&[outer, inner, 2])?, DType::F32, "arg_reduce output")?;
    let n = outer * inner;
    if n == 0 {
        return Ok(());
    }
    if r == 0 {
        return Err(shape_err("arg_reduce over an empty dimension"));
    }
    let k = ctx.kernel(&ARG, Spec::new().bool("MAX", max))?;
    let p = Params::new().uz(n)?.uz(r)?.uz(inner)?;
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// Softmax (or log-softmax) over the middle axis of dense `[outer, d, inner]` of `dtype` (f32 or f16, f32
/// arithmetic).
pub fn softmax_dim(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    (outer, d, inner): (usize, usize, usize),
    log: bool,
    dtype: DType,
) -> Result<()> {
    if inner == 1 {
        return softmax_rows(ctx, x, y, outer, d, log, dtype);
    }
    check_buf(x, prod(&[outer, d, inner])?, dtype, "softmax input")?;
    check_buf(y, prod(&[outer, d, inner])?, dtype, "softmax output")?;
    let n = outer * inner;
    if n * d == 0 {
        return Ok(());
    }
    let k = ctx.kernel(
        &SOFTMAX_COLS,
        typed_spec(ctx, dtype, dtype, "softmax")?.bool("LOG", log).bool("SMALL", d <= SOFTMAX_COL_SMALL_MAX),
    )?;
    let p = Params::new().uz(n)?.uz(d)?.uz(inner)?.f(f32::NEG_INFINITY);
    ctx.dispatch(&k, &[x, y], &p, ctx.grid_1d(n as u64, 256)?)
}

/// The `k` largest (`largest`) or smallest entries of each of `rows` dense f32 rows of length `d`, sorted,
/// ties broken by lower index (total order: `-NaN < -inf < .. < -0 < +0 < .. < inf < NaN`, as
/// `f32::total_cmp`). Values go to `values` (`[rows, k]`), `i64` indices to `indices`. One 256-thread workgroup
/// per row: rows up to [`TOPK_BITONIC_MAX`] run a single-read shared-memory bitonic sort; longer rows run `k`
/// coalesced workgroup-parallel selection passes (with 32-wide subgroup reduction when available).
#[allow(clippy::too_many_arguments)]
pub fn topk(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    values: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    rows: usize,
    d: usize,
    k: usize,
    largest: bool,
) -> Result<()> {
    if k > d {
        return Err(shape_err(format!("topk: k = {k} exceeds the row length {d}")));
    }
    check_buf(x, prod(&[rows, d])?, DType::F32, "topk input")?;
    check_buf(values, prod(&[rows, k])?, DType::F32, "topk values")?;
    check_buf(indices, prod(&[rows, k, 2])?, DType::F32, "topk indices")?;
    if rows * k == 0 {
        return Ok(());
    }
    let bitonic = d <= TOPK_BITONIC_MAX;
    let spec = Spec::new()
        .bool("LARGEST", largest)
        .bool("BITONIC", bitonic)
        .subgroups(!bitonic && use_subgroups(ctx));
    let kern = ctx.kernel(&TOPK, spec)?;
    let p = Params::new().uz(rows)?.uz(d)?.uz(k)?;
    ctx.dispatch(&kern, &[x, values, indices], &p, ctx.grid_1d(rows as u64 * 256, 256)?)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] = &[&ROWS, &COLS, &SOFTMAX, &ARG, &SOFTMAX_COLS, &TOPK];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_plan_covers_arbitrary_r_without_empty_chunks() {
        assert_eq!(split_plan(0, 16384, 1), (1, 16384));
        assert_eq!(split_plan(1, 8191, 1), (1, 8191));
        assert_eq!(split_plan(256, 16384, 1), (1, 16384));
        assert_eq!(split_plan(4, 16384, 64 * 64), (1, 16384));

        for &(outer, r, inner) in &[
            (1usize, 8192usize, 1usize),
            (1, 8193, 1),
            (3, 10007, 1),
            (2, 16129, 5),
            (2, 100_003, 70),
            (1, 1_000_000, 1),
            (7, 9973, 3),
        ] {
            let (splits, chunk) = split_plan(outer, r, inner);
            assert!(splits >= 2, "expected split for ({outer}, {r}, {inner}), got ({splits}, {chunk})");
            assert!(chunk >= MIN_SPLIT_CHUNK, "chunk {chunk} < {MIN_SPLIT_CHUNK}");
            assert_eq!(splits, r.div_ceil(chunk), "no wasted empty tail chunks");
            assert!((splits - 1) * chunk < r && r <= splits * chunk);
        }
    }
}
