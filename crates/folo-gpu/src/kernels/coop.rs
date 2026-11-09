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

use std::fmt::Write as _;
use std::sync::Arc;

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::UNARY_FN;
use crate::kernels::conv::Conv2dDesc;
use crate::kernels::gemm::{BatchStrides, GemmBias, GemmDesc, gemm_params};
use crate::pipeline::{Access, Kernel, KernelDef, Params, Source, Spec};

/// Invocations per workgroup.
const THREADS: u32 = 256;
/// Subgroups per workgroup (32-wide subgroups are required, see [`unavailable_reason`]).
const SUBGROUPS: u32 = 8;
/// Halves of padding at the end of every workgroup-tile row.
const PAD: u32 = 8;
/// f32 elements of one 16×16 fragment.
const FRAG: u32 = 256;

/// Whether `adapter` offers cooperative matrices with the configuration the kernels use: 16×16×16, f16 `A` and
/// `B`, f32 accumulator.
pub(crate) fn adapter_supports(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && adapter.cooperative_matrix_properties().iter().any(|p| {
            p.m_size == 16
                && p.n_size == 16
                && p.k_size == 16
                && p.ab_type == wgpu::CooperativeScalarType::F16
                && p.cr_type == wgpu::CooperativeScalarType::F32
        })
}

/// Why the tensor-core kernels cannot run on `adapter` with the device `features` FOLO is about to request, or
/// `None` when they can be tried.
pub(crate) fn unavailable_reason(adapter: &wgpu::Adapter, features: wgpu::Features) -> Option<String> {
    let info = adapter.get_info();
    if !features.contains(wgpu::Features::SHADER_F16) {
        return Some("needs shader-f16 (missing, or FOLO_GPU_NO_F16 is set)".into());
    }
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return Some(format!("{} ({:?}) has no cooperative matrices in wgpu", info.name, info.backend));
    }
    if !adapter_supports(adapter) {
        return Some(format!("{} has no 16x16x16 f16 -> f32 cooperative matrix configuration", info.name));
    }
    if !adapter.features().contains(wgpu::Features::SUBGROUP) {
        return Some(format!("{} has no subgroup support", info.name));
    }
    if info.subgroup_min_size != 32 || info.subgroup_max_size != 32 {
        return Some(format!(
            "the kernels assume 32-wide subgroups; {} has {}..{}",
            info.name, info.subgroup_min_size, info.subgroup_max_size
        ));
    }
    None
}

/// Why the tensor-core kernels cannot run on `adapter` at all (no `shader-f16`, no 16×16×16 f16 → f32
/// cooperative matrices, no 32-wide subgroups), or `None` when it qualifies: a context created on it with
/// [`crate::GpuOptions::tensor_cores`] then has them unless the device refuses or the self-test fails (which tests
/// treat as a bug, not a skip).
pub fn adapter_unavailable_reason(adapter: &wgpu::Adapter) -> Option<String> {
    unavailable_reason(adapter, adapter.features() & wgpu::Features::SHADER_F16)
}

/// Device features the tensor-core kernels need on top of FOLO's usual ones.
pub(crate) const FEATURES: wgpu::Features =
    wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX.union(wgpu::Features::SUBGROUP);

/// wgpu's opt-in for `EXPERIMENTAL_*` features. This is the one place FOLO enables them.
///
/// # Safety conditions (why this is sound for FOLO)
///
/// `ExperimentalFeatures::enabled` is `unsafe` because wgpu's experimental features are not fully validated:
/// misusing them can be undefined behaviour on the GPU (out-of-bounds memory access, driver faults) even from
/// safe code. FOLO uses exactly one of them, `EXPERIMENTAL_COOPERATIVE_MATRIX`, and only under these
/// conditions, which together keep every cooperative-matrix access in bounds and well defined:
///
/// 1. The device is created with it only when the user opted in (`FOLO_GPU_TENSOR_CORES=1`, `--tensor-cores`,
///    or `GpuOptions::tensor_cores`) and the adapter reports the exact 16×16×16 f16 × f16 → f32 configuration
///    with 32-wide subgroups ([`unavailable_reason`]).
/// 2. Only the kernels generated in this module (and the self-test probe) contain
///    `enable wgpu_cooperative_matrix;`, and they only `coopLoadT` / `coopStoreT` through **workgroup** arrays of
///    compile-time size. Every fragment's offset is a compile-time constant plus a subgroup index clamped to
///    `0..8`, and its stride is a constant; the generator asserts that all 16 rows of every fragment lie inside
///    the array ([`gen_coop`]). No cooperative load or store touches a storage buffer, so the missing bounds
///    checks on cooperative accesses cannot reach outside a binding: global memory is read and written only by
///    ordinary WGSL loads and stores, which wgpu bounds-checks (robust buffer access on Vulkan).
/// 3. Every `coopLoadT`, `coopMultiplyAdd` and `coopStoreT` runs in workgroup-uniform control flow: the trip
///    counts and conditions around them depend only on uniform parameters, never on the invocation, as
///    subgroup-scope operations require. Workgroups are 256 invocations, a multiple of the subgroup size, and
///    which fragments a subgroup owns comes from `@builtin(subgroup_id)`, so it is uniform within the subgroup.
/// 4. A device must pass [`self_test`] (the generated kernels against a CPU reference, plus a probe) before the
///    kernels are used, so a driver that mis-compiles them is caught at start-up; on failure the experimental
///    device is dropped and the context is rebuilt on a plain one.
/// 5. No other code gets the experimental device through a safe API: `GpuContext::device()` is `None` on such a
///    context, and the raw device is only reachable through the `unsafe` `GpuContext::device_unchecked`, whose
///    contract is conditions 2 and 3.
pub(crate) fn experimental_opt_in() -> wgpu::ExperimentalFeatures {
    // SAFETY: the only experimental feature FOLO requests is `EXPERIMENTAL_COOPERATIVE_MATRIX`, and it is used
    // only by this module's kernels under conditions 1-4 above.
    unsafe { wgpu::ExperimentalFeatures::enabled() }
}

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

/// One operand-tile load plan: element `u` of this invocation covers inner unit `tid % inner` of line
/// `tid / inner + u · (256 / inner)`; `guard` is set when fewer than 256 units exist.
struct Lines {
    inner: u32,
    count: u32,
    guard: Option<u32>,
}

fn lines(inner: u32, units: u32) -> Lines {
    Lines { inner, count: units.div_ceil(THREADS), guard: (units < THREADS).then_some(units) }
}

impl Lines {
    fn outer(&self, u: u32) -> String {
        format!("(tid / {}u + {}u)", self.inner, u * (THREADS / self.inner))
    }

    fn inner_expr(&self) -> String {
        format!("(tid % {}u)", self.inner)
    }

    /// Wraps `body` in the "fewer than 256 units" guard.
    fn guarded(&self, body: String) -> String {
        match self.guard {
            Some(n) => format!("    if (tid < {n}u) {{\n{body}    }}\n"),
            None => body,
        }
    }
}

pub(crate) fn gen_coop(spec: &Spec) -> String {
    let cfg = CoopConfig {
        bm: spec.get_u32("BM", 128),
        bn: spec.get_u32("BN", 128),
        bk: spec.get_u32("BK", 32),
        wgm: spec.get_u32("WGM", 2),
        wgn: spec.get_u32("WGN", 4),
    };
    assert!(cfg.valid(), "invalid tensor-core config {cfg:?}");
    let CoopConfig { bm, bn, bk, wgm: _, wgn } = cfg;
    let (fm, fnn) = (cfg.fm(), cfg.fnn());
    let (sa, sb) = (bk + PAD, bn + PAD);
    let (va, vb, vc) = (spec.get_bool("VEC_A"), spec.get_bool("VEC_B"), spec.get_bool("VEC_C"));
    let (ta, tb) = (spec.get_bool("TRANS_A"), spec.get_bool("TRANS_B"));
    let check = spec.get_bool("CHECK");
    // 0 = plain GEMM, 1 = conv forward, 2 = conv input gradient, 3 = conv weight gradient (as `super::gemm`).
    let mode = spec.get_u32("CONV_MODE", 0);
    let conv = mode != 0;
    let split = spec.get_bool("SPLIT");
    let bias = spec.get_u32("BIAS_KIND", 0);
    let ct = if spec.get_bool("OUT_F16") { "f16" } else { "f32" };
    let (a_len, b_len) = (bm * sa, bk * sb);

    // Condition 2 of `experimental_opt_in`: every fragment lies inside its workgroup array. The largest
    // subgroup index is 7 (clamped in the shader), so these are the furthest fragments.
    let a_last = ((cfg.wgm - 1) * fm * 16 + (fm - 1) * 16) * sa + (bk - 16);
    assert!(a_last + 15 * sa + 16 <= a_len, "A fragment out of bounds");
    let b_last = (bk - 16) * sb + (wgn - 1) * fnn * 16 + (fnn - 1) * 16;
    assert!(b_last + 15 * sb + 16 <= b_len, "B fragment out of bounds");
    const { assert!((SUBGROUPS - 1) * FRAG + 15 * 16 + 16 <= SUBGROUPS * FRAG, "scratch fragment out of bounds") };

    let vty = |v: bool, scalar: &str| if v { format!("vec4<{scalar}>") } else { scalar.to_string() };
    let mut s = String::new();
    s.push_str(UNARY_FN);
    let _ = writeln!(s, "alias CT = {ct};");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> MA: array<{}>;", vty(va, "f16"));
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read> MB: array<{}>;", vty(vb && !conv, "f16"));
    s.push_str("@group(0) @binding(3) var<storage, read> MBIAS: array<CT>;\n");
    let _ = writeln!(s, "@group(0) @binding(4) var<storage, read_write> MC: array<{}>;", vty(vc, "CT"));
    for b in 0..2 {
        let _ = writeln!(s, "var<workgroup> As{b}: array<f16, {a_len}>;");
        let _ = writeln!(s, "var<workgroup> Bs{b}: array<f16, {b_len}>;");
    }
    let _ = writeln!(s, "var<workgroup> Cs: array<f32, {}>;", SUBGROUPS * FRAG);
    let _ = writeln!(s, "@compute @workgroup_size({THREADS})");
    s.push_str(
        "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32,\n\
         \x20       @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {\n\
         \x20 let M = pu(0u); let N = pu(1u);\n\
         \x20 let lda = pu(3u); let ldb = pu(4u); let ldc = pu(5u);\n",
    );
    if split {
        // wid.z = batch index * S + chunk; chunk `sidx` covers k in [kbeg, kbeg + KS) and writes its own C slab.
        // All of it derives from `wid` and uniforms, so the K loop stays workgroup-uniform.
        s.push_str(
            "  let S = pu(28u); let zz = wid.z / S; let sidx = wid.z % S;\n\
             \x20 let b1 = zz / pu(6u); let b2 = zz % pu(6u);\n\
             \x20 let kbeg = sidx * pu(29u); let K = min(pu(2u), kbeg + pu(29u));\n\
             \x20 let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u) + sidx * pu(30u);\n",
        );
    } else {
        s.push_str(
            "  let b1 = wid.z / pu(6u); let b2 = wid.z % pu(6u);\n\
             \x20 let kbeg = 0u; let K = pu(2u);\n\
             \x20 let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u);\n",
        );
    }
    s.push_str(
        "  let offA = pu(8u) + b1 * pu(9u) + b2 * pu(10u);\n\
         \x20 let offB = pu(11u) + b1 * pu(12u) + b2 * pu(13u);\n\
         \x20 let offBias = pu(18u) + b1 * pu(19u) + b2 * pu(20u);\n",
    );
    let _ = writeln!(s, "  let m0 = wid.y * {bm}u; let n0 = wid.x * {bn}u;");
    // The clamp keeps the fragment offsets in bounds whatever the driver reports (safety condition 2).
    let _ =
        writeln!(s, "  let sg = min(sgid, {}u); let wm = sg / {wgn}u; let wn = sg % {wgn}u;", SUBGROUPS - 1);

    // --- tile-load plans ------------------------------------------------------------------------------------
    let pa = match (va, ta) {
        (true, false) => lines(bk / 4, bm * bk / 4),
        (false, false) => lines(bk, bm * bk),
        (true, true) => lines(bm / 4, bm * bk / 4),
        (false, true) => lines(bm, bm * bk),
    };
    let pb = if conv {
        lines(bn, bk * bn)
    } else {
        match (vb, tb) {
            (true, false) => lines(bn / 4, bn * bk / 4),
            (false, false) => lines(bn, bn * bk),
            (true, true) => lines(bk / 4, bn * bk / 4),
            (false, true) => lines(bk, bn * bk),
        }
    };
    // One B column per invocation for the whole K loop, so its index math is done once.
    match mode {
        // Forward: column = output pixel; B[k = (ci, kh, kw)][j] = X[ci][oh·SH − PH + kh·DH][ow·SW − PW + kw·DW].
        1 => {
            let _ = writeln!(
                s,
                "  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let coh = cj / OW; let cow = cj % OW;\n\
                 \x20 let cih0 = i32(coh * SH) - PH; let ciw0 = i32(cow * SW) - PW;",
                pb.inner_expr()
            );
        }
        // Input gradient: column = input pixel; B[k = (co, kh, kw)][j] =
        // G[co][(ih + PH − kh·DH) / SH][(iw + PW − kw·DW) / SW] where both divisions are exact and in range, else 0.
        2 => {
            let _ = writeln!(
                s,
                "  let IW = pu(25u); let OH = pu(26u); let OW = pu(27u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let cih = i32(cj / IW) + PH; let ciw = i32(cj % IW) + PW;",
                pb.inner_expr()
            );
        }
        // Weight gradient: k = (n, oh, ow) runs over the whole batch, column = (ci, kh, kw);
        // A[m][k] = G[n][m][oh, ow], B[k][j] = X[n][ci][oh·SH − PH + kh·DH][ow·SW − PW + kw·DW].
        3 => {
            let _ = writeln!(
                s,
                "  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u);\n\
                 \x20 let OHOW = pu(31u); let SAN = pu(32u); let SBN = pu(33u);\n\
                 \x20 let cj = n0 + {}; let cjok = cj < N;\n\
                 \x20 let cci = cj / (KH * KW); let cr = cj % (KH * KW);\n\
                 \x20 let cdh = i32((cr / KW) * DH) - PH; let cdw = i32((cr % KW) * DW) - PW;",
                pb.inner_expr()
            );
        }
        _ => {}
    }
    let a_reg = if va { "vec4<f16>" } else { "f16" };
    let b_reg = if vb && !conv { "vec4<f16>" } else { "f16" };
    for u in 0..pa.count {
        let _ = writeln!(s, "  var ra{u} = {a_reg}(0.0);");
    }
    for u in 0..pb.count {
        let _ = writeln!(s, "  var rb{u} = {b_reg}(0.0);");
    }
    for i in 0..fm {
        for j in 0..fnn {
            let _ = writeln!(s, "  var acc{i}_{j} = coop_mat16x16<f32, C>();");
        }
    }
    let _ = writeln!(s, "  let nt = (K - kbeg + {bk}u - 1u) / {bk}u;");

    // --- global → registers (ordinary bounds-checked loads) --------------------------------------------------
    let mut load = String::new();
    for u in 0..pa.count {
        let (o, i) = (pa.outer(u), pa.inner_expr());
        let (idx, cond) = match (va, ta) {
            // Weight gradient: k = (n, pixel) folds the batch into K and the gradient rows are `OHOW` long (a
            // multiple of 4 for the vec4 form, so four consecutive k stay in one image).
            (true, false) if mode == 3 => (
                format!(
                    "(offA + ((k0 + {i} * 4u) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + {i} * 4u) % OHOW) / 4u"
                ),
                format!("m0 + {o} < M && k0 + {i} * 4u < K"),
            ),
            (false, false) if mode == 3 => (
                format!("offA + ((k0 + {i}) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + {i}) % OHOW"),
                format!("m0 + {o} < M && k0 + {i} < K"),
            ),
            (true, false) => (
                format!("offA / 4u + (m0 + {o}) * (lda / 4u) + k0 / 4u + {i}"),
                format!("m0 + {o} < M && k0 + {i} * 4u < K"),
            ),
            (false, false) => {
                (format!("offA + (m0 + {o}) * lda + k0 + {i}"), format!("m0 + {o} < M && k0 + {i} < K"))
            }
            (true, true) => (
                format!("offA / 4u + (k0 + {o}) * (lda / 4u) + m0 / 4u + {i}"),
                format!("k0 + {o} < K && m0 + {i} * 4u < M"),
            ),
            (false, true) => {
                (format!("offA + (k0 + {o}) * lda + m0 + {i}"), format!("k0 + {o} < K && m0 + {i} < M"))
            }
        };
        let body = if check {
            format!("    if ({cond}) {{ ra{u} = MA[{idx}]; }} else {{ ra{u} = {a_reg}(0.0); }}\n")
        } else {
            format!("    ra{u} = MA[{idx}];\n")
        };
        load.push_str(&pa.guarded(body));
    }
    for u in 0..pb.count {
        let (o, i) = (pb.outer(u), pb.inner_expr());
        let body = if mode == 1 {
            let _ = i;
            format!(
                "    {{ let k = k0 + {o}; let ci = k / (KH * KW); let r = k % (KH * KW);\n\
                 \x20     let ih = cih0 + i32((r / KW) * DH); let iw = ciw0 + i32((r % KW) * DW);\n\
                 \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                 \x20       rb{u} = MB[offB + (ci * IH + u32(ih)) * IW + u32(iw)];\n\
                 \x20     }} else {{ rb{u} = f16(0.0); }} }}\n"
            )
        } else if mode == 2 {
            format!(
                "    {{ let k = k0 + {o}; let co = k / (KH * KW); let r = k % (KH * KW);\n\
                 \x20     let th = cih - i32((r / KW) * DH); let tw = ciw - i32((r % KW) * DW);\n\
                 \x20     var v = f16(0.0);\n\
                 \x20     if (cjok && k < K && th >= 0 && tw >= 0) {{\n\
                 \x20       let uh = u32(th); let uw = u32(tw);\n\
                 \x20       if (uh % SH == 0u && uw % SW == 0u && uh / SH < OH && uw / SW < OW) {{\n\
                 \x20         v = MB[offB + (co * OH + uh / SH) * OW + uw / SW];\n\
                 \x20       }}\n\
                 \x20     }}\n\
                 \x20     rb{u} = v; }}\n"
            )
        } else if mode == 3 {
            format!(
                "    {{ let k = k0 + {o}; let nn = k / OHOW; let r = k % OHOW;\n\
                 \x20     let ih = i32((r / OW) * SH) + cdh; let iw = i32((r % OW) * SW) + cdw;\n\
                 \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                 \x20       rb{u} = MB[offB + nn * SBN + (cci * IH + u32(ih)) * IW + u32(iw)];\n\
                 \x20     }} else {{ rb{u} = f16(0.0); }} }}\n"
            )
        } else {
            let (idx, cond) = match (vb, tb) {
                (true, false) => (
                    format!("offB / 4u + (k0 + {o}) * (ldb / 4u) + n0 / 4u + {i}"),
                    format!("k0 + {o} < K && n0 + {i} * 4u < N"),
                ),
                (false, false) => {
                    (format!("offB + (k0 + {o}) * ldb + n0 + {i}"), format!("k0 + {o} < K && n0 + {i} < N"))
                }
                (true, true) => (
                    format!("offB / 4u + (n0 + {o}) * (ldb / 4u) + k0 / 4u + {i}"),
                    format!("n0 + {o} < N && k0 + {i} * 4u < K"),
                ),
                (false, true) => {
                    (format!("offB + (n0 + {o}) * ldb + k0 + {i}"), format!("n0 + {o} < N && k0 + {i} < K"))
                }
            };
            if check {
                format!("    if ({cond}) {{ rb{u} = MB[{idx}]; }} else {{ rb{u} = {b_reg}(0.0); }}\n")
            } else {
                format!("    rb{u} = MB[{idx}];\n")
            }
        };
        load.push_str(&pb.guarded(body));
    }

    // --- registers → workgroup tiles (row-major `As[m][k]`, `Bs[k][n]`) --------------------------------------
    let store = |buf: u32| -> String {
        let mut st = String::new();
        for u in 0..pa.count {
            let (o, i) = (pa.outer(u), pa.inner_expr());
            let mut body = String::new();
            match (va, ta) {
                (true, false) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    As{buf}[{o} * {sa}u + {i} * 4u + {c}u] = ra{u}[{c}];");
                    }
                }
                (false, false) => {
                    let _ = writeln!(body, "    As{buf}[{o} * {sa}u + {i}] = ra{u};");
                }
                (true, true) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    As{buf}[({i} * 4u + {c}u) * {sa}u + {o}] = ra{u}[{c}];");
                    }
                }
                (false, true) => {
                    let _ = writeln!(body, "    As{buf}[{i} * {sa}u + {o}] = ra{u};");
                }
            }
            st.push_str(&pa.guarded(body));
        }
        for u in 0..pb.count {
            let (o, i) = (pb.outer(u), pb.inner_expr());
            let mut body = String::new();
            match (vb && !conv, tb && !conv) {
                (true, false) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    Bs{buf}[{o} * {sb}u + {i} * 4u + {c}u] = rb{u}[{c}];");
                    }
                }
                (false, false) => {
                    let _ = writeln!(body, "    Bs{buf}[{o} * {sb}u + {i}] = rb{u};");
                }
                (true, true) => {
                    for c in 0..4 {
                        let _ = writeln!(body, "    Bs{buf}[({i} * 4u + {c}u) * {sb}u + {o}] = rb{u}[{c}];");
                    }
                }
                (false, true) => {
                    let _ = writeln!(body, "    Bs{buf}[{i} * {sb}u + {o}] = rb{u};");
                }
            }
            st.push_str(&pb.guarded(body));
        }
        st
    };

    // --- tensor-core multiply-accumulate on buffer `buf` (uniform control flow) ------------------------------
    let compute = |buf: u32| -> String {
        let mut c = String::new();
        for kk in 0..bk / 16 {
            c.push_str("    {\n");
            for i in 0..fm {
                let _ = writeln!(
                    c,
                    "      let fa{i} = coopLoadT<coop_mat16x16<f16, A>>(&As{buf}[(wm * {}u + {}u) * {sa}u + {}u], {sa}u);",
                    fm * 16,
                    i * 16,
                    kk * 16
                );
            }
            for j in 0..fnn {
                let _ = writeln!(
                    c,
                    "      let fb{j} = coopLoadT<coop_mat16x16<f16, B>>(&Bs{buf}[{}u + wn * {}u + {}u], {sb}u);",
                    kk * 16 * sb,
                    fnn * 16,
                    j * 16
                );
            }
            for i in 0..fm {
                for j in 0..fnn {
                    let _ = writeln!(c, "      acc{i}_{j} = coopMultiplyAdd(fa{i}, fb{j}, acc{i}_{j});");
                }
            }
            c.push_str("    }\n");
        }
        c
    };

    // --- main loop: two K tiles per iteration so the tile buffers are named statically -----------------------
    // The tensor-core ops must be in uniform control flow, and naga treats anything derived from the loop
    // counter as non-uniform. A `break` does not taint the statements after it, so each half starts with a
    // `break` check instead of being wrapped in an `if`.
    let _ = writeln!(s, "  {{\n    let k0 = kbeg;\n{load}  }}");
    let _ = writeln!(s, "  {{\n{}  }}", store(0));
    s.push_str("  workgroupBarrier();\n  var t = 0u;\n  loop {\n");
    for (cur, next) in [(0, 1), (1, 0)] {
        let _ = writeln!(
            s,
            "   if (t >= nt) {{ break; }}\n   {{\n    let more = t + 1u < nt;\n    if (more) {{\n    let k0 = kbeg + (t + 1u) * {bk}u;\n{load}    }}\n{}\
             \x20   if (more) {{\n{}    }}\n    workgroupBarrier();\n   }}\n   t += 1u;",
            compute(cur),
            store(next)
        );
    }
    s.push_str("  }\n\n");

    // --- epilogue: one fragment at a time through the subgroup's scratch slice ---------------------------------
    s.push_str(
        "  let alpha = pf(7u); let act_alpha = pf(17u);\n\
         \x20 let er = (lane & 31u) >> 1u; let ec = (lane & 1u) * 8u;\n",
    );
    let _ = writeln!(s, "  let cbase = sg * {FRAG}u;");
    for i in 0..fm {
        for j in 0..fnn {
            let _ = writeln!(s, "  coopStoreT(acc{i}_{j}, &Cs[cbase], 16u);");
            s.push_str("  workgroupBarrier();\n  {\n");
            let _ = writeln!(
                s,
                "    let row = m0 + wm * {}u + {}u + er; let col = n0 + wn * {}u + {}u + ec;",
                fm * 16,
                i * 16,
                fnn * 16,
                j * 16
            );
            s.push_str(
                "    let e = cbase + er * 16u + ec;\n\
                 \x20   var v0 = vec4<f32>(Cs[e], Cs[e + 1u], Cs[e + 2u], Cs[e + 3u]) * alpha;\n\
                 \x20   var v1 = vec4<f32>(Cs[e + 4u], Cs[e + 5u], Cs[e + 6u], Cs[e + 7u]) * alpha;\n",
            );
            if check {
                s.push_str("    if (row < M) {\n");
            } else {
                s.push_str("    {\n");
            }
            match bias {
                1 => s.push_str(
                    "      let bv = vec4<f32>(f32(MBIAS[offBias + row]));\n      v0 += bv; v1 += bv;\n",
                ),
                2 => {
                    for (v, base) in [("v0", 0), ("v1", 4)] {
                        let terms: Vec<String> = (0..4)
                            .map(|c| format!("f32(MBIAS[offBias + min(col + {}u, N - 1u)])", base + c))
                            .collect();
                        let _ = writeln!(s, "      {v} += vec4<f32>({});", terms.join(", "));
                    }
                }
                _ => {}
            }
            s.push_str("      v0 = unary4(ACT, v0, act_alpha); v1 = unary4(ACT, v1, act_alpha);\n");
            let at = "offC + row * ldc + col";
            if vc {
                if check {
                    let _ = writeln!(s, "      if (col < N) {{ MC[({at}) / 4u] = vec4<CT>(v0); }}");
                    let _ = writeln!(s, "      if (col + 4u < N) {{ MC[({at}) / 4u + 1u] = vec4<CT>(v1); }}");
                } else {
                    let _ = writeln!(s, "      MC[({at}) / 4u] = vec4<CT>(v0);");
                    let _ = writeln!(s, "      MC[({at}) / 4u + 1u] = vec4<CT>(v1);");
                }
            } else {
                for c in 0..8 {
                    let (v, x) = if c < 4 { ("v0", c) } else { ("v1", c - 4) };
                    if check {
                        let _ =
                            writeln!(s, "      if (col + {c}u < N) {{ MC[{at} + {c}u] = CT({v}[{x}]); }}");
                    } else {
                        let _ = writeln!(s, "      MC[{at} + {c}u] = CT({v}[{x}]);");
                    }
                }
            }
            s.push_str("    }\n  }\n  workgroupBarrier();\n");
        }
    }
    s.push_str("}\n");
    s
}

pub(crate) static COOP: KernelDef = KernelDef {
    name: "coop_gemm",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_coop),
};

fn base_spec(d: &GemmDesc, out: DType, cfg: CoopConfig) -> Spec {
    Spec::new()
        .u32("BM", cfg.bm)
        .u32("BN", cfg.bn)
        .u32("BK", cfg.bk)
        .u32("WGM", cfg.wgm)
        .u32("WGN", cfg.wgn)
        .u32("ACT", d.act.code())
        .u32(
            "BIAS_KIND",
            match d.bias {
                GemmBias::None => 0,
                GemmBias::Row => 1,
                GemmBias::Col => 2,
            },
        )
        .bool("OUT_F16", out == DType::F16)
        .f16(true)
        .coop(true)
}

/// Builds (or fetches) a tensor-core pipeline. A build failure turns tensor cores off for the context and returns
/// `None`, so the caller can fall back to the regular kernel.
fn build(ctx: &GpuContext, spec: Spec) -> Option<Arc<Kernel>> {
    match ctx.kernel(&COOP, spec) {
        Ok(k) => Some(k),
        Err(e) => {
            ctx.disable_tensor_cores(format!("a tensor-core kernel failed to build ({e})"));
            None
        }
    }
}

fn check_io(ctx: &GpuContext, input: DType, out: DType) -> Result<()> {
    if !ctx.caps().coop_f16 {
        return Err(GpuError::Unsupported("tensor cores (cooperative matrices; see FOLO_GPU_TENSOR_CORES)"));
    }
    if input != DType::F16 {
        return Err(GpuError::DType { op: "tensor-core gemm input", dtype: input });
    }
    let _ = out; // f16 or f32: both are supported.
    Ok(())
}

/// Highest element index (exclusive) a strided batched matrix operand touches.
fn extent(s: &BatchStrides, batch: (usize, usize), rows: usize, cols: usize, ld: usize) -> Result<usize> {
    if rows == 0 || cols == 0 || batch.0 == 0 || batch.1 == 0 {
        return Ok(0);
    }
    (|| {
        s.offset
            .checked_add((batch.0 - 1).checked_mul(s.outer)?)?
            .checked_add((batch.1 - 1).checked_mul(s.inner)?)?
            .checked_add((rows - 1).checked_mul(ld)?)?
            .checked_add(cols)
    })()
    .ok_or_else(|| GpuError::Limit(format!("gemm operand extent overflows ({s:?}, {rows}x{cols}, ld {ld})")))
}

fn aligned4(s: &BatchStrides) -> bool {
    s.offset.is_multiple_of(4) && s.outer.is_multiple_of(4) && s.inner.is_multiple_of(4)
}

/// Tensor-core GEMM: `C = alpha · op(A) · op(B) (+ bias) → act` with f16 `A` and `B` (`d.dtype` must be
/// [`DType::F16`]), f32 accumulation, and `C` and the bias stored as `out` (f16 or f32). Fails with
/// [`GpuError::Unsupported`] unless the context has tensor cores ([`crate::Capabilities::coop_f16`]); ignores the
/// run-time switch ([`GpuContext::set_tensor_cores`]), which only governs the automatic routing in
/// [`super::gemm::gemm`].
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    ctx: &GpuContext,
    d: &GemmDesc,
    out: DType,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<()> {
    check_io(ctx, d.dtype, out)?;
    let (ar, ac) = if d.trans_a { (d.k, d.m) } else { (d.m, d.k) };
    let (br, bc) = if d.trans_b { (d.n, d.k) } else { (d.k, d.n) };
    if d.lda < ac || d.ldb < bc || d.ldc < d.n {
        return Err(shape_err(format!("gemm: leading dimensions too small in {d:?}")));
    }
    if d.batch.1 == 0 {
        return Err(shape_err("gemm: inner batch count must be non-zero"));
    }
    if d.m == 0 || d.n == 0 || d.batch.0 == 0 {
        return Ok(());
    }
    check_buf(a, extent(&d.a, d.batch, ar, ac, d.lda)?, DType::F16, "gemm A")?;
    check_buf(b, extent(&d.b, d.batch, br, bc, d.ldb)?, DType::F16, "gemm B")?;
    check_buf(c, extent(&d.c, d.batch, d.m, d.n, d.ldc)?, out, "gemm C")?;
    let bias_len = match d.bias {
        GemmBias::None => 0,
        GemmBias::Row => d.m,
        GemmBias::Col => d.n,
    };
    let bias_buf = match (d.bias, bias) {
        (GemmBias::None, _) => ctx.dummy(),
        (_, Some(bb)) => {
            check_buf(bb, extent(&d.bias_strides, d.batch, 1, bias_len, bias_len)?, out, "gemm bias")?;
            bb
        }
        (_, None) => return Err(shape_err("gemm: bias kind set but no bias buffer")),
    };
    if !encode_gemm(ctx, d, out, a, b, bias_buf, c, cfg)? {
        return Err(GpuError::Unsupported("tensor cores (a tensor-core kernel failed to build)"));
    }
    Ok(())
}

/// Encodes a validated GEMM on tensor cores. Returns `false` (nothing encoded) when the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gemm(
    ctx: &GpuContext,
    d: &GemmDesc,
    out: DType,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    c: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<bool> {
    let batch = d.batch.0.saturating_mul(d.batch.1);
    let Some(cfg) = resolve(ctx, cfg, || pick_config(d.m, d.n, batch))? else { return Ok(false) };
    let vec_a = d.lda.is_multiple_of(4)
        && aligned4(&d.a)
        && if d.trans_a { d.m.is_multiple_of(4) } else { d.k.is_multiple_of(4) };
    let vec_b = d.ldb.is_multiple_of(4)
        && aligned4(&d.b)
        && if d.trans_b { d.k.is_multiple_of(4) } else { d.n.is_multiple_of(4) };
    let vec_c = d.ldc.is_multiple_of(4) && aligned4(&d.c) && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, out, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", d.trans_a)
        .bool("TRANS_B", d.trans_b)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0);
    let Some(k) = build(ctx, spec) else { return Ok(false) };
    let groups = [
        u32::try_from(d.n.div_ceil(cfg.bn as usize))
            .map_err(|_| GpuError::Limit("gemm N too large".into()))?,
        u32::try_from(d.m.div_ceil(cfg.bm as usize))
            .map_err(|_| GpuError::Limit("gemm M too large".into()))?,
        u32::try_from(batch).map_err(|_| GpuError::Limit("gemm batch too large".into()))?,
    ];
    ctx.dispatch(&k, &[a, b, bias, c], &gemm_params(d)?, groups)?;
    Ok(true)
}

/// Tensor-core conv2d: `y = act(conv2d(x, weight) + bias)` with f16 `x` and `weight` (`d.dtype` must be
/// [`DType::F16`]), f32 accumulation, and `y` and the bias stored as `out` (f16 or f32). 1×1 stride-1 convs run as
/// the tensor-core GEMM, other non-depthwise convs as the implicit-GEMM conv; depthwise convs have no tensor-core
/// path (one output row per group) and fail with [`GpuError::Unsupported`], as does a context without tensor cores.
/// Ignores the run-time switch like [`gemm`]. Returns `(OH, OW)`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    out: DType,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<(usize, usize)> {
    check_io(ctx, d.dtype, out)?;
    let (oh, ow) = d.check_buffers(x, weight, bias, y, out)?;
    if d.is_depthwise() {
        return Err(GpuError::Unsupported("tensor-core depthwise conv (use conv::conv2d)"));
    }
    if d.n * d.c_out * oh * ow == 0 {
        return Ok((oh, ow));
    }
    let g = d.gemm_view(oh, ow, bias.is_some());
    let bias = bias.unwrap_or(ctx.dummy());
    let built = if d.is_pointwise() {
        encode_gemm(ctx, &g, out, weight, x, bias, y, cfg)?
    } else {
        encode_conv(ctx, d, &g, out, x, weight, bias, y, cfg)?
    };
    if !built {
        return Err(GpuError::Unsupported("tensor cores (a tensor-core kernel failed to build)"));
    }
    Ok((oh, ow))
}

/// The implicit-GEMM conv on tensor cores for a validated, non-depthwise, non-pointwise f16 conv `d` whose GEMM
/// view is `g` (as built by [`super::conv::conv2d`]). Returns `false` (nothing encoded) when the kernel failed to
/// build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    out: DType,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    bias: &wgpu::Buffer,
    y: &wgpu::Buffer,
    cfg: Option<CoopConfig>,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let Some(cfg) = resolve(ctx, cfg, || pick_config(g.m, g.n, d.n * d.groups))? else { return Ok(false) };
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let mut spec = base_spec(g, out, cfg)
        .bool("VEC_A", g.k.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 1);
    spec = with_window(spec, d);
    let Some(kernel) = build(ctx, spec) else { return Ok(false) };
    let p = gemm_params(g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    let groups = grid(g, cfg, d.n * d.groups)?;
    ctx.dispatch(&kernel, &[weight, x, bias, y], &p, groups)?;
    Ok(true)
}

/// The conv window as the generator's specialisation constants.
fn with_window(spec: Spec, d: &Conv2dDesc) -> Spec {
    let (kh, kw) = d.win.kernel;
    let mut spec = spec;
    for (name, v) in [
        ("KH", kh as u32),
        ("KW", kw as u32),
        ("SH", d.win.stride.0 as u32),
        ("SW", d.win.stride.1 as u32),
        ("DH", d.win.dilation.0 as u32),
        ("DW", d.win.dilation.1 as u32),
    ] {
        spec = spec.u32(name, v);
    }
    spec.i32("PH", d.win.padding.0 as i32).i32("PW", d.win.padding.1 as i32)
}

/// Workgroup grid of a GEMM view `g` with `layers` z-layers.
fn grid(g: &GemmDesc, cfg: CoopConfig, layers: usize) -> Result<[u32; 3]> {
    Ok([
        u32::try_from(g.n.div_ceil(cfg.bn as usize))
            .map_err(|_| GpuError::Limit("conv output too large".into()))?,
        u32::try_from(g.m.div_ceil(cfg.bm as usize))
            .map_err(|_| GpuError::Limit("conv rows too large".into()))?,
        u32::try_from(layers)
            .ok()
            .filter(|&z| z <= 65535)
            .ok_or_else(|| GpuError::Limit("conv batch too large".into()))?,
    ])
}

/// The input gradient of the f16 conv `d` on tensor cores (`CONV_MODE` 2, a transposed conv as implicit GEMM):
/// `g` is the GEMM view built by [`super::conv::conv2d_backward_input`] (`M = C_in/groups`, `N = H·W`,
/// `K = C_out/groups·KH·KW`), `wt` the weight permuted to `[groups, C_in/groups, C_out/groups, KH·KW]`, `grad` the
/// output gradient and `gx` the f16 result. Returns `false` (nothing encoded) when the kernel failed to build.
pub(crate) fn encode_conv_backward_input(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    wt: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    gx: &wgpu::Buffer,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let Some(cfg) = resolve(ctx, None, || pick_config(g.m, g.n, d.n * d.groups))? else { return Ok(false) };
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !g.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(g, DType::F16, cfg)
        .bool("VEC_A", g.k.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 2);
    let Some(kernel) = build(ctx, with_window(spec, d)) else { return Ok(false) };
    let p = gemm_params(g)?.at(24).uz(d.h)?.uz(d.w)?.uz(oh)?.uz(ow)?;
    ctx.dispatch(&kernel, &[wt, grad, ctx.dummy(), gx], &p, grid(g, cfg, d.n * d.groups)?)?;
    Ok(true)
}

/// Tile configuration for a weight gradient (`M = C_out/groups` rows, `N = C_in/groups·KH·KW` columns): by `M`
/// alone, since split-K supplies the occupancy.
fn pick_config_weight_grad(m: usize, n: usize) -> CoopConfig {
    match m {
        0..=16 => CoopConfig::C16X256,
        17..=32 => CoopConfig::C32X128,
        _ if m.div_ceil(128) * 128 * 4 <= m * 5 && n >= 128 => CoopConfig::C128X128,
        _ if n >= 128 => CoopConfig::C64X128,
        _ => CoopConfig::C64X64,
    }
}

/// Upper bound for split-K partial slabs, in f32 elements (64 MiB), as for the regular kernels.
const MAX_PARTIAL_ELEMS: usize = 16 << 20;

/// The weight gradient of the f16 conv `d` on tensor cores (`CONV_MODE` 3): `K = N·OH·OW` folds the batch into
/// the reduction, `A` rows are output-gradient planes read across images, `B` the image gathered as a transposed
/// im2col. `g` is the GEMM view built by [`super::conv::conv2d_backward_weight`]. `K` is split into chunks (one
/// z-layer each, f32 partial slabs summed by a column reduction) when the weight alone has too few tiles to fill
/// the GPU. `gw` is written as `out` (f32 or f16; partial sums stay f32). Returns `false` (nothing encoded) when
/// the kernel failed to build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_conv_backward_weight(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    g: &GemmDesc,
    out: DType,
    grad: &wgpu::Buffer,
    x: &wgpu::Buffer,
    gw: &wgpu::Buffer,
) -> Result<bool> {
    let (oh, ow) = d.output()?;
    let Some(cfg) = resolve(ctx, None, || pick_config_weight_grad(g.m, g.n))? else { return Ok(false) };
    let (ohow, np, wn) = (oh * ow, g.k, g.m * g.n * d.groups);
    let bk = cfg.bk as usize;
    // Aim for ~512 workgroups, each at least 16 K tiles deep.
    let tiles = g.m.div_ceil(cfg.bm as usize) * g.n.div_ceil(cfg.bn as usize) * d.groups;
    let want = 512usize
        .div_ceil(tiles)
        .min(np.div_ceil(16 * bk))
        .min(MAX_PARTIAL_ELEMS / wn.max(1))
        .min(65535 / d.groups)
        .max(1);
    let ks = np.div_ceil(want).next_multiple_of(bk);
    let splits = np.div_ceil(ks);
    // Partial slabs (and an f16 result) go through an f32 buffer and a column reduction / cast.
    let tmp = if splits > 1 || out != DType::F32 {
        Some(ctx.alloc(DType::F32.bytes(super::prod(&[splits, wn])?))?)
    } else {
        None
    };
    let target = tmp.as_ref().map_or(gw, |t| t.raw());
    let check = !g.m.is_multiple_of(cfg.bm as usize)
        || !g.n.is_multiple_of(cfg.bn as usize)
        || !np.is_multiple_of(bk);
    let spec = base_spec(g, DType::F32, cfg)
        .bool("VEC_A", ohow.is_multiple_of(4))
        .bool("VEC_B", false)
        .bool("VEC_C", g.n.is_multiple_of(4))
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 3)
        .bool("SPLIT", splits > 1);
    let Some(kernel) = build(ctx, with_window(spec, d)) else { return Ok(false) };
    let p = gemm_params(g)?
        .at(24)
        .uz(d.h)?
        .uz(d.w)?
        .uz(oh)?
        .uz(ow)?
        .uz(splits)?
        .uz(ks)?
        .uz(wn)?
        .uz(ohow)?
        .uz(d.c_out * ohow)?
        .uz(d.c_in * d.h * d.w)?;
    ctx.dispatch(&kernel, &[grad, x, ctx.dummy(), target], &p, grid(g, cfg, d.groups * splits)?)?;
    if let Some(t) = &tmp {
        if splits > 1 {
            let op = super::reduce::ReduceOp::Sum;
            super::reduce::reduce_as(ctx, op, (t.raw(), DType::F32), (gw, out), 1, splits, wn)?;
        } else {
            super::layout::cast(ctx, t.raw(), gw, wn, true)?;
        }
    }
    Ok(true)
}

static PROBE: KernelDef = KernelDef {
    name: "coop_probe",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Static(
        "@group(0) @binding(1) var<storage, read> GA: array<f16>;\n\
         @group(0) @binding(2) var<storage, read> GB: array<f16>;\n\
         @group(0) @binding(3) var<storage, read_write> GC: array<f32>;\n\
         var<workgroup> sa: array<f16, 256>;\n\
         var<workgroup> sb: array<f16, 256>;\n\
         var<workgroup> sc: array<f32, 256>;\n\
         @compute @workgroup_size(32)\n\
         fn main(@builtin(local_invocation_index) lid: u32) {\n\
           for (var i = lid; i < 256u; i += 32u) { sa[i] = GA[i]; sb[i] = GB[i]; }\n\
           workgroupBarrier();\n\
           let a = coopLoadT<coop_mat16x16<f16, A>>(&sa[0], 16u);\n\
           let b = coopLoadT<coop_mat16x16<f16, B>>(&sb[0], 16u);\n\
           var c = coop_mat16x16<f32, C>();\n\
           c = coopMultiplyAdd(a, b, c);\n\
           coopStoreT(c, &sc[0], 16u);\n\
           workgroupBarrier();\n\
           for (var i = lid; i < 256u; i += 32u) { GC[i] = sc[i]; }\n\
         }\n",
    ),
};

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
}

/// Runs the tensor-core kernels against a CPU reference: a single-fragment probe, then a ragged GEMM through the
/// generated kernel with a row bias and SiLU, f16 and f32 output. Called while creating a context that has
/// cooperative matrices, before they are used. Values are small multiples of powers of two, so the f16 inputs
/// and every product are exact and only the f32 summation order and the output rounding differ.
pub(crate) fn self_test(ctx: &GpuContext) -> Result<()> {
    let a: Vec<f32> = (0..256).map(|i| ((i * 7 % 13) as f32 - 6.0) / 4.0).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0).collect();
    let ga = ctx.upload(&to_f16_bytes(&a))?;
    let gb = ctx.upload(&to_f16_bytes(&b))?;
    let gc = ctx.alloc(256 * 4)?;
    let k = ctx.kernel(&PROBE, Spec::new().f16(true).coop(true))?;
    ctx.dispatch(&k, &[ga.raw(), gb.raw(), gc.raw()], &Params::new(), [1, 1, 1])?;
    let got: Vec<f32> = bytemuck::pod_collect_to_vec(&ctx.read(gc.raw(), 0, 256 * 4)?);
    for i in 0..16 {
        for j in 0..16 {
            let want: f32 = (0..16).map(|k| a[i * 16 + k] * b[k * 16 + j]).sum();
            let g = got[i * 16 + j];
            if (g - want).abs() > 1e-3 * (1.0 + want.abs()) {
                return Err(GpuError::Wgpu(format!("probe C[{i}][{j}] = {g}, expected {want}")));
            }
        }
    }

    // A ragged GEMM through the generated kernel: every tile edge, the bias and the activation.
    let (m, n, kd) = (70usize, 90usize, 75usize);
    let a: Vec<f32> = (0..m * kd).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
    let b: Vec<f32> = (0..kd * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 16.0).collect();
    let bias: Vec<f32> = (0..m).map(|i| (i % 5) as f32 / 4.0 - 0.5).collect();
    let mut d = GemmDesc::new(m, n, kd, false, false, 1);
    d.dtype = DType::F16;
    d.bias = GemmBias::Row;
    d.act = crate::UnaryOp::Silu;
    let (ga, gb) = (ctx.upload(&to_f16_bytes(&a))?, ctx.upload(&to_f16_bytes(&b))?);
    for out in [DType::F32, DType::F16] {
        let gbias = if out == DType::F16 {
            ctx.upload(&to_f16_bytes(&bias))?
        } else {
            ctx.upload(bytemuck::cast_slice(&bias))?
        };
        let gc = ctx.alloc(out.bytes(m * n))?;
        let built = encode_gemm(ctx, &d, out, ga.raw(), gb.raw(), gbias.raw(), gc.raw(), None)?;
        if !built {
            // `build` recorded the compiler's message in the status.
            return Err(GpuError::Wgpu(ctx.tensor_cores_status()));
        }
        let bytes = ctx.read(gc.raw(), 0, out.bytes(m * n))?;
        let got: Vec<f32> = match out {
            DType::F16 => {
                bytes.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect()
            }
            _ => bytemuck::pod_collect_to_vec(&bytes),
        };
        for i in 0..m {
            for j in 0..n {
                let z: f32 = (0..kd).map(|k| a[i * kd + k] * b[k * n + j]).sum::<f32>() + bias[i];
                let want = z / (1.0 + (-z).exp());
                let g = got[i * n + j];
                // f16 output: one rounding (2^-11 relative); f32: summation order and the SiLU's `exp`.
                let tol = if out == DType::F16 { 2e-3 } else { 1e-4 };
                let close = (g - want).abs() <= tol * (1.0 + want.abs()); // false for NaN
                if !close {
                    return Err(GpuError::Wgpu(format!("{out:?} GEMM C[{i}][{j}] = {g}, expected {want}")));
                }
            }
        }
    }
    Ok(())
}
