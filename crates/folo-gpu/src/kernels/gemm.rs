//! Tiled GEMM: `C = alpha · op(A) · op(B) (+ bias) → act`, batched, with transposed variants and f16 storage.
//!
//! One WGSL generator covers every variant. A workgroup computes a `BM × BN` tile of `C`; each thread owns a
//! `TM × TN` block held in registers as `vec4<f32>` accumulators. `K` is walked in steps of `BK`: the next
//! `A` and `B` tiles are fetched from global memory into registers while the current tiles in workgroup
//! memory are consumed (double-buffered workgroup memory, one barrier per step). Tiles are stored `k`-major
//! (`As[k][m]`, `Bs[k][n]`) so the inner loop reads `vec4`s of both operands. Global loads are `vec4` when the
//! leading dimension and offsets allow it; bounds checks are compiled out when `M`, `N`, `K` are tile
//! multiples. The same generator also does implicit-GEMM convolution (see [`super::conv`]): `CONV_MODE` 1
//! gathers `B` straight from an NCHW image (forward), 2 gathers it from the output gradient with the
//! transposed-conv rule (input gradient), and 3 gathers an im2col-transposed image tile while reading `A` from
//! the output gradient across the batch (weight gradient). `SPLIT` splits `K` into chunks, one per extra
//! workgroup layer, each writing its own partial result (split-K; the caller sums the partials).

use std::fmt::Write as _;

use super::check_buf;
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::elementwise::UnaryOp;
use crate::kernels::{UNARY_FN, coop};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};

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

    fn aligned4(&self) -> bool {
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

/// Tile configuration (exposed for benchmarks; [`gemm`] picks one automatically).
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

fn wgsl_ty(f16: bool, vec: bool) -> &'static str {
    match (f16, vec) {
        (false, false) => "f32",
        (false, true) => "vec4<f32>",
        (true, false) => "f16",
        (true, true) => "vec4<f16>",
    }
}

/// How one operand tile is fetched: `inner` units along the contiguous direction per `outer` line.
struct Plan {
    vec: bool,
    /// Units (vec4 or scalar) along the contiguous direction of the tile.
    inner: u32,
    /// Units per thread.
    count: u32,
    /// Outer lines advanced per unit step.
    step: u32,
}

fn plan(units_inner: u32, total_units: u32, threads: u32, vec: bool) -> Plan {
    Plan { vec, inner: units_inner, count: total_units / threads, step: threads / units_inner }
}

pub(crate) fn gen_gemm(spec: &Spec) -> String {
    let bm = spec.get_u32("BM", 128);
    let bn = spec.get_u32("BN", 128);
    let bk = spec.get_u32("BK", 16);
    let tm = spec.get_u32("TM", 8);
    let tn = spec.get_u32("TN", 8);
    let (va, vb, vc) = (spec.get_bool("VEC_A"), spec.get_bool("VEC_B"), spec.get_bool("VEC_C"));
    let (ta, tb) = (spec.get_bool("TRANS_A"), spec.get_bool("TRANS_B"));
    let check = spec.get_bool("CHECK");
    // 0 = plain GEMM, 1 = conv forward, 2 = conv input gradient, 3 = conv weight gradient.
    let mode = spec.get_u32("CONV_MODE", 0);
    let conv = mode != 0;
    // Input gradient of one stride phase (`super::conv`'s stride-phase decomposition).
    let phase = mode == 2 && spec.get_bool("PHASE");
    let split = spec.get_bool("SPLIT");
    let bias = spec.get_u32("BIAS_KIND", 0);
    // `CAT` > 1: `B` is that many row blocks in separate buffers (`super::gemm_cat`; plain GEMM, `B` not transposed).
    let cat = spec.get_u32("CAT", 1);
    let f16 = spec.is_f16();
    // `OUT_F32`: f16 operands but an f32 `C` (the weight gradient's f32 partial slabs / f32 result).
    let c_f16 = f16 && !spec.get_bool("OUT_F32");
    let wgx = bn / tn;
    let wgy = bm / tm;
    let threads = wgx * wgy;
    let (bm4, bn4) = (bm / 4, bn / 4);

    // Operand tile fetch plans (see the module docs for the layouts).
    let pa = match (va, ta) {
        (true, false) => plan(bk / 4, bm * bk / 4, threads, true),
        (true, true) => plan(bm4, bm * bk / 4, threads, true),
        (false, false) => plan(bk, bm * bk, threads, false),
        (false, true) => plan(bm, bm * bk, threads, false),
    };
    let pb = match (vb, tb) {
        (true, false) => plan(bn4, bn * bk / 4, threads, true),
        (true, true) => plan(bk / 4, bn * bk / 4, threads, true),
        (false, false) => plan(bn, bn * bk, threads, false),
        (false, true) => plan(bk, bn * bk, threads, false),
    };

    let mut s = String::new();
    s.push_str(UNARY_FN);
    if phase {
        s.push_str(super::conv::PHASE_PIXEL);
    }
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> A: array<{}>;", wgsl_ty(f16, va));
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read> B: array<{}>;", wgsl_ty(f16, vb && !conv));
    s.push_str("@group(0) @binding(3) var<storage, read> BIAS: array<T>;\n");
    let _ = writeln!(s, "@group(0) @binding(4) var<storage, read_write> C: array<{}>;", wgsl_ty(c_f16, vc));
    if cat > 1 {
        s.push_str(&super::gemm_cat::cat_source(cat, "B", wgsl_ty(f16, vb), vb));
    }
    let _ = writeln!(s, "var<workgroup> As: array<array<vec4<f32>, {bm4}>, {}>;", 2 * bk);
    // `WG_PAD`: the weight gradient pads each B row by one vec4 (`2 · BK · 16` bytes, which
    // `TileConfig::workgroup_bytes` leaves out, so the caller sets it only where that still fits): its stores go
    // down short runs of rows (see `wgrad_b_units`), which would otherwise all hit the same banks.
    let wg_pad = mode == 3 && spec.get_bool("WG_PAD");
    let _ =
        writeln!(s, "var<workgroup> Bs: array<array<vec4<f32>, {}>, {}>;", bn4 + u32::from(wg_pad), 2 * bk);
    let _ = writeln!(s, "@compute @workgroup_size({wgx}, {wgy}, 1)");
    s.push_str(
        "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {\n",
    );
    let _ = writeln!(s, "  let tx = lid3.x; let ty = lid3.y; let tid = ty * {wgx}u + tx;");
    // Runs of 16 pixels where the tile is that deep (two 32-byte sectors per column and load).
    let wb =
        (mode == 3).then(|| super::conv::wgrad_b_units(threads, bk, bn, spec.get_u32("WG_RUN", bk.min(16))));
    if let Some(wb) = &wb {
        let _ = writeln!(s, "  let wl = tid % {0}u; let wq = tid / {0}u;", wb.run);
    }
    s.push_str(
        "  let M = pu(0u); let N = pu(1u);\n\
         let lda = pu(3u); let ldb = pu(4u); let ldc = pu(5u);\n",
    );
    if split {
        // wid.z = batch index * S + chunk; chunk `sidx` covers k in [kbeg, kbeg + KS) and writes its own C slab.
        s.push_str(
            "  let S = pu(28u); let zz = wid.z / S; let sidx = wid.z % S;\n\
             let b1 = zz / pu(6u); let b2 = zz % pu(6u);\n\
             let kbeg = sidx * pu(29u); let K = min(pu(2u), kbeg + pu(29u));\n\
             let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u) + sidx * pu(30u);\n",
        );
    } else {
        s.push_str(
            "  let b1 = wid.z / pu(6u); let b2 = wid.z % pu(6u);\n\
             let kbeg = 0u; let K = pu(2u);\n\
             let offC = pu(14u) + b1 * pu(15u) + b2 * pu(16u);\n",
        );
    }
    s.push_str(
        "  let offA = pu(8u) + b1 * pu(9u) + b2 * pu(10u);\n\
         let offB = pu(11u) + b1 * pu(12u) + b2 * pu(13u);\n\
         let offBias = pu(18u) + b1 * pu(19u) + b2 * pu(20u);\n",
    );
    let _ = writeln!(s, "  let m0 = wid.y * {bm}u; let n0 = wid.x * {bn}u;");
    match mode {
        // Forward gather: B[k][j] = X[ci][oh*SH - PH + kh*DH][ow*SW - PW + kw*DW].
        1 => s.push_str("  let IH = pu(24u); let IW = pu(25u); let OW = pu(27u);\n"),
        // Input gradient: B[k = (co, kh, kw)][j = (ih, iw)] = G[co][(ih + PH - kh*DH) / SH][(iw + PW - kw*DW) / SW]
        // where the divisions are exact and in range, else 0.
        2 => {
            s.push_str("  let IH = pu(24u); let IW = pu(25u); let OH = pu(26u); let OW = pu(27u);\n");
            if phase {
                s.push_str("  let WR = pu(33u);\n");
            }
        }
        // Weight gradient: k = (n, oh, ow) runs over the whole batch; A[m][k] = G[n][m][oh, ow] and
        // B[k][j = (ci, kh, kw)] = X[n][ci][oh*SH - PH + kh*DH][ow*SW - PW + kw*DW].
        3 => {
            s.push_str(&super::conv::weight_grad_dims(spec));
            s.push_str("  let SAN = pu(32u); let SBN = pu(33u);\n");
        }
        _ => {}
    }
    // Per-thread tile coordinates for the loads.
    let _ = writeln!(s, "  let ai = tid % {}u; let ao = tid / {}u;", pa.inner, pa.inner);
    let _ = writeln!(s, "  let bi = tid % {}u; let bo = tid / {}u;", pb.inner, pb.inner);
    // One B column per thread for the whole K loop, so its index math is done once.
    match mode {
        1 => s.push_str(
            "  let cj = n0 + bi;\n  let coh = cj / OW; let cow = cj % OW;\n\
             let cih0 = i32(coh * SH) - PH; let ciw0 = i32(cow * SW) - PW;\n\
             let cjok = cj < N;\n",
        ),
        2 if phase => s.push_str(
            "  let cj = n0 + bi; let cjok = cj < N;\n\
             let ca = pi(30u) + i32(cj / WR); let cb = pi(31u) + i32(cj % WR);\n",
        ),
        2 => s.push_str(
            "  let cj = n0 + bi; let cjok = cj < N;\n\
             let cih = i32(cj / IW) + PH; let ciw = i32(cj % IW) + PW;\n",
        ),
        _ => {}
    }
    let _ =
        writeln!(s, "  var ra: array<vec4<f32>, {}>;", pa.count.div_ceil(if pa.vec { 1 } else { 4 }).max(1));
    let _ =
        writeln!(s, "  var rb: array<vec4<f32>, {}>;", pb.count.div_ceil(if pb.vec { 1 } else { 4 }).max(1));
    for i in 0..tm {
        for j in 0..tn / 4 {
            let _ = writeln!(s, "  var acc_{i}_{j} = vec4<f32>(0.0);");
        }
    }
    s.push_str("  let nt = (K - kbeg + BK - 1u) / BK;\n");

    // --- global → registers -------------------------------------------------------------------------------
    let mut load = String::new();
    for u in 0..pa.count {
        let o = format!("(ao + {}u)", u * pa.step);
        let (reg, comp) = if pa.vec {
            (format!("ra[{u}]"), String::new())
        } else {
            (format!("ra[{}]", u / 4), format!("[{}]", u % 4))
        };
        let (idx, cond) = match (pa.vec, ta) {
            // Weight gradient: k = (n, pixel) folds the batch into K; the gradient rows are `OHOW` long.
            (true, false) if mode == 3 => (
                format!(
                    "(offA + ((k0 + ai * 4u) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + ai * 4u) % OHOW) / 4u"
                ),
                format!("m0 + {o} < M && k0 + ai * 4u < K"),
            ),
            (false, false) if mode == 3 => (
                format!("offA + ((k0 + ai) / OHOW) * SAN + (m0 + {o}) * OHOW + (k0 + ai) % OHOW"),
                format!("m0 + {o} < M && k0 + ai < K"),
            ),
            (true, false) => (
                format!("(offA / 4u) + (m0 + {o}) * (lda / 4u) + ((k0 >> 2u) + ai)"),
                format!("m0 + {o} < M && k0 + ai * 4u < K"),
            ),
            (true, true) => (
                format!("(offA / 4u) + (k0 + {o}) * (lda / 4u) + ((m0 >> 2u) + ai)"),
                format!("k0 + {o} < K && m0 + ai * 4u < M"),
            ),
            (false, false) => {
                (format!("offA + (m0 + {o}) * lda + k0 + ai"), format!("m0 + {o} < M && k0 + ai < K"))
            }
            (false, true) => {
                (format!("offA + (k0 + {o}) * lda + m0 + ai"), format!("k0 + {o} < K && m0 + ai < M"))
            }
        };
        let val = if pa.vec { format!("vec4<f32>(A[{idx}])") } else { format!("f32(A[{idx}])") };
        let zero = if pa.vec { "vec4<f32>(0.0)" } else { "0.0" };
        if check {
            let _ =
                writeln!(load, "    if ({cond}) {{ {reg}{comp} = {val}; }} else {{ {reg}{comp} = {zero}; }}");
        } else {
            let _ = writeln!(load, "    {reg}{comp} = {val};");
        }
    }
    if let Some(wb) = &wb {
        let reg = |u: usize| format!("rb[{}][{}]", u / 4, u % 4);
        load.push_str(&super::conv::wgrad_b_load(wb, "B", reg, "f32", "0.0"));
    }
    for u in 0..pb.count {
        let o = format!("(bo + {}u)", u * pb.step);
        let (reg, comp) = if pb.vec {
            (format!("rb[{u}]"), String::new())
        } else {
            (format!("rb[{}]", u / 4), format!("[{}]", u % 4))
        };
        match mode {
            1 => {
                // Unit (k = k0 + o, j = cj): gather from the image.
                let _ = writeln!(
                    load,
                    "    {{ let k = k0 + {o}; let ci = k / (KH * KW); let r = k % (KH * KW);\n\
                     \x20     let ih = cih0 + i32((r / KW) * DH); let iw = ciw0 + i32((r % KW) * DW);\n\
                     \x20     if (cjok && k < K && ih >= 0 && ih < i32(IH) && iw >= 0 && iw < i32(IW)) {{\n\
                     \x20       {reg}{comp} = f32(B[offB + (ci * IH + u32(ih)) * IW + u32(iw)]);\n\
                     \x20     }} else {{ {reg}{comp} = 0.0; }} }}"
                );
                continue;
            }
            2 if phase => {
                // Unit (k = (co, th, tw), j = phase pixel (a, b)): output pixel (BH + a - th, BW + b - tw).
                let _ = writeln!(
                    load,
                    "    {{ let k = k0 + {o}; let co = k / (PTH * PTW); let r = k % (PTH * PTW);\n\
                     \x20     let oh = ca - i32(r / PTW); let ow = cb - i32(r % PTW);\n\
                     \x20     var v = 0.0;\n\
                     \x20     if (cjok && k < K && oh >= 0 && ow >= 0 && oh < i32(OH) && ow < i32(OW)) {{\n\
                     \x20       v = f32(B[offB + (co * OH + u32(oh)) * OW + u32(ow)]);\n\
                     \x20     }}\n\
                     \x20     {reg}{comp} = v; }}"
                );
                continue;
            }
            2 => {
                // Unit (k = (co, kh, kw), j = input pixel): the output pixel that read it, if any.
                let _ = writeln!(
                    load,
                    "    {{ let k = k0 + {o}; let co = k / (KH * KW); let r = k % (KH * KW);\n\
                     \x20     let th = cih - i32((r / KW) * DH); let tw = ciw - i32((r % KW) * DW);\n\
                     \x20     var v = 0.0;\n\
                     \x20     if (cjok && k < K && th >= 0 && tw >= 0) {{\n\
                     \x20       let uh = u32(th); let uw = u32(tw);\n\
                     \x20       if (uh % SH == 0u && uw % SW == 0u && uh / SH < OH && uw / SW < OW) {{\n\
                     \x20         v = f32(B[offB + (co * OH + uh / SH) * OW + uw / SW]);\n\
                     \x20       }}\n\
                     \x20     }}\n\
                     \x20     {reg}{comp} = v; }}"
                );
                continue;
            }
            // Loaded above (`wgrad_b_load`).
            3 => continue,
            _ => {}
        }
        let (idx, cond) = match (pb.vec, tb) {
            (true, false) => (
                format!("(offB / 4u) + (k0 + {o}) * (ldb / 4u) + ((n0 >> 2u) + bi)"),
                format!("k0 + {o} < K && n0 + bi * 4u < N"),
            ),
            (true, true) => (
                format!("(offB / 4u) + (n0 + {o}) * (ldb / 4u) + ((k0 >> 2u) + bi)"),
                format!("n0 + {o} < N && k0 + bi * 4u < K"),
            ),
            (false, false) => {
                (format!("offB + (k0 + {o}) * ldb + n0 + bi"), format!("k0 + {o} < K && n0 + bi < N"))
            }
            (false, true) => {
                (format!("offB + (n0 + {o}) * ldb + k0 + bi"), format!("n0 + {o} < N && k0 + bi < K"))
            }
        };
        let fetch = match (cat > 1, pb.vec) {
            (true, true) => format!("cat_b(k0 + {o}, (n0 >> 2u) + bi, b1)"),
            (true, false) => format!("cat_b(k0 + {o}, n0 + bi, b1)"),
            (false, _) => format!("B[{idx}]"),
        };
        let val = if pb.vec { format!("vec4<f32>({fetch})") } else { format!("f32({fetch})") };
        let zero = if pb.vec { "vec4<f32>(0.0)" } else { "0.0" };
        if check {
            let _ =
                writeln!(load, "    if ({cond}) {{ {reg}{comp} = {val}; }} else {{ {reg}{comp} = {zero}; }}");
        } else {
            let _ = writeln!(load, "    {reg}{comp} = {val};");
        }
    }

    // --- registers → workgroup memory (buffer `sb`, a constant) -------------------------------------------
    // Tiles are 2-D arrays `As[k][m / 4]`: with the buffer index and `k` constant and the column index
    // loop-invariant, the index clamps naga adds for workgroup arrays fold or hoist out of the main loop.
    let store = |sb: u32| -> String {
        let mut st = String::new();
        let (ka, kb) = (sb * bk, sb * bk);
        for u in 0..pa.count {
            let o = format!("(ao + {}u)", u * pa.step);
            let (x, y) = (u / 4, u % 4);
            match (pa.vec, ta) {
                (true, false) => {
                    // Unit (m = o, k = 4·ai .. 4·ai+3): scatter four scalars down the k-major tile.
                    for c in 0..4 {
                        let _ = writeln!(
                            st,
                            "    As[{ka}u + ai * 4u + {c}u][{o} >> 2u][{o} & 3u] = ra[{u}][{c}];"
                        );
                    }
                }
                (true, true) => {
                    let _ = writeln!(st, "    As[{ka}u + {o}][ai] = ra[{u}];");
                }
                (false, false) => {
                    let _ = writeln!(st, "    As[{ka}u + ai][{o} >> 2u][{o} & 3u] = ra[{x}][{y}];");
                }
                (false, true) => {
                    let _ = writeln!(st, "    As[{ka}u + {o}][ai >> 2u][ai & 3u] = ra[{x}][{y}];");
                }
            }
        }
        for (u, w) in wb.iter().flat_map(|wb| wb.units.iter().map(move |w| (wb.run, w))).enumerate() {
            let (run, w) = w;
            let row = format!("{kb}u + {} * {run}u + wl", w.kq);
            let _ = writeln!(st, "    Bs[{row}][{0} >> 2u][{0} & 3u] = rb[{1}][{2}];", w.jj, u / 4, u % 4);
        }
        for u in (0..pb.count).filter(|_| mode != 3) {
            let o = format!("(bo + {}u)", u * pb.step);
            let (x, y) = (u / 4, u % 4);
            match (pb.vec, tb) {
                (true, false) => {
                    let _ = writeln!(st, "    Bs[{kb}u + {o}][bi] = rb[{u}];");
                }
                (true, true) => {
                    for c in 0..4 {
                        let _ = writeln!(
                            st,
                            "    Bs[{kb}u + bi * 4u + {c}u][{o} >> 2u][{o} & 3u] = rb[{u}][{c}];"
                        );
                    }
                }
                (false, false) => {
                    let _ = writeln!(st, "    Bs[{kb}u + {o}][bi >> 2u][bi & 3u] = rb[{x}][{y}];");
                }
                (false, true) => {
                    let _ = writeln!(st, "    Bs[{kb}u + bi][{o} >> 2u][{o} & 3u] = rb[{x}][{y}];");
                }
            }
        }
        st
    };

    // --- compute on buffer `sb` ------------------------------------------------------------------------------
    // The k loop is unrolled here rather than left to the driver: naga guards every WGSL loop with an
    // iteration counter (infinite-loop protection), which blocks the driver's unrolling and costs 20-70%.
    let compute = |sb: u32| -> String {
        let mut c = String::new();
        for kk in 0..bk {
            let row = sb * bk + kk;
            c.push_str("    {\n");
            for i in 0..tm / 4 {
                let _ = writeln!(c, "      let a{i} = As[{row}u][ty + {}u];", i * wgy);
            }
            for j in 0..tn / 4 {
                let _ = writeln!(c, "      let b{j} = Bs[{row}u][tx + {}u];", j * wgx);
            }
            for i in 0..tm / 4 {
                for r in 0..4 {
                    for j in 0..tn / 4 {
                        let acc = format!("acc_{}_{j}", i * 4 + r);
                        let _ = writeln!(c, "      {acc} = fma(vec4<f32>(a{i}[{r}]), b{j}, {acc});");
                    }
                }
            }
            c.push_str("    }\n");
        }
        c
    };

    // --- main loop: two K tiles per iteration so every workgroup-memory buffer index is a constant ----------
    let _ = writeln!(s, "  {{\n    let k0 = kbeg;\n{load}  }}");
    let _ = writeln!(s, "  {{\n{}  }}", store(0));
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  for (var t = 0u; t < nt; t += 2u) {\n");
    let _ = writeln!(
        s,
        "   {{\n    let more = t + 1u < nt;\n    if (more) {{\n    let k0 = kbeg + (t + 1u) * BK;\n{load}    }}\n{}\
         \x20   if (more) {{\n{}    }}\n    workgroupBarrier();\n   }}",
        compute(0),
        store(1)
    );
    let _ = writeln!(
        s,
        "   if (t + 1u < nt) {{\n    let more = t + 2u < nt;\n    if (more) {{\n    let k0 = kbeg + (t + 2u) * BK;\n{load}    }}\n{}\
         \x20   if (more) {{\n{}    }}\n    workgroupBarrier();\n   }}",
        compute(1),
        store(0)
    );
    s.push_str("  }\n\n");

    // --- epilogue --------------------------------------------------------------------------------------------
    s.push_str("  let alpha = pf(7u); let act_alpha = pf(17u);\n");
    for i in 0..tm / 4 {
        for r in 0..4 {
            let row_i = i * 4 + r;
            let _ = writeln!(s, "  {{\n    let row = m0 + ({}u + ty) * 4u + {r}u;", i * wgy);
            s.push_str("    if (row < M) {\n");
            for j in 0..tn / 4 {
                let _ = writeln!(s, "    {{\n      let col = n0 + ({}u + tx) * 4u;", j * wgx);
                let _ = writeln!(s, "      var v = acc_{row_i}_{j} * alpha;");
                match bias {
                    1 => s.push_str("      v += vec4<f32>(f32(BIAS[offBias + row]));\n"),
                    2 => s.push_str(
                        "      v += vec4<f32>(f32(BIAS[offBias + min(col, N - 1u)]), f32(BIAS[offBias + min(col + 1u, N - 1u)]),\n\
                         \x20                f32(BIAS[offBias + min(col + 2u, N - 1u)]), f32(BIAS[offBias + min(col + 3u, N - 1u)]));\n",
                    ),
                    _ => {}
                }
                s.push_str("      v = unary4(ACT, v, act_alpha);\n");
                if vc {
                    let cast = if c_f16 { "vec4<f16>(v)" } else { "v" };
                    let _ =
                        writeln!(s, "      if (col < N) {{ C[(offC + row * ldc + col) / 4u] = {cast}; }}");
                } else {
                    if phase {
                        s.push_str("      var pp = ppix_at(col);\n");
                    }
                    for c in 0..4 {
                        let cast = if c_f16 { format!("f16(v[{c}])") } else { format!("v[{c}]") };
                        let at = if phase { "ppix_off(pp)".to_string() } else { format!("col + {c}u") };
                        let _ = writeln!(
                            s,
                            "      if (col + {c}u < N) {{ C[offC + row * ldc + {at}] = {cast}; }}"
                        );
                        if phase && c < 3 {
                            s.push_str("      ppix_next(&pp);\n");
                        }
                    }
                }
                s.push_str("    }\n");
            }
            s.push_str("    }\n  }\n");
        }
    }
    s.push_str("}\n");
    s
}

pub(crate) static GEMM: KernelDef = KernelDef {
    name: "gemm",
    storage: &[Access::Read, Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_gemm),
};

/// [`GEMM`] with `B` in up to [`super::gemm_cat::MAX_CAT_PARTS`] buffers (`CAT`; unused parts bind the dummy).
pub(crate) static GEMM_CAT: KernelDef = KernelDef {
    name: "gemm_cat",
    storage: &[
        Access::Read,
        Access::Read,
        Access::Read,
        Access::ReadWrite,
        Access::Read,
        Access::Read,
        Access::Read,
    ],
    source: Source::Generated(gen_gemm),
};

/// Highest element index (exclusive) a strided batched matrix operand touches (0 when it touches none).
pub(crate) fn extent(
    s: &BatchStrides,
    batch: (usize, usize),
    rows: usize,
    cols: usize,
    ld: usize,
) -> Result<usize> {
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

pub(crate) fn base_spec(d: &GemmDesc, cfg: TileConfig) -> Spec {
    Spec::new()
        .u32("BM", cfg.bm)
        .u32("BN", cfg.bn)
        .u32("BK", cfg.bk)
        .u32("TM", cfg.tm)
        .u32("TN", cfg.tn)
        .u32("ACT", d.act.code())
        .u32(
            "BIAS_KIND",
            match d.bias {
                GemmBias::None => 0,
                GemmBias::Row => 1,
                GemmBias::Col => 2,
            },
        )
        .f16(d.dtype == DType::F16)
}

pub(crate) fn gemm_params(d: &GemmDesc) -> Result<Params> {
    let p = Params::new()
        .uz(d.m)?
        .uz(d.n)?
        .uz(d.k)?
        .uz(d.lda)?
        .uz(d.ldb)?
        .uz(d.ldc)?
        .uz(d.batch.1)?
        .f(d.alpha)
        .uz(d.a.offset)?
        .uz(d.a.outer)?
        .uz(d.a.inner)?
        .uz(d.b.offset)?
        .uz(d.b.outer)?
        .uz(d.b.inner)?
        .uz(d.c.offset)?
        .uz(d.c.outer)?
        .uz(d.c.inner)?
        .f(d.act.alpha())
        .uz(d.bias_strides.offset)?
        .uz(d.bias_strides.outer)?
        .uz(d.bias_strides.inner)?;
    Ok(p)
}

pub(crate) fn check_dtype(ctx: &GpuContext, dtype: DType, op: &'static str) -> Result<()> {
    if dtype == DType::F16 && !ctx.caps().shader_f16 {
        return Err(GpuError::Unsupported("shader-f16"));
    }
    let _ = op;
    Ok(())
}

/// Encodes a GEMM described by `d`. `bias` is required when `d.bias != GemmBias::None`.
pub fn gemm(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
) -> Result<()> {
    gemm_impl(ctx, d, a, b, bias, c, None)
}

/// [`gemm`] with an explicit tile configuration (benchmarks and tuning).
pub fn gemm_with_config(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: TileConfig,
) -> Result<()> {
    gemm_impl(ctx, d, a, b, bias, c, Some(cfg))
}

/// [`gemm`] (`cfg == None`: tensor cores for f16 when the context has them on, else a picked tile config) or
/// [`gemm_with_config`].
fn gemm_impl(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: Option<TileConfig>,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "gemm")?;
    if let Some(cfg) = cfg {
        check_config(ctx, cfg)?;
    }
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
    check_buf(a, extent(&d.a, d.batch, ar, ac, d.lda)?, d.dtype, "gemm A")?;
    check_buf(b, extent(&d.b, d.batch, br, bc, d.ldb)?, d.dtype, "gemm B")?;
    check_buf(c, extent(&d.c, d.batch, d.m, d.n, d.ldc)?, d.dtype, "gemm C")?;
    let bias_len = match d.bias {
        GemmBias::None => 0,
        GemmBias::Row => d.m,
        GemmBias::Col => d.n,
    };
    let bias_buf = match (d.bias, bias) {
        (GemmBias::None, _) => ctx.dummy(),
        (_, Some(bb)) => {
            check_buf(bb, extent(&d.bias_strides, d.batch, 1, bias_len, bias_len)?, d.dtype, "gemm bias")?;
            bb
        }
        (_, None) => return Err(shape_err("gemm: bias kind set but no bias buffer")),
    };
    if cfg.is_none()
        && d.dtype == DType::F16
        && ctx.tensor_cores()
        && coop::encode_gemm(ctx, d, DType::F16, a, b, bias_buf, c, None)?
    {
        return Ok(());
    }
    let cfg = cfg.unwrap_or_else(|| pick_config(ctx, d));
    check_config(ctx, cfg)?;
    let vec_a = d.lda.is_multiple_of(4)
        && d.a.aligned4()
        && if d.trans_a { d.m.is_multiple_of(4) } else { d.k.is_multiple_of(4) };
    let vec_b = d.ldb.is_multiple_of(4)
        && d.b.aligned4()
        && if d.trans_b { d.k.is_multiple_of(4) } else { d.n.is_multiple_of(4) };
    let vec_c = d.ldc.is_multiple_of(4) && d.c.aligned4() && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", d.trans_a)
        .bool("TRANS_B", d.trans_b)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0);
    let k = ctx.kernel(&GEMM, spec)?;
    let groups = [
        d.n.div_ceil(cfg.bn as usize) as u32,
        d.m.div_ceil(cfg.bm as usize) as u32,
        d.batch
            .0
            .checked_mul(d.batch.1)
            .and_then(|b| u32::try_from(b).ok())
            .ok_or_else(|| GpuError::Limit("gemm batch too large".into()))?,
    ];
    ctx.dispatch(&k, &[a, b, bias_buf, c], &gemm_params(d)?, groups)
}
