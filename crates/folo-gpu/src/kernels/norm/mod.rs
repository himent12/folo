//! Train-mode BatchNorm over the channels of NCHW data, with a fused activation, forward and backward. The data
//! (`x`, `y`, the gradients `g` and `dx`) is f32 or f16 ([`BnDesc::dtype`], mixed-precision training); the
//! parameters, the statistics and every sum are f32.
//!
//! Dispatch tiers (both forward and backward, `vec4` when `H·W` and view offsets/strides are multiples of 4):
//! - **Single-dispatch register-cached** (`fused.rs`, `bn_fused_fwd`/`bn_fused_bwd` with `REGS = 4 | 8`, up to
//!   2,048 units = 8,192 elements/channel under `VEC4`): one workgroup per channel holds its channel slice in
//!   registers and reads `X` (and `G`) once (`WG = 512, REGS = 4` in backward for `1024 < m_u <= 2048, C <= 256`
//!   to halve per-thread register pressure; `WG = 256` otherwise).
//! - **Single-dispatch streaming two-pass** (`fused.rs`, `REGS = 0`, `WG = 512` when supported else `256`, up to
//!   6,400 units with `C >= 32` and 25,600 units = 102,400 elements/channel with `C >= 64`; backward requires
//!   `max_storage_buffers >= 9`): one workgroup per channel runs anchor-centered Welford/Chan statistics (or
//!   2×-unrolled `dw`/`db` reduction) in Pass 1 and normalizes / computes `dx` from L2-warm tensors in Pass 2
//!   with zero intermediate buffers.
//! - **Three-dispatch split-K** (larger planes or narrower `C`):
//!   - Forward: `bn_stats` (grid `(C, chunks)`, per-chunk Welford/Chan partials) -> `bn_finalize` (merge,
//!     `mean`/`invstd`, optional running stats) -> `bn_apply` (division-free 2-units/thread 3D grid `(ceil(hw/512), C, N)`).
//!   - Backward: `bn_bwd_reduce` (grid `(C, chunks)`, 2×-unrolled `dz` and `dz · x̂` sums) -> `bn_bwd_finalize`
//!     (`db`, `dw`) -> `bn_bwd_dx` (division-free 2-units/thread 3D grid `(ceil(hw/512), C, N)`).
//!
//! All three tiers support zero-copy channel-slice views (`STRIDED` specialization via [`BnView`]) so `narrow` /
//! `chunk` gradients from `cat` backprop skip materializing contiguous copies.
//!
//! Reductions use 32-wide subgroup operations (`subgroupShuffleDown` / `subgroupAdd`, 1-barrier in `bn_fused_bwd`
//! and the 3-dispatch reduction kernels) when the native device supports fixed 32-wide subgroups, or a portable
//! shared-memory tree fallback otherwise (browser WebGPU, WARP, Lavapipe).

mod fused;

#[cfg(test)]
pub(crate) use super::FORCE_NO_SUBGROUPS;
use super::{check_buf, prod, use_subgroups, vec_prelude, vec_prelude_f16};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
pub(crate) use fused::fused_fits;
use fused::{
    FUSED_BWD, FUSED_BWD_BUFFERS, FUSED_FWD, FUSED_MAX_UNITS, FUSED_STREAM_MAX_UNITS, FUSED_STREAM_MED_UNITS,
    FUSED_STREAM_MIN_CHANNELS_LARGE, FUSED_STREAM_MIN_CHANNELS_MED,
};

pub(super) const COMMON: &str = include_str!("../../shaders/bn_common.wgsl");
pub(super) const WELFORD: &str = include_str!("../../shaders/bn_welford.wgsl");

const STATS_MAIN_SUBGROUPS: &str = r#"
var<workgroup> wn: array<f32, 8>;
var<workgroup> wm: array<f32, 8>;
var<workgroup> wq: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let c = wid.x;
    let chunk = wid.y;
    let t = bn_stats_thread(c, chunk, lid);
    var na = t.x;
    var ma = t.y;
    var qa = t.z;
    // Exact pairwise Chan tree merge over the 32-wide subgroup: skips empty lanes and keeps `d == 0`
    // bit-exact on constant inputs (even partial subgroups of large F16/F32 constants).
    for (var off = 16u; off > 0u; off = off >> 1u) {
        let nb = subgroupShuffleDown(na, off);
        let mb = subgroupShuffleDown(ma, off);
        let qb = subgroupShuffleDown(qa, off);
        if (lane < off) {
            if (na == 0.0) {
                na = nb;
                ma = mb;
                qa = qb;
            } else if (nb > 0.0) {
                let n = na + nb;
                let d = mb - ma;
                let delta = d * (nb / n);
                ma += delta;
                qa += qb + d * (delta * na);
                na = n;
            }
        }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) {
        wn[sg] = na;
        wm[sg] = ma;
        wq[sg] = qa;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var na = wn[0];
        var ma = wm[0];
        var qa = wq[0];
        for (var i = 1u; i < 8u; i += 1u) {
            let nb = wn[i];
            if (na == 0.0) {
                na = nb;
                ma = wm[i];
                qa = wq[i];
            } else if (nb > 0.0) {
                let n = na + nb;
                let d = wm[i] - ma;
                let delta = d * (nb / n);
                ma += delta;
                qa += wq[i] + d * (delta * na);
                na = n;
            }
        }
        let o = (c * pu(4u) + chunk) * 3u;
        PART[o] = na;
        PART[o + 1u] = ma;
        PART[o + 2u] = qa;
    }
}
"#;

const STATS_MAIN_PORTABLE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x;
    let chunk = wid.y;
    let t = bn_stats_thread(c, chunk, lid);
    wn[lid] = t.x;
    wm[lid] = t.y;
    wq[lid] = t.z;
    workgroupBarrier();
    bn_tree_merge(lid);
    if (lid == 0u) {
        let o = (c * pu(4u) + chunk) * 3u;
        PART[o] = wn[0];
        PART[o + 1u] = wm[0];
        PART[o + 2u] = wq[0];
    }
}
"#;

const BWD_REDUCE_MAIN_SUBGROUPS: &str = r#"
var<workgroup> s1: array<f32, 8>;
var<workgroup> s2: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let c = wid.x;
    let chunk = wid.y;
    let a = bn_bwd_reduce_thread(c, chunk, lid);
    let r1 = subgroupAdd(a.x);
    let r2 = subgroupAdd(a.y);
    let sg = min(sgid, 7u);
    if (lane == 0u) {
        s1[sg] = r1;
        s2[sg] = r2;
    }
    workgroupBarrier();
    if (lid == 0u) {
        let o = (c * pu(4u) + chunk) * 2u;
        PART[o] = ((s1[0] + s1[1]) + (s1[2] + s1[3])) + ((s1[4] + s1[5]) + (s1[6] + s1[7]));
        PART[o + 1u] = ((s2[0] + s2[1]) + (s2[2] + s2[3])) + ((s2[4] + s2[5]) + (s2[6] + s2[7]));
    }
}
"#;

const BWD_REDUCE_MAIN_PORTABLE: &str = r#"
var<workgroup> s1: array<f32, 256>;
var<workgroup> s2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x;
    let chunk = wid.y;
    let a = bn_bwd_reduce_thread(c, chunk, lid);
    s1[lid] = a.x;
    s2[lid] = a.y;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            s1[lid] += s1[lid + s];
            s2[lid] += s2[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let o = (c * pu(4u) + chunk) * 2u;
        PART[o] = s1[0];
        PART[o + 1u] = s2[0];
    }
}
"#;

/// The activation fused after the normalisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BnAct {
    /// None.
    Identity,
    /// `z · sigmoid(z)`.
    Silu,
    /// `max(z, 0)`.
    Relu,
}

impl BnAct {
    fn code(self) -> u32 {
        match self {
            BnAct::Identity => 0,
            BnAct::Silu => 1,
            BnAct::Relu => 2,
        }
    }

    /// CPU reference of the activation.
    pub fn apply(self, z: f32) -> f32 {
        match self {
            BnAct::Identity => z,
            BnAct::Silu => z / (1.0 + (-z).exp()),
            BnAct::Relu => z.max(0.0),
        }
    }
}

/// A BatchNorm problem: `[n, c, hw]` values, normalised per channel over `n · hw`.
#[derive(Clone, Copy, Debug)]
pub struct BnDesc {
    /// Batch size.
    pub n: usize,
    /// Channels.
    pub c: usize,
    /// Values per plane (`H·W`).
    pub hw: usize,
    /// Activation after the affine transform.
    pub act: BnAct,
    /// Storage type of the data (`x`, `y`, `g`, `dx`): f32 or f16. Weight, bias, statistics and running statistics
    /// are always f32.
    pub dtype: DType,
}

/// The running statistics the forward updates in place.
pub struct BnRunning<'a> {
    /// Running mean, `[c]`.
    pub mean: &'a wgpu::Buffer,
    /// Running unbiased variance, `[c]`.
    pub var: &'a wgpu::Buffer,
    /// Weight of the batch statistic.
    pub momentum: f32,
}

/// An NCHW input buffer view for [`batch_norm_forward_strided`] / [`batch_norm_backward_strided`]: dense
/// channel planes (`stride[1] == H·W`, `stride[2] == W`, `stride[3] == 1`) with an element `offset` and
/// inter-image `n_stride >= C·H·W` (covering both dense tensors and `narrow`/`chunk` channel slices).
#[derive(Clone, Copy, Debug)]
pub struct BnView<'a> {
    /// Underlying storage buffer.
    pub buf: &'a wgpu::Buffer,
    /// Element offset of `[0, 0, 0, 0]`.
    pub offset: usize,
    /// Element stride between consecutive batch images (`>= c · hw`).
    pub n_stride: usize,
}

impl<'a> BnView<'a> {
    /// Dense NCHW view starting at element 0 with `n_stride = d.c * d.hw`.
    pub fn dense(buf: &'a wgpu::Buffer, d: &BnDesc) -> Self {
        Self { buf, offset: 0, n_stride: d.c.saturating_mul(d.hw) }
    }

    fn is_dense(&self, d: &BnDesc) -> bool {
        self.offset == 0 && self.n_stride == d.c * d.hw
    }

    fn validate(&self, d: &BnDesc, what: &'static str) -> Result<()> {
        let image = prod(&[d.c, d.hw])?;
        if self.n_stride < image {
            return Err(shape_err(format!("{what}: batch stride {} < image size {image}", self.n_stride)));
        }
        let span = (d.n - 1)
            .checked_mul(self.n_stride)
            .and_then(|s| s.checked_add(image))
            .and_then(|s| s.checked_add(self.offset))
            .ok_or_else(|| GpuError::Limit(format!("{what}: strided span overflows usize")))?;
        if span >= (1usize << 31) {
            return Err(GpuError::Limit(format!("{what}: strided span {span} exceeds 31-bit indexing")));
        }
        check_buf(self.buf, span, d.dtype, what)
    }
}

/// The element prelude of the passes: f32 or f16 storage (`spec.is_f16()`), scalar or vec4.
pub(super) fn prelude(spec: &Spec) -> &'static str {
    let vec = spec.get_bool("VEC4");
    if spec.is_f16() { vec_prelude_f16(vec) } else { vec_prelude(vec) }
}

fn gen_stats(spec: &Spec) -> String {
    let strided = spec.get_bool("STRIDED");
    let body = include_str!("../../shaders/bn_stats.wgsl");
    if spec.get_bool("SUBGROUPS") {
        format!("{}const IS_STRIDED: bool = {strided};\n{COMMON}{body}{STATS_MAIN_SUBGROUPS}", prelude(spec))
    } else {
        format!(
            "{}const IS_STRIDED: bool = {strided};\n{COMMON}{WELFORD}{body}{STATS_MAIN_PORTABLE}",
            prelude(spec)
        )
    }
}

const FINALIZE_MAIN_SUBGROUPS: &str = r#"
var<workgroup> wn: array<f32, 8>;
var<workgroup> wm: array<f32, 8>;
var<workgroup> wq: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let c = wid.x;
    let t = bn_finalize_thread(c, lid);
    var na = t.x;
    var ma = t.y;
    var qa = t.z;
    for (var off = 16u; off > 0u; off = off >> 1u) {
        let nb = subgroupShuffleDown(na, off);
        let mb = subgroupShuffleDown(ma, off);
        let qb = subgroupShuffleDown(qa, off);
        if (lane < off) {
            if (na == 0.0) {
                na = nb;
                ma = mb;
                qa = qb;
            } else if (nb > 0.0) {
                let n = na + nb;
                let d = mb - ma;
                let delta = d * (nb / n);
                ma += delta;
                qa += qb + d * (delta * na);
                na = n;
            }
        }
    }
    let sg = min(sgid, 7u);
    if (lane == 0u) {
        wn[sg] = na;
        wm[sg] = ma;
        wq[sg] = qa;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var na = wn[0];
        var ma = wm[0];
        var qa = wq[0];
        for (var i = 1u; i < 8u; i += 1u) {
            let nb = wn[i];
            if (na == 0.0) {
                na = nb;
                ma = wm[i];
                qa = wq[i];
            } else if (nb > 0.0) {
                let n = na + nb;
                let d = wm[i] - ma;
                let delta = d * (nb / n);
                ma += delta;
                qa += wq[i] + d * (delta * na);
                na = n;
            }
        }
        bn_finalize_write(c, na, ma, qa);
    }
}
"#;

const FINALIZE_MAIN_PORTABLE: &str = r#"
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x;
    let t = bn_finalize_thread(c, lid);
    wn[lid] = t.x;
    wm[lid] = t.y;
    wq[lid] = t.z;
    workgroupBarrier();
    bn_tree_merge(lid);
    if (lid == 0u) {
        bn_finalize_write(c, wn[0], wm[0], wq[0]);
    }
}
"#;

const BWD_FINALIZE_MAIN_SUBGROUPS: &str = r#"
var<workgroup> s1: array<f32, 8>;
var<workgroup> s2: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32,
        @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
    let c = wid.x;
    let a = bn_bwd_finalize_thread(c, lid);
    let r1 = subgroupAdd(a.x);
    let r2 = subgroupAdd(a.y);
    let sg = min(sgid, 7u);
    if (lane == 0u) {
        s1[sg] = r1;
        s2[sg] = r2;
    }
    workgroupBarrier();
    if (lid == 0u) {
        let db = ((s1[0] + s1[1]) + (s1[2] + s1[3])) + ((s1[4] + s1[5]) + (s1[6] + s1[7]));
        let dw = ((s2[0] + s2[1]) + (s2[2] + s2[3])) + ((s2[4] + s2[5]) + (s2[6] + s2[7]));
        bn_bwd_finalize_write(c, db, dw);
    }
}
"#;

const BWD_FINALIZE_MAIN_PORTABLE: &str = r#"
var<workgroup> s1: array<f32, 256>;
var<workgroup> s2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x;
    let a = bn_bwd_finalize_thread(c, lid);
    s1[lid] = a.x;
    s2[lid] = a.y;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            s1[lid] += s1[lid + s];
            s2[lid] += s2[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        bn_bwd_finalize_write(c, s1[0], s2[0]);
    }
}
"#;

fn gen_finalize(spec: &Spec) -> String {
    let body = include_str!("../../shaders/bn_finalize.wgsl");
    if spec.is_subgroups() {
        format!("{COMMON}{body}{FINALIZE_MAIN_SUBGROUPS}")
    } else {
        format!("{COMMON}{WELFORD}{body}{FINALIZE_MAIN_PORTABLE}")
    }
}

fn gen_apply(spec: &Spec) -> String {
    let grid3d = spec.get_bool("GRID3D");
    let strided = spec.get_bool("STRIDED");
    format!(
        "{}const USE_GRID3D: bool = {grid3d};\nconst IS_STRIDED: bool = {strided};\n{COMMON}{}",
        prelude(spec),
        include_str!("../../shaders/bn_apply.wgsl")
    )
}

fn gen_bwd_reduce(spec: &Spec) -> String {
    let strided = spec.get_bool("STRIDED");
    let body = include_str!("../../shaders/bn_bwd_reduce.wgsl");
    let main = if spec.get_bool("SUBGROUPS") { BWD_REDUCE_MAIN_SUBGROUPS } else { BWD_REDUCE_MAIN_PORTABLE };
    format!("{}const IS_STRIDED: bool = {strided};\n{COMMON}{body}{main}", prelude(spec))
}

fn gen_bwd_finalize(spec: &Spec) -> String {
    let body = include_str!("../../shaders/bn_bwd_finalize.wgsl");
    let main = if spec.is_subgroups() { BWD_FINALIZE_MAIN_SUBGROUPS } else { BWD_FINALIZE_MAIN_PORTABLE };
    format!("{COMMON}{body}{main}")
}

fn gen_bwd_dx(spec: &Spec) -> String {
    let grid3d = spec.get_bool("GRID3D");
    let strided = spec.get_bool("STRIDED");
    format!(
        "{}const USE_GRID3D: bool = {grid3d};\nconst IS_STRIDED: bool = {strided};\n{COMMON}{}",
        prelude(spec),
        include_str!("../../shaders/bn_bwd_dx.wgsl")
    )
}

use Access::{Read as R, ReadWrite as RW};

static STATS: KernelDef =
    KernelDef { name: "bn_stats", storage: &[R, RW], source: Source::Generated(gen_stats) };
static FINALIZE: KernelDef = KernelDef {
    name: "bn_finalize",
    storage: &[R, R, R, RW, RW, RW, RW, RW],
    source: Source::Generated(gen_finalize),
};
static APPLY: KernelDef =
    KernelDef { name: "bn_apply", storage: &[R, R, RW], source: Source::Generated(gen_apply) };
static BWD_REDUCE: KernelDef = KernelDef {
    name: "bn_bwd_reduce",
    storage: &[R, R, R, R, R, R, RW],
    source: Source::Generated(gen_bwd_reduce),
};
static BWD_FINALIZE: KernelDef = KernelDef {
    name: "bn_bwd_finalize",
    storage: &[R, R, R, R, R, RW, RW, RW],
    source: Source::Generated(gen_bwd_finalize),
};
static BWD_DX: KernelDef =
    KernelDef { name: "bn_bwd_dx", storage: &[R, R, R, RW], source: Source::Generated(gen_bwd_dx) };

/// Chunking of each channel's `m` units (`n · hw` scalar or `n · (hw/4)` vec4): `(chunks, chunk length)`.
fn plan(m: usize, c: usize, vec4: bool) -> (usize, usize) {
    let want = 2048usize.div_ceil(c.max(1));
    let min_per_wg = if vec4 { 256 * 2 } else { 256 * 4 };
    let chunks = want.min(m.div_ceil(min_per_wg)).clamp(1, 65535);
    let len = m.div_ceil(chunks);
    (m.div_ceil(len), len)
}

impl BnDesc {
    /// Validates the sizes and the dtype and returns `(total elements, n · hw)`.
    fn check(&self, ctx: &GpuContext) -> Result<(usize, usize)> {
        super::gemm::check_dtype(ctx, self.dtype, "batch_norm")?;
        let total = prod(&[self.n, self.c, self.hw])?;
        let m = self.n * self.hw;
        if m == 0 || self.c == 0 {
            return Err(shape_err(format!("batch_norm: empty problem {self:?}")));
        }
        if total >= (1usize << 31) || self.c > 65535 {
            return Err(GpuError::Limit(format!("batch_norm: {self:?} is too large for one dispatch")));
        }
        Ok((total, m))
    }

    fn walk_params(&self, hw_units: usize, m_units: usize, chunks: usize, len: usize) -> Result<Params> {
        Params::new()
            .uz(self.c)?
            .uz(hw_units)?
            .uz(m_units)?
            .uz(len)?
            .uz(chunks)?
            .uz(256 / hw_units)?
            .uz(256 % hw_units)
    }
}

/// Selects `(workgroup_size, regs)` for the single-dispatch fused BatchNorm kernels, or `None` for the
/// three-dispatch split-K path.
fn pick_fused(ctx: &GpuContext, m_u: usize, c: usize, chunks: usize, is_bwd: bool) -> Option<(u32, u32)> {
    if is_bwd && ctx.caps().max_storage_buffers < FUSED_BWD_BUFFERS {
        return None;
    }
    let caps = ctx.caps();
    let wg512 = caps.max_workgroup_invocations >= 512 && caps.max_workgroup_size_x >= 512;
    if m_u <= FUSED_MAX_UNITS && (chunks == 1 || c >= 32) {
        if m_u <= 256 * 4 {
            Some((256, 4))
        } else if is_bwd && wg512 && c <= 256 {
            Some((512, 4))
        } else {
            Some((256, 8))
        }
    } else if (m_u <= FUSED_STREAM_MED_UNITS && c >= FUSED_STREAM_MIN_CHANNELS_MED)
        || (m_u <= FUSED_STREAM_MAX_UNITS && c >= FUSED_STREAM_MIN_CHANNELS_LARGE)
    {
        Some((if wg512 { 512 } else { 256 }, 0))
    } else {
        None
    }
}

/// Whether `[ceil(hw_u / 512), c, n]` fits the device's per-dimension workgroup limit for division-free 3D apply/dx.
fn fits_grid3d(ctx: &GpuContext, hw_u: usize, c: usize, n: usize) -> Option<[u32; 3]> {
    let lim = ctx.caps().max_workgroups_per_dim as usize;
    let gx = hw_u.div_ceil(512);
    (gx <= lim && c <= lim && n <= lim).then_some([gx as u32, c as u32, n as u32])
}

/// Train-mode forward: `y = act((x − mean) · invstd · w + b)`, writing `mean` and `invstd` (`[c]` each) and
/// updating `running` if given.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward(
    ctx: &GpuContext,
    d: &BnDesc,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    eps: f32,
    running: Option<BnRunning<'_>>,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    invstd: &wgpu::Buffer,
) -> Result<()> {
    batch_norm_forward_strided(ctx, d, BnView::dense(x, d), w, b, eps, running, y, mean, invstd)
}

/// Train-mode forward reading `x` from a dense or channel-slice [`BnView`] without copying it first.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward_strided(
    ctx: &GpuContext,
    d: &BnDesc,
    x: BnView<'_>,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    eps: f32,
    running: Option<BnRunning<'_>>,
    y: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    invstd: &wgpu::Buffer,
) -> Result<()> {
    let (total, m) = d.check(ctx)?;
    x.validate(d, "batch_norm input")?;
    check_buf(y, total, d.dtype, "batch_norm output")?;
    for (buf, what) in [(w, "weight"), (b, "bias"), (mean, "mean"), (invstd, "invstd")] {
        check_buf(buf, d.c, DType::F32, what)?;
    }
    if let Some(r) = &running {
        check_buf(r.mean, d.c, DType::F32, "running_mean")?;
        check_buf(r.var, d.c, DType::F32, "running_var")?;
    }
    let strided = !x.is_dense(d);
    let vec = d.hw.is_multiple_of(4) && x.offset.is_multiple_of(4) && x.n_stride.is_multiple_of(4);
    let q = if vec { 4 } else { 1 };
    let (units, hw_u, m_u) = (total / q, d.hw / q, m / q);
    let (x_off_u, x_ns_u) = (x.offset / q, x.n_stride / q);
    let (chunks, len) = plan(m_u, d.c, vec);
    let half = d.dtype == DType::F16;
    let sg = use_subgroups(ctx);
    let mom = running.as_ref().map_or(0.0, |r| r.momentum);
    let unbias = if m > 1 { m as f64 / (m as f64 - 1.0) } else { 1.0 };
    let (rm, rv) = running.as_ref().map_or((ctx.dummy(), ctx.dummy()), |r| (r.mean, r.var));

    if let Some((wg, regs)) = pick_fused(ctx, m_u, d.c, chunks, false) {
        let mut spec = Spec::new()
            .u32("ACT", d.act.code())
            .bool("VEC4", vec)
            .bool("RUNNING", running.is_some())
            .u32("REGS", regs)
            .u32("WG", wg)
            .f16(half)
            .subgroups(sg);
        if strided {
            spec = spec.bool("STRIDED", true);
        }
        let k = ctx.kernel(&FUSED_FWD, spec)?;
        let wgz = wg as usize;
        let p = Params::new()
            .uz(d.c)?
            .uz(hw_u)?
            .uz(m_u)?
            .uz(x_off_u)?
            .uz(x_ns_u)?
            .uz(wgz / hw_u)?
            .uz(wgz % hw_u)?
            .f(eps)
            .f(1.0 - mom)
            .f(mom)
            .f(mom * unbias as f32);
        return ctx.dispatch(&k, &[x.buf, w, b, y, mean, invstd, rm, rv], &p, [d.c as u32, 1, 1]);
    }

    let part = ctx.alloc(DType::F32.bytes(prod(&[d.c, chunks, 3])?))?;
    let coef = ctx.alloc(DType::F32.bytes(4 * d.c))?;

    let mut stats_spec =
        Spec::new().u32("ACT", 0).bool("VEC4", vec).bool("SUBGROUPS", sg).f16(half).subgroups(sg);
    if strided {
        stats_spec = stats_spec.bool("STRIDED", true);
    }
    let k = ctx.kernel(&STATS, stats_spec)?;
    let mut p = d.walk_params(hw_u, m_u, chunks, len)?;
    if strided {
        p = p.uz(x_off_u)?.uz(x_ns_u)?;
    }
    ctx.dispatch(&k, &[x.buf, part.raw()], &p, [d.c as u32, chunks as u32, 1])?;

    let k =
        ctx.kernel(&FINALIZE, Spec::new().u32("ACT", 0).bool("RUNNING", running.is_some()).subgroups(sg))?;
    let p = Params::new().uz(d.c)?.uz(chunks)?.f(eps).f(1.0 - mom).f(mom).f(mom * unbias as f32);
    ctx.dispatch(&k, &[part.raw(), w, b, mean, invstd, coef.raw(), rm, rv], &p, [d.c as u32, 1, 1])?;

    let grid3d = fits_grid3d(ctx, hw_u, d.c, d.n);
    let mut apply_spec = Spec::new().u32("ACT", d.act.code()).bool("VEC4", vec).f16(half);
    if grid3d.is_some() {
        apply_spec = apply_spec.bool("GRID3D", true);
    }
    if strided {
        apply_spec = apply_spec.bool("STRIDED", true);
    }
    let k = ctx.kernel(&APPLY, apply_spec)?;
    let mut p = Params::new().uz(units)?.uz(d.c)?.uz(hw_u)?;
    if strided {
        p = p.uz(x_off_u)?.uz(x_ns_u)?;
    }
    let grid = match grid3d {
        Some(g) => g,
        None => ctx.grid_1d(units as u64, 256)?,
    };
    ctx.dispatch(&k, &[x.buf, coef.raw(), y], &p, grid)
}

/// Backward of [`batch_norm_forward`]: `dx` (like `x`), `dw` and `db` (`[c]`) from the output gradient `g`
/// and the forward's `mean` / `invstd`.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward(
    ctx: &GpuContext,
    d: &BnDesc,
    x: &wgpu::Buffer,
    g: &wgpu::Buffer,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    invstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dw: &wgpu::Buffer,
    db: &wgpu::Buffer,
) -> Result<()> {
    batch_norm_backward_strided(
        ctx,
        d,
        BnView::dense(x, d),
        BnView::dense(g, d),
        w,
        b,
        mean,
        invstd,
        dx,
        dw,
        db,
    )
}

/// Backward of [`batch_norm_forward`] reading `x` and `g` from dense or channel-slice [`BnView`]s without
/// pre-copying non-contiguous slices.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward_strided(
    ctx: &GpuContext,
    d: &BnDesc,
    x: BnView<'_>,
    g: BnView<'_>,
    w: &wgpu::Buffer,
    b: &wgpu::Buffer,
    mean: &wgpu::Buffer,
    invstd: &wgpu::Buffer,
    dx: &wgpu::Buffer,
    dw: &wgpu::Buffer,
    db: &wgpu::Buffer,
) -> Result<()> {
    let (total, m) = d.check(ctx)?;
    x.validate(d, "batch_norm input")?;
    g.validate(d, "batch_norm output gradient")?;
    check_buf(dx, total, d.dtype, "batch_norm input gradient")?;
    for (buf, what) in
        [(w, "weight"), (b, "bias"), (mean, "mean"), (invstd, "invstd"), (dw, "dw"), (db, "db")]
    {
        check_buf(buf, d.c, DType::F32, what)?;
    }
    let strided = !x.is_dense(d) || !g.is_dense(d);
    let vec = d.hw.is_multiple_of(4)
        && x.offset.is_multiple_of(4)
        && x.n_stride.is_multiple_of(4)
        && g.offset.is_multiple_of(4)
        && g.n_stride.is_multiple_of(4);
    let q = if vec { 4 } else { 1 };
    let (units, hw_u, m_u) = (total / q, d.hw / q, m / q);
    let (x_off_u, x_ns_u, g_off_u, g_ns_u) = (x.offset / q, x.n_stride / q, g.offset / q, g.n_stride / q);
    let (chunks, len) = plan(m_u, d.c, vec);
    let act = d.act.code();
    let half = d.dtype == DType::F16;
    let sg = use_subgroups(ctx);
    let inv_m = (1.0 / m as f64) as f32;

    if let Some((wg, regs)) = pick_fused(ctx, m_u, d.c, chunks, true) {
        let mut spec = Spec::new()
            .u32("ACT", act)
            .bool("VEC4", vec)
            .u32("REGS", regs)
            .u32("WG", wg)
            .f16(half)
            .subgroups(sg);
        if strided {
            spec = spec.bool("STRIDED", true);
        }
        let k = ctx.kernel(&FUSED_BWD, spec)?;
        let wgz = wg as usize;
        let mut p = Params::new().uz(d.c)?.uz(hw_u)?.uz(m_u)?.f(inv_m).at(5).uz(wgz / hw_u)?.uz(wgz % hw_u)?;
        if strided {
            p = p.uz(x_off_u)?.uz(x_ns_u)?.uz(g_off_u)?.uz(g_ns_u)?;
        }
        return ctx.dispatch(&k, &[x.buf, g.buf, mean, invstd, w, b, dx, dw, db], &p, [d.c as u32, 1, 1]);
    }

    let part = ctx.alloc(DType::F32.bytes(prod(&[d.c, chunks, 2])?))?;
    let coef = ctx.alloc(DType::F32.bytes(6 * d.c))?;

    let mut red_spec =
        Spec::new().u32("ACT", act).bool("VEC4", vec).bool("SUBGROUPS", sg).f16(half).subgroups(sg);
    if strided {
        red_spec = red_spec.bool("STRIDED", true);
    }
    let k = ctx.kernel(&BWD_REDUCE, red_spec)?;
    let mut p = d.walk_params(hw_u, m_u, chunks, len)?;
    if strided {
        p = p.uz(x_off_u)?.uz(x_ns_u)?.uz(g_off_u)?.uz(g_ns_u)?;
    }
    ctx.dispatch(&k, &[x.buf, g.buf, mean, invstd, w, b, part.raw()], &p, [d.c as u32, chunks as u32, 1])?;

    let k = ctx.kernel(&BWD_FINALIZE, Spec::new().u32("ACT", 0).subgroups(sg))?;
    let p = Params::new().uz(d.c)?.uz(chunks)?.f(inv_m);
    ctx.dispatch(&k, &[part.raw(), mean, invstd, w, b, dw, db, coef.raw()], &p, [d.c as u32, 1, 1])?;

    let grid3d = fits_grid3d(ctx, hw_u, d.c, d.n);
    let mut dx_spec = Spec::new().u32("ACT", act).bool("VEC4", vec).f16(half);
    if grid3d.is_some() {
        dx_spec = dx_spec.bool("GRID3D", true);
    }
    if strided {
        dx_spec = dx_spec.bool("STRIDED", true);
    }
    let k = ctx.kernel(&BWD_DX, dx_spec)?;
    let mut p = Params::new().uz(units)?.uz(d.c)?.uz(hw_u)?;
    if strided {
        p = p.uz(x_off_u)?.uz(x_ns_u)?.uz(g_off_u)?.uz(g_ns_u)?;
    }
    let grid = match grid3d {
        Some(g) => g,
        None => ctx.grid_1d(units as u64, 256)?,
    };
    ctx.dispatch(&k, &[x.buf, g.buf, coef.raw(), dx], &p, grid)
}

/// This module's kernels, for the precompile registry (`kernels::find_def`).
pub(crate) static DEFS: &[&KernelDef] =
    &[&STATS, &FINALIZE, &APPLY, &FUSED_FWD, &BWD_REDUCE, &BWD_FINALIZE, &BWD_DX, &FUSED_BWD];
