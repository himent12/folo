//! Split-K for the forward implicit-GEMM conv: layers with few output tiles and a long `K` (folo's 3×3 convs at
//! 40 and 20 px: 64–256 workgroups for 80 SMs) split `K` into chunks, one workgroup layer each, every chunk
//! writing an f32 partial slab of the whole output; [`SPLIT_FINISH`] then sums the slabs and applies the bias
//! and activation.

use std::fmt::Write as _;

use crate::context::GpuContext;
use crate::error::Result;
use crate::kernels::UNARY_FN;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
use crate::{DType, UnaryOp};

/// Workgroups a split conv aims for: six per SM of an 80-SM GPU, so the last partial wave of 2–3 resident
/// workgroups per SM costs little (measured on an RTX 4080 SUPER: folo-s's 256-workgroup stride-2 conv at 40 px
/// runs 13% faster as three chunks, 10% as two).
const TARGET_GROUPS: usize = 480;
/// A conv with at least this many workgroups is not split.
const MIN_GROUPS: usize = 320;
/// `BK`-deep tiles of `K` per chunk the automatic count allows (`splits <= k_tiles / MIN_CHUNK_TILES`): a cap on
/// the count, not a minimum for every chunk, since the last chunk ends short (`K = 441` at `BK = 16`: 28 tiles,
/// 3 chunks of 160, the last 121 long).
const MIN_CHUNK_TILES: usize = 9;
/// Most chunks.
const MAX_SPLITS: usize = 8;
/// Upper bound for the partial slabs, in f32 elements (64 MiB).
const MAX_PARTIAL_ELEMS: usize = 16 << 20;

/// `(splits, chunk)`: how many `K` chunks a forward conv with `tiles` output tiles per (image, group), `batch`
/// (image, group) pairs and `out_elems` outputs runs as, and the chunk length (a multiple of `cfg.bk`, so only
/// the last chunk can end short; every chunk is non-empty) for a tile depth of `bk`. `want` forces the number of chunks (tests and
/// tuning), clamped to what `K` and the dispatch limits allow; `None` picks it from the shape.
pub(crate) fn plan(
    bk: usize,
    k: usize,
    tiles: usize,
    batch: usize,
    out_elems: usize,
    want: Option<usize>,
) -> (usize, usize) {
    let k_tiles = k.div_ceil(bk);
    let groups = tiles.saturating_mul(batch);
    let cap = MAX_SPLITS.min(k_tiles).min(MAX_PARTIAL_ELEMS / out_elems.max(1)).min(65535 / batch.max(1));
    let splits = match want {
        Some(s) => s.min(cap),
        None if groups >= MIN_GROUPS => 1,
        None => TARGET_GROUPS.div_ceil(groups.max(1)).min(k_tiles / MIN_CHUNK_TILES).min(cap),
    }
    .max(1);
    if splits == 1 {
        return (1, k);
    }
    let chunk = k_tiles.div_ceil(splits) * bk;
    (k.div_ceil(chunk), chunk)
}

/// `Y[i] = act(Σ_s PART[s · slab + i] + bias[channel of i])`, the `S` slabs unrolled. Spec: `S`, `VEC` (four
/// outputs per invocation; the plane is a multiple of 4, so they share a channel), `HAS_BIAS`, `ACT`.
/// Params: 0 = units (vec4 or scalar), 1 = slab stride in units, 2 = plane (`OH·OW`), 3 = channels,
/// 4 = activation alpha.
fn gen_split_finish(spec: &Spec) -> String {
    let splits = spec.get_u32("S", 2);
    let vec = spec.get_bool("VEC");
    let (pt, yt, wide, out) = if vec {
        ("vec4<f32>", "vec4<T>", 4, "Y[i] = vec4<T>(unary4(ACT, v, pf(4u)));")
    } else {
        ("f32", "T", 1, "Y[i] = T(unary1(ACT, v, pf(4u)));")
    };
    let mut s = String::from(UNARY_FN);
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> PART: array<{pt}>;");
    s.push_str("@group(0) @binding(2) var<storage, read> BIAS: array<T>;\n");
    let _ = writeln!(s, "@group(0) @binding(3) var<storage, read_write> Y: array<{yt}>;");
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,\n\
         \x20       @builtin(local_invocation_index) lid: u32) {\n\
         \x20 let i = linear_id(wid, nwg, lid, 256u);\n\
         \x20 if (i >= pu(0u)) { return; }\n\
         \x20 let slab = pu(1u);\n\
         \x20 var v = PART[i];\n",
    );
    for k in 1..splits {
        let _ = writeln!(s, "  v += PART[i + {k}u * slab];");
    }
    let _ = writeln!(s, "  if (HAS_BIAS) {{ v += {pt}(f32(BIAS[((i * {wide}u) / pu(2u)) % pu(3u)])); }}");
    let _ = writeln!(s, "  {out}\n}}");
    s
}

/// Whether a kernel list may ask for [`SPLIT_FINISH`] with `spec` (`kernels::bounds`): the slabs are unrolled, and
/// [`plan`] makes at most [`MAX_SPLITS`].
pub(super) fn finish_fits(spec: &Spec) -> bool {
    (1..=MAX_SPLITS).contains(&(spec.get_u32("S", 2) as usize))
}

pub(super) static SPLIT_FINISH: KernelDef = KernelDef {
    name: "conv_split_finish",
    storage: &[Access::Read, Access::Read, Access::ReadWrite],
    source: Source::Generated(gen_split_finish),
};

/// The output side of a split conv: `y` (`out_elems` values stored as `out`, NCHW with `channels` planes of
/// `plane` pixels) is `act(Σ partial slabs + bias)`.
pub(crate) struct Finish {
    pub splits: usize,
    pub out_elems: usize,
    pub plane: usize,
    pub channels: usize,
    pub act: UnaryOp,
    pub out: DType,
}

/// Encodes [`SPLIT_FINISH`]: sums the `f.splits` f32 slabs of `part` (each `f.out_elems` long) into `y` with the
/// bias (stored as `f.out`, `None` for none) and the activation.
pub(crate) fn finish(
    ctx: &GpuContext,
    f: &Finish,
    part: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    y: &wgpu::Buffer,
) -> Result<()> {
    let vec = f.plane.is_multiple_of(4);
    let units = if vec { f.out_elems / 4 } else { f.out_elems };
    let spec = Spec::new()
        .u32("S", f.splits as u32)
        .bool("VEC", vec)
        .bool("HAS_BIAS", bias.is_some())
        .u32("ACT", f.act.code())
        .f16(f.out == DType::F16);
    let kernel = ctx.kernel(&SPLIT_FINISH, spec)?;
    let p = Params::new().uz(units)?.uz(units)?.uz(f.plane)?.uz(f.channels)?.f(f.act.alpha());
    ctx.dispatch(&kernel, &[part, bias.unwrap_or(ctx.dummy()), y], &p, ctx.grid_1d(units as u64, 256)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::gemm::TileConfig;

    #[test]
    fn plan_covers_k_with_non_empty_chunks() {
        for cfg in [TileConfig::LARGE, TileConfig::MEDIUM, TileConfig::ROWS16] {
            for k in [1, 7, 27, 45, 288, 576, 1152, 4608] {
                for want in [None, Some(1), Some(2), Some(3), Some(8), Some(100)] {
                    let (s, chunk) = plan(cfg.bk as usize, k, 4, 16, 16 * 64 * 400, want);
                    assert!((1..=MAX_SPLITS).contains(&s), "{cfg:?} k {k} {want:?}: {s} chunks");
                    if s == 1 {
                        assert_eq!(chunk, k);
                        continue;
                    }
                    assert!(chunk.is_multiple_of(cfg.bk as usize));
                    assert!((s - 1) * chunk < k && s * chunk >= k, "{cfg:?} k {k} {want:?}: {s} x {chunk}");
                }
            }
        }
        // folo-n's 3x3 64 -> 64 at 20 px, batch 16: 64 tiles of 64x128, K = 576 (36 tiles of 16).
        assert_eq!(plan(16, 576, 4, 16, 16 * 64 * 400, None).0, 4);
        // folo-s's 256 -> 512 stride-2 conv at 40 px: 256 tiles of 128x128, K = 2304.
        assert_eq!(plan(8, 2304, 16, 16, 16 * 512 * 400, None).0, 2);
        // Enough tiles: no split. Batch limits: at most 65535 workgroup layers.
        assert_eq!(plan(16, 576, 50, 16, 16 * 64 * 6400, None).0, 1);
        assert_eq!(plan(16, 4608, 1, 30000, 30000 * 64, Some(8)).0, 2);
    }
}
