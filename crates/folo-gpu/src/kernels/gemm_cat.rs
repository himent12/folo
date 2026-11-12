//! GEMMs whose `B` is a concatenation of row blocks in separate buffers, read in place (`CAT` in the GEMM and
//! tensor-core generators). A 1×1 conv over the channel concatenation of several NCHW tensors is such a GEMM:
//! `B` is `ΣC_i × H·W` per image and part `i` is rows `C_0 + .. + C_{i-1} ..` of it, so the concatenation is never
//! written (docs/GPU.md, Concatenations read in place).
//!
//! Part 0 binds where `B` always binds (`@binding(2)`); parts 1.. bind at `@binding(5..)`. Uniform words
//! [`CAT_WORD`]`..` hold the table: the cumulative row ends (4 words), each part's element offset (4 words) and
//! its stride per outer batch index (4 words). The generated `cat_b(k, i, b1)` returns element (or `vec4` word)
//! `i` of row `k` for outer batch `b1`; the branch on `k` is uniform across a row of the tile.

use std::fmt::Write as _;

use super::check_buf;
use super::gemm::{
    BatchStrides, GEMM_CAT, GemmBias, GemmDesc, TileConfig, base_spec, check_config, check_dtype, extent,
    gemm_params,
};
use crate::DType;
use crate::context::GpuContext;
use crate::error::{GpuError, Result, shape_err};
use crate::kernels::coop;

/// Most parts one GEMM reads in place.
pub const MAX_CAT_PARTS: usize = 4;
/// Storage bindings of the `CAT` kernels (`A`, part 0, bias, `C`, parts 1..): a device must allow this many.
pub const CAT_STORAGE_BUFFERS: u32 = 4 + MAX_CAT_PARTS as u32 - 1;
/// First uniform word of the part table.
pub(crate) const CAT_WORD: usize = 40;

/// One row block of a concatenated `B` ([`gemm_cat`]): `rows` rows at element `offset` of `buf` for outer batch 0,
/// `outer` elements further per outer batch index; rows are the GEMM's `ldb` apart.
#[derive(Clone, Copy, Debug)]
pub struct CatPart<'a> {
    /// The buffer holding the rows.
    pub buf: &'a wgpu::Buffer,
    /// Rows of `B` in this part.
    pub rows: usize,
    /// Element offset of the first row of outer batch 0.
    pub offset: usize,
    /// Elements between outer batches.
    pub outer: usize,
}

/// The WGSL of the extra part bindings (`{name}1..`) and `cat_b`, for a generator whose `B` binding is `name`
/// with element type `ty` (`vec` when `ty` is a `vec4`).
pub(crate) fn cat_source(parts: u32, name: &str, ty: &str, vec: bool) -> String {
    let mut s = String::new();
    for p in 1..parts {
        let _ = writeln!(s, "@group(0) @binding({}) var<storage, read> {name}{p}: array<{ty}>;", 4 + p);
    }
    let w = if vec { " / 4u" } else { "" };
    // One return at the end: an early return under a branch would make the caller's control flow non-uniform to
    // naga, and the tensor-core kernel's cooperative operations need it uniform.
    let _ =
        writeln!(s, "fn cat_b(k: u32, i: u32, b1: u32) -> {ty} {{\n  let ld = pu(4u){w};\n  var v: {ty};");
    let word = CAT_WORD as u32;
    for p in 0..parts {
        let arr = if p == 0 { name.to_string() } else { format!("{name}{p}") };
        let row = if p == 0 { "k".to_string() } else { format!("(k - pu({}u))", word + p - 1) };
        let at = format!("{arr}[(pu({}u) + b1 * pu({}u)){w} + {row} * ld + i]", word + 4 + p, word + 8 + p);
        let head = match p {
            0 => format!("  if (k < pu({word}u)) {{"),
            _ if p + 1 < parts => format!(" else if (k < pu({}u)) {{", word + p),
            _ => " else {".to_string(),
        };
        let _ = write!(s, "{head} v = {at}; }}");
    }
    s.push_str("\n  return v;\n}\n");
    s
}

/// Checks the parts against `d` and returns the uniform part table (row ends, offsets, outer strides) and
/// whether every part allows `vec4` loads.
pub(crate) fn part_table(d: &GemmDesc, parts: &[CatPart<'_>]) -> Result<([u32; 12], bool)> {
    if parts.len() < 2 || parts.len() > MAX_CAT_PARTS {
        return Err(shape_err(format!("gemm_cat: {} parts (2 to {MAX_CAT_PARTS})", parts.len())));
    }
    if d.trans_b || d.batch.1 != 1 {
        return Err(shape_err("gemm_cat: B must not be transposed and the inner batch must be 1"));
    }
    let rows: usize = parts.iter().map(|p| p.rows).sum();
    if rows != d.k {
        return Err(shape_err(format!("gemm_cat: parts hold {rows} rows of B, K is {}", d.k)));
    }
    if parts.iter().any(|p| p.rows == 0) {
        return Err(shape_err("gemm_cat: a part holds no rows (leave it out)"));
    }
    if d.ldb < d.n {
        return Err(shape_err(format!("gemm_cat: ldb {} < N {}", d.ldb, d.n)));
    }
    let u =
        |v: usize| u32::try_from(v).map_err(|_| GpuError::Limit(format!("{v} does not fit a u32 parameter")));
    let mut table = [0u32; 12];
    let mut end = 0;
    let mut vec = d.ldb.is_multiple_of(4) && d.n.is_multiple_of(4);
    for (i, p) in parts.iter().enumerate() {
        let at = BatchStrides { offset: p.offset, outer: p.outer, inner: 0 };
        check_buf(p.buf, extent(&at, d.batch, p.rows, d.n, d.ldb)?, d.dtype, "gemm_cat B part")?;
        end += p.rows;
        table[i] = u(end)?;
        table[4 + i] = u(p.offset)?;
        table[8 + i] = u(p.outer)?;
        vec &= p.offset.is_multiple_of(4) && p.outer.is_multiple_of(4);
    }
    Ok((table, vec))
}

/// `C = alpha · A · cat(parts) (+ bias) → act` for the GEMM `d` (its `b` strides are ignored; `B` is the
/// concatenation of `parts` along the rows), on tensor cores for f16 when the context has them on.
///
/// Every part holds at least one row. `c` is written while the others are read in the same dispatch, so it
/// must not be `a`, the bias or any part's buffer. The kernel binds [`CAT_STORAGE_BUFFERS`] storage buffers;
/// on a device that allows fewer this is [`GpuError::Unsupported`].
pub fn gemm_cat(
    ctx: &GpuContext,
    d: &GemmDesc,
    a: &wgpu::Buffer,
    parts: &[CatPart<'_>],
    bias: Option<&wgpu::Buffer>,
    c: &wgpu::Buffer,
    cfg: Option<TileConfig>,
) -> Result<()> {
    check_dtype(ctx, d.dtype, "gemm_cat")?;
    if ctx.caps().max_storage_buffers < CAT_STORAGE_BUFFERS {
        return Err(GpuError::Unsupported("gemm_cat: 7 storage buffers per shader stage"));
    }
    if d.m == 0 || d.n == 0 || d.batch.0 == 0 {
        return Ok(());
    }
    let (table, vec_b) = part_table(d, parts)?;
    if d.trans_a || d.lda < d.k || d.ldc < d.n {
        return Err(shape_err(format!("gemm_cat: unsupported A or C layout in {d:?}")));
    }
    if parts.iter().map(|p| p.buf).chain([a]).chain(bias).any(|b| b == c) {
        return Err(shape_err("gemm_cat: C must not share a buffer with A, the bias or a part of B"));
    }
    check_buf(a, extent(&d.a, d.batch, d.m, d.k, d.lda)?, d.dtype, "gemm_cat A")?;
    check_buf(c, extent(&d.c, d.batch, d.m, d.n, d.ldc)?, d.dtype, "gemm_cat C")?;
    let bias_buf = match (d.bias, bias) {
        (GemmBias::None, _) => ctx.dummy(),
        (GemmBias::Row, Some(b)) => {
            // The shaders read `BIAS[offBias + b1·outer + b2·inner + row]`.
            check_buf(b, extent(&d.bias_strides, d.batch, 1, d.m, d.m)?, d.dtype, "gemm_cat bias")?;
            b
        }
        _ => return Err(shape_err("gemm_cat: only a row bias is supported")),
    };
    let bufs: Vec<&wgpu::Buffer> =
        (1..MAX_CAT_PARTS).map(|i| parts.get(i).map_or(ctx.dummy(), |p| p.buf)).collect();
    let mut p = gemm_params(d)?.at(CAT_WORD);
    for v in table {
        p = p.u(v);
    }
    if cfg.is_none()
        && d.dtype == DType::F16
        && ctx.tensor_cores()
        && coop::encode_gemm_cat(ctx, d, a, parts[0].buf, bias_buf, c, &bufs, &p, parts.len() as u32, vec_b)?
    {
        return Ok(());
    }
    let cfg = cfg.unwrap_or_else(|| super::gemm::pick_config_conv(ctx, d));
    check_config(ctx, cfg)?;
    let vec_a = d.lda.is_multiple_of(4)
        && d.a.offset.is_multiple_of(4)
        && d.a.outer.is_multiple_of(4)
        && d.k.is_multiple_of(4);
    let vec_c = d.ldc.is_multiple_of(4)
        && d.c.offset.is_multiple_of(4)
        && d.c.outer.is_multiple_of(4)
        && d.n.is_multiple_of(4);
    let check = !d.m.is_multiple_of(cfg.bm as usize)
        || !d.n.is_multiple_of(cfg.bn as usize)
        || !d.k.is_multiple_of(cfg.bk as usize);
    let spec = base_spec(d, cfg)
        .bool("VEC_A", vec_a)
        .bool("VEC_B", vec_b)
        .bool("VEC_C", vec_c)
        .bool("TRANS_A", false)
        .bool("TRANS_B", false)
        .bool("CHECK", check)
        .u32("CONV_MODE", 0)
        .u32("CAT", parts.len() as u32);
    let k = ctx.kernel(&GEMM_CAT, spec)?;
    let groups = [
        d.n.div_ceil(cfg.bn as usize) as u32,
        d.m.div_ceil(cfg.bm as usize) as u32,
        u32::try_from(d.batch.0).map_err(|_| GpuError::Limit("gemm_cat batch too large".into()))?,
    ];
    let mut all = vec![a, parts[0].buf, bias_buf, c];
    all.extend(bufs);
    ctx.dispatch(&k, &all, &p, groups)
}
