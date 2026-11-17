//! BatchNorm statistics in the tensor-core forward conv's epilogue (`STATS`): a train-mode `ConvBnAct` runs its
//! conv with this flag and the BatchNorm skips its statistics pass (`bn_stats`, a full read of the conv output).
//!
//! The conv's GEMM is `C[n][co][p]`: rows are output channels, columns the pixels of one image. In the epilogue each
//! lane already holds 8 f32 results of one row of a 16×16 fragment (before any rounding to f16 storage); it folds
//! them into a running Welford partial `(count, mean, M2)` of its row (an exact two-pass mean and `M2` of the 8,
//! then Chan et al.'s merge). After the last fragment of a row block the lanes' partials go through the scratch
//! array `Cs`, and one invocation per row merges the `2 · WGN` partials of its row. The partial of channel `co`,
//! image `n` and column tile `t` lands at `PART[(co · chunks + n · tiles + t) · 3]`, the layout `norm`'s
//! `bn_finalize` merges, with `chunks = N · tiles`.
//!
//! Only tensor cores (mixed-precision training) have it: on the regular f32 kernels the epilogue cost two thirds of
//! the pass it replaced and the conv + BatchNorm pair came out even (`docs/notes/hill/r5-gputrain.md`).

use std::fmt::Write as _;

use crate::DType;
use crate::context::GpuContext;
use crate::error::Result;
use crate::pool::GpuBuffer;

/// The statistics a conv wrote for the BatchNorm after it: `chunks` Welford partials per channel in `buf`.
pub struct BnPartials {
    /// The partials, `[C, chunks, 3]` f32 as `(count, mean, M2)`.
    pub buf: GpuBuffer,
    /// Partials per channel.
    pub chunks: usize,
}

/// Where a conv kernel puts its statistics: it allocates the partials once its tile config (and so its column
/// tiles) is known. `None`: no statistics wanted.
pub(crate) type StatsSlot<'a> = Option<&'a mut Option<BnPartials>>;

/// Allocates the partials of a conv with `c_out` channels, `batch` images and `tiles` column tiles per image.
pub(crate) fn alloc(ctx: &GpuContext, c_out: usize, batch: usize, tiles: usize) -> Result<BnPartials> {
    let chunks = batch * tiles;
    let buf = ctx.alloc(DType::F32.bytes(c_out * chunks * 3))?;
    Ok(BnPartials { buf, chunks })
}

/// The statistics of one fragment (WGSL, inside the fragment's block after `v0`, `v1` are read): lane (`er`, `ec`)
/// folds its 8 values (columns `col ..`, masked past `N` when `check`) into its running partial `(sn, sm, sq)`.
pub(crate) fn coop_fragment(check: bool) -> String {
    let mut s = String::from("    {\n");
    for (k, base) in [("k0", 0), ("k1", 4)] {
        if check {
            let terms: Vec<String> =
                (0..4).map(|c| format!("select(0.0, 1.0, col + {}u < N)", base + c)).collect();
            let _ = writeln!(s, "      let {k} = vec4<f32>({});", terms.join(", "));
        } else {
            let _ = writeln!(s, "      let {k} = vec4<f32>(1.0);");
        }
    }
    s.push_str(
        "      let nb = dot(k0, vec4<f32>(1.0)) + dot(k1, vec4<f32>(1.0));\n\
         \x20     let mb = select(0.0, (dot(k0, v0) + dot(k1, v1)) / max(nb, 1.0), nb > 0.0);\n\
         \x20     let d0 = (v0 - vec4<f32>(mb)) * k0; let d1 = (v1 - vec4<f32>(mb)) * k1;\n\
         \x20     let qb = dot(d0, d0) + dot(d1, d1);\n",
    );
    s.push_str(&merge(("sn", "sm", "sq")));
    s.push_str("    }\n");
    s
}

/// After the fragments of row block `i` (WGSL): the lanes' partials go through `Cs` (free between fragments; 3
/// floats per invocation, `(subgroup · 32 + lane) · 3`, so lane `2 · er + ec / 8` of subgroup `wm · WGN + wn`), and
/// invocation `wm · 16 + er` merges the `2 · WGN` partials of its row and writes them. Params: 34 = chunks per row,
/// 35 = column tiles per image.
pub(crate) fn coop_merge(wgm: u32, wgn: u32, fm: u32, i: u32) -> String {
    let mut s = String::from(
        "  { let so = (sg * 32u + lane) * 3u; Cs[so] = sn; Cs[so + 1u] = sm; Cs[so + 2u] = sq; }\n\
         \x20 sn = 0.0; sm = 0.0; sq = 0.0;\n  workgroupBarrier();\n",
    );
    let _ = writeln!(s, "  if (tid < {}u) {{", wgm * 16);
    s.push_str(
        "    let sr = tid % 16u; let swm = tid / 16u;\n    var cn = 0.0; var cm = 0.0; var cq = 0.0;\n",
    );
    let _ = writeln!(s, "    for (var se = 0u; se < {}u; se += 1u) {{", wgn * 2);
    let _ = writeln!(
        s,
        "      let so = ((swm * {wgn}u + se / 2u) * 32u + sr * 2u + se % 2u) * 3u;\n\
         \x20     let nb = Cs[so]; let mb = Cs[so + 1u]; let qb = Cs[so + 2u];"
    );
    s.push_str(&merge(("cn", "cm", "cq")));
    s.push_str("    }\n");
    let _ = writeln!(s, "    let srow = m0 + swm * {}u + {}u + sr;", fm * 16, i * 16);
    s.push_str(
        "    if (srow < M) {\n\
         \x20     let o = (srow * pu(34u) + wid.z * pu(35u) + wid.x) * 3u;\n\
         \x20     PART[o] = cn; PART[o + 1u] = cm; PART[o + 2u] = cq;\n    }\n  }\n  workgroupBarrier();\n",
    );
    s
}

/// One Chan merge of the partial `(nb, mb, qb)` into the running `(n, m, q)` (WGSL statements).
fn merge((n, m, q): (&str, &str, &str)) -> String {
    format!(
        "      if ({n} == 0.0) {{ {n} = nb; {m} = mb; {q} = qb; }} else if (nb > 0.0) {{\n\
         \x20       let tot = {n} + nb; let dd = mb - {m};\n\
         \x20       {m} += dd * (nb / tot); {q} += qb + dd * dd * ({n} * nb / tot); {n} = tot;\n\
         \x20     }}\n"
    )
}
