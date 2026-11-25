//! What a kernel list may ask a generator for (`crate::kernel_list`). A list is checksummed, but nothing stops a
//! valid one (from another build, or a bug) from naming constants no call site sets, and some generators' output
//! grows with a constant: `conv_split_finish` with `S = 4294967295` would emit billions of statements, an
//! allocation failure no `catch_unwind` stops. So every integer constant read from a list is bounded
//! ([`MAX_INT`], in `Spec::decode_fields`), and every kernel whose source or unrolled loops scale with a constant
//! is bounded to what its call sites set ([`precompilable`]) before a worker is given it. A kernel outside its
//! bound is still compiled lazily when a run asks for it; only the list skips it.
// Precompile and kernel lists are native only (no threads in the browser); in wasm32 builds most of this is unused.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports, unused_variables))]

use crate::pipeline::Spec;

/// Largest magnitude of an integer constant read from a kernel list. Call sites set much less: the largest are
/// spatial sizes (the weight gradient's `SP_IH` .. `SP_OW`, at most the image size, so `imgsz 4096` still fits
/// sixteen times over) and channel ratios.
pub(crate) const MAX_INT: u32 = 1 << 16;

/// Whether a kernel list may schedule kernel `name` with `spec`: for every generator whose output or unrolled
/// loops grow with a constant, that constant within what the kernel's call sites set (the generators' defaults
/// for absent ones). Every other kernel only has the [`MAX_INT`] bound of the list's decoder.
pub(crate) fn precompilable(name: &str, spec: &Spec) -> bool {
    use super::detect::{MAX_REG_MAX, MAX_TOPK};
    use super::elementwise::MAX_RANK;
    let u = |n: &str, default: u32| spec.get_u32(n, default) as usize;
    let rank = u("RANK", 0) <= MAX_RANK;
    match name {
        "conv2d_depthwise" => super::conv::depthwise_fits(spec),
        "conv2d_depthwise_grad" => super::conv::depthwise_grad_fits(spec),
        "conv_split_finish" => super::conv::split_fits(spec),
        "gemm" | "gemm_cat" => {
            let cfg = super::gemm::TileConfig {
                bm: spec.get_u32("BM", 128),
                bn: spec.get_u32("BN", 128),
                bk: spec.get_u32("BK", 16),
                tm: spec.get_u32("TM", 8),
                tn: spec.get_u32("TN", 8),
            };
            cfg.valid() && cat_fits(spec) && (1..=cfg.bk).contains(&spec.get_u32("WG_RUN", cfg.bk.min(16)))
        }
        "coop_gemm" | "coop_gemm_cat" => {
            let cfg = super::coop::CoopConfig {
                bm: spec.get_u32("BM", 128),
                bn: spec.get_u32("BN", 128),
                bk: spec.get_u32("BK", 32),
                wgm: spec.get_u32("WGM", 2),
                wgn: spec.get_u32("WGN", 4),
            };
            cfg.valid() && cat_fits(spec) && (1..=cfg.bk).contains(&spec.get_u32("WG_RUN", 8))
        }
        "ew1" | "ew2" | "ew3" => rank && (1..=4).contains(&u("KIND", 1)) && u("BCAST", 0) < 8,
        "binary" => rank && u("BCAST", 0) < 4,
        "permute" => rank,
        "move" => rank && eb_fits(spec) && u("MODE", 0) <= 2,
        "scatter" => rank && eb_fits(spec),
        "tal_topk" => {
            (1..=MAX_TOPK).contains(&u("K", 0)) && [1, 2, 4, 8, 16, 32, 64, 128].contains(&u("WG", 0))
        }
        _ if name.starts_with("tal_") || name.starts_with("det_") => {
            u("K", 0) <= MAX_TOPK && u("K2", 0) <= MAX_TOPK && u("R", 0) <= MAX_REG_MAX
        }
        "muon_gemm" => super::muon::gemm_fits(spec),
        // The multi-tensor kernels (`J` tensors) are only found through their resolvers, which bound `J`.
        _ => true,
    }
}

/// `CAT`: the row blocks of a concatenated `B` (`super::gemm_cat`).
fn cat_fits(spec: &Spec) -> bool {
    (1..=super::gemm_cat::MAX_CAT_PARTS).contains(&(spec.get_u32("CAT", 1) as usize))
}

/// `EB`: the element bytes of the strided copies.
fn eb_fits(spec: &Spec) -> bool {
    [1, 2, 4, 8].contains(&spec.get_u32("EB", 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_site_constants_fit_and_others_do_not() {
        let ok = [
            (
                "conv2d_depthwise",
                Spec::new().u32("KH", 3).u32("KW", 3).u32("SW", 2).u32("DH", 1).u32("DW", 1),
            ),
            (
                "conv2d_depthwise",
                Spec::new().u32("KH", 31).u32("KW", 31).u32("SW", 1).u32("DH", 1).u32("DW", 1),
            ),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 7).u32("KW", 7).u32("MODE", 1)),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 99).u32("KW", 99).u32("MODE", 0)),
            ("conv_split_finish", Spec::new().u32("S", 8)),
            ("gemm", Spec::new().u32("BM", 64).u32("BN", 128).u32("BK", 16).u32("TM", 4).u32("TN", 8)),
            (
                "gemm_cat",
                Spec::new().u32("BM", 64).u32("BN", 64).u32("BK", 16).u32("TM", 4).u32("TN", 4).u32("CAT", 4),
            ),
            ("coop_gemm", Spec::new().u32("BM", 16).u32("BN", 256).u32("BK", 32).u32("WGM", 1).u32("WGN", 8)),
            ("ew2", Spec::new().u32("KIND", 3).u32("RANK", 8)),
            ("move", Spec::new().u32("EB", 2).u32("MODE", 1).u32("RANK", 0)),
            ("tal_topk", Spec::new().u32("K", 64).u32("WG", 16)),
            ("det_box", Spec::new().u32("R", 16).u32("K", 10)),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 2)
                    .u32("BM", 64)
                    .u32("BN", 128)
                    .u32("TM", 4)
                    .u32("TN", 8)
                    .u32("DEPTH", 16),
            ),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 2)
                    .u32("BM", 64)
                    .u32("BN", 64)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("DEPTH", 16)
                    .bool("S4", true),
            ),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 0)
                    .u32("BM", 32)
                    .u32("BN", 32)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("DEPTH", 32)
                    .bool("DBUF", true),
            ),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 1)
                    .u32("BM", 32)
                    .u32("BN", 32)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("DEPTH", 32)
                    .bool("S4", true),
            ),
            ("unary", Spec::new().u32("OP", 40)),
        ];
        for (name, spec) in &ok {
            assert!(precompilable(name, spec), "{name} {}", spec.describe());
        }
        let bad = [
            ("conv_split_finish", Spec::new().u32("S", u32::MAX)),
            ("conv_split_finish", Spec::new().u32("S", 9)),
            ("conv2d_depthwise", Spec::new().u32("KH", 65536).u32("KW", 65536)),
            ("conv2d_depthwise", Spec::new().u32("KH", 3).u32("KW", 3).u32("SW", 65536)),
            ("conv2d_depthwise", Spec::new().u32("KH", 3).u32("KW", 0)),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 8).u32("KW", 8).u32("MODE", 1)),
            ("gemm", Spec::new().u32("BM", 65536)),
            ("gemm", Spec::new().u32("TM", 0)),
            (
                "gemm_cat",
                Spec::new().u32("BM", 64).u32("BN", 64).u32("BK", 16).u32("TM", 4).u32("TN", 4).u32("CAT", 5),
            ),
            (
                "gemm",
                Spec::new()
                    .u32("BM", 64)
                    .u32("BN", 64)
                    .u32("BK", 16)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("WG_RUN", 0),
            ),
            ("coop_gemm", Spec::new().u32("BK", 65536)),
            ("ew1", Spec::new().u32("RANK", 9)),
            ("move", Spec::new().u32("EB", 0)),
            ("scatter", Spec::new().u32("RANK", 65536)),
            ("tal_topk", Spec::new().u32("K", 65)),
            ("tal_targets", Spec::new().u32("K", 10).u32("K2", 65536)),
            ("det_box", Spec::new().u32("R", 33)),
            ("muon_gemm", Spec::new().u32("MODE", 3)),
            ("muon_gemm", Spec::new().u32("MODE", 0).u32("BM", 64).u32("BN", 128).u32("TM", 4).u32("TN", 8)),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 1)
                    .u32("BM", 64)
                    .u32("BN", 64)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("DEPTH", 16),
            ),
            ("muon_gemm", Spec::new().u32("MODE", 2).u32("BM", 32).u32("BN", 32).u32("TM", 8).u32("TN", 8)),
        ];
        for (name, spec) in &bad {
            assert!(!precompilable(name, spec), "{name} {}", spec.describe());
        }
    }
}
