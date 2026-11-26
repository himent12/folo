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
        "gemm" | "gemm_cat" | "gemm_epilogue" => {
            let cfg = super::gemm::TileConfig {
                bm: spec.get_u32("BM", 128),
                bn: spec.get_u32("BN", 128),
                bk: spec.get_u32("BK", 16),
                tm: spec.get_u32("TM", 8),
                tn: spec.get_u32("TN", 8),
            };
            cfg.valid()
                && cat_fits(spec)
                && spec.get_u32("RES_MODE", 0) <= 2
                && (1..=cfg.bk).contains(&spec.get_u32("WG_RUN", cfg.bk.min(16)))
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
        "move_vec4" => rank && u("MODE", 0) <= 1,
        "scatter" => rank && eb_fits(spec),
        "scatter_vec4" => rank,
        "strided_transpose" => {
            let eb = spec.get_u32("EB", 4);
            eb == 4 || eb == 8 || (eb == 2 && spec.is_f16())
        }
        "tal_topk" => {
            (1..=MAX_TOPK).contains(&u("K", 0)) && [1, 2, 4, 8, 16, 32, 64, 128].contains(&u("WG", 0))
        }
        "det_dfl_stats" => {
            use crate::pipeline::SpecVal;
            matches!(spec.get("R"), Some(SpecVal::U32(r)) if (1..=MAX_REG_MAX).contains(&(r as usize)))
                && matches!(spec.get("MOMENTS"), Some(SpecVal::Bool(_)))
                && matches!(spec.get("ENT"), Some(SpecVal::Bool(_)))
        }
        "detect" | "sites_decode" => {
            use crate::pipeline::SpecVal;
            matches!(spec.get("R"), Some(SpecVal::U32(r)) if (r as usize) <= MAX_REG_MAX)
        }
        _ if name.starts_with("tal_") || name.starts_with("det_") => {
            u("K", 0) <= MAX_TOPK && u("K2", 0) <= MAX_TOPK && u("R", 0) <= MAX_REG_MAX
        }
        "max_pool2d" => super::pooling::maxpool_fits(spec),
        "pool" | "pool_idx" | "pool_bw_idx" => super::pool_grad::pool_tiled_fits(name, spec),
        "bn_fused_fwd" | "bn_fused_bwd" => super::norm::fused_fits(spec),
        "arg_reduce_rows" => spec.get_u32("MODE", 0) <= 2,
        "reduce_var_rows" => super::reduce::var_rows_fits(spec),
        "sdpa_fwd" | "sdpa_bwd_dkdv" | "sdpa_bwd_dq" => super::attention::sdpa_fits(spec),
        "bias_add_silu" => super::fused_act::bias_add_silu_fits(spec),
        "bias_residual_act" => super::fused_act::bias_residual_act_fits(spec),
        "where_select_vec4" | "where_select_scalar" => super::select::where_select_fits(spec),
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
                Spec::new()
                    .u32("KH", 7)
                    .u32("KW", 7)
                    .u32("SH", 1)
                    .u32("SW", 1)
                    .u32("DH", 1)
                    .u32("DW", 1)
                    .bool("TILED", true)
                    .bool("VEC4_OUT", true),
            ),
            (
                "norm_fwd",
                Spec::new()
                    .bool("RMS", false)
                    .bool("HAS_W", true)
                    .bool("HAS_B", true)
                    .bool("VEC4", true)
                    .bool("CACHE", true)
                    .subgroups(true),
            ),
            (
                "norm_bwd_dx",
                Spec::new().bool("RMS", true).bool("HAS_W", true).bool("VEC4", true).bool("CACHE", true),
            ),
            ("norm_bwd_dwb", Spec::new().bool("RMS", false).bool("VEC4", true)),
            ("norm_bwd_param", Spec::new().bool("RMS", true).bool("VEC4", false).u32("MODE", 0)),
            ("softmax_bwd_rows", Spec::new().bool("LOG", false).bool("VEC4", true).bool("CACHE", true)),
            ("softmax_bwd_cols", Spec::new().bool("LOG", true).bool("VEC4", true).f16(true)),
            (
                "conv2d_depthwise",
                Spec::new().u32("KH", 31).u32("KW", 31).u32("SW", 1).u32("DH", 1).u32("DW", 1),
            ),
            (
                "conv2d_depthwise_grad",
                Spec::new().u32("KH", 7).u32("KW", 7).u32("MODE", 1).bool("VEC4", true).subgroups(true),
            ),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 7).u32("KW", 7).u32("MODE", 0).bool("VX4", true)),
            (
                "conv2d_depthwise_grad",
                Spec::new()
                    .u32("KH", 5)
                    .u32("KW", 5)
                    .u32("SH", 1)
                    .u32("SW", 1)
                    .u32("DH", 1)
                    .u32("DW", 1)
                    .u32("MODE", 0)
                    .bool("TILED", true)
                    .bool("VEC4_OUT", true),
            ),
            (
                "conv2d_depthwise_grad",
                Spec::new()
                    .u32("KH", 3)
                    .u32("KW", 3)
                    .u32("SH", 2)
                    .u32("SW", 2)
                    .u32("DH", 1)
                    .u32("DW", 1)
                    .u32("MODE", 0)
                    .bool("S2", true)
                    .bool("VEC2_OUT", true),
            ),
            ("bn_fused_fwd", Spec::new().u32("REGS", 4).bool("VEC4", true).subgroups(true)),
            ("bn_fused_bwd", Spec::new().u32("REGS", 8).bool("VEC4", true).subgroups(false)),
            (
                "arg_reduce_rows",
                Spec::new().bool("MAX", true).u32("MODE", 1).bool("VEC4", true).subgroups(true),
            ),
            ("reduce_var_rows", Spec::new().u32("MODE", 1).bool("VEC4", true).subgroups(true)),
            ("reduce_var_cols", Spec::new().bool("VEC4", true).bool("SMALL", true)),
            (
                "sdpa_fwd",
                Spec::new()
                    .bool("CAUSAL", true)
                    .bool("HAS_MASK", true)
                    .bool("WRITE_STATS", true)
                    .bool("CACHE_Q", true)
                    .u32("D_CHUNKS", 2)
                    .subgroups(true),
            ),
            ("sdpa_bwd_delta", Spec::new().subgroups(true)),
            (
                "sdpa_bwd_dkdv",
                Spec::new().bool("CAUSAL", true).bool("HAS_MASK", true).u32("D_CHUNKS", 4).subgroups(true),
            ),
            (
                "sdpa_bwd_dq",
                Spec::new()
                    .bool("CAUSAL", false)
                    .bool("HAS_MASK", true)
                    .bool("CACHE_Q", true)
                    .u32("D_CHUNKS", 2)
                    .subgroups(false),
            ),
            ("silu_bwd", Spec::new().bool("VEC4", true).f16(true)),
            ("gelu_bwd", Spec::new().bool("VEC4", true)),
            ("swiglu_fwd", Spec::new().bool("VEC4", true)),
            ("swiglu_bwd", Spec::new().bool("VEC4", true)),
            ("bias_add_silu", Spec::new().u32("MODE", 1).bool("HAS_BIAS", true)),
            (
                "bias_residual_act",
                Spec::new().u32("MODE", 0).u32("ACT", 2).u32("RES_MODE", 1).bool("HAS_BIAS", true),
            ),
            ("where_select_vec4", Spec::new().u32("MC", 0).u32("MA", 2).u32("MB", 3).bool("COND_U8", true)),
            ("where_select_scalar", Spec::new().u32("MC", 1).u32("MA", 0).u32("MB", 0)),
            (
                "rope2d_table",
                Spec::new().bool("INTERLEAVED", true).bool("BACKWARD", false).bool("VEC4", true),
            ),
            (
                "rope2d_trig",
                Spec::new().bool("INTERLEAVED", false).bool("BACKWARD", true).bool("VEC4", false),
            ),
            ("q8_matmul", Spec::new().u32("ACT", 8).bool("HAS_BIAS", true).bool("GROUPED", true)),
            (
                "q8_gemv",
                Spec::new().u32("ACT", 11).bool("HAS_BIAS", true).bool("GROUPED", false).subgroups(true),
            ),
            ("scatter_add_vec4", Spec::new().bool("IDX64", true)),
            (
                "gemm_epilogue",
                Spec::new()
                    .u32("BM", 64)
                    .u32("BN", 64)
                    .u32("BK", 16)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("RES_MODE", 2),
            ),
            (
                "res_norm_fwd",
                Spec::new()
                    .bool("RMS", false)
                    .bool("HAS_W", true)
                    .bool("HAS_B", true)
                    .bool("VEC4", true)
                    .bool("CACHE", true)
                    .subgroups(true),
            ),
            (
                "res_norm_bwd_dx",
                Spec::new()
                    .bool("RMS", true)
                    .bool("HAS_W", true)
                    .bool("HAS_GRES", true)
                    .bool("VEC4", true)
                    .bool("CACHE", true),
            ),
            (
                "res_norm_bwd_dwb",
                Spec::new().bool("RMS", false).bool("NEED_W", true).bool("NEED_B", true).bool("VEC4", true),
            ),
            (
                "group_norm_fwd",
                Spec::new().bool("HAS_W", true).bool("HAS_B", true).bool("VEC4", true).bool("CACHE", true),
            ),
            ("gather_vec4", Spec::new().bool("IDX64", true)),
            ("index_add_vec4", Spec::new().bool("IDX64", false)),
            ("max_pool2d", Spec::new().bool("S2_VEC4", true).f16(true)),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 99).u32("KW", 99).u32("MODE", 0)),
            ("upsample_2x_vec4", Spec::new().f16(true)),
            ("max_pool2d", Spec::new().bool("TILED", true).u32("KH", 5).u32("KW", 5).f16(true)),
            ("pool_idx", Spec::new().u32("MODE", 0).bool("TILED", true).u32("KH", 5).u32("KW", 5)),
            ("pool_bw_idx", Spec::new().u32("MODE", 1).bool("TILED", true).u32("KH", 3).u32("KW", 3)),
            ("pool_2x2_s2_vec4", Spec::new().f16(true)),
            ("avg_2x2_s2_bwd_vec4", Spec::new().f16(true)),
            ("adaptive_gap_bwd", Spec::new().bool("VEC4", true)),
            ("conv_split_finish", Spec::new().u32("S", 8)),
            ("gemm", Spec::new().u32("BM", 64).u32("BN", 128).u32("BK", 16).u32("TM", 4).u32("TN", 8)),
            (
                "gemm_cat",
                Spec::new().u32("BM", 64).u32("BN", 64).u32("BK", 16).u32("TM", 4).u32("TN", 4).u32("CAT", 4),
            ),
            ("coop_gemm", Spec::new().u32("BM", 16).u32("BN", 256).u32("BK", 32).u32("WGM", 1).u32("WGN", 8)),
            ("ew2", Spec::new().u32("KIND", 3).u32("RANK", 8)),
            ("move", Spec::new().u32("EB", 2).u32("MODE", 1).u32("RANK", 0)),
            ("move_vec4", Spec::new().u32("MODE", 1).u32("RANK", 0)),
            ("scatter_vec4", Spec::new().u32("RANK", 8)),
            ("strided_transpose", Spec::new().u32("EB", 4)),
            ("strided_transpose", Spec::new().u32("EB", 2).f16(true)),
            ("cast_f16_vec4", Spec::new().bool("TO_F16", true)),
            ("cast_u8_vec4", Spec::new().bool("TO_F16", true)),
            ("u8_to_float_scale", Spec::new().bool("VEC4", true).f16(true)),
            ("rgb8_hwc_to_chw_norm", Spec::new().bool("VEC4", true)),
            ("letterbox", Spec::new().bool("PREFILTER", false)),
            ("tal_topk", Spec::new().u32("K", 64).u32("WG", 16)),
            ("det_box", Spec::new().u32("R", 16).u32("K", 10)),
            ("det_dfl_stats", Spec::new().f16(true).u32("R", 16).bool("MOMENTS", true).bool("ENT", true)),
            ("detect", Spec::new().f16(true).u32("R", 16)),
            ("sites_decode", Spec::new().f16(false).u32("R", 16)),
            ("det_bce_fwd", Spec::new().subgroups(true)),
            ("det_combine", Spec::new().subgroups(true)),
            ("reduce_rows", Spec::new().u32("OP", 0).bool("VEC4", true).subgroups(true)),
            ("reduce_cols", Spec::new().u32("OP", 0).bool("SPLIT", true).bool("VEC4", true)),
            (
                "optim",
                Spec::new().u32("MODE", 2).bool("FLAG", true).bool("NESTEROV", false).bool("VEC4", true),
            ),
            ("unscale_check", Spec::new().bool("VEC4", true).subgroups(true)),
            ("muon_s8", Spec::new()),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 1)
                    .u32("BM", 32)
                    .u32("BN", 32)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("DEPTH", 32)
                    .bool("DIAG_BIAS", true),
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
                    .bool("S8", true)
                    .bool("INPLACE_SR", true),
            ),
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
            ("max_pool2d", Spec::new().bool("TILED", true).u32("KH", 8).u32("KW", 5)),
            ("max_pool2d", Spec::new().bool("TILED", true).u32("KH", 1).u32("KW", 3)),
            ("max_pool2d", Spec::new().bool("TILED", true).i32("KH", 5).u32("KW", 5)),
            ("pool", Spec::new().bool("TILED", true).u32("MODE", 0).u32("KH", 5).u32("KW", 5)),
            ("pool", Spec::new().bool("TILED", true).u32("MODE", 2).u32("KH", 5).u32("KW", 5)),
            ("pool_idx", Spec::new().u32("MODE", 0).bool("TILED", true).u32("KH", 8).u32("KW", 5)),
            ("pool_idx", Spec::new().u32("MODE", 1).bool("TILED", true).u32("KH", 5).u32("KW", 5)),
            ("pool_idx", Spec::new().u32("MODE", 0).bool("TILED", true).i32("KH", 5).u32("KW", 5)),
            ("pool_bw_idx", Spec::new().u32("MODE", 0).bool("TILED", true).u32("KH", 3).u32("KW", 3)),
            ("pool_bw_idx", Spec::new().u32("MODE", 2).bool("TILED", true).u32("KH", 3).u32("KW", 3)),
            ("conv2d_depthwise", Spec::new().u32("KH", 65536).u32("KW", 65536)),
            ("conv2d_depthwise", Spec::new().u32("KH", 3).u32("KW", 3).u32("SW", 65536)),
            ("conv2d_depthwise", Spec::new().u32("KH", 3).u32("KW", 0)),
            (
                "conv2d_depthwise",
                Spec::new().u32("KH", 8).u32("KW", 3).u32("SH", 1).u32("SW", 1).bool("TILED", true),
            ),
            (
                "conv2d_depthwise",
                Spec::new().u32("KH", 3).u32("KW", 3).u32("SH", 2).u32("SW", 1).bool("TILED", true),
            ),
            (
                "conv2d_depthwise",
                Spec::new().u32("KH", 1).u32("KW", 3).u32("SH", 1).u32("SW", 1).bool("TILED", true),
            ),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 8).u32("KW", 8).u32("MODE", 1)),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 8).u32("KW", 8).u32("MODE", 0).bool("VX4", true)),
            ("conv2d_depthwise_grad", Spec::new().u32("KH", 0).u32("KW", 3).u32("MODE", 0).bool("VX4", true)),
            (
                "conv2d_depthwise_grad",
                Spec::new()
                    .u32("KH", 8)
                    .u32("KW", 3)
                    .u32("SH", 1)
                    .u32("SW", 1)
                    .u32("MODE", 0)
                    .bool("TILED", true),
            ),
            (
                "conv2d_depthwise_grad",
                Spec::new()
                    .u32("KH", 3)
                    .u32("KW", 3)
                    .u32("SH", 1)
                    .u32("SW", 1)
                    .u32("MODE", 0)
                    .bool("S2", true),
            ),
            ("bn_fused_fwd", Spec::new().u32("REGS", 16)),
            ("bn_fused_bwd", Spec::new().u32("REGS", 0)),
            ("arg_reduce_rows", Spec::new().u32("MODE", 3)),
            ("reduce_var_rows", Spec::new().u32("MODE", 3)),
            ("sdpa_fwd", Spec::new().u32("D_CHUNKS", 0)),
            ("sdpa_fwd", Spec::new().u32("D_CHUNKS", 5)),
            ("sdpa_bwd_dkdv", Spec::new().u32("D_CHUNKS", 0)),
            ("sdpa_bwd_dkdv", Spec::new().u32("D_CHUNKS", 5)),
            ("sdpa_bwd_dq", Spec::new().u32("D_CHUNKS", 0)),
            ("sdpa_bwd_dq", Spec::new().u32("D_CHUNKS", 5)),
            ("bias_add_silu", Spec::new().u32("MODE", 3)),
            ("bias_residual_act", Spec::new().u32("MODE", 0).u32("ACT", 4)),
            ("bias_residual_act", Spec::new().u32("MODE", 0).u32("RES_MODE", 3)),
            ("where_select_vec4", Spec::new().u32("MC", 4)),
            ("where_select_scalar", Spec::new().u32("MB", 4)),
            (
                "gemm_epilogue",
                Spec::new()
                    .u32("BM", 64)
                    .u32("BN", 64)
                    .u32("BK", 16)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("RES_MODE", 3),
            ),
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
            ("move_vec4", Spec::new().u32("MODE", 2)),
            ("move_vec4", Spec::new().u32("RANK", 9)),
            ("scatter", Spec::new().u32("RANK", 65536)),
            ("scatter_vec4", Spec::new().u32("RANK", 9)),
            ("strided_transpose", Spec::new().u32("EB", 1)),
            ("strided_transpose", Spec::new().u32("EB", 2)),
            ("tal_topk", Spec::new().u32("K", 65)),
            ("tal_targets", Spec::new().u32("K", 10).u32("K2", 65536)),
            ("det_box", Spec::new().u32("R", 33)),
            ("det_dfl_stats", Spec::new()),
            ("det_dfl_stats", Spec::new().u32("R", 16)),
            ("det_dfl_stats", Spec::new().u32("R", 0).bool("MOMENTS", true).bool("ENT", true)),
            ("det_dfl_stats", Spec::new().u32("R", 33).bool("MOMENTS", true).bool("ENT", true)),
            ("detect", Spec::new()),
            ("detect", Spec::new().u32("R", 33)),
            ("sites_decode", Spec::new()),
            ("sites_decode", Spec::new().u32("R", 33)),
            ("muon_gemm", Spec::new().u32("MODE", 3)),
            (
                "muon_gemm",
                Spec::new()
                    .u32("MODE", 0)
                    .u32("BM", 32)
                    .u32("BN", 32)
                    .u32("TM", 4)
                    .u32("TN", 4)
                    .u32("DEPTH", 32)
                    .bool("S8", true),
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
                    .bool("S4", true)
                    .bool("S8", true),
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
                    .bool("DIAG_BIAS", true),
            ),
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
