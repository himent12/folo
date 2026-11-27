//! Analytic GPU training and validation memory footprint estimator for [`crate::model::FoloModel`] and
//! [`crate::model::v2::FoloV2Model`].
//!
//! Walks the exact stage widths, repeats, and head branches of `FoloModel::with_rng` / `FoloV2Model::with_rng`
//! without allocating tensors, accounting for `f32`/`amp` dtypes (including `F32` autocast attention softmax),
//! `o2m`/`o2o` heads, EMA, optimizer states (`SGD`, `MuSGD` momentum + Muon conv buffers, `AdamW`), frozen
//! distillation teachers (weights, retained outputs, and eval/forward-plan intermediates), and `folo-gpu`
//! convolution backward weight-gradient split-K scratch buffers.

use folo_nn::io::ModelConfig;
use folo_nn::optim::OptimKind;
use serde::{Deserialize, Serialize};

use super::{AutoBatchRequest, parse_arch};
use crate::error::Result;
use crate::model::Scaling;
use crate::model::v2::{FoloV2Config, ScalingV2};

/// Pool size-class rounding overhead factor (power-of-2 / 25% buckets in `folo-gpu::pool::size_class`).
const POOL_OVERHEAD_NUM: u64 = 5;
const POOL_OVERHEAD_DEN: u64 = 4;
/// Fixed baseline device workspace (cached anchor tables, uniform arenas, baseline kernel scratch).
pub(super) const FIXED_WORKSPACE_BYTES: u64 = 16 * 1024 * 1024;
/// Upper bound on split-K partial weight-gradient slabs in `folo-gpu` (`16 Mi` `f32` elems = 64 MiB).
const MAX_PARTIAL_ELEMS: usize = 16 << 20;
/// Largest `KH * KW` handled by `folo-gpu`'s direct depthwise weight-gradient kernel.
const MAX_DEPTHWISE_TAPS: usize = 49;

/// Analytic GPU training memory footprint for a [`crate::model::FoloModel`] configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFootprint {
    /// Exact trainable parameter count of the student [`crate::model::FoloModel`].
    pub model_params: usize,
    /// Fixed resident bytes (weights, BN buffers, grad accumulator, optimizer state, optional EMA & teacher
    /// weights, plus pool overhead and fixed workspace).
    pub fixed_bytes: u64,
    /// Estimated per-sample activation, autograd backward, loss/assigner, and optional teacher working bytes
    /// (including 25% `folo-gpu` pool size-class rounding overhead).
    pub per_sample_bytes: u64,
    /// Largest single per-sample GPU storage tensor in bytes (linear in batch size), checked alongside
    /// convolution weight-gradient scratch against `min(max_buffer_size, max_storage_buffer_binding_size)`.
    pub max_single_tensor_per_sample: u64,
}

/// Internal footprint breakdown used by [`super::auto_batch_plan`] for both training (including batch-dependent
/// conv backward scratch and fixed weight buffers) and epoch-end `F32` validation batch sizing.
pub(super) struct FootprintDetails {
    pub fp: MemoryFootprint,
    pub max_fixed_tensor_bytes: u64,
    pub eval_fixed_bytes: u64,
    pub eval_per_sample_bytes: u64,
    pub eval_max_single_tensor: u64,
    pub convs: Vec<ConvWgradShape>,
}

/// Minimal conv geometry needed to bound `folo-gpu`'s `conv2d_backward_weight_as` split-K scratch buffer.
#[derive(Clone, Copy, Debug)]
pub(super) struct ConvWgradShape {
    cin: usize,
    cout: usize,
    k: usize,
    groups: usize,
    oh: usize,
    ow: usize,
}

/// Analytically estimate the GPU memory footprint (exact parameter count, fixed resident bytes, per-sample
/// activation/backward bytes, and largest single per-sample storage tensor) for a [`crate::model::FoloModel`]
/// training run.
///
/// Allocates no tensors and completes in microseconds, matching `FoloModel::with_rng`'s exact stage widths,
/// repeats, and head branches.
pub fn estimate_training_footprint(req: &AutoBatchRequest) -> Result<MemoryFootprint> {
    Ok(estimate_footprint_details(req)?.fp)
}

pub(super) fn estimate_footprint_details(req: &AutoBatchRequest) -> Result<FootprintDetails> {
    req.validate()?;
    let (is_v2, scale) = parse_arch(&req.arch)?;
    let mut mconf = if is_v2 {
        ModelConfig::new_v2(scale, req.num_classes)?
    } else {
        ModelConfig::new(scale, req.num_classes)?
    };
    mconf.head = req.head.as_str().into();
    let student = analyze_model(&mconf, req.imgsz as usize, req.amp, true)?;

    let params_bytes = student.params.saturating_mul(4);
    let buffers_bytes = student.bn_buffers.saturating_mul(4);
    // `Sgd`: 1× F32 momentum buffer per param.
    // `MuSgd`: embeds `Sgd` (1× F32 momentum for all params) + 1× F32 Muon buffer for every 2-D/4-D conv/linear weight.
    // `AdamW`: 2× F32 moment buffers (m, v) per param.
    let opt_state_bytes = match req.optimizer {
        OptimKind::Sgd => params_bytes,
        OptimKind::MuSgd => student.params.saturating_add(student.conv_weight_params).saturating_mul(4),
        OptimKind::AdamW => params_bytes.saturating_mul(2),
    };
    // Live weights + BN buffers + grad accumulator + optimizer state + optional EMA shadow (weights + BN buffers).
    let mut fixed =
        params_bytes.saturating_mul(2).saturating_add(opt_state_bytes).saturating_add(buffers_bytes);
    if req.ema {
        fixed = fixed.saturating_add(params_bytes).saturating_add(buffers_bytes);
    }

    let mut per_sample = student.per_sample_bytes;
    let mut max_single = student.max_single_tensor;
    let mut max_fixed_tensor = student.max_weight_elems.saturating_mul(4);

    if req.teacher_arch.is_some() || req.teacher_params.is_some() {
        let teacher_elem: u64 = if req.amp { 2 } else { 4 };
        let t_stats = match &req.teacher_arch {
            Some(t_arch) => {
                let (t_v2, ts) = parse_arch(t_arch)?;
                let mut tconf = if t_v2 {
                    ModelConfig::new_v2(ts, req.num_classes)?
                } else {
                    ModelConfig::new(ts, req.num_classes)?
                };
                tconf.head = req.teacher_head.as_str().into();
                Some(analyze_model(&tconf, req.imgsz as usize, req.amp, false)?)
            }
            None => None,
        };
        let teacher_params = match (req.teacher_params, &t_stats) {
            (Some(p), _) => p as u64,
            (None, Some(ts)) => ts.params,
            (None, None) => 0,
        };
        // Fused frozen teacher weights (`F16` under AMP, `F32` otherwise), including both branches if `o2o`.
        fixed = fixed.saturating_add(teacher_params.saturating_mul(teacher_elem));
        // Retained `TeacherBatch` raw outputs (`box_raw` + `cls_raw`) stay live across student forward+backward.
        let anchors = total_anchors(req.imgsz as usize);
        let t_box =
            (4u64.saturating_mul(mconf.reg_max as u64)).saturating_mul(anchors).saturating_mul(teacher_elem);
        let t_cls = (req.num_classes as u64).saturating_mul(anchors).saturating_mul(teacher_elem);
        per_sample = per_sample.saturating_add(t_box).saturating_add(t_cls);
        if let Some(ts) = &t_stats {
            // `Teacher::load` enables `ForwardPlans`, pinning eval intermediates while student steps run (and even
            // without a plan, a large teacher's eval forward peak must fit sequentially). Guaranteeing
            // `t_eval_peak >= ts.max_single_tensor` also keeps `per_sample_bytes >= max_single_tensor_per_sample`.
            let t_eval_peak = (ts.eval_forward_bytes / 2).max(ts.max_single_tensor);
            per_sample = per_sample.saturating_add(t_eval_peak);
            max_single = max_single.max(ts.max_single_tensor);
            max_fixed_tensor = max_fixed_tensor.max(ts.max_weight_elems.saturating_mul(teacher_elem));
        }
    }

    // Apply pool size-class rounding overhead (+25% per buffer) and fixed workspace floor.
    let fixed_bytes = pool_round(fixed).saturating_add(FIXED_WORKSPACE_BYTES);
    let per_sample_bytes = pool_round(per_sample).max(max_single);

    // Epoch validation (`Trainer::validate_epoch`) runs a temporary `F32` `fused_eval_clone()` while live model,
    // optimizer, EMA, and any teacher remain resident.
    let mut eval_mconf = if is_v2 {
        ModelConfig::new_v2(scale, req.num_classes)?
    } else {
        ModelConfig::new(scale, req.num_classes)?
    };
    eval_mconf.head = "o2m".into();
    let eval_stats = analyze_model(&eval_mconf, req.imgsz as usize, false, false)?;
    let eval_fixed_bytes = fixed_bytes.saturating_add(pool_round(params_bytes));
    let eval_per_sample_bytes =
        pool_round((eval_stats.eval_forward_bytes / 2).max(eval_stats.max_single_tensor));
    let eval_max_single_tensor = eval_stats.max_single_tensor;

    let fp = MemoryFootprint {
        model_params: usize::try_from(student.params).unwrap_or(usize::MAX),
        fixed_bytes,
        per_sample_bytes,
        max_single_tensor_per_sample: max_single,
    };

    Ok(FootprintDetails {
        fp,
        max_fixed_tensor_bytes: max_fixed_tensor,
        eval_fixed_bytes,
        eval_per_sample_bytes,
        eval_max_single_tensor,
        convs: student.convs,
    })
}

/// Worst-case single `F32` split-K partial-slab scratch buffer (`ctx.alloc(F32.bytes(slabs * wn))`) allocated by
/// `folo-gpu`'s `conv2d_backward_weight_as` across all model conv layers for physical micro-batch `batch`.
///
/// Under `amp`, takes the maximum of both the regular and cooperative-matrix (tensor-core) tile routes because
/// tensor cores may be enabled or fall back to regular kernels at runtime.
pub(super) fn max_wgrad_scratch_bytes(convs: &[ConvWgradShape], batch: usize, amp: bool) -> u64 {
    let mut max_bytes = 0u64;
    for c in convs {
        max_bytes = max_bytes.max(conv_wgrad_scratch_bytes(*c, batch, amp));
    }
    max_bytes
}

fn conv_wgrad_scratch_bytes(c: ConvWgradShape, batch: usize, amp: bool) -> u64 {
    if batch == 0 || c.groups == 0 {
        return 0;
    }
    let cin_g = c.cin / c.groups;
    let cout_g = c.cout / c.groups;
    let khkw = c.k.saturating_mul(c.k);
    let wn = c.cout.saturating_mul(cin_g).saturating_mul(khkw);
    let ohow = c.oh.saturating_mul(c.ow);
    let np = batch.saturating_mul(ohow);
    if wn == 0 || np == 0 {
        return 0;
    }

    // 1. Direct depthwise weight gradient (`weight_grad.rs:480-497`).
    if c.groups == c.cin && c.cin == c.cout && khkw <= MAX_DEPTHWISE_TAPS {
        let vec4 = c.ow.is_multiple_of(4);
        let np_u = if vec4 { np / 4 } else { np };
        if np_u == 0 {
            return 0;
        }
        let chunks = 1024usize
            .div_ceil(c.cout.max(1))
            .clamp(1, np.div_ceil(4096))
            .min(MAX_PARTIAL_ELEMS / wn)
            .min(np_u)
            .max(1);
        let ps = np_u.div_ceil(chunks);
        let chunks = np_u.div_ceil(ps);
        return if chunks > 1 { (chunks as u64).saturating_mul(wn as u64).saturating_mul(4) } else { 0 };
    }

    let reg = gemm_wgrad_scratch_bytes(
        c,
        batch,
        cin_g,
        cout_g,
        khkw,
        wn,
        ohow,
        np,
        pick_pointwise_wgrad_tile(cout_g, cin_g),
        pick_weight_grad_tile(cout_g, cin_g.saturating_mul(khkw)),
    );
    if !amp {
        return reg;
    }
    let coop = gemm_wgrad_scratch_bytes(
        c,
        batch,
        cin_g,
        cout_g,
        khkw,
        wn,
        ohow,
        np,
        pick_coop_pointwise_wgrad_tile(cout_g, cin_g),
        pick_coop_weight_grad_tile(cout_g, cin_g.saturating_mul(khkw)),
    );
    reg.max(coop)
}

#[allow(clippy::too_many_arguments)]
fn gemm_wgrad_scratch_bytes(
    c: ConvWgradShape,
    batch: usize,
    cin_g: usize,
    cout_g: usize,
    khkw: usize,
    wn: usize,
    ohow: usize,
    np: usize,
    pw_tile: (usize, usize, usize, usize),
    win_tile: (usize, usize, usize, usize),
) -> u64 {
    // 2. Pointwise 1×1 stride-1 unpadded per-image split-K GEMM (`weight_grad.rs:530-559`).
    if c.k == 1 && c.groups == 1 {
        let (bm, bn, bk, threads) = pw_tile;
        if ohow >= bk && batch <= 65_535 && batch.saturating_mul(wn) <= MAX_PARTIAL_ELEMS {
            let tiles = cout_g.div_ceil(bm).saturating_mul(cin_g.div_ceil(bn));
            let sp = plan_wgrad_per_img_sp(batch, c.groups, ohow, tiles, bk, threads, wn, 8);
            let total_slabs = batch.saturating_mul(sp);
            return if total_slabs > 1 {
                (total_slabs as u64).saturating_mul(wn as u64).saturating_mul(4)
            } else {
                0
            };
        }
    }

    // 3. General implicit-GEMM weight gradient (`weight_grad.rs:591-627` & global split-K fallback).
    let ncols = cin_g.saturating_mul(khkw);
    let (bm, bn, bk, threads) = win_tile;
    let tiles = cout_g.div_ceil(bm).saturating_mul(ncols.div_ceil(bn)).saturating_mul(c.groups);
    let target = (512 * 256 / threads).min(1024);
    let want = target
        .div_ceil(tiles.max(1))
        .min(np.div_ceil(16 * bk))
        .min(MAX_PARTIAL_ELEMS / wn)
        .min(65_535 / c.groups)
        .max(1);
    let per_img = ohow >= bk
        && batch.saturating_mul(c.groups) <= 65_535
        && batch.saturating_mul(wn) <= MAX_PARTIAL_ELEMS
        && want >= batch;
    let total_slabs = if per_img {
        let sp = plan_wgrad_per_img_sp(batch, c.groups, ohow, tiles, bk, threads, wn, 16);
        batch.saturating_mul(sp)
    } else {
        want
    };
    if total_slabs > 1 { (total_slabs as u64).saturating_mul(wn as u64).saturating_mul(4) } else { 0 }
}

/// Mirrors `folo-gpu::kernels::coop::convolution::pick_config_pointwise_wgrad` (`(bm, bn, bk, threads)`).
fn pick_coop_pointwise_wgrad_tile(m: usize, n: usize) -> (usize, usize, usize, usize) {
    const MAX_WG: usize = 65_535;
    if m.div_ceil(64) > MAX_WG {
        (128, 128, 32, 256)
    } else if m <= 16 && n <= 32 {
        (16, 32, 32, 64)
    } else if m <= 16 && n.div_ceil(64) <= MAX_WG {
        (16, 64, 32, 128)
    } else if m <= 16 {
        (16, 256, 32, 256)
    } else if (n <= 32 || (m <= 32 && n <= 64)) && m.div_ceil(32) <= MAX_WG && n.div_ceil(32) <= MAX_WG {
        (32, 32, 32, 128)
    } else if m <= 32 && n <= 192 && n.div_ceil(64) <= MAX_WG {
        (32, 64, 32, 128)
    } else if m <= 32 {
        (32, 128, 32, 128)
    } else if n >= 128 || n.div_ceil(64) > MAX_WG {
        (64, 128, 32, 256)
    } else {
        (64, 64, 32, 128)
    }
}

/// Mirrors `folo-gpu::kernels::coop::convolution::pick_config_weight_grad` (`(bm, bn, bk, threads)`).
fn pick_coop_weight_grad_tile(m: usize, n: usize) -> (usize, usize, usize, usize) {
    const MAX_WG: usize = 65_535;
    if m <= 16 && n <= 32 && m.div_ceil(16) <= MAX_WG && n.div_ceil(32) <= MAX_WG {
        return (16, 32, 32, 64);
    }
    if (n <= 32 || (m <= 32 && n <= 64 && m > 16)) && m.div_ceil(32) <= MAX_WG && n.div_ceil(32) <= MAX_WG {
        return (32, 32, 32, 128);
    }
    if m.div_ceil(64) > MAX_WG {
        return (128, 128, 32, 256);
    }
    match m {
        0..=16 if n.div_ceil(64) <= MAX_WG => (16, 64, 32, 128),
        0..=16 => (16, 256, 32, 256),
        17..=32 if n <= 192 && n.div_ceil(64) <= MAX_WG => (32, 64, 32, 128),
        17..=32 => (32, 128, 32, 128),
        _ if n >= 128 || n.div_ceil(64) > MAX_WG => (64, 128, 32, 256),
        _ => (64, 64, 32, 128),
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_wgrad_per_img_sp(
    batch: usize,
    groups: usize,
    ohow: usize,
    tiles: usize,
    bk: usize,
    threads: usize,
    wn: usize,
    min_chunk_tiles: usize,
) -> usize {
    let target = (512 * 256 / threads).min(1024);
    let base_groups = tiles.saturating_mul(batch);
    let max_sp = ohow
        .div_ceil(min_chunk_tiles * bk)
        .min((MAX_PARTIAL_ELEMS / batch.saturating_mul(wn).max(1)).max(1))
        .min(65_535 / batch.saturating_mul(groups).max(1))
        .max(1);
    let want = target.div_ceil(base_groups.max(1)).min(max_sp).max(1);
    if want <= 1 {
        return 1;
    }
    let chunk_p = ohow.div_ceil(want).next_multiple_of(bk);
    ohow.div_ceil(chunk_p)
}

/// Mirrors `folo-gpu::kernels::conv::weight_grad::pick_config_pointwise_wgrad` (`(bm, bn, bk, threads)`).
fn pick_pointwise_wgrad_tile(m: usize, n: usize) -> (usize, usize, usize, usize) {
    match (m, n) {
        (0..=32, 0..=32) => (32, 32, 16, 64),
        (0..=32, 33..=64) => (32, 64, 16, 128),
        (33..=64, 0..=32) => (64, 32, 16, 128),
        _ => pick_weight_grad_tile(m, n),
    }
}

/// Mirrors `folo-gpu::kernels::conv::weight_grad::pick_config_weight_grad` (`(bm, bn, bk, threads)`).
fn pick_weight_grad_tile(m: usize, n: usize) -> (usize, usize, usize, usize) {
    match m {
        0..=16 if n <= 32 => (16, 32, 32, 32),                         // STEM
        _ if n < 128 => (64, 64, 16, 256),                             // SMALL
        0..=32 => (32, 128, 32, 256),                                  // ROWS32
        _ if m.div_ceil(128) * 128 * 4 <= m * 5 => (128, 128, 8, 256), // LARGE
        _ => (64, 128, 16, 256),                                       // MEDIUM
    }
}

fn pool_round(bytes: u64) -> u64 {
    bytes.saturating_mul(POOL_OVERHEAD_NUM).div_ceil(POOL_OVERHEAD_DEN)
}

struct ModelStats {
    amp: bool,
    record_convs: bool,
    params: u64,
    conv_weight_params: u64,
    max_weight_elems: u64,
    bn_buffers: u64,
    per_sample_bytes: u64,
    eval_forward_bytes: u64,
    max_single_tensor: u64,
    convs: Vec<ConvWgradShape>,
}

impl ModelStats {
    fn new(amp: bool, record_convs: bool) -> Self {
        ModelStats {
            amp,
            record_convs,
            params: 0,
            conv_weight_params: 0,
            max_weight_elems: 0,
            bn_buffers: 0,
            per_sample_bytes: 0,
            eval_forward_bytes: 0,
            max_single_tensor: 0,
            convs: Vec::new(),
        }
    }

    fn elem_bytes(&self) -> u64 {
        if self.amp { 2 } else { 4 }
    }

    fn record_single(&mut self, bytes: u64) {
        self.max_single_tensor = self.max_single_tensor.max(bytes);
    }

    /// Account for one `ConvBnAct(cin -> cout, k, s)` producing `[cout, h, w]`.
    fn conv_bn_act(&mut self, cin: usize, cout: usize, k: usize, groups: usize, h: usize, w: usize) {
        let w_params = (cout as u64)
            .saturating_mul((cin / groups) as u64)
            .saturating_mul(k as u64)
            .saturating_mul(k as u64);
        let bn_p = (cout as u64).saturating_mul(2);
        self.conv_weight_params = self.conv_weight_params.saturating_add(w_params);
        self.max_weight_elems = self.max_weight_elems.max(w_params).max(cout as u64);
        self.params = self.params.saturating_add(w_params).saturating_add(bn_p);
        self.bn_buffers = self.bn_buffers.saturating_add(bn_p);
        let elems = (cout as u64).saturating_mul(h as u64).saturating_mul(w as u64);
        let elem = self.elem_bytes();
        // Under AMP: conv out, fused BN+SiLU activations, and dx bind `F16` storage buffers (2 B), with `F32`
        // 1-D per-channel BN stats; under F32: saved conv/BN/SiLU activations + backward grad buffer ≈ 3 * F32.
        let bytes_per_elem: u64 = if self.amp { 8 } else { 12 };
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(elems.saturating_mul(bytes_per_elem));
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(elems.saturating_mul(elem));
        self.record_single(elems.saturating_mul(elem));
        if self.record_convs {
            self.convs.push(ConvWgradShape { cin, cout, k, groups, oh: h, ow: w });
        }
    }

    /// Account for a bare `Conv2d(cin -> cout, k)` with bias.
    fn conv2d_bias(&mut self, cin: usize, cout: usize, k: usize, h: usize, w: usize) {
        let w_params =
            (cout as u64).saturating_mul(cin as u64).saturating_mul(k as u64).saturating_mul(k as u64);
        self.conv_weight_params = self.conv_weight_params.saturating_add(w_params);
        self.max_weight_elems = self.max_weight_elems.max(w_params).max(cout as u64);
        self.params = self.params.saturating_add(w_params).saturating_add(cout as u64);
        let elems = (cout as u64).saturating_mul(h as u64).saturating_mul(w as u64);
        let elem = self.elem_bytes();
        let out_bytes = elems.saturating_mul(elem);
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(out_bytes.saturating_mul(2));
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(out_bytes);
        self.record_single(out_bytes);
        if self.record_convs {
            self.convs.push(ConvWgradShape { cin, cout, k, groups: 1, oh: h, ow: w });
        }
    }

    fn bottleneck(&mut self, c1: usize, c2: usize, e: f64, h: usize, w: usize) {
        let hid = (c2 as f64 * e) as usize;
        self.conv_bn_act(c1, hid, 3, 1, h, w);
        self.conv_bn_act(hid, c2, 3, 1, h, w);
    }

    fn c3k(&mut self, c1: usize, c2: usize, n: usize, h: usize, w: usize) {
        let mid = c2 / 2;
        self.conv_bn_act(c1, mid, 1, 1, h, w);
        self.conv_bn_act(c1, mid, 1, 1, h, w);
        for _ in 0..n {
            self.bottleneck(mid, mid, 1.0, h, w);
        }
        let elem = self.elem_bytes();
        let cat_bytes = (2u64.saturating_mul(mid as u64))
            .saturating_mul(h as u64)
            .saturating_mul(w as u64)
            .saturating_mul(elem);
        self.record_single(cat_bytes);
        self.conv_bn_act(2 * mid, c2, 1, 1, h, w);
    }

    #[allow(clippy::too_many_arguments)]
    fn c3k2(&mut self, c1: usize, c2: usize, n: usize, use_c3k: bool, e: f64, h: usize, w: usize) {
        let mid = ((c2 as f64) * e) as usize;
        self.conv_bn_act(c1, 2 * mid, 1, 1, h, w);
        for _ in 0..n {
            if use_c3k {
                self.c3k(mid, mid, 2, h, w);
            } else {
                self.bottleneck(mid, mid, 0.5, h, w);
            }
        }
        let cat_c = (2 + n).saturating_mul(mid);
        let elem = self.elem_bytes();
        let cat_bytes = (cat_c as u64).saturating_mul(h as u64).saturating_mul(w as u64).saturating_mul(elem);
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(cat_bytes);
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(cat_bytes);
        self.record_single(cat_bytes);
        self.conv_bn_act(cat_c, c2, 1, 1, h, w);
    }

    fn sppf(&mut self, c1: usize, c2: usize, h: usize, w: usize) {
        let mid = c1 / 2;
        self.conv_bn_act(c1, mid, 1, 1, h, w);
        let elem = self.elem_bytes();
        // Three MaxPool outputs + 4-way concat.
        let pool_bytes = (mid as u64).saturating_mul(h as u64).saturating_mul(w as u64).saturating_mul(elem);
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(pool_bytes.saturating_mul(7));
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(pool_bytes.saturating_mul(7));
        self.record_single(pool_bytes.saturating_mul(4));
        self.conv_bn_act(4 * mid, c2, 1, 1, h, w);
    }

    fn c2psa(&mut self, c: usize, n: usize, h: usize, w: usize) {
        let mid = c / 2;
        self.conv_bn_act(c, 2 * mid, 1, 1, h, w);
        let heads = (mid / 64).max(1);
        let head_dim = mid / heads;
        let key_dim = head_dim / 2;
        let qkv_out = heads * (2 * key_dim + head_dim);
        let tokens = (h as u64).saturating_mul(w as u64);
        let attn_elems = (heads as u64).saturating_mul(tokens).saturating_mul(tokens);
        // `Tensor::softmax` promotes `F16` inputs to `F32` under autocast (`folo-core/src/ops/reduce.rs`), so the
        // softmax input, saved output, and backward grad are always `F32` (4 B/elem).
        let attn_f32 = attn_elems.saturating_mul(4);
        let attn_train = if self.amp {
            // F16 QK^T logits (2 B) + F32 cast (4 B) + F32 softmax output (4 B) + F32 backward grad (4 B).
            attn_elems.saturating_mul(14)
        } else {
            // F32 QK^T logits (4 B) + F32 softmax output (4 B) + F32 backward grad (4 B).
            attn_elems.saturating_mul(12)
        };
        let attn_eval = if self.amp { attn_elems.saturating_mul(10) } else { attn_elems.saturating_mul(8) };
        for _ in 0..n {
            self.conv_bn_act(mid, qkv_out, 1, 1, h, w);
            self.per_sample_bytes = self.per_sample_bytes.saturating_add(attn_train);
            self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(attn_eval);
            self.record_single(attn_f32);
            self.conv_bn_act(mid, mid, 3, mid, h, w); // pe depthwise
            self.conv_bn_act(mid, mid, 1, 1, h, w); // proj
            self.conv_bn_act(mid, 2 * mid, 1, 1, h, w); // FFN expand
            self.conv_bn_act(2 * mid, mid, 1, 1, h, w); // FFN contract
        }
        self.conv_bn_act(2 * mid, c, 1, 1, h, w);
    }

    /// Account for a `Linear(cin -> cout, bias=true)` layer on `tokens` rows.
    fn linear_bias(&mut self, cin: usize, cout: usize, tokens: usize) {
        let w_params = (cout as u64).saturating_mul(cin as u64);
        self.conv_weight_params = self.conv_weight_params.saturating_add(w_params);
        self.max_weight_elems = self.max_weight_elems.max(w_params).max(cout as u64);
        self.params = self.params.saturating_add(w_params).saturating_add(cout as u64);
        let elems = (tokens as u64).saturating_mul(cout as u64);
        let out_bytes = elems.saturating_mul(4); // HyperContext/FullPadGate run Linear in F32
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(out_bytes.saturating_mul(2));
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(out_bytes);
        self.record_single(out_bytes);
    }

    /// Account for one `WsQARepConv3x3(cin -> cout)` at `(h, w)`: 4 multi-branch conv weights (`3×3`, `1×3`,
    /// `3×1`, `1×1` = `16 * cout * cin`), optional `gamma_id` (`cout` when `cin == cout`), `BN_post`, and
    /// effective `3×3` conv activations + wgrad scratch.
    fn ws_qarep_3x3(&mut self, cin: usize, cout: usize, h: usize, w: usize) {
        let w33 = (cout as u64).saturating_mul(cin as u64).saturating_mul(9);
        let w_branches = (cout as u64).saturating_mul(cin as u64).saturating_mul(16);
        let gamma_id = if cin == cout { cout as u64 } else { 0 };
        let bn_p = (cout as u64).saturating_mul(2);
        self.conv_weight_params = self.conv_weight_params.saturating_add(w_branches);
        self.max_weight_elems = self.max_weight_elems.max(w33).max(cout as u64);
        self.params = self.params.saturating_add(w_branches).saturating_add(gamma_id).saturating_add(bn_p);
        self.bn_buffers = self.bn_buffers.saturating_add(bn_p);
        let elems = (cout as u64).saturating_mul(h as u64).saturating_mul(w as u64);
        let elem = self.elem_bytes();
        let bytes_per_elem: u64 = if self.amp { 8 } else { 12 };
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(elems.saturating_mul(bytes_per_elem));
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(elems.saturating_mul(elem));
        self.record_single(elems.saturating_mul(elem));
        if self.record_convs {
            self.convs.push(ConvWgradShape { cin, cout, k: 3, groups: 1, oh: h, ow: w });
        }
    }

    /// Account for one FOLO v2 `FoloBlock(c1 -> c2, m, c_mid)` at `(h, w)`.
    fn folo_block(&mut self, c1: usize, c2: usize, m: usize, c_mid: usize, h: usize, w: usize) {
        let taps: usize = if m <= 2 { 1 } else { 2 };
        self.conv_bn_act(c1, 2 * c_mid, 1, 1, h, w);
        for _ in 0..m {
            self.ws_qarep_3x3(c_mid, c_mid, h, w);
        }
        let cat_c = (2usize + taps).saturating_mul(c_mid);
        let cat_bytes = (cat_c as u64)
            .saturating_mul(h as u64)
            .saturating_mul(w as u64)
            .saturating_mul(self.elem_bytes());
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(cat_bytes);
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(cat_bytes);
        self.record_single(cat_bytes);
        self.conv_bn_act(cat_c, c2, 1, 1, h, w);
    }

    /// Account for FOLO v2 `HyperContext(c5, m_edges)` at P5 `(h, w)`.
    fn hyper_context(&mut self, c5: usize, m_edges: usize, h: usize, w: usize) {
        let c_g = c5 / 2;
        self.conv_bn_act(c5, m_edges + c_g, 1, 1, h, w);
        let tokens = (h as u64).saturating_mul(w as u64);
        let attn_f32 = (m_edges as u64).saturating_mul(tokens).saturating_mul(4);
        self.per_sample_bytes = self.per_sample_bytes.saturating_add(attn_f32.saturating_mul(6));
        self.eval_forward_bytes = self.eval_forward_bytes.saturating_add(attn_f32.saturating_mul(3));
        self.record_single(attn_f32);
        self.linear_bias(c_g, c_g, m_edges);
        self.linear_bias(c_g, c_g, m_edges);
        self.conv_bn_act(c_g, c5, 1, 1, h, w);
    }

    /// Account for one FOLO v2 `ClsBranchV2(cin -> c_cls -> nc)` at `(hw, hw)`.
    fn cls_branch_v2(&mut self, cin: usize, c_cls: usize, nc: usize, hw: usize) {
        if cin != c_cls {
            self.conv_bn_act(cin, c_cls, 1, 1, hw, hw);
        }
        self.ws_qarep_3x3(c_cls, c_cls, hw, hw);
        self.conv2d_bias(c_cls, nc, 1, hw, hw);
    }

    fn detect_branch(&mut self, chs: [usize; 3], imgsz: usize, nc: usize, reg_max: usize) {
        let c2 = 16usize.max(chs[0] / 4).max(4 * reg_max);
        let c3 = chs[0].max(nc.min(100));
        let strides = [8usize, 16, 32];
        for (&cin, &s) in chs.iter().zip(&strides) {
            let hw = imgsz / s;
            // Box branch: conv0(3x3), conv1(3x3), out(1x1)
            self.conv_bn_act(cin, c2, 3, 1, hw, hw);
            self.conv_bn_act(c2, c2, 3, 1, hw, hw);
            self.conv2d_bias(c2, 4 * reg_max, 1, hw, hw);
            // Cls branch: dw0(3x3, g=cin), pw0(1x1), dw1(3x3, g=c3), pw1(1x1), out(1x1)
            self.conv_bn_act(cin, cin, 3, cin, hw, hw);
            self.conv_bn_act(cin, c3, 1, 1, hw, hw);
            self.conv_bn_act(c3, c3, 3, c3, hw, hw);
            self.conv_bn_act(c3, c3, 1, 1, hw, hw);
            self.conv2d_bias(c3, nc, 1, hw, hw);
        }
    }
}

fn analyze_model(cfg: &ModelConfig, imgsz: usize, amp: bool, record_convs: bool) -> Result<ModelStats> {
    if cfg.is_v2() {
        let v2 = FoloV2Config::from_model_config(cfg)?;
        Ok(analyze_folo_v2_model(&v2, imgsz, amp, record_convs))
    } else {
        Ok(analyze_folo_model(cfg, imgsz, amp, record_convs))
    }
}

fn analyze_folo_v2_model(cfg: &FoloV2Config, imgsz: usize, amp: bool, record_convs: bool) -> ModelStats {
    let s = ScalingV2::of(cfg);
    let mut st = ModelStats::new(amp, record_convs);

    let in_bytes =
        (cfg.in_channels as u64).saturating_mul(imgsz as u64).saturating_mul(imgsz as u64).saturating_mul(4);
    st.per_sample_bytes = st.per_sample_bytes.saturating_add(in_bytes);
    st.eval_forward_bytes = st.eval_forward_bytes.saturating_add(in_bytes);
    st.record_single(in_bytes);

    let (h2, h4, h8, h16, h32) = (imgsz / 2, imgsz / 4, imgsz / 8, imgsz / 16, imgsz / 32);
    let (c0, c1, c3, c4) = (s.ch(64), s.ch(128), s.ch(256), s.ch(512));
    let (c5, c6, c7, c8) = (s.ch(512), s.ch(512), s.ch(1024), s.ch(1024));
    let c9 = c8;
    let (c13, c16, c19, c22) = (s.ch(512), s.ch(256), s.ch(512), s.ch(1024));
    let c_p2 = c1;
    let c_g = c9 / 2;

    // BackboneV2
    if cfg.spd_stem {
        st.conv_bn_act(4 * cfg.in_channels, c0, 1, 1, h2, h2);
    } else {
        st.conv_bn_act(cfg.in_channels, c0, 3, 1, h2, h2);
    }
    st.conv_bn_act(c0, c1, 3, 1, h4, h4);
    for _ in 0..s.stage2_m {
        st.ws_qarep_3x3(c1, c1, h4, h4);
    }
    st.conv_bn_act(c1, c3, 3, 1, h8, h8);
    st.folo_block(c3, c4, s.bb_m, c3 / 2, h8, h8);
    st.conv_bn_act(c4, c5, 3, 1, h16, h16);
    st.folo_block(c5, c6, s.bb_m, c6 / 2, h16, h16);
    st.conv_bn_act(c6, c7, 3, 1, h32, h32);
    st.folo_block(c7, c8, s.bb_m, c8 / 2, h32, h32);
    st.sppf(c8, c9, h32, h32);
    st.hyper_context(c9, s.hyper_edges, h32, h32);

    // NeckV2
    let elem = st.elem_bytes();
    let cat_bytes = |c_sum: usize, h: usize| -> u64 {
        (c_sum as u64).saturating_mul(h as u64).saturating_mul(h as u64).saturating_mul(elem)
    };
    st.record_single(cat_bytes(c9 + c6, h16));
    st.folo_block(c9 + c6, c13, s.neck_m, c13 / 2, h16, h16);
    st.conv_bn_act(4 * c1, c_p2, 1, 1, h8, h8); // p2_proj
    st.record_single(cat_bytes(c13 + c4 + c_p2, h8));
    st.folo_block(c13 + c4 + c_p2, c16, s.out3_m, c16 / 2, h8, h8);
    st.conv_bn_act(c16, c16, 3, 1, h16, h16); // down3
    st.record_single(cat_bytes(c16 + c13, h16));
    st.folo_block(c16 + c13, c19, s.neck_m, c19 / 2, h16, h16);
    st.conv_bn_act(c19, c19, 3, 1, h32, h32); // down4
    st.record_single(cat_bytes(c19 + c9, h32));
    st.folo_block(c19 + c9, c22, s.out5_m, c22 / 2, h32, h32);
    if cfg.fullpad {
        st.linear_bias(c_g, c16, 1);
        st.linear_bias(c_g, c19, 1);
    }

    // LazyAlignHead
    let chs = [c16, c19, c22];
    let strides = [8usize, 16, 32];
    for (&cin, &stride) in chs.iter().zip(&strides) {
        let hw = imgsz / stride;
        st.ws_qarep_3x3(cin, s.c_box, hw, hw);
        st.ws_qarep_3x3(s.c_box, s.c_box, hw, hw);
        st.conv2d_bias(s.c_box, 4 * cfg.reg_max, 1, hw, hw);
        if cfg.iou_head {
            st.conv2d_bias(s.c_box, 1, 1, hw, hw);
        }
        st.cls_branch_v2(cin, s.c_cls, cfg.num_classes, hw);
    }
    if cfg.has_o2o() {
        let saved_eval = st.eval_forward_bytes;
        for (&cin, &stride) in chs.iter().zip(&strides) {
            let hw = imgsz / stride;
            st.cls_branch_v2(cin, s.c_cls, cfg.num_classes, hw);
            st.conv2d_bias(s.c_cls, 1, 1, hw, hw); // PeakSuppress::gate
            st.params = st.params.saturating_add(cfg.num_classes as u64); // PeakSuppress::alpha_raw
            st.max_weight_elems = st.max_weight_elems.max(cfg.num_classes as u64);
        }
        // Under default `o2o_detach == true`, `LazyAlignHead::forward_o2o` re-runs the shared box branch
        // (`forward_box_only`) on detached features during training: zero extra parameters or eval memory,
        // but a second set of box-branch autograd activations.
        let saved_params = st.params;
        let saved_conv_weights = st.conv_weight_params;
        let saved_bn_buffers = st.bn_buffers;
        let saved_convs_len = st.convs.len();
        for (&cin, &stride) in chs.iter().zip(&strides) {
            let hw = imgsz / stride;
            st.ws_qarep_3x3(cin, s.c_box, hw, hw);
            st.ws_qarep_3x3(s.c_box, s.c_box, hw, hw);
            st.conv2d_bias(s.c_box, 4 * cfg.reg_max, 1, hw, hw);
        }
        st.params = saved_params;
        st.conv_weight_params = saved_conv_weights;
        st.bn_buffers = saved_bn_buffers;
        st.convs.truncate(saved_convs_len);
        st.eval_forward_bytes = saved_eval;
    }

    let anchors = total_anchors(imgsz);
    let box_raw_f32 = (4u64.saturating_mul(cfg.reg_max as u64)).saturating_mul(anchors).saturating_mul(4);
    let cls_raw_f32 = (cfg.num_classes as u64).saturating_mul(anchors).saturating_mul(4);
    let head_branches: u64 = if cfg.has_o2o() { 2 } else { 1 };
    let loss_work =
        box_raw_f32.saturating_add(cls_raw_f32).saturating_mul(3).saturating_add(anchors.saturating_mul(32));
    st.per_sample_bytes = st.per_sample_bytes.saturating_add(loss_work.saturating_mul(head_branches));
    st.record_single(box_raw_f32);
    st.record_single(cls_raw_f32);
    st
}

fn analyze_folo_model(cfg: &ModelConfig, imgsz: usize, amp: bool, record_convs: bool) -> ModelStats {
    let s = Scaling::of(cfg);
    let (ch, rep) = (|c| s.ch(c), |n| s.rep(n));
    let mut st = ModelStats::new(amp, record_convs);

    // Input image tensor [3, H, W] in F32 (uploaded as u8/f32 before autocast).
    let in_bytes =
        (cfg.in_channels as u64).saturating_mul(imgsz as u64).saturating_mul(imgsz as u64).saturating_mul(4);
    st.per_sample_bytes = st.per_sample_bytes.saturating_add(in_bytes);
    st.eval_forward_bytes = st.eval_forward_bytes.saturating_add(in_bytes);
    st.record_single(in_bytes);

    let (h2, h4, h8, h16, h32) = (imgsz / 2, imgsz / 4, imgsz / 8, imgsz / 16, imgsz / 32);
    let (c0, c1, c2, c3, c4) = (ch(64), ch(128), ch(256), ch(256), ch(512));
    let (c5, c6, c7, c8, c9) = (ch(512), ch(512), ch(1024), ch(1024), ch(1024));
    let c10 = c9;
    let (c13, c16, c17, c19, c20, c22) = (ch(512), ch(256), ch(256), ch(512), ch(512), ch(1024));

    // Backbone (layers 0-10)
    st.conv_bn_act(cfg.in_channels, c0, 3, 1, h2, h2); // stem
    st.conv_bn_act(c0, c1, 3, 1, h4, h4); // down2
    st.c3k2(c1, c2, rep(2), s.c3k(false), 0.25, h4, h4); // stage2
    st.conv_bn_act(c2, c3, 3, 1, h8, h8); // down3
    st.c3k2(c3, c4, rep(2), s.c3k(false), 0.25, h8, h8); // stage3
    st.conv_bn_act(c4, c5, 3, 1, h16, h16); // down4
    st.c3k2(c5, c6, rep(2), s.c3k(true), 0.5, h16, h16); // stage4
    st.conv_bn_act(c6, c7, 3, 1, h32, h32); // down5
    st.c3k2(c7, c8, rep(2), s.c3k(true), 0.5, h32, h32); // stage5
    st.sppf(c8, c9, h32, h32); // sppf
    st.c2psa(c9, rep(2), h32, h32); // psa

    // Neck (layers 11-22)
    let elem = st.elem_bytes();
    let cat_bytes = |c_sum: usize, h: usize| -> u64 {
        (c_sum as u64).saturating_mul(h as u64).saturating_mul(h as u64).saturating_mul(elem)
    };
    st.record_single(cat_bytes(c10 + c6, h16));
    st.c3k2(c10 + c6, c13, rep(2), s.c3k(false), 0.5, h16, h16); // td4
    st.record_single(cat_bytes(c13 + c4, h8));
    st.c3k2(c13 + c4, c16, rep(2), s.c3k(false), 0.5, h8, h8); // out3
    st.conv_bn_act(c16, c17, 3, 1, h16, h16); // down3
    st.record_single(cat_bytes(c17 + c13, h16));
    st.c3k2(c17 + c13, c19, rep(2), s.c3k(false), 0.5, h16, h16); // out4
    st.conv_bn_act(c19, c20, 3, 1, h32, h32); // down4
    st.record_single(cat_bytes(c20 + c10, h32));
    st.c3k2(c20 + c10, c22, rep(2), s.c3k(true), 0.5, h32, h32); // out5

    // Head (layer 23): o2m + optional o2o branch
    let reg_max = cfg.reg_max as usize;
    let nc = cfg.num_classes;
    let o2o = cfg.has_o2o();
    st.detect_branch([c16, c19, c22], imgsz, nc, reg_max);
    if o2o {
        // Under eval/teacher distillation, only one branch executes per forward, but both sets of weights stay
        // resident. Record params and training activations for the second branch without doubling eval forward.
        let saved_eval = st.eval_forward_bytes;
        st.detect_branch([c16, c19, c22], imgsz, nc, reg_max);
        st.eval_forward_bytes = saved_eval;
    }

    // Loss & TAL working buffers (always F32, even under AMP; doubled for o2o dual loss).
    let anchors = total_anchors(imgsz);
    let box_raw_f32 = (4u64.saturating_mul(reg_max as u64)).saturating_mul(anchors).saturating_mul(4);
    let cls_raw_f32 = (nc as u64).saturating_mul(anchors).saturating_mul(4);
    let head_branches: u64 = if o2o { 2 } else { 1 };
    let loss_work =
        box_raw_f32.saturating_add(cls_raw_f32).saturating_mul(3).saturating_add(anchors.saturating_mul(32));
    st.per_sample_bytes = st.per_sample_bytes.saturating_add(loss_work.saturating_mul(head_branches));
    st.record_single(box_raw_f32);
    st.record_single(cls_raw_f32);
    st
}

fn total_anchors(imgsz: usize) -> u64 {
    let a = |s: usize| {
        let g = (imgsz / s) as u64;
        g.saturating_mul(g)
    };
    a(8).saturating_add(a(16)).saturating_add(a(32))
}
