//! Candidate-first prediction on a GPU (docs/MODEL.md §9.5): per level the class branch runs densely, a class pass
//! lists the candidates (and the pixels their box branch reads) into fixed-capacity lists on the device, the box
//! branch runs at those sites only ([`Tensor::conv2d_act_sites`]: `conv0` on the candidates' 3×3 windows, written
//! back into a map; `conv1` and `out` at the candidates), and one decode appends every level's boxes to one record
//! list per image, read back once. The site convs are the dense convs' values bit for bit (same kernels, same
//! split-K chunks), so the records are those of the dense head's [`Tensor::detect_candidates`].
//!
//! The lists' capacities fix the dispatches, whatever an image's candidates; a level with more candidates than
//! slots in some image is flagged, and those images take their candidates from the dense head (a second round
//! trip on the same features).
//!
//! DFL box fusion (`dfl`, `sebf`) and DFL-based score rules (`dfl-consensus`, `dfl-repulse`, `dfl-cgr`) need per-candidate side moments
//! and/or Shannon entropy: [`dfl_stats_with`] runs on the candidates' box-logit columns, which hold the dense
//! logits bit for bit, so the stats are the dense ones bit for bit. When any DFL stats are requested, all three
//! levels' site lists are concatenated with the records into one `u32` buffer so the host downloads at most two
//! buffers (the packed `u32` buffer and the single `[B, ch, Σ cap]` `f32` stats buffer).
//!
//! Everything from the input to the records (and the DFL stats) is a pure function of the input with a fixed
//! dispatch list, so with the model's forward plans on ([`FoloModel::set_forward_plans`], as in a `Predictor`) the
//! whole device side is one recorded plan ([`folo_gpu::ForwardPlans`]): a replayed prediction is one flat list of
//! dispatches, then at most two readbacks (the records packed with the site lists in one `u32` buffer; with DFL
//! fusion/scoring also the single `[B, ch, Σ cap]` `f32` stats buffer).

use std::collections::HashMap;
use std::sync::Mutex;

use folo_core::{ClassSites, Device, Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_nn::{ConvBnAct, Forward, Result};

use super::head::{HeadLevel, HeadMode};
use super::predict::{
    DflWants, bin_precision_from_moments, candidates_of_records, dfl_stats_channels, dfl_stats_with,
    entropy_from_stats, moment_bins, precision_from_moments,
};
use super::{FoloModel, HeadOutput, ImageCandidates, each_tensor};
use crate::nms::NmsConfig;

/// One level's site outputs for the decode: the raw box outputs `[B, 4 · reg_max, cap]`, the site list
/// `[B, 1 + cap]`, the class records, the map width, the stride and the level's first anchor.
type SitePart = (Tensor, Tensor, Tensor, usize, f32, usize);

/// Tags the planned device side's key apart from the forward's ([`FoloModel::forward`]).
const SITES_PLAN: u64 = 1;

/// Candidate slots per image of the P3, P4 and P5 levels (at most the level's pixels). At `conf = 0.25` over COCO
/// val2017 with the published weights the most any image has is far below (docs/notes/hill/r3-lazy-head.md); the
/// slots past an image's candidates cost a workgroup that returns at once.
pub const GPU_SITE_SLOTS: [usize; 3] = [1024, 512, 512];

/// `c` (a fused `Conv` block) at the sites `sites` ([`Tensor::conv2d_act_sites`]); `None` when the device has no
/// kernel for it.
fn conv_sites(c: &ConvBnAct, x: &Tensor, sites: &Tensor, to_map: bool) -> Result<Option<Tensor>> {
    debug_assert!(c.is_fused(), "candidate-first prediction needs a fused model");
    let p = c.conv.config().params();
    Ok(x.conv2d_act_sites(&c.conv.weight, c.conv.bias.as_ref(), &p, c.act.unary_op(), sites, to_map)?)
}

/// [`moment_bins`] on `dev`, one tensor per registered context and `reg_max` for the process: a planned call reads
/// it (listed with the model's tensors) instead of uploading it. Keyed by the context serving `dev`, not by the
/// ordinal: a context registered again at the same device ([`folo_gpu::register`]) gets bins of its own, and the
/// replaced context's are dropped. Without a registered context the bins are uploaded for the call.
pub(crate) fn device_bins(dev: Device, rm: usize) -> Result<Tensor> {
    static BINS: Mutex<Vec<(Device, GpuContext, usize, Tensor)>> = Mutex::new(Vec::new());
    let Some(ctx) = folo_gpu::context_for(dev) else { return Ok(moment_bins(rm)?.to_device(dev)?) };
    let mut bins = BINS.lock().unwrap_or_else(|p| p.into_inner());
    bins.retain(|(d, c, _, _)| *d != dev || c.same_device(&ctx));
    if let Some((.., k)) = bins.iter().find(|(d, _, r, _)| *d == dev && *r == rm) {
        return Ok(k.clone());
    }
    let k = moment_bins(rm)?.to_device(dev)?;
    bins.push((dev, ctx, rm, k.clone()));
    Ok(k)
}

/// Stride of global `anchor` from per-level `(cap, stride, start)` layout.
fn stride_of(layout: &[(usize, f32, usize)], anchor: usize) -> f32 {
    for w in layout.windows(2) {
        if anchor < w[1].2 {
            return w[0].1;
        }
    }
    layout.last().map_or(0.0, |l| l.1)
}

impl FoloModel {
    /// [`FoloModel::predict_candidates`] on a GPU: the candidates at the sites, with the dense head serving the
    /// images whose candidates did not fit (and every image on a device without the site kernels).
    pub(super) fn predict_candidates_gpu(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<ImageCandidates>> {
        let _g = no_grad();
        let (h, w) = self.check_input(x)?;
        let levels = self.site_levels();
        let o2o = self.head_mode == HeadMode::O2o && self.head.o2o.is_some();
        let wants = self.dfl_wants(nms);
        let want_strides = self.wants_strides(nms);
        let cand_conf = nms.candidate_conf_for(self.head_mode);
        let bins = wants.moments().then(|| device_bins(x.device(), self.head.reg_max)).transpose()?;
        let tensors = |f: &mut dyn FnMut(&Tensor)| {
            each_tensor(self, f);
            bins.iter().for_each(&mut *f);
        };
        // What reaches a kernel besides the input and the tensors: the branch, the class pass's threshold (a
        // parameter block), the slots, and which DFL stats (moments / entropy) are computed and returned.
        let [s3, s4, s5] = GPU_SITE_SLOTS.map(|s| s as u64);
        let stats_bits = (wants.moments() as u64) | ((wants.ent as u64) << 1);
        let key = [SITES_PLAN, o2o as u64, stats_bits, cand_conf.to_bits() as u64, s3, s4, s5];
        // The features of a call that found no site kernel, for the dense head. Such a call returns no results, so
        // it is never a plan, and every call of it runs (and fills this).
        let mut unsited = None;
        let out = self.plans.run(&key, x, &tensors, |x| {
            let feats = self.features(x, None)?;
            let out = self.sites_on_device(levels, &feats, cand_conf, wants, bins.as_ref())?;
            if out.is_none() {
                unsited = Some(feats);
            }
            Ok::<_, folo_nn::Error>(out.unwrap_or_default())
        })?;
        if out.is_empty() {
            // No site kernel on this device: the dense head on the same features.
            return self.dense_candidates(x, unsited, nms, h, w);
        }
        let (cands, fits) = self.read_sites(&out, levels, h, w, wants, want_strides)?;
        if fits.iter().all(|&f| f) {
            return Ok(cands);
        }
        // The images that did not fit, from the dense head (the features again, unplanned).
        let dense = self.dense_candidates(x, None, nms, h, w)?;
        Ok(cands.into_iter().zip(fits).zip(dense).map(|((c, f), d)| if f { c } else { d }).collect())
    }

    /// The head levels of the [`FoloModel::head_mode`] branch.
    fn site_levels(&self) -> &[HeadLevel] {
        match (self.head_mode, &self.head.o2o) {
            (HeadMode::O2o, Some(o2o)) => o2o,
            _ => &self.head.levels,
        }
    }

    /// Every image's candidates from the dense head ([`HeadOutput::candidates_on_device`]) on `feats`, the features
    /// of `x` (computed here when `None`).
    fn dense_candidates(
        &self,
        x: &Tensor,
        feats: Option<[Tensor; 3]>,
        nms: &NmsConfig,
        h: usize,
        w: usize,
    ) -> Result<Vec<ImageCandidates>> {
        let feats = match feats {
            Some(f) => f,
            None => self.features(x, None)?,
        };
        let feats = [&feats[0], &feats[1], &feats[2]];
        let mode = self.head_mode;
        let (box_raw, cls_raw) = match mode {
            HeadMode::O2o => self
                .head
                .forward_o2o(feats)?
                .ok_or_else(|| folo_nn::Error::Invalid(format!("{} has no one-to-one head", self.arch())))?,
            HeadMode::O2m => self.head.forward(feats)?,
        };
        let out = HeadOutput {
            box_raw,
            cls_raw,
            mode,
            o2o: None,
            anchors: self.anchors(h, w),
            reg_max: self.head.reg_max,
            num_classes: self.head.num_classes,
        };
        out.candidates_on_device(nms)?
            .ok_or_else(|| folo_nn::Error::Invalid(format!("no candidate kernel on {:?}", x.device())))
    }

    /// The device side of candidate-first prediction on `feats`, nothing read back: the records tensor of
    /// [`Tensor::detect_decode_sites`], or when `wants.any_stats()` two tensors: the records concatenated along
    /// dim 1 with all three levels' `u32` site lists (one `u32` readback instead of four) and the combined
    /// `[B, ch, Σ cap]` DFL stats tensor. `None` when the device lacks a site kernel.
    fn sites_on_device(
        &self,
        levels: &[HeadLevel],
        feats: &[Tensor; 3],
        conf: f32,
        wants: DflWants,
        bins: Option<&Tensor>,
    ) -> Result<Option<Vec<Tensor>>> {
        let (b, rm) = (feats[0].dims()[0], self.head.reg_max);
        let mut parts: Vec<SitePart> = Vec::with_capacity(3);
        let mut start = 0;
        for (l, (lv, x)) in levels.iter().zip(feats).enumerate() {
            let (_, _, h, w) = x.dims4()?;
            let cap = GPU_SITE_SLOTS[l].min(h * w);
            let cls = lv.cls_branch.forward(x)?;
            let Some((sites, recs, field)) = cls.detect_class_sites(&ClassSites { conf, cap })? else {
                return Ok(None);
            };
            let bx = &lv.box_branch;
            let Some(h0) = conv_sites(&bx.conv0, x, &field, true)? else { return Ok(None) };
            let Some(h1) = conv_sites(&bx.conv1, &h0, &sites, false)? else { return Ok(None) };
            drop(h0);
            // `out` (1×1) on the candidates' columns as a 1-pixel-high map: the dense GEMM's sums, column by column.
            let c = h1.dims()[1];
            let raw = bx.out.forward(&h1.reshape([b, c, 1, cap])?)?.reshape([b, 4 * rm, cap])?;
            parts.push((raw, sites, recs, w, lv.stride as f32, start));
            start += h * w;
        }
        let refs: Vec<_> = parts.iter().map(|(r, s, c, w, st, a)| (r, s, c, *w, *st, *a)).collect();
        let Some(out) = Tensor::detect_decode_sites(&refs, rm)? else { return Ok(None) };
        if !wants.any_stats() {
            return Ok(Some(vec![out]));
        }
        // `Σ cap ≥ 3` columns, so the column kernels the dense `A` takes.
        let raws: Vec<&Tensor> = parts.iter().map(|p| &p.0).collect();
        let stats = dfl_stats_with(&Tensor::cat(&raws, 2)?, rm, bins, wants.moments(), wants.ent)?;
        let mut u32_parts: Vec<&Tensor> = Vec::with_capacity(1 + parts.len());
        u32_parts.push(&out);
        u32_parts.extend(parts.iter().map(|p| &p.1));
        let packed = Tensor::cat(&u32_parts, 1)?;
        Ok(Some(vec![packed, stats]))
    }

    /// Every image's candidates in anchor order from [`FoloModel::sites_on_device`]'s results (with DFL stats
    /// when requested), and whether they fit the slots (`false`: some level had more; that image's candidates are
    /// partial and carry no stats). `h × w` is the input.
    fn read_sites(
        &self,
        out: &[Tensor],
        levels: &[HeadLevel],
        h: usize,
        w: usize,
        wants: DflWants,
        want_strides: bool,
    ) -> Result<(Vec<ImageCandidates>, Vec<bool>)> {
        let strides = [levels[0].stride, levels[1].stride, levels[2].stride];
        read_sites_for_strides(out, strides, h, w, wants, want_strides)
    }
}

/// Shared site-record and DFL-stats unpacker for GPU candidate-first prediction on both `FoloModel` and
/// [`super::v2::FoloV2Model`].
pub(crate) fn read_sites_for_strides(
    out: &[Tensor],
    strides: [u32; 3],
    h: usize,
    w: usize,
    wants: DflWants,
    want_strides: bool,
) -> Result<(Vec<ImageCandidates>, Vec<bool>)> {
    let mut start = 0usize;
    let layout: Vec<(usize, f32, usize)> = strides
        .iter()
        .enumerate()
        .map(|(l, &stride)| {
            let at = start;
            let st = stride as usize;
            let hw = (h / st) * (w / st);
            start += hw;
            (GPU_SITE_SLOTS[l].min(hw), stride as f32, at)
        })
        .collect();
    let total_words = out[0].dims()[1];
    let has_stats = out.len() == 2;
    let site_words: usize = if has_stats { layout.iter().map(|l| 1 + l.0).sum() } else { 0 };
    let rec_words = total_words - site_words;
    let v = out[0].to_vec::<u32>()?;
    let mut cands = Vec::with_capacity(out[0].dims()[0]);
    let mut fits = Vec::with_capacity(cands.capacity());
    for row in v.chunks_exact(total_words) {
        let rec = &row[..rec_words];
        let n = (rec[0] as usize).min((rec_words - 2) / 7);
        let fit = rec[1] == 0;
        let list = candidates_of_records(&rec[2..2 + 7 * n]);
        let strides = (want_strides && fit)
            .then(|| list.iter().map(|c| stride_of(&layout, c.anchor as usize)).collect());
        cands.push(ImageCandidates {
            cands: list,
            precisions: None,
            bin_precisions: None,
            entropies: None,
            strides,
        });
        fits.push(fit);
    }
    if has_stats && wants.any_stats() {
        unpack_site_stats(&out[1], &v, total_words, rec_words, &layout, &mut cands, &fits, wants)?;
    }
    Ok((cands, fits))
}

/// DFL precisions and/or entropies of the fitting images' candidates from the packed `u32` buffer's site lists
/// and the single `stats` (`[B, ch, Σ cap]`) tensor. Skips the `stats` readback when no fitting image has any
/// candidate.
#[allow(clippy::too_many_arguments)]
fn unpack_site_stats(
    stats: &Tensor,
    packed_u32: &[u32],
    total_words: usize,
    rec_words: usize,
    layout: &[(usize, f32, usize)],
    cands: &mut [ImageCandidates],
    fits: &[bool],
    wants: DflWants,
) -> Result<()> {
    let any_needed = fits.iter().zip(cands.iter()).any(|(&fit, c)| fit && !c.cands.is_empty());
    if !any_needed {
        for (c, &fit) in cands.iter_mut().zip(fits) {
            if fit {
                c.precisions = wants.side.then(Vec::new);
                c.bin_precisions = wants.bin.then(Vec::new);
                c.entropies = wants.ent.then(Vec::new);
            }
        }
        return Ok(());
    }
    let cols: usize = layout.iter().map(|l| l.0).sum();
    let ch = dfl_stats_channels(wants.moments(), wants.ent);
    let m = stats.to_vec::<f32>()?;
    for (i, (c, &fit)) in cands.iter_mut().zip(fits).enumerate() {
        if !fit {
            continue;
        }
        // Global anchor -> (column, stride) of every slotted candidate, read straight from packed_u32.
        let mut at = HashMap::with_capacity(c.cands.len());
        let mut col0 = 0;
        let mut off = i * total_words + rec_words;
        for &(cap, stride, start) in layout {
            let row = &packed_u32[off..off + 1 + cap];
            for (k, &px) in row[1..].iter().take((row[0] as usize).min(cap)).enumerate() {
                at.insert(start as u32 + px, (col0 + k, stride));
            }
            col0 += cap;
            off += 1 + cap;
        }
        let img = &m[i * ch * cols..(i + 1) * ch * cols];
        let mut prec = wants.side.then(|| Vec::with_capacity(c.cands.len()));
        let mut bprec = wants.bin.then(|| Vec::with_capacity(c.cands.len()));
        let mut ent = wants.ent.then(|| Vec::with_capacity(c.cands.len()));
        for cand in &c.cands {
            let &(j, stride) = at.get(&cand.anchor).ok_or_else(|| {
                folo_nn::Error::Invalid(format!("candidate anchor {} is in no site list", cand.anchor))
            })?;
            if let Some(p) = prec.as_mut() {
                p.push(precision_from_moments(img, cols, j, stride));
            }
            if let Some(bp) = bprec.as_mut() {
                bp.push(bin_precision_from_moments(img, cols, j));
            }
            if let Some(ev) = ent.as_mut() {
                ev.push(entropy_from_stats(img, cols, j, wants.moments()));
            }
        }
        c.precisions = prec;
        c.bin_precisions = bprec;
        c.entropies = ent;
    }
    Ok(())
}
