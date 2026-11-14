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
//! DFL box fusion needs each candidate's side precisions: the dense path's moment ops
//! ([`HeadOutput::moments_on_device`]) run on the candidates' box-logit columns, which hold the dense logits bit
//! for bit, so the precisions are the dense ones bit for bit (only then are the moments and the site lists read
//! back too).
//!
//! Everything from the input to the records (and the moments) is a pure function of the input with a fixed
//! dispatch list, so with the model's forward plans on ([`FoloModel::set_forward_plans`], as in a `Predictor`) the
//! whole device side is one recorded plan ([`folo_gpu::ForwardPlans`]): a replayed prediction is one flat list of
//! dispatches, then the readbacks (the records; with DFL fusion also the moments and the three site lists).

use std::collections::HashMap;
use std::sync::Mutex;

use folo_core::{ClassSites, Device, Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_nn::{ConvBnAct, Forward, Result};

use super::head::{HeadLevel, HeadMode};
use super::predict::{candidates_of_records, dfl_moments_with, moment_bins, precision_from_moments};
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
pub const GPU_SITE_SLOTS: [usize; 3] = [1024, 512, 256];

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
fn device_bins(dev: Device, rm: usize) -> Result<Tensor> {
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

impl FoloModel {
    /// [`FoloModel::predict_candidates`] on a GPU: the candidates at the sites, with the dense head serving the
    /// images whose candidates did not fit (and every image on a device without the site kernels).
    pub(super) fn predict_candidates_gpu(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<ImageCandidates>> {
        let _g = no_grad();
        let (h, w) = self.check_input(x)?;
        let levels = self.site_levels();
        let o2o = self.head_mode == HeadMode::O2o && self.head.o2o.is_some();
        let bins =
            self.wants_precisions(nms).then(|| device_bins(x.device(), self.head.reg_max)).transpose()?;
        let tensors = |f: &mut dyn FnMut(&Tensor)| {
            each_tensor(self, f);
            bins.iter().for_each(&mut *f);
        };
        // What reaches a kernel besides the input and the tensors: the branch, the class pass's threshold (a
        // parameter block), the slots, and whether the moments and site lists are results. The rest of `nms` is
        // the host's post-process.
        let [s3, s4, s5] = GPU_SITE_SLOTS.map(|s| s as u64);
        let key = [SITES_PLAN, o2o as u64, bins.is_some() as u64, nms.conf.to_bits() as u64, s3, s4, s5];
        let out = self.plans.run(&key, x, &tensors, |x| {
            let feats = self.features(x, None)?;
            Ok::<_, folo_nn::Error>(
                self.sites_on_device(levels, &feats, nms.conf, bins.as_ref())?.unwrap_or_default(),
            )
        })?;
        if out.is_empty() {
            // No site kernel on this device.
            return self.dense_candidates(x, nms, h, w);
        }
        let (cands, fits) = self.read_sites(&out, levels, h, w)?;
        if fits.iter().all(|&f| f) {
            return Ok(cands);
        }
        // The images that did not fit, from the dense head (the features again, unplanned).
        let dense = self.dense_candidates(x, nms, h, w)?;
        Ok(cands.into_iter().zip(fits).zip(dense).map(|((c, f), d)| if f { c } else { d }).collect())
    }

    /// The head levels of the [`FoloModel::head_mode`] branch.
    fn site_levels(&self) -> &[HeadLevel] {
        match (self.head_mode, &self.head.o2o) {
            (HeadMode::O2o, Some(o2o)) => o2o,
            _ => &self.head.levels,
        }
    }

    /// Every image's candidates from the dense head ([`HeadOutput::candidates_on_device`]) on freshly computed
    /// features.
    fn dense_candidates(
        &self,
        x: &Tensor,
        nms: &NmsConfig,
        h: usize,
        w: usize,
    ) -> Result<Vec<ImageCandidates>> {
        let feats = self.features(x, None)?;
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
    /// [`Tensor::detect_decode_sites`], and with the moment bins `bins` (DFL fusion) the moments of the levels'
    /// box-logit columns side by side and every level's site list. `None` when the device lacks a site kernel.
    fn sites_on_device(
        &self,
        levels: &[HeadLevel],
        feats: &[Tensor; 3],
        conf: f32,
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
        let mut results = vec![out];
        if let Some(k) = bins {
            // `Σ cap ≥ 3` columns, so the column kernels the dense `A` takes.
            let raws: Vec<&Tensor> = parts.iter().map(|p| &p.0).collect();
            results.push(dfl_moments_with(&Tensor::cat(&raws, 2)?, rm, k)?);
            results.extend(parts.into_iter().map(|p| p.1));
        }
        Ok(Some(results))
    }

    /// Every image's candidates in anchor order from [`FoloModel::sites_on_device`]'s results (with the moments,
    /// also their side precisions), and whether they fit the slots (`false`: some level had more; that image's
    /// candidates are partial and carry no precisions). `h × w` is the input.
    fn read_sites(
        &self,
        out: &[Tensor],
        levels: &[HeadLevel],
        h: usize,
        w: usize,
    ) -> Result<(Vec<ImageCandidates>, Vec<bool>)> {
        let words = out[0].dims()[1];
        let v = out[0].to_vec::<u32>()?;
        let mut cands = Vec::with_capacity(out[0].dims()[0]);
        let mut fits = Vec::with_capacity(cands.capacity());
        for row in v.chunks_exact(words) {
            let n = (row[0] as usize).min((words - 2) / 7);
            cands
                .push(ImageCandidates { cands: candidates_of_records(&row[2..2 + 7 * n]), precisions: None });
            fits.push(row[1] == 0);
        }
        if let [_, moments, sites @ ..] = out {
            // Each level's (cap, stride, first anchor): its map is the input over its stride.
            let mut start = 0;
            let layout: Vec<(usize, f32, usize)> = levels
                .iter()
                .zip(sites)
                .map(|(lv, s)| {
                    let at = start;
                    let stride = lv.stride as usize;
                    start += (h / stride) * (w / stride);
                    (s.dims()[1] - 1, lv.stride as f32, at)
                })
                .collect();
            self.site_precisions(moments, sites, &layout, &mut cands, &fits)?;
        }
        Ok((cands, fits))
    }

    /// DFL fusion's side precisions of the fitting images' candidates: [`precision_from_moments`] at each
    /// candidate's column of `moments` ([`super::predict::dfl_moments`] of the levels' box-logit columns side by
    /// side; `layout` holds each level's cap, stride and first anchor). A candidate's column holds its anchor's
    /// dense box logits bit for bit, so its precisions are the dense path's bit for bit.
    fn site_precisions(
        &self,
        moments: &Tensor,
        site_lists: &[Tensor],
        layout: &[(usize, f32, usize)],
        cands: &mut [ImageCandidates],
        fits: &[bool],
    ) -> Result<()> {
        let cols: usize = layout.iter().map(|l| l.0).sum();
        let m = moments.to_vec::<f32>()?;
        let mut sites = Vec::with_capacity(site_lists.len());
        for s in site_lists {
            sites.push(s.to_vec::<u32>()?);
        }
        for (i, (c, &fit)) in cands.iter_mut().zip(fits).enumerate() {
            if !fit {
                continue;
            }
            // Global anchor -> (column, stride) of every slotted candidate.
            let mut at = HashMap::new();
            let mut col0 = 0;
            for (&(cap, stride, start), s) in layout.iter().zip(&sites) {
                let row = &s[i * (1 + cap)..(i + 1) * (1 + cap)];
                for (k, &px) in row[1..].iter().take((row[0] as usize).min(cap)).enumerate() {
                    at.insert(start as u32 + px, (col0 + k, stride));
                }
                col0 += cap;
            }
            let img = &m[i * 8 * cols..(i + 1) * 8 * cols];
            let prec = c
                .cands
                .iter()
                .map(|c| {
                    let &(j, stride) = at.get(&c.anchor).ok_or_else(|| {
                        folo_nn::Error::Invalid(format!("candidate anchor {} is in no site list", c.anchor))
                    })?;
                    Ok(precision_from_moments(img, cols, j, stride))
                })
                .collect::<Result<_>>()?;
            c.precisions = Some(prec);
        }
        Ok(())
    }
}
