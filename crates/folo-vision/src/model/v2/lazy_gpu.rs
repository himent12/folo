//! Candidate-first prediction on a GPU for [`FoloV2Model`] (`LazyAlignHead`, docs/MODEL.md §9.5):
//! per level the class branch (`ClsBranchV2` for `o2m`, or `O2oClsLevel` with `PeakSuppress` for `o2o`) runs
//! densely, a class pass lists above-threshold sites and their `3×3` receptive fields on the device, the fused
//! `BoxBranchV2` (`2 × WS-QARepConv3x3` + `1×1 out`) runs only at surviving sites via [`Tensor::conv2d_act_sites`],
//! and one decode emits the candidate record buffer (plus optional DFL stats) in a single recorded
//! [`folo_gpu::ForwardPlans`] replay when plans are enabled.

use folo_core::{ClassSites, Tensor, no_grad};
use folo_nn::{Error, Forward, Result};

use super::blocks::WsQARepConv3x3;
use super::{FoloV2Model, HeadMode};
use crate::model::lazy_gpu::{GPU_SITE_SLOTS, device_bins, read_sites_for_strides};
use crate::model::predict::{DflWants, dfl_stats_with};
use crate::model::{HeadOutput, ImageCandidates, each_tensor};
use crate::nms::NmsConfig;

/// One level's site outputs for GPU candidate decode:
/// `(box_raw [B, 4·reg_max, cap], sites [B, 1 + cap], class_records, map_w, stride, level_anchor_offset)`.
type SitePart = (Tensor, Tensor, Tensor, usize, f32, usize);

/// Tags the GPU site-plan key apart from dense [`FoloV2Model::forward_planned`] (`0`).
const SITES_PLAN: u64 = 1;

/// Run a fused [`WsQARepConv3x3`] at `sites` on a GPU (`None` when the device has no site kernel for it).
fn rep_conv_sites(rep: &WsQARepConv3x3, x: &Tensor, sites: &Tensor, to_map: bool) -> Result<Option<Tensor>> {
    let conv = rep.fused.as_ref().ok_or_else(|| {
        Error::Invalid("FoloV2Model GPU candidate-first prediction needs a fused model".into())
    })?;
    let p = conv.config().params();
    Ok(x.conv2d_act_sites(&conv.weight, conv.bias.as_ref(), &p, rep.act.unary_op(), sites, to_map)?)
}

impl FoloV2Model {
    /// GPU candidate-first prediction for [`FoloV2Model`]: evaluates `BoxBranchV2` only at above-threshold
    /// class sites (recorded into [`Self::plans`] when enabled), falling back to the dense GPU head for any
    /// image that overflows [`GPU_SITE_SLOTS`] or on devices lacking site kernels.
    pub(super) fn predict_candidates_gpu(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<ImageCandidates>> {
        let _g = no_grad();
        let (h, w) = self.check_input(x)?;
        let o2o = self.head_mode == HeadMode::O2o && self.head.o2o_cls.is_some();
        let wants = self.dfl_wants(nms);
        let want_strides = self.wants_strides(nms);
        let cand_conf = nms.candidate_conf_for(self.head_mode);
        let bins = wants.moments().then(|| device_bins(x.device(), self.head.reg_max)).transpose()?;
        let tensors = |f: &mut dyn FnMut(&Tensor)| {
            each_tensor(self, f);
            bins.iter().for_each(&mut *f);
        };
        let [s3, s4, s5] = GPU_SITE_SLOTS.map(|s| s as u64);
        let stats_bits = (wants.moments() as u64) | ((wants.ent as u64) << 1);
        let key = [SITES_PLAN, o2o as u64, stats_bits, cand_conf.to_bits() as u64, s3, s4, s5];
        let mut unsited = None;
        let out = self.plans.run(&key, x, &tensors, |x| {
            let feats = self.forward_features(x)?;
            let out = self.sites_on_device(&feats, o2o, cand_conf, wants, bins.as_ref())?;
            if out.is_none() {
                unsited = Some(feats);
            }
            Ok::<_, Error>(out.unwrap_or_default())
        })?;
        if out.is_empty() {
            return self.dense_candidates_gpu(x, unsited, nms, h, w);
        }
        let (cands, fits) = read_sites_for_strides(&out, self.strides(), h, w, wants, want_strides)?;
        if fits.iter().all(|&f| f) {
            return Ok(cands);
        }
        // Any image whose candidates overflowed a level's fixed slot budget falls back to the dense head.
        let dense = self.dense_candidates_gpu(x, None, nms, h, w)?;
        Ok(cands.into_iter().zip(fits).zip(dense).map(|((c, f), d)| if f { c } else { d }).collect())
    }

    fn dense_candidates_gpu(
        &self,
        x: &Tensor,
        feats: Option<[Tensor; 3]>,
        nms: &NmsConfig,
        h: usize,
        w: usize,
    ) -> Result<Vec<ImageCandidates>> {
        let feats = match feats {
            Some(f) => f,
            None => self.forward_features(x)?,
        };
        let feats = [&feats[0], &feats[1], &feats[2]];
        let mode = self.head_mode;
        let (box_raw, cls_raw) = match mode {
            HeadMode::O2o => {
                let (b, c, _) = self.head.forward_o2o(feats, None)?.ok_or_else(|| {
                    Error::Invalid(format!("{} has no one-to-one head branch", self.arch()))
                })?;
                (b, c)
            }
            HeadMode::O2m => {
                let (b, c, _) = self.head.forward(feats)?;
                (b, c)
            }
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
            .ok_or_else(|| Error::Invalid(format!("no candidate kernel on {:?}", x.device())))
    }

    fn sites_on_device(
        &self,
        feats: &[Tensor; 3],
        o2o: bool,
        conf: f32,
        wants: DflWants,
        bins: Option<&Tensor>,
    ) -> Result<Option<Vec<Tensor>>> {
        let (b, rm) = (feats[0].dims()[0], self.head.reg_max);
        let o2o_levels = o2o.then_some(self.head.o2o_cls.as_deref()).flatten();
        let mut parts: Vec<SitePart> = Vec::with_capacity(3);
        let mut start = 0usize;
        for (l, (lv, x)) in self.head.levels.iter().zip(feats).enumerate() {
            let (_, _, h, w) = x.dims4()?;
            let cap = GPU_SITE_SLOTS[l].min(h * w);
            let cls = match o2o_levels {
                Some(ol) => {
                    let (logits, feat) = ol[l].cls_branch.forward_with_feat(x)?;
                    ol[l].suppress.apply(&logits, &feat)?
                }
                None => lv.cls_branch.forward(x)?,
            };
            let Some((sites, recs, field)) = cls.detect_class_sites(&ClassSites { conf, cap })? else {
                return Ok(None);
            };
            let bx = &lv.box_branch;
            let Some(h0) = rep_conv_sites(&bx.conv0, x, &field, true)? else { return Ok(None) };
            let Some(h1) = rep_conv_sites(&bx.conv1, &h0, &sites, false)? else { return Ok(None) };
            drop(h0);
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
        let raws: Vec<&Tensor> = parts.iter().map(|p| &p.0).collect();
        let stats = dfl_stats_with(&Tensor::cat(&raws, 2)?, rm, bins, wants.moments(), wants.ent)?;
        let mut u32_parts: Vec<&Tensor> = Vec::with_capacity(1 + parts.len());
        u32_parts.push(&out);
        u32_parts.extend(parts.iter().map(|p| &p.1));
        let packed = Tensor::cat(&u32_parts, 1)?;
        Ok(Some(vec![packed, stats]))
    }
}
