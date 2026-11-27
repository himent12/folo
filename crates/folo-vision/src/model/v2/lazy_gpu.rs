//! Candidate-first prediction on a GPU for [`FoloV2Model`] (`LazyAlignHead`, docs/MODEL.md §9.5):
//! per level the class branch (`ClsBranchV2` for `o2m`, or `O2oClsLevel` with `PeakSuppress` for `o2o`) runs
//! densely, a class pass lists above-threshold sites and their `3×3` receptive fields on the device, the fused
//! `BoxBranchV2` (`2 × WS-QARepConv3x3` + `1×1 out`) runs only at surviving sites via [`Tensor::conv2d_act_sites`],
//! and one decode emits the candidate record buffer (plus optional DFL stats) in a single recorded
//! [`folo_gpu::ForwardPlans`] replay when plans are enabled.

use folo_core::{ClassSites, DType, Device, Tensor, no_grad};
use folo_nn::{Activation, Conv2d, ConvBnAct, Error, Forward, Result};

use super::blocks::{FoloBlock, WsQARepConv3x3};
use super::head::ClsBranchV2;
use super::{FoloV2Model, HeadMode};
use crate::model::lazy_gpu::{GPU_SITE_SLOTS, device_bins, read_sites_for_strides};
use crate::model::predict::{DflWants, dfl_stats_with};
use crate::model::{HeadOutput, ImageCandidates, each_tensor};
use crate::nms::NmsConfig;

/// FNV-1a mixer for non-tensor mutable layer settings in [`FoloV2Model::plan_signature`].
struct Sig(u64);

impl Sig {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn u64(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(0x0100_0000_01b3);
    }

    fn act(&mut self, a: Activation) {
        let (tag, extra) = match a {
            Activation::Identity => (0, 0),
            Activation::Silu => (1, 0),
            Activation::Relu => (2, 0),
            Activation::LeakyRelu(s) => (3, s.to_bits() as u64),
            Activation::Sigmoid => (4, 0),
            Activation::Tanh => (5, 0),
            Activation::Hardswish => (6, 0),
        };
        self.u64(tag);
        self.u64(extra);
    }

    fn conv2d(&mut self, c: &Conv2d) {
        let cfg = c.config();
        self.u64(cfg.in_channels as u64);
        self.u64(cfg.out_channels as u64);
        self.u64(cfg.kernel as u64);
        self.u64(cfg.stride as u64);
        self.u64(cfg.padding as u64);
        self.u64(cfg.dilation as u64);
        self.u64(cfg.groups as u64);
        self.u64(cfg.bias as u64);
    }

    fn conv_bn_act(&mut self, c: &ConvBnAct) {
        self.act(c.act);
        self.u64(c.bn.is_some() as u64);
        self.conv2d(&c.conv);
    }

    fn ws_qarep(&mut self, r: &WsQARepConv3x3) {
        self.act(r.act);
        self.u64(r.in_channels as u64);
        self.u64(r.out_channels as u64);
        self.u64(r.bn.is_some() as u64);
        self.u64(r.branches.is_some() as u64);
        self.u64(r.fused.is_some() as u64);
        if let Some(f) = &r.fused {
            self.conv2d(f);
        }
    }

    fn folo_block(&mut self, b: &FoloBlock) {
        self.conv_bn_act(&b.cv1);
        self.u64(b.c_mid as u64);
        self.u64(b.reps.len() as u64);
        for r in b.reps.iter() {
            self.ws_qarep(r);
        }
        self.conv_bn_act(&b.cv2);
    }

    fn cls_branch(&mut self, c: &ClsBranchV2) {
        self.u64(c.align.is_some() as u64);
        if let Some(a) = &c.align {
            self.conv_bn_act(a);
        }
        self.ws_qarep(&c.conv0);
        self.conv2d(&c.out);
    }
}

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
    /// Hash of every mutable non-tensor kernel setting on this model (activations, conv configs, `stem.spd`,
    /// `sppf.k`, `hyper` dims, `pad_p3`/`pad_p4` presence, head strides/flags) mixed into GPU
    /// [`folo_gpu::ForwardPlans`] keys so mutating public layer fields never replays stale dispatches.
    pub(super) fn plan_signature(&self) -> u64 {
        let mut s = Sig::new();
        let b = &self.backbone;
        s.u64(b.stem.spd as u64);
        s.conv_bn_act(&b.stem.conv);
        s.conv_bn_act(&b.down2);
        s.u64(b.stage2.len() as u64);
        for r in b.stage2.iter() {
            s.ws_qarep(r);
        }
        s.conv_bn_act(&b.down3);
        s.folo_block(&b.stage3);
        s.conv_bn_act(&b.down4);
        s.folo_block(&b.stage4);
        s.conv_bn_act(&b.down5);
        s.folo_block(&b.stage5);
        s.conv_bn_act(&b.sppf.cv1);
        s.conv_bn_act(&b.sppf.cv2);
        s.u64(b.sppf.k as u64);
        s.conv_bn_act(&b.hyper.av_proj);
        s.conv_bn_act(&b.hyper.out_proj);
        s.u64(b.hyper.m_edges as u64);
        s.u64(b.hyper.c_g as u64);
        let n = &self.neck;
        s.folo_block(&n.td4);
        s.conv_bn_act(&n.p2_proj);
        s.folo_block(&n.out3);
        s.conv_bn_act(&n.down3);
        s.folo_block(&n.out4);
        s.conv_bn_act(&n.down4);
        s.folo_block(&n.out5);
        for g in [&n.pad_p3, &n.pad_p4] {
            s.u64(g.is_some() as u64);
            if let Some(g) = g {
                s.u64(g.c_tgt as u64);
            }
        }
        let h = &self.head;
        s.u64(h.reg_max as u64);
        s.u64(h.num_classes as u64);
        s.u64(h.has_iou_head as u64);
        s.u64(h.o2o_detach as u64);
        s.u64(h.levels.len() as u64);
        for lv in &h.levels {
            s.u64(lv.stride as u64);
            s.ws_qarep(&lv.box_branch.conv0);
            s.ws_qarep(&lv.box_branch.conv1);
            s.conv2d(&lv.box_branch.out);
            s.u64(lv.box_branch.iou_out.is_some() as u64);
            if let Some(iou) = &lv.box_branch.iou_out {
                s.conv2d(iou);
            }
            s.cls_branch(&lv.cls_branch);
        }
        s.u64(h.o2o_cls.is_some() as u64);
        if let Some(o2o) = &h.o2o_cls {
            s.u64(o2o.len() as u64);
            for ol in o2o {
                s.cls_branch(&ol.cls_branch);
                s.conv2d(&ol.suppress.gate);
                s.u64(ol.suppress.num_classes as u64);
            }
        }
        s.0
    }

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
        let sig = self.plan_signature();
        let key = [SITES_PLAN, o2o as u64, stats_bits, cand_conf.to_bits() as u64, s3, s4, s5, sig];
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
        if let Some(cands) = out.candidates_on_device(nms)? {
            return Ok(cands);
        }
        // Host fallback for non-CPU backends that implement dense inference without candidate kernels.
        let host = HeadOutput {
            box_raw: out.box_raw.to_device(Device::Cpu)?.to_dtype(DType::F32)?,
            cls_raw: out.cls_raw.to_device(Device::Cpu)?.to_dtype(DType::F32)?,
            ..out
        };
        host.candidates_on_cpu(nms)?
            .ok_or_else(|| Error::Invalid(format!("candidate fallback failed on {:?}", x.device())))
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
