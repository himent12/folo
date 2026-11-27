//! CPU candidate-first prediction for [`FoloV2Model`] (docs/MODEL.md §9.5): the `LazyAlignHead` class branches
//! (`ClsBranchV2` or `O2oClsLevel` with `PeakSuppress`) run densely, surviving candidates are extracted straight
//! from each level's class map, and the fused `BoxBranchV2` (`2 × WS-QARepConv3x3` + `1×1 out`) runs only at
//! surviving candidate sites via [`Tensor::conv2d_act_at_tiles`] / [`Tensor::conv2d_act_at`].

use folo_core::{DType, Device, Tensor, no_grad, with_no_grad};
use folo_nn::{Error, Forward, Module, Result};
use rayon::prelude::*;

use super::blocks::WsQARepConv3x3;
use super::head::{HeadLevelV2, O2oClsLevel};
use super::{FoloV2Model, HeadMode};
use crate::model::ImageCandidates;
use crate::model::lazy::{level_candidates, needing_boxes};
use crate::model::predict::{DflWants, Found, dfl_box, finish_candidates};
use crate::nms::{Candidate, Detection, DflInputs, NmsConfig};

/// Run a fused [`WsQARepConv3x3`] at output sites `at` via [`Tensor::conv2d_act_at_tiles`].
fn rep_conv_at(rep: &WsQARepConv3x3, x: &Tensor, at: &Tensor) -> Result<Tensor> {
    let conv = rep
        .fused
        .as_ref()
        .ok_or_else(|| Error::Invalid("FoloV2Model candidate-first prediction needs a fused model".into()))?;
    let p = conv.config().params();
    Ok(x.conv2d_act_at_tiles(&conv.weight, conv.bias.as_ref(), &p, rep.act.unary_op(), at)?)
}

impl FoloV2Model {
    /// Whether [`FoloV2Model::predict_candidates`] serves an input on `device` with `nms`: a fused eval-mode
    /// model, single-label, without `iou_head` (which fuses per-anchor localization quality logits with class
    /// probabilities before thresholding), on CPU (`F32`) or a GPU (`F32` / `F16`, excluding DX12 software
    /// adapters where WARP crashes in site kernels).
    pub fn candidates_first_applies(&self, device: Device, nms: &NmsConfig) -> bool {
        self.fused
            && !self.is_training()
            && !nms.multi_label
            && !self.head.has_iou_head
            && !crate::model::lazy_gpu::is_dx12_software_adapter(device)
            && ((device.is_cpu() && self.dtype() == DType::F32) || !device.is_cpu())
    }

    /// Candidate-first prediction of `[B, in_channels, H, W]` with [`FoloV2Model::head_mode`]: bit-for-bit
    /// identical to the dense [`super::HeadOutputV2::detections`] path on both CPU and GPU while skipping
    /// `BoxBranchV2` on non-candidate anchors.
    pub fn predict_candidates(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<ImageCandidates>> {
        if !self.candidates_first_applies(x.device(), nms) {
            return Err(Error::Invalid(
                "FoloV2Model::predict_candidates needs a fused eval-mode model (CPU F32 or GPU), single-label, without iou_head"
                    .into(),
            ));
        }
        if !x.device().is_cpu() {
            return self.predict_candidates_gpu(x, nms);
        }
        folo_core::cpu::in_pool(|| {
            let _g = no_grad();
            let _ = self.check_input(x)?;
            let feats = self.forward_features(x)?;
            self.predict_candidates_from_features(feats, x.dims()[0], nms)
        })
    }

    /// Candidate-first prediction reading a pre-packed `[b, h, w]` `u32` `[R, G, B, 0]` canvas straight into the
    /// int8 `u8` stem (`SPD(2×2) + 1×1` or `3×3 s=2`, zero `f32` image materialisation).
    pub(crate) fn predict_candidates_rgb0_u8(
        &self,
        rgb0: &[u32],
        b: usize,
        h: usize,
        w: usize,
        nms: &NmsConfig,
    ) -> Result<Vec<ImageCandidates>> {
        if !self.candidates_first_applies(Device::Cpu, nms) {
            return Err(Error::Invalid(
                "FoloV2Model::predict_candidates needs a fused CPU F32 eval-mode model, single-label, without iou_head"
                    .into(),
            ));
        }
        let plan = self
            .active_int8()
            .ok_or_else(|| Error::Invalid("predict_candidates_rgb0_u8 requires an int8 plan".into()))?;
        folo_core::cpu::in_pool(|| {
            let _g = no_grad();
            self.check_hw(b, h, w)?;
            let feats = crate::model::quant::exec::features_v2_rgb0_u8(self, plan, rgb0, b, h, w)?;
            self.predict_candidates_from_features(feats, b, nms)
        })
    }

    fn predict_candidates_from_features(
        &self,
        feats: [Tensor; 3],
        b: usize,
        nms: &NmsConfig,
    ) -> Result<Vec<ImageCandidates>> {
        let mode = self.head_mode;
        let wants = self.dfl_wants(nms);
        let want_strides = nms.score.wants_strides() && mode == HeadMode::O2m;
        let cand_conf = nms.candidate_conf_for(mode);
        let hw = [
            feats[0].dims()[2] * feats[0].dims()[3],
            feats[1].dims()[2] * feats[1].dims()[3],
            feats[2].dims()[2] * feats[2].dims()[3],
        ];
        let offsets = [0, hw[0], hw[0] + hw[1]];

        let o2o_levels = if mode == HeadMode::O2o { self.head.o2o_cls.as_deref() } else { None };
        let found: Vec<Vec<Vec<Found>>> = (0..3)
            .into_par_iter()
            .map(|l| {
                with_no_grad(|| match o2o_levels {
                    Some(o2o) => self.class_candidates_o2o(&o2o[l], &feats[l], cand_conf),
                    None => self.class_candidates_o2m(&self.head.levels[l], &feats[l], cand_conf),
                })
            })
            .collect::<Result<_>>()?;

        let mut need: [Vec<Vec<Found>>; 3] = Default::default();
        let mut found: Vec<_> = found.into_iter().map(Vec::into_iter).collect();
        for _ in 0..b {
            let per_level = [0, 1, 2].map(|l| found[l].next().unwrap_or_default());
            for (l, v) in needing_boxes(per_level, offsets, mode, nms).into_iter().enumerate() {
                need[l].push(v);
            }
        }

        if need.iter().all(|lv| lv.iter().all(Vec::is_empty)) {
            let empty = ImageCandidates {
                cands: Vec::new(),
                precisions: wants.side.then(Vec::new),
                bin_precisions: wants.bin.then(Vec::new),
                entropies: wants.ent.then(Vec::new),
                strides: want_strides.then(Vec::new),
            };
            return Ok(vec![empty; b]);
        }

        let boxed: Vec<Vec<ImageCandidates>> = (0..3)
            .into_par_iter()
            .map(|l| {
                with_no_grad(|| {
                    self.boxes_at(&self.head.levels[l], &feats[l], &need[l], offsets[l], wants, want_strides)
                })
            })
            .collect::<Result<_>>()?;

        Ok((0..b)
            .map(|i| ImageCandidates {
                cands: boxed.iter().flat_map(|lv| lv[i].cands.iter().copied()).collect(),
                precisions: wants.side.then(|| {
                    boxed.iter().flat_map(|lv| lv[i].precisions.iter().flatten().copied()).collect()
                }),
                bin_precisions: wants.bin.then(|| {
                    boxed.iter().flat_map(|lv| lv[i].bin_precisions.iter().flatten().copied()).collect()
                }),
                entropies: wants
                    .ent
                    .then(|| boxed.iter().flat_map(|lv| lv[i].entropies.iter().flatten().copied()).collect()),
                strides: want_strides
                    .then(|| boxed.iter().flat_map(|lv| lv[i].strides.iter().flatten().copied()).collect()),
            })
            .collect())
    }

    /// Post-process one image's candidates from [`FoloV2Model::predict_candidates`] with [`FoloV2Model::head_mode`].
    pub fn finish_candidates(&self, c: &ImageCandidates, nms: &NmsConfig) -> Vec<Detection> {
        let dfl = DflInputs {
            precision: c.precisions.as_deref(),
            bin_precision: c.bin_precisions.as_deref(),
            entropy: c.entropies.as_deref(),
            strides: c.strides.as_deref(),
        };
        finish_candidates(self.head_mode, &c.cands, dfl, nms)
    }

    pub(super) fn dfl_wants(&self, nms: &NmsConfig) -> DflWants {
        if self.head_mode != HeadMode::O2m {
            return DflWants::default();
        }
        DflWants {
            side: nms.fusion.wants_side_precision(),
            bin: nms.score.wants_bin_precisions(),
            ent: nms.score.wants_dfl_entropy(),
        }
    }

    pub(super) fn wants_strides(&self, nms: &NmsConfig) -> bool {
        nms.score.wants_strides() && self.head_mode == HeadMode::O2m
    }

    fn extract_cls_candidates(&self, cls: &Tensor, b: usize, a: usize, conf: f32) -> Result<Vec<Vec<Found>>> {
        let nc = self.head.num_classes;
        if a == 0 || nc == 0 {
            return Ok(vec![Vec::new(); b]);
        }
        Ok(cls.with_f32_slice(|cl| {
            (0..b)
                .into_par_iter()
                .map(|i| level_candidates(&cl[i * nc * a..(i + 1) * nc * a], a, conf))
                .collect()
        })?)
    }

    fn class_candidates_o2m(&self, level: &HeadLevelV2, x: &Tensor, conf: f32) -> Result<Vec<Vec<Found>>> {
        let (b, _, h, w) = x.dims4()?;
        let cls = level.cls_branch.forward(x)?;
        self.extract_cls_candidates(&cls, b, h * w, conf)
    }

    fn class_candidates_o2o(&self, level: &O2oClsLevel, x: &Tensor, conf: f32) -> Result<Vec<Vec<Found>>> {
        let (b, _, h, w) = x.dims4()?;
        let (logits, feat) = level.cls_branch.forward_with_feat(x)?;
        let cls = level.suppress.apply(&logits, &feat)?;
        self.extract_cls_candidates(&cls, b, h * w, conf)
    }

    fn boxes_at(
        &self,
        level: &HeadLevelV2,
        x: &Tensor,
        need: &[Vec<Found>],
        offset: usize,
        wants: DflWants,
        want_strides: bool,
    ) -> Result<Vec<ImageCandidates>> {
        let (b, _, h, w) = x.dims4()?;
        let (hw, rm) = (h * w, self.head.reg_max);
        let st = level.stride as f32;
        if need.iter().all(Vec::is_empty) {
            let empty = ImageCandidates {
                cands: Vec::new(),
                precisions: wants.side.then(Vec::new),
                bin_precisions: wants.bin.then(Vec::new),
                entropies: wants.ent.then(Vec::new),
                strides: want_strides.then(Vec::new),
            };
            return Ok(vec![empty; b]);
        }
        let bx = &level.box_branch;
        let wino4 = h >= 24 && w >= 24;
        let mut sites = Vec::new();
        let mut field = vec![false; b * hw];
        for (i, v) in need.iter().enumerate() {
            for &(j, _, _) in v {
                let j = j as usize;
                sites.push((i * hw + j) as u32);
                let (y0, x0) = (j / w, j % w);
                // `conv1` is a 3×3 s=1 p=1 conv that may dispatch Winograd F(4×4) (reading a tile-aligned
                // 6×6 patch when `h >= 24 && w >= 24`) or Winograd F(2×2) (reading a tile-aligned 4×4 patch
                // for smaller maps, which also strictly contains the 3×3 direct receptive field).
                let (y_lo, y_hi, x_lo, x_hi) = if wino4 {
                    let (ty, tx) = (y0 / 4, x0 / 4);
                    (
                        (4 * ty).saturating_sub(1),
                        (4 * ty + 5).min(h),
                        (4 * tx).saturating_sub(1),
                        (4 * tx + 5).min(w),
                    )
                } else {
                    let (ty, tx) = (y0 / 2, x0 / 2);
                    (
                        (2 * ty).saturating_sub(1),
                        (2 * ty + 3).min(h),
                        (2 * tx).saturating_sub(1),
                        (2 * tx + 3).min(w),
                    )
                };
                for y in y_lo..y_hi {
                    field[i * hw + y * w + x_lo..i * hw + y * w + x_hi].fill(true);
                }
            }
        }
        let field: Vec<u32> = (0..b * hw).filter(|&p| field[p]).map(|p| p as u32).collect();
        let (n1, n0) = (sites.len(), field.len());
        // When >= 2/3 of `conv0`'s receptive field is active, dense `BoxBranchV2` avoids sparse-tile overhead
        // while staying bit-identical to the dense forward path.
        let raw = if n0 * 3 >= 2 * b * hw {
            bx.forward_box_only(x)?
        } else {
            let (sites, field) = (Tensor::from_vec(sites, [n1])?, Tensor::from_vec(field, [n0])?);
            let h0 = rep_conv_at(&bx.conv0, x, &field)?;
            let h1 = rep_conv_at(&bx.conv1, &h0, &sites)?;
            drop(h0);
            let p = bx.out.config().params();
            h1.conv2d_act_at(&bx.out.weight, bx.out.bias.as_ref(), &p, None, &sites)?
        };
        let per = 4 * rm * hw;
        Ok(raw.with_f32_slice(|rg| {
            need.iter()
                .enumerate()
                .map(|(i, v)| {
                    let rg = &rg[i * per..(i + 1) * per];
                    let mut bins = vec![0f32; rm];
                    let mut out = ImageCandidates {
                        cands: Vec::with_capacity(v.len()),
                        precisions: wants.side.then(|| Vec::with_capacity(v.len())),
                        bin_precisions: wants.bin.then(|| Vec::with_capacity(v.len())),
                        entropies: wants.ent.then(|| Vec::with_capacity(v.len())),
                        strides: want_strides.then(|| Vec::with_capacity(v.len())),
                    };
                    for &(j, class, score) in v {
                        let (j, k) = (j as usize, offset + j as usize);
                        let point = [(j % w) as f32 + 0.5, (j / w) as f32 + 0.5];
                        let (bbox, p, bp, ent) = dfl_box(&mut bins, |r| rg[r * hw + j], point, st, wants);
                        out.cands.push(Candidate { anchor: k as u32, class, score, bbox });
                        out.precisions.iter_mut().zip(p).for_each(|(v, p)| v.push(p));
                        out.bin_precisions.iter_mut().zip(bp).for_each(|(v, p)| v.push(p));
                        out.entropies.iter_mut().zip(ent).for_each(|(v, e)| v.push(e));
                        if let Some(sv) = out.strides.as_mut() {
                            sv.push(st);
                        }
                    }
                    out
                })
                .collect()
        })?)
    }
}
