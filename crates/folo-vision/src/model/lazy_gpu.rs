//! Candidate-first prediction on a GPU (docs/MODEL.md §9.4): per level the class branch runs densely, a class pass
//! lists the candidates (and the pixels their box branch reads) into fixed-capacity lists on the device, the box
//! branch runs at those sites only ([`Tensor::conv2d_act_sites`]: `conv0` on the candidates' 3×3 windows, written
//! back into a map; `conv1` and `out` at the candidates), and one decode appends every level's boxes to one record
//! list per image, read back once. The site convs are the dense convs' values bit for bit (same kernels, same
//! split-K chunks), so the records are those of the dense head's [`Tensor::detect_candidates`].
//!
//! The lists' capacities fix the dispatches, whatever an image's candidates; a level with more candidates than
//! slots in some image is flagged, and those images take their candidates from the dense head (a second round
//! trip on the same features).

use folo_core::{ClassSites, Tensor, no_grad};
use folo_nn::{ConvBnAct, Forward, Result};

use super::head::{HeadLevel, HeadMode};
use super::predict::candidates_of_records;
use super::{FoloModel, HeadOutput, ImageCandidates};
use crate::nms::{Candidate, NmsConfig};

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

impl FoloModel {
    /// [`FoloModel::predict_candidates`] on a GPU: the candidates at the sites, with the dense head serving the
    /// images whose candidates did not fit (and every image on a device without the site kernels).
    pub(super) fn predict_candidates_gpu(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<ImageCandidates>> {
        let _g = no_grad();
        let (h, w) = self.check_input(x)?;
        let feats = self.features(x, None)?;
        let mode = self.head_mode;
        let levels = match (mode, &self.head.o2o) {
            (HeadMode::O2o, Some(o2o)) => o2o,
            _ => &self.head.levels,
        };
        let sited = self.candidates_at_sites(levels, &feats, nms)?;
        if let Some((cands, fits)) = &sited
            && fits.iter().all(|&f| f)
        {
            return Ok(cands
                .iter()
                .map(|c| ImageCandidates { cands: c.clone(), precisions: None })
                .collect());
        }
        // The dense head on the same features, for the images that did not fit.
        let feats = [&feats[0], &feats[1], &feats[2]];
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
        let dense = out
            .candidates_on_device(nms)?
            .ok_or_else(|| folo_nn::Error::Invalid(format!("no candidate kernel on {:?}", x.device())))?;
        Ok(match sited {
            Some((cands, fits)) => {
                let sited = cands.into_iter().map(|cands| ImageCandidates { cands, precisions: None });
                sited.zip(fits).zip(dense).map(|((c, f), d)| if f { c } else { d }).collect()
            }
            None => dense,
        })
    }

    /// Every image's candidates in anchor order, and whether they fit the slots (`false`: some level had more);
    /// `None` when the device lacks a site kernel.
    #[allow(clippy::type_complexity)]
    fn candidates_at_sites(
        &self,
        levels: &[HeadLevel],
        feats: &[Tensor; 3],
        nms: &NmsConfig,
    ) -> Result<Option<(Vec<Vec<Candidate>>, Vec<bool>)>> {
        let (b, rm) = (feats[0].dims()[0], self.head.reg_max);
        let mut parts = Vec::with_capacity(3);
        let mut start = 0;
        for (l, (lv, x)) in levels.iter().zip(feats).enumerate() {
            let (_, _, h, w) = x.dims4()?;
            let cap = GPU_SITE_SLOTS[l].min(h * w);
            let cls = lv.cls_branch.forward(x)?;
            let Some((sites, recs, field)) = cls.detect_class_sites(&ClassSites { conf: nms.conf, cap })?
            else {
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
        let words = out.dims()[1];
        let v = out.to_vec::<u32>()?;
        let mut cands = Vec::with_capacity(b);
        let mut fits = Vec::with_capacity(b);
        for row in v.chunks_exact(words) {
            let n = (row[0] as usize).min((words - 2) / 7);
            cands.push(candidates_of_records(&row[2..2 + 7 * n]));
            fits.push(row[1] == 0);
        }
        Ok(Some((cands, fits)))
    }
}
