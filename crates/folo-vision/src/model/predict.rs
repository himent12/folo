//! Decoding and the raw prediction path (docs/MODEL.md §6.4, §9.3): forward, decode (on the CPU with
//! [`crate::decode`], or as tensor ops on the outputs' device), then NMS on the host.

use folo_core::backend::DETECT_RECORD_WORDS;
use folo_core::{DType, DetectDecode, Device, Tensor, no_grad};
use folo_nn::{Error, Module, Result};
use rayon::prelude::*;

use super::{FoloModel, HeadMode, HeadOutput, ImageCandidates};
use crate::boxes::Xyxy;
use crate::decode::{
    bin_precision, decode_head, dist2bbox, side_precision, side_precisions, sigmoid, softmax_expectation,
    softmax_moments,
};
use crate::fusion::BoxFusion;
use crate::nms::{
    Candidate, Detection, NmsConfig, nms_candidates_with_dfl, postprocess_with_dfl, topk_candidates,
};

/// Records per image in the first download of [`HeadOutput::detections`]' device path: the counts and up to
/// this many candidates of every image come back in one readback; an image with more takes a second one.
pub const CANDIDATE_READBACK: usize = 1024;

/// Anchors per task of the CPU candidate extraction ([`HeadOutput::detections`] on the CPU): 8400 anchors at
/// 640 px make nine blocks, and a block's running class maximum (4 KiB) stays in L1.
pub(super) const CPU_DECODE_BLOCK: usize = 1024;

/// An anchor that passes the single-label confidence test: `(anchor, best class, its score)`.
pub(super) type Found = (u32, u32, f32);

/// The single-label candidates among anchors `j0..j1` (at most [`CPU_DECODE_BLOCK`] of them) of one image's
/// channel-major class logits `cl` (`[nc, a]`), appended to `out` in anchor order: a running maximum of the
/// logits per anchor, then the full extraction of `non_max_suppression` (sigmoid of every class, first maximum,
/// `> conf`) only for anchors whose largest logit could pass. The sigmoid is monotonic, so a skipped anchor has no
/// score above `conf` (a NaN first score never passes either).
pub(super) fn single_label_block(
    cl: &[f32],
    a: usize,
    j0: usize,
    j1: usize,
    conf: f32,
    out: &mut Vec<Found>,
) {
    let nc = cl.len() / a;
    let mut best = [0f32; CPU_DECODE_BLOCK];
    let best = &mut best[..j1 - j0];
    best.copy_from_slice(&cl[j0..j1]);
    for row in cl.chunks_exact(a).skip(1) {
        for (m, &v) in best.iter_mut().zip(&row[j0..j1]) {
            *m = m.max(v);
        }
    }
    for (j, &m) in (j0..j1).zip(best.iter()) {
        // A margin of a few ulp, so even a libm `exp` that is not monotonic cannot skip a candidate; the exact
        // test follows.
        let may_pass = sigmoid(m) * (1.0 + 1e-6) + 1e-30 > conf;
        if !may_pass {
            continue;
        }
        // Exactly the extraction of `non_max_suppression`.
        let (mut bc, mut bs) = (0usize, sigmoid(cl[j]));
        for c in 1..nc {
            let s = sigmoid(cl[c * a + j]);
            if s > bs {
                (bc, bs) = (c, s);
            }
        }
        if bs > conf {
            out.push((j as u32, bc as u32, bs)); // never for NaN
        }
    }
}

/// The DFL decode of one anchor (MODEL.md §6.4) with the functions of [`crate::decode::decode_head`]: `read(i)`
/// is its raw box output `i` (`side · reg_max + bin`), `bins` is `reg_max` long scratch. With `precisions`, also
/// its side precisions for DFL evidence fusion ([`crate::decode::side_precision`] of each side's mean and
/// variance at `stride`), the box then from the same moments' means.
pub(super) fn dfl_box(
    bins: &mut [f32],
    read: impl Fn(usize) -> f32,
    point: [f32; 2],
    stride: f32,
    want_side: bool,
    want_bin: bool,
) -> (Xyxy, Option<[f32; 4]>, Option<[f32; 4]>) {
    let rm = bins.len();
    let (mut d, mut p, mut bp) = ([0f32; 4], [0f32; 4], [0f32; 4]);
    let moments = want_side || want_bin;
    for side in 0..4 {
        for (k, v) in bins.iter_mut().enumerate() {
            *v = read(side * rm + k);
        }
        if moments {
            // The mean is the expectation of `decode::softmax_expectation`, bit for bit.
            let (mean, var) = softmax_moments(bins);
            d[side] = mean;
            if want_side {
                p[side] = side_precision(mean, var, stride);
            }
            if want_bin {
                bp[side] = bin_precision(mean, var);
            }
        } else {
            d[side] = softmax_expectation(bins);
        }
    }
    (dist2bbox(d, point).scale(stride, stride), want_side.then_some(p), want_bin.then_some(bp))
}

/// The single-label post-process of one image's candidates (sorted by anchor) for the branch `mode`: NMS for
/// `o2m` (with the candidates' side and/or scale-free bin precisions, parallel to `cands`), §6.6's top-k for `o2o`.
pub(super) fn finish_candidates(
    mode: HeadMode,
    cands: &[Candidate],
    prec: Option<&[[f32; 4]]>,
    bin_prec: Option<&[[f32; 4]]>,
    nms: &NmsConfig,
) -> Vec<Detection> {
    match mode {
        HeadMode::O2m => nms_candidates_with_dfl(cands, prec, bin_prec, nms),
        HeadMode::O2o => topk_candidates(cands, nms),
    }
}

/// One image's decoded output, in network-input pixels (no letterbox inverse).
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePrediction {
    /// `[A]` decoded boxes, `xyxy`.
    pub boxes: Vec<Xyxy>,
    /// `[A, nc]` row-major class scores (post-sigmoid).
    pub scores: Vec<f32>,
    /// The post-processing output (NMS, or the top-k of an `o2o` head), sorted by descending score.
    pub detections: Vec<Detection>,
}

/// What [`FoloModel::predict_raw`] returns.
#[derive(Clone)]
pub struct RawPrediction {
    /// The raw head outputs.
    pub output: HeadOutput,
    /// Per image: decoded boxes and scores, and the detections after post-processing.
    pub images: Vec<ImagePrediction>,
}

/// `[B, C, A]` → host `[B, A, C]` (anchor-major, the layout [`crate::decode`] takes). `to_vec` walks the
/// transposed view in logical order, so this is one copy.
fn anchor_major(t: &Tensor) -> Result<Vec<f32>> {
    let t = if t.device() == Device::Cpu { t.clone() } else { t.to_device(Device::Cpu)? };
    Ok(t.to_dtype(DType::F32)?.transpose(1, 2)?.to_vec::<f32>()?)
}

impl HeadOutput {
    /// Decode every image (MODEL.md §6.4): `(boxes [A], scores [A, nc])` in input pixels. CPU outputs are
    /// decoded on the CPU; outputs on another device are decoded there ([`HeadOutput::decode_on_device`])
    /// and downloaded once.
    pub fn decode_images(&self) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        let _g = no_grad();
        if !self.box_raw.device().is_cpu() {
            return self.decode_on_device();
        }
        let (a, nc, rm) = (self.num_anchors(), self.num_classes, self.reg_max);
        let reg = anchor_major(&self.box_raw)?;
        let cls = anchor_major(&self.cls_raw)?;
        if reg.len() != self.batch() * a * 4 * rm || cls.len() != self.batch() * a * nc {
            return Err(Error::Invalid(format!(
                "HeadOutput: box_raw {:?} / cls_raw {:?} do not match {a} anchors",
                self.box_raw.dims(),
                self.cls_raw.dims()
            )));
        }
        Ok(reg
            .chunks_exact(a * 4 * rm)
            .zip(cls.chunks_exact(a * nc))
            .map(|(r, c)| decode_head(r, c, &self.anchors, rm, nc))
            .collect())
    }

    /// [`HeadOutput::decode_images`] as device tensor ops: the DFL softmax expectation, `ltrb` → `xyxy`
    /// around the anchor points times the stride, and the class sigmoid, assembled anchor-major as
    /// `[B, A, 4 + nc]` on the device so that only the decoded values are downloaded (about 60% of the raw
    /// head) and the host does no transpose. Same formulas as the CPU path (max-subtracted softmax). `F16`
    /// outputs (half-precision inference) are decoded in `F32`.
    pub fn decode_on_device(&self) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        let all = self.decoded_on_device()?.to_vec::<f32>()?;
        Ok(Self::split_decoded(&all, self.num_anchors(), self.num_classes))
    }

    /// The device half of [`HeadOutput::decode_on_device`]: the decoded `[B, A, 4 + nc]` tensor (a transposed
    /// view) on the outputs' device, not read back yet. Readers that cannot block (the browser) download it
    /// asynchronously and hand the values to [`HeadOutput::split_decoded`].
    pub fn decoded_on_device(&self) -> Result<Tensor> {
        let _g = no_grad();
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] || self.cls_raw.dims() != [b, nc, a] {
            return Err(Error::Invalid(format!(
                "HeadOutput: box_raw {:?} / cls_raw {:?} do not match {a} anchors",
                self.box_raw.dims(),
                self.cls_raw.dims()
            )));
        }
        let dev = self.box_raw.device();
        let (box_raw, cls_raw) = (self.box_raw.to_dtype(DType::F32)?, self.cls_raw.to_dtype(DType::F32)?);
        // One upload: anchor x, anchor y (grid units), stride, then the bin indices 0..reg_max.
        let mut consts = Vec::with_capacity(3 * a + rm);
        consts.extend(self.anchors.points.iter().map(|p| p[0]));
        consts.extend(self.anchors.points.iter().map(|p| p[1]));
        consts.extend_from_slice(&self.anchors.strides);
        consts.extend((0..rm).map(|i| i as f32));
        let consts = Tensor::from_vec(consts, [3 * a + rm])?.to_device(dev)?;
        let points = consts.narrow(0, 0, 2 * a)?.reshape([1, 2, a])?;
        let stride = consts.narrow(0, 2 * a, a)?.reshape([1, 1, a])?;
        let bins = consts.narrow(0, 3 * a, rm)?.reshape([1, 1, rm, 1])?;
        let dist = box_raw.reshape([b, 4, rm, a])?.softmax(2)?.mul(&bins)?.sum(2, false)?; // [B, 4, A]
        let xy1 = points.sub(&dist.narrow(1, 0, 2)?)?;
        let xy2 = points.add(&dist.narrow(1, 2, 2)?)?;
        let boxes = Tensor::cat(&[&xy1, &xy2], 1)?.mul(&stride)?;
        let scores = cls_raw.sigmoid()?;
        Ok(Tensor::cat(&[&boxes, &scores], 1)?.transpose(1, 2)?) // [B, A, 4 + nc]
    }

    /// Per image `(boxes [A], scores [A, nc])` from the host values of [`HeadOutput::decoded_on_device`]
    /// (`[B, A, 4 + nc]`, row-major).
    pub fn split_decoded(all: &[f32], a: usize, nc: usize) -> Vec<(Vec<Xyxy>, Vec<f32>)> {
        let row = 4 + nc;
        all.par_chunks_exact(a * row)
            .map(|img| {
                let boxes = img.chunks_exact(row).map(|r| Xyxy::new(r[0], r[1], r[2], r[3])).collect();
                let mut scores = Vec::with_capacity(a * nc);
                for r in img.chunks_exact(row) {
                    scores.extend_from_slice(&r[4..]);
                }
                (boxes, scores)
            })
            .collect()
    }

    /// The detections of every image (MODEL.md §6.4 decode, then §9.3 NMS with `nms`, or for one-to-one
    /// outputs, [`HeadOutput::mode`] `o2o`, §6.6's top-k with `nms.conf` and `nms.max_det` and no NMS), in
    /// network-input pixels, highest score first.
    ///
    /// On a device with the candidate kernel (the GPU) and without `multi_label`, the decode runs there in
    /// one kernel ([`Tensor::detect_candidates`], which reads `F16` outputs as they are and decodes in `F32`)
    /// and only the anchors whose best class scores `> conf` are
    /// downloaded: one readback of the counts plus up to [`CANDIDATE_READBACK`] records per image, a second
    /// one only for an image with more. On the CPU (single-label) the anchors are prefiltered by their largest
    /// logit and only the candidates are decoded, with the same functions as [`HeadOutput::decode_images`], so
    /// the result is bit for bit that of the full decode. Everything else decodes every anchor and runs
    /// [`crate::nms::non_max_suppression`] (or [`crate::nms::topk_detections`]). All paths give the same
    /// detections up to the device's float rounding.
    pub fn detections(&self, nms: &NmsConfig) -> Result<Vec<Vec<Detection>>> {
        let _g = no_grad();
        if !nms.multi_label {
            let fast = if self.box_raw.device().is_cpu() {
                self.detections_on_cpu(nms)?
            } else {
                self.detections_on_device(nms)?
            };
            if let Some(d) = fast {
                return Ok(d);
            }
        }
        let (nc, mode) = (self.num_classes, self.mode);
        let (prec, bin_prec) = self.dfl_precisions_for(nms)?;
        Ok(self
            .decode_images()?
            .into_par_iter()
            .enumerate()
            .map(|(i, (b, s))| {
                postprocess_with_dfl(
                    &b,
                    &s,
                    prec.as_ref().map(|p| p[i].as_slice()),
                    bin_prec.as_ref().map(|p| p[i].as_slice()),
                    nc,
                    nms,
                    mode,
                )
            })
            .collect())
    }

    /// Whether `nms` fuses kept boxes with DFL evidence ([`BoxFusion::Dfl`]) on this output's branch: only then
    /// are the side precisions needed.
    fn wants_precisions(&self, nms: &NmsConfig) -> bool {
        nms.fusion == BoxFusion::Dfl && self.mode == HeadMode::O2m
    }

    /// Whether `nms` scores kept boxes with scale-free DFL bin precisions ([`crate::fusion::ScoreRule::DflConsensus`]).
    fn wants_bin_precisions(&self, nms: &NmsConfig) -> bool {
        nms.score.wants_bin_precisions() && self.mode == HeadMode::O2m
    }

    /// Per-image side precisions (for [`BoxFusion::Dfl`]) and/or scale-free bin precisions (for
    /// [`crate::fusion::ScoreRule::DflConsensus`]), computed in a single pass / single device readback.
    #[allow(clippy::type_complexity)]
    pub(crate) fn dfl_precisions_for(
        &self,
        nms: &NmsConfig,
    ) -> Result<(Option<Vec<Vec<[f32; 4]>>>, Option<Vec<Vec<[f32; 4]>>>)> {
        let want_side = self.wants_precisions(nms);
        let want_bin = self.wants_bin_precisions(nms);
        if !want_side && !want_bin {
            return Ok((None, None));
        }
        if want_side && !want_bin {
            return Ok((Some(self.side_precisions()?), None));
        }
        let _g = no_grad();
        let (b, a, rm) = (self.batch(), self.num_anchors(), self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] {
            return Err(Error::Invalid(format!(
                "HeadOutput: box_raw {:?} does not match {a} anchors",
                self.box_raw.dims()
            )));
        }
        let strides = &self.anchors.strides;
        if !self.box_raw.device().is_cpu() {
            let m = self.moments_on_device()?.to_vec::<f32>()?;
            let side = want_side.then(|| {
                m.chunks_exact(8 * a)
                    .map(|img| (0..a).map(|j| precision_from_moments(img, a, j, strides[j])).collect())
                    .collect()
            });
            let bin = m
                .chunks_exact(8 * a)
                .map(|img| (0..a).map(|j| bin_precision_from_moments(img, a, j)).collect())
                .collect();
            return Ok((side, Some(bin)));
        }
        let reg = self.box_raw.to_dtype(DType::F32)?.to_vec::<f32>()?;
        let both: Vec<(Option<Vec<[f32; 4]>>, Vec<[f32; 4]>)> = reg
            .par_chunks_exact(4 * rm * a)
            .map(|rg| {
                let mut row = vec![0f32; 4 * rm];
                let mut sp = want_side.then(|| Vec::with_capacity(a));
                let mut bp = Vec::with_capacity(a);
                for j in 0..a {
                    for (k, v) in row.iter_mut().enumerate() {
                        *v = rg[k * a + j];
                    }
                    let stride = strides[j];
                    let mut s_arr = [0f32; 4];
                    let mut b_arr = [0f32; 4];
                    for side in 0..4 {
                        let (mean, var) = softmax_moments(&row[side * rm..(side + 1) * rm]);
                        if want_side {
                            s_arr[side] = side_precision(mean, var, stride);
                        }
                        b_arr[side] = bin_precision(mean, var);
                    }
                    if let Some(sp) = sp.as_mut() {
                        sp.push(s_arr);
                    }
                    bp.push(b_arr);
                }
                (sp, bp)
            })
            .collect();
        let mut side = want_side.then(|| Vec::with_capacity(both.len()));
        let mut bin = Vec::with_capacity(both.len());
        for (s, b) in both {
            if let (Some(sv), Some(s)) = (side.as_mut(), s) {
                sv.push(s);
            }
            bin.push(b);
        }
        Ok((side, Some(bin)))
    }

    /// Per image, every anchor's side precisions for DFL evidence fusion (`[A]`,
    /// [`crate::decode::side_precision`] of each side's DFL distribution at the anchor's stride). CPU outputs are
    /// read on the host; on another device the two softmax moments of every side are computed there
    /// ([`HeadOutput::moments_on_device`]) and only they are downloaded.
    pub fn side_precisions(&self) -> Result<Vec<Vec<[f32; 4]>>> {
        let _g = no_grad();
        let (b, a, rm) = (self.batch(), self.num_anchors(), self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] {
            return Err(Error::Invalid(format!(
                "HeadOutput: box_raw {:?} does not match {a} anchors",
                self.box_raw.dims()
            )));
        }
        let strides = &self.anchors.strides;
        if !self.box_raw.device().is_cpu() {
            let m = self.moments_on_device()?.to_vec::<f32>()?;
            return Ok(m
                .chunks_exact(8 * a)
                .map(|img| (0..a).map(|j| precision_from_moments(img, a, j, strides[j])).collect())
                .collect());
        }
        let reg = self.box_raw.to_dtype(DType::F32)?.to_vec::<f32>()?;
        Ok(reg
            .par_chunks_exact(4 * rm * a)
            .map(|rg| {
                let mut row = vec![0f32; 4 * rm];
                (0..a)
                    .map(|j| {
                        for (k, v) in row.iter_mut().enumerate() {
                            *v = rg[k * a + j];
                        }
                        side_precisions(&row, rm, strides[j])
                    })
                    .collect()
            })
            .collect())
    }

    /// `[B, 8, A]` on the outputs' device: per side `(l, t, r, b)` the mean, then per side the second moment, of
    /// the DFL softmax over the bin indices (bins and bins², f32): what [`HeadOutput::side_precisions`] downloads
    /// instead of the `4 · reg_max` logits.
    pub fn moments_on_device(&self) -> Result<Tensor> {
        dfl_moments(&self.box_raw, self.reg_max)
    }

    /// The CPU half of [`HeadOutput::detections`]: straight from the channel-major outputs (borrowed, not
    /// copied), [`single_label_block`]'s extraction, and the DFL decode ([`dfl_box`]) only for the anchors that
    /// pass. Like the GPU candidate kernel, anchors are processed in independent blocks
    /// ([`CPU_DECODE_BLOCK`]) on the rayon pool, and the blocks' candidates are joined in anchor order, so the
    /// result does not depend on the split. `None` for shapes (or dtypes) the full path should report.
    fn detections_on_cpu(&self, nms: &NmsConfig) -> Result<Option<Vec<Vec<Detection>>>> {
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] || self.cls_raw.dims() != [b, nc, a] || a == 0 || nc == 0 {
            return Ok(None);
        }
        if self.box_raw.dtype() != DType::F32 || self.cls_raw.dtype() != DType::F32 {
            return Ok(None);
        }
        let anchors = &self.anchors;
        let want_side = self.wants_precisions(nms);
        let want_bin = self.wants_bin_precisions(nms);
        let cand_conf = nms.candidate_conf_for(self.mode);
        // The candidates among anchors `j0..j1` of one image, plus any requested DFL precisions.
        let block = |cl: &[f32], rg: &[f32], j0: usize, j1: usize| {
            let mut found = Vec::new();
            single_label_block(cl, a, j0, j1, cand_conf, &mut found);
            let mut bins = vec![0f32; rm];
            let mut precs = Vec::new();
            let mut bprecs = Vec::new();
            let cands: Vec<Candidate> = found
                .into_iter()
                .map(|(j, class, score)| {
                    let j = j as usize;
                    let (point, st) = (anchors.points[j], anchors.strides[j]);
                    let (bbox, p, bp) = dfl_box(&mut bins, |i| rg[i * a + j], point, st, want_side, want_bin);
                    precs.extend(p);
                    bprecs.extend(bp);
                    Candidate { anchor: j as u32, class, score, bbox }
                })
                .collect();
            (cands, precs, bprecs)
        };
        let blocks = a.div_ceil(CPU_DECODE_BLOCK);
        let images = |reg: &[f32], cls: &[f32]| -> Vec<Vec<Detection>> {
            (0..b)
                .into_par_iter()
                .map(|i| {
                    let cl = &cls[i * nc * a..(i + 1) * nc * a];
                    let rg = &reg[i * 4 * rm * a..(i + 1) * 4 * rm * a];
                    let parts: Vec<_> = (0..blocks)
                        .into_par_iter()
                        .map(|t| block(cl, rg, t * CPU_DECODE_BLOCK, ((t + 1) * CPU_DECODE_BLOCK).min(a)))
                        .collect();
                    let mut cands = Vec::new();
                    let mut precs = Vec::new();
                    let mut bprecs = Vec::new();
                    for (c, p, bp) in parts {
                        cands.extend(c);
                        precs.extend(p);
                        bprecs.extend(bp);
                    }
                    finish_candidates(
                        self.mode,
                        &cands,
                        want_side.then_some(precs.as_slice()),
                        want_bin.then_some(bprecs.as_slice()),
                        nms,
                    )
                })
                .collect()
        };
        // Views of one storage (e.g. channel slices of one `[B, 4·reg_max + nc, A]` tensor) are not borrowed
        // twice: a second read lock on a storage while a writer queues for it could deadlock. Copy one instead.
        let dets = if self.box_raw.same_storage(&self.cls_raw) {
            let cls = self.cls_raw.to_vec::<f32>()?;
            self.box_raw.with_f32_slice(|reg| images(reg, &cls))?
        } else {
            self.box_raw.with_f32_slice(|reg| self.cls_raw.with_f32_slice(|cls| images(reg, cls)))??
        };
        Ok(Some(dets))
    }

    /// The anchors as `[3, A]` on `dev`: point x, point y (grid units), stride.
    fn anchors_tensor(&self, dev: Device) -> Result<Tensor> {
        let a = self.num_anchors();
        let mut v = Vec::with_capacity(3 * a);
        v.extend(self.anchors.points.iter().map(|p| p[0]));
        v.extend(self.anchors.points.iter().map(|p| p[1]));
        v.extend_from_slice(&self.anchors.strides);
        Ok(Tensor::from_vec(v, [3, a])?.to_device(dev)?)
    }

    /// The device half of [`HeadOutput::detections`]; `None` when the device has no candidate kernel.
    fn detections_on_device(&self, nms: &NmsConfig) -> Result<Option<Vec<Vec<Detection>>>> {
        Ok(self.candidates_on_device(nms)?.map(|c| {
            c.into_par_iter()
                .map(|c| {
                    finish_candidates(
                        self.mode,
                        &c.cands,
                        c.precisions.as_deref(),
                        c.bin_precisions.as_deref(),
                        nms,
                    )
                })
                .collect()
        }))
    }

    /// Every image's candidates (best class score `> nms.candidate_conf()`) with their boxes, in anchor order,
    /// decoded on the outputs' device ([`Tensor::detect_candidates`]), with DFL fusion / DFL consensus also their
    /// precisions (from [`HeadOutput::moments_on_device`]); `None` when the device has no candidate kernel.
    pub(super) fn candidates_on_device(&self, nms: &NmsConfig) -> Result<Option<Vec<ImageCandidates>>> {
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] || self.cls_raw.dims() != [b, nc, a] || a == 0 || nc == 0 {
            return Ok(None);
        }
        let dev = self.box_raw.device();
        let p = DetectDecode { reg_max: rm, num_classes: nc, conf: nms.candidate_conf_for(self.mode) };
        let anchors = self.anchors_tensor(dev)?;
        let Some(cands) = Tensor::detect_candidates(&self.box_raw, &self.cls_raw, &anchors, &p)? else {
            return Ok(None);
        };
        const W: usize = DETECT_RECORD_WORDS;
        let cap = CANDIDATE_READBACK.min(a);
        let head = cands.narrow(1, 0, 1 + W * cap)?.to_vec::<u32>()?;
        let mut records: Vec<Vec<u32>> = Vec::with_capacity(b);
        for (i, row) in head.chunks_exact(1 + W * cap).enumerate() {
            let n = row[0] as usize;
            if n > a {
                return Err(Error::Invalid(format!("detect_candidates: {n} candidates for {a} anchors")));
            }
            records.push(if n <= cap {
                row[1..1 + W * n].to_vec()
            } else {
                cands.narrow(0, i, 1)?.narrow(1, 1, W * n)?.to_vec::<u32>()?
            });
        }
        let want_side = self.wants_precisions(nms);
        let want_bin = self.wants_bin_precisions(nms);
        // DFL fusion / DFL consensus: the side moments of every anchor, one download (`[B, 8, A]`).
        let moments =
            if want_side || want_bin { Some(self.moments_on_device()?.to_vec::<f32>()?) } else { None };
        let strides = &self.anchors.strides;
        Ok(Some(
            records
                .into_par_iter()
                .enumerate()
                .map(|(i, r)| {
                    let cands = candidates_of_records(&r);
                    let precisions = if want_side {
                        moments.as_ref().map(|m| {
                            let img = &m[i * 8 * a..(i + 1) * 8 * a];
                            cands
                                .iter()
                                .map(|c| c.anchor as usize)
                                .map(|j| precision_from_moments(img, a, j, strides[j]))
                                .collect()
                        })
                    } else {
                        None
                    };
                    let bin_precisions = if want_bin {
                        moments.as_ref().map(|m| {
                            let img = &m[i * 8 * a..(i + 1) * 8 * a];
                            cands
                                .iter()
                                .map(|c| c.anchor as usize)
                                .map(|j| bin_precision_from_moments(img, a, j))
                                .collect()
                        })
                    } else {
                        None
                    };
                    ImageCandidates { cands, precisions, bin_precisions }
                })
                .collect(),
        ))
    }

    /// The decoded output `[B, A, 4 + nc]` (MODEL.md §6.4): per anchor `x1, y1, x2, y2` in input pixels, then
    /// the `nc` class scores. A CPU tensor without gradient.
    pub fn decode(&self) -> Result<Tensor> {
        let (a, nc) = (self.num_anchors(), self.num_classes);
        let images = self.decode_images()?;
        let mut out = Vec::with_capacity(images.len() * a * (4 + nc));
        for (boxes, scores) in &images {
            for (b, s) in boxes.iter().zip(scores.chunks_exact(nc.max(1))) {
                out.extend_from_slice(&[b.x1, b.y1, b.x2, b.y2]);
                out.extend_from_slice(s);
            }
        }
        Ok(Tensor::from_vec(out, [images.len(), a, 4 + nc])?)
    }
}

/// The candidates of a device decode's records (`DETECT_RECORD_WORDS` words each, in arrival order), in anchor
/// order (NMS tie-breaks by anchor).
pub(super) fn candidates_of_records(r: &[u32]) -> Vec<Candidate> {
    let mut c: Vec<Candidate> = r
        .as_chunks::<DETECT_RECORD_WORDS>()
        .0
        .iter()
        .map(|w| Candidate {
            anchor: w[0],
            class: w[1],
            score: f32::from_bits(w[2]),
            bbox: Xyxy::new(
                f32::from_bits(w[3]),
                f32::from_bits(w[4]),
                f32::from_bits(w[5]),
                f32::from_bits(w[6]),
            ),
        })
        .collect();
    c.sort_unstable_by_key(|c| c.anchor);
    c
}

impl FoloModel {
    /// The predictor's model step: an eval-mode forward under `no_grad` (of the [`FoloModel::head_mode`]
    /// branch), then decode and class-aware NMS (`nms`, MODEL.md §9.3; for an `o2o` head the top-k of §6.6) on
    /// the host, images in parallel; outputs on a GPU are decoded there and downloaded
    /// once ([`HeadOutput::decode_on_device`]). Boxes are in network-input
    /// pixels; mapping back through the letterbox is the caller's job. Fails in training mode, where the
    /// forward would update the BatchNorm statistics.
    pub fn predict_raw(&self, x: &Tensor, nms: &NmsConfig) -> Result<RawPrediction> {
        self.predict_raw_as(x, nms, self.head_mode())
    }

    /// [`FoloModel::predict_raw`] with the `mode` branch and its post-processing ([`FoloModel::forward_head`]).
    pub fn predict_raw_as(&self, x: &Tensor, nms: &NmsConfig, mode: HeadMode) -> Result<RawPrediction> {
        if self.is_training() {
            return Err(Error::Invalid(
                "FoloModel::predict_raw: the model is in training mode; call eval() or fuse() first".into(),
            ));
        }
        let _g = no_grad();
        let output = self.forward_head(x, mode)?;
        let (nc, mode) = (output.num_classes, output.mode);
        let (prec, bin_prec) = output.dfl_precisions_for(nms)?;
        let images = output
            .decode_images()?
            .into_par_iter()
            .enumerate()
            .map(|(i, (boxes, scores))| {
                let p = prec.as_ref().map(|p| p[i].as_slice());
                let bp = bin_prec.as_ref().map(|p| p[i].as_slice());
                let detections = postprocess_with_dfl(&boxes, &scores, p, bp, nc, nms, mode);
                ImagePrediction { boxes, scores, detections }
            })
            .collect();
        Ok(RawPrediction { output, images })
    }

    /// The predictor's model step without the per-anchor outputs: an eval-mode forward under `no_grad`,
    /// then [`HeadOutput::detections`] (decoded on the outputs' device where it can be, so a GPU prediction
    /// downloads only the candidates). Where [`FoloModel::candidates_first_applies`] (a fused model on the CPU),
    /// [`FoloModel::predict_candidates`] then [`FoloModel::finish_candidates`] instead: the same detections, bit
    /// for bit, with the box branch run only at the candidates. Boxes are in network-input pixels. Fails in
    /// training mode.
    pub fn predict_detections(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<Vec<Detection>>> {
        if self.is_training() {
            return Err(Error::Invalid(
                "FoloModel::predict_detections: the model is in training mode; call eval() or fuse() first"
                    .into(),
            ));
        }
        if self.candidates_first_applies(x.device(), nms) {
            let cands = self.predict_candidates(x, nms)?;
            return Ok(cands.into_par_iter().map(|c| self.finish_candidates(&c, nms)).collect());
        }
        let _g = no_grad();
        self.forward(x)?.detections(nms)
    }
}

/// [`HeadOutput::moments_on_device`] of raw box outputs `box_raw` (`[B, 4 · reg_max, A]`, on any device). For
/// `A > 1` every op runs column by column (an anchor's values read its own logits only, in the same order for
/// any `A`), so any such `box_raw` holding an anchor's logits gives that anchor the same bits.
pub(super) fn dfl_moments(box_raw: &Tensor, rm: usize) -> Result<Tensor> {
    dfl_moments_with(box_raw, rm, &moment_bins(rm)?.to_device(box_raw.device())?)
}

/// The `[2 · reg_max]` bin values [`dfl_moments`] weighs the distribution with: `i`, then `i²`.
pub(super) fn moment_bins(rm: usize) -> Result<Tensor> {
    let mut k = Vec::with_capacity(2 * rm);
    k.extend((0..rm).map(|i| i as f32));
    k.extend((0..rm).map(|i| (i * i) as f32));
    Ok(Tensor::from_vec(k, [2 * rm])?)
}

/// [`dfl_moments`] with its bins `k` ([`moment_bins`]) already on the device, so it uploads nothing.
pub(super) fn dfl_moments_with(box_raw: &Tensor, rm: usize, k: &Tensor) -> Result<Tensor> {
    let _g = no_grad();
    let (b, a) = (box_raw.dims()[0], box_raw.dims()[2]);
    let bins = k.narrow(0, 0, rm)?.reshape([1, 1, rm, 1])?;
    let bins2 = k.narrow(0, rm, rm)?.reshape([1, 1, rm, 1])?;
    let p = box_raw.to_dtype(DType::F32)?.reshape([b, 4, rm, a])?.softmax(2)?;
    let mean = p.mul(&bins)?.sum(2, false)?;
    let m2 = p.mul(&bins2)?.sum(2, false)?;
    Ok(Tensor::cat(&[&mean, &m2], 1)?)
}

/// The side precisions of anchor `j` from one image's [`HeadOutput::moments_on_device`] values (`[8, A]`).
pub(super) fn precision_from_moments(img: &[f32], a: usize, j: usize, stride: f32) -> [f32; 4] {
    std::array::from_fn(|side| {
        let mean = img[side * a + j];
        let var = (img[(4 + side) * a + j] - mean * mean).max(0.0);
        side_precision(mean, var, stride)
    })
}

/// The scale-free bin precisions of anchor `j` from one image's [`HeadOutput::moments_on_device`] values (`[8, A]`).
pub(super) fn bin_precision_from_moments(img: &[f32], a: usize, j: usize) -> [f32; 4] {
    std::array::from_fn(|side| {
        let mean = img[side * a + j];
        let var = (img[(4 + side) * a + j] - mean * mean).max(0.0);
        bin_precision(mean, var)
    })
}
