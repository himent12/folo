//! Candidate-first prediction (docs/MODEL.md §9.5): the head's class branches run densely, the candidates are
//! read straight from each level's class map, and the box branch runs only where a candidate needs a box.
//!
//! The box branch (`conv0`, `conv1`: 3×3, `out`: 1×1) and the class branch are independent functions of the same
//! level feature, so the class scores, and with them the candidate set, are the dense forward's. A candidate's
//! box needs `out` at its own pixel, `conv1` at that pixel and `conv0` on that pixel's receptive field. The convs
//! run through [`Tensor::conv2d_act_at`], which on the CPU computes only the Winograd tiles (and 1×1 columns)
//! that hold a site with the dense conv's own arithmetic, and whose site values read only their receptive
//! field: every value a candidate reads is then the dense forward's, and every box is the dense path's bit for
//! bit.

use folo_core::{DType, Device, Tensor, no_grad, with_no_grad};
use folo_nn::{ConvBnAct, Forward, Module, Result};
use rayon::prelude::*;

use super::FoloModel;
use super::head::{HeadLevel, HeadMode};
use super::predict::{CPU_DECODE_BLOCK, Found, dfl_box, finish_candidates, single_label_block};
use crate::decode::Anchors;
use crate::fusion::BoxFusion;
use crate::nms::{Candidate, Detection, NmsConfig, class_allowed};

/// One image's candidates from [`FoloModel::predict_candidates`], in anchor order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageCandidates {
    /// The candidates that [`FoloModel::finish_candidates`] can keep, with their boxes.
    pub cands: Vec<Candidate>,
    /// With DFL evidence fusion (`NmsConfig::fusion == BoxFusion::Dfl`, an `o2m` head): each candidate's side
    /// precisions, parallel to `cands`, the dense path's bit for bit. `None` otherwise.
    pub precisions: Option<Vec<[f32; 4]>>,
    /// With DFL consensus scoring (`NmsConfig::score == ScoreRule::DflConsensus`, an `o2m` head): each candidate's
    /// scale-free bin precisions, parallel to `cands`, the dense path's bit for bit. `None` otherwise.
    pub bin_precisions: Option<Vec<[f32; 4]>>,
}

/// `c` (a fused `Conv` block) at the output sites `at` ([`Tensor::conv2d_act_at`]).
fn conv_at(c: &ConvBnAct, x: &Tensor, at: &Tensor) -> Result<Tensor> {
    debug_assert!(c.is_fused(), "candidate-first prediction needs a fused model");
    let p = c.conv.config().params();
    Ok(x.conv2d_act_at(&c.conv.weight, c.conv.bias.as_ref(), &p, c.act.unary_op(), at)?)
}

/// The single-label candidates of one image's level map `cl` (`[nc, a]`), in anchor order, in blocks of
/// [`CPU_DECODE_BLOCK`] anchors on the rayon pool (the dense path's split, so the same result).
fn level_candidates(cl: &[f32], a: usize, conf: f32) -> Vec<Found> {
    let blocks = a.div_ceil(CPU_DECODE_BLOCK);
    let parts: Vec<Vec<Found>> = (0..blocks)
        .into_par_iter()
        .map(|t| {
            let mut v = Vec::new();
            single_label_block(
                cl,
                a,
                t * CPU_DECODE_BLOCK,
                ((t + 1) * CPU_DECODE_BLOCK).min(a),
                conf,
                &mut v,
            );
            v
        })
        .collect();
    parts.concat()
}

/// Of one image's candidates `found[level]` (in anchor order), the ones that need a box, in anchor order: those
/// [`finish_candidates`] can keep. `o2m`: every candidate of an allowed class (NMS drops the others first).
/// `o2o`: of the `max_det` best by score, then anchor (§6.6, ties by anchor), those of an allowed class (the
/// class filter runs after the selection). `offsets[level]` is the level's first anchor.
fn needing_boxes(
    found: [Vec<Found>; 3],
    offsets: [usize; 3],
    mode: HeadMode,
    nms: &NmsConfig,
) -> [Vec<Found>; 3] {
    let allowed = |f: &Found| class_allowed(nms, f.1);
    match mode {
        HeadMode::O2m => found.map(|v| v.into_iter().filter(allowed).collect()),
        HeadMode::O2o => {
            // (score, global anchor, level, found): every global anchor is distinct, so the order is total.
            let mut all: Vec<(f32, usize, usize, Found)> = Vec::new();
            for (l, v) in found.iter().enumerate() {
                all.extend(v.iter().map(|f| (f.2, offsets[l] + f.0 as usize, l, *f)));
            }
            let rank = |x: &(f32, usize, usize, Found), y: &(f32, usize, usize, Found)| {
                y.0.total_cmp(&x.0).then(x.1.cmp(&y.1))
            };
            if all.len() > nms.max_det {
                if nms.max_det == 0 {
                    all.clear();
                } else {
                    all.select_nth_unstable_by(nms.max_det - 1, rank);
                    all.truncate(nms.max_det);
                }
            }
            all.sort_unstable_by_key(|x| x.1);
            let mut out: [Vec<Found>; 3] = Default::default();
            for (_, _, l, f) in all.into_iter().filter(|x| allowed(&x.3)) {
                out[l].push(f);
            }
            out
        }
    }
}

impl FoloModel {
    /// Whether [`FoloModel::predict_candidates`] serves an input on `device` with the settings `nms`: a fused
    /// model in eval mode, on the CPU in `F32` or on a GPU, single-label (multi-label validation at
    /// `conf = 0.001` passes thousands of anchors per image and stays on the dense path).
    pub fn candidates_first_applies(&self, device: Device, nms: &NmsConfig) -> bool {
        self.fused
            && !self.is_training()
            && !nms.multi_label
            && (!device.is_cpu() || self.dtype() == DType::F32)
    }

    /// Candidate-first prediction (MODEL.md §9.5) of a `[B, 3, H, W]` batch with the [`FoloModel::head_mode`]
    /// branch: per image the candidates that [`FoloModel::finish_candidates`] can keep, with their boxes, in
    /// anchor order, each bit for bit the record the dense path ([`HeadOutput::detections`] on the same device)
    /// makes for that anchor. The backbone, the neck and the class branches run densely; the box branch only at
    /// the candidates; the dense raw outputs ([`HeadOutput`]) are never made (on a GPU, only for an image with
    /// more candidates than [`GPU_SITE_SLOTS`] on some level). Fails unless
    /// [`FoloModel::candidates_first_applies`].
    ///
    /// [`GPU_SITE_SLOTS`]: super::lazy_gpu::GPU_SITE_SLOTS
    ///
    /// [`HeadOutput`]: super::HeadOutput
    /// [`HeadOutput::detections`]: super::HeadOutput::detections
    pub fn predict_candidates(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<ImageCandidates>> {
        if !self.candidates_first_applies(x.device(), nms) {
            return Err(folo_nn::Error::Invalid(
                "FoloModel::predict_candidates needs a fused model in eval mode (F32 on the CPU), single-label"
                    .into(),
            ));
        }
        if !x.device().is_cpu() {
            return self.predict_candidates_gpu(x, nms);
        }
        folo_core::cpu::in_pool(|| {
            let _g = no_grad();
            let (h, w) = self.check_input(x)?;
            let feats = self.features(x, None)?;
            self.predict_candidates_from_features(feats, x.dims()[0], h, w, nms)
        })
    }

    /// Candidate-first prediction reading a pre-packed `[b, h, w]` `u32` `[R, G, B, 0]` canvas directly into the
    /// int8 `u8` stem (zero `f32` image materialisation).
    pub(crate) fn predict_candidates_rgb0_u8(
        &self,
        rgb0: &[u32],
        b: usize,
        h: usize,
        w: usize,
        nms: &NmsConfig,
    ) -> Result<Vec<ImageCandidates>> {
        if !self.candidates_first_applies(Device::Cpu, nms) {
            return Err(folo_nn::Error::Invalid(
                "FoloModel::predict_candidates needs a fused model in eval mode (F32 on the CPU), single-label"
                    .into(),
            ));
        }
        let plan = self.int8.as_ref().ok_or_else(|| {
            folo_nn::Error::Invalid("predict_candidates_rgb0_u8 requires an int8 plan".into())
        })?;
        folo_core::cpu::in_pool(|| {
            let _g = no_grad();
            self.check_hw(b, h, w)?;
            let feats = super::quant::exec::features_rgb0_u8(self, plan, rgb0, b, h, w)?;
            self.predict_candidates_from_features(feats, b, h, w, nms)
        })
    }

    fn predict_candidates_from_features(
        &self,
        feats: [Tensor; 3],
        b: usize,
        h: usize,
        w: usize,
        nms: &NmsConfig,
    ) -> Result<Vec<ImageCandidates>> {
        let mode = self.head_mode;
        let want_side = self.wants_precisions(nms);
        let want_bin = self.wants_bin_precisions(nms);
        let cand_conf = nms.candidate_conf_for(mode);
        let levels = match (mode, &self.head.o2o) {
            (HeadMode::O2o, Some(o2o)) => o2o,
            _ => &self.head.levels,
        };
        let anchors = self.anchors(h, w);
        let hw: Vec<usize> = feats.iter().map(|f| f.dims()[2] * f.dims()[3]).collect();
        let offsets = [0, hw[0], hw[0] + hw[1]];
        // The class branches and the candidates of every level (side by side, like the dense head).
        let found: Vec<Vec<Vec<Found>>> = (0..3)
            .into_par_iter()
            .map(|l| with_no_grad(|| self.class_candidates(&levels[l], &feats[l], cand_conf)))
            .collect::<Result<_>>()?;
        // Per image, the candidates that need a box.
        let mut need: [Vec<Vec<Found>>; 3] = Default::default();
        let mut found: Vec<_> = found.into_iter().map(Vec::into_iter).collect();
        for _ in 0..b {
            let per_level = [0, 1, 2].map(|l| found[l].next().unwrap_or_default());
            for (l, v) in needing_boxes(per_level, offsets, mode, nms).into_iter().enumerate() {
                need[l].push(v);
            }
        }
        // Their boxes, level by level (side by side), then every image's records in anchor order.
        let boxed: Vec<Vec<ImageCandidates>> = (0..3)
            .into_par_iter()
            .map(|l| {
                with_no_grad(|| {
                    self.boxes_at(&levels[l], &feats[l], &need[l], &anchors, offsets[l], want_side, want_bin)
                })
            })
            .collect::<Result<_>>()?;
        Ok((0..b)
            .map(|i| ImageCandidates {
                cands: boxed.iter().flat_map(|lv| lv[i].cands.iter().copied()).collect(),
                precisions: want_side.then(|| {
                    boxed.iter().flat_map(|lv| lv[i].precisions.iter().flatten().copied()).collect()
                }),
                bin_precisions: want_bin.then(|| {
                    boxed.iter().flat_map(|lv| lv[i].bin_precisions.iter().flatten().copied()).collect()
                }),
            })
            .collect())
    }

    /// The single-label post-process of one image's candidates from [`FoloModel::predict_candidates`] for the
    /// [`FoloModel::head_mode`] branch: class-aware NMS (§9.3, with the box fusion `nms.fusion`, §9.4) for
    /// `o2m`, §6.6's top-k for `o2o`. Highest score first, in network-input pixels. `c` must come from
    /// [`FoloModel::predict_candidates`] with the same `nms` (DFL fusion needs its precisions).
    pub fn finish_candidates(&self, c: &ImageCandidates, nms: &NmsConfig) -> Vec<Detection> {
        finish_candidates(self.head_mode, &c.cands, c.precisions.as_deref(), c.bin_precisions.as_deref(), nms)
    }

    /// Whether the post-process of `nms` needs each candidate's side precisions (DFL evidence fusion, an `o2m`
    /// head): the dense path's `HeadOutput::wants_precisions`.
    pub(super) fn wants_precisions(&self, nms: &NmsConfig) -> bool {
        nms.fusion == BoxFusion::Dfl && self.head_mode == HeadMode::O2m
    }

    /// Whether the post-process of `nms` needs each candidate's scale-free DFL bin precisions (`o2m` head).
    pub(super) fn wants_bin_precisions(&self, nms: &NmsConfig) -> bool {
        nms.score.wants_bin_precisions() && self.head_mode == HeadMode::O2m
    }

    /// One level's class branch on `x` (`[B, C, h, w]`) and each image's candidates in it (local anchors).
    fn class_candidates(&self, level: &HeadLevel, x: &Tensor, conf: f32) -> Result<Vec<Vec<Found>>> {
        let (b, _, h, w) = x.dims4()?;
        let (a, nc) = (h * w, self.head.num_classes);
        let cls = level.cls_branch.forward(x)?;
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

    /// One level's box branch at the anchors `need[img]` (local, ascending) of `x` (`[B, C, h, w]`), decoded into
    /// per-image candidate records (global anchors from `offset`), with `dfl` also their side precisions.
    #[allow(clippy::too_many_arguments)]
    fn boxes_at(
        &self,
        level: &HeadLevel,
        x: &Tensor,
        need: &[Vec<Found>],
        anchors: &Anchors,
        offset: usize,
        want_side: bool,
        want_bin: bool,
    ) -> Result<Vec<ImageCandidates>> {
        let (b, _, h, w) = x.dims4()?;
        let (hw, rm) = (h * w, self.head.reg_max);
        if need.iter().all(Vec::is_empty) {
            let empty = ImageCandidates {
                cands: Vec::new(),
                precisions: want_side.then(Vec::new),
                bin_precisions: want_bin.then(Vec::new),
            };
            return Ok(vec![empty; b]);
        }
        // `out` and `conv1` at the candidates; `conv0` on the patch `conv1` reads: the 3×3 receptive field, or
        // the full 6×6 input patch of each site's 4×4 tile when `conv1` may take Winograd F(4×4, 3×3).
        let bx = &level.box_branch;
        let c1 = bx.conv1.conv.config();
        let wino4 = c1.kernel == 3 && c1.stride == 1 && c1.padding == 1 && h >= 24 && w >= 24;
        let mut sites = Vec::new();
        let mut field = vec![false; b * hw];
        for (i, v) in need.iter().enumerate() {
            for &(j, _, _) in v {
                let j = j as usize;
                sites.push((i * hw + j) as u32);
                let (y0, x0) = (j / w, j % w);
                let (y_lo, y_hi, x_lo, x_hi) = if wino4 {
                    let (ty, tx) = (y0 / 4, x0 / 4);
                    (
                        (4 * ty).saturating_sub(1),
                        (4 * ty + 5).min(h),
                        (4 * tx).saturating_sub(1),
                        (4 * tx + 5).min(w),
                    )
                } else {
                    (y0.saturating_sub(1), (y0 + 2).min(h), x0.saturating_sub(1), (x0 + 2).min(w))
                };
                for y in y_lo..y_hi {
                    field[i * hw + y * w + x_lo..i * hw + y * w + x_hi].fill(true);
                }
            }
        }
        let field: Vec<u32> = (0..b * hw).filter(|&p| field[p]).map(|p| p as u32).collect();
        let (n1, n0) = (sites.len(), field.len());
        let (sites, field) = (Tensor::from_vec(sites, [n1])?, Tensor::from_vec(field, [n0])?);
        let h0 = conv_at(&bx.conv0, x, &field)?;
        let h1 = conv_at(&bx.conv1, &h0, &sites)?;
        drop(h0);
        let p = bx.out.config().params();
        let raw = h1.conv2d_act_at(&bx.out.weight, bx.out.bias.as_ref(), &p, None, &sites)?;
        let per = 4 * rm * hw;
        Ok(raw.with_f32_slice(|rg| {
            need.iter()
                .enumerate()
                .map(|(i, v)| {
                    let rg = &rg[i * per..(i + 1) * per];
                    let mut bins = vec![0f32; rm];
                    let mut out = ImageCandidates {
                        cands: Vec::with_capacity(v.len()),
                        precisions: want_side.then(|| Vec::with_capacity(v.len())),
                        bin_precisions: want_bin.then(|| Vec::with_capacity(v.len())),
                    };
                    for &(j, class, score) in v {
                        let (j, k) = (j as usize, offset + j as usize);
                        let (point, st) = (anchors.points[k], anchors.strides[k]);
                        let (bbox, p, bp) =
                            dfl_box(&mut bins, |r| rg[r * hw + j], point, st, want_side, want_bin);
                        out.cands.push(Candidate { anchor: k as u32, class, score, bbox });
                        out.precisions.iter_mut().zip(p).for_each(|(v, p)| v.push(p));
                        out.bin_precisions.iter_mut().zip(bp).for_each(|(v, p)| v.push(p));
                    }
                    out
                })
                .collect()
        })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use folo_core::rng::Rng;

    /// The box branch at isolated anchors (every corner, every border, even and odd coordinates, neighbours
    /// across a Winograd tile edge) of a rectangular input whose P5 map has odd sides, in two images with
    /// different anchors: the dense forward's boxes bit for bit, with every buffer the branch did not write
    /// poisoned to NaN.
    #[test]
    fn isolated_sites_get_the_dense_boxes() {
        folo_core::cpu::poison_uninit(true);
        let mut m = FoloModel::with_rng(ModelConfig::new("n", 3).unwrap(), &mut Rng::seed(21)).unwrap();
        m.fuse().unwrap();
        let x = Tensor::rand_uniform([2, 3, 352, 160], 0.0, 1.0, &mut Rng::seed(22));
        let _g = no_grad();
        let out = m.forward(&x).unwrap();
        let dense: Vec<Vec<crate::boxes::Xyxy>> =
            out.decode_images().unwrap().into_iter().map(|d| d.0).collect();
        let feats = m.features(&x, None).unwrap();
        for (l, lv) in out.anchors.levels.iter().enumerate() {
            let (h, w) = (lv.height, lv.width);
            let first =
                [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (0, w / 2), (h / 2, 0), (h - 1, w / 2)];
            let second =
                [(h / 2, w - 1), (1, 1), (2, 3), (3, 2), (h - 2, w - 2), (h / 2, w / 2), (h / 2 + 1, w / 2)];
            let need: Vec<Vec<Found>> = [&first[..], &second[..]]
                .iter()
                .map(|pts| {
                    let mut v: Vec<u32> = pts.iter().map(|&(y, x)| (y * w + x) as u32).collect();
                    v.sort_unstable();
                    v.dedup();
                    v.into_iter().map(|j| (j, 0, 1.0)).collect()
                })
                .collect();
            let got = m
                .boxes_at(&m.head.levels[l], &feats[l], &need, &out.anchors, lv.start, false, false)
                .unwrap();
            for (i, (g, n)) in got.iter().zip(&need).enumerate() {
                assert_eq!(g.cands.len(), n.len());
                assert_eq!(g.precisions, None);
                for c in &g.cands {
                    let want = dense[i][c.anchor as usize];
                    let bits = |b: &crate::boxes::Xyxy| [b.x1, b.y1, b.x2, b.y2].map(f32::to_bits);
                    assert_eq!(bits(&c.bbox), bits(&want), "level {l}, image {i}, anchor {}", c.anchor);
                }
            }
        }
    }

    /// Mixed Winograd/direct dispatch (`conv0` on direct `8->64` or F(2×2) `16->64` at `40×40`, `conv1` on
    /// F(4×4) `64->64` at `40×40`) under `poison_uninit(true)`: `boxes_at` populates the full 6×6 patch of each
    /// requested 4×4 tile in `conv0`, so `conv1` never reads NaN and matches the dense forward bit for bit.
    #[test]
    fn mixed_f2_conv0_f4_conv1_sites_match_dense_under_poison() {
        use crate::model::blocks::Init;

        folo_core::cpu::poison_uninit(true);
        let mut rng = Rng::seed(31);
        let mut m = FoloModel::with_rng(ModelConfig::new("n", 3).unwrap(), &mut rng).unwrap();
        m.fuse().unwrap();
        let anchors = m.anchors(320, 320);
        let lv = &anchors.levels[0]; // P3: 40×40
        let (h, w) = (lv.height, lv.width);
        assert_eq!((h, w), (40, 40));
        assert_eq!(m.head.levels[0].box_branch.conv1.conv.config().in_channels, 64);
        assert_eq!(m.head.levels[0].box_branch.conv1.conv.config().out_channels, 64);
        // Isolated sites covering tile corners, tile interior, borders and adjacent tile boundaries.
        let pts = [(0, 0), (4, 4), (5, 5), (7, 7), (16, 20), (23, 19), (39, 39)];
        let need: Vec<Vec<Found>> = vec![pts.iter().map(|&(y, x)| ((y * w + x) as u32, 0, 1.0)).collect()];
        let _g = no_grad();
        let (hw, rm) = (h * w, m.head.reg_max);
        // `c_in = 8` puts `conv0` on direct conv; `c_in = 16` puts `conv0` on F(2×2) (`16*64 < 2048`,
        // `tiles = 100 < 200`); in both cases `conv1` (`64->64` at `40×40`) runs F(4×4).
        for c_in in [8usize, 16] {
            let mut conv0 = Init::new(&mut rng).conv(c_in, 64, 3, 1);
            conv0.fuse_().unwrap();
            m.head.levels[0].box_branch.conv0 = conv0;
            let feat = Tensor::rand_uniform([1, c_in, h, w], -1.0, 1.0, &mut rng);
            let dense_raw = m.head.levels[0].box_branch.forward(&feat).unwrap();
            let got = m.boxes_at(&m.head.levels[0], &feat, &need, &anchors, lv.start, false, false).unwrap();
            assert_eq!(got[0].cands.len(), pts.len());
            dense_raw
                .with_f32_slice(|rg| {
                    let mut bins = vec![0f32; rm];
                    for c in &got[0].cands {
                        let k = c.anchor as usize;
                        let j = k - lv.start;
                        let (want, _, _) = dfl_box(
                            &mut bins,
                            |r| rg[r * hw + j],
                            anchors.points[k],
                            anchors.strides[k],
                            false,
                            false,
                        );
                        let bits = |b: &crate::boxes::Xyxy| [b.x1, b.y1, b.x2, b.y2].map(f32::to_bits);
                        assert_eq!(bits(&c.bbox), bits(&want), "c_in={c_in} -> F(4x4) anchor {k}");
                    }
                })
                .unwrap();
        }
    }
}
