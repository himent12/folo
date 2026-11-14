//! Detector pre- and post-processing behind optional `Backend` hooks: letterboxing `u8` images into the
//! network input, decoding the head into above-threshold candidates, and the class pass and box decode of
//! candidate-first prediction. All four return `Ok(None)` on backends without the kernels (the CPU), so callers
//! keep their host implementations as the fallback and reference.

use super::with_views_dyn;
use crate::backend::{ClassSites, DETECT_RECORD_WORDS, DetectDecode, LetterboxImage, SiteLevel};
use crate::{DType, Error, Result, Shape, Tensor};

fn unsupported(e: &Error) -> bool {
    matches!(e, Error::NotSupported { .. })
}

impl Tensor {
    /// Letterbox `U8` RGB images (`images[i]` is `[H_i, W_i, 3]`, all on one device) into a `[B, 3, OH, OW]`
    /// network input of `dtype` (`F32`, or `F16` on backends that compute in it; see
    /// [`crate::backend::Backend::letterbox_rgb8`] for the exact sampling). `taps` (`U32`, same device) holds
    /// the tap arrays `geoms[i].taps` points at. `Ok(None)` when the device has no kernel for it. Not recorded
    /// by autograd.
    #[allow(clippy::too_many_arguments)]
    pub fn letterbox_rgb8(
        images: &[&Tensor],
        taps: &Tensor,
        geoms: &[LetterboxImage],
        out_hw: [usize; 2],
        pad_value: u8,
        scale: f32,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        const OP: &str = "letterbox_rgb8";
        let first = *images.first().ok_or_else(|| Error::invalid_arg(OP, "no images given"))?;
        if images.len() != geoms.len() {
            return Err(Error::invalid_arg(
                OP,
                format!("{} images but {} geometries", images.len(), geoms.len()),
            ));
        }
        if taps.dtype() != DType::U32 || taps.rank() != 1 {
            return Err(Error::invalid_shape(OP, taps.shape(), "taps must be a U32 vector"));
        }
        let [oh, ow] = out_hw;
        for (im, g) in images.iter().zip(geoms) {
            first.check_same_device(im, OP)?;
            if im.dtype() != DType::U8 {
                return Err(Error::DTypeMismatch { op: OP, expected: DType::U8, got: im.dtype() });
            }
            if im.dims() != [g.src[1], g.src[0], 3] {
                return Err(Error::invalid_shape(
                    OP,
                    im.shape(),
                    format!("expected [{}, {}, 3]", g.src[1], g.src[0]),
                ));
            }
            let fits = g.src.iter().chain(&g.resized).chain(&g.prefilter).all(|&v| v > 0)
                && g.pad[0] + g.resized[0] <= ow
                && g.pad[1] + g.resized[1] <= oh
                && g.taps + 3 * (g.resized[0] + g.resized[1]) <= taps.numel();
            if !fits {
                return Err(Error::invalid_arg(
                    OP,
                    format!("geometry {g:?} does not fit a {oh}x{ow} canvas"),
                ));
            }
        }
        first.check_same_device(taps, OP)?;
        let b = first.backend()?;
        if !(dtype == DType::F32 || dtype == DType::F16 && b.computes_f16()) {
            return Err(Error::UnsupportedDType { op: OP, dtype });
        }
        let mut ts: Vec<&Tensor> = images.to_vec();
        ts.push(taps);
        let k = images.len();
        match with_views_dyn(&ts, |v| b.letterbox_rgb8(&v[..k], v[k], geoms, out_hw, pad_value, scale, dtype))
        {
            Ok(s) => Ok(Some(Tensor::wrap(s, Shape::new(&[k, 3, oh, ow]), OP)?)),
            Err(e) if unsupported(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The detection candidates of raw head outputs `box_raw` `F32 [B, 4·reg_max, A]` and `cls_raw`
    /// `F32 [B, nc, A]` with `anchors` `F32 [3, A]` (see [`crate::backend::Backend::detect_candidates`]):
    /// `U32 [B, 1 + 7·A]`, per image a count and that many `(anchor, class, score, x1, y1, x2, y2)` records.
    /// `Ok(None)` when the device has no kernel for it. Not recorded by autograd.
    pub fn detect_candidates(
        box_raw: &Tensor,
        cls_raw: &Tensor,
        anchors: &Tensor,
        p: &DetectDecode,
    ) -> Result<Option<Tensor>> {
        const OP: &str = "detect_candidates";
        box_raw.check_float(OP)?;
        cls_raw.check_float_like(box_raw, OP)?;
        anchors.check_f32(OP)?;
        for t in [cls_raw, anchors] {
            box_raw.check_same_device(t, OP)?;
        }
        let bad = |t: &Tensor| Error::invalid_shape(OP, t.shape(), format!("does not match {p:?}"));
        let [b, r, a] = *box_raw.dims() else { return Err(bad(box_raw)) };
        if p.reg_max == 0 || r != 4 * p.reg_max || a == 0 {
            return Err(bad(box_raw));
        }
        if cls_raw.dims() != [b, p.num_classes, a] || p.num_classes == 0 {
            return Err(bad(cls_raw));
        }
        if anchors.dims() != [3, a] {
            return Err(bad(anchors));
        }
        let be = box_raw.backend()?;
        match with_views_dyn(&[box_raw, cls_raw, anchors], |v| be.detect_candidates(v[0], v[1], v[2], p)) {
            Ok(s) => Ok(Some(Tensor::wrap(s, Shape::new(&[b, 1 + DETECT_RECORD_WORDS * a]), OP)?)),
            Err(e) if unsupported(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The class pass of one head level for candidate-first prediction (docs/MODEL.md §9.5,
    /// [`crate::backend::Backend::detect_class_sites`]): from `self` (`[B, nc, H, W]` logits) the `U32` tensors
    /// `(sites [B, 1 + cap], recs [B, 2·cap], field [B, 1 + min(9·cap, H·W)])`. `Ok(None)` when the device has no
    /// kernel for it. Not recorded by autograd.
    pub fn detect_class_sites(&self, p: &ClassSites) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        const OP: &str = "detect_class_sites";
        self.check_float(OP)?;
        let (b, nc, h, w) =
            self.dims4().map_err(|_| Error::invalid_shape(OP, self.shape(), "must be NCHW"))?;
        if nc == 0 || h * w == 0 || p.cap == 0 {
            return Err(Error::invalid_shape(OP, self.shape(), format!("empty level or no slots ({p:?})")));
        }
        let fcap = (9 * p.cap).min(h * w);
        let be = self.backend()?;
        match with_views_dyn(&[self], |v| be.detect_class_sites(v[0], p)) {
            Ok((s, r, f)) => Ok(Some((
                Tensor::wrap(s, Shape::new(&[b, 1 + p.cap]), OP)?,
                Tensor::wrap(r, Shape::new(&[b, 2 * p.cap]), OP)?,
                Tensor::wrap(f, Shape::new(&[b, 1 + fcap]), OP)?,
            ))),
            Err(e) if unsupported(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The boxes of every level's candidates ([`crate::backend::Backend::detect_decode_sites`]): `levels[l]` is
    /// `(raw, sites, recs, width, stride, start)`, `raw` the level's box logits at its slots (`[B, 4·reg_max,
    /// cap]`, as [`Tensor::conv2d_act_sites`] gives them), `sites` and `recs` from [`Tensor::detect_class_sites`].
    /// `U32 [B, 2 + 7·Σ cap]`: per image a record count, the overflowing levels' bits, then the records.
    /// `Ok(None)` when the device has no kernel for it. Not recorded by autograd.
    pub fn detect_decode_sites(
        levels: &[(&Tensor, &Tensor, &Tensor, usize, f32, usize)],
        reg_max: usize,
    ) -> Result<Option<Tensor>> {
        const OP: &str = "detect_decode_sites";
        let Some(&(first, ..)) = levels.first() else {
            return Err(Error::invalid_arg(OP, "no levels given"));
        };
        first.check_float(OP)?;
        let b = first.dims().first().copied().unwrap_or(0);
        let mut slots = 0;
        let mut ts = Vec::with_capacity(3 * levels.len());
        for &(raw, sites, recs, ..) in levels {
            raw.check_float_like(first, OP)?;
            let cap = sites.dims().get(1).map_or(0, |w| w.saturating_sub(1));
            if raw.dims() != [b, 4 * reg_max, cap] || cap == 0 || reg_max == 0 {
                return Err(Error::invalid_shape(OP, raw.shape(), "box logits must be [B, 4·reg_max, cap]"));
            }
            if sites.dtype() != DType::U32 || recs.dtype() != DType::U32 || recs.dims() != [b, 2 * cap] {
                return Err(Error::invalid_shape(OP, recs.shape(), "sites and records must be U32 lists"));
            }
            for t in [raw, sites, recs] {
                first.check_same_device(t, OP)?;
                ts.push(t);
            }
            slots += cap;
        }
        let be = first.backend()?;
        let r = with_views_dyn(&ts, |v| {
            let lv: Vec<SiteLevel<'_>> = levels
                .iter()
                .enumerate()
                .map(|(l, &(_, _, _, width, stride, start))| SiteLevel {
                    raw: v[3 * l],
                    sites: v[3 * l + 1],
                    recs: v[3 * l + 2],
                    width,
                    stride,
                    start,
                })
                .collect();
            be.detect_decode_sites(&lv, reg_max)
        });
        match r {
            Ok(s) => Ok(Some(Tensor::wrap(s, Shape::new(&[b, 2 + DETECT_RECORD_WORDS * slots]), OP)?)),
            Err(e) if unsupported(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
