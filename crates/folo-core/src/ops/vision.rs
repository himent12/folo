//! Detector pre- and post-processing behind optional `Backend` hooks: letterboxing `u8` images into the
//! network input, and decoding the head into above-threshold candidates. Both return `Ok(None)` on backends
//! without the kernels (the CPU), so callers keep their host implementations as the fallback and reference.

use super::with_views_dyn;
use crate::backend::{DETECT_RECORD_WORDS, DetectDecode, LetterboxImage};
use crate::{DType, Error, Result, Shape, Tensor};

fn unsupported(e: &Error) -> bool {
    matches!(e, Error::NotSupported { .. })
}

impl Tensor {
    /// Letterbox `U8` RGB images (`images[i]` is `[H_i, W_i, 3]`, all on one device) into an `F32
    /// [B, 3, OH, OW]` network input (see [`crate::backend::Backend::letterbox_rgb8`] for the exact sampling).
    /// `taps` (`U32`, same device) holds the tap arrays `geoms[i].taps` points at. `Ok(None)` when the
    /// device has no kernel for it. Not recorded by autograd.
    pub fn letterbox_rgb8(
        images: &[&Tensor],
        taps: &Tensor,
        geoms: &[LetterboxImage],
        out_hw: [usize; 2],
        pad_value: u8,
        scale: f32,
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
        let mut ts: Vec<&Tensor> = images.to_vec();
        ts.push(taps);
        let k = images.len();
        match with_views_dyn(&ts, |v| b.letterbox_rgb8(&v[..k], v[k], geoms, out_hw, pad_value, scale)) {
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
        for t in [box_raw, cls_raw, anchors] {
            t.check_f32(OP)?;
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
}
