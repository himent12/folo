//! The flip consensus's exact map (docs/MODEL.md §8.9): the head outputs of a forward on the input mirrored along
//! its width, moved onto the anchors of the unmirrored input.

use folo_core::{Device, Result, Tensor};

use crate::decode::Anchors;

/// For each anchor `a` of `anchors`, the anchor of the mirrored input that sees the same place: in every level the
/// anchor `(y, x)` takes `(y, W_l − 1 − x)`, `W_l` that level's width. A permutation and its own inverse. It is exact
/// because a level's anchor centres `x + 0.5` (grid units) mirror to `W_l − x − 0.5` when the input width is
/// `W_l · stride` (every training size: a multiple of 32).
pub fn flip_permutation(anchors: &Anchors) -> Vec<usize> {
    let mut perm = Vec::with_capacity(anchors.len());
    for l in &anchors.levels {
        for y in 0..l.height {
            let row = l.start + y * l.width;
            perm.extend((0..l.width).rev().map(|x| row + x));
        }
    }
    perm
}

/// The map of one input shape on one device: the reversed column order of the input, the anchor permutation of
/// [`flip_permutation`] and the side swap `l t r b → r t l b` of the box outputs (left and right trade places).
#[derive(Clone, Debug)]
pub struct FlipMap {
    columns: Tensor,
    anchors: Tensor,
    sides: Tensor,
    num_anchors: usize,
}

impl FlipMap {
    /// The map of an input `width` pixels wide whose head has `anchors`, with its index tensors on `device`. Errors
    /// unless every level is `width / stride` cells wide (the input width a multiple of every stride).
    pub fn new(anchors: &Anchors, width: usize, device: Device) -> Result<Self> {
        if let Some(l) = anchors.levels.iter().find(|l| l.width * l.stride as usize != width) {
            return Err(super::invalid(format!(
                "the flip map needs an input width that is a multiple of every stride; width {width}, stride {} has \
                 {} cells",
                l.stride, l.width
            )));
        }
        let index = |v: Vec<i64>| -> Result<Tensor> {
            let n = v.len();
            Tensor::from_vec(v, [n])?.to_device(device)
        };
        let perm = flip_permutation(anchors);
        Ok(FlipMap {
            columns: index((0..width as i64).rev().collect())?,
            anchors: index(perm.iter().map(|&i| i as i64).collect())?,
            sides: index(vec![2, 1, 0, 3])?,
            num_anchors: perm.len(),
        })
    }

    /// The input `[B, C, H, W]` mirrored along `W`.
    pub fn flip_input(&self, x: &Tensor) -> Result<Tensor> {
        x.index_select(3, &self.columns)
    }

    /// Per-anchor outputs `[B, C, A]` (class logits or probabilities) moved to the mirrored anchors.
    pub fn map_cls(&self, t: &Tensor) -> Result<Tensor> {
        self.check(t, 3)?;
        t.index_select(2, &self.anchors)
    }

    /// Box outputs moved to the mirrored anchors with left and right swapped: `[B, 4, reg_max, A]`, or
    /// `[B, 4 · reg_max, A]` (the head's layout, channel `side · reg_max + bin`), kept in the shape given.
    pub fn map_box(&self, t: &Tensor) -> Result<Tensor> {
        match t.rank() {
            4 => {
                self.check(t, 4)?;
                t.index_select(1, &self.sides)?.index_select(3, &self.anchors)
            }
            _ => {
                let (b, ch, a) = t.dims3()?;
                if ch % 4 != 0 {
                    return Err(super::invalid(format!(
                        "box outputs {} are not [B, 4 · reg_max, A]",
                        t.shape()
                    )));
                }
                self.map_box(&t.reshape([b, 4, ch / 4, a])?)?.reshape([b, ch, a])
            }
        }
    }

    fn check(&self, t: &Tensor, rank: usize) -> Result<()> {
        if t.rank() != rank || t.dims()[rank - 1] != self.num_anchors {
            return Err(super::invalid(format!(
                "{} is not a rank-{rank} tensor over the map's {} anchors",
                t.shape(),
                self.num_anchors
            )));
        }
        Ok(())
    }
}
