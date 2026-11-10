//! FOLO's blocks as graph nodes, mirroring [`FoloModel::forward`]: C3k2, SPPF, C2PSA and attention, the
//! backbone and neck (layers 0–22), and the head's raw outputs (layer 23).

use folo_nn::ConvBnAct;

use super::graph::{Graph, attrs, ints};
use super::proto::AttrValue;
use crate::Result;
use crate::model::blocks::{Attention, Bottleneck, C2Psa, C3k, C3k2, C3k2Block, PsaBlock, Sppf};
use crate::model::head::LEVELS;
use crate::model::{FoloModel, HeadMode};

impl Graph {
    fn bottleneck(&mut self, p: &str, b: &Bottleneck, x: &str) -> Result<String> {
        let y = self.conv(&format!("{p}.cv1"), &b.cv1, x)?;
        let y = self.conv(&format!("{p}.cv2"), &b.cv2, &y)?;
        Ok(if b.residual { self.add(&format!("{p}.add"), x, &y) } else { y })
    }

    fn c3k(&mut self, p: &str, c: &C3k, x: &str) -> Result<String> {
        let mut a = self.conv(&format!("{p}.cv1"), &c.cv1, x)?;
        for (j, b) in c.blocks.iter().enumerate() {
            a = self.bottleneck(&format!("{p}.block{j}"), b, &a)?;
        }
        let b = self.conv(&format!("{p}.cv2"), &c.cv2, x)?;
        let cat = self.concat(&format!("{p}.cat"), &[&a, &b], 1);
        self.conv(&format!("{p}.cv3"), &c.cv3, &cat)
    }

    fn c3k2(&mut self, p: &str, c: &C3k2, x: &str) -> Result<String> {
        let y = self.conv(&format!("{p}.cv1"), &c.cv1, x)?;
        let half = (out_channels(&c.cv1) / 2) as i64;
        let mut parts = self.split(&format!("{p}.split"), &y, &[half, half], 1);
        for (i, b) in c.blocks.iter().enumerate() {
            let prev = parts.last().expect("two halves").clone();
            let q = format!("{p}.block{i}");
            let next = match b {
                C3k2Block::Bottleneck(b) => self.bottleneck(&q, b, &prev)?,
                C3k2Block::C3k(b) => self.c3k(&q, b, &prev)?,
            };
            parts.push(next);
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        let cat = self.concat(&format!("{p}.cat"), &refs, 1);
        self.conv(&format!("{p}.cv2"), &c.cv2, &cat)
    }

    fn sppf(&mut self, p: &str, s: &Sppf, x: &str) -> Result<String> {
        let y0 = self.conv(&format!("{p}.cv1"), &s.cv1, x)?;
        let (k, pad) = (s.k as i64, (s.k / 2) as i64);
        let a = || {
            attrs([("kernel_shape", ints(&[k, k])), ("strides", ints(&[1, 1])), ("pads", ints(&[pad; 4]))])
        };
        let y1 = self.node("MaxPool", &format!("{p}.pool1"), &[&y0], a());
        let y2 = self.node("MaxPool", &format!("{p}.pool2"), &[&y1], a());
        let y3 = self.node("MaxPool", &format!("{p}.pool3"), &[&y2], a());
        let cat = self.concat(&format!("{p}.cat"), &[&y0, &y1, &y2, &y3], 1);
        self.conv(&format!("{p}.cv2"), &s.cv2, &cat)
    }

    fn attention(&mut self, p: &str, at: &Attention, x: &str) -> Result<String> {
        let (heads, kd, hd) = (at.heads as i64, at.key_dim as i64, at.head_dim as i64);
        let qkv = self.conv(&format!("{p}.qkv"), &at.qkv, x)?;
        let qkv = self.reshape(&format!("{p}.qkv_heads"), &qkv, &[0, heads, 2 * kd + hd, -1]);
        let [q, k, v]: [String; 3] =
            self.split(&format!("{p}.qkv_split"), &qkv, &[kd, kd, hd], 2).try_into().expect("three parts");
        // A = softmax_j(scale · qᵀk): [B, heads, N, N].
        let qt = self.transpose(&format!("{p}.q_t"), &q, &[0, 1, 3, 2]);
        let logits = self.node("MatMul", &format!("{p}.qk"), &[&qt, &k], vec![]);
        let scale = self.float_init(&format!("{p}.scale"), &[], &[(kd as f64).powf(-0.5) as f32]);
        let logits = self.node("Mul", &format!("{p}.qk_scaled"), &[&logits, &scale], vec![]);
        let attn =
            self.node("Softmax", &format!("{p}.softmax"), &[&logits], attrs([("axis", AttrValue::Int(-1))]));
        // o = v · Aᵀ: [B, heads, hd, N], viewed as the input's [B, C, H, W].
        let at_t = self.transpose(&format!("{p}.attn_t"), &attn, &[0, 1, 3, 2]);
        let o = self.node("MatMul", &format!("{p}.av"), &[&v, &at_t], vec![]);
        let shape = self.node("Shape", &format!("{p}.shape"), &[x], vec![]);
        let o = self.node("Reshape", &format!("{p}.o"), &[&o, &shape], vec![]);
        let v = self.node("Reshape", &format!("{p}.v"), &[&v, &shape], vec![]);
        let pe = self.conv(&format!("{p}.pe"), &at.pe, &v)?;
        let sum = self.add(&format!("{p}.o_pe"), &o, &pe);
        self.conv(&format!("{p}.proj"), &at.proj, &sum)
    }

    fn psa_block(&mut self, p: &str, b: &PsaBlock, x: &str) -> Result<String> {
        let a = self.attention(&format!("{p}.attn"), &b.attn, x)?;
        let x = self.add(&format!("{p}.attn_add"), x, &a);
        let y = self.conv(&format!("{p}.ffn1"), &b.ffn1, &x)?;
        let y = self.conv(&format!("{p}.ffn2"), &b.ffn2, &y)?;
        Ok(self.add(&format!("{p}.ffn_add"), &x, &y))
    }

    fn c2psa(&mut self, p: &str, c: &C2Psa, x: &str) -> Result<String> {
        let y = self.conv(&format!("{p}.cv1"), &c.cv1, x)?;
        let half = (out_channels(&c.cv1) / 2) as i64;
        let [a, mut b]: [String; 2] =
            self.split(&format!("{p}.split"), &y, &[half, half], 1).try_into().expect("two halves");
        for (i, blk) in c.blocks.iter().enumerate() {
            b = self.psa_block(&format!("{p}.block{i}"), blk, &b)?;
        }
        let cat = self.concat(&format!("{p}.cat"), &[&a, &b], 1);
        self.conv(&format!("{p}.cv2"), &c.cv2, &cat)
    }

    fn upsample(&mut self, name: &str, x: &str) -> String {
        let scales = self.f32_init("const.f32.upsample_scales", &[4], &[1.0, 1.0, 2.0, 2.0]);
        let a = attrs([
            ("mode", AttrValue::Str("nearest".into())),
            ("coordinate_transformation_mode", AttrValue::Str("asymmetric".into())),
            ("nearest_mode", AttrValue::Str("floor".into())),
        ]);
        self.node("Resize", name, &[x, "", &scales], a)
    }

    /// Layers 0–22 (MODEL.md §4): the P3, P4 and P5 features.
    pub(super) fn body(&mut self, m: &FoloModel, x: &str) -> Result<[String; 3]> {
        let (b, n) = (&m.backbone, &m.neck);
        let y = self.conv("backbone.stem", &b.stem, x)?;
        let y = self.conv("backbone.down2", &b.down2, &y)?;
        let y = self.c3k2("backbone.stage2", &b.stage2, &y)?;
        let y = self.conv("backbone.down3", &b.down3, &y)?;
        let x4 = self.c3k2("backbone.stage3", &b.stage3, &y)?;
        let y = self.conv("backbone.down4", &b.down4, &x4)?;
        let x6 = self.c3k2("backbone.stage4", &b.stage4, &y)?;
        let y = self.conv("backbone.down5", &b.down5, &x6)?;
        let y = self.c3k2("backbone.stage5", &b.stage5, &y)?;
        let y = self.sppf("backbone.sppf", &b.sppf, &y)?;
        let x10 = self.c2psa("backbone.psa", &b.psa, &y)?;
        let y = self.upsample("neck.up11", &x10);
        let y = self.concat("neck.cat12", &[&y, &x6], 1);
        let x13 = self.c3k2("neck.td4", &n.td4, &y)?;
        let y = self.upsample("neck.up14", &x13);
        let y = self.concat("neck.cat15", &[&y, &x4], 1);
        let x16 = self.c3k2("neck.out3", &n.out3, &y)?;
        let y = self.conv("neck.down3", &n.down3, &x16)?;
        let y = self.concat("neck.cat18", &[&y, &x13], 1);
        let x19 = self.c3k2("neck.out4", &n.out4, &y)?;
        let y = self.conv("neck.down4", &n.down4, &x19)?;
        let y = self.concat("neck.cat21", &[&y, &x10], 1);
        let x22 = self.c3k2("neck.out5", &n.out5, &y)?;
        Ok([x16, x19, x22])
    }

    /// Layer 23: the raw head outputs `(box_raw, cls_raw)` of the predicting branch ([`FoloModel::head_mode`]),
    /// named so when they are graph outputs.
    pub(super) fn head(
        &mut self,
        m: &FoloModel,
        feats: &[String; 3],
        are_outputs: bool,
    ) -> Result<(String, String)> {
        let h = &m.head;
        let (r4, nc) = ((4 * h.reg_max) as i64, h.num_classes as i64);
        let (levels, root) = match (&h.o2o, m.head_mode()) {
            (Some(o2o), HeadMode::O2o) => (o2o, "head.o2o"),
            _ => (&h.levels, "head"),
        };
        let (mut boxes, mut classes) = (vec![], vec![]);
        for ((l, name), x) in levels.iter().zip(LEVELS).zip(feats) {
            let p = format!("{root}.box.{name}");
            let bb = &l.box_branch;
            let y = self.conv(&format!("{p}.conv0"), &bb.conv0, x)?;
            let y = self.conv(&format!("{p}.conv1"), &bb.conv1, &y)?;
            let y = self.conv2d(&format!("{p}.out"), &bb.out, &y)?;
            boxes.push(self.reshape(&format!("{p}.flat"), &y, &[0, r4, -1]));
            let p = format!("{root}.cls.{name}");
            let cb = &l.cls_branch;
            let y = self.conv(&format!("{p}.dw0"), &cb.dw0, x)?;
            let y = self.conv(&format!("{p}.pw0"), &cb.pw0, &y)?;
            let y = self.conv(&format!("{p}.dw1"), &cb.dw1, &y)?;
            let y = self.conv(&format!("{p}.pw1"), &cb.pw1, &y)?;
            let y = self.conv2d(&format!("{p}.out"), &cb.out, &y)?;
            classes.push(self.reshape(&format!("{p}.flat"), &y, &[0, nc, -1]));
        }
        let names = if are_outputs { ["box_raw", "cls_raw"] } else { ["head.box_raw", "head.cls_raw"] };
        let br: Vec<&str> = boxes.iter().map(String::as_str).collect();
        let cr: Vec<&str> = classes.iter().map(String::as_str).collect();
        let box_raw = self.concat(names[0], &br, 2);
        let cls_raw = self.concat(names[1], &cr, 2);
        Ok((box_raw, cls_raw))
    }
}

fn out_channels(c: &ConvBnAct) -> usize {
    c.conv.weight.dims()[0]
}
