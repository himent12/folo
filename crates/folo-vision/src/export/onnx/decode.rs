//! `--decode` in the graph: the anchor grid (constant when spatial dimensions are fixed, or computed from the
//! feature maps when `axes.hw` is enabled), the DFL decode of MODEL.md §6.4 (with optional FOLO v2 `iou_out`
//! geometric score fusion), and the end-to-end top-k selection of an `o2o` head (§6.6).

use super::graph::{Graph, attrs, ints};
use super::proto::{AttrValue, DataType};
use super::{DynamicAxes, OnnxOptions};
use crate::Result;
use crate::decode::make_anchors;
use crate::model::head::LEVELS;

impl Graph {
    /// The anchor points `[2, A]` (x row, then y row, in grid units) and strides `[1, A]`: constants when spatial
    /// dimensions are fixed (`!axes.hw`, including batch-only dynamic graphs), computed from the feature-map
    /// shapes when `axes.hw` is true.
    fn anchors(
        &mut self,
        strides: &[u32; 3],
        opts: &OnnxOptions,
        axes: DynamicAxes,
        feats: &[String; 3],
    ) -> (String, String) {
        if !axes.hw {
            let an = make_anchors(opts.height, opts.width, strides, 0.5);
            let mut pts: Vec<f32> = an.points.iter().map(|p| p[0]).collect();
            pts.extend(an.points.iter().map(|p| p[1]));
            let a = an.len();
            let points = self.float_init("decode.anchors", &[2, a], &pts);
            let strides = self.float_init("decode.strides", &[1, a], &an.strides);
            return (points, strides);
        }
        let (zero, one) = (self.i64_const(&[0], true), self.i64_const(&[1], true));
        let (two, four) = (self.i64_const(&[2], false), self.i64_const(&[4], false));
        let to = AttrValue::Int(self.elem.code());
        let half = self.float_init("decode.half", &[], &[0.5]);
        let (mut pts, mut strides_out) = (vec![], vec![]);
        for ((x, name), &s) in feats.iter().zip(LEVELS).zip(strides) {
            let p = format!("decode.{name}");
            let shape = self.node("Shape", &format!("{p}.shape"), &[x], vec![]);
            let hw = self.node("Slice", &format!("{p}.hw"), &[&shape, &two, &four], vec![]);
            let [h, w]: [String; 2] = [2i64, 3].map(|i| {
                let idx = self.i64_const(&[i], true);
                self.node(
                    "Gather",
                    &format!("{p}.dim{i}"),
                    &[&shape, &idx],
                    attrs([("axis", AttrValue::Int(0))]),
                )
            });
            // Cell centres along each axis: (0.5, 1.5, …).
            let mut centres = |len: &str, axis: &str| {
                let r = self.node("Range", &format!("{p}.range_{axis}"), &[&zero, len, &one], vec![]);
                let r =
                    self.node("Cast", &format!("{p}.range_{axis}_f"), &[&r], vec![("to".into(), to.clone())]);
                self.node("Add", &format!("{p}.centres_{axis}"), &[&r, &half], vec![])
            };
            let (cx, cy) = (centres(&w, "x"), centres(&h, "y"));
            let ax0 = self.i64_const(&[0], false);
            let ax1 = self.i64_const(&[1], false);
            let cx = self.node("Unsqueeze", &format!("{p}.cx_row"), &[&cx, &ax0], vec![]); // [1, W]
            let cy = self.node("Unsqueeze", &format!("{p}.cy_col"), &[&cy, &ax1], vec![]); // [H, 1]
            let gx = self.node("Expand", &format!("{p}.grid_x"), &[&cx, &hw], vec![]);
            let gy = self.node("Expand", &format!("{p}.grid_y"), &[&cy, &hw], vec![]);
            let gx = self.reshape(&format!("{p}.flat_x"), &gx, &[1, -1]);
            let gy = self.reshape(&format!("{p}.flat_y"), &gy, &[1, -1]);
            pts.push(self.concat(&format!("{p}.points"), &[&gx, &gy], 0));
            let sv = self.float_init(&format!("{p}.stride"), &[1, 1], &[s as f32]);
            let n = self.node("Shape", &format!("{p}.count"), &[&gx], vec![]);
            strides_out.push(self.node("Expand", &format!("{p}.strides"), &[&sv, &n], vec![]));
        }
        let pr: Vec<&str> = pts.iter().map(String::as_str).collect();
        let sr: Vec<&str> = strides_out.iter().map(String::as_str).collect();
        (self.concat("decode.anchors", &pr, 1), self.concat("decode.strides", &sr, 1))
    }

    /// MODEL.md §6.4 in the graph: the decoded `[B, 4 + nc, A]` (boxes in input pixels, then class scores).
    /// When `iou_raw` (`[B, 1, A]`) is supplied (FOLO v2 with `iou_head = true`), class probabilities are fused
    /// via geometric mean `sqrt(sigmoid(cls_raw) * sigmoid(iou_raw))` (`DEFAULT_IOU_FUSION_ALPHA = 0.5`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode(
        &mut self,
        strides: &[u32; 3],
        reg_max: usize,
        opts: &OnnxOptions,
        axes: DynamicAxes,
        feats: &[String; 3],
        box_raw: &str,
        cls_raw: &str,
        iou_raw: Option<&str>,
    ) -> Result<String> {
        let r = reg_max;
        let logits = self.reshape("decode.dfl_logits", box_raw, &[0, 4, r as i64, -1]); // [B, 4, R, A]
        let prob =
            self.node("Softmax", "decode.dfl_softmax", &[&logits], attrs([("axis", AttrValue::Int(2))]));
        let bins: Vec<f32> = (0..r).map(|i| i as f32).collect();
        let proj = self.float_init("decode.dfl_bins", &[1, 1, r, 1], &bins);
        let weighted = self.node("Mul", "decode.dfl_weighted", &[&prob, &proj], vec![]);
        let axis = self.i64_const(&[2], false);
        let dist = self.node(
            "ReduceSum",
            "decode.dfl_dist",
            &[&weighted, &axis],
            attrs([("keepdims", AttrValue::Int(0))]),
        ); // [B, 4, A]
        let [lt, rb]: [String; 2] =
            self.split("decode.ltrb", &dist, &[2, 2], 1).try_into().expect("two halves");
        let (points, strides) = self.anchors(strides, opts, axes, feats);
        let x1y1 = self.node("Sub", "decode.x1y1_cells", &[&points, &lt], vec![]);
        let x2y2 = self.node("Add", "decode.x2y2_cells", &[&points, &rb], vec![]);
        let x1y1 = self.node("Mul", "decode.x1y1", &[&x1y1, &strides], vec![]);
        let x2y2 = self.node("Mul", "decode.x2y2", &[&x2y2, &strides], vec![]);
        let boxes = self.concat("decode.boxes", &[&x1y1, &x2y2], 1); // [B, 4, A]
        let mut scores = self.node("Sigmoid", "decode.scores", &[cls_raw], vec![]);
        if let Some(iou_logits) = iou_raw {
            let p_iou = self.node("Sigmoid", "decode.iou_scores", &[iou_logits], vec![]);
            if self.elem == DataType::Float16 {
                let s32 = self.cast("decode.scores_f32", &scores, DataType::Float);
                let i32 = self.cast("decode.iou_f32", &p_iou, DataType::Float);
                let prod = self.mul("decode.iou_prod", &s32, &i32);
                let fused = self.node("Sqrt", "decode.iou_fused_f32", &[&prod], vec![]);
                scores = self.cast("decode.iou_fused", &fused, DataType::Float16);
            } else {
                let prod = self.mul("decode.iou_prod", &scores, &p_iou);
                scores = self.node("Sqrt", "decode.iou_fused", &[&prod], vec![]);
            }
        }
        Ok(self.concat("decode.channels", &[&boxes, &scores], 1)) // [B, 4 + nc, A]
    }

    /// MODEL.md §6.6's selection in the graph, from the decoded `all: [B, 4 + nc, A]`: the end-to-end
    /// `detections: [B, K, 6]` output (`x1, y1, x2, y2, score, class`, by descending score). `TopK` puts equal
    /// values in index order, the tie rule of [`crate::nms::topk_detections`]. An `F16` graph selects in `F32`
    /// (the CPU `TopK` and `ReduceMax` of onnxruntime take no `float16`) and its rows stay `F32`, so class ids above
    /// 2048 stay exact.
    pub(super) fn topk(&mut self, all: &str, nc: usize, opts: &OnnxOptions) {
        let half = self.elem == DataType::Float16;
        let to_f32 = || vec![("to".to_string(), AttrValue::Int(DataType::Float.code()))];
        let dets = self.transpose("e2e.anchor_major", all, &[0, 2, 1]); // [B, A, 4 + nc]
        let dets = if half { self.node("Cast", "e2e.f32", &[&dets], to_f32()) } else { dets };
        let [_, scores]: [String; 2] =
            self.split("e2e.split_anchors", &dets, &[4, nc as i64], 2).try_into().expect("two parts");
        let best = self.node(
            "ReduceMax",
            "e2e.best_score",
            &[&scores],
            attrs([("axes", ints(&[2])), ("keepdims", AttrValue::Int(0))]),
        ); // [B, A]
        // K = min(max_det, A), as a 1-element int64 tensor.
        let max_det = self.i64_const(&[opts.max_det as i64], false);
        let shape = self.node("Shape", "e2e.best_shape", &[&best], vec![]);
        let (one, two) = (self.i64_const(&[1], false), self.i64_const(&[2], false));
        let count = self.node("Slice", "e2e.anchor_count", &[&shape, &one, &two], vec![]);
        let k = self.node("Min", "e2e.k", &[&count, &max_det], vec![]);
        let top = |g: &mut Self, name: &str, x: &str| -> [String; 2] {
            let a = attrs([
                ("axis", AttrValue::Int(-1)),
                ("largest", AttrValue::Int(1)),
                ("sorted", AttrValue::Int(1)),
            ]);
            g.node_n("TopK", name, &[x, &k], a, 2).try_into().expect("values and indices")
        };
        let gather = |g: &mut Self, name: &str, data: &str, idx: &str| -> String {
            let ax = g.i64_const(&[2], false);
            let idx = g.node("Unsqueeze", &format!("{name}.index"), &[idx, &ax], vec![]); // [B, K, 1]
            g.node("GatherND", name, &[data, &idx], attrs([("batch_dims", AttrValue::Int(1))]))
        };
        let [_, anchors] = top(self, "e2e.top_anchors", &best); // [B, K]
        let sel = gather(self, "e2e.selected", &dets, &anchors); // [B, K, 4 + nc]
        let [sel_boxes, sel_scores]: [String; 2] =
            self.split("e2e.split_selected", &sel, &[4, nc as i64], 2).try_into().expect("two parts");
        let flat = self.reshape("e2e.pairs", &sel_scores, &[0, -1]); // [B, K · nc]
        let [score, pair] = top(self, "e2e.top_pairs", &flat); // [B, K]
        let ncs = self.i64_const(&[nc as i64], true);
        let slot = self.node("Div", "e2e.pair_anchor", &[&pair, &ncs], vec![]);
        let class = self.node("Mod", "e2e.pair_class", &[&pair, &ncs], vec![]);
        let class = self.node("Cast", "e2e.class_f32", &[&class], to_f32());
        let boxes = gather(self, "e2e.boxes", &sel_boxes, &slot); // [B, K, 4]
        let ax = self.i64_const(&[2], false);
        let score = self.node("Unsqueeze", "e2e.score_col", &[&score, &ax], vec![]);
        let class = self.node("Unsqueeze", "e2e.class_col", &[&class, &ax], vec![]);
        self.concat("detections", &[&boxes, &score, &class], 2);
    }
}
