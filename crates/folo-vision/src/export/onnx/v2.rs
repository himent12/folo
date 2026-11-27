//! ONNX export (opset 17) of a fused [`FoloV2Model`] (`folo2-n/s/m/l/x`, docs/MODEL.md §11).
//!
//! Mirrors [`FoloV2Model::forward`] op for op:
//! - [`StemV2`]: optional `SpaceToDepth(blocksize=2)` + fused `Conv`.
//! - [`WsQARepConv3x3`]: fused dense `3×3` `Conv` + activation (`{p}.conv.{weight, bias}`).
//! - [`FoloBlock`]: `cv1` → `Slice` second half → sequential `WsQARepConv3x3` (`rep0..rep{m-1}`) with
//!   optional mid-tap (`m >= 3`) → `Concat` → `cv2`.
//! - [`HyperContext`]: `av_proj` → `Split` + dynamic `Reshape` → `F32` `Softmax` / `MatMul` / `ffn1` + `SiLU` +
//!   `ffn2` / residual `Add` / `ReduceMean` (when `FullPadGate` is enabled) → dynamic `Shape` + `Reshape` →
//!   `out_proj` + residual `Add`.
//! - [`FullPadGate`]: `F32` `feat * (1 + tanh(proj(g_scene)))` with dynamic batch `Reshape([0, c, 1, 1])`.
//! - [`LazyAlignHead`]: shared `BoxBranchV2` (with optional `iou_out`), `ClsBranchV2`, and `o2o`
//!   [`PeakSuppress`] (`Sigmoid(gate)`, `F32` `Softplus(alpha_raw)`, `MaxPool(3×3)`).

use std::collections::BTreeMap;

use folo_nn::Module;

use super::graph::{Graph, attrs, ints};
use super::proto::{AttrValue, DataType, Dim};
use super::{DynamicAxes, INPUT, OPSET, OnnxExport, OnnxOptions, OnnxValue};
use crate::model::HeadMode;
use crate::model::head::LEVELS;
use crate::model::v2::{
    ClsBranchV2, DEFAULT_IOU_FUSION_ALPHA, FoloBlock, FoloV2Model, FullPadGate, HyperContext, PeakSuppress,
    StemV2, WsQARepConv3x3,
};
use crate::{Error, Result};

impl FoloV2Model {
    /// Export this FOLO v2 model as an ONNX graph (opset 17), resolving dynamic axes from `opts.dynamic`.
    /// The model must be fused ([`FoloV2Model::fuse`]); weights are read from any device.
    pub fn to_onnx(&self, opts: &OnnxOptions) -> Result<OnnxExport> {
        export_v2(self, opts)
    }

    /// Export this FOLO v2 model as an ONNX graph with independent [`DynamicAxes`] (`batch` and `hw`),
    /// overriding `opts.dynamic`.
    pub fn to_onnx_with_axes(&self, opts: &OnnxOptions, axes: DynamicAxes) -> Result<OnnxExport> {
        export_v2_with_axes(self, opts, axes)
    }
}

/// Build the ONNX model of a fused [`FoloV2Model`], deriving dynamic axes from `opts.dynamic`.
pub fn export_v2(model: &FoloV2Model, opts: &OnnxOptions) -> Result<OnnxExport> {
    export_v2_with_axes(model, opts, opts.dynamic_axes())
}

/// Build the ONNX model of a fused [`FoloV2Model`] with explicit [`DynamicAxes`] (`batch` and `hw`),
/// overriding `opts.dynamic`.
pub fn export_v2_with_axes(model: &FoloV2Model, opts: &OnnxOptions, axes: DynamicAxes) -> Result<OnnxExport> {
    if !model.is_fused() {
        return Err(Error::Unsupported(
            "ONNX export needs a fused model (WS-QARepConv3x3 and Conv+BN folded): call FoloV2Model::fuse first"
                .into(),
        ));
    }
    let cfg = model.config();
    let e2e = model.head_mode() == HeadMode::O2o;
    if e2e && !model.has_o2o() {
        return Err(Error::Config(format!("ONNX export: {} has no one-to-one head branch", model.arch())));
    }
    let strides = model.strides();
    let stride = *strides.iter().max().expect("three strides");
    opts.validate(stride)?;

    let elem = if opts.half { DataType::Float16 } else { DataType::Float };
    let mut g = Graph::new(elem);
    let (bdim, hdim, wdim) = axes.input_dims(opts);
    let input = OnnxValue {
        name: INPUT.into(),
        dtype: elem.name(),
        shape: vec![bdim.clone(), Dim::Value(cfg.in_channels), hdim, wdim],
    };

    let feats = g.body_v2(model, INPUT)?;
    let (box_raw, cls_raw, iou_raw) = g.head_v2(model, &feats, !opts.decode)?;
    let (reg_max, nc) = (model.head.reg_max, model.num_classes());
    let a = if axes.hw {
        Dim::Param("anchors".into())
    } else {
        Dim::Value(model.anchors(opts.height as usize, opts.width as usize).len())
    };
    let outputs = if opts.decode && e2e {
        let k = match &a {
            Dim::Value(a) => Dim::Value(opts.max_det.min(*a)),
            Dim::Param(_) => Dim::Param("max_det".into()),
        };
        let all = g.decode(&strides, reg_max, opts, axes, &feats, &box_raw, &cls_raw, iou_raw.as_deref())?;
        g.topk(&all, nc, opts);
        // Always f32: an f16 class id is exact only up to 2048.
        let dtype = DataType::Float.name();
        vec![OnnxValue { name: "detections".into(), dtype, shape: vec![bdim, k, Dim::Value(6)] }]
    } else if opts.decode {
        let all = g.decode(&strides, reg_max, opts, axes, &feats, &box_raw, &cls_raw, iou_raw.as_deref())?;
        g.transpose("detections", &all, &[0, 2, 1]);
        vec![OnnxValue {
            name: "detections".into(),
            dtype: elem.name(),
            shape: vec![bdim, a, Dim::Value(4 + nc)],
        }]
    } else {
        let mut outs = vec![
            OnnxValue {
                name: box_raw,
                dtype: elem.name(),
                shape: vec![bdim.clone(), Dim::Value(4 * reg_max), a.clone()],
            },
            OnnxValue {
                name: cls_raw,
                dtype: elem.name(),
                shape: vec![bdim.clone(), Dim::Value(nc), a.clone()],
            },
        ];
        if let Some(iou_name) = iou_raw {
            outs.push(OnnxValue { name: iou_name, dtype: elem.name(), shape: vec![bdim, Dim::Value(1), a] });
        }
        outs
    };

    // Every model tensor must have been consumed except the unused classification branch (`head.cls.*` in o2o
    // mode, `head.o2o.*` in o2m mode; `head.box.*` is shared by both).
    let dropped = |n: &str| {
        if e2e { n.starts_with("head.cls.") } else { n.starts_with("head.o2o.") }
    };
    for t in model.named_tensors() {
        if !g.used.contains(&t.name) && !dropped(&t.name) {
            return Err(Error::Unsupported(format!("ONNX export: tensor `{}` is not mapped", t.name)));
        }
    }

    let metadata = metadata_v2(model, opts, axes, &outputs);
    let ops = g.nodes.iter().map(|n| n.op.to_string()).collect();
    let initializers = g.inits.len();
    let bytes = g.finish(&input, &outputs, &metadata);
    Ok(OnnxExport { bytes, inputs: vec![input], outputs, ops, initializers, metadata })
}

impl Graph {
    fn space_to_depth_2x2(&mut self, name: &str, x: &str) -> String {
        self.node("SpaceToDepth", name, &[x], attrs([("blocksize", AttrValue::Int(2))]))
    }

    fn stem_v2(&mut self, p: &str, s: &StemV2, x: &str) -> Result<String> {
        let inp = if s.spd { self.space_to_depth_2x2(&format!("{p}.spd"), x) } else { x.to_string() };
        self.conv(&format!("{p}.conv"), &s.conv, &inp)
    }

    fn ws_qarep(&mut self, p: &str, c: &WsQARepConv3x3, x: &str) -> Result<String> {
        let Some(conv) = &c.fused else {
            return Err(Error::Unsupported(format!(
                "ONNX export: `{p}` is not fused (WsQARepConv3x3 still has branches/BatchNorm)"
            )));
        };
        if c.bn.is_some() || c.branches.is_some() {
            return Err(Error::Unsupported(format!(
                "ONNX export: `{p}` still has unfused branches or BatchNorm"
            )));
        }
        let y = self.conv2d(&format!("{p}.conv"), conv, x)?;
        self.act(p, c.act, &y)
    }

    fn folo_block(&mut self, p: &str, blk: &FoloBlock, xs: &[&str]) -> Result<String> {
        let inp = if xs.len() == 1 { xs[0].to_string() } else { self.concat(&format!("{p}.in_cat"), xs, 1) };
        let y = self.conv(&format!("{p}.cv1"), &blk.cv1, &inp)?;
        let c_mid = blk.c_mid as i64;
        let m = blk.reps.len();
        let mid_idx = if m >= 3 { Some(m / 2 - 1) } else { None };
        let mut cur = self.slice_axis(&format!("{p}.half"), &y, 1, c_mid, 2 * c_mid);
        let mut mid_tap = None;
        for (j, rep) in blk.reps.iter().enumerate() {
            cur = self.ws_qarep(&format!("{p}.rep{j}"), rep, &cur)?;
            if mid_idx == Some(j) {
                mid_tap = Some(cur.clone());
            }
        }
        let cat = match &mid_tap {
            Some(mid) => self.concat(&format!("{p}.cat"), &[&y, mid, &cur], 1),
            None => self.concat(&format!("{p}.cat"), &[&y, &cur], 1),
        };
        self.conv(&format!("{p}.cv2"), &blk.cv2, &cat)
    }

    fn hyper_context(
        &mut self,
        p: &str,
        hc: &HyperContext,
        x: &str,
        need_scene: bool,
    ) -> Result<(String, Option<String>)> {
        let half = self.elem == DataType::Float16;
        let (m, cg) = (hc.m_edges as i64, hc.c_g as i64);
        let av = self.conv(&format!("{p}.av_proj"), &hc.av_proj, x)?;
        let [a_sp, v_sp]: [String; 2] =
            self.split(&format!("{p}.av_split"), &av, &[m, cg], 1).try_into().expect("two parts");
        let a_flat = self.reshape(&format!("{p}.a_flat"), &a_sp, &[0, m, -1]);
        let v_flat = self.reshape(&format!("{p}.v_flat"), &v_sp, &[0, cg, -1]);
        let a_f32 = if half { self.cast(&format!("{p}.a_f32"), &a_flat, DataType::Float) } else { a_flat };
        let v_f32 = if half { self.cast(&format!("{p}.v_f32"), &v_flat, DataType::Float) } else { v_flat };

        let a_gather =
            self.node("Softmax", &format!("{p}.a_gather"), &[&a_f32], attrs([("axis", AttrValue::Int(-1))]));
        let a_bcast =
            self.node("Softmax", &format!("{p}.a_bcast"), &[&a_f32], attrs([("axis", AttrValue::Int(1))]));
        let a_gather_t = self.transpose(&format!("{p}.a_gather_t"), &a_gather, &[0, 2, 1]);
        let e = self.node("MatMul", &format!("{p}.gather"), &[&v_f32, &a_gather_t], vec![]); // [B, C_g, M]
        let e_t = self.transpose(&format!("{p}.e_t"), &e, &[0, 2, 1]); // [B, M, C_g] in F32

        let ffn1_in =
            if half { self.cast(&format!("{p}.ffn1_in"), &e_t, DataType::Float16) } else { e_t.clone() };
        let h1 = self.linear(&format!("{p}.ffn1"), &hc.ffn1, &ffn1_in)?;
        let h1 = if half { self.cast(&format!("{p}.h1_f32"), &h1, DataType::Float) } else { h1 };
        let h1 = self.act(&format!("{p}.ffn_act"), folo_nn::Activation::Silu, &h1)?;

        let ffn2_in = if half { self.cast(&format!("{p}.ffn2_in"), &h1, DataType::Float16) } else { h1 };
        let e_ffn = self.linear(&format!("{p}.ffn2"), &hc.ffn2, &ffn2_in)?;
        let e_ffn = if half { self.cast(&format!("{p}.e_ffn_f32"), &e_ffn, DataType::Float) } else { e_ffn };

        let e_prime_t = self.add(&format!("{p}.e_res"), &e_t, &e_ffn); // [B, M, C_g] in F32
        let g_scene = need_scene.then(|| {
            self.node(
                "ReduceMean",
                &format!("{p}.g_scene"),
                &[&e_prime_t],
                attrs([("axes", ints(&[1])), ("keepdims", AttrValue::Int(0))]),
            )
        });
        let e_prime = self.transpose(&format!("{p}.e_prime"), &e_prime_t, &[0, 2, 1]); // [B, C_g, M]
        let y_flat = self.node("MatMul", &format!("{p}.bcast"), &[&e_prime, &a_bcast], vec![]); // [B, C_g, N]
        // `v_sp` already has exact dynamic shape `[B, C_g, H, W]`.
        let v_shape = self.node("Shape", &format!("{p}.v_shape"), &[&v_sp], vec![]);
        let y_ctx = self.node("Reshape", &format!("{p}.y_ctx"), &[&y_flat, &v_shape], vec![]);
        let y_ctx =
            if half { self.cast(&format!("{p}.y_ctx_f16"), &y_ctx, DataType::Float16) } else { y_ctx };
        let proj = self.conv(&format!("{p}.out_proj"), &hc.out_proj, &y_ctx)?;
        let out = self.add(&format!("{p}.out_add"), x, &proj);
        Ok((out, g_scene))
    }

    fn fullpad_gate(&mut self, p: &str, gate: &FullPadGate, feat: &str, g_scene: &str) -> Result<String> {
        let half = self.elem == DataType::Float16;
        let g_in = if half {
            self.cast(&format!("{p}.g_in"), g_scene, DataType::Float16)
        } else {
            g_scene.to_string()
        };
        let raw = self.linear(&format!("{p}.proj"), &gate.proj, &g_in)?;
        let raw_f32 = if half { self.cast(&format!("{p}.raw_f32"), &raw, DataType::Float) } else { raw };
        let th = self.node("Tanh", &format!("{p}.tanh"), &[&raw_f32], vec![]);
        let one = self.f32_init(&format!("{p}.one"), &[], &[1.0]);
        let g = self.add(&format!("{p}.scale"), &th, &one);
        let g_4d = self.reshape(&format!("{p}.scale_4d"), &g, &[0, gate.c_tgt as i64, 1, 1]);
        Ok(if half {
            let feat_f32 = self.cast(&format!("{p}.feat_f32"), feat, DataType::Float);
            let out_f32 = self.mul(&format!("{p}.mul_f32"), &feat_f32, &g_4d);
            self.cast(&format!("{p}.out"), &out_f32, DataType::Float16)
        } else {
            self.mul(&format!("{p}.mul"), feat, &g_4d)
        })
    }

    fn body_v2(&mut self, m: &FoloV2Model, x: &str) -> Result<[String; 3]> {
        let (b, n) = (&m.backbone, &m.neck);
        let y = self.stem_v2("backbone.stem", &b.stem, x)?;
        let mut p2 = self.conv("backbone.down2", &b.down2, &y)?;
        for (j, rep) in b.stage2.iter().enumerate() {
            p2 = self.ws_qarep(&format!("backbone.stage2.rep{j}"), rep, &p2)?;
        }
        let y = self.conv("backbone.down3", &b.down3, &p2)?;
        let p3 = self.folo_block("backbone.stage3", &b.stage3, &[&y])?;
        let y = self.conv("backbone.down4", &b.down4, &p3)?;
        let p4 = self.folo_block("backbone.stage4", &b.stage4, &[&y])?;
        let y = self.conv("backbone.down5", &b.down5, &p4)?;
        let y = self.folo_block("backbone.stage5", &b.stage5, &[&y])?;
        let y = self.sppf("backbone.sppf", &b.sppf, &y)?;
        let need_scene = n.pad_p3.is_some() || n.pad_p4.is_some();
        let (p5, g_scene) = self.hyper_context("backbone.hyper", &b.hyper, &y, need_scene)?;

        let up11 = self.upsample("neck.up11", &p5);
        let td4 = self.folo_block("neck.td4", &n.td4, &[&up11, &p4])?;
        let up14 = self.upsample("neck.up14", &td4);
        let p2_s2d = self.space_to_depth_2x2("neck.p2_spd", &p2);
        let p2_spd = self.conv("neck.p2_proj", &n.p2_proj, &p2_s2d)?;
        let out3_raw = self.folo_block("neck.out3", &n.out3, &[&up14, &p3, &p2_spd])?;

        let nd3 = self.conv("neck.down3", &n.down3, &out3_raw)?;
        let out4_raw = self.folo_block("neck.out4", &n.out4, &[&nd3, &td4])?;
        let nd4 = self.conv("neck.down4", &n.down4, &out4_raw)?;
        let out5 = self.folo_block("neck.out5", &n.out5, &[&nd4, &p5])?;

        let p3_head = match &n.pad_p3 {
            Some(gate) => self.fullpad_gate(
                "neck.pad_p3",
                gate,
                &out3_raw,
                g_scene.as_deref().expect("g_scene present"),
            )?,
            None => out3_raw,
        };
        let p4_head = match &n.pad_p4 {
            Some(gate) => self.fullpad_gate(
                "neck.pad_p4",
                gate,
                &out4_raw,
                g_scene.as_deref().expect("g_scene present"),
            )?,
            None => out4_raw,
        };
        Ok([p3_head, p4_head, out5])
    }

    fn cls_branch_v2(&mut self, p: &str, cb: &ClsBranchV2, x: &str) -> Result<(String, String)> {
        let h = match &cb.align {
            Some(a) => self.conv(&format!("{p}.align"), a, x)?,
            None => x.to_string(),
        };
        let feat = self.ws_qarep(&format!("{p}.conv0"), &cb.conv0, &h)?;
        let logits = self.conv2d(&format!("{p}.out"), &cb.out, &feat)?;
        Ok((logits, feat))
    }

    fn peak_suppress(&mut self, p: &str, sup: &PeakSuppress, logits: &str, feat: &str) -> Result<String> {
        let half = self.elem == DataType::Float16;
        let g = self.conv2d(&format!("{p}.gate"), &sup.gate, feat)?;
        let g = self.node("Sigmoid", &format!("{p}.gate_sig"), &[&g], vec![]);
        let alpha_raw = self.weight(&format!("{p}.alpha_raw"), &sup.alpha_raw)?;
        let alpha = if half {
            let a32 = self.cast(&format!("{p}.alpha_f32"), &alpha_raw, DataType::Float);
            let sp = self.node("Softplus", &format!("{p}.alpha_sp"), &[&a32], vec![]);
            self.cast(&format!("{p}.alpha_f16"), &sp, DataType::Float16)
        } else {
            self.node("Softplus", &format!("{p}.alpha_sp"), &[&alpha_raw], vec![])
        };
        let alpha = self.reshape(&format!("{p}.alpha_4d"), &alpha, &[1, sup.num_classes as i64, 1, 1]);
        let z_max = self.node(
            "MaxPool",
            &format!("{p}.maxpool"),
            &[logits],
            attrs([("kernel_shape", ints(&[3, 3])), ("strides", ints(&[1, 1])), ("pads", ints(&[1; 4]))]),
        );
        let diff = self.sub(&format!("{p}.diff"), &z_max, logits);
        let pen = self.mul(&format!("{p}.pen_g"), &diff, &g);
        let pen = self.mul(&format!("{p}.pen"), &pen, &alpha);
        Ok(self.sub(&format!("{p}.out"), logits, &pen))
    }

    fn head_v2(
        &mut self,
        m: &FoloV2Model,
        feats: &[String; 3],
        are_outputs: bool,
    ) -> Result<(String, String, Option<String>)> {
        let h = &m.head;
        let (r4, nc) = ((4 * h.reg_max) as i64, h.num_classes as i64);
        let e2e = m.head_mode() == HeadMode::O2o;
        let (mut boxes, mut classes, mut ious) = (vec![], vec![], vec![]);
        for (i, ((l, name), x)) in h.levels.iter().zip(LEVELS).zip(feats).enumerate() {
            let bp = format!("head.box.{name}");
            let bb = &l.box_branch;
            let y = self.ws_qarep(&format!("{bp}.conv0"), &bb.conv0, x)?;
            let feat = self.ws_qarep(&format!("{bp}.conv1"), &bb.conv1, &y)?;
            let bx = self.conv2d(&format!("{bp}.out"), &bb.out, &feat)?;
            boxes.push(self.reshape(&format!("{bp}.flat"), &bx, &[0, r4, -1]));
            if let Some(iou_out) = &bb.iou_out {
                let iou = self.conv2d(&format!("{bp}.iou_out"), iou_out, &feat)?;
                ious.push(self.reshape(&format!("{bp}.iou_flat"), &iou, &[0, 1, -1]));
            }
            if e2e {
                let ol = &h.o2o_cls.as_ref().expect("o2o branch")[i];
                let op = format!("head.o2o.{name}");
                let (logits, cls_feat) = self.cls_branch_v2(&format!("{op}.cls"), &ol.cls_branch, x)?;
                let z_o2o =
                    self.peak_suppress(&format!("{op}.suppress"), &ol.suppress, &logits, &cls_feat)?;
                classes.push(self.reshape(&format!("{op}.flat"), &z_o2o, &[0, nc, -1]));
            } else {
                let cp = format!("head.cls.{name}");
                let (logits, _) = self.cls_branch_v2(&cp, &l.cls_branch, x)?;
                classes.push(self.reshape(&format!("{cp}.flat"), &logits, &[0, nc, -1]));
            }
        }
        let names = if are_outputs {
            ["box_raw", "cls_raw", "iou_raw"]
        } else {
            ["head.box_raw", "head.cls_raw", "head.iou_raw"]
        };
        let br: Vec<&str> = boxes.iter().map(String::as_str).collect();
        let cr: Vec<&str> = classes.iter().map(String::as_str).collect();
        let box_raw = self.concat(names[0], &br, 2);
        let cls_raw = self.concat(names[1], &cr, 2);
        let iou_raw = if h.has_iou_head {
            let ir: Vec<&str> = ious.iter().map(String::as_str).collect();
            Some(self.concat(names[2], &ir, 2))
        } else {
            None
        };
        Ok((box_raw, cls_raw, iou_raw))
    }
}

fn metadata_v2(
    model: &FoloV2Model,
    opts: &OnnxOptions,
    axes: DynamicAxes,
    outputs: &[OnnxValue],
) -> BTreeMap<String, String> {
    let mut m = opts.metadata.clone();
    let cfg = model.config();
    let s = |v: serde_json::Value| v.to_string();
    let e2e = model.head_mode() == HeadMode::O2o;
    let kind = match (opts.decode, e2e) {
        (false, _) => "raw",
        (true, false) => "decoded",
        (true, true) => "end2end",
    };
    let classes: Vec<String> = (0..cfg.num_classes).map(|i| i.to_string()).collect();
    let names: serde_json::Map<String, serde_json::Value> =
        classes.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone().into())).collect();
    m.entry("names".into()).or_insert_with(|| serde_json::Value::Object(names).to_string());
    m.entry("folo.classes".into()).or_insert_with(|| s(serde_json::json!(classes)));
    m.insert("folo.arch".into(), model.arch());
    m.insert("folo.dtype".into(), if opts.half { "F16" } else { "F32" }.into());
    m.insert("folo.fused".into(), "true".into());
    m.insert("folo.head".into(), model.head_mode().as_str().into());
    m.insert("folo.imgsz".into(), opts.height.max(opts.width).to_string());
    m.insert("folo.v2.plus".into(), cfg.plus.to_string());
    m.insert("folo.v2.spd_stem".into(), cfg.spd_stem.to_string());
    m.insert("folo.v2.fullpad".into(), cfg.fullpad.to_string());
    m.insert("folo.v2.iou_head".into(), cfg.iou_head.to_string());
    m.insert("folo.onnx.outputs".into(), kind.into());
    m.insert("folo.onnx.head".into(), model.head_mode().as_str().into());
    if opts.decode && e2e {
        m.insert("folo.onnx.max_det".into(), opts.max_det.to_string());
    }
    m.insert("folo.onnx.output_names".into(), s(outputs.iter().map(|o| o.name.clone()).collect()));
    m.insert("folo.onnx.imgsz".into(), s(serde_json::json!([opts.height, opts.width])));
    m.insert("folo.onnx.batch".into(), if axes.batch { "dynamic".into() } else { opts.batch.to_string() });
    m.insert("folo.onnx.dynamic".into(), axes.is_any().to_string());
    m.insert("folo.onnx.dynamic_batch".into(), axes.batch.to_string());
    m.insert("folo.onnx.dynamic_hw".into(), axes.hw.to_string());
    m.insert("folo.onnx.opset".into(), OPSET.to_string());
    m.insert("folo.onnx.strides".into(), s(serde_json::json!(model.strides())));
    m.insert("folo.onnx.reg_max".into(), model.head.reg_max.to_string());
    if cfg.iou_head {
        m.insert("folo.onnx.iou_head".into(), "true".into());
        if opts.decode {
            m.insert("folo.onnx.iou_fusion_alpha".into(), DEFAULT_IOU_FUSION_ALPHA.to_string());
        }
    }
    m
}
