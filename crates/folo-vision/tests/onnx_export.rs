//! ONNX export tests for independent [`DynamicAxes`] (`batch` vs `hw`) and FOLO v2-Iso ([`FoloV2Model`]).
//!
//! Includes both deterministic structural/metadata checks (always run) and real `onnx.checker` + `onnxruntime`
//! forward-pass parity checks across multiple batch sizes and stride-32 resolutions when the Python ONNX venv
//! (`FOLO_ONNX_PYTHON` or `D:/folo-venvs/onnx`) is available.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::{Module, init};
use folo_test_support::TestDir;
use folo_vision::Error;
use folo_vision::export::onnx::proto::Dim;
use folo_vision::export::onnx::{DynamicAxes, OnnxOptions, export_v2, export_v2_with_axes};
use folo_vision::model::v2::{
    DEFAULT_IOU_FUSION_ALPHA, FoloV2Config, FoloV2Model, HeadOutputV2, WsQARepConv3x3,
};
use folo_vision::model::{FoloModel, HeadMode, HeadOutput, ModelConfig};
use folo_vision::nms::{NmsConfig, topk_detections};
use serde_json::{Value, json};

/// Element-wise relative tolerance `(max, mean)` for `f32` raw head tensors.
const TOL_F32_RAW: (f64, f64) = (2e-4, 1e-5);
/// Element-wise relative tolerance `(max, mean)` for `f32` decoded pixel boxes (stride-32 coordinates near 0
/// amplify ~1e-5 grid-unit FMA differences by 32× while mean error stays ~1e-6).
const TOL_F32_DEC: (f64, f64) = (5e-4, 1e-5);
/// Element-wise relative tolerance `(max, mean)` for `f16` raw/e2e graphs (11-bit mantissa vs FOLO's `f32`).
const TOL_F16_RAW: (f64, f64) = (1e-1, 5e-3);
/// Element-wise relative tolerance `(max, mean)` for `f16` decoded dense boxes (stride-32 pixel edges near 0
/// can shift by ~0.4 px from 11-bit mantissa rounding while mean error stays ~1e-3).
const TOL_F16_DEC: (f64, f64) = (0.5, 5e-3);

fn python() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FOLO_ONNX_PYTHON") {
        return Some(p.into());
    }
    let venv =
        std::env::var_os("FOLO_ONNX_VENV").map_or_else(|| PathBuf::from("D:/folo-venvs/onnx"), PathBuf::from);
    ["Scripts/python.exe", "bin/python"].iter().map(|p| venv.join(p)).find(|p| p.is_file())
}

fn required() -> bool {
    std::env::var("FOLO_ONNX_REQUIRED").is_ok_and(|v| !v.is_empty() && v != "0")
}

#[test]
fn v1_independent_dynamic_axes_shapes_and_ops() {
    let mut m = FoloModel::new(ModelConfig::new("n", 4).unwrap()).unwrap();
    m.fuse().unwrap();
    let opts = OnnxOptions { decode: true, batch: 2, height: 96, width: 128, ..OnnxOptions::new(96) };
    let expected_a = (96 / 8) * (128 / 8) + (96 / 16) * (128 / 16) + (96 / 32) * (128 / 32);

    // Static default via `to_onnx` vs `DynamicAxes::STATIC` are bit-identical.
    let s1 = m.to_onnx(&opts).unwrap();
    let s2 = m.to_onnx_with_axes(&opts, DynamicAxes::STATIC).unwrap();
    assert_eq!(s1.bytes, s2.bytes);
    assert_eq!(s1.inputs[0].shape, [2, 3, 96, 128].map(Dim::Value));
    assert_eq!(s1.outputs[0].shape, [2, expected_a, 8].map(Dim::Value));
    assert!(!s1.ops.iter().any(|op| op == "Range"));
    assert_eq!(s1.metadata["folo.onnx.dynamic"], "false");
    assert_eq!(s1.metadata["folo.onnx.dynamic_batch"], "false");
    assert_eq!(s1.metadata["folo.onnx.dynamic_hw"], "false");

    // Batch-only dynamic: symbolic `"batch"`, fixed H/W and constant anchor grid (no `Range`).
    let b = m.to_onnx_with_axes(&opts, DynamicAxes::BATCH).unwrap();
    assert_eq!(
        b.inputs[0].shape,
        [Dim::Param("batch".into()), Dim::Value(3), Dim::Value(96), Dim::Value(128)]
    );
    assert_eq!(b.outputs[0].shape, [Dim::Param("batch".into()), Dim::Value(expected_a), Dim::Value(8)]);
    assert!(!b.ops.iter().any(|op| op == "Range"), "batch-only dynamic keeps constant anchors");
    assert_eq!(b.metadata["folo.onnx.batch"], "dynamic");
    assert_eq!(b.metadata["folo.onnx.dynamic"], "true");
    assert_eq!(b.metadata["folo.onnx.dynamic_batch"], "true");
    assert_eq!(b.metadata["folo.onnx.dynamic_hw"], "false");

    // H/W-only dynamic: fixed batch `2`, symbolic `"height"` / `"width"` / `"anchors"`, dynamic anchor grid.
    let hw = m.to_onnx_with_axes(&opts, DynamicAxes::HW).unwrap();
    assert_eq!(
        hw.inputs[0].shape,
        [Dim::Value(2), Dim::Value(3), Dim::Param("height".into()), Dim::Param("width".into())]
    );
    assert_eq!(hw.outputs[0].shape, [Dim::Value(2), Dim::Param("anchors".into()), Dim::Value(8)]);
    assert!(hw.ops.iter().any(|op| op == "Range"), "hw dynamic computes anchor grid in graph");
    assert_eq!(hw.metadata["folo.onnx.batch"], "2");
    assert_eq!(hw.metadata["folo.onnx.dynamic"], "true");
    assert_eq!(hw.metadata["folo.onnx.dynamic_batch"], "false");
    assert_eq!(hw.metadata["folo.onnx.dynamic_hw"], "true");

    // Full dynamic matches `OnnxOptions { dynamic: true, .. }`.
    let d1 = m.to_onnx(&OnnxOptions { dynamic: true, ..opts.clone() }).unwrap();
    let d2 = m.to_onnx_with_axes(&opts, DynamicAxes::ALL).unwrap();
    assert_eq!(d1.bytes, d2.bytes);
    assert_eq!(
        d1.inputs[0].shape,
        [Dim::Param("batch".into()), Dim::Value(3), Dim::Param("height".into()), Dim::Param("width".into())]
    );
    assert_eq!(
        d1.outputs[0].shape,
        [Dim::Param("batch".into()), Dim::Param("anchors".into()), Dim::Value(8)]
    );
}

#[test]
fn v2_export_rejects_unfused_models_and_blocks() {
    let m = FoloV2Model::new(FoloV2Config::new("n", 3).unwrap()).unwrap();
    let err = export_v2(&m, &OnnxOptions::new(64)).err().unwrap();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(err.to_string().contains("fused"), "{err}");

    let mut m = FoloV2Model::new(FoloV2Config::new("n", 3).unwrap()).unwrap();
    m.fuse().unwrap();
    let mut rng = Rng::seed(1);
    let mut init_ctx = folo_vision::model::blocks::Init { rng: &mut rng, bn_eps: 1e-3, bn_momentum: 0.03 };
    m.head.levels[0].box_branch.conv0 = WsQARepConv3x3::new(64, 48, &mut init_ctx);
    let err = m.to_onnx(&OnnxOptions::new(64)).err().unwrap().to_string();
    assert!(err.contains("head.box.p3.conv0") && err.contains("fused"), "{err}");
}

#[test]
fn v2_export_covers_all_scales_and_variants() {
    for scale in ["n", "s", "m", "l", "x"] {
        let mut m = FoloV2Model::new(FoloV2Config::new(scale, 5).unwrap()).unwrap();
        m.fuse().unwrap();
        let x = m.to_onnx(&OnnxOptions::new(64)).unwrap();
        let a = 8 * 8 + 4 * 4 + 2 * 2;
        assert_eq!(x.inputs[0].shape, [1, 3, 64, 64].map(Dim::Value));
        assert_eq!(x.outputs.len(), 2);
        assert_eq!(x.outputs[0].shape, [1, 64, a].map(Dim::Value));
        assert_eq!(x.outputs[1].shape, [1, 5, a].map(Dim::Value));
        assert_eq!(x.metadata["folo.arch"], format!("folo2-{scale}"));
        assert!(x.ops.iter().any(|op| op == "SpaceToDepth"), "folo2-{scale} uses SpaceToDepth");
        assert!(x.ops.iter().any(|op| op == "ReduceMean"), "folo2-{scale} FullPadGate uses g_scene");
    }

    // Variant matrix on `folo2-n`: spd_stem=false, fullpad=false, plus=true, iou_head=true, o2o.
    let cfg = FoloV2Config::new("n", 4)
        .unwrap()
        .with_spd_stem(false)
        .with_fullpad(false)
        .with_plus(true)
        .with_iou_head(true)
        .with_head(HeadMode::O2o);
    let mut m = FoloV2Model::new(cfg).unwrap();
    m.fuse().unwrap();
    assert_eq!(m.head_mode(), HeadMode::O2o);

    // Raw o2o with iou_head emits [box_raw, cls_raw, iou_raw], no ReduceMean when fullpad=false.
    let raw = m.to_onnx(&OnnxOptions::new(96)).unwrap();
    let a = 12 * 12 + 6 * 6 + 3 * 3;
    assert_eq!(raw.outputs.len(), 3);
    assert_eq!(raw.outputs[2].name, "iou_raw");
    assert_eq!(raw.outputs[2].shape, [1, 1, a].map(Dim::Value));
    assert_eq!(raw.metadata["folo.onnx.iou_head"], "true");
    assert_eq!(raw.metadata["folo.v2.fullpad"], "false");
    assert!(!raw.ops.iter().any(|op| op == "ReduceMean"), "no dead ReduceMean when fullpad=false");
    assert!(raw.ops.iter().any(|op| op == "Softplus"), "o2o PeakSuppress uses Softplus");

    // Decoded o2o with iou_head fuses scores via Sqrt and outputs end-to-end [B, K, 6] in F32 even for F16.
    let e2e = export_v2_with_axes(
        &m,
        &OnnxOptions { decode: true, half: true, max_det: 25, ..OnnxOptions::new(96) },
        DynamicAxes::BATCH,
    )
    .unwrap();
    assert_eq!(e2e.outputs.len(), 1);
    assert_eq!(e2e.outputs[0].dtype, "float");
    assert_eq!(e2e.outputs[0].shape, [Dim::Param("batch".into()), Dim::Value(25), Dim::Value(6)]);
    assert!(e2e.ops.iter().any(|op| op == "Sqrt"), "iou_head decode fuses scores via Sqrt");
    assert_eq!(e2e.metadata["folo.onnx.iou_fusion_alpha"], DEFAULT_IOU_FUSION_ALPHA.to_string());
    assert_eq!(e2e.metadata["folo.onnx.outputs"], "end2end");

    // Switch to o2m on the same model: drops o2o tensors cleanly.
    m.set_head_mode(HeadMode::O2m).unwrap();
    let dec =
        m.to_onnx_with_axes(&OnnxOptions { decode: true, ..OnnxOptions::new(96) }, DynamicAxes::HW).unwrap();
    assert_eq!(dec.outputs[0].shape, [Dim::Value(1), Dim::Param("anchors".into()), Dim::Value(8)]);
    assert!(!dec.ops.iter().any(|op| op == "Softplus"));
    assert_eq!(dec.metadata["folo.onnx.head"], "o2m");
}

// ---------------------------------------------------------------------------------------------------------
// Real onnx.checker + onnxruntime numerical parity tests
// ---------------------------------------------------------------------------------------------------------

fn calibrated_v1(scale: &str, nc: usize, seed: u64, head: &str) -> FoloModel {
    let mut cfg = ModelConfig::new(scale, nc).unwrap();
    cfg.head = head.into();
    let base = cfg.clone();
    cfg.bn_momentum = 0.999;
    let cal = FoloModel::new(cfg).unwrap();
    let mut rng = Rng::seed(seed);
    for t in cal.named_tensors() {
        let n = &t.name;
        let (lo, hi) = if n.ends_with("bn.weight") {
            (0.4, 0.5)
        } else if n.ends_with("bn.bias") {
            (-0.1, 0.1)
        } else if n.ends_with("out.bias") {
            (-1.0, 1.0)
        } else {
            continue;
        };
        init::copy_(&t.tensor, &Tensor::rand_uniform(t.tensor.shape().clone(), lo, hi, &mut rng)).unwrap();
    }
    {
        let _g = no_grad();
        cal.forward(&Tensor::rand_uniform([4, 3, 128, 128], 0.0, 1.0, &mut rng)).unwrap();
    }
    let m = FoloModel::new(base).unwrap();
    let src: HashMap<String, Tensor> = cal.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    for t in m.named_tensors() {
        init::copy_(&t.tensor, &src[&t.name]).unwrap();
    }
    m
}

fn calibrated_v2(cfg: FoloV2Config, seed: u64) -> FoloV2Model {
    let mut cal_cfg = cfg.clone();
    cal_cfg.bn_momentum = 0.999;
    let cal = FoloV2Model::with_rng(cal_cfg, &mut Rng::seed(seed)).unwrap();
    let mut rng = Rng::seed(seed + 1000);
    for t in cal.named_tensors() {
        let n = &t.name;
        let (lo, hi) = if n.ends_with("bn.weight") {
            (0.4, 0.5)
        } else if n.ends_with("bn.bias") {
            (-0.1, 0.1)
        } else if n.ends_with("out.bias") || n.ends_with("iou_out.bias") {
            (-1.0, 1.0)
        } else if n.contains("pad_p") && n.ends_with(".weight") {
            // Non-zero FullPadGate weights so the FiLM gate actively modulates head inputs!
            (-0.15, 0.15)
        } else if n.contains("pad_p") && n.ends_with(".bias") {
            (-0.1, 0.1)
        } else {
            continue;
        };
        init::copy_(&t.tensor, &Tensor::rand_uniform(t.tensor.shape().clone(), lo, hi, &mut rng)).unwrap();
    }
    {
        let _g = no_grad();
        cal.forward(&Tensor::rand_uniform([4, 3, 128, 128], 0.0, 1.0, &mut rng)).unwrap();
    }
    let m = FoloV2Model::with_rng(cfg, &mut Rng::seed(seed)).unwrap();
    let src: HashMap<String, Tensor> = cal.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    for t in m.named_tensors() {
        init::copy_(&t.tensor, &src[&t.name]).unwrap();
    }
    m
}

fn dump(dir: &Path, name: &str, t: &Tensor) -> Value {
    let bytes: Vec<u8> = t.to_vec::<f32>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(name), bytes).unwrap();
    json!({ "file": name, "shape": t.dims() })
}

fn ref_v1(out: &HeadOutput, head: HeadMode, decode: bool, max_det: usize, dir: &Path, tag: &str) -> Value {
    let _g = no_grad();
    if decode && head == HeadMode::O2o {
        let nc = out.num_classes;
        let k = max_det.min(out.num_anchors());
        let pool_k = (k + 16).min(out.num_anchors());
        let select = |lim: usize| -> Tensor {
            let nms = NmsConfig {
                conf: f32::NEG_INFINITY,
                max_det: lim,
                multi_label: true,
                ..NmsConfig::default()
            };
            let mut rows = Vec::new();
            for (boxes, scores) in out.decode_images().unwrap() {
                let d = topk_detections(&boxes, &scores, nc, &nms);
                assert_eq!(d.len(), lim, "{tag}: {lim} rows");
                for d in d {
                    rows.extend([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.score, d.class as f32]);
                }
            }
            Tensor::from_vec(rows, [out.batch(), lim, 6]).unwrap()
        };
        let mut spec = dump(dir, &format!("{tag}-detections.f32"), &select(k));
        spec["match"] = json!("rows");
        spec["pool"] = dump(dir, &format!("{tag}-pool.f32"), &select(pool_k));
        json!({ "detections": spec })
    } else if decode {
        json!({ "detections": dump(dir, &format!("{tag}-detections.f32"), &out.decode().unwrap()) })
    } else {
        json!({
            "box_raw": dump(dir, &format!("{tag}-box_raw.f32"), &out.box_raw),
            "cls_raw": dump(dir, &format!("{tag}-cls_raw.f32"), &out.cls_raw),
        })
    }
}

fn ref_v2(out: &HeadOutputV2, decode: bool, max_det: usize, dir: &Path, tag: &str) -> Value {
    let _g = no_grad();
    let alpha = if out.iou_raw.is_some() { DEFAULT_IOU_FUSION_ALPHA } else { 0.0 };
    if decode && out.mode == HeadMode::O2o {
        let nc = out.num_classes;
        let k = max_det.min(out.num_anchors());
        let pool_k = (k + 16).min(out.num_anchors());
        let select = |lim: usize| -> Tensor {
            let nms = NmsConfig {
                conf: f32::NEG_INFINITY,
                max_det: lim,
                multi_label: true,
                ..NmsConfig::default()
            };
            let mut rows = Vec::new();
            for (boxes, scores) in out.decode_images(alpha).unwrap() {
                let d = topk_detections(&boxes, &scores, nc, &nms);
                assert_eq!(d.len(), lim, "{tag}: {lim} rows");
                for d in d {
                    rows.extend([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.score, d.class as f32]);
                }
            }
            Tensor::from_vec(rows, [out.batch(), lim, 6]).unwrap()
        };
        let mut spec = dump(dir, &format!("{tag}-detections.f32"), &select(k));
        spec["match"] = json!("rows");
        spec["pool"] = dump(dir, &format!("{tag}-pool.f32"), &select(pool_k));
        json!({ "detections": spec })
    } else if decode {
        let nc = out.num_classes;
        let a = out.num_anchors();
        let mut flat = Vec::with_capacity(out.batch() * a * (4 + nc));
        for (boxes, scores) in out.decode_images(alpha).unwrap() {
            for (i, b) in boxes.iter().enumerate() {
                flat.extend([b.x1, b.y1, b.x2, b.y2]);
                flat.extend_from_slice(&scores[i * nc..(i + 1) * nc]);
            }
        }
        let dets = Tensor::from_vec(flat, [out.batch(), a, 4 + nc]).unwrap();
        json!({ "detections": dump(dir, &format!("{tag}-detections.f32"), &dets) })
    } else {
        let mut map = serde_json::Map::new();
        map.insert("box_raw".into(), dump(dir, &format!("{tag}-box_raw.f32"), &out.box_raw));
        map.insert("cls_raw".into(), dump(dir, &format!("{tag}-cls_raw.f32"), &out.cls_raw));
        if let Some(iou) = &out.iou_raw {
            map.insert("iou_raw".into(), dump(dir, &format!("{tag}-iou_raw.f32"), iou));
        }
        Value::Object(map)
    }
}

#[test]
fn dynamic_axes_and_v2_match_folo_in_onnxruntime() {
    let Some(py) = python() else {
        let why = "no ONNX Python (set FOLO_ONNX_PYTHON or create the venv at D:/folo-venvs/onnx)";
        assert!(!required(), "FOLO_ONNX_REQUIRED is set but there is {why}");
        eprintln!("skipped: {why}");
        return;
    };
    let dir = TestDir::new("vision-onnx-v2");
    let mut cases = Vec::new();
    let mut rng = Rng::seed(77);

    // --- Part 1: FoloModel independent DynamicAxes (BATCH, HW, ALL) ---
    let mut v1 = calibrated_v1("n", 5, 11, "o2m");
    v1.fuse().unwrap();
    let v1_batch_xs = [
        Tensor::rand_uniform([1, 3, 128, 128], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([2, 3, 128, 128], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([4, 3, 128, 128], 0.0, 1.0, &mut rng),
    ];
    let v1_hw_xs = [
        Tensor::rand_uniform([2, 3, 96, 96], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([2, 3, 320, 320], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([2, 3, 480, 640], 0.0, 1.0, &mut rng),
    ];
    let v1_all_xs = [
        Tensor::rand_uniform([1, 3, 96, 128], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([4, 3, 64, 96], 0.0, 1.0, &mut rng),
    ];
    for (name, axes, opts, xs) in [
        (
            "v1-dyn-batch-decoded",
            DynamicAxes::BATCH,
            OnnxOptions { decode: true, ..OnnxOptions::new(128) },
            v1_batch_xs.as_slice(),
        ),
        (
            "v1-dyn-hw-decoded",
            DynamicAxes::HW,
            OnnxOptions { decode: true, batch: 2, ..OnnxOptions::new(128) },
            v1_hw_xs.as_slice(),
        ),
        (
            "v1-dyn-all-decoded",
            DynamicAxes::ALL,
            OnnxOptions { decode: true, ..OnnxOptions::new(128) },
            v1_all_xs.as_slice(),
        ),
    ] {
        let exp = v1.to_onnx_with_axes(&opts, axes).unwrap();
        exp.write_to(dir.join(format!("{name}.onnx"))).unwrap();
        let inputs: Vec<Value> =
            xs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("{name}-in{i}.f32"), x)).collect();
        let expected: Vec<Value> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let _g = no_grad();
                let out = v1.forward(x).unwrap();
                ref_v1(&out, v1.head_mode(), opts.decode, opts.max_det, &dir, &format!("{name}-{i}"))
            })
            .collect();
        let tol = if opts.decode { TOL_F32_DEC } else { TOL_F32_RAW };
        cases.push(json!({
            "name": name,
            "model": format!("{name}.onnx"),
            "dtype": "float32",
            "tolerance": tol.0,
            "mean_tolerance": tol.1,
            "inputs": inputs,
            "expected": expected,
        }));
    }

    // --- Part 2: FoloV2Model full parity across variants, dtypes, and dynamic axes ---
    let mut v2_default = calibrated_v2(FoloV2Config::new("n", 5).unwrap(), 21);
    v2_default.fuse().unwrap();
    // Sanity-check that outputs depend non-trivially on the input image (so parity is never vacuous).
    {
        let _g = no_grad();
        let x0 = Tensor::rand_uniform([1, 3, 96, 96], 0.0, 1.0, &mut rng);
        let a = v2_default.forward(&x0).unwrap();
        let b = v2_default.forward(&x0.rsub_scalar(1.0).unwrap()).unwrap();
        let mean_diff = |p: &Tensor, q: &Tensor| {
            let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
            p.iter().zip(&q).map(|(u, v)| (u - v).abs()).sum::<f32>() / p.len() as f32
        };
        assert!(mean_diff(&a.box_raw, &b.box_raw) > 0.05 && mean_diff(&a.cls_raw, &b.cls_raw) > 0.05);
    }
    let v2_xs = [
        Tensor::rand_uniform([1, 3, 128, 128], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([2, 3, 96, 160], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([4, 3, 64, 64], 0.0, 1.0, &mut rng),
    ];
    for (name, axes, opts, xs) in [
        ("v2-n-f32-raw-static", DynamicAxes::STATIC, OnnxOptions::new(128), &v2_xs[..1]),
        (
            "v2-n-f16-raw-static",
            DynamicAxes::STATIC,
            OnnxOptions { half: true, ..OnnxOptions::new(128) },
            &v2_xs[..1],
        ),
        (
            "v2-n-f32-decoded-dyn-all",
            DynamicAxes::ALL,
            OnnxOptions { decode: true, ..OnnxOptions::new(128) },
            &v2_xs[..],
        ),
        (
            "v2-n-f16-decoded-dyn-all",
            DynamicAxes::ALL,
            OnnxOptions { decode: true, half: true, ..OnnxOptions::new(128) },
            &v2_xs[..2],
        ),
    ] {
        let exp = v2_default.to_onnx_with_axes(&opts, axes).unwrap();
        exp.write_to(dir.join(format!("{name}.onnx"))).unwrap();
        let inputs: Vec<Value> =
            xs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("{name}-in{i}.f32"), x)).collect();
        let expected: Vec<Value> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let _g = no_grad();
                let out = v2_default.forward(x).unwrap();
                ref_v2(&out, opts.decode, opts.max_det, &dir, &format!("{name}-{i}"))
            })
            .collect();
        let tol = match (opts.half, opts.decode) {
            (true, true) => TOL_F16_DEC,
            (true, false) => TOL_F16_RAW,
            (false, true) => TOL_F32_DEC,
            (false, false) => TOL_F32_RAW,
        };
        cases.push(json!({
            "name": name,
            "model": format!("{name}.onnx"),
            "dtype": if opts.half { "float16" } else { "float32" },
            "tolerance": tol.0,
            "mean_tolerance": tol.1,
            "inputs": inputs,
            "expected": expected,
        }));
    }

    // FOLO v2 with `o2o` (`PeakSuppress`), `iou_head`, `plus=true`, `spd_stem=false`, `fullpad=false`.
    let mut v2_o2o_iou = calibrated_v2(
        FoloV2Config::new("n", 5)
            .unwrap()
            .with_head(HeadMode::O2o)
            .with_iou_head(true)
            .with_plus(true)
            .with_spd_stem(false)
            .with_fullpad(false),
        31,
    );
    v2_o2o_iou.fuse().unwrap();
    let v2_o2o_xs = [
        Tensor::rand_uniform([1, 3, 128, 128], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([2, 3, 128, 128], 0.0, 1.0, &mut rng),
    ];
    for (name, axes, opts, xs) in [
        ("v2-o2o-iou-f32-raw", DynamicAxes::BATCH, OnnxOptions::new(128), &v2_o2o_xs[..]),
        (
            "v2-o2o-iou-f32-e2e-dyn-batch",
            DynamicAxes::BATCH,
            OnnxOptions { decode: true, max_det: 20, ..OnnxOptions::new(128) },
            &v2_o2o_xs[..],
        ),
        (
            "v2-o2o-iou-f16-e2e-dyn-all",
            DynamicAxes::ALL,
            OnnxOptions { decode: true, half: true, max_det: 20, ..OnnxOptions::new(128) },
            &v2_xs[..2],
        ),
    ] {
        let exp = v2_o2o_iou.to_onnx_with_axes(&opts, axes).unwrap();
        exp.write_to(dir.join(format!("{name}.onnx"))).unwrap();
        let inputs: Vec<Value> =
            xs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("{name}-in{i}.f32"), x)).collect();
        let expected: Vec<Value> = xs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let _g = no_grad();
                let out = v2_o2o_iou.forward(x).unwrap();
                ref_v2(&out, opts.decode, opts.max_det, &dir, &format!("{name}-{i}"))
            })
            .collect();
        let tol = if opts.half {
            TOL_F16_RAW
        } else if opts.decode {
            TOL_F32_DEC
        } else {
            TOL_F32_RAW
        };
        cases.push(json!({
            "name": name,
            "model": format!("{name}.onnx"),
            "dtype": if opts.half { "float16" } else { "float32" },
            "tolerance": tol.0,
            "mean_tolerance": tol.1,
            "inputs": inputs,
            "expected": expected,
        }));
    }

    // Also test `folo2-s` with `iou_head=true` in `o2m` decoded mode + `DynamicAxes::HW`.
    let mut v2_s = calibrated_v2(FoloV2Config::new("s", 5).unwrap().with_iou_head(true), 41);
    v2_s.fuse().unwrap();
    let v2_s_xs = [
        Tensor::rand_uniform([1, 3, 96, 128], 0.0, 1.0, &mut rng),
        Tensor::rand_uniform([1, 3, 128, 96], 0.0, 1.0, &mut rng),
    ];
    let opts = OnnxOptions { decode: true, ..OnnxOptions::new(128) };
    let exp = v2_s.to_onnx_with_axes(&opts, DynamicAxes::HW).unwrap();
    exp.write_to(dir.join("v2-s-iou-f32-decoded-dyn-hw.onnx")).unwrap();
    let inputs: Vec<Value> =
        v2_s_xs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("v2-s-in{i}.f32"), x)).collect();
    let expected: Vec<Value> = v2_s_xs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let _g = no_grad();
            let out = v2_s.forward(x).unwrap();
            ref_v2(&out, true, opts.max_det, &dir, &format!("v2-s-{i}"))
        })
        .collect();
    cases.push(json!({
        "name": "v2-s-iou-f32-decoded-dyn-hw",
        "model": "v2-s-iou-f32-decoded-dyn-hw.onnx",
        "dtype": "float32",
        "tolerance": TOL_F32_DEC.0,
        "mean_tolerance": TOL_F32_DEC.1,
        "inputs": inputs,
        "expected": expected,
    }));

    let manifest = dir.join("manifest.json");
    std::fs::write(&manifest, serde_json::to_vec_pretty(&json!({ "cases": cases })).unwrap()).unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity/onnx_check.py");
    let o = Command::new(&py).arg(&script).arg(&manifest).output().unwrap();
    let stdout = String::from_utf8_lossy(&o.stdout);
    let report: Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("onnx_check.py printed no report ({e}): {stdout}\n{}", String::from_utf8_lossy(&o.stderr))
    });
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    assert!(o.status.success() && report["ok"] == true, "ONNX parity failed: {report}");
}
