//! `folo export --format onnx` parity: the exported graphs run in onnxruntime (bench/parity/onnx_check.py) and
//! must match FOLO's own forward pass on the same inputs, for folo-n and folo-s, f32 and f16, raw and decoded
//! outputs, and dynamic graphs (f32 raw and decoded, f16 decoded) at two other sizes. `onnx.checker.check_model`
//! runs on every file.
//!
//! Needs a Python with `onnx`, `onnxruntime` and `numpy`: `FOLO_ONNX_PYTHON`, else the venv at `FOLO_ONNX_VENV`
//! (default `D:/folo-venvs/onnx`). Without one the test prints why and passes, unless `FOLO_ONNX_REQUIRED=1`
//! (CI's `onnx` job sets it, so a broken Python setup fails there instead of skipping).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::{Module, init};
use folo_vision::model::{FoloModel, ModelConfig};
use serde_json::{Value, json};

/// Element-wise relative tolerance (max, mean) of an f32 graph against FOLO's f32 forward: docs/MODEL.md §9.1's
/// fused-model bound. The test models below reach about 5e-5 (max) and 1e-6 (mean).
const TOL_F32: (f64, f64) = (1e-4, 1e-5);
/// The same for an f16 graph, which rounds every activation to half precision (11-bit mantissa) while FOLO
/// computes in f32: precision-bound, about 6e-2 (max) and 3e-3 (mean) on the test models. A mapping error
/// (a wrong constant, cast or layout) moves the mean by orders of magnitude more.
const TOL_F16: (f64, f64) = (1e-1, 5e-3);

fn python() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FOLO_ONNX_PYTHON") {
        return Some(p.into());
    }
    let venv =
        std::env::var_os("FOLO_ONNX_VENV").map_or_else(|| PathBuf::from("D:/folo-venvs/onnx"), PathBuf::from);
    ["Scripts/python.exe", "bin/python"].iter().map(|p| venv.join(p)).find(|p| p.is_file())
}

/// `FOLO_ONNX_REQUIRED` set to anything but `0` or empty: a missing Python fails the test instead of skipping it.
fn required() -> bool {
    std::env::var("FOLO_ONNX_REQUIRED").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// A folo-`scale` whose outputs depend on the image the way a trained model's do. PyTorch's default init with
/// identity BatchNorms shrinks the signal layer by layer until the outputs no longer depend on the image (a mean
/// change of about 1e-7 when it is inverted), which would hide most export bugs. So the running statistics are
/// measured on a batch of images (one training-mode pass with momentum 0.999, which normalises every layer), with
/// γ ≈ 0.45: at γ ≈ 1 the normalised random network is close to chaotic and amplifies f32 rounding about 3000×
/// (onnxruntime with and without graph optimisations then differs from itself by 2e-4), at γ ≈ 0.45 about 300×.
fn calibrated(scale: &str, nc: usize, seed: u64) -> FoloModel {
    let mut cfg = ModelConfig::new(scale, nc).unwrap();
    cfg.bn_momentum = 0.999; // the running statistics become (almost exactly) the batch statistics
    let cal = FoloModel::new(cfg).unwrap(); // training mode
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
    let m = FoloModel::new(ModelConfig::new(scale, nc).unwrap()).unwrap();
    let src: HashMap<String, Tensor> = cal.named_tensors().into_iter().map(|t| (t.name, t.tensor)).collect();
    for t in m.named_tensors() {
        init::copy_(&t.tensor, &src[&t.name]).unwrap();
    }
    m
}

/// Writes `t` as raw little-endian f32 and returns its manifest entry.
fn dump(dir: &Path, name: &str, t: &Tensor) -> Value {
    let bytes: Vec<u8> = t.to_vec::<f32>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(name), bytes).unwrap();
    json!({ "file": name, "shape": t.dims() })
}

/// FOLO's outputs for `x`: the raw head, or the `[B, A, 4 + nc]` decode.
fn reference(m: &FoloModel, x: &Tensor, decode: bool, dir: &Path, tag: &str) -> Value {
    let _g = no_grad();
    let out = m.forward(x).unwrap();
    // The outputs must depend on the image, or the comparison proves little (see `calibrated`).
    let other = m.forward(&x.rsub_scalar(1.0).unwrap()).unwrap();
    let delta = |p: &Tensor, q: &Tensor| {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f32>() / p.len() as f32
    };
    let (db, dc) = (delta(&out.box_raw, &other.box_raw), delta(&out.cls_raw, &other.cls_raw));
    assert!(db > 0.1 && dc > 0.1, "{tag}: the outputs barely depend on the image ({db}, {dc})");
    if decode {
        json!({ "detections": dump(dir, &format!("{tag}-detections.f32"), &out.decode().unwrap()) })
    } else {
        json!({
            "box_raw": dump(dir, &format!("{tag}-box_raw.f32"), &out.box_raw),
            "cls_raw": dump(dir, &format!("{tag}-cls_raw.f32"), &out.cls_raw),
        })
    }
}

fn export(model: &Path, out: &Path, flags: &[&str]) -> Value {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_folo"));
    cmd.args(["export", "--json", "--format", "onnx", "--model", s(model), "--output", s(out)]).args(flags);
    let o = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "export {flags:?}: {stderr}");
    serde_json::from_slice(&o.stdout).unwrap()
}

#[test]
fn onnx_matches_folo_in_onnxruntime() {
    let Some(py) = python() else {
        let why = "no ONNX Python (set FOLO_ONNX_PYTHON or create the venv at D:/folo-venvs/onnx)";
        assert!(!required(), "FOLO_ONNX_REQUIRED is set but there is {why}");
        eprintln!("skipped: {why}");
        return;
    };
    let dir = std::env::temp_dir().join(format!("folo-cli-onnx-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let classes: Vec<String> = ["circle", "square", "triangle", "star", "ring"].map(String::from).into();
    let mut cases = vec![];
    for (k, scale) in ["n", "s"].into_iter().enumerate() {
        let mut m = calibrated(scale, classes.len(), 7 + k as u64);
        m.set_classes(classes.clone()).unwrap();
        m.set_imgsz(128);
        let weights = dir.join(format!("folo-{scale}.folo"));
        m.save(&weights).unwrap();
        // The reference is the fused model, loaded as the exporter loads it.
        let mut r = FoloModel::open(&weights).unwrap();
        r.fuse().unwrap();
        let x = Tensor::rand_uniform([1, 3, 128, 128], 0.0, 1.0, &mut Rng::seed(100 + k as u64));
        let input = dump(&dir, &format!("in-{scale}.f32"), &x);
        for (half, decode) in [(false, false), (false, true), (true, false), (true, true)] {
            let name = format!(
                "folo-{scale}-{}-{}",
                if half { "f16" } else { "f32" },
                if decode { "decoded" } else { "raw" }
            );
            let mut flags = vec![];
            if half {
                flags.push("--half");
            }
            if decode {
                flags.push("--decode");
            }
            let tol = if half { TOL_F16 } else { TOL_F32 };
            let rec = export(&weights, &dir.join(format!("{name}.onnx")), &flags);
            assert_eq!(rec["imgsz"], 128, "the model's imgsz is the default");
            cases.push(json!({
                "name": name,
                "model": format!("{name}.onnx"),
                "dtype": if half { "float16" } else { "float32" },
                "tolerance": tol.0,
                "mean_tolerance": tol.1,
                "inputs": [input],
                "expected": [reference(&r, &x, decode, &dir, &name)],
            }));
        }
        if scale == "n" {
            // A dynamic graph at sizes it was not exported at: another batch, rectangular inputs.
            let xs = [
                Tensor::rand_uniform([2, 3, 96, 160], 0.0, 1.0, &mut Rng::seed(200)),
                Tensor::rand_uniform([1, 3, 64, 32], 0.0, 1.0, &mut Rng::seed(201)),
            ];
            // f16 + decode also covers the anchor grid's cast to f16.
            for (half, decode) in [(false, false), (false, true), (true, true)] {
                let dtype = if half { "f16" } else { "f32" };
                let name = format!("folo-n-{dtype}-dynamic-{}", if decode { "decoded" } else { "raw" });
                let mut flags = vec!["--dynamic"];
                flags.extend(half.then_some("--half"));
                flags.extend(decode.then_some("--decode"));
                export(&weights, &dir.join(format!("{name}.onnx")), &flags);
                let inputs: Vec<Value> =
                    xs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("{name}-in{i}.f32"), x)).collect();
                let expected: Vec<Value> = xs
                    .iter()
                    .enumerate()
                    .map(|(i, x)| reference(&r, x, decode, &dir, &format!("{name}-{i}")))
                    .collect();
                let tol = if half { TOL_F16 } else { TOL_F32 };
                cases.push(json!({
                    "name": name, "model": format!("{name}.onnx"),
                    "dtype": if half { "float16" } else { "float32" },
                    "tolerance": tol.0, "mean_tolerance": tol.1,
                    "inputs": inputs, "expected": expected,
                }));
            }
        }
    }
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
    assert_eq!(report["cases"].as_array().unwrap().len(), 11);
    let _ = std::fs::remove_dir_all(&dir);
}
