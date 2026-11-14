//! `folo export --format onnx` parity: the exported graphs run in onnxruntime (bench/parity/onnx_check.py) and
//! must match FOLO's own forward pass on the same inputs, for folo-n and folo-s, f32 and f16, raw and decoded
//! outputs, and dynamic graphs (f32 raw and decoded, f16 decoded) at two other sizes. `onnx.checker.check_model`
//! runs on every file.
//!
//! Needs a Python with `onnx`, `onnxruntime` and `numpy`: `FOLO_ONNX_PYTHON`, else the venv at `FOLO_ONNX_VENV`
//! (default `D:/folo-venvs/onnx`). Without one the test prints why and passes, unless `FOLO_ONNX_REQUIRED=1`
//! (CI's `rust (linux)` job sets it, so a broken Python setup fails there instead of skipping).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::JoinHandle;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::{Module, init};
use folo_test_support::TestDir;
use folo_vision::HeadOutput;
use folo_vision::model::{FoloModel, HeadMode, ModelConfig};
use folo_vision::nms::{NmsConfig, topk_detections};
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
    calibrated_head(scale, nc, seed, "o2m")
}

/// [`calibrated`] with a `head` (`o2o`: the training-mode pass also normalises the one-to-one branch).
fn calibrated_head(scale: &str, nc: usize, seed: u64, head: &str) -> FoloModel {
    let mut cfg = ModelConfig::new(scale, nc).unwrap();
    cfg.head = head.into();
    let base = cfg.clone();
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
    let m = FoloModel::new(base).unwrap();
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

/// FOLO's forward pass of `x`, checked to depend on the image: otherwise the comparison proves little (see
/// [`calibrated`]). Computed once per model and input, for every case that runs that input.
fn forward(m: &FoloModel, x: &Tensor, tag: &str) -> HeadOutput {
    let _g = no_grad();
    let out = m.forward(x).unwrap();
    let other = m.forward(&x.rsub_scalar(1.0).unwrap()).unwrap();
    let delta = |p: &Tensor, q: &Tensor| {
        let (p, q) = (p.to_vec::<f32>().unwrap(), q.to_vec::<f32>().unwrap());
        p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f32>() / p.len() as f32
    };
    let (db, dc) = (delta(&out.box_raw, &other.box_raw), delta(&out.cls_raw, &other.cls_raw));
    assert!(db > 0.1 && dc > 0.1, "{tag}: the outputs barely depend on the image ({db}, {dc})");
    out
}

/// The expected outputs from FOLO's forward `out` (of a model predicting with `head`): the raw head, or the
/// `[B, A, 4 + nc]` decode; for the `o2o` head, the decode is the end-to-end `[B, K, 6]` of docs/MODEL.md §6.6
/// (`K = min(max_det, A)`, no `conf`), compared row by row (`"match": "rows"`).
fn reference(out: &HeadOutput, head: HeadMode, decode: bool, dir: &Path, tag: &str) -> Value {
    reference_k(out, head, decode, dir, tag, 300)
}

fn reference_k(
    out: &HeadOutput,
    head: HeadMode,
    decode: bool,
    dir: &Path,
    tag: &str,
    max_det: usize,
) -> Value {
    let _g = no_grad();
    if decode && head == HeadMode::O2o {
        let nc = out.num_classes;
        let k = max_det.min(out.num_anchors());
        // The top K, and a longer pool (the selection with K + 16) that rows swapped across the cutoff come from.
        let pool_k = (k + 16).min(out.num_anchors());
        let select = |max_det: usize| -> Tensor {
            let nms =
                NmsConfig { conf: f32::NEG_INFINITY, max_det, multi_label: true, ..NmsConfig::default() };
            let mut rows = Vec::new();
            for (boxes, scores) in out.decode_images().unwrap() {
                let d = topk_detections(&boxes, &scores, nc, &nms);
                assert_eq!(d.len(), max_det, "{tag}: {max_det} rows");
                for d in d {
                    rows.extend([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.score, d.class as f32]);
                }
            }
            Tensor::from_vec(rows, [out.batch(), max_det, 6]).unwrap()
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

/// Starts `folo export --format onnx` of `model` to `out` on its own thread; the handle gives the export record.
/// The exports run while the next models and references are computed.
fn export(model: &Path, out: &Path, flags: &[&str]) -> JoinHandle<Value> {
    let mut cmd = folo_cmd();
    cmd.args(["export", "--json", "--format", "onnx", "--model", s(model), "--output", s(out)]).args(flags);
    let flags: Vec<String> = flags.iter().map(|f| f.to_string()).collect();
    std::thread::spawn(move || {
        let o = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success(), "export {flags:?}: {stderr}");
        serde_json::from_slice(&o.stdout).unwrap()
    })
}

/// The result of a thread, or its panic.
fn joined<T>(handle: JoinHandle<T>) -> T {
    handle.join().unwrap_or_else(|e| std::panic::resume_unwind(e))
}

/// Runs bench/parity/onnx_check.py with `py` on `cases` on its own thread, once their `exports` are written (each
/// with the fields its record must have), with the manifest `dir/manifest-{part}.json`. The handle gives the
/// number of cases checked; the checkers of the first cases run while the later ones are prepared.
fn check(
    py: &Path,
    dir: &Path,
    part: &str,
    cases: Vec<Value>,
    exports: Vec<(JoinHandle<Value>, Value)>,
) -> JoinHandle<usize> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity/onnx_check.py");
    let manifest = dir.join(format!("manifest-{part}.json"));
    let mut cmd = Command::new(py);
    cmd.arg(&script).arg(&manifest);
    std::thread::spawn(move || {
        for (handle, want) in exports {
            let rec = joined(handle);
            for (k, v) in want.as_object().unwrap() {
                assert_eq!(&rec[k], v, "{k} of {rec}");
            }
        }
        std::fs::write(&manifest, serde_json::to_vec_pretty(&json!({ "cases": cases })).unwrap()).unwrap();
        let o = cmd.output().unwrap();
        let stdout = String::from_utf8_lossy(&o.stdout);
        let report: Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
            panic!("onnx_check.py printed no report ({e}): {stdout}\n{}", String::from_utf8_lossy(&o.stderr))
        });
        eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
        assert!(o.status.success() && report["ok"] == true, "ONNX parity failed: {report}");
        report["cases"].as_array().unwrap().len()
    })
}

#[test]
fn onnx_matches_folo_in_onnxruntime() {
    let Some(py) = python() else {
        let why = "no ONNX Python (set FOLO_ONNX_PYTHON or create the venv at D:/folo-venvs/onnx)";
        assert!(!required(), "FOLO_ONNX_REQUIRED is set but there is {why}");
        eprintln!("skipped: {why}");
        return;
    };
    let dir = TestDir::new("cli-onnx");
    let classes: Vec<String> = ["circle", "square", "triangle", "star", "ring"].map(String::from).into();
    // The cases of the next checker, their running exports with the fields each record must have, and the
    // running checkers.
    let mut cases = vec![];
    let mut exports: Vec<(JoinHandle<Value>, Value)> = vec![];
    let mut checkers = vec![];
    let mut checker = |part: &str, cases: &mut Vec<Value>, exports: &mut Vec<_>| {
        checkers.push(check(&py, &dir, part, std::mem::take(cases), std::mem::take(exports)));
    };
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
        let fwd = forward(&r, &x, &format!("folo-{scale}"));
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
            // The model's imgsz is the default.
            exports
                .push((export(&weights, &dir.join(format!("{name}.onnx")), &flags), json!({ "imgsz": 128 })));
            cases.push(json!({
                "name": name,
                "model": format!("{name}.onnx"),
                "dtype": if half { "float16" } else { "float32" },
                "tolerance": tol.0,
                "mean_tolerance": tol.1,
                "inputs": [input],
                "expected": [reference(&fwd, r.head_mode(), decode, &dir, &name)],
            }));
        }
        checker(scale, &mut cases, &mut exports);
        if scale == "n" {
            // A dynamic graph at sizes it was not exported at: another batch, rectangular inputs.
            let xs = [
                Tensor::rand_uniform([2, 3, 96, 160], 0.0, 1.0, &mut Rng::seed(200)),
                Tensor::rand_uniform([1, 3, 64, 32], 0.0, 1.0, &mut Rng::seed(201)),
            ];
            let fwds: Vec<HeadOutput> =
                xs.iter().enumerate().map(|(i, x)| forward(&r, x, &format!("folo-n-dynamic-{i}"))).collect();
            // f16 + decode also covers the anchor grid's cast to f16.
            for (half, decode) in [(false, false), (false, true), (true, true)] {
                let dtype = if half { "f16" } else { "f32" };
                let name = format!("folo-n-{dtype}-dynamic-{}", if decode { "decoded" } else { "raw" });
                let mut flags = vec!["--dynamic"];
                flags.extend(half.then_some("--half"));
                flags.extend(decode.then_some("--decode"));
                exports.push((export(&weights, &dir.join(format!("{name}.onnx")), &flags), json!({})));
                let inputs: Vec<Value> =
                    xs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("{name}-in{i}.f32"), x)).collect();
                let expected: Vec<Value> = fwds
                    .iter()
                    .enumerate()
                    .map(|(i, f)| reference(f, r.head_mode(), decode, &dir, &format!("{name}-{i}")))
                    .collect();
                let tol = if half { TOL_F16 } else { TOL_F32 };
                cases.push(json!({
                    "name": name, "model": format!("{name}.onnx"),
                    "dtype": if half { "float16" } else { "float32" },
                    "tolerance": tol.0, "mean_tolerance": tol.1,
                    "inputs": inputs, "expected": expected,
                }));
            }
            checker("n-dynamic", &mut cases, &mut exports);
        }
    }
    // An `o2o` model (docs/MODEL.md §6.6): its one-to-one branch raw and end to end (f32, f16, dynamic, a small
    // max_det), and its one-to-many branch with `--head o2m`.
    let mut m = calibrated_head("n", classes.len(), 9, "o2o");
    m.set_classes(classes.clone()).unwrap();
    m.set_imgsz(128);
    let weights = dir.join("folo-n-o2o.folo");
    m.save(&weights).unwrap();
    let mut r = FoloModel::open(&weights).unwrap();
    r.fuse().unwrap();
    assert_eq!(r.head_mode(), HeadMode::O2o);
    let x = Tensor::rand_uniform([2, 3, 128, 128], 0.0, 1.0, &mut Rng::seed(300));
    let xs = [x.clone(), Tensor::rand_uniform([1, 3, 96, 160], 0.0, 1.0, &mut Rng::seed(301))];
    let fwds: Vec<HeadOutput> =
        xs.iter().enumerate().map(|(i, x)| forward(&r, x, &format!("folo-n-o2o-{i}"))).collect();
    for (name, flags, max_det, inputs) in [
        ("folo-n-o2o-f32-raw", vec!["--batch", "2"], 300, &xs[..1]),
        ("folo-n-o2o-f32-e2e", vec!["--batch", "2", "--decode"], 300, &xs[..1]),
        ("folo-n-o2o-f32-e2e-k20", vec!["--batch", "2", "--decode", "--max-det", "20"], 20, &xs[..1]),
        ("folo-n-o2o-f16-e2e", vec!["--batch", "2", "--decode", "--half"], 300, &xs[..1]),
        ("folo-n-o2o-f32-dynamic-e2e", vec!["--dynamic", "--decode"], 300, &xs[..]),
    ] {
        let half = flags.contains(&"--half");
        let decode = flags.contains(&"--decode");
        exports.push((export(&weights, &dir.join(format!("{name}.onnx")), &flags), json!({ "head": "o2o" })));
        let ins: Vec<Value> =
            inputs.iter().enumerate().map(|(i, x)| dump(&dir, &format!("{name}-in{i}.f32"), x)).collect();
        let expected: Vec<Value> = fwds[..inputs.len()]
            .iter()
            .enumerate()
            .map(|(i, f)| reference_k(f, HeadMode::O2o, decode, &dir, &format!("{name}-{i}"), max_det))
            .collect();
        let tol = if half { TOL_F16 } else { TOL_F32 };
        cases.push(json!({
            "name": name, "model": format!("{name}.onnx"),
            "dtype": if half { "float16" } else { "float32" },
            "tolerance": tol.0, "mean_tolerance": tol.1,
            "inputs": ins, "expected": expected,
        }));
    }
    let as_o2m = export(&weights, &dir.join("folo-n-o2o-as-o2m.onnx"), &["--batch", "2", "--head", "o2m"]);
    exports.push((as_o2m, json!({ "head": "o2m" })));
    r.set_head_mode(HeadMode::O2m).unwrap();
    let o2m = forward(&r, &x, "folo-n-o2o-as-o2m");
    cases.push(json!({
        "name": "folo-n-o2o-as-o2m-raw", "model": "folo-n-o2o-as-o2m.onnx", "dtype": "float32",
        "tolerance": TOL_F32.0, "mean_tolerance": TOL_F32.1,
        "inputs": [dump(&dir, "o2m-in.f32", &x)],
        "expected": [reference(&o2m, HeadMode::O2m, false, &dir, "folo-n-o2o-as-o2m")],
    }));

    checker("o2o", &mut cases, &mut exports);
    let checked: usize = checkers.into_iter().map(joined).sum();
    assert_eq!(checked, 17);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1");
    c
}
