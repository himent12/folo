"""Parity check of `folo export --format onnx` against FOLO's own forward pass (docs/CLI.md, export).

Runs an exported model in onnxruntime (CPU) on fixed inputs and compares every output with tensors FOLO dumped
for the same inputs. The Rust integration test `crates/folo-cli/tests/onnx.rs` writes the manifest, the inputs
and the expected outputs, then runs this script; it can also be run by hand on such a directory.

Usage:
    python bench/parity/onnx_check.py MANIFEST.json [--bench N]

The manifest:
    {"cases": [{"name": "folo-n-f32-raw", "model": "x.onnx", "dtype": "float32" | "float16",
                "tolerance": 1e-4, "mean_tolerance": null,             # optional bound on the mean error
                "inputs": [{"file": "in.f32", "shape": [B, 3, H, W]}, ...],      # one run per input
                "expected": [{"box_raw": {"file": "a.f32", "shape": [...]}, ...}, ...]}]}  # one per input
Data files are raw little-endian float32, row-major. A model is checked with `onnx.checker.check_model`
(`full_check=True`, which also runs shape inference), then every output must satisfy
`|onnx - folo| <= tolerance * max(|onnx|, |folo|, 1)` element-wise (relative, with a floor of one unit so values
near zero are not held to a tighter absolute bound than their neighbours; the same rule as the Rust model tests),
and, when the case has a `mean_tolerance`, the mean of that element-wise error must be within it too.

An expected output with `"match": "rows"` is an end-to-end detection list `[B, K, 6]` (`x1, y1, x2, y2, score,
class`, sorted by score; the one-to-one head, docs/MODEL.md §6.6), compared so that runtimes may order near-equal
scores differently: per image the sorted score columns must agree element-wise, and every FOLO row must have an
unused ONNX row of the same class whose score and box agree (the same relative rule); a FOLO row may only be missing
when its score is within the tolerance of the list's last score (it may have dropped out of the top K).

`--bench N` also times N onnxruntime CPU runs of each case's first input (after 5 warm-up runs) and reports the
median latency in milliseconds.

Prints one JSON object (`{"ok": bool, "cases": [...]}`) on stdout; exits 1 when any case fails.
Needs `pip install onnx onnxruntime numpy` (the FOLO dev venv is D:/folo-venvs/onnx).
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort


def load(base, spec):
    data = np.fromfile(os.path.join(base, spec["file"]), dtype="<f4")
    return data.reshape(spec["shape"])


def compare(got, want, tol, mean_tol):
    got = got.astype(np.float64)
    want = want.astype(np.float64)
    if got.shape != want.shape:
        return {"ok": False, "why": f"shape {list(got.shape)} != {list(want.shape)}"}
    if not np.all(np.isfinite(got)):
        return {"ok": False, "why": "non-finite ONNX output"}
    err = np.abs(got - want) / np.maximum(np.maximum(np.abs(got), np.abs(want)), 1.0)
    i = int(np.argmax(err))
    worst = float(err.flat[i])
    mean = float(np.mean(err))
    return {
        "ok": bool(worst <= tol and (mean_tol is None or mean <= mean_tol)),
        "max_rel_err": worst,
        "mean_rel_err": mean,
        "at": i,
        "onnx": float(got.flat[i]),
        "folo": float(want.flat[i]),
    }


def rel_err(a, b):
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)


def compare_rows(got, want, tol):
    got = got.astype(np.float64)
    want = want.astype(np.float64)
    if got.shape != want.shape or got.ndim != 3 or got.shape[2] != 6:
        return {"ok": False, "why": f"shape {list(got.shape)} != {list(want.shape)} (or not [B, K, 6])"}
    if not np.all(np.isfinite(got)):
        return {"ok": False, "why": "non-finite ONNX output"}
    worst_score = float(np.max(rel_err(got[..., 4], want[..., 4]), initial=0.0))
    worst_box, unmatched, boundary = 0.0, [], 0
    for b in range(want.shape[0]):
        g, w = got[b], want[b]
        last = min(g[-1, 4], w[-1, 4]) if len(w) else 0.0
        used = np.zeros(len(g), dtype=bool)
        for i, row in enumerate(w):
            ok = (~used) & (g[:, 5] == row[5]) & (rel_err(g[:, 4], row[4]) <= tol)
            cand = np.flatnonzero(ok)
            errs = [float(np.max(rel_err(g[j, :4], row[:4]))) for j in cand]
            good = [(e, j) for e, j in zip(errs, cand) if e <= tol]
            if good:
                e, j = min(good)
                used[j] = True
                worst_box = max(worst_box, e)
            elif rel_err(row[4], last) <= tol:
                boundary += 1
            else:
                unmatched.append([b, i, row.tolist(), (min(errs) if errs else None)])
    return {
        "ok": bool(worst_score <= tol and not unmatched),
        "max_rel_err_score": worst_score,
        "max_rel_err_box": worst_box,
        "boundary_rows": boundary,
        "unmatched": unmatched[:5],
    }


def check_case(base, case, bench):
    path = os.path.join(base, case["model"])
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    dtype = np.float16 if case["dtype"] == "float16" else np.float32
    result = {"name": case["name"], "outputs": out_names, "runs": []}
    ok = True
    for inp, expected in zip(case["inputs"], case["expected"]):
        x = load(base, inp).astype(dtype)
        outs = dict(zip(out_names, sess.run(None, {in_name: x})))
        if set(outs) != set(expected):
            result["runs"].append({"ok": False, "why": f"outputs {sorted(outs)} != {sorted(expected)}"})
            ok = False
            continue
        run = {"shape": list(x.shape)}
        for name, spec in expected.items():
            if spec.get("match") == "rows":
                c = compare_rows(outs[name], load(base, spec), case["tolerance"])
            else:
                c = compare(outs[name], load(base, spec), case["tolerance"], case.get("mean_tolerance"))
            run[name] = c
            ok = ok and c["ok"]
        result["runs"].append(run)
    if bench:
        x = load(base, case["inputs"][0]).astype(dtype)
        for _ in range(5):
            sess.run(None, {in_name: x})
        times = []
        for _ in range(bench):
            t = time.perf_counter()
            sess.run(None, {in_name: x})
            times.append((time.perf_counter() - t) * 1e3)
        result["latency_ms"] = {"median": statistics.median(times), "min": min(times), "runs": bench}
    result["ok"] = ok
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--bench", type=int, default=0, help="time N onnxruntime CPU runs per case")
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(args.manifest))
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    cases = []
    for case in manifest["cases"]:
        try:
            cases.append(check_case(base, case, args.bench))
        except Exception as e:  # a checker or runtime failure is a failed case, with its message
            cases.append({"name": case["name"], "ok": False, "why": f"{type(e).__name__}: {e}"})
    ok = all(c["ok"] for c in cases)
    versions = {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__}
    print(json.dumps({"ok": ok, "versions": versions, "cases": cases}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
