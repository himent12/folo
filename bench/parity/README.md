# bench/parity: FOLO against reference implementations

Scripts that tie FOLO's numbers to the tools people trust: the COCO evaluator against pycocotools, Ultralytics' mAP
method against a port of it, and `folo export --format onnx` against FOLO's own forward pass. The results are
reported in [docs/METRICS.md](../../docs/METRICS.md)
([Parity with pycocotools](../../docs/METRICS.md#parity-with-pycocotools),
[Full COCOeval](../../docs/METRICS.md#full-cocoeval-crowd-regions-area-buckets-ar),
[Differences from Ultralytics' mAP](../../docs/METRICS.md#differences-from-ultralytics-map)) and
[docs/CLI.md, ONNX export](../../docs/CLI.md#onnx-export).

| file | what it does |
|---|---|
| [`coco_reference.py`](coco_reference.py) | generates seeded detection scenarios, scores them with pycocotools' `COCOeval` and writes the fixtures `crates/folo-vision/tests/fixtures/coco/*.json` (read by `tests/coco_parity.rs`) and `tests/fixtures/coco_full/*.json` (read by `tests/coco_full.rs`); also stores Ultralytics' AP of each scenario, through the port |
| [`ultralytics_port.py`](ultralytics_port.py) | a numpy port of Ultralytics' detection AP: version 8.4.198 and the older 8.x algorithm (a module, not run on its own) |
| [`check_ultralytics_port.py`](check_ultralytics_port.py) | checks the port against a real `ultralytics` 8.4.198 install on every fixture: TP matrices and mAP |
| [`onnx_check.py`](onnx_check.py) | runs exported ONNX models in onnxruntime (CPU) on fixed inputs and compares every output with the tensors FOLO wrote; prints one JSON report, exits 1 on a failure |
| [`test_onnx_check.py`](test_onnx_check.py) | tests of `onnx_check.py`'s end-to-end row comparison |

## The COCO fixtures

Needs a Python with `pycocotools` and `numpy`. From the repository root:

```sh
python -I bench/parity/coco_reference.py
```

It rewrites the fixtures in `crates/folo-vision/tests/fixtures/coco/` and `coco_full/`, which `cargo test` then checks
FOLO against; the script's docstring explains why both sides see bit-identical inputs. `check_ultralytics_port.py`
needs a Python with `ultralytics` (and torch) at exactly 8.4.198, the version the port asserts; it installs nothing
and writes nothing:

```sh
<python-with-ultralytics> bench/parity/check_ultralytics_port.py
```

## ONNX export

`crates/folo-cli/tests/onnx.rs` writes the inputs, FOLO's outputs and a manifest per group of cases, then runs
`onnx_check.py MANIFEST.json` (the manifest format is in the script's docstring). It needs a Python with `onnx`,
`onnxruntime` and `numpy`:

- the test takes it from `FOLO_ONNX_PYTHON`, else from the venv at `FOLO_ONNX_VENV` (default `D:/folo-venvs/onnx`, the
  maintainers' machine);
- without one it prints why and passes, unless `FOLO_ONNX_REQUIRED` is set (to anything but `0` or empty);
- CI's `rust (linux)` job in `.github/workflows/ci.yml` installs pinned versions on Python 3.12 (step
  "onnx / onnxruntime / numpy (pinned)") and sets both variables, so there a missing Python fails the test.

It also runs by hand on such a directory: `python bench/parity/onnx_check.py MANIFEST.json [--bench N]` (`--bench N`
also times N onnxruntime runs of each case's first input). The comparison's own tests:

```sh
python -m unittest discover -s bench/parity -p "test_*.py"
```

[Up: bench/](../README.md)
