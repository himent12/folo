# bench: measurement harnesses

The scripts behind the numbers in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) and [docs/METRICS.md](../docs/METRICS.md):
Python and shell programs that run FOLO and Ultralytics side by side, check FOLO's evaluator and ONNX export against
reference tools, and keep the raw results. None of it is part of the cargo workspace. `cargo test` runs one script
from here: `crates/folo-cli/tests/onnx.rs` calls [`parity/onnx_check.py`](parity/onnx_check.py).

Three different things in this repository are called "bench":

| what | where | documented in |
|---|---|---|
| these measurement harnesses | `bench/` | [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) |
| vision benchmarks, the Studio and API feature (stress conditions, plans, leaderboards) | `crates/folo-bench`, `/api/bench/...` | [docs/BENCH.md](../docs/BENCH.md) |
| Rust speed benchmarks | `crates/<crate>/benches/*.rs`, run with `cargo bench -p <crate> --bench <name>` | [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) |

## Folders

| folder | the question it answers | where the results are reported |
|---|---|---|
| [`baselines/`](baselines/README.md) | How fast is the incumbent (Ultralytics YOLO26) on this machine: forward pass, end to end, CPU, training step, whole trainer? | docs/BENCHMARKS.md, [Results](../docs/BENCHMARKS.md#results) ([Reproduce](../docs/BENCHMARKS.md#reproduce) has the commands) |
| [`finetune/`](finetune/README.md) | Fine-tuned from their COCO weights on four small public datasets, how accurate are FOLO and YOLO26 under one scorer? | docs/BENCHMARKS.md, [Fine-tuning on small datasets](../docs/BENCHMARKS.md#fine-tuning-on-small-datasets) |
| [`parity/`](parity/README.md) | Does FOLO's mAP match pycocotools, how far is Ultralytics' mAP from it, and does `folo export --format onnx` compute what FOLO computes? | docs/METRICS.md, [Parity with pycocotools](../docs/METRICS.md#parity-with-pycocotools) and [Differences from Ultralytics' mAP](../docs/METRICS.md#differences-from-ultralytics-map); docs/CLI.md, [ONNX export](../docs/CLI.md#onnx-export) |
| [`trainer/`](trainer/README.md) | How long do `folo train` and `folo val` take (start-up, first epoch, sustained img/s), one build against another? | docs/BENCHMARKS.md, the hill-climb sections, e.g. [The trainer (row 10)](../docs/BENCHMARKS.md#the-trainer-row-10) |
| [`learn/`](learn/README.md) | Trained from scratch on the same COCO subset, how do FOLO and Ultralytics compare, and what do their augmentations produce? | [docs/notes/hill/learn.md](../docs/notes/hill/learn.md) |
| `fusion/` | Which post-processing (NMS variants, box fusion) scores best on FOLO's pre-NMS candidates? | [docs/notes/hill/r3-fusion.md](../docs/notes/hill/r3-fusion.md) |
| `refine/` | Can a small learned box refiner (PyTorch, on FOLO's feature dump) tighten the boxes? | [docs/notes/hill/r3-refine.md](../docs/notes/hill/r3-refine.md) |
| [`distill/`](distill/README.md) | How much of the folo-n to folo-s gap is in the scores and how much in the boxes, and is folo-s a better localiser on the same objects? | [docs/notes/hill/r4-distill.md](../docs/notes/hill/r4-distill.md) |

`fusion/` and `refine/` are the working code of two lanes of hill-climb 3. Each script's docstring says what
it does; the notebooks above say how they were used. `fusion/` reads the dump that
`cargo run -p folo-vision --example candidate_dump` writes, and `fusion/make_fixture.py` writes
`crates/folo-vision/tests/fixtures/fusion/dfl.json`, which `crates/folo-vision/tests/fusion_reference.rs` reads.

## Conventions

- **Run from the repository root.** The usage lines and default output paths (`bench/baselines/results.json`,
  `crates/folo-vision/tests/fixtures/coco/`) are relative to it.
- **No pinned Python environment.** There is no requirements file; each folder's README lists the packages its
  scripts import. The scripts that drive Ultralytics confine its settings, downloads and runs to a `--work-dir` and
  turn off its auto-install and telemetry.
- **`D:/` paths are the maintainers' machine** (AGENTS.md, On the maintainers' machine). Where a usage line or a
  default has one, pass your own path; where a shell script hard-codes one, the folder's README says so.
- **Tests:** `baselines/`, `finetune/` and `parity/` have `test_*.py` files, run with
  `python -m unittest discover -s bench/<folder> -p "test_*.py"`. No workflow runs them; CI runs
  `parity/onnx_check.py` only, through `crates/folo-cli/tests/onnx.rs` in the `rust (linux)` job of `ci.yml`.
- **Scratch output** goes to `runs-bench/`, `runs/` and `datasets/`, which `.gitignore` excludes. Committed results
  are `baselines/results-rtx4080s.json` and `finetune/results/`.

[Back to the repository README](../README.md)
