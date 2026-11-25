# examples

Complete, reproducible runs of FOLO with the commands that produced them, and one script for outside models.

| path | what it is | prerequisites & expected runtime |
|---|---|---|
| [`shapes/`](shapes/README.md) | the convergence smoke test ([docs/MODEL.md §10.6](../docs/MODEL.md#106-convergence-smoke-test-m2-gate)): folo-n from scratch on folo-data's synthetic shapes, 320 px, with the bar it must reach (`mAP@0.5 ≥ 0.90`, `mAP@0.5:0.95 ≥ 0.75` by epoch 30, with no non-finite loss) and measured CPU and GPU runs | Rust stable only, no downloads (~60 MB generated in ~3 s); historical AdamW reference runs took ~2.8 min on an RTX 4080 SUPER GPU and ~72 min on CPU (current `auto` selects MuSGD) |
| [`coco/`](coco/README.md) | folo-n and folo-s trained from scratch on COCO 2017: data, training, evaluation (cross-checked with pycocotools) and how the published weights (`folo-n-coco*`, `folo-s-coco`) were made | COCO 2017 (~26 GB extracted, plus ~20 GB if keeping ZIPs), a local or cloud GPU ([docs/CLOUD.md](../docs/CLOUD.md); `--amp` recipes require GPU tensor cores), and Python with `pycocotools` for the optional cross-check; 50–128 epochs for `folo-n` (0.314 → 0.370 val2017 AP) or 56 epochs for `folo-s` (0.416 val2017 AP, ~9.5 h on an RTX PRO 4000 Blackwell) |
| [`bench_external.py`](bench_external.py) | runs any model on a FOLO vision benchmark and submits it as an external entry over the HTTP API (chunks passes into ≤ 32 MiB parts before the JSON wrapper — the server accepts up to 64 MiB — and ≤ 1,000 boxes/image); fill in its `predict` function ([docs/BENCH.md, Who can run](../docs/BENCH.md#who-can-run)) | Python 3.9+ standard library + your model's runtime, and a running `folo serve` with a benchmark (`bn_*`); runtime scales with your detector and benchmark pass count |

The two READMEs are pages of the docs site as well ([site/README.md](../site/README.md)). `bench_external.py` needs only the Python
standard library plus whatever the model needs:

```sh
python examples/bench_external.py --bench <benchmark id> --name my-detector
python examples/bench_external.py --server http://127.0.0.1:7878 --bench <benchmark id> --name fast --no-conditions
```

`--server` defaults to `$FOLO_SERVER`, else `http://127.0.0.1:7878`, and `--token` to `$FOLO_TOKEN`. For FOLO's own
models, a benchmark entry needs no script: see [docs/BENCH.md](../docs/BENCH.md). The measurement harnesses that
compare FOLO with Ultralytics are in [bench/](../bench/README.md).

Related guides: [docs/QUICKSTART.md](../docs/QUICKSTART.md) · [docs/MODEL.md](../docs/MODEL.md) ·
[docs/BENCH.md](../docs/BENCH.md) · [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)

[Back to the repository README](../README.md)
