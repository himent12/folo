# examples

Complete, reproducible runs of FOLO with the commands that produced them, and one script for outside models.

| path | what it is |
|---|---|
| [`shapes/`](shapes/README.md) | the convergence smoke test (docs/MODEL.md §10.6): folo-n from scratch on folo-data's synthetic shapes, 320 px, with the bar it must reach and measured CPU and GPU runs |
| [`coco/`](coco/README.md) | folo-n and folo-s trained from scratch on COCO 2017: data, training, evaluation (cross-checked with pycocotools) and how the published weights were made |
| [`bench_external.py`](bench_external.py) | runs any model on a FOLO vision benchmark and submits it as an external entry over the HTTP API; fill in its `predict` function ([docs/BENCH.md, Who can run](../docs/BENCH.md#who-can-run)) |

The two READMEs are pages of the docs site as well (site/README.md). `bench_external.py` needs only the Python
standard library plus whatever the model needs:

```sh
python examples/bench_external.py --bench <benchmark id> --name my-detector
python examples/bench_external.py --server http://127.0.0.1:7878 --bench <benchmark id> --name fast --no-conditions
```

`--server` defaults to `$FOLO_SERVER`, else `http://127.0.0.1:7878`, and `--token` to `$FOLO_TOKEN`. For FOLO's own
models, a benchmark entry needs no script: see docs/BENCH.md. The measurement harnesses that compare FOLO with
Ultralytics are in [bench/](../bench/README.md).

[Back to the repository README](../README.md)
