# bench/baselines: the incumbent's numbers

Measures Ultralytics YOLO26 (PyTorch eager) on one machine, so FOLO has numbers to beat: parameters and GFLOPs, the
GPU forward pass, the Ultralytics predictor end to end, the CPU, one training step and the whole trainer. The tables
and the method are in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md): [Reproduce](../../docs/BENCHMARKS.md#reproduce),
[Method](../../docs/BENCHMARKS.md#method) and [Results](../../docs/BENCHMARKS.md#results).

| file | what it does |
|---|---|
| [`ultralytics_bench.py`](ultralytics_bench.py) | the harness: runs the sections picked with `--sections` (`params`, `gpu_fwd`, `gpu_e2e`, `cpu`, `train_step`, `train`) and writes one results JSON |
| [`bench_timing.py`](bench_timing.py) | timing helpers and statistics: per-call latency and throughput (CUDA or host), training throughput, CPU use around a measurement |
| [`bench_workloads.py`](bench_workloads.py) | what is measured: the models (`yolo26n`, `yolo26s`, `yolo26m`; training `yolo26n` and `yolo26s`), 640 px, the predictor end to end, and the synthetic dataset (shapes on coloured noise, 3 classes, YOLO format) |
| [`render_tables.py`](render_tables.py) | prints the Markdown tables from a results JSON; `--update docs/BENCHMARKS.md` replaces the text between `<!-- tables:begin -->` and `<!-- tables:end -->` |
| [`results-rtx4080s.json`](results-rtx4080s.json) | the committed raw results behind every incumbent number in docs/BENCHMARKS.md ([Machine](../../docs/BENCHMARKS.md#machine)) |
| [`test_ultralytics_bench.py`](test_ultralytics_bench.py) | CPU-only tests of the harness's bookkeeping (no torch, no GPU, no Ultralytics import) |
| [`test_render_tables.py`](test_render_tables.py) | tests of `render_tables.py` (pure Python) |

## What it needs

- A Python with `torch`, `ultralytics`, `opencv-python`, `numpy` and `psutil` (the imports of the three scripts).
  docs/BENCHMARKS.md, Machine, lists the versions the committed results were measured with.
- An NVIDIA GPU with a CUDA build of torch for the GPU sections, and `nvidia-smi` on `PATH` (the script records the
  GPU's state with it). On Windows, the load of other processes on the GPU is read from the per-process
  "GPU Engine" performance counters through PowerShell; elsewhere `nvidia-smi` only says which processes use it.
- `yolo26n.pt`, `yolo26s.pt` and `yolo26m.pt` in `--weights-dir`. They are not committed, and the script exits if one
  is missing rather than let Ultralytics download it.

## Run it

From the repository root (the commands of [Reproduce](../../docs/BENCHMARKS.md#reproduce)):

```sh
python bench/baselines/ultralytics_bench.py --weights-dir <dir with yolo26n.pt, yolo26s.pt, yolo26m.pt> \
    --work-dir <work dir> --out bench/baselines/results-rtx4080s.json
python bench/baselines/render_tables.py bench/baselines/results-rtx4080s.json --update docs/BENCHMARKS.md
python -m unittest discover -s bench/baselines -p "test_*.py"
```

- `--work-dir` defaults to `D:/folo-bench` (the maintainers' machine): pass your own. Ultralytics' config dir,
  downloads and runs, and the synthetic dataset (`<work dir>/data-<train>-<val>-seed0`, 512 train and 64 val images by
  default) go there; the script writes nothing outside `--work-dir` and `--out`.
- `--out` defaults to `bench/baselines/results.json`. Re-running some sections merges them into an existing file;
  `--quick` is a smoke test with fewer iterations.
- `python bench/baselines/ultralytics_bench.py --help` lists every option.

[Up: bench/](../README.md)
