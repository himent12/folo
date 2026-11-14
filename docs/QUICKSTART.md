# FOLO quickstart

This walkthrough takes you from a fresh checkout to a trained detector: on synthetic data first, then on your own
YOLO or COCO folder. After that it covers fine-tuning, evaluation, export, FOLO Studio, and running the model in a
browser. Every command here was run on our test machine (Windows 11, RTX 4080 SUPER, i7-14700KF); the timings come
from it. For the full flag lists, see [CLI.md](CLI.md) or `folo describe --json`.

## 1. Get `folo`

You need Rust stable and, for training at speed, a GPU with Vulkan, DX12 or Metal. The GPU kernels are FOLO's own
WGSL on wgpu, so there is no CUDA or PyTorch to install (`folo doctor` shows whether your adapter qualifies). Without a usable GPU, `--device auto` falls back to the
CPU, which is about 25× slower to train.

```sh
git clone https://github.com/himent12/folo && cd folo
cargo folo doctor        # builds the CLI on first use (a few minutes), then prints the machine report
```

`cargo folo ...` is an alias for `cargo run --release -p folo-cli -- ...` (`.cargo/config.toml`). Every `folo`
command below works as `cargo folo` from the checkout. You can also put the built binary on your `PATH`
(`target/release/folo`, or `cargo install --path crates/folo-cli`).

`folo doctor` lists your GPU adapters and the one FOLO picks (`FOLO uses [0] NVIDIA GeForce RTX 4080 SUPER
(Vulkan)`). To choose a different one, set `FOLO_GPU_ADAPTER=<index or name substring>`, or limit the backends with
`FOLO_GPU_BACKEND=vulkan,dx12,metal` ([GPU.md](GPU.md)).

Or skip the build: releases ship `folo` for x86_64 Windows and Linux with Studio embedded (and FOLO Desktop for
Windows). On Linux:
`curl -fsSL https://github.com/himent12/folo/releases/latest/download/install.sh | sh`. In PowerShell on Windows:
`irm https://github.com/himent12/folo/releases/latest/download/install.ps1 | iex`. For FOLO Desktop too, download
`install.ps1` and run `powershell -ExecutionPolicy Bypass -File install.ps1 -Desktop`. The commands below then work
as plain `folo`. See the README's [Install](../README.md#install) and [RELEASE.md](RELEASE.md).

## 2. A first model on synthetic data

```sh
folo dataset synth datasets/shapes --images 1000 --imgsz 320
folo dataset stats datasets/shapes
folo train --data datasets/shapes --model folo-n --imgsz 320 --epochs 30
```

- `dataset synth` writes a labelled YOLO dataset of circles, squares and triangles: 800 train and 200 val images,
  in about 1 s. `--classes` takes up to 8 names, and `circle`, `square`, `triangle`, `star`, `ring`, `cross`,
  `hexagon` and `heart` each get their own shape.
- `train` uses the GPU when it finds one. Each epoch shows a progress bar (losses, img/s, ETA) and then one row of
  validation metrics. On our test machine this run took 83 s and ended at **mAP50 0.955, mAP50-95 0.823** (two runs
  a day apart gave the same numbers; with the GPU shared with other jobs it took up to 139 s).

The run folder is `runs/<dataset>-<size>/`, here `runs/shapes-n/`. If that folder already exists, a numeric suffix
is added (`--project` and `--name` override both parts). It contains:

| file | what |
|---|---|
| `weights/best.folo` | the checkpoint with the best validation mAP50-95: EMA weights, optimizer and schedule |
| `weights/last.folo` | the latest epoch's checkpoint, for `--resume` |
| `metrics.jsonl` | every event of the run: each step's losses, each epoch's metrics, the logs |
| `config.json`, `samples/` | the resolved settings, and validation predictions per epoch |

If you stop a run with Ctrl+C, it stops after the current batch and keeps the last finished epoch. Continue it,
bit-exactly, with `folo train --resume runs/shapes-n/weights/last.folo`.

## 3. Predict and evaluate

```sh
folo predict --model runs/shapes-n/weights/best.folo datasets/shapes/images/val --save runs/shapes-n/pred
folo val --model runs/shapes-n/weights/best.folo --data datasets/shapes
```

`predict` takes images or folders (it searches folders recursively) and prints the boxes for each image.
`--save DIR` writes copies of the images with the boxes drawn on them. The defaults are `--conf 0.25 --iou 0.7
--max-det 300`. On the 200 val images above, the mean was 9.2 ms per image end to end.

`val` reports mAP50, mAP50-95, precision and recall, overall and per class, computed as pycocotools does
([METRICS.md](METRICS.md)). Add `--split test` to evaluate on a test split. On a COCO dataset, `--coco` adds the
full COCOeval table (AP small / medium / large, AR); a YOLO dataset converts with `folo dataset convert`. `--half`
runs `val` and `predict` in f16 on the GPU (`--tensor-cores` puts it on the tensor cores), within 0.002 mAP of f32.

## 4. Train on your own data

FOLO reads YOLO and COCO datasets as they are, including Roboflow exports. You can pass a folder, a `data.yaml` or a
COCO instances `.json` ([DATA.md](DATA.md) has the detection rules). Images but no labels yet? Studio can have
Claude Code or Codex draw the boxes for you to review ([LABELING.md](LABELING.md)).

**YOLO**: images, plus one `.txt` per image holding `class cx cy w h` lines (normalised to 0–1):

```
my-data/
  data.yaml            # names: [cat, dog]   (optional: path, train, val, test)
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt     # same file stems as the images; a missing file = background image
```

**COCO**: `annotations/instances_<split>.json` with the images under `images/<split>/`, next to the JSON, or in a
folder named after the split:

```
my-coco/
  annotations/instances_train.json   annotations/instances_val.json
  images/train/*.jpg                 images/val/*.jpg
```

Check the data before you spend GPU time on it:

```sh
folo dataset check my-data --fix-dry-run   # every broken label, corrupt image, duplicate, out-of-range box; exit 1 if any
folo dataset stats my-data                 # class balance, box sizes, boxes per image
folo dataset convert my-coco my-yolo --to yolo   # convert between YOLO and COCO (images are copied unchanged)
```

Then train. The defaults follow [MODEL.md](MODEL.md) §8: 100 epochs, batch 16, 640 px, `--optimizer auto`, mosaic
with the last 10 epochs mosaic-free, HSV, flips, scale and translate jitter, and early stopping after
`--patience 100` epochs:

```sh
folo train --data my-data --model folo-s --epochs 100 --imgsz 640
```

Things to know:
- **Sizes**: the sizes are `folo-n`, `folo-s`, `folo-m`, `folo-l` and `folo-x`. folo-n is 2.6 M parameters and
  6.5 GFLOPs at 640; folo-s is 9.5 M and 21.6 GFLOPs.
- **Small objects**: use a larger `--imgsz` (a multiple of 32) when targets are only a few pixels across at 640 px.
- **Data loading**: with large JPEGs and mosaic, the data loader can set the pace. `--workers N` (default
  `min(cores, 8)`) adds decode threads.
- **Device**: `--device cpu` or `--device gpu:<n>` pins the device. `gpu:<n>` without a usable adapter fails with
  `device_not_found` (exit 3), and the hint lists the devices this machine has.
- **Mixed precision**: `--amp` trains with f16 activations on the GPU's tensor cores and f32 master weights, about
  1.6× (folo-n) to 1.85× (folo-s) faster per step with about 40% less memory ([MODEL.md](MODEL.md) §8.8). On a
  machine without tensor cores it fails with `not_supported` (exit 4) before anything is written.
- **NMS-free head**: `--head o2o` also trains a one-to-one head, so the model predicts end to end without NMS
  ([MODEL.md](MODEL.md) §6.6); `val`, `predict` and `export` use it automatically.
- **Watching a run**: add `--json` and read stdout line by line. Each `epoch` record carries the validation metrics,
  and the run ends with a `done` record that names the weights.

Two complete recipes with their results: [examples/shapes](../examples/shapes/README.md) (the convergence gate, CPU
vs GPU) and [examples/coco](../examples/coco/README.md) (folo-n and folo-s pretrained from scratch on COCO 2017). The
convergence gate is FOLO's convergence test: a synthetic shapes dataset that a correct trainer must learn to a fixed
mAP bar.

## 5. Fine-tune

Pass a `.folo` file instead of an architecture name to start from those weights:

```sh
folo train --data my-data --model runs/shapes-n/weights/best.folo --epochs 30 --imgsz 320
```

The tensors that match are loaded. If your dataset's class list differs from the weights' classes, the class outputs
are re-initialised, and everything else is kept. As a check: fine-tuning the shapes model above on a new
`star, ring, heart` dataset reached mAP50 0.32 after 2 epochs, while a from-scratch model is near 0 at that point.

### Fine-tune from pretrained weights

`folo-n-coco-v3` is folo-n trained from scratch on COCO ([examples/coco](../examples/coco/README.md)): val2017 AP
0.340 after 78 epochs (`folo-n-coco`, AP 0.314 after 50 epochs, and `folo-n-coco-v2`, AP 0.317, are the earlier
ones). `folo-s-coco` is the larger folo-s, trained from scratch on COCO for 56 epochs: val2017 AP 0.416 (fine-tuning from
it takes the weights' architecture, folo-s). `folo weights list` shows every published name. Two commands fine-tune
one on your dataset:

```sh
folo weights pull folo-n-coco-v3                                  # into ~/.folo/weights/, checksum verified
folo train --data my-data --weights folo-n-coco-v3 --epochs 20    # takes the weights' architecture (folo-n)
```

The weights are published GitHub releases, and `folo weights pull` downloads them over HTTPS: the `folo` binary is
all it takes. `train` never downloads by itself: without the pull it stops with
`weights_not_found` and the hint to run `folo weights pull folo-n-coco-v3` first.

Studio and MCP list downloaded published weights as models once pulled (`folo weights pull`), or you can copy a `.folo`
file into `<data dir>/models/` ([CLI.md, Weights](CLI.md#weights)).

`--resume` is different: it continues an *interrupted* run with its stored settings. To train further after a run
**completed**, start a new run with `--model .../best.folo`.

## 6. Export

```sh
folo export --model runs/shapes-n/weights/best.folo                  # → runs/shapes-n/weights/best-f16.folo
folo export --model runs/shapes-n/weights/best.folo --format folo    # → best-fused.folo (f32)
folo export --model runs/shapes-n/weights/best.folo --format onnx    # → best.onnx (ONNX opset 17, f32)
```

Export fuses BatchNorm into the convolutions and keeps only the EMA weights. The default `folo-f16` stores them as
f16: 5.2 MB for folo-n, against 42 MB for the full checkpoint. Exports load anywhere a model is expected:
`folo predict`, `folo val`, `folo serve`'s model list and the browser. `.folo` files are valid safetensors files
([FORMAT.md](FORMAT.md)).

`--format onnx` writes an ONNX graph for other runtimes (onnxruntime, TensorRT, OpenVINO, the CoreML converters):
input `images: [1, 3, S, S]` (RGB in `[0, 1]`, `S` the model's imgsz), outputs the raw head `box_raw` / `cls_raw`.
`--decode` puts the box decode in the graph (one `detections: [B, A, 4 + nc]` output: `x1 y1 x2 y2` in input pixels,
then class scores; you still run NMS), `--dynamic` makes batch, height and width symbolic, `--half` writes f16. Class
names are in the model's metadata (`names`). Details: [CLI.md, ONNX export](CLI.md#onnx-export).

```sh
folo export --model runs/shapes-n/weights/best.folo --format onnx --decode --output best-decoded.onnx
# the shapes model is 320 px and the graph is static, so the input is [1, 3, 320, 320]; this prints (1, 2100, 7)
pip install onnxruntime numpy   # the check below needs them
python -c "import onnxruntime as ort, numpy as np; s = ort.InferenceSession('best-decoded.onnx'); print(s.run(None, {'images': np.zeros((1, 3, 320, 320), np.float32)})[0].shape)"
```

## 7. FOLO Studio

Studio is the web app for datasets, runs, models and the Playground. Build it once, then serve it:

```sh
sh scripts/build-wasm.sh        # optional: the in-browser runtime for the Playground (PowerShell: scripts\build-wasm.ps1)
                                # it needs wasm-bindgen-cli at Cargo.lock's version; the script prints the install command
cd web && npm ci && npm run build && cd ..
folo serve                      # http://127.0.0.1:7878
```

- **Data dir.** `folo serve` keeps its datasets, runs, models and exports in the data dir: `--data-dir`, or
  `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`.
- **In Studio.** Register a dataset folder, or generate a synthetic one. Start a run and follow it live, with
  metrics, validation samples, per-class mAP and logs.
- **Models.** Models are the finished runs' weights, plus every `.folo` file under `<data dir>/models/`. To bring in
  a model you trained with the CLI, copy it there: `mkdir -p ~/.folo/models && cp runs/shapes-n/weights/best.folo
  ~/.folo/models/shapes-n.folo` (with `--data-dir D`, use `D/models/`).
- **Mock mode.** `folo serve --mock` serves deterministic demo data instead, which is handy for a look around.
- **Ports.** If port 7878 is taken, `folo serve` says so and exits; `--port 0` picks a free one. Its first line
  (`FOLO 0.1.0 … serving the real store on http://127.0.0.1:<port>`), or with `--json` the `listening` record's
  `url`, shows the address.

The desktop app (`apps/desktop`, `cargo run --release` there) shows the same screens natively and connects to the
same server ([apps/desktop/README.md](../apps/desktop/README.md)).

## 8. Run the model in a browser

`crates/folo-wasm` compiles the predictor to wasm. It runs on **WebGPU** (folo-n at 640 in about 14 ms in Chrome)
or, without WebGPU, on the wasm CPU (single-threaded, about 0.5 s) ([WASM.md](WASM.md)).

- **In Studio**: build the wasm module (`sh scripts/build-wasm.sh`) *before* `npm run build`. Then the
  Playground's "Run on" switch offers **WebGPU** and **CPU** next to **Server**. Studio downloads the model's f16
  export from `GET /api/models/{id}/weights?format=f16` once and runs every image in the page.
- **In your own page**:

  ```js
  import init, { FoloDetector } from "/folo-wasm/folo_wasm.js";
  await init();
  const bytes = new Uint8Array(await (await fetch("shapes-n-f16.folo")).arrayBuffer());   // any .folo file
  const det = await FoloDetector.load(bytes, { backend: "auto" });                       // WebGPU, else CPU
  const res = await det.detect(imageData, undefined, undefined, { conf: 0.25 });          // /api/predict's shape
  ```

- **Check it against the native predictor.** The page should find the same boxes as `folo predict`:

  ```sh
  folo predict --model runs/shapes-n/weights/best.folo img.jpg --imgsz 640 --device cpu --json > native.json
  node crates/folo-wasm/demo/test.mjs --model runs/shapes-n/weights/best.folo --image img.jpg --imgsz 640 --expect native.json
  ```

  `test.mjs` serves the wasm module that `sh scripts/build-wasm.sh` builds and uses `playwright-core` from
  `web/node_modules`, so run both (the script, then `npm ci` in `web/`) first.

## 9. From an agent

Everything above is also available to agents in two ways:
- **MCP**: `folo mcp`, with 22 tools, `folo://` resources and prompts. It keeps its datasets and runs in the same
  data dir as `folo serve` (`~/.folo` by default, `--data-dir` to change it). To register it:
  `claude mcp add folo -- folo mcp`.
- **CLI**: any command with `--json`. stdout then carries JSON Lines only, and the exit codes are stable.

[AGENT_COOKBOOK.md](AGENT_COOKBOOK.md) has the recipes in both forms. [MCP.md](MCP.md) is the contract.

## Troubleshooting

| symptom | fix |
|---|---|
| `folo serve` says the web app is not built | `(cd web && npm ci && npm run build)` from the repo root, or pass `--web-dir` |
| `folo serve` cannot bind its port | another server holds port 7878: `--port 0` picks a free port |
| Playground shows "does not include the in-browser runtime" | run `scripts/build-wasm.sh`, then rebuild Studio |
| training is on the CPU | `folo doctor`: is a GPU listed? Try `FOLO_GPU_BACKEND=dx12` or `vulkan` |
| a command failed in a script | the last stdout line of `--json` is `{"error":{"code","message","hint"}}`; exit codes are 1 runtime, 2 usage, 3 not found, 4 not implemented |
