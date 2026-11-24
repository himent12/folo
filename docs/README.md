# FOLO documentation

Every file in `docs/`, grouped by what you want to do, with the question each one answers. The same pages, with
search, are the [docs website](../site/README.md). The repository's front page is the [README](../README.md); the
code's own map is [crates/README.md](../crates/README.md) and the README in each crate.

## Use FOLO

| doc | what it answers |
|---|---|
| [QUICKSTART](QUICKSTART.md) | How do I install FOLO, train on my own YOLO or COCO folder, fine-tune, evaluate, export and run in the browser? |
| [CLI](CLI.md) | What does each `folo` command do, with which flags, `--json` records and exit codes? |
| [LABELING](LABELING.md) | How do I turn unlabeled images into a dataset with Claude Code, Codex or Antigravity, and what does it cost? |
| [CLOUD](CLOUD.md) | How do I train on a rented GPU (RunPod) or another FOLO server, and how are instances never left behind? |
| [BENCH](BENCH.md) | How do I build a vision benchmark from my own images and score any model on it? |

The examples are worked runs: [shapes](../examples/shapes/README.md) (the convergence test) and
[coco](../examples/coco/README.md) (pretraining on COCO 2017).

## Drive FOLO from an agent or a program

| doc | what it answers |
|---|---|
| [AGENT_COOKBOOK](AGENT_COOKBOOK.md) | How does an agent do a common task, as MCP tool calls and as `folo ... --json` commands? |
| [MCP](MCP.md) | What are `folo mcp`'s tools, resources, prompts and errors? |
| [API](API.md) | What is the HTTP + SSE contract of `folo serve`, used by Studio, Desktop and `folo mcp --server`? |

[llms.txt](../llms.txt) is an index of these docs written for agents.

## Use the apps

- [FOLO Studio](../web/README.md): the web app that `folo serve` hosts; its dev server, checks and tests.
- [FOLO Desktop](../apps/desktop/README.md): the native app; running it, connecting it to `folo serve`, `--mock`.

Both apps talk to `folo serve` over the HTTP API, so [API](API.md) is the contract they share.

## Understand the internals

Read [ARCHITECTURE](ARCHITECTURE.md) first: it is the map the others hang from.

| doc | what it answers |
|---|---|
| [ARCHITECTURE](ARCHITECTURE.md) | Which crates and folders exist, how a request or a training run flows through them, and where do I change X? |
| [CORE](CORE.md) | How do folo-core's tensors, CPU kernels and autograd work? |
| [GPU](GPU.md) | How does the wgpu backend work: the buffer pool, the WGSL kernels, adapter selection, testing? |
| [NN](NN.md) | What do folo-nn's layers, init, optimizers, LR schedules, EMA and `.folo` IO do? |
| [DATA](DATA.md) | How are YOLO and COCO datasets read, checked, converted, augmented and loaded? |
| [VISION](VISION.md) | Where in folo-vision do the model, loss, assigner, trainer and predictor live, and how do they fit? |
| [MODEL](MODEL.md) | What exactly is the detector: blocks, head, loss, assigner, training recipe, inference path (the normative spec)? |
| [METRICS](METRICS.md) | How are mAP, precision, recall and the confusion matrix computed, and how do they match pycocotools? |
| [FORMAT](FORMAT.md) | What is in a `.folo` weights or checkpoint file? |
| [SERVER](SERVER.md) | How is folo-server built: the router, the Store and the services behind the API? |
| [WASM](WASM.md) | How does folo-wasm run the detector in the browser: the build, the JS API, performance? |

## Measure and compare

| doc | what it answers |
|---|---|
| [BENCHMARKS](BENCHMARKS.md) | How fast and how accurate is FOLO against Ultralytics on one machine, measured how? |

FOLO's speed numbers come from `cargo bench` (the benches live in `crates/<crate>/benches`); the Ultralytics
baselines and the accuracy harnesses are Python scripts in [bench/](../bench/README.md). [BENCH](BENCH.md) is a
different thing: vision benchmarks that grade models on your own images.

## Contribute and release

| doc | what it answers |
|---|---|
| [CONTRIBUTING](../CONTRIBUTING.md) | How do I build FOLO, and what must a pull request pass? |
| [AGENTS](../AGENTS.md) | What rules every contributor follows, how the agentic subagent team is organized, and what is the repo layout? |
| [CI](CI.md) | What does each CI workflow run, on which runners, and how are the tests kept fast? |
| [RELEASE](RELEASE.md) | How is a release cut: versioning, the tag to draft release workflow, the install scripts? |
| [ROADMAP](ROADMAP.md) | What is done, in progress and next? |
| [CHANGELOG](../CHANGELOG.md) | What did each release change? |

## The engineering notebooks

[notes/](notes/README.md) holds the maintainers' agent process, the optimisation notebooks behind the benchmark
numbers, negative results and the dogfooding reports. They are not needed to use FOLO.

## A reading order for a new contributor

1. The [README](../README.md): what FOLO is, its architecture at a glance and where each folder is.
2. [CONTRIBUTING](../CONTRIBUTING.md), then [AGENTS](../AGENTS.md): the build, the checks and the rules.
3. [ARCHITECTURE](ARCHITECTURE.md): the map, the flows and the "Where do I change ...?" table.
4. [crates/README.md](../crates/README.md), then the README of the crate or folder you will change.
5. The doc for that part: [CORE](CORE.md), [GPU](GPU.md), [NN](NN.md), [DATA](DATA.md), [VISION](VISION.md) with
   [MODEL](MODEL.md), [SERVER](SERVER.md) with [API](API.md), or [WASM](WASM.md).
