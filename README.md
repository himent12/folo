# FOLO

**FOLO Only Looks Once.** A from-scratch, very fast vision framework in Rust: tensors, autograd, CPU and GPU (wgpu)
kernels, a YOLO-style real-time object detector, training, evaluation and export, and **FOLO Studio**, a web app
for datasets, training runs and live inference.

> Status: early. See [docs/ROADMAP.md](docs/ROADMAP.md).

```sh
cargo folo --help                 # the CLI (alias for: cargo run --release -p folo-cli --)
cargo folo dataset synth shapes   # make a synthetic dataset
cargo folo train --data shapes --model folo-n --epochs 50
cargo folo predict --model runs/shapes-n/best.folo image.jpg
cargo folo serve                  # FOLO Studio on http://127.0.0.1:7878
```

## Running the server

```sh
cargo run -p folo-cli -- serve --mock   # mock data on http://127.0.0.1:7878 (API under /api)
cargo run -p folo-cli -- serve          # real store: datasets from disk; training / inference answer 501 for now
```

`--mock` serves deterministic datasets, models, finished runs and one live training run whose SSE stream advances in
real time, so FOLO Studio (`web/`, Vite dev server on :5173) and FOLO Desktop can be built before the engine exists.
The server also serves the built Studio from `web/dist` (`--web-dir`) when it exists. Flags: `--host`, `--port`
(`0` picks a free one), `--data-dir`, `--cors-any`, and the global `--json`. See [API](docs/API.md) and
[CLI](docs/CLI.md).

- Why: fast (Rust, hand-written kernels, GPU through wgpu everywhere including the browser), small, hackable, and
  built to be driven by agents (`--json`, MCP server, stable error codes).
- Read: [ARCHITECTURE](docs/ARCHITECTURE.md) · [API](docs/API.md) · [AGENTS](AGENTS.md)

Licensed under either of MIT or Apache-2.0 at your option.
