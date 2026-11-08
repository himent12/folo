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

- Why: fast (Rust, hand-written kernels, GPU through wgpu everywhere including the browser), small, hackable, and
  built to be driven by agents (`--json`, MCP server, stable error codes).
- Read: [ARCHITECTURE](docs/ARCHITECTURE.md) · [API](docs/API.md) · [AGENTS](AGENTS.md)

Licensed under either of MIT or Apache-2.0 at your option.
