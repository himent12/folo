# The crates

The root cargo workspace (`members = ["crates/*"]` in the root `Cargo.toml`): the engine at the bottom, the services
and the `folo` binary on top. FOLO Desktop is not here: it is its own workspace in
[apps/desktop/](../apps/desktop/README.md). [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) shows how the crates work
together; each crate's README says what is in it and which file holds what.

## In dependency order

Bottom of the stack first. "Uses" lists the other FOLO crates in the crate's `[dependencies]` (dev-dependencies are
left out).

| crate | what it is | uses |
|---|---|---|
| [folo-core](folo-core/README.md) | Tensors, dtypes, shapes, the CPU kernels, reverse-mode autograd, the `Backend` trait | none |
| [folo-api](folo-api/README.md) | The serde types of docs/API.md (JSON Schemas behind the `schema` feature) | none |
| [folo-data](folo-data/README.md) | Image IO, YOLO and COCO datasets, augmentation, the data loader, synthetic datasets | none |
| [folo-gpu](folo-gpu/README.md) | The wgpu device, the buffer pool and the WGSL kernels; implements folo-core's `Backend` | core |
| [folo-nn](folo-nn/README.md) | Layers, init, optimizers, LR schedules, the `.folo` weight format | core |
| [folo-vision](folo-vision/README.md) | The detector (folo-n/s/m/l/x), loss, assigner, NMS, mAP, trainer, predictor, ONNX export | core, nn, gpu, data, api |
| [folo-bench](folo-bench/README.md) | Vision benchmarks (docs/BENCH.md): stress conditions, plans, the scores of an entry | core, data, vision, api |
| [folo-wasm](folo-wasm/README.md) | The detectors in the browser (wasm-bindgen; WebGPU or the wasm CPU), docs/WASM.md | core, nn, gpu, data, vision, api |
| [folo-client](folo-client/README.md) | A blocking HTTP client of docs/API.md, for `folo mcp --server` and cloud workers | api |
| [folo-cloud](folo-cloud/README.md) | Cloud compute providers behind one trait: RunPod and remote FOLO servers (docs/CLOUD.md) | client, api |
| [folo-server](folo-server/README.md) | `folo serve`: the axum HTTP + SSE API behind Studio and Desktop; serves the built Studio | core, nn, gpu, data, vision, bench, cloud, client, api |
| [folo-mcp](folo-mcp/README.md) | `folo mcp`: the MCP server (stdio) over folo-server's Store, in process or proxying `folo serve` | server, data, client, api |
| [folo-cli](folo-cli/README.md) | The `folo` binary: every command, `--json` output, exit codes | server, mcp, vision, nn, gpu, data, client, api |
| [folo-test-support](folo-test-support/README.md) | Test helpers: `TestDir`, `TestFile`, `ChildGuard`, `FOLO_TEST_TMP` | none (a dev-dependency only) |

FOLO Desktop (`apps/desktop/`) uses folo-api, folo-bench and the engine crates (folo-core, folo-data, folo-gpu,
folo-nn, folo-vision) by path, the last of them to run models in its Playground.

## Dependency rules

These hold in today's `Cargo.toml` files; a change that breaks one needs a reason in its pull request.

- **The engine knows nothing of the services.** folo-core, folo-gpu, folo-nn, folo-data and folo-vision depend on
  none of folo-bench, folo-client, folo-cloud, folo-server, folo-mcp and folo-cli.
- **folo-core is the bottom.** It depends on no other FOLO crate. folo-gpu plugs in from above by implementing
  folo-core's `Backend` trait (`GpuBackend`), so folo-core has no dependency on it.
- **folo-data stands alone.** It depends on no other FOLO crate, folo-core included.
- **folo-api is types only.** Its dependencies are serde, serde_json and, behind the `schema` feature, schemars; the
  feature is off by default to keep the crate light for FOLO Desktop.
- **folo-client is transport only.** It depends on folo-api alone and knows the API's conventions, not its routes.
- **folo-test-support is a dev-dependency**, never a normal one.
- **What folo-wasm uses must build for `wasm32-unknown-unknown`**: folo-core, folo-gpu, folo-nn, folo-data,
  folo-vision and folo-api ([docs/WASM.md](../docs/WASM.md), `scripts/build-wasm.sh`).
- **No ML frameworks, anywhere** ([AGENTS.md](../AGENTS.md)). A new dependency goes only into the crate that needs
  it and is declared in the root `[workspace.dependencies]`.

## Where to start reading the engine

1. [folo-core](folo-core/README.md): tensors, the CPU kernels and the autograd tape everything else builds on
   ([docs/CORE.md](../docs/CORE.md)).
2. [folo-nn](folo-nn/README.md): layers and optimizers on top of those tensors ([docs/NN.md](../docs/NN.md)).
3. [folo-gpu](folo-gpu/README.md): the same operations as WGSL kernels ([docs/GPU.md](../docs/GPU.md)).
4. [folo-data](folo-data/README.md): how images and labels become batches ([docs/DATA.md](../docs/DATA.md)).
5. [folo-vision](folo-vision/README.md): the detector, its loss, the trainer and the predictor
   ([docs/VISION.md](../docs/VISION.md), with [docs/MODEL.md](../docs/MODEL.md) as the spec).

For the services, start at [folo-server](folo-server/README.md) and [docs/SERVER.md](../docs/SERVER.md), then
[folo-cli](folo-cli/README.md).
