# Contributing to FOLO

Bug reports, measurements on other hardware and pull requests are all welcome. This page is the short version;
[AGENTS.md](AGENTS.md) has the full rules (they apply to people and to coding agents alike) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the crates fit together.

## Finding your way around

- `crates/`: the Rust engine (folo-core, folo-gpu, folo-nn, folo-data, folo-vision) and the services and the `folo`
  binary on top of it (folo-server, folo-mcp, folo-cli and their helpers), one cargo workspace.
- `web/`: FOLO Studio, the web app. `apps/desktop/`: FOLO Desktop and its UI framework, a second cargo workspace.
- `docs/`: the documentation. `examples/`, `bench/`: worked runs and the benchmark harnesses.
- `tools/`, `scripts/`, `site/`, `design/`: CI helpers, the build and install scripts, the docs site, the shared
  design tokens.

The [README's map](README.md#architecture-at-a-glance) links every folder's README, and
[crates/README.md](crates/README.md) lists the crates in dependency order. To find the files a change touches, use
the "Where do I change ...?" table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); [docs/README.md](docs/README.md)
lists every doc by what it answers.

A first contribution is easiest where no GPU is needed: the docs and the examples (checked by `npm run check` in
`site/`), the CLI, the server and the dataset tools, which CI tests on runners without a GPU. A change to
`crates/folo-gpu` or a GPU path needs a machine with a GPU to test (see Building).

## Reporting a problem

Open an issue with what you ran, what happened and what you expected. `folo doctor --json` prints the version, the
OS, the CPU and the GPU adapters, which answers most first questions; `--json` on the failing command gives the error
code and hint. Numbers from other machines are useful too: say which GPU, driver and OS, and how much else was running.

For a vulnerability, do not open an issue: see [SECURITY.md](SECURITY.md).

## Building

You need Rust stable (`rust-toolchain.toml` pins the channel and the components). Node is needed only for Studio and
the docs site: CI uses Node 22 for Studio and Node 24 for the site, whose scripts are TypeScript run by `node` itself.

```sh
cargo folo doctor                                     # builds the CLI (`cargo folo` = `cargo run --release -p folo-cli --`)
cargo build --profile fast -p folo-cli                # fast local CLI/server rebuilds (no cross-crate LTO, incremental; target/fast/)
cargo test --workspace                                # the engine, the server, the CLI
cargo test -p folo-core                               # fast inner loop: test one crate while iterating
bash tools/test-tmp.sh cargo test --workspace         # same as CI: fails on leftover top-level folo-* scratch dirs
(cd web && npm ci && npm run check && npm test)       # FOLO Studio
(cd apps/desktop && cargo test --workspace)           # FOLO Desktop and folo-ui: their own cargo workspace
(cd site && npm ci && npm run check && npm run build) # the docs site: every link, anchor, image and full build
```

The GPU tests need a Vulkan, DX12 or Metal adapter and skip without one. CI's hosted runners have no GPU, so if your
change touches `crates/folo-gpu` or a GPU path, run the tests on a machine that has one and say so in the pull request.
If you work across multiple git worktrees or want build output on a secondary disk, set `CARGO_TARGET_DIR` and
`FOLO_TEST_TMP` (or `TEST_TMP` when running via `tools/test-tmp.sh`, which overrides `FOLO_TEST_TMP`) to keep scratch
and build directories isolated.

## What a pull request must pass

- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace`.
- `bash tools/file-size.sh`: no source file of 1000 lines or more. Split by responsibility, not by line count.
- For `web/`: `npm run check` and `npm test` (plus `npm run qa` for layout and load-fault checks on the production build).
- For `apps/desktop/`: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo test --workspace` in that workspace (pass `--workspace` so `folo-ui` is checked alongside the app).
- For `crates/folo-wasm/` (or its engine dependencies): install the `wasm32-unknown-unknown` target
  (`rustup target add wasm32-unknown-unknown`) and the `wasm-bindgen-cli` version pinned in `Cargo.lock`, then run
  `cargo clippy --target wasm32-unknown-unknown -p folo-wasm -- -D warnings`, `sh scripts/build-wasm.sh`, and
  `npx --yes -p typescript@6.0 tsc -p crates/folo-wasm/demo/tsconfig.json`.
- For any Markdown: `npm run check` and `npm run build` in `site/`, which fails on a broken link, anchor or image.
- For a number in a chart: `node tools/charts/render.mjs` after changing `tools/charts/data.json`
  ([docs/BENCHMARKS.md, Charts](docs/BENCHMARKS.md#charts)); CI runs `node tools/charts/render.mjs --check`.

### First PR checklist

Before opening a pull request:
1. Branch from `main` as `<area>/<short-topic>` and title the PR `area: imperative summary`.
2. Run the checks above for every workspace you touched (plus `bash tools/file-size.sh`).
3. Update the matching `docs/*.md` when behaviour or an interface changes, the crate or folder `README.md` when
   adding, moving or removing a module, and `docs/ARCHITECTURE.md` when the map or change table no longer matches, then
   run `(cd site && npm run check)`.
4. If you changed a hot path, include before/after alternated benchmark numbers in the PR description.

[docs/CI.md](docs/CI.md) lists what each workflow runs.

## Conventions

- **The ML stack is FOLO's own.** Tensors, kernels, autograd, layers, optimizers, the detector, losses, NMS, metrics,
  augmentation and the GPU kernels are written here. Do not add an ML framework; commodity crates do commodity jobs.
- **Correct first.** Every op has tests, every differentiable op a finite-difference gradient check, every GPU kernel
  a test against the CPU reference within a stated tolerance.
- **Tests are quick and deterministic**: seeded, about a second each at most, the smallest model on a few small
  images. A test that needs a folder takes a `folo_test_support::TestDir`.
- **Fast is a feature.** A change to a hot path reports numbers, measured against the build before it on the same
  machine, the two alternated (`cargo bench -p folo-vision --bench gpu`, `--bench cpu_e2e`, `--bench train_step`;
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md) describes the method). Negative results are worth writing down as well.
- **Docs in the same pull request.** A change to behaviour or an interface updates the matching file in `docs/`; a
  module added, moved or removed updates that crate's README.
- **Names.** Branches `<area>/<short-topic>` (`core/autograd`, `ui/run-view`); commit and pull request titles
  `area: imperative summary` (`core: add broadcasting binary ops`).
- **No secrets, ever**, in code, tests, logs or pull requests.

A pull request says what changed, why, how it was tested, and the benchmark numbers if it touches a hot path. Keep
it focused; a large feature can be a stack of pull requests. Every pull request is reviewed before it merges.

## Using AI tools

Most of FOLO is built by a team of specialized coding agents (orchestration, implementation, app play-testing and
review, and writing), and contributions made with AI tools are welcome on the same terms as any other: you have read
the change, it passes the checks above, and the pull request says how it was tested. [AGENTS.md](AGENTS.md) holds
the rules and subagent role breakdown; point your agent at it (and see [docs/notes/PROCESS.md](docs/notes/PROCESS.md)
for the maintainers' T3 Code subagent loop).

## Licence

FOLO is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. Unless you state
otherwise, any contribution you intentionally submit for inclusion in FOLO, as defined in the Apache-2.0 licence, is
dual licensed as above, without any additional terms or conditions.
