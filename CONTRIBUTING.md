# Contributing to FOLO

Bug reports, measurements on other hardware and pull requests are all welcome. This page is the short version;
[AGENTS.md](AGENTS.md) has the full rules (they apply to people and to coding agents alike) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the crates fit together.

## Reporting a problem

Open an issue with what you ran, what happened and what you expected. `folo doctor --json` prints the version, the
OS, the CPU and the GPU adapters, which answers most first questions; `--json` on the failing command gives the error
code and hint. Numbers from other machines are useful too: say which GPU, driver and OS, and how much else was running.

For a vulnerability, do not open an issue: see [SECURITY.md](SECURITY.md).

## Building

You need Rust stable (`rust-toolchain.toml` pins the channel and the components). Node 22 or newer is needed only for
Studio and the docs site.

```sh
cargo folo doctor                         # builds the CLI (`cargo folo` = `cargo run --release -p folo-cli --`)
cargo test --workspace                    # the engine, the server, the CLI
(cd web && npm ci && npm run check && npm test)       # FOLO Studio
(cd apps/desktop && cargo test --workspace)           # FOLO Desktop and folo-ui: their own cargo workspace
(cd site && npm ci && npm run check)                  # the docs site: every link, anchor and image in the Markdown
```

The GPU tests need a Vulkan, DX12 or Metal adapter and skip without one. CI's hosted runners have no GPU, so if your
change touches `crates/folo-gpu` or a GPU path, run the tests on a machine that has one and say so in the pull request.

## What a pull request must pass

- `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace`.
- `bash tools/file-size.sh`: no source file of 1000 lines or more. Split by responsibility, not by line count.
- For `web/`: `npm run check` and `npm test`. For `apps/desktop/`: fmt, clippy and the tests in that workspace.
- For any Markdown: `npm run check` in `site/`, which fails on a broken link or anchor.
- For a number in a chart: `node tools/charts/render.mjs` after changing `tools/charts/data.json`
  ([docs/BENCHMARKS.md, Charts](docs/BENCHMARKS.md#charts)).

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
- **Docs in the same pull request.** A change to behaviour or an interface updates the matching file in `docs/`.
- **Names.** Branches `<area>/<short-topic>` (`core/autograd`, `ui/run-view`); commit and pull request titles
  `area: imperative summary` (`core: add broadcasting binary ops`).
- **No secrets, ever**, in code, tests, logs or pull requests.

A pull request says what changed, why, how it was tested, and the benchmark numbers if it touches a hot path. Keep
it focused; a large feature can be a stack of pull requests. Every pull request is reviewed before it merges.

## Using AI tools

Most of FOLO was written by coding agents, and contributions made with them are welcome on the same terms as any
other: you have read the change, it passes the checks above, and the pull request says how it was tested. `AGENTS.md`
is written to be read by an agent; point yours at it.

## Licence

FOLO is licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. Unless you state
otherwise, any contribution you intentionally submit for inclusion in FOLO, as defined in the Apache-2.0 licence, is
dual licensed as above, without any additional terms or conditions.
