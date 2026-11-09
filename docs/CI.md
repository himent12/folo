# CI

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`: `cargo fmt --check` (Linux), `cargo clippy -D
warnings` and `cargo test --workspace` on Linux and Windows, the web app's `npm ci && npm run check && npm run build`, and
the browser build (`wasm`, Linux: `cargo clippy --target wasm32-unknown-unknown -p folo-wasm` and
`scripts/build-wasm.sh` with the wasm-bindgen CLI matching `Cargo.lock`, docs/WASM.md), and ONNX export parity
(`onnx`, Linux: a Python 3.12 venv with pinned `onnx`, `onnxruntime` and `numpy`, then `cargo test -p folo-cli --test
onnx` with `FOLO_ONNX_REQUIRED=1`, so the test fails instead of skipping when the venv is broken; docs/CLI.md, ONNX
export).

## Runners
The repository is private, so GitHub-hosted minutes are limited (Windows counts double). FOLO therefore runs CI on its
own runners, selected by two repository variables holding a JSON `runs-on` value:

| variable | value | runner |
|---|---|---|
| `RUNS_ON_LINUX` | `["self-hosted","folo-linux"]` | `folo-linux-boat`, `folo-linux-boat-2..4`: boat.dev `large` sandboxes (8 vCPU, 16 GB, Ubuntu 24.04), runner in `~/gha` |
| `RUNS_ON_WINDOWS` | `["self-hosted","folo-win"]` | `folo-win-local`, `folo-win-local-2`: the dev PC, runners in `D:\gha\win`, `D:\gha\win2` (each with its own target dir `D:\gha\folo-target-<runner name>`) |

Linux runners are managed with `python tools/gha_boat_runner.py up|renew|down <n>` (sandbox ids in
`~/.folo/gha_linux<n>_sid`).

To fall back to GitHub-hosted runners, delete the variable (`gh variable delete RUNS_ON_LINUX`); the workflow then
uses `ubuntu-latest` / `windows-latest`. Self-hosted jobs keep a persistent `CARGO_TARGET_DIR` outside the checkout,
so incremental builds stay warm.

The same variables drive `desktop.yml` and `release.yml` (its Linux and Windows jobs, including the CLI build
matrix that its `plan` job writes). Release builds are the exception to the persistent target dir: they always start
from an empty one in `$RUNNER_TEMP` so no stale artifact reaches a shipped binary, and they need no tools
preinstalled beyond the basics listed in docs/RELEASE.md, Runners. macOS release builds have no self-hosted runner
and use GitHub-hosted `macos-latest`; set `FOLO_RELEASE_MACOS=false` while hosted runners are unavailable (hosted
billing must be fixed before macOS release builds can run).

## Operating the runners
- Status: `gh api repos/himent12/folo/actions/runners -q '.runners[] | .name + " " + .status'`.
- Windows (dev PC): start with `Start-Process D:\gha\win\run.cmd -WindowStyle Hidden`. After a reboot, start it again.
  Prerequisite: Git for Windows' `bin` and `usr\bin` on the runner's PATH (actions such as `dtolnay/rust-toolchain`
  run bash). It is set in `D:\gha\win\.env` (`PATH=...`), which the runner reads at start.
- Linux (boat.dev): each sandbox has a 24 h TTL. Renew with `python tools/gha_boat_runner.py renew <n>` (it extends
  the TTL and restarts the listener if needed), replace with `down <n>` then `up <n>`. Runner `n = 1` is the original
  `folo-linux-boat`; `n >= 2` are `folo-linux-boat-<n>`. Sandbox ids are in `~/.folo/gha_linux<n>_sid` on the dev PC.
  The script fetches registration tokens from `gh` at runtime and never prints them. Never print or commit tokens or
  the boat key (docs/BOAT.md).
- **Security.** Self-hosted runners execute code from PRs (tests, build scripts, npm lifecycle scripts) with the
  runner account's access. The Windows runner uses the developer account, which can read `~/.folo/boat.key` and the
  gh credentials. That is the same trust boundary FOLO's agents already work in, since they build and run the same
  code as the same user on the same machine, so the runner adds no new principal. It still means: never enable these
  runners for a public repository or for PRs from forks; keep dependencies pinned through Cargo.lock and
  package-lock.json; and review new dependencies in PRs. If FOLO ever accepts outside contributions, move CI to a
  dedicated account or VM with no developer credentials (or back to GitHub-hosted runners) first.
- An offline runner does not trigger the fallback: jobs queue (up to 24 h). If a runner is down for long, delete the
  variable to switch to GitHub-hosted runners.

## Unstable self-hosted Windows host
The dev PC (an i7-14700KF) occasionally crashes rustc or rustfmt under heavy parallel load (`STATUS_ACCESS_VIOLATION`
0xc0000005, `STATUS_ILLEGAL_INSTRUCTION` 0xc000001d, `STATUS_STACK_BUFFER_OVERRUN` 0xc0000409), a pattern consistent
with the Intel 13th/14th-gen instability issue. Until the host is fixed (BIOS / microcode update), the Windows steps in
`ci.yml`, `desktop.yml` and `release.yml` (its Windows `cargo build` steps) run through `tools/ci-retry.sh`, which on
self-hosted runners retries a step once **only when its output shows one of those crash codes**, and logs a warning.
Compile errors and failing (including flaky) tests are never retried.
