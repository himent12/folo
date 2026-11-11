# CI

Five workflows check the code, one per part of the repo. A pull request runs only the ones whose files it touches
(`paths` filters); every push to `main` runs all of them, so a filter that misses a dependency shows up there.

| workflow | runs for | what it does |
|---|---|---|
| `ci.yml` | everything except `web/`, `site/`, `design/`, `apps/desktop/`, `docs/img/` and the other workflows (plus the few files in those folders that a Rust test reads: the package versions and Studio's command-line fixtures) | `rust (linux)`: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace` with the ONNX parity test required. `rust (windows)`: clippy and the tests on the runner with a GPU, so the GPU tests run there |
| `web.yml` | `web/`, `design/` | `check`: `npm ci`, `npm run check`, `npm test`. `qa (1/3)` to `qa (3/3)`: `npm run qa`, the layout checks on the production build (blank sections, overflow, covered controls and cut selects at tablet and phone sizes, web/DESIGN_DIRECTION.md §6; the runner's Chrome, else Playwright's headless Chromium), a third of the states each (`FOLO_QA_SHARD`) |
| `site.yml` | every pull request (the Markdown links to files all over the repo; about 20 s) | `bash tools/file-size.sh` (no source file of 1000 lines or more, AGENTS.md), then the docs site: `npm ci && npm run check && npm run build` in `site/`, which fails on a broken link, anchor or image in the Markdown (site/README.md) |
| `wasm.yml` | `crates/folo-wasm/` and the crates it depends on, `Cargo.toml`, `Cargo.lock`, `scripts/build-wasm.sh` | the browser build: `cargo clippy --target wasm32-unknown-unknown -p folo-wasm` and `scripts/build-wasm.sh` with the wasm-bindgen CLI matching `Cargo.lock` (docs/WASM.md) |
| `desktop.yml` | `apps/desktop/` (the app and folo-ui), `design/`, the crates the app builds and the root `Cargo.toml` they inherit from | FOLO Desktop and folo-ui (below). `desktop (windows)`: fmt, clippy and the tests of both, again for folo-ui with its portable-text and winit features, and the temp-dir hygiene test. `desktop (linux)`: clippy and the tests, a headless screenshot, and folo-ui type-checked for macOS |

ONNX export parity (docs/CLI.md, ONNX export) is part of `cargo test --workspace`: `crates/folo-cli/tests/onnx.rs`
runs the exported graphs in onnxruntime and skips without a Python. `rust (linux)` gives it one (a Python 3.12 venv
with pinned `onnx`, `onnxruntime` and `numpy`, kept between runs on a self-hosted runner) and sets
`FOLO_ONNX_REQUIRED=1`, so there the test fails instead of skipping when the venv is broken.

## FOLO Desktop
`desktop.yml` checks the desktop workspace (`apps/desktop`: the app and `ui/`, folo-ui) with `--workspace`, since a
bare `cargo test` there builds only the app.

- **`desktop (windows)`**, where the app ships: `cargo fmt --check`; clippy of the workspace; clippy of folo-ui with
  `--features portable-text,winit,test-support` (the cosmic-text backend the deterministic text tests use, and
  the winit backend, both opt-in on Windows); the tests of the workspace and of folo-ui with those features,
  through `tools/test-tmp.sh`; and the temp-dir hygiene test (ci.yml does not run for a change to the app alone).
- **`desktop (linux)`**: clippy and the tests of the workspace on Linux, where the winit backend and cosmic-text are
  plain dependencies. Every test is headless, so no display is needed. Building needs no system package: wayland
  and Xlib are loaded at run time (`wayland-sys` and `tiny-xlib` on their `dlopen` features; pkg-config is only
  asked for the library folder and may be missing), `drm-sys` ships its bindings, and the only C code (ring, for
  the HTTP client's TLS) needs the runner's C compiler. Then a smoke test: `folo-desktop --scenario overview
  --shot <png>` must exit 0 and write a 1280 x 800 PNG, which runs the whole UI stack (runtime, layout,
  cosmic-text, the renderer) on the headless platform.
- **macOS, in the Linux job:** `tools/cross-check.sh` with `CROSS_TARGETS="aarch64-apple-darwin
  x86_64-apple-darwin"` runs `cargo check` and `cargo clippy -D warnings` on folo-ui for both targets. This proves
  that folo-ui (the winit backend and cosmic-text on macOS, AppKit through objc2) type-checks and lints for macOS.
  It does not prove that it links or runs there: nothing is linked or run on macOS, and no macOS runner is used.
  The app crate is not checked for macOS: ring (the HTTP client's TLS) compiles C for the target in its build
  script, so `cargo check -p folo-desktop --target aarch64-apple-darwin` needs a C compiler for macOS. On the
  Windows dev PC it fails there (`failed to find tool "cc"`); a Linux runner has no macOS SDK either.

## Keeping CI fast
- **Tests are quick or they are not in the suite.** A test trains the smallest model on a handful of small images
  for a few steps; several tests that need the same run or server share it. Aim for under a second per test.
  `#[ignore]` is for tests that need what CI does not have (network, credentials, a download) and for a documented
  manual acceptance run (docs/MODEL.md §10.4, the overfit test), not for slow ones.
- **The dev profile** (root `Cargo.toml`) builds folo-core, folo-nn, folo-data and folo-vision at `opt-level = 3`
  like the dependencies, because tests train and predict with them, and keeps only line tables as debug info
  (file and line in backtraces): less to generate and to link into each of the test binaries.
- **A new job needs a reason to run on every pull request.** Give a new part of the repo its own workflow with a
  `paths` filter, as above. If a Rust test starts reading a file under `web/`, `site/`, `design/` or
  `apps/desktop/`, add that file to `ci.yml`'s `paths`.
- **The layout QA runs on the production build**, served as static files: every state opens a fresh browser context
  with an empty cache, and the build is one script where the dev server answers a request per module. Phone states
  are checked once, under a coarse pointer (the stricter case). A state still costs a few CPU seconds of browser
  work (the page runs seconds of simulated time), so CI splits the states over three runners.

## Test temp dirs
Tests make their scratch folders with `folo_test_support::TestDir` (`crates/folo-test-support`): a fresh
`folo-<tag>-<pid>-<n>` folder under `FOLO_TEST_TMP` when that is set, else the system temp dir, removed when the guard
drops, also when the test panics. A spawned `folo serve` / `folo train` / `folo mcp` sits in a `ChildGuard` that kills
it first, since Windows cannot delete files a running process holds open. `crates/folo-test-support/tests/hygiene.rs`
fails on any `std::env::temp_dir()` in a crate's `src/` or `tests/` (and in `apps/desktop/src`, and folo-ui's `src/`,
`tests/` and `examples/`), so a new test cannot
bypass the guard by accident. (Before this, `cargo test --workspace` runs left `folo-*` folders behind; 3,209 of them,
about 20 GB, once filled the dev PC's C: drive.)

CI runs `cargo test` (the `rust` jobs) through `tools/test-tmp.sh`. The script points `FOLO_TEST_TMP` and
the temp dir itself (`TMPDIR`, `TMP`, `TEMP`) at a new `$RUNNER_TEMP/test-tmp.XXXXXX`, one per run, so concurrent runs
never share it. Test files therefore land on the runner's work drive (D: on the dev PC), not in the user's %TEMP%.
After the run it fails the step, listing the entries, if any `folo-*` entry is left there. Because the temp dir is
redirected too, the check also catches a binary that writes to the temp dir. Locally:
`TEST_TMP=D:/folo-build/test-tmp bash tools/test-tmp.sh cargo test --workspace` (`TEST_TMP` is the parent folder).
`FOLO_TEST_TMP` must be an absolute path, since cargo runs each test binary in its own package folder.

## Runners
Linux jobs run on GitHub-hosted runners (`ubuntu-latest`). Windows jobs run on the dev PC, which has the GPU the
GPU tests need. Two repository variables choose, each a JSON `runs-on` value for a self-hosted runner; an unset
variable means GitHub-hosted (`ubuntu-latest` / `windows-latest`):

| variable | value | runner |
|---|---|---|
| `RUNS_ON_LINUX` | unset | GitHub-hosted `ubuntu-latest` |
| `RUNS_ON_WINDOWS` | `["self-hosted","folo-win"]` | `folo-win-local`, `folo-win-local-2`: the dev PC, runners in `D:\gha\win`, `D:\gha\win2` (each with its own target dir `D:\gha\folo-target-<runner name>`) |

The repository is private, so GitHub-hosted jobs use the account's Actions minutes (Windows counts double, which
is the other reason Windows stays on the dev PC). When the account's billing is blocked, GitHub refuses to start
a hosted job: it fails within seconds with "recent account payments have failed or your spending limit needs to
be increased", and nothing in the repository can fix that (GitHub, Settings, Billing & plans).

Hosted jobs cache their build with `Swatinem/rust-cache` and the npm cache of `actions/setup-node`. To move a
platform to a self-hosted runner, set its variable (`gh variable set RUNS_ON_LINUX --body '["self-hosted","<label>"]'`);
to move it back, delete the variable (`gh variable delete RUNS_ON_LINUX`). Self-hosted jobs keep a persistent
`CARGO_TARGET_DIR` outside the checkout, so incremental builds stay warm. That folder only grows, so on a
self-hosted Linux runner `rust (linux)` starts from an empty one when the runner has under 20 GB free.

Until November 2026 the Linux jobs ran on boat.dev sandboxes registered as self-hosted runners. Each sandbox
expired after 24 hours and took its runner offline, and jobs for an offline runner queue instead of failing (a
pull request's checks once sat queued for an hour and a half), so they were removed.

The same variables drive every workflow, `release.yml` included (its Linux and Windows jobs, including the CLI
build matrix that its `plan` job writes). Release builds are the exception to the persistent target dir: they always start
from an empty one in `$RUNNER_TEMP` so no stale artifact reaches a shipped binary, and they need no tools
preinstalled beyond the basics listed in docs/RELEASE.md, Runners. macOS release builds have no self-hosted runner
and use GitHub-hosted `macos-latest`; set `FOLO_RELEASE_MACOS=false` while hosted runners are unavailable (hosted
billing must be fixed before macOS release builds can run).

## Operating the runners
- Status: `gh api repos/himent12/folo/actions/runners -q '.runners[] | .name + " " + .status'`.
- Windows (dev PC): start with `Start-Process D:\gha\win\run.cmd -WindowStyle Hidden`. After a reboot, start it again.
  Prerequisite: Git for Windows' `bin` and `usr\bin` on the runner's PATH (actions such as `dtolnay/rust-toolchain`
  run bash). It is set in `D:\gha\win\.env` (`PATH=...`), which the runner reads at start.
- Linux: nothing to operate, the runners are GitHub's. A hosted job that fails in seconds with no steps was
  refused by GitHub (billing, above), not broken by the change under test.
- **Security.** Self-hosted runners execute code from PRs (tests, build scripts, npm lifecycle scripts) with the
  runner account's access. The Windows runner uses the developer account, which can read `~/.folo/boat.key` and the
  gh credentials. That is the same trust boundary FOLO's agents already work in, since they build and run the same
  code as the same user on the same machine, so the runner adds no new principal. It still means: never enable these
  runners for a public repository or for PRs from forks; keep dependencies pinned through Cargo.lock and
  package-lock.json; and review new dependencies in PRs. If FOLO ever accepts outside contributions, move CI to a
  dedicated account or VM with no developer credentials (or back to GitHub-hosted runners) first.
- An offline self-hosted runner does not trigger the fallback: its jobs queue (up to 24 h). If the dev PC is off
  for long, delete `RUNS_ON_WINDOWS` to run the Windows jobs on GitHub-hosted runners (which have no GPU).

## Unstable self-hosted Windows host
The dev PC (an i7-14700KF) occasionally crashes rustc or rustfmt under heavy parallel load (`STATUS_ACCESS_VIOLATION`
0xc0000005, `STATUS_ILLEGAL_INSTRUCTION` 0xc000001d, `STATUS_STACK_BUFFER_OVERRUN` 0xc0000409), a pattern consistent
with the Intel 13th/14th-gen instability issue. Until the host is fixed (BIOS / microcode update), the Windows steps in
`ci.yml`, `desktop.yml` and `release.yml` (its Windows `cargo build` steps) run through `tools/ci-retry.sh`, which on
self-hosted runners retries a step once **only when its output shows one of those crash codes**, and logs a warning.
Compile errors and failing (including flaky) tests are never retried.
