# CI

Five workflows check the code, one per part of the repo (a sixth, `release.yml`, builds releases: docs/RELEASE.md).
A pull request runs only the ones whose files it touches (`paths` filters); every push to `main` runs all of them, so
a filter that misses a dependency shows up there.

| workflow | runs for | what it does |
|---|---|---|
| `ci.yml` | everything except `web/`, `site/`, `design/`, `apps/desktop/`, `docs/img/` and the other workflows (plus the few files in those folders that a Rust test reads: the package versions and Studio's command-line fixtures) | `rust (linux)`: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace` with the ONNX parity test required. `rust (windows)`: clippy and the tests on the runner with a GPU, so the GPU tests run there |
| `web.yml` | `web/`, `design/` | `check`: `npm ci`, `npm run check`, `npm test`. `qa (1/3)` to `qa (3/3)`: `npm run qa`, the layout checks on the production build (blank sections, overflow, covered controls and cut selects at tablet and phone sizes, and Overview at 1280 px, web/DESIGN_DIRECTION.md §6; the runner's Chrome, else Playwright's headless Chromium), a third of the states each (`FOLO_QA_SHARD`) |
| `site.yml` | every pull request (the Markdown links to files all over the repo; about 20 s) | `bash tools/file-size.sh` (no source file of 1000 lines or more, AGENTS.md), `node tools/charts/render.mjs --check` (the committed charts match their data, docs/BENCHMARKS.md, Charts), then the docs site: `npm ci && npm run check && npm run build` in `site/`, which fails on a broken link, anchor or image in the Markdown (site/README.md) |
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
  script, so `cargo check -p folo-desktop --target aarch64-apple-darwin` needs a C compiler for macOS. On
  Windows it fails there (`failed to find tool "cc"`); a Linux runner has no macOS SDK either.

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
about 20 GB, once filled a system drive.)

CI runs `cargo test` (the `rust` jobs) through `tools/test-tmp.sh`. The script points `FOLO_TEST_TMP` and
the temp dir itself (`TMPDIR`, `TMP`, `TEMP`) at a new `$RUNNER_TEMP/test-tmp.XXXXXX`, one per run, so concurrent runs
never share it. Test files therefore land on the runner's work drive, not in the user's %TEMP%.
After the run it fails the step, listing the entries, if any `folo-*` entry is left there. Because the temp dir is
redirected too, the check also catches a binary that writes to the temp dir. Locally:
`TEST_TMP=/tmp/folo-test-tmp bash tools/test-tmp.sh cargo test --workspace` (`TEST_TMP` is the parent folder).
`FOLO_TEST_TMP` must be an absolute path, since cargo runs each test binary in its own package folder.

## Runners
Jobs run on GitHub-hosted runners unless a repository variable says otherwise: `ubuntu-latest` for Linux (the release
workflow's Linux CLI build and its Linux install test use `ubuntu-22.04`, docs/RELEASE.md, Runners) and
`windows-latest` for Windows. A hosted runner has no GPU, so the tests that need one skip there. Two repository
variables move a platform to a self-hosted runner, each a JSON `runs-on` value:

| variable | example | effect |
|---|---|---|
| `RUNS_ON_LINUX` | `["self-hosted","linux"]` | the Linux jobs of every workflow run on that runner |
| `RUNS_ON_WINDOWS` | `["self-hosted","folo-win"]` | the Windows jobs run on that runner; with a GPU there, the GPU tests run in CI |

The maintainers ran the Windows jobs this way, on a machine with an RTX 4080 SUPER, while the repository was private:
that is where the GPU tests ran on every pull request. The Linux jobs ran on the same machine, on runners inside a
WSL2 distribution (Self-hosted runners, below). Set a variable with
`gh variable set RUNS_ON_WINDOWS --body '["self-hosted","<label>"]'` and delete it to go back
(`gh variable delete RUNS_ON_WINDOWS`).

Hosted jobs cache their build with `Swatinem/rust-cache` and the npm cache of `actions/setup-node`. Self-hosted jobs
keep a persistent `CARGO_TARGET_DIR` outside the checkout, so incremental builds stay warm. That folder only grows, so
on a self-hosted Linux runner `rust (linux)` starts from an empty one when the runner has under 20 GB free.

The same variables drive every workflow, `release.yml` included (its Linux and Windows jobs, including the CLI
build matrix that its `plan` job writes). Release builds are the exception to the persistent target dir: they always start
from an empty one in `$RUNNER_TEMP` so no stale artifact reaches a shipped binary, and they need no tools
preinstalled beyond the basics listed in docs/RELEASE.md, Runners. macOS release builds have no self-hosted runner
and use GitHub-hosted `macos-latest`; `FOLO_RELEASE_MACOS=false` skips them.

## Self-hosted runners
- **Security.** A self-hosted runner executes the code of a pull request (tests, build scripts, npm lifecycle
  scripts) with the runner account's access to that machine, and its persistent target dir carries build output from
  one run to the next. That is acceptable for a private repository whose contributors already work on that machine.
  It is not for a public one: **pull requests from forks must never reach a self-hosted runner.** For a pull
  request from a fork every `runs-on` here ignores the variables and uses the hosted runner, but that is a
  convenience, not a protection: a fork's pull request runs the fork's copy of the workflow files, and that copy can
  name any runner registered to the repository. So before a repository with these workflows goes public, **remove
  the self-hosted runners from it** (Settings, Actions, Runners) and delete `RUNS_ON_LINUX` and `RUNS_ON_WINDOWS`;
  deleting the variables alone is not enough. If a public repository must keep one, put it in a dedicated account or
  VM with no credentials and nothing worth keeping, and require approval for every outside contributor's workflow
  runs (Settings, Actions, General). Keep dependencies pinned through `Cargo.lock` and `package-lock.json`,
  and review new dependencies in pull requests.
- Status: `gh api repos/<owner>/<repo>/actions/runners -q '.runners[] | .name + " " + .status'`.
- A Windows runner needs Git for Windows' `bin` and `usr\bin` on its PATH (actions such as `dtolnay/rust-toolchain`
  run bash); set it in the runner's `.env` file (`PATH=...`), which the runner reads at start. Start it again after a
  reboot.
- An offline self-hosted runner does not trigger the fallback: its jobs queue (up to 24 h). If the machine is off for
  long, delete the variable so the jobs run on GitHub-hosted runners.
- **A Linux runner** needs, besides the usual shell tools of a Linux system (`bash`, `tar`, gzip, `sha256sum`), a C
  toolchain (`build-essential`), `git`, `curl`, `jq` and `python3`; `docker` and `gh` for `release.yml`
  (docs/RELEASE.md, Runners); and Chrome for the layout QA, or passwordless `sudo` so the job can install
  Playwright's Chromium with its libraries. The jobs install Rust, Node and the Python of the ONNX test themselves.
- **One Linux user per runner.** On a self-hosted runner the Rust jobs of `ci.yml`, `desktop.yml` and `wasm.yml`
  keep a persistent target dir under `$HOME` (`~/gha/folo-target`, `~/gha/folo-desktop-target`,
  `~/gha/folo-target-wasm`; release builds use a fresh one in `$RUNNER_TEMP`). Two runners of one user on one
  machine can run the same job in the same folder: each waits for the other's cargo lock, and the under-20-GB
  cleanup of one deletes the folder under the other.
- **Linux runners in WSL2**, on a Windows machine that already runs the Windows jobs: a distribution with systemd
  on (the runners are services, and docker needs it), without the Windows drives and programs (`automount` and
  `interop` off in `/etc/wsl.conf`, so a job sees only the Linux file system and the Windows `PATH` cannot shadow
  the Linux tools), and with a size limit on its disk (`wsl --manage <distro> --resize 300GB`): the default virtual
  disk reports about 1 TB free whatever the drive holds, so the cleanup above would never start.
  - **Memory.** Builds fill the VM's memory with page cache. Under that load the kernel of a 14 GB VM logged
    `page allocation failure: order:7` (512 KiB of contiguous memory, with 4 KiB pages) under `vmbus_alloc_ring`,
    the ring storage of a new Hyper-V socket, which is what a `wsl.exe` call opens. That fits fragmented free
    memory, and it is a likely reason for the `wsl.exe` calls that timed out (`Wsl/Service/WSAETIMEDOUT`) while the
    jobs inside still ran; it is not proven. The settings since then, in `/etc/sysctl.d/`:
    `vm.min_free_kbytes = 524288` (512 MiB; it was 45056, and it sets the minimum free-page watermarks) and
    `vm.watermark_scale_factor = 100` (it was 10; a wider gap between the watermarks, so background reclaim starts
    earlier). Neither guarantees a contiguous block. In the one run of every workflow from cold after the change
    there was no allocation failure, and all 70 `wsl.exe` probes answered. `compact_stall` in `/proc/vmstat` read
    318 after it, against 7,172 before; the two totals cover different jobs and durations, so they are an
    indication, not a controlled comparison.
  - **Keep-alive and hangs.** Systemd services alone do not keep a WSL distribution alive: it can shut down once it
    is idle, so keep a long-running `wsl.exe` session attached. A VM
    that hangs keeps its runners "online" for a while and its jobs running forever (seen once, a kernel soft lockup
    in the first minutes of the first run), so let that `wsl.exe` print a heartbeat and watch it from Windows;
    a new `wsl.exe` started as the probe may time out as described above. After a forced stop
    (`wsl --shutdown`), delete the runners' toolchains and target dirs: a file written just before can be cut short
    (a `librustc_driver` that then failed with "cannot read file data").

## Unstable self-hosted Windows host
The maintainers' self-hosted Windows machine (an i7-14700KF) occasionally crashes rustc or rustfmt under heavy
parallel load (`STATUS_ACCESS_VIOLATION` 0xc0000005, `STATUS_ILLEGAL_INSTRUCTION` 0xc000001d,
`STATUS_STACK_BUFFER_OVERRUN` 0xc0000409), a pattern consistent with the Intel 13th/14th-gen instability issue. The
Windows steps in `ci.yml`, `desktop.yml` and `release.yml` (its Windows `cargo build` steps) therefore run through
`tools/ci-retry.sh`, which on self-hosted runners retries a step once **only when cargo's own line shows one of those
crash codes** (`process didn't exit successfully: ... (exit code: 0xc0000005, ...`), and logs a warning. Compile
errors and failing (including flaky) tests are never retried, also when they come with a crash line or print one, and
on hosted runners nothing is.

The self-hosted Linux runners (the WSL VMs on the same machine) show the same thing in Node: `npm run check` died
three times in one evening on unrelated heads with V8's `# Fatal error in , line 0` / `unreachable code` banner and
`Trace/breakpoint trap` (exit 133), and a layout-QA shard once with `Illegal instruction` (exit 132), with Node
v22 from `actions/setup-node`. So `web.yml` runs `npm run check`, `npm test` and `npm run qa` through the same
script, which also counts as a crash a process killed by SIGILL, SIGTRAP, SIGBUS or SIGSEGV (exit 132 / 133 / 135 /
139, npm's whole `npm error signal SIG...` line, or V8's `# Fatal error in` banner at the start of a line). The
crash words inside a line (a test's message) do not count, and a step whose output shows an ordinary failure (vitest's
`Tests  N failed` summary, a `FAIL` line, `error TS...`, `test result: FAILED`, `error[E...]`, `due to N previous
errors`, the QA's `N problem(s):`) is never retried, crash or not, so a failing or flaky test cannot be retried into a
pass. `tools/ci-retry-test.sh` holds these cases and runs first in `web.yml`'s `check` job. Not determined: whether the Node crashes come from the CPU (as the rustc ones appear to) or from that Node
build; a crash that repeats on the retry, or one seen on hosted runners, points at Node and should be reported
there.
