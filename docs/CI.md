# CI

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`: `cargo fmt --check` (Linux), `cargo clippy -D
warnings` and `cargo test --workspace` on Linux and Windows, and the web app's `npm ci && npm run check && npm run build`.

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

## Operating the runners
- Status: `gh api repos/himent12/folo/actions/runners -q '.runners[] | .name + " " + .status'`.
- Windows (dev PC): start with `Start-Process D:\gha\win\run.cmd -WindowStyle Hidden`. After a reboot, start it again.
  Prerequisite: Git for Windows' `bin` and `usr\bin` on the runner's PATH (actions such as `dtolnay/rust-toolchain`
  run bash). It is set in `D:\gha\win\.env` (`PATH=...`), which the runner reads at start.
- Linux (boat.dev): the sandbox has a TTL (24 h) and is renewed or recreated by the orchestrator. The sandbox id is in
  `~/.folo/gha_linux_sid` on the dev PC. To recreate it: create a `large` sandbox named `folo-gha-linux`, download the
  runner tarball into `~/gha`, `./config.sh --url https://github.com/himent12/folo --token <registration token>
  --labels folo-linux --unattended --replace`, then `nohup ./run.sh &`. Get a registration token with
  `gh api -X POST repos/himent12/folo/actions/runners/registration-token -q .token`. Never print or commit tokens or
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
