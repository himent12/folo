# CI

`.github/workflows/ci.yml` runs on every PR and on pushes to `main`: `cargo fmt --check` (Linux), `cargo clippy -D
warnings` and `cargo test --workspace` on Linux and Windows, and the web app's `npm ci && npm run check && npm run build`.

## Runners
The repository is private, so GitHub-hosted minutes are limited (Windows counts double). FOLO therefore runs CI on its
own runners, selected by two repository variables holding a JSON `runs-on` value:

| variable | value | runner |
|---|---|---|
| `RUNS_ON_LINUX` | `["self-hosted","folo-linux"]` | `folo-linux-boat`: a boat.dev `large` sandbox (8 vCPU, 16 GB, Ubuntu 24.04), runner in `~/gha` |
| `RUNS_ON_WINDOWS` | `["self-hosted","folo-win"]` | `folo-win-local`: the dev PC, runner in `D:\gha\win` |

To fall back to GitHub-hosted runners, delete the variable (`gh variable delete RUNS_ON_LINUX`); the workflow then
uses `ubuntu-latest` / `windows-latest`. Self-hosted jobs keep a persistent `CARGO_TARGET_DIR` outside the checkout,
so incremental builds stay warm.

## Operating the runners
- Status: `gh api repos/himent12/folo/actions/runners -q '.runners[] | .name + " " + .status'`.
- Windows (dev PC): start with `Start-Process D:\gha\win\run.cmd -WindowStyle Hidden`. After a reboot, start it again.
- Linux (boat.dev): the sandbox has a TTL (24 h) and is renewed or recreated by the orchestrator. The sandbox id is in
  `~/.folo/gha_linux_sid` on the dev PC. To recreate it: create a `large` sandbox named `folo-gha-linux`, download the
  runner tarball into `~/gha`, `./config.sh --url https://github.com/himent12/folo --token <registration token>
  --labels folo-linux --unattended --replace`, then `nohup ./run.sh &`. Get a registration token with
  `gh api -X POST repos/himent12/folo/actions/runners/registration-token -q .token`. Never print or commit tokens or
  the boat key (docs/BOAT.md).
- Self-hosted runners execute code from PRs. This is acceptable only because the repository is private and every PR
  comes from FOLO's own agents. Never enable these runners on a public fork.
