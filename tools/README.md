# tools: CI and development helpers

Small scripts that CI runs and that contributors run by hand. [docs/CI.md](../docs/CI.md) lists what each workflow
runs; [CONTRIBUTING.md](../CONTRIBUTING.md) says which checks a pull request must pass. FOLO Desktop has its own
helpers in `apps/desktop/tools/`.

| file | what it does | who runs it |
|---|---|---|
| [`file-size.sh`](file-size.sh) | fails when a tracked source file (`.rs`, `.ts`, `.py`, `.wgsl`, `.sh`, ...; not `web/framework/vendor`) has 1000 lines or more, and lists those files | `site.yml`, step "file sizes", on every pull request; AGENTS.md and CONTRIBUTING.md ask for it before a PR |
| [`test-tmp.sh`](test-tmp.sh) | runs a test command with `FOLO_TEST_TMP`, `TMPDIR`, `TMP` and `TEMP` pointed at one fresh folder, then fails if the run left a `folo-*` entry there | `ci.yml`, job `rust`, step "test"; `desktop.yml`, the "test" steps of both jobs ([docs/CI.md, Test temp dirs](../docs/CI.md#test-temp-dirs)) |
| [`ci-retry.sh`](ci-retry.sh) | runs a command up to `ATTEMPTS` times (default 1), retrying only when the output shows a Windows crash code (access violation, illegal instruction, stack-buffer overrun) | `ci.yml` (clippy, test), `desktop.yml` (job `desktop`), `release.yml` (the "build" steps of `cli` and `desktop`); `ATTEMPTS` is 2 only on self-hosted runners ([docs/CI.md, Unstable self-hosted Windows host](../docs/CI.md#unstable-self-hosted-windows-host)) |
| [`charts/`](charts/README.md) | draws the SVG charts of the README and docs from `charts/data.json` | `site.yml`, step "charts" (`--check`); by hand after changing a number |
| [`bench-ab.ps1`](bench-ab.ps1) | Windows: runs benchmark binaries alternately, round by round, at a process priority and CPU affinity, and reports the CPU time the rest of the machine used during each run and the run's peak working set | by hand, for the CPU sections of the hill-climbs in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) ([docs/notes/hill/cpu.md](../docs/notes/hill/cpu.md) has the protocol) |
| [`boat.py`](boat.py) | a client for boat.dev CPU sandboxes, as a Python module and a small CLI (`ls`, `run`, `stop`, `rm`) | by hand, by the maintainers ([docs/notes/BOAT.md](../docs/notes/BOAT.md)) |

## Run them by hand

From the repository root, in bash (Git Bash on Windows):

```sh
bash tools/file-size.sh                      # the check CI runs
bash tools/file-size.sh 800                  # the same with another limit, to see what is getting close
bash tools/test-tmp.sh cargo test --workspace
TEST_TMP=/path/to/scratch bash tools/test-tmp.sh cargo test -p folo-cli   # the fresh folder under TEST_TMP
ATTEMPTS=2 bash tools/ci-retry.sh cargo test --workspace
node tools/charts/render.mjs                 # see charts/README.md
```

`test-tmp.sh` creates its folder under `$TEST_TMP`, else `$RUNNER_TEMP`, else `$TMPDIR`, else `/tmp`, and removes it
when nothing was left behind.

`bench-ab.ps1` takes `-Runs "label=path;label=path"`, `-Rounds` (3), `-Threads` (`RAYON_NUM_THREADS`, 0 = unset),
`-Affinity` (a hex mask), `-Priority` (`High`), `-Env "NAME=value;..."` and `-ExeArgs` (`--bench`); the header of the
script has an example. `boat.py` needs `requests` and reads its key from `~/.folo/boat.key` (outside the repository)
when it is imported.

[Back to the repository README](../README.md)
