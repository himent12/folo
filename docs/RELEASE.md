# Releasing FOLO

Releases are built by `.github/workflows/release.yml`. Nothing is published automatically: a tag produces a
**draft** GitHub Release that a person reviews and publishes.

## Version
`[workspace.package] version` in the root `Cargo.toml` is the single source of truth. Every workspace crate
inherits it (`version.workspace = true`), `folo --version` and `folo-desktop --version` print it, and
`crates/folo-cli/tests/version.rs` fails when `web/package.json`, `site/package.json`, their lock files or
`apps/desktop/Cargo.toml` (its own cargo workspace, so it cannot inherit) disagree. The release workflow refuses a tag that is not `v<version>`; a version
with a pre-release suffix (`0.2.0-rc.1`) makes a pre-release.

## Cutting a release
1. **Bump the version** on a branch (`release/v0.2.0`):
   - `Cargo.toml`: `[workspace.package] version = "0.2.0"`
   - `web/package.json` and `site/package.json`: `"version": "0.2.0"`, then `npm install` in `web/` and in `site/`
     to refresh their `package-lock.json`
   - `apps/desktop/Cargo.toml`: `[package] version = "0.2.0"`
   - refresh both lock files, which the release builds use with `--locked`: `cargo check --workspace` at the root
     and `cargo check` in `apps/desktop/`
   - `cargo test -p folo-cli --test version` checks they agree.
   - add the version's section to `CHANGELOG.md` (grouped as Engine, Training, Inference and export, Server/API/MCP,
     Studio, Desktop, Docs, Known issues; from `gh pr list --state merged`). Open the PR as usual and land it.
2. **Rehearse the tag** on the merged `main`, without creating anything:
   ```sh
   gh workflow run release.yml --ref main -f tag=v0.2.0          # add -f macos=false to skip the macOS build
   gh run watch "$(gh run list --workflow release.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')"
   ```
   This is the tag build minus the release: the tag check against the version, every build and smoke test, the
   checksums, the install-script test, and (in the `release` job's summary) a preview of the generated release
   notes and what would happen to the draft. The assets are uploaded as the run's `release-<version>` artifact.
   Download it (`gh run download <run id> -n release-0.2.0 -D dist`) to try the binaries before tagging (see
   [Checking a build](#checking-a-build)).
3. **Create the draft**, either way:
   - **Tag `main`** and push the tag:
     ```sh
     git switch main && git pull --ff-only
     git tag -a v0.2.0 -m "FOLO 0.2.0" && git push origin v0.2.0
     ```
     The workflow runs again for the tag and creates the draft release `FOLO 0.2.0` with generated notes and every
     asset.
   - **Or dispatch with `draft`**, which creates the same draft without creating the tag:
     ```sh
     gh workflow run release.yml --ref main -f tag=v0.2.0 -f draft=true
     ```
     The draft targets the dispatched commit, and GitHub creates the tag `v0.2.0` there when the draft is published.
     This also works from the release branch, to review a draft before the PR lands; then dispatch again on the
     merged `main` before publishing, so the draft (and the tag publishing creates) points at `main`'s commit. A
     re-run for an existing draft moves its target to the new commit and replaces its assets. It refuses when the
     tag already exists at another commit.
4. **Review the draft** (`gh release view v0.2.0`, or Releases on GitHub):
   - the assets: three CLI archives (macOS may be missing if its optional build failed or was skipped, see
     [Runners](#runners); the run's summary says so), `folo-desktop-x86_64-pc-windows-msvc.zip`,
     `folo-wasm.tar.gz`, `install.sh` and `install.ps1`, a `.sha256` per asset and `SHA256SUMS`;
   - the notes: edit them into the release notes (highlights, benchmarks, known issues; from the CHANGELOG section),
     `gh release edit v0.2.0 --notes-file notes.md`. A re-run of the workflow for the draft keeps them;
   - optionally download an archive and run the [checks below](#checking-a-build).
5. **Publish**: `gh release edit v0.2.0 --draft=false --latest` (or "Publish release" on GitHub, with "Set as the
   latest release" ticked; [Weights releases](#weights-releases) says why the flag matters). Only now do
   `releases/latest/download/...` and the install scripts (`scripts/install.sh`, `scripts/install.ps1`) see it.
   Check with a fresh install: `sh scripts/install.sh --dir /tmp/folo-check && /tmp/folo-check/folo --version`.
   Users install with `curl -fsSL https://github.com/himent12/folo/releases/latest/download/install.sh | sh`
   (Linux) or `irm https://github.com/himent12/folo/releases/latest/download/install.ps1 | iex` (Windows
   PowerShell); no account is needed. The scripts' token path (`GH_TOKEN`, `GITHUB_TOKEN`, else a logged-in `gh`)
   is for private forks and mirrors.

Re-running the workflow for a tag whose draft already exists replaces that draft's assets (and removes any asset
the run did not build, so a draft never mixes commits); it refuses to touch a release that is already published. To abandon a draft, delete it and the tag
(`gh release delete v0.2.0 --cleanup-tag`) and fix forward with a new version: a published version is never
rebuilt.

## Dry runs
| how | what runs | creates a release |
|---|---|---|
| a PR touching the workflow, the install scripts, `scripts/build-wasm.sh` or the build scripts behind `--version` and `embed-web` | everything except macOS | no |
| `gh workflow run release.yml --ref <branch or tag>` (Actions → release → Run workflow) | everything; `macos` (default on) adds the macOS build unless `FOLO_RELEASE_MACOS` is `false` | no |
| the same with `-f tag=vX.Y.Z` | also the tag check and the notes preview | no |
| the same with `-f tag=vX.Y.Z -f draft=true` | everything | a draft (the tag is created on publish) |
| pushing a tag `v*` | everything (macOS unless `FOLO_RELEASE_MACOS` is `false`) | a draft |

A dispatch runs the workflow file of the ref it is given, so a branch can rehearse changes to the workflow itself.
To rehearse a commit that is not on a branch, put a throwaway tag on it that does not start with `v` (pushing a `v*`
tag starts a real tag build), dispatch on it, and delete it afterwards:
```sh
git tag dryrun-0.2.0 <commit> && git push origin dryrun-0.2.0
gh workflow run release.yml --ref dryrun-0.2.0 -f tag=v0.2.0
git push origin :refs/tags/dryrun-0.2.0 && git tag -d dryrun-0.2.0     # once the run is done
```

## What the workflow builds
| job | runner | output |
|---|---|---|
| `plan` | Linux | the version, the tag check (tag builds, and dispatches with `tag`) and the build matrix |
| `web` | Linux | `web/dist`: folo-wasm first (`scripts/build-wasm.sh` with SIMD128, wasm-bindgen-cli at the `Cargo.lock` version, binaryen's `wasm-opt` pinned by version and checksum), then `npm ci && npm run build`; fails unless `dist/folo-wasm/` has the module. Shared with every CLI build. Also `folo-wasm.tar.gz`: a `folo-wasm/` folder with the ES module, the `.wasm`, the TypeScript declarations, docs/WASM.md as `README.md` and the licenses |
| `cli` | Windows, Linux (built in an `ubuntu:22.04` container), macOS (hosted `macos-latest`) | `folo-<target>.zip` (Windows) / `.tar.gz`: a `folo-<target>/` folder with the binary (built with `--features embed-web`), `README.md` and the licenses |
| `desktop` | Windows | `folo-desktop-x86_64-pc-windows-msvc.zip` with `folo-desktop.exe`, built with `--profile dist` (thin LTO, one codegen unit; apps/desktop's plain `release` profile is the faster-building development one) (skipped while `apps/desktop/Cargo.toml` does not exist) |
| `release` | Linux | adds `install.sh` and `install.ps1` from `scripts/`, checks every `.sha256`, writes `SHA256SUMS`, uploads everything as the `release-<version>` artifact, creates or refreshes the draft release (tags, and dispatches with `tag` and `draft`) or previews it (dispatches with `tag` only) |
| `install-test` | Linux, Windows | runs the `install.sh` / `install.ps1` assets against those assets (`--base-url`), checks the installed `folo` (and `folo-desktop`), checks that a tampered archive is refused, and runs the installed Linux `folo` in an `ubuntu:22.04` container, where it must start without loader warnings |

Targets: `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu` (runs on glibc 2.35 and newer, see below) and
`aarch64-apple-darwin` (macOS 11+).
The macOS build is optional: it can fail without blocking the release, and tag builds and dispatches skip it when
the repository variable `FOLO_RELEASE_MACOS` is `false`. The desktop job is separate for the same reason; the
release goes out with whatever desktop build succeeded.

## Runners
The Linux and Windows jobs use the runners of `ci.yml` (docs/CI.md): the repository variables `RUNS_ON_LINUX` and
`RUNS_ON_WINDOWS` (JSON `runs-on` values) select self-hosted runners.
When a variable is unset the jobs run on GitHub-hosted runners: `ubuntu-22.04` for the Linux CLI build and
the Linux install test, `ubuntu-latest` for the other Linux jobs, `windows-latest` for Windows. The `plan` job
writes the chosen `runs-on` into each CLI matrix entry, and its summary lists them.

- **macOS has no self-hosted runner**: the macOS build always runs on GitHub-hosted `macos-latest`. Set
  `FOLO_RELEASE_MACOS` to `false` (`gh variable set FOLO_RELEASE_MACOS --body false`) whenever hosted runners are
  unavailable, so tag builds and dispatches skip it instead of queueing a job GitHub refuses to start (for example
  when the account's billing blocks hosted runners). A release built that way ships without the macOS archive, so
  `scripts/install.sh` fails on macOS for it and macOS users build from source.
- **Clean builds.** Every release build starts from an empty `CARGO_TARGET_DIR` in `$RUNNER_TEMP` (the runner
  deletes it after the job), on hosted and self-hosted runners alike, and there is no `rust-cache`. Release builds
  are rare and their binaries ship, so they never reuse CI's persistent target dirs: nothing stale (an older
  embedded Studio, another commit's `FOLO_GIT_SHA`, a different profile's build-script output) can reach an asset.
  The cost is a full build per run (about as long as a hosted runner took).
- **Tools.** The jobs install what they need: the Rust toolchain and targets (`dtolnay/rust-toolchain`),
  wasm-bindgen-cli at the `Cargo.lock` version (`taiki-e/install-action`, checksummed), binaryen's `wasm-opt`
  (pinned and checksummed) and Node 22 (`actions/setup-node`). They rely on the runner for `bash`, `curl`, `tar`,
  `sha256sum`, `jq` (Linux), `objdump` (Linux), `docker` (Linux: the Ubuntu 22.04 build and check, with network
  access for apt and crates.io) and `gh` (Linux, the `release` job); the hosted images have them. Windows zips are written with Windows' own
  `tar.exe` (bsdtar), so 7-Zip is not needed.
- **glibc 2.35.** The Linux CLI is linked inside an `ubuntu:22.04` container (pinned by digest; `gcc` and
  `libc6-dev` from apt, the job's Rust toolchain mounted read-only, cargo running as the runner's user), so it gets
  Ubuntu 22.04's glibc whatever the runner runs. Linked on Ubuntu 24.04, std's weak `pidfd_spawnp` /
  `pidfd_getpid` imports carry a `GLIBC_2.39` version, and the loader then prints ``weak version `GLIBC_2.39' not
  found`` on every start on older systems. Two checks hold the line: the `cli` job fails when `objdump -T` shows any
  symbol version above 2.35 (weak ones included), and `install-test` runs the installed binary in `ubuntu:22.04` and
  fails if `folo --version` writes anything to stderr.
- **Windows retries.** The Windows `cargo build` steps run through `tools/ci-retry.sh`, which retries once on
  the maintainers' self-hosted Windows runner when rustc crashes (docs/CI.md, Unstable self-hosted Windows host).
- **Isolation.** The CLI smoke test and the install test use a throwaway `FOLO_HOME` in `$RUNNER_TEMP`, so they
  never read the runner account's own `~/.folo` (on a self-hosted runner, a person's own). The `release` job runs
  with a `contents: write` token on the Linux runner; if that is ever a self-hosted one, docs/CI.md, Self-hosted runners, applies.

Each CLI build is smoke-tested before packaging: `folo --version`; `folo doctor --json` must report the embedded
Studio (`web_source: embedded`) with folo-wasm in it (`web_folo_wasm: true`); `folo serve --mock --port 0` must
serve `/api/health`, `/`, an SPA route, and `/folo-wasm/folo_wasm_bg.wasm` as `application/wasm` (a real module,
not the SPA fallback). The desktop build must answer `--version` with the release version and `--help`.

Asset names carry no version, so `https://github.com/himent12/folo/releases/latest/download/<asset>` always points
at the newest published release. Every archive has a `<asset>.sha256` (`<hex>  <asset>`, the `sha256sum` format);
`SHA256SUMS` lists them all.

## Weights releases
Pretrained weights are releases of their own, tagged `weights-v<n>` and made by hand, not by the workflow (the recipes
are in examples/coco/README.md, Publish): the `.folo` files and a `SHA256SUMS` (`sha256sum` output) listing them, plus
an entry per file in `folo_api::weights::PUBLISHED`, which is what `folo weights pull <name>` and `folo weights list`
read. A weights release must be created with `--latest=false` (`gh release create weights-vN --latest=false ...`,
or `gh release edit weights-vN --latest=false` after the fact) so that `releases/latest`, which the install scripts
use, keeps pointing at the newest FOLO version: otherwise GitHub marks the newest published release *Latest*, and
`releases/latest/download/...`, the install scripts and `gh release download --pattern install.sh` look for FOLO's
binaries in the weights release. A published weights release downloads over plain HTTPS
(`https://github.com/<repo>/releases/download/<tag>/<file>`), which is how `folo weights pull` gets it, with no
account; `gh` is only its fallback for a private fork or mirror that answers `404` (docs/CLI.md, Weights). Do not
replace a published file: caches keep the copy they have (`pull` skips a cached name without `--force`), so a new
result is a new name.

| tag | names | published |
|---|---|---|
| `weights-v1` | `folo-n-coco`, `folo-n-coco-f16` | 2026-11-11 (created as a draft) |
| `weights-v2` | `folo-n-coco-v2`, `folo-n-coco-v2-f16` | 2026-11-11 (created as a draft) |
| `weights-v3` | `folo-s-coco`, `folo-s-coco-f16` | 2026-11-11 |
| `weights-v4` | `folo-n-coco-v3`, `folo-n-coco-v3-f16` | 2026-11-14 |
| `weights-v5` | `folo-n-coco-v4`, `folo-n-coco-v4-f16` | 2026-11-16 |
| `weights-v6` | `folo-n-coco-v5`, `folo-n-coco-v5-f16` | 2026-11-18 |

All six carry `--latest=false`.

**If "latest" is a weights release.** When no published release is marked latest, GitHub picks one, which can be
a weights release. Then `releases/latest/download/...` and `install.sh` without `--version` do not find the
binaries. The fix is one flag on the newest FOLO release: `gh release edit v0.2.0 --latest`.

## Building a release binary locally
```sh
sh scripts/build-wasm.sh --locked        # web/public/folo-wasm (or: powershell -File scripts\build-wasm.ps1 -Locked)
(cd web && npm ci && npm run build)      # web/dist, with folo-wasm/ copied in
cargo build --release --locked -p folo-cli --features embed-web
```
Without the first step the binary still builds (cargo prints a warning) but its Playground can only run models on
the server. `FOLO_WEB_DIST=/path/to/dist` embeds another build of the Studio; `FOLO_GIT_SHA` and
`SOURCE_DATE_EPOCH` override the commit and build date that `folo --version` reports.

## Checking a build
On a release binary (from a dry run's artifact, a draft's assets or a local build):
```sh
folo --version && folo doctor              # "browser   folo-wasm in the embedded Studio; ..."
folo-desktop --version
```
The in-browser Playground end to end: `web/scripts/release-smoke.mjs` starts the binary's `folo serve` on a
throwaway data dir with a synthetic dataset and a model, opens the embedded Studio's Playground in headless Chrome
(`playwright-core` from `web/node_modules`, so `npm ci` in `web/` first) and detects a sample image on the server,
then in the browser on WebGPU and on the CPU:
```sh
node web/scripts/release-smoke.mjs --folo ./folo-x86_64-pc-windows-msvc/folo.exe --weights path/to/model.folo
```
Any `.folo` weights or checkpoint works (a run's `weights/best.folo`). It exits non-zero on a page error, a failed
request or a backend that does not produce detections.
