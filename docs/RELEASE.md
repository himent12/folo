# Releasing FOLO

Releases are built by `.github/workflows/release.yml`. Nothing is published automatically: a tag produces a
**draft** GitHub Release that a person reviews and publishes.

## Version
`[workspace.package] version` in the root `Cargo.toml` is the single source of truth. Every workspace crate
inherits it (`version.workspace = true`), `folo --version` and `folo-desktop --version` print it, and
`crates/folo-cli/tests/version.rs` fails when `web/package.json` or `apps/desktop/Cargo.toml` (its own cargo
workspace, so it cannot inherit) disagree. The release workflow refuses a tag that is not `v<version>`; a version
with a pre-release suffix (`0.2.0-rc.1`) makes a pre-release.

## Cutting a release
1. **Bump the version** on a branch (`release/v0.2.0`):
   - `Cargo.toml`: `[workspace.package] version = "0.2.0"`
   - `web/package.json`: `"version": "0.2.0"`, then `npm install` in `web/` to refresh `package-lock.json`
   - `apps/desktop/Cargo.toml`: `[package] version = "0.2.0"`
   - refresh both lock files, which the release builds use with `--locked`: `cargo check --workspace` at the root
     and `cargo check` in `apps/desktop/`
   - `cargo test -p folo-cli --test version` checks they agree. Open the PR as usual and land it.
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
3. **Tag `main`** and push the tag:
   ```sh
   git switch main && git pull --ff-only
   git tag -a v0.2.0 -m "FOLO 0.2.0" && git push origin v0.2.0
   ```
   The workflow runs again for the tag and creates the draft release `FOLO 0.2.0` with generated notes and every
   asset.
4. **Review the draft** (`gh release view v0.2.0`, or Releases on GitHub):
   - the assets: three CLI archives (macOS may be missing if its optional build failed; the run's summary says so),
     `folo-desktop-x86_64-pc-windows-msvc.zip`, a `.sha256` per archive and `SHA256SUMS`;
   - the notes: edit them into a short summary of what changed (`gh release edit v0.2.0 --notes-file notes.md`);
   - optionally download an archive and run the [checks below](#checking-a-build).
5. **Publish**: `gh release edit v0.2.0 --draft=false` (or "Publish release" on GitHub). Only now do
   `releases/latest/download/...` and the install scripts (`scripts/install.sh`, `scripts/install.ps1`) see it.
   Check with a fresh install: `sh scripts/install.sh --dir /tmp/folo-check && /tmp/folo-check/folo --version`.

Re-running the workflow for a tag whose draft already exists replaces that draft's assets; it refuses to touch a
release that is already published. To abandon a draft, delete it and the tag
(`gh release delete v0.2.0 --cleanup-tag`) and fix forward with a new version: a published version is never
rebuilt.

## Dry runs
| how | what runs | creates a release |
|---|---|---|
| a PR touching the workflow, the install scripts, `scripts/build-wasm.sh` or the build scripts behind `--version` and `embed-web` | everything except macOS | no |
| `gh workflow run release.yml --ref <branch or tag>` (Actions → release → Run workflow) | everything; `macos` (default on) adds the macOS build | no |
| the same with `-f tag=vX.Y.Z` | also the tag check and the notes preview | no |
| pushing a tag `v*` | everything | a draft |

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
| `plan` | ubuntu-latest | the version, the tag check (tag builds, and dispatches with `tag`) and the build matrix |
| `web` | ubuntu-latest | `web/dist`: folo-wasm first (`scripts/build-wasm.sh` with SIMD128, wasm-bindgen-cli at the `Cargo.lock` version, binaryen's `wasm-opt` pinned by version and checksum), then `npm ci && npm run build`; fails unless `dist/folo-wasm/` has the module. Shared with every CLI build |
| `cli` | windows-latest, ubuntu-22.04, macos-latest | `folo-<target>.zip` (Windows) / `.tar.gz`: a `folo-<target>/` folder with the binary (built with `--features embed-web`), `README.md` and the licenses |
| `desktop` | windows-latest | `folo-desktop-x86_64-pc-windows-msvc.zip` with `folo-desktop.exe`, built with `--profile dist` (thin LTO, one codegen unit; apps/desktop's plain `release` profile is the faster-building development one) (skipped while `apps/desktop/Cargo.toml` does not exist) |
| `release` | ubuntu-latest | checks every `.sha256`, writes `SHA256SUMS`, uploads everything as the `release-<version>` artifact, creates the draft release (tags only) or previews it (dispatches with `tag`) |
| `install-test` | ubuntu-22.04, windows-latest | runs `scripts/install.sh` / `scripts/install.ps1` against those assets (`--base-url`), checks the installed `folo` (and `folo-desktop`), and checks that a tampered archive is refused |

Targets: `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu` (built on Ubuntu 22.04 so it runs on glibc 2.35 and
newer) and `aarch64-apple-darwin` (macOS 11+). The macOS build is optional: it can fail without blocking the
release, and tag builds skip it when the repository variable `FOLO_RELEASE_MACOS` is `false`. The desktop job is
separate for the same reason; the release goes out with whatever desktop build succeeded.

Each CLI build is smoke-tested before packaging: `folo --version`; `folo doctor --json` must report the embedded
Studio (`web_source: embedded`) with folo-wasm in it (`web_folo_wasm: true`); `folo serve --mock --port 0` must
serve `/api/health`, `/`, an SPA route, and `/folo-wasm/folo_wasm_bg.wasm` as `application/wasm` (a real module,
not the SPA fallback). The desktop build must answer `--version` with the release version and `--help`.

Asset names carry no version, so `https://github.com/himent12/folo/releases/latest/download/<asset>` always points
at the newest published release. Every archive has a `<asset>.sha256` (`<hex>  <asset>`, the `sha256sum` format);
`SHA256SUMS` lists them all.

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
