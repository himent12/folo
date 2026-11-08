# Releasing FOLO

Releases are built by `.github/workflows/release.yml`. Nothing is published automatically: a tag produces a
**draft** GitHub Release that a person reviews and publishes.

## Version
`[workspace.package] version` in the root `Cargo.toml` is the single source of truth. Every workspace crate
inherits it (`version.workspace = true`), `folo --version` prints it, and `crates/folo-cli/tests/version.rs` fails
when `web/package.json` or `apps/desktop/Cargo.toml` (its own cargo workspace, so it cannot inherit) disagree. The
release workflow refuses a tag that is not `v<version>`; a version with a pre-release suffix (`0.2.0-rc.1`) makes a
pre-release.

## Cutting a release
1. Bump the version in `Cargo.toml`, `web/package.json` (then `npm install` in `web/` to refresh the lock file) and
   `apps/desktop/Cargo.toml`; `cargo test --workspace` checks they agree. Land it through a PR as usual.
2. Optional dry run: run the `release` workflow by hand (Actions → release → Run workflow). It builds and smoke-tests
   everything and uploads the assets as run artifacts, without creating a release. Untick `macos` to save macOS
   minutes (billed at 10x on private repositories).
3. Tag `main` and push the tag: `git tag v0.2.0 && git push origin v0.2.0`.
4. The workflow creates the draft release with generated notes. Check the assets and notes, then publish it. The
   install scripts (`scripts/install.sh`, `scripts/install.ps1`) only see published releases.

Re-running the workflow for a tag whose draft already exists replaces that draft's assets.

## What the workflow builds
| job | runner | output |
|---|---|---|
| `web` | ubuntu-latest | `web/dist` (`npm ci && npm run build`), shared with every CLI build |
| `cli` | windows-latest, ubuntu-22.04, macos-latest | `folo-<target>.zip` (Windows) / `.tar.gz`: a `folo-<target>/` folder with the binary (built with `--features embed-web`), `README.md` and the licenses |
| `desktop` | windows-latest | `folo-desktop-x86_64-pc-windows-msvc.zip` with `folo-desktop.exe` (skipped while `apps/desktop/Cargo.toml` does not exist) |
| `release` | ubuntu-latest | checks every `.sha256`, writes `SHA256SUMS`, uploads everything as the `release-<version>` artifact, creates the draft release (tags only) |
| `install-test` | ubuntu-22.04, windows-latest | runs `scripts/install.sh` / `scripts/install.ps1` against those assets (`--base-url`), checks the installed `folo`, and checks that a tampered archive is refused |

Pull requests that change the workflow, the install scripts or the build scripts behind `--version` and
`embed-web` run the whole thing as a dry run (no macOS, no release), so release breakage shows up before a tag.

Targets: `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu` (built on Ubuntu 22.04 so it runs on glibc 2.35 and
newer) and `aarch64-apple-darwin` (macOS 11+). The macOS build is optional: it can fail without blocking the
release, and tag builds skip it when the repository variable `FOLO_RELEASE_MACOS` is `false`. The desktop job is
separate for the same reason; the release goes out with whatever desktop build succeeded.

Each CLI build is smoke-tested before packaging: `folo --version`, `folo doctor --json` must report the embedded
Studio, and `folo serve --mock --port 0` must serve `/api/health`, `/` and an SPA route.

Asset names carry no version, so `https://github.com/himent12/folo/releases/latest/download/<asset>` always points
at the newest published release. Every archive has a `<asset>.sha256` (`<hex>  <asset>`, the `sha256sum` format);
`SHA256SUMS` lists them all.

## Building a release binary locally
```sh
(cd web && npm ci && npm run build)
cargo build --release --locked -p folo-cli --features embed-web
```
`FOLO_WEB_DIST=/path/to/dist` embeds another build of the Studio; `FOLO_GIT_SHA` and `SOURCE_DATE_EPOCH` override
the commit and build date that `folo --version` reports.
