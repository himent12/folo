# scripts: install and build scripts

| file | what it does | who runs it |
|---|---|---|
| [`install.sh`](install.sh) | installs the `folo` CLI from a GitHub release on Linux x86_64 or macOS arm64 (when the release has a macOS archive, docs/RELEASE.md, Runners) | users ([README.md, Quickstart](../README.md#quickstart)); `release.yml` ships it as a release asset and its `install-test` job runs it |
| [`install.ps1`](install.ps1) | the same on Windows x64 (Windows PowerShell 5.1 or PowerShell 7), optionally with FOLO Desktop | the same |
| [`build-wasm.sh`](build-wasm.sh) | builds `folo-wasm` into `web/public/folo-wasm/` | `wasm.yml`; `release.yml`, job `web` (with `--locked`); developers before building Studio |
| [`build-wasm.ps1`](build-wasm.ps1) | the same for PowerShell on Windows | developers |

Releases and what the workflow builds: [docs/RELEASE.md](../docs/RELEASE.md). The browser module:
[docs/WASM.md](../docs/WASM.md).

## install.sh and install.ps1

Both pick the archive for the platform (`folo-<target>.tar.gz`, or `folo-x86_64-pc-windows-msvc.zip`), download it
and its `.sha256` file, refuse it when the SHA-256 does not match, and install the binary. The binary embeds FOLO
Studio, so `folo serve` works at once. A release ships both scripts as assets, so they can run without a checkout:

```sh
curl -fsSL https://github.com/himent12/folo/releases/latest/download/install.sh | sh
sh scripts/install.sh --version v0.2.0 --dir ~/bin
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Version v0.2.0 -Desktop
```

| `install.sh` | `install.ps1` | environment | default |
|---|---|---|---|
| `--version TAG` | `-Version` | `FOLO_VERSION` | `latest` |
| `--dir DIR` | `-InstallDir` | `FOLO_INSTALL_DIR` | `~/.local/bin`; `%LOCALAPPDATA%\Programs\folo\bin` |
| `--repo OWNER/NAME` | `-Repo` | `FOLO_REPO` | `himent12/folo` |
| `--base-url URL` | `-BaseUrl` | `FOLO_BASE_URL` | GitHub; else `<URL>/<asset>` from a mirror, a `file://` URL or a local folder |
| `--target TRIPLE` | | `FOLO_TARGET` | detected from `uname` |
| | `-Desktop` | | also install `folo-desktop.exe` |
| | `-NoModifyPath` | | leave the user `PATH` alone (by default the install folder is added to it) |

- `install.sh` stops on any other platform (and on Windows shells, pointing to `install.ps1`) and needs `curl`, `tar`
  and `sha256sum` or `shasum`. It copies the binary next to the destination and renames it, so an update never
  leaves a half-written file, runs `folo --version`, and says when the folder is not on `PATH`.
- `install.ps1` installs the x64 build on ARM64 Windows too (which runs it emulated). It moves a running `folo.exe`
  aside before replacing it, and adds the folder to the user `PATH` unless `-NoModifyPath` is given.
- A token from `GH_TOKEN` or `GITHUB_TOKEN`, else `gh auth token` when `gh` is logged in, makes both read the release
  through the GitHub API, for private forks and mirrors.

## build-wasm.sh and build-wasm.ps1

Both build `folo-wasm` for `wasm32-unknown-unknown` with `-C target-feature=+simd128` added to `RUSTFLAGS`, run
`wasm-bindgen --target web`, and run `wasm-opt -O3` on release builds when it is on `PATH`. The output is
`folo_wasm.js` (an ES module), `folo_wasm_bg.wasm` and their TypeScript declarations in `web/public/folo-wasm/`
(ignored by git), which Studio's `vite build` copies into `web/dist/`.

```sh
sh scripts/build-wasm.sh                 # release build into web/public/folo-wasm
sh scripts/build-wasm.sh --out DIR       # somewhere else      (PowerShell: -Out DIR)
sh scripts/build-wasm.sh --dev           # unoptimised, faster (PowerShell: -Dev)
sh scripts/build-wasm.sh --locked        # fail instead of updating Cargo.lock (PowerShell: -Locked)
```

They need the `wasm32-unknown-unknown` target (added with `rustup` when it is missing) and `wasm-bindgen-cli` at
exactly the `wasm-bindgen` version in `Cargo.lock`, on `PATH` or in `$FOLO_TOOLS/bin`; otherwise they stop and print
the `cargo install wasm-bindgen-cli --version <version> --locked` command. `CARGO_TARGET_DIR` is honoured.

[Back to the repository README](../README.md)
