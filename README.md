# FOLO

**FOLO Only Looks Once.** A from-scratch, very fast vision framework in Rust: tensors, autograd, CPU and GPU (wgpu)
kernels, a YOLO-style real-time object detector, training, evaluation and export, and **FOLO Studio**, a web app
for datasets, training runs and live inference.

> Status: early. See [docs/ROADMAP.md](docs/ROADMAP.md).

```sh
cargo folo --help                 # the CLI (alias for: cargo run --release -p folo-cli --)
cargo folo dataset synth shapes   # make a synthetic dataset
cargo folo train --data shapes --model folo-n --epochs 50
cargo folo predict --model runs/shapes-n/best.folo image.jpg
cargo folo serve                  # FOLO Studio on http://127.0.0.1:7878
```

## Install

Prebuilt `folo` binaries (x86_64 Windows and Linux, Apple-silicon macOS) ship with FOLO Studio built in, so
`folo serve` works out of the box. The install scripts download the latest release for your platform, check its
SHA-256 and put `folo` in a user bin directory:

```sh
sh scripts/install.sh                     # Linux / macOS: into ~/.local/bin
sh scripts/install.sh --version v0.2.0 --dir ~/bin
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1            # into %LOCALAPPDATA%\Programs\folo\bin, added to PATH
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Desktop   # also FOLO Desktop (folo-desktop.exe)
```

While the repository is private, the scripts need a token that can read it: `GH_TOKEN` / `GITHUB_TOKEN`, or a
logged-in [`gh`](https://cli.github.com) (they use `gh auth token`). Options: `--version` / `-Version` (a tag,
default `latest`), `--dir` / `-InstallDir`, `--repo` / `-Repo`, `--base-url` / `-BaseUrl` (a mirror or a local
folder of release assets); each also reads `FOLO_VERSION`, `FOLO_INSTALL_DIR`, `FOLO_REPO`, `FOLO_BASE_URL`.
`folo --version` shows the version, commit and build date. To build from source instead:
`cargo install --path crates/folo-cli` (add `--features embed-web` after building `web/`). Releases:
[docs/RELEASING.md](docs/RELEASING.md).

## Running the server

```sh
cargo run -p folo-cli -- serve --mock   # mock data on http://127.0.0.1:7878 (API under /api)
cargo run -p folo-cli -- serve          # real store: datasets from disk; training / inference answer 501 for now
```

`--mock` serves deterministic datasets, models, finished runs and one live training run whose SSE stream advances in
real time, so FOLO Studio (`web/`, Vite dev server on :5173) and FOLO Desktop can be built before the engine exists.
The server also serves the built Studio from `web/dist` (`--web-dir`) when it exists. Flags: `--host`, `--port`
(`0` picks a free one), `--data-dir`, `--cors-any`, and the global `--json`. See [API](docs/API.md) and
[CLI](docs/CLI.md).

## Use FOLO from Claude Code / Codex

`folo mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio
([docs/MCP.md](docs/MCP.md)): agents get tools to build datasets, train, follow runs, evaluate, predict and export,
`folo://` resources (dataset images, run status, metrics, sample grids, the docs) and two prompts. It speaks MCP
2026-07-28 and 2025-11-25.

**Claude Code**: register it once (`--scope project` writes `.mcp.json` into the repository instead):

```sh
claude mcp add folo -- folo mcp --data-dir D:/folo-data
```

or put this in `.mcp.json` (Claude Desktop: `claude_desktop_config.json`):

```json
{ "mcpServers": { "folo": { "command": "folo", "args": ["mcp", "--data-dir", "D:/folo-data"] } } }
```

**Codex**: add to `~/.codex/config.toml` (or run `codex mcp add folo -- folo mcp --data-dir D:/folo-data`):

```toml
[mcp_servers.folo]
command = "folo"
args = ["mcp", "--data-dir", "D:/folo-data"]
```

Flags: `--mock` (deterministic mock data, so the whole dataset → train → predict flow works today, before the
engine is wired into the real store), `--read-only` (only tools that do not change state), `--confine` (only read
files inside `--data-dir`), `--server http://127.0.0.1:7878` (forward to a running `folo serve`, so the agent and
Studio share runs). From a checkout, use `"command": "cargo", "args": ["run", "-q", "--release", "-p",
"folo-cli", "--", "mcp", "--mock"]`.

- Why: fast (Rust, hand-written kernels, GPU through wgpu everywhere including the browser), small, hackable, and
  built to be driven by agents (`--json`, MCP server, stable error codes).
- Read: [ARCHITECTURE](docs/ARCHITECTURE.md) · [API](docs/API.md) · [AGENTS](AGENTS.md)

Licensed under either of MIT or Apache-2.0 at your option.
