# FOLO Desktop

The native FOLO app, built in Rust with [GPUI](https://gpui.rs) (Zed's GPU-accelerated UI framework). Same screens and
design language as FOLO Studio (`web/`): Overview, Datasets, Runs (with a live training dashboard), Models,
Playground and Settings, plus a command palette, keyboard navigation, toasts, and dark and light themes.

It is a separate cargo workspace, so the root workspace and its CI stay GPUI-free. See `docs/ARCHITECTURE.md`
(Front ends) and `docs/API.md` (its data contract).

## Run

```powershell
$env:CARGO_TARGET_DIR = "D:\folo-build\desktop"   # keep build output off C:
cargo run                                          # finds or starts folo serve, else demo data
cargo run -- --server http://127.0.0.1:7878        # a specific server
cargo run -- --mock                                # built-in demo data
cargo run -- --theme light
cargo run -- --scenario run-live+hover             # a fixed, deterministic state (see below)
```

Keyboard: `Ctrl K` palette · `Ctrl N` new run · `Ctrl O` open image · `Ctrl 1–5` screens · `Ctrl ,` settings ·
`Ctrl Shift L` theme · `Alt ←` back · `Esc` close or go up · `Tab` / arrows move focus.

## Layout

```
build.rs          generates the theme tables from design/tokens.json and embeds assets/icons/*.svg
src/main.rs       args, fonts (design/fonts), window, keymap
src/theme.rs      the active theme (Colors from tokens, dark/light), class palette, shadows
src/api/          Backend trait (the client seam), wire types for docs/API.md, the in-app mock
src/synth.rs      procedural "shapes" scenes behind the mock's images
src/local.rs      local inference: folo-vision's Predictor in process, on folo-gpu or the CPU
src/store.rs      fetched state, live run data (SSE events), the decoded-image cache
src/app.rs        root view: chrome, sidebar, routing, fetching, live event pump, actions
src/controls.rs   focusable controls: buttons, segmented, chips, switch, slider, select, tooltip
src/ui.rs         stateless pieces: type, panels, badges, metrics, progress, skeletons, empty states
src/charts.rs     canvas charts: line (raw + smoothed, area, markers, linked hover), sparkline, histogram
src/viz.rs        images with detection boxes and label chips (design/README.md box spec)
src/screens/      one module per screen
src/overlays.rs   palette, dialogs, image inspector, toasts
src/scenario.rs   `--scenario` states for screenshots
tools/            shoot.ps1 / shoot-all.ps1 (screenshots)
```

## Data

Everything comes through `api::Backend`, typed with the shared `crates/folo-api` wire types:

- **`HttpBackend`** talks to `folo serve` (docs/API.md): JSON over HTTP with `ureq`, live runs over SSE on a
  background thread, image bytes for the grid, multipart uploads for the Playground.
- **Local inference** (`src/local.rs`): Playground's *Run on* switch runs the model in this process instead of on the
  server: Local GPU (folo-gpu, wgpu) or Local CPU. Weights are the f16 export from
  `GET /api/models/{id}/weights?format=f16`, held in memory and revalidated by ETag before each run (a retrained model
  reloads; an unreachable server still runs the loaded copy). The demo data has no weights, so there it is Server only.
  `FOLO_SERVER=… cargo test real_server -- --ignored` checks local results against the server's.
- **`MockBackend`** is built-in demo data (procedural shapes datasets, models and runs with a live simulator) used
  for screenshots, tests and when no server is available.

By default the app looks for `folo serve` at `$FOLO_SERVER` (or `http://127.0.0.1:7878`); if none answers it starts
`folo serve --mock` on a free port (`$FOLO_BIN`, next to the executable, or on `PATH`) and stops it on exit; if that
fails it falls back to the built-in demo data and says so. `--server URL` and `--mock` pick a source explicitly;
Settings › Connection switches at runtime. `--open run:<id>` (or `dataset:`, `model:`, `runs`, …) starts on a screen.

## Design tokens

`build.rs` reads `../../design/tokens.json`, the source of truth shared with Studio, and generates `Colors` (both
themes), the type scale, spacing, radii, sizes, motion and shadow tables. Never hard-code a colour or size in the app;
add a token instead. GPUI has no letter-spacing, so `tracking` values are not applied.

## Screenshots and the design hill-climb

`--scenario <name>` freezes the mock clock and puts the app in a state: a screen, an overlay, a hover. Names compose
with `+` (`dataset+light`, `overview+empty`). `folo-desktop --list-scenarios` prints the review set.

```powershell
tools\shoot-all.ps1 -Dir shots\round1      # every scenario + minimum and wide window sizes, at the display's scale
tools\shoot.ps1 -Scenario palette -Out shots\palette.png -Width 960 -Height 620
```

`shots/` is gitignored. Design reviews score the shots against `REVIEW_RUBRIC.md`; `HILLCLIMB.md` logs each round.
