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
cargo run -- --help                                # every option (`--version`: the version); prints and exits
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
  for screenshots, tests and when no server is available. It also has cloud compute (docs/CLOUD.md), as
  `folo serve --mock` does: the real providers' fields, one RunPod connection with a fixed offer list, one cloud
  run released long ago (with its final cost and terminated instance), and runs created with a GPU under Run on
  going through every phase. `MockBackend::with_cloud_runs` adds one uploading its data and one training (the
  cloud scenarios: `+cloud`, `cloud-…`, `run-cloud+…`); `+empty` scenarios have no connection and `+trouble` ones
  a revoked key and a termination the provider hasn't confirmed.

## Cloud compute

With a server that lists the `cloud` feature, Settings › Cloud connects providers (the fields come from the
server; secrets are typed into masked fields, sent once and never kept by the app) and lists every instance with
its cost and a Terminate; New run's Compute section picks where a run trains (this machine or a connection, its
GPU offer and Stop after, with the cost beside Start training); a cloud run's page shows its phases, instance,
cost and time limit. `src/cloud.rs` holds the state and the rules those surfaces share.

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
tools\shoot.ps1 -Scenario none -Out shots\run.png -ExtraArgs "--server http://127.0.0.1:7878 --open run:run_9d9d"   # real data
```

The scripts run `$CARGO_TARGET_DIR/debug/folo-desktop.exe` unless `-Exe` names another build (a release one, say).

`shots/` is gitignored. Design reviews score the shots against `REVIEW_RUBRIC.md`; `HILLCLIMB.md` logs each round.

## Vendored GPUI

`vendor/gpui` is GPUI 0.2.2 from crates.io, used via `[patch.crates-io]` in `Cargo.toml`, with one fix and two
trims:

- **The fix, `src/elements/text.rs`.**
  - **The bug:** the text element built its `runs` once per layout and let `truncate_line` shorten them in place.
    A later measure in the same layout (another width, or none) shaped the whole text with run lengths cut for the
    truncated one. DirectWrite then sliced the text at that byte, and inside a multi-byte character ("·", "é", CJK)
    that panicked (`direct_write.rs`: "byte index … is not a char boundary"). It crashed the finished-run page
    on "best F1 · conf 0.4xx · val".
  - **The change:** `truncate_for_measure` now truncates a copy of the element's runs on every measure. Truncation
    still cuts on character boundaries, so the runs always cover the text exactly.
  - **The test:** `folo_truncation_tests` measures truncated, then untruncated, text containing "·", "×", "é", "–",
    "…" and CJK at several widths, and checks the runs cover each text on character boundaries.
- **The trims:** `examples/` (4.8 MB of demo assets) and their 30 `[[example]]` entries in `Cargo.toml`, and
  `Cargo.toml.orig`. Nothing else differs from the published crate.

To check it on a GPUI upgrade:

1. **See whether upstream still needs the fix.** In the new crate's `src/elements/text.rs`, look at the closure in
   `TextLayout::layout`. If it still passes the element's own `runs` mutably to `truncate_line`, the fix is needed.
2. **Re-vendor.** Copy the new crate into `vendor/gpui` without `examples/`, drop its `[[example]]` entries, and
   re-apply `truncate_for_measure` and its test. `diff -ru` against the registry copy should show only those.
3. **Run the test and a panic sweep:**
   - `cargo test --manifest-path vendor/gpui/Cargo.toml --lib folo_truncation` (use its own target dir);
   - every scenario at 960, 1280 and 1680 widths with stderr captured, checking for no "panicked".
4. **Drop the vendor if upstream fixed it:** remove `vendor/gpui` and the `[patch]` entry.
