# FOLO Desktop

The native FOLO app, built in Rust with [GPUI](https://gpui.rs) (Zed's GPU-accelerated UI framework). Same screens and
design language as FOLO Studio (`web/`): Overview, Datasets, Runs (with a live training dashboard), Models,
Playground and Settings, plus a command palette, keyboard navigation, toasts, and dark and light themes.

It is a separate cargo workspace, so the root workspace and its CI stay GPUI-free. See `docs/ARCHITECTURE.md`
(Front ends) and `docs/API.md` (its data contract).

## Run

```powershell
$env:CARGO_TARGET_DIR = "D:\folo-build\desktop"   # keep build output off C:
cargo run                                          # live mock data, dark theme
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

Everything comes through `api::Backend`. Until `folo serve` lands, the app runs on `MockBackend`: three synthetic
datasets, four models, and six runs (one live, one queued, completed, stopped and failed ones) whose metrics are pure
functions of seeds and a clock, so a frozen clock gives pixel-identical screenshots. Two endpoints the screens need are
not in API v1 yet and are proposed: `GET /api/models/{id}/eval` (PR curves, confusion matrix) and
`GET /api/runs/{id}/logs` (log history; the SSE stream only carries new lines).

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
