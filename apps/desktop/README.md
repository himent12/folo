# FOLO Desktop

The native FOLO app, built in Rust with [GPUI](https://gpui.rs) (Zed's GPU-accelerated UI framework). Same screens and
design language as FOLO Studio (`web/`): Overview, Datasets, Runs (with a live training dashboard), Models,
Playground and Settings, plus a command palette, keyboard navigation, toasts, and dark and light themes.

It is a separate cargo workspace, so the root workspace and its CI stay GPUI-free. See `docs/ARCHITECTURE.md`
(Front ends) and `docs/API.md` (its data contract).

## Run

```powershell
$env:CARGO_TARGET_DIR = "D:\folo-build\desktop"   # keep build output off C:
cargo run                                          # connects to or starts folo serve (real store, bare)
cargo run -- --server http://127.0.0.1:7878        # a specific server
cargo run -- --mock                                # built-in demo data
cargo run -- --theme light
cargo run -- --scenario run-live+hover             # a fixed, deterministic state (see below)
cargo run -- --help                                # every option (`--version`: the version); prints and exits
```

Keyboard: `Ctrl K` palette · `Ctrl N` new run · `Ctrl O` open image · `Ctrl 1–5` screens · `Ctrl ,` settings ·
`Ctrl Shift L` theme · `F11` full screen · `Alt ←` back · `Esc` close or go up · `Tab` / arrows move focus ·
`F6` next region.

## Window

The app draws its own title bar (`titlebar()` in `src/app.rs`) and tells Windows which parts of it are caption
areas: Minimize, Maximize / Restore and Close, and two drag areas, the title bar's empty space and the sidebar's
FOLO row (drag to move, double-click to maximize or restore, hover Maximize for the snap layouts). The breadcrumbs
and the search field in the title bar take their own clicks. The top few pixels (`layout.desktop_resize_edge`)
resize the window, as the other edges do, except when it is maximized or full screen.

Two GPUI details keep this working; `tools/window-check.ps1` catches a regression in either:

- **Caption areas are `.occlude()`d.** GPUI sends a caption press to the app first and passes it to Windows only
  if nothing prevented its default. A focus-tracked element under the pointer does (it focuses itself), and the
  root view is one, so without the occlusion no button, drag or double-click ever reached Windows.
- **The top edge is a strip over the title bar** (`resize-top`, on top and occluding). Windows resizes only where
  GPUI's hit test finds no caption area, and the title bar's drag area reaches the top of the window.

Hidden from the root, the chrome also leaves keyboard mode itself on a press (`leave_kb`), as the root does for
any other press.

An overlay's scrim (the palette, a dialog, the inspector) dims the title bar too, and so hides its caption areas
from the hit test. Each scrim carries the caption areas again (`caption_controls`, on the same row and button
tokens as `titlebar()`), over the scrim and under the overlay's own panel: with an overlay open the window still
moves, maximizes, minimizes and closes, and that doesn't dismiss the overlay. The whole dimmed title bar and brand
row are one drag band then (the crumbs and the search field take no clicks); a press on the dimmed page still
dismisses. The caption buttons' width and Close's hover colours are tokens (`layout.desktop_caption_button`,
`window.close`, `window.close_text`).

`F11` (or Toggle full screen in the palette) toggles full screen. There the window has no caption areas, so the
three buttons are plain buttons: Minimize, Restore (leaves full screen) and Close.

The window opens as it was closed: its restored size and place, maximized or full screen, and the theme
(`src/window_state.rs`), kept in `<FOLO_HOME or ~/.folo>/desktop.json` (`$FOLO_DESKTOP_STATE` names another file).
It reopens on the display it was on (GPUI's display UUID, kept with the place: each display's bounds are in its
own logical pixels, so with mixed scale factors their rectangles overlap and the place alone can't say which), if
that display still shows enough of the title bar to grab; else on the display that shows the most of it (a tie
goes to the primary display). A place no connected display shows enough of opens centred on the primary display. Launches for screenshots
(`--scenario`, `--width`, `--height`) neither read nor write it, and `--theme` beats the remembered theme.
Closed while minimized, it keeps the mode it had before (`window_state::closing_mode`).

The app icon (taskbar, Alt-Tab, Explorer) is `assets/folo.ico`, drawn from `design/logo.svg` and the tokens by
`tools/make-icon.ps1` (run it again when either changes, and commit the .ico). GPUI loads icon resource 1 from the
executable; `build.rs` turns the .ico into a compiled resource file in plain Rust and hands it to link.exe, so the
build needs no resource compiler or extra crate (MSVC targets only).

## Layout

```
build.rs          generates the theme tables from design/tokens.json and embeds assets/icons/*.svg
src/main.rs       args, fonts (design/fonts), window, keymap
src/window_state.rs  the remembered window and theme (desktop.json), where a saved window reopens
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
tools/            shoot.ps1 / shoot-all.ps1 (screenshots), window-check.ps1 (the window chrome, real input),
                  chrome-check.ps1 (its hit tests and buttons, by window message), make-icon.ps1 (assets/folo.ico)
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

By default the app looks for `folo serve` at `$FOLO_SERVER` (or `http://127.0.0.1:7878`); if none answers it starts a
local `folo serve` on a free port (`$FOLO_BIN`, next to the executable, or on `PATH`) and stops it on exit, opening
bare against the real data dir (`$FOLO_HOME` or `~/.folo`). `--server URL` and `--mock` pick a source explicitly;
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

### Window check

```powershell
tools\window-check.ps1                     # about a minute; leave the mouse alone
tools\window-check.ps1 -Exe D:\folo-build\desktop\release\folo-desktop.exe
```

It drives the window chrome with real mouse and keyboard input (Windows only) and prints PASS or FAIL per step:
the breadcrumbs and search field in the title bar, Maximize / Restore, double-click and drag on both drag areas,
the top resize edge, Minimize, `F11` on and off, the three buttons in full screen, Close, a title-bar press leaving
keyboard mode, the window's icon, the buttons and both drags with the palette and with the New run dialog open
(each still open after them, and dismissed by a click on the dimmed page), and the next launches (maximized, windowed and full screen again at the last size
and place, also after closing while minimized; a saved place off every display; a corrupt state file). It exits 1 if any step fails. The remembered state goes to `$FOLO_DESKTOP_STATE`, a fresh file
under `$FOLO_TEST_TMP\window-check` (or `-StateDir`), never the real profile, and the last check proves that. It
also saves a screenshot of the snap layouts over Maximize there, to look at.

`tools\chrome-check.ps1 -Exe <build> [-AppArgs "--scenario palette"]` is the quick check that needs no mouse: it
asks the window what `WM_NCHITTEST` answers over each caption area, the top edge and the page, and posts a
non-client press and release on Maximize, Minimize and Close. It works when something on the desktop swallows
injected input (a system prompt), and with any `--scenario` (an overlay open). It does not replace window-check:
it cannot drag, and it never proves that a real click arrives.

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
