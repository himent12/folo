# FOLO Desktop

The native FOLO app, built in Rust on [folo-ui](ui/README.md), its own UI framework (GPUI's API, rendered on the
CPU; the UI uses no GPU). Same screens and design language as FOLO Studio (`web/`): Overview, Datasets, Runs (with a
live training dashboard), Models, Benchmarks, Playground and Settings, plus a command palette, keyboard navigation,
toasts, and dark and light themes.

It is a separate cargo workspace with two members, the app and `ui/` (folo-ui), so the root workspace does not
build them. See `docs/ARCHITECTURE.md` (Front ends, folo-ui) and `docs/API.md` (its data contract).

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

Keyboard: `Ctrl K` palette · `Ctrl N` new run · `Ctrl O` open image · `Ctrl 1–5` screens · `Ctrl 6` benchmarks · `Ctrl ,` settings ·
`Ctrl Shift L` theme · `F11` full screen · `Alt ←` back · `Esc` close or go up · `Tab` / arrows move focus ·
`F6` next region.

## Window

The app draws its own title bar (`titlebar()` in `src/app/titlebar.rs`) and tells Windows which parts of it are caption
areas: Minimize, Maximize / Restore and Close, and two drag areas, the title bar's empty space and the sidebar's
FOLO row (drag to move, double-click to maximize or restore, hover Maximize for the snap layouts). The breadcrumbs
and the search field in the title bar take their own clicks. The top few pixels (`layout.desktop_resize_edge`)
resize the window, as the other edges do, except when it is maximized or full screen.

Two folo-ui details (both GPUI's behaviour, kept) shape this; `tools/window-check.ps1` catches a regression in
either:

- **Caption areas are `.occlude()`d.** folo-ui sends a caption press to the app first and passes it to Windows
  only if nothing handled it or prevented its default (`ui/src/platform/windows/chrome.rs`). A focus-tracked
  element under the pointer prevents it (it focuses itself), and the root view is one, so without the occlusion no
  button, drag or double-click would reach Windows.
- **The top edge is a strip over the title bar** (`resize-top`, on top and occluding). The hit test answers with
  the app's caption areas first and resizes only where it finds none, and the title bar's drag area reaches the
  top of the window.

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
It reopens on the display it was on (the display's UUID, derived from its device name as GPUI did, so a file
written by an older build still matches; kept with the place: each display's bounds are in its own logical pixels,
so with mixed scale factors their rectangles overlap and the place alone can't say which), if
that display still shows enough of the title bar to grab; else on the display that shows the most of it (a tie
goes to the primary display). A place no connected display shows enough of opens centred on the primary display. Launches for screenshots
(`--scenario`, `--width`, `--height`) neither read nor write it, and `--theme` beats the remembered theme.
Closed while minimized, it keeps the mode it had before (`window_state::closing_mode`).

The app icon (taskbar, Alt-Tab, Explorer) is `assets/folo.ico`, the logo of `design/logo-icon.svg` drawn with the
tokens by `tools/make-icon.ps1` (run it again when either changes, and commit the .ico). folo-ui loads icon resource
1 from the executable; `build.rs` turns the .ico into a compiled resource file in plain Rust and hands it to link.exe, so the
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
src/store/        fetched state, live run data (SSE events), the decoded-image cache
src/bench/        vision benchmarks: what the Benchmarks screens hold, the open benchmark's stream, their actions
src/app/          root view: chrome, sidebar, routing, fetching, live event pump, actions
src/controls/     focusable controls: buttons, segmented, chips, switch, slider, select, tooltip
src/ui/           stateless pieces: type, panels, badges, metrics, progress, skeletons, empty states
src/charts/       canvas charts: line (raw + smoothed, area, markers, linked hover), sparkline, histogram
src/viz.rs        images with detection boxes and label chips (design/README.md box spec)
src/screens/      one module per screen
src/overlays/     palette, dialogs, image inspector, toasts
src/scenario.rs   `--scenario` states for screenshots
ui/               folo-ui, the UI framework (ui/README.md)
tools/            developer scripts (below)
```

Tools (`tools/`, PowerShell 5.1 unless noted; each has a usage header). Two of them can take over the screen
in their default mode: `window-check.ps1` without `-Sandbox` moves the real cursor and takes focus, and
`chrome-check.ps1` without `-Offscreen` or `-Sandbox` opens a visible window and maximizes it. Every other tool,
and those two in their `-Sandbox` / `-Offscreen` modes, never shows a window on your desktop, moves the cursor
or takes focus.

- `shoot.ps1`: one headless screenshot (`folo-desktop --scenario … --shot`; Screenshots, below).
- `shoot-all.ps1`: every review scenario headless, in parallel (98 PNGs in a few seconds on a release build),
  as `<Dir>\<W>x<H>\<scenario>.png`: all scenarios at 1280 x 800, the densest at 960 x 620 and 1680 x 1050.
- `parity.py` (Python with numpy and Pillow): compares two such trees pixel by pixel (the per-pixel difference
  is the largest channel difference): per pair the share of pixels above `--tol`, the largest difference, the
  regions that differ, and a triptych (A | B | difference) for the worst; `summary.md` and `summary.json`.
  `--fail-pct P` exits 1 above P; an empty comparison or a missing folder exits 2. `ui/PARITY.md` uses it
  against the GPUI captures.
- `capture.ps1`, `capture-all.ps1`: the same scenarios captured from the real window, opened off screen
  (`FOLO_UI_OFFSCREEN=1`) and never focused; what the Win32 backend presents, at the display's scale, through
  PrintWindow into a plain DIB. `capture-all.ps1` writes the same tree as `shoot-all.ps1`, a few at a time.
- `interact-check.ps1`: drives the real window with window messages only, off screen (default) or on a
  private desktop (`-Sandbox`, which adds the file dialog): its captures equal the headless shot of the
  same scenario, and hover, clicks, the palette and its text field, Tab and F6 focus, wheel scrolling, a
  slider, a select, tooltips, the New run dialog, a toast, the theme toggle and resizing behave. PASS / FAIL
  per step, exit 1 on a failure. Copy and paste run only with `-Sandbox -Clipboard`, off by default:
  `-Clipboard` replaces what is on your clipboard with test text and does not put it back (a private desktop
  shares your clipboard); run it when you have nothing on the clipboard you need. Without `-Clipboard` it
  never opens the clipboard.
- `chrome-check.ps1`: the window chrome through window messages: what `WM_NCHITTEST` answers over each caption
  area, the top edge and the page, and a non-client press and release on Maximize, Minimize (restored with
  `SC_RESTORE`, as the taskbar does) and Close; any `-AppArgs "--scenario …"` (an overlay open). Modes:
  `-Offscreen` (never on a display, never active; maximize is the app's off-screen emulation, checked by size),
  `-Sandbox` (on a private desktop), or neither (on your screen, it maximizes).
- `window-check.ps1`: the whole window chrome (Window check, below); `-Sandbox` on a private desktop.
- `drive.ps1`: the window-message driver the checks dot-source: posted mouse and non-client messages, keys
  with their modifiers in the UI thread's key state, a sync, PrintWindow captures. Nothing in it moves the
  cursor, activates a window or injects real input.
- `sandbox-run.ps1`: runs a command or an exe on a private Win32 desktop that is never shown, inside a job
  object that kills what it started; output in a log printed at the end, exit code the command's (124 on a
  timeout). The checks' `-Sandbox` modes and `win_smoke`'s sandbox checks use it.
- `win.ps1`: the Win32 helpers the capture and perf tools dot-source (find a process's window, rectangles,
  launch with extra environment, the DIB capture).
- `perf.ps1`: start-up, memory, CPU and GPU numbers of a build, off screen, as JSON (`ui/PERF.md` has them for
  this build and GPUI's).
- `bench-frames.sh` (bash): the frame bench (`--bench-frames`: the time and allocations of each frame phase)
  over the main screens at two window sizes; build with `--features bench-alloc` for allocation counts.
- `cross-check.sh` (bash): type-checks and lints folo-ui for Linux and both macOS targets.
- `make-icon.ps1`: draws `assets/folo.ico`, the logo of `design/logo-icon.svg`, with the tokens.
- `review-brief.md`: the brief a design reviewer gets with a round's shots.

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
  for screenshots and tests, and when picked (`--mock`, `--scenario`, Settings › Connection); an unreachable server
  shows as unreachable, never as demo data. It also has cloud compute (docs/CLOUD.md), as
  `folo serve --mock` does: the real providers' fields, one RunPod connection with a fixed offer list, one cloud
  run released long ago (with its final cost and terminated instance), and runs created with a GPU under Run on
  going through every phase. `MockBackend::with_cloud_runs` adds one uploading its data and one training (the
  cloud scenarios: `+cloud`, `cloud-…`, `run-cloud+…`); `+empty` scenarios have no connection and `+trouble` ones
  a revoked key and a termination the provider hasn't confirmed.

## Benchmarks

With a server that lists the `vision_bench` feature (docs/BENCH.md), **Benchmarks** in the sidebar (`Ctrl 6`) lists
the server's benchmarks and makes new ones (from a dataset split, or the labeled images of a labeling job, with the
default plan). A benchmark's page has three tabs:

- **Leaderboard**: the entries ranked by score, with the five axes on the score's 0–100 scale, running entries with
  their progress (live over the benchmark's event stream), and Run: a model of the server that knows the
  benchmark's classes (the others are listed, off), or a coding agent (servers with `agent_labeling`).
- **Plan**: each condition with the benchmark's first image under it, its severities and its share of robustness;
  **Design with an agent** proposes a plan beside the current one while no entry has run.
- **Loops**: the self-improvement loops with their verdicts, and **New loop** (a teacher with the benchmark's
  classes in its order, the pool it labels, generations, epochs per student).

An entry's page shows the score and its axes, accuracy, self-labeling, robustness per condition and severity, per
class, calibration and localization, then its images worst first with the truth (dashed) and its detections.
Without the feature the section says so. The built-in demo has three benchmarks (`src/api/mock_bench/`, scored by
`folo-bench` itself); `FOLO_SERVER=… cargo test real_server_bench -- --ignored` checks every benchmark call against
a real server.

## Cloud compute

With a server that lists the `cloud` feature, Settings › Cloud connects providers (the fields come from the
server; secrets are typed into masked fields, sent once and never kept by the app) and lists every instance with
its cost and a Terminate; New run's Compute section picks where a run trains (this machine or a connection, its
GPU offer and Stop after, with the cost beside Start training); a cloud run's page shows its phases, instance,
cost and time limit. `src/cloud.rs` holds the state and the rules those surfaces share.

By default the app looks for `folo serve` at `$FOLO_SERVER` (or `http://127.0.0.1:7878`); if none answers it starts a
local `folo serve` on a free port (`$FOLO_BIN`, next to the executable, or on `PATH`) and stops it on exit, opening
bare against the real data dir (`$FOLO_HOME` or `~/.folo`). When that data dir already has a `folo serve` (another
window's), it connects to that one instead (`data_dir_in_use`, docs/CLI.md). `--server URL` and `--mock` pick a source explicitly;
Settings › Connection switches at runtime. `--open run:<id>` (or `dataset:`, `model:`, `runs`, …) starts on a screen.

## Design tokens

`build.rs` reads `../../design/tokens.json`, the source of truth shared with Studio, and generates `Colors` (both
themes), the type scale, spacing, radii, sizes, motion and shadow tables. Never hard-code a colour or size in the app;
add a token instead. folo-ui has no letter-spacing (GPUI had none either, and the design was tuned without it), so
`tracking` values are not applied.

## Screenshots and the design hill-climb

`--scenario <name>` freezes the mock clock and puts the app in a state: a screen, an overlay, a hover. Names compose
with `+` (`dataset+light`, `overview+empty`). `folo-desktop --list-scenarios` prints the review set.

```powershell
tools\shoot-all.ps1 -Dir shots\round1      # every scenario + the minimum and wide window sizes (98 PNGs in <W>x<H> folders; a few seconds on a release build)
tools\shoot.ps1 -Scenario palette -Out shots\palette.png -Width 960 -Height 620
tools\shoot.ps1 -Scenario none -Out shots\run.png -ExtraArgs "--server http://127.0.0.1:7878 --open run:run_9d9d"   # real data
```

Shots are headless: the scripts run `folo-desktop --scenario … --shot <png>`, which renders on folo-ui's headless
platform (no window, no cursor, no focus), settles animations on a fake clock and writes the frame; one takes
well under a second. `-Width` / `-Height` are the window's logical size and `-Scale` its scale factor, 1.5 by
default: the dev PC's display, at which every design review has been shot (the PNG is 1920 x 1200 at the default
1280 x 800). `shoot.ps1` runs `$CARGO_TARGET_DIR/debug/folo-desktop.exe` and `shoot-all.ps1` the release build (the debug one
if there is none) unless `-Exe` names another.

A real-data shot (`-Scenario none` and `--server URL`, with `--open`) waits for what the screen fetches: the
harness pumps the app's queues and waits for its background requests before it draws, and a live run's event
stream is read up to that point. Checked against `folo serve --mock`: a live run's page, Runs and Overview show
their data; an unreachable server shows the unreachable state. Always give `--server` or `--mock` with
`-Scenario none`; without them the app looks for, or starts, a local `folo serve`, as a normal launch does.
`tools\capture.ps1` takes the same arguments for a real window off screen.

### Window check

```powershell
tools\window-check.ps1 -Exe D:\folo-build\desktop\release\folo-desktop.exe -Sandbox          # private desktop, about a minute
tools\interact-check.ps1 -Exe D:\folo-build\desktop\release\folo-desktop.exe                  # off screen; -Sandbox -Clipboard adds copy and paste (overwrites your clipboard)
tools\chrome-check.ps1 -Exe D:\folo-build\desktop\release\folo-desktop.exe -Offscreen         # or -Sandbox
```

`window-check.ps1` drives the window chrome and prints PASS or FAIL per step: the breadcrumbs and search field
in the title bar, Maximize / Restore, double-click and drag on both drag areas, the top resize edge, Minimize,
`F11` on and off, the three buttons in full screen, Close, a title-bar press leaving keyboard mode, the window's
icon, the buttons and both drags with the palette and with the New run dialog open (each still open after them,
and dismissed by a click on the dimmed page), and the next launches (maximized, windowed and full screen again
at the last size and place, also after closing while minimized; a saved place off every display; a corrupt
state file). It exits 1 if any step fails. The remembered state goes to `$FOLO_DESKTOP_STATE`, a fresh file
under `$FOLO_TEST_TMP\window-check` (or `-StateDir`), never the real profile, and the last check proves that.

- With `-Sandbox` it runs on a private desktop nobody sees (`sandbox-run.ps1`) with window messages only
  (`drive.ps1`): each real action becomes the messages Windows would produce for it (a hit-tested
  `WM_(NC)LBUTTONDOWN` / `UP`, `WM_NCLBUTTONDBLCLK` for a double click on a caption, keys with their modifiers,
  `SC_RESTORE` for the taskbar). Two things only Windows does with the real mouse are replaced: a drag on a
  caption area or the top edge is the press (which the app must leave to Windows) followed by the move or
  resize Windows' loop would make, so the app's part (the hit test, the press, following the new place) is
  what is checked; and the snap-layouts flyout over Maximize is not shown.
- Without it, it uses real mouse and keyboard input on your screen (it moves the pointer and takes focus;
  leave the machine alone for about a minute), drags for real and saves a screenshot of the snap layouts
  over Maximize under the state folder.

`chrome-check.ps1` is the quick check of the same caption areas through messages alone: what `WM_NCHITTEST`
answers over each caption area, the top edge and the page, and a non-client press and release on Maximize,
Minimize and Close, with any `--scenario` (an overlay open). It cannot drag. `interact-check.ps1` covers the
page itself (Tools, above). All three find the window by its process and kill what they start by PID.

`shots/` is gitignored. Design reviews score the shots against `REVIEW_RUBRIC.md`; `HILLCLIMB.md` logs each round.

## folo-ui

`ui/` is the app's UI framework: GPUI's API for the subset the app uses, its own runtime, a CPU tile renderer,
DirectWrite text on Windows and cosmic-text elsewhere, and Win32, winit and headless platforms. How to build and
test it, what is verified on each platform and its developer switches (`FOLO_UI_OFFSCREEN`, `FOLO_UI_TRACE`,
`FOLO_UI_PLATFORM`) are in `ui/README.md`; the design and its contracts in `ui/DESIGN.md`; the files adapted from
GPUI in `ui/NOTICE.md`.
