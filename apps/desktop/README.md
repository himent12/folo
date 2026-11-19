# FOLO Desktop

The native FOLO app, built in Rust on [folo-ui](ui/README.md), its own UI framework (GPUI's API, rendered on the
CPU; the UI uses no GPU). Same screens and design language as FOLO Studio ([web/](../../web/README.md)): Overview,
Datasets, Runs (with a live training dashboard), Models, Benchmarks, Playground and Settings, plus a command palette,
keyboard navigation, toasts, and the theme system of [design/THEMES.md](../../design/THEMES.md) (Dark, Light or
System, a theme per mode, custom themes).

It holds no data of its own: everything comes from a `folo serve` over [docs/API.md](../../docs/API.md) (Data,
below), or from the built-in demo. The repository's [README](../../README.md) says what FOLO is;
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) how the engine and the front ends fit together;
[design/README.md](../../design/README.md) the design system both front ends share.

## Where things are

Two folders are called `ui`: `src/ui/` is a module of this app (its stateless building blocks), and `ui/` next to
`src/` is the folo-ui crate, the UI framework underneath everything.

| Path | What it holds |
| --- | --- |
| `src/lib.rs` | `run`, the app: arguments (`USAGE` is `--help`), the bundled fonts (`design/fonts`), the data source, the window, key bindings |
| `src/main.rs` | the `folo-desktop` binary: `fn main` calls `folo_desktop::run`, and the `bench-alloc` allocator |
| `src/app/` | `FoloApp`, the root view: `Route` and navigation, title bar, sidebar, routing a route to its page (`page.rs`), fetching and the live event pump (`data.rs`), finding or starting a server (`connection.rs`), actions and key bindings (`mod.rs`, `render.rs`), the forms' state; most of the app's behaviour tests (`tests.rs`, `*_tests.rs`) |
| `src/screens/` | one folder per page, each `impl FoloApp` methods that return it: `overview/`, `datasets/` (the index, a dataset's images and statistics), `runs/` (the list, the live training dashboard), `models/` (the list, a model's evaluation and export), `bench/` (Benchmarks: the list, a benchmark's leaderboard, plan and loops, an entry), `playground/`, `settings/` (connection, throughput, cloud, appearance, shortcuts); `first_step.rs` is what an empty Runs or Models page asks for |
| `src/overlays/` | what sits above the page: the command palette, the New run and Generate synthetic dialogs, the confirmations, the cloud dialogs, the image inspector, toasts |
| `src/controls/` | focusable controls, each an `impl FoloApp` method taking a stable key: buttons, segmented, tabs, chips and switches, slider, select, tooltip, scroll |
| `src/ui/` | the app's stateless pieces: type, surfaces, badges and status, metrics, legends, number and text formatting, motion |
| `src/charts/` | canvas charts: line (raw + smoothed, area, markers, hover cursor), sparkline, histogram |
| `src/viz.rs` | images with detection boxes and label chips (the box spec of design/README.md) |
| `src/store/` | client state: fetched values (`Load<T>`), live run data from the SSE events, the decoded-image cache, the "best model" rule |
| `src/api/` | the `Backend` trait (`mod.rs`) and the wire types (`types.rs`, from `crates/folo-api`); `http/`, the client of `folo serve` (transport, SSE streams, images, inference, `server.rs`, which starts a local `folo serve`, and `refused.rs`, which tells a closed port on this machine at once); `mock/` and `mock_bench/`, the built-in demo |
| `src/bench/` | the state and actions behind the Benchmarks screens, and the open benchmark's event stream |
| `src/cloud.rs` | cloud compute: the state and rules Settings › Cloud, New run and a cloud run's page share |
| `src/local.rs` | Playground's local inference: folo-vision's predictor in this process, on folo-gpu or the CPU |
| `src/input.rs` | the single-line text input (IME, selection, clipboard) and its key bindings |
| `src/theme/` | the active theme over the generated token tables: the mode, colour helpers, the class palette (`mod.rs`); the themes to pick from and the mode / slot rules (`catalog.rs`); custom theme files (`custom.rs`) |
| `src/window_state.rs` | the window and theme choice remembered in `desktop.json`, and where a saved window reopens |
| `src/scenario.rs` | `--scenario` states for screenshots (`ALL` is the review set) |
| `src/synth.rs` | procedural "shapes" scenes behind the demo's images |
| `build.rs` | generates the token tables from `design/tokens.json` (Design tokens, below), embeds `assets/icons/*.svg`, and on MSVC links `assets/folo.ico` and `assets/folo-desktop.manifest` into the executable |
| `assets/` | the SVG icons (`LICENSE-lucide.txt` for the Lucide ones), the app icon, the application manifest (DPI awareness, Common Controls 6, the segment heap) |
| `ui/` | folo-ui, the UI framework crate ([ui/README.md](ui/README.md)) |
| `tools/` | developer scripts (Tools, below) |
| `notes/` | the design review record: `REVIEW_RUBRIC.md` and `HILLCLIMB.md` |
| `DIRECTION.md` | the design direction the screens are built against |

## Workspace, build, test

`apps/desktop` is its own cargo workspace with two members, the app (`folo-desktop`) and `ui/` (`folo-ui`), its own
`Cargo.lock` and its own profiles. The root workspace excludes it (`exclude = ["apps/desktop"]` in the root
`Cargo.toml`), so `cargo build` and `cargo test` at the root never build the app or folo-ui, which are a slow build;
their CI (below) runs only when something they depend on changes. From `crates/` the app takes, by path, `folo-api`
(the wire types), `folo-bench` (the demo's benchmarks), and `folo-core`, `folo-data`, `folo-gpu`, `folo-nn` and
`folo-vision` (local inference, plus the engine's image decoding and its automatic optimizer choice, which New run
shows); `folo-test-support` and `folo-cloud` (the real providers, which the demo's list must match) are
dev-dependencies. The app does not link folo-server: it starts the `folo` binary instead.

Run every command from `apps/desktop`. A bare `cargo test` there builds only the app, so pass `--workspace`:

```powershell
cargo run -- --mock                                             # the demo data (Run, below, for the other sources)
cargo test --workspace                                          # the app and folo-ui
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run -- --scenario overview --shot overview.png            # a headless screenshot, no window
cargo build --profile dist                                      # the shipped build (thin LTO, one codegen unit)
```

Build speed. What keeps the app's build short, each measured A/B on the dev PC (i7-14700KF, `jobs = 8`, medians;
other builds were running, so take the clean-build numbers as rough):

- The app is a library (`src/lib.rs`, the whole app) and a thin `folo-desktop` binary (`src/main.rs`). Cargo starts
  compiling a library once its dependencies' metadata exists, a binary only after their code is built, so the app's
  front end and code generation overlap folo-ui's and folo-vision's.
- The HTTP client parses every JSON body from bytes (`json` in `src/api/http/transport.rs`): ureq's `read_json` compiles
  two serde parsers per response type, which were most of the client's machine code. Decode bodies through `json`, not
  `read_json` or `serde_json::from_str`. Both changes together: a release build from an empty target dir 122 s before,
  93 s after (a quiet machine); a release rebuild after one screen file changes 62 s, 42 s. Moving the API layer
  (`src/api`, `src/synth.rs`) into a crate of its own gained about 3 s more on that rebuild, nothing on a clean build,
  and made a clean `cargo test --workspace` 15 to 25% slower (one more test binary), so the app stays one crate.
- On Windows every build links with rust-lld, which ships with the toolchain (`linker` in the root
  `.cargo/config.toml`), and the dev profile keeps line tables only (file and line in backtraces, nothing for a
  debugger). Against link.exe and full debug info: a clean `cargo build` about 2 min 45 s instead of 3 min 55 s,
  `cargo build` after an edit of the app 3.4 s instead of 7 s, `cargo test --workspace --no-run` 3 min 17 s instead
  of 7 min 8 s from clean and 5 s instead of 14 s after an edit. `--release` and `dist` spend their time in code
  generation and thin LTO, so the linker barely shows there.
- The engine crates compile rayon's parallel drivers once instead of once per kernel (`cpu::par_chunks` /
  `cpu::par_for`, docs/CORE.md): folo-core in a clean release build of the app about 35 to 46 s instead of 70 to
  82 s, and a rebuild after an edit of folo-core 13% shorter. folo-data reads available memory itself on Windows and
  Linux, so the app no longer builds sysinfo and a second `windows` crate there (19 fewer compile units, 6% less CPU).

Cargo runs at most 8 jobs (`jobs = 8` in the root `.cargo/config.toml`, because several builds share that machine).
On an otherwise idle machine, `$env:CARGO_BUILD_JOBS=28` (or `cargo build -j 28`) gives cargo every core: a clean
`--release` build took 1 min 49 s instead of 2 min 8 s (measured before the library split). The gain is small because
the longest chain (wgpu, then folo-ui and the engine, then the app) runs one crate at a time.

Cargo features of the app: `bench-alloc` counts allocations for `--bench-frames`; `winit` and `portable-text` turn
on folo-ui's features of the same names, so that on Windows the app runs on the Linux / macOS backend
(`--features winit,portable-text`, then `FOLO_UI_PLATFORM=winit`). folo-ui's own features and tests are in
[ui/README.md](ui/README.md).

CI is `.github/workflows/desktop.yml`, on a pull request that touches `apps/desktop`, `design/`, the engine crates
the app uses, or the shared build files, and on every push to main:

- **Windows:** fmt, clippy of the workspace, clippy of folo-ui with `portable-text,winit,test-support`, the tests
  of both (each through `tools/test-tmp.sh`), and the root workspace's temp-dir hygiene test.
- **Linux:** clippy and the tests of the workspace, a headless `--shot` of `overview` checked to be a 1280 x 800
  PNG, and `tools/cross-check.sh` for folo-ui on both macOS targets.

The release workflow (`.github/workflows/release.yml`) builds `--profile dist` on Windows and ships
`folo-desktop.exe`.

## Common changes

Each recipe names an existing example to copy.

- **A screen.** Add a `Route` variant (and a `Section`, for a new sidebar entry) with its `--open` name in
  `Route::parse` (`src/app/mod.rs`); a folder under `src/screens/` declared in `screens/mod.rs`, whose
  `impl FoloApp` method returns the page; its arm in `FoloApp::page` (`src/app/page.rs`). The compiler lists every
  other exhaustive `match` on `Route` or `Section`. Then the sidebar's `nav` list (`src/app/sidebar.rs`), the
  palette's (`palette_items` in `src/overlays/palette.rs`), `USAGE` in `src/lib.rs`, and a scenario in
  `src/scenario.rs` (`ALL` and `apply`). Example: Benchmarks (`Route::Benches`, `screens/bench/`).
- **An API call.** Check the endpoint in docs/API.md and its types in `crates/folo-api`. `src/api/types.rs`
  re-exports those types by name, so a new type goes into one of its `pub use folo_api::{…}` lists. Add the method
  to `Backend` (`src/api/mod.rs`), with a default that answers `not_supported` when only some servers have it;
  implement it on `HttpBackend` (`src/api/http/`, through `self.get` / `self.post`) and on `MockBackend`
  (`src/api/mock/backend.rs`). A page calls it on the background executor with `FoloApp::fetch` (`src/app/data.rs`)
  and keeps the answer in a `Load<T>` (`src/store/mod.rs`).
  Example: `throughput`, loaded by `load_throughput` in `src/screens/settings/throughput.rs`.
- **A control.** An `impl FoloApp` method in a file under `src/controls/` that builds a `Stateful<Div>` with the
  key as its id and passes it through `FoloApp::bind` (`controls/focus.rs`), which makes it focusable and
  activatable by click, Enter and Space. Example: `chip` in `src/controls/toggle.rs`.
- **A command-palette command.** Push a `PaletteItem` (group, icon, title, keys, and an `act(...)` closure) in
  `palette_items` (`src/overlays/palette.rs`). Example: Toggle full screen. Icons are paths under
  `assets/icons/`, which `build.rs` embeds.
- **A key binding.** Declare the action in `actions!` and bind it in `bind_keys` (`src/app/mod.rs`; the text
  input's own bindings are in `src/input.rs`, under its context), then handle it with `.on_action` on the root in
  `src/app/render.rs`. List it in Settings' shortcuts (`src/screens/settings/page.rs`), on the palette item if
  there is one, and in Run, below. Example: `ToggleFullscreen` on `F11`.
- **A colour or size.** A token in `design/tokens.json`, never a literal (Design tokens, below).

## Run

```powershell
cargo run                                          # connects to or starts folo serve (real store, bare)
cargo run -- --server http://127.0.0.1:7878        # a specific server
cargo run -- --mock                                # built-in demo data
cargo run -- --theme light                         # a mode (dark, light, system) or a theme id (midnight)
cargo run -- --scenario run-live+hover             # a fixed, deterministic state (see below)
cargo run -- --help                                # every option (`--version`: the version); prints and exits
```

An unknown option, an option missing its value, a `--theme` that is no mode and no installed theme's id, and an
unknown `--scenario` each print one `folo-desktop: …` line on stderr; the app still opens (a bad `--theme` opens in
the remembered theme, else dark).

Build output goes to `target\` unless `CARGO_TARGET_DIR` names another folder; the tools below look there too.

Keyboard: `Ctrl K` palette · `Ctrl N` new run · `Ctrl O` open image · `Ctrl 1–5` screens · `Ctrl 6` benchmarks · `Ctrl ,` settings ·
`Ctrl Shift L` dark / light mode · `F11` full screen · `Alt ←` back · `Esc` close (a dialog, an inline form, a delete waiting
for its second click) or go up · `Ctrl Enter` submit the open dialog or form · `Tab` / arrows move focus ·
`F6` next region. Text fields: `Home` / `End`, `Ctrl ←/→` word, `Ctrl Backspace` / `Ctrl Delete` delete a word,
`Shift` with any of these moves select, `Ctrl A` all.

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
dismisses. An overlay takes no press that continues a double-click begun before it came up (`OverlayPresses`, by
the platform's click count, so the system's double-click time and distance): double-clicking the button that
opens a dialog leaves it open, and the second press cannot press what opened under the pointer (a confirmation's
Delete); any later press on the scrim dismisses. Menus and selects are not overlays: a double-click on a select
opens and closes it, as on a Windows combo box. The caption buttons' width and Close's hover colours are tokens (`layout.desktop_caption_button`,
`window.close`, `window.close_text`).

`F11` (or Toggle full screen in the palette) toggles full screen. There the window has no caption areas, so the
three buttons are plain buttons: Minimize, Restore (leaves full screen) and Close.

The window opens as it was closed: its restored size and place, maximized or full screen, the theme choice (the
mode and each mode's theme; Themes, below) and Reduce motion when it was set against the OS's own setting (`src/window_state.rs`), kept in `<FOLO_HOME or ~/.folo>/desktop.json` (`$FOLO_DESKTOP_STATE` names another file). It is read field by
field: one value this build can't read (a wrong type, a window mode from a newer build) is dropped alone.
It reopens on the display it was on (the display's UUID, derived from its device name as GPUI did, so a file
written by an older build still matches; kept with the place: each display's bounds are in its own logical pixels,
so with mixed scale factors their rectangles overlap and the place alone can't say which), if
that display still shows enough of the title bar to grab; else on the display that shows the most of it (a tie
goes to the primary display). A place no connected display shows enough of opens centred on the primary display. Launches for screenshots
(`--scenario`, `--width`, `--height`) neither read nor write it, and `--theme` beats the remembered theme.
Closed while minimized, it keeps the mode it had before (`window_state::closing_mode`). Closing the window stops the
`folo serve` the app started: when a run of this machine is training or queued on it, the app asks first (Keep open,
or Stop and close).

The app icon (taskbar, Alt-Tab, Explorer) is `assets/folo.ico`, the logo of `design/logo-icon.svg` drawn with the
tokens by `tools/make-icon.ps1` (run it again when either changes, and commit the .ico). folo-ui loads icon resource
1 from the executable; `build.rs` turns the .ico into a compiled resource file in plain Rust and hands it to the linker, so the
build needs no resource compiler or extra crate (MSVC targets only).

## Tools

The scripts in `tools/` are Windows PowerShell 5.1, and so Windows only, unless noted: `parity.py` is Python, and
`bench-frames.sh` and `cross-check.sh` are bash. Each has a usage header. Two of them can take over the screen
in their default mode: `window-check.ps1` without `-Sandbox` moves the real cursor and takes focus, and
`chrome-check.ps1` without `-Offscreen` or `-Sandbox` opens a visible window and maximizes it. Every other tool,
and those two in their `-Sandbox` / `-Offscreen` modes, never shows a window on your desktop, moves the cursor
or takes focus.

Off screen (`FOLO_UI_OFFSCREEN=1`) the app opens no file dialog, Explorer or browser: Add dataset, Open image
and Ctrl O answer with a toast ("The file dialog didn't open"), Show in Explorer, Show in folder and links
do nothing but say so on stderr, on either backend (Win32, or winit with `FOLO_UI_PLATFORM=winit`); on
Win32 any other window of the process that reaches a display is moved off it too (folo-ui's README,
`FOLO_UI_OFFSCREEN`, says what each backend guarantees). On a private desktop without it the dialogs are real, but
never click Show in Explorer, Show in folder or a link there: Explorer and the browser already running
on your desktop take the request and open their window on your screen.

- `shoot.ps1`: one headless screenshot (`folo-desktop --scenario … --shot`; Screenshots, below).
- `shoot-all.ps1`: every review scenario headless, in parallel (109 PNGs in a few seconds on a release build),
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
  slider, a select, tooltips, the New run dialog, a toast, the theme toggle and resizing behave (with
  `-Sandbox` also the file dialog, and that off screen Ctrl O opens none). PASS / FAIL
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
  cursor, activates a window or injects real input. `[FoloDrive]::Find` returns the app's window by its
  class (`FOLO::Window`), never another window of the process; `[FoloDrive]::OnDisplay(pid)` lists every
  visible window of the process that a display shows a part of, and `Assert-FoloOffscreen $p` kills the
  app and throws when that list is not empty or the app took the foreground. Guard with it after every
  step that could open something: a check of the app's window alone misses a dialog, which is a window
  of its own (a file dialog once stayed on the screen for minutes that way).
- `sandbox-run.ps1`: runs a command or an exe on a private Win32 desktop that is never shown, inside a job
  object that kills what it started; output in a log printed at the end, exit code the command's (124 on a
  timeout). The checks' `-Sandbox` modes and `win_smoke`'s sandbox checks use it.
- `win.ps1`: the Win32 helpers the capture and perf tools dot-source (find the app's window by class,
  every window of a process on a display, rectangles, launch with extra environment, the DIB capture).
- `perf.ps1`: start-up, memory, CPU and GPU numbers of a build, off screen, as JSON (`ui/PERF.md` has them for
  this build and GPUI's).
- `bench-frames.sh` (bash): the frame bench (`--bench-frames`: the time and allocations of each frame phase, for
  full rebuilds, animation ticks and 40 px wheel scrolls over the window's centre) over the main screens at two window sizes; build with `--features bench-alloc` for allocation counts. Its default
  executable is `$CARGO_TARGET_DIR/release/folo-desktop.exe`; pass another as the first argument.
- `cross-check.sh` (bash, any host): type-checks and lints folo-ui for Linux and both macOS targets (CI runs it on
  Linux).
- `make-icon.ps1`: draws `assets/folo.ico`, the logo of `design/logo-icon.svg`, with the tokens.
- `review-brief.md`: the brief a design reviewer gets with a round's shots.

## Data

Everything comes through `api::Backend`, typed with the shared `crates/folo-api` wire types:

- **`HttpBackend`** talks to `folo serve` (docs/API.md): JSON over HTTP with `ureq`, live runs over SSE on a
  background thread, image bytes for the grid, multipart uploads for the Playground.
- **Local inference** (`src/local.rs`): Playground's *Run on* switch runs the model in this process instead of on the
  server: Local GPU (folo-gpu, wgpu) or Local CPU. Weights are the f16 export from
  `GET /api/models/{id}/weights?format=f16`, held in memory and revalidated by ETag before each run (a retrained model
  reloads; an unreachable server still runs the loaded copy, on the image on screen: a dataset image's file is kept
  as fetched). A local run still waiting for the model when a newer one is asked for (a threshold stepped again) is
  skipped, so the newest answer comes after one inference. The demo data has no weights, so there it is Server only.
  Images are shown upright by their EXIF orientation, as the engine reads them (`folo_data::decode`), so the boxes
  land on what is shown.
  `FOLO_SERVER=… cargo test real_server -- --ignored` checks local results against the server's.
- **`MockBackend`** is built-in demo data (procedural shapes datasets, models and runs with a live simulator) used
  for screenshots and tests, and when picked (`--mock`, `--scenario`, Settings › Connection); an unreachable server
  shows as unreachable, never as demo data. It also has cloud compute (docs/CLOUD.md), as
  `folo serve --mock` does: the real providers' fields, one RunPod connection with a fixed offer list, one cloud
  run released long ago (with its final cost and terminated instance), and runs created with a GPU under Run on
  going through every phase. `MockBackend::with_cloud_runs` adds one uploading its data and one training (the
  cloud scenarios: `+cloud`, `cloud-…`, `run-cloud+…`); `+empty` scenarios have no connection and `+trouble` ones
  a revoked key and a termination the provider hasn't confirmed.

By default the app looks for `folo serve` at `$FOLO_SERVER` (or `http://127.0.0.1:7878`; on Windows a closed port on this
machine is told at once, not after the probe's 0.5 s timeout, unless a proxy from the environment is used for it); if none answers it starts a
local `folo serve` on a free port (`$FOLO_BIN`, next to the executable, or on `PATH`) and stops it on exit, opening
bare against the real data dir (`$FOLO_HOME` or `~/.folo`). When that data dir already has a `folo serve` (another
window's), it connects to that one instead (`data_dir_in_use`, docs/CLI.md). If that local server stops answering (the
window that started it closed, or it crashed), the app finds or starts one again the same way and says so. Why a
local server couldn't start stays on the pages' error panels. `--server URL` and `--mock` pick a source explicitly;
Settings › Connection switches at runtime: an address on this machine that doesn't answer falls back to the local
server with a note saying so, and another machine's address is never replaced by the local server. A server started
with `--token` shows as needing one (the app can't send a token); the server the app starts itself never inherits a
`FOLO_TOKEN` from the app's environment, so it is always one the app can use. A run's live event stream that stays
silent for 45 s (three of the server's 15 s keep-alives) counts as dropped: the app opens it again and reads the history
anew, and when the run list says the run ended meanwhile it loads the final history. When the server answers again
after being unreachable, every list that failed (runs too) is read again. `--open run:<id>` (or `dataset:`, `model:`, `runs`,
…) starts on a screen. The deciding is `FoloApp::autoconnect` (`src/app/connection.rs`); finding the binary,
starting and stopping it is `src/api/http/server.rs`.

## Benchmarks

With a server that lists the `vision_bench` feature (docs/BENCH.md), **Benchmarks** in the sidebar (`Ctrl 6`) lists
the server's benchmarks and makes new ones (from a dataset split, or the labeled images of a labeling job, with the
default plan). A benchmark's page has three underline tabs, each with its count:

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
cost and time limit. As Studio's, it says when the worker stops answering ("Worker not answering" beside the
status, and since when in the Cloud panel), and when a released run's instance is still alive in the instance list:
"Termination not confirmed: the instance may still be billed.", with Releasing still the current step and the
instance's own cost so far. `src/cloud.rs` holds the state and the rules those surfaces share; the demo's
`run-cloud+silent` and `run-cloud+trouble` scenarios show both.

## Design tokens

`build.rs` reads `../../design/tokens.json`, the source of truth shared with Studio, and generates `Colors`, the type
scale, spacing, radii, sizes, motion and shadow tables. Never hard-code a colour or size in the app; add a token
instead. folo-ui has no letter-spacing (GPUI had none either, and the design was tuned without it), so `tracking`
values are not applied.

For colours it resolves every built-in theme of `themes` (its mode's base palette, `color.dark` or `color.light`, with
its `color` merged over it key by key, arrays whole) into one constructor each (`theme_midnight()`, …), the `THEMES`
table (id, name, mode, description, constructor, in tokens.json's order), `TOKENS` (every colour token's path and
kind) and `Colors::set` / `Colors::set_number` by token path (`bg.app`, `chart.series.0`, `chart.area_opacity`) for
custom themes. A malformed theme (an unknown path, a wrong type or length, a mode's default that isn't first or has
overrides) fails the build with its path.

## Themes

As [design/THEMES.md](../../design/THEMES.md) says, with Desktop's parts:

- **Choosing.** Settings › Appearance: Theme (Dark / Light / System; with System its description says what the OS is
  now), Dark theme and Light theme (selects: built-ins in order, then custom themes by name), Custom themes, Reduce
  motion (`src/screens/settings/appearance.rs`). Picking a different theme of the other mode switches to that mode,
  except with System; re-picking the theme a select shows does nothing, by pointer as by Enter and Space (the slot's
  callback ignores its own theme; the select itself calls back for a click on its chosen option, which other selects
  need, such as New run asking about its dataset again). The palette has "Theme: <name>" for every theme, offered for a query only (type "theme"); Ctrl Shift L and "Switch to light / dark
  theme" switch the mode only, to an explicit Dark or Light. The rules are `ThemePrefs` and `Catalog` in
  `src/theme/catalog.rs`; `FoloApp::apply_theme` (`src/app/preferences.rs`) sets the `Theme` global.
- **System** follows the OS appearance live: folo-ui's `Window::on_appearance_changed` (Win32 `WM_SETTINGCHANGE`
  "ImmersiveColorSet"; winit's theme-changed event), registered by `app::follow_system_appearance`; the headless
  platform reports dark, so `--shot` with System shows the dark theme, and a test turns it with
  `HeadlessApp::set_system_appearance`, delivered through the same callback (`app/theme_tests.rs`). On Windows the frame DWM draws follows the mode shown (`Window::set_frame_appearance`),
  not the OS.
- **Custom themes** are every `*.json` in `<FOLO_HOME or ~/.folo>/themes/`, read at start and on Reload (off the UI
  thread), in file name order (a later file with the same id replaces an earlier one), each at most 1 MiB (the read
  stops one byte past it; an unreadable one is `(file): couldn't read (…)`, as in Studio). `src/theme/custom.rs`
  (`parse_bytes`: one byte order mark stripped, strict UTF-8, JSON; serde_json's own refusals, such as nesting past
  127 levels, are what `design/json-scan.mjs` makes the JS readers refuse too; serde_json's `float_roundtrip`
  feature rounds numbers as JavaScript does, so the range limit and every value agree)
  validates them with the messages of `design/theme-cases.json`, which its tests replay through the folder reader
  (`catalog::read_theme`), so Studio, `design/contrast.mjs` and Desktop agree. Under the Custom themes row, as Studio lists its imports, come the themes
  loaded (name; mode · based on its base · file name) and then each file that failed with its first error; the
  folder shows in the platform's form (`catalog::display_path`). There is no Remove: deleting the file and Reload
  does it. The app never opens a file picker or Explorer for them.
- **Stored** in `desktop.json`: `theme` is the mode (`dark`, `light`, `system`; older files read the same),
  `dark_theme` and `light_theme` the ids. An id no theme has (a removed custom theme) shows the mode's default and
  stays stored, so installing the theme again brings it back.
- **`--theme`** takes a mode or a theme id, custom ones too (`--theme midnight`: mode Dark with Midnight).
  `--scenario …+light` / `+dark` still force the mode, so `--scenario overview --theme paper` and
  `--scenario overview+light --theme paper` both shoot Paper.
- `Theme` keeps `mode` (`is_dark()`, the mode's shadows, the few mode rules such as the olive `accent_fill` in light)
  and gains `id`. Text on `accent_fill` is `Theme::on_accent_fill` (white in High contrast light, where `accent.on`
  holds only 3.56:1 on its darker olive).

## Screenshots and the design hill-climb

`--scenario <name>` freezes the mock clock and puts the app in a state: a screen, an overlay, a hover. Names compose
with `+` (`dataset+light`, `overview+empty`). `folo-desktop --list-scenarios` prints the review set.

```powershell
tools\shoot-all.ps1 -Dir shots\round1      # every scenario + the minimum and wide window sizes (109 PNGs in <W>x<H> folders; a few seconds on a release build)
tools\shoot.ps1 -Scenario palette -Out shots\palette.png -Width 960 -Height 620
tools\shoot.ps1 -Scenario settings+bottom -Out shots\appearance.png -ExtraArgs "--theme midnight"   # any theme
tools\shoot.ps1 -Scenario none -Out shots\run.png -ExtraArgs "--server http://127.0.0.1:7878 --open run:run_9d9d"   # real data
```

Shots are headless: the scripts run `folo-desktop --scenario … --shot <png>`, which renders on folo-ui's headless
platform (no window, no cursor, no focus), settles animations on a fake clock and writes the frame; one takes
well under a second. `-Width` / `-Height` are the window's logical size and `-Scale` its scale factor, 1.5 by
default: our test machine's display, at which every design review has been shot (the PNG is 1920 x 1200 at the default
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
tools\window-check.ps1 -Exe target\release\folo-desktop.exe -Sandbox          # private desktop, about a minute
tools\interact-check.ps1 -Exe target\release\folo-desktop.exe                  # off screen; -Sandbox -Clipboard adds copy and paste (overwrites your clipboard)
tools\chrome-check.ps1 -Exe target\release\folo-desktop.exe -Offscreen         # or -Sandbox
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

`shots/` is gitignored. `notes/REVIEW_RUBRIC.md` and `notes/HILLCLIMB.md` are the record of the design review rounds:
the rubric each round's shots were scored against, and the log of each round.

## folo-ui

`ui/` is the app's UI framework: GPUI's API for the subset the app uses, its own runtime, a CPU tile renderer,
DirectWrite text on Windows and cosmic-text elsewhere, and Win32, winit and headless platforms. How to build and
test it, what is verified on each platform and its developer switches (`FOLO_UI_OFFSCREEN`, `FOLO_UI_TRACE`,
`FOLO_UI_PLATFORM`) are in [ui/README.md](ui/README.md); the design and its contracts in
[ui/DESIGN.md](ui/DESIGN.md); the pixel comparison with the GPUI app in [ui/PARITY.md](ui/PARITY.md), its cost
against GPUI in [ui/PERF.md](ui/PERF.md), and the files adapted from GPUI in [ui/NOTICE.md](ui/NOTICE.md).
