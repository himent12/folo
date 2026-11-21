# folo-ui

FOLO Desktop's own UI framework. It has GPUI's API for exactly the subset the app uses (same names, signatures
and behaviour, so the app reads as it did on GPUI 0.2.2), and its own engine: it renders on the CPU, in 64 px
tiles, and redraws only the tiles that changed. The UI uses no GPU device, so the GPU and its memory stay free
for training and inference.

What it is not: a general UI toolkit, or GPUI. Only what `apps/desktop/src` calls exists (a GPUI method the app
does not use is not here), and there is no `gpui` dependency: the files that adapt GPUI source (Adapted from
GPUI, below) are part of this crate. The crate is a member of the FOLO Desktop workspace
([apps/desktop/README.md](../README.md)), not of the repository's root workspace ([README](../../../README.md),
[docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md)).

This file is how to work on the crate. The others next to it:

- [DESIGN.md](DESIGN.md): the design record: the decisions, the contracts between the parts (scene, renderer,
  text, platform), and how the port is verified. When code and DESIGN.md disagree, fix one of them.
- [PARITY.md](PARITY.md): how close the headless shots are to the GPUI app's, pixel by pixel, and why each
  remaining difference stays.
- [PERF.md](PERF.md): what FOLO Desktop costs on folo-ui against the GPUI build (start-up, memory, CPU and GPU,
  frames, the renderer, the build), and how to measure it again.
- [NOTICE.md](NOTICE.md): the attribution for GPUI (Apache-2.0) and every file adapted from it.

## The shape of it

```
 app code (apps/desktop/src)
 ──────────────────────────────────────────────────────────────────────────────
 elements + styles     div, text, svg, img, canvas, uniform_list, anchored, deferred, animation
 window                frame: render -> layout (taffy) -> prepaint -> paint -> Scene; hit test; focus; keys
 app runtime           App, Entity, Context, globals, subscriptions, actions + keymap, executors
 ──────────────────────────────────────────────────────────────────────────────
 text                  TextSystem (fonts, line layout cache, wrapping)  ─►  PlatformTextSystem
                         DirectWrite (Windows)  |  portable: cosmic-text (Linux, macOS, tests)
 raster                Scene ─► CPU tile renderer ─► Framebuffer + damage rects
 platform              Platform / PlatformWindow / PlatformDispatcher
                         windows (Win32, GDI present)  |  winit + softbuffer (Linux, macOS)  |  headless
```

Where things live, under `src/`:

| Folder or file | What it holds |
| --- | --- |
| `lib.rs`, `prelude.rs` | the module list, and the flat re-exports of everything public; the traits app code imports with `use folo_ui::prelude::*` |
| `geometry/`, `color.rs`, `shared_string.rs`, `input_events.rs`, `keystroke.rs` | value types, events, keystrokes |
| `opacity.rs` | opacity groups: an opacity that changes after paint without repainting the UI |
| `style/`, `styled/`, `layout.rs`, `path_builder.rs`, `path_cache.rs`, `svg.rs`, `image.rs`, `easing.rs`, `assets.rs` | `Style` and its refinement, the `Styled` methods, the mapping to taffy, paths (and the built ones kept by input), SVG icons, images |
| `app/`, `executor.rs`, `action.rs`, `keymap/`, `view.rs` | the runtime: `Application`, `App`, entities, contexts, effects, tasks, actions and key bindings, `--shot` |
| `window/`, `interactive/`, `element.rs`, `elements/` | the window and its frame pipeline, dispatch, focus, hit testing; the elements |
| `text/` | `TextSystem`, the line layout cache, wrapping and truncation, painting; `dwrite/` and `portable/` backends |
| `scene.rs`, `raster/` | the display list and the tile renderer with its caches and thread pool |
| `platform.rs`, `platform/` | the platform contract; `windows/` (Win32), `winit/`, `headless/` |
| `headless_app.rs` | `HeadlessApp`, the harness over the headless platform (tests and `--shot`) |

A frame: a view's `render` builds elements; the window lays them out with taffy, prepaints (bounds, hitboxes,
scroll offsets), paints a `Scene` (quads, shadows, glyph and SVG sprites, images, paths, underlines); the renderer
bins the scene into tiles, hashes each tile's primitives, rasterizes only the tiles whose hash changed into the
window's `Framebuffer`, and the platform presents those rectangles. A window draws only when something made it
dirty, at most once per display refresh; while nothing changes there is no timer and the process sleeps.

Two things keep a frame cheap without changing a pixel (`DESIGN.md`, Frames). The layout tree keeps what the
previous frame computed (`window/layout_tree.rs`): a subtree whose nodes, styles and texts are unchanged answers
the same layout requests from last frame's records instead of running taffy again, so the result is exactly a
fresh layout's. Within a frame, a node's cached final layout is not reused once the node has been measured
again since it was stored: taffy's block algorithm lays a block's children out at the width it measures with.
And an element that only animates its opacity (`with_opacity_animation`: the breathing status dot, fades, shimmers) is painted once inside an opacity group (`opacity.rs`, `window/paint_animation.rs`); at
each refresh the window re-fades that group's primitives in the scene and repaints only their tiles, with no
render, layout or paint.

## Build, test, bench

From `apps/desktop` (the crate is a member of that workspace). `CARGO_TARGET_DIR` and `FOLO_TEST_TMP` move the
build output and the tests' temporary files to another disk when the system one is short of space (`AGENTS.md`).

```
cargo build -p folo-ui
cargo test -p folo-ui                                                     # Windows: DirectWrite and Win32 tests too
cargo test -p folo-ui --features portable-text,winit,test-support         # adds the cosmic-text and winit code
cargo clippy -p folo-ui --all-targets -- -D warnings
cargo clippy -p folo-ui --all-targets --features portable-text,winit,test-support -- -D warnings
cargo bench -p folo-ui --bench raster                                     # the renderer on a busy frame
bash tools/cross-check.sh                                                 # type-check and lint for Linux and macOS
```

Features:

- `portable-text`: the cosmic-text text backend (`text/portable`). Always on off Windows. On Windows it is
  opt-in; the deterministic text tests use it, since it gives the same layout and masks on every OS.
- `winit`: the winit + softbuffer backend (`platform/winit`, with arboard and rfd). Always on off Windows. On
  Windows it lets that backend's shared code build and run on a Windows machine (`FOLO_UI_PLATFORM=winit`).
- `test-support`: exposes `FixedTextSystem` (`text/fixed.rs`), a fake text backend with fixed metrics, to other
  crates. Unit tests inside the crate get it through `cfg(test)`.

Code gates on `#[cfg(any(not(windows), feature = "winit"))]` (and the same with `"portable-text"`);
`platform/windows` and `text/dwrite` are `#[cfg(windows)]`.

Tests are deterministic and quick. Most run on the headless platform through `HeadlessApp`: a fixed window size
and scale, synthetic input, a fake clock that moves only when the test advances it, and a framebuffer to read
back. The renderer's tests check its invariant on random scenes (rendering A then B into one buffer gives the
same bytes as B into a fresh buffer, for every thread count) and its exactness against GPUI's formulas.

Examples (Windows, off screen, never focused, driven by posted window messages, never by real input):

```
cargo run -p folo-ui --release --example win_smoke [-- --quick]            # the Win32 backend
# on a private desktop only (they show windows, or would if off screen failed):
tools\sandbox-run.ps1 -Exe <target>\release\examples\win_smoke.exe -ExeArgs "--dialog" -Env @{ FOLO_SMOKE_SANDBOX = "1" }
#   --dialog: the modal file dialog keeps tasks and frames running; --offscreen-dialog: off screen
#   the dialogs are refused and a stray window is moved off the displays
cargo run -p folo-ui --example winit_smoke --features winit,portable-text   # the winit backend
cargo run -p folo-ui --example text_probe -- <lines.txt> <out-dir> [scale]  # DirectWrite layout and masks
```

Where the tests are:

| Path | What it covers |
| --- | --- |
| `src/app/tests.rs`, `src/elements/tests/`, `src/window/tests/` | unit tests on the headless platform: the runtime (entities, effects, subscriptions, executors); the elements; the window's layout, frames, keys and mouse (asserting on the `Scene` and on callbacks, never on pixels) |
| `src/text/tests/`, `src/geometry/tests.rs` | the text front against `FixedTextSystem` (layout cache, wrapping, truncation, painting); the geometry types |
| `src/raster/quad_tests.rs` | the quads' span decomposition against the shader formula at every pixel |
| `src/platform/windows/tests.rs` | the Win32 backend on real windows that are never shown, driven by their messages |
| `tests/raster/` | the renderer: the incremental invariant on random scenes, exactness against GPUI's formulas, degenerate input, cache bounds |
| `tests/style/` | style refinement, the `Styled` methods the app calls (`styled.rs` fails to compile if one is missing), the taffy mapping, painting, paths, SVG icons, images, easing |
| `tests/portable_text.rs` | the cosmic-text backend with the bundled Geist fonts (off Windows, or with `portable-text`) |
| `tests/text_dwrite.rs`, `tests/text_dwrite_startup.rs` | the DirectWrite backend (Windows); the start-up one is `#[ignore]`d, its own binary |
| `tests/fonts_added_late.rs` | fonts added after a window has drawn |
| `tests/windows_platform.rs` | the Win32 run loop with no window on screen (Windows) |
| `benches/raster.rs` | the renderer on a synthetic busy frame (`cargo bench -p folo-ui --bench raster`) |
| `examples/win_smoke/`, `examples/winit_smoke.rs`, `examples/text_probe.rs` | real-window checks of the Win32 and winit backends; DirectWrite layout and masks for comparison with GPUI captures (below) |

A test opens a window on the headless platform through `HeadlessApp` (`src/headless_app.rs`): build state with
`update`, feed input with `simulate_input`, move the fake clock with `advance_clock` (`advance_clock_held` while
the test holds background jobs back), pump with `run_until_idle`, then draw with `frame` (the framebuffer) or read
the `scene`. `open` in `src/window/tests/mod.rs` is the smallest
example: a `HeadlessApp` with `HeadlessOptions { size, scale_factor, text: TextBackend::Fixed }`, one window
whose view renders a closure, and its first frame drawn. The app's own tests use the same harness
(`apps/desktop/src/app/tests.rs`). File dialogs answer at once with what `HeadlessPlatform::answer_prompts` set
(the paths picked, or `None`: cancelled, the default); a file dropped on a window is
`PlatformInput::FileDrop` with its paths built by `ExternalPaths::new`.

## Platforms

| Platform | Backend | What is verified |
| --- | --- | --- |
| Windows | Win32 (`platform/windows`), DirectWrite text | built, linted and tested in CI; the app runs and ships here; headless shots within 0.004 % of the GPUI app's pixels (`PARITY.md`); real-window checks with posted messages, off screen and on a private desktop (`win_smoke`; the app's `interact-check.ps1`, `chrome-check.ps1`, `window-check.ps1 -Sandbox`; copy and paste, opt-in with `interact-check.ps1 -Sandbox -Clipboard`, passed once in a run on 2026-11-11); cost against GPUI in `PERF.md` |
| Linux | winit + softbuffer, cosmic-text | CI builds the app, runs every test (all headless) and renders a headless `--shot`; no window has been opened on Linux |
| macOS | winit + softbuffer, cosmic-text | type-checked and linted for `aarch64-apple-darwin` and `x86_64-apple-darwin` by `tools/cross-check.sh` (in CI too); nothing is linked or run on macOS |

The winit backend's shared code (events, keys, the custom caption, presenting) also runs on Windows through the
`winit` feature, which is how it is exercised without a Linux or macOS desktop.

## System appearance

The app follows the system's light / dark appearance through four calls (FOLO Desktop's System theme mode, and its
window frame in the mode it shows):

- `App::window_appearance()`: the system's appearance now (the headless platform: dark, until a test calls
  `HeadlessApp::set_system_appearance`).
- `Window::appearance()`: the same, as that window sees it.
- `Window::on_appearance_changed(cx, f)`: `f` runs after the system's appearance changes (Win32: `WM_SETTINGCHANGE`
  "ImmersiveColorSet"; winit: its theme-changed event; headless: `HeadlessApp::set_system_appearance`, through the
  same platform callback); the window is redrawn either way.
- `Window::set_frame_appearance(Some(appearance) | None)`: draws the system frame (DWM's caption and border on
  Windows) dark or light whatever the system says, or following it again; a no-op on winit, recorded headless
  (`HeadlessPlatform::frame_appearances`).

## Developer switches

| Variable | Effect |
| --- | --- |
| `FOLO_UI_OFFSCREEN=1` | real-window checks on a machine someone is using. Both OS backends (Win32 and winit, on any OS): the app's windows open outside every display, are never activated and take no focus (no taskbar button on Windows), and nothing is launched that would open a window of its own: the file dialogs answer with an error (one with no active window to own it opened centred on the primary display), `open_url` and `reveal_path` do nothing but say so on stderr. That refusal is one gate both backends call (`platform/services.rs`). Win32 only: any other window of the process that shows on a display (a library's dialog) is moved off it, named on stderr. winit does not move such windows, and places its own by position only: where the window system ignores a client's position (Wayland), the compositor decides where they open |
| `FOLO_UI_TRACE=1` | prints `first-render <ms>`, `first-present <ms>` and `first-frame <ms>` to stderr (ms since the process was created), and the app's own points: `trace_point(label)` prints `label <ms>` at once, `trace_at_next_frame(label)` when the next frame is presented (FOLO Desktop: `probe-found`, `probe-answered` (something answered, not as a FOLO server) or `probe-none` at once; after starting `folo serve`, `server-started` at once when the new server said it is listening, `server-in-use` when the data dir's server (already running) is used instead, `server-failed` when the start failed; `first-lists` and `system` at once and each again with `-frame` when it is on screen; `images-drained` at once when the image queue first runs dry (every image asked for so far answered, failures included), and `images-drained-frame` at the next frame presented after it, in which a thumbnail that failed can still show its placeholder) |
| `FOLO_UI_TRACE=frames` | prints one line per frame: the time of each phase (build, layout, prepaint, paint, raster, present) and the primitive and hitbox counts |
| `FOLO_UI_PLATFORM=winit` | on Windows, with the `winit` feature, use the winit backend instead of Win32 |

`Application::new` also reads `--shot <png>` from the process arguments: the app then runs on the headless
platform, settles (queues pumped, animations advanced on the fake clock up to `--shot-time`, default 2200 ms),
writes its first window as a PNG and exits. `--width` / `--height` (logical px, default 1280 x 800), `--scale`
(default 1) and `--text portable` (cosmic-text instead of DirectWrite) set the window. Sizes are checked up
front: finite, at least 1 px, 1 to 16384 device pixels a side (rounded as the framebuffer is), a scale above 0
and at most 8, a `--shot-time` of at most
600000 ms (`--bench-frames` checks the same sizes). An error exits 2 with a message. `apps/desktop/tools/shoot.ps1` uses it.

## Adapted from GPUI

GPUI is Apache-2.0. The API shape is GPUI's by design, and some files adapt GPUI 0.2.2 source (value types,
style fields, keystroke parsing, easing, coverage formulas, the line wrapper, the style-to-taffy mapping, parts
of the Win32 backend). Each says so in its first line and is listed in `NOTICE.md`. The rest (the runtime, the
window pipeline, the elements, the renderer, the text backends, the winit and headless platforms) is written for
this design and matches GPUI's observable behaviour.

Things to know when editing:

- Module names clash with crate names (`image`, `platform::windows`, `platform::winit`): inside the crate write
  `::image::…`, `::windows::…`, `::winit::…` for the crates.
- `AlphaMask`, `ColorMask` and `PathData` take ids from one shared atomic counter; tile hashes use the ids,
  never pixel bytes.
- `FontMetrics.descent` is negative (font units, as GPUI); `LineLayout.descent` is positive.
- A new dependency needs a line in `DESIGN.md` (Decisions) saying why.
