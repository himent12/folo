# folo-ui

FOLO Desktop's own UI framework. It has GPUI's API for exactly the subset the app uses (same names, signatures
and behaviour, so the app reads as it did on GPUI 0.2.2), and its own engine: it renders on the CPU, in 64 px
tiles, and redraws only the tiles that changed. The UI uses no GPU device, so the GPU and its memory stay free
for training and inference.

`DESIGN.md` is the design record: the decisions, the contracts between the parts, and how the port is verified.
This file is how to work on the crate.

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
| `geometry/`, `color.rs`, `shared_string.rs`, `input_events.rs`, `keystroke.rs` | value types, events, keystrokes |
| `style/`, `styled/`, `layout.rs`, `path_builder.rs`, `svg.rs`, `image.rs`, `easing.rs`, `assets.rs` | `Style` and its refinement, the `Styled` methods, the mapping to taffy, paths, SVG icons, images |
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
fresh layout's. And an element that only animates its opacity (`with_opacity_animation`: the breathing status
dot, fades, shimmers) is painted once inside an opacity group (`opacity.rs`, `window/paint_animation.rs`); at
each refresh the window re-fades that group's primitives in the scene and repaints only their tiles, with no
render, layout or paint.

## Build, test, bench

From `apps/desktop` (the crate is a member of that workspace). On the dev machine, set `CARGO_TARGET_DIR` and
`FOLO_TEST_TMP` off C: first (`AGENTS.md`).

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
  Windows it lets that backend's shared code build and run on the dev machine (`FOLO_UI_PLATFORM=winit`).
- `test-support`: exposes `FixedTextSystem` (`text/types.rs`), a fake text backend with fixed metrics, to other
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
cargo run -p folo-ui --example winit_smoke --features winit,portable-text   # the winit backend
cargo run -p folo-ui --example text_probe -- <lines.txt> <out-dir> [scale]  # DirectWrite layout and masks
```

## Platforms

| Platform | Backend | What is verified |
| --- | --- | --- |
| Windows | Win32 (`platform/windows`), DirectWrite text | built, linted and tested in CI; the app runs and ships here; headless shots within 0.004 % of the GPUI app's pixels (`PARITY.md`); real-window checks with posted messages, off screen and on a private desktop (`win_smoke`; the app's `interact-check.ps1`, `chrome-check.ps1`, `window-check.ps1 -Sandbox`, copy and paste included); cost against GPUI in `PERF.md` |
| Linux | winit + softbuffer, cosmic-text | CI builds the app, runs every test (all headless) and renders a headless `--shot`; no window has been opened on Linux |
| macOS | winit + softbuffer, cosmic-text | type-checked and linted for `aarch64-apple-darwin` and `x86_64-apple-darwin` by `tools/cross-check.sh` (in CI too); nothing is linked or run on macOS |

The winit backend's shared code (events, keys, the custom caption, presenting) also runs on Windows through the
`winit` feature, which is how it is exercised without a Linux or macOS desktop.

## Developer switches

| Variable | Effect |
| --- | --- |
| `FOLO_UI_OFFSCREEN=1` | windows open outside every display, are never activated and take no focus (no taskbar button on Windows): real-window checks on a machine someone is using |
| `FOLO_UI_TRACE=1` | prints `first-render <ms>`, `first-present <ms>` and `first-frame <ms>` to stderr (ms since the process was created) |
| `FOLO_UI_TRACE=frames` | prints one line per frame: the time of each phase (build, layout, prepaint, paint, raster, present) and the primitive and hitbox counts |
| `FOLO_UI_PLATFORM=winit` | on Windows, with the `winit` feature, use the winit backend instead of Win32 |

`Application::new` also reads `--shot <png>` from the process arguments: the app then runs on the headless
platform, settles (queues pumped, animations advanced on the fake clock up to `--shot-time`, default 2200 ms),
writes its first window as a PNG and exits. `--width` / `--height` (logical px, default 1280 x 800), `--scale`
(default 1) and `--text portable` (cosmic-text instead of DirectWrite) set the window. An error exits 2 with a
message. `apps/desktop/tools/shoot.ps1` uses it.

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
