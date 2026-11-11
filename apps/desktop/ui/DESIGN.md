# folo-ui: design

`folo-ui` is FOLO Desktop's own UI framework; it replaced GPUI 0.2.2 under the app. This file is the design
record and the contract every part is built against; when code and this file disagree, fix one of them in the
same change. `README.md` is how to build, test and run it.

## Goals, in order

1. **The app looks and behaves exactly as it does on GPUI 0.2.2.** Same layout to the pixel, same colours, same
   text, same hover / focus / keyboard / scroll / drag / IME behaviour, same window chrome.
2. **Much lighter.** No GPU device for the UI (the GPU and its memory belong to training and inference), a
   fraction of the memory, a window on screen in tens of milliseconds, zero work while nothing changes, a
   small binary and a short build.
3. **Windows, Linux and macOS.** Windows ships first and is where parity is measured; Linux and macOS use the
   same core, renderer and tests behind a different window backend.

## The shape of it

```
 app code (apps/desktop/src)          unchanged apart from `gpui` -> `folo_ui`
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

### Decisions (do not relitigate; raise a problem with evidence instead)

- **Language: Rust.** One toolchain with the app, no FFI seam on the hot path, SIMD through `core::arch` where a
  loop needs it (SSE2 is the x86-64 baseline, NEON the aarch64 one; always keep a scalar path).
- **The public API is GPUI's, for exactly the subset the app uses.** Same names, signatures and semantics,
  re-exported flat from the crate root (and `folo_ui::prelude`). The app is the specification of the surface:
  if `apps/desktop/src` does not use it, it does not exist. GPUI 0.2.2 (the `gpui` crate on crates.io) is the
  specification of behaviour.
- **Rendering is on the CPU, in tiles, and only where the picture changed.** The window keeps one framebuffer.
  Each frame's display list (`Scene`) is binned into 64 px tiles; a tile whose content hash equals last frame's
  is not touched; changed tiles are rasterized (in parallel when there are many) and only their rectangles are
  presented. There is no GPU device, swap chain, atlas texture or shader anywhere in the UI.
- **Layout is taffy, pinned to `=0.9.0`** (the version GPUI 0.2.2 uses), with GPUI's style-to-taffy mapping
  and rounding, so every box lands where it did.
- **Text on Windows is DirectWrite**, used the way GPUI uses it (line layout through `IDWriteTextLayout`,
  glyph masks through `CreateGlyphRunAnalysis`, grayscale, 4x2 subpixel variants, the same contrast and gamma
  correction applied at blend time), so glyphs are identical. **Elsewhere it is cosmic-text**, as GPUI does
  on Linux; that backend also compiles on Windows (feature `portable-text`) so tests are deterministic.
- **Windows backend is plain Win32** (`windows` crate, same version as wgpu-hal's so there is one copy):
  our own window class and message loop, custom title bar through `WM_NCCALCSIZE` / `WM_NCHITTEST`, IME,
  clipboard, file dialogs, per-monitor DPI. Presents with `SetDIBitsToDevice` of the damage rectangles.
- **Linux and macOS backend is winit 0.30 + softbuffer** (plus arboard for the clipboard and rfd for file
  dialogs). It also compiles on Windows (feature `winit`) so its shared code runs on the dev machine.
- **Commodity crates do commodity jobs:** taffy (flexbox maths), resvg/usvg (SVG icons, as GPUI), tiny-skia
  (path coverage, comes with resvg), cosmic-text, winit, softbuffer, arboard, rfd, image, async-task, smallvec.
  Everything else is ours. No new dependency without a line in this file saying why.

### What GPUI code may be reused

GPUI is Apache-2.0. Its API shape is ours by design, and these may be adapted from its source: the value types
(`geometry`, `color`), the `Style` / `StyleRefinement` fields and the `Styled` method list, keystroke parsing,
easing functions, and exact formulas (quad / shadow coverage, glyph contrast and gamma correction, line-wrap
and truncation rules, the style-to-taffy mapping). Every file that adapts GPUI source says so in its first
line (`// Adapted from GPUI (Apache-2.0): <path>`) and is listed in `ui/NOTICE.md`. The engine (runtime,
window pipeline, element and interactivity code, renderer, platforms, text backends) is written for this
design; match GPUI's observable behaviour, do not transplant its modules.

## Crate layout

`apps/desktop/ui/`, crate `folo-ui`, a member of the `apps/desktop` workspace. One crate, in these parts (a
module's doc comment may name its part with an `Owner tag`):

| Files (under `ui/src/`) | Contents |
| --- | --- |
| `geometry/` `color.rs` `shared_string.rs` `util.rs` | `Pixels` `ScaledPixels` `DevicePixels` `Rems` `Length`… `Point` `Size` `Bounds` `Edges` `Corners`, `Hsla` `Rgba` `Background` gradients |
| `input_events.rs` `keystroke.rs` | `PlatformInput` and every event struct, `Modifiers`, `Keystroke` (parse / unparse), `MouseButton`, `ScrollDelta`, `ExternalPaths`, `CursorStyle`, `WindowControlArea` |
| `platform.rs` `platform/` (shared parts) | `Platform`, `PlatformWindow`, `PlatformDisplay`, `PlatformDispatcher`, `PlatformInputHandler`, `WindowOptions` and friends, `ClipboardItem`, `PathPromptOptions`, `Framebuffer` |
| `scene.rs` | `Scene`, primitives, `AlphaMask`, `ColorMask` |
| `text/types.rs` | `Font` `FontId` `FontWeight` `FontStyle` `FontFeatures` `FontMetrics` `GlyphId` `TextRun` `FontRun` `LineLayout` `ShapedRun` `ShapedGlyph` `RenderGlyphParams`, `PlatformTextSystem` |
| `raster/` | the tile renderer, mask / shadow / path / scaled-image caches, its thread pool (bench: `benches/raster.rs`) |
| `text/` (rest) | `TextSystem`, `WindowTextSystem`, line layout cache, `LineWrapper`, truncation, `ShapedLine` / `WrappedLine` and their painting through `GlyphPainter` |
| `text/dwrite/` | DirectWrite `PlatformTextSystem` |
| `text/portable/` | cosmic-text `PlatformTextSystem` |
| `platform/windows/` | Win32 backend |
| `platform/winit/` | winit + softbuffer backend |
| `platform/headless/` `headless_app.rs` | the headless platform and `HeadlessApp` (below) |
| `style/` `styled/` `layout.rs` `path_builder.rs` `assets.rs` `svg.rs` `image.rs` `easing.rs` | `Style`, `StyleRefinement`, `Styled`, `TextStyle`, taffy mapping, `PathBuilder` / `Path`, `AssetSource`, SVG to mask, `RenderImage` |
| `executor.rs` `action.rs` `keymap/` `app/` `view.rs` | the runtime: `Application`, `App`, entities, contexts, effects, tasks, actions, key bindings, `--shot` |
| `window/` `interactive/` `element.rs` `elements/` | the window, its frame pipeline and dispatch, the elements |

Rules:

- Rust 2024, `cargo fmt`, `cargo clippy --all-targets -- -D warnings` clean, **no source file of 1000 lines
  or more** (`tools/file-size.sh` fails CI), `unsafe` only in blocks with a `// SAFETY:` line.
- Public items get a doc comment. Tests are deterministic and quick (about a second each at most).
- Never open a window on the user's screen from a test or a script without being told to. Use the headless
  platform. A real-window check runs off screen and without taking focus (`FOLO_UI_OFFSCREEN=1`, below).

## Differences from GPUI (on purpose)

The API is GPUI's where the app uses it. Below it, these differ by design:

- **Frames.** `PlatformWindow::request_frame(&self)` means "I am dirty, call `on_request_frame` once at the next
  refresh"; calls coalesce, and a backend runs no timer while none is pending.
  `PlatformWindow::present(&mut self, frame, damage)` replaces `draw(scene)`; there is no atlas.
- `Platform::dispatcher()` replaces the executor getters; the runtime builds its executors from it.
  `Platform::open_window(WindowParams)` takes no window handle; `WindowParams` has `app_id`. `Platform` has no
  `on_quit`, menus, credentials, keyboard layouts or screen capture.
- File prompts return `platform::oneshot::Receiver<T>` (awaiting gives `Result<T, Canceled>`; also `try_recv`).
- Scene primitives carry `clip: Bounds<ScaledPixels>` (no `ContentMask`, no draw order). `GlyphBlend` has
  `apply: bool`; `GlyphBlend::IDENTITY` / `none()` mean no correction.
- `ClipboardItem` holds strings only. `PathBuilder::build` returns a `BuiltPath` (logical px; `Path` is the
  scene primitive).
- Dropped: serde, schemars, derive_more, refineable (operators written by hand), string parsing of lengths and
  colours, grid placement.
- The text element truncates a copy of its runs on every measure (GPUI 0.2.2 shortened them in place, which
  sliced multi-byte characters on a later measure and panicked; the app carried a patched GPUI for it).

## Units and colour

- App code works in logical `Pixels`; the scene is in device pixels (`ScaledPixels`, `f32`); a window has one
  scale factor (1.0, 1.25, 1.5, 2.0 …). Scale exactly where GPUI scales and round exactly where GPUI rounds.
- Colours are `Hsla` in the API and are converted with GPUI's `From<Hsla> for Rgba`. Blending is plain
  source-over **in gamma space on 8-bit channels**, one primitive at a time, as GPUI's DX11 renderer does
  (`B8G8R8A8_UNORM` target, `SRC_ALPHA / INV_SRC_ALPHA`): after each primitive the destination is 8-bit again.
- `Framebuffer`: `width`, `height` (device px), `Vec<u32>` rows top-down, pixel `0x00RRGGBB`.

## Scene (the seam between the window and the renderer)

Painter's order: a primitive is drawn over every earlier one. Each primitive carries its own clip rectangle
(the content mask, already intersected) in device pixels.

- `Quad { bounds, clip, background: Background, border_color, corner_radii, border_widths, border_style }`
- `Shadow { bounds, clip, corner_radii, color, blur_radius }`
- `MonoSprite { bounds, clip, color, mask: Arc<AlphaMask>, kind: Glyph | Svg }` (a glyph gets the text
  backend's contrast / gamma correction, an SVG mask is plain alpha)
- `ColorSprite { bounds, clip, mask: Arc<ColorMask>, opacity }` (colour emoji)
- `Image { bounds, clip, corner_radii, image: Arc<RenderImage>, frame, grayscale, opacity }`
- `Path { bounds, clip, path: Arc<PathData>, paint: Background }` (fill or stroke, already resolved to
  contours + fill rule or stroke style by `PathBuilder`)
- `Underline { … }` as GPUI's (straight and wavy)

Masks and images are identified by a process-unique `id: u64` assigned at creation; tile hashes use ids, never
pixel bytes. `Scene::clear()` keeps capacity. Element opacity is applied to colours at paint time (as GPUI).

## Renderer contract (`raster`)

```rust
pub struct Renderer { /* tile hashes of the last frame, caches, thread pool */ }
impl Renderer {
    pub fn new() -> Self;
    /// Makes `frame` show `scene`. Returns the rectangles that changed (device px, tile-aligned, merged).
    /// `frame` must be the buffer the previous call drew, or the call must follow `invalidate`.
    pub fn render(&mut self, scene: &Scene, glyph: GlyphBlend, frame: &mut Framebuffer) -> Vec<Bounds<DevicePixels>>;
    pub fn invalidate(&mut self);            // next frame repaints everything (resize, new buffer)
}
```

Beyond the contract, for tests, benches and the window's frame timings: `Renderer::with_threads(n)` (1 draws
on the calling thread; `new` uses up to `min(8, cores)`), `threads()`, `stats() -> &RenderStats` (what the last
frame did), `cache_bytes()` and `set_cache_budget(bytes)` (default about 108 MB in all, of which a frame
usually needs a small part); `raster::MaskCache` (the window's glyph and icon masks), `raster::TILE_SIZE`, and
`raster::debug::{write_png, read_png}` (a `Framebuffer` to and from a PNG; `--shot` writes through it).

- **The invariant, tested with random scenes:** rendering frame A then frame B into the same buffer gives
  the same bytes as rendering B into a fresh buffer, and the same bytes for every thread count.
- Tiles are 64x64 device px. Bin primitives by `bounds ∩ clip`; hash each tile's ordered primitive
  signatures; skip equal tiles; drop primitives hidden behind a later opaque quad that covers the tile.
- Few dirty tiles are drawn on the calling thread; many go to a small dedicated pool (not rayon's global
  pool: it belongs to the engine). The pool is created the first time it is needed.
- Quads, borders (solid and dashed), rounded corners and shadows use GPUI's coverage formulas
  (GPUI 0.2.2's `src/platform/windows/shaders.hlsl`) on edge pixels and solid spans inside. Shadows are cached
  as masks. Images are sampled as a linear GPU sampler would (texel centres), with a cache of resampled
  copies for images drawn at the same size frame after frame. Path coverage is cached by geometry.
- Caches are bounded in bytes and drop what recent frames did not use. Memory is a goal, not an afterthought.
- Targets on the dev machine (i7-14700KF), release build: a full 1920x1200 frame of a busy screen in under
  4 ms; a frame that changes one small region in under 0.3 ms including binning and hashing.

## Text contract

- `PlatformTextSystem` is GPUI's trait (add_fonts, font_id, font_metrics, typographic_bounds, advance,
  glyph_for_char, glyph_raster_bounds, rasterize_glyph, layout_line) plus `glyph_blend() -> GlyphBlend`
  (the contrast / gamma parameters the renderer applies to glyph masks; the portable backend gives GPUI's
  Linux renderer defaults, gamma 1.8 and contrast 1.0, as `shaders.wgsl` applies them).
- `TextSystem` (app-wide: font ids, metrics, fallback stack) and `WindowTextSystem` (per window: the line
  layout cache that keeps what the previous frame used) expose what the app and the text element call:
  `shape_line`, `shape_text`, `layout_line`, `line_wrapper`, `ShapedLine::{paint, width, x_for_index,
  index_for_x, closest_index_for_x, len}`, `WrappedLine`. Wrapping and truncation follow GPUI's rules exactly
  (they decide where every line of the UI breaks).
- Painting a line goes through `trait GlyphPainter` (paint_glyph, paint_emoji, paint_underline,
  paint_strikethrough, paint_background); the window implements it, so `text/` does not depend on `window/`.
- Fonts come from `add_fonts` (the app bundles Geist and Geist Mono). A glyph the bundled fonts lack falls
  back to system fonts as GPUI's backend for that platform does, looked up lazily, never enumerated at start.

## Platform contract

- `Platform`: run loop, quit, displays, open_window, clipboard, prompt_for_paths / prompt_for_new_path,
  open_url, reveal_path, cursor style, dispatcher, text system, activate.
- `PlatformWindow`: bounds, window_bounds, content_size, scale_factor, display, mouse_position, modifiers,
  is_maximized / is_fullscreen / is_active, minimize, zoom, toggle_fullscreen, activate, set_title,
  set / take input handler, update_ime_position, the `on_*` callbacks GPUI has (request_frame, input, resize,
  moved, active_status_change, should_close, close, hit_test_window_control), and
  `present(&mut self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>])` in place of `draw(scene)`.
- `PlatformDispatcher`: `dispatch` (background, a pool that grows on demand and lets idle threads exit),
  `dispatch_on_main_thread`, `dispatch_after`, `is_main_thread`.
- **Frames.** The window draws when it is dirty, at most once per display refresh; while a frame asks for
  another (`request_animation_frame`) the backend keeps a refresh-rate tick, otherwise there is no timer and
  the process sleeps in the message loop. Resizing repaints synchronously (no black edges).
- **Headless** (`platform/headless/`): no OS window; a fixed size and scale; synthetic input; renders to a
  `Framebuffer`. `HeadlessApp` (`headless_app.rs`) drives it: `update` changes state, `run_until_idle` pumps
  the main queue and waits for background jobs, `advance_clock` moves the fake clock (timers and animations see
  only that clock), `simulate_input` delivers events, `frame` draws a dirty window and returns its pixels,
  `settle` / `settle_advancing` draw until the window stops asking for frames. A frame is a function of the
  state and the time the harness advanced to. It is what tests, `folo-desktop --shot`, and CI use.
- **`--shot <png>`** (`app/shot.rs`): `Application::new` sees it in the process arguments and runs the app on
  the headless platform instead (`--width`, `--height`, `--scale`, `--text portable`, `--shot-time`), settles
  its first window (queues pumped, then 50 ms steps of the fake clock until it stops asking for frames or
  `--shot-time` has passed, default 2200 ms) and writes the PNG. The app needs no code for it.
- `FOLO_UI_OFFSCREEN=1`: a real window opens off every display, is never activated and takes no focus
  (developer checks on a machine someone is using).

## How it is verified

- **The app builds on folo-ui alone.** `apps/desktop` has no `gpui` dependency and no vendored copy; the
  only GPUI left is the attribution in `NOTICE.md` and the files' first lines.
- **Pixels.** Every `--scenario` rendered headless at the GPUI captures' sizes and scale (1280 x 800 for all,
  960 x 620 and 1680 x 1050 for the densest, at 1.5) is compared with GPUI 0.2.2's captures, region by
  region: boxes, colours and text at the same pixels. The GPUI captures and the comparison were made outside
  the repo, on the dev PC. (Placeholder: the parity branch adds its tool and the residual differences here.)
- **Behaviour.** The crate's tests drive the headless platform with synthetic input: layout, text wrapping and
  truncation, hover, focus and keyboard, scrolling, drag and drop, IME ranges, animations on the fake clock,
  the renderer's invariant on random scenes and its exactness against GPUI's formulas. The app's own tests run
  on it too.
- **The real window** (Windows): `win_smoke` and the app's `tools/chrome-check.ps1` drive an off-screen window
  with posted messages; `tools/window-check.ps1` drives it with real input when someone can leave the machine
  alone.
- **Cost.** `tools/perf.ps1` measures time to first frame, working set and private bytes, GPU memory, idle CPU
  and CPU over a live run, off screen; compare two builds back to back on the same machine.
- **Platforms.** CI runs fmt, clippy and the tests on Windows and Linux, a headless `--shot` on Linux, and
  `tools/cross-check.sh` (check and clippy of the crate for Linux and both macOS targets). Nothing runs on
  macOS.
