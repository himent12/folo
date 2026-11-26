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
  dialogs). It also compiles on Windows (feature `winit`) so its shared code runs on a Windows development machine too.
- **Commodity crates do commodity jobs:** taffy (flexbox maths), resvg/usvg (SVG icons, as GPUI), tiny-skia
  (comes with resvg), lyon `=1.0.19` (path tessellation and dashing, the versions GPUI 0.2.2 uses, so a path
  covers the same samples), cosmic-text, winit, softbuffer, arboard, rfd, image, async-task, smallvec,
  unicode-segmentation (UAX #29 extended grapheme clusters so text wrapping and truncation never split a base
  character from its combining marks).
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
| `style/` `styled/` `layout.rs` `path_builder.rs` `path_cache.rs` `assets.rs` `svg.rs` `image.rs` `easing.rs` | `Style`, `StyleRefinement`, `Styled`, `TextStyle`, taffy mapping, `PathBuilder` / `Path` (and the built paths kept by input), `AssetSource`, SVG to mask, `RenderImage` |
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
- The line layout cache's key for a wrapped line holds the line limit (`line_clamp`), so a text drawn
  with two clamps, in one frame or one after the other, is wrapped to each (GPUI 0.2.2's key left it
  out: the first clamp asked for a text, font and width was reused, also in later frames).
- Fonts added after text was drawn are used from the next frame: `TextSystem::add_fonts` forgets font
  resolutions (the backends forget their family lookups too) and moves a font generation on, which the
  line layout caches' keys, the line wrapper pool and layout reuse carry (GPUI 0.2.2 kept the
  resolutions, so a family drawn before its font was added stayed on its fallback). Font ids handed
  out stay valid: an id always names the same face, so metrics, raster bounds and masks keyed by id
  stay right.
- A shown tooltip hides on the mouse move that leaves its element (GPUI 0.2.2 waited for the next
  prepaint, so a tooltip on an element without a hover style stayed up), and a tooltip's timer and
  visibility callback hold its state weakly (GPUI's cycle leaked the state and the tooltip view once the
  element went away).
- Text wrapping and truncation (`LineLayout::compute_wrap_boundaries`, `LineWrapper::wrap_line`,
  `LineWrapper::truncate_line`) respect Unicode extended grapheme cluster boundaries (and multi-glyph shaped
  clusters), so a combining accent or modifier never wraps or truncates apart from its base character
  (GPUI 0.2.2 broke per scalar value / per glyph).

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
- `MonoSprite { bounds, clip, color, mask: Arc<AlphaMask>, kind: Glyph | Svg }` (both get the text
  backend's contrast / gamma correction, as GPUI's one monochrome sprite shader applies it; an SVG mask
  is sampled linearly over the bounds, a glyph is drawn 1:1)
- `ColorSprite { bounds, clip, mask: Arc<ColorMask>, opacity }` (colour emoji)
- `Image { bounds, clip, corner_radii, image: Arc<RenderImage>, frame, grayscale, opacity }`
- `Path { bounds, clip, path: Arc<PathData>, paint: Background }` (`PathBuilder` makes lyon's triangles,
  as GPUI; contours with a fill rule or stroke style are also accepted)
- `Underline { … }` as GPUI's (straight and wavy)

Masks and images are identified by a process-unique `id: u64` assigned at creation; tile hashes use ids, never
pixel bytes. `Scene::clear()` keeps capacity. Element opacity is applied to colours at paint time (as GPUI),
except inside an opacity group (below), where it is applied right after paint by the same multiplications.

## Frames: opacity groups, layout reuse and built paths

A full frame is GPUI's pipeline (`window/draw.rs`): render the root view, lay it out with taffy, prepaint,
paint into a `Scene`, then rasterize and present the damage. Three things keep it cheap without changing a pixel:

- **Opacity groups** (`ui/src/opacity.rs`, `window/paint_animation.rs`, `elements/animation.rs`). An element
  whose only animated property is its opacity uses `AnimationExt::with_opacity_animation` (the app's breathing
  status dot, its appear fade and its skeleton shimmer). A full frame paints it once inside an opacity group:
  its primitives at their base colours, the scene recording which span of primitives belongs to which group
  (a group nests in the group it starts inside; an element opacity set inside a group becomes a nested group).
  After paint, `Scene::seal_opacity_groups` keeps the base copies and writes the faded primitives, each colour
  multiplied by the group's opacity (`parent * outer * factor`) in the same order as element opacity is at paint
  time, so the pixels are the same bytes. The element registers a paint animation (group, start time,
  `Animation`, its closure). When a refresh comes and the window is not dirty, `tick_paint_animations`
  computes every running animation's factor; only if one moved by at least 1/255 (or finished) does it set the
  factors and re-fade the spans from the base copies (`apply_opacity_groups`), and the renderer repaints only
  the tiles those primitives touch. No render, layout or paint runs, and every frame shown is the picture of
  one moment of the animations. A full redraw for any other reason paints the groups again; the elements keep
  their start times in element state, so the animations go on where they were. The window asks for refreshes
  while a paint animation runs (a repeating one never finishes). Tests: `window/tests` and the app's
  `ui/motion_tests.rs` (every frame equals the `with_animation` version's frame at that moment).
- **Layout reuse** (`window/layout_tree.rs`). The elements still add their layout nodes every frame (children
  before parents), but into our own node storage running taffy's algorithms, which keeps what the previous frame
  computed. A new node whose style, kind, measure key (the text element's text and font) and children equal a
  node of the previous frame (at the same scale, with no fonts added since) takes that node's place and keeps
  its records: each layout request it received last frame with its answer. While this frame asks the same
  requests in the same order, they are answered from the records and nothing under the node runs; its
  measured leaves replay last frame's measure calls in order, so their state (the text element's shaped
  lines) is what a fresh layout leaves. The first request that differs makes the node diverge: its subtree
  forgets everything and is laid out from scratch, the requests it already answered included. Every result,
  every measure call and every final position is therefore exactly what laying out a fresh tree gives
  (GPUI's behaviour); only the cost differs. Nodes measured without a key are never reused.
- **Built paths** (`path_cache.rs`). `PathBuilder::build` (lyon's tessellation) is a pure function of the
  builder, so a builder equal word for word to one built recently (its contours, transform, dashes and options,
  every float compared by its bits) gets a copy of that result instead of being tessellated again: the live run
  page's chart lines were about a quarter of its full frame. Per thread, the cache's stored path entries and
  lookup table stay at or under 8 MB after every call: keys without spare capacity, the triangles, and its
  table at the size the table has allocated (all its slots, not only the used ones); a separate reusable
  key-serialization scratch buffer per thread is retained up to 64 KiB (`16,384` `u32` words) so cache hits
  avoid allocating a temporary key vector. While the table grows, the old one is held too for the
  copy. Over 8 MB, one pass drops the least recently used until the rest, with a table for a quarter more
  entries than are left (room for the table's next growth, so the next inserts don't go over again), is down
  to 6 MB. A path whose entry alone would be over 1 MB is turned away before it is copied. An error is never
  kept. A path built while the thread is torn down, after its cache is gone, is built without it. Tests:
  `path_cache.rs` (a hit equals a fresh build, any change in the input builds anew, what is held is what is
  counted, the allocator's count of many tiny paths stays within the count and the budget, an oversized path
  is not copied, one big path evicts in one pass, a cache of small paths is not cut on every insert, a path
  built after the cache is gone, the budget holds).
- **Zero-allocation steady-frame paths**. Steady frames also avoid transient heap traffic across the pipeline:
  two-generation `GlobalElementId` interning (`window/frame.rs`), per-frame `Interactivity` style reuse
  (`window/mod.rs`, `interactive/phases.rs`), borrowed font/SVG/glyph cache lookups (`text/`, `raster/cache.rs`),
  throttled idle cache sweeps (`raster/cache.rs`), early clip culling and precomputed clipped hitboxes
  (`style/paint.rs`, `text/paint.rs`, `window/element_api.rs`), pooled `UniformList` row scratch buffers and
  clamped wheel scroll no-ops (`elements/uniform_list.rs`, `interactive/listeners.rs`), and fast 1:1 / integer
  opaque bilinear image resampling and SIMD span fills (`raster/image.rs`, `raster/blend.rs`).

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
  as masks. Images are sampled as a linear GPU sampler would (texel centres; past the image's edge GPUI's
  zeroed atlas), with a cache of resampled copies for images drawn at the same size frame after frame.
  Paths are sampled as GPUI's 4x MSAA path target (standard pattern, top-left rule, 1/256 px snapping,
  per-sample clip and overlap counts); their coverage is cached by geometry. Clip edges follow the
  top-left rule too. `ui/PARITY.md` lists every rule and what still differs.
- Caches are bounded in bytes and drop what recent frames did not use. Memory is a goal, not an afterthought.
- Targets on our test machine (an i7-14700KF), release build: a full 1920x1200 frame of a busy screen in under
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
  open_url, reveal_path, cursor style, dispatcher, text system, activate, window_appearance (the system's light /
  dark appearance; the headless platform reports dark until a test changes it with `set_appearance`, which runs each
  window's `on_appearance_changed` callback).
- `PlatformWindow`: bounds, window_bounds, content_size, scale_factor, display, mouse_position, modifiers,
  is_maximized / is_fullscreen / is_active, minimize, zoom, toggle_fullscreen, activate, set_title,
  set / take input handler, update_ime_position, the `on_*` callbacks GPUI has (request_frame, input, resize,
  moved, active_status_change, should_close, close, hit_test_window_control, appearance_changed), appearance,
  `set_frame_appearance(Option<WindowAppearance>)` (a default no-op; Win32 draws DWM's frame dark or light and keeps
  that override across `WM_SETTINGCHANGE`), and
  `present(&mut self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>])` in place of `draw(scene)`.
- **A cancelled press.** When a button pressed in the window will never be released there (the mouse
  capture went elsewhere, the window was deactivated), the backend reports a mouse up outside the window,
  where a real release lands when the button comes up elsewhere: it ends drags and selections
  (`on_mouse_up_out`, the pressed style) and clicks and drops nothing, even with the cursor still over the
  element pressed.
- `PlatformDispatcher`: `dispatch` (background, a pool that grows on demand and lets idle threads exit),
  `dispatch_on_main_thread`, `dispatch_after`, `is_main_thread`. When the loop ends the backend shuts its
  dispatcher down on the main thread: queued runnables are dropped, the foreground tasks that are waiting
  have their futures dropped there too (a foreground future lives in its thread's registry, and its
  runnable holds only its ids: `executor.rs`), later runnables are dropped on whatever thread they
  arrive, and the pool and timer threads exit.
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
  (developer checks on a machine someone is using). Nothing else of the process shows either: on Windows
  the file dialogs answer with an error, `open_url` and `reveal_path` do nothing (both say so on stderr),
  and any other window of the process that shows or moves onto a display is moved off at once (a
  WinEvent hook, `platform/windows/offscreen.rs`).

## How it is verified

- **The app builds on folo-ui alone.** `apps/desktop` has no `gpui` dependency and no vendored copy. GPUI is
  named only as the specification (comments saying which GPUI rule or formula a piece matches) and in the
  attribution (`NOTICE.md` and the adapted files' first lines).
- **Pixels.** Every `--scenario` rendered headless at the GPUI captures' sizes and scale (1280 x 800 for all,
  960 x 620 and 1680 x 1050 for the densest, at 1.5) is compared with GPUI 0.2.2's captures, region by
  region: boxes, colours and text at the same pixels. The GPUI captures and the comparison were made outside
  the repo, on our test machine. `apps/desktop/tools/shoot-all.ps1` renders the 98 shots and `tools/parity.py`
  compares the trees (`ui/PARITY.md`: how, every rule, and every residual difference with its cause). Result,
  as the percentage of pixels whose largest channel difference exceeds 8 of 255: worst pair 0.0040 %, median
  0 %, mean 0.0001 %; 71 of 98 pairs have no such pixel, and 62 none above 2 levels. Against GPUI captures
  retaken without the old capture script's GDI+ artifact (it blackened every RGB(13, 11, 12) pixel), 76 and 68.
  The largest single difference (102 levels, one pixel) is in chart paths (GPUI resolves overlapping paths per
  MSAA sample). The rest is the breathing status dot's phase, path edge
  samples, image-atlas neighbours in thumbnails, and 1 level on glyph and icon edges.
- **Behaviour.** The crate's tests drive the headless platform with synthetic input: layout, text wrapping and
  truncation, hover, focus and keyboard, scrolling, drag and drop, IME ranges, animations on the fake clock,
  the renderer's invariant on random scenes and its exactness against GPUI's formulas. The app's own tests run
  on it too.
- **The real window** (Windows): `win_smoke`, and the app's `tools/interact-check.ps1` (its captures equal the
  headless shot of the same scenario; hover, clicks, keys, focus, scrolling, controls, dialogs, resizing) and
  `tools/chrome-check.ps1` (the caption areas' hit test and the caption buttons), drive a real window with
  posted messages, off screen (`FOLO_UI_OFFSCREEN=1`) or on a private desktop nobody sees
  (`tools/sandbox-run.ps1`); `tools/window-check.ps1 -Sandbox` drives the whole chrome there (maximize, drags,
  full screen, the remembered window). Without `-Sandbox`, window-check uses real input on the screen.
- **Cost.** `tools/perf.ps1` measures time to first frame, working set and private bytes, GPU memory, idle CPU
  and CPU over a live run, off screen; compare two builds back to back on the same machine. `ui/PERF.md` has
  the numbers against the GPUI build, the frame bench (`tools/bench-frames.sh`) and the renderer bench.
- **Platforms.** CI runs clippy and the tests on Windows and Linux, fmt on Windows, a headless `--shot` on Linux, and
  `tools/cross-check.sh` (check and clippy of the crate for Linux and both macOS targets). Nothing runs on
  macOS.
