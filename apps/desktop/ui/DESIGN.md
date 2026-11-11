# folo-ui: design

`folo-ui` is FOLO Desktop's own UI framework. It replaces GPUI. This file is the contract every part is
built against; when code and this file disagree, fix one of them in the same change.

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
  if `apps/desktop/src` does not use it, it does not exist. GPUI 0.2.2 (`apps/desktop/vendor/gpui`, in the tree
  until the port lands) is the specification of behaviour.
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
  (comes with resvg), lyon `=1.0.19` (path tessellation and dashing, the versions GPUI 0.2.2 uses, so a path
  covers the same samples), cosmic-text, winit, softbuffer, arboard, rfd, image, async-task, smallvec.
  Everything else is ours. No new dependency without a line in this file saying why.

### What GPUI code may be reused

GPUI is Apache-2.0. Its API shape is ours by design, and these may be adapted from its source: the value types
(`geometry`, `color`), the `Style` / `StyleRefinement` fields and the `Styled` method list, keystroke parsing,
easing functions, and exact formulas (quad / shadow coverage, glyph contrast and gamma correction, line-wrap
and truncation rules, the style-to-taffy mapping). Every file that adapts GPUI source says so in its first
line (`// Adapted from GPUI (Apache-2.0): <path>`) and is listed in `ui/NOTICE.md`. The engine (runtime,
window pipeline, element and interactivity code, renderer, platforms, text backends) is written for this
design; match GPUI's observable behaviour, do not transplant its modules.

## Crate layout and owners

`apps/desktop/ui/`, crate `folo-ui`, a member of the `apps/desktop` workspace. One crate; each work package
owns the files listed and touches others only to fix a real integration problem (say so in your report).

| Files (under `ui/src/`) | Package | Contents |
| --- | --- | --- |
| `geometry.rs` `color.rs` `shared_string.rs` `util.rs` | foundation | `Pixels` `ScaledPixels` `DevicePixels` `Rems` `Length`… `Point` `Size` `Bounds` `Edges` `Corners`, `Hsla` `Rgba` `Background` gradients |
| `input_events.rs` `keystroke.rs` | foundation | `PlatformInput` and every event struct, `Modifiers`, `Keystroke` (parse / unparse), `MouseButton`, `ScrollDelta`, `ExternalPaths`, `CursorStyle`, `WindowControlArea` |
| `platform.rs` | foundation | `Platform`, `PlatformWindow`, `PlatformDisplay`, `PlatformDispatcher`, `PlatformInputHandler`, `WindowOptions` and friends, `ClipboardItem`, `PathPromptOptions`, `Framebuffer` |
| `scene.rs` | foundation (raster may extend) | `Scene`, primitives, `AlphaMask`, `ColorMask` |
| `text/types.rs` | foundation | `Font` `FontId` `FontWeight` `FontStyle` `FontFeatures` `FontMetrics` `GlyphId` `TextRun` `FontRun` `LineLayout` `ShapedRun` `ShapedGlyph` `RenderGlyphParams`, `PlatformTextSystem` |
| `raster/` | raster | the tile renderer, mask / shadow / path / scaled-image caches, benches |
| `text/` (rest) | text | `TextSystem`, `WindowTextSystem`, line layout cache, `LineWrapper`, truncation, `ShapedLine` / `WrappedLine` and their painting through `GlyphPainter` |
| `text/dwrite/` | text | DirectWrite `PlatformTextSystem` |
| `text/portable/` | unix | cosmic-text `PlatformTextSystem` |
| `platform/windows/` | windows | Win32 backend |
| `platform/winit/` | unix | winit + softbuffer backend |
| `style.rs` `styled.rs` `layout.rs` `path_builder.rs` `assets.rs` `svg.rs` `image.rs` `easing.rs` | style | `Style`, `StyleRefinement`, `Styled`, `TextStyle`, taffy mapping, `PathBuilder` / `Path`, `AssetSource`, SVG to mask, `RenderImage` |
| everything else: `executor.rs` `action.rs` `keymap/` `app/` `window/` `element.rs` `elements/` `interactive.rs` `platform/headless.rs` … | core | the runtime, the window, the elements |

Rules for every package:

- Rust 2024, `cargo fmt`, `cargo clippy --all-targets -- -D warnings` clean, **no source file of 1000 lines
  or more** (`tools/file-size.sh` fails CI), `unsafe` only in blocks with a `// SAFETY:` line.
- Public items get a doc comment. Tests are deterministic and quick (about a second each at most).
- Build output goes to `D:/folo-build/<your dir>` (`CARGO_TARGET_DIR`), never under `C:`.
- Never open a window on the user's screen from a test or a script without being told to. Use the headless
  platform. A real-window check runs off screen and without taking focus (`FOLO_UI_OFFSCREEN=1`, below).

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

- **The invariant, tested with random scenes:** rendering frame A then frame B into the same buffer gives
  the same bytes as rendering B into a fresh buffer, and the same bytes for every thread count.
- Tiles are 64x64 device px. Bin primitives by `bounds ∩ clip`; hash each tile's ordered primitive
  signatures; skip equal tiles; drop primitives hidden behind a later opaque quad that covers the tile.
- Few dirty tiles are drawn on the calling thread; many go to a small dedicated pool (not rayon's global
  pool: it belongs to the engine). The pool is created the first time it is needed.
- Quads, borders (solid and dashed), rounded corners and shadows use GPUI's coverage formulas
  (`vendor/gpui/src/platform/windows/shaders.hlsl`) on edge pixels and solid spans inside. Shadows are cached
  as masks. Images are sampled as a linear GPU sampler would (texel centres; past the image's edge GPUI's
  zeroed atlas), with a cache of resampled copies for images drawn at the same size frame after frame.
  Paths are sampled as GPUI's 4x MSAA path target (standard pattern, top-left rule, 1/256 px snapping,
  per-sample clip and overlap counts); their coverage is cached by geometry. Clip edges follow the
  top-left rule too. `ui/PARITY.md` lists every rule and what still differs.
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
- **Headless** (`platform/headless.rs`): no OS window; a fixed size and scale; synthetic input; renders to a
  `Framebuffer`. It is what tests, `folo-desktop --shot`, and CI use.
- `FOLO_UI_OFFSCREEN=1`: a real window opens off every display, is never activated and takes no focus
  (developer checks on a machine someone is using).

## Acceptance (the whole port)

- `apps/desktop` builds against `folo-ui` with no `gpui` anywhere, and `vendor/gpui` is gone.
- Every `--scenario` rendered headless at the baseline's size and scale matches the GPUI capture: boxes,
  colours and text at the same pixels; any residual difference is listed and explained.
- `tools/chrome-check.ps1` and the window checks pass on the real window.
- Measured against the GPUI build on the same machine: time to first frame, working set and private bytes,
  GPU memory, idle CPU, CPU time over a scripted live run, binary size, clean build time.
- `cargo fmt`, clippy and tests are green on Windows and Linux; the crate checks for macOS.
