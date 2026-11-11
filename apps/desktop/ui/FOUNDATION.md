# folo-ui foundation: what exists and how to build it

Working notes for the packages built on the foundation (`DESIGN.md` is the contract; this file is the facts).
It is removed or folded into the README before the port lands.

## Cargo features

- On non-Windows targets cosmic-text, winit, softbuffer, arboard and rfd are plain dependencies: always built.
- On Windows the same crates are optional: `portable-text = ["dep:cosmic-text"]`,
  `winit = ["dep:winit", "dep:softbuffer", "dep:arboard", "dep:rfd"]`.
- `test-support` exposes `FixedTextSystem` (a deterministic fake text backend) to other crates; unit tests
  inside the crate get it through `cfg(test)`.
- Module gates: `#[cfg(any(not(windows), feature = "winit"))]` for `platform/winit`, the same with
  `"portable-text"` for `text/portable`; `platform/windows` and `text/dwrite` are `#[cfg(windows)]`.
- `windows` is 0.62.2, the copy wgpu-hal and gpu-allocator already use.

Commands, from `apps/desktop`:

```
cargo check -p folo-ui
cargo clippy -p folo-ui --all-targets -- -D warnings
cargo fmt -p folo-ui -- --check
cargo test -p folo-ui
cargo clippy -p folo-ui --all-targets --features portable-text,winit,test-support -- -D warnings
cargo test -p folo-ui --features portable-text,winit
```

## Differences from GPUI in the foundation (on purpose)

- `PlatformWindow::request_frame(&self)`: "I am dirty, call `on_request_frame` once at the next refresh".
  Calls coalesce. A backend runs no timer while none is pending.
- `PlatformWindow::present(&mut self, frame, damage)` replaces `draw(scene)`; there is no atlas.
- `Platform::dispatcher()` replaces the executor getters; the core builds its executors from it.
- `Platform::open_window(WindowParams)` takes no window handle; `WindowParams` has `app_id`.
- `Platform` has no `on_quit`, menus, credentials, keyboard layouts or screen capture.
- File prompts return `platform::oneshot::Receiver<T>` (ours; awaiting gives `Result<T, Canceled>`, also
  `try_recv`; `oneshot::channel()` makes a pair).
- Scene primitives carry `clip: Bounds<ScaledPixels>` (no `ContentMask`, no draw order). `GlyphBlend` has
  `apply: bool`; `GlyphBlend::IDENTITY` / `none()` mean no correction.
- `ClipboardItem` holds strings only.
- Dropped: serde, schemars, derive_more, refineable (operators written by hand), string parsing of lengths
  and colours, grid placement.

## Things to know

- `lib.rs` declares every module; stubs are one `//!` line. Owners add their own `pub use x::*;` lines in
  `lib.rs`. Platform backends are declared in `platform.rs`, text backends in `text/mod.rs`.
- Module names clash with crate names (`image`, `platform::windows`, `platform::winit`): in the crate
  root, in `platform.rs` and inside those modules write `::image::…`, `::windows::…`, `::winit::…`.
- Style package: `Fill`, `UnderlineStyle`, `StrikethroughStyle` already exist (`color.rs`,
  `text/types.rs`). No refinement types exist for `Point`, `Size`, `Edges`, `Corners`: add them in `style`.
  `image.rs` holds a minimal `RenderImage` (GPUI's shape, BGRA `image::Frame`s): extend that file.
- Text package: `LineLayout` is only a struct; add its methods in your own files. `FontMetrics.descent` is
  negative (font units, as GPUI), `LineLayout.descent` is positive.
- Core: `ContentMask` is not defined yet (the app uses `ContentMask { bounds }`); `Bounds::centered(None,
  size, cx)` and `Bounds::maximized` take `&App` and are the core's; the input events' `Sealed` trait is
  private to `input_events.rs`.
- `AlphaMask`, `ColorMask` and `PathData` take ids from one shared atomic counter. Paths are
  `Contour { points, closed }` plus `PathKind::{Fill(FillRule), Stroke(StrokeStyle { width, cap, join,
  miter_limit, dash })}`.
- `NOTICE.md` lists every file adapted from GPUI. Add a line when you adapt another.
