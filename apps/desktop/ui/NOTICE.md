# NOTICE

folo-ui adapts parts of GPUI, the UI framework of the Zed editor:

> GPUI, Copyright 2022 - 2025 Zed Industries, Inc. Licensed under the Apache License, Version 2.0
> (https://www.apache.org/licenses/LICENSE-2.0). Source: https://github.com/zed-industries/zed (crates/gpui),
> version 0.2.2 (the copy in `apps/desktop/vendor/gpui` while the port is under way).

`ui/DESIGN.md` ("What GPUI code may be reused") says what may be adapted. Each file below says so in its first
line (`// Adapted from GPUI (Apache-2.0): <path>`), names the GPUI source it came from, and was changed (trimmed
to what FOLO Desktop uses, serde / schemars / derive_more / refineable removed, docs shortened). Add a line here
when you adapt another file.

| folo-ui file (`ui/src/`) | GPUI source (`crates/gpui/src/`) |
| --- | --- |
| `geometry/mod.rs`, `geometry/point.rs`, `geometry/size.rs`, `geometry/bounds.rs`, `geometry/edges.rs`, `geometry/corners.rs`, `geometry/units.rs`, `geometry/length.rs` | `geometry.rs` |
| `color.rs` | `color.rs`, `style.rs` (`Fill`) |
| `shared_string.rs` | `shared_string.rs` |
| `keystroke.rs` | `platform/keystroke.rs` |
| `input_events.rs` | `interactive.rs`, `platform.rs` (`CursorStyle`), `window.rs` (`WindowControlArea`, `DispatchEventResult`) |
| `platform.rs`, `platform/options.rs`, `platform/clipboard.rs` | `platform.rs` |
| `scene.rs` | `scene.rs` (`BorderStyle`, primitive field names) |
| `image.rs` | `assets.rs` (`RenderImage`) |
| `text/types.rs` | `text_system.rs`, `text_system/line_layout.rs`, `text_system/font_features.rs`, `text_system/font_fallbacks.rs`, `style.rs` (`UnderlineStyle`, `StrikethroughStyle`), `platform.rs` (`PlatformTextSystem`) |
| `platform/windows/{chrome, clipboard, display, drop_target, events, keyboard, pacer, platform, platform_window, shell, util, window}.rs` | `platform/windows/` (`events.rs`, `window.rs`, `platform.rs`, `keyboard.rs`, `display.rs`, `clipboard.rs`, `util.rs`, `vsync.rs`, `system_settings.rs`): message handling, title bar, key names, display UUID, window placement formulas |

The Apache License 2.0 text is in the repository root, `LICENSE-APACHE`.
