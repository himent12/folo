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
| `element.rs` | `element.rs` (the `Element` trait), `util.rs` (`FluentBuilder`) |
| `window/paint.rs` | `window.rs` (`PaintQuad`, `quad`, `fill`, `outline`) |
| `style/mod.rs`, `style/layout.rs`, `style/text.rs` | `style.rs` |
| `style/to_taffy.rs` | `taffy.rs` (`ToTaffy`) |
| `style/paint.rs` | `style.rs` (`Style::paint`, `overflow_mask`), `window.rs` (`paint_quad`, `paint_shadows`, `ContentMask`) |
| `style/refine.rs` | `refineable` (`Refineable`) and `refineable/derive_refineable` (the generated refinements) |
| `styled/mod.rs`, `styled/macros.rs` | `styled.rs`, and `gpui_macros/src/styles.rs` (the generated method families) |
| `styled/fluent.rs` | `util.rs` (`FluentBuilder`) |
| `path_builder.rs` | `path_builder.rs` |
| `svg.rs` | `svg_renderer.rs` |
| `assets.rs` | `assets.rs` (`AssetSource`) |
| `easing.rs` | `elements/animation.rs` (`easing`) |
| `text/system.rs`, `text/window_text_system.rs` | `text_system.rs` |
| `text/line_layout.rs`, `text/layout_cache.rs` | `text_system/line_layout.rs` |
| `text/line_wrapper.rs`, `text/tests/wrapper.rs` | `text_system/line_wrapper.rs`, `elements/text.rs` (FOLO's truncation tests) |
| `text/line.rs`, `text/paint.rs` | `text_system/line.rs`, `style.rs` (`TextAlign`) |
| `text/dwrite/mod.rs`, `text/dwrite/fonts.rs`, `text/dwrite/layout.rs`, `text/dwrite/raster.rs` | `platform/windows/direct_write.rs`, `platform/windows/color_text_raster.hlsl` |
| `text/dwrite/blend.rs` | `platform/windows/directx_renderer.rs` (`get_font_info`, `get_gamma_ratios`), `platform/windows/alpha_correction.hlsl` |
| `text/portable/fonts.rs`, `text/portable/layout.rs` | `platform/linux/text_system.rs` |
| `text/portable/raster.rs` | `platform/linux/text_system.rs`, `platform/blade/blade_renderer.rs` (`get_gamma_ratios`) |
| `platform/windows/{chrome, clipboard, display, drop_target, events, keyboard, pacer, platform, platform_window, shell, util, window}.rs` | `platform/windows/` (`events.rs`, `window.rs`, `platform.rs`, `keyboard.rs`, `display.rs`, `clipboard.rs`, `util.rs`, `vsync.rs`, `system_settings.rs`): message handling, title bar, key names, display UUID, window placement formulas |
| `raster/quad_eval.rs`, `raster/gradient.rs`, `raster/shadow.rs`, `raster/image.rs`, `raster/path.rs` (underlines) | `platform/windows/shaders.hlsl` (quad, gradient, Oklab, shadow, polychrome sprite and underline fragment maths) |
| `raster/sprite.rs` | `platform/windows/alpha_correction.hlsl` (glyph contrast and gamma correction) |
| `elements/anchored.rs`, `elements/uniform_list.rs`, `elements/text.rs`, `elements/animation.rs` | `elements/anchored.rs`, `elements/uniform_list.rs`, `elements/text.rs`, `elements/animation.rs` (layout, measure and scroll rules) |

The Apache License 2.0 text is in the repository root, `LICENSE-APACHE`.
