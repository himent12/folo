# folo-ui: the API the app uses

The checklist the port is measured against: everything `apps/desktop/src` takes from the UI crate (it used
`gpui`; it now imports `folo_ui`), one line per item, grouped by the package that owns it. Signatures are
GPUI 0.2.2's unless a line says otherwise. "core: done" means the skeleton has it with its final signature
(the body may be `todo!`); every other line is what that package must add for the app to compile.

Inventory method: every `use gpui::…` and `gpui::…` path in `apps/desktop/src`, every method call on
`Window`, `App`, `Context`, `AsyncApp`, entities, handles and elements whose name GPUI defines, and the
traits the app implements (`Element`, `EntityInputHandler`, `Render`, `Focusable`, `EventEmitter`,
`Global`, `AssetSource`).

## foundation (landed)

- `Pixels`, `px`, `Point`, `point`, `Size`, `size`, `Bounds` (`new`, `from_corners`, `centered_at`,
  `top/bottom/left/right`, `center`, `intersect`, `intersects`, `dilate`, `localize`, `contains`), `Edges`,
  `Corners`, `Corner`, `relative`, `DefiniteLength`, `Length`, `AbsoluteLength::to_pixels(rem)`, `IsZero::is_zero`
- `Hsla`, `hsla`, `Rgba`, `rgba`, `black`, `white`, `transparent_black`, `Background`, `linear_gradient`,
  `linear_color_stop`
- `SharedString` (`From<String/&'static str/…>`, `as_str`, `Deref<str>`)
- `KeyDownEvent` (`keystroke.key`, `keystroke.modifiers.{shift,control,…}`), `MouseDownEvent`
  (`position`, `button`, `click_count`, `modifiers`), `MouseUpEvent`, `MouseMoveEvent` (`position`,
  `pressed_button`, `dragging()`), `ScrollWheelEvent` (`delta.pixel_delta(line_height)`), `MouseButton::Left`,
  `ClickEvent`, `ExternalPaths::paths()`, `CursorStyle::{Arrow, OpenHand, …}`, `WindowControlArea`
- `ClipboardItem::new_string`, `ClipboardItem::text`, `PathPromptOptions { files, directories, multiple, prompt }`,
  `oneshot::Receiver` (awaited as `Ok(Ok(Some(paths)))`), `DisplayId`, `PlatformDisplay::{id, uuid, bounds}`,
  `WindowOptions { window_bounds, display_id, titlebar, window_min_size, app_id, ..Default }`,
  `TitlebarOptions { title, appears_transparent, traffic_light_position }`,
  `WindowBounds::{Windowed, Maximized, Fullscreen}`, `UTF16Selection { range, reversed }`
- `BorderStyle::{Solid, default}`, `RenderImage` (`Arc<RenderImage>`, `RenderImage::new(frames)`, `size(0)`)
- `FontWeight` (consts), `FontFeatures(Arc<Vec<(String, u32)>>)`, `Font`, `font(family)`, `TextRun { len, font,
  color, background_color, underline, strikethrough }`, `UnderlineStyle { color, thickness, wavy }`

## runtime (core: done unless marked)

- `Application::new()`, `.with_assets(impl AssetSource)`, `.run(|cx: &mut App| …)`
- `App`: `new` (via `AppContext`), `open_window(options, |window, cx| Entity<V>) -> Result<WindowHandle<V>>`,
  `text_system()`, `set_global`, `global::<G>()`, `bind_keys`, `background_executor`, `spawn`, `defer`,
  `refresh_windows`, `on_window_closed(|cx| …) -> Subscription`, `quit`, `activate(bool)`, `displays`,
  `primary_display`, `write_to_clipboard`, `read_from_clipboard`, `prompt_for_paths`, `open_url`, `reveal_path`,
  `focus_handle`, `stop_propagation`, `propagate`, `drop_image(Arc<RenderImage>, None)`
- `Bounds::<Pixels>::centered(Option<DisplayId>, Size<Pixels>, &App)` (core, `app/platform_ops.rs`: written)
- `AppContext` (imported as a trait: `cx.new(|cx| …)` on `App`, `Context<T>`)
- `Context<T>`: `notify`, `emit`, `subscribe(&entity, |this, emitter, ev, cx| …)`, `listener`, `processor`,
  `spawn(async move |this, cx| …)`, `entity`, `weak_entity`, `on_app_quit(|this, cx| async {})`, deref to `App`
- `AsyncApp` (inside `spawn`): `background_executor().spawn(..)` / `.timer(..)`; `weak.update(cx, …)` returns `Result`
- `Entity<T>`: `read(cx)`, `update(cx, |t, cx| …)`, `downgrade`, `entity_id`; `WeakEntity<T>`: `update`, `upgrade`
- `Subscription` (kept in `Vec<Subscription>`), `.detach()`
- `Task<T>` (stored, `detach`), `BackgroundExecutor::{spawn, timer}`
- `Global` (impl for `Theme`), `EventEmitter<E>` (impl for `TextInput`)
- `actions!(ns, [A, B, …])`; `KeyBinding::new("ctrl-shift-l", Action, None | Some("TextInput"))`
- Key contexts used: `"FoloApp"`, `"TextInput"` (bare identifiers); bindings are single keystrokes

## window (core: done)

- `Window`: `focus(&handle)`, `focused(cx)`, `focus_next`, `focus_prev`, `viewport_size`, `text_system`,
  `request_animation_frame`, `with_content_mask(Some(ContentMask { bounds }), …)`, `content_mask`,
  `is_fullscreen`, `is_maximized`, `toggle_fullscreen`, `minimize_window`, `remove_window`, `paint_quad`,
  `paint_path(path, impl Into<Background>)`, `paint_image(bounds, corners, Arc<RenderImage>, frame, grayscale)`,
  `line_height`, `text_style`, `rem_size`, `scale_factor`, `request_layout(style, [], cx)`, `refresh`,
  `mouse_position`, `handle_input(&fh, ElementInputHandler::new(bounds, entity), cx)`, `window_bounds`,
  `display(cx)`, `on_window_should_close(cx, |window, cx| bool)`
- `ContentMask<Pixels> { bounds }`, `PaintQuad` (+ `.corner_radii(..)`), `quad`, `fill`, `outline`
- `FocusHandle`: `tab_stop(bool)`, `is_focused(window)`, `clone`; `Focusable` (impl for views)
- `Div` (`div()`), `Stateful<Div>`; `InteractiveElement`: `id`, `group`, `track_focus`, `key_context`, `hover`,
  `group_hover`, `on_mouse_down`, `capture_any_mouse_down`, `on_mouse_up`, `on_mouse_down_out`,
  `on_mouse_up_out`, `on_mouse_move`, `on_drag_move::<ExternalPaths>`, `on_scroll_wheel`, `on_action`,
  `on_key_down`, `capture_key_down`, `on_drop::<ExternalPaths>`, `occlude`, `window_control_area`;
  `StatefulInteractiveElement`: `overflow_y_scroll`, `track_scroll(&ScrollHandle)`, `active`, `group_active`,
  `on_click`, `on_hover`, `tooltip(|window, cx| AnyView)`
- `ScrollHandle`: `new`, `offset`, `set_offset`, `max_offset`, `bounds`, `scroll_to_bottom`, `scroll_to_item`
- `DragMoveEvent<ExternalPaths>`

## elements (core: done)

- `Element` (implemented by hand in `src/input.rs`: `id`, `source_location`, `request_layout`, `prepaint`,
  `paint` with `Option<&GlobalElementId>`, `Option<&InspectorElementId>`), `IntoElement`, `AnyElement`
  (`into_any_element`), `Render`, `ParentElement` (`child`, `children`), `LayoutId`, `GlobalElementId`,
  `InspectorElementId`, `ElementId::Name`, `From<&'static str | SharedString | (&'static str, usize)>`
- `prelude::*` / `prelude::FluentBuilder` (`when`, `when_some`)
- Text children: `&'static str`, `String`, `SharedString`, `StyledText::new(..).with_runs(runs)`
- `svg().path(..)`, `Svg`; `canvas(prepaint, paint)`, `Canvas`; `uniform_list(id, count, cx.processor(..))`,
  `UniformList::track_scroll(UniformListScrollHandle)`, `UniformListScrollHandle::{new, scroll_to_item}`,
  `ScrollStrategy::{Top, Bottom}`; `anchored().snap_to_window_with_margin(px)`; `deferred(..)` (and
  `.with_priority`); `Animation::new(d).repeat().with_easing(f)`, `AnimationExt::with_animation(id, anim, |e, p| e)`
- `AnyView` (`From<Entity<V: Render>>`, `cx.new(..).into()`), `Entity<V: Render>` as a child
- `EntityInputHandler` (implemented by `TextInput`), `ElementInputHandler::new(bounds, entity)`

## style (to do: names the app still fails on)

- `Styled` trait (`fn style(&mut self) -> &mut StyleRefinement` is the core's seam; `Div`, `Stateful<E>`,
  `Svg`, `Canvas<T>`, `UniformList` implement it) with these methods, as GPUI's `styled.rs` and
  `style_helpers!`: zero-argument — `flex flex_1 flex_col flex_none flex_grow flex_shrink flex_wrap grid
  absolute relative items_baseline items_center items_end items_start justify_between justify_center
  justify_end w_full h_full size_full max_w_full min_w_0 min_h_0 mx_auto inset_0 top_0 bottom_0 left_0
  right_0 border_1 border_2 border_t_1 border_b_0 border_b_1 border_b_2 border_l_1 border_r_1 border_dashed
  rounded_full rounded_l_full overflow_hidden whitespace_normal whitespace_nowrap text_left text_center
  text_right text_ellipsis truncate cursor_pointer cursor_default cursor_not_allowed visible invisible`;
  one argument — `w h size min_w min_h max_w max_h p px py pt pb pl pr m mx my mt mb ml mr gap gap_x gap_y
  top bottom left right rounded rounded_t rounded_b flex_basis grid_cols line_clamp bg border_color
  text_color text_size font_family font_weight line_height opacity shadow(Vec<BoxShadow>) cursor(CursorStyle)`;
  and `text_style(&mut self) -> &mut Option<TextStyleRefinement>` (the app's `tnum()` sets `font_features`)
- `impl Styled for StyleRefinement` (the `hover(|s| s.bg(..))` closures call Styled methods on it)
- `Style` (`Default`, `size.width = relative(1.).into()`, `size.height = Pixels.into()`), `StyleRefinement`
  (`aspect_ratio: Option<f32>` field written directly), `TextStyle` (`color: Hsla`, `font_size:
  AbsoluteLength`, `font() -> Font`), `TextStyleRefinement` (`font_features: Option<FontFeatures>`)
- `BoxShadow { color: Hsla, offset: Point<Pixels>, blur_radius: Pixels, spread_radius: Pixels }`
- `AvailableSpace` (GPUI's `taffy.rs`; the core's layout API takes `Size<AvailableSpace>`)
- `PathBuilder::{fill(), stroke(Pixels)}`, `move_to`, `line_to`, `close` (`&mut self`),
  `dash_array(self, &[Pixels]) -> Self` (the app writes `b = b.dash_array(&d)`), `build(self) ->
  anyhow::Result<P>` where `P` is what `Window::paint_path` takes: the core names it **`BuiltPath`** (logical
  px; `Path` is taken by the scene primitive). Rename in `window/paint.rs` if you pick another name.
- `AssetSource` (`load(&self, &str) -> Result<Option<Cow<'static, [u8]>>>`, `list(&self, &str) ->
  Result<Vec<SharedString>>`; implemented by the app's `Assets`)
- `ease_out_quint() -> impl Fn(f32) -> f32`
- SVG to mask for `Window::paint_svg` and the icon `MaskCache`

## text (to do)

- `TextSystem::add_fonts(&self, Vec<Cow<'static, [u8]>>) -> anyhow::Result<()>` (`cx.text_system()` is
  `&Arc<TextSystem>`)
- `WindowTextSystem::shape_line(&self, SharedString, font_size: Pixels, &[TextRun], force_width:
  Option<Pixels>) -> ShapedLine` (`window.text_system()` is `&Arc<WindowTextSystem>`)
- `ShapedLine` (stored in `TextInput`): `width: Pixels` (field), `x_for_index(usize) -> Pixels`,
  `index_for_x(Pixels) -> Option<usize>`, `closest_index_for_x(Pixels) -> usize`, `len()`,
  `paint(&self, origin: Point<Pixels>, line_height: Pixels, window: &mut Window, cx: &mut App) -> Result<()>`
- `WrappedLine` (the text element's measured lines; core refers to it in `elements/text.rs`)
- `GlyphPainter`: the window implements it with these existing `Window` methods (`window/paint.rs`):
  `paint_glyph(origin, FontId, GlyphId, font_size, Hsla) -> Result<()>`, `paint_emoji(origin, FontId,
  GlyphId, font_size) -> Result<()>`, `paint_underline(origin, width, &UnderlineStyle)`,
  `paint_strikethrough(origin, width, &StrikethroughStyle)`, and `paint_quad(fill(bounds, color))` for
  backgrounds. Add the `impl GlyphPainter for Window` next to them when the trait lands.

## raster (to do)

- `raster::Renderer` (`new`, `render(&scene, glyph_blend, &mut framebuffer) -> Vec<Bounds<DevicePixels>>`,
  `invalidate`) and `raster::MaskCache` (glyph masks and icon masks; the window owns two). The window refers
  to stand-ins in `pending.rs` (`raster_pending`) until then: switch the `use` in `window/mod.rs`.
