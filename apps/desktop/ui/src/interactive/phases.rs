//! `Interactivity`'s three phases, which `Div` calls around its children (GPUI 0.2.2's `div.rs`): the style
//! (base refined by group hover, hover, group active and active, in that order), the dispatch node and
//! focus, the hitbox, scroll offsets clamped to the content, and at paint the background, listeners, key
//! context, cursor and window control area, then the children, then the border. Owner tag: `"window"`.

use std::{collections::HashMap, rc::Rc};

use smallvec::SmallVec;

use crate::{
    App, Bounds, Display, GlobalElementId, Hitbox, HitboxBehavior, HitboxId, InspectorElementId,
    InteractiveElementState, Interactivity, LayoutId, Overflow, Pixels, Point, Refineable, SharedString,
    Size, Style, Visibility, Window,
    interactive::{ElementClickedState, tooltip},
    px, size,
};

impl Interactivity {
    /// Computes the style for this frame and requests the layout (the closure adds children).
    pub fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
        f: impl FnOnce(Style, &mut Window, &mut App) -> LayoutId,
    ) -> LayoutId {
        self.request_layout_ref(global_id, inspector_id, window, cx, |style, window, cx| {
            f(style.clone(), window, cx)
        })
    }

    /// Borrowed-style variant of [`Self::request_layout`]: computes the base+active style once and
    /// caches it in `Window::frame_styles` for `prepaint` and `paint` while drawing.
    pub(crate) fn request_layout_ref(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
        f: impl FnOnce(&Style, &mut Window, &mut App) -> LayoutId,
    ) -> LayoutId {
        window.with_sparse_element_state::<InteractiveElementState, _>(
            global_id,
            InteractiveElementState::is_empty,
            |mut element_state, window| {
                // A drag in progress cancels a pending click and the pressed state (GPUI).
                if let Some(state) = element_state.as_ref()
                    && window.active_drag.is_some()
                {
                    if let Some(pending) = state.pending_mouse_down.as_ref() {
                        *pending.borrow_mut() = None;
                    }
                    if let Some(clicked) = state.clicked_state.as_ref() {
                        *clicked.borrow_mut() = Default::default();
                    }
                }

                // A focusable element without a tracked handle keeps one in its state.
                if self.focusable
                    && self.tracked_focus_handle.is_none()
                    && let Some(state) = element_state.as_mut()
                {
                    let mut handle = state
                        .focus_handle
                        .get_or_insert_with(|| cx.focus_handle())
                        .clone()
                        .tab_stop(self.tab_stop);
                    if let Some(index) = self.tab_index {
                        handle = handle.tab_index(index);
                    }
                    self.tracked_focus_handle = Some(handle);
                }

                if let Some(scroll_handle) = self.tracked_scroll_handle.as_ref() {
                    self.scroll_offset = Some(scroll_handle.0.borrow().offset.clone());
                } else if (self.base_style.overflow.x == Some(Overflow::Scroll)
                    || self.base_style.overflow.y == Some(Overflow::Scroll))
                    && let Some(state) = element_state.as_mut()
                {
                    self.scroll_offset = Some(state.scroll_offset.get_or_insert_with(Rc::default).clone());
                }

                let clicked = element_clicked_state(element_state.as_ref());
                let style = self.compute_style_internal(None, element_state.as_ref(), window);
                let layout_id = f(&style, window, cx);
                if window.drawing
                    && let Ok(index) = u32::try_from(window.frame_styles.len())
                {
                    window.frame_styles.push(style);
                    self.cached_style = Some(index);
                    self.cached_clicked = clicked;
                } else {
                    self.cached_style = None;
                }
                (layout_id, element_state)
            },
        )
    }

    /// Runs `f` with the prepaint `Style` (reusing the cached style from `request_layout` when unchanged).
    pub(crate) fn with_prepaint_style<R>(
        &self,
        global_id: Option<&GlobalElementId>,
        window: &mut Window,
        f: impl FnOnce(&Style, &Window) -> R,
    ) -> R {
        window.with_sparse_element_state::<InteractiveElementState, _>(
            global_id,
            InteractiveElementState::is_empty,
            |element_state, window| {
                let clicked = element_clicked_state(element_state.as_ref());
                let result = if let Some(index) = self.cached_style
                    && window.drawing
                    && clicked == self.cached_clicked
                    && let Some(style) = window.frame_styles.get(index as usize)
                {
                    f(style, window)
                } else {
                    let style = self.compute_style_internal(None, element_state.as_ref(), window);
                    f(&style, window)
                };
                (result, element_state)
            },
        )
    }

    /// Pushes the element's dispatch node and focus id, inserts the hitbox, clamps the scroll offset, and
    /// prepaints children through `f` (which gets the style, the scroll offset and the hitbox).
    #[allow(clippy::too_many_arguments)]
    pub fn prepaint<R>(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        content_size: Size<Pixels>,
        window: &mut Window,
        cx: &mut App,
        f: impl FnOnce(&Style, Point<Pixels>, Option<Hitbox>, &mut Window, &mut App) -> R,
    ) -> R {
        self.content_size = content_size;
        self.dispatch_node = Some(window.push_dispatch_node());
        if let Some(focus_handle) = self.tracked_focus_handle.as_ref() {
            window.set_focus_handle(focus_handle, cx);
        }
        let result = window.with_sparse_element_state::<InteractiveElementState, _>(
            global_id,
            InteractiveElementState::is_empty,
            |mut element_state, window| {
                let clicked = element_clicked_state(element_state.as_ref());
                let reused_index = self.cached_style.take().filter(|&index| {
                    window.drawing
                        && clicked == self.cached_clicked
                        && (index as usize) < window.frame_styles.len()
                });
                let style = match reused_index {
                    Some(index) => std::mem::take(&mut window.frame_styles[index as usize]),
                    None => self.compute_style_internal(None, element_state.as_ref(), window),
                };

                if let Some(state) = element_state.as_mut() {
                    if state.clicked_state.is_some() {
                        self.active = Some(clicked.element);
                    }
                    if let Some(active_tooltip) = state.active_tooltip.as_ref() {
                        if self.tooltip_builder.is_some() {
                            self.tooltip_id = tooltip::set_tooltip_on_window(active_tooltip, window);
                        } else {
                            state.active_tooltip.take();
                        }
                    }
                }

                let has_active_click = clicked.is_clicked();
                let result = window.with_text_style_ref(style.text_style(), |window| {
                    window.with_content_mask(style.overflow_mask(bounds, window.rem_size()), |window| {
                        let hitbox = (self.should_insert_hitbox(&style) || has_active_click)
                            .then(|| window.insert_hitbox(bounds, self.hitbox_behavior));
                        let scroll_offset = self.clamp_scroll_position(bounds, &style, window);
                        f(&style, scroll_offset, hitbox, window, cx)
                    })
                });
                if let Some(index) = reused_index
                    && (index as usize) < window.frame_styles.len()
                {
                    window.frame_styles[index as usize] = style;
                    self.cached_style = Some(index);
                }
                (result, element_state)
            },
        );
        window.pop_dispatch_node();
        result
    }

    /// Whether the element needs a hitbox (GPUI's `should_insert_hitbox`).
    fn should_insert_hitbox(&self, style: &Style) -> bool {
        self.hitbox_behavior != HitboxBehavior::Normal
            || self.window_control.is_some()
            || style.mouse_cursor.is_some()
            || self.group.is_some()
            || self.scroll_offset.is_some()
            || self.tracked_focus_handle.is_some()
            || self.hover_style.is_some()
            || self.group_hover_style.is_some()
            || self.active_style.is_some()
            || self.group_active_style.is_some()
            || self.hover_listener.is_some()
            || !self.mouse_up_listeners.is_empty()
            || !self.mouse_down_listeners.is_empty()
            || !self.mouse_move_listeners.is_empty()
            || !self.click_listeners.is_empty()
            || !self.scroll_wheel_listeners.is_empty()
            || !self.drop_listeners.is_empty()
            || self.tooltip_builder.is_some()
    }

    /// Clamps the scroll offset to `-max..=0` (max = content plus padding minus bounds, two decimals) and
    /// reports bounds and max to the scroll handle (GPUI's `clamp_scroll_position`).
    fn clamp_scroll_position(
        &self,
        bounds: Bounds<Pixels>,
        style: &Style,
        window: &mut Window,
    ) -> Point<Pixels> {
        fn round_to_two_decimals(pixels: Pixels) -> Pixels {
            (pixels * 100.).round() / 100.
        }
        let Some(scroll_offset) = self.scroll_offset.as_ref() else { return Point::default() };
        let mut handle_state = self.tracked_scroll_handle.as_ref().map(|handle| handle.0.borrow_mut());
        let mut scroll_to_bottom = false;
        if let Some(state) = handle_state.as_deref_mut() {
            state.overflow = style.overflow;
            scroll_to_bottom = std::mem::take(&mut state.scroll_to_bottom);
        }
        let padding = style.padding.to_pixels(bounds.size.into(), window.rem_size());
        let padding_size = size(padding.left + padding.right, padding.top + padding.bottom);
        let padded = self.content_size + padding_size;
        let scroll_max = size(
            round_to_two_decimals(padded.width - bounds.size.width).max(px(0.)),
            round_to_two_decimals(padded.height - bounds.size.height).max(px(0.)),
        );
        let mut offset = scroll_offset.borrow_mut();
        offset.x = offset.x.clamp(-scroll_max.width, px(0.));
        offset.y =
            if scroll_to_bottom { -scroll_max.height } else { offset.y.clamp(-scroll_max.height, px(0.)) };
        if let Some(mut state) = handle_state {
            state.max_offset = scroll_max;
            state.bounds = bounds;
        }
        *offset
    }

    /// Paints background, borders and shadows, registers listeners, key context, focus and cursor, and
    /// paints children through `f`.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        hitbox: Option<&Hitbox>,
        window: &mut Window,
        cx: &mut App,
        f: impl FnOnce(&Style, &mut Window, &mut App),
    ) {
        if let Some(node) = self.dispatch_node {
            window.set_active_dispatch_node(node);
        }
        self.hovered = hitbox.map(|hitbox| hitbox.is_hovered(window));
        window.with_sparse_element_state::<InteractiveElementState, _>(
            global_id,
            InteractiveElementState::is_empty,
            |mut element_state, window| {
                let clicked = element_clicked_state(element_state.as_ref());
                let reused_index = self.cached_style.take().filter(|&index| {
                    window.drawing
                        && clicked == self.cached_clicked
                        && (index as usize) < window.frame_styles.len()
                });
                let style =
                    self.resolve_paint_style(reused_index, clicked, hitbox, element_state.as_ref(), window);
                self.paint_hover_group_handler(window);
                if style.visibility == Visibility::Hidden {
                    return ((), element_state);
                }
                if let Some(focus_handle) = self.tracked_focus_handle.as_ref() {
                    window.insert_tab_stop(focus_handle);
                }
                window.with_element_opacity(style.opacity, |window| {
                    let paint = window.style_paint();
                    style.paint_before_children(bounds, &paint, &mut window.next_frame.scene);
                    window.with_text_style_ref(style.text_style(), |window| {
                        window.with_content_mask(style.overflow_mask(bounds, window.rem_size()), |window| {
                            if let Some(hitbox) = hitbox {
                                if let Some(cursor) = style.mouse_cursor
                                    && window.active_drag.is_none()
                                {
                                    window.set_cursor_style(cursor, hitbox);
                                }
                                if let Some(group) = self.group.clone() {
                                    window.group_hitboxes.push(group, hitbox.id);
                                }
                                if let Some(area) = self.window_control {
                                    window.insert_window_control_hitbox(area, hitbox.clone());
                                }
                                self.paint_mouse_listeners(hitbox, element_state.as_mut(), window);
                                self.paint_scroll_listener(hitbox, &style, window);
                            }
                            self.paint_keyboard_listeners(window);
                            f(&style, window, cx);
                            if hitbox.is_some()
                                && let Some(group) = self.group.as_ref()
                            {
                                window.group_hitboxes.pop(group);
                            }
                        });
                    });
                    let paint = window.style_paint();
                    style.paint_after_children(bounds, &paint, &mut window.next_frame.scene);
                });
                ((), element_state)
            },
        );
    }

    /// The style for this frame: the base refined by focus, hover, group hover, active and group active.
    pub fn compute_style(
        &self,
        global_id: Option<&GlobalElementId>,
        hitbox: Option<&Hitbox>,
        window: &mut Window,
        _cx: &mut App,
    ) -> Style {
        window.with_sparse_element_state::<InteractiveElementState, _>(
            global_id,
            InteractiveElementState::is_empty,
            |element_state, window| {
                let style = self.compute_style_internal(hitbox, element_state.as_ref(), window);
                (style, element_state)
            },
        )
    }

    /// Resolves the paint style, reusing `window.frame_styles[reused_index]` without re-refining
    /// `base_style` when neither hover nor active precedence invalidates it.
    fn resolve_paint_style(
        &self,
        reused_index: Option<u32>,
        clicked: ElementClickedState,
        hitbox: Option<&Hitbox>,
        element_state: Option<&InteractiveElementState>,
        window: &mut Window,
    ) -> Style {
        let mut group_hover = None;
        let mut self_hover = None;
        if let Some(hitbox) = hitbox
            && window.active_drag.is_none()
        {
            if let Some(group_hover_style) = self.group_hover_style.as_ref()
                && let Some(group_hitbox) = window.group_hitboxes.get(&group_hover_style.group)
                && group_hitbox.is_hovered(window)
            {
                group_hover = Some(&group_hover_style.style);
            }
            if let Some(hover_style) = self.hover_style.as_ref()
                && hitbox.is_hovered(window)
            {
                self_hover = Some(hover_style.as_ref());
            }
        }
        if let Some(index) = reused_index {
            if group_hover.is_none() && self_hover.is_none() {
                return std::mem::take(&mut window.frame_styles[index as usize]);
            }
            // Active styles refine after hover styles; only apply hover deltas on top of the cached
            // style when no active style was already merged into it.
            if !clicked.is_clicked() {
                let mut style = std::mem::take(&mut window.frame_styles[index as usize]);
                if let Some(group_hover_style) = group_hover {
                    style.refine(group_hover_style);
                }
                if let Some(hover_style) = self_hover {
                    style.refine(hover_style);
                }
                return style;
            }
        }
        self.compute_style_internal(hitbox, element_state, window)
    }

    /// GPUI's `compute_style_internal`: hover styles need a hitbox (so only paint sees them) and no drag;
    /// the pressed styles come from the element state.
    fn compute_style_internal(
        &self,
        hitbox: Option<&Hitbox>,
        element_state: Option<&InteractiveElementState>,
        window: &Window,
    ) -> Style {
        let mut style = Style::default();
        style.refine(&self.base_style);
        if let Some(hitbox) = hitbox
            && window.active_drag.is_none()
        {
            if let Some(group_hover) = self.group_hover_style.as_ref()
                && let Some(group_hitbox) = window.group_hitboxes.get(&group_hover.group)
                && group_hitbox.is_hovered(window)
            {
                style.refine(&group_hover.style);
            }
            if let Some(hover_style) = self.hover_style.as_ref()
                && hitbox.is_hovered(window)
            {
                style.refine(hover_style);
            }
        }
        if let Some(state) = element_state
            && let Some(clicked_state) = state.clicked_state.as_ref()
        {
            let clicked = *clicked_state.borrow();
            if clicked.group
                && let Some(group) = self.group_active_style.as_ref()
            {
                style.refine(&group.style);
            }
            if clicked.element
                && let Some(active_style) = self.active_style.as_ref()
            {
                style.refine(active_style);
            }
        }
        style
    }
}

/// The hitboxes of the named groups being painted (for `group_hover` / `group_active`): a stack per name,
/// pushed by the group's element around its children (GPUI keeps it in a global; here the window owns it).
#[derive(Default)]
pub(crate) struct GroupHitboxes(pub(crate) HashMap<SharedString, SmallVec<[HitboxId; 1]>>);

impl GroupHitboxes {
    /// The innermost hitbox painted with `name` right now.
    pub(crate) fn get(&self, name: &SharedString) -> Option<HitboxId> {
        self.0.get(name).and_then(|stack| stack.last()).copied()
    }

    /// Enters a group.
    pub(crate) fn push(&mut self, name: SharedString, hitbox: HitboxId) {
        self.0.entry(name).or_default().push(hitbox);
    }

    /// Leaves a group.
    pub(crate) fn pop(&mut self, name: &SharedString) {
        if let Some(stack) = self.0.get_mut(name) {
            stack.pop();
        }
    }
}

/// Whether a `Display::None` style skips its children (Div's phases ask).
pub(crate) fn skips_children(style: &Style) -> bool {
    style.display == Display::None
}

/// Reads the current pressed state from `element_state`, or defaults to unpressed.
#[inline]
fn element_clicked_state(element_state: Option<&InteractiveElementState>) -> ElementClickedState {
    element_state
        .and_then(|state| state.clicked_state.as_ref())
        .map_or_else(ElementClickedState::default, |clicked| *clicked.borrow())
}
