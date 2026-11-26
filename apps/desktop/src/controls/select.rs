//! Selects: a dropdown trigger and its listbox.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, ClickEvent, Context, InteractiveElement, IntoElement, KeyDownEvent, MouseMoveEvent,
    ParentElement, SharedString, StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::tooltip::tip;
use crate::app::{FoloApp, focus_ring};
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::*;

/// Incremental keyboard typeahead buffer for open [`FoloApp::select`] menus.
#[derive(Default)]
pub struct SelectTypeahead {
    query: String,
    last: Option<std::time::Instant>,
    menu: Option<&'static str>,
    /// Enclosing container's vertical scroll offset when the active popup opened; if it changes
    /// while the popup is open (wheel, scrollbar, or programmatic scroll), the popup closes so it
    /// never detaches over a dialog header or the title bar.
    scroll_anchor: Option<Option<folo_ui::Pixels>>,
}

/// Pause after which the next typed character starts a fresh typeahead prefix.
const TYPEAHEAD_WINDOW: std::time::Duration = std::time::Duration::from_millis(800);

impl SelectTypeahead {
    /// Clears any buffered typeahead prefix without resetting the popup's scroll anchor.
    pub fn clear(&mut self) {
        self.query.clear();
        self.last = None;
        self.menu = None;
    }

    /// Clears the recorded scroll anchor (called on popup open/close and when no popup is open).
    pub(crate) fn reset_scroll_anchor(&mut self) {
        self.scroll_anchor = None;
    }

    /// Records the enclosing scroll offset on the first frame a popup is open, and returns `true`
    /// once that scroll container moves to a different offset.
    pub(crate) fn scroll_moved(&mut self, current: Option<folo_ui::Pixels>) -> bool {
        match self.scroll_anchor {
            None => {
                self.scroll_anchor = Some(current);
                false
            }
            Some(initial) => initial != current,
        }
    }

    /// Appends `ch` to the active prefix for `menu` and returns the matching enabled option index.
    ///
    /// Multi-character prefixes search from `current` (wrapping around), falling back to a fresh
    /// single-character search if the extended prefix misses. Repeating the same letter cycles
    /// through matching options when no option matches the doubled prefix.
    pub fn match_option(
        &mut self,
        menu: &'static str,
        ch: char,
        now: std::time::Instant,
        labels: &[SharedString],
        disabled: &[bool],
        current: usize,
    ) -> Option<usize> {
        let n = labels.len();
        if n == 0 {
            self.clear();
            return None;
        }
        let expired = self.menu != Some(menu)
            || self.last.is_none_or(|t| now.saturating_duration_since(t) > TYPEAHEAD_WINDOW);
        if expired {
            self.query.clear();
            self.menu = Some(menu);
        }
        self.last = Some(now);
        let ch_lower: String = ch.to_lowercase().collect();
        let cur = current % n;
        let find = |prefix: &str, start: usize| -> Option<usize> {
            for off in 0..n {
                let i = (start + off) % n;
                if disabled.get(i).copied().unwrap_or(false) {
                    continue;
                }
                if labels[i].to_lowercase().starts_with(prefix) {
                    return Some(i);
                }
            }
            None
        };
        if self.query.is_empty() {
            self.query = ch_lower;
            let cur_matches = !disabled.get(cur).copied().unwrap_or(false)
                && labels[cur].to_lowercase().starts_with(&self.query);
            let start = if cur_matches { (cur + 1) % n } else { 0 };
            return find(&self.query, start);
        }
        if self.query == ch_lower {
            return find(&ch_lower, (cur + 1) % n);
        }
        let mut extended = self.query.clone();
        extended.push_str(&ch_lower);
        if let Some(hit) = find(&extended, cur) {
            self.query = extended;
            return Some(hit);
        }
        self.query = ch_lower;
        find(&self.query, (cur + 1) % n)
    }
}

/// The narrowest a menu opens: wide enough for an option's name and detail side by side, also under
/// a narrow trigger (a leaderboard's Run select is 220 px).
const MENU_MIN_W: f32 = 300.;

/// A menu's width and horizontal offset relative to its trigger's left edge: as wide as the
/// trigger (at least [`MENU_MIN_W`], capped to the viewport's 16 px horizontal margin), left-aligned
/// when it fits on the right, right-aligned when it fits on the left, otherwise clamped inside
/// `[8., viewport_w - 8.]` (`trigger`: its left and right edge, `viewport_w`: the window's width).
fn menu_box(trigger: (f32, f32), viewport_w: f32) -> (f32, f32) {
    let trig_w = (trigger.1 - trigger.0).max(0.);
    let max_w = (viewport_w - 16.).max(0.);
    let w = if viewport_w > 16. { trig_w.max(MENU_MIN_W).min(max_w) } else { trig_w };
    let max_left = (viewport_w - 8. - w).max(8.);
    let raw_left = if trigger.0 + w <= viewport_w - 8. {
        trigger.0
    } else if trigger.1 - w >= 8. {
        trigger.1 - w
    } else {
        max_left
    };
    let menu_left = raw_left.clamp(8., max_left);
    (w, menu_left - trigger.0)
}

/// Called with the chosen option index.
pub type OnSelect = std::rc::Rc<dyn Fn(&mut FoloApp, usize, &mut Window, &mut Context<FoloApp>)>;

impl FoloApp {
    /// Closes any open select dropdown or model export menu and clears its anchor/typeahead state.
    pub(crate) fn close_menus(&mut self) -> bool {
        let was_open = self.menu_open.is_some() || self.export_open;
        if was_open {
            self.menu_open = None;
            self.menu_trigger.set(None);
            self.menu_typeahead.clear();
            self.menu_typeahead.reset_scroll_anchor();
            self.export_open = false;
        }
        was_open
    }

    /// The enclosing scroll container for the currently active popup, if any.
    pub(crate) fn menu_scroll_handle(&self) -> Option<folo_ui::ScrollHandle> {
        if self.new_run.is_some() || self.synth.is_some() || self.cloud.connect.is_some() {
            Some(self.dialog_scroll.clone())
        } else if self.modal_open() || matches!(self.route, crate::app::Route::Dataset(_)) {
            None
        } else if self.route == crate::app::Route::Playground {
            Some(self.play_side_scroll.clone())
        } else {
            Some(self.page_scroll.clone())
        }
    }

    /// The viewport bounds clipping an open popup's trigger, when its enclosing scroll container
    /// is mounted and laid out.
    pub(crate) fn menu_clip_bounds(&self) -> Option<folo_ui::Bounds<folo_ui::Pixels>> {
        let b = self.menu_scroll_handle()?.bounds();
        (b.size.height > px(0.)).then_some(b)
    }

    /// Vertical scroll offset of the active popup's enclosing scroll container, if any.
    pub(crate) fn menu_scroll_offset(&self) -> Option<folo_ui::Pixels> {
        self.menu_scroll_handle().map(|h| h.offset().y)
    }

    /// True once the active popup's trigger has been measured and lies completely outside its
    /// enclosing scroll viewport.
    pub(crate) fn menu_trigger_clipped_out(&self) -> bool {
        let Some(b) = self.menu_trigger.get() else { return false };
        let Some(clip) = self.menu_clip_bounds() else { return false };
        b.bottom() <= clip.top() || b.top() >= clip.bottom()
    }

    /// Dropdown trigger that opens a listbox under itself.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        self.select_with(key, current, options, selected, &[], window, cx, on_select)
    }

    /// [`Self::select`] with options that can't be chosen now: `disabled[i]` says why option `i`
    /// can't (its tooltip); it shows dimmed, the pointer can't pick it and the arrows skip it.
    #[allow(clippy::too_many_arguments)]
    pub fn select_with(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        disabled: &[Option<SharedString>],
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        self.select_full(key, current, options, selected, disabled, &[], false, window, cx, on_select)
    }

    /// A compact (28 px) dropdown trigger for toolbars, matching segmented controls beside it.
    #[allow(clippy::too_many_arguments)]
    pub fn compact_select(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        self.select_full(key, current, options, selected, &[], &[], true, window, cx, on_select)
    }

    /// A compact (28 px) select of classes, for a toolbar: `classes[i]` is option `i`'s class (its
    /// swatch before the name; `None` for an option that is no class, "All classes"), and the trigger
    /// shows the chosen one's.
    #[allow(clippy::too_many_arguments)]
    pub fn class_select(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        classes: &[Option<u32>],
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        self.select_full(key, current, options, selected, &[], classes, true, window, cx, on_select)
    }

    #[allow(clippy::too_many_arguments)]
    fn select_full(
        &self,
        key: &'static str,
        current: SharedString,
        options: Vec<(SharedString, Option<SharedString>)>,
        selected: usize,
        disabled: &[Option<SharedString>],
        classes: &[Option<u32>],
        compact: bool,
        window: &Window,
        cx: &mut Context<Self>,
        on_select: OnSelect,
    ) -> AnyElement {
        let t = theme(cx);
        let trigger_h = if compact { tsize::CONTROL_MD } else { tsize::CONTROL_LG };
        let class_of = |i: usize| classes.get(i).copied().flatten();
        let trigger_class = class_of(selected);
        let open = self.menu_open == Some(key);
        let fh = self.fh(key, cx);
        let ring = self.ring(&fh, window);
        let n = options.len();
        let empty = n == 0;
        let keys = on_select.clone();
        let off: Vec<Option<SharedString>> = (0..n).map(|i| disabled.get(i).cloned().flatten()).collect();
        let usable: Vec<usize> = (0..n).filter(|&i| off[i].is_none()).collect();
        let (first, last) = (usable.first().copied().unwrap_or(0), usable.last().copied().unwrap_or(0));
        let initial_hl = if selected < n && off[selected].is_none() { selected } else { first };
        // The usable option after (or before) `i`, staying put at the ends.
        let step_usable = usable.clone();
        let step = move |i: usize, down: bool| {
            let next = if down {
                step_usable.iter().find(|&&u| u > i)
            } else {
                step_usable.iter().rev().find(|&&u| u < i)
            };
            next.copied().unwrap_or(i)
        };
        let page_step = move |i: usize, down: bool| {
            if usable.is_empty() {
                return i;
            }
            let pos = usable.binary_search(&i).unwrap_or_else(|p| p.min(usable.len() - 1));
            let next = if down { (pos + 8).min(usable.len() - 1) } else { pos.saturating_sub(8) };
            usable[next]
        };
        // While open, arrows move a highlight; Enter commits it, Esc closes without a change.
        let hl = if open { self.menu_hl.min(n.saturating_sub(1)) } else { selected };
        let menu_scroll = self.menu_scroll.clone();
        let dead: Vec<bool> = off.iter().map(Option::is_some).collect();
        let labels: Vec<SharedString> = options.iter().map(|(label, _)| label.clone()).collect();
        let trigger = div()
            .id(key)
            .track_focus(&fh)
            .relative()
            .h(px(trigger_h))
            .w_full()
            .px(px(10.))
            .flex()
            .items_center()
            .gap(px(8.))
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if open { t.accent_line } else { t.line_control })
            .cursor_pointer()
            .hover(|s| s.border_color(t.text_tertiary))
            .active(|s| s.bg(t.bg_active))
            .when(open, |d| {
                let cell = self.menu_trigger.clone();
                let scroll_handle = self.menu_scroll_handle();
                // Where the trigger is decides where the menu fits; if opened while clipped out of
                // its scroll container (e.g. keyboard activation after scrolling away), reveal it
                // before recording its anchor bounds and mounting the popup.
                d.child(
                    folo_ui::canvas(
                        move |b, window, _| {
                            if cell.get().is_none()
                                && let Some(h) = &scroll_handle
                            {
                                let vp = h.bounds();
                                if vp.size.height > px(0.)
                                    && (b.top() < vp.top() || b.bottom() > vp.bottom())
                                    && crate::app::reveal_bounds_in_scroll(b, h, window)
                                {
                                    return;
                                }
                            }
                            if cell.get() != Some(b) {
                                cell.set(Some(b));
                                // refresh() is ignored mid-draw; this schedules the next frame.
                                window.request_animation_frame();
                            }
                        },
                        |_, _, _, _| {},
                    )
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full(),
                )
            })
            .on_click(cx.listener(move |this, ev: &ClickEvent, window, cx| {
                // Enter and Space work the menu on the press (below); the release's keyboard click
                // would open what they closed, or close what they opened.
                if ev.is_keyboard() || empty {
                    return;
                }
                // The clicked select takes focus, so the keys then work its menu.
                window.focus(&this.fh(key, cx));
                let opening = this.menu_open != Some(key);
                this.menu_open = if opening { Some(key) } else { None };
                this.menu_trigger.set(None);
                this.menu_typeahead.reset_scroll_anchor();
                if opening {
                    this.export_open = false;
                }
                this.menu_hl = initial_hl;
                this.menu_typeahead.clear();
                this.menu_scroll.scroll_to_item(initial_hl);
                cx.notify();
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                let is_open = this.menu_open == Some(key);
                let mods = ev.keystroke.modifiers;
                if mods.control || mods.alt || mods.platform {
                    return;
                }
                match ev.keystroke.key.as_str() {
                    _ if n == 0 => return,
                    "enter" | "space" if is_open => {
                        let visible = this.menu_trigger.get().is_some() && !this.menu_trigger_clipped_out();
                        let pick = this.menu_hl.min(n - 1);
                        this.close_menus();
                        if visible && pick != selected && !dead[pick] {
                            keys(this, pick, window, cx);
                        }
                    }
                    "enter" | "space" | "down" | "up" if !is_open => {
                        this.menu_open = Some(key);
                        this.menu_trigger.set(None);
                        this.menu_typeahead.reset_scroll_anchor();
                        this.export_open = false;
                        this.menu_hl = initial_hl;
                        this.menu_typeahead.clear();
                        this.menu_scroll.scroll_to_item(initial_hl);
                    }
                    "down" => {
                        this.menu_typeahead.clear();
                        this.menu_hl = step(this.menu_hl.min(n - 1), true);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "up" => {
                        this.menu_typeahead.clear();
                        this.menu_hl = step(this.menu_hl.min(n - 1), false);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "pagedown" if is_open => {
                        this.menu_typeahead.clear();
                        this.menu_hl = page_step(this.menu_hl.min(n - 1), true);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "pageup" if is_open => {
                        this.menu_typeahead.clear();
                        this.menu_hl = page_step(this.menu_hl.min(n - 1), false);
                        this.menu_scroll.scroll_to_item(this.menu_hl);
                    }
                    "home" if is_open => {
                        this.menu_typeahead.clear();
                        this.menu_hl = first;
                        this.menu_scroll.scroll_to_item(first);
                    }
                    "end" if is_open => {
                        this.menu_typeahead.clear();
                        this.menu_hl = last;
                        this.menu_scroll.scroll_to_item(last);
                    }
                    "escape" if is_open => {
                        this.close_menus();
                    }
                    "tab" if is_open => {
                        this.close_menus();
                        return;
                    }
                    _ if is_open => {
                        let typed = ev.keystroke.key_char.as_deref().unwrap_or(ev.keystroke.key.as_str());
                        let mut chars = typed.chars();
                        let Some(ch) = chars
                            .next()
                            .filter(|c| chars.next().is_none() && !c.is_control() && !c.is_whitespace())
                        else {
                            return;
                        };
                        if let Some(next) = this.menu_typeahead.match_option(
                            key,
                            ch,
                            cx.background_executor().now(),
                            &labels,
                            &dead,
                            this.menu_hl.min(n - 1),
                        ) {
                            this.menu_hl = next;
                            this.menu_scroll.scroll_to_item(next);
                        }
                    }
                    _ => return,
                }
                cx.stop_propagation();
                cx.notify();
            }))
            .when(ring, |d| d.child(focus_ring(&t, radius::MD)))
            .children(trigger_class.map(|c| class_swatch(&t, c, 8.)))
            .child(
                text(if compact { ty::SM } else { ty::BASE }, t.text_primary)
                    .flex_1()
                    .ellipsis()
                    .child(current),
            )
            .child(icon(
                if open { "icons/chevron-up.svg" } else { "icons/chevron-down.svg" },
                14.,
                t.text_tertiary,
            ));
        // Below the trigger when it fits, else above; never taller than the room there (8 px margin
        // plus 4 px gap on both sides), and never extending over the title bar or dialog header.
        let want = (n as f32 * 33. + 8.).min(288.);
        let top_edge =
            self.menu_clip_bounds().map_or(tsize::TOPBAR, |c| f32::from(c.top()).max(tsize::TOPBAR)) + 12.;
        let (up, max_h) = match self.menu_trigger.get() {
            Some(b) if open => {
                let vh = f32::from(window.viewport_size().height);
                let below = (vh - f32::from(b.bottom()) - 12.).max(0.);
                let above = (f32::from(b.top()) - top_edge).max(0.);
                if below >= want || below >= above {
                    (false, below.min(288.))
                } else {
                    (true, above.min(288.))
                }
            }
            _ => (false, 288.),
        };
        let show_menu =
            open && !empty && self.menu_trigger.get().is_some() && !self.menu_trigger_clipped_out();
        // Measured on the first open frame: until then, the trigger's own width.
        let wide = self.menu_trigger.get().filter(|_| show_menu).map(|b| {
            menu_box((f32::from(b.left()), f32::from(b.right())), f32::from(window.viewport_size().width))
        });
        let menu_chrome = if show_menu {
            self.scroll_chrome(menu_scroll.clone(), t.bg_overlay, 0., radius::LG - 1., cx)
        } else {
            Vec::new()
        };
        let menu = show_menu.then(|| {
            folo_ui::deferred(
                div()
                    .absolute()
                    .when(!up, |d| d.top(px(trigger_h + 4.)))
                    .when(up, |d| d.bottom(px(trigger_h + 4.)))
                    .map(|d| match wide {
                        Some((w, left)) => d.left(px(left)).w(px(w)),
                        None => d.left_0().right_0(),
                    })
                    .occlude()
                    .on_mouse_down_out(cx.listener(move |this, ev: &folo_ui::MouseDownEvent, _, cx| {
                        let on_trigger = this.menu_trigger.get().is_some_and(|b| b.contains(&ev.position));
                        if this.menu_open == Some(key) && !on_trigger {
                            this.close_menus();
                            cx.notify();
                        }
                    }))
                    .rounded(px(radius::LG))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    .child(
                        div()
                            .id(SharedString::from(format!("{key}-menu")))
                            .max_h(px(max_h.max(64.)))
                            .overflow_y_scroll()
                            .track_scroll(&menu_scroll)
                            .p(px(4.))
                            .flex()
                            .flex_col()
                            .gap(px(1.))
                            .children(options.into_iter().enumerate().map(|(i, (label, detail))| {
                                let f = on_select.clone();
                                let on = i == selected;
                                let lit = i == hl;
                                let why = off[i].clone();
                                let dead = why.is_some();
                                let has_detail = detail.is_some();
                                let class = class_of(i);
                                div()
                                    .id(SharedString::from(format!("{key}-opt-{i}")))
                                    .flex_none()
                                    .h(px(32.))
                                    .px(px(8.))
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .rounded(px(radius::SM))
                                    .when_some(why, |d, why| d.tooltip(tip(why, None)))
                                    .when(on, |d| d.bg(t.bg_selected))
                                    .when(lit && !on, |d| d.bg(t.bg_hover))
                                    .when(lit && on, |d| d.bg(crate::theme::over(t.bg_selected, t.bg_hover)))
                                    .when(!dead, |d| {
                                        d.cursor_pointer()
                                            .hover(|s| s.bg(t.bg_hover))
                                            .active(|s| s.bg(t.bg_active))
                                            .on_mouse_move(cx.listener(
                                                move |this, _: &MouseMoveEvent, _, cx| {
                                                    if this.menu_hl != i {
                                                        this.menu_hl = i;
                                                        cx.notify();
                                                    }
                                                },
                                            ))
                                            .on_click(cx.listener(move |this, _, window, cx| {
                                                f(this, i, window, cx);
                                                this.close_menus();
                                                cx.notify();
                                            }))
                                    })
                                    .children(class.map(|c| class_swatch(&t, c, 8.)))
                                    // The name is what you choose by: it keeps its width (up to most
                                    // of the row) and the detail beside it is cut first.
                                    .child(
                                        text(ty::BASE, if dead { t.text_tertiary } else { t.text_primary })
                                            .map(|d| {
                                                if has_detail {
                                                    d.flex_none().max_w(relative(0.7))
                                                } else {
                                                    d.flex_1()
                                                }
                                            })
                                            .ellipsis()
                                            .child(label),
                                    )
                                    .when_some(detail, |d, s| {
                                        d.child(
                                            text(ty::SM, t.text_tertiary)
                                                .tnum()
                                                .flex_1()
                                                .min_w(px(0.))
                                                .text_right()
                                                .ellipsis()
                                                .child(s),
                                        )
                                    })
                                    .child(
                                        div().w(px(14.)).flex_none().when(on, |d| {
                                            d.child(icon("icons/check.svg", 14., t.accent_text))
                                        }),
                                    )
                            })),
                    )
                    .children(menu_chrome),
            )
            .with_priority(2)
        });
        div().relative().w_full().child(trigger).children(menu).into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_narrow_select_opens_a_wider_menu_toward_the_room() {
        // A 220 px select by the window's right edge: 300 px wide, right-aligned (offset -80).
        assert_eq!(menu_box((1000., 1220.), 1280.), (300., -80.));
        // The same select further left: left-aligned (offset 0).
        assert_eq!(menu_box((600., 820.), 1280.), (300., 0.));
        // A wide select keeps its own width.
        assert_eq!(menu_box((900., 1240.), 1280.), (340., 0.));
        // Narrow viewport where right-alignment would overflow the left edge: clamps inside [8, w-8].
        let (w, off) = menu_box((80., 220.), 360.);
        assert_eq!(w, 300.);
        assert!(80. + off >= 8. && 80. + off + w <= 352., "menu stays inside viewport margins: off={off}");
        // Very narrow viewport narrower than MENU_MIN_W + 16: width caps to viewport - 16.
        let (w, off) = menu_box((20., 200.), 260.);
        assert_eq!(w, 244.);
        assert_eq!(20. + off, 8.);
        assert_eq!(20. + off + w, 252.);
        // Trigger partially past the right viewport edge still clamps menu strictly inside [8, w-8].
        let (w, off) = menu_box((200., 300.), 260.);
        assert_eq!(w, 244.);
        assert_eq!(200. + off, 8.);
        assert_eq!(200. + off + w, 252.);
    }

    #[test]
    fn typeahead_matches_prefix_cycles_repeats_skips_disabled_and_expires() {
        let labels: Vec<SharedString> =
            ["Alpha", "Beta", "Bison", "Bravo", "Gamma"].into_iter().map(SharedString::from).collect();
        let disabled = [false, false, true, false, false];
        let mut ta = SelectTypeahead::default();
        let t0 = std::time::Instant::now();

        // Single letter jumps to first enabled match ("Beta").
        assert_eq!(ta.match_option("m", 'b', t0, &labels, &disabled, 0), Some(1));
        // Repeated same letter skips disabled "Bison" and cycles to "Bravo".
        assert_eq!(
            ta.match_option("m", 'b', t0 + std::time::Duration::from_millis(120), &labels, &disabled, 1),
            Some(3)
        );
        // Next press wraps back to "Beta".
        assert_eq!(
            ta.match_option("m", 'b', t0 + std::time::Duration::from_millis(240), &labels, &disabled, 3),
            Some(1)
        );

        // Multi-character prefix ("br") jumps directly to "Bravo".
        ta.clear();
        assert_eq!(ta.match_option("m", 'b', t0, &labels, &disabled, 0), Some(1));
        assert_eq!(
            ta.match_option("m", 'r', t0 + std::time::Duration::from_millis(150), &labels, &disabled, 1),
            Some(3)
        );

        // Extended prefix miss ("brg") falls back to fresh single-letter lookup ("Gamma").
        assert_eq!(
            ta.match_option("m", 'g', t0 + std::time::Duration::from_millis(250), &labels, &disabled, 3),
            Some(4)
        );

        // After the 800 ms window, typing starts a fresh prefix from index 0.
        assert_eq!(
            ta.match_option("m", 'a', t0 + std::time::Duration::from_millis(1200), &labels, &disabled, 4),
            Some(0)
        );

        // Repeated identical letter cycles through every match even when one starts with doubled letters.
        let a_labels: Vec<SharedString> =
            ["Aardvark", "Alpaca", "Ant"].into_iter().map(SharedString::from).collect();
        let a_disabled = [false, false, false];
        ta.clear();
        assert_eq!(ta.match_option("a", 'a', t0, &a_labels, &a_disabled, 0), Some(1));
        assert_eq!(
            ta.match_option("a", 'a', t0 + std::time::Duration::from_millis(100), &a_labels, &a_disabled, 1),
            Some(2)
        );
        assert_eq!(
            ta.match_option("a", 'a', t0 + std::time::Duration::from_millis(200), &a_labels, &a_disabled, 2),
            Some(0)
        );
    }
}
