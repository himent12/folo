//! Keyboard focus: a stable handle per control, focus rings and scrolling a focused control into
//! view, Tab and F6 cycling (trapped inside the topmost overlay), and leaving keyboard mode on a
//! press.

use std::cell::RefCell;

use folo_ui::{
    App, Context, FocusHandle, Focusable, ParentElement, Pixels, ScrollHandle, SharedString, Styled, Window,
    div, px,
};

use crate::theme::{Theme, size as tsize};

use super::{FoloApp, Route, Section};

impl FoloApp {
    // ---------- focus ----------

    /// A stable focus handle for a control key; tab order follows paint order.
    pub fn fh(&self, key: impl Into<SharedString>, cx: &mut Context<Self>) -> FocusHandle {
        let key = key.into();
        self.focus.borrow_mut().entry(key).or_insert_with(|| cx.focus_handle().tab_stop(true)).clone()
    }

    /// The handle of a control that comes and goes, as a failed image's Retry: a Tab stop while it
    /// is `shown`. Gone, one it had stays for the caller to track as no stop, so focus left on it
    /// (Enter retried and the Retry went) keeps its place in Tab order instead of starting over.
    pub fn passing_stop(&self, key: &str, shown: bool, cx: &mut Context<Self>) -> Option<FocusHandle> {
        if !shown && !self.focus.borrow().contains_key(key) {
            return None;
        }
        // The flag lives in the handle's shared entry, so `bind`'s clone reads it back.
        Some(self.fh(key.to_string(), cx).tab_stop(shown))
    }

    /// Names the keys of the focused control in the strip under the page (see `key_hint`).
    pub fn hint_keys(&self, keys: impl Into<SharedString>) {
        *self.key_hint.borrow_mut() = Some(keys.into());
    }

    /// Keyboard focus is on this handle and should show a ring.
    pub fn ring(&self, fh: &FocusHandle, window: &Window) -> bool {
        self.kb && fh.is_focused(window)
    }

    pub fn focus_key(&mut self, key: &str, window: &mut Window, cx: &mut Context<Self>) {
        let fh = self.fh(key.to_string(), cx);
        window.focus(&fh);
    }

    /// Focuses `key` in keyboard mode and schedules a focus-ring scroll reveal on `page_scroll`.
    pub fn focus_key_and_reveal(&mut self, key: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.kb = true;
        self.focus_key(key, window, cx);
        REVEAL.with(|r| *r.borrow_mut() = Some(vec![self.page_scroll.clone()]));
        cx.notify();
    }

    /// A press leaves keyboard mode (focus rings off). The root listens for presses anywhere, but
    /// the window chrome hides the root from the hit test (`titlebar-drag`), so it listens too.
    /// Only a capture-phase listener: it never prevents the press's default.
    pub(super) fn leave_kb(&self, cx: &mut Context<Self>) -> PressListener {
        Box::new(cx.listener(|this, _: &folo_ui::MouseDownEvent, _, cx| {
            if this.kb {
                this.kb = false;
                cx.notify();
            }
        }))
    }

    /// Whether the focused element belongs to the topmost overlay (always true without one).
    fn focus_in_scope(&self, window: &Window, cx: &App) -> bool {
        let has_prefix = |prefixes: &[&str]| {
            self.focus
                .borrow()
                .iter()
                .any(|(k, fh)| prefixes.iter().any(|p| k.starts_with(p)) && fh.is_focused(window))
        };
        if self.confirm_close.is_some() {
            has_prefix(&["close-", "dialog-close"])
        } else if let Some(p) = &self.palette {
            p.input.read(cx).focus_handle(cx).is_focused(window)
        } else if self.confirm_delete.is_some() {
            has_prefix(&["del-", "dialog-close"])
        } else if self.confirm_stop.is_some() {
            has_prefix(&["stop-", "dialog-close"])
        } else if self.cloud.disconnect.is_some() {
            has_prefix(&["cd-", "dialog-close"])
        } else if self.cloud.terminate.is_some() {
            has_prefix(&["ct-", "dialog-close"])
        } else if let Some(f) = &self.cloud.connect {
            has_prefix(&["cc-", "dialog-close"])
                || f.fields.iter().chain([&f.name]).any(|i| i.read(cx).focus_handle(cx).is_focused(window))
        } else if let Some(f) = &self.new_run {
            has_prefix(&["nr-", "dialog-close"])
                || [&f.name, &f.epochs_input, &f.lr_input, &f.close_input, &f.hours_input]
                    .iter()
                    .any(|i| i.read(cx).focus_handle(cx).is_focused(window))
        } else if let Some(f) = &self.synth {
            has_prefix(&["syn-", "dialog-close"])
                || [&f.name, &f.images].iter().any(|i| i.read(cx).focus_handle(cx).is_focused(window))
        } else if self.inspector.is_some() {
            has_prefix(&["insp-"]) || self.inspector_focus.is_focused(window)
        } else {
            true
        }
    }

    /// Tab / Shift-Tab, trapped inside the topmost overlay.
    /// F6 / Shift+F6: the sidebar (the current section's item), the search field, then the page
    /// (its first control). Dialogs and the palette keep focus to themselves.
    pub(super) fn cycle_region(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.modal_open() || self.palette.is_some() {
            return;
        }
        self.menu_open = None;
        self.menu_typeahead.clear();
        self.export_open = false;
        let nav = format!(
            "nav-{}",
            match self.route.section() {
                Section::Overview => "Overview",
                Section::Datasets => "Datasets",
                Section::Runs => "Runs",
                Section::Models => "Models",
                Section::Benchmarks => "Benchmarks",
                Section::Playground => "Playground",
                Section::Settings => "Settings",
            }
        );
        // The sidebar region: its navigation and the connection line at its foot.
        let in_nav = self
            .focus
            .borrow()
            .iter()
            .any(|(k, f)| (k.starts_with("nav-") || k.as_ref() == "connection") && f.is_focused(window));
        let in_search = self.fh("titlebar-search", cx).is_focused(window);
        let at = if in_nav {
            0
        } else if in_search {
            1
        } else {
            2
        };
        let next = if forward { (at + 1) % 3 } else { (at + 2) % 3 };
        self.kb = true;
        match next {
            0 => window.focus(&self.fh(nav, cx)),
            1 => window.focus(&self.fh("titlebar-search", cx)),
            _ => {
                window.focus(&self.fh("page-region", cx));
                self.cycle_focus(true, window, cx);
            }
        }
        cx.notify();
    }

    pub(super) fn cycle_focus(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        // The newly focused control scrolls into view inside whichever of these holds it.
        let containers = if self.inspector.is_some() {
            vec![self.insp_list_scroll.clone()]
        } else if self.new_run.is_some() || self.synth.is_some() || self.cloud.connect.is_some() {
            vec![self.dialog_scroll.clone()]
        } else if matches!(self.route, Route::Playground) {
            vec![self.play_side_scroll.clone(), self.page_scroll.clone()]
        } else if matches!(self.route, Route::Model(_)) {
            // A class row scrolls into view inside its table first.
            vec![self.pc_scroll.clone(), self.page_scroll.clone()]
        } else {
            vec![self.page_scroll.clone()]
        };
        REVEAL.with(|r| *r.borrow_mut() = Some(containers));
        cx.notify();
        for _ in 0..512 {
            if forward {
                window.focus_next()
            } else {
                window.focus_prev()
            }
            if self.focus_in_scope(window, cx) {
                return;
            }
        }
    }
}

thread_local! {
    /// Scroll containers to search after a keyboard focus move, innermost first. The focus ring
    /// that paints next scrolls its control into view in the first container that holds it.
    pub(super) static REVEAL: RefCell<Option<Vec<ScrollHandle>>> = const { RefCell::new(None) };
}

/// Scroll each candidate container holding `b` (horizontally), innermost first, so `b` is fully
/// visible, clear of the edge fades (24 px, so 32 at the top) and of the key-hint strip over the
/// page's bottom (32 px, so 40 there): a row in a list that scrolls inside the page comes into the
/// list's view, then the page brings the list's row into its own. Runs once per request.
fn reveal(mut b: folo_ui::Bounds<Pixels>, window: &mut Window) {
    let Some(handles) = REVEAL.with(|r| r.borrow_mut().take()) else { return };
    let margin = px(8.);
    let (top_m, bottom_m) = (px(32.), px(40.));
    // The innermost clip around the control: only a container that contains it owns the control
    // (a dialog's fixed header and footer sit outside the scrolling body).
    let mask = window.content_mask().bounds;
    let tol = px(1.);
    for h in handles {
        let vp = h.bounds();
        let inside = mask.left() >= vp.left() - tol
            && mask.right() <= vp.right() + tol
            && mask.top() >= vp.top() - tol
            && mask.bottom() <= vp.bottom() + tol;
        if vp.size.width <= px(0.) || !inside {
            continue;
        }
        let tall = b.size.height + top_m + bottom_m > vp.size.height;
        let dy = if b.top() - top_m < vp.top() && !tall {
            b.top() - top_m - vp.top()
        } else if tall {
            // Taller than the view (the run page's linked chart group at minimum size): its top
            // leads. Already in view, it stays; elsewhere, it comes to the top edge, never past.
            if b.top() < vp.top() || b.top() + margin > vp.bottom() - margin * 4. {
                b.top() - top_m - vp.top()
            } else {
                px(0.)
            }
        } else if b.bottom() + bottom_m > vp.bottom() {
            b.bottom() + bottom_m - vp.bottom()
        } else {
            px(0.)
        };
        if dy != px(0.) {
            let off = h.offset();
            let y = (off.y - dy).clamp(-h.max_offset().height, px(0.));
            h.set_offset(folo_ui::point(off.x, y));
            window.refresh();
            // Where the control is now, for the containers around this one.
            b.origin.y -= off.y - y;
        }
    }
}

/// Scroll `h` so `b` is visible within its viewport (clear of edge fades and the bottom key-hint
/// strip), independent of keyboard focus rings. Runs during prepaint and schedules a follow-up
/// frame when the scroll offset changes.
pub(crate) fn reveal_bounds_in_scroll(b: folo_ui::Bounds<Pixels>, h: &ScrollHandle, window: &mut Window) {
    let vp = h.bounds();
    if vp.size.width <= px(0.) || vp.size.height <= px(0.) {
        return;
    }
    let margin = px(8.);
    let (top_m, bottom_m) = (px(16.), px(40.));
    let tall = b.size.height + top_m + bottom_m > vp.size.height;
    let dy = if b.top() - top_m < vp.top() && !tall {
        b.top() - top_m - vp.top()
    } else if tall {
        if b.top() - top_m < vp.top() || b.top() + margin > vp.bottom() - margin * 4. {
            b.top() - top_m - vp.top()
        } else {
            px(0.)
        }
    } else if b.bottom() + bottom_m > vp.bottom() {
        b.bottom() + bottom_m - vp.bottom()
    } else {
        px(0.)
    };
    if dy != px(0.) {
        let off = h.offset();
        // Clamp only against top (0) here; prepaint clamps against the freshly laid-out max_offset.
        let y = (off.y - dy).min(px(0.));
        if (f32::from(y) - f32::from(off.y)).abs() > 0.5 {
            h.set_offset(folo_ui::point(off.x, y));
            window.request_animation_frame();
        }
    }
}

/// Keyboard focus ring: 2 px in the focus colour, 2 px outside the control.
pub fn focus_ring(t: &Theme, radius: f32) -> folo_ui::Div {
    let off = tsize::FOCUS_OFFSET + tsize::FOCUS_RING;
    div()
        .absolute()
        .top(px(-off))
        .left(px(-off))
        .right(px(-off))
        .bottom(px(-off))
        .rounded(px(radius + off))
        .border_2()
        .border_color(t.focus)
        // Keyboard focus moved here: make sure the control (and its ring) is on screen.
        .child(folo_ui::canvas(|b, window, _| reveal(b, window), |_, _, _, _| {}).size_full())
}

/// A mouse-press listener, as `capture_any_mouse_down` takes it.
type PressListener = Box<dyn Fn(&folo_ui::MouseDownEvent, &mut Window, &mut App)>;
