//! The sidebar: the FOLO mark, section navigation with its shortcuts, the live run card and the
//! connection line.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, ClickEvent, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, WindowControlArea, div, px,
};

use crate::api::*;
use crate::store::*;
use crate::theme::{Theme, size as tsize, theme, ty};
use crate::ui::*;

use super::focus_ring;
use super::titlebar::TITLEBAR;
use super::{FoloApp, Route, Section, act};

impl FoloApp {
    pub(super) fn sidebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        // Narrow windows collapse the sidebar to an icon rail.
        let compact = window.viewport_size().width < px(1120.);
        let section = self.route.section();
        let live = self.store.live_run().cloned();
        // The Runs badge: how many runs train and how many wait (UNIFIED.md §1, as Studio counts).
        let counts = self.store.runs.ready().map_or(RunsBadge::default(), |r| RunsBadge::of(r));
        let nav = [
            (Section::Overview, "Overview", "icons/layout-dashboard.svg", Route::Overview, "1"),
            (Section::Datasets, "Datasets", "icons/images.svg", Route::Datasets, "2"),
            (Section::Runs, "Runs", "icons/activity.svg", Route::Runs, "3"),
            (Section::Models, "Models", "icons/box.svg", Route::Models, "4"),
            (Section::Benchmarks, "Benchmarks", "icons/target.svg", Route::Benches, "6"),
            (Section::Playground, "Playground", "icons/scan-eye.svg", Route::Playground, "5"),
        ];
        let items: Vec<AnyElement> = nav
            .into_iter()
            .map(|(sec, label, icon_path, route, key)| {
                let badge = (sec == Section::Runs && counts.total() > 0).then_some(counts);
                self.nav_item(label, icon_path, section == sec, route, key, badge, compact, window, cx)
            })
            .collect();
        let settings = self.nav_item(
            "Settings",
            "icons/settings.svg",
            section == Section::Settings,
            Route::Settings,
            ",",
            None,
            compact,
            window,
            cx,
        );
        // The live run's row on every page while a run trains, its own page and Overview included:
        // the chrome doesn't change as you move between pages (as in Studio).
        let live_card = live
            .map(|r| if compact { self.live_rail(&r, window, cx) } else { self.live_mini(&r, window, cx) });
        let brand_press = self.leave_kb(cx);
        let health = self.connection_line(compact, window, cx);
        div()
            .w(px(if compact { tsize::SIDEBAR_COLLAPSED } else { tsize::SIDEBAR }))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .bg(t.bg_panel)
            .border_r_1()
            .border_color(t.line_subtle)
            .child(
                // Brand row doubles as a drag area for the window.
                div()
                    .id("brand")
                    .h(px(TITLEBAR))
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .px(px(if compact { 18. } else { 16. }))
                    .window_control_area(WindowControlArea::Drag)
                    // See `titlebar-drag`.
                    .occlude()
                    .capture_any_mouse_down(brand_press)
                    .when(compact, |d| d.child(logo(&t, 20.)))
                    .when(!compact, |d| {
                        d.child(
                            // The wordmark has no text baseline to align on: bottoms together, and it is
                            // lifted by what the `md` line keeps under its baseline.
                            div()
                                .flex()
                                .items_end()
                                .gap(px(6.))
                                .child(div().pb(px(MD_UNDER_BASELINE)).child(wordmark(&t, 16.)))
                                .child(text(ty::MD, t.text_tertiary).child("Desktop")),
                        )
                    }),
            )
            .child(div().flex().flex_col().gap(px(2.)).px(px(8.)).pt(px(8.)).children(items))
            .child(div().flex_1())
            .children(live_card)
            .child(div().px(px(8.)).pb(px(4.)).child(settings))
            .child(health)
    }

    #[allow(clippy::too_many_arguments)]
    fn nav_item(
        &mut self,
        label: &'static str,
        icon_path: &'static str,
        active: bool,
        route: Route,
        key: &'static str,
        badge: Option<RunsBadge>,
        compact: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(format!("nav-{label}"), cx);
        let ring = self.ring(&fh, window);
        let r2 = route.clone();
        div()
            .id(SharedString::from(format!("nav-{label}")))
            .track_focus(&fh)
            .relative()
            .h(px(30.))
            .px(px(8.))
            .flex()
            .items_center()
            .gap(px(10.))
            .rounded(px(6.))
            .cursor_pointer()
            .when(active, |d| d.bg(t.bg_active))
            .when(!active, |d| d.hover(|s| s.bg(t.bg_hover)))
            .active(|s| s.bg(t.bg_active))
            // Enter and Space navigate on the press (below), not again on the release's keyboard click.
            .on_click(cx.listener(move |this, ev: &ClickEvent, w, cx| {
                if !ev.is_keyboard() {
                    this.navigate(route.clone(), w, cx)
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, w, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    this.navigate(r2.clone(), w, cx);
                    cx.stop_propagation();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, 6.)))
            // The icon rail names the focused item, as its tooltip does for the pointer.
            .when(compact && ring, |d| {
                d.child(crate::controls::focus_tip(&t, format!("{label} · {}", shortcut_label(key)), 34.))
            })
            .when(compact, |d| {
                d.justify_center().px(px(0.)).tooltip(crate::controls::tip(
                    badge.map_or(label.to_string(), |b| b.text(label)),
                    Some(shortcut_label(key)),
                ))
            })
            .child(icon(icon_path, 16., if active { t.text_primary } else { t.text_tertiary }))
            // The icon rail has no room for the count: a lime dot while a run trains (now, so lime).
            .when(compact && badge.is_some_and(|b| b.training > 0), |d| {
                d.child(div().absolute().top(px(6.)).right(px(8.)).child(dot(
                    t.accent_fill(),
                    6.,
                    false,
                    "nav-live-c",
                )))
            })
            .when(!compact, |d| {
                d.child(
                    text(ty::BASE, if active { t.text_primary } else { t.text_secondary })
                        .when(active, |d| d.font_weight(FontWeight::MEDIUM))
                        .flex_1()
                        .child(label),
                )
                // How many runs train or wait, as a 20 px count chip in the accent: the number alone,
                // since the sidebar's live row names the run and the tooltip says what is counted.
                .when_some(badge, |d, b| {
                    d.child(
                        text(ty::SM, t.accent_text)
                            .tnum()
                            .font_weight(FontWeight::MEDIUM)
                            .h(px(20.))
                            .min_w(px(20.))
                            .px(px(6.))
                            .flex()
                            .flex_none()
                            .items_center()
                            .justify_center()
                            .rounded(px(10.))
                            .bg(t.accent_muted)
                            .child(b.total().to_string()),
                    )
                })
                .tooltip(crate::controls::tip(
                    badge.map_or(label.to_string(), |b| b.text(label)),
                    Some(shortcut_label(key)),
                ))
            })
            .into_any_element()
    }

    fn live_mini(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let id = r.id.clone();
        let fh = self.fh("live-mini", cx);
        let ring = self.ring(&fh, window);
        let id2 = id.clone();
        // A single footer row: the run's name, how far it is, and its progress in the accent (the bar is the live
        // mark; the nav's count says how many train).
        div()
            .id("live-mini")
            .track_focus(&fh)
            .relative()
            .mx(px(8.))
            .mb(px(6.))
            .h(px(36.))
            .px(px(10.))
            .flex()
            .flex_col()
            .justify_center()
            .gap(px(6.))
            .rounded(px(crate::theme::radius::MD))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(format!("{} · epoch {} / {}", r.name, r.epoch, r.epochs), None))
            .on_click(cx.listener(move |this, ev: &ClickEvent, w, cx| {
                if !ev.is_keyboard() {
                    this.navigate(Route::Run(id.clone()), w, cx)
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, w, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    this.navigate(Route::Run(id2.clone()), w, cx);
                    cx.stop_propagation();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, crate::theme::radius::MD)))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_primary).flex_1().min_w_0().ellipsis().child(r.name.clone()))
                    .child(
                        text(ty::SM, t.text_secondary).tnum().child(format!("{:.0}%", r.progress() * 100.0)),
                    ),
            )
            .child(progress(&t, r.progress(), t.accent_fill(), 2.))
            .into_any_element()
    }

    /// The live run as a rail button: how far it is, over its progress bar.
    fn live_rail(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let id = r.id.clone();
        let el = div()
            .id("live-rail")
            .mx(px(10.))
            .mb(px(8.))
            .h(px(40.))
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(px(5.))
            .rounded(px(8.))
            .bg(t.bg_raised)
            .border_1()
            .border_color(t.line_subtle)
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_default))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(format!("{} · epoch {} / {}", r.name, r.epoch, r.epochs), None))
            .child(text(ty::SM, t.text_secondary).tnum().child(format!("{:.0}%", r.progress() * 100.)))
            .child(div().w(px(24.)).child(progress(&t, r.progress(), t.accent_fill(), 2.)));
        self.bind(
            el,
            "live-rail",
            8.,
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
        )
        .into_any_element()
    }

    fn connection_line(&self, compact: bool, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        let (color, glyph, label) = match &self.store.health {
            _ if self.connecting.is_some() => (
                t.warning_solid,
                "icons/refresh-cw.svg",
                self.connecting.clone().unwrap_or_default().to_string(),
            ),
            Load::Ready(h) => (
                t.success_solid,
                "icons/circle-check.svg",
                format!("{} · v{}", self.store.api.describe(), h.version),
            ),
            Load::Failed(e) if e.code == "unauthorized" => {
                (t.danger_solid, "icons/circle-x.svg", "Server needs a token".to_string())
            }
            // As Settings says: nothing was sent, so no server is down.
            Load::Failed(e) if e.is_bad_address() => {
                (t.danger_solid, "icons/circle-x.svg", "Not a server address".to_string())
            }
            Load::Failed(_) => (t.danger_solid, "icons/circle-x.svg", "Server unreachable".to_string()),
            _ => (t.text_disabled, "icons/refresh-cw.svg", "Connecting…".to_string()),
        };
        // A control, by pointer and keyboard: it opens Connection settings, and the collapsed rail's
        // glyph (a shape, not colour alone) shows the full status on hover and on focus.
        let tip = format!("Connection: {label}");
        let ring = self.ring(&self.fh("connection", cx), window);
        let el = div()
            .id("connection")
            .relative()
            .h(px(36.))
            .flex_none()
            .flex()
            .items_center()
            .gap(px(8.))
            .px(px(16.))
            .border_t_1()
            .border_color(t.line_subtle)
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(tip.clone(), None))
            .when(compact, |d| {
                d.justify_center()
                    .px(px(0.))
                    .child(icon(glyph, 14., color))
                    .when(ring, |d| d.child(crate::controls::focus_tip(&t, tip.clone(), 30.)))
            })
            .when(!compact, |d| {
                d.child(div().size(px(6.)).rounded_full().bg(color))
                    .child(text(ty::SM, t.text_tertiary).ellipsis().child(label))
            });
        self.bind(el, "connection", 0., window, cx, act(|this, w, cx| this.navigate(Route::Settings, w, cx)))
    }
}

fn shortcut_label(key: &str) -> &'static str {
    match key {
        "1" => "Ctrl 1",
        "2" => "Ctrl 2",
        "3" => "Ctrl 3",
        "4" => "Ctrl 4",
        "5" => "Ctrl 5",
        "6" => "Ctrl 6",
        _ => "Ctrl ,",
    }
}

/// What a line of `ty::MD` text keeps under its baseline, in px (the descent and half the leading).
const MD_UNDER_BASELINE: f32 = 5.;

/// The FOLO mark (design/logo.svg): the f and the signal dot, the last o of "folo".
pub fn logo(t: &Theme, size: f32) -> impl IntoElement {
    div()
        .relative()
        .size(px(size))
        .flex_none()
        .child(folo_ui::svg().path("icons/logo-mark.svg").size(px(size)).text_color(t.text_primary))
        .child(
            div()
                .absolute()
                .top(px(size * (14.75 - 6.375) / 24.0))
                .left(px(size * (17.25 - 6.375) / 24.0))
                .size(px(size * 12.75 / 24.0))
                .rounded_full()
                .bg(t.accent_solid),
        )
}

/// The FOLO wordmark (design/logo-wordmark.svg), `height` px tall: "folo" with its last o as the signal dot.
/// Its bottom edge is the letters' baseline.
fn wordmark(t: &Theme, height: f32) -> impl IntoElement {
    let width = height * 207.0 / 93.0;
    div()
        .relative()
        .w(px(width))
        .h(px(height))
        .flex_none()
        .child(
            folo_ui::svg()
                .path("icons/logo-wordmark.svg")
                .w(px(width))
                .h(px(height))
                .text_color(t.text_primary),
        )
        .child(
            div()
                .absolute()
                .top(px(height * (61.0 - 31.0) / 93.0))
                .left(px(height * (176.0 - 31.0) / 93.0))
                .size(px(height * 62.0 / 93.0))
                .rounded_full()
                .bg(t.accent_solid),
        )
}

/// The Runs badge's counts: runs training (stopping included) and runs waiting in the queue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RunsBadge {
    pub training: usize,
    pub queued: usize,
}

impl RunsBadge {
    pub fn of(runs: &[Run]) -> Self {
        let n = |f: fn(RunStatus) -> bool| runs.iter().filter(|r| f(r.status)).count();
        RunsBadge {
            training: n(|s| matches!(s, RunStatus::Running | RunStatus::Stopping)),
            queued: n(|s| s == RunStatus::Queued),
        }
    }

    /// The number on the badge.
    pub fn total(self) -> usize {
        self.training + self.queued
    }

    /// The row's name with the badge in words ("Runs · 1 training · 2 queued"), for its tooltip.
    pub fn text(self, label: &str) -> String {
        let mut s = label.to_string();
        if self.training > 0 {
            s += &format!(" · {} training", self.training);
        }
        if self.queued > 0 {
            s += &format!(" · {} queued", self.queued);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::RunsBadge;

    #[test]
    fn the_runs_badge_counts_training_and_queued_runs() {
        let b = RunsBadge { training: 1, queued: 2 };
        assert_eq!(b.total(), 3);
        assert_eq!(b.text("Runs"), "Runs · 1 training · 2 queued");
        assert_eq!(RunsBadge { training: 0, queued: 1 }.text("Runs"), "Runs · 1 queued");
        assert_eq!(RunsBadge::default().total(), 0);
    }
}
