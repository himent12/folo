//! The title bar: breadcrumbs, search, the window's caption buttons, and the caption areas that
//! stay above a scrim.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    ClickEvent, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, WindowControlArea, div, px,
};

use crate::theme::{size as tsize, theme, ty};
use crate::ui::*;

use super::focus_ring;
use super::{FoloApp, Route, act};

/// The title bar's height, and the sidebar's brand row's (the other drag area).
pub(super) const TITLEBAR: f32 = tsize::TOPBAR - 8.;

/// The title bar's row, its hairline (uncoloured) included. `titlebar()` and `caption_controls`
/// both lay out on it, so their caption buttons are the same size in the same place.
fn titlebar_row() -> folo_ui::Div {
    div().h(px(TITLEBAR)).flex_none().flex().items_center().border_b_1()
}

impl FoloApp {
    pub(super) fn titlebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        let crumbs = self.breadcrumbs();
        let maximized = window.is_maximized();
        let fullscreen = window.is_fullscreen();
        let win_btn = |base: folo_ui::Stateful<folo_ui::Div>,
                       id: &'static str,
                       icon_path: &'static str,
                       close: bool| {
            base.group(id)
                .flex()
                .items_center()
                .justify_center()
                .hover(move |s| if close { s.bg(t.window_close) } else { s.bg(t.bg_hover) })
                .active(|s| s.opacity(0.8))
                .child(
                    folo_ui::svg()
                        .path(icon_path)
                        .size(px(10.))
                        .flex_none()
                        .text_color(t.text_secondary)
                        .group_hover(id, move |s| {
                            s.text_color(if close { t.window_close_text } else { t.text_primary })
                        }),
                )
        };
        let search_fh = self.fh("titlebar-search", cx);
        let search_ring = self.ring(&search_fh, window);
        let min_btn = self.caption_button("win-min", WindowControlArea::Min, fullscreen, cx);
        let max_btn = self.caption_button("win-max", WindowControlArea::Max, fullscreen, cx);
        let close_btn = self.caption_button("win-close", WindowControlArea::Close, fullscreen, cx);
        titlebar_row()
            .border_color(t.line_subtle)
            .child(
                // Breadcrumbs (draggable area).
                div()
                    .id("titlebar-drag")
                    .flex_1()
                    .h_full()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    // The crumbs' 4 px pads put their text on the page gutter.
                    .pl(px(self.gutter() - 4.))
                    .window_control_area(WindowControlArea::Drag)
                    // A caption press goes to the app first and only reaches Windows (drag, double-click
                    // maximize, the caption buttons) if nothing prevents its default. The focus-tracked
                    // root would (focusing itself does), so the chrome hides the root from the hit
                    // test. The crumbs above it still take their own clicks. Hidden from the root, it
                    // leaves keyboard mode itself, as the root does for any other press.
                    .occlude()
                    .capture_any_mouse_down(self.leave_kb(cx))
                    .children(crumbs.into_iter().enumerate().map(|(i, (label, route))| {
                        let last = route.is_none();
                        let crumb = div().id(("crumb", i)).px(px(4.)).py(px(2.)).rounded(px(4.)).child(
                            text(ty::BASE, if last { t.text_primary } else { t.text_tertiary })
                                .when(last, |d| d.font_weight(FontWeight::MEDIUM))
                                .whitespace_nowrap()
                                .child(label),
                        );
                        let crumb = match route {
                            Some(r) => self
                                .bind(
                                    crumb
                                        .cursor_pointer()
                                        .hover(|s| s.bg(t.bg_hover))
                                        .active(|s| s.bg(t.bg_active)),
                                    &format!("crumb-{i}"),
                                    4.,
                                    window,
                                    cx,
                                    act(move |this, w, cx| this.navigate(r.clone(), w, cx)),
                                )
                                .into_any_element(),
                            None => crumb.into_any_element(),
                        };
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .when(i > 0, |d| d.child(icon("icons/chevron-right.svg", 12., t.text_disabled)))
                            .child(crumb)
                    })),
            )
            .child(
                div()
                    .id("titlebar-search")
                    .track_focus(&search_fh)
                    .relative()
                    .w(px(280.))
                    .h(px(28.))
                    .mr(px(12.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(6.))
                    .bg(t.bg_raised)
                    .border_1()
                    .border_color(t.line_default)
                    .cursor_pointer()
                    .hover(|s| s.border_color(t.line_strong))
                    .active(|s| s.bg(t.bg_active))
                    // Enter and Space open it on the press (below), not again on the release.
                    .on_click(cx.listener(|this, ev: &ClickEvent, w, cx| {
                        if !ev.is_keyboard() {
                            this.open_palette(w, cx)
                        }
                    }))
                    .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, w, cx| {
                        if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                            this.open_palette(w, cx);
                            cx.stop_propagation();
                        }
                    }))
                    .when(search_ring, |d| d.child(focus_ring(&t, 6.)))
                    .child(icon("icons/search.svg", 14., t.text_tertiary))
                    .child(text(ty::SM, t.text_tertiary).flex_1().child("Search or jump to…"))
                    .child(shortcut(&t, &["Ctrl", "K"])),
            )
            .child(win_btn(min_btn, "win-min", "icons/win-min.svg", false))
            .child(win_btn(
                max_btn,
                "win-max",
                if maximized || fullscreen { "icons/win-restore.svg" } else { "icons/win-max.svg" },
                false,
            ))
            .child(win_btn(close_btn, "win-close", "icons/win-close.svg", true))
    }

    /// A caption button's hit area, `layout.desktop_caption_button` wide and the title bar's
    /// height: Minimize, Maximize / Restore or Close for Windows. In full screen folo-ui reports no
    /// caption areas, so Windows doesn't act on it: it takes its own click then. Only then, since
    /// otherwise the caption press already acts (on mouse up) and a click handler would act a
    /// second time. `titlebar()` draws on it; `caption_controls` uses it bare.
    fn caption_button(
        &self,
        id: &'static str,
        area: WindowControlArea,
        fullscreen: bool,
        cx: &mut Context<Self>,
    ) -> folo_ui::Stateful<folo_ui::Div> {
        div()
            .id(id)
            .w(px(crate::theme::layout::DESKTOP_CAPTION_BUTTON))
            .h_full()
            .window_control_area(area)
            // See `titlebar-drag`: the press must reach Windows, not focus the root.
            .occlude()
            .capture_any_mouse_down(self.leave_kb(cx))
            .when(fullscreen, |d| match area {
                WindowControlArea::Min => d.on_click(|_, w, _| w.minimize_window()),
                // Full screen's "restore" leaves full screen.
                WindowControlArea::Max => d.on_click(|_, w, _| w.toggle_fullscreen()),
                // Closing this way skips the window's should-close hook: ask it here (it remembers the
                // window, or asks first when closing would stop training).
                WindowControlArea::Close => d.on_click(cx.listener(|this, _, w, cx| {
                    if this.should_close(w, cx) {
                        w.remove_window();
                    }
                })),
                WindowControlArea::Drag => d,
            })
    }

    /// The window's caption areas again, bare, for an overlay to put over its scrim and under its
    /// own panel. The scrim dims the title bar, but it also hides the title bar from the hit test,
    /// and Windows acts only on caption areas in it: a window must still move, minimize, maximize
    /// and close. One drag band across the top (the sidebar's brand row and the title bar, whose
    /// dimmed crumbs and search field take no clicks under an overlay) and the three buttons,
    /// where `titlebar()` puts them. A press here neither reaches the scrim (which would dismiss
    /// the overlay) nor the page.
    pub fn caption_controls(
        &self,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> folo_ui::Stateful<folo_ui::Div> {
        let fullscreen = window.is_fullscreen();
        titlebar_row()
            .id("caption-controls")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .child(
                div()
                    .id("caption-drag")
                    .flex_1()
                    .h_full()
                    .window_control_area(WindowControlArea::Drag)
                    // See `titlebar-drag`.
                    .occlude()
                    .capture_any_mouse_down(self.leave_kb(cx)),
            )
            .child(self.caption_button("caption-min", WindowControlArea::Min, fullscreen, cx))
            .child(self.caption_button("caption-max", WindowControlArea::Max, fullscreen, cx))
            .child(self.caption_button("caption-close", WindowControlArea::Close, fullscreen, cx))
    }

    fn breadcrumbs(&self) -> Vec<(SharedString, Option<Route>)> {
        match &self.route {
            // Top-level pages carry their own title; the titlebar stays quiet.
            Route::Overview
            | Route::Datasets
            | Route::Runs
            | Route::Models
            | Route::Benches
            | Route::Playground
            | Route::Settings => {
                vec![]
            }
            Route::Dataset(id) => {
                vec![("Datasets".into(), Some(Route::Datasets)), (self.store.dataset_name(id).into(), None)]
            }
            Route::Run(id) => vec![
                ("Runs".into(), Some(Route::Runs)),
                (self.store.run(id).map_or(id.clone(), |r| r.name.clone()).into(), None),
            ],
            Route::Model(id) => vec![
                ("Models".into(), Some(Route::Models)),
                (self.store.model(id).map_or(id.clone(), |m| m.name.clone()).into(), None),
            ],
            Route::Bench(id) => {
                vec![("Benchmarks".into(), Some(Route::Benches)), (self.bench_name(id).into(), None)]
            }
            Route::BenchEntry(b, e) => vec![
                ("Benchmarks".into(), Some(Route::Benches)),
                (self.bench_name(b).into(), Some(Route::Bench(b.clone()))),
                (self.bench_entry_name(b, e).into(), None),
            ],
        }
    }
}
