//! The page area: routing to each screen, and the frame, gutter and content width every page
//! uses.

use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, StatefulInteractiveElement, Styled,
    Window, div, px,
};

use crate::theme::{Theme, alpha, size as tsize};

use super::{FoloApp, Route};

impl FoloApp {
    pub(super) fn page(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        match self.route.clone() {
            Route::Overview => self.overview(window, cx),
            Route::Datasets => self.datasets(window, cx),
            Route::Dataset(id) => self.dataset(&id, window, cx),
            Route::Runs => self.runs(window, cx),
            Route::Run(id) => self.run_view(&id, window, cx),
            Route::Models => self.models(window, cx),
            Route::Model(id) => self.model_view(&id, window, cx),
            Route::Benches => self.benches(window, cx),
            Route::Bench(id) => self.bench_view(&id, window, cx),
            Route::BenchEntry(b, e) => self.bench_entry_view(&b, &e, window, cx),
            Route::Playground => self.playground(window, cx),
            Route::Settings => self.settings(window, cx),
        }
    }
    /// A page frame: scrollable column with the standard gutter and an optional header.
    /// The page's side gutter: 16 px below 1120 px wide, else 24. Every page, and the breadcrumbs
    /// (which sit 4 px further in for their hover pads), use it.
    pub fn gutter(&self) -> f32 {
        if self.narrow { 16. } else { 24. }
    }

    /// Whether the sidebar is shown as the compact icon rail (either collapsed manually via
    /// `Ctrl+B` or forced compact below 1120 px).
    pub fn sidebar_compact(&self, window: &Window) -> bool {
        self.sidebar_collapsed || window.viewport_size().width < px(1120.)
    }

    /// The page's content width: the window less the sidebar (an icon rail when collapsed or below
    /// 1120 px), capped at `layout.desktop_content_max` (Desktop's own cap, apart from Studio's
    /// `size.content_max`; border-box, so it includes the gutters), less the gutters. Layouts that
    /// pick columns by the room they get measure it here, not by the window width.
    pub fn content_width(&self, window: &Window) -> f32 {
        let vw = f32::from(window.viewport_size().width);
        let side = if self.sidebar_compact(window) { tsize::SIDEBAR_COLLAPSED } else { tsize::SIDEBAR };
        (vw - side).min(crate::theme::layout::DESKTOP_CONTENT_MAX) - 2. * self.gutter()
    }

    pub fn page_frame(
        &self,
        t: &Theme,
        header: impl IntoElement,
        body: impl IntoElement,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        div()
            .relative()
            .size_full()
            .child(
                div()
                    .id("page")
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.page_scroll)
                    .on_scroll_wheel(cx.listener(|this, _, window, cx| {
                        if window.default_prevented() && this.close_menus() {
                            cx.notify();
                        }
                    }))
                    .child(
                        div()
                            .w_full()
                            .min_w_0()
                            .max_w(px(crate::theme::layout::DESKTOP_CONTENT_MAX))
                            .mx_auto()
                            .flex()
                            .flex_col()
                            .gap(px(16.))
                            .px(px(self.gutter()))
                            .pt(px(20.))
                            .pb(px(32.))
                            .child(header)
                            .child(body),
                    )
                    .child(div().absolute().top_0().left_0().right_0().h(px(1.)).bg(alpha(t.bg_app, 0.0))),
            )
            // The page is a scroll area like the rest: edge fades where content lies past the
            // window (the at-rest cue, thumbs being hidden), and the thumb.
            .children(self.scroll_chrome(self.page_scroll.clone(), t.bg_app, 0., 0., cx))
            .into_any_element()
    }
}
