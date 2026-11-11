//! The benchmark list, and the panel for a server without benchmarks.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::cells::{name_with_badges, num_cell};
use super::format::source_line;
use crate::api::time;
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::{loading_rows, page_header, th};
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// Without `vision_bench`: what the section needs, in place of its content.
    pub(super) fn bench_unsupported_panel(
        &self,
        t: &Theme,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let settings = self.button(
            "bench-settings",
            "Connection settings",
            Btn::Secondary,
            Some("icons/settings.svg"),
            window,
            cx,
            act(|this, w, cx| this.navigate(Route::Settings, w, cx)),
        );
        panel(t)
            .child(
                empty_state(
                    t,
                    "icons/lock.svg",
                    "This server has no vision benchmarks",
                    "Benchmarks need a real folo serve: the demo server (folo serve --mock) has none. Start folo serve without --mock, connect to another server, or use the built-in demo.",
                    Tone::Neutral,
                )
                .child(settings),
            )
            .into_any_element()
    }

    // ---------- the list ----------

    pub fn benches(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_benches(cx);
        if std::mem::take(&mut self.bench.list_stale) && self.bench.list.ready().is_some() {
            self.refresh_benches(cx);
        }
        let list = self.bench.list.ready().cloned();
        let subtitle = match (&list, self.bench.list.error()) {
            (Some(l), _) => format!(
                "{} · your images under hard conditions, scored on five axes",
                plural(l.len(), "benchmark")
            ),
            (None, Some(_)) => "The benchmark list didn't load".to_string(),
            _ if self.bench_unsupported() => "Not available on this server".to_string(),
            _ => "Loading…".to_string(),
        };
        let mut actions = Vec::new();
        if self.bench_on() && self.bench.new_bench.is_none() {
            actions.push(
                self.button(
                    "bench-new",
                    "New benchmark",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_bench(w, cx)),
                )
                .into_any_element(),
            );
        }
        let header = page_header(&t, "Benchmarks", subtitle, actions);
        if self.bench_unsupported() {
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        if let Some(e) = self.bench.list.error().cloned() {
            let err = self.error_panel(&e, "benches-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }
        let form = self.bench.new_bench.is_some().then(|| self.new_bench_form(&t, window, cx));
        // The name column's width: what the fixed columns, their gaps and the row padding leave.
        let name_w =
            (self.content_width(window) - (72. + 84. + 64. + 220. + 84.) - 5. * 16. - 42.).clamp(120., 640.);
        let now = self.store.api.now();
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Benchmark").flex_1().min_w(px(200.)))
            .child(num_cell(72., th(&t, "Images")))
            .child(num_cell(84., th(&t, "Conditions")))
            .child(num_cell(64., th(&t, "Entries")))
            .child(th(&t, "Leader").w(px(220.)).flex_none())
            .child(num_cell(84., th(&t, "Created")));
        let table = match list {
            None => panel(&t).child(head).child(loading_rows(&t, "benches", 3)).into_any_element(),
            Some(l) if l.is_empty() => panel(&t)
                .child(empty_state(
                    &t,
                    "icons/target.svg",
                    "No benchmarks yet",
                    "A benchmark copies a dataset split, freezes it, and grades any model on it: accuracy, how much survives low light, blur or haze, whether its confidences can be trusted, and whether it could label new images by itself.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Some(l) => {
                let rows: Vec<AnyElement> = l
                    .iter()
                    .map(|b| {
                        let key = format!("bench-row-{}", b.id);
                        let id = b.id.clone();
                        let leader = match &b.best {
                            Some(best) => div()
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .min_w_0()
                                .child(text(ty::SM, t.text_primary).tnum().font_weight(FontWeight::MEDIUM).child(format!("{:.1}", best.score)))
                                .child(text(ty::SM, t.text_secondary).ellipsis().min_w_0().child(best.name.clone()))
                                .when(best.partial, |d| d.child(badge(&t, "partial", Tone::Neutral))),
                            None => text(ty::SM, t.text_tertiary).child(if b.active { "Running…" } else { "No finished entry" }),
                        };
                        let at = time::parse(&b.created);
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .flex()
                            .items_center()
                            .gap(px(16.))
                            .px(px(16.))
                            .h(px(52.))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(200.))
                                    .flex()
                                    .flex_col()
                                    .child(name_with_badges(
                                        &t,
                                        window,
                                        &b.name,
                                        if b.active { &[("Running", Tone::Accent)] } else { &[] },
                                        name_w,
                                    ))
                                    .child(text(ty::SM, t.text_tertiary).ellipsis().child(source_line(b))),
                            )
                            .child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(int(b.images))))
                            .child(num_cell(84., text(ty::SM, t.text_secondary).tnum().child(b.plan.conditions.len().to_string())))
                            .child(num_cell(64., text(ty::SM, t.text_secondary).tnum().child(b.entries.to_string())))
                            .child(div().w(px(220.)).flex_none().min_w_0().child(leader))
                            .child(num_cell(
                                84.,
                                text(ty::SM, t.text_tertiary).tnum().child(at.map_or(String::new(), |c| time::ago(c, now))),
                            ));
                        self.bind(row, &key, 0., window, cx, act(move |this, w, cx| this.navigate(Route::Bench(id.clone()), w, cx)))
                            .into_any_element()
                    })
                    .collect();
                panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
            }
        };
        let body = div().flex().flex_col().gap(px(16.)).children(form).child(table);
        self.page_frame(&t, header, body, cx)
    }
}
