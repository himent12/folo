//! A benchmark's page: its header and its tabs.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use super::format::source_line;
use crate::api::time;
use crate::app::{FoloApp, act};
use crate::bench::BenchTab;
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    // ---------- a benchmark ----------

    pub fn bench_view(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_bench(id, cx);
        if self.bench_unsupported() {
            let header = page_header(&t, "Benchmark", "Not available on this server", vec![]);
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let Some(b) = self.bench.list.ready().and_then(|l| l.iter().find(|b| b.id == id)).cloned() else {
            let header = page_header(&t, "Benchmark", "", vec![]);
            let body = if let Some(e) = self.bench.list.error().cloned() {
                self.error_panel(&e, "bench-retry", window, cx)
            } else if self.bench.list.is_loading() {
                panel(&t).child(loading_rows(&t, "bench", 5)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Benchmark not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body, cx);
        };
        let now = self.store.api.now();
        let created = time::parse(&b.created).map_or(String::new(), |c| time::ago(c, now));
        let title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).ellipsis().child(b.name.clone()))
            .when(b.active, |d| d.child(badge(&t, "Running", Tone::Accent)));
        let subtitle = format!(
            "{} · {} images · {} boxes · {} · created {created}",
            source_line(&b),
            int(b.images),
            int(b.boxes),
            plural(b.plan.conditions.len(), "condition"),
        );
        let confirming = self.bench.confirm_delete.as_ref().is_some_and(|(x, i)| *x == b.id && i.is_none());
        let mut actions = Vec::new();
        if confirming {
            actions.push(
                self.button(
                    "bench-del-cancel",
                    "Keep it",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.bench.confirm_delete = None;
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
            actions.push(
                self.button(
                    "bench-del-confirm",
                    "Delete benchmark and entries",
                    Btn::DangerCommit,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.confirm_bench_delete(cx)),
                )
                .into_any_element(),
            );
        } else if b.active {
            actions.push(
                self.button_disabled(
                    "bench-del",
                    "Delete",
                    Btn::Secondary,
                    Some("icons/trash-2.svg"),
                    "Stop its running entries and loops first",
                    window,
                    cx,
                )
                .into_any_element(),
            );
        } else {
            let bid = b.id.clone();
            actions.push(
                self.button(
                    "bench-del",
                    "Delete",
                    Btn::Secondary,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.bench.confirm_delete = Some((bid.clone(), None));
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        let header = detail_header(&t, title, subtitle, actions);

        let entries = self.bench.entries.get(id).cloned().unwrap_or_default();
        let loops = self.bench.loops.get(id).cloned().unwrap_or_default();
        let n_entries = entries.ready().map_or(b.entries as usize, |l| l.len());
        let n_loops = loops.ready().map(|l| l.len());
        let tab = self.bench.tab;
        let tab_names = [
            format!("Leaderboard · {n_entries}"),
            format!("Plan · {}", b.plan.conditions.len()),
            n_loops.map_or_else(|| "Loops".to_string(), |n| format!("Loops · {n}")),
        ];
        let tab_refs: Vec<&str> = tab_names.iter().map(String::as_str).collect();
        let tabs = self.segmented(
            "bench-tabs",
            &tab_refs,
            tab as usize,
            window,
            cx,
            std::rc::Rc::new(|this, i, _, cx| {
                this.bench.tab = [BenchTab::Leaderboard, BenchTab::Plan, BenchTab::Loops][i.min(2)];
                cx.notify();
            }),
        );
        let description = (!b.description.is_empty())
            .then(|| text(ty::BASE, t.text_secondary).max_w(px(820.)).child(b.description.clone()));
        let content = match tab {
            BenchTab::Leaderboard => self.bench_leaderboard(&t, &b, &entries, window, cx),
            BenchTab::Plan => self.bench_plan(&t, &b, window, cx),
            BenchTab::Loops => self.bench_loops(&t, &b, &loops, window, cx),
        };
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .children(description)
            .child(div().flex().child(tabs))
            .child(content);
        self.page_frame(&t, header, body, cx)
    }
}

#[cfg(test)]
mod tests {
    use crate::app::Route;

    #[test]
    fn routes_round_trip() {
        assert_eq!(Route::parse("benchmarks"), Some(Route::Benches));
        assert_eq!(Route::parse("bench:bn_1"), Some(Route::Bench("bn_1".into())));
        assert_eq!(
            Route::parse("bench-entry:bn_1/be_2"),
            Some(Route::BenchEntry("bn_1".into(), "be_2".into()))
        );
        assert_eq!(Route::parse("bench-entry:bn_1"), None);
        assert_eq!(Route::BenchEntry("b".into(), "e".into()).section(), crate::app::Section::Benchmarks);
    }
}
