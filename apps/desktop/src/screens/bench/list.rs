//! The benchmark list, and the panel for a server without benchmarks.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::cells::num_cell;
use super::format::source_line;
use crate::api::{Bench, time};
use crate::app::{FoloApp, Route, act};
use crate::bench::BenchListFilter;
use crate::controls::Btn;
use crate::screens::{loading_rows, page_header, th};
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

/// Filters benchmarks by status (`All`, `Running`, `Scored`) and a case-insensitive substring match
/// across name, id, source label, and current leader name.
pub(crate) fn filter_benches<'a>(list: &'a [Bench], filter: BenchListFilter, query: &str) -> Vec<&'a Bench> {
    let q = query.trim().to_lowercase();
    list.iter()
        .filter(|b| match filter {
            BenchListFilter::All => true,
            BenchListFilter::Running => b.active,
            BenchListFilter::Scored => b.best.is_some(),
        })
        .filter(|b| {
            if q.is_empty() {
                return true;
            }
            b.name.to_lowercase().contains(&q)
                || b.id.to_lowercase().contains(&q)
                || source_line(b).to_lowercase().contains(&q)
                || b.best.as_ref().is_some_and(|best| best.name.to_lowercase().contains(&q))
        })
        .collect()
}

/// Compares optional finite `f32` values so missing/non-finite entries always sort last in both
/// ascending and descending order.
fn cmp_opt_f32(a: Option<f32>, b: Option<f32>, desc: bool) -> std::cmp::Ordering {
    match (a.filter(|v| v.is_finite()), b.filter(|v| v.is_finite())) {
        (Some(av), Some(bv)) => {
            if desc {
                bv.total_cmp(&av)
            } else {
                av.total_cmp(&bv)
            }
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Compares optional timestamps so unparseable/missing timestamps always sort last in both orders.
fn cmp_opt_i64(a: Option<i64>, b: Option<i64>, desc: bool) -> std::cmp::Ordering {
    match (a, b) {
        (Some(av), Some(bv)) => {
            if desc {
                bv.cmp(&av)
            } else {
                av.cmp(&bv)
            }
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// Sorts benchmark rows by `(column_key, descending)`. Missing leader scores or timestamps always
/// sort last in both directions; ties preserve deterministic name/id order.
pub(crate) fn sort_benches(rows: &mut [&Bench], sort: Option<(&'static str, bool)>) {
    let Some((key, desc)) = sort else { return };
    rows.sort_by(|a, b| {
        let primary = match key {
            "images" => {
                if desc {
                    b.images.cmp(&a.images)
                } else {
                    a.images.cmp(&b.images)
                }
            }
            "entries" => {
                if desc {
                    b.entries.cmp(&a.entries)
                } else {
                    a.entries.cmp(&b.entries)
                }
            }
            "leader" => cmp_opt_f32(a.best.as_ref().map(|x| x.score), b.best.as_ref().map(|x| x.score), desc),
            "created" => cmp_opt_i64(time::parse(&a.created), time::parse(&b.created), desc),
            _ => std::cmp::Ordering::Equal,
        };
        primary.then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase())).then_with(|| a.id.cmp(&b.id))
    });
}

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
            // What the page holds (§2 page header): how many, and how many have an entry running.
            (Some(l), _) => {
                let running = l.iter().filter(|b| b.active).count();
                let images: u64 = l.iter().map(|b| b.images).sum();
                let mut facts = vec![plural(l.len(), "benchmark")];
                if running > 0 {
                    facts.push(format!("{running} running"));
                }
                facts.push(counted(images, "image", "images"));
                facts.join(" · ")
            }
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
        // Single-line rows; the optional columns drop whole where the window is narrow (`BenchCols`),
        // and the name takes what the others leave.
        let content = self.content_width(window);
        let c = BenchCols::fit(content);
        let name_w = (content - c.fixed() - BENCH_SLACK).clamp(BENCH_NAME_MIN, 480.);
        // The leader's name is cut before a benchmark's name, never while the Benchmark column has
        // room it doesn't use: the room past the longest name (and the running dot) goes to Leader.
        let name_need = list
            .iter()
            .flat_map(|l| l.iter())
            .map(|b| {
                let w = f32::from(crate::charts::text_width(window, &b.name, ty::SM.size)) * 1.05;
                w.ceil() + if b.active { 20. } else { 0. }
            })
            .fold(BENCH_NAME_FLOOR, f32::max);
        let leader_w = LEADER_W + (name_w - name_need).max(0.).floor();
        let name_w = name_w.min(name_need);
        let now = self.store.api.now();
        let sort_col =
            |key: &'static str, label: &'static str, w: f32, right: bool, cx: &mut Context<Self>| {
                self.sort_th_on(
                    self.bench.list_sort,
                    |a| &mut a.bench.list_sort,
                    "bench-list-sort",
                    key,
                    label,
                    w,
                    right,
                    window,
                    cx,
                )
            };
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Benchmark").flex_1().min_w(px(name_w)))
            .child(th(&t, "Images from").w(px(FROM_W)).flex_none())
            .child(sort_col("images", "Images", 72., true, cx))
            .when(c.classes, |d| d.child(num_cell(64., th(&t, "Classes"))))
            .when(c.conditions, |d| d.child(num_cell(84., th(&t, "Conditions"))))
            .child(sort_col("entries", "Entries", 64., true, cx))
            .child(sort_col("leader", "Leader", leader_w, false, cx))
            .when(c.created, |d| d.child(sort_col("created", "Created", 84., true, cx)));
        let query = self.bench_search.read(cx).text().to_string();
        let filter = self.bench.list_filter;
        let filters_active =
            filter != BenchListFilter::All || !query.trim().is_empty() || self.bench.list_sort.is_some();
        let toolbar = list.as_ref().filter(|l| !l.is_empty()).map(|l| {
            let running_n = l.iter().filter(|b| b.active).count();
            let scored_n = l.iter().filter(|b| b.best.is_some()).count();
            let labels = [
                format!("All\u{a0}{}", int(l.len() as u64)),
                format!("Running\u{a0}{}", int(running_n as u64)),
                format!("Scored\u{a0}{}", int(scored_n as u64)),
            ];
            let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
            let filters = [BenchListFilter::All, BenchListFilter::Running, BenchListFilter::Scored];
            let sel = filters.iter().position(|f| *f == filter).unwrap_or(0);
            let seg = self.segmented(
                "bench-filter",
                &refs,
                sel,
                window,
                cx,
                Rc::new(move |this, i, _, cx| {
                    this.bench.list_filter = filters[i.min(2)];
                    cx.notify();
                }),
            );
            let narrow = window.viewport_size().width < px(1120.);
            let search = self.search_field(
                "bench-search",
                self.bench_search.clone(),
                if narrow { 180. } else { 220. },
                window,
                cx,
            );
            let reset_stop = self.passing_stop("bench-filter-reset", filters_active, cx);
            let search_wrap = div()
                .flex()
                .flex_none()
                .items_center()
                .child(search)
                .when(!filters_active, |d| d.children(reset_stop.map(|fh| div().track_focus(&fh))));
            let reset_btn = filters_active.then(|| {
                self.button(
                    "bench-filter-reset",
                    "Reset",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.bench.list_filter = BenchListFilter::All;
                        this.bench.list_sort = None;
                        this.bench_search.update(cx, |i, cx| i.set_text("", cx));
                        cx.notify();
                    }),
                )
                .tooltip(crate::controls::tip("Reset filter, search and sort", None))
                .into_any_element()
            });
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap(px(8.))
                .child(seg)
                .child(div().flex_1())
                .child(search_wrap)
                .children(reset_btn)
        });
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
                let mut shown = filter_benches(&l, filter, &query);
                sort_benches(&mut shown, self.bench.list_sort);
                if shown.is_empty() {
                    let reset = self.button(
                        "bench-empty-reset",
                        "Reset filters",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, w, cx| {
                            this.bench.list_filter = BenchListFilter::All;
                            this.bench.list_sort = None;
                            this.bench_search.update(cx, |i, cx| i.set_text("", cx));
                            w.focus(&this.bench_search.read(cx).focus_handle(cx));
                            cx.notify();
                        }),
                    );
                    panel(&t)
                        .overflow_hidden()
                        .child(head)
                        .child(
                            empty_state(
                                &t,
                                "icons/list-filter.svg",
                                "No benchmarks match",
                                "Try another status filter or search term.",
                                Tone::Neutral,
                            )
                            .child(reset),
                        )
                        .into_any_element()
                } else {
                    let rows: Vec<AnyElement> = shown
                        .into_iter()
                        .map(|b| {
                            let key = format!("bench-row-{}", b.id);
                            let id = b.id.clone();
                            // The leader: its score (what the board ranks by), then its name.
                            let leader = match &b.best {
                                Some(best) => div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .min_w_0()
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .tnum()
                                            .font_weight(FontWeight::MEDIUM)
                                            .flex_none()
                                            .child(format!("{:.1}", best.score)),
                                    )
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .ellipsis()
                                            .min_w_0()
                                            .child(best.name.clone()),
                                    )
                                    .when(best.partial, |d| d.child(badge(&t, "partial", Tone::Neutral))),
                                None => text(ty::SM, t.text_tertiary)
                                    .child(if b.active { "Running…" } else { "No finished entry" }),
                            };
                            let at = time::parse(&b.created);
                            // A running entry or loop: the ringed lime dot after the name, its word in the tip.
                            let active_w = if b.active { 14. } else { 0. };
                            let name = div()
                                .id(SharedString::from(format!("bench-name-{}", b.id)))
                                .flex_1()
                                .min_w(px(name_w))
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .when(crate::ui::may_overflow(&b.name, ty::SM.size, name_w - active_w), |d| {
                                    d.tooltip(crate::controls::tip(b.name.clone(), None))
                                })
                                .child(
                                    text(ty::SM, t.text_primary)
                                        .font_weight(FontWeight::MEDIUM)
                                        .max_w(px(name_w - active_w))
                                        .ellipsis()
                                        .child(b.name.clone()),
                                )
                                .when(b.active, |d| {
                                    d.child(
                                        div()
                                            .id(SharedString::from(format!("bench-live-{}", b.id)))
                                            .tooltip(crate::controls::tip(
                                                "An entry or loop is running",
                                                None,
                                            ))
                                            .child(status_mark(
                                                StatusKind::Running,
                                                StatusKind::Running.hue(&t),
                                                format!("bench-{}", b.id),
                                            )),
                                    )
                                });
                            let row = div()
                                .id(SharedString::from(key.clone()))
                                .flex()
                                .items_center()
                                .gap(px(16.))
                                .px(px(16.))
                                .h(px(36.))
                                .border_b_1()
                                .border_color(t.line_subtle)
                                .cursor_pointer()
                                .hover(|s| s.bg(t.bg_hover))
                                .active(|s| s.bg(t.bg_active))
                                .child(name)
                                .child(
                                    text(ty::SM, t.text_secondary)
                                        .w(px(FROM_W))
                                        .flex_none()
                                        .ellipsis()
                                        .child(source_line(b)),
                                )
                                .child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(int(b.images))))
                                .when(c.classes, |d| {
                                    d.child(num_cell(
                                        64.,
                                        text(ty::SM, t.text_secondary)
                                            .tnum()
                                            .child(b.classes.len().to_string()),
                                    ))
                                })
                                .when(c.conditions, |d| {
                                    d.child(num_cell(
                                        84.,
                                        text(ty::SM, t.text_secondary)
                                            .tnum()
                                            .child(b.plan.conditions.len().to_string()),
                                    ))
                                })
                                .child(num_cell(
                                    64.,
                                    text(ty::SM, t.text_secondary).tnum().child(b.entries.to_string()),
                                ))
                                .child(div().w(px(leader_w)).flex_none().min_w_0().child(leader))
                                .when(c.created, |d| {
                                    d.child(num_cell(
                                        84.,
                                        text(ty::SM, t.text_tertiary)
                                            .tnum()
                                            .whitespace_nowrap()
                                            .child(at.map_or(String::new(), |c| time::ago(c, now))),
                                    ))
                                });
                            self.bind(
                                row,
                                &key,
                                0.,
                                window,
                                cx,
                                act(move |this, w, cx| this.navigate(Route::Bench(id.clone()), w, cx)),
                            )
                            .into_any_element()
                        })
                        .collect();
                    panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
                }
            }
        };
        let body = div().flex().flex_col().gap(px(16.)).children(form).children(toolbar).child(table);
        self.page_frame(&t, header, body, cx)
    }
}

/// The narrowest the Benchmark column gets; optional columns drop before it would go under.
const BENCH_NAME_MIN: f32 = 140.;
/// The narrowest it gets when its names are short and Leader takes its spare room ("Benchmark").
const BENCH_NAME_FLOOR: f32 = 96.;
/// Row padding, the panel's borders and slack for rounding.
const BENCH_SLACK: f32 = 32. + 2. + 8.;
const FROM_W: f32 = 140.;
const LEADER_W: f32 = 180.;

/// Which of the Benchmarks table's optional columns show at a content width: Conditions, then
/// Classes, then Created, each while the row still fits with the name at [`BENCH_NAME_MIN`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BenchCols {
    conditions: bool,
    classes: bool,
    created: bool,
}

/// Which of the optional columns a step of the fit turns on.
type Flag = fn(&mut BenchCols) -> &mut bool;
impl BenchCols {
    /// Images from, Images, Entries and Leader always show.
    const BASE: [f32; 4] = [FROM_W, 72., 64., LEADER_W];

    fn fit(content: f32) -> Self {
        let mut c = BenchCols::default();
        let order: [(Flag, f32); 3] =
            [(|c| &mut c.conditions, 84.), (|c| &mut c.classes, 64.), (|c| &mut c.created, 84.)];
        for (flag, w) in order {
            if c.fixed() + w + 16. + BENCH_NAME_MIN + BENCH_SLACK > content {
                break;
            }
            *flag(&mut c) = true;
        }
        c
    }

    /// The fixed columns' widths and their gaps.
    fn fixed(&self) -> f32 {
        let opt = [(self.conditions, 84.), (self.classes, 64.), (self.created, 84.)];
        Self::BASE.iter().map(|w| w + 16.).sum::<f32>()
            + opt.iter().filter(|(on, _)| *on).map(|(_, w)| w + 16.).sum::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::{BenchCols, filter_benches, sort_benches};
    use crate::api::{Backend, Bench, BenchBest, mock::MockBackend};
    use crate::bench::BenchListFilter;

    #[test]
    fn every_column_at_1280_and_whole_columns_drop_below() {
        assert_eq!(BenchCols::fit(1000.), BenchCols { conditions: true, classes: true, created: true });
        assert_eq!(BenchCols::fit(872.), BenchCols { conditions: true, classes: false, created: false });
    }

    #[test]
    fn bench_list_filters_and_sorts_missing_leaders_last() {
        let base = MockBackend::new(4200, true).benches().unwrap().remove(0);
        let a = Bench {
            id: "bn_a".into(),
            name: "Alpha street".into(),
            images: 200,
            entries: 3,
            active: true,
            best: Some(BenchBest {
                entry_id: "e1".into(),
                name: "folo-s-best".into(),
                score: 82.4,
                partial: false,
            }),
            created: "2026-11-10T10:00:00Z".into(),
            ..base.clone()
        };
        let b = Bench {
            id: "bn_b".into(),
            name: "Beta lab".into(),
            images: 50,
            entries: 1,
            active: false,
            best: None,
            created: "invalid-ts".into(),
            ..base.clone()
        };
        let c = Bench {
            id: "bn_c".into(),
            name: "Gamma indoor".into(),
            images: 400,
            entries: 5,
            active: false,
            best: Some(BenchBest {
                entry_id: "e2".into(),
                name: "folo-n-fast".into(),
                score: 64.0,
                partial: false,
            }),
            created: "2026-11-12T10:00:00Z".into(),
            ..base
        };
        let all = [a, b, c];

        // Status filter & query matching name and leader.
        let running = filter_benches(&all, BenchListFilter::Running, "");
        assert_eq!(running.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_a"]);
        let scored = filter_benches(&all, BenchListFilter::Scored, "");
        assert_eq!(scored.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_a", "bn_c"]);
        let by_leader = filter_benches(&all, BenchListFilter::All, "folo-n");
        assert_eq!(by_leader.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_c"]);

        // Missing leader score sorts LAST in both desc and asc.
        let mut refs: Vec<&Bench> = all.iter().collect();
        sort_benches(&mut refs, Some(("leader", true)));
        assert_eq!(refs.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_a", "bn_c", "bn_b"]);
        sort_benches(&mut refs, Some(("leader", false)));
        assert_eq!(refs.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_c", "bn_a", "bn_b"]);

        // Invalid/missing created timestamp sorts LAST in both desc and asc.
        sort_benches(&mut refs, Some(("created", true)));
        assert_eq!(refs.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_c", "bn_a", "bn_b"]);
        sort_benches(&mut refs, Some(("created", false)));
        assert_eq!(refs.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_a", "bn_c", "bn_b"]);

        // Images and entries sort numerically.
        sort_benches(&mut refs, Some(("images", true)));
        assert_eq!(refs.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_c", "bn_a", "bn_b"]);
        sort_benches(&mut refs, Some(("entries", false)));
        assert_eq!(refs.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["bn_b", "bn_a", "bn_c"]);
    }
}
