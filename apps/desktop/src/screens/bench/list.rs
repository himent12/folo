//! The benchmark list, and the panel for a server without benchmarks.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::cells::num_cell;
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
            .child(num_cell(72., th(&t, "Images")))
            .when(c.classes, |d| d.child(num_cell(64., th(&t, "Classes"))))
            .when(c.conditions, |d| d.child(num_cell(84., th(&t, "Conditions"))))
            .child(num_cell(64., th(&t, "Entries")))
            .child(th(&t, "Leader").w(px(leader_w)).flex_none())
            .when(c.created, |d| d.child(num_cell(84., th(&t, "Created"))));
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
                                .child(text(ty::SM, t.text_secondary).ellipsis().min_w_0().child(best.name.clone()))
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
                                        .tooltip(crate::controls::tip("An entry or loop is running", None))
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
                                    text(ty::SM, t.text_secondary).tnum().child(b.classes.len().to_string()),
                                ))
                            })
                            .when(c.conditions, |d| {
                                d.child(num_cell(
                                    84.,
                                    text(ty::SM, t.text_secondary).tnum().child(b.plan.conditions.len().to_string()),
                                ))
                            })
                            .child(num_cell(64., text(ty::SM, t.text_secondary).tnum().child(b.entries.to_string())))
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
    use super::BenchCols;

    #[test]
    fn every_column_at_1280_and_whole_columns_drop_below() {
        assert_eq!(BenchCols::fit(1000.), BenchCols { conditions: true, classes: true, created: true });
        assert_eq!(BenchCols::fit(872.), BenchCols { conditions: true, classes: false, created: false });
    }
}
