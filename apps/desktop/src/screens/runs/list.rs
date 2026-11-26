//! The run list: its status filters, sorting and the runs table.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::duration_of;
use crate::api::{Arch, Label, Run, RunStatus, time};
use crate::app::{FoloApp, Route, RunFilter, act};
use crate::controls::Btn;
use crate::screens::{FirstStep, loading_rows, page_header, th};
use crate::theme::{theme, ty};
use crate::ui::*;

/// Filters `runs` by status, optional architecture, and case-insensitive substring query across
/// run name, id, dataset name, architecture, and status label.
pub fn filter_runs<'a>(
    runs: &'a [Run],
    status: RunFilter,
    arch: Option<Arch>,
    query: &str,
    dataset_name: impl Fn(&Run) -> String,
) -> Vec<&'a Run> {
    let q = query.trim().to_lowercase();
    runs.iter()
        .filter(|r| matches_filter(r, status))
        .filter(|r| arch.is_none_or(|a| r.config.arch == a))
        .filter(|r| {
            q.is_empty()
                || r.name.to_lowercase().contains(&q)
                || r.id.to_lowercase().contains(&q)
                || r.config.arch.as_str().to_lowercase().contains(&q)
                || r.status.label().to_lowercase().contains(&q)
                || dataset_name(r).to_lowercase().contains(&q)
        })
        .collect()
}

impl FoloApp {
    pub fn runs(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let runs = self.store.runs.ready().cloned().unwrap_or_default();
        let total = runs.len();
        let count = |f: RunFilter| runs.iter().filter(|r| matches_filter(r, f)).count();
        let active = count(RunFilter::Active);
        let subtitle = if self.store.runs.is_loading() {
            "Loading…".to_string()
        } else if self.store.runs.error().is_some() {
            // Not "0 runs" over the error.
            "The run list didn't load".to_string()
        } else {
            format!("{} · {} active · {} failed", plural(runs.len(), "run"), active, count(RunFilter::Failed))
        };
        let ds_err = self.store.datasets.error().cloned();
        let show_retry_ds = ds_err.is_some() && self.store.runs.error().is_none();
        let retry_ds_stop = self.passing_stop("runs-retry-ds", show_retry_ds, cx);
        let mut header_actions = Vec::new();
        if let Some(err) = ds_err.as_ref()
            && self.store.runs.error().is_none()
        {
            header_actions.push(
                self.button(
                    "runs-retry-ds",
                    "Retry datasets",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_datasets(cx)),
                )
                .tooltip(crate::controls::tip(format!("Datasets didn't load: {}", err.message), None))
                .into_any_element(),
            );
        } else if let Some(fh) = retry_ds_stop {
            header_actions.push(div().track_focus(&fh).into_any_element());
        }
        let new = if self.store.runs.error().is_some() {
            self.button_disabled(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                "Connect to folo serve first",
                window,
                cx,
            )
            .into_any_element()
        } else if ds_err.is_some() {
            self.button_disabled(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                "Datasets didn't load · retry to start a run",
                window,
                cx,
            )
            .into_any_element()
        } else if self.store.datasets.ready().is_none_or(|d| d.is_empty()) {
            self.button_disabled(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                if self.store.datasets.ready().is_none() {
                    "Loading datasets…"
                } else {
                    "Add a dataset first"
                },
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_new_run(None, w, cx)),
            )
            .tooltip(crate::controls::tip("Start training", Some("Ctrl N")))
            .into_any_element()
        };
        header_actions.push(new);
        let header = page_header(&t, "Runs", subtitle, header_actions);
        if let Some(e) = self.store.runs.error().cloned() {
            let err = self.error_panel(&e, "runs-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }
        let filters = [RunFilter::All, RunFilter::Active, RunFilter::Completed, RunFilter::Failed];
        let labels: Vec<String> = filters
            .iter()
            .map(|f| {
                let name = match f {
                    RunFilter::All => "All",
                    RunFilter::Active => "Active",
                    RunFilter::Completed => "Finished",
                    RunFilter::Failed => "Failed",
                };
                // A no-break space marks the count for the segmented control's quieter style.
                format!("{name}\u{a0}{}", count(*f))
            })
            .collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let sel = filters.iter().position(|f| *f == self.run_filter).unwrap_or(0);
        let seg = self.segmented(
            "runs-filter",
            &label_refs,
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, _| this.run_filter = filters[i]),
        );
        let archs = [None, Some(Arch::N), Some(Arch::S), Some(Arch::M), Some(Arch::L), Some(Arch::X)];
        let arch_labels = arch_filter_labels(&runs);
        let arch_refs: Vec<&str> = arch_labels.iter().map(String::as_str).collect();
        let arch_sel = archs.iter().position(|a| *a == self.run_arch_filter).unwrap_or(0);
        let arch_seg = self.segmented(
            "runs-arch-filter",
            &arch_refs,
            arch_sel,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                this.run_arch_filter = archs[i];
                cx.notify();
            }),
        );
        let query = self.run_search.read(cx).text().to_string();
        let search = self.search_field("runs-search", self.run_search.clone(), 220., window, cx);
        let extra_filter = self.run_arch_filter.is_some() || !query.trim().is_empty();
        let now = self.store.api.now();
        let mut visible: Vec<Run> = filter_runs(&runs, self.run_filter, self.run_arch_filter, &query, |r| {
            self.store.run_dataset_name(r)
        })
        .into_iter()
        .cloned()
        .collect();
        // Sorted like Models (stable, so ties keep the server's newest-first order); unmeasured or
        // missing values always sort last in both ascending and descending order.
        sort_runs(&mut visible, self.runs_sort, now);
        let ids: Vec<String> = visible.iter().map(|r| r.id.clone()).collect();
        let cols = |d: folo_ui::Div| d.flex().items_center().gap(px(16.)).px(px(16.));
        // Single-line rows; secondary facts get their own columns as the window allows, whole
        // columns at a time (`RunCols::fit`), and the name takes what they leave.
        let content = self.content_width(window);
        let c = RunCols::fit(content);
        let name_w = (content - c.fixed() - RUN_SLACK).clamp(RUN_NAME_MIN, 360.);
        let head = cols(div())
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Run").flex_1().min_w(px(RUN_NAME_MIN)))
            .child(th(&t, "Architecture").w(px(96.)).flex_none())
            .when(c.dataset, |d| d.child(th(&t, "Dataset").w(px(120.)).flex_none()))
            .when(c.setup, |d| d.child(th(&t, "Image size · batch").w(px(120.)).flex_none()))
            .child(th(&t, "Status").w(px(96.)).flex_none())
            .child(th(&t, "Progress").w(px(120.)).flex_none())
            .child(self.sort_th_on(
                self.runs_sort,
                |a| &mut a.runs_sort,
                "rsort",
                "best",
                "Best mAP50-95",
                96.,
                true,
                window,
                cx,
            ))
            .when(c.trend, |d| d.child(th(&t, "Trend").w(px(TREND_W)).flex_none()))
            .when(c.duration, |d| {
                d.child(self.sort_th_on(
                    self.runs_sort,
                    |a| &mut a.runs_sort,
                    "rsort",
                    "duration",
                    "Duration",
                    80.,
                    true,
                    window,
                    cx,
                ))
            })
            .when(c.created, |d| {
                d.child(self.sort_th_on(
                    self.runs_sort,
                    |a| &mut a.runs_sort,
                    "rsort",
                    "created",
                    "Created",
                    80.,
                    true,
                    window,
                    cx,
                ))
            });
        let rows: Vec<AnyElement> = visible
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let ds = self.store.run_dataset_name(r);
                let dur = duration_of(r, now).map_or("–".into(), time::duration);
                let created_at = time::parse(&r.created);
                let created = created_at.map_or(String::new(), |c| time::ago(c, now));
                // A queued run says where it waits; every other one its epochs on a 2 px rail.
                let progress_cell = match r.status {
                    RunStatus::Queued => text(ty::SM, t.text_tertiary)
                        .whitespace_nowrap()
                        .ellipsis()
                        .child(waiting_text(r))
                        .into_any_element(),
                    _ => run_progress_cell(&t, r, true, 56.).into_any_element(),
                };
                // The trend needs the run's epochs (fetched once per run, kept current while it trains).
                let trend = c.trend.then(|| {
                    if r.epoch > 0 {
                        self.ensure_run_data(&r.id, cx);
                    }
                    let values: Vec<f32> = self
                        .store
                        .run_data
                        .get(&r.id)
                        .and_then(|l| l.ready())
                        .map(|d| d.metrics.epochs.iter().map(|e| e.val.map50_95).collect())
                        .unwrap_or_default();
                    run_trend(&t, values, r.epochs, r.status == RunStatus::Running)
                });
                // The number the list is ranked by: primary ink, medium.
                let best = r.best.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |b| {
                    text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).tnum().child(f3(b.map50_95))
                });
                let id = r.id.clone();
                let key = format!("run-row-{}", r.id);
                let ids_up = ids.clone();
                // Keyboard focus shows what hover does: the full name and the exact time.
                let e2e = r.config.head == crate::api::Head::O2o;
                let focus_text = self.ring(&self.fh(key.clone(), cx), window).then(|| {
                    let base = created_at.map_or(r.name.clone(), |c| {
                        format!("{} · created {}", r.name, time::local_datetime(c))
                    });
                    if e2e { base + " · end-to-end head: predicts without NMS" } else { base }
                });
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .when_some(focus_text, |d, s| d.child(crate::controls::focus_tip(&t, s, 36.)))
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
                    .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, w, cx| {
                        let m = &ev.keystroke.modifiers;
                        if m.control || m.alt || m.platform || m.shift {
                            return;
                        }
                        let next = match ev.keystroke.key.as_str() {
                            "down" => i + 1,
                            "up" if i > 0 => i - 1,
                            "home" if i > 0 => 0,
                            "end" if i + 1 < ids_up.len() => ids_up.len() - 1,
                            _ => return,
                        };
                        if let Some(id) = ids_up.get(next) {
                            this.focus_key_and_reveal(&format!("run-row-{id}"), w, cx);
                            cx.stop_propagation();
                        }
                    }))
                    // A name that may be cut shows in full on hover (and on focus, above).
                    .child(
                        div()
                            .id(SharedString::from(format!("rn-{}", r.id)))
                            .flex_1()
                            .min_w(px(RUN_NAME_MIN))
                            .when(crate::ui::may_overflow(&r.name, ty::SM.size, name_w), |d| {
                                d.tooltip(crate::controls::tip(r.name.clone(), None))
                            })
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .w(px(name_w))
                                    .ellipsis()
                                    .child(r.name.clone()),
                            ),
                    )
                    // The architecture, and the same e2e tag as the model tables for an end-to-end run.
                    .child(
                        div()
                            .w(px(96.))
                            .flex_none()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_secondary).child(r.config.arch.as_str()))
                            .when(r.config.head == crate::api::Head::O2o, |d| {
                                d.child(e2e_tag(&t, format!("e2e-r-{}", r.id)))
                            }),
                    )
                    .when(c.dataset, |d| {
                        d.child(
                            text(ty::SM, t.text_secondary)
                                .w(px(120.))
                                .flex_none()
                                .ellipsis()
                                .child(ds.clone()),
                        )
                    })
                    .when(c.setup, |d| {
                        d.child(
                            text(ty::SM, t.text_secondary)
                                .w(px(120.))
                                .flex_none()
                                .tnum()
                                .whitespace_nowrap()
                                .child(format!("{} px · {}", r.config.imgsz, r.config.batch)),
                        )
                    })
                    .child(div().w(px(96.)).flex_none().child(status_dot(&t, r, &format!("row-{}", r.id))))
                    .child(div().w(px(120.)).flex_none().child(progress_cell))
                    .child(div().w(px(96.)).flex_none().flex().justify_end().child(best))
                    .children(trend.map(|el| div().w(px(TREND_W)).flex_none().child(el)))
                    .when(c.duration, |d| {
                        d.child(
                            text(ty::SM, t.text_secondary)
                                .tnum()
                                .w(px(80.))
                                .flex_none()
                                .text_right()
                                .child(dur),
                        )
                    })
                    .when(c.created, |d| {
                        // The exact local time on hover; the column reads relative.
                        d.child(
                            text(ty::SM, t.text_tertiary)
                                .id(SharedString::from(format!("rc-{}", r.id)))
                                .when_some(created_at, |d, c| {
                                    d.tooltip(crate::controls::tip(time::local_datetime(c), None))
                                })
                                .tnum()
                                .w(px(80.))
                                .flex_none()
                                .whitespace_nowrap()
                                .text_right()
                                .child(created),
                        )
                    });
                self.bind(
                    row,
                    &key,
                    0.,
                    window,
                    cx,
                    act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
                )
                .into_any_element()
            })
            .collect();
        let step = self.next_first_step();
        let mut first = self.first_step_button("runs-first", step, window, cx);
        // A filter that hides every run says how to see them again.
        let show_all = (rows.is_empty() && total > 0).then(|| {
            let label = if extra_filter {
                "Clear filters".to_string()
            } else if total == 1 {
                "Show the 1 run".to_string()
            } else {
                format!("Show all {total} runs")
            };
            self.button(
                "runs-show-all",
                label,
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, w, cx| {
                    this.run_filter = RunFilter::All;
                    this.run_arch_filter = None;
                    this.run_search.update(cx, |i, cx| i.set_text("", cx));
                    w.focus(&this.run_search.read(cx).focus_handle(cx));
                    cx.notify();
                }),
            )
        });
        let empty_retry_ds_stop = self.passing_stop(
            "runs-empty-retry-ds",
            !self.store.runs.is_loading() && total == 0 && ds_err.is_some(),
            cx,
        );
        let table = panel(&t).overflow_hidden().child(head).map(|d| {
            if self.store.runs.is_loading() {
                d.child(loading_rows(&t, "runs", 6))
            } else if total == 0 {
                if let Some(err) = ds_err.as_ref() {
                    let retry = self.button(
                        "runs-empty-retry-ds",
                        "Retry datasets",
                        Btn::Secondary,
                        Some("icons/refresh-cw.svg"),
                        window,
                        cx,
                        act(|this, _, cx| this.retry_datasets(cx)),
                    );
                    d.child(
                        empty_state(
                            &t,
                            "icons/circle-alert.svg",
                            "Datasets didn't load",
                            &format!("{} · retry datasets to start a run.", err.message),
                            Tone::Danger,
                        )
                        .child(retry),
                    )
                } else {
                    // An empty server: what to do first, and the button that does it.
                    let body = match step {
                        FirstStep::AddDataset => {
                            "Training needs a dataset first: add a YOLO or COCO folder, or generate a synthetic one."
                        }
                        _ => "Start one with New run: pick a dataset, an architecture and how long to train.",
                    };
                    d.child(
                        empty_state(&t, "icons/activity.svg", "No runs yet", body, Tone::Neutral)
                            .when_some(empty_retry_ds_stop.clone(), |d, fh| d.child(div().track_focus(&fh)))
                            .children(first.take()),
                    )
                }
            } else if rows.is_empty() {
                d.child(
                    empty_state(
                        &t,
                        "icons/activity.svg",
                        "No runs here",
                        "Runs matching this filter will show up here.",
                        Tone::Neutral,
                    )
                    .children(show_all),
                )
            } else {
                d.children(rows)
            }
        });
        // A filter that narrows the list says what is shown, under the table (§2: no totals line
        // otherwise).
        let narrowed = shown_line(ids.len(), total)
            .map(|line| text(ty::SM, t.text_tertiary).tnum().px(px(16.)).child(line));
        let toolbar = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(12.))
            .child(seg)
            .child(arch_seg)
            .child(div().flex_1().min_w(px(8.)))
            .child(search);
        let body = div().flex().flex_col().gap(px(12.)).child(toolbar).child(table).children(narrowed);
        self.page_frame(&t, header, body, cx)
    }
}

/// Sorts `runs` stably by the given `(column_key, descending)` pair. Missing, unmeasured, or
/// non-finite values always sort last in both ascending and descending order.
pub(crate) fn sort_runs(runs: &mut [Run], sort: Option<(&str, bool)>, now: i64) {
    let Some((k, desc)) = sort else { return };
    let score = |r: &Run| -> Option<f64> {
        match k {
            "best" => r.best.as_ref().map(|b| b.map50_95 as f64),
            "duration" => duration_of(r, now),
            _ => time::parse(&r.created).map(|t| t as f64),
        }
    };
    runs.sort_by(|a, b| cmp_opt_f64(score(a), score(b), desc));
}

/// Compares optional numeric sort keys so `None` (and non-finite values) always sort last,
/// regardless of ascending or descending direction.
fn cmp_opt_f64(a: Option<f64>, b: Option<f64>, desc: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let a = a.filter(|v| v.is_finite());
    let b = b.filter(|v| v.is_finite());
    match (a, b) {
        (Some(va), Some(vb)) => {
            let o = va.total_cmp(&vb);
            if desc { o.reverse() } else { o }
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Labels with per-architecture counts for `runs-arch-filter` (`"All\u{a0}N"`, `"n\u{a0}K"`, ...).
fn arch_filter_labels(runs: &[Run]) -> Vec<String> {
    const ARCHS: [(Option<Arch>, &str); 6] = [
        (None, "All"),
        (Some(Arch::N), "n"),
        (Some(Arch::S), "s"),
        (Some(Arch::M), "m"),
        (Some(Arch::L), "l"),
        (Some(Arch::X), "x"),
    ];
    ARCHS
        .iter()
        .map(|&(arch, name)| {
            let n = runs.iter().filter(|r| arch.is_none_or(|w| r.config.arch == w)).count();
            // A no-break space marks the count for the segmented control's quieter style.
            format!("{name}\u{a0}{n}")
        })
        .collect()
}

/// The line under a narrowed list: "3 of 10 runs", "0 of 10 runs" when nothing matches; none
/// when everything is shown.
fn shown_line(shown: usize, total: usize) -> Option<String> {
    (shown < total).then(|| format!("{shown} of {}", plural(total, "run")))
}

fn matches_filter(r: &Run, f: RunFilter) -> bool {
    match f {
        RunFilter::All => true,
        RunFilter::Active => r.status.is_active(),
        RunFilter::Completed => matches!(r.status, RunStatus::Completed | RunStatus::Stopped),
        RunFilter::Failed => r.status == RunStatus::Failed,
    }
}

/// The narrowest the Run column gets; the optional columns drop before it would go under.
const RUN_NAME_MIN: f32 = 140.;
/// Row padding, the panel's borders, and slack for rounding: a row exactly as wide as its panel can
/// overflow it by a rounding error.
const RUN_SLACK: f32 = 32. + 2. + 8.;
/// The trend sparkline's width (design §4 Runs: 96 × 20).
const TREND_W: f32 = 96.;

/// Which of the Runs table's optional columns show at a content width. They join one at a time,
/// most telling first (the trend, then the age, the dataset, the duration, the setup), while the
/// row still fits with the name at [`RUN_NAME_MIN`]; a column never squeezes (text truncates against
/// its pre-shrink width), it drops whole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RunCols {
    pub trend: bool,
    pub created: bool,
    pub dataset: bool,
    pub duration: bool,
    pub setup: bool,
}

/// Which of the optional columns a step of the fit turns on.
type Flag = fn(&mut RunCols) -> &mut bool;
impl RunCols {
    /// Architecture, Status, Progress and Best mAP50-95 always show.
    const BASE: [f32; 4] = [96., 96., 120., 96.];

    pub fn fit(content: f32) -> Self {
        let mut c = RunCols::default();
        let mut used = c.fixed();
        let order: [(Flag, f32); 5] = [
            (|c| &mut c.trend, TREND_W),
            (|c| &mut c.created, 80.),
            (|c| &mut c.dataset, 120.),
            (|c| &mut c.duration, 80.),
            (|c| &mut c.setup, 120.),
        ];
        for (flag, w) in order {
            if used + w + 16. + RUN_NAME_MIN + RUN_SLACK > content {
                break;
            }
            *flag(&mut c) = true;
            used += w + 16.;
        }
        c
    }

    /// The fixed columns' widths and every gap (the name's included).
    fn fixed(&self) -> f32 {
        let opt = [
            (self.trend, TREND_W),
            (self.created, 80.),
            (self.dataset, 120.),
            (self.duration, 80.),
            (self.setup, 120.),
        ];
        Self::BASE.iter().map(|w| w + 16.).sum::<f32>()
            + opt.iter().filter(|(on, _)| *on).map(|(_, w)| w + 16.).sum::<f32>()
    }
}

/// A run's mAP50-95 per epoch as a 96 × 20 sparkline, on the run's planned length (a run halfway
/// through draws over the left half), neutral, with a lime end dot while it trains. Nothing under two
/// epochs.
fn run_trend(t: &crate::theme::Theme, values: Vec<f32>, epochs: u32, live: bool) -> AnyElement {
    const H: f32 = 20.;
    // The sparkline insets its line 3 px on every side.
    const INSET: f32 = 3.;
    let values: Vec<f32> = values.into_iter().filter(|v| v.is_finite()).collect();
    let n = values.len();
    if n < 2 {
        return div().h(px(H)).into_any_element();
    }
    let hi = values.iter().copied().fold(0.1f32, f32::max);
    let span = (epochs.max(n as u32) - 1).max(1) as f32;
    let w = 2. * INSET + (TREND_W - 2. * INSET) * ((n - 1) as f32 / span);
    let last = values[n - 1];
    let frac = (last / hi).clamp(0.0, 1.0);
    let end = (w - INSET, INSET + (H - 2. * INSET) * (1. - frac));
    div()
        .relative()
        .w(px(TREND_W))
        .h(px(H))
        .child(div().w(px(w)).h(px(H)).child(crate::charts::sparkline_in(
            t,
            std::sync::Arc::new(values),
            t.text_tertiary,
            H,
            false,
            Some((0., hi)),
        )))
        .when(live, |d| {
            d.child(
                div()
                    .absolute()
                    .left(px(end.0 - 2.5))
                    .top(px(end.1 - 2.5))
                    .size(px(5.))
                    .rounded_full()
                    .bg(t.accent_fill()),
            )
        })
        .into_any_element()
}

#[cfg(test)]
mod tests {
    use super::{RunCols, arch_filter_labels, shown_line, sort_runs};
    use crate::api::{Arch, Backend, BestEpoch, Run};

    #[test]
    fn missing_run_sort_values_always_sort_last_in_both_directions() {
        let base = crate::api::mock::MockBackend::new(4200, true).runs().unwrap().remove(0);
        let mk =
            |id: &str, best: Option<f32>, started: Option<&str>, finished: Option<&str>, created: &str| Run {
                id: id.into(),
                name: id.into(),
                best: best.map(|m| BestEpoch { epoch: 1, map50_95: m }),
                started: started.map(str::to_string),
                finished: finished.map(str::to_string),
                paused_s: None,
                created: created.into(),
                ..base.clone()
            };
        let now = crate::api::time::parse("2026-11-10T12:00:00Z").unwrap();
        let initial = vec![
            mk("none_first", None, None, None, "not-a-timestamp"),
            mk(
                "low",
                Some(0.20),
                Some("2026-11-10T11:59:00Z"),
                Some("2026-11-10T11:59:10Z"),
                "2026-11-01T00:00:00Z",
            ),
            mk("nan", Some(f32::NAN), None, None, ""),
            mk(
                "high",
                Some(0.80),
                Some("2026-11-10T11:00:00Z"),
                Some("2026-11-10T11:10:00Z"),
                "2026-11-05T00:00:00Z",
            ),
        ];

        for key in ["best", "duration", "created"] {
            let mut desc = initial.clone();
            sort_runs(&mut desc, Some((key, true)), now);
            let desc_ids: Vec<&str> = desc.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(desc_ids, ["high", "low", "none_first", "nan"], "{key} desc");

            let mut asc = initial.clone();
            sort_runs(&mut asc, Some((key, false)), now);
            let asc_ids: Vec<&str> = asc.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(asc_ids, ["low", "high", "none_first", "nan"], "{key} asc");
        }
    }

    #[test]
    fn arch_filter_labels_include_counts_with_nobreak_space() {
        let base = crate::api::mock::MockBackend::new(4200, true).runs().unwrap().remove(0);
        let mk = |arch: Arch| {
            let mut r = base.clone();
            r.config.arch = arch;
            r
        };
        let runs = vec![mk(Arch::N), mk(Arch::N), mk(Arch::S), mk(Arch::L)];
        assert_eq!(
            arch_filter_labels(&runs),
            vec!["All\u{a0}4", "n\u{a0}2", "s\u{a0}1", "m\u{a0}0", "l\u{a0}1", "x\u{a0}0"]
        );
    }

    #[test]
    fn a_narrowed_list_says_what_it_shows_even_when_nothing_matches() {
        assert_eq!(shown_line(3, 10).as_deref(), Some("3 of 10 runs"));
        assert_eq!(shown_line(0, 10).as_deref(), Some("0 of 10 runs"), "Failed on a server with no failures");
        assert_eq!(shown_line(0, 1).as_deref(), Some("0 of 1 run"));
        assert_eq!(shown_line(10, 10), None, "the whole list: no totals line");
        assert_eq!(shown_line(0, 0), None, "an empty server says so in its empty state");
    }

    #[test]
    fn columns_drop_whole_and_in_order() {
        // 960 x 620 (content 872): the trend and the age beside the core columns.
        let c = RunCols::fit(872.);
        assert!(c.trend && c.created && !c.dataset && !c.duration && !c.setup, "{c:?}");
        // 1280 x 800 (content 1000): the dataset joins.
        let c = RunCols::fit(1000.);
        assert!(c.trend && c.created && c.dataset && !c.duration, "{c:?}");
        // 1440 (content 1160): the duration; 1680 (content 1392): everything.
        assert!(RunCols::fit(1160.).duration && !RunCols::fit(1160.).setup);
        assert_eq!(
            RunCols::fit(1392.),
            RunCols { trend: true, created: true, dataset: true, duration: true, setup: true }
        );
        // Whatever shows from the 960 px window up, the name keeps its minimum.
        for w in [872., 1000., 1160., 1392.] {
            let c = RunCols::fit(w);
            assert!(w - c.fixed() - super::RUN_SLACK >= super::RUN_NAME_MIN, "{w}");
        }
    }
}
