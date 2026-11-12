//! The run list: its status filters, sorting and the runs table.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::duration_of;
use crate::api::{Run, RunExt, RunStatus, time};
use crate::app::{FoloApp, Route, RunFilter, act};
use crate::controls::Btn;
use crate::screens::{FirstStep, loading_rows, page_header, th};
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn runs(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let runs = self.store.runs.ready().cloned().unwrap_or_default();
        let total = runs.len();
        let count = |f: RunFilter| runs.iter().filter(|r| matches_filter(r, f)).count();
        let active = count(RunFilter::Active);
        let subtitle = if self.store.runs.is_loading() {
            "Loading…".to_string()
        } else {
            format!("{} · {} active · {} failed", plural(runs.len(), "run"), active, count(RunFilter::Failed))
        };
        let new = if self.store.datasets.ready().is_none_or(|d| d.is_empty()) {
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
        let header = page_header(&t, "Runs", subtitle, vec![new]);
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
        let now = self.store.api.now();
        let mut visible: Vec<Run> = runs.into_iter().filter(|r| matches_filter(r, self.run_filter)).collect();
        // Sorted like Models (stable, so ties keep the server's newest-first order).
        if let Some((k, desc)) = self.runs_sort {
            let score = |r: &Run| -> f64 {
                match k {
                    "best" => r.best.as_ref().map_or(-1., |b| b.map50_95 as f64),
                    "duration" => duration_of(r, now).unwrap_or(-1.),
                    _ => time::parse(&r.created).unwrap_or(0) as f64,
                }
            };
            visible.sort_by(|a, b| {
                let o = score(a).total_cmp(&score(b));
                if desc { o.reverse() } else { o }
            });
        }
        let ids: Vec<String> = visible.iter().map(|r| r.id.clone()).collect();
        let cols = |d: folo_ui::Div| d.flex().items_center().gap(px(16.)).px(px(16.));
        // Single-line rows; secondary facts get their own columns as the window allows, age before
        // setup (so the default 1280 window tells an hour-old run from a week-old one). Budget
        // (content width, Run keeps its 148 px and Arch its 96 for an e2e tag; 1000 px at 1280 with
        // Progress at 140): Dataset +136
        // from 1240, Created +96 from 1280, Image size · batch +136 from 1440.
        let vw = window.viewport_size().width;
        let show_ds = vw >= px(1240.);
        let show_created = vw >= px(1280.);
        let wide = vw >= px(1440.);
        let head = cols(div())
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Run").flex_1().min_w(px(148.)))
            .child(th(&t, "Architecture").w(px(96.)).flex_none())
            .when(show_ds, |d| d.child(th(&t, "Dataset").w(px(120.)).flex_none()))
            .when(wide, |d| d.child(th(&t, "Image size · batch").w(px(120.)).flex_none()))
            .child(th(&t, "Status").w(px(96.)).flex_none())
            .child(th(&t, "Progress").w(px(140.)).flex_none())
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
            .child(self.sort_th_on(
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
            .when(show_created, |d| {
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
                let progress_cell = match r.status {
                    RunStatus::Queued => {
                        text(ty::SM, t.text_tertiary).whitespace_nowrap().child(waiting_text(r))
                    }
                    _ => div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(div().flex_1().child(progress(
                            &t,
                            r.progress(),
                            if r.status == RunStatus::Running {
                                t.accent_fill()
                            } else if r.status == RunStatus::Failed {
                                t.danger_solid
                            } else {
                                t.text_tertiary
                            },
                            4.,
                        )))
                        .child(
                            text(ty::SM, t.text_secondary)
                                .tnum()
                                .w(px(72.))
                                .flex_none()
                                .whitespace_nowrap()
                                .text_right()
                                .child(format!("{} / {}", r.epoch, r.epochs)),
                        ),
                };
                let best = r.best.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |b| {
                    text(ty::SM, t.text_primary).tnum().child(f3(b.map50_95))
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
                        let next = match ev.keystroke.key.as_str() {
                            "down" => i + 1,
                            "up" if i > 0 => i - 1,
                            _ => return,
                        };
                        if let Some(id) = ids_up.get(next) {
                            this.focus_key(&format!("run-row-{id}"), w, cx);
                            cx.stop_propagation();
                        }
                    }))
                    // A name that may be cut shows in full on hover (and on focus, above).
                    .child(
                        div()
                            .id(SharedString::from(format!("rn-{}", r.id)))
                            .flex_1()
                            .min_w(px(148.))
                            .when(crate::ui::may_overflow(&r.name, ty::SM.size, 148.), |d| {
                                d.tooltip(crate::controls::tip(r.name.clone(), None))
                            })
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .w(px(148.))
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
                    .when(show_ds, |d| {
                        d.child(
                            text(ty::SM, t.text_secondary)
                                .w(px(120.))
                                .flex_none()
                                .ellipsis()
                                .child(ds.clone()),
                        )
                    })
                    .when(wide, |d| {
                        d.child(
                            text(ty::SM, t.text_tertiary)
                                .w(px(120.))
                                .flex_none()
                                .tnum()
                                .whitespace_nowrap()
                                .child(format!("{} px · {}", r.config.imgsz, r.config.batch)),
                        )
                    })
                    .child(div().w(px(96.)).flex_none().child(status_dot(&t, r, &format!("row-{}", r.id))))
                    .child(div().w(px(140.)).flex_none().child(progress_cell))
                    .child(div().w(px(96.)).flex_none().flex().justify_end().child(best))
                    .child(
                        text(ty::SM, t.text_secondary).tnum().w(px(80.)).flex_none().text_right().child(dur),
                    )
                    .when(show_created, |d| {
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
            self.button(
                "runs-show-all",
                if total == 1 { "Show the 1 run".to_string() } else { format!("Show all {total} runs") },
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.run_filter = RunFilter::All;
                    cx.notify();
                }),
            )
        });
        let table = panel(&t).overflow_hidden().child(head).map(|d| {
            if self.store.runs.is_loading() {
                d.child(loading_rows(&t, "runs", 6))
            } else if total == 0 {
                // An empty server: what to do first, and the button that does it.
                let body = match step {
                    FirstStep::AddDataset => {
                        "Training needs a dataset first: add a YOLO or COCO folder, or generate a synthetic one."
                    }
                    _ => "Start one with New run: pick a dataset, an architecture and how long to train.",
                };
                d.child(
                    empty_state(&t, "icons/activity.svg", "No runs yet", body, Tone::Neutral)
                        .children(first.take()),
                )
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
        let body = div().flex().flex_col().gap(px(12.)).child(div().flex().child(seg)).child(table);
        self.page_frame(&t, header, body, cx)
    }
}

fn matches_filter(r: &Run, f: RunFilter) -> bool {
    match f {
        RunFilter::All => true,
        RunFilter::Active => r.status.is_active(),
        RunFilter::Completed => matches!(r.status, RunStatus::Completed | RunStatus::Stopped),
        RunFilter::Failed => r.status == RunStatus::Failed,
    }
}
