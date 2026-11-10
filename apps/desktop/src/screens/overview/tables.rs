//! The Recent runs and Best models tables.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::SIDE_W;
use crate::api::{Run, RunExt, time};
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::{loading_rows, th};
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn recent_runs(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        // The list is never squeezed beside Best model (`tables_beside` stacks them first), so
        // the architecture and dataset always have their column.
        let created = true;
        // The list's width: the content (stacked) or what Best model leaves beside it.
        // The wider name and dataset columns need 918 px (812 of columns, six 12 px gaps, row
        // padding, margins, borders); with less, the narrower set (798 px) keeps Created inside.
        let content = self.content_width(window);
        let beside = self.tables_beside(window);
        let list_w = if beside { content - (SIDE_W + 40.) - 16. } else { content };
        // 8 px of slack on every budget: a row exactly as wide as its slot can overflow it by a
        // rounding error, and GPUI then lays the whole page out at its min-content width (the
        // hero chart collapsed to its axis labels at 1680).
        const SLACK: f32 = 8.;
        let full = list_w >= 918. + SLACK;
        let (name_w, prog_w, best_w, when_w) =
            if full { (200., 160., 96., 80.) } else { (150., 120., 96., 80.) };
        // At least its heading, "Architecture · dataset".
        let meta_w = if full { 180. } else { 150. };
        // Stacked, width past the set's need goes to the name, arch · dataset and
        // progress (2 : 1 : 1, to their caps); only what's left opens before the numbers. Beside
        // Best model the set stays as is: widened there, GPUI 0.2.2 laid the page out at its
        // min-content width (the hero chart collapsed to its axis at 1680).
        let spare = if beside { 0. } else { list_w - SLACK - if full { 918. } else { 798. } };
        let [name_w, meta_w, prog_w] =
            crate::ui::share_spare(spare, [(name_w, 2., 280.), (meta_w, 1., 240.), (prog_w, 1., 200.)]);
        let now = self.store.api.now();
        let all = self.view_all("ov-runs-all", Route::Runs, window, cx);
        let header = panel_header(&t, "Recent runs").child(all);
        // Column heads on the rows' grid, so the metric label sits over its values.
        let heads = div()
            .h(px(28.))
            .mx(px(4.))
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(12.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Run").w(px(name_w)).flex_none())
            .child(th(&t, "Architecture · dataset").w(px(meta_w)).flex_none())
            .child(th(&t, "Status").w(px(96.)).flex_none())
            .child(th(&t, "Progress").w(px(prog_w)).flex_none())
            .child(div().flex_1())
            .child(div().w(px(best_w)).flex_none().flex().justify_end().child(th(&t, "Best mAP50-95")))
            .when(created, |d| {
                d.child(div().w(px(when_w)).flex_none().flex().justify_end().child(th(&t, "Created")))
            });
        let body = match &self.store.runs {
            Load::Ready(runs) if runs.is_empty() => div()
                .child(empty_state(
                    &t,
                    "icons/activity.svg",
                    "No runs yet",
                    "Training runs appear here with their status and best mAP.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Load::Ready(runs) => {
                let runs: Vec<Run> = runs.iter().take(5).cloned().collect();
                div()
                    .flex()
                    .flex_col()
                    .pb(px(4.))
                    .child(heads)
                    .children(runs.into_iter().map(|r| {
                        let created_at = time::parse(&r.created);
                        let when = created_at.map_or(String::new(), |c| time::ago(c, now));
                        // Progress (training runs only) right after Status; best mAP50-95 at the end.
                        let training = r.status.is_active() && r.status != crate::api::RunStatus::Queued;
                        let progress_el =
                            div().w(px(prog_w)).flex_none().flex().items_center().gap(px(8.)).map(|d| {
                                if r.status == crate::api::RunStatus::Queued {
                                    // The Runs table's phrase, so a queued run reads the same everywhere.
                                    return d.child(
                                        text(ty::SM, t.text_tertiary)
                                            .whitespace_nowrap()
                                            .child(waiting_text(&r)),
                                    );
                                }
                                let fill = match r.status {
                                    _ if training => t.accent_fill(),
                                    crate::api::RunStatus::Failed => t.danger_solid,
                                    _ => t.text_tertiary,
                                };
                                // A bar under 48 px reads as a stub: narrow columns keep the
                                // number only.
                                d.when(prog_w - 64. >= 48., |d| {
                                    d.child(div().flex_1().child(progress(&t, r.progress(), fill, 4.)))
                                })
                                .child(
                                    text(ty::SM, t.text_secondary)
                                        .tnum()
                                        .w(px(56.))
                                        .flex_none()
                                        .whitespace_nowrap()
                                        .text_right()
                                        .child(format!("{} / {}", r.epoch, r.epochs)),
                                )
                            });
                        let best = div().w(px(best_w)).flex_none().flex().justify_end().child(
                            text(ty::SM, if r.best.is_some() { t.text_primary } else { t.text_tertiary })
                                .tnum()
                                .child(r.best.as_ref().map_or("–".to_string(), |b| f3(b.map50_95))),
                        );
                        let id = r.id.clone();
                        let key = format!("ov-run-{}", r.id);
                        // Keyboard focus shows what hover does: the full name and the exact time.
                        let e2e = r.config.head == crate::api::Head::O2o;
                        // The focus tip says in full what the row may cut: name, architecture,
                        // dataset, created time, and the head.
                        let focus_text = self.ring(&self.fh(key.clone(), cx), window).then(|| {
                            let who = format!("{} · {}", r.name, self.run_meta(&r));
                            let base = created_at.map_or(who.clone(), |c| {
                                format!("{who} · created {}", time::local_datetime(c))
                            });
                            if e2e { base + " · end-to-end head: predicts without NMS" } else { base }
                        });
                        // Architecture first (short, so it survives a narrow cell), then the
                        // dataset, then the e2e tag for an end-to-end run.
                        // The e2e tag rides its architecture, as in the Runs and Models lists; the
                        // dataset after them gives way first.
                        let ds_name = self
                            .store
                            .dataset_summary(&r.config.dataset_id)
                            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
                        let meta = |w: f32| {
                            div().w(px(w)).flex_none().child(meta_row(
                                ty::SM,
                                t.text_tertiary,
                                r.config.arch.as_str(),
                                e2e.then(|| e2e_tag(&t, format!("e2e-ovr-{}-{w}", r.id))),
                                std::slice::from_ref(&ds_name),
                            ))
                        };
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .when_some(focus_text, |d, s| d.child(crate::controls::focus_tip(&t, s, 36.)))
                            .h(px(36.))
                            .mx(px(4.))
                            .px(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            // Same order as the Runs table: name, then status.
                            // One line per run: name, then dataset and architecture, then status as a dot.
                            // A name that may be cut shows in full on hover (and on focus, above).
                            .child(
                                div()
                                    .id(SharedString::from(format!("ov-rn-{}", r.id)))
                                    .w(px(name_w))
                                    .flex_none()
                                    .when(crate::ui::may_overflow(&r.name, ty::SM.size, name_w), |d| {
                                        d.tooltip(crate::controls::tip(r.name.clone(), None))
                                    })
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .w_full()
                                            .ellipsis()
                                            .child(r.name.clone()),
                                    ),
                            )
                            // Wide: the dataset and architecture in their own column.
                            .child(meta(meta_w))
                            .child(div().w(px(96.)).flex_none().child(status_dot(
                                &t,
                                &r,
                                &format!("ov-{}", r.id),
                            )))
                            .child(progress_el)
                            .child(div().flex_1())
                            .child(best)
                            .when(created, |d| {
                                d.child(
                                    div().w(px(when_w)).flex_none().flex().justify_end().child(
                                        text(ty::SM, t.text_tertiary)
                                            .id(SharedString::from(format!("ov-rc-{}", r.id)))
                                            .when_some(created_at, |d, c| {
                                                d.tooltip(crate::controls::tip(time::local_datetime(c), None))
                                            })
                                            .tnum()
                                            .whitespace_nowrap()
                                            .child(when),
                                    ),
                                )
                            });
                        self.bind(
                            row,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    }))
                    .into_any_element()
            }
            _ => loading_rows(&t, "ov-runs", 5).into_any_element(),
        };
        panel(&t)
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(header)
            .child(body)
            .into_any_element()
    }

    pub(super) fn top_models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-models-all", Route::Models, window, cx);
        // mAP compares only within one dataset's validation split, so the card ranks within each
        // dataset and shows its best: one row per dataset, never one list across them.
        let header = panel_header(&t, "Best model per dataset").child(all);
        let body = match &self.store.models {
            Load::Ready(models) if models.is_empty() => div()
                .child(empty_state(
                    &t,
                    "icons/box.svg",
                    "No models yet",
                    "Finished runs save their best weights here.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Load::Ready(models) => 'card: {
                // Each evaluated model's dataset (from the run that trained it; an imported model
                // without one is left out: there's nothing to compare it with).
                let ds_of = |m: &crate::api::Model| {
                    m.run_id.as_ref().and_then(|r| self.store.run(r)).map(|r| r.config.dataset_id.clone())
                };
                let mut best: Vec<(String, crate::api::Model)> = Vec::new();
                for m in models.iter().filter(|m| m.metrics.is_some()) {
                    let Some(ds) = ds_of(m) else { continue };
                    let map = |m: &crate::api::Model| m.metrics.as_ref().map_or(0., |x| x.map50_95);
                    match best.iter_mut().find(|(d, _)| *d == ds) {
                        Some((_, b)) if map(m) > map(b) => *b = m.clone(),
                        Some(_) => {}
                        None => best.push((ds, m.clone())),
                    }
                }
                // Datasets in name order (their mAPs aren't comparable, so no rank between rows).
                let name_of =
                    |ds: &str| self.store.dataset_summary(ds).map_or(ds.to_string(), |d| d.name.clone());
                best.sort_by_key(|(ds, _)| name_of(ds));
                let models: Vec<(String, crate::api::Model)> =
                    best.into_iter().map(|(ds, m)| (name_of(&ds), m)).collect();
                // Models, but none to rank yet: unevaluated, imported, or their runs still loading.
                if models.is_empty() {
                    let body = if self.store.runs.error().is_some() {
                        "Runs didn't load, so the datasets models were trained on aren't known: retry above."
                    } else if self.store.runs.ready().is_none() {
                        "Waiting for runs to say which dataset each model was trained on."
                    } else {
                        "Evaluated models trained here show up per dataset."
                    };
                    break 'card div()
                        .child(empty_state(
                            &t,
                            "icons/box.svg",
                            "Nothing to compare yet",
                            body,
                            Tone::Neutral,
                        ))
                        .into_any_element();
                }
                // Stacked under Recent runs (full width), the table carries the Models list's
                // architecture, mAP50 and age too, rather than three columns across 1,100 px.
                let wide = !self.tables_beside(window);
                let now = self.store.api.now();
                // Column heads like Recent runs beside it, so the two lists share row lines.
                let heads = div()
                    .h(px(28.))
                    .mx(px(4.))
                    .px(px(12.))
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    // The dataset leads: it's what the rows are, one each, in name order.
                    .child(th(&t, "Dataset").w(px(120.)).flex_none())
                    .child(th(&t, "Best model").flex_1())
                    .when(wide, |d| d.child(th(&t, "Architecture").w(px(120.)).flex_none()))
                    .child(div().w(px(64.)).flex_none().flex().justify_end().child(th(&t, "mAP50-95")))
                    .when(wide, |d| {
                        d.child(div().w(px(64.)).flex_none().flex().justify_end().child(th(&t, "mAP50")))
                            .child(div().w(px(80.)).flex_none().flex().justify_end().child(th(&t, "Created")))
                    });
                div()
                    .flex()
                    .flex_col()
                    .pb(px(4.))
                    .child(heads)
                    .children(models.into_iter().take(5).map(|(ds_name, m)| {
                        let id = m.id.clone();
                        let key = format!("ov-model-{}", m.id);
                        let map = m.metrics.as_ref().map_or(0., |m| m.map50_95);
                        let map50 = m.metrics.as_ref().map(|m| m.map50);
                        let arch = format!("{} · {} px", m.arch.as_str(), m.imgsz);
                        let created = time::parse(&m.created).map_or(String::new(), |c| time::ago(c, now));
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .h(px(36.))
                            .mx(px(4.))
                            .px(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .w(px(120.))
                                    .flex_none()
                                    .ellipsis()
                                    .child(ds_name.clone()),
                            )
                            // Its best model, then the same compact e2e tag as the Models table.
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .min_w_0()
                                            .ellipsis()
                                            .child(m.name.clone()),
                                    )
                                    .when(self.model_e2e(&m), |d| {
                                        d.child(e2e_tag(&t, format!("e2e-ov-{}", m.id)))
                                    }),
                            )
                            .when(wide, |d| {
                                d.child(
                                    text(ty::SM, t.text_secondary)
                                        .tnum()
                                        .w(px(120.))
                                        .flex_none()
                                        .ellipsis()
                                        .child(arch),
                                )
                            })
                            // The number alone: a bar would set rows from different datasets against
                            // each other, which their mAPs can't support.
                            .child(
                                div()
                                    .w(px(64.))
                                    .flex_none()
                                    .flex()
                                    .justify_end()
                                    .child(text(ty::SM, t.text_secondary).tnum().child(f3(map))),
                            )
                            .when(wide, |d| {
                                d.child(div().w(px(64.)).flex_none().flex().justify_end().child(
                                    text(ty::SM, t.text_tertiary).tnum().child(map50.map_or("–".into(), f3)),
                                ))
                                .child(
                                    div().w(px(80.)).flex_none().flex().justify_end().child(
                                        text(ty::SM, t.text_tertiary).whitespace_nowrap().child(created),
                                    ),
                                )
                            });
                        self.bind(
                            row,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    }))
                    .into_any_element()
            }
            // A failed list says so and can be fetched again (it would otherwise read as loading
            // forever: nothing else retries a failed load).
            Load::Failed(e) => {
                let retry = self.button(
                    "ov-models-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_models(cx)),
                );
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .items_start()
                    .gap(px(8.))
                    .child(text(ty::SM, t.danger_text).whitespace_normal().child(e.message.clone()))
                    .child(retry)
                    .into_any_element()
            }
            _ => loading_rows(&t, "ov-models", 5).into_any_element(),
        };
        panel(&t)
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(header)
            .child(body)
            .into_any_element()
    }
}
