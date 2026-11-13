//! The model list: the models table and its sortable column headers.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::class_table::num_cell;
use crate::api::{Model, time};
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::{FirstStep, loading_rows, page_header, th};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let models = self.store.models.ready().cloned();
        let subtitle = match (&models, self.store.models.error()) {
            (Some(m), _) => format!(
                "{} · runs' best and last checkpoints, plus published and imported weights",
                plural(m.len(), "model")
            ),
            (None, Some(_)) => "The model list didn't load".to_string(),
            (None, None) => "Loading…".to_string(),
        };
        let open = self
            .button(
                "models-open",
                // No model in hand on the index: it opens Playground (a model's page tries that one).
                "Open Playground",
                Btn::Secondary,
                Some("icons/scan-eye.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Playground, w, cx)),
            )
            .into_any_element();
        let header = page_header(&t, "Models", subtitle, vec![open]);
        if let Some(e) = self.store.models.error().cloned() {
            let err = self.error_panel(&e, "models-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }
        let now = self.store.api.now();
        // Narrow windows drop whole columns (text truncates against pre-shrink widths).
        // Budget: every column at full width needs about 1,030 px of content; Params goes first
        // (below 1360), then From run (below 1240).
        let vw = window.viewport_size().width;
        let wide = vw >= px(1240.);
        let params_col = vw >= px(1360.);
        let cols = |d: folo_ui::Stateful<folo_ui::Div>| d.flex().items_center().gap(px(16.)).px(px(16.));
        // The name takes what the other columns leave (text truncates against a known width, so
        // it's computed: content width less the fixed columns, their 16 px gaps, row padding).
        let name_w = {
            let content = self.content_width(window);
            // Architecture is 150 px (its "folo-s · 640 px" plus the e2e tag).
            let fixed = [150., 130., 72., 84., 84.].iter().sum::<f32>()
                + if wide { 150. } else { 0. }
                + if params_col { 72. } else { 0. };
            let columns = 6. + f32::from(u8::from(wide)) + f32::from(u8::from(params_col));
            // Row padding, panel borders, and a few px of slack for rounding.
            (content - fixed - (columns - 1.) * 16. - 32. - 2. - 8.).clamp(200., 520.)
        };
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Model").flex_1().min_w(px(200.)))
            .child(th(&t, "Architecture").w(px(150.)).flex_none())
            .when(wide, |d| d.child(th(&t, "From run").w(px(150.)).flex_none()))
            .child(self.sort_th("map", "mAP50-95", 130., true, window, cx))
            .child(self.sort_th("map50", "mAP50", 72., true, window, cx))
            .when(params_col, |d| d.child(num_cell(72., th(&t, "Params"))))
            .child(self.sort_th("size", "File size", 84., true, window, cx))
            .child(self.sort_th("created", "Created", 84., true, window, cx));
        // Keyed by the list's content (ids and the sorted fields), so a reload or another backend
        // with the same length never reuses a stale order.
        let src = self.store.models.ready().map_or(0, |v| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for m in v {
                (
                    &m.id,
                    &m.created,
                    m.bytes,
                    m.metrics.as_ref().map(|x| (x.map50_95.to_bits(), x.map50.to_bits())),
                )
                    .hash(&mut h);
            }
            h.finish() as usize
        });
        let models = models.map(|m| {
            let Some((k, desc)) = self.models_sort else { return m };
            let key = (src, m.len(), k, desc);
            let mut cache = self.models_order.borrow_mut();
            if cache.as_ref().is_none_or(|(c, _)| *c != key) {
                let score = |m: &Model| -> i64 {
                    let v = match k {
                        "map" => m.metrics.as_ref().map_or(-1, |x| (x.map50_95 * 1e6) as i64),
                        "map50" => m.metrics.as_ref().map_or(-1, |x| (x.map50 * 1e6) as i64),
                        "size" => m.bytes as i64,
                        _ => time::parse(&m.created).unwrap_or(0),
                    };
                    if desc { -v } else { v }
                };
                let mut idx: Vec<usize> = (0..m.len()).collect();
                idx.sort_by_cached_key(|&i| score(&m[i]));
                *cache = Some((key, idx));
            }
            cache.as_ref().map_or(m.clone(), |(_, idx)| idx.iter().map(|&i| m[i].clone()).collect())
        });
        let body = match models {
            None => panel(&t).child(head).child(loading_rows(&t, "models", 4)).into_any_element(),
            Some(m) if m.is_empty() => {
                // An empty server: what to do first, and the button that does it.
                let step = self.next_first_step();
                let body = match step {
                    FirstStep::AddDataset => {
                        "Models come from training runs, and a run needs a dataset first."
                    }
                    FirstStep::NewRun => {
                        "Train a run: when it finishes, its best weights appear here, ready to evaluate, export or try in the Playground."
                    }
                    FirstStep::Wait => {
                        "When a run finishes, its best weights appear here, ready to evaluate, export or try in the Playground."
                    }
                };
                let first = self.first_step_button("models-first", step, window, cx);
                panel(&t)
                    .child(
                        empty_state(&t, "icons/box.svg", "No models yet", body, Tone::Neutral)
                            .children(first),
                    )
                    .into_any_element()
            }
            Some(m) => {
                let ids: Vec<String> = m.iter().map(|m| m.id.clone()).collect();
                let rows: Vec<AnyElement> = m
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let metrics = m.metrics;
                        let (p, pu) = count_si(m.params);
                        let (b, bu) = bytes(m.bytes);
                        let published = m.source == Some(crate::api::ModelSource::Published);
                        let run = m.run_id.as_ref().and_then(|r| self.store.run(r)).map_or(
                            if published { "published".to_string() } else { "imported".to_string() },
                            |r| format!("from {}", r.name),
                        );
                        let id = m.id.clone();
                        let key = format!("model-row-{}", m.id);
                        let ids_k = ids.clone();
                        // Keyboard focus shows what hover does: the full name, the exact time, and
                        // what the e2e tag means.
                        let e2e = self.model_e2e(m);
                        let focus_text = self.ring(&self.fh(key.clone(), cx), window).then(|| {
                            let base = time::parse(&m.created).map_or(m.name.clone(), |c| {
                                format!("{} · created {}", m.name, time::local_datetime(c))
                            });
                            if e2e { base + " · end-to-end head: predicts without NMS" } else { base }
                        });
                        // What its numbers were measured on: the dataset of the run that trained it
                        // (mAPs compare only within one dataset), or that it isn't known.
                        let eval_on = m
                            .run_id
                            .as_ref()
                            .and_then(|r| self.store.run(r))
                            .map(|r| {
                                self.store.run_dataset_name(r)
                            })
                            .map_or_else(
                                // Published weights say what they are (`about`), not "unknown".
                                || {
                                    m.about
                                        .clone()
                                        .filter(|_| published)
                                        .unwrap_or_else(|| "Evaluation dataset unknown".to_string())
                                },
                                |d| format!("on {d}"),
                            )
                            // Then the weights as stored, so an f16 export reads as one by its record,
                            // not its name.
                            + &model_storage(m).map_or(String::new(), |s| format!(" · {s}"));
                        let row = cols(div().id(SharedString::from(key.clone())))
                            .when_some(focus_text, |d, s| d.child(crate::controls::focus_tip(&t, s, 48.)))
                            .h(px(48.))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, w, cx| {
                                let next = match ev.keystroke.key.as_str() {
                                    "down" => i + 1,
                                    "up" if i > 0 => i - 1,
                                    _ => return,
                                };
                                if let Some(id) = ids_k.get(next) {
                                    this.focus_key(&format!("model-row-{id}"), w, cx);
                                    cx.stop_propagation();
                                }
                            }))
                            .child(
                                div()
                                    .id(SharedString::from(format!("m-name-{}", m.id)))
                                    .flex_1()
                                    .min_w(px(200.))
                                    .when(crate::ui::may_overflow(&m.name, ty::SM.size, name_w), |d| {
                                        d.tooltip(crate::controls::tip(m.name.clone(), None))
                                    })
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .w(px(name_w))
                                            .ellipsis()
                                            .child(m.name.clone()),
                                    )
                                    .child(
                                        text(ty::SM, t.text_tertiary).w(px(name_w)).ellipsis().child(eval_on),
                                    ),
                            )
                            .child(
                                div()
                                    .w(px(150.))
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .tnum()
                                            .whitespace_nowrap()
                                            .child(format!("{} · {} px", m.arch.as_str(), m.imgsz)),
                                    )
                                    .when(self.model_e2e(m), |d| {
                                        d.child(e2e_tag(&t, format!("e2e-m-{}", m.id)))
                                    }),
                            )
                            .when(wide, |d| {
                                d.child(
                                    text(ty::SM, t.text_tertiary)
                                        .w(px(150.))
                                        .flex_none()
                                        .ellipsis()
                                        .child(run.trim_start_matches("from ").to_string()),
                                )
                            })
                            .child(num_cell(
                                130.,
                                metrics.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |mm| {
                                    text(ty::SM, t.text_primary).tnum().child(f3(mm.map50_95))
                                }),
                            ))
                            .child(num_cell(
                                72.,
                                text(ty::SM, t.text_secondary)
                                    .tnum()
                                    .child(metrics.as_ref().map_or("–".into(), |mm| f3(mm.map50))),
                            ))
                            .when(params_col, |d| {
                                d.child(num_cell(
                                    72.,
                                    text(ty::SM, t.text_secondary).tnum().child(format!("{p} {pu}")),
                                ))
                            })
                            .child(num_cell(
                                84.,
                                text(ty::SM, t.text_secondary).tnum().child(format!("{b} {bu}")),
                            ))
                            .child(num_cell(84., {
                                // Relative in the column, the exact local time on hover.
                                let at = time::parse(&m.created);
                                text(ty::SM, t.text_tertiary)
                                    .id(SharedString::from(format!("mc-{}", m.id)))
                                    .when_some(at, |d, c| {
                                        d.tooltip(crate::controls::tip(time::local_datetime(c), None))
                                    })
                                    .tnum()
                                    .child(at.map_or(String::new(), |c| time::ago(c, now)))
                            }));
                        self.bind(
                            row,
                            &key,
                            0.,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    })
                    .collect();
                panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
            }
        };
        self.page_frame(&t, header, body, cx)
    }

    /// A sortable column header of the Models table.
    fn sort_th(
        &self,
        key: &'static str,
        label: &'static str,
        w: f32,
        right: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        self.sort_th_on(self.models_sort, |a| &mut a.models_sort, "sort", key, label, w, right, window, cx)
    }

    /// A sortable column header: click or Enter/Space sorts by it (descending first, then flips).
    /// `current` is the table's sort and `field` where it lives; `prefix` keeps element ids
    /// apart between tables.
    #[allow(clippy::too_many_arguments)]
    pub fn sort_th_on(
        &self,
        current: Option<(&'static str, bool)>,
        field: fn(&mut Self) -> &mut Option<(&'static str, bool)>,
        prefix: &str,
        key: &'static str,
        label: &'static str,
        w: f32,
        right: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let on = current.filter(|s| s.0 == key);
        let chevron = |desc: bool| {
            icon(if desc { "icons/chevron-down.svg" } else { "icons/chevron-up.svg" }, 12., t.text_primary)
        };
        // Unsorted columns: a faint up-down mark, a different shape from the active direction.
        let mark = |on: Option<(&'static str, bool)>| match on {
            Some((_, desc)) => chevron(desc),
            None => icon("icons/chevrons-up-down.svg", 12., t.text_disabled),
        };
        let id = format!("{prefix}-{key}");
        let el = div()
            .id(SharedString::from(id.clone()))
            .w(px(w))
            .flex_none()
            .h(px(24.))
            .flex()
            .items_center()
            .when(right, |d| d.justify_end())
            .gap(px(2.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(format!("Sort by {label}"), None))
            // The chevron sits on the side away from the values, so the label stays aligned with them.
            // Unsorted columns show a faint one, so they read as sortable before the first click.
            .when(right, |d| d.child(mark(on)))
            .child(
                text(ty::SM, if on.is_some() { t.text_primary } else { t.text_tertiary })
                    .font_weight(FontWeight::MEDIUM)
                    .whitespace_nowrap()
                    .child(label),
            )
            .when(!right, |d| d.child(mark(on)));
        self.bind(
            el,
            &id,
            radius::SM,
            window,
            cx,
            act(move |this, _, cx| {
                let sort = field(this);
                *sort = match *sort {
                    Some((k, desc)) if k == key => Some((key, !desc)),
                    _ => Some((key, true)),
                };
                cx.notify();
            }),
        )
        .into_any_element()
    }
}
