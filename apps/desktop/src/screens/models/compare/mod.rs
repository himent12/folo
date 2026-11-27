//! Side-by-side model comparison (`Route::ModelCompare` / Versus) with summary metrics, per-class
//! AP deltas, and copy/save CSV & Markdown export.

pub(crate) mod data;
#[cfg(test)]
mod tests;

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, IntoElement, ParentElement, SharedString, Styled, Window, div, px,
};

use crate::api::{Head, Model, ModelEval};
use crate::app::{FoloApp, Route, act, export_default_dir, picked_save};
use crate::arch::ModelArch;
use crate::controls::Btn;
use crate::screens::models::groups::eval_set_with_report;
use crate::screens::playground::write_export_bytes;
use crate::screens::{loading_rows, page_header};
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;

pub use data::{
    MODEL_COMPARE_MAX, ModelCompareState, build_model_compare_csv, build_model_compare_markdown,
    build_per_class_rows, fmt_signed_delta, fmt_signed_delta2, model_compare_filename,
    reconcile_compare_models, same_eval_set_with_evals,
};

struct CompareExportInputs {
    models: Vec<Model>,
    archs: Vec<ModelArch>,
    sets: Vec<String>,
    evals: Vec<Option<ModelEval>>,
    same: bool,
}

impl FoloApp {
    /// Opens side-by-side Model Compare (`Route::ModelCompare`), optionally seeding `seed` as
    /// baseline slot 0. Preserves `seed` when `store.models` is not loaded yet.
    pub fn open_model_compare(&mut self, seed: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(sid) = seed {
            self.model_compare.ids.retain(|id| id != &sid);
            self.model_compare.ids.insert(0, sid);
        }
        if let Some(models) = self.store.models.ready().cloned() {
            reconcile_compare_models(&mut self.model_compare.ids, &self.store, &models);
        }
        self.navigate(Route::ModelCompare, window, cx);
    }

    pub fn model_compare_add(&mut self, id: String, cx: &mut Context<Self>) {
        if self.model_compare.ids.len() >= MODEL_COMPARE_MAX || self.model_compare.ids.contains(&id) {
            return;
        }
        self.model_compare.ids.push(id);
        cx.notify();
    }

    pub fn model_compare_remove(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.model_compare.ids.len() <= 2 {
            return;
        }
        self.model_compare.ids.retain(|x| x != id);
        cx.notify();
    }

    pub fn model_compare_make_baseline(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(pos) = self.model_compare.ids.iter().position(|x| x == id) {
            let item = self.model_compare.ids.remove(pos);
            self.model_compare.ids.insert(0, item);
            cx.notify();
        }
    }

    /// Snapshot of currently compared models, canonical architectures, evaluation set labels, and
    /// loaded evals for export actions. Uses loaded [`ModelEval`] provenance when available.
    fn model_compare_export_inputs(&self) -> Option<CompareExportInputs> {
        let all = self.store.models.ready()?;
        let mut ids = self.model_compare.ids.clone();
        reconcile_compare_models(&mut ids, &self.store, all);
        let models: Vec<Model> =
            ids.iter().filter_map(|id| all.iter().find(|m| m.id == *id).cloned()).collect();
        if models.is_empty() {
            return None;
        }
        let archs: Vec<ModelArch> = models.iter().map(|m| self.store.model_arch(m)).collect();
        let evals: Vec<Option<ModelEval>> =
            models.iter().map(|m| self.store.evals.get(&m.id).and_then(|l| l.ready()).cloned()).collect();
        let sets: Vec<String> = models
            .iter()
            .zip(evals.iter())
            .map(|(m, ev)| eval_set_with_report(&self.store, m, ev.as_ref()).label)
            .collect();
        let refs: Vec<&Model> = models.iter().collect();
        let e_refs: Vec<Option<&ModelEval>> = evals.iter().map(Option::as_ref).collect();
        let same = same_eval_set_with_evals(&self.store, &refs, &e_refs);
        Some(CompareExportInputs { models, archs, sets, evals, same })
    }

    pub fn copy_model_compare_csv(&mut self, cx: &mut Context<Self>) {
        let Some(inp) = self.model_compare_export_inputs() else {
            self.toast(Tone::Neutral, "No models to compare yet", None, cx);
            return;
        };
        let m_refs: Vec<&Model> = inp.models.iter().collect();
        let e_refs: Vec<Option<&ModelEval>> = inp.evals.iter().map(Option::as_ref).collect();
        let csv = build_model_compare_csv(&m_refs, &inp.archs, &inp.sets, &e_refs, inp.same);
        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(csv));
        self.toast(
            Tone::Success,
            "Copied comparison CSV",
            Some(
                format!("{} compared with summary & per-class AP deltas.", plural(inp.models.len(), "model"))
                    .into(),
            ),
            cx,
        );
    }

    pub fn copy_model_compare_markdown(&mut self, cx: &mut Context<Self>) {
        let Some(inp) = self.model_compare_export_inputs() else {
            self.toast(Tone::Neutral, "No models to compare yet", None, cx);
            return;
        };
        let m_refs: Vec<&Model> = inp.models.iter().collect();
        let e_refs: Vec<Option<&ModelEval>> = inp.evals.iter().map(Option::as_ref).collect();
        let md = build_model_compare_markdown(&m_refs, &inp.archs, &inp.sets, &e_refs, inp.same);
        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(md));
        self.toast(
            Tone::Success,
            "Copied comparison Markdown",
            Some(
                format!(
                    "{} compared with GFM summary & per-class AP tables.",
                    plural(inp.models.len(), "model")
                )
                .into(),
            ),
            cx,
        );
    }

    pub fn save_model_compare_csv(&mut self, cx: &mut Context<Self>) {
        let Some(inp) = self.model_compare_export_inputs() else {
            self.toast(Tone::Neutral, "No models to compare yet", None, cx);
            return;
        };
        let m_refs: Vec<&Model> = inp.models.iter().collect();
        let e_refs: Vec<Option<&ModelEval>> = inp.evals.iter().map(Option::as_ref).collect();
        let filename = model_compare_filename(&m_refs, "csv");
        let bytes = build_model_compare_csv(&m_refs, &inp.archs, &inp.sets, &e_refs, inp.same).into_bytes();
        self.save_model_compare_file("CSV", filename, bytes, cx);
    }

    pub fn save_model_compare_markdown(&mut self, cx: &mut Context<Self>) {
        let Some(inp) = self.model_compare_export_inputs() else {
            self.toast(Tone::Neutral, "No models to compare yet", None, cx);
            return;
        };
        let m_refs: Vec<&Model> = inp.models.iter().collect();
        let e_refs: Vec<Option<&ModelEval>> = inp.evals.iter().map(Option::as_ref).collect();
        let filename = model_compare_filename(&m_refs, "md");
        let bytes =
            build_model_compare_markdown(&m_refs, &inp.archs, &inp.sets, &e_refs, inp.same).into_bytes();
        self.save_model_compare_file("Markdown", filename, bytes, cx);
    }

    fn save_model_compare_file(
        &mut self,
        kind: &'static str,
        filename: String,
        bytes: Vec<u8>,
        cx: &mut Context<Self>,
    ) {
        let dir = export_default_dir();
        let rx = cx.prompt_for_new_path(&dir, Some(&filename));
        cx.spawn(async move |this, cx| {
            let Some(path) = picked_save(&this, rx.await, cx) else { return };
            let written = cx
                .background_executor()
                .spawn(async move { write_export_bytes(&path, &bytes).map(|()| path) })
                .await;
            this.update(cx, |this, cx| match written {
                Ok(saved) => {
                    let reveal = saved.clone();
                    this.toast_with(
                        Tone::Success,
                        format!("Saved comparison {kind}"),
                        Some(saved.display().to_string().into()),
                        Some(("Show in folder".into(), act(move |_, _, cx| cx.reveal_path(&reveal)))),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(
                        Tone::Danger,
                        format!("Couldn't save comparison {kind}"),
                        Some(e.to_string().into()),
                        cx,
                    );
                }
            })
            .ok();
        })
        .detach();
    }

    pub fn model_compare_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all_models = self.store.models.ready().cloned();
        let back_btn = self
            .button(
                "model-cmp-back",
                "All models",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Models, w, cx)),
            )
            .into_any_element();

        let Some(all) = all_models else {
            let header = page_header(&t, "Compare models", "Loading…".to_string(), vec![back_btn]);
            if let Some(e) = self.store.models.error().cloned() {
                let err = self.error_panel(&e, "model-cmp-retry", window, cx);
                return self.page_frame(&t, header, err, cx);
            }
            return self.page_frame(&t, header, panel(&t).child(loading_rows(&t, "model-cmp", 4)), cx);
        };

        reconcile_compare_models(&mut self.model_compare.ids, &self.store, &all);
        let chosen: Vec<Model> = self
            .model_compare
            .ids
            .iter()
            .filter_map(|id| all.iter().find(|m| m.id == *id).cloned())
            .collect();

        for m in &chosen {
            self.ensure_eval(&m.id, cx);
        }

        if chosen.len() < 2 {
            let header = page_header(
                &t,
                "Compare models",
                "Need at least 2 trained models to compare side by side".to_string(),
                vec![back_btn],
            );
            let empty = panel(&t).child(empty_state(
                &t,
                "icons/git-compare-arrows.svg",
                "Need 2 models to compare",
                "Finish at least two training runs to compare summary metrics and per-class AP deltas.",
                Tone::Neutral,
            ));
            return self.page_frame(&t, header, empty, cx);
        }

        let copy_csv = self
            .button(
                "model-cmp-copy-csv",
                "Copy CSV",
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(|this, _, cx| this.copy_model_compare_csv(cx)),
            )
            .into_any_element();
        let save_csv = self
            .button(
                "model-cmp-save-csv",
                "Save CSV",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(|this, _, cx| this.save_model_compare_csv(cx)),
            )
            .into_any_element();
        let copy_md = self
            .button(
                "model-cmp-copy-md",
                "Copy Markdown",
                Btn::Secondary,
                Some("icons/file-text.svg"),
                window,
                cx,
                act(|this, _, cx| this.copy_model_compare_markdown(cx)),
            )
            .into_any_element();
        let save_md = self
            .button(
                "model-cmp-save-md",
                "Save Markdown",
                Btn::Primary,
                Some("icons/download.svg"),
                window,
                cx,
                act(|this, _, cx| this.save_model_compare_markdown(cx)),
            )
            .into_any_element();

        let subtitle = format!("{} · baseline: {}", plural(chosen.len(), "model"), chosen[0].name);
        let header =
            page_header(&t, "Compare models", subtitle, vec![copy_csv, save_csv, copy_md, save_md, back_btn]);

        let refs: Vec<&Model> = chosen.iter().collect();
        let eval_refs: Vec<Option<&ModelEval>> =
            chosen.iter().map(|m| self.store.evals.get(&m.id).and_then(|l| l.ready())).collect();
        let same_set = same_eval_set_with_evals(&self.store, &refs, &eval_refs);
        let chips = self.model_compare_chips(&t, &chosen, &all, window, cx);
        let summary_panel = self.model_compare_summary(&t, &chosen, same_set, window, cx);
        let per_class_panel = self.model_compare_per_class(&t, &chosen, window, cx);

        let caveat = (!same_set).then(|| {
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(14.))
                .py(px(10.))
                .rounded(px(radius::MD))
                .bg(t.warning_muted)
                .border_1()
                .border_color(t.warning_muted)
                .child(icon("icons/triangle-alert.svg", 14., t.warning_solid))
                .child(text(ty::SM, t.warning_text).child(
                    "Compared models come from different evaluation sets or unranked splits — compare mAP deltas with caution.",
                ))
        });

        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(chips)
            .children(caveat)
            .child(summary_panel)
            .child(per_class_panel);
        self.page_frame(&t, header, body, cx)
    }

    fn model_compare_chips(
        &mut self,
        t: &Theme,
        chosen: &[Model],
        all: &[Model],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let can_remove = chosen.len() > 2;
        let mut bar = div().flex().flex_wrap().items_center().gap(px(8.));

        for (slot, m) in chosen.iter().enumerate() {
            let is_base = slot == 0;
            let arch = self.store.model_arch(m);
            let mid_open = m.id.clone();
            let open_btn = self.icon_button(
                &format!("model-cmp-open-{}", m.id),
                "icons/arrow-up-right.svg",
                "Open model",
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Model(mid_open.clone()), w, cx)),
            );
            let base_btn = (!is_base).then(|| {
                let mid_base = m.id.clone();
                self.button(
                    &format!("model-cmp-base-{}", m.id),
                    "Baseline",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(move |this, w, cx| {
                        this.model_compare_make_baseline(&mid_base, cx);
                        w.focus(&this.fh(format!("model-cmp-open-{mid_base}"), cx));
                    }),
                )
            });
            let rm_btn = can_remove.then(|| {
                let mid_rm = m.id.clone();
                let fallback_focus = chosen
                    .get(slot.saturating_sub(1))
                    .filter(|c| c.id != m.id)
                    .or_else(|| chosen.get(slot + 1))
                    .map(|c| c.id.clone());
                self.icon_button(
                    &format!("model-cmp-rm-{}", m.id),
                    "icons/x.svg",
                    "Remove from comparison",
                    window,
                    cx,
                    act(move |this, w, cx| {
                        this.model_compare_remove(&mid_rm, cx);
                        if let Some(next_id) = fallback_focus.as_deref() {
                            w.focus(&this.fh(format!("model-cmp-open-{next_id}"), cx));
                        }
                    }),
                )
            });
            let chip = div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(10.))
                .py(px(6.))
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if is_base { t.accent_line } else { t.line_subtle })
                .child(
                    text(ty::SM, t.text_primary)
                        .font_weight(FontWeight::MEDIUM)
                        .max_w(px(180.))
                        .ellipsis()
                        .child(m.name.clone()),
                )
                .child(badge(&t, arch.as_str(), if arch.is_v2() { Tone::Accent } else { Tone::Neutral }))
                .when(is_base, |d| d.child(badge(&t, "baseline", Tone::Accent)))
                .children(base_btn)
                .child(open_btn)
                .children(rm_btn);
            bar = bar.child(chip);
        }

        let candidates: Vec<&Model> = all.iter().filter(|m| !chosen.iter().any(|c| c.id == m.id)).collect();
        if chosen.len() < MODEL_COMPARE_MAX && !candidates.is_empty() {
            let cand_ids: Vec<String> = candidates.iter().map(|c| c.id.clone()).collect();
            let options: Vec<(SharedString, Option<SharedString>)> = candidates
                .iter()
                .map(|c| {
                    let arch = self.store.model_arch(c).as_str();
                    let detail = match &c.metrics {
                        Some(x) => format!("{arch} · mAP {}", f3(x.map50_95)),
                        None => arch.to_string(),
                    };
                    (c.name.clone().into(), Some(detail.into()))
                })
                .collect();
            let adder = self.select(
                "model-cmp-add",
                "Add model…".into(),
                options,
                candidates.len(),
                window,
                cx,
                Rc::new(move |this, i, w, cx| {
                    if let Some(mid) = cand_ids.get(i).cloned() {
                        let will_unmount =
                            this.model_compare.ids.len() + 1 >= MODEL_COMPARE_MAX || cand_ids.len() <= 1;
                        this.model_compare_add(mid.clone(), cx);
                        if will_unmount {
                            w.focus(&this.fh(format!("model-cmp-open-{mid}"), cx));
                        }
                    }
                }),
            );
            bar = bar.child(div().w(px(220.)).child(adder));
        }

        bar.into_any_element()
    }

    fn model_compare_summary(
        &self,
        t: &Theme,
        chosen: &[Model],
        same_set: bool,
        window: &Window,
        _cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let lead_w = if self.content_width(window) < 880. { 130. } else { 164. };

        let mut head = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .px(px(16.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(lead_w)).flex_none().child(caption(&t, "Metric")));
        for (slot, m) in chosen.iter().enumerate() {
            head = head.child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .justify_end()
                    .gap(px(6.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .ellipsis()
                            .child(m.name.clone()),
                    )
                    .when(slot == 0, |d| d.child(badge(&t, "base", Tone::Accent))),
            );
        }

        let row_frame = |label: &'static str, cells: Vec<AnyElement>| {
            let mut r = div()
                .flex()
                .items_center()
                .gap(px(12.))
                .px(px(16.))
                .min_h(px(36.))
                .py(px(6.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(div().w(px(lead_w)).flex_none().child(text(ty::SM, t.text_secondary).child(label)));
            for cell in cells {
                r = r.child(div().flex_1().min_w_0().flex().items_center().justify_end().child(cell));
            }
            r
        };

        let arch_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                let ma = self.store.model_arch(m);
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(text(ty::SM, t.text_primary).font_family(crate::theme::MONO).child(ma.as_str()))
                    .when(ma.is_v2(), |d| d.child(badge(&t, "v2", Tone::Accent)))
                    .into_any_element()
            })
            .collect();

        let eval_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                let ev = self.store.evals.get(&m.id).and_then(|l| l.ready());
                text(ty::SM, t.text_secondary)
                    .ellipsis()
                    .child(eval_set_with_report(&self.store, m, ev).label)
                    .into_any_element()
            })
            .collect();

        let input_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                text(ty::SM, t.text_secondary).tnum().child(format!("{} px", m.imgsz)).into_any_element()
            })
            .collect();

        let head_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                text(ty::SM, t.text_secondary)
                    .child(if m.head == Head::O2o { "end-to-end" } else { "NMS (o2m)" })
                    .into_any_element()
            })
            .collect();

        let param_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                let (v, u) = count_si(m.params);
                text(ty::SM, t.text_secondary).tnum().child(format!("{v} {u}")).into_any_element()
            })
            .collect();

        let flops_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                text(ty::SM, t.text_secondary)
                    .tnum()
                    .child(format!("{:.1} GFLOPs", m.flops_g))
                    .into_any_element()
            })
            .collect();

        let size_cells: Vec<AnyElement> = chosen
            .iter()
            .map(|m| {
                let (v, u) = bytes(m.bytes);
                text(ty::SM, t.text_secondary).tnum().child(format!("{v} {u}")).into_any_element()
            })
            .collect();

        let metric_cells = |extract: fn(&Model) -> Option<f32>,
                            decimals: usize,
                            highlight_best: bool,
                            directional: bool|
         -> Vec<AnyElement> {
            let eps = if decimals == 2 { 0.005 } else { 0.0005 };
            let vals: Vec<Option<f32>> = chosen.iter().map(extract).collect();
            let base = vals.first().copied().flatten();
            let best = if highlight_best && same_set {
                vals.iter().filter_map(|v| *v).filter(|v| v.is_finite()).reduce(f32::max)
            } else {
                None
            };
            vals.into_iter()
                .enumerate()
                .map(|(i, val)| {
                    let is_best = best.is_some_and(|b| val.is_some_and(|v| (v - b).abs() < eps));
                    let main = match val.filter(|v| v.is_finite()) {
                        Some(v) => {
                            let s = if decimals == 2 { format!("{v:.2}") } else { format!("{v:.3}") };
                            text(ty::SM, if is_best { t.accent_text } else { t.text_primary })
                                .tnum()
                                .when(is_best, |d| d.font_weight(FontWeight::SEMIBOLD))
                                .child(s)
                        }
                        None => text(ty::SM, t.text_tertiary).child("—"),
                    };
                    let delta = if i > 0 {
                        match (val.filter(|v| v.is_finite()), base.filter(|v| v.is_finite())) {
                            (Some(v), Some(b)) => {
                                let d = v - b;
                                let s =
                                    if decimals == 2 { fmt_signed_delta2(d) } else { fmt_signed_delta(d) };
                                let color = if !directional {
                                    t.text_tertiary
                                } else if d > eps {
                                    t.success_text
                                } else if d < -eps {
                                    t.danger_text
                                } else {
                                    t.text_tertiary
                                };
                                Some(text(ty::XS, color).tnum().child(s))
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    div().flex().items_baseline().gap(px(6.)).child(main).children(delta).into_any_element()
                })
                .collect()
        };

        panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Summary metrics"))
            .child(head)
            .child(row_frame("Architecture", arch_cells))
            .child(row_frame("Evaluation set", eval_cells))
            .child(row_frame("Input size", input_cells))
            .child(row_frame("Head", head_cells))
            .child(row_frame("Parameters", param_cells))
            .child(row_frame("Compute", flops_cells))
            .child(row_frame("File size", size_cells))
            .child(row_frame(
                "mAP50-95",
                metric_cells(|m| m.metrics.as_ref().map(|x| x.map50_95), 3, true, true),
            ))
            .child(row_frame("mAP50", metric_cells(|m| m.metrics.as_ref().map(|x| x.map50), 3, true, true)))
            .child(row_frame(
                "Precision",
                metric_cells(|m| m.metrics.as_ref().map(|x| x.precision), 3, false, true),
            ))
            .child(row_frame(
                "Recall",
                metric_cells(|m| m.metrics.as_ref().map(|x| x.recall), 3, false, true),
            ))
            .child(row_frame(
                "Best F1 conf",
                metric_cells(|m| m.metrics.as_ref().and_then(|x| x.best_f1_conf), 2, false, false),
            ))
            .into_any_element()
    }

    fn model_compare_per_class(
        &self,
        t: &Theme,
        chosen: &[Model],
        window: &Window,
        _cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let lead_w = if self.content_width(window) < 880. { 130. } else { 164. };
        let any_loading = chosen.iter().any(|m| self.store.evals.get(&m.id).is_some_and(|l| l.is_loading()));
        let any_failed =
            chosen.iter().any(|m| self.store.evals.get(&m.id).is_some_and(|l| l.error().is_some()));
        let refs: Vec<&Model> = chosen.iter().collect();
        let eval_refs: Vec<Option<&ModelEval>> =
            chosen.iter().map(|m| self.store.evals.get(&m.id).and_then(|l| l.ready())).collect();
        let rows = build_per_class_rows(&refs, &eval_refs);

        let status_hint = if any_loading {
            "AP50-95 · signed Δ vs baseline · loading evals…"
        } else if any_failed {
            "AP50-95 · signed Δ vs baseline · some evals failed to load"
        } else {
            "AP50-95 · signed Δ vs baseline"
        };
        let mut p = panel(&t).flex().flex_col().overflow_hidden().child(
            panel_header(&t, "Per-class AP deltas").child(
                text(ty::SM, if any_failed && !any_loading { t.warning_text } else { t.text_tertiary })
                    .child(status_hint),
            ),
        );

        if rows.is_empty() && any_loading {
            return p.child(loading_rows(&t, "model-cmp-pc", 4)).into_any_element();
        }
        if rows.is_empty() {
            let (title, desc, tone) = if any_failed {
                (
                    "Couldn't load per-class evaluation",
                    "Evaluation reports for the selected models failed to load from the server.",
                    Tone::Warning,
                )
            } else {
                (
                    "No per-class evaluation yet",
                    "Per-class AP deltas appear once models have saved evaluation reports.",
                    Tone::Neutral,
                )
            };
            return p.child(empty_state(&t, "icons/bar-chart-3.svg", title, desc, tone)).into_any_element();
        }

        let mut head = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .px(px(16.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(lead_w)).flex_none().child(caption(&t, "Class")));
        for (slot, m) in chosen.iter().enumerate() {
            head = head.child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .justify_end()
                    .gap(px(6.))
                    .child(text(ty::SM, t.text_secondary).ellipsis().child(m.name.clone()))
                    .when(slot == 0, |d| d.child(badge(&t, "base", Tone::Accent))),
            );
        }
        p = p.child(head);

        for row in rows {
            let mut r = div()
                .flex()
                .items_center()
                .gap(px(12.))
                .px(px(16.))
                .min_h(px(34.))
                .py(px(6.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    div()
                        .w(px(lead_w))
                        .flex_none()
                        .child(text(ty::SM, t.text_primary).ellipsis().child(row.name.clone())),
                );
            for (i, cell) in row.cells.iter().enumerate() {
                let main = match cell {
                    Some((ap, _)) => text(ty::SM, t.text_primary).tnum().child(f3(*ap)),
                    None => text(ty::SM, t.text_tertiary).child("—"),
                };
                let delta = row.deltas_50_95.get(i).copied().flatten().map(|d| {
                    let color = if d > 0.0005 {
                        t.success_text
                    } else if d < -0.0005 {
                        t.danger_text
                    } else {
                        t.text_tertiary
                    };
                    text(ty::XS, color).tnum().child(fmt_signed_delta(d))
                });
                r = r.child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .items_baseline()
                        .justify_end()
                        .gap(px(6.))
                        .child(main)
                        .children(delta),
                );
            }
            p = p.child(r);
        }
        p.into_any_element()
    }
}
