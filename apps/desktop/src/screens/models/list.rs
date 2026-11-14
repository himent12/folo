//! The model list: the models grouped by what they were evaluated on, and the sortable column
//! headers.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::class_table::num_cell;
use super::groups::{best_mark, best_word, group_models, same_as_best};
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
        let groups = models.as_ref().map(|m| group_models(&self.store, m, self.models_sort));
        // What the page holds (§2 page header): how many models, over how many evaluation sets.
        let subtitle = match (&models, &groups, self.store.models.error()) {
            (Some(m), Some(g), _) => {
                let ranked = g.iter().filter(|g| g.set.ranked).count();
                let mut facts = vec![plural(m.len(), "model")];
                if ranked > 0 {
                    facts.push(counted(ranked as u64, "evaluation dataset", "evaluation datasets"));
                }
                let (v, u) = bytes(m.iter().map(|m| m.bytes).sum());
                facts.push(format!("{v} {u}"));
                facts.join(" · ")
            }
            (None, _, Some(_)) => "The model list didn't load".to_string(),
            _ => "Loading…".to_string(),
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
        // Single-line rows; columns drop whole where the window is narrow (`ModelCols`), and the
        // name takes what they leave (text truncates against a known width).
        let content = self.content_width(window);
        let c = ModelCols::fit(content);
        let name_w = (content - c.fixed() - MODEL_SLACK).clamp(MODEL_NAME_MIN, 520.);
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Model").flex_1().min_w(px(MODEL_NAME_MIN)))
            .child(th(&t, "Architecture").w(px(ARCH_W)).flex_none())
            .when(c.classes, |d| d.child(self.sort_th("classes", "Classes", 64., true, window, cx)))
            .child(self.sort_th("size", "Size", 84., true, window, cx))
            .child(self.sort_th("map", "mAP50-95", MAP_W, true, window, cx))
            .when(c.map50, |d| d.child(self.sort_th("map50", "mAP50", 72., true, window, cx)))
            .when(c.source, |d| d.child(th(&t, "Source run").w(px(SOURCE_W)).flex_none()))
            .child(self.sort_th("created", "Created", 84., true, window, cx));
        // The score the list is ranked by leads in primary ink; the other in secondary.
        let strong = |key: &str| self.models_sort.map_or("map", |s| s.0) == key;
        let body = match groups {
            None => panel(&t).child(head).child(loading_rows(&t, "models", 4)).into_any_element(),
            Some(g) if g.is_empty() => {
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
            Some(groups) => {
                // Up and down walk the rows across the groups, in the order shown.
                let ids: Vec<String> =
                    groups.iter().flat_map(|g| g.rows.iter().map(|m| m.id.clone())).collect();
                let mut out: Vec<AnyElement> = Vec::new();
                let mut i = 0;
                for g in groups {
                    // The group's best (none on an unranked group): a tertiary word before its score.
                    let contenders: Vec<Option<f32>> =
                        g.rows
                            .iter()
                            .map(|m| {
                                if same_as_best(&self.store, m) {
                                    None
                                } else {
                                    m.metrics.map(|x| x.map50_95)
                                }
                            })
                            .collect();
                    let mark = g.set.ranked.then(|| best_mark(&contenders, 3, 6));
                    out.push(group_head(&t, &g.set.label, g.rows.len(), g.set.ranked).into_any_element());
                    for (j, m) in g.rows.iter().enumerate() {
                        let word = mark.as_ref().and_then(|b| {
                            if b.best.contains(&j) {
                                Some(best_word(b))
                            } else if same_as_best(&self.store, m) {
                                Some("same as best")
                            } else {
                                None
                            }
                        });
                        let decimals = mark.as_ref().map_or(3, |b| b.decimals);
                        out.push(
                            self.model_row(m, i, &ids, word, decimals, &c, name_w, now, &strong, window, cx),
                        );
                        i += 1;
                    }
                }
                panel(&t).overflow_hidden().child(head).children(out).into_any_element()
            }
        };
        self.page_frame(&t, header, body, cx)
    }

    /// One model's row: name, architecture, classes, size, scores, source run, age.
    #[allow(clippy::too_many_arguments)]
    fn model_row(
        &self,
        m: &Model,
        i: usize,
        ids: &[String],
        best: Option<&'static str>,
        decimals: usize,
        c: &ModelCols,
        name_w: f32,
        now: i64,
        strong: &dyn Fn(&str) -> bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let metrics = m.metrics;
        let (b, bu) = bytes(m.bytes);
        let published = m.source == Some(crate::api::ModelSource::Published);
        let run = m.run_id.as_ref().and_then(|r| self.store.run(r)).map(|r| r.name.clone());
        let id = m.id.clone();
        let key = format!("model-row-{}", m.id);
        let ids_k = ids.to_vec();
        // Keyboard focus shows what hover does: the full name, the exact time, what the e2e tag
        // means, and for published weights what they are.
        let e2e = self.model_e2e(m);
        let about = m.about.clone().filter(|_| published);
        let full = {
            let mut s = m.name.clone();
            if let Some(a) = &about {
                s += &format!(" · {a}");
            }
            s
        };
        let focus_text = self.ring(&self.fh(key.clone(), cx), window).then(|| {
            let base = time::parse(&m.created)
                .map_or(full.clone(), |c| format!("{full} · created {}", time::local_datetime(c)));
            if e2e { base + " · end-to-end head: predicts without NMS" } else { base }
        });
        let ink = |on: bool| if on { t.text_primary } else { t.text_secondary };
        let score = |v: Option<f32>, d: usize, on: bool| match v {
            Some(v) => text(ty::SM, ink(on))
                .tnum()
                .when(on, |d| d.font_weight(FontWeight::MEDIUM))
                .child(format!("{v:.d$}")),
            None => text(ty::SM, t.text_tertiary).child("–"),
        };
        let row = div()
            .id(SharedString::from(key.clone()))
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .when_some(focus_text, |d, s| d.child(crate::controls::focus_tip(&t, s, 36.)))
            .h(px(36.))
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
                    .min_w(px(MODEL_NAME_MIN))
                    .when(about.is_some() || crate::ui::may_overflow(&m.name, ty::SM.size, name_w), |d| {
                        d.tooltip(crate::controls::tip(full.clone(), None))
                    })
                    .child(
                        text(ty::SM, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .w(px(name_w))
                            .ellipsis()
                            .child(m.name.clone()),
                    ),
            )
            // The architecture is a name (sans), with the e2e tag of an end-to-end model.
            .child(
                div()
                    .w(px(ARCH_W))
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(text(ty::SM, t.text_secondary).whitespace_nowrap().child(m.arch.as_str()))
                    .when(e2e, |d| d.child(e2e_tag(&t, format!("e2e-m-{}", m.id)))),
            )
            .when(c.classes, |d| {
                d.child(num_cell(
                    64.,
                    text(ty::SM, t.text_secondary).tnum().child(m.classes.len().to_string()),
                ))
            })
            // The weights as stored ("f16, fused") on hover: an export reads as one by its record.
            .child(num_cell(
                84.,
                div()
                    .id(SharedString::from(format!("m-size-{}", m.id)))
                    .when_some(model_storage(m), |d, s| {
                        d.tooltip(crate::controls::tip(format!("Stored as {s}"), None))
                    })
                    .child(
                        text(ty::SM, t.text_secondary).tnum().whitespace_nowrap().child(format!("{b} {bu}")),
                    ),
            ))
            .child(num_cell(
                MAP_W,
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .children(best.map(|w| text(ty::SM, t.text_tertiary).whitespace_nowrap().child(w)))
                    .child(score(metrics.map(|x| x.map50_95), decimals, strong("map"))),
            ))
            .when(c.map50, |d| d.child(num_cell(72., score(metrics.map(|x| x.map50), 3, strong("map50")))))
            .when(c.source, |d| {
                d.child(
                    text(ty::SM, if run.is_some() { t.text_secondary } else { t.text_tertiary })
                        .w(px(SOURCE_W))
                        .flex_none()
                        .ellipsis()
                        .child(run.clone().unwrap_or_else(|| "—".to_string())),
                )
            })
            .child(num_cell(84., {
                // Relative in the column, the exact local time on hover.
                let at = time::parse(&m.created);
                text(ty::SM, t.text_tertiary)
                    .id(SharedString::from(format!("mc-{}", m.id)))
                    .when_some(at, |d, c| d.tooltip(crate::controls::tip(time::local_datetime(c), None)))
                    .tnum()
                    .whitespace_nowrap()
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
                text(ty::SM, if on.is_some() { t.text_primary } else { t.text_secondary })
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

/// The narrowest the Model column gets; optional columns drop before it would go under.
const MODEL_NAME_MIN: f32 = 160.;
/// Row padding, the panel's borders and slack for rounding.
const MODEL_SLACK: f32 = 32. + 2. + 8.;
const ARCH_W: f32 = 96.;
/// mAP50-95 with "joint best" before it.
const MAP_W: f32 = 130.;
const SOURCE_W: f32 = 150.;

/// A group's 32 px head: what its scores are on, how many models, and "not ranked" where they
/// can't be (published, imported, unknown split).
fn group_head(t: &crate::theme::Theme, label: &str, n: usize, ranked: bool) -> folo_ui::Div {
    div()
        .h(px(32.))
        .px(px(16.))
        .flex()
        .items_center()
        .gap(px(8.))
        .bg(t.bg_raised)
        .border_b_1()
        .border_color(t.line_subtle)
        .child(
            text(ty::SM, t.text_secondary)
                .font_weight(FontWeight::MEDIUM)
                .ellipsis()
                .child(label.to_string()),
        )
        .child(text(ty::SM, t.text_tertiary).tnum().child(n.to_string()))
        .when(!ranked, |d| d.child(text(ty::SM, t.text_tertiary).whitespace_nowrap().child("not ranked")))
}

/// Which of the Models table's optional columns show at a content width: mAP50, then Classes, then
/// Source run, each while the row still fits with the name at [`MODEL_NAME_MIN`]. Model,
/// Architecture, Size, mAP50-95 and Created always show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ModelCols {
    map50: bool,
    classes: bool,
    source: bool,
}

/// Which of the optional columns a step of the fit turns on.
type Flag = fn(&mut ModelCols) -> &mut bool;
impl ModelCols {
    const BASE: [f32; 4] = [ARCH_W, 84., MAP_W, 84.];

    fn fit(content: f32) -> Self {
        let mut c = ModelCols::default();
        let order: [(Flag, f32); 3] =
            [(|c| &mut c.map50, 72.), (|c| &mut c.classes, 64.), (|c| &mut c.source, SOURCE_W)];
        for (flag, w) in order {
            if c.fixed() + w + 16. + MODEL_NAME_MIN + MODEL_SLACK > content {
                break;
            }
            *flag(&mut c) = true;
        }
        c
    }

    /// The fixed columns' widths and their gaps.
    fn fixed(&self) -> f32 {
        let opt = [(self.map50, 72.), (self.classes, 64.), (self.source, SOURCE_W)];
        Self::BASE.iter().map(|w| w + 16.).sum::<f32>()
            + opt.iter().filter(|(on, _)| *on).map(|(_, w)| w + 16.).sum::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::ModelCols;

    #[test]
    fn every_column_from_1280_and_source_run_drops_first() {
        assert_eq!(ModelCols::fit(1000.), ModelCols { map50: true, classes: true, source: true });
        assert_eq!(ModelCols::fit(872.), ModelCols { map50: true, classes: true, source: false });
    }
}
