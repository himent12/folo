//! A run's Per class panel: each class's AP50-95 at the latest validated epoch and its change since
//! the epoch before, from the run's `epoch` events (`ValMetrics.per_class`).

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{ClassMetrics, Epoch, Run};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::{loading_rows, th};
use crate::store::{Load, RunData};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// One row of the panel: a class, its AP50-95 and its change since the previous epoch.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ClassRow {
    pub class: u32,
    pub ap: f32,
    /// None at the first epoch, or when the previous epoch didn't report the class.
    pub delta: Option<f32>,
}

/// The latest epoch's classes, best first (ties in class order), each with its change since the
/// epoch before it. Empty when the server sent no per-class metrics.
pub(super) fn class_rows(epochs: &[Epoch]) -> Vec<ClassRow> {
    let Some(last) = epochs.last() else { return Vec::new() };
    let prev = epochs.len().checked_sub(2).map(|i| &epochs[i].val.per_class);
    let before =
        |c: &ClassMetrics| prev?.iter().find(|p| p.class == c.class).map(|p| c.map50_95 - p.map50_95);
    let mut rows: Vec<ClassRow> = last
        .val
        .per_class
        .iter()
        .map(|c| ClassRow { class: c.class, ap: c.map50_95, delta: before(c) })
        .collect();
    rows.sort_by(|a, b| b.ap.total_cmp(&a.ap).then(a.class.cmp(&b.class)));
    rows
}

/// What the panel's body shows, from the run's history as loaded.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum PerClass {
    Loading,
    /// The history couldn't be read: the server's words (the charts above say the same, with
    /// the page's retry).
    Failed(String),
    /// No epoch validated yet.
    NoEval,
    /// The latest epoch, which reported no classes.
    NoMetrics(u32),
    /// The latest epoch and its classes.
    Rows(u32, Vec<ClassRow>),
}

pub(super) fn per_class_state(load: Option<&Load<RunData>>) -> PerClass {
    match load {
        Some(Load::Ready(d)) => match d.metrics.epochs.last() {
            None => PerClass::NoEval,
            Some(last) => match class_rows(&d.metrics.epochs) {
                rows if rows.is_empty() => PerClass::NoMetrics(last.epoch),
                rows => PerClass::Rows(last.epoch, rows),
            },
        },
        Some(Load::Failed(e)) => PerClass::Failed(e.message.clone()),
        _ => PerClass::Loading,
    }
}

/// A change as the panel writes it: "▲0.003", "▼0.013", "0.000" when it rounds to none.
pub(super) fn delta_text(d: f32) -> String {
    let shown = (d * 1000.).round() / 1000.;
    if shown > 0. {
        format!("▲{shown:.3}")
    } else if shown < 0. {
        format!("▼{:.3}", -shown)
    } else {
        "0.000".to_string()
    }
}

/// The AP50-95 column's width, and the change column's.
const AP_W: f32 = 64.;
const DELTA_W: f32 = 64.;

impl FoloApp {
    /// The Per class panel. `beside`: it sits beside the samples, so its list takes their height and
    /// scrolls inside it; stacked, it is at most 360 px tall.
    pub(super) fn per_class_panel(
        &mut self,
        r: &Run,
        beside: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let names: Vec<String> = r
            .dataset
            .as_ref()
            .map(|d| d.classes.clone())
            .or_else(|| self.store.dataset_summary(&r.config.dataset_id).map(|d| d.classes.clone()))
            .unwrap_or_default();
        let state = per_class_state(self.store.run_data.get(&r.id));
        let epoch = match &state {
            PerClass::NoMetrics(e) | PerClass::Rows(e, _) => Some(*e),
            _ => None,
        };
        let body = match state {
            // The same retry as the charts' error panel above (it reloads the history).
            PerClass::Failed(message) => {
                let retry = self.button(
                    "per-class-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_all(cx)),
                );
                empty_state(
                    &t,
                    "icons/chart-line.svg",
                    "Per-class metrics unavailable",
                    &message,
                    Tone::Danger,
                )
                .child(retry)
                .into_any_element()
            }
            PerClass::NoEval => empty_state(
                &t,
                "icons/chart-line.svg",
                "No evaluation yet",
                "Each class's AP appears after the first epoch.",
                Tone::Neutral,
            )
            .into_any_element(),
            PerClass::NoMetrics(_) => empty_state(
                &t,
                "icons/chart-line.svg",
                "No per-class metrics",
                "The server sent none with this run's epochs.",
                Tone::Neutral,
            )
            .into_any_element(),
            PerClass::Rows(_, rows) => self.per_class_list(&rows, &names, beside, window, cx),
            PerClass::Loading => loading_rows(&t, "per-class", 6).into_any_element(),
        };
        let qualifier = epoch.map(|e| format!("epoch {e}"));
        let export_counts = self
            .store
            .run_data
            .get(&r.id)
            .and_then(|l| l.ready())
            .filter(|d| !d.metrics.epochs.is_empty())
            .map(|d| {
                (d.metrics.epochs.len(), d.metrics.epochs.last().is_some_and(|e| !e.val.per_class.is_empty()))
            });
        let export_footer =
            export_counts.map(|(n, has_pc)| self.run_metrics_export_footer(r, n, has_pc, window, cx));
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                panel_header(&t, "Per class")
                    .when_some(qualifier, |d, q| d.child(text(ty::SM, t.text_tertiary).tnum().child(q))),
            )
            .child(body)
            .children(export_footer)
            .into_any_element()
    }

    fn per_class_list(
        &mut self,
        rows: &[ClassRow],
        names: &[String],
        beside: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // The list is one Tab stop when it scrolls: arrows step a row, Page Up/Down a screen.
        let fh = self.fh("per-class-scroll", cx);
        let ring = self.ring(&fh, window);
        if ring {
            self.hint_keys("↑ ↓ a row · PgUp / PgDn a screen · Home / End the first and last classes");
        }
        let head = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .h(px(HEAD))
            .flex_none()
            .px(px(16.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().flex_1().min_w_0().child(th(&t, "Class")))
            .child(div().w(px(AP_W)).flex_none().flex().justify_end().child(th(&t, "AP50-95")))
            .child(div().w(px(DELTA_W)).flex_none().flex().justify_end().child(th(&t, "Δ prev")));
        let row = |c: &ClassRow| {
            let name = names.get(c.class as usize).cloned().unwrap_or_else(|| format!("class {}", c.class));
            div()
                .flex()
                .items_center()
                .gap(px(12.))
                .h(px(ROW))
                .px(px(16.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(class_swatch(&t, c.class, 8.))
                        .child(
                            text(ty::BASE, t.text_primary)
                                .font_weight(FontWeight::MEDIUM)
                                .min_w_0()
                                .ellipsis()
                                .child(name),
                        ),
                )
                // Ranked by this: primary ink and weight; the change beside it is secondary.
                .child(
                    text(ty::BASE, t.text_primary)
                        .font_weight(FontWeight::MEDIUM)
                        .tnum()
                        .w(px(AP_W))
                        .flex_none()
                        .text_right()
                        .child(f3(c.ap)),
                )
                .child(
                    text(ty::SM, t.text_secondary)
                        .tnum()
                        .w(px(DELTA_W))
                        .flex_none()
                        .text_right()
                        .child(c.delta.map_or("–".to_string(), delta_text)),
                )
        };
        let scroll = self.per_class_scroll.clone();
        let list = div()
            .id("per-class-scroll")
            .track_focus(&fh)
            .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                let s = &this.per_class_scroll;
                let (y, max) = (-s.offset().y, s.max_offset().height);
                let page = (s.bounds().size.height - px(ROW)).max(px(ROW));
                let to = match ev.keystroke.key.as_str() {
                    "up" => y - px(ROW),
                    "down" => y + px(ROW),
                    "pageup" => y - page,
                    "pagedown" => y + page,
                    "home" => px(0.),
                    "end" => max,
                    _ => return,
                };
                s.set_offset(folo_ui::point(s.offset().x, -to.clamp(px(0.), max)));
                cx.stop_propagation();
                cx.notify();
            }))
            .overflow_y_scroll()
            .track_scroll(&scroll)
            // Beside the samples the list takes the rest of the body's height (below). Stacked it
            // stops at 360 px.
            .when(beside, |d| d.flex_1().flex_basis(px(0.)).min_h(px(0.)))
            .when(!beside, |d| d.max_h(px(360.)))
            .children(rows.iter().map(row));
        // The body (header and list) beside the samples takes their height, but never less than
        // the header and the list's first `BESIDE_ROWS` rows: beside a short samples panel (no
        // images yet) the row grows so every class of a small dataset shows whole, not one row and
        // a sliver. On the body, not the list: the body's zero-basis flex would otherwise let it
        // shrink below the list, and the panel clips what overflows it.
        div()
            .relative()
            .flex_1()
            .min_h(px(if beside { beside_min_h(rows.len()) } else { 0. }))
            .flex()
            .flex_col()
            .child(head)
            .child(list)
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::SM)))
            .into_any_element()
    }
}

/// A class row's height.
const ROW: f32 = 32.;

/// The column header's height.
const HEAD: f32 = 32.;

/// The rows the list shows at least beside the samples; more scroll inside it.
const BESIDE_ROWS: usize = 8;

/// The least height of the list's body (column header and rows) beside the samples, px: the header
/// and the rows up to `BESIDE_ROWS`, whole.
fn beside_min_h(rows: usize) -> f32 {
    HEAD + rows.min(BESIDE_ROWS) as f32 * ROW
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, mock::MockBackend};

    fn epoch(n: u32, aps: &[(u32, f32)]) -> Epoch {
        let mut e = MockBackend::new(4200, true).metrics("run_52e8").unwrap().epochs.remove(0);
        e.epoch = n;
        e.val.per_class = aps
            .iter()
            .map(|&(class, ap)| ClassMetrics { class, map50: ap, map50_95: ap, precision: 0.5, recall: 0.5 })
            .collect();
        e
    }

    #[test]
    fn classes_rank_best_first_with_their_change_since_the_epoch_before() {
        let rows = class_rows(&[epoch(1, &[(0, 0.5), (1, 0.4), (2, 0.3)]), epoch(2, &[(0, 0.49), (1, 0.6)])]);
        assert_eq!(rows.iter().map(|r| r.class).collect::<Vec<_>>(), vec![1, 0]);
        assert!((rows[0].delta.unwrap() - 0.2).abs() < 1e-6);
        assert!((rows[1].delta.unwrap() + 0.01).abs() < 1e-6);
    }

    #[test]
    fn the_first_epoch_and_a_class_new_to_the_epoch_have_no_change() {
        let first = class_rows(&[epoch(1, &[(0, 0.5)])]);
        assert_eq!(first, vec![ClassRow { class: 0, ap: 0.5, delta: None }]);
        let rows = class_rows(&[epoch(1, &[(0, 0.5)]), epoch(2, &[(0, 0.5), (3, 0.2)])]);
        assert_eq!(rows[1], ClassRow { class: 3, ap: 0.2, delta: None });
        assert_eq!(rows[0].delta, Some(0.));
    }

    #[test]
    fn no_per_class_metrics_is_an_empty_list() {
        assert!(class_rows(&[]).is_empty());
        assert!(class_rows(&[epoch(4, &[])]).is_empty());
    }

    #[test]
    fn a_history_that_failed_to_load_says_so_instead_of_loading() {
        let down = crate::api::ApiError::new("server_unreachable", "Could not reach folo serve");
        assert_eq!(
            per_class_state(Some(&Load::Failed(down))),
            PerClass::Failed("Could not reach folo serve".into())
        );
        assert_eq!(per_class_state(Some(&Load::Loading)), PerClass::Loading);
        assert_eq!(per_class_state(None), PerClass::Loading);
        let m = MockBackend::new(4200, true).metrics("run_52e8").unwrap();
        let last = m.epochs.last().unwrap().epoch;
        let PerClass::Rows(e, rows) = per_class_state(Some(&Load::Ready(RunData::new(m, 1)))) else {
            panic!("rows")
        };
        assert_eq!(e, last);
        assert!(!rows.is_empty());
    }

    #[test]
    fn beside_the_samples_a_small_datasets_classes_all_show_whole() {
        // Three classes: three whole rows, however short the samples panel beside them.
        assert_eq!(beside_min_h(3), HEAD + 3. * ROW);
        assert_eq!(beside_min_h(0), HEAD);
        // Eighty: the first eight, the rest scroll.
        assert_eq!(beside_min_h(80), HEAD + BESIDE_ROWS as f32 * ROW);
    }

    #[test]
    fn changes_read_with_their_direction() {
        assert_eq!(delta_text(0.0031), "▲0.003");
        assert_eq!(delta_text(-0.013), "▼0.013");
        assert_eq!(delta_text(0.0002), "0.000");
        assert_eq!(delta_text(-0.0004), "0.000");
    }
}
