//! The Overview page's layout, and its hero chart's height.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::RunStatus;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::page_header;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn overview(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let n_ds = self.store.datasets.ready().map(|d| d.len());
        let n_models = self.store.models.ready().map(|m| m.len());
        let n_live =
            self.store.runs.ready().map(|r| r.iter().filter(|r| r.status == RunStatus::Running).count());
        let subtitle = match (n_ds, n_models, n_live) {
            (Some(d), Some(m), Some(l)) => {
                let live = match l {
                    0 => "nothing training".to_string(),
                    1 => "1 run training".to_string(),
                    n => format!("{n} runs training"),
                };
                format!("{} · {} · {live}", plural(d, "dataset"), plural(m, "model"))
            }
            // The header keeps its subtitle row: where the app tried to connect (the panel below
            // says what went wrong).
            _ if self.store.runs.error().is_some() => {
                format!("Not connected · {}", self.store.api.describe())
            }
            // Runs came back but something else failed: not "Loading…" for ever.
            _ if self.store.models.error().is_some() || self.store.datasets.error().is_some() => {
                "Some data didn't load · retry in its panel".to_string()
            }
            // The sidebar says "Connecting…" until the server answers; the page says the same.
            _ if self.store.health.is_loading() => "Connecting…".to_string(),
            _ => "Loading…".to_string(),
        };
        // Training needs a server and a dataset; until then the action explains itself instead.
        let no_ds = self.store.datasets.ready().is_some_and(|d| d.is_empty());
        let blocked = if self.store.runs.error().is_some() {
            Some("Connect to folo serve first")
        } else if no_ds {
            Some("Add a dataset first")
        } else if self.store.datasets.error().is_some() {
            // Settled, not loading: say so, and where to retry.
            Some("Datasets didn't load · retry on the Datasets page")
        } else if self.store.datasets.ready().is_none() {
            Some("Loading datasets…")
        } else {
            None
        };
        let new_run = match blocked {
            Some(reason) => self
                .button_disabled(
                    "ov-new",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    reason,
                    window,
                    cx,
                )
                .into_any_element(),
            None => self
                .button(
                    "ov-new",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_run(None, w, cx)),
                )
                .tooltip(crate::controls::tip("Start training", Some("Ctrl N")))
                .into_any_element(),
        };
        // One rule for Open image: it needs a model to run (not connected, still loading or none
        // yet, it says why, as New run beside it does).
        let open_blocked = self.open_image_blocked();
        let open = if let Some(why) = open_blocked {
            self.button_disabled(
                "ov-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                why,
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "ov-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, w, cx| this.pick_image(w, cx)),
            )
            .tooltip(crate::controls::tip("Run a model on an image", Some("Ctrl O")))
            .into_any_element()
        };
        let actions = vec![open, new_run];
        let nothing_yet = no_ds
            && self.store.runs.ready().is_some_and(|r| r.is_empty())
            && self.store.models.ready().is_some_and(|m| m.is_empty());
        // One rule for the header: actions show once the page has settled and has a body of its own
        // to act on. First run (the getting-started panel's "Add dataset" is the call to action),
        // not connected (the error panel's Retry is) and still loading show none, so nothing
        // flashes in disabled and then vanishes. A list that failed has settled: each action then
        // says for itself whether it can run (a failed models list doesn't hide New run).
        let pending = [
            self.store.runs.ready().is_none() && self.store.runs.error().is_none(),
            self.store.models.ready().is_none() && self.store.models.error().is_none(),
            self.store.datasets.ready().is_none() && self.store.datasets.error().is_none(),
        ]
        .contains(&true);
        let hide = nothing_yet || pending || self.store.runs.error().is_some();
        let actions = if hide { Vec::new() } else { actions };
        let header = page_header(&t, "Overview", text(ty::BASE, t.text_secondary).child(subtitle), actions);

        if let Some(e) = self.store.runs.error().cloned() {
            let err = self.error_panel(&e, "ov-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }

        if nothing_yet {
            let body = self.getting_started(window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let live = self.store.live_run().cloned();
        let hero = match &live {
            Some(r) => self.live_panel(r, window, cx),
            None => self.idle_panel(window, cx),
        };
        // With a live run the machine sits in the hero's footer; otherwise a slim strip says it.
        // While runs load, the hero skeleton already reserves the machine footer.
        let strip = (live.is_none() && !self.store.runs.is_loading()).then(|| self.system_strip(cx));
        let recent = self.recent_runs(window, cx);
        let models = self.top_models(window, cx);
        let datasets = self.dataset_strip(window, cx);
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(div().flex().child(hero))
            .when_some(strip, |d, s| d.child(s))
            // From 1280 px Recent runs (two thirds) sits beside Best models (one third, at least
            // 380 px); narrower windows stack the two lists full width rather than squeeze the
            // runs table.
            .map(|d| {
                if !self.tables_beside(window) {
                    d.child(div().flex().child(recent)).child(div().flex().child(models))
                } else {
                    d.child(
                        div()
                            .flex()
                            .gap(px(COLUMN_GAP))
                            // Recent runs takes what is left (its columns are budgeted on
                            // `main_w`, with slack): two fixed slots summing to the content's
                            // width overflow it by a rounding error, and the layout then puts the
                            // whole page at its min-content width.
                            .child(div().flex_1().min_w_0().flex().child(recent))
                            .child(div().w(px(self.side_w(window))).flex_none().flex().child(models)),
                    )
                }
            })
            .child(datasets);
        self.page_frame(&t, header, body, cx)
    }

    /// Whether the page has two columns: Recent runs beside Best models, and the live run's KPI
    /// block beside its chart. From 1280 px.
    pub(super) fn tables_beside(&self, window: &Window) -> bool {
        two_columns(window)
    }

    /// The one-third column's width (Best models), in whole pixels.
    pub(super) fn side_w(&self, window: &Window) -> f32 {
        side_width(self.content_width(window))
    }

    /// Recent runs' width: the two thirds beside Best models, or the content when they stack.
    pub(super) fn main_w(&self, window: &Window) -> f32 {
        let content = self.content_width(window);
        if self.tables_beside(window) { content - COLUMN_GAP - self.side_w(window) } else { content }
    }
}

/// Between the Overview's two columns.
const COLUMN_GAP: f32 = 16.;

/// Best models' least width beside Recent runs (UNIFIED.md §4): its three columns need a little
/// more than a third of the row at 1280 px.
const SIDE_MIN: f32 = 380.;

/// Best models' column in a row `content` wide: a third of it, at least `SIDE_MIN`, in whole
/// pixels. Recent runs takes the rest.
pub(super) fn side_width(content: f32) -> f32 {
    ((content - COLUMN_GAP) / 3.).floor().max(SIDE_MIN)
}

/// Whether the Overview has two columns (UNIFIED.md §4: from 1280 px).
pub(super) fn two_columns(window: &Window) -> bool {
    window.viewport_size().width >= px(1280.)
}

/// The Overview hero chart's plot height: lower in short windows (620 px minimum, so two Recent
/// runs rows show under it), taller in tall, wide ones so the curve isn't a flat band across 1,100
/// px. The skeleton and the live card share it.
pub(super) fn hero_height(window: &Window) -> f32 {
    let v = window.viewport_size();
    if v.height < px(700.) {
        108.
    } else if v.width >= px(1600.) && v.height >= px(960.) {
        180.
    } else {
        132.
    }
}
