//! Chart groups: one Tab stop that steps a shared chart cursor.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, Div, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    Styled, Window,
};

use crate::app::{FoloApp, focus_ring};
use crate::theme::{radius, theme};

/// Which shared chart cursor a chart group drives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cursor {
    /// The linked cursor of the run dashboard and the Overview hero.
    Run,
    /// The PR curve on a model page.
    Pr,
}

impl FoloApp {
    fn cursor_mut(&mut self, c: Cursor) -> &mut Option<f32> {
        match c {
            Cursor::Run => &mut self.chart_hover,
            Cursor::Pr => &mut self.pr_hover,
        }
    }

    /// Make a group of charts one Tab stop: ←/→ step the shared cursor by `step` (a fraction of the
    /// x range), Shift steps ×10, Home/End jump, Esc clears; Page Up/Down pass on and page the page. The readouts follow the cursor. `end` is
    /// where recorded data stops (a live run's latest epoch): the cursor starts there and never
    /// steps past it into the epochs still to come.
    #[allow(clippy::too_many_arguments)]
    pub fn chart_group(
        &self,
        key: &str,
        step: f32,
        end: f32,
        cursor: Cursor,
        content: Div,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        content
            .id(SharedString::from(key.to_string()))
            .track_focus(&fh)
            .relative()
            // Esc is bound globally to Dismiss (go back), and actions run before key handlers: with
            // a cursor showing, Esc clears it here instead; otherwise it passes on.
            .on_action(cx.listener(move |this, _: &crate::app::Dismiss, _, cx| {
                let cur = this.cursor_mut(cursor);
                if cur.is_some() {
                    *cur = None;
                    cx.notify();
                } else {
                    cx.propagate();
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let big = if ev.keystroke.modifiers.shift { 10. } else { 1. };
                let cur = this.cursor_mut(cursor);
                let end = end.clamp(0.0, 1.0);
                let at = cur.map_or(end, |c| c.min(end));
                let next = match ev.keystroke.key.as_str() {
                    // The first press inspects the latest point itself, then steps from there.
                    "left" if cur.is_none() => Some(end),
                    "left" => Some((at - step * big).max(0.0)),
                    "right" => Some((at + step * big).min(end)),
                    "home" => Some(0.0),
                    "end" => Some(end),
                    "escape" if cur.is_some() => None,
                    _ => return,
                };
                *cur = next;
                cx.stop_propagation();
                cx.notify();
            }))
            .when(ring, |d| {
                // The keys go in the strip under the page: in view at any size, over no chart.
                self.hint_keys(
                    "← → inspect · Shift ×10 · Home / End · PgUp / PgDn scroll the page · Esc clears",
                );
                d.child(focus_ring(&t, radius::LG))
            })
            .into_any_element()
    }
}
