//! The confirmation before stopping a run.

use gpui::{AnyElement, Context, ParentElement, Styled, Window, div, px};

use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

use super::dialog_footer;

impl FoloApp {
    pub(super) fn stop_dialog(
        &mut self,
        id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let run = self.store.run(id).cloned();
        let name = run.as_ref().map_or(id.to_string(), |r| r.name.clone());
        // The checkpoint kept is the last validated epoch's: name it from the history when it's
        // loaded (Run.epoch alone can't tell "epoch 12 finished" from "epoch 12 under way").
        let finished = self
            .store
            .run_data
            .get(id)
            .and_then(|d| d.ready())
            .map(|d| d.metrics.epochs.last().map(|e| e.epoch));
        let rid = id.to_string();
        let cancel = self.button(
            "stop-cancel",
            "Keep training",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| {
                this.confirm_stop = None;
                this.restore_focus(w);
                cx.notify();
            }),
        );
        let stop = self.button(
            "stop-confirm",
            "Stop run",
            Btn::DangerCommit,
            Some("icons/stop.svg"),
            window,
            cx,
            act(move |this, w, cx| {
                this.restore_focus(w);
                this.stop_run(rid.clone(), cx)
            }),
        );
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .child(
                text(ty::BASE, t.text_secondary)
                    .w_full()
                    .min_w_0()
                    .whitespace_normal()
                    // The epoch in progress is discarded; last.folo is the last finished epoch's.
                    .child(match finished {
                        Some(Some(e)) => format!("Stops after the current batch and keeps the epoch {e} checkpoint for export or resume. Progress after epoch {e} is lost."),
                        Some(None) => "Stops after the current batch. No epoch has finished yet, so there is no checkpoint to keep.".to_string(),
                        None => "Stops after the current batch and keeps the last finished epoch's checkpoint for export or resume. Progress after it is lost.".to_string(),
                    }),
            );
        let footer = dialog_footer(&t).justify_end().child(cancel).child(stop);
        self.dialog_frame(
            &t,
            "stop-dialog",
            440.,
            &format!("Stop {name}?"),
            None,
            body,
            footer,
            window,
            cx,
            // X, scrim and Esc dismiss like Keep training: focus goes back to what opened it.
            act(|this, w, cx| {
                this.confirm_stop = None;
                this.restore_focus(w);
                cx.notify();
            }),
        )
    }
}
