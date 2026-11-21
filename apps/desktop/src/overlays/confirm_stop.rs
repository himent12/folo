//! The confirmation before stopping a run.

use folo_ui::{AnyElement, Context, ParentElement, Styled, Window, div, px};

use crate::api::RunExt;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

use super::dialog_footer;

/// The epoch whose checkpoint a stop keeps now: the last validated one, or the epoch whose batches
/// are all done (it is being validated, and the trainer checks for a stop only after its checkpoint).
fn kept_epoch(validated: Option<u32>, run: Option<&crate::api::Run>) -> Option<u32> {
    let done =
        run.filter(|r| r.epoch > 0 && r.steps_per_epoch > 0 && r.step >= r.epoch as u64 * r.steps_per_epoch);
    match done {
        Some(r) if validated.is_none_or(|v| v < r.epoch) => Some(r.epoch),
        _ => validated,
    }
}

/// What the confirmation says a stop does, with `finished` the epoch whose checkpoint it keeps (`None`
/// while the history is loading, `Some(None)` when no epoch has finished).
fn stop_text(finished: Option<Option<u32>>, never_started: bool, waiting: bool) -> String {
    // The epoch in progress is discarded; last.folo is the last finished epoch's.
    match finished {
        _ if never_started => "It hasn't started, so nothing is lost: it leaves the queue and the runs list, and the runs behind it move up.".to_string(),
        _ if waiting => "It stops waiting and keeps what it trained so far. You can resume it later.".to_string(),
        Some(Some(e)) => format!("Stops after the current batch and keeps the epoch {e} checkpoint for export or resume. Progress after epoch {e} is lost."),
        // An epoch whose last batch is done when the stop comes still validates and saves.
        Some(None) => "Stops after the current batch. No epoch has finished yet: unless one finishes first, nothing is saved.".to_string(),
        None => "Stops after the current batch and keeps the last finished epoch's checkpoint for export or resume. Progress after it is lost.".to_string(),
    }
}

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
        // loaded (Run.epoch alone can't tell "epoch 12 finished" from "epoch 12 under way"), and
        // follow the run while the dialog is open (an epoch being validated is kept too).
        let finished = self
            .store
            .run_data
            .get(id)
            .and_then(|d| d.ready())
            .map(|d| kept_epoch(d.metrics.epochs.last().map(|e| e.epoch), run.as_ref()));
        let rid = id.to_string();
        // A run still waiting in the queue leaves it: one that never started is deleted (nothing is
        // lost); a resumed one waiting to continue is stopped and keeps what it trained.
        let waiting =
            run.as_ref().is_some_and(|r| r.status == crate::api::RunStatus::Queued && r.cloud.is_none());
        let never_started = run.as_ref().is_some_and(|r| r.never_started());
        let cancel = self.button(
            "stop-cancel",
            if waiting { "Keep in queue" } else { "Keep training" },
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
            if waiting { "Remove from queue" } else { "Stop run" },
            Btn::DangerCommit,
            Some("icons/stop.svg"),
            window,
            cx,
            act(move |this, w, cx| {
                this.restore_focus(w);
                if never_started {
                    this.remove_queued(rid.clone(), cx)
                } else {
                    this.stop_run(rid.clone(), cx)
                }
            }),
        );
        let body = div().px(px(20.)).pb(px(20.)).child(
            text(ty::BASE, t.text_secondary).w_full().min_w_0().whitespace_normal().child(stop_text(
                finished,
                never_started,
                waiting,
            )),
        );
        let footer = dialog_footer(&t).justify_end().child(cancel).child(stop);
        self.dialog_frame(
            &t,
            "stop-dialog",
            440.,
            &if waiting { format!("Remove {name} from the queue?") } else { format!("Stop {name}?") },
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

#[cfg(test)]
mod tests {
    use crate::api::Backend;

    #[test]
    fn a_stop_with_no_finished_epoch_says_nothing_is_saved_only_unless_one_finishes() {
        assert_eq!(
            super::stop_text(Some(None), false, false),
            "Stops after the current batch. No epoch has finished yet: unless one finishes first, nothing is saved."
        );
        assert!(super::stop_text(Some(Some(3)), false, false).contains("keeps the epoch 3 checkpoint"));
    }

    #[test]
    fn a_stop_keeps_the_epoch_being_validated() {
        let api = crate::api::mock::MockBackend::new(4200, true);
        let run = api.run("run_a19f").unwrap();
        let spe = run.steps_per_epoch;
        let at = |epoch: u32, step: u64| crate::api::Run { epoch, step, ..run.clone() };
        // Epoch 17 under way: epoch 16's checkpoint is kept.
        assert_eq!(super::kept_epoch(Some(16), Some(&at(17, 16 * spe + 3))), Some(16));
        // Its batches done, it is being validated: a stop now keeps epoch 17.
        assert_eq!(super::kept_epoch(Some(16), Some(&at(17, 17 * spe))), Some(17));
        // Validated already: nothing changes.
        assert_eq!(super::kept_epoch(Some(17), Some(&at(17, 17 * spe))), Some(17));
        assert_eq!(super::kept_epoch(None, Some(&at(1, 2))), None);
    }
}
