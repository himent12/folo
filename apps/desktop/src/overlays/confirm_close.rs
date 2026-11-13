//! The confirmation before closing the window would stop training: the `folo serve` this app started
//! runs only as long as the app.

use folo_ui::{AnyElement, Context, ParentElement, Styled, Window, div, px};

use crate::api::Run;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

use super::dialog_footer;

/// The runs closing would stop: those of this machine (not a cloud run, which trains elsewhere) that
/// are training, stopping or waiting, when the server they are on is the one this app started (`owned`).
pub fn closing_stops(runs: &[Run], owned: bool) -> Vec<String> {
    if !owned {
        return Vec::new();
    }
    runs.iter().filter(|r| r.cloud.is_none() && r.status.is_active()).map(|r| r.name.clone()).collect()
}

/// What the dialog says about `names` (one run, or how many).
fn what_stops(names: &[String]) -> String {
    let training = |n: &[String]| match n {
        [one] => format!("{one} is training or waiting to train"),
        many => format!("{} runs are training or waiting to train", many.len()),
    };
    format!(
        "{} on the folo serve FOLO Desktop started, which stops when it closes. A stopped run keeps its last finished epoch, and you can resume it.",
        training(names)
    )
}

impl FoloApp {
    pub(super) fn close_dialog(
        &mut self,
        names: &[String],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let keep = self.button(
            "close-cancel",
            "Keep open",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| {
                this.confirm_close = None;
                this.restore_focus(w);
                cx.notify();
            }),
        );
        let close = self.button(
            "close-confirm",
            "Stop and close",
            Btn::DangerCommit,
            Some("icons/stop.svg"),
            window,
            cx,
            act(|this, w, cx| {
                this.confirm_close = None;
                this.close_now(w, cx);
            }),
        );
        let body = div().px(px(20.)).pb(px(20.)).child(
            text(ty::BASE, t.text_secondary).w_full().min_w_0().whitespace_normal().child(what_stops(names)),
        );
        let footer = dialog_footer(&t).justify_end().child(keep).child(close);
        self.dialog_frame(
            &t,
            "close-dialog",
            440.,
            "Close FOLO Desktop?",
            None,
            body,
            footer,
            window,
            cx,
            // X, scrim and Esc keep the window open, like Keep open.
            act(|this, w, cx| {
                this.confirm_close = None;
                this.restore_focus(w);
                cx.notify();
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, RunStatus};

    #[test]
    fn only_this_machines_active_runs_on_the_apps_own_server_stop_with_it() {
        let api = crate::api::mock::MockBackend::with_cloud_runs(4200, true);
        let runs = api.runs().unwrap();
        let stops = closing_stops(&runs, true);
        let active: Vec<&Run> = runs.iter().filter(|r| r.cloud.is_none() && r.status.is_active()).collect();
        assert!(!active.is_empty() && stops.len() == active.len(), "{stops:?}");
        assert!(
            runs.iter().any(|r| r.cloud.is_some() && r.status.is_active()),
            "the demo has a cloud run training"
        );
        assert!(stops.iter().all(|n| runs.iter().any(|r| &r.name == n && r.cloud.is_none())));
        // Finished runs never count, and a server the app didn't start keeps training.
        assert!(!stops.iter().any(|n| runs.iter().any(|r| &r.name == n && r.status == RunStatus::Completed)));
        assert!(closing_stops(&runs, false).is_empty());
        assert!(what_stops(&stops[..1]).starts_with(&format!("{} is training", stops[0])));
        assert!(what_stops(&["a".into(), "b".into()]).starts_with("2 runs are training"));
    }
}
