//! The confirmation before closing the window would stop training: the `folo serve` this app started
//! runs only as long as the app.

use folo_ui::{AnyElement, Context, ParentElement, Styled, Window, div, px};

use crate::api::Run;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

use super::dialog_footer;

/// The runs closing would stop, of the server this app started: those of this machine (not a cloud
/// run, which trains elsewhere) that are training, stopping or waiting.
pub fn closing_stops(runs: &[Run]) -> Vec<String> {
    runs.iter().filter(|r| r.cloud.is_none() && r.status.is_active()).map(|r| r.name.clone()).collect()
}

/// What the dialog says: the runs that would stop (`names`, as the server said when asked: a run that
/// ends meanwhile stays named, a warning, never a false "nothing"), or, with none, that the server
/// couldn't say.
fn what_stops(names: &[String]) -> String {
    const SERVER: &str = "the folo serve FOLO Desktop started, which stops when it closes";
    const KEPT: &str = "A stopped run keeps its last finished epoch, and you can resume it.";
    match names {
        [] => format!("FOLO Desktop couldn't check whether a run is training on {SERVER}. {KEPT}"),
        [one] => format!("{one} is training or waiting to train on {SERVER}. {KEPT}"),
        many => format!("{} runs are training or waiting to train on {SERVER}. {KEPT}", many.len()),
    }
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
            act(|this, w, cx| this.keep_open(w, cx)),
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
            act(|this, w, cx| this.keep_open(w, cx)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, RunStatus};

    #[test]
    fn only_this_machines_active_runs_stop_with_the_apps_server() {
        let api = crate::api::mock::MockBackend::with_cloud_runs(4200, true);
        let runs = api.runs().unwrap();
        let stops = closing_stops(&runs);
        let active: Vec<&Run> = runs.iter().filter(|r| r.cloud.is_none() && r.status.is_active()).collect();
        assert!(!active.is_empty() && stops.len() == active.len(), "{stops:?}");
        assert!(
            runs.iter().any(|r| r.cloud.is_some() && r.status.is_active()),
            "the demo has a cloud run training"
        );
        assert!(stops.iter().all(|n| runs.iter().any(|r| &r.name == n && r.cloud.is_none())));
        assert!(!stops.iter().any(|n| runs.iter().any(|r| &r.name == n && r.status == RunStatus::Completed)));
        assert!(what_stops(&stops[..1]).starts_with(&format!("{} is training", stops[0])));
        assert!(what_stops(&["a".into(), "b".into()]).starts_with("2 runs are training"));
        assert!(what_stops(&[]).starts_with("FOLO Desktop couldn't check"));
    }
}
