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

/// What the dialog says: the runs that would stop (`names`), none any more, or that the server
/// couldn't say (`unchecked`).
fn what_stops(names: &[String], unchecked: bool) -> String {
    const SERVER: &str = "the folo serve FOLO Desktop started, which stops when it closes";
    const KEPT: &str = "A stopped run keeps its last finished epoch, and you can resume it.";
    match names {
        [] if unchecked => {
            format!("FOLO Desktop couldn't check whether a run is training on {SERVER}. {KEPT}")
        }
        [] => "Nothing is training on the folo serve FOLO Desktop started any more.".to_string(),
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
        // The runs as the window now sees them, when it shows that server: one that ended while asking
        // no longer stops.
        let own = self.shows_own_server();
        let names: Vec<String> = match self.store.runs.ready().filter(|_| own) {
            Some(runs) => names
                .iter()
                .filter(|n| runs.iter().find(|r| &r.name == *n).is_none_or(|r| r.status.is_active()))
                .cloned()
                .collect(),
            None => names.to_vec(),
        };
        let unchecked = self.close_unchecked;
        let keep = self.button(
            "close-cancel",
            "Keep open",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.keep_open(w, cx)),
        );
        let nothing_left = names.is_empty() && !unchecked;
        let close = self.button(
            "close-confirm",
            if nothing_left { "Close" } else { "Stop and close" },
            if nothing_left { Btn::Primary } else { Btn::DangerCommit },
            (!nothing_left).then_some("icons/stop.svg"),
            window,
            cx,
            act(|this, w, cx| {
                this.confirm_close = None;
                this.close_now(w, cx);
            }),
        );
        let body = div().px(px(20.)).pb(px(20.)).child(
            text(ty::BASE, t.text_secondary)
                .w_full()
                .min_w_0()
                .whitespace_normal()
                .child(what_stops(&names, unchecked)),
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
        assert!(what_stops(&stops[..1], false).starts_with(&format!("{} is training", stops[0])));
        assert!(what_stops(&["a".into(), "b".into()], false).starts_with("2 runs are training"));
        assert!(what_stops(&[], true).starts_with("FOLO Desktop couldn't check"));
        assert!(what_stops(&[], false).starts_with("Nothing is training"));
    }
}
