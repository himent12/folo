//! The confirmation before deleting a run or a dataset.

use folo_ui::{AnyElement, Context, ParentElement, Styled, Window, div, px};

use crate::app::{DeleteTarget, FoloApp, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

use super::dialog_footer;

impl FoloApp {
    pub(super) fn delete_dialog(
        &mut self,
        target: &DeleteTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let (name, what, body) = match target {
            DeleteTarget::Run(id) => {
                let name = self.store.run(id).map_or(id.clone(), |r| r.name.clone());
                let models = self
                    .store
                    .models
                    .ready()
                    .map_or(0, |m| m.iter().filter(|m| m.run_id.as_deref() == Some(id)).count());
                let body = match models {
                    0 => {
                        "Deletes its folder: metrics, log and checkpoints. This can't be undone.".to_string()
                    }
                    n => format!(
                        "Deletes its folder: metrics, log and checkpoints, and its {} with them. This can't be undone.",
                        plural(n, "model")
                    ),
                };
                (name, "run", body)
            }
            DeleteTarget::Dataset(id) => {
                let ds = self.store.dataset_summary(id);
                let name = ds.map_or(id.clone(), |d| d.name.clone());
                let body = match ds {
                    Some(d) => format!(
                        "FOLO forgets it; its files stay where they are ({}). Runs trained on it keep their results.",
                        display_path(&d.path)
                    ),
                    None => "FOLO forgets it; its files stay where they are. Runs trained on it keep their results."
                        .to_string(),
                };
                (name, "dataset", body)
            }
        };
        let close = act(|this, w, cx| {
            this.confirm_delete = None;
            this.restore_focus(w);
            cx.notify();
        });
        let cancel = self.button("del-cancel", "Cancel", Btn::Secondary, None, window, cx, close.clone());
        let confirm = self.button(
            "del-confirm",
            format!("Delete {what}"),
            Btn::DangerCommit,
            Some("icons/trash-2.svg"),
            window,
            cx,
            act(|this, w, cx| {
                this.restore_focus(w);
                this.delete_confirmed(cx)
            }),
        );
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .child(text(ty::BASE, t.text_secondary).w_full().min_w_0().whitespace_normal().child(body));
        let footer = dialog_footer(&t).justify_end().child(cancel).child(confirm);
        // X, scrim and Esc dismiss like Cancel: focus goes back to what opened it.
        self.dialog_frame(
            &t,
            "delete-dialog",
            440.,
            &format!("Delete {name}?"),
            None,
            body,
            footer,
            window,
            cx,
            close,
        )
    }
}
