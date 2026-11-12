//! The confirmation before deleting a run, or removing a dataset from FOLO.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

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
        let (title, confirm_label, body) = match target {
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
                let body =
                    text(ty::BASE, t.text_secondary).w_full().min_w_0().whitespace_normal().child(body);
                (format!("Delete {name}?"), "Delete run", body.into_any_element())
            }
            DeleteTarget::Dataset(id) => {
                let body = self.remove_dataset_body(id, window, cx);
                let files = self.delete_files && self.dataset_files_deletable(id);
                let confirm = if files { "Remove and delete files" } else { "Remove from FOLO" };
                (format!("Remove {} from FOLO?", self.store.dataset_name(id)), confirm, body)
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
            confirm_label,
            Btn::DangerCommit,
            Some("icons/trash-2.svg"),
            window,
            cx,
            act(|this, w, cx| {
                this.restore_focus(w);
                this.delete_confirmed(cx)
            }),
        );
        let body = div().px(px(20.)).pb(px(20.)).child(body);
        let footer = dialog_footer(&t).justify_end().child(cancel).child(confirm);
        // X, scrim and Esc dismiss like Cancel: focus goes back to what opened it.
        self.dialog_frame(&t, "delete-dialog", 440., &title, None, body, footer, window, cx, close)
    }

    /// Whether a dataset's folder can go with it: the server wrote it into its data dir (a generated
    /// set) and no run on the server, on any dataset, trains or waits now (the server checks both
    /// too: it deletes files only while no run is active, docs/API.md), as in Studio.
    pub(crate) fn dataset_files_deletable(&self, id: &str) -> bool {
        let Some(d) = self.store.dataset_summary(id) else { return false };
        let Some(runs) = self.store.runs.ready() else { return false };
        self.store.written_by_server(d) && !runs.iter().any(|r| r.status.is_active())
    }

    /// What removing a dataset means, in Studio's words: it is unregistered and its files stay (or,
    /// asked for, its folder goes), what that does to the runs that trained on it, and the files
    /// switch for a dataset FOLO generated.
    fn remove_dataset_body(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let ds = self.store.dataset_summary(id).cloned();
        let name = self.store.dataset_name(id);
        let deletable = self.dataset_files_deletable(id);
        let files = self.delete_files && deletable;
        let used: Vec<(String, bool)> = self
            .store
            .runs
            .ready()
            .into_iter()
            .flatten()
            .filter(|r| r.config.dataset_id == id)
            .map(|r| (r.name.clone(), r.status.is_active()))
            .collect();
        let para =
            |s: String| text(ty::BASE, t.text_secondary).w_full().min_w_0().whitespace_normal().child(s);
        let lead = match (&ds, files) {
            (Some(_), true) => format!("{name} will be unregistered, and its folder deleted:"),
            (Some(_), false) => {
                format!("{name} will be unregistered. The files in this folder stay on disk:")
            }
            (None, _) => format!("{name} will be unregistered. Its files stay on disk."),
        };
        let impact = match used.len() {
            0 => None,
            1 => Some(
                "1 run trained on it. It keeps its metrics and models, but its sample images will not show \
                 until the folder is registered again."
                    .to_string(),
            ),
            n => Some(format!(
                "{n} runs trained on it. They keep their metrics and models, but their sample images will not \
                 show until the folder is registered again."
            )),
        };
        let active: Vec<&str> = used.iter().filter(|(_, a)| *a).map(|(n, _)| n.as_str()).collect();
        let active = match active.as_slice() {
            [] => None,
            [one] => Some(format!("{one} is training or queued on it now: it carries on.")),
            many => Some(format!("{} runs are training or queued on it now: they carry on.", many.len())),
        };
        // The body's width: the dialog's 440 px less its 20 px sides.
        let path =
            ds.as_ref().map(|ds| path_text(window, ty::SM, t.text_primary, &display_path(&ds.path), 400.));
        let images = ds.as_ref().map_or_else(
            || "generated images".to_string(),
            |d| counted(d.splits.train + d.splits.val + d.splits.test, "generated image", "generated images"),
        );
        let switch_row = deletable.then(|| {
            let toggle = act(|this, _, cx| {
                this.delete_files = !this.delete_files;
                cx.notify();
            });
            let switch = self.switch("del-files", files, window, cx, toggle);
            div()
                .mt(px(4.))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(
                    // The label toggles too (the switch keeps the one Tab stop).
                    div()
                        .id("del-files-label")
                        .cursor_pointer()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.delete_files = !this.delete_files;
                            cx.notify();
                        }))
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(
                            text(ty::BASE, t.text_primary)
                                .font_weight(FontWeight::MEDIUM)
                                .child(format!("Also delete the {images}")),
                        )
                        .child(
                            text(ty::SM, t.text_tertiary)
                                .whitespace_normal()
                                .child("FOLO made them in its own data folder. This can't be undone."),
                        ),
                )
                .child(switch)
        });
        div()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(para(lead))
            .children(path)
            .when_some(impact, |d, s| d.child(para(s)))
            .children(switch_row)
            .when_some(active, |d, s| d.child(para(s)))
            .into_any_element()
    }
}
