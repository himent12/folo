//! The Labeling job detail screen (`Route::LabelJob(id)`): header, inline forms, image review
//! workspace, and agent runner / logs rail.

use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use super::{job_status, unlabeled_count};
use crate::api::LabelImageStatus;
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn label_job_view(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_label_job(id, cx);

        if self.label_unsupported() {
            let header = page_header(&t, "Labeling", "Not available on this server", vec![]);
            let body = self.label_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        if let Some(e) = self
            .label
            .jobs
            .error()
            .or_else(|| self.label.jobs.ready().is_none().then(|| self.label_blocked_error()).flatten())
            .cloned()
        {
            let header = page_header(&t, "Labeling job", "Could not load job", vec![]);
            let body = self.error_panel(&e, "label-job-retry", window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let Some(jobs) = self.label.jobs.ready() else {
            let header = page_header(&t, "Labeling job", "Loading…", vec![]);
            let body = panel(&t).child(loading_rows(&t, "label-job", 4)).into_any_element();
            return self.page_frame(&t, header, body, cx);
        };
        let Some(job) = jobs.iter().find(|j| j.id == id).cloned() else {
            let back = self.button(
                "label-not-found-back",
                "All labeling jobs",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::LabelJobs, w, cx)),
            );
            let header = page_header(&t, "Labeling job", "Not found", vec![back.into_any_element()]);
            let body = panel(&t)
                .child(empty_state(
                    &t,
                    "icons/circle-alert.svg",
                    "Labeling job not found",
                    &format!("No labeling job `{id}` exists on this server."),
                    Tone::Neutral,
                ))
                .into_any_element();
            return self.page_frame(&t, header, body, cx);
        };

        let reviewed_n = self.label.images.get(id).and_then(|l| l.ready()).map(|imgs| {
            imgs.iter().filter(|i| i.status == LabelImageStatus::Labeled && i.reviewed).count() as u64
        });
        let (st_label, st_tone, st_kind) = job_status(&job);
        let title_row = div()
            .flex()
            .items_center()
            .gap(px(8.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).ellipsis().child(job.name.clone()))
            .child(status_mark(st_kind, st_kind.hue(&t), format!("label-hd-{}", job.id)))
            .child(badge(&t, st_label, st_tone));

        let mut facts = vec![
            counted(job.counts.images, "image", "images"),
            format!("{} labeled", int(job.counts.labeled)),
        ];
        if job.counts.labeled > 0
            && let Some(rev) = reviewed_n
        {
            facts.push(format!("{} / {} reviewed", int(rev), int(job.counts.labeled)));
        }
        facts.push(counted(job.classes.len() as u64, "class", "classes"));
        let made_link = job.dataset_id.as_ref().and_then(|ds_id| {
            let ds = self.store.dataset_summary(ds_id)?;
            let target = ds.id.clone();
            let label = format!("Dataset: {}", ds.name);
            Some(self.button(
                "label-open-ds",
                &label,
                Btn::Secondary,
                Some("icons/database.svg"),
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Dataset(target.clone()), w, cx)),
            ))
        });
        let subtitle = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(8.))
            .child(text(ty::SM, t.text_secondary).child(facts.join(" · ")))
            .children(made_link);

        let jid_add = job.id.clone();
        let add_btn = self.button(
            "label-add-folder",
            "Add folder",
            Btn::Secondary,
            Some("icons/folder-open.svg"),
            window,
            cx,
            act(move |this, w, cx| this.open_add_label_folder(&jid_add, w, cx)),
        );

        let export_why = if job.status.is_active() {
            Some("Wait for the agent to finish before exporting")
        } else if job.counts.labeled == 0 {
            Some("Label at least one image first")
        } else {
            None
        };
        let export_primary = export_why.is_none() && job.counts.images > 0 && unlabeled_count(&job) == 0;
        let export_variant = if export_primary { Btn::Primary } else { Btn::Secondary };
        let export_btn = match export_why {
            Some(why) => self.button_disabled(
                "label-export-open",
                "Create dataset",
                export_variant,
                Some("icons/database.svg"),
                why,
                window,
                cx,
            ),
            None => {
                let job_for_export = job.clone();
                self.button(
                    "label-export-open",
                    "Create dataset",
                    export_variant,
                    Some("icons/database.svg"),
                    window,
                    cx,
                    act(move |this, w, cx| this.open_export_label_dataset(&job_for_export, w, cx)),
                )
            }
        };

        let confirming_del = self.label.confirm_delete_job.as_deref() == Some(job.id.as_str());
        let busy_del = self.label.busy.contains(&job.id);
        let del_btn = if job.status.is_active() {
            self.button_disabled(
                "label-del-job",
                "Delete",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                "Stop the active agent run before deleting this job",
                window,
                cx,
            )
        } else if busy_del {
            self.button_disabled("label-del-job", "Deleting…", Btn::Danger, None, "Deleting job", window, cx)
        } else if confirming_del {
            let jid = job.id.clone();
            self.button(
                "label-del-job",
                "Confirm delete",
                Btn::Danger,
                Some("icons/trash-2.svg"),
                window,
                cx,
                act(move |this, _, cx| this.delete_label_job(&jid, cx)),
            )
        } else {
            let jid = job.id.clone();
            self.button(
                "label-del-job",
                "Delete",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    this.label.confirm_delete_job = Some(jid.clone());
                    cx.notify();
                }),
            )
        };

        let header = detail_header(
            &t,
            title_row,
            subtitle,
            vec![add_btn.into_any_element(), export_btn.into_any_element(), del_btn.into_any_element()],
        );

        let add_form = self
            .label
            .add_folder
            .as_ref()
            .is_some_and(|f| f.job == job.id)
            .then(|| self.add_label_folder_form(&t, window, cx));
        let export_form = self
            .label
            .export_ds
            .as_ref()
            .is_some_and(|f| f.job == job.id)
            .then(|| self.export_label_dataset_form(&t, &job, window, cx));

        let content_w = self.content_width(window);
        let stacked = content_w < 920.;
        let rail_w = if content_w >= 1120. { 320. } else { 288. };
        let review_w = if stacked { content_w } else { (content_w - rail_w - 16.).max(240.) };
        let main = self.label_review_workspace(&t, &job, review_w, window, cx);
        let rail = self.label_agent_rail(&t, &job, window, cx);

        let workspace = if stacked {
            div().flex().flex_col().gap(px(16.)).child(main).child(rail)
        } else {
            div()
                .flex()
                .items_start()
                .gap(px(16.))
                .child(div().flex_1().min_w_0().child(main))
                .child(div().w(px(rail_w)).flex_none().min_w_0().child(rail))
        };

        let body =
            div().flex().flex_col().gap(px(16.)).children(add_form).children(export_form).child(workspace);
        self.page_frame(&t, header, body, cx)
    }
}
