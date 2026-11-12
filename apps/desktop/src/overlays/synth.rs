//! The Generate synthetic dataset dialog.

use std::rc::Rc;

use folo_ui::Focusable;
use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, FontWeight, IntoElement, ParentElement, Styled, Window, div, px};

use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

use super::dialog_footer;

impl FoloApp {
    /// Generate a synthetic shapes dataset on folo serve: name, size, image size and classes.
    pub(super) fn synth_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let f = self.synth.as_ref().unwrap();
        let (name, images, imgsz, classes, submitting, error) =
            (f.name.clone(), f.images.clone(), f.imgsz, f.classes, f.submitting, f.error.clone());
        let (name_err, images_err) = crate::app::synth_errors(&name.read(cx).text(), &images.read(cx).text());
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        let input_box = |this: &Self,
                         input: folo_ui::Entity<crate::input::TextInput>,
                         bad: bool,
                         w: f32,
                         window: &Window,
                         cx: &mut Context<Self>| {
            let fh = input.read(cx).focus_handle(cx);
            let ring = this.ring(&fh, window);
            let focused = fh.is_focused(window);
            div()
                .relative()
                .w(px(w))
                .h(px(32.))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if bad {
                    t.danger_solid
                } else if focused {
                    t.accent_line
                } else {
                    t.line_control
                })
                .text_size(px(ty::BASE.size))
                .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(input)
                .into_any_element()
        };
        // A field's problem takes its help line's place, in red: shown at once, and nothing moves.
        let row = |name: &str, control: AnyElement, help: &str, bad: Option<&str>| {
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(div().flex().flex_col().gap(px(2.)).child(label(name)).child(match bad {
                    Some(msg) => text(ty::SM, t.danger_text).child(msg.to_string()),
                    None => text(ty::SM, t.text_tertiary).child(help.to_string()),
                }))
                .child(control)
        };
        let name_box = input_box(self, name, name_err.is_some(), 200., window, cx);
        let images_box = input_box(self, images, images_err.is_some(), 120., window, cx);
        let size_labels: Vec<String> = crate::app::SYNTH_IMGSZ.iter().map(|s| format!("{s} px")).collect();
        let size_refs: Vec<&str> = size_labels.iter().map(String::as_str).collect();
        let size_seg = self.segmented(
            "syn-size",
            &size_refs,
            imgsz,
            window,
            cx,
            Rc::new(|this, i, _, _| {
                if let Some(f) = this.synth.as_mut() {
                    f.imgsz = i;
                }
            }),
        );
        let class_seg = self.segmented(
            "syn-classes",
            &["3 shapes", "5 shapes", "8 shapes"],
            classes,
            window,
            cx,
            Rc::new(|this, i, _, _| {
                if let Some(f) = this.synth.as_mut() {
                    f.classes = i;
                }
            }),
        );
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(row("Name", name_box, "Folder under the server's data dir", name_err))
            .child(row("Images", images_box, "Split 80 / 20 into train and val", images_err))
            .child(row("Image size", size_seg.into_any_element(), "Square JPEGs", None))
            .child(row(
                "Classes",
                class_seg.into_any_element(),
                crate::app::SYNTH_CLASSES[classes].join(", ").as_str(),
                None,
            ))
            .when_some(error, |d, e| {
                d.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
                        .child(text(ty::SM, t.danger_text).child(e.message)),
                )
            });
        let cancel = self.button(
            "syn-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_synth(w, cx)),
        );
        let go = if submitting {
            self.button_disabled(
                "syn-go",
                "Generating…",
                Btn::Primary,
                Some("icons/zap.svg"),
                "Writing images…",
                window,
                cx,
            )
        } else if name_err.is_some() || images_err.is_some() {
            self.button_disabled(
                "syn-go",
                "Generate",
                Btn::Primary,
                Some("icons/zap.svg"),
                "Fix the highlighted fields first",
                window,
                cx,
            )
        } else {
            self.button(
                "syn-go",
                "Generate",
                Btn::Primary,
                Some("icons/zap.svg"),
                window,
                cx,
                act(|this, w, cx| this.submit_synth(w, cx)),
            )
        };
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "synth-dialog",
            520.,
            "Generate a synthetic dataset",
            Some("Procedural shapes with exact labels: a quick way to try training end to end.".into()),
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_synth(w, cx)),
        )
    }
}
