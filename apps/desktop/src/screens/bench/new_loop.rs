//! The New loop form on the Loops tab.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, SharedString, Styled, Window, div, px};

use crate::api::Bench;
use crate::app::{FoloApp, act};
use crate::bench::{
    LOOP_EPOCHS, LOOP_GENERATIONS, SourceChoice, SourcePick, loop_pool_likely_empty, loop_pool_overlaps,
};
use crate::controls::Btn;
use crate::theme::{Theme, ty};
use crate::ui::*;

impl FoloApp {
    /// The New loop form: the teacher, the pool it labels, how many generations and how long each
    /// student trains (its architecture and image size are the teacher's).
    pub(super) fn new_loop_form(
        &mut self,
        t: &Theme,
        b: &Bench,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        // The models that can teach first; the rest are listed, off, with the reason.
        let teachers = self.bench_teachers(b);
        let others: Vec<crate::api::Model> = self
            .store
            .models
            .ready()
            .into_iter()
            .flatten()
            .filter(|m| !teachers.iter().any(|t| t.id == m.id))
            .cloned()
            .collect();
        let off: Vec<Option<SharedString>> = (0..teachers.len() + others.len())
            .map(|i| {
                (i >= teachers.len()).then(|| "Its classes aren't this benchmark's, in the same order".into())
            })
            .collect();
        let models: Vec<crate::api::Model> = teachers.iter().chain(&others).cloned().collect();
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_loop.as_ref() else { return div().into_any_element() };
        let (teacher, generations, epochs, submitting, error) =
            (f.teacher.clone(), f.generations, f.epochs, f.submitting, f.error.clone());
        let choice = self.source_choice(&sources, f.pool.as_ref());
        let teacher_gone = self.loop_teacher_deleted();
        let teacher_i = teacher.as_ref().and_then(|id| teachers.iter().position(|m| &m.id == id));
        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        let teacher_select = self.select_with(
            "bench-loop-teacher",
            teacher_i.and_then(|i| teachers.get(i)).map_or_else(
                // A teacher that appeared while the form was open (the list refreshed) is offered.
                || {
                    if teacher_gone || !teachers.is_empty() { "Pick a teacher" } else { "No model can teach" }
                        .into()
                },
                |m| m.name.clone().into(),
            ),
            models
                .iter()
                .map(|m| {
                    (m.name.clone().into(), Some(format!("{} · {} px", m.arch.as_str(), m.imgsz).into()))
                })
                .collect(),
            // None set, or deleted: none of them is the choice, so the first one can be picked.
            teacher_i.unwrap_or(models.len()),
            &off,
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| {
                if let Some(f) = this.bench.new_loop.as_mut() {
                    f.teacher = ids.get(i).cloned();
                    f.error = None;
                }
                cx.notify();
            }),
        );
        let pool_select =
            self.source_select("bench-loop-pool", &sources, choice, window, cx, |this, id, cx| {
                if let Some(f) = this.bench.new_loop.as_mut() {
                    f.pool = Some(id);
                    f.error = None;
                }
                cx.notify();
            });
        let gen_names: Vec<String> = LOOP_GENERATIONS.iter().map(u32::to_string).collect();
        let gen_refs: Vec<&str> = gen_names.iter().map(String::as_str).collect();
        let gen_ctl = self.segmented(
            "bench-loop-gens",
            &gen_refs,
            generations,
            window,
            cx,
            std::rc::Rc::new(|this, i, _, cx| {
                if let Some(f) = this.bench.new_loop.as_mut() {
                    f.generations = i.min(LOOP_GENERATIONS.len() - 1);
                }
                cx.notify();
            }),
        );
        let ep_names: Vec<String> = LOOP_EPOCHS.iter().map(u32::to_string).collect();
        let ep_refs: Vec<&str> = ep_names.iter().map(String::as_str).collect();
        let ep_ctl = self.segmented(
            "bench-loop-epochs",
            &ep_refs,
            epochs,
            window,
            cx,
            std::rc::Rc::new(|this, i, _, cx| {
                if let Some(f) = this.bench.new_loop.as_mut() {
                    f.epochs = i.min(LOOP_EPOCHS.len() - 1);
                }
                cx.notify();
            }),
        );
        let picked = choice.index().map(|i| &sources[i]);
        let pool_images = picked.map(SourcePick::pool_images);
        // The benchmark's own images must stay out of training: the server leaves them out of the pool.
        let overlaps = picked.is_some_and(|p| loop_pool_overlaps(b, p));
        let likely_empty = picked.is_some_and(|p| loop_pool_likely_empty(b, p));
        let gens = LOOP_GENERATIONS[generations];
        let (note, note_color) = match (teacher_i.and_then(|i| teachers.get(i)), pool_images) {
            (None, _) if teacher_gone => {
                ("Its teacher was deleted. Pick another.".to_string(), t.text_secondary)
            }
            (_, None) if choice == SourceChoice::Deleted => {
                ("Its pool was deleted. Pick another.".to_string(), t.text_secondary)
            }
            (None, _) if !teachers.is_empty() => {
                ("Pick a teacher: it labels the pool.".to_string(), t.text_secondary)
            }
            (None, _) => (
                "No model can teach yet: a teacher is trained on this benchmark's classes, in its order."
                    .to_string(),
                t.text_secondary,
            ),
            (_, None) => (
                "Add a dataset or label images in a labeling job first: the teacher labels its images."
                    .to_string(),
                t.text_secondary,
            ),
            (_, Some(0)) => ("That pool has no images: pick another.".to_string(), t.text_secondary),
            (Some(_), Some(n)) if likely_empty => (
                format!(
                    "All {} {} come from this benchmark's source and will be excluded unless files changed on disk. Pick another pool.",
                    int(n),
                    if matches!(picked, Some(SourcePick::Job { .. })) {
                        "images of the job"
                    } else {
                        "train images"
                    },
                ),
                t.warning_text,
            ),
            (Some(m), Some(n)) => (
                format!(
                    "{} labels {} {} by itself; a {} student trains {} epochs on those labels alone and runs here. {}{}",
                    m.name,
                    int(n),
                    if matches!(picked, Some(SourcePick::Job { .. })) {
                        "images of the job"
                    } else {
                        "train images"
                    },
                    m.arch.as_str(),
                    LOOP_EPOCHS[epochs],
                    if gens > 1 {
                        format!("Each student teaches the next, {gens} generations in all.")
                    } else {
                        "One generation.".to_string()
                    },
                    if overlaps {
                        " Images that are also in this benchmark are left out of the pool."
                    } else {
                        ""
                    },
                ),
                t.text_secondary,
            ),
        };
        let bid = b.id.clone();
        let start = if submitting {
            self.button_disabled(
                "bench-loop-start",
                "Starting…",
                Btn::Primary,
                Some("icons/repeat.svg"),
                "Starting the loop",
                window,
                cx,
            )
        } else if teacher_i.is_none() || pool_images.unwrap_or(0) == 0 {
            self.button_disabled(
                "bench-loop-start",
                "Start loop",
                Btn::Primary,
                Some("icons/repeat.svg"),
                "It needs a model to teach and a pool with images",
                window,
                cx,
            )
        } else {
            self.button(
                "bench-loop-start",
                "Start loop",
                Btn::Primary,
                Some("icons/repeat.svg"),
                window,
                cx,
                act(move |this, _, cx| this.submit_new_loop(&bid, cx)),
            )
        };
        let cancel = self.button(
            "bench-loop-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_bench_form(crate::bench::BenchForm::NewLoop, w, cx)),
        );
        let field = |label: &str, control: AnyElement| {
            div().flex().flex_col().gap(px(6.)).child(caption(&t, label)).child(control)
        };
        panel(&t)
            .child(panel_header(&t, "New loop").child(
                text(ty::SM, t.text_tertiary).child("Does a model's own labeling train a better model?"),
            ))
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(14.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .gap(px(16.))
                            .items_start()
                            .child(field(
                                "Teacher",
                                div().w(px(220.)).child(teacher_select).into_any_element(),
                            ))
                            .child(field(
                                "Pool it labels",
                                div().w(px(220.)).child(pool_select).into_any_element(),
                            ))
                            .child(field("Generations", gen_ctl.into_any_element()))
                            .child(field("Epochs per student", ep_ctl.into_any_element())),
                    )
                    .child(text(ty::SM, note_color).child(note))
                    .when_some(error, |d, e| {
                        d.child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                                .child(text(ty::SM, t.danger_text).child(e.message)),
                        )
                        .when_some(e.hint, |d, h| d.child(text(ty::SM, t.text_tertiary).child(h)))
                    })
                    .child(div().flex().gap(px(8.)).child(start).child(cancel)),
            )
            .into_any_element()
    }
}
