//! The New benchmark form, and the source picker it shares with the New loop form.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, InteractiveElement, IntoElement, ParentElement, SharedString, Styled,
    Window, div, px,
};

use crate::app::{FoloApp, act};
use crate::bench::{BENCH_SPLITS, SourceChoice, SourceId, SourcePick};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// A picker of image sources: the datasets (their split counts beside them), then labeling jobs.
    pub(super) fn source_select(
        &self,
        key: &'static str,
        sources: &[SourcePick],
        choice: SourceChoice,
        window: &Window,
        cx: &mut Context<Self>,
        pick: impl Fn(&mut Self, SourceId, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        let label: SharedString = match choice {
            SourceChoice::At(i) => sources[i].name().to_string().into(),
            SourceChoice::Missing => "No datasets".into(),
            SourceChoice::Deleted => "Pick a source".into(),
        };
        let ids: Vec<SourceId> = sources.iter().map(SourcePick::id).collect();
        let options: Vec<(SharedString, Option<SharedString>)> = sources
            .iter()
            .map(|s| {
                let meta = match s {
                    SourcePick::Dataset { splits, .. } => format!(
                        "{} train · {} val · {} test",
                        int(splits.train),
                        int(splits.val),
                        int(splits.test)
                    ),
                    SourcePick::Job { labeled, .. } => format!("labeling job · {} labeled", int(*labeled)),
                };
                (s.name().to_string().into(), Some(meta.into()))
            })
            .collect();
        self.select(
            key,
            label,
            options,
            // Deleted, none of them is the choice.
            choice.index().unwrap_or(sources.len()),
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| {
                if let Some(id) = ids.get(i) {
                    pick(this, id.clone(), cx)
                }
            }),
        )
    }

    pub(super) fn new_bench_form(
        &mut self,
        t: &Theme,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_bench.as_ref() else { return div().into_any_element() };
        let (split, submitting, error) = (f.split, f.submitting, f.error.clone());
        let choice = self.source_choice(&sources, f.source.as_ref());
        let name_input = f.name.clone();
        let name_fh = name_input.read(cx).focus_handle(cx);
        let name_ring = self.ring(&name_fh, window);
        let name = div()
            .relative()
            .w(px(280.))
            .h(px(32.))
            .px(px(10.))
            .flex()
            .items_center()
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if name_fh.is_focused(window) { t.accent_line } else { t.line_control })
            .text_size(px(ty::SM.size))
            // Enter in the name creates it, as the Create button does.
            .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                if ev.keystroke.key == "enter" && !ev.keystroke.modifiers.control {
                    this.submit_new_bench(cx);
                    cx.stop_propagation();
                }
            }))
            .when(name_ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .child(name_input);
        let src_select = self.source_select("bench-new-ds", &sources, choice, window, cx, |this, id, cx| {
            this.pick_bench_source(Some(id), None, cx)
        });
        let picked = choice.index().map(|i| &sources[i]);
        let deleted = choice == SourceChoice::Deleted;
        let is_dataset = matches!(picked, Some(SourcePick::Dataset { .. }));
        // Each split with its image count (a no-break space: the control's quieter count style);
        // an empty one is there but off, with the reason.
        let counts: Vec<u64> = BENCH_SPLITS.iter().map(|s| picked.map_or(0, |p| p.images(*s))).collect();
        let split_names: Vec<String> = BENCH_SPLITS
            .iter()
            .zip(&counts)
            .map(|(s, n)| format!("{}\u{a0}{}", s.as_str(), int(*n)))
            .collect();
        let split_refs: Vec<&str> = split_names.iter().map(String::as_str).collect();
        let split_off: Vec<Option<SharedString>> =
            counts.iter().map(|n| (*n == 0).then(|| "No images in this split".into())).collect();
        let split_ctl = is_dataset.then(|| {
            self.segmented_with(
                "bench-new-split",
                &split_refs,
                split,
                &split_off,
                window,
                cx,
                std::rc::Rc::new(|this, i, _, cx| this.pick_bench_source(None, Some(i), cx)),
            )
        });
        let empty = picked.is_none_or(|p| p.images(BENCH_SPLITS[split]) == 0);
        // How many images it would copy (up to 500 by default, a seeded sample of more).
        let note = match picked.map(|p| (p, p.images(BENCH_SPLITS[split]))) {
            Some((_, n)) if n > 500 => {
                format!("{} images: a seeded sample of 500 is copied and frozen with their boxes.", int(n))
            }
            Some((SourcePick::Job { .. }, n)) => {
                format!("{} labeled images of the job, copied and frozen with their boxes.", int(n))
            }
            Some((_, 0)) => "This split has no images: pick another.".to_string(),
            Some((_, n)) => format!("{}, copied and frozen with their boxes.", counted(n, "image", "images")),
            None if deleted => "That source was deleted. Pick another.".to_string(),
            None => "Add a dataset first: a benchmark copies its images.".to_string(),
        };
        let create = if submitting {
            self.button_disabled(
                "bench-new-create",
                "Creating…",
                Btn::Primary,
                Some("icons/plus.svg"),
                "Creating the benchmark",
                window,
                cx,
            )
        } else if empty {
            self.button_disabled(
                "bench-new-create",
                "Create benchmark",
                Btn::Primary,
                Some("icons/plus.svg"),
                if deleted {
                    "Pick another source"
                } else if picked.is_none() {
                    "Add a dataset first"
                } else {
                    "This split has no images: pick another"
                },
                window,
                cx,
            )
        } else {
            self.button(
                "bench-new-create",
                "Create benchmark",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, _, cx| this.submit_new_bench(cx)),
            )
        };
        let cancel = self.button(
            "bench-new-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, _, cx| {
                this.bench.new_bench = None;
                cx.notify();
            }),
        );
        let field = |label: &str, control: AnyElement| {
            div().flex().flex_col().gap(px(6.)).child(caption(&t, label)).child(control)
        };
        panel(&t)
            .child(panel_header(&t, "New benchmark").child(
                text(ty::SM, t.text_tertiary).child("Default plan: twelve conditions at three severities"),
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
                            .child(field("Name", name.into_any_element()))
                            .child(field(
                                "Images from",
                                div().w(px(240.)).child(src_select).into_any_element(),
                            ))
                            .when_some(split_ctl, |d, s| d.child(field("Split", s.into_any_element()))),
                    )
                    .child(text(ty::SM, t.text_secondary).child(note))
                    .when_some(error, |d, e| {
                        d.child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                                .child(text(ty::SM, t.danger_text).child(e.message)),
                        )
                    })
                    .child(div().flex().gap(px(8.)).child(create).child(cancel)),
            )
            .into_any_element()
    }
}
