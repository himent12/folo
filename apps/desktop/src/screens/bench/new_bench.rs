//! The New benchmark form, and the source picker it shares with the New loop form.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, Focusable, IntoElement, ParentElement, SharedString, Styled, Window, div, px,
};

use crate::app::{FoloApp, act};
use crate::bench::{BENCH_SPLITS, SourcePick};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// A picker of image sources: the datasets (their split counts beside them), then labeling jobs.
    pub(super) fn source_select(
        &self,
        key: &'static str,
        sources: &[SourcePick],
        selected: usize,
        window: &Window,
        cx: &mut Context<Self>,
        pick: impl Fn(&mut Self, usize, &mut Context<Self>) + 'static,
    ) -> AnyElement {
        let label: SharedString =
            sources.get(selected).map_or_else(|| "No datasets".into(), |s| s.name().to_string().into());
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
            selected,
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| pick(this, i, cx)),
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
        let (dataset, split, submitting, error) = (f.dataset, f.split, f.submitting, f.error.clone());
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
            .when(name_ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .child(name_input);
        let src_select = self.source_select("bench-new-ds", &sources, dataset, window, cx, |this, i, cx| {
            this.pick_bench_source(Some(i), None, cx)
        });
        let picked = sources.get(dataset);
        let is_dataset = matches!(picked, Some(SourcePick::Dataset { .. }));
        let split_names: Vec<&str> = BENCH_SPLITS.iter().map(|s| s.as_str()).collect();
        let split_ctl = is_dataset.then(|| {
            self.segmented(
                "bench-new-split",
                &split_names,
                split,
                window,
                cx,
                std::rc::Rc::new(|this, i, _, cx| this.pick_bench_source(None, Some(i), cx)),
            )
        });
        // How many images it would copy (up to 500 by default, a seeded sample of more).
        let note = match picked.map(|p| (p, p.images(BENCH_SPLITS[split]))) {
            Some((_, n)) if n > 500 => {
                format!("{} images: a seeded sample of 500 is copied and frozen with their boxes.", int(n))
            }
            Some((SourcePick::Job { .. }, n)) => {
                format!("{} labeled images of the job, copied and frozen with their boxes.", int(n))
            }
            Some((_, 0)) => "This split has no images: pick another.".to_string(),
            Some((_, n)) => format!("{} images, copied and frozen with their boxes.", int(n)),
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
