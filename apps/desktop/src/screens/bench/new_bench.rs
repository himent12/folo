//! The New benchmark form, and the source picker it shares with the New loop form.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Entity, Focusable, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{FoloApp, SYNTH_CLASSES, act};
use crate::bench::{
    BENCH_GEN_IMGSZ, BENCH_SPLITS, BenchSourceKind, SourceChoice, SourceId, SourcePick,
    validate_generated_bench,
};
use crate::controls::Btn;
use crate::input::TextInput;
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
            SourceChoice::Missing => "No datasets or labeling jobs".into(),
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
        let generated_on = self.bench_generated_on();
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_bench.as_ref() else { return div().into_any_element() };
        let (source_kind, split, gen_imgsz, gen_vary_aspect, submitting, error) =
            (f.source_kind, f.split, f.gen_imgsz, f.gen_vary_aspect, f.submitting, f.error.clone());
        let choice = self.source_choice(&sources, f.source.as_ref());
        let name_input = f.name.clone();
        let gen_images_input = f.gen_images.clone();
        let gen_classes_input = f.gen_classes.clone();
        let gen_seed_input = f.gen_seed.clone();

        let text_field =
            |this: &Self, input: Entity<TextInput>, w: f32, bad: bool, cx: &mut Context<Self>| {
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
                    .text_size(px(ty::SM.size))
                    // Enter in any field submits, as the Create button does.
                    .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                        if ev.keystroke.key == "enter" && !ev.keystroke.modifiers.control {
                            this.submit_new_bench(cx);
                            cx.stop_propagation();
                        }
                    }))
                    .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                    .child(input)
                    .into_any_element()
            };

        let name_text = name_input.read(cx).text().to_string();
        let name_bad = name_text.trim().is_empty() || name_text.trim().chars().count() > 100;
        let name = text_field(self, name_input, 260., false, cx);

        let kind_idx = usize::from(source_kind == BenchSourceKind::Generated);
        let kind_ctl = generated_on.then(|| {
            self.segmented(
                "bench-new-kind",
                &["Dataset / Job", "Generated"],
                kind_idx,
                window,
                cx,
                Rc::new(|this, i, _, cx| {
                    let kind = if i == 1 { BenchSourceKind::Generated } else { BenchSourceKind::Existing };
                    this.pick_bench_source_kind(kind, cx);
                }),
            )
        });

        let field = |label: &str, control: AnyElement| {
            div().flex().flex_col().gap(px(6.)).child(caption(&t, label)).child(control)
        };
        let field_err = |label: &str, control: AnyElement, err: Option<String>| {
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .child(caption(&t, label))
                .child(control)
                .when_some(err, |d, msg| d.child(text(ty::XS, t.danger_text).child(msg)))
        };

        let (source_fields, note, disabled_reason): (AnyElement, String, Option<&'static str>) =
            match source_kind {
                BenchSourceKind::Existing => {
                    let src_select =
                        self.source_select("bench-new-ds", &sources, choice, window, cx, |this, id, cx| {
                            this.pick_bench_source(Some(id), None, cx)
                        });
                    let picked = choice.index().map(|i| &sources[i]);
                    let deleted = choice == SourceChoice::Deleted;
                    let is_dataset = matches!(picked, Some(SourcePick::Dataset { .. }));
                    let counts: Vec<u64> =
                        BENCH_SPLITS.iter().map(|s| picked.map_or(0, |p| p.images(*s))).collect();
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
                            Rc::new(|this, i, _, cx| this.pick_bench_source(None, Some(i), cx)),
                        )
                    });
                    let empty = picked.is_none_or(|p| p.images(BENCH_SPLITS[split]) == 0);
                    let note = match picked.map(|p| (p, p.images(BENCH_SPLITS[split]))) {
                        Some((_, n)) if n > 500 => format!(
                            "{} images: a seeded sample of 500 is copied and frozen with their boxes.",
                            int(n)
                        ),
                        Some((SourcePick::Job { .. }, n)) => format!(
                            "{} of the job, copied and frozen with their boxes.",
                            counted(n, "labeled image", "labeled images")
                        ),
                        Some((_, 0)) => "This split has no images: pick another.".to_string(),
                        Some((_, n)) => {
                            format!("{}, copied and frozen with their boxes.", counted(n, "image", "images"))
                        }
                        None if deleted => "That source was deleted. Pick another.".to_string(),
                        None if generated_on => {
                            "Add a dataset or label images in a labeling job first, or switch Source to Generated."
                                .to_string()
                        }
                        None => "Add a dataset or label images in a labeling job first.".to_string(),
                    };
                    let why_off = if empty {
                        Some(if deleted {
                            "Pick another source"
                        } else if picked.is_none() {
                            "Add a dataset or labeling job first"
                        } else {
                            "This split has no images: pick another"
                        })
                    } else if name_bad {
                        Some("Enter a benchmark name (1 to 100 characters)")
                    } else {
                        None
                    };
                    let row = div()
                        .flex()
                        .flex_wrap()
                        .gap(px(16.))
                        .items_start()
                        .child(field("Name", name))
                        .when_some(kind_ctl, |d, k| d.child(field("Source", k.into_any_element())))
                        .child(field("Images from", div().w(px(240.)).child(src_select).into_any_element()))
                        .when_some(split_ctl, |d, s| d.child(field("Split", s.into_any_element())))
                        .into_any_element();
                    (row, note, why_off)
                }
                BenchSourceKind::Generated => {
                    let imgsz = BENCH_GEN_IMGSZ[gen_imgsz.min(BENCH_GEN_IMGSZ.len() - 1)];
                    let imgs_text = gen_images_input.read(cx).text().to_string();
                    let seed_text = gen_seed_input.read(cx).text().to_string();
                    let cls_text = gen_classes_input.read(cx).text().to_string();
                    let checked =
                        validate_generated_bench(&imgs_text, &seed_text, &cls_text, imgsz, gen_vary_aspect);
                    let errs = checked.as_ref().err().cloned().unwrap_or_default();

                    let images_ctl = text_field(self, gen_images_input, 104., errs.images.is_some(), cx);
                    let seed_ctl = text_field(self, gen_seed_input, 120., errs.seed.is_some(), cx);
                    let classes_ctl = text_field(self, gen_classes_input, 320., errs.classes.is_some(), cx);

                    let size_labels: Vec<String> =
                        BENCH_GEN_IMGSZ.iter().map(|s| format!("{s}\u{a0}px")).collect();
                    let size_refs: Vec<&str> = size_labels.iter().map(String::as_str).collect();
                    let size_ctl = self.segmented(
                        "bench-new-gen-imgsz",
                        &size_refs,
                        gen_imgsz.min(BENCH_GEN_IMGSZ.len() - 1),
                        window,
                        cx,
                        Rc::new(|this, i, _, cx| {
                            if let Some(f) = this.bench.new_bench.as_mut() {
                                f.gen_imgsz = i.min(BENCH_GEN_IMGSZ.len() - 1);
                                f.error = None;
                                cx.notify();
                            }
                        }),
                    );
                    let vary_ctl = self
                        .chip(
                            "bench-new-gen-vary-aspect",
                            "Varied aspect",
                            gen_vary_aspect,
                            None,
                            None,
                            window,
                            cx,
                            act(|this, _, cx| {
                                if let Some(f) = this.bench.new_bench.as_mut() {
                                    f.gen_vary_aspect = !f.gen_vary_aspect;
                                    f.error = None;
                                    cx.notify();
                                }
                            }),
                        )
                        .tooltip(crate::controls::tip(
                            "Mix 1:1, 4:3, 3:4, 16:9, and 3:2 aspect ratios instead of square frames",
                            None,
                        ));
                    let preset_chips: Vec<AnyElement> = ["3 shapes", "5 shapes", "8 shapes"]
                        .iter()
                        .enumerate()
                        .map(|(idx, label)| {
                            let active = SYNTH_CLASSES[idx].join(", ") == cls_text.trim();
                            self.chip(
                                match idx {
                                    0 => "bench-new-gen-cls-0",
                                    1 => "bench-new-gen-cls-1",
                                    _ => "bench-new-gen-cls-2",
                                },
                                *label,
                                active,
                                None,
                                None,
                                window,
                                cx,
                                act(move |this, _, cx| this.set_bench_gen_class_preset(idx, cx)),
                            )
                            .into_any_element()
                        })
                        .collect();

                    let note = match &checked {
                        Ok(spec) => {
                            let geom = if spec.vary_aspect {
                                format!("up to {} px (varied aspect)", spec.imgsz)
                            } else {
                                format!("{}×{} px", spec.imgsz, spec.imgsz)
                            };
                            format!(
                                "{} deterministic {geom} scenes ({}, seed {}), generated on the server with exact boxes.",
                                int(spec.images as u64),
                                plural(spec.classes.len(), "class"),
                                spec.seed,
                            )
                        }
                        Err(_) => {
                            "Deterministic procedural scenes (1 to 5,000 images, 1 to 8 classes) generated directly on the server without registering a dataset."
                                .to_string()
                        }
                    };
                    let why_off = if !generated_on {
                        Some("This server doesn't advertise bench_generated_source")
                    } else if errs.any() {
                        Some("Fix the highlighted generated-source fields first")
                    } else if name_bad {
                        Some("Enter a benchmark name (1 to 100 characters)")
                    } else {
                        None
                    };
                    let rows = div()
                        .flex()
                        .flex_col()
                        .gap(px(12.))
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .gap(px(16.))
                                .items_start()
                                .child(field("Name", name))
                                .when_some(kind_ctl, |d, k| d.child(field("Source", k.into_any_element())))
                                .child(field_err("Images", images_ctl, errs.images.map(str::to_string)))
                                .child(field_err("Seed", seed_ctl, errs.seed.map(str::to_string))),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_wrap()
                                .gap(px(16.))
                                .items_start()
                                .child(field("Image size", size_ctl.into_any_element()))
                                .child(field("Aspect ratio", vary_ctl.into_any_element()))
                                .child(field_err(
                                    "Classes (comma-separated, 1–8)",
                                    div()
                                        .flex()
                                        .flex_wrap()
                                        .items_center()
                                        .gap(px(8.))
                                        .child(classes_ctl)
                                        .children(preset_chips)
                                        .into_any_element(),
                                    errs.classes,
                                )),
                        )
                        .into_any_element();
                    (rows, note, why_off)
                }
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
        } else if let Some(why) = disabled_reason {
            self.button_disabled(
                "bench-new-create",
                "Create benchmark",
                Btn::Primary,
                Some("icons/plus.svg"),
                why,
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
            act(|this, w, cx| this.close_bench_form(crate::bench::BenchForm::NewBench, w, cx)),
        );
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
                    .child(source_fields)
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
                        .when_some(e.hint, |d, h| d.child(text(ty::SM, t.text_tertiary).child(h)))
                    })
                    .child(div().flex().gap(px(8.)).child(create).child(cancel)),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use crate::app::Route;
    use crate::app::tests::{drive, open};
    use crate::bench::{
        BenchSourceKind, parse_bench_gen_classes, parse_bench_gen_images, parse_bench_gen_seed,
        validate_generated_bench,
    };
    use crate::screens::bench::format::{is_generated_bench, source_line};

    #[test]
    fn generated_bench_validators_enforce_server_bounds() {
        assert_eq!(parse_bench_gen_images("64"), Some(64));
        assert_eq!(parse_bench_gen_images("1,000"), Some(1000));
        assert_eq!(parse_bench_gen_images("5000"), Some(5000));
        assert_eq!(parse_bench_gen_images("0"), None);
        assert_eq!(parse_bench_gen_images("5001"), None);
        assert_eq!(parse_bench_gen_images("1,5"), None);

        assert_eq!(parse_bench_gen_seed("0"), Some(0));
        assert_eq!(parse_bench_gen_seed("9007199254740991"), Some(folo_bench::source::MAX_SEED));
        assert_eq!(parse_bench_gen_seed("9007199254740992"), None);
        assert_eq!(parse_bench_gen_seed("-1"), None);

        assert_eq!(parse_bench_gen_classes("bolt, nut").unwrap(), vec!["bolt", "nut"]);
        assert!(parse_bench_gen_classes("").is_err());
        assert!(parse_bench_gen_classes("circle, , square").is_err());
        assert!(parse_bench_gen_classes("circle, circle").is_err());
        assert!(parse_bench_gen_classes("a,b,c,d,e,f,g,h,i").is_err());

        let spec = validate_generated_bench("32", "42", "circle, square", 320, true).unwrap();
        assert_eq!(spec.images, 32);
        assert_eq!(spec.seed, 42);
        assert_eq!(spec.imgsz, 320);
        assert!(spec.vary_aspect);
        assert_eq!(spec.classes, vec!["circle", "square"]);
    }

    #[test]
    fn new_bench_generated_source_creates_benchmark_and_renders_header_badge() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Benches, w, cx);
            this.open_new_bench(w, cx);
            this.pick_bench_source_kind(BenchSourceKind::Generated, cx);
            let f = this.bench.new_bench.as_mut().unwrap();
            f.name.update(cx, |i, cx| i.set_text("Synthetic Smoke", cx));
            f.gen_images.update(cx, |i, cx| i.set_text("12", cx));
            f.gen_seed.update(cx, |i, cx| i.set_text("7", cx));
            f.gen_vary_aspect = true;
            this.set_bench_gen_class_preset(1, cx);
        });
        app.frame(window.into());
        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            for k in
                ["bench-new-kind", "bench-new-gen-imgsz", "bench-new-gen-vary-aspect", "bench-new-gen-cls-1"]
            {
                assert!(keys.iter().any(|x| x == k), "missing {k} in {keys:?}");
            }
        });
        drive(&mut app, window, |this, _, cx| this.submit_new_bench(cx));
        app.run_until_idle();
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());

        let created_id = drive(&mut app, window, |this, _, _| {
            let Route::Bench(id) = &this.route else {
                panic!("expected navigation to created benchmark, got {:?}", this.route);
            };
            let b =
                this.bench.list.ready().unwrap().iter().find(|x| x.id == *id).expect("created bench in list");
            assert_eq!(b.name, "Synthetic Smoke");
            assert_eq!(b.images, 12);
            assert_eq!(b.seed, 7);
            assert_eq!(b.classes.len(), 5);
            assert!(is_generated_bench(b));
            let src = source_line(b);
            assert!(src.contains("generated (12×640px, varied aspect, seed 7)"), "{src}");
            assert!(!src.contains("· val"), "generated source must not append split suffix: {src}");
            id.clone()
        });
        assert!(!created_id.is_empty());
    }

    #[test]
    fn generated_bench_source_hidden_when_server_feature_absent() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            this.store.features.retain(|f| f != crate::api::FEATURE_BENCH_GENERATED_SOURCE);
            assert!(!this.bench_generated_on());
            this.navigate(Route::Benches, w, cx);
            this.open_new_bench(w, cx);
            this.pick_bench_source_kind(BenchSourceKind::Generated, cx);
            assert_eq!(this.bench.new_bench.as_ref().unwrap().source_kind, BenchSourceKind::Existing);
        });
        app.frame(window.into());
        drive(&mut app, window, |this, _, _| {
            let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
            assert!(
                !keys.iter().any(|k| k == "bench-new-kind"),
                "kind picker shown without feature: {keys:?}"
            );
        });
    }
}
