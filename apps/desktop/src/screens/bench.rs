//! Benchmarks (servers with `vision_bench`, docs/BENCH.md): the benchmark list with a New benchmark
//! form, a benchmark's leaderboard, plan and loops, and an entry's result on the five axes.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::*;
use crate::api::time;
use crate::api::{
    Bench, BenchAxis, BenchContestant, BenchEntry, BenchEntryStatus, BenchLoop, BenchLoopStatus,
    BenchLoopVerdict, BenchReadiness, BenchResult, BenchSource, Label, LabelAgentId,
};
use crate::app::{FoloApp, Route, act};
use crate::bench::{
    BENCH_SPLITS, BenchTab, Contender, ImageSort, LOOP_EPOCHS, LOOP_GENERATIONS, SourcePick, model_fits,
    preview_severity,
};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

/// An axis's short name, as the leaderboard heads its column.
pub fn axis_short(a: BenchAxis) -> &'static str {
    match a {
        BenchAxis::Accuracy => "Accuracy",
        BenchAxis::Robustness => "Robust.",
        BenchAxis::SelfLabeling => "Self-label",
        BenchAxis::Calibration => "Calib.",
        BenchAxis::Localization => "Local.",
    }
}

/// An axis's full name.
pub fn axis_name(a: BenchAxis) -> &'static str {
    match a {
        BenchAxis::Accuracy => "Accuracy",
        BenchAxis::Robustness => "Robustness",
        BenchAxis::SelfLabeling => "Self-labeling",
        BenchAxis::Calibration => "Calibration",
        BenchAxis::Localization => "Localization",
    }
}

/// An axis value (0..=1) on the score's 0–100 scale; "–" when it was not measured.
pub fn axis_text(v: Option<f32>) -> String {
    v.map_or_else(|| "–".to_string(), |v| format!("{:.0}", (v * 100.).clamp(0., 100.)))
}

/// A share as a percentage: 0.834 → "83%".
pub fn pct(v: f32) -> String {
    format!("{:.0}%", (v * 100.).clamp(0., 100.))
}

/// Who an entry is, in a few words: "folo-n · 640 px", "Claude Code · sonnet", "External model".
pub fn contestant_line(e: &BenchEntry) -> String {
    match &e.contestant {
        BenchContestant::Model { arch, imgsz, .. } => format!("{arch} · {imgsz} px"),
        BenchContestant::Agent { agent, model, effort } => {
            let who = agent_name(*agent);
            let model = model.as_deref().unwrap_or("default model");
            format!("{who} · {model} · {} effort", effort.label().to_lowercase())
        }
        BenchContestant::External => "External model".to_string(),
    }
}

/// A coding agent's product name.
pub fn agent_name(a: LabelAgentId) -> &'static str {
    match a {
        LabelAgentId::Claude => "Claude Code",
        LabelAgentId::Codex => "Codex",
    }
}

/// Where a benchmark's images came from: "shapes-1k · val".
pub fn source_line(b: &Bench) -> String {
    match &b.source {
        BenchSource::Dataset { split, .. } => {
            format!("{} · {}", b.source_name, split.map_or("val", |s| s.as_str()))
        }
        BenchSource::LabelJob { .. } => format!("{} · labeling job", b.source_name),
    }
}

/// The badge tone of an entry's status.
pub fn entry_tone(s: BenchEntryStatus) -> Tone {
    match s {
        BenchEntryStatus::Queued | BenchEntryStatus::Stopping => Tone::Neutral,
        BenchEntryStatus::Running => Tone::Accent,
        BenchEntryStatus::Done => Tone::Success,
        BenchEntryStatus::Failed => Tone::Danger,
        BenchEntryStatus::Stopped => Tone::Warning,
    }
}

/// The badge tone of a loop's status.
pub fn loop_tone(s: BenchLoopStatus) -> Tone {
    match s {
        BenchLoopStatus::Running => Tone::Accent,
        BenchLoopStatus::Stopping => Tone::Neutral,
        BenchLoopStatus::Done => Tone::Success,
        BenchLoopStatus::Failed => Tone::Danger,
        BenchLoopStatus::Stopped => Tone::Warning,
    }
}

/// How far a running entry is, 0..=1.
pub fn entry_frac(e: &BenchEntry) -> f32 {
    if e.progress.total == 0 { 0. } else { (e.progress.done as f32 / e.progress.total as f32).clamp(0., 1.) }
}

/// The leaderboard's ranks: finished entries with the benchmark's fingerprint, in the server's order
/// (complete ones first, then partial ones); the rest are not ranked.
pub fn ranks(b: &Bench, entries: &[BenchEntry]) -> Vec<Option<usize>> {
    let mut n = 0;
    entries
        .iter()
        .map(|e| {
            (e.status == BenchEntryStatus::Done && e.fingerprint == b.fingerprint).then(|| {
                n += 1;
                n
            })
        })
        .collect()
}

/// A name and its badges on one line `width` px wide. GPUI truncates text only against a width it
/// knows, so the name gets one: its own width, or what the badges leave of the line.
fn name_with_badges(
    t: &Theme,
    window: &mut Window,
    name: &str,
    badges: &[(&str, Tone)],
    width: f32,
) -> gpui::Div {
    use crate::charts::text_width;
    // Medium weight runs a little wider than the regular face text_width measures.
    let mut measure = |s: &str| f32::from(text_width(window, s, ty::SM.size)) * 1.04;
    let badge_w: f32 = badges.iter().map(|(l, _)| measure(l) + 12. + 8.).sum();
    let name_w = (measure(name) + 2.).min((width - badge_w).max(48.));
    div()
        .flex()
        .items_center()
        .gap(px(8.))
        .w(px(width))
        .child(
            text(ty::SM, t.text_primary)
                .font_weight(FontWeight::MEDIUM)
                .w(px(name_w))
                .ellipsis()
                .child(name.to_string()),
        )
        .children(badges.iter().map(|(l, tone)| badge(t, l.to_string(), *tone)))
}

fn num_cell(w: f32, content: impl IntoElement) -> gpui::Div {
    div().w(px(w)).flex_none().flex().justify_end().child(content)
}

/// A labelled bar: a 0..=1 value as a filled track with its text on the right.
fn bar_row(
    t: &Theme,
    label: impl Into<SharedString>,
    frac: f32,
    value: String,
    color: gpui::Hsla,
) -> gpui::Div {
    div()
        .flex()
        .items_center()
        .gap(px(12.))
        .h(px(28.))
        .child(text(ty::SM, t.text_secondary).w(px(140.)).flex_none().ellipsis().child(label.into()))
        .child(div().flex_1().min_w_0().child(progress(t, frac, color, 6.)))
        .child(text(ty::SM, t.text_primary).tnum().w(px(56.)).flex_none().flex().justify_end().child(value))
}

impl FoloApp {
    /// The display name of a benchmark (its id until the list is in).
    pub fn bench_name(&self, id: &str) -> String {
        self.bench
            .list
            .ready()
            .and_then(|l| l.iter().find(|b| b.id == id))
            .map_or(id.to_string(), |b| b.name.clone())
    }

    /// The display name of an entry (its id until the leaderboard is in).
    pub fn bench_entry_name(&self, bench: &str, entry: &str) -> String {
        self.bench
            .entries
            .get(bench)
            .and_then(|l| l.ready())
            .and_then(|l| l.iter().find(|e| e.id == entry))
            .map_or(entry.to_string(), |e| e.name.clone())
    }

    /// A picker of image sources: the datasets (their split counts beside them), then labeling jobs.
    fn source_select(
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

    /// A contestant's name in the Run picker, and what it runs.
    fn contender_option(&self, c: &Contender) -> (SharedString, Option<SharedString>) {
        match c {
            Contender::Model(id) => self.store.model(id).map_or_else(
                || (id.clone().into(), None),
                |m| (m.name.clone().into(), Some(format!("{} · {} px", m.arch.as_str(), m.imgsz).into())),
            ),
            Contender::Agent(a) => {
                let agent = self.bench_agents().into_iter().find(|x| x.id == *a);
                let name = agent.as_ref().map_or_else(|| agent_name(*a).to_string(), |x| x.name.clone());
                let model = agent.and_then(|x| x.default_model).unwrap_or_else(|| "default model".into());
                (name.into(), Some(format!("agent · {model} · first images, unchanged").into()))
            }
        }
    }

    /// Without `vision_bench`: what the section needs, in place of its content.
    fn bench_unsupported_panel(&self, t: &Theme, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let settings = self.button(
            "bench-settings",
            "Connection settings",
            Btn::Secondary,
            Some("icons/settings.svg"),
            window,
            cx,
            act(|this, w, cx| this.navigate(Route::Settings, w, cx)),
        );
        panel(t)
            .child(
                empty_state(
                    t,
                    "icons/lock.svg",
                    "This server has no vision benchmarks",
                    "Benchmarks need a real folo serve: the demo server (folo serve --mock) has none. Start folo serve without --mock, connect to another server, or use the built-in demo.",
                    Tone::Neutral,
                )
                .child(settings),
            )
            .into_any_element()
    }

    // ---------- the list ----------

    pub fn benches(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_benches(cx);
        if std::mem::take(&mut self.bench.list_stale) && self.bench.list.ready().is_some() {
            self.refresh_benches(cx);
        }
        let list = self.bench.list.ready().cloned();
        let subtitle = match (&list, self.bench.list.error()) {
            (Some(l), _) => format!(
                "{} · your images under hard conditions, scored on five axes",
                plural(l.len(), "benchmark")
            ),
            (None, Some(_)) => "The benchmark list didn't load".to_string(),
            _ if self.bench_unsupported() => "Not available on this server".to_string(),
            _ => "Loading…".to_string(),
        };
        let mut actions = Vec::new();
        if self.bench_on() && self.bench.new_bench.is_none() {
            actions.push(
                self.button(
                    "bench-new",
                    "New benchmark",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_bench(w, cx)),
                )
                .into_any_element(),
            );
        }
        let header = page_header(&t, "Benchmarks", subtitle, actions);
        if self.bench_unsupported() {
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        if let Some(e) = self.bench.list.error().cloned() {
            let err = self.error_panel(&e, "benches-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }
        let form = self.bench.new_bench.is_some().then(|| self.new_bench_form(&t, window, cx));
        // The name column's width: what the fixed columns, their gaps and the row padding leave.
        let name_w =
            (self.content_width(window) - (72. + 84. + 64. + 220. + 84.) - 5. * 16. - 42.).clamp(120., 640.);
        let now = self.store.api.now();
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Benchmark").flex_1().min_w(px(200.)))
            .child(num_cell(72., th(&t, "Images")))
            .child(num_cell(84., th(&t, "Conditions")))
            .child(num_cell(64., th(&t, "Entries")))
            .child(th(&t, "Leader").w(px(220.)).flex_none())
            .child(num_cell(84., th(&t, "Created")));
        let table = match list {
            None => panel(&t).child(head).child(loading_rows(&t, "benches", 3)).into_any_element(),
            Some(l) if l.is_empty() => panel(&t)
                .child(empty_state(
                    &t,
                    "icons/target.svg",
                    "No benchmarks yet",
                    "A benchmark copies a dataset split, freezes it, and grades any model on it: accuracy, how much survives low light, blur or haze, whether its confidences can be trusted, and whether it could label new images by itself.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Some(l) => {
                let rows: Vec<AnyElement> = l
                    .iter()
                    .map(|b| {
                        let key = format!("bench-row-{}", b.id);
                        let id = b.id.clone();
                        let leader = match &b.best {
                            Some(best) => div()
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .min_w_0()
                                .child(text(ty::SM, t.text_primary).tnum().font_weight(FontWeight::MEDIUM).child(format!("{:.1}", best.score)))
                                .child(text(ty::SM, t.text_secondary).ellipsis().min_w_0().child(best.name.clone()))
                                .when(best.partial, |d| d.child(badge(&t, "partial", Tone::Neutral))),
                            None => text(ty::SM, t.text_tertiary).child(if b.active { "Running…" } else { "No finished entry" }),
                        };
                        let at = time::parse(&b.created);
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .flex()
                            .items_center()
                            .gap(px(16.))
                            .px(px(16.))
                            .h(px(52.))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(200.))
                                    .flex()
                                    .flex_col()
                                    .child(name_with_badges(
                                        &t,
                                        window,
                                        &b.name,
                                        if b.active { &[("Running", Tone::Accent)] } else { &[] },
                                        name_w,
                                    ))
                                    .child(text(ty::SM, t.text_tertiary).ellipsis().child(source_line(b))),
                            )
                            .child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(int(b.images))))
                            .child(num_cell(84., text(ty::SM, t.text_secondary).tnum().child(b.plan.conditions.len().to_string())))
                            .child(num_cell(64., text(ty::SM, t.text_secondary).tnum().child(b.entries.to_string())))
                            .child(div().w(px(220.)).flex_none().min_w_0().child(leader))
                            .child(num_cell(
                                84.,
                                text(ty::SM, t.text_tertiary).tnum().child(at.map_or(String::new(), |c| time::ago(c, now))),
                            ));
                        self.bind(row, &key, 0., window, cx, act(move |this, w, cx| this.navigate(Route::Bench(id.clone()), w, cx)))
                            .into_any_element()
                    })
                    .collect();
                panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
            }
        };
        let body = div().flex().flex_col().gap(px(16.)).children(form).child(table);
        self.page_frame(&t, header, body, cx)
    }

    fn new_bench_form(&mut self, t: &Theme, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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

    // ---------- a benchmark ----------

    pub fn bench_view(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_bench(id, cx);
        if self.bench_unsupported() {
            let header = page_header(&t, "Benchmark", "Not available on this server", vec![]);
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let Some(b) = self.bench.list.ready().and_then(|l| l.iter().find(|b| b.id == id)).cloned() else {
            let header = page_header(&t, "Benchmark", "", vec![]);
            let body = if let Some(e) = self.bench.list.error().cloned() {
                self.error_panel(&e, "bench-retry", window, cx)
            } else if self.bench.list.is_loading() {
                panel(&t).child(loading_rows(&t, "bench", 5)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Benchmark not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body, cx);
        };
        let now = self.store.api.now();
        let created = time::parse(&b.created).map_or(String::new(), |c| time::ago(c, now));
        let title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).ellipsis().child(b.name.clone()))
            .when(b.active, |d| d.child(badge(&t, "Running", Tone::Accent)));
        let subtitle = format!(
            "{} · {} images · {} boxes · {} · created {created}",
            source_line(&b),
            int(b.images),
            int(b.boxes),
            plural(b.plan.conditions.len(), "condition"),
        );
        let confirming = self.bench.confirm_delete.as_ref().is_some_and(|(x, i)| *x == b.id && i.is_none());
        let mut actions = Vec::new();
        if confirming {
            actions.push(
                self.button(
                    "bench-del-cancel",
                    "Keep it",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.bench.confirm_delete = None;
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
            actions.push(
                self.button(
                    "bench-del-confirm",
                    "Delete benchmark and entries",
                    Btn::DangerCommit,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.confirm_bench_delete(cx)),
                )
                .into_any_element(),
            );
        } else if b.active {
            actions.push(
                self.button_disabled(
                    "bench-del",
                    "Delete",
                    Btn::Secondary,
                    Some("icons/trash-2.svg"),
                    "Stop its running entries and loops first",
                    window,
                    cx,
                )
                .into_any_element(),
            );
        } else {
            let bid = b.id.clone();
            actions.push(
                self.button(
                    "bench-del",
                    "Delete",
                    Btn::Secondary,
                    Some("icons/trash-2.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.bench.confirm_delete = Some((bid.clone(), None));
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        let header = detail_header(&t, title, subtitle, actions);

        let entries = self.bench.entries.get(id).cloned().unwrap_or_default();
        let loops = self.bench.loops.get(id).cloned().unwrap_or_default();
        let n_entries = entries.ready().map_or(b.entries as usize, |l| l.len());
        let n_loops = loops.ready().map(|l| l.len());
        let tab = self.bench.tab;
        let tab_names = [
            format!("Leaderboard · {n_entries}"),
            format!("Plan · {}", b.plan.conditions.len()),
            n_loops.map_or_else(|| "Loops".to_string(), |n| format!("Loops · {n}")),
        ];
        let tab_refs: Vec<&str> = tab_names.iter().map(String::as_str).collect();
        let tabs = self.segmented(
            "bench-tabs",
            &tab_refs,
            tab as usize,
            window,
            cx,
            std::rc::Rc::new(|this, i, _, cx| {
                this.bench.tab = [BenchTab::Leaderboard, BenchTab::Plan, BenchTab::Loops][i.min(2)];
                cx.notify();
            }),
        );
        let description = (!b.description.is_empty())
            .then(|| text(ty::BASE, t.text_secondary).max_w(px(820.)).child(b.description.clone()));
        let content = match tab {
            BenchTab::Leaderboard => self.bench_leaderboard(&t, &b, &entries, window, cx),
            BenchTab::Plan => self.bench_plan(&t, &b, window, cx),
            BenchTab::Loops => self.bench_loops(&t, &b, &loops, window, cx),
        };
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .children(description)
            .child(div().flex().child(tabs))
            .child(content);
        self.page_frame(&t, header, body, cx)
    }

    fn bench_leaderboard(
        &mut self,
        t: &Theme,
        b: &Bench,
        entries: &Load<Vec<BenchEntry>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        // Run a contestant: the picked one (the newest model by default), or a coding agent.
        let contenders = self.bench_contenders(b);
        let has_models = contenders.iter().any(|c| matches!(c, Contender::Model(_)));
        let picked = self.bench_run_model(b);
        let picked_i = picked.as_ref().and_then(|p| contenders.iter().position(|c| c == p)).unwrap_or(0);
        // The models that can't run here are listed too, off, with the reason.
        let unfit: Vec<Contender> = self
            .store
            .models
            .ready()
            .into_iter()
            .flatten()
            .filter(|m| !model_fits(&m.classes, &b.classes))
            .map(|m| Contender::Model(m.id.clone()))
            .collect();
        let mut run_controls = div().flex().items_center().gap(px(8.));
        if !contenders.is_empty() {
            let list: Vec<Contender> = contenders.iter().chain(&unfit).cloned().collect();
            let options: Vec<(SharedString, Option<SharedString>)> =
                list.iter().map(|c| self.contender_option(c)).collect();
            let off: Vec<Option<SharedString>> = (0..list.len())
                .map(|i| (i >= contenders.len()).then(|| "It knows none of this benchmark's classes".into()))
                .collect();
            let label = options.get(picked_i).map_or_else(|| "Model".into(), |o| o.0.clone());
            let select = self.select_with(
                "bench-run-model",
                label,
                options,
                picked_i,
                &off,
                window,
                cx,
                std::rc::Rc::new(move |this, i, _, cx| {
                    this.bench.run_model = list.get(i).cloned();
                    cx.notify();
                }),
            );
            let bid = b.id.clone();
            let run = if self.bench.starting {
                self.button_disabled(
                    "bench-run",
                    "Starting…",
                    Btn::Primary,
                    Some("icons/play.svg"),
                    "Starting the entry",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "bench-run",
                    "Run",
                    Btn::Primary,
                    Some("icons/play.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.run_bench_model(&bid, cx)),
                )
            };
            run_controls = run_controls.child(div().w(px(220.)).child(select)).child(run);
        }
        let head_row = panel_header(&t, "Leaderboard").child(run_controls);
        let cols = |d: gpui::Stateful<gpui::Div>| d.flex().items_center().gap(px(16.)).px(px(16.));
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(num_cell(24., th(&t, "#")))
            .child(th(&t, "Entry").flex_1().min_w(px(180.)))
            .child(num_cell(64., th(&t, "Score")))
            .children(BenchAxis::ALL.iter().map(|a| num_cell(68., th(&t, axis_short(*a)))))
            .child(div().w(px(116.)).flex_none());
        let list = match entries {
            Load::Ready(l) => l.clone(),
            Load::Failed(e) => {
                return panel(&t)
                    .child(head_row)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "The leaderboard didn't load",
                        &e.message,
                        Tone::Danger,
                    ))
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(head_row)
                    .child(head)
                    .child(loading_rows(&t, "board", 4))
                    .into_any_element();
            }
        };
        if list.is_empty() {
            return panel(&t)
                .child(head_row)
                .child(empty_state(
                    &t,
                    "icons/play.svg",
                    "No entries yet",
                    if has_models {
                        "Pick a model and run it: every image, unchanged and under each condition of the plan."
                    } else {
                        "Train a model first, then run it here: every image, unchanged and under each condition of the plan."
                    },
                    Tone::Neutral,
                ))
                .into_any_element();
        }
        let rank = ranks(b, &list);
        let rows: Vec<AnyElement> = list
            .iter()
            .zip(rank)
            .map(|(e, rank)| self.leaderboard_row(&t, b, e, rank, cols, window, cx))
            .collect();
        let note = "Axes on the score's 0–100 scale. Partial entries ran a part of the benchmark or missed an axis; they rank after complete ones.";
        panel(&t)
            .overflow_hidden()
            .child(head_row)
            .child(head)
            .children(rows)
            .child(div().px(px(16.)).py(px(10.)).child(text(ty::SM, t.text_tertiary).child(note)))
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn leaderboard_row(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        rank: Option<usize>,
        cols: impl Fn(gpui::Stateful<gpui::Div>) -> gpui::Stateful<gpui::Div>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        // The entry column's width: what the number columns, their gaps and the row padding leave.
        let name_w =
            (self.content_width(window) - (24. + 64. + 5. * 68. + 116.) - 8. * 16. - 42.).clamp(120., 640.);
        let key = format!("bench-entry-{}", e.id);
        let done = e.status == BenchEntryStatus::Done;
        let stale = done && e.fingerprint != b.fingerprint;
        let busy = self.bench.busy.contains(&e.id);
        let confirming = self
            .bench
            .confirm_delete
            .as_ref()
            .is_some_and(|(x, i)| *x == b.id && i.as_deref() == Some(e.id.as_str()));
        let axis = |a: BenchAxis| {
            e.score.as_ref().and_then(|s| s.axes.iter().find(|x| x.axis == a)).and_then(|x| x.value)
        };
        let mut badges: Vec<(&str, Tone)> = Vec::new();
        if !done {
            badges.push((e.status.label(), entry_tone(e.status)));
        }
        if e.score.as_ref().is_some_and(|s| s.partial) {
            badges.push(("partial", Tone::Neutral));
        }
        if stale {
            badges.push(("earlier version", Tone::Warning));
        }
        let name_line = name_with_badges(&t, window, &e.name, &badges, name_w);
        // The second line: who it is, or how far it is, or why it failed.
        let detail: AnyElement = match e.status {
            BenchEntryStatus::Running | BenchEntryStatus::Queued | BenchEntryStatus::Stopping => {
                let at = e.progress.condition.as_ref().map_or_else(
                    || "unchanged images".to_string(),
                    |c| format!("{c} · severity {}", e.progress.severity.unwrap_or(1)),
                );
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(div().w(px(120.)).flex_none().child(progress(
                        &t,
                        entry_frac(e),
                        t.accent_fill(),
                        4.,
                    )))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .ellipsis()
                            .child(format!("{} · {at}", pct(entry_frac(e)))),
                    )
                    .into_any_element()
            }
            BenchEntryStatus::Failed => text(ty::SM, t.danger_text)
                .ellipsis()
                .child(e.error.clone().unwrap_or_else(|| "Failed".to_string()))
                .into_any_element(),
            _ => text(ty::SM, t.text_tertiary)
                .ellipsis()
                .child({
                    let mut s = contestant_line(e);
                    if let Some(sp) = &e.speed {
                        s += &format!(" · {:.1} ms/image", sp.ms_per_image);
                    }
                    s
                })
                .into_any_element(),
        };
        let score = e.score.as_ref().map_or_else(
            || text(ty::SM, t.text_tertiary).child("–"),
            |s| {
                text(ty::MD, t.text_primary)
                    .tnum()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(format!("{:.1}", s.total))
            },
        );
        // Actions: stop while it runs; delete (a second click confirms) once it has ended.
        let actions: AnyElement = if busy {
            text(ty::SM, t.text_tertiary).child("…").into_any_element()
        } else if confirming {
            let k = div()
                .flex()
                .gap(px(4.))
                .child(self.button(
                    &format!("{key}-keep"),
                    "Keep",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.bench.confirm_delete = None;
                        cx.notify();
                    }),
                ))
                .child(self.button(
                    &format!("{key}-del-ok"),
                    "Delete",
                    Btn::DangerCommit,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| this.confirm_bench_delete(cx)),
                ));
            k.into_any_element()
        } else if e.status.is_active() && e.status != BenchEntryStatus::Stopping {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.icon_button(
                &format!("{key}-stop"),
                "icons/stop.svg",
                "Stop this entry",
                window,
                cx,
                act(move |this, _, cx| this.stop_bench_entry(&bid, &eid, cx)),
            )
            .into_any_element()
        } else if e.status.is_active() {
            div().into_any_element()
        } else {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.icon_button(
                &format!("{key}-del"),
                "icons/trash-2.svg",
                "Delete this entry",
                window,
                cx,
                act(move |this, _, cx| {
                    this.bench.confirm_delete = Some((bid.clone(), Some(eid.clone())));
                    cx.notify();
                }),
            )
            .into_any_element()
        };
        let faded = !done || stale;
        let row = cols(div().id(SharedString::from(key.clone())))
            .h(px(52.))
            .border_b_1()
            .border_color(t.line_subtle)
            .when(done, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active)))
            .child(num_cell(
                24.,
                text(ty::SM, t.text_tertiary).tnum().child(rank.map_or("–".to_string(), |r| r.to_string())),
            ))
            .child(
                div().flex_1().min_w(px(180.)).flex().flex_col().gap(px(2.)).child(name_line).child(detail),
            )
            .child(num_cell(64., score))
            .children(BenchAxis::ALL.iter().map(|a| {
                let v = axis(*a);
                num_cell(
                    68.,
                    text(ty::SM, if faded || v.is_none() { t.text_tertiary } else { t.text_secondary })
                        .tnum()
                        .child(axis_text(v)),
                )
            }))
            .child(div().w(px(116.)).flex_none().flex().justify_end().child(actions));
        if done {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.bind(
                row,
                &key,
                0.,
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::BenchEntry(bid.clone(), eid.clone()), w, cx)),
            )
            .into_any_element()
        } else {
            row.into_any_element()
        }
    }

    /// The plan can still change: no entry has finished or is under way (docs/BENCH.md).
    fn plan_open(&self, b: &Bench) -> bool {
        !b.active
            && self.bench.entries.get(&b.id).and_then(|l| l.ready()).is_some_and(|l| {
                l.iter().all(|e| e.status != BenchEntryStatus::Done && !e.status.is_active())
            })
    }

    /// A 72 × 48 thumbnail of the benchmark's first image, unchanged or under a condition.
    fn condition_thumb(
        &mut self,
        t: &Theme,
        b: &Bench,
        condition: Option<(&str, u8)>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let first =
            self.bench.previews.get(&b.id).and_then(|l| l.ready()).and_then(|p| p.images.first()).cloned();
        let frame =
            div().w(px(72.)).h(px(48.)).flex_none().rounded(px(radius::SM)).overflow_hidden().bg(t.bg_inset);
        let Some(im) = first else { return frame.into_any_element() };
        let img = self.bench_image(&b.id, &im.id, condition, cx);
        frame
            .child(image_canvas(
                t,
                img,
                (im.width as f32, im.height as f32),
                Vec::new(),
                Fit::Cover,
                false,
                false,
            ))
            .into_any_element()
    }

    fn bench_plan(
        &mut self,
        t: &Theme,
        b: &Bench,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        self.ensure_previews(&b.id, cx);
        let open = self.plan_open(b);
        let design_here = self.bench.design.as_ref().is_some_and(|f| f.bench == b.id);
        // Design with an agent: while the plan can change, on servers that run agents.
        let design_btn = (open && self.agents_on() && !design_here).then(|| {
            let bid = b.id.clone();
            self.button(
                "bench-design",
                "Design with an agent",
                Btn::Secondary,
                Some("icons/sparkles.svg"),
                window,
                cx,
                act(move |this, w, cx| this.open_design(&bid, w, cx)),
            )
        });
        let lock = (!open).then(|| {
            div()
                .flex()
                .items_center()
                .gap(px(6.))
                .child(icon("icons/lock.svg", 12., t.text_tertiary))
                .child(text(ty::SM, t.text_tertiary).child("Fixed: entries have run on it"))
        });
        let header = panel_header(&t, "Plan")
            .child(div().flex().items_center().gap(px(8.)).children(lock).children(design_btn));
        let catalog = self.bench.catalog.ready().cloned();
        let cond =
            |id: &str| catalog.as_ref().and_then(|c| c.conditions.iter().find(|x| x.id == id).cloned());
        let total: f32 = b.plan.conditions.iter().map(|c| c.weight).sum::<f32>().max(1e-6);
        let who = b.plan.designed_by.as_ref().map(|d| {
            format!(
                "Designed by {}{}",
                agent_name(d.agent),
                d.model.as_ref().map_or(String::new(), |m| format!(" ({m})"))
            )
        });
        let mut intro = div()
            .px(px(16.))
            .py(px(12.))
            .flex()
            .flex_col()
            .gap(px(4.))
            .border_b_1()
            .border_color(t.line_subtle);
        intro = intro.child(
            text(ty::SM, t.text_secondary).child(match &who {
                Some(w) => {
                    format!("{w}. Each condition changes every image the same way for every contestant.")
                }
                None => {
                    "The default plan. Each condition changes every image the same way for every contestant."
                        .to_string()
                }
            }),
        );
        if let Some(r) = &b.plan.rationale {
            intro = intro.child(text(ty::SM, t.text_tertiary).child(r.clone()));
        }
        let design = design_here.then(|| self.design_panel(&t, b, window, cx));
        if b.plan.conditions.is_empty() {
            let plan = panel(&t).child(header).child(empty_state(
                &t,
                "icons/sliders-horizontal.svg",
                "Unchanged images only",
                "This plan runs no conditions, so no entry has a robustness axis.",
                Tone::Neutral,
            ));
            return div().flex().flex_col().gap(px(16.)).children(design).child(plan).into_any_element();
        }
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(72.)).flex_none().child(th(&t, "Preview")))
            .child(th(&t, "Condition").flex_1().min_w(px(220.)))
            .child(th(&t, "Group").w(px(84.)).flex_none())
            .child(th(&t, "Severities").w(px(150.)).flex_none())
            .child(num_cell(110., th(&t, "Share of robustness")));
        let row =
            |thumb: AnyElement, name: String, what: String, group: &str, sev: gpui::Div, share: String| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(16.))
                    .px(px(16.))
                    .py(px(8.))
                    .min_h(px(64.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(thumb)
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(220.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).child(name))
                            .child(text(ty::SM, t.text_tertiary).child(what)),
                    )
                    .child(text(ty::SM, t.text_secondary).w(px(84.)).flex_none().child(group.to_string()))
                    .child(sev.w(px(150.)).flex_none())
                    .child(num_cell(110., text(ty::SM, t.text_secondary).tnum().child(share)))
                    .into_any_element()
            };
        let mut rows = vec![{
            let thumb = self.condition_thumb(&t, b, None, cx);
            row(
                thumb,
                "Unchanged".into(),
                "The images as they were frozen: accuracy is measured on these.".into(),
                "",
                div(),
                String::new(),
            )
        }];
        for c in &b.plan.conditions {
            let info = cond(&c.condition);
            let name = info.as_ref().map_or(c.condition.clone(), |i| i.name.clone());
            let what =
                c.why.clone().or_else(|| info.as_ref().map(|i| i.description.clone())).unwrap_or_default();
            let group = info.as_ref().map_or("", |i| i.group.label());
            let shown = preview_severity(&c.severities);
            let thumb = self.condition_thumb(&t, b, Some((&c.condition, shown)), cx);
            let sev = div().flex().gap(px(4.)).children((1..=crate::api::BENCH_SEVERITIES).map(|s| {
                let on = c.severities.contains(&s);
                div()
                    .size(px(20.))
                    .rounded(px(radius::XS))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(if on { t.accent_muted } else { t.bg_inset })
                    .when(s == shown && on, |d| d.border_1().border_color(t.accent_line))
                    .child(
                        text(ty::SM, if on { t.accent_text } else { t.text_disabled })
                            .tnum()
                            .child(s.to_string()),
                    )
            }));
            rows.push(row(thumb, name, what, group, sev, pct(c.weight / total)));
        }
        let plan = panel(&t).overflow_hidden().child(header).child(intro).child(head).children(rows).child(
            div().px(px(16.)).py(px(10.)).child(
                text(ty::SM, t.text_tertiary)
                    .child("Previews show the first image at the outlined severity."),
            ),
        );
        div().flex().flex_col().gap(px(16.)).children(design).child(plan).into_any_element()
    }

    /// "Design with an agent": pick the agent, add a note, ask; then the proposal, to use or discard.
    fn design_panel(
        &mut self,
        t: &Theme,
        b: &Bench,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let agents = self.bench_agents();
        let Some(f) = self.bench.design.as_ref() else { return div().into_any_element() };
        let (agent, asking, applying, error, proposal) =
            (f.agent, f.asking, f.applying, f.error.clone(), f.proposal.clone());
        let note_input = f.note.clone();
        let note_fh = note_input.read(cx).focus_handle(cx);
        let close = self.icon_button(
            "bench-design-close",
            "icons/x.svg",
            "Close",
            window,
            cx,
            act(|this, _, cx| {
                this.bench.design = None;
                cx.notify();
            }),
        );
        let title = if proposal.is_some() { "Proposed plan" } else { "Design the plan with an agent" };
        let header = panel_header(&t, title).child(close);
        let mut body = div().p(px(16.)).flex().flex_col().gap(px(14.));
        if let Some(d) = &proposal {
            let catalog = self.bench.catalog.ready().cloned();
            let cond_name = |id: &str| {
                catalog
                    .as_ref()
                    .and_then(|x| x.conditions.iter().find(|i| i.id == id))
                    .map_or(id.to_string(), |i| i.name.clone())
            };
            let total: f32 = d.plan.conditions.iter().map(|c| c.weight).sum::<f32>().max(1e-6);
            let current: Vec<&str> = b.plan.conditions.iter().map(|c| c.condition.as_str()).collect();
            if let Some(r) = &d.plan.rationale {
                body = body.child(text(ty::SM, t.text_secondary).child(r.clone()));
            }
            let rows = d.plan.conditions.iter().map(|c| {
                let sev = c.severities.iter().map(u8::to_string).collect::<Vec<_>>().join(", ");
                let new = !current.contains(&c.condition.as_str());
                div()
                    .flex()
                    .items_start()
                    .gap(px(12.))
                    .py(px(6.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child(cond_name(&c.condition)),
                                    )
                                    .when(new, |d| d.child(badge(&t, "new", Tone::Accent))),
                            )
                            .children(c.why.clone().map(|w| text(ty::SM, t.text_tertiary).child(w))),
                    )
                    .child(
                        text(ty::SM, t.text_secondary)
                            .tnum()
                            .w(px(140.))
                            .flex_none()
                            .child(format!("severities {sev}")),
                    )
                    .child(num_cell(56., text(ty::SM, t.text_secondary).tnum().child(pct(c.weight / total))))
            });
            let dropped: Vec<String> = b
                .plan
                .conditions
                .iter()
                .filter(|c| !d.plan.conditions.iter().any(|x| x.condition == c.condition))
                .map(|c| cond_name(&c.condition))
                .collect();
            body = body.child(div().flex().flex_col().children(rows));
            if !dropped.is_empty() {
                body = body.child(
                    text(ty::SM, t.text_tertiary).child(format!("Leaves out: {}.", dropped.join(", "))),
                );
            }
            let cost = d.usage.cost_usd.map_or(String::new(), |c| format!(" · {}", fmt_cost(c)));
            body = body.child(
                text(ty::SM, t.text_tertiary)
                    .tnum()
                    .child(format!("{}{cost}", plural(d.usage.calls as usize, "call"))),
            );
            let use_btn = if applying {
                self.button_disabled(
                    "bench-design-use",
                    "Saving…",
                    Btn::Primary,
                    Some("icons/check.svg"),
                    "Saving the plan",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "bench-design-use",
                    "Use this plan",
                    Btn::Primary,
                    Some("icons/check.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.apply_design(cx)),
                )
            };
            let discard = self.button(
                "bench-design-discard",
                "Discard",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.bench.design = None;
                    cx.notify();
                }),
            );
            body = body.child(div().flex().gap(px(8.)).child(use_btn).child(discard));
        } else {
            let names: Vec<String> = agents.iter().map(|a| a.name.clone()).collect();
            let ids: Vec<LabelAgentId> = agents.iter().map(|a| a.id).collect();
            let sel = ids.iter().position(|a| *a == agent).unwrap_or(0);
            let picker = (!names.is_empty()).then(|| {
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                self.segmented(
                    "bench-design-agent",
                    &refs,
                    sel,
                    window,
                    cx,
                    std::rc::Rc::new(move |this, i, _, cx| {
                        if let (Some(f), Some(a)) = (this.bench.design.as_mut(), ids.get(i)) {
                            f.agent = *a;
                        }
                        cx.notify();
                    }),
                )
            });
            let note = div()
                .relative()
                .w(px(420.))
                .h(px(32.))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if note_fh.is_focused(window) { t.accent_line } else { t.line_control })
                .text_size(px(ty::SM.size))
                .when(self.ring(&note_fh, window), |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(note_input);
            body = body
                .child(text(ty::SM, t.text_secondary).child(
                    "One request, no tools: the agent reads the description, the classes and a few of the images, and answers with a plan from the catalog. You see it here before anything changes.",
                ))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_start()
                        .gap(px(16.))
                        .when_some(picker, |d, p| {
                            d.child(div().flex().flex_col().gap(px(6.)).child(caption(&t, "Agent")).child(p))
                        })
                        .child(div().flex().flex_col().gap(px(6.)).child(caption(&t, "Note")).child(note)),
                );
            let who =
                agents.iter().find(|a| a.id == agent).map_or("the agent".to_string(), |a| a.name.clone());
            let ask = if agents.is_empty() {
                self.button_disabled(
                    "bench-design-ask",
                    "Ask",
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    "Install Claude Code or Codex on the server's machine",
                    window,
                    cx,
                )
            } else if asking {
                self.button_disabled(
                    "bench-design-ask",
                    format!("Asking {who}…"),
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    "Waiting for the plan",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "bench-design-ask",
                    format!("Ask {who}"),
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.ask_design(cx)),
                )
            };
            body = body.child(div().flex().child(ask));
        }
        if let Some(e) = error {
            body = body.child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                    .child(text(ty::SM, t.danger_text).child(e.message)),
            );
        }
        panel(&t).child(header).child(body).into_any_element()
    }

    fn bench_loops(
        &mut self,
        t: &Theme,
        b: &Bench,
        loops: &Load<Vec<BenchLoop>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let form = self.bench.new_loop.is_some().then(|| self.new_loop_form(t, b, window, cx));
        let table = self.bench_loops_table(t, b, loops, window, cx);
        div().flex().flex_col().gap(px(16.)).children(form).child(table).into_any_element()
    }

    fn bench_loops_table(
        &mut self,
        t: &Theme,
        b: &Bench,
        loops: &Load<Vec<BenchLoop>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let new_btn = (self.bench.new_loop.is_none()).then(|| {
            self.button(
                "bench-loop-new",
                "New loop",
                Btn::Secondary,
                Some("icons/repeat.svg"),
                window,
                cx,
                act(|this, _, cx| this.open_new_loop(cx)),
            )
        });
        let header = panel_header(&t, "Self-improvement loops").child(
            div()
                .flex()
                .items_center()
                .gap(px(12.))
                .child(
                    text(ty::SM, t.text_tertiary)
                        .child("A model labels a pool, a student trains on those labels"),
                )
                .children(new_btn),
        );
        let list = match loops {
            Load::Ready(l) => l.clone(),
            Load::Failed(e) => {
                return panel(&t)
                    .child(header)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Loops didn't load",
                        &e.message,
                        Tone::Danger,
                    ))
                    .into_any_element();
            }
            _ => return panel(&t).child(header).child(loading_rows(&t, "loops", 3)).into_any_element(),
        };
        if list.is_empty() {
            return panel(&t)
                .child(header)
                .child(empty_state(
                    &t,
                    "icons/repeat.svg",
                    "No loops yet",
                    "A loop asks whether a model's own labels can train a better one: it labels a pool of images, a student trains on those labels alone, and this benchmark grades the student.",
                    Tone::Neutral,
                ))
                .into_any_element();
        }
        let now = self.store.api.now();
        let name_w = (self.content_width(window) - (220. + 84. + 116.) - 3. * 16. - 42.).clamp(120., 640.);
        let rows: Vec<AnyElement> = list
            .iter()
            .map(|l| {
                let key = format!("bench-loop-{}", l.id);
                let busy = self.bench.busy.contains(&l.id);
                let confirming = self
                    .bench
                    .confirm_delete
                    .as_ref()
                    .is_some_and(|(x, i)| *x == b.id && i.as_deref() == Some(l.id.as_str()));
                let total = l.config.generations.unwrap_or(1);
                let done_gens =
                    l.generations.iter().filter(|g| g.generation > 0 && g.score.is_some()).count();
                let first = l.generations.first().and_then(|g| g.map50_95);
                let last = l.generations.iter().rev().find(|g| g.generation > 0).and_then(|g| g.map50_95);
                let trend = match (first, last) {
                    (Some(a), Some(z)) => format!("mAP50-95 {} → {}", f3(a), f3(z)),
                    (Some(a), None) => format!("teacher mAP50-95 {}", f3(a)),
                    _ => "–".to_string(),
                };
                let state = match (l.status, l.phase) {
                    (BenchLoopStatus::Running, Some(p)) => format!(
                        "{} · generation {} of {total}",
                        p.label(),
                        (done_gens + 1).min(total as usize)
                    ),
                    (BenchLoopStatus::Failed, _) => l.error.clone().unwrap_or_else(|| "Failed".into()),
                    _ => format!("{done_gens} of {total} generations"),
                };
                let verdict = l.verdict.map(|v| {
                    let tone = match v {
                        BenchLoopVerdict::Improved => Tone::Success,
                        BenchLoopVerdict::Held => Tone::Neutral,
                        BenchLoopVerdict::Regressed => Tone::Danger,
                    };
                    (v.label(), tone)
                });
                let actions: AnyElement = if busy {
                    text(ty::SM, t.text_tertiary).child("…").into_any_element()
                } else if confirming {
                    div()
                        .flex()
                        .gap(px(4.))
                        .child(self.button(
                            &format!("{key}-keep"),
                            "Keep",
                            Btn::Secondary,
                            None,
                            window,
                            cx,
                            act(|this, _, cx| {
                                this.bench.confirm_delete = None;
                                cx.notify();
                            }),
                        ))
                        .child(self.button(
                            &format!("{key}-del-ok"),
                            "Delete",
                            Btn::DangerCommit,
                            None,
                            window,
                            cx,
                            act(|this, _, cx| this.confirm_bench_delete(cx)),
                        ))
                        .into_any_element()
                } else if l.status == BenchLoopStatus::Running {
                    let (bid, lid) = (b.id.clone(), l.id.clone());
                    self.icon_button(
                        &format!("{key}-stop"),
                        "icons/stop.svg",
                        "Stop this loop",
                        window,
                        cx,
                        act(move |this, _, cx| this.stop_bench_loop(&bid, &lid, cx)),
                    )
                    .into_any_element()
                } else if l.status.is_active() {
                    div().into_any_element()
                } else {
                    let (bid, lid) = (b.id.clone(), l.id.clone());
                    self.icon_button(
                        &format!("{key}-del"),
                        "icons/trash-2.svg",
                        "Delete this loop",
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.bench.confirm_delete = Some((bid.clone(), Some(lid.clone())));
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                };
                let at = time::parse(&l.created);
                div()
                    .flex()
                    .items_center()
                    .gap(px(16.))
                    .px(px(16.))
                    .h(px(56.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(200.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child({
                                let mut badges = vec![(l.status.label(), loop_tone(l.status))];
                                badges.extend(verdict);
                                name_with_badges(&t, window, &l.name, &badges, name_w)
                            })
                            .child(
                                text(
                                    ty::SM,
                                    if l.status == BenchLoopStatus::Failed {
                                        t.danger_text
                                    } else {
                                        t.text_tertiary
                                    },
                                )
                                .ellipsis()
                                .child(state),
                            ),
                    )
                    .child(text(ty::SM, t.text_secondary).tnum().w(px(220.)).flex_none().child(trend))
                    .child(num_cell(
                        84.,
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .child(at.map_or(String::new(), |c| time::ago(c, now))),
                    ))
                    .child(div().w(px(116.)).flex_none().flex().justify_end().child(actions))
                    .into_any_element()
            })
            .collect();
        panel(&t).overflow_hidden().child(header).children(rows).into_any_element()
    }

    /// The New loop form: the teacher, the pool it labels, how many generations and how long each
    /// student trains (its architecture and image size are the teacher's).
    fn new_loop_form(
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
        let (teacher, pool, generations, epochs, submitting, error) =
            (f.teacher.clone(), f.pool, f.generations, f.epochs, f.submitting, f.error.clone());
        let teacher_i = teacher.as_ref().and_then(|id| teachers.iter().position(|m| &m.id == id));
        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        let teacher_select = self.select_with(
            "bench-loop-teacher",
            teacher_i
                .and_then(|i| teachers.get(i))
                .map_or_else(|| "No model can teach".into(), |m| m.name.clone().into()),
            models
                .iter()
                .map(|m| {
                    (m.name.clone().into(), Some(format!("{} · {} px", m.arch.as_str(), m.imgsz).into()))
                })
                .collect(),
            teacher_i.unwrap_or(0),
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
        let pool_select = self.source_select("bench-loop-pool", &sources, pool, window, cx, |this, i, cx| {
            if let Some(f) = this.bench.new_loop.as_mut() {
                f.pool = i;
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
        let picked = sources.get(pool);
        let pool_images = picked.map(|p| p.images(crate::api::Split::Train));
        // The benchmark's own images must stay out of training: the server leaves them out of the pool.
        let overlaps = match (picked, &b.source) {
            (Some(SourcePick::Dataset { id, .. }), BenchSource::Dataset { dataset_id, split }) => {
                id == dataset_id && split.unwrap_or(crate::api::Split::Val) == crate::api::Split::Train
            }
            (Some(SourcePick::Job { id, .. }), BenchSource::LabelJob { job_id }) => id == job_id,
            _ => false,
        };
        let gens = LOOP_GENERATIONS[generations];
        let note = match (teacher_i.and_then(|i| teachers.get(i)), pool_images) {
            (None, _) => {
                "No model can teach yet: a teacher is trained on this benchmark's classes, in its order."
                    .to_string()
            }
            (_, None) => "Add a dataset first: the teacher labels its images.".to_string(),
            (_, Some(0)) => "That pool has no images: pick another.".to_string(),
            (Some(m), Some(n)) => format!(
                "{} labels {} {} by itself; a {} student trains {} epochs on those labels alone and runs here. {}{}",
                m.name,
                int(n),
                if matches!(picked, Some(SourcePick::Job { .. })) {
                    "labeled images of the job"
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
            act(|this, _, cx| {
                this.bench.new_loop = None;
                cx.notify();
            }),
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
                    .child(div().flex().gap(px(8.)).child(start).child(cancel)),
            )
            .into_any_element()
    }

    /// An entry's images: the truth (dashed) and its detections at its best-F1 confidence, worst first
    /// or in the benchmark's order, a page at a time.
    fn entry_images_panel(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        best_conf: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        self.ensure_entry_images(&b.id, &e.id, cx);
        let sort = self.bench.image_sort;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let sorts = [ImageSort::Worst, ImageSort::Order];
        let seg = self.segmented(
            "entry-img-sort",
            &["Worst first", "In order"],
            sorts.iter().position(|s| *s == sort).unwrap_or(0),
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| {
                this.bench.image_sort = sorts[i.min(1)];
                this.ensure_entry_images(&bid, &eid, cx);
                cx.notify();
            }),
        );
        let header = panel_header(&t, "Images").child(seg);
        let state = self
            .bench
            .entry_images
            .get(&(b.id.clone(), e.id.clone()))
            .map(|x| (x.page.clone(), x.more, x.more_error.clone()));
        let (page, loading_more, more_error) = match state {
            Some((Load::Ready(p), more, err)) => (p, more, err),
            Some((Load::Failed(err), _, _)) => {
                return panel(&t)
                    .child(header)
                    .child(div().p(px(16.)).child(text(ty::SM, t.danger_text).child(err.message)))
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(header)
                    .child(div().p(px(16.)).flex().flex_wrap().gap(px(8.)).children(
                        (0..8).map(|i| div().w(px(152.)).h(px(152.)).child(skeleton(&t, format!("eimg{i}")))),
                    ))
                    .into_any_element();
            }
        };
        let tiles: Vec<AnyElement> = page
            .images
            .iter()
            .map(|im| {
                let img = self.bench_image(&b.id, &im.image.id, None, cx);
                let mut boxes: Vec<DrawBox> = im
                    .image
                    .boxes
                    .iter()
                    .map(|x| DrawBox { truth: true, ..DrawBox::new(x.class, x.x1, x.y1, x.x2, x.y2) })
                    .collect();
                boxes.extend(im.detections.iter().map(|d| DrawBox {
                    label: b.classes.get(d.class as usize).cloned().map(Into::into),
                    score: e.scored.then_some(d.score),
                    ..DrawBox::new(d.class, d.x1, d.y1, d.x2, d.y2)
                }));
                let dims = (im.image.width as f32, im.image.height as f32);
                let ok = im.missed == 0 && im.wrong == 0;
                let caption_text = if ok {
                    "all right".to_string()
                } else {
                    [(im.missed, "missed"), (im.wrong, "wrong")]
                        .iter()
                        .filter(|(n, _)| *n > 0)
                        .map(|(n, w)| format!("{n} {w}"))
                        .collect::<Vec<_>>()
                        .join(" · ")
                };
                div()
                    .id(SharedString::from(format!("eimg-{}", im.image.id)))
                    .w(px(152.))
                    .flex_none()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .tooltip(crate::controls::tip(im.image.file.clone(), None))
                    .child(
                        div()
                            .w(px(152.))
                            .h(px(152.))
                            .rounded(px(radius::MD))
                            .overflow_hidden()
                            .bg(t.bg_inset)
                            .child(image_canvas(&t, img, dims, boxes, Fit::Contain, true, false)),
                    )
                    .child(
                        text(ty::SM, if ok { t.text_tertiary } else { t.text_secondary })
                            .tnum()
                            .child(caption_text),
                    )
                    .into_any_element()
            })
            .collect();
        let more = (page.images.len() as u64) < page.total;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let footer = div()
            .px(px(16.))
            .pb(px(12.))
            .flex()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                "{} of {} images · dashed: the truth · solid: its detections{}",
                int(page.images.len() as u64),
                int(page.total),
                if e.scored {
                    format!(" at confidence {} and up", fmt_conf(best_conf))
                } else {
                    String::new()
                },
            )))
            .when_some(more_error, |d, err| d.child(text(ty::SM, t.danger_text).child(err.message)))
            .when(more, |d| {
                d.child(if loading_more {
                    self.button_disabled(
                        "entry-img-more",
                        "Loading…",
                        Btn::Secondary,
                        Some("icons/chevron-down.svg"),
                        "Loading the next images",
                        window,
                        cx,
                    )
                } else {
                    self.button(
                        "entry-img-more",
                        "Show more",
                        Btn::Secondary,
                        Some("icons/chevron-down.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.more_entry_images(&bid, &eid, cx)),
                    )
                })
            });
        panel(&t)
            .child(header)
            .child(div().p(px(16.)).flex().flex_wrap().gap(px(12.)).children(tiles))
            .child(footer)
            .into_any_element()
    }

    // ---------- an entry ----------

    pub fn bench_entry_view(
        &mut self,
        bench: &str,
        entry: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        self.ensure_bench(bench, cx);
        let e = self
            .bench
            .entries
            .get(bench)
            .and_then(|l| l.ready())
            .and_then(|l| l.iter().find(|x| x.id == entry))
            .cloned();
        let b = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == bench)).cloned();
        let (Some(e), Some(b)) = (e, b) else {
            let header = page_header(&t, "Entry", "", vec![]);
            let failed =
                self.bench.entries.get(bench).and_then(|l| l.error()).or(self.bench.list.error()).cloned();
            let body = if let Some(err) = failed {
                self.error_panel(&err, "bench-entry-retry", window, cx)
            } else if self.bench.list.is_loading()
                || self.bench.entries.get(bench).is_none_or(|l| l.is_loading())
            {
                panel(&t).child(loading_rows(&t, "entry", 5)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Entry not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body, cx);
        };
        let now = self.store.api.now();
        let mut title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).ellipsis().child(e.name.clone()))
            .child(badge(&t, e.status.label(), entry_tone(e.status)));
        if e.score.as_ref().is_some_and(|s| s.partial) {
            title = title.child(badge(&t, "partial", Tone::Neutral));
        }
        let finished = e
            .finished
            .as_deref()
            .and_then(time::parse)
            .map_or(String::new(), |c| format!(" · finished {}", time::ago(c, now)));
        let subtitle = format!(
            "{} · on {} · {} images · {} of {} conditions{finished}",
            contestant_line(&e),
            b.name,
            int(e.images),
            e.conditions_run,
            b.plan.conditions.len(),
        );
        let bid = b.id.clone();
        let back = self
            .button(
                "entry-board",
                "Leaderboard",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Bench(bid.clone()), w, cx)),
            )
            .into_any_element();
        let header = detail_header(&t, title, subtitle, vec![back]);
        if e.status != BenchEntryStatus::Done {
            let msg = match e.status {
                BenchEntryStatus::Failed => e.error.clone().unwrap_or_else(|| "It failed.".into()),
                BenchEntryStatus::Stopped => {
                    "It was stopped before it finished, so nothing was scored.".into()
                }
                _ => format!("{} done. Its result shows here when it finishes.", pct(entry_frac(&e))),
            };
            let body = panel(&t)
                .child(empty_state(
                    &t,
                    "icons/clock.svg",
                    e.status.label(),
                    &msg,
                    if e.status == BenchEntryStatus::Failed { Tone::Danger } else { Tone::Neutral },
                ))
                .into_any_element();
            return self.page_frame(&t, header, body, cx);
        }
        self.ensure_bench_result(bench, entry, cx);
        let result =
            self.bench.results.get(&(bench.to_string(), entry.to_string())).cloned().unwrap_or_default();
        let body = match result {
            Load::Ready(r) => {
                let images = self.entry_images_panel(&t, &b, &e, r.accuracy.best_conf, window, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(self.entry_result(&t, &b, &r))
                    .child(images)
                    .into_any_element()
            }
            Load::Failed(err) => panel(&t)
                .child(empty_state(
                    &t,
                    "icons/circle-alert.svg",
                    "The result didn't load",
                    &err.message,
                    Tone::Danger,
                ))
                .into_any_element(),
            _ => panel(&t).child(loading_rows(&t, "result", 6)).into_any_element(),
        };
        self.page_frame(&t, header, body, cx)
    }

    fn entry_result(&self, t: &Theme, b: &Bench, r: &BenchResult) -> AnyElement {
        let t = *t;
        let e = &r.entry;
        let score = e.score.as_ref();
        let axis = |a: BenchAxis| score.and_then(|s| s.axes.iter().find(|x| x.axis == a)).copied();
        // The score and its five axes, each with its weight.
        let kpi = |content: gpui::Div, basis: f32| {
            div().flex_grow().flex_basis(relative(basis)).min_w_0().px(px(16.)).py(px(14.)).child(content)
        };
        let mut kpis = panel(&t).flex().flex_wrap().child(kpi(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .child(caption(&t, "Score"))
                .child(metric(
                    &t,
                    score.map_or("–".to_string(), |s| format!("{:.1}", s.total)),
                    "/ 100",
                    ty::X3XL,
                    ty::SM,
                ))
                .child(text(ty::SM, t.text_tertiary).child(if score.is_some_and(|s| s.partial) {
                    "Partial: over what was measured"
                } else {
                    "All five axes measured"
                })),
            0.2,
        ));
        for a in BenchAxis::ALL {
            let s = axis(a);
            let v = s.and_then(|s| s.value);
            let detail = match a {
                BenchAxis::Accuracy => format!("mAP50-95 {}", f3(r.accuracy.map50_95)),
                BenchAxis::Robustness => r.robustness.as_ref().map_or("conditions not run".into(), |x| {
                    format!("keeps {} of its accuracy", pct(x.retention))
                }),
                BenchAxis::SelfLabeling => match r.self_labeling.readiness {
                    BenchReadiness::Ready => "ready to self-label".into(),
                    BenchReadiness::Partial => "partly ready".into(),
                    BenchReadiness::NotReady => "not ready".into(),
                },
                BenchAxis::Calibration => {
                    r.calibration.as_ref().map_or("no confidences".into(), |c| format!("ECE {}", f3(c.ece)))
                }
                BenchAxis::Localization => r
                    .localization
                    .mean_iou
                    .map_or("nothing correct".into(), |m| format!("mean IoU {}", f3(m))),
            };
            kpis = kpis.child(kpi(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .child(caption(
                        &t,
                        &format!("{} · {}", axis_name(a), pct(s.map_or(a.weight(), |s| s.weight))),
                    ))
                    .child(div().h(px(ty::X3XL.line)).flex().items_end().child(
                        metric(&t, axis_text(v), "", ty::XL, ty::SM).pb(px(baseline_lift(ty::X3XL, ty::XL))),
                    ))
                    .child(text(ty::SM, t.text_tertiary).ellipsis().child(detail)),
                0.12,
            ));
        }

        // Accuracy and self-labeling, side by side.
        let acc = &r.accuracy;
        let opt = |v: Option<f32>| v.map_or("–".to_string(), f3);
        let accuracy = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(
                panel_header(&t, "Accuracy")
                    .child(text(ty::SM, t.text_tertiary).child("on the unchanged images")),
            )
            .child(
                div()
                    .px(px(16.))
                    .py(px(8.))
                    .child(kv(
                        &t,
                        "mAP50-95",
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{}  ({} – {}, 95%)",
                            f3(acc.map50_95),
                            f3(acc.ci95[0]),
                            f3(acc.ci95[1])
                        )),
                    ))
                    .child(kv(
                        &t,
                        "mAP50 · mAP75",
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{} · {}",
                            f3(acc.map50),
                            f3(acc.map75)
                        )),
                    ))
                    .child(kv(
                        &t,
                        &format!("P · R · F1 at {}", fmt_conf(acc.best_conf)),
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{} · {} · {}",
                            f3(acc.precision),
                            f3(acc.recall),
                            f3(acc.f1)
                        )),
                    ))
                    .child(kv(
                        &t,
                        "Small · medium · large",
                        text(ty::SM, t.text_primary).tnum().child(format!(
                            "{} · {} · {}",
                            opt(acc.map_small),
                            opt(acc.map_medium),
                            opt(acc.map_large)
                        )),
                    ))
                    .when_some(e.speed.as_ref(), |d, s| {
                        d.child(kv(
                            &t,
                            "Speed",
                            text(ty::SM, t.text_primary).tnum().child(format!(
                                "{:.1} ms/image on {}",
                                s.ms_per_image,
                                device_label(&s.device)
                            )),
                        ))
                    }),
            );
        let sl = &r.self_labeling;
        let tone = match sl.readiness {
            BenchReadiness::Ready => Tone::Success,
            BenchReadiness::Partial => Tone::Warning,
            BenchReadiness::NotReady => Tone::Neutral,
        };
        let self_label = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(panel_header(&t, "Self-labeling").child(badge(&t, sl.readiness.label(), tone)))
            .child(
                div()
                    .px(px(16.))
                    .py(px(8.))
                    .child(kv(
                        &t,
                        "Threshold",
                        text(ty::SM, t.text_primary).tnum().child(
                            sl.threshold.map_or("never reaches the target".into(), |c| {
                                format!("confidence {}", fmt_conf(c))
                            }),
                        ),
                    ))
                    .child(kv(
                        &t,
                        &format!("Precision (target {})", pct(sl.target_precision)),
                        text(ty::SM, t.text_primary).tnum().child(pct(sl.precision)),
                    ))
                    .child(kv(
                        &t,
                        "Objects it labels",
                        text(ty::SM, t.text_primary).tnum().child(pct(sl.recall)),
                    ))
                    .child(kv(
                        &t,
                        "Labels per image",
                        text(ty::SM, t.text_primary).tnum().child(format!("{:.1}", sl.labels_per_image)),
                    ))
                    .child(kv(
                        &t,
                        "Mean IoU of right labels",
                        text(ty::SM, t.text_primary).tnum().child(opt(sl.mean_iou)),
                    )),
            );
        let pair = div().flex().flex_wrap().gap(px(16.)).child(accuracy).child(self_label);

        // Robustness: what each condition keeps.
        let catalog = self.bench.catalog.ready();
        let cond_name = |id: &str| {
            catalog
                .and_then(|c| c.conditions.iter().find(|x| x.id == id))
                .map_or(id.to_string(), |c| c.name.clone())
        };
        let robustness = r.robustness.as_ref().map(|rob| {
            let rows = rob.conditions.iter().map(|c| {
                let sev = c
                    .points
                    .iter()
                    .map(|p| format!("S{} {}", p.severity, f3(p.map50_95)))
                    .collect::<Vec<_>>()
                    .join(" · ");
                div()
                    .flex()
                    .flex_col()
                    .py(px(4.))
                    .child(bar_row(
                        &t,
                        cond_name(&c.condition),
                        c.retention,
                        pct(c.retention),
                        if c.retention >= 0.8 {
                            t.success_solid
                        } else if c.retention >= 0.5 {
                            t.warning_solid
                        } else {
                            t.danger_solid
                        },
                    ))
                    .child(
                        div().pl(px(152.)).child(text(ty::SM, t.text_tertiary).tnum().ellipsis().child(sev)),
                    )
            });
            panel(&t)
                .child(panel_header(&t, "Robustness").child(text(ty::SM, t.text_tertiary).child(format!(
                    "keeps {} of its accuracy under the plan's conditions",
                    pct(rob.retention)
                ))))
                .child(div().px(px(16.)).py(px(8.)).children(rows))
        });

        // Per class.
        let classes = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(panel_header(&t, "Per class").child(text(ty::SM, t.text_tertiary).child("AP50-95")))
            .child(div().px(px(16.)).py(px(8.)).children(acc.per_class.iter().map(|c| {
                let name =
                    b.classes.get(c.class as usize).cloned().unwrap_or_else(|| format!("class {}", c.class));
                div().flex().items_center().gap(px(8.)).child(class_swatch(&t, c.class, 10.)).child(
                    div().flex_1().min_w_0().child(bar_row(
                        &t,
                        format!("{name} · {}", int(c.truths)),
                        c.map50_95,
                        f3(c.map50_95),
                        t.class_on(c.class, crate::theme::Surface::Panel),
                    )),
                )
            })))
            .when(!e.missing_classes.is_empty(), |d| {
                d.child(div().px(px(16.)).pb(px(10.)).child(text(ty::SM, t.warning_text).child(format!(
                    "The model doesn't know {}: AP 0 on those.",
                    e.missing_classes.join(", ")
                ))))
            });

        // Calibration and localization.
        let calib = panel(&t).flex_1().min_w(px(320.)).child(panel_header(&t, "Calibration").child(
            text(ty::SM, t.text_tertiary).child(
                r.calibration.as_ref().map_or("no confidences to check".to_string(), |c| {
                    format!("ECE {} over {} detections", f3(c.ece), int(c.detections))
                }),
            ),
        ));
        let calib = match &r.calibration {
            Some(c) => calib.child(div().px(px(16.)).py(px(8.)).children(c.bins.iter().filter(|b| b.count > 0).map(|bin| {
                let p = bin.precision.unwrap_or(0.);
                bar_row(
                    &t,
                    format!("{:.1}–{:.1} · {}", bin.lo, bin.hi, int(bin.count)),
                    p,
                    pct(p),
                    if bin.confidence.is_some_and(|c| (c - p).abs() <= 0.1) { t.success_solid } else { t.warning_solid },
                )
            })))
            .child(div().px(px(16.)).pb(px(10.)).child(text(ty::SM, t.text_tertiary).child("Share right per confidence band: a trustworthy model is right about as often as it is sure."))),
            None => calib.child(div().px(px(16.)).py(px(12.)).child(text(ty::SM, t.text_tertiary).child("Without confidences (or none from 0.1 up), calibration is left out of the score."))),
        };
        let loc = &r.localization;
        let peak = loc.histogram.iter().copied().max().unwrap_or(0).max(1);
        let hist = div().h(px(64.)).flex().items_end().gap(px(3.)).children(loc.histogram.iter().map(|n| {
            div()
                .flex_1()
                .h(relative(*n as f32 / peak as f32))
                .min_h(px(1.))
                .rounded_t(px(2.))
                .bg(t.accent_fill())
        }));
        let localization = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(panel_header(&t, "Localization").child(text(ty::SM, t.text_tertiary).child(
                loc.mean_iou.map_or("nothing correct".to_string(), |m| {
                    format!("mean IoU {} over {} matches", f3(m), int(loc.matches))
                }),
            )))
            .child(
                div().px(px(16.)).py(px(12.)).flex().flex_col().gap(px(6.)).child(hist).child(
                    div()
                        .flex()
                        .justify_between()
                        .child(text(ty::SM, t.text_tertiary).child("IoU 0.5"))
                        .child(text(ty::SM, t.text_tertiary).child("1.0")),
                ),
            );

        div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(kpis)
            .child(pair)
            .children(robustness)
            .child(
                div().flex().flex_wrap().gap(px(16.)).child(classes).child(
                    div()
                        .flex_1()
                        .min_w(px(320.))
                        .flex()
                        .flex_col()
                        .gap(px(16.))
                        .child(calib)
                        .child(localization),
                ),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_read_on_the_score_scale() {
        assert_eq!(axis_text(Some(0.834)), "83");
        assert_eq!(axis_text(None), "–");
        assert_eq!(axis_text(Some(1.2)), "100");
        assert_eq!(pct(0.5), "50%");
    }

    #[test]
    fn routes_round_trip() {
        assert_eq!(Route::parse("benchmarks"), Some(Route::Benches));
        assert_eq!(Route::parse("bench:bn_1"), Some(Route::Bench("bn_1".into())));
        assert_eq!(
            Route::parse("bench-entry:bn_1/be_2"),
            Some(Route::BenchEntry("bn_1".into(), "be_2".into()))
        );
        assert_eq!(Route::parse("bench-entry:bn_1"), None);
        assert_eq!(Route::BenchEntry("b".into(), "e".into()).section(), crate::app::Section::Benchmarks);
    }
}
