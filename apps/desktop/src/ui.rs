//! Stateless building blocks: type, surfaces, badges, metrics, progress, placeholders.
//!
//! Interactive controls that need focus handles live on the app (`controls.rs`); everything
//! here is a pure function of the theme and its arguments.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use gpui::prelude::FluentBuilder;
use gpui::{
    Animation, AnimationExt, Bounds, Div, FontFeatures, FontWeight, Hsla, IntoElement, ParentElement, Pixels,
    ScrollHandle, SharedString, Styled, Svg, div, fill, point, px, relative, size, svg,
};

use crate::api::{CloudOffer, CloudPhase, Label, Run, RunCloud, RunExt, RunStatus, TransferProgress};
use crate::theme::{Theme, TypeStep, radius, ty};

static REDUCE_MOTION: AtomicBool = AtomicBool::new(false);

/// Settings › Reduce motion: stops pulses and shimmers and makes transitions instant.
/// Whether the OS asks for less motion (Windows: Settings › Accessibility › Animation effects off).
pub fn os_prefers_reduced_motion() -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            SPI_GETCLIENTAREAANIMATION, SystemParametersInfoW,
        };
        let mut on: i32 = 1;
        // SAFETY: SPI_GETCLIENTAREAANIMATION writes one BOOL into the pointed-to value.
        let ok =
            unsafe { SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, (&mut on as *mut i32).cast(), 0) };
        ok != 0 && on == 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

pub fn set_reduce_motion(on: bool) {
    REDUCE_MOTION.store(on, Ordering::Relaxed);
}

pub fn reduce_motion() -> bool {
    REDUCE_MOTION.load(Ordering::Relaxed)
}

pub fn weight(w: u16) -> FontWeight {
    FontWeight(w as f32)
}

pub trait StyledExt: Styled + Sized {
    /// Tabular numerals, so live counters and table columns don't jitter.
    fn tnum(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_features =
            Some(FontFeatures(Arc::new(vec![("tnum".into(), 1)])));
        self
    }

    fn step(self, s: TypeStep) -> Self {
        self.text_size(px(s.size)).line_height(px(s.line)).font_weight(weight(s.weight))
    }
}

impl<T: Styled> StyledExt for T {}

/// Sans text at a type step.
pub fn text(s: TypeStep, color: Hsla) -> Div {
    div().font_family(crate::theme::SANS).step(s).text_color(color)
}

/// A group heading (palette groups, form sections). Sentence case: GPUI has no letter-spacing, and
/// untracked capitals look cramped next to sentence-case titles.
pub fn overline(t: &Theme, label: &str) -> Div {
    text(ty::SM, t.text_tertiary).font_weight(FontWeight::MEDIUM).child(label.to_string())
}

/// A metric or field label: `xs` medium, tertiary, in its own case.
pub fn caption(t: &Theme, label: &str) -> Div {
    text(ty::SM, t.text_tertiary).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label.to_string())
}

/// Centred text (`.text_left()` to align it left) where `backticked` spans render as inline code:
/// one text element, so words space and wrap as body text does, with code in Geist Mono on a
/// chip fill (a no-break space either side for its padding).
pub fn rich(t: &Theme, s: TypeStep, color: Hsla, content: &str) -> Div {
    let (sans, mono) = (gpui::font(crate::theme::SANS), gpui::font(crate::theme::MONO));
    let mut text = String::new();
    let mut runs = Vec::new();
    for (i, part) in content.split('`').enumerate() {
        if part.is_empty() {
            continue;
        }
        let code = i % 2 == 1;
        // No-break spaces keep a code span together where they can, but GPUI's wrapper can still
        // break at one, so a command that must stay whole belongs on its own chip line.
        let seg =
            if code { format!("\u{a0}{}\u{a0}", part.replace(' ', "\u{a0}")) } else { part.to_string() };
        runs.push(gpui::TextRun {
            len: seg.len(),
            font: if code { mono.clone() } else { sans.clone() },
            color: if code { t.text_primary } else { color },
            background_color: code.then_some(t.bg_active),
            underline: None,
            strikethrough: None,
        });
        text.push_str(&seg);
    }
    div()
        .text_size(px(s.size))
        .line_height(px(s.line))
        .text_color(color)
        .text_center()
        .child(gpui::StyledText::new(text).with_runs(runs))
}

pub fn icon(path: &'static str, size: f32, color: Hsla) -> Svg {
    svg().path(path).size(px(size)).flex_none().text_color(color)
}

/// Raised content surface with a hairline edge.
pub fn panel(t: &Theme) -> Div {
    div().bg(t.bg_panel).border_1().border_color(t.line_default).rounded(px(radius::LG))
}

/// Panel header row: overline title on the left, optional extras on the right.
/// A chart's anatomy, everywhere one shows: a header (title, caption) on a rule, a 28 px legend
/// row whose items start under the title, then the plot inset 12 px. The run page frames it as a
/// card; the Overview hero sets it under its KPIs. `short` windows trade 8 px of header and 4 of
/// legend for plot.
pub fn chart_section(
    t: &Theme,
    title: &str,
    sub: &str,
    legend: impl IntoElement,
    plot: impl IntoElement,
    short: bool,
) -> Div {
    div()
        .w_full()
        .flex()
        .flex_col()
        .min_w_0()
        .child(
            // A caption's power of ten ("× 10⁻³") in the same raised run as every value's.
            panel_header(t, title)
                .child(sci(ty::SM, t.text_tertiary, sub.to_string()).font_weight(FontWeight::MEDIUM))
                .when(short, |d| d.h(px(32.))),
        )
        .child(
            div()
                .h(px(if short { 24. } else { 28. }))
                // Legends start under the title (16 px); the loss toggles pull their 6 px hover
                // pads back out so their swatches land there too.
                .px(px(16.))
                .pt(px(if short { 4. } else { 6. }))
                .flex()
                .items_center()
                .child(legend),
        )
        .child(div().px(px(12.)).pb(px(6.)).child(plot))
}

pub fn panel_header(t: &Theme, title: &str) -> Div {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap(px(8.))
        .h(px(40.))
        .px(px(16.))
        .border_b_1()
        .border_color(t.line_subtle)
        .child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).child(title.to_string()))
}

pub fn vline(t: &Theme) -> Div {
    div().w(px(1.)).h_full().flex_none().bg(t.line_subtle)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger,
    Info,
}

impl Tone {
    pub fn colors(self, t: &Theme) -> (Hsla, Hsla, Hsla) {
        // (background, text, dot)
        match self {
            Tone::Neutral => (t.bg_active, t.text_secondary, t.text_tertiary),
            Tone::Accent => (t.accent_muted, t.accent_text, t.accent_fill()),
            Tone::Success => (t.success_muted, t.success_text, t.success_solid),
            Tone::Warning => (t.warning_muted, t.warning_text, t.warning_solid),
            Tone::Danger => (t.danger_muted, t.danger_text, t.danger_solid),
            Tone::Info => (t.info_muted, t.info_text, t.info_solid),
        }
    }
}

pub fn status_tone(s: RunStatus) -> Tone {
    match s {
        RunStatus::Running => Tone::Accent,
        RunStatus::Completed => Tone::Success,
        RunStatus::Failed => Tone::Danger,
        RunStatus::Stopping => Tone::Warning,
        // Waiting is not a status hue (design/README): neutral grey.
        RunStatus::Queued => Tone::Neutral,
        RunStatus::Stopped => Tone::Neutral,
    }
}

/// A small status dot; `live` pulses a halo around it.
pub fn dot(color: Hsla, size: f32, live: bool, id: impl Into<SharedString>) -> impl IntoElement {
    let id: SharedString = id.into();
    div()
        .relative()
        .size(px(size))
        .flex_none()
        .child(div().absolute().size_full().rounded_full().bg(color))
        .when(live && !reduce_motion(), |d| {
            d.child(div().absolute().size_full().rounded_full().bg(color).with_animation(
                gpui::ElementId::Name(format!("pulse-{id}").into()),
                Animation::new(Duration::from_millis(1600)).repeat(),
                move |d, p| {
                    let grow = size * 1.6 * p;
                    d.top(px(-grow / 2.)).left(px(-grow / 2.)).size(px(size + grow)).opacity(0.45 * (1.0 - p))
                },
            ))
        })
}

/// Tinted label chip, 20 px tall.
/// A waiting run's place, as every list writes it: "Next to start", "2nd in queue", or (a server
/// without queue positions) "Waiting to start".
pub fn queue_text(position: Option<u32>) -> String {
    match position {
        Some(1) => "Next to start".to_string(),
        Some(n) => {
            let suffix = match (n % 10, n % 100) {
                (1, x) if x != 11 => "st",
                (2, x) if x != 12 => "nd",
                (3, x) if x != 13 => "rd",
                _ => "th",
            };
            format!("{n}{suffix} in queue")
        }
        None => "Waiting to start".to_string(),
    }
}

/// A confidence threshold as every surface writes it: "0.31", or "0.314" when the third decimal
/// carries information (the API's best-F1 grid is 0.001).
pub fn fmt_conf(c: f32) -> String {
    let three = format!("{c:.3}");
    if three.ends_with('0') { format!("{c:.2}") } else { three }
}

/// One inference's total latency as every readout writes it: "6.4 ms", or "6.4 ms simulated" when the
/// built-in demo simulated it.
pub fn latency_text(ms: f64, simulated: bool) -> String {
    if simulated { format!("{ms:.1} ms simulated") } else { format!("{ms:.1} ms") }
}

/// What a run that hasn't started training waits on, as every list writes it: a cloud run's phase
/// ("Uploading data · 25%"), else its place in the training queue.
pub fn waiting_text(r: &Run) -> String {
    match r.cloud_pending() {
        Some(c) => phase_text(c),
        None => queue_text(r.queue_position),
    }
}

// ---------- cloud compute ----------

/// A price per hour, one way everywhere: "$0.69/h"; "—" for compute that isn't billed.
pub fn fmt_price(per_hour: Option<f64>) -> String {
    match per_hour {
        Some(p) => format!("${p:.2}/h"),
        None => "—".to_string(),
    }
}

/// Money spent or to be spent: "$0.42", "<$0.01" under a cent, whole dollars from $100 ("$1,280").
pub fn fmt_cost(usd: f64) -> String {
    let usd = usd.max(0.);
    if usd > 0. && usd < 0.01 {
        "<$0.01".to_string()
    } else if usd >= 100. {
        format!("${}", int(usd.round() as u64))
    } else {
        format!("${usd:.2}")
    }
}

/// A cost range, "$0.12–$0.93"; one figure when both ends read the same.
pub fn fmt_cost_range(lo: f64, hi: f64) -> String {
    let (lo, hi) = (fmt_cost(lo.min(hi)), fmt_cost(lo.max(hi)));
    if lo == hi { lo } else { format!("{lo}–{hi}") }
}

/// Hours as typed in Stop after: "12", "0.5", "0.25" (no trailing zeros).
pub fn hours_value(h: f32) -> String {
    let s = format!("{h:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Hours as a limit reads: "12 h", "0.5 h".
pub fn fmt_hours(h: f32) -> String {
    format!("{} h", hours_value(h))
}

/// An offer in one line: "RTX 4090 · 24.0 GiB · Secure Cloud" (memory and qualifier when known).
pub fn offer_line(o: &CloudOffer) -> String {
    let mut parts =
        vec![if o.gpu_count > 1 { format!("{}× {}", o.gpu_count, o.gpu) } else { o.gpu.clone() }];
    if o.vram_bytes > 0 {
        let (v, u) = bytes(o.vram_bytes);
        parts.push(format!("{v} {u}"));
    }
    parts.extend(o.detail.clone());
    parts.join(" · ")
}

/// A transfer's progress: "1.1 of 4.5 GiB" (one unit when both sides share it), or what has moved
/// when the total is unknown.
pub fn fmt_transfer(p: &TransferProgress) -> String {
    let (done, done_u) = bytes(p.done_bytes);
    if p.total_bytes == 0 {
        return format!("{done} {done_u}");
    }
    let (total, total_u) = bytes(p.total_bytes);
    if done_u == total_u {
        format!("{done} of {total} {total_u}")
    } else {
        format!("{done} {done_u} of {total} {total_u}")
    }
}

/// The fraction of a transfer done, when its total is known.
pub fn transfer_frac(p: &TransferProgress) -> Option<f32> {
    (p.total_bytes > 0).then(|| (p.done_bytes as f64 / p.total_bytes as f64).clamp(0., 1.) as f32)
}

/// A cloud run's phase in a few words: "Training", "Uploading data · 25%" while a transfer with a
/// known size runs.
pub fn phase_text(c: &RunCloud) -> String {
    match c.progress.as_ref().and_then(transfer_frac) {
        Some(f) => format!("{} · {:.0}%", c.phase.label(), f * 100.),
        None => c.phase.label().to_string(),
    }
}

/// What a cloud run that hasn't started training is doing, as a sentence-case title:
/// "Provisioning an RTX 4090 at RunPod", "Booting the instance", "Uploading data · 25%".
pub fn cloud_wait_title(c: &RunCloud, provider: &str) -> String {
    match c.phase {
        CloudPhase::Provisioning => format!("Provisioning {} at {provider}", with_article(&c.offer.gpu)),
        CloudPhase::Booting => "Booting the instance".to_string(),
        _ => phase_text(c),
    }
}

/// A GPU name with its article, as the name is read aloud: "an RTX 4090", "an A100", "a T4" (GPU
/// names start with initials, read letter by letter).
pub fn with_article(name: &str) -> String {
    let an = name.chars().next().is_some_and(|c| "AEFHILMNORSXaeiou".contains(c));
    format!("{} {name}", if an { "an" } else { "a" })
}

/// An action inside a line of text ("Show them ›", "Use 0.31 (best F1) ›"): secondary ink and a
/// trailing chevron, so every inline action reads as one at rest, not only on hover.
pub fn inline_link(t: &Theme, label: impl Into<SharedString>) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(2.))
        .child(text(ty::SM, t.text_secondary).child(label.into()))
        .child(icon("icons/chevron-right.svg", 12., t.text_secondary))
}

/// How a model's weights are stored, from its own record (`precision`, `fused`): "f16, fused",
/// "f32", or `None` when the server doesn't say. Every surface that names it uses this.
pub fn model_storage(m: &crate::api::Model) -> Option<String> {
    let p = m.precision?.as_str();
    Some(if m.fused == Some(true) { format!("{p}, fused") } else { p.to_string() })
}

/// Truncation that ends in "…" wherever the element sits. GPUI 0.2.2's `truncate()` measures
/// no-wrap text once at its unbounded width and keeps that, so a flex-sized text clips without
/// an ellipsis; wrapping text clamped to one line re-measures at each width and truncates there.
/// Every truncated text uses this (a test keeps `.truncate()` out).
pub trait Ellipsis: Styled + Sized {
    fn ellipsis(self) -> Self {
        self.overflow_hidden().whitespace_normal().line_clamp(1).text_ellipsis()
    }
}

impl<T: Styled> Ellipsis for T {}

/// Two values as one figure, "0.861 / 0.792": both at `size`, the slash a step down in tertiary
/// ink. The run and model pages' Precision / Recall both use it.
pub fn metric_pair(t: &Theme, size: TypeStep, a: String, b: String) -> Div {
    div()
        .flex()
        .items_baseline()
        .gap(px(4.))
        .child(text(size, t.text_primary).tnum().child(a))
        .child(text(ty::LG, t.text_tertiary).child("/"))
        .child(text(size, t.text_primary).tnum().child(b))
}

/// The data a model was evaluated on, one way everywhere: "shapes-hard · val · 800 images" (the
/// count when known).
pub fn split_label(dataset: &str, split: &str, images: Option<u64>) -> String {
    match images {
        Some(n) => format!("{dataset} · {split} · {} images", int(n)),
        None => format!("{dataset} · {split}"),
    }
}

/// Where precision and recall were taken, as both pages caption them: the best-F1 confidence when
/// known, the match rule, and the split ("best F1 · conf 0.31 · IoU 0.50 · val"). A tile whose
/// chart states the match rule beside it leaves it out (`iou: false`), so the line fits.
pub fn pr_caption(conf: Option<f32>, split: &str, iou: bool) -> String {
    let mut s = "best F1".to_string();
    if let Some(c) = conf {
        s += &format!(" · conf {}", fmt_conf(c));
    }
    if iou {
        s += " · IoU 0.50";
    }
    // A surface that names the split elsewhere (the model page's headline) leaves it out.
    if !split.is_empty() {
        s += &format!(" · {split}");
    }
    s
}

/// One metadata line: `lead` (an architecture, say), an optional tag right after it, then the rest
/// as " · "-joined text. The line owns its separators (4 px either side of each dot) and only the
/// rest gives way, with a real ellipsis, so every meta line spaces and truncates alike.
pub fn meta_row(
    size: TypeStep,
    color: Hsla,
    lead: impl Into<SharedString>,
    tag: Option<gpui::Stateful<Div>>,
    rest: &[String],
) -> Div {
    let rest: Vec<&str> = rest.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
    // The lead and its tag hold their width; everything after them is one group that gives way
    // as a whole, ending in "…", so no part can push the row past its bounds.
    div()
        .w_full()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .child(text(size, color).flex_none().whitespace_nowrap().child(lead.into()))
        .children(tag)
        // One separator rule: every dot is the font's " · " inside the group, the first included
        // (a flex gap beside a lone "·" spaced it wider than the rest).
        .when(!rest.is_empty(), |d| {
            d.child(
                text(size, color)
                    .flex_1()
                    .min_w_0()
                    .ml(px(-4.))
                    .ellipsis()
                    .child(format!(" · {}", rest.join(" · "))),
            )
        })
}

/// The compact end-to-end tag in tables (the title rows carry the full badge and its help).
pub fn e2e_tag(t: &Theme, id: impl Into<SharedString>) -> gpui::Stateful<Div> {
    use gpui::{InteractiveElement, StatefulInteractiveElement};
    div()
        .id(id.into())
        .tooltip(crate::controls::tip(crate::app::FoloApp::E2E_HELP, None))
        .flex_none()
        .px(px(4.))
        .rounded(px(crate::theme::radius::XS))
        .bg(t.bg_active)
        .child(text(ty::SM, t.text_secondary).child("e2e"))
}

pub fn badge(t: &Theme, label: impl Into<SharedString>, tone: Tone) -> Div {
    let (bg, fg, _) = tone.colors(t);
    text(ty::SM, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .px(px(6.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(5.))
        .rounded(px(radius::XS))
        .bg(bg)
        .whitespace_nowrap()
        .child(label.into())
}

/// Status chip for a run: a dot (pulsing while running) and the status word.
/// Status for dense rows: a dot and a word, no tinted pill.
pub fn status_dot(t: &Theme, s: RunStatus, id: &str) -> Div {
    let (_, fg, dotc) = status_tone(s).colors(t);
    div()
        .flex()
        .items_center()
        .gap(px(6.))
        .child(if s == RunStatus::Queued {
            // Waiting, not halted: a ring where a stopped run has a filled dot.
            div().size(px(6.)).flex_none().rounded_full().border_1().border_color(dotc).into_any_element()
        } else {
            dot(dotc, 6., s == RunStatus::Running, format!("sd-{id}")).into_any_element()
        })
        .child(
            text(ty::SM, if s == RunStatus::Running { t.accent_text } else { fg })
                .whitespace_nowrap()
                .child(s.label()),
        )
}

pub fn status_badge(t: &Theme, s: RunStatus, id: &str) -> Div {
    let tone = status_tone(s);
    let (_, _, dotc) = tone.colors(t);
    let (bg, fg, _) = tone.colors(t);
    text(ty::SM, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .pl(px(6.))
        .pr(px(7.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(6.))
        .rounded(px(radius::XS))
        .bg(bg)
        .whitespace_nowrap()
        .child(dot(dotc, 6., s == RunStatus::Running, format!("badge-{id}")))
        .child(s.label())
}

/// Keycap for shortcut hints.
pub fn kbd(t: &Theme, key: impl Into<SharedString>) -> Div {
    text(ty::SM, t.text_secondary)
        .font_weight(FontWeight::MEDIUM)
        .h(px(18.))
        .min_w(px(18.))
        .px(px(4.))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .rounded(px(radius::SM))
        .bg(t.bg_active)
        .border_1()
        .border_color(t.line_subtle)
        .child(key.into())
}

/// A shortcut as keycaps ("Ctrl", "K").
pub fn shortcut(t: &Theme, keys: &[&str]) -> Div {
    div().flex().items_center().gap(px(3.)).children(keys.iter().map(|k| kbd(t, k.to_string())))
}

/// A number with its unit one step smaller in tertiary ("0.612", "ms").
pub fn metric(
    t: &Theme,
    value: impl Into<SharedString>,
    unit: &str,
    s: TypeStep,
    unit_step: TypeStep,
) -> Div {
    // GPUI's flex baseline doesn't align mixed sizes, so bottom-align and lift the unit by the
    // difference between the two lines' baselines. A space's width apart ("9.42 M", as text
    // writes a number and its unit everywhere).
    div()
        .flex()
        .items_end()
        .gap(px(4.))
        .whitespace_nowrap()
        .child(text(s, t.text_primary).tnum().child(value.into()))
        .when(!unit.is_empty(), |d| {
            d.child(
                text(unit_step, t.text_tertiary)
                    .tnum()
                    .pb(px(baseline_lift(s, unit_step)))
                    .child(unit.to_string()),
            )
        })
}

/// Labelled metric tile body: overline label, value, optional detail line.
/// A stat at a given value size (no unit).
pub fn stat_sized(
    t: &Theme,
    label: &str,
    value: impl Into<SharedString>,
    size: TypeStep,
    detail: Option<Div>,
) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(6.))
        .min_w_0()
        .child(caption(t, label))
        .child(metric(t, value, "", size, ty::SM))
        .children(detail)
}

/// A labelled stat with its value at `step` (X2XL; compact strips on short windows use XL).
pub fn stat_at(
    t: &Theme,
    label: &str,
    value: impl Into<SharedString>,
    unit: &str,
    detail: Option<Div>,
    step: TypeStep,
) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(if step.size < ty::X2XL.size { 4. } else { 6. }))
        .min_w_0()
        .child(caption(t, label))
        .child(metric(t, value, unit, step, ty::SM))
        .children(detail)
}

/// A KPI tile's first detail line, at the caption line's height, for a graphic (a progress or
/// split bar) that stands in for text, so its tile's later lines stay on the strip's grid.
pub fn detail_line(content: impl IntoElement) -> Div {
    div().h(px(ty::SM.line)).w_full().flex().items_center().child(content)
}

/// A duration as a metric: numbers at value size, units ("h", "m", "s") one step smaller.
/// How far above the bottom of its line box a smaller step's baseline must move to meet a larger
/// step's baseline (both bottom-aligned): baseline ≈ half the leading plus 0.2 em of descent.
pub fn baseline_lift(big: TypeStep, small: TypeStep) -> f32 {
    let from_bottom = |s: TypeStep| (s.line - s.size) / 2. + 0.2 * s.size;
    (from_bottom(big) - from_bottom(small)).max(0.)
}

pub fn duration_metric(t: &Theme, secs: Option<f64>, s: TypeStep, unit_step: TypeStep) -> Div {
    let Some(secs) = secs else { return text(s, t.text_primary).child("–") };
    let total = secs.max(0.0).round() as i64;
    let parts: Vec<(String, &str)> = if total < 60 {
        vec![(total.to_string(), "s")]
    } else if total < 3600 {
        vec![((total / 60).to_string(), "m"), (format!("{:02}", total % 60), "s")]
    } else {
        vec![((total / 3600).to_string(), "h"), (format!("{:02}", total / 60 % 60), "m")]
    };
    // "14m 55s": each unit hugs its number; the pair is one reading.
    div().flex().items_end().gap(px(5.)).whitespace_nowrap().children(parts.into_iter().map(|(v, u)| {
        div()
            .flex()
            .items_end()
            .gap(px(1.))
            .child(text(s, t.text_primary).tnum().child(v))
            .child(text(unit_step, t.text_tertiary).pb(px(baseline_lift(s, unit_step))).child(u))
    }))
}

/// A filesystem path as the platform writes it (backslashes on Windows).
pub fn display_path(p: &str) -> String {
    // URLs and pseudo paths (`mock://shapes`) keep their slashes.
    if cfg!(windows) && !p.contains("://") { p.replace('/', "\\") } else { p.to_string() }
}

#[cfg(test)]
mod path_tests {
    use super::display_path;

    #[test]
    fn urls_keep_slashes() {
        assert_eq!(display_path("mock://traffic-cams"), "mock://traffic-cams");
        if cfg!(windows) {
            assert_eq!(display_path("D:/data/shapes"), r"D:\data\shapes");
        }
    }
}

/// A thin progress track with a filled part.
pub fn progress(t: &Theme, frac: f32, color: Hsla, height: f32) -> Div {
    div()
        .w_full()
        .h(px(height))
        .rounded(px(height / 2.))
        .bg(t.bg_active)
        .overflow_hidden()
        .child(div().h_full().w(relative(frac.clamp(0.0, 1.0))).rounded(px(height / 2.)).bg(color))
}

/// A placeholder block that shimmers while data loads.
pub fn skeleton(t: &Theme, id: impl Into<SharedString>) -> gpui::AnyElement {
    let base = t.bg_active;
    let id: SharedString = id.into();
    let block = div().size_full().rounded(px(radius::SM)).bg(base);
    if reduce_motion() {
        return block.into_any_element();
    }
    block
        .with_animation(
            gpui::ElementId::Name(format!("skel-{id}").into()),
            Animation::new(Duration::from_millis(1400)).repeat(),
            move |d, p| {
                let w = (p * std::f32::consts::TAU).sin() * 0.5 + 0.5;
                d.opacity(0.55 + 0.45 * w)
            },
        )
        .into_any_element()
}

/// Centered empty / error message with an icon, a title, a line of help and optional actions.
pub fn empty_state(t: &Theme, icon_path: &'static str, title: &str, body: &str, tone: Tone) -> Div {
    let (bg, _, dotc) = tone.colors(t);
    div()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap(px(12.))
        .py(px(40.))
        .px(px(24.))
        .child(
            div()
                .size(px(40.))
                .rounded(px(radius::LG))
                .bg(if tone == Tone::Neutral { t.bg_raised } else { bg })
                .border_1()
                .border_color(t.line_subtle)
                .flex()
                .items_center()
                .justify_center()
                .child(icon(icon_path, 18., if tone == Tone::Neutral { t.text_secondary } else { dotc })),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .items_center()
                .gap(px(4.))
                .child(text(ty::MD, t.text_primary).font_weight(FontWeight::MEDIUM).child(title.to_string()))
                .child(rich(t, ty::SM, t.text_secondary, body).max_w(px(400.))),
        )
}

/// One legend entry with its value: swatch, name, number (tabular).
/// A 10 px line sample matching a series stroke: solid, dashed (two dashes) or dotted (three dots).
pub fn line_swatch(color: Hsla, dashed: bool, dotted: bool, weight: f32) -> Div {
    if dotted {
        div()
            .w(px(10.))
            .flex()
            .gap(px(2.))
            .children((0..3).map(move |_| div().w(px(2.)).h(px(2.)).rounded_full().bg(color)))
    } else if dashed {
        div().w(px(10.)).flex().gap(px(2.)).children((0..2).map(move |_| div().w(px(4.)).h(px(2.)).bg(color)))
    } else {
        div().w(px(10.)).h(px(weight)).rounded_full().bg(color)
    }
}

pub fn legend_value(t: &Theme, name: &str, color: Hsla, dashed: bool, value: Option<String>) -> Div {
    let swatch = line_swatch(color, dashed, false, 2.);
    div()
        .flex()
        .items_center()
        .gap(px(5.))
        .child(swatch)
        .child(text(ty::SM, t.text_secondary).whitespace_nowrap().child(name.to_string()))
        .when_some(value, |d, v| d.child(sci(ty::SM, t.text_primary, v)))
}

/// A class swatch: a small square in the class colour.
pub fn swatch(color: Hsla, size: f32) -> Div {
    div().size(px(size)).flex_none().rounded(px(2.)).bg(color)
}

/// A class swatch: the hue its boxes have on the image, so the key matches the photo. Its 1 px
/// edge is `chart.swatch_edge` in dark; in light, the class's own panel shade (3:1 on white), so
/// the swatch also matches that class's bars and curves beside it.
pub fn class_swatch(t: &Theme, class: u32, size: f32) -> Div {
    use crate::theme::Surface;
    let edge = if t.is_dark() { t.chart_swatch_edge } else { t.class_on(class, Surface::Panel) };
    swatch(t.class_on(class, Surface::Photo), size).border_1().border_color(edge)
}

/// A row of key/value pairs for detail panes.
pub fn kv(t: &Theme, key: &str, value: impl IntoElement) -> Div {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap(px(12.))
        .min_h(px(28.))
        .child(text(ty::SM, t.text_tertiary).flex_none().child(key.to_string()))
        .child(div().min_w_0().flex().justify_end().child(value))
}

// ---------- number formatting ----------

/// 4210 → "4,210".
pub fn int(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// 5_400_000 → "5.1 MiB".
/// Whether `s` set at `size` px in Geist may be wider than `width` px: a per-glyph estimate that
/// errs wide (capitals and w / m count as wide), so a name that might be cut gets its full-name tip.
/// A table's spare width, shared by weight: each column takes `weight` shares of `spare` on top
/// of its base, up to its cap. What the caps leave stays where the table keeps its gap (before the
/// numbers), so wide windows fill the columns first rather than open one dead band.
pub fn share_spare<const N: usize>(spare: f32, cols: [(f32, f32, f32); N]) -> [f32; N] {
    let weights: f32 = cols.iter().map(|c| c.1).sum();
    let share = if weights > 0. { spare.max(0.) / weights } else { 0. };
    // Whole pixels: a fractional column width can leave text measured at one width and laid out
    // at another.
    cols.map(|(base, weight, cap)| (base + share * weight).min(cap.max(base)).floor())
}

pub fn may_overflow(s: &str, size: f32, width: f32) -> bool {
    let em: f32 = s
        .chars()
        .map(|c| match c {
            'W' | 'M' | 'm' | 'w' => 0.9,
            c if c.is_uppercase() => 0.7,
            'i' | 'l' | 'j' | 'I' | '.' | ',' | ':' | ';' | '\'' | '|' | '!' => 0.3,
            c if c.is_ascii() => 0.58,
            // CJK and other wide scripts.
            _ => 1.0,
        })
        .sum();
    em * size > width
}

pub fn bytes(n: u64) -> (String, &'static str) {
    // Binary units, as Studio shows them by default (a 16 GiB card reads 16.0, not 17.2).
    const K: f64 = 1024.;
    let f = n as f64;
    if f >= K * K * K {
        (format!("{:.1}", f / (K * K * K)), "GiB")
    } else if f >= K * K {
        (format!("{:.1}", f / (K * K)), "MiB")
    } else if f >= K {
        (format!("{:.0}", f / K), "KiB")
    } else {
        (n.to_string(), "B")
    }
}

/// 2_630_000 → "2.63 M".
pub fn count_si(n: u64) -> (String, &'static str) {
    let f = n as f64;
    if f >= 1e9 {
        (format!("{:.2}", f / 1e9), "B")
    } else if f >= 1e6 {
        (format!("{:.2}", f / 1e6), "M")
    } else if f >= 1e3 {
        (format!("{:.1}", f / 1e3), "K")
    } else {
        (n.to_string(), "")
    }
}

/// A learning rate, one way everywhere: 0.01 and up as typed ("0.01", "0.0125"); below that in
/// its power of ten ("1.11×10⁻³"), as the learning-rate chart's legend writes it.
pub fn fmt_lr(v: f64) -> String {
    // Seven significant digits first: a rate that was an `f32` (0.01 is 0.0099999998 widened)
    // reads as the value it was typed as.
    let v = format!("{v:.6e}").parse::<f64>().unwrap_or(v);
    if v != 0. && v.abs() < 0.01 {
        return crate::charts::in_pow(v, v.abs().log10().floor() as i32);
    }
    let s = format!("{v:.6}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Text that may hold a power of ten ("lr0 1.11×10⁻³"): the exponent drawn as an 11 px raised
/// run ("−3") rather than the font's superscript glyphs, which render at 7–8 px. Text without
/// one is plain `text`.
pub fn sci(size: TypeStep, color: Hsla, s: String) -> Div {
    const SUP: &str = "⁻⁰¹²³⁴⁵⁶⁷⁸⁹";
    // "1.11×10⁻³" in values, "× 10⁻³" in a chart's caption.
    let found = ["×10", "× 10"].iter().find_map(|p| s.find(p).map(|at| at + p.len()));
    let Some(head_end) = found else { return text(size, color).tnum().whitespace_nowrap().child(s) };
    let exp_len: usize = s[head_end..].chars().take_while(|c| SUP.contains(*c)).map(char::len_utf8).sum();
    if exp_len == 0 {
        return text(size, color).tnum().whitespace_nowrap().child(s);
    }
    let exp: String = s[head_end..head_end + exp_len]
        .chars()
        .map(|c| match c {
            '⁻' => '−',
            c => char::from_digit(SUP.chars().skip(1).position(|d| d == c).unwrap_or(0) as u32, 10)
                .unwrap_or('0'),
        })
        .collect();
    div()
        .flex()
        // Tops aligned and the exponent's line tight: it sits raised by about a third of the size.
        .items_start()
        .whitespace_nowrap()
        .child(text(size, color).tnum().child(s[..head_end].to_string()))
        // small: an exponent beside its value, 11 px (the superscript glyphs it replaces are smaller).
        .child(text(ty::XS, color).line_height(px(ty::XS.size)).tnum().child(exp))
        .when(head_end + exp_len < s.len(), |d| {
            d.child(text(size, color).tnum().child(s[head_end + exp_len..].to_string()))
        })
}

/// An augmentation strength (mosaic, mixup, flip): two decimals, as New run's sliders show it.
pub fn fmt_aug(v: f32) -> String {
    format!("{v:.2}")
}

/// The scale augmentation's range: "±0.50", as New run's slider shows it.
pub fn fmt_scale(v: f32) -> String {
    format!("±{v:.2}")
}

pub fn f3(v: f32) -> String {
    format!("{v:.3}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_goes_through_ellipsis() {
        // GPUI's `truncate()` clips flex-sized text without an ellipsis; `Ellipsis::ellipsis` doesn't.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src];
        let mut bad = Vec::new();
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs")
                    && p.file_name().is_some_and(|n| n != "ui.rs")
                    && std::fs::read_to_string(&p).unwrap().contains(".truncate()")
                {
                    bad.push(p.file_name().unwrap().to_string_lossy().to_string());
                }
            }
        }
        assert!(bad.is_empty(), ".truncate() in {bad:?}: use .ellipsis()");
    }

    #[test]
    fn confidences_keep_a_meaningful_third_decimal() {
        assert_eq!(fmt_conf(0.31), "0.31");
        assert_eq!(fmt_conf(0.314), "0.314");
        assert_eq!(fmt_conf(0.024), "0.024");
        assert_eq!(fmt_conf(0.35), "0.35");
    }

    #[test]
    fn queue_places_read_as_ordinals() {
        assert_eq!(queue_text(Some(1)), "Next to start");
        assert_eq!(queue_text(Some(2)), "2nd in queue");
        assert_eq!(queue_text(Some(3)), "3rd in queue");
        assert_eq!(queue_text(Some(11)), "11th in queue");
        assert_eq!(queue_text(Some(22)), "22nd in queue");
        assert_eq!(queue_text(None), "Waiting to start");
    }

    #[test]
    fn overflow_is_by_width() {
        // The inspector's 70 px class column at 12 px: short names fit, eight W's don't.
        assert!(!may_overflow("hexagon", 12., 70.));
        assert!(!may_overflow("triangle", 12., 70.));
        assert!(may_overflow("WWWWWWWW", 12., 70.));
        assert!(may_overflow("construction vehicle", 12., 70.));
    }

    #[test]
    fn formats() {
        assert_eq!(int(4210), "4,210");
        assert_eq!(int(999), "999");
        assert_eq!(int(1_000_000), "1,000,000");
        assert_eq!(bytes(5_400_000), ("5.1".into(), "MiB"));
        assert_eq!(bytes(17_171_480_576), ("16.0".into(), "GiB"));
        assert_eq!(count_si(2_630_000), ("2.63".into(), "M"));
    }
}

/// Where a scroll container's thumb sits, from its handle (None when nothing overflows).
#[derive(Clone, Copy)]
pub struct ThumbGeo {
    pub top: Pixels,
    pub right: Pixels,
    pub h: Pixels,
    pub thumb: Pixels,
    pub y: Pixels,
    pub max: Pixels,
}

impl ThumbGeo {
    pub fn of(handle: &ScrollHandle) -> Option<Self> {
        let b = handle.bounds();
        let max = handle.max_offset().height;
        let h = b.size.height;
        if max <= px(0.5) || h <= px(8.) {
            return None;
        }
        let thumb = (h * (h / (h + max))).max(px(28.)).min(h);
        let off = (-handle.offset().y).clamp(px(0.), max);
        Some(Self { top: b.top(), right: b.right(), h, thumb, y: b.top() + (h - thumb) * (off / max), max })
    }
}

/// Paint the thumb (GPUI draws none), so overflow is visible.
pub fn paint_thumb(t: &Theme, handle: &ScrollHandle, hot: bool, window: &mut gpui::Window) {
    let Some(g) = ThumbGeo::of(handle) else { return };
    // An overlay thumb: shown while dragged, while the pointer is over its area, and for a moment
    // after the content scrolls; otherwise the content has the space to itself. Movement is
    // recorded on every paint, whatever else shows the thumb.
    let scrolled = recently_scrolled(handle);
    let over = handle.bounds().contains(&window.mouse_position());
    if !(hot || over || scrolled) {
        return;
    }
    // Shown only by a recent scroll: keep painting until that lapses, so it hides on time.
    if scrolled && !hot && !over {
        window.request_animation_frame();
    }
    // 3:1 against the surface in both themes (45 % read about 1.9:1).
    let color = crate::theme::alpha(t.text_tertiary, if hot { 1.0 } else { 0.8 });
    window.paint_quad(
        fill(Bounds::new(point(g.right - px(6.), g.y + px(2.)), size(px(4.), g.thumb - px(4.))), color)
            .corner_radii(px(2.)),
    );
}

/// A scroll area's origin, its last vertical offset, and when that last changed.
type ScrollSeen = ((f32, f32), f32, std::time::Instant);

thread_local! {
    static SCROLLED: std::cell::RefCell<Vec<ScrollSeen>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Edge fades over a scroll area where more content lies past it (thumbs being hidden at
/// rest, the cue that there's more).
pub fn scroll_edges(
    handle: &ScrollHandle,
    bg: Hsla,
    inset: f32,
    radius: f32,
) -> impl Iterator<Item = gpui::Canvas<()>> {
    // Read at paint time, after this frame's layout set the scroll extent: a fade decided while
    // building the frame used the last frame's extent, so a window's first frames (and a still
    // capture) showed no fade over content below the fold.
    let h = handle.clone();
    let painter = gpui::canvas(
        |_, _, _| (),
        move |b, _, window, _| {
            let y = h.offset().y;
            let above = y < px(-1.);
            let below = h.max_offset().height + y > px(1.);
            // 24 px: the gradient holds its full colour for the outer 6 px (its first stop at
            // 25%), so a row cut at the edge leaves no sliver of glyphs, then ramps out. It
            // paints its own rounded corners.
            let (inset, r) = (px(inset), px(radius));
            let w = b.size.width - inset * 2.;
            let fade = |top: bool| {
                let at = if top { b.top() + inset } else { b.bottom() - inset - px(24.) };
                let corners = if top {
                    gpui::Corners { top_left: r, top_right: r, ..Default::default() }
                } else {
                    gpui::Corners { bottom_left: r, bottom_right: r, ..Default::default() }
                };
                gpui::quad(
                    Bounds::new(point(b.left() + inset, at), size(w, px(24.))),
                    corners,
                    gpui::linear_gradient(
                        if top { 180. } else { 0. },
                        gpui::linear_color_stop(bg, 0.25),
                        gpui::linear_color_stop(crate::theme::alpha(bg, 0.), 1.),
                    ),
                    px(0.),
                    gpui::transparent_black(),
                    gpui::BorderStyle::default(),
                )
            };
            if above {
                window.paint_quad(fade(true));
            }
            if below {
                window.paint_quad(fade(false));
            }
        },
    );
    std::iter::once(painter.absolute().top_0().left_0().size_full())
}

/// Whether `handle`'s content moved in the last 800 ms.
fn recently_scrolled(handle: &ScrollHandle) -> bool {
    let b = handle.bounds();
    let key = (f32::from(b.origin.x), f32::from(b.origin.y));
    let y = f32::from(handle.offset().y);
    let now = std::time::Instant::now();
    SCROLLED.with(|s| {
        let mut s = s.borrow_mut();
        match s.iter_mut().find(|(k, _, _)| *k == key) {
            // A jump back to the top is a page change resetting its scroll, not the user scrolling.
            Some(e) if (e.1 - y).abs() > 0.5 && y.abs() < 0.5 => {
                *e = (key, y, now - std::time::Duration::from_secs(1));
                false
            }
            Some(e) if (e.1 - y).abs() > 0.5 => {
                *e = (key, y, now);
                true
            }
            Some(e) => now.duration_since(e.2) < std::time::Duration::from_millis(800),
            None => {
                // First sight: not a scroll.
                s.push((key, y, now - std::time::Duration::from_secs(1)));
                if s.len() > 64 {
                    s.remove(0);
                }
                false
            }
        }
    })
}

/// A run's device as shown: "RTX 4080 SUPER · Vulkan" from "NVIDIA GeForce RTX 4080 SUPER (vulkan)".
pub fn device_label(device: &str) -> String {
    match device.split_once(" (") {
        Some((name, rest)) => {
            let api = rest.trim_end_matches(')');
            let api = match api {
                "vulkan" => "Vulkan",
                "metal" => "Metal",
                "dx12" => "DirectX 12",
                "cuda" => "CUDA",
                other => other,
            };
            format!("{} · {api}", short_gpu(name))
        }
        None => short_gpu(device),
    }
}

/// A GPU name as people say it: vendor marketing prefixes dropped ("NVIDIA GeForce RTX 4080" → "RTX 4080").
pub fn short_gpu(name: &str) -> String {
    let mut n = name.trim();
    for p in ["NVIDIA GeForce ", "NVIDIA ", "AMD ", "Intel(R) "] {
        n = n.strip_prefix(p).unwrap_or(n);
    }
    n.to_string()
}

/// Fade an overlay in over 160 ms when it first appears (instant under Reduce motion).
pub fn appear<E: IntoElement + Styled + 'static>(el: E, id: &str) -> gpui::AnyElement {
    if reduce_motion() {
        return el.into_any_element();
    }
    el.with_animation(
        SharedString::from(id.to_string()),
        Animation::new(Duration::from_millis(160)).with_easing(gpui::ease_out_quint()),
        |d, p| d.opacity(p),
    )
    .into_any_element()
}

/// The 2 px accent bar on the selected row of a box list (design/README).
pub fn select_bar(t: &Theme) -> Div {
    div().absolute().left_0().top(px(4.)).bottom(px(4.)).w(px(2.)).rounded_full().bg(t.accent_fill())
}

/// "1 dataset", "3 datasets".
pub fn plural(n: usize, noun: &str) -> String {
    if n == 1 { format!("1 {noun}") } else { format!("{n} {noun}s") }
}

#[cfg(test)]
mod cloud_format_tests {
    use super::*;
    use crate::api::CloudAvailability;

    fn offer(gpu: &str, vram: u64, detail: Option<&str>) -> CloudOffer {
        CloudOffer {
            id: "o".into(),
            gpu: gpu.into(),
            gpu_count: 1,
            vram_bytes: vram,
            price_per_hour: Some(0.69),
            region: None,
            detail: detail.map(Into::into),
            availability: CloudAvailability::High,
        }
    }

    #[test]
    fn money_reads_one_way() {
        assert_eq!(fmt_price(Some(0.69)), "$0.69/h");
        assert_eq!(fmt_price(Some(2.5)), "$2.50/h");
        assert_eq!(fmt_price(None), "—");
        assert_eq!(fmt_cost(0.42), "$0.42");
        assert_eq!(fmt_cost(0.0), "$0.00");
        assert_eq!(fmt_cost(0.004), "<$0.01");
        assert_eq!(fmt_cost(99.994), "$99.99");
        assert_eq!(fmt_cost(1280.4), "$1,280");
        assert_eq!(fmt_cost(-1.0), "$0.00");
        assert_eq!(fmt_cost_range(0.12, 0.93), "$0.12–$0.93");
        assert_eq!(fmt_cost_range(0.93, 0.12), "$0.12–$0.93");
        assert_eq!(fmt_cost_range(0.001, 0.004), "<$0.01");
    }

    #[test]
    fn hours_and_offers() {
        assert_eq!(
            (fmt_hours(12.0), fmt_hours(0.5), fmt_hours(0.25)),
            ("12 h".into(), "0.5 h".into(), "0.25 h".into())
        );
        assert_eq!(hours_value(720.0), "720");
        assert_eq!(
            offer_line(&offer("RTX 4090", 25_769_803_776, Some("Secure Cloud"))),
            "RTX 4090 · 24.0 GiB · Secure Cloud"
        );
        assert_eq!(offer_line(&offer("RTX 4080 SUPER", 0, None)), "RTX 4080 SUPER");
        assert_eq!(offer_line(&CloudOffer { gpu_count: 2, ..offer("A100", 0, None) }), "2× A100");
        assert_eq!(with_article("RTX 4090"), "an RTX 4090");
        assert_eq!(with_article("H100 SXM"), "an H100 SXM");
        assert_eq!(with_article("T4"), "a T4");
    }

    #[test]
    fn phases_and_transfers() {
        let gib = 1u64 << 30;
        let p = TransferProgress { done_bytes: gib + gib / 5, total_bytes: 4 * gib + gib * 4 / 5 };
        assert_eq!(fmt_transfer(&p), "1.2 of 4.8 GiB");
        assert_eq!(
            fmt_transfer(&TransferProgress { done_bytes: 300 << 20, total_bytes: 2 * gib }),
            "300.0 MiB of 2.0 GiB"
        );
        assert_eq!(fmt_transfer(&TransferProgress { done_bytes: 300 << 20, total_bytes: 0 }), "300.0 MiB");
        let mut c = RunCloud {
            connection_id: "cc".into(),
            provider: "runpod".into(),
            offer: offer("RTX 4090", 0, None),
            instance_id: None,
            phase: CloudPhase::Provisioning,
            progress: None,
            max_hours: 12.,
            deadline: None,
            cost_usd: None,
            synced: None,
        };
        assert_eq!(cloud_wait_title(&c, "RunPod"), "Provisioning an RTX 4090 at RunPod");
        c.phase = CloudPhase::Booting;
        assert_eq!(cloud_wait_title(&c, "RunPod"), "Booting the instance");
        c.phase = CloudPhase::Syncing;
        c.progress = Some(TransferProgress { done_bytes: 1, total_bytes: 4 });
        assert_eq!(cloud_wait_title(&c, "RunPod"), "Uploading data · 25%");
        c.progress = Some(TransferProgress { done_bytes: 1, total_bytes: 0 });
        assert_eq!(phase_text(&c), "Uploading data");
    }
}

#[cfg(test)]
mod format_tests {
    use super::{fmt_aug, fmt_lr, fmt_scale};

    #[test]
    fn rates_read_one_way() {
        assert_eq!(fmt_lr(0.01), "0.01");
        assert_eq!(fmt_lr(0.0125), "0.0125");
        assert_eq!(fmt_lr(0.001111), "1.11×10⁻³");
        assert_eq!(fmt_lr(0.001), "1.00×10⁻³");
        assert_eq!(fmt_lr(1e-7), "1.00×10⁻⁷");
        // An f32 rate widened: 0.01 stays "0.01", not "10.00×10⁻³".
        assert_eq!(fmt_lr(0.01_f32 as f64), "0.01");
        assert_eq!(fmt_lr(0.001111_f32 as f64), "1.11×10⁻³");
        assert_eq!(fmt_aug(1.0), "1.00");
        assert_eq!(fmt_scale(0.5), "±0.50");
    }
}
