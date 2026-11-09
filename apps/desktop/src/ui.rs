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

use crate::api::{Label, RunStatus};
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
    text(ty::XS, t.text_tertiary).font_weight(FontWeight::MEDIUM).child(label.to_string())
}

/// A metric or field label: `xs` medium, tertiary, in its own case.
pub fn caption(t: &Theme, label: &str) -> Div {
    text(ty::XS, t.text_tertiary).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label.to_string())
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

pub fn hairline(t: &Theme) -> Div {
    div().h(px(1.)).w_full().flex_none().bg(t.line_subtle)
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

/// One inference's total latency as every readout writes it: "6.4 ms", or "Sim. 6.4 ms" when the
/// built-in demo simulated it.
pub fn latency_text(ms: f64, simulated: bool) -> String {
    if simulated { format!("Sim. {ms:.1} ms") } else { format!("{ms:.1} ms") }
}

/// How a model's weights are stored, from its own record (`precision`, `fused`): "f16, fused",
/// "f32", or `None` when the server doesn't say. Every surface that names it uses this.
pub fn model_storage(m: &crate::api::Model) -> Option<String> {
    let p = m.precision?.as_str();
    Some(if m.fused == Some(true) { format!("{p}, fused") } else { p.to_string() })
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
    div()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .child(text(size, color).flex_none().whitespace_nowrap().child(lead.into()))
        .children(tag)
        .when(!rest.is_empty(), |d| {
            d.child(text(size, color).flex_none().child("·"))
                .child(text(size, color).flex_1().min_w_0().truncate().child(rest.join(" · ")))
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
        .child(text(ty::XS, t.text_secondary).child("e2e"))
}

pub fn badge(t: &Theme, label: impl Into<SharedString>, tone: Tone) -> Div {
    let (bg, fg, _) = tone.colors(t);
    text(ty::XS, fg)
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
            text(ty::XS, if s == RunStatus::Running { t.accent_text } else { fg })
                .whitespace_nowrap()
                .child(s.label()),
        )
}

pub fn status_badge(t: &Theme, s: RunStatus, id: &str) -> Div {
    let tone = status_tone(s);
    let (_, _, dotc) = tone.colors(t);
    let (bg, fg, _) = tone.colors(t);
    text(ty::XS, fg)
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
    text(ty::XS, t.text_secondary)
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
    // difference between the two lines' baselines.
    div()
        .flex()
        .items_end()
        .gap(px(2.))
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
        .child(text(ty::XS, t.text_secondary).whitespace_nowrap().child(name.to_string()))
        .when_some(value, |d, v| d.child(text(ty::XS, t.text_primary).tnum().whitespace_nowrap().child(v)))
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

pub fn f3(v: f32) -> String {
    format!("{v:.3}")
}

#[cfg(test)]
mod tests {
    use super::*;

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
    let color = crate::theme::alpha(t.text_tertiary, if hot { 0.7 } else { 0.45 });
    window.paint_quad(
        fill(Bounds::new(point(g.right - px(6.), g.y + px(2.)), size(px(4.), g.thumb - px(4.))), color)
            .corner_radii(px(2.)),
    );
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
