//! Text formatting: queue places, confidences, paths, counts, sizes, rates, devices and plurals.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{Div, Hsla, ParentElement, Styled, div, px};

use super::cloud::fmt_transfer;
use super::text::{StyledExt, text};
use crate::api::{Run, RunExt};
use crate::theme::{TypeStep, ty};

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

/// What a run waiting in the training queue waits for, as its page says it: the run that holds
/// the device (`ahead`: its name and time left) and the queued runs in front of it.
pub fn queue_wait_text(position: Option<u32>, ahead: Option<(&str, Option<f64>)>) -> String {
    let left = |eta: Option<f64>| {
        eta.map_or("time left not known".to_string(), |s| format!("{} left", crate::api::time::duration(s)))
    };
    match (position.filter(|p| *p > 1), ahead) {
        (Some(p), Some((name, eta))) => {
            format!("Starts after {name} ({}) and {}.", left(eta), plural(p as usize - 1, "queued run"))
        }
        (Some(p), None) => format!("Starts after {}.", plural(p as usize - 1, "queued run")),
        (None, Some((name, Some(eta)))) => {
            format!("Starts when {name} finishes, in about {}.", crate::api::time::duration(eta))
        }
        (None, Some((name, None))) => format!("Starts when {name} finishes."),
        (None, None) => "Starts as soon as the device is free.".to_string(),
    }
}

/// A GPU as Overview's system meters name it: its short name, and its memory when the platform
/// reports it ("RTX 4080 SUPER · 16.0 GiB VRAM"); never "VRAM unknown". Its load is the meter
/// beside it, when the platform reports one.
pub fn gpu_label(g: &crate::api::GpuInfo) -> String {
    if g.memory_bytes > 0 {
        let (vram, unit) = bytes(g.memory_bytes);
        format!("{} · {vram} {unit} VRAM", short_gpu(&g.name))
    } else {
        short_gpu(&g.name)
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

/// What a run that hasn't started training waits on, as every list writes it beside its status: a
/// cloud run's transfer ("1.2 of 4.8 GiB"; "–" before there is one, the status names the phase),
/// else its place in the training queue.
pub fn waiting_text(r: &Run) -> String {
    match r.cloud_pending() {
        Some(c) => c.progress.as_ref().map_or("–".to_string(), fmt_transfer),
        None => queue_text(r.queue_position),
    }
}

/// A GPU name with its article, as the name is read aloud: "an RTX 4090", "an A100", "a T4" (GPU
/// names start with initials, read letter by letter).
pub fn with_article(name: &str) -> String {
    let an = name.chars().next().is_some_and(|c| "AEFHILMNORSXaeiou".contains(c));
    format!("{} {name}", if an { "an" } else { "a" })
}

/// How a model's weights are stored, from its own record (`precision`, `fused`): "f16, fused",
/// "f32", or `None` when the server doesn't say. Every surface that names it uses this.
pub fn model_storage(m: &crate::api::Model) -> Option<String> {
    let p = m.precision?.as_str();
    Some(if m.fused == Some(true) { format!("{p}, fused") } else { p.to_string() })
}

/// The data a model was evaluated on, one way everywhere: "shapes-hard · val · 800 images" (the
/// count when known).
pub fn split_label(dataset: &str, split: &str, images: Option<u64>) -> String {
    match images {
        Some(n) => format!("{dataset} · {split} · {}", counted(n, "image", "images")),
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

/// A learning rate in normalised scientific notation, as Studio writes it (`lib/format` `lr`):
/// "1.43×10⁻⁵", a mantissa that would round to 10 moving to the next power ("1.00×10⁻²").
pub fn lr_sci(x: f64) -> String {
    if x == 0. || !x.is_finite() {
        return if x == 0. { "0".into() } else { "–".into() };
    }
    let mut e = x.abs().log10().floor() as i32;
    if (x / 10f64.powi(e) * 100.).round().abs() >= 1000. {
        e += 1;
    }
    crate::charts::in_pow(x, e)
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

/// A 0–1 score in the legend of a chart whose axis is fitted to `axis_max` (charts::domain), as
/// Studio's `axisMetric` writes it: three decimals, or the axis labels' when a zoomed axis has more
/// (on 0–0.00042, "0.00039", not "0.000"), and two significant figures for a nonzero score still
/// under the last digit, so a small score never reads as zero.
pub fn fmt_score_on(v: f32, axis_max: f32) -> String {
    let mut d = crate::charts::domain::unit_domain(axis_max, 0.).decimals.max(3);
    let a = v.abs() as f64;
    if a > 0. && a < 0.5 * 10f64.powi(-(d as i32)) {
        d = (1 - a.log10().floor() as i32) as usize;
    }
    format!("{v:.d$}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// d1-datasets: a one-image split read "1 images".
    #[test]
    fn a_split_label_counts_one_image_in_the_singular() {
        assert_eq!(split_label("shapes", "val", Some(1)), "shapes · val · 1 image");
        assert_eq!(split_label("shapes", "val", Some(1200)), "shapes · val · 1,200 images");
        assert_eq!(split_label("shapes", "val", Some(0)), "shapes · val · 0 images");
        assert_eq!(split_label("shapes", "val", None), "shapes · val");
    }

    #[test]
    fn confidences_keep_a_meaningful_third_decimal() {
        assert_eq!(fmt_conf(0.31), "0.31");
        assert_eq!(fmt_conf(0.314), "0.314");
        assert_eq!(fmt_conf(0.024), "0.024");
        assert_eq!(fmt_conf(0.35), "0.35");
    }

    /// #338: a one-epoch run's mAP50 0.000388 and mAP50-95 0.000068, on an axis zoomed to
    /// 0–0.00042, both read "0.000" in the legend.
    #[test]
    fn a_zoomed_legend_reads_small_scores_apart() {
        let max = 0.000_387_725_9;
        assert_eq!(fmt_score_on(max, max), "0.00039");
        assert_eq!(fmt_score_on(0.000_067_994_26, max), "0.00007");
        // Under the axis's last digit: two significant figures, never zero.
        assert_eq!(fmt_score_on(0.000_001_23, max), "0.0000012");
        assert_eq!(fmt_score_on(0., max), "0.00000");
        // A full-scale or lightly zoomed axis keeps three decimals; a finer one its own.
        assert_eq!(fmt_score_on(0.741_23, 0.8), "0.741");
        assert_eq!(fmt_score_on(0.051_2, 0.06), "0.051");
        assert_eq!(fmt_score_on(0.004_12, 0.004_9), "0.0041");
        assert_eq!(fmt_score_on(0.000_2, 0.6), "0.00020");
    }

    #[test]
    fn a_gpu_is_named_by_what_is_known() {
        let g = crate::api::GpuInfo {
            ordinal: 0,
            name: "NVIDIA GeForce RTX 4080 SUPER".into(),
            backend: crate::api::GpuBackend::Vulkan,
            memory_bytes: 16 << 30,
            util: None,
        };
        assert_eq!(super::gpu_label(&g), "RTX 4080 SUPER · 16.0 GiB VRAM");
        assert_eq!(super::gpu_label(&crate::api::GpuInfo { memory_bytes: 0, ..g }), "RTX 4080 SUPER");
    }

    #[test]
    fn learning_rates_read_as_studio_writes_them() {
        use super::lr_sci;
        // The dogfood case: Desktop said "0.014×10⁻³", Studio "1.43×10⁻⁵".
        assert_eq!(lr_sci(1.43e-5), "1.43×10⁻⁵");
        assert_eq!(lr_sci(0.0067), "6.70×10⁻³");
        assert_eq!(lr_sci(0.009996), "1.00×10⁻²");
        assert_eq!(lr_sci(0.01), "1.00×10⁻²");
        assert_eq!(lr_sci(0.), "0");
    }

    #[test]
    fn counts_scores_and_operating_points_read_right() {
        use super::{counted, operating_conf, score_delta};
        use std::cmp::Ordering;
        assert_eq!(counted(1, "box", "boxes"), "1 box");
        assert_eq!(counted(1200, "box", "boxes"), "1,200 boxes");
        assert_eq!(counted(0, "result", "results"), "0 results");
        assert_eq!(score_delta(0.0004), ("±0.000".to_string(), Ordering::Equal));
        assert_eq!(score_delta(-0.0004), ("±0.000".to_string(), Ordering::Equal));
        assert_eq!(score_delta(0.012), ("+0.012".to_string(), Ordering::Greater));
        assert_eq!(score_delta(-0.004), ("-0.004".to_string(), Ordering::Less));
        assert_eq!(operating_conf(Some(1.0), 0., 0.), None);
        assert_eq!(operating_conf(Some(0.31), 0.5, 0.), Some(0.31));
    }

    #[test]
    fn a_queued_run_says_what_it_waits_for() {
        use super::queue_wait_text;
        assert_eq!(queue_wait_text(Some(1), None), "Starts as soon as the device is free.");
        assert_eq!(
            queue_wait_text(Some(1), Some(("shapes-s-4", Some(1080.)))),
            "Starts when shapes-s-4 finishes, in about 18m 00s."
        );
        assert_eq!(queue_wait_text(None, Some(("a", None))), "Starts when a finishes.");
        assert_eq!(
            queue_wait_text(Some(3), Some(("a", Some(90.)))),
            "Starts after a (1m 30s left) and 2 queued runs."
        );
        assert_eq!(queue_wait_text(Some(2), None), "Starts after 1 queued run.");
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

/// "1 dataset", "3 datasets".
pub fn plural(n: usize, noun: &str) -> String {
    if n == 1 { format!("1 {noun}") } else { format!("{n} {noun}s") }
}

/// A count with thousands separators and the noun that fits it: "1 box", "1,200 boxes".
pub fn counted(n: u64, one: &str, many: &str) -> String {
    format!("{} {}", int(n), if n == 1 { one } else { many })
}

/// An epoch-over-epoch change of a score, as the KPI writes it ("+0.012", "-0.004") and which way
/// it went: a change that rounds to nothing reads "±0.000", neither good nor bad.
pub fn score_delta(d: f32) -> (String, std::cmp::Ordering) {
    let shown = (d * 1000.).round() / 1000.;
    if shown == 0. {
        ("±0.000".into(), std::cmp::Ordering::Equal)
    } else if shown > 0. {
        (format!("+{shown:.3}"), std::cmp::Ordering::Greater)
    } else {
        (format!("{shown:.3}"), std::cmp::Ordering::Less)
    }
}

/// The best-F1 confidence worth naming: none when nothing was detected (precision and recall both
/// 0, where the engine's "best" threshold is just its top end and finds nothing).
pub fn operating_conf(conf: Option<f32>, precision: f32, recall: f32) -> Option<f32> {
    conf.filter(|_| precision > 0. || recall > 0.)
}

#[cfg(test)]
mod format_tests {
    use super::fmt_lr;

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
    }
}
