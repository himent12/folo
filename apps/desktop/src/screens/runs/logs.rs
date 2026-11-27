//! A run's log panel: its lines, level and search filters, truncation notice, keyboard scrolling and copying.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{LogLevel, LogLine, Run, RunStatus, time};
use crate::app::{FoloApp, LogFilter, act};
use crate::controls::Btn;
use crate::screens::loading_rows;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Maximum matching lines rendered in the scrollback at once (tail window).
pub(crate) const LOG_DISPLAY_CAP: usize = 300;

/// Cached per-run log level counts and filtered line indices, invalidated when the run, log
/// revision, line buffer identity, level filter, or search query changes. Avoids per-frame O(N)
/// lowercase string allocations while scrolling or repainting a run page.
#[derive(Default)]
pub(crate) struct LogFilterCache {
    counts_key: Option<(String, u64, usize, usize)>,
    counts: (usize, usize),
    filter_key: Option<(String, u64, usize, usize, LogFilter, String)>,
    indices: Vec<usize>,
    #[cfg(test)]
    recomputes: usize,
}

impl LogFilterCache {
    /// Ensures cached level counts and filtered line indices match the given inputs.
    pub(crate) fn ensure(
        &mut self,
        run_id: &str,
        log_rev: u64,
        lines: &[LogLine],
        filter: LogFilter,
        query: &str,
    ) {
        let ptr = lines.as_ptr() as usize;
        let len = lines.len();
        let counts_hit = self
            .counts_key
            .as_ref()
            .is_some_and(|(id, rev, p, l)| id == run_id && *rev == log_rev && *p == ptr && *l == len);
        if !counts_hit {
            self.counts = level_counts(lines);
            self.counts_key = Some((run_id.to_string(), log_rev, ptr, len));
        }
        let filter_hit = self.filter_key.as_ref().is_some_and(|(id, rev, p, l, f, q)| {
            id == run_id && *rev == log_rev && *p == ptr && *l == len && *f == filter && q == query
        });
        if !filter_hit {
            self.indices = filter_log_indices(lines, filter, query);
            self.filter_key = Some((run_id.to_string(), log_rev, ptr, len, filter, query.to_string()));
            #[cfg(test)]
            {
                self.recomputes += 1;
            }
        }
    }

    pub(crate) fn counts(&self) -> (usize, usize) {
        self.counts
    }

    pub(crate) fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub(crate) fn clear(&mut self) {
        self.counts_key = None;
        self.counts = (0, 0);
        self.filter_key = None;
        self.indices.clear();
    }
}

impl FoloApp {
    /// The log panel. `beside`: it sits beside Config, so its scrollback takes the row's height
    /// (at least its usual 372 px, or 220 px when sharing the column with Checkpoint CLI); stacked,
    /// it is 372 px tall.
    pub(super) fn logs_panel(
        &mut self,
        r: &Run,
        beside: bool,
        with_ckpt_cli: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let filter = self.log_filter;
        let query = self.log_search.read(cx).text().to_string();
        let filtered = filter != LogFilter::All || !query.trim().is_empty();
        // The scrollback is one Tab stop: arrows step a line, Page Up/Down a screen, Home/End the ends.
        let log_fh = self.fh("log-scroll", cx);
        let log_ring = self.ring(&log_fh, window);
        if log_ring {
            self.hint_keys("↑ ↓ a line · PgUp / PgDn a screen · Home / End the first and last lines");
        }
        let log_chrome = self.scroll_chrome(self.log_scroll.clone(), t.bg_inset, 0., 0., cx);
        if let Some(Load::Ready(hist)) = self.store.logs.get(&r.id) {
            self.log_cache.ensure(&r.id, self.store.log_rev, &hist.lines, filter, &query);
        }
        let logs_failed = self.store.logs.get(&r.id).is_some_and(|l| l.error().is_some());
        let retry_anchor = match self.passing_stop("run-logs-retry", logs_failed, cx) {
            Some(fh) if !logs_failed => Some(div().track_focus(&fh)),
            _ => None,
        };
        let (notice, copy_btn, warns, errors, body) = match self.store.logs.get(&r.id) {
            Some(Load::Ready(hist)) => {
                let total_in_buf = hist.lines.len();
                let dropped = hist.dropped;
                let (warns, errors) = self.log_cache.counts();
                let shown = self.log_cache.indices();
                let matched = shown.len();
                let start = matched.saturating_sub(LOG_DISPLAY_CAP);
                let displayed = matched - start;
                let notice = truncation_notice(total_in_buf, dropped, matched, displayed, filtered);
                let copy_btn = if total_in_buf == 0 {
                    None
                } else if matched == 0 {
                    Some(
                        self.button_disabled(
                            "log-copy",
                            "Copy matching",
                            Btn::Secondary,
                            Some("icons/copy.svg"),
                            "No matching log lines to copy",
                            window,
                            cx,
                        )
                        .into_any_element(),
                    )
                } else {
                    let rid = r.id.clone();
                    let label = if filtered { "Copy matching" } else { "Copy log" };
                    let tip_msg = if filtered {
                        format!("Copy {}", counted(matched as u64, "matching log line", "matching log lines"))
                    } else if dropped > 0 {
                        format!(
                            "Copy the latest {}",
                            counted(matched as u64, "loaded log line", "loaded log lines")
                        )
                    } else {
                        format!("Copy {}", counted(matched as u64, "log line", "log lines"))
                    };
                    Some(
                        self.button(
                            "log-copy",
                            label,
                            Btn::Secondary,
                            Some("icons/copy.svg"),
                            window,
                            cx,
                            act(move |this, _, cx| this.copy_log(&rid, cx)),
                        )
                        .tooltip(crate::controls::tip(tip_msg, None))
                        .into_any_element(),
                    )
                };
                let empty_msg =
                    empty_log_message(total_in_buf, dropped, r.status.is_active(), filter, &query);
                let clear_btn = (total_in_buf > 0 && matched == 0).then(|| {
                    self.button(
                        "log-clear-filters",
                        "Clear filters",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, w, cx| {
                            this.log_filter = LogFilter::All;
                            this.log_search.update(cx, |i, cx| i.set_text("", cx));
                            this.log_follow = true;
                            this.log_scroll.scroll_to_bottom();
                            w.focus(&this.log_search.read(cx).focus_handle(cx));
                            cx.notify();
                        }),
                    )
                });
                let scroll = div()
                    .relative()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("log-scroll")
                            .track_focus(&log_fh)
                            .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                                let s = &this.log_scroll;
                                let (y, max) = (-s.offset().y, s.max_offset().height);
                                let page = (s.bounds().size.height - px(LOG_LINE)).max(px(LOG_LINE));
                                let to = match ev.keystroke.key.as_str() {
                                    "up" => y - px(LOG_LINE),
                                    "down" => y + px(LOG_LINE),
                                    "pageup" => y - page,
                                    "pagedown" => y + page,
                                    "home" => px(0.),
                                    "end" => max,
                                    _ => return,
                                };
                                let to = to.clamp(px(0.), max);
                                s.set_offset(folo_ui::point(s.offset().x, -to));
                                // The key decides following now (a reflow before the next frame
                                // could otherwise hide the move).
                                this.log_follow = max - to < px(8.);
                                this.log_last_y = to;
                                cx.stop_propagation();
                                cx.notify();
                            }))
                            .flex_1()
                            .when(beside, |d| {
                                d.flex_basis(px(0.)).min_h(px(if with_ckpt_cli { 220. } else { LOG_H }))
                            })
                            .when(!beside, |d| d.h(px(LOG_H)))
                            .overflow_y_scroll()
                            .track_scroll(&self.log_scroll)
                            .bg(t.bg_inset)
                            .py(px(10.))
                            .when(shown.is_empty(), |d| {
                                d.child(
                                    div()
                                        .px(px(16.))
                                        .py(px(4.))
                                        .flex()
                                        .flex_wrap()
                                        .items_center()
                                        .gap(px(10.))
                                        .child(text(ty::SM, t.text_tertiary).child(empty_msg))
                                        .children(clear_btn),
                                )
                            })
                            .children(shown[start..].iter().map(|&idx| {
                                let l = &hist.lines[idx];
                                let (tag, color) = match l.level {
                                    LogLevel::Debug => ("DBG", t.text_tertiary),
                                    LogLevel::Info => ("INF", t.text_tertiary),
                                    LogLevel::Warn => ("WRN", t.warning_text),
                                    LogLevel::Error => ("ERR", t.danger_text),
                                };
                                let clock = time::parse(&l.t).map(time::local_clock).unwrap_or_default();
                                div()
                                    .flex()
                                    .gap(px(12.))
                                    .px(px(16.))
                                    .py(px(2.))
                                    .font_family(crate::theme::MONO)
                                    .text_size(px(ty::SM.size))
                                    .line_height(px(17.))
                                    .when(l.level == LogLevel::Error, |d| d.bg(t.danger_muted))
                                    .child(div().flex_none().text_color(t.text_tertiary).child(clock))
                                    .child(div().flex_none().w(px(24.)).text_color(color).child(tag))
                                    .child(
                                        // Long entries wrap under the message column, so nothing is clipped.
                                        div()
                                            .flex_1()
                                            .min_w_0()
                                            .text_color(if l.level == LogLevel::Debug {
                                                t.text_tertiary
                                            } else {
                                                t.text_secondary
                                            })
                                            // Fields wrap as whole units, a space apart; a field
                                            // longer than the column still wraps inside itself.
                                            .flex()
                                            .flex_wrap()
                                            .gap_x(px(7.))
                                            .children(log_fields(&l.message).into_iter().map(|f| {
                                                let long = f.chars().count() > 36;
                                                div()
                                                    .max_w_full()
                                                    .when(!long, |d| d.whitespace_nowrap())
                                                    .child(f)
                                            })),
                                    )
                            })),
                    )
                    // Lines above the window fade at its top; at the start of the log, none do.
                    .children(log_chrome)
                    .when(log_ring, |d| d.child(crate::app::focus_ring(&t, radius::SM)))
                    .into_any_element();
                (notice, copy_btn, warns, errors, scroll)
            }
            Some(Load::Failed(e)) => {
                let msg = e.message.clone();
                let rid = r.id.clone();
                let retry = self.button(
                    "run-logs-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_logs(&rid, cx)),
                );
                (
                    None,
                    None,
                    0,
                    0,
                    div()
                        .p(px(16.))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap(px(12.))
                        .child(text(ty::SM, t.danger_text).flex_1().min_w_0().whitespace_normal().child(msg))
                        .child(retry)
                        .into_any_element(),
                )
            }
            _ => (None, None, 0, 0, loading_rows(&t, "logs", 8).into_any_element()),
        };
        // Follow the tail; scrolling up pauses it, scrolling back to the bottom resumes. Decided
        // only when the offset moved, so a reflow (a wrapped line after a resize) doesn't pause it.
        let switched = self.log_run.as_deref() != Some(r.id.as_str());
        if switched {
            self.log_run = Some(r.id.clone());
            self.log_follow = true;
            self.log_last_y = -self.log_scroll.offset().y;
        }
        let (y, max) = (-self.log_scroll.offset().y, self.log_scroll.max_offset().height);
        if !switched && y != self.log_last_y {
            // Tighter than one keyboard line (21 px), so a single Up press pauses following.
            self.log_follow = max - y < px(8.);
            self.log_last_y = y;
        }
        let rev = self.store.log_rev;
        if self.log_follow && (switched || rev != self.log_seen || max - y >= px(8.)) {
            self.log_scroll.scroll_to_bottom();
        }
        self.log_seen = rev;
        let at_bottom = self.log_follow;
        // A count reads in tertiary after its label (the no-break space marks it).
        let count = |label: &str, n: usize| format!("{label}\u{a0}{}", crate::ui::int(n as u64));
        let filters = [LogFilter::All, LogFilter::Warn, LogFilter::Error];
        let levels = self.segmented(
            "log-level",
            &["All", &count("Warn", warns), &count("Error", errors)],
            filters.iter().position(|f| *f == filter).unwrap_or(0),
            window,
            cx,
            Rc::new(move |this, i, _, _| {
                this.log_filter = filters[i];
                // A new filter starts at the newest lines, following.
                this.log_follow = true;
                this.log_scroll.scroll_to_bottom();
            }),
        );
        let search = self.search_field("log-search", self.log_search.clone(), 210., window, cx);
        let follow = if r.status != RunStatus::Running {
            div().into_any_element()
        } else if at_bottom {
            div()
                .flex()
                .items_center()
                .gap(px(6.))
                .child(dot(t.accent_fill(), 6., true, "log-live"))
                .child(text(ty::SM, t.text_tertiary).child("Following"))
                .into_any_element()
        } else {
            let el = div()
                .id("log-resume")
                .flex()
                .items_center()
                .gap(px(4.))
                .px(px(6.))
                .h(px(20.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .active(|s| s.bg(t.bg_active))
                .child(text(ty::SM, t.text_secondary).child("Resume following"))
                .child(icon("icons/chevron-down.svg", 12., t.text_secondary));
            self.bind(
                el,
                "log-resume",
                radius::SM,
                window,
                cx,
                act(|this, _, cx| {
                    this.log_follow = true;
                    this.log_scroll.scroll_to_bottom();
                    cx.notify();
                }),
            )
            .into_any_element()
        };
        let toolbar = div()
            .px(px(16.))
            .py(px(8.))
            .border_b_1()
            .border_color(t.line_subtle)
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(8.))
            .child(levels)
            .child(search);
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                panel_header(&t, "Logs")
                    .child(div().flex().items_center().gap(px(12.)).child(follow).children(copy_btn)),
            )
            .child(toolbar)
            .children(retry_anchor)
            .when_some(notice, |d, msg| {
                d.child(
                    div()
                        .id("log-truncation-notice")
                        .px(px(16.))
                        .py(px(6.))
                        .border_b_1()
                        .border_color(t.line_subtle)
                        .bg(t.bg_panel)
                        .flex()
                        .items_start()
                        .gap(px(6.))
                        .child(div().pt(px(2.)).flex_none().child(icon(
                            "icons/info.svg",
                            12.,
                            t.text_tertiary,
                        )))
                        .child(
                            text(ty::SM, t.text_tertiary).flex_1().min_w_0().whitespace_normal().child(msg),
                        ),
                )
            })
            .child(body)
            .into_any_element()
    }

    /// Puts the run's currently filtered log lines on the clipboard, one line per entry as shown
    /// (local time, level, message).
    pub(crate) fn copy_log(&mut self, id: &str, cx: &mut Context<Self>) {
        let query = self.log_search.read(cx).text().to_string();
        let filtered = self.log_filter != LogFilter::All || !query.trim().is_empty();
        if let Some(Load::Ready(hist)) = self.store.logs.get(id) {
            self.log_cache.ensure(id, self.store.log_rev, &hist.lines, self.log_filter, &query);
        } else {
            return;
        }
        let Some(hist) = self.store.logs.get(id).and_then(|l| l.ready()) else { return };
        let indices = self.log_cache.indices();
        if indices.is_empty() {
            return;
        }
        let shown: Vec<&LogLine> = indices.iter().map(|&i| &hist.lines[i]).collect();
        let text = format_log_lines(&shown);
        let n = shown.len() as u64;
        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(text));
        let label = if filtered {
            counted(n, "matching log line", "matching log lines")
        } else {
            counted(n, "log line", "log lines")
        };
        self.toast(Tone::Neutral, format!("Copied {label}"), None, cx);
    }
}

/// Formats log lines for the clipboard (local time, level tag, message).
pub(crate) fn format_log_lines(lines: &[&LogLine]) -> String {
    lines
        .iter()
        .map(|l| {
            let clock = time::parse(&l.t).map(time::local_clock).unwrap_or_default();
            let level = match l.level {
                LogLevel::Debug => "DBG",
                LogLevel::Info => "INF",
                LogLevel::Warn => "WRN",
                LogLevel::Error => "ERR",
            };
            format!("{clock} {level} {}\n", l.message)
        })
        .collect()
}

/// Whether `q` (trimmed, lowercased) matches `level`'s severity token (`ERROR`/`ERR`, `WARN`/`WRN`,
/// `INFO`/`INF`, `DEBUG`/`DBG`).
fn level_matches_query(level: LogLevel, q: &str) -> bool {
    match level {
        LogLevel::Debug => matches!(q, "dbg" | "debug"),
        LogLevel::Info => matches!(q, "inf" | "info"),
        LogLevel::Warn => matches!(q, "wrn" | "warn" | "warning"),
        LogLevel::Error => matches!(q, "err" | "error"),
    }
}

/// Indices into `lines` matching `filter` and optional case-insensitive search across severity
/// level tokens and message text (with `/` / `\` path normalization).
pub(crate) fn filter_log_indices(lines: &[LogLine], filter: LogFilter, query: &str) -> Vec<usize> {
    let q = query.trim().to_lowercase();
    let q_slash = if q.contains('\\') { Some(q.replace('\\', "/")) } else { None };
    let q_norm = q_slash.as_deref().unwrap_or(&q);
    lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            if !shows(filter, l.level) {
                return None;
            }
            if q.is_empty() || level_matches_query(l.level, &q) {
                return Some(i);
            }
            let lower = l.message.to_lowercase();
            let hit = lower.contains(&q)
                || (q_slash.is_some() && lower.contains(q_norm))
                || (lower.contains('\\') && lower.replace('\\', "/").contains(q_norm));
            hit.then_some(i)
        })
        .collect()
}

/// Filters log lines by severity level and optional case-insensitive substring search.
#[cfg(test)]
pub(crate) fn filter_log_lines<'a>(lines: &'a [LogLine], filter: LogFilter, query: &str) -> Vec<&'a LogLine> {
    filter_log_indices(lines, filter, query).into_iter().map(|i| &lines[i]).collect()
}

/// Empty-state message for the log scrollback, distinguishing 0-line logs from filter/search misses
/// and scoping level-only misses to the loaded history when earlier lines were omitted.
pub(crate) fn empty_log_message(
    total_lines: usize,
    dropped: u64,
    active: bool,
    filter: LogFilter,
    query: &str,
) -> &'static str {
    if total_lines == 0 {
        return if active { "No log entries yet." } else { "No log entries in this run." };
    }
    if !query.trim().is_empty() {
        return match filter {
            LogFilter::All => "No log lines match your search.",
            LogFilter::Warn => "No warnings or errors match your search.",
            LogFilter::Error => "No errors match your search.",
        };
    }
    match (filter, dropped > 0) {
        (LogFilter::Error, true) => "No errors in the loaded log lines.",
        (LogFilter::Error, false) => "No errors in this run's log.",
        (LogFilter::Warn, true) => "No warnings or errors in the loaded log lines.",
        (LogFilter::Warn, false) => "No warnings or errors in this run's log.",
        (LogFilter::All, _) => "No log entries yet.",
    }
}

/// Banner text when earlier log lines are omitted by the server limit, the client ring buffer, or
/// the 300-line display tail window.
pub(crate) fn truncation_notice(
    buf_len: usize,
    dropped: u64,
    matched: usize,
    displayed: usize,
    filtered: bool,
) -> Option<String> {
    let hidden_display = matched.saturating_sub(displayed) as u64;
    if filtered {
        if hidden_display > 0 && dropped > 0 {
            Some(format!(
                "Showing the latest {} of {} matching lines ({} earlier omitted from history).",
                int(displayed as u64),
                int(matched as u64),
                int(dropped)
            ))
        } else if hidden_display > 0 {
            Some(format!(
                "Showing the latest {} of {} matching lines.",
                int(displayed as u64),
                int(matched as u64)
            ))
        } else if dropped > 0 {
            Some(format!(
                "Filtered over the latest {} lines ({} earlier omitted from history).",
                int(buf_len as u64),
                int(dropped)
            ))
        } else {
            None
        }
    } else {
        let omitted = dropped.saturating_add(hidden_display);
        if omitted > 0 {
            let total = dropped.saturating_add(buf_len as u64);
            Some(format!(
                "Showing the latest {} of {} lines ({} earlier omitted).",
                int(displayed as u64),
                int(total),
                int(omitted)
            ))
        } else {
            None
        }
    }
}

/// A log message as shown: Windows paths with `/`, split into its " · " fields, each but the
/// first led by its dot (a wrapped line starts "· mAP50 0.745" instead of ending on "·").
/// Rows lay the fields out unbroken in a wrapping row, so a long entry breaks between fields,
/// never inside "mAP50 0.745" (the line wrapper breaks before any non-word character, so a
/// no-break space can't hold a pair together).
fn log_fields(s: &str) -> Vec<String> {
    let s = slash_paths(s);
    s.split(" · ").enumerate().map(|(i, f)| if i > 0 { format!("· {f}") } else { f.to_string() }).collect()
}

/// `D:\data\x.folo` → `D:/data/x.folo`: only whitespace-separated words that start with a drive
/// (a letter, `:`, a separator) change, so a Unix path or escaped text with `\` keeps it.
fn slash_paths(s: &str) -> String {
    let drive = |w: &str| {
        let b = w.trim_start_matches(['"', '\'', '(']).as_bytes();
        b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/')
    };
    if !s.contains('\\') {
        return s.to_string();
    }
    s.split_inclusive(' ').map(|w| if drive(w) { w.replace('\\', "/") } else { w.to_string() }).collect()
}

/// One log line with its padding: an arrow key's scroll step in the log.
const LOG_LINE: f32 = 21.;
/// The scrollback's height stacked, and its least beside Config.
const LOG_H: f32 = 372.;

/// Whether the level filter shows a line: Warn shows warnings and errors, Error errors alone.
fn shows(filter: LogFilter, level: LogLevel) -> bool {
    match filter {
        LogFilter::All => true,
        LogFilter::Warn => matches!(level, LogLevel::Warn | LogLevel::Error),
        LogFilter::Error => level == LogLevel::Error,
    }
}

/// The log's warnings and its errors, as the filter's counts say them.
fn level_counts(lines: &[LogLine]) -> (usize, usize) {
    let n = |lv: LogLevel| lines.iter().filter(|l| l.level == lv).count();
    (n(LogLevel::Warn), n(LogLevel::Error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_level_filter_shows_warnings_and_up_or_errors_alone_and_counts_each() {
        let line = |level| LogLine { t: String::new(), level, message: String::new() };
        let lines =
            [LogLevel::Info, LogLevel::Warn, LogLevel::Debug, LogLevel::Error, LogLevel::Warn].map(line);
        assert_eq!(level_counts(&lines), (2, 1));
        let n = |f| lines.iter().filter(|l| shows(f, l.level)).count();
        assert_eq!((n(LogFilter::All), n(LogFilter::Warn), n(LogFilter::Error)), (5, 3, 1));
    }

    #[test]
    fn search_and_level_filters_combine_and_match_normalized_windows_paths() {
        let lines = vec![
            LogLine {
                t: String::new(),
                level: LogLevel::Info,
                message: "loaded weights from C:\\Program Files\\FOLO\\models\\best.folo".into(),
            },
            LogLine { t: String::new(), level: LogLevel::Warn, message: "lr warmup spike detected".into() },
            LogLine {
                t: String::new(),
                level: LogLevel::Error,
                message: "GPU Out Of Memory during batch 16".into(),
            },
        ];
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "C:/Program Files/FOLO/models").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "C:\\Program Files\\FOLO").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "out of memory").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::Warn, "WARMUP").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::Error, "warmup").len(), 0);
        // Severity level tokens match case-insensitively without false positives on short substrings.
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "ERROR").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "err").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "WARN").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "wrn").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "INFO").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::All, "inf").len(), 1);
        assert_eq!(filter_log_lines(&lines, LogFilter::Error, "INFO").len(), 0);
        // Searching for "in" does not match Info lines whose message lacks "in".
        let plain = vec![LogLine { t: String::new(), level: LogLevel::Info, message: "epoch 1 done".into() }];
        assert_eq!(filter_log_lines(&plain, LogFilter::All, "in").len(), 0);
    }

    #[test]
    fn log_filter_cache_reuses_results_until_key_changes() {
        let mut lines = vec![
            LogLine { t: String::new(), level: LogLevel::Info, message: "epoch 1".into() },
            LogLine { t: String::new(), level: LogLevel::Warn, message: "slow io".into() },
            LogLine { t: String::new(), level: LogLevel::Error, message: "oom".into() },
        ];
        let mut cache = LogFilterCache::default();
        cache.ensure("run_1", 1, &lines, LogFilter::All, "ERROR");
        assert_eq!(cache.counts(), (1, 1));
        assert_eq!(cache.indices(), &[2]);
        assert_eq!(cache.recomputes, 1);

        // Identical key: cache hit, no recompute.
        cache.ensure("run_1", 1, &lines, LogFilter::All, "ERROR");
        assert_eq!(cache.recomputes, 1);

        // Query change recomputes indices but keeps level counts.
        cache.ensure("run_1", 1, &lines, LogFilter::Warn, "");
        assert_eq!(cache.indices(), &[1, 2]);
        assert_eq!(cache.recomputes, 2);

        // Filter change recomputes.
        cache.ensure("run_1", 1, &lines, LogFilter::Error, "");
        assert_eq!(cache.indices(), &[2]);
        assert_eq!(cache.recomputes, 3);

        // New log revision recomputes counts and indices.
        lines.push(LogLine { t: String::new(), level: LogLevel::Error, message: "cuda fail".into() });
        cache.ensure("run_1", 2, &lines, LogFilter::Error, "");
        assert_eq!(cache.counts(), (1, 2));
        assert_eq!(cache.indices(), &[2, 3]);
        assert_eq!(cache.recomputes, 4);

        // Different run with same rev invalidates cache.
        cache.ensure("run_2", 2, &lines, LogFilter::Error, "");
        assert_eq!(cache.recomputes, 5);
    }

    #[test]
    fn empty_log_messages_distinguish_zero_lines_from_filters_and_search() {
        assert_eq!(empty_log_message(0, 0, true, LogFilter::All, ""), "No log entries yet.");
        assert_eq!(empty_log_message(0, 0, false, LogFilter::All, ""), "No log entries in this run.");
        assert_eq!(
            empty_log_message(10, 0, true, LogFilter::Warn, ""),
            "No warnings or errors in this run's log."
        );
        assert_eq!(
            empty_log_message(10, 50, true, LogFilter::Warn, ""),
            "No warnings or errors in the loaded log lines."
        );
        assert_eq!(empty_log_message(10, 0, true, LogFilter::Error, ""), "No errors in this run's log.");
        assert_eq!(
            empty_log_message(10, 50, true, LogFilter::Error, ""),
            "No errors in the loaded log lines."
        );
        assert_eq!(empty_log_message(10, 0, true, LogFilter::All, "oom"), "No log lines match your search.");
        assert_eq!(
            empty_log_message(10, 0, true, LogFilter::Warn, "oom"),
            "No warnings or errors match your search."
        );
        assert_eq!(empty_log_message(10, 0, true, LogFilter::Error, "oom"), "No errors match your search.");
    }

    #[test]
    fn truncation_notice_reports_display_window_and_dropped_history() {
        assert_eq!(truncation_notice(50, 0, 50, 50, false), None);
        assert_eq!(
            truncation_notice(500, 0, 500, 300, false).as_deref(),
            Some("Showing the latest 300 of 500 lines (200 earlier omitted).")
        );
        assert_eq!(
            truncation_notice(1000, 400, 1000, 300, false).as_deref(),
            Some("Showing the latest 300 of 1,400 lines (1,100 earlier omitted).")
        );
        assert_eq!(
            truncation_notice(1000, 400, 12, 12, true).as_deref(),
            Some("Filtered over the latest 1,000 lines (400 earlier omitted from history).")
        );
        assert_eq!(
            truncation_notice(1000, 0, 450, 300, true).as_deref(),
            Some("Showing the latest 300 of 450 matching lines.")
        );
        assert_eq!(
            truncation_notice(1000, 200, 450, 300, true).as_deref(),
            Some("Showing the latest 300 of 450 matching lines (200 earlier omitted from history).")
        );
    }

    #[test]
    fn log_lines_break_between_fields() {
        let f = super::log_fields("epoch 32/100 · mAP50-95 0.594 · mAP50 0.745 · 14.3 s");
        assert_eq!(f, ["epoch 32/100", "· mAP50-95 0.594", "· mAP50 0.745", "· 14.3 s"]);
        assert_eq!(super::log_fields("loaded 3 classes"), ["loaded 3 classes"]);
    }

    #[test]
    fn windows_paths_only() {
        use super::slash_paths;
        assert_eq!(
            slash_paths("transferred 417 of 417 tensors from D:/folo/data\\models\\n.folo"),
            "transferred 417 of 417 tensors from D:/folo/data/models/n.folo"
        );
        assert_eq!(slash_paths("dataset /data/a\\b ok"), "dataset /data/a\\b ok");
        assert_eq!(slash_paths("loaded \"C:\\x\\y\""), "loaded \"C:/x/y\"");
    }
}
