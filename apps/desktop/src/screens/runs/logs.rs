//! A run's log panel: its lines, keyboard scrolling and copying.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, StatefulInteractiveElement, Styled,
    Window, div, px,
};

use crate::api::{LogLevel, Run, RunStatus, time};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::loading_rows;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn logs_panel(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        // The scrollback is one Tab stop: arrows step a line, Page Up/Down a screen, Home/End the ends.
        let log_fh = self.fh("log-scroll", cx);
        let log_ring = self.ring(&log_fh, window);
        if log_ring {
            self.hint_keys("↑ ↓ a line · PgUp / PgDn a screen · Home / End the first and last lines");
        }
        let log_chrome = self.scroll_chrome(self.log_scroll.clone(), t.bg_inset, 0., 0., cx);
        let body = match self.store.logs.get(&r.id) {
            Some(Load::Ready(lines)) => {
                let start = lines.len().saturating_sub(300);
                div()
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
                            .h(px(372.))
                            .overflow_y_scroll()
                            .track_scroll(&self.log_scroll)
                            .bg(t.bg_inset)
                            .py(px(10.))
                            .children(lines[start..].iter().map(|l| {
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
                    .into_any_element()
            }
            Some(Load::Failed(e)) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            _ => loading_rows(&t, "logs", 8).into_any_element(),
        };
        // Follow the tail; scrolling up pauses it, scrolling back to the bottom resumes. Decided
        // only when the offset moved, so a reflow (a wrapped line after a resize) doesn't pause it.
        if self.log_run.as_deref() != Some(r.id.as_str()) {
            self.log_run = Some(r.id.clone());
            self.log_follow = true;
            self.log_last_y = -self.log_scroll.offset().y;
        }
        let (y, max) = (-self.log_scroll.offset().y, self.log_scroll.max_offset().height);
        if y != self.log_last_y {
            // Tighter than one keyboard line (21 px), so a single Up press pauses following.
            self.log_follow = max - y < px(8.);
            self.log_last_y = y;
        }
        let rev = self.store.log_rev;
        if self.log_follow && (rev != self.log_seen || max - y >= px(8.)) {
            self.log_scroll.scroll_to_bottom();
        }
        self.log_seen = rev;
        let at_bottom = self.log_follow;
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
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Log").child(
                div().flex().items_center().gap(px(12.)).child(follow).when(
                    self.store.logs.get(&r.id).is_some_and(|l| l.ready().is_some_and(|v| !v.is_empty())),
                    |d| {
                        let rid = r.id.clone();
                        d.child(self.button(
                            "log-copy",
                            "Copy log",
                            Btn::Secondary,
                            Some("icons/copy.svg"),
                            window,
                            cx,
                            act(move |this, _, cx| this.copy_log(&rid, cx)),
                        ))
                    },
                ),
            ))
            .child(body)
            .into_any_element()
    }

    /// Puts the run's whole log on the clipboard, a line per entry as the panel shows it (local
    /// time, level, message).
    fn copy_log(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(lines) = self.store.logs.get(id).and_then(|l| l.ready()) else { return };
        let text: String = lines
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
            .collect();
        let n = lines.len();
        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(text));
        self.toast(Tone::Neutral, format!("Copied {} log lines", crate::ui::int(n as u64)), None, cx);
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

#[cfg(test)]
mod tests {
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
