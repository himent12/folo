//! winit pointer input to GPUI's: buttons, click counts (GPUI's Linux double-click rule), wheel deltas
//! (lines with GPUI's line factor, or logical pixels), touch phases, cursor icons, and the file-drop
//! sequence (winit reports one event per file; GPUI wants entered / pending / submit / exited).

use crate::{
    CursorStyle, ExternalPaths, FileDropEvent, MouseButton, NavigationDirection, Pixels, Point, ScrollDelta,
    TouchPhase, point, px,
};
use smallvec::SmallVec;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use winit::event::{MouseButton as WinitButton, MouseScrollDelta, TouchPhase as WinitPhase};
use winit::window::CursorIcon;

/// Two presses closer than this in time (and `DOUBLE_CLICK_DISTANCE` in space) count as one more click.
/// GPUI's Linux value is GTK's default; macOS's default double-click interval is half a second.
pub(crate) const DOUBLE_CLICK_INTERVAL: Duration =
    Duration::from_millis(if cfg!(target_os = "macos") { 500 } else { 400 });
/// GPUI's double-click distance, in logical pixels on each axis.
pub(crate) const DOUBLE_CLICK_DISTANCE: Pixels = px(5.0);
/// Lines per wheel notch (GPUI's Linux `SCROLL_LINES`; on macOS GPUI passes the system's line delta on).
pub(crate) const SCROLL_LINES: f32 = if cfg!(target_os = "macos") { 1.0 } else { 3.0 };

/// GPUI's button for a winit button (`None` for buttons GPUI has no name for).
pub(crate) fn button(button: WinitButton) -> Option<MouseButton> {
    Some(match button {
        WinitButton::Left => MouseButton::Left,
        WinitButton::Right => MouseButton::Right,
        WinitButton::Middle => MouseButton::Middle,
        WinitButton::Back => MouseButton::Navigate(NavigationDirection::Back),
        WinitButton::Forward => MouseButton::Navigate(NavigationDirection::Forward),
        WinitButton::Other(_) => return None,
    })
}

/// Counts clicks: a press continues the count when it is the same button, soon enough and close enough.
#[derive(Debug, Default)]
pub(crate) struct ClickState {
    last: Option<(MouseButton, Instant, Point<Pixels>)>,
    count: usize,
}

impl ClickState {
    /// Records a press; returns its click count.
    pub fn press(&mut self, button: MouseButton, position: Point<Pixels>, now: Instant) -> usize {
        let continues = self.last.is_some_and(|(last_button, at, last_position)| {
            last_button == button
                && now.saturating_duration_since(at) < DOUBLE_CLICK_INTERVAL
                && is_within_click_distance(last_position, position)
        });
        self.count = if continues { self.count + 1 } else { 1 };
        self.last = Some((button, now, position));
        self.count
    }

    /// The count of the last press (what its release reports).
    pub fn count(&self) -> usize {
        self.count.max(1)
    }
}

fn is_within_click_distance(a: Point<Pixels>, b: Point<Pixels>) -> bool {
    let diff = a - b;
    diff.x.abs() <= DOUBLE_CLICK_DISTANCE && diff.y.abs() <= DOUBLE_CLICK_DISTANCE
}

/// GPUI's scroll delta for a winit one; winit's pixel deltas are physical.
pub(crate) fn scroll_delta(delta: MouseScrollDelta, scale_factor: f32) -> ScrollDelta {
    match delta {
        MouseScrollDelta::LineDelta(x, y) => ScrollDelta::Lines(point(x * SCROLL_LINES, y * SCROLL_LINES)),
        MouseScrollDelta::PixelDelta(p) => {
            ScrollDelta::Pixels(point(px(p.x as f32 / scale_factor), px(p.y as f32 / scale_factor)))
        }
    }
}

/// GPUI's touch phase for a winit one.
pub(crate) fn touch_phase(phase: WinitPhase) -> TouchPhase {
    match phase {
        WinitPhase::Started => TouchPhase::Started,
        WinitPhase::Moved => TouchPhase::Moved,
        WinitPhase::Ended | WinitPhase::Cancelled => TouchPhase::Ended,
    }
}

/// The winit cursor for a GPUI style (`None`: hide the cursor). The names follow GPUI's Linux table.
pub(crate) fn cursor_icon(style: CursorStyle) -> Option<CursorIcon> {
    Some(match style {
        CursorStyle::Arrow => CursorIcon::Default,
        CursorStyle::IBeam => CursorIcon::Text,
        CursorStyle::Crosshair => CursorIcon::Crosshair,
        CursorStyle::ClosedHand => CursorIcon::Grabbing,
        CursorStyle::OpenHand => CursorIcon::Grab,
        CursorStyle::PointingHand => CursorIcon::Pointer,
        CursorStyle::ResizeLeft => CursorIcon::WResize,
        CursorStyle::ResizeRight => CursorIcon::EResize,
        CursorStyle::ResizeLeftRight => CursorIcon::EwResize,
        CursorStyle::ResizeUp => CursorIcon::NResize,
        CursorStyle::ResizeDown => CursorIcon::SResize,
        CursorStyle::ResizeUpDown => CursorIcon::NsResize,
        CursorStyle::ResizeUpLeftDownRight => CursorIcon::NwseResize,
        CursorStyle::ResizeUpRightDownLeft => CursorIcon::NeswResize,
        CursorStyle::ResizeColumn => CursorIcon::ColResize,
        CursorStyle::ResizeRow => CursorIcon::RowResize,
        CursorStyle::IBeamCursorForVerticalLayout => CursorIcon::VerticalText,
        CursorStyle::OperationNotAllowed => CursorIcon::NotAllowed,
        CursorStyle::DragLink => CursorIcon::Alias,
        CursorStyle::DragCopy => CursorIcon::Copy,
        CursorStyle::ContextualMenu => CursorIcon::ContextMenu,
        CursorStyle::None => return None,
    })
}

/// Turns winit's per-file drag events into GPUI's file-drop sequence. Events of one batch (winit sends
/// one per file, back to back) are collected and turned into events by `flush` at the end of the batch.
#[derive(Debug, Default)]
pub(crate) struct FileDropTracker {
    /// Paths hovering, not yet reported.
    hovered: SmallVec<[PathBuf; 2]>,
    /// Paths dropped, not yet reported.
    dropped: SmallVec<[PathBuf; 2]>,
    /// `Entered` was reported and no `Submit` / `Exited` since.
    active: bool,
    cancelled: bool,
}

impl FileDropTracker {
    /// A file is being dragged over the window.
    pub fn hovered(&mut self, path: PathBuf) {
        self.hovered.push(path);
    }

    /// A file was dropped on the window.
    pub fn dropped(&mut self, path: PathBuf) {
        self.dropped.push(path);
    }

    /// The drag left the window or was cancelled.
    pub fn cancelled(&mut self) {
        self.hovered.clear();
        self.cancelled = true;
    }

    /// Whether files are being dragged over the window (the mouse moves become `Pending`).
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// The events the batch adds up to, at `position`.
    pub fn flush(&mut self, position: Point<Pixels>) -> SmallVec<[FileDropEvent; 2]> {
        let mut events = SmallVec::new();
        if !self.dropped.is_empty() {
            if !self.active {
                let paths = ExternalPaths(std::mem::take(&mut self.dropped));
                events.push(FileDropEvent::Entered { position, paths });
            }
            self.dropped.clear();
            self.hovered.clear();
            events.push(FileDropEvent::Submit { position });
            self.active = false;
            self.cancelled = false;
            return events;
        }
        if std::mem::take(&mut self.cancelled) {
            if self.active {
                events.push(FileDropEvent::Exited);
            }
            self.active = false;
        }
        if !self.hovered.is_empty() {
            let paths = ExternalPaths(std::mem::take(&mut self.hovered));
            events.push(FileDropEvent::Entered { position, paths });
            self.active = true;
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalPosition;

    #[test]
    fn click_counts_follow_time_distance_and_button() {
        let mut clicks = ClickState::default();
        let t0 = Instant::now();
        let at = |x: f32| point(px(x), px(10.));
        assert_eq!(clicks.press(MouseButton::Left, at(10.), t0), 1);
        assert_eq!(clicks.press(MouseButton::Left, at(14.), t0 + Duration::from_millis(150)), 2);
        assert_eq!(clicks.press(MouseButton::Left, at(15.), t0 + Duration::from_millis(300)), 3);
        assert_eq!(clicks.count(), 3);
        // Too far.
        assert_eq!(clicks.press(MouseButton::Left, at(21.), t0 + Duration::from_millis(350)), 1);
        // Another button.
        assert_eq!(clicks.press(MouseButton::Right, at(21.), t0 + Duration::from_millis(360)), 1);
        // Too late.
        let late = t0 + Duration::from_millis(360) + DOUBLE_CLICK_INTERVAL;
        assert_eq!(clicks.press(MouseButton::Right, at(21.), late), 1);
    }

    #[test]
    fn wheel_lines_and_pixels() {
        match scroll_delta(MouseScrollDelta::LineDelta(0., -1.), 1.5) {
            ScrollDelta::Lines(p) => assert_eq!((p.x, p.y), (0., -SCROLL_LINES)),
            other => panic!("{other:?}"),
        }
        match scroll_delta(MouseScrollDelta::PixelDelta(PhysicalPosition::new(3., 30.)), 1.5) {
            ScrollDelta::Pixels(p) => assert_eq!((p.x, p.y), (px(2.), px(20.))),
            other => panic!("{other:?}"),
        }
        assert!(matches!(touch_phase(WinitPhase::Cancelled), TouchPhase::Ended));
    }

    #[test]
    fn buttons_and_cursors() {
        assert_eq!(button(WinitButton::Back), Some(MouseButton::Navigate(NavigationDirection::Back)));
        assert_eq!(button(WinitButton::Other(9)), None);
        assert_eq!(cursor_icon(CursorStyle::IBeam), Some(CursorIcon::Text));
        assert_eq!(cursor_icon(CursorStyle::ResizeUpLeftDownRight), Some(CursorIcon::NwseResize));
        assert_eq!(cursor_icon(CursorStyle::None), None);
    }

    #[test]
    fn file_drops_become_gpui_events() {
        let mut drops = FileDropTracker::default();
        let at = point(px(5.), px(6.));
        drops.hovered("a.png".into());
        drops.hovered("b.png".into());
        let events = drops.flush(at);
        assert!(matches!(&events[..], [FileDropEvent::Entered { paths, .. }] if paths.paths().len() == 2));
        assert!(drops.is_active());
        drops.dropped("a.png".into());
        drops.dropped("b.png".into());
        assert!(matches!(&drops.flush(at)[..], [FileDropEvent::Submit { .. }]));
        assert!(!drops.is_active());

        // A drop with no hover first (some systems) still enters, then submits.
        drops.dropped("c.png".into());
        let events = drops.flush(at);
        assert!(matches!(&events[..], [FileDropEvent::Entered { .. }, FileDropEvent::Submit { .. }]));

        drops.hovered("d.png".into());
        drops.flush(at);
        drops.cancelled();
        assert!(matches!(&drops.flush(at)[..], [FileDropEvent::Exited]));
        assert!(drops.flush(at).is_empty());
    }
}
