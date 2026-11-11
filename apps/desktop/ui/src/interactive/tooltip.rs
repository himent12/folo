//! Tooltips, as GPUI 0.2.2's `div.rs`: hovering an element with a tooltip starts a 500 ms timer; when it
//! fires (and the element is still hovered, no mouse button down) the tooltip becomes visible and the
//! window redraws; the window asks it each frame (`check_visible_and_update`) whether it is still wanted,
//! and it hides as soon as the mouse leaves its element (the app's tooltips are not hoverable). A mouse
//! down or a scroll anywhere outside the tooltip hides it too. Owner tag: `"window"`.
//!
//! Two departures from GPUI 0.2.2, both fixing its bugs: a mouse move that leaves the element hides a
//! shown tooltip at once (GPUI's mouse handler leaves that to the next prepaint, so without a hover style
//! or another redraw the tooltip stayed up), and the show timer and the shown tooltip hold the state
//! weakly (GPUI's held it strongly from inside it, a cycle that leaked the state and the tooltip's view
//! once the element went away).

use std::{
    cell::RefCell,
    rc::{Rc, Weak},
    time::Duration,
};

use crate::{
    AnyTooltip, App, AsyncApp, Bounds, DispatchPhase, MouseDownEvent, MouseMoveEvent, Pixels,
    ScrollWheelEvent, Task, TooltipId, Window, interactive::TooltipBuilder,
};

/// GPUI's `TOOLTIP_SHOW_DELAY`.
pub(crate) const TOOLTIP_SHOW_DELAY: Duration = Duration::from_millis(500);

/// A tooltip's state (GPUI's `ActiveTooltip`, without the hoverable variants the app does not use).
pub(crate) enum ActiveTooltip {
    /// Waiting for the show delay.
    WaitingForShow {
        /// The timer (dropping it cancels the show).
        _task: Task<()>,
    },
    /// Shown.
    Visible {
        /// What to show.
        tooltip: AnyTooltip,
    },
}

/// The shared state cell of one element's tooltip.
pub(crate) type TooltipState = Rc<RefCell<Option<ActiveTooltip>>>;

/// Hides the tooltip (redrawing if it was shown).
fn clear_active_tooltip(active_tooltip: &TooltipState, window: &mut Window) {
    if let Some(ActiveTooltip::Visible { .. }) = active_tooltip.borrow_mut().take() {
        window.refresh();
    }
}

/// Asks the window to show the tooltip this frame if it is visible; returns its request id.
pub(crate) fn set_tooltip_on_window(active_tooltip: &TooltipState, window: &mut Window) -> Option<TooltipId> {
    let tooltip = match active_tooltip.borrow().as_ref() {
        Some(ActiveTooltip::Visible { tooltip }) => tooltip.clone(),
        _ => return None,
    };
    Some(window.set_tooltip(tooltip))
}

/// Registers the tooltip's mouse listeners for this frame (GPUI's `register_tooltip_mouse_handlers`).
/// `check_is_hovered` tests the element's hitbox; `check_is_hovered_during_prepaint` tests its bounds
/// (hitboxes are not final during prepaint).
pub(crate) fn register_tooltip_mouse_handlers(
    active_tooltip: &TooltipState,
    tooltip_id: Option<TooltipId>,
    build_tooltip: TooltipBuilder,
    check_is_hovered: Rc<dyn Fn(&Window) -> bool>,
    check_is_hovered_during_prepaint: Rc<dyn Fn(&Window) -> bool>,
    window: &mut Window,
) {
    window.on_mouse_event({
        let active_tooltip = active_tooltip.clone();
        move |_: &MouseMoveEvent, phase, window, cx| {
            handle_tooltip_mouse_move(
                &active_tooltip,
                &build_tooltip,
                &check_is_hovered,
                &check_is_hovered_during_prepaint,
                phase,
                window,
                cx,
            )
        }
    });
    window.on_mouse_event({
        let active_tooltip = active_tooltip.clone();
        move |_: &MouseDownEvent, _, window, _| {
            if !tooltip_id.is_some_and(|id| id.is_hovered(window)) {
                clear_active_tooltip(&active_tooltip, window);
            }
        }
    });
    window.on_mouse_event({
        let active_tooltip = active_tooltip.clone();
        move |_: &ScrollWheelEvent, _, window, _| {
            if !tooltip_id.is_some_and(|id| id.is_hovered(window)) {
                clear_active_tooltip(&active_tooltip, window);
            }
        }
    });
}

/// A mouse move: start the show timer when the element becomes hovered (bubble phase), cancel it when
/// the mouse leaves before it fires, hide the tooltip when it leaves after.
fn handle_tooltip_mouse_move(
    active_tooltip: &TooltipState,
    build_tooltip: &TooltipBuilder,
    check_is_hovered: &Rc<dyn Fn(&Window) -> bool>,
    check_is_hovered_during_prepaint: &Rc<dyn Fn(&Window) -> bool>,
    phase: DispatchPhase,
    window: &mut Window,
    cx: &mut App,
) {
    // Read the state into a value first: a `match` on `borrow()` would hold the borrow through the arms.
    let (waiting, visible) = match active_tooltip.borrow().as_ref() {
        None => (false, false),
        Some(ActiveTooltip::WaitingForShow { .. }) => (true, false),
        Some(ActiveTooltip::Visible { .. }) => (false, true),
    };
    if visible {
        if !check_is_hovered(window) {
            clear_active_tooltip(active_tooltip, window);
        }
        return;
    }
    let schedule = if waiting {
        if !check_is_hovered(window) {
            // Cancelling: dropping the task stops the timer.
            drop(active_tooltip.borrow_mut().take());
        }
        false
    } else {
        check_is_hovered(window) && phase.bubble()
    };
    if !schedule {
        return;
    }
    let handle = window.window_handle();
    // The task is stored in the state it updates: it holds the state weakly, so an element that goes
    // away before the delay ends frees both (and nothing is built).
    let task = cx.spawn({
        let weak_tooltip = Rc::downgrade(active_tooltip);
        let build_tooltip = build_tooltip.clone();
        let check_is_hovered = check_is_hovered_during_prepaint.clone();
        async move |cx: &mut AsyncApp| {
            cx.background_executor().timer(TOOLTIP_SHOW_DELAY).await;
            let Some(active_tooltip) = weak_tooltip.upgrade() else { return };
            let _ = handle.update(cx, |_, window, cx| {
                let view = build_tooltip(window, cx);
                // The callback is stored in the state too: weak for the same reason.
                let state = weak_tooltip.clone();
                let check_visible_and_update: crate::interactive::TooltipVisibility =
                    Rc::new(move |bounds, window, _cx| {
                        check_visible_and_update(&state, &check_is_hovered, bounds, window)
                    });
                let tooltip =
                    AnyTooltip { view, mouse_position: window.mouse_position(), check_visible_and_update };
                *active_tooltip.borrow_mut() = Some(ActiveTooltip::Visible { tooltip });
                window.refresh();
            });
        }
    });
    *active_tooltip.borrow_mut() = Some(ActiveTooltip::WaitingForShow { _task: task });
}

/// Called by the window at each prepaint of a shown tooltip: hides it once its element is not hovered.
fn check_visible_and_update(
    active_tooltip: &Weak<RefCell<Option<ActiveTooltip>>>,
    check_is_hovered: &Rc<dyn Fn(&Window) -> bool>,
    _tooltip_bounds: Bounds<Pixels>,
    window: &mut Window,
) -> bool {
    let Some(active_tooltip) = active_tooltip.upgrade() else { return false };
    let active_tooltip = &active_tooltip;
    let visible = matches!(active_tooltip.borrow().as_ref(), Some(ActiveTooltip::Visible { .. }));
    if visible && !check_is_hovered(window) {
        clear_active_tooltip(active_tooltip, window);
    }
    active_tooltip.borrow().is_some()
}

impl TooltipId {
    /// Whether the mouse is over the tooltip shown this frame with this id.
    pub fn is_hovered(&self, window: &Window) -> bool {
        window
            .tooltip_bounds
            .as_ref()
            .is_some_and(|(id, bounds)| id == self && bounds.contains(&window.mouse_position()))
    }
}
