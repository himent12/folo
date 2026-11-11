//! Regressions found by driving the real app in a real window (`apps/desktop/tools/interact-check.ps1`),
//! each reduced to what the headless harness can express.

use std::time::Duration;

use super::*;
use crate::{InteractiveElement, ParentElement, StatefulInteractiveElement, Styled, div};

/// A tooltip element the mouse leaves before the show delay: cancelling the waiting timer re-borrowed
/// the tooltip state inside a `match` on its own borrow and panicked ("already borrowed"), which aborted
/// the app on the first sidebar hover.
#[test]
fn leaving_a_tooltip_element_before_the_delay_cancels_it() {
    let (mut app, window) = open(size(px(300.), px(200.)), 1.0, |_, _| {
        div()
            .size_full()
            .child(div().id("tipped").w(px(50.)).h(px(20.)).hover(|s| s.bg(color(3))).tooltip(|_, cx| {
                let render: RenderFn =
                    Rc::new(|_, _| div().w(px(30.)).h(px(10.)).bg(color(9)).into_any_element());
                cx.new(|_| TestView { render }).into()
            }))
            .into_any_element()
    });
    mouse_move(&mut app, window, 200., 150.);
    app.frame(window);
    mouse_move(&mut app, window, 10., 10.);
    app.frame(window);
    app.advance_clock(Duration::from_millis(200));
    mouse_move(&mut app, window, 200., 150.);
    app.frame(window);
    app.advance_clock(Duration::from_millis(600));
    app.frame(window);
    assert!(quads_of(&mut app, window, color(9)).is_empty(), "the cancelled tooltip never shows");
    // And it still works the next time.
    mouse_move(&mut app, window, 10., 10.);
    app.advance_clock(Duration::from_millis(500));
    app.frame(window);
    assert_eq!(quads_of(&mut app, window, color(9)).len(), 1);
}
