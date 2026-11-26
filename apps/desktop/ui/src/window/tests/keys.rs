//! Keyboard and focus: GPUI 0.2.2's order (bindings under the focus path's contexts first, then key
//! listeners capture root to leaf and bubble leaf to root, then text input), context-dependent bindings,
//! focus by mouse down, Tab order with `focus_next` / `focus_prev`.

use super::*;
use crate::{
    App, Bounds, Element, ElementId, FocusHandle, GlobalElementId, InputHandler, InspectorElementId,
    InteractiveElement, KeyBinding, KeyDownEvent, Keystroke, LayoutId, ParentElement, Style, Styled,
    UTF16Selection, div,
};

crate::actions!(window_tests, [Alpha, Beta]);

/// An element that registers a text input handler for `focus` (what the app's text field does).
struct InputProbe {
    focus: FocusHandle,
    log: Log,
}

impl IntoElement for InputProbe {
    type Element = Self;

    fn into_element(self) -> Self {
        self
    }
}

impl Element for InputProbe {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (window.request_layout(Style::default(), [], cx), ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut Window,
        _: &mut App,
    ) {
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        window.handle_input(&self.focus, Probe(self.log.clone()), cx);
    }
}

/// Records inserted text.
struct Probe(Log);

impl InputHandler for Probe {
    fn selected_text_range(&mut self, _: bool, _: &mut Window, _: &mut App) -> Option<UTF16Selection> {
        None
    }
    fn marked_text_range(&mut self, _: &mut Window, _: &mut App) -> Option<std::ops::Range<usize>> {
        None
    }
    fn text_for_range(
        &mut self,
        _: std::ops::Range<usize>,
        _: &mut Option<std::ops::Range<usize>>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<String> {
        None
    }
    fn replace_text_in_range(
        &mut self,
        _: Option<std::ops::Range<usize>>,
        text: &str,
        _: &mut Window,
        _: &mut App,
    ) {
        self.0.borrow_mut().push(format!("text {text}"));
    }
    fn replace_and_mark_text_in_range(
        &mut self,
        _: Option<std::ops::Range<usize>>,
        _: &str,
        _: Option<std::ops::Range<usize>>,
        _: &mut Window,
        _: &mut App,
    ) {
    }
    fn unmark_text(&mut self, _: &mut Window, _: &mut App) {}
    fn bounds_for_range(
        &mut self,
        _: std::ops::Range<usize>,
        _: &mut Window,
        _: &mut App,
    ) -> Option<Bounds<Pixels>> {
        None
    }
    fn character_index_for_point(&mut self, _: Point<Pixels>, _: &mut Window, _: &mut App) -> Option<usize> {
        None
    }
}

fn key(app: &mut HeadlessApp, window: AnyWindowHandle, source: &str, key_char: Option<&str>) {
    let mut keystroke = Keystroke::parse(source).unwrap();
    keystroke.key_char = key_char.map(Into::into);
    app.simulate_input(window, PlatformInput::KeyDown(KeyDownEvent { keystroke, is_held: false }));
}

#[test]
fn bindings_then_key_listeners_then_text_input_in_gpui_order() {
    let events = log();
    let mut app = HeadlessApp::new(HeadlessOptions {
        size: size(px(200.), px(100.)),
        scale_factor: 1.,
        text: TextBackend::Fixed,
    });
    let (outer, inner) = app.update(|cx| {
        cx.bind_keys([KeyBinding::new("a", Alpha, Some("Inner")), KeyBinding::new("b", Beta, None)]);
        (cx.focus_handle(), cx.focus_handle())
    });
    let render: RenderFn = Rc::new({
        let (events, outer, inner) = (events.clone(), outer.clone(), inner.clone());
        move |_, _| {
            let l = |s: &'static str| {
                let events = events.clone();
                move || events.borrow_mut().push(s.to_string())
            };
            let (rc, rb, ra, cb, ca) =
                (l("root capture"), l("root key"), l("root beta"), l("child key"), l("child alpha"));
            div()
                .key_context("Root")
                .track_focus(&outer)
                .size_full()
                .capture_key_down(move |_, _, _| rc())
                .on_key_down(move |_, _, _| rb())
                .on_action(move |_: &Beta, _, _| ra())
                .child(
                    div()
                        .key_context("Inner")
                        .track_focus(&inner)
                        .w(px(50.))
                        .h(px(20.))
                        .on_key_down(move |_, _, _| cb())
                        .on_action(move |_: &Alpha, _, _| ca())
                        .child(InputProbe { focus: inner.clone(), log: events.clone() }),
                )
                .into_any_element()
        }
    });
    let window: AnyWindowHandle = app
        .update(|cx| cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| TestView { render })))
        .unwrap()
        .into();
    with_window(&mut app, window, |window, _| window.focus(&inner));
    app.frame(window);

    key(&mut app, window, "a", Some("a"));
    assert_eq!(*events.borrow(), ["child alpha"], "a binding in the focused context wins; nothing else runs");
    events.borrow_mut().clear();
    key(&mut app, window, "b", Some("b"));
    assert_eq!(*events.borrow(), ["root beta"], "a context-free binding, handled up the path");
    events.borrow_mut().clear();
    key(&mut app, window, "x", Some("x"));
    assert_eq!(*events.borrow(), ["root capture", "child key", "root key", "text x"]);
    events.borrow_mut().clear();
    key(&mut app, window, "ctrl-x", Some("x"));
    assert_eq!(*events.borrow(), ["root capture", "child key", "root key"], "no text with ctrl held");

    // Focus on the outer element: the "Inner" binding no longer applies, so "a" is just a key.
    events.borrow_mut().clear();
    with_window(&mut app, window, |window, _| window.focus(&outer));
    key(&mut app, window, "a", Some("a"));
    assert_eq!(*events.borrow(), ["root capture", "root key"]);
}

#[test]
fn tab_order_and_focus_on_mouse_down() {
    let mut app = HeadlessApp::new(HeadlessOptions {
        size: size(px(300.), px(100.)),
        scale_factor: 1.,
        text: TextBackend::Fixed,
    });
    let handles: Vec<FocusHandle> =
        app.update(|cx| (0..4).map(|i| cx.focus_handle().tab_stop(i != 2)).collect());
    let render: RenderFn = Rc::new({
        let handles = handles.clone();
        move |_, _| {
            div()
                .flex()
                .size_full()
                .children(handles.iter().map(|handle| div().track_focus(handle).w(px(50.)).h(px(20.))))
                .into_any_element()
        }
    });
    let window: AnyWindowHandle = app
        .update(|cx| cx.open_window(WindowOptions::default(), |_, cx| cx.new(|_| TestView { render })))
        .unwrap()
        .into();
    app.frame(window);
    let focused = |app: &mut HeadlessApp| {
        with_window(app, window, |window, cx| {
            window.focused(cx).map(|handle| handles.iter().position(|h| *h == handle).unwrap())
        })
    };
    let next = |app: &mut HeadlessApp, reverse: bool| {
        with_window(app, window, |window, _| if reverse { window.focus_prev() } else { window.focus_next() });
        app.frame(window);
        focused(app)
    };
    assert_eq!(next(&mut app, false), Some(0));
    assert_eq!(next(&mut app, false), Some(1));
    assert_eq!(next(&mut app, false), Some(3), "a handle that is not a tab stop is skipped");
    assert_eq!(next(&mut app, false), Some(0), "wraps");
    assert_eq!(next(&mut app, true), Some(3));
    // A mouse down on a focusable element focuses it (even one Tab skips).
    mouse_down(&mut app, window, 110., 10.);
    assert_eq!(focused(&mut app), Some(2));
    app.frame(window);
    assert_eq!(next(&mut app, false), Some(3), "Tab continues from a focused non-stop");
    with_window(&mut app, window, |window, cx| {
        assert!(handles[3].is_focused(window));
        assert!(!handles[0].contains_focused(window, cx));
        assert!(window.is_focus_mounted(&handles[0]));
        let orphan = cx.focus_handle();
        assert!(!window.is_focus_mounted(&orphan));
    });
}
