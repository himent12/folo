// Adapted from GPUI (Apache-2.0): crates/gpui/src/elements/text.rs (the measure function)
//! Text as elements: `&'static str`, `String`, `SharedString`, and `StyledText` (text with runs). Owner tag:
//! `"elements"`. Shaping, wrapping and truncation are the text package's (`WindowTextSystem::shape_text`,
//! `LineWrapper`); this element measures through `request_measured_layout`, keeps the shaped lines from
//! prepaint to paint, and paints them through the window's `GlyphPainter`.
//!
//! Performance: text is `SharedString` (never copied per frame); shaped layouts come from the window's line
//! layout cache, which keeps what the previous frame used.

use std::{cell::RefCell, panic, rc::Rc};

use smallvec::SmallVec;

use crate::{
    App, AvailableSpace, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement,
    LayoutId, Pixels, SharedString, Size, TextAlign, TextOverflow, TextRun, WhiteSpace, Window, WrappedLine,
    text::fx_hash::FxHasher, window::MeasureKey,
};

/// Text with explicit runs (fonts, colours, underlines per byte range).
pub struct StyledText {
    /// The text.
    pub(crate) text: SharedString,
    /// The runs, if given (else one run in the inherited text style).
    pub(crate) runs: Option<Vec<TextRun>>,
    /// The layout measured in `request_layout`, shared with prepaint / paint.
    pub(crate) layout: TextLayout,
}

impl StyledText {
    /// Text in the inherited style.
    pub fn new(text: impl Into<SharedString>) -> Self {
        StyledText { text: text.into(), runs: None, layout: TextLayout::default() }
    }

    /// Sets the runs (their lengths must sum to the text's byte length).
    pub fn with_runs(mut self, runs: Vec<TextRun>) -> Self {
        self.runs = Some(runs);
        self
    }

    /// The layout of this element (its lines once measured).
    pub fn layout(&self) -> &TextLayout {
        &self.layout
    }
}

/// The shaped, wrapped lines of a text element this frame, shared between its phases.
#[derive(Clone, Default)]
pub struct TextLayout(pub(crate) Rc<RefCell<Option<TextLayoutInner>>>);

/// What `TextLayout` holds once measured.
pub(crate) struct TextLayoutInner {
    /// The wrapped lines (the text package's type).
    pub(crate) lines: SmallVec<[WrappedLine; 1]>,
    /// The line height used.
    pub(crate) line_height: Pixels,
    /// The width the lines were wrapped at (a later measure at the same width reuses them, as GPUI).
    pub(crate) wrap_width: Option<Pixels>,
    /// The width the text was truncated to, if any (a later measure at a different width re-measures).
    pub(crate) truncate_width: Option<Pixels>,
    /// The measured size.
    pub(crate) size: Option<Size<Pixels>>,
    /// The element's bounds once prepainted.
    pub(crate) bounds: Option<Bounds<Pixels>>,
}

impl TextLayout {
    /// Requests a measured layout node for `text` in the window's current text style (GPUI's measure:
    /// wrap at the known or definite width when white space is normal; truncate to the width, times the
    /// line clamp, when the style truncates; the size is the widest line rounded up and the lines' height).
    fn layout(
        &self,
        text: SharedString,
        runs: Option<Vec<TextRun>>,
        window: &mut Window,
        _cx: &mut App,
    ) -> LayoutId {
        let text_style = window.text_style();
        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = text_style.line_height.to_pixels(font_size.into(), window.rem_size());
        let element_state = self.clone();
        let key = measure_key(&text, font_size, line_height, runs.as_deref(), &text_style);
        let reset_state = self.clone();
        let reset = move || *reset_state.0.borrow_mut() = None;

        window.request_keyed_measured_layout(
            Default::default(),
            key,
            reset,
            move |known, available, window, cx| {
                let definite_width = match available.width {
                    AvailableSpace::Definite(width) => Some(width),
                    AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
                };
                let wrap_width = if text_style.white_space == WhiteSpace::Normal {
                    known.width.or(definite_width)
                } else {
                    None
                };
                let (truncate_width, suffix): (Option<Pixels>, &str) = match &text_style.text_overflow {
                    Some(TextOverflow::Truncate(suffix)) => {
                        let width = known.width.or(definite_width.map(|width| match text_style.line_clamp {
                            Some(max_lines) => width * max_lines as f32,
                            None => width,
                        }));
                        (width, suffix.as_str())
                    }
                    None => (None, ""),
                };

                if let Some(layout) = element_state.0.borrow().as_ref()
                    && let Some(size) = layout.size
                    && wrap_width == layout.wrap_width
                    && truncate_width == layout.truncate_width
                {
                    return size;
                }

                let shaped = match runs.as_deref() {
                    None => {
                        let text = match truncate_width {
                            Some(width) => {
                                let mut wrapper = cx
                                    .text_system()
                                    .line_wrapper_for_key(text_style.font_key_ref(), font_size);
                                wrapper.truncate_line_text(text.clone(), width, suffix).0
                            }
                            None => text.clone(),
                        };
                        window.text_system().shape_text_with_style(
                            text,
                            font_size,
                            &text_style,
                            wrap_width,
                            text_style.line_clamp,
                        )
                    }
                    Some(explicit_runs) => {
                        let (text, run_slice) = match truncate_width {
                            Some(width) => {
                                let mut wrapper = cx
                                    .text_system()
                                    .line_wrapper_for_key(text_style.font_key_ref(), font_size);
                                wrapper.truncate_line_cow(text.clone(), width, suffix, explicit_runs)
                            }
                            None => (text.clone(), std::borrow::Cow::Borrowed(explicit_runs)),
                        };
                        window.text_system().shape_text(
                            text,
                            font_size,
                            run_slice.as_ref(),
                            wrap_width,
                            text_style.line_clamp,
                        )
                    }
                };
                let lines = match shaped {
                    Ok(lines) => lines,
                    Err(error) => {
                        eprintln!("folo-ui: shaping text failed: {error:#}");
                        SmallVec::new()
                    }
                };

                let mut size: Size<Pixels> = Size::default();
                for line in &lines {
                    let line_size = line.size(line_height);
                    size.height += line_size.height;
                    size.width = size.width.max(line_size.width).ceil();
                }
                element_state.0.borrow_mut().replace(TextLayoutInner {
                    lines,
                    line_height,
                    wrap_width,
                    truncate_width,
                    size: Some(size),
                    bounds: None,
                });
                size
            },
        )
    }

    fn prepaint(&self, bounds: Bounds<Pixels>, text: &str) {
        let mut state = self.0.borrow_mut();
        let state = state.as_mut().unwrap_or_else(|| panic!("text {text:?} was not measured"));
        state.bounds = Some(bounds);
    }

    fn paint(&self, text: &str, window: &mut Window, cx: &mut App) {
        let state = self.0.borrow();
        let state = state.as_ref().unwrap_or_else(|| panic!("text {text:?} was not measured"));
        let bounds = state.bounds.unwrap_or_else(|| panic!("text {text:?} was not prepainted"));
        let line_height = state.line_height;
        let align = window.text_style_ref().map_or(TextAlign::default(), |style| style.text_align);
        let mut origin = bounds.origin;
        for line in &state.lines {
            if line.decoration_runs().iter().any(|run| run.background_color.is_some())
                && let Err(error) =
                    line.paint_background(origin, line_height, align, Some(bounds), window, cx)
            {
                eprintln!("folo-ui: painting text background failed: {error:#}");
            }
            if let Err(error) = line.paint(origin, line_height, align, Some(bounds), window, cx) {
                eprintln!("folo-ui: painting text failed: {error:#}");
            }
            origin.y += line.size(line_height).height;
        }
    }

    /// The bounds the text was prepainted at, if it was.
    pub fn bounds(&self) -> Option<Bounds<Pixels>> {
        self.0.borrow().as_ref().and_then(|state| state.bounds)
    }

    /// The line height the text was measured with, if it was.
    pub fn line_height(&self) -> Option<Pixels> {
        self.0.borrow().as_ref().map(|state| state.line_height)
    }

    /// How many visual lines the text wrapped into (0 before it is measured).
    pub fn line_count(&self) -> usize {
        self.0
            .borrow()
            .as_ref()
            .map_or(0, |state| state.lines.iter().map(|line| line.wrap_boundaries().len() + 1).sum())
    }
}

/// What the measure function's results depend on besides the layout requests: the text, its size and
/// line height, the fonts of its runs, and the wrapping and truncation style (colours do not change sizes).
fn measure_key(
    text: &SharedString,
    font_size: Pixels,
    line_height: Pixels,
    runs: Option<&[TextRun]>,
    style: &crate::TextStyle,
) -> MeasureKey {
    use std::hash::{Hash, Hasher};
    let mut h = FxHasher::default();
    text.as_str().hash(&mut h);
    h.write_u32(font_size.0.to_bits());
    h.write_u32(line_height.0.to_bits());
    let style_font = style.font_key_ref();
    match runs {
        Some(runs) => {
            for run in runs {
                h.write_usize(run.len);
                run.font.as_key_ref().hash(&mut h);
            }
        }
        None => {
            h.write_usize(text.len());
            style_font.hash(&mut h);
        }
    }
    style_font.hash(&mut h);
    (style.white_space == WhiteSpace::Normal).hash(&mut h);
    match &style.text_overflow {
        Some(TextOverflow::Truncate(suffix)) => suffix.as_str().hash(&mut h),
        None => h.write_u8(0),
    }
    style.line_clamp.hash(&mut h);
    MeasureKey::Text { hash: h.finish(), text: text.clone() }
}

/// Implements `Element` for a text type: `$text` turns `self` into the `SharedString` to lay out.
macro_rules! text_element {
    ($ty:ty, |$this:ident| $text:expr) => {
        impl Element for $ty {
            type RequestLayoutState = TextLayout;
            type PrepaintState = ();

            fn id(&self) -> Option<ElementId> {
                None
            }

            fn source_location(&self) -> Option<&'static panic::Location<'static>> {
                None
            }

            fn request_layout(
                &mut self,
                _id: Option<&GlobalElementId>,
                _inspector_id: Option<&InspectorElementId>,
                window: &mut Window,
                cx: &mut App,
            ) -> (LayoutId, TextLayout) {
                let $this = &*self;
                let layout = TextLayout::default();
                let layout_id = layout.layout($text, None, window, cx);
                (layout_id, layout)
            }

            fn prepaint(
                &mut self,
                _id: Option<&GlobalElementId>,
                _inspector_id: Option<&InspectorElementId>,
                bounds: Bounds<Pixels>,
                layout: &mut TextLayout,
                _window: &mut Window,
                _cx: &mut App,
            ) {
                layout.prepaint(bounds, self)
            }

            fn paint(
                &mut self,
                _id: Option<&GlobalElementId>,
                _inspector_id: Option<&InspectorElementId>,
                _bounds: Bounds<Pixels>,
                layout: &mut TextLayout,
                _prepaint: &mut (),
                window: &mut Window,
                cx: &mut App,
            ) {
                layout.paint(self, window, cx)
            }
        }

        impl IntoElement for $ty {
            type Element = Self;

            fn into_element(self) -> Self::Element {
                self
            }
        }
    };
}

text_element!(&'static str, |this| SharedString::from(*this));
text_element!(SharedString, |this| this.clone());

impl IntoElement for String {
    type Element = SharedString;

    fn into_element(self) -> Self::Element {
        self.into()
    }
}

impl Element for StyledText {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.layout.layout(self.text.clone(), self.runs.take(), window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        _window: &mut Window,
        _cx: &mut App,
    ) {
        self.layout.prepaint(bounds, &self.text)
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.layout.paint(&self.text, window, cx)
    }
}

impl IntoElement for StyledText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
