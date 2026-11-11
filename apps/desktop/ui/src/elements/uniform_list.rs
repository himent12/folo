//! `uniform_list`: a virtual list of equal-height rows that renders only the visible range, with
//! `UniformListScrollHandle` and `ScrollStrategy`. Owner tag: `"elements"`. Row height is measured from the
//! first row, as GPUI does.

use std::{cell::RefCell, ops::Range, panic, rc::Rc};

use smallvec::SmallVec;

use crate::{
    AnyElement, App, AvailableSpace, Bounds, ContentMask, Edges, Element, ElementId, GlobalElementId, Hitbox,
    InspectorElementId, InteractiveElement, Interactivity, IntoElement, IsZero, LayoutId, Overflow, Pixels,
    ScrollHandle, Size, StyleRefinement, Styled, Window, point, size,
};

/// A list of `item_count` rows; `f` renders the rows in a range.
pub fn uniform_list<R>(
    id: impl Into<ElementId>,
    item_count: usize,
    f: impl 'static + Fn(Range<usize>, &mut Window, &mut App) -> Vec<R>,
) -> UniformList
where
    R: IntoElement,
{
    // It scrolls vertically (GPUI sets `overflow.y = Scroll` on its base style).
    let mut interactivity = Interactivity { element_id: Some(id.into()), ..Default::default() };
    interactivity.base_style.overflow.y = Some(Overflow::Scroll);
    UniformList {
        item_count,
        render_items: Box::new(move |range, window, cx| {
            f(range, window, cx).into_iter().map(|item| item.into_any_element()).collect()
        }),
        interactivity,
        scroll_handle: None,
    }
}

/// Renders the rows in a range.
pub(crate) type RenderItems =
    Box<dyn Fn(Range<usize>, &mut Window, &mut App) -> smallvec::SmallVec<[AnyElement; 64]>>;

/// The list element.
pub struct UniformList {
    /// How many rows.
    pub(crate) item_count: usize,
    /// Renders a range of rows.
    pub(crate) render_items: RenderItems,
    /// Styles and listeners (it scrolls vertically).
    pub(crate) interactivity: Interactivity,
    /// The handle it reports to.
    pub(crate) scroll_handle: Option<UniformListScrollHandle>,
}

/// What `UniformList::request_layout` keeps.
pub struct UniformListFrameState {
    /// The measured row height.
    pub(crate) item_height: Pixels,
    /// The rows rendered this frame.
    pub(crate) items: smallvec::SmallVec<[AnyElement; 32]>,
}

/// Where `scroll_to_item` puts the item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScrollStrategy {
    /// At the top (or as close as possible).
    #[default]
    Top,
    /// In the centre.
    Center,
    /// At the bottom.
    Bottom,
}

/// The state behind a `UniformListScrollHandle`.
#[derive(Debug, Default)]
pub struct UniformListScrollState {
    /// The underlying scroll handle.
    pub base_handle: ScrollHandle,
    /// A scroll-to request for the next frame.
    pub deferred_scroll_to_item: Option<(usize, ScrollStrategy)>,
    /// The last row height.
    pub last_item_height: Option<Pixels>,
}

/// A handle to a `uniform_list`'s scroll state.
#[derive(Clone, Debug, Default)]
pub struct UniformListScrollHandle(pub Rc<RefCell<UniformListScrollState>>);

impl UniformListScrollHandle {
    /// A new handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Scrolls row `ix` into view with `strategy` at the next frame.
    pub fn scroll_to_item(&self, ix: usize, strategy: ScrollStrategy) {
        self.0.borrow_mut().deferred_scroll_to_item = Some((ix, strategy));
    }
}

impl UniformList {
    /// Reports to (and takes scroll requests from) `handle`.
    pub fn track_scroll(mut self, handle: UniformListScrollHandle) -> Self {
        self.interactivity.tracked_scroll_handle = Some(handle.0.borrow().base_handle.clone());
        self.scroll_handle = Some(handle);
        self
    }
}

impl Styled for UniformList {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl InteractiveElement for UniformList {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

impl UniformList {
    /// Lays out the first row on its own (GPUI 0.2.2 measures row 0: every row is assumed that tall).
    fn measure_item(&self, list_width: Option<Pixels>, window: &mut Window, cx: &mut App) -> Size<Pixels> {
        if self.item_count == 0 {
            return Size::default();
        }
        let mut items = (self.render_items)(0..1, window, cx);
        let Some(mut item) = items.pop() else {
            return Size::default();
        };
        let available_space = size(
            list_width.map_or(AvailableSpace::MinContent, AvailableSpace::Definite),
            AvailableSpace::MinContent,
        );
        item.layout_as_root(available_space, window, cx)
    }
}

impl Element for UniformList {
    type RequestLayoutState = UniformListFrameState;
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, UniformListFrameState) {
        let max_items = self.item_count;
        let item_size = self.measure_item(None, window, cx);
        // GPUI's `ListSizingBehavior::Infer`: as tall as all rows, capped by the space available.
        let layout_id = self.interactivity.request_layout(global_id, inspector_id, window, cx, |style, window, _| {
            window.with_text_style(style.text_style().cloned(), |window| {
                window.request_measured_layout(style, move |known, available, _, _| {
                    let desired_height = item_size.height * max_items as f32;
                    let width = known.width.unwrap_or(match available.width {
                        AvailableSpace::Definite(width) => width,
                        AvailableSpace::MinContent | AvailableSpace::MaxContent => item_size.width,
                    });
                    let height = match available.height {
                        AvailableSpace::Definite(height) => desired_height.min(height),
                        AvailableSpace::MinContent | AvailableSpace::MaxContent => desired_height,
                    };
                    size(width, height)
                })
            })
        });
        (layout_id, UniformListFrameState { item_height: item_size.height, items: SmallVec::new() })
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        frame_state: &mut UniformListFrameState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Hitbox> {
        let style = self.interactivity.compute_style(global_id, None, window, cx);
        let border = style.border_widths.to_pixels(window.rem_size());
        let padding = style.padding.to_pixels(bounds.size.into(), window.rem_size());
        let padded_bounds = Bounds::from_corners(
            bounds.origin + point(border.left + padding.left, border.top + padding.top),
            bounds.bottom_right() - point(border.right + padding.right, border.bottom + padding.bottom),
        );

        let item_height = self.measure_item(None, window, cx).height;
        frame_state.item_height = item_height;
        let content_height = item_height * self.item_count as f32 + padding.top + padding.bottom;
        let content_size = Size { width: padded_bounds.size.width, height: content_height };

        let shared_scroll_offset =
            self.interactivity.scroll_offset.clone().expect("a uniform list scrolls (overflow-y: scroll)");
        let deferred_scroll = self.scroll_handle.as_ref().and_then(|handle| {
            let mut state = handle.0.borrow_mut();
            state.last_item_height = Some(item_height);
            state.deferred_scroll_to_item.take()
        });
        let item_count = self.item_count;
        let render_items = &self.render_items;
        let items = &mut frame_state.items;

        self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            content_size,
            window,
            cx,
            |_style, mut scroll_offset, hitbox, window, cx| {
                if item_count == 0 {
                    return hitbox;
                }
                let list_height = padded_bounds.size.height;
                let min_scroll_offset = list_height - content_height;
                if !scroll_offset.y.is_zero() && scroll_offset.y < min_scroll_offset {
                    shared_scroll_offset.borrow_mut().y = min_scroll_offset;
                    scroll_offset.y = min_scroll_offset;
                }
                if let Some((ix, strategy)) = deferred_scroll {
                    let mut offset = shared_scroll_offset.borrow_mut();
                    let geometry = ListGeometry { item_height, padding, list_height, content_height };
                    offset.y = geometry.scroll_to_item(ix, strategy, offset.y);
                    scroll_offset = *offset;
                }

                let first = (-(scroll_offset.y + padding.top) / item_height).floor().max(0.) as usize;
                let last = ((-scroll_offset.y + list_height) / item_height).ceil().max(0.) as usize;
                let visible_range = first..last.min(item_count);
                let rendered = render_items(visible_range.clone(), window, cx);

                window.with_content_mask(Some(ContentMask { bounds }), |window| {
                    for (mut item, ix) in rendered.into_iter().zip(visible_range) {
                        let item_origin = padded_bounds.origin
                            + point(scroll_offset.x, item_height * ix as f32 + scroll_offset.y + padding.top);
                        let available_space = size(
                            AvailableSpace::Definite(padded_bounds.size.width),
                            AvailableSpace::Definite(item_height),
                        );
                        item.layout_as_root(available_space, window, cx);
                        item.prepaint_at(item_origin, window, cx);
                        items.push(item);
                    }
                });
                hitbox
            },
        )
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        frame_state: &mut UniformListFrameState,
        hitbox: &mut Option<Hitbox>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let items = &mut frame_state.items;
        self.interactivity.paint(global_id, inspector_id, bounds, hitbox.as_ref(), window, cx, |_, window, cx| {
            for item in items.iter_mut() {
                item.paint(window, cx);
            }
        })
    }
}

impl IntoElement for UniformList {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// A list's vertical geometry for `scroll_to_item`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ListGeometry {
    /// Every row's height.
    pub(crate) item_height: Pixels,
    /// The list's padding.
    pub(crate) padding: Edges<Pixels>,
    /// The height inside border and padding.
    pub(crate) list_height: Pixels,
    /// All rows plus the vertical padding.
    pub(crate) content_height: Pixels,
}

impl ListGeometry {
    /// The scroll offset (`y`, negative going down) after `scroll_to_item(ix, strategy)`: GPUI 0.2.2's rule
    /// (no extra offset, not strict). A row already in view does not move the list; a row that was out of view
    /// is placed where `strategy` says, clamped to the content.
    pub(crate) fn scroll_to_item(&self, ix: usize, strategy: ScrollStrategy, offset_y: Pixels) -> Pixels {
        let Self { item_height, padding, list_height, content_height } = *self;
        let item_top = item_height * ix as f32 + padding.top;
        let item_bottom = item_top + item_height;
        let scroll_top = -offset_y;
        let mut offset_y = offset_y;
        let mut scrolled = false;
        if item_top < scroll_top + padding.top {
            scrolled = true;
            offset_y = -item_top + padding.top;
        } else if item_bottom > scroll_top + list_height - padding.bottom {
            scrolled = true;
            offset_y = -(item_bottom - list_height) - padding.bottom;
        }
        if scrolled && (item_top < scroll_top || item_bottom > scroll_top + list_height) {
            let max_scroll = content_height - list_height;
            let target = match strategy {
                ScrollStrategy::Top => item_top,
                ScrollStrategy::Center => item_top + item_height / 2.0 - list_height / 2.0,
                ScrollStrategy::Bottom => item_bottom - list_height,
            };
            offset_y = -target.max(Pixels::ZERO).min(max_scroll).max(Pixels::ZERO);
        }
        offset_y
    }
}
