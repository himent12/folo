//! `uniform_list`: a virtual list of equal-height rows that renders only the visible range, with
//! `UniformListScrollHandle` and `ScrollStrategy`. Owner tag: `"elements"`. Row height is measured from the
//! first row, as GPUI does.
#![allow(dead_code)] // skeleton

use std::{cell::RefCell, ops::Range, rc::Rc};

use crate::{
    AnyElement, App, ElementId, InteractiveElement, Interactivity, IntoElement, Pixels, ScrollHandle,
    StyleRefinement, Styled, Window, elements::skeleton_element,
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
    // It scrolls vertically: `request_layout` sets `overflow.y = Scroll` on its style.
    let interactivity = Interactivity { element_id: Some(id.into()), ..Default::default() };
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

skeleton_element!([] UniformList, UniformListFrameState, Option<crate::Hitbox>, "elements");
