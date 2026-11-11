//! What one paint produces: `Frame` (the scene, hitboxes, listeners, the key dispatch tree, cursor styles,
//! the input handler, tooltip requests, window control areas, element states) and the types in it.
//! Owner tag: `"window"`.
//!
//! Performance: the window keeps two frames (`rendered_frame`, `next_frame`) and swaps them after each draw;
//! `Frame::clear` empties every list but keeps its capacity, so a steady-state frame allocates nothing a
//! reused `Vec` avoids. Element states are moved, not cloned, from the previous frame when touched again.
#![allow(dead_code)] // skeleton

use std::{any::Any, collections::HashMap};

use smallvec::SmallVec;

use crate::{
    AnyTooltip, App, Bounds, CursorStyle, DispatchTree, FocusId, GlobalElementId, Pixels, Scene,
    WindowControlArea, window::input_handler::InputHandlerSlot,
};

/// A hitbox's id in its frame (its index in insertion order, so a later id is on top).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HitboxId(pub(crate) usize);

impl HitboxId {
    /// Whether the mouse is over this hitbox and nothing above it blocks it (last frame's hit test).
    pub fn is_hovered(self, window: &crate::Window) -> bool {
        window.mouse_hit_test.ids.contains(&self)
    }
}

/// How a hitbox affects the ones under it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HitboxBehavior {
    /// Others under it still receive mouse events.
    #[default]
    Normal,
    /// Hitboxes under it are not hovered and get no mouse events (`occlude`).
    BlockMouse,
    /// As `BlockMouse`, except scroll wheel events pass through.
    BlockMouseExceptScroll,
}

/// A rectangle that takes part in hit testing, clipped by the content mask it was inserted under.
#[derive(Clone, Debug)]
pub struct Hitbox {
    /// Its id.
    pub id: HitboxId,
    /// Its bounds (window logical px).
    pub bounds: Bounds<Pixels>,
    /// The content mask in effect.
    pub content_mask: crate::ContentMask<Pixels>,
    /// How it blocks hitboxes below.
    pub behavior: HitboxBehavior,
}

impl Hitbox {
    /// Whether the mouse is over it (and not blocked).
    pub fn is_hovered(&self, window: &crate::Window) -> bool {
        self.id.is_hovered(window)
    }
}

/// The result of hit testing the mouse position against the rendered frame: hovered hitboxes, topmost
/// first, up to and including the first blocking one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct HitTest {
    /// Hovered hitbox ids, topmost first.
    pub(crate) ids: SmallVec<[HitboxId; 8]>,
    /// How many of `ids` get scroll events (all, unless a `BlockMouseExceptScroll` cut the list).
    pub(crate) hover_hitbox_count: usize,
}

/// Capture runs root to leaf, bubble leaf to root (GPUI's `DispatchPhase`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DispatchPhase {
    /// Leaf to root; the default for listeners.
    #[default]
    Bubble,
    /// Root to leaf, before bubble.
    Capture,
}

impl DispatchPhase {
    /// Whether this is the bubble phase.
    pub fn bubble(self) -> bool {
        self == DispatchPhase::Bubble
    }

    /// Whether this is the capture phase.
    pub fn capture(self) -> bool {
        self == DispatchPhase::Capture
    }
}

/// A mouse listener registered during paint (`Window::on_mouse_event`).
pub(crate) type AnyMouseListener =
    Box<dyn FnMut(&dyn Any, DispatchPhase, &mut crate::Window, &mut App) + 'static>;

/// A cursor style request for a hitbox (or for the whole window when `hitbox_id` is `None`).
#[derive(Clone, Debug)]
pub(crate) struct CursorStyleRequest {
    /// The hitbox the style applies over; `None` for window-wide.
    pub(crate) hitbox_id: Option<HitboxId>,
    /// The style.
    pub(crate) style: CursorStyle,
}

/// A tooltip an element asked for this frame (`set_tooltip`).
#[derive(Clone)]
pub(crate) struct TooltipRequest {
    /// The hitbox it belongs to.
    pub(crate) id: TooltipId,
    /// What to show and where.
    pub(crate) tooltip: AnyTooltip,
}

/// A tooltip request's id in its frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TooltipId(pub(crate) usize);

/// A window control area (caption, minimize, maximize, close) painted this frame.
#[derive(Clone, Debug)]
pub(crate) struct WindowControlHitbox {
    /// Which control.
    pub(crate) area: WindowControlArea,
    /// Its hitbox.
    pub(crate) hitbox: Hitbox,
}

/// A deferred draw (`deferred(..)`), painted after the main tree in priority order.
pub(crate) struct DeferredDraw {
    /// Higher draws later (on top).
    pub(crate) priority: usize,
    /// The dispatch node of the element that deferred it.
    pub(crate) parent_node: crate::window::dispatch::DispatchNodeId,
    /// The element id path at the point it was deferred.
    pub(crate) element_id_stack: crate::element::ElementIdStack,
    /// The text style stack at that point.
    pub(crate) text_style_stack: Vec<crate::TextStyleRefinement>,
    /// The element.
    pub(crate) element: Option<crate::AnyElement>,
    /// Its absolute offset.
    pub(crate) absolute_offset: crate::Point<Pixels>,
}

/// Element state kept across frames, by element id path and state type.
pub(crate) type ElementStateMap = HashMap<(GlobalElementId, std::any::TypeId), Box<dyn Any>>;

/// Everything one paint produced.
pub(crate) struct Frame {
    /// The focused element when this frame was painted.
    pub(crate) focus: Option<FocusId>,
    /// Element states touched this frame (the rest of the previous frame's are dropped at the swap).
    pub(crate) element_states: ElementStateMap,
    /// Hit-testable rectangles, in paint order.
    pub(crate) hitboxes: Vec<Hitbox>,
    /// Mouse listeners, in paint order (dispatched topmost first in bubble).
    pub(crate) mouse_listeners: Vec<Option<AnyMouseListener>>,
    /// The key dispatch tree: key contexts, key / action listeners, focus ids, tab stops.
    pub(crate) dispatch_tree: DispatchTree,
    /// The display list for the renderer.
    pub(crate) scene: Scene,
    /// Cursor styles by hitbox.
    pub(crate) cursor_styles: Vec<CursorStyleRequest>,
    /// The text input handler of the focused element, if it painted one.
    pub(crate) input_handlers: Vec<Option<InputHandlerSlot>>,
    /// Tooltips asked for this frame.
    pub(crate) tooltip_requests: Vec<Option<TooltipRequest>>,
    /// Window control areas (caption, buttons).
    pub(crate) window_control_hitboxes: Vec<WindowControlHitbox>,
    /// `deferred` elements, drawn after the tree.
    pub(crate) deferred_draws: Vec<DeferredDraw>,
}

impl Frame {
    /// An empty frame.
    pub(crate) fn new() -> Self {
        Frame {
            focus: None,
            element_states: HashMap::new(),
            hitboxes: Vec::new(),
            mouse_listeners: Vec::new(),
            dispatch_tree: DispatchTree::new(),
            scene: Scene::new(),
            cursor_styles: Vec::new(),
            input_handlers: Vec::new(),
            tooltip_requests: Vec::new(),
            window_control_hitboxes: Vec::new(),
            deferred_draws: Vec::new(),
        }
    }

    /// Empties every list, keeping capacity (element states are handled by the swap).
    pub(crate) fn clear(&mut self) {
        todo!("window")
    }

    /// Hit tests `position` against this frame's hitboxes.
    pub(crate) fn hit_test(&self, _position: crate::Point<Pixels>) -> HitTest {
        todo!("window")
    }

    /// The focus path (root first) of the focused element in this frame's dispatch tree.
    pub(crate) fn focus_path(&self) -> SmallVec<[FocusId; 8]> {
        todo!("window")
    }
}
