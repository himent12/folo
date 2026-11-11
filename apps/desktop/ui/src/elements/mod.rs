//! The built-in elements. Owner tag: `"elements"`, except `div.rs` (`"window"`, with `interactive/`).
//!
//! Files: `div.rs` (`Div`, `Stateful`), `text.rs` (strings as elements, `StyledText`), `svg.rs` (`Svg`),
//! `canvas.rs`, `uniform_list.rs` (+ `UniformListScrollHandle`, `ScrollStrategy`), `anchored.rs`,
//! `deferred.rs`, `animation.rs` (`Animation`, `AnimationExt`), `input_handler.rs` (`InputHandler`,
//! `EntityInputHandler`, `ElementInputHandler`).

mod anchored;
mod animation;
mod canvas;
mod deferred;
mod div;
mod input_handler;
mod svg;
mod text;
mod uniform_list;

pub use anchored::*;
pub use animation::*;
pub use canvas::*;
pub use deferred::*;
pub use div::*;
pub use input_handler::*;
pub use svg::*;
pub use text::*;
pub use uniform_list::*;

/// Skeleton only: implements `Element` for `$ty` with the given phase states and `todo!($owner)` bodies, and
/// `IntoElement` as itself. The owner replaces each use with the real impl (the trait fixes the signatures;
/// the associated types written here are the design).
macro_rules! skeleton_element {
    ([$($generics:tt)*] $ty:ty, $request_layout:ty, $prepaint:ty, $owner:literal) => {
        impl<$($generics)*> $crate::Element for $ty {
            type RequestLayoutState = $request_layout;
            type PrepaintState = $prepaint;

            fn id(&self) -> Option<$crate::ElementId> {
                todo!($owner)
            }

            fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
                None
            }

            fn request_layout(
                &mut self,
                _id: Option<&$crate::GlobalElementId>,
                _inspector_id: Option<&$crate::InspectorElementId>,
                _window: &mut $crate::Window,
                _cx: &mut $crate::App,
            ) -> ($crate::LayoutId, Self::RequestLayoutState) {
                todo!($owner)
            }

            fn prepaint(
                &mut self,
                _id: Option<&$crate::GlobalElementId>,
                _inspector_id: Option<&$crate::InspectorElementId>,
                _bounds: $crate::Bounds<$crate::Pixels>,
                _request_layout: &mut Self::RequestLayoutState,
                _window: &mut $crate::Window,
                _cx: &mut $crate::App,
            ) -> Self::PrepaintState {
                todo!($owner)
            }

            fn paint(
                &mut self,
                _id: Option<&$crate::GlobalElementId>,
                _inspector_id: Option<&$crate::InspectorElementId>,
                _bounds: $crate::Bounds<$crate::Pixels>,
                _request_layout: &mut Self::RequestLayoutState,
                _prepaint: &mut Self::PrepaintState,
                _window: &mut $crate::Window,
                _cx: &mut $crate::App,
            ) {
                todo!($owner)
            }
        }

        impl<$($generics)*> $crate::IntoElement for $ty {
            type Element = Self;

            fn into_element(self) -> Self::Element {
                self
            }
        }
    };
}

pub(crate) use skeleton_element;
