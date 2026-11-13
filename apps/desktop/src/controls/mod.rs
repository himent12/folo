//! Interactive controls. Each takes a stable key: it names the element, owns its focus handle (so
//! Tab reaches it and Enter/Space activate it) and shows the focus ring under keyboard use.

mod button;
mod chart_group;
mod focus;
#[cfg(test)]
mod key_tests;
mod scroll;
mod segmented;
mod select;
mod slider;
mod toggle;
mod tooltip;

pub use button::Btn;
pub use chart_group::Cursor;
pub use tooltip::{focus_tip, tip};
