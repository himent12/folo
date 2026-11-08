//! Placeholder.
use gpui::{AnyElement, Context, Window};
use crate::app::FoloApp;
impl FoloApp {
    pub fn overlays(&mut self, _w: &mut Window, _cx: &mut Context<Self>) -> Vec<AnyElement> { Vec::new() }
}
