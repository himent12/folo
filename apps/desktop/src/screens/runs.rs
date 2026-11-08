//! Placeholder.
use gpui::{AnyElement, Context, IntoElement, Window, div};
use crate::app::FoloApp;
impl FoloApp {
    pub fn runs(&mut self, _w: &mut Window, _cx: &mut Context<Self>) -> AnyElement { div().into_any_element() }
    pub fn run_view(&mut self, _id: &str, _w: &mut Window, _cx: &mut Context<Self>) -> AnyElement { div().into_any_element() }
}
