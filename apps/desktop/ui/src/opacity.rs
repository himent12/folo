//! Opacity groups: an opacity that can change after paint without repainting the UI (the breathing status
//! dot, fades, shimmers: `AnimationExt::with_opacity_animation`). Owner tag: `"window"`.
//!
//! The window paints a group's content at its base colours (element opacity 1) and records which spans of
//! the scene's primitives belong to which group. After paint, `seal_opacity_groups` keeps those base
//! primitives aside and writes the faded ones into the scene; a later factor change re-fades the spans from
//! the base copies (`apply_opacity_groups`). The renderer sees ordinary primitives, so their tile signatures
//! change exactly where the group's primitives are, and only those tiles are repainted.
//!
//! Exactness: a group's opacity is `(parent's opacity * outer) * factor`, where `outer` is the element
//! opacity in effect where a top-level group starts (1 for a nested one); an element opacity set inside a
//! group becomes a nested group of its own (`Window::with_element_opacity`). Every primitive's colour is
//! therefore faded by the same product, multiplied in the same order, as when the element opacity was
//! applied at paint time, so the pixels are identical.

use crate::{Primitive, Scene};

/// One opacity group of a scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpacityGroup {
    /// The group it is nested in.
    pub parent: Option<u32>,
    /// The element opacity where it started (a top-level group's; 1 for a nested group).
    pub outer: f32,
    /// Its own factor (the animated value).
    pub factor: f32,
}

/// A span of the scene's primitives painted inside a group (its innermost one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupSpan {
    /// The first primitive.
    pub(crate) start: u32,
    /// One past the last primitive.
    pub(crate) end: u32,
    /// The group.
    pub(crate) group: u32,
}

impl Scene {
    /// Adds a group and returns its index.
    pub(crate) fn push_opacity_group(&mut self, parent: Option<u32>, outer: f32, factor: f32) -> u32 {
        self.opacity_groups.push(OpacityGroup { parent, outer, factor });
        self.opacity_groups.len() as u32 - 1
    }

    /// Records that the primitives from `start` to the end of the scene belong to `group` (nothing when
    /// that is empty).
    pub(crate) fn push_group_span(&mut self, group: u32, start: usize) {
        let end = self.primitives.len();
        if end > start {
            self.group_spans.push(GroupSpan { start: start as u32, end: end as u32, group });
        }
    }

    /// The opacity group `group`'s primitives are faded by: its factor times its parents', multiplied as
    /// the window multiplies element opacity.
    pub fn group_opacity(&self, group: u32) -> f32 {
        let g = self.opacity_groups[group as usize];
        let parent = g.parent.map_or(1.0, |parent| self.group_opacity(parent));
        parent * g.outer * g.factor
    }

    /// Sets a group's factor (the scene changes at the next `apply_opacity_groups`).
    pub(crate) fn set_group_factor(&mut self, group: u32, factor: f32) {
        self.opacity_groups[group as usize].factor = factor;
    }

    /// After paint: keeps the grouped primitives at their base colours and fades them in the scene.
    pub(crate) fn seal_opacity_groups(&mut self) {
        self.group_base.clear();
        for span in &self.group_spans {
            self.group_base.extend_from_slice(&self.primitives[span.start as usize..span.end as usize]);
        }
        self.apply_opacity_groups();
    }

    /// Fades every grouped primitive from its base copy by its group's current opacity.
    pub(crate) fn apply_opacity_groups(&mut self) {
        if self.group_spans.is_empty() {
            return;
        }
        let mut opacity: Vec<f32> = Vec::with_capacity(self.opacity_groups.len());
        for g in &self.opacity_groups {
            // Parents come first (a group is added when it starts).
            let parent = g.parent.map_or(1.0, |parent| opacity[parent as usize]);
            opacity.push(parent * g.outer * g.factor);
        }
        let mut base = self.group_base.iter();
        for span in &self.group_spans {
            let factor = opacity[span.group as usize];
            for dst in &mut self.primitives[span.start as usize..span.end as usize] {
                fade(dst, base.next().expect("a base copy per grouped primitive"), factor);
            }
        }
    }
}

/// `dst` becomes `base` with every colour faded by `factor`, as the paint functions fade by the element
/// opacity (`Hsla::opacity`, `Background::opacity`; an image's or emoji's opacity is multiplied).
fn fade(dst: &mut Primitive, base: &Primitive, factor: f32) {
    dst.clone_from(base);
    match dst {
        Primitive::Quad(quad) => {
            quad.background = quad.background.opacity(factor);
            quad.border_color = quad.border_color.opacity(factor);
        }
        Primitive::Shadow(shadow) => shadow.color = shadow.color.opacity(factor),
        Primitive::MonoSprite(sprite) => sprite.color = sprite.color.opacity(factor),
        Primitive::ColorSprite(sprite) => sprite.opacity *= factor,
        Primitive::Image(image) => image.opacity *= factor,
        Primitive::Path(path) => path.paint = path.paint.opacity(factor),
        Primitive::Underline(underline) => underline.color = underline.color.opacity(factor),
    }
}
