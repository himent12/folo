// Adapted from GPUI (Apache-2.0): crates/gpui/src/scene.rs (BorderStyle, primitive field names)
//! The scene: one frame's display list, in painter's order, in device pixels (`ui/DESIGN.md`, Scene).
//! It is the seam between the window (which paints into it) and the renderer (`raster`).

use crate::{
    Background, Bounds, Corners, DevicePixels, Edges, Hsla, Point, ScaledPixels, Size, image::RenderImage,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// One id sequence for masks and paths, so ids never collide across kinds.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// A frame's primitives, drawn in order (each over every earlier one).
#[derive(Default, Debug, Clone)]
pub struct Scene {
    /// The primitives, in painter's order.
    pub primitives: Vec<Primitive>,
}

impl Scene {
    /// An empty scene.
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes every primitive, keeping the capacity.
    pub fn clear(&mut self) {
        self.primitives.clear();
    }

    /// Appends any primitive.
    pub fn push(&mut self, primitive: impl Into<Primitive>) {
        self.primitives.push(primitive.into());
    }

    /// Appends a quad.
    pub fn push_quad(&mut self, quad: Quad) {
        self.primitives.push(Primitive::Quad(quad));
    }

    /// Appends a shadow.
    pub fn push_shadow(&mut self, shadow: Shadow) {
        self.primitives.push(Primitive::Shadow(shadow));
    }

    /// Appends a monochrome sprite (a glyph or an SVG).
    pub fn push_mono_sprite(&mut self, sprite: MonoSprite) {
        self.primitives.push(Primitive::MonoSprite(sprite));
    }

    /// Appends a colour sprite (an emoji).
    pub fn push_color_sprite(&mut self, sprite: ColorSprite) {
        self.primitives.push(Primitive::ColorSprite(sprite));
    }

    /// Appends an image.
    pub fn push_image(&mut self, image: Image) {
        self.primitives.push(Primitive::Image(image));
    }

    /// Appends a path.
    pub fn push_path(&mut self, path: Path) {
        self.primitives.push(Primitive::Path(path));
    }

    /// Appends an underline.
    pub fn push_underline(&mut self, underline: Underline) {
        self.primitives.push(Primitive::Underline(underline));
    }

    /// How many primitives the scene holds.
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Whether the scene is empty.
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }
}

/// One thing to draw. Every primitive carries its clip (the content mask, already intersected).
#[derive(Debug, Clone)]
pub enum Primitive {
    /// A filled, bordered, rounded rectangle.
    Quad(Quad),
    /// A blurred box shadow.
    Shadow(Shadow),
    /// An alpha mask tinted with one colour (glyphs, SVG icons).
    MonoSprite(MonoSprite),
    /// A colour mask (emoji).
    ColorSprite(ColorSprite),
    /// A decoded image.
    Image(Image),
    /// A filled or stroked path.
    Path(Path),
    /// A straight or wavy underline (or strikethrough).
    Underline(Underline),
}

impl Primitive {
    /// The primitive's bounds.
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Quad(p) => &p.bounds,
            Primitive::Shadow(p) => &p.bounds,
            Primitive::MonoSprite(p) => &p.bounds,
            Primitive::ColorSprite(p) => &p.bounds,
            Primitive::Image(p) => &p.bounds,
            Primitive::Path(p) => &p.bounds,
            Primitive::Underline(p) => &p.bounds,
        }
    }

    /// The primitive's clip rectangle.
    pub fn clip(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Quad(p) => &p.clip,
            Primitive::Shadow(p) => &p.clip,
            Primitive::MonoSprite(p) => &p.clip,
            Primitive::ColorSprite(p) => &p.clip,
            Primitive::Image(p) => &p.clip,
            Primitive::Path(p) => &p.clip,
            Primitive::Underline(p) => &p.clip,
        }
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A rectangle with a background, a border and rounded corners (GPUI's `Quad`).
#[derive(Debug, Clone)]
pub struct Quad {
    /// The outer bounds.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The fill (solid, gradient or pattern), converted by the renderer with GPUI's formulas.
    pub background: Background,
    /// The border colour.
    pub border_color: Hsla,
    /// The corner radii.
    pub corner_radii: Corners<ScaledPixels>,
    /// The border widths.
    pub border_widths: Edges<ScaledPixels>,
    /// Solid or dashed.
    pub border_style: BorderStyle,
}

/// A box shadow (GPUI's `Shadow`): the shape of `bounds` with `corner_radii`, blurred.
#[derive(Debug, Clone)]
pub struct Shadow {
    /// The shadow's box, before blurring.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The box's corner radii.
    pub corner_radii: Corners<ScaledPixels>,
    /// The colour.
    pub color: Hsla,
    /// The blur radius.
    pub blur_radius: ScaledPixels,
}

/// What a monochrome sprite's mask is: a glyph gets the text backend's contrast / gamma correction,
/// an SVG mask is plain alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpriteKind {
    /// A text glyph.
    Glyph,
    /// An SVG icon.
    Svg,
}

/// An alpha mask tinted with `color`, drawn at `bounds` (the mask's size in device pixels).
#[derive(Debug, Clone)]
pub struct MonoSprite {
    /// Where the mask lands.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The tint.
    pub color: Hsla,
    /// The coverage mask.
    pub mask: Arc<AlphaMask>,
    /// Glyph or SVG.
    pub kind: SpriteKind,
}

/// A colour mask (an emoji), drawn at `bounds`.
#[derive(Debug, Clone)]
pub struct ColorSprite {
    /// Where the mask lands.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The pixels.
    pub mask: Arc<ColorMask>,
    /// Opacity multiplied into every pixel.
    pub opacity: f32,
}

/// A decoded image drawn into `bounds` (GPUI's `PolychromeSprite` of an image).
#[derive(Debug, Clone)]
pub struct Image {
    /// Where the image lands (it is scaled to fit exactly).
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The corner radii the image is rounded with.
    pub corner_radii: Corners<ScaledPixels>,
    /// The image.
    pub image: Arc<RenderImage>,
    /// Which frame of the image.
    pub frame: usize,
    /// Draw in grayscale.
    pub grayscale: bool,
    /// Opacity multiplied into every pixel.
    pub opacity: f32,
}

/// A filled or stroked path.
#[derive(Debug, Clone)]
pub struct Path {
    /// The path's bounds (stroke included).
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The geometry.
    pub path: Arc<PathData>,
    /// The paint.
    pub paint: Background,
}

/// An underline or strikethrough (GPUI's `Underline`).
#[derive(Debug, Clone)]
pub struct Underline {
    /// The line's box.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip rectangle.
    pub clip: Bounds<ScaledPixels>,
    /// The colour.
    pub color: Hsla,
    /// The line's thickness.
    pub thickness: ScaledPixels,
    /// A wavy (spell-check) line rather than a straight one.
    pub wavy: bool,
}

/// A coverage mask: one byte of alpha per device pixel, rows top-down.
#[derive(Debug)]
pub struct AlphaMask {
    /// Process-unique id (tile hashes and caches use it, never the bytes).
    pub id: u64,
    /// The mask's size in device pixels.
    pub size: Size<DevicePixels>,
    /// `width * height` alpha bytes.
    pub data: Box<[u8]>,
}

impl AlphaMask {
    /// Wraps `data` (`width * height` bytes) with a new id.
    pub fn new(size: Size<DevicePixels>, data: impl Into<Box<[u8]>>) -> Self {
        let data = data.into();
        debug_assert_eq!(data.len(), size.width.0.max(0) as usize * size.height.0.max(0) as usize);
        Self { id: next_id(), size, data }
    }
}

/// A colour mask: four bytes per device pixel, BGRA premultiplied, rows top-down.
#[derive(Debug)]
pub struct ColorMask {
    /// Process-unique id (tile hashes and caches use it, never the bytes).
    pub id: u64,
    /// The mask's size in device pixels.
    pub size: Size<DevicePixels>,
    /// `width * height * 4` bytes.
    pub data: Box<[u8]>,
}

impl ColorMask {
    /// Wraps `data` (`width * height * 4` bytes, BGRA premultiplied) with a new id.
    pub fn new(size: Size<DevicePixels>, data: impl Into<Box<[u8]>>) -> Self {
        let data = data.into();
        debug_assert_eq!(data.len(), size.width.0.max(0) as usize * size.height.0.max(0) as usize * 4);
        Self { id: next_id(), size, data }
    }
}

/// A path's geometry, already flattened by `PathBuilder`: contours of device-pixel points, filled
/// with a rule or stroked with a style.
#[derive(Debug)]
pub struct PathData {
    /// Process-unique id (path coverage is cached by it).
    pub id: u64,
    /// The contours.
    pub contours: Vec<Contour>,
    /// Fill or stroke.
    pub kind: PathKind,
}

impl PathData {
    /// Wraps the geometry with a new id.
    pub fn new(contours: Vec<Contour>, kind: PathKind) -> Self {
        Self { id: next_id(), contours, kind }
    }
}

/// One contour: points in device pixels, and whether it closes back to its first point.
#[derive(Debug, Clone, Default)]
pub struct Contour {
    /// The points, in order.
    pub points: Vec<Point<ScaledPixels>>,
    /// Closed (the last point joins the first).
    pub closed: bool,
}

/// How a path is painted.
#[derive(Debug, Clone, PartialEq)]
pub enum PathKind {
    /// Filled with a rule.
    Fill(FillRule),
    /// Stroked with a style.
    Stroke(StrokeStyle),
}

/// Which points are inside a filled path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum FillRule {
    /// Non-zero winding (lyon's and GPUI's default).
    #[default]
    NonZero,
    /// Even-odd.
    EvenOdd,
}

/// How a path is stroked (lyon's `StrokeOptions` as GPUI's `PathBuilder` uses them).
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeStyle {
    /// The line width in device pixels.
    pub width: ScaledPixels,
    /// The cap at open ends.
    pub cap: LineCap,
    /// The join between segments.
    pub join: LineJoin,
    /// The miter limit (lyon's default 4).
    pub miter_limit: f32,
    /// Dash lengths in device pixels, alternating on / off (`None`: solid).
    pub dash: Option<Vec<ScaledPixels>>,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            width: ScaledPixels(1.),
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 4.,
            dash: None,
        }
    }
}

/// The shape at the open ends of a stroke.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LineCap {
    /// Flat, at the end point.
    #[default]
    Butt,
    /// Flat, half a width past the end point.
    Square,
    /// A half circle.
    Round,
}

/// The shape where stroke segments meet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum LineJoin {
    /// A sharp corner, up to the miter limit.
    #[default]
    Miter,
    /// A cut corner.
    Bevel,
    /// A rounded corner.
    Round,
}

/// The contrast / gamma correction applied to glyph masks at blend time (as GPUI's DirectWrite
/// renderer does). `apply == false` means none: the mask is used as plain alpha.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphBlend {
    /// GPUI's gamma ratios (`get_gamma_correction_ratios` of the gamma).
    pub gamma_ratios: [f32; 4],
    /// GPUI's grayscale enhanced contrast.
    pub grayscale_enhanced_contrast: f32,
    /// Whether to correct at all.
    pub apply: bool,
}

impl GlyphBlend {
    /// No correction (the portable and fixed text backends).
    pub const IDENTITY: GlyphBlend =
        GlyphBlend { gamma_ratios: [0.; 4], grayscale_enhanced_contrast: 0., apply: false };

    /// No correction.
    pub const fn none() -> Self {
        Self::IDENTITY
    }

    /// GPUI's correction with these parameters.
    pub const fn new(gamma_ratios: [f32; 4], grayscale_enhanced_contrast: f32) -> Self {
        Self { gamma_ratios, grayscale_enhanced_contrast, apply: true }
    }

    /// Whether this applies no correction.
    pub fn is_identity(&self) -> bool {
        !self.apply
    }
}

impl Default for GlyphBlend {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    #[test]
    fn ids_are_unique_and_clear_keeps_capacity() {
        let a = AlphaMask::new(size(DevicePixels(2), DevicePixels(1)), vec![0u8, 255]);
        let b = ColorMask::new(size(DevicePixels(1), DevicePixels(1)), vec![0u8; 4]);
        let c = PathData::new(
            vec![Contour { points: vec![point(ScaledPixels(0.), ScaledPixels(0.))], closed: true }],
            PathKind::Fill(FillRule::NonZero),
        );
        assert!(a.id != b.id && b.id != c.id && a.id != c.id);

        let mut scene = Scene::new();
        let r =
            Bounds::new(point(ScaledPixels(0.), ScaledPixels(0.)), size(ScaledPixels(4.), ScaledPixels(4.)));
        scene.push_underline(Underline {
            bounds: r,
            clip: r,
            color: crate::black(),
            thickness: ScaledPixels(1.),
            wavy: false,
        });
        assert_eq!(scene.len(), 1);
        assert_eq!(*scene.primitives[0].clip(), r);
        let cap = scene.primitives.capacity();
        scene.clear();
        assert!(scene.is_empty() && scene.primitives.capacity() == cap);
        assert!(GlyphBlend::none().is_identity() && !GlyphBlend::new([1.; 4], 1.).is_identity());
    }
}
