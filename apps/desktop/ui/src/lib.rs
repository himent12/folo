//! folo-ui: FOLO Desktop's own UI framework (`ui/DESIGN.md`). GPUI's API for the subset the app uses,
//! rendered on the CPU in tiles, with Win32, winit and headless platforms.
//!
//! Everything public is re-exported flat from the crate root (and `prelude`), as GPUI does. The modules,
//! bottom-up (`ui/README.md` has the table with what each holds):
//!
//! - foundation: `geometry`, `color`, `shared_string`, `input_events`, `keystroke`, `scene`, `opacity`;
//!   the platform contract (`platform`, backends in `platform/windows`, `platform/winit`,
//!   `platform/headless`) and the text system ([`text`], backends in `text/dwrite` and `text/portable`);
//! - style and layout: `style`, `styled`, `layout`, `path_builder`, `svg`, `image`, `easing`, `assets`;
//! - the renderer: [`raster`], the CPU tile renderer that turns a `Scene` into a `Framebuffer`;
//! - the runtime: `app`, `executor`, `action`, `keymap`, `view`;
//! - the window and elements: `window`, `interactive`, `element`, `elements`;
//! - `headless_app`: `HeadlessApp`, the harness over the headless platform for tests and `--shot`.

// Foundation: value types, events, the platform and text contracts, the scene.
mod color;
mod geometry;
mod input_events;
mod keystroke;
mod opacity;
mod platform;
mod scene;
mod shared_string;
pub mod text;
mod util;

// Style, layout, assets.
mod assets;
mod easing;
mod image;
mod layout;
mod path_builder;
mod style;
mod styled;
mod svg;

// The renderer.
pub mod raster;

// The runtime, the window, the elements.
mod action;
mod app;
mod element;
mod elements;
mod executor;
mod headless_app;
mod interactive;
mod keymap;
pub mod prelude;
mod view;
mod window;

pub use assets::*;
pub use color::*;
pub use easing::*;
pub use geometry::*;
pub use image::*;
pub use input_events::*;
pub use keystroke::*;
pub use opacity::OpacityGroup;
pub use path_builder::*;
pub use platform::*;
pub use scene::*;
pub use shared_string::*;
pub use style::*;
pub use styled::*;
pub use svg::*;
pub use text::*;

pub use action::*;
pub use app::*;
pub use element::*;
pub use elements::*;
pub use executor::*;
pub use headless_app::*;
pub use interactive::*;
pub use keymap::*;
pub use view::*;
pub use window::*;
