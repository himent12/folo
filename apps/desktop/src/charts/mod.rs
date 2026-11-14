//! Charts painted with folo-ui canvas primitives: line charts (raw + smoothed series, gradient
//! areas, markers, end marks and the best ring, lines named at their ends, fitted 0–1 axes, hover
//! crosshair with a readout), sparklines and histograms.
//!
//! Charts are plain functions of their data. Hover state is owned by the caller: the chart reports
//! the pointer's position as a fraction of the plot width through `on_hover` and draws the cursor
//! at the `hover` fraction it is given, so several charts can share one cursor.

mod axis;
mod cursor;
pub mod domain;
mod ease;
mod end_labels;
mod histogram;
mod line;
pub mod marks;
mod smooth;
mod sparkline;
mod spec;
mod text;

pub use axis::{axis_pow, in_pow, nice_step, superscript};
pub use cursor::{nearest, value_at, value_near};
pub use ease::{Points, ease_key};
pub use histogram::histogram;
pub use line::line_chart;
pub use marks::End;
pub use smooth::{ema, smoothed};
pub use sparkline::{sparkline, sparkline_in};
pub use spec::{LineChart, Marker, Series};
pub use text::{mono_width, text_width};

use std::rc::Rc;

use folo_ui::{App, Window};

pub type HoverFn = Rc<dyn Fn(Option<f32>, &mut Window, &mut App)>;
