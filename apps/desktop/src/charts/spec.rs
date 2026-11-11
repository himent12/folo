//! What a line chart draws: its series, markers and options.

use std::sync::Arc;

use gpui::{Hsla, SharedString};

use super::HoverFn;
use super::axis::format_tick;

#[derive(Clone)]
pub struct Series {
    pub name: SharedString,
    pub color: Hsla,
    /// (x, y), sorted by x.
    pub points: Arc<Vec<(f32, f32)>>,
    /// Exponential smoothing weight (0 = none). Raw data is then drawn faint underneath.
    pub smooth: f32,
    pub area: bool,
    pub width: f32,
    pub dashed: bool,
    /// Dotted (wins over dashed).
    pub dotted: bool,
    /// Any other dash pattern (wins over both), for sets of more than three lines.
    pub pattern: Option<&'static [f32]>,
    /// Draw a live dot with a halo on the last point.
    pub live: bool,
}

impl Series {
    pub fn dashed(mut self) -> Self {
        self.dashed = true;
        self
    }

    pub fn dotted(mut self) -> Self {
        self.dotted = true;
        self
    }

    pub fn pattern(mut self, dash: &'static [f32]) -> Self {
        self.pattern = Some(dash);
        self
    }

    pub fn new(name: impl Into<SharedString>, color: Hsla, points: Arc<Vec<(f32, f32)>>) -> Self {
        Self {
            name: name.into(),
            color,
            points,
            smooth: 0.0,
            area: false,
            width: 1.5,
            dashed: false,
            dotted: false,
            pattern: None,
            live: false,
        }
    }

    pub fn smooth(mut self, w: f32) -> Self {
        self.smooth = w;
        self
    }

    pub fn area(mut self) -> Self {
        self.area = true;
        self
    }

    pub fn live(mut self, live: bool) -> Self {
        self.live = live;
        self
    }

    pub fn width(mut self, w: f32) -> Self {
        self.width = w;
        self
    }
}

#[derive(Clone)]
pub struct Marker {
    pub x: f32,
    pub label: SharedString,
    pub color: Hsla,
    /// A fixed label lane, so the same marker sits at the same height in sibling charts.
    pub lane: Option<usize>,
}

pub struct LineChart {
    pub id: SharedString,
    pub series: Vec<Series>,
    pub x_range: Option<(f32, f32)>,
    pub y_range: Option<(f32, f32)>,
    /// Tick label format; `None` picks decimals from the tick step so labels line up.
    pub y_fmt: Option<fn(f32) -> String>,
    pub x_fmt: fn(f32) -> String,
    /// Readout values use this format (defaults to `y_fmt`).
    pub value_fmt: Option<fn(f32) -> String>,
    pub x_name: &'static str,
    pub markers: Vec<Marker>,
    pub hover: Option<f32>,
    pub on_hover: Option<HoverFn>,
    pub height: f32,
    /// y ticks to aim for.
    pub y_ticks: usize,
    /// Exact y tick values instead (e.g. an axis that starts above 0 labels its floor).
    pub y_tick_values: Option<Vec<f32>>,
    pub log_y: bool,
    /// Axis titles (x, y): y above the plot's top-left, x centred under the tick labels.
    pub axis_titles: Option<(&'static str, &'static str)>,
    /// Marker label lanes to reserve even when fewer are used, so charts in a row share a plot top.
    pub min_lanes: usize,
    /// X is a count (epochs): ticks land on whole numbers and the last one labels the end.
    pub x_int: bool,
    /// Draw the floating readout box on hover. Off when the panel's legend shows the values.
    pub readout: bool,
    /// Fixed x tick count (shared across linked charts); None picks one from the width.
    pub x_ticks: Option<usize>,
    /// What the data belongs to (`ease_key(run or model id)`): easing never blends two owners.
    pub ease_key: u64,
    /// Print a shared power of ten above the y axis. Off when the panel caption carries it.
    pub pow_caption: bool,
    /// The part of x not reached yet (a live run's remaining epochs): tinted, with a label.
    pub future: Option<(f32, SharedString)>,
    /// Points to mark with a dot (x, y, colour), e.g. the best epoch.
    pub dots: Vec<(f32, f32, Hsla)>,
}

impl LineChart {
    pub fn new(id: impl Into<SharedString>, series: Vec<Series>) -> Self {
        Self {
            id: id.into(),
            series,
            x_range: None,
            y_range: None,
            y_tick_values: None,
            y_fmt: None,
            x_fmt: |v| format_tick(v),
            value_fmt: None,
            x_name: "",
            markers: Vec::new(),
            hover: None,
            on_hover: None,
            height: 180.,
            y_ticks: 4,
            log_y: false,
            axis_titles: None,
            min_lanes: 0,
            x_int: false,
            readout: true,
            x_ticks: None,
            future: None,
            ease_key: 0,
            pow_caption: true,
            dots: Vec::new(),
        }
    }
}
