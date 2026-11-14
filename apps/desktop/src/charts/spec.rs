//! What a line chart draws: its series, markers and options.

use std::sync::Arc;

use folo_ui::{Hsla, SharedString};

use super::HoverFn;
use super::axis::format_tick;
use super::marks::End;

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
    /// How the line ends: the mark on its last point (`charts::marks`).
    pub end: End,
    /// The ink of the line's name at its end, on a chart that names its lines there
    /// (`LineChart::end_labels`); None leaves it unnamed.
    pub end_label: Option<Hsla>,
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
            end: End::None,
            end_label: None,
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

    /// A live run's line, ending in the lime dot (no mark otherwise).
    pub fn live(mut self, live: bool) -> Self {
        if live {
            self.end = End::Live;
        }
        self
    }

    pub fn end(mut self, end: End) -> Self {
        self.end = end;
        self
    }

    /// Named at its end in `ink` (`chart.text` of its hue, or the text ink).
    pub fn end_label(mut self, ink: Hsla) -> Self {
        self.end_label = Some(ink);
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
    /// Points to mark with a dot (x, y, colour), e.g. a class's operating point.
    pub dots: Vec<(f32, f32, Hsla)>,
    /// The best point (x, y): a hollow ring in ink, drawn around a line's end mark when they meet.
    pub best: Option<(f32, f32)>,
    /// A 0–1 metric whose highest value is this: y from 0 to a little above it with 3 to 5 round
    /// gridlines (`charts::domain::unit_domain`), instead of `y_range`.
    pub unit_max: Option<f32>,
    /// Name the lines that have an `end_label` at their ends, where the plot has room.
    pub end_labels: bool,
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
            best: None,
            unit_max: None,
            end_labels: false,
        }
    }
}
