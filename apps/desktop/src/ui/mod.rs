//! Stateless building blocks: type, surfaces, badges, metrics, progress, placeholders.
//!
//! Interactive controls that need focus handles live on the app (`controls/`); everything
//! here is a pure function of the theme and its arguments.

mod cloud;
mod format;
mod legend;
mod metric;
mod motion;
mod scroll;
mod status;
mod surface;
mod text;

pub use cloud::{
    cloud_wait_title, fmt_cost, fmt_cost_range, fmt_hours, fmt_price, fmt_transfer, hours_value, offer_line,
    transfer_frac,
};
pub use format::{
    bytes, count_si, device_label, display_path, f3, fmt_aug, fmt_conf, fmt_lr, fmt_scale, int, latency_text,
    may_overflow, model_storage, plural, pr_caption, sci, share_spare, short_gpu, split_label, waiting_text,
};
pub use legend::{class_swatch, legend_value, line_swatch, swatch};
pub use metric::{baseline_lift, detail_line, duration_metric, kv, metric, metric_pair, stat_at, stat_sized};
pub use motion::{appear, os_prefers_reduced_motion, reduce_motion, set_reduce_motion};
pub use scroll::{ThumbGeo, paint_thumb, scroll_edges};
pub use status::{Tone, badge, dot, e2e_tag, status_badge, status_dot, status_label};
pub use surface::{chart_section, empty_state, panel, panel_header, progress, select_bar, skeleton, vline};
pub use text::{
    Ellipsis, StyledExt, caption, icon, inline_link, kbd, meta_row, overline, rich, shortcut, text,
};
