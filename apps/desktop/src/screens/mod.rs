//! One module per screen, each an `impl FoloApp` block returning the page element.

pub mod datasets;
pub mod models;
pub mod overview;
pub mod playground;
pub mod runs;
pub mod settings;

use std::sync::Arc;

use gpui::{
    AnyElement, Context, Div, FontWeight, IntoElement, ParentElement, RenderImage, SharedString, Styled,
    Window, canvas, div, px,
};

use crate::api::{ApiError, ImageInfo};
pub use crate::api::{Label, RunExt, SplitsExt};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::store::ImgKind;
use crate::theme::{Theme, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, paint_image_with_boxes};

/// Title, one line of context, and actions on the right.
pub fn page_header(
    t: &Theme,
    title: impl Into<SharedString>,
    subtitle: impl IntoElement,
    actions: Vec<AnyElement>,
) -> Div {
    let title: SharedString = title.into();
    detail_header(t, text(ty::XL, t.text_primary).child(title), subtitle, actions)
}

/// The page header with any title row (a name plus badges): every page shares its geometry, and the
/// actions sit centred on the title line whatever the subtitle below holds.
pub fn detail_header(
    t: &Theme,
    title: impl IntoElement,
    subtitle: impl IntoElement,
    actions: Vec<AnyElement>,
) -> Div {
    div()
        .flex()
        .items_start()
        .justify_between()
        .gap(px(16.))
        .pb(px(4.))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .min_w_0()
                .child(div().h(px(ty::XL.line)).flex().items_center().child(title))
                .child(div().text_color(t.text_secondary).text_size(px(ty::SM.size)).child(subtitle)),
        )
        .child(div().h(px(ty::XL.line)).flex().items_center().gap(px(8.)).flex_none().children(actions))
}

impl FoloApp {
    /// A dataset image painted with optional boxes, at a fixed tile size.
    pub fn tile(
        &mut self,
        dataset: &str,
        meta: &ImageInfo,
        boxes: bool,
        fit: Fit,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let img = self.image(dataset, &meta.id, ImgKind::Thumb, cx);
        let dims = (meta.width as f32, meta.height as f32);
        let draw: Vec<DrawBox> = if boxes {
            meta.boxes.iter().map(|b| DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)).collect()
        } else {
            Vec::new()
        };
        image_canvas(&t, img, dims, draw, fit, false)
    }

    /// A centred error panel with retry.
    pub fn error_panel(
        &self,
        e: &ApiError,
        retry_key: &str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let hint = e.hint.clone().unwrap_or_else(|| format!("Error code {}", e.code));
        let retry = self.button(
            retry_key,
            "Retry",
            Btn::Primary,
            Some("icons/refresh-cw.svg"),
            window,
            cx,
            act(|this, _, cx| this.retry_all(cx)),
        );
        let settings = self.button(
            &format!("{retry_key}-settings"),
            "Connection settings",
            Btn::Secondary,
            Some("icons/settings.svg"),
            window,
            cx,
            act(|this, w, cx| this.navigate(crate::app::Route::Settings, w, cx)),
        );
        panel(&t)
            .child(
                empty_state(&t, "icons/plug.svg", &e.message, &hint, Tone::Danger)
                    .child(div().flex().gap(px(8.)).child(settings).child(retry)),
            )
            .into_any_element()
    }

    pub fn retry_all(&mut self, cx: &mut Context<Self>) {
        use crate::store::Load;
        *self.ds_filter_cache.borrow_mut() = None;
        *self.ds_split_counts.borrow_mut() = None;
        let s = &mut self.store;
        s.health = Load::Idle;
        s.system = Load::Idle;
        s.datasets = Load::Idle;
        s.models = Load::Idle;
        s.runs = Load::Idle;
        s.dataset.clear();
        s.images.clear();
        s.evals.clear();
        s.run_data.clear();
        s.samples.clear();
        s.logs.clear();
        cx.notify();
    }
}

/// Paint an image with boxes filling its parent.
pub fn image_canvas(
    t: &Theme,
    img: Option<Arc<RenderImage>>,
    dims: (f32, f32),
    boxes: Vec<DrawBox>,
    fit: Fit,
    labels: bool,
) -> AnyElement {
    let t = *t;
    canvas(
        |_, _, _| (),
        move |b, _, window, cx| {
            paint_image_with_boxes(
                &t,
                window,
                cx,
                b,
                img.as_ref(),
                dims,
                &boxes,
                fit,
                1.0,
                (0., 0.),
                labels,
                0.,
            );
        },
    )
    .size_full()
    .into_any_element()
}

/// Shimmering placeholder blocks shaped like a panel of rows.
pub fn loading_rows(t: &Theme, id: &str, rows: usize) -> Div {
    div().flex().flex_col().gap(px(10.)).p(px(16.)).children((0..rows).map(|i| {
        div()
            .flex()
            .gap(px(12.))
            .items_center()
            .child(div().size(px(16.)).flex_none().child(skeleton(t, format!("{id}-a{i}"))))
            .child(
                // Relative widths so placeholders fit any panel.
                div()
                    .h(px(12.))
                    .w(gpui::relative(0.45 + (i % 3) as f32 * 0.12))
                    .min_w_0()
                    .child(skeleton(t, format!("{id}-b{i}"))),
            )
            .child(div().flex_1())
            .child(div().h(px(12.)).w(gpui::relative(0.15)).child(skeleton(t, format!("{id}-c{i}"))))
    }))
}

/// A two-line table cell: primary text over a tertiary detail.
pub fn cell2(t: &Theme, primary: impl Into<SharedString>, secondary: impl Into<SharedString>) -> Div {
    div()
        .flex()
        .flex_col()
        .min_w_0()
        .child(
            text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).truncate().child(primary.into()),
        )
        .child(text(ty::XS, t.text_tertiary).truncate().child(secondary.into()))
}

/// Table header cell.
pub fn th(t: &Theme, label: &str) -> Div {
    text(ty::XS, t.text_tertiary).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label.to_string())
}

/// A small inline bar (0..1) with its value, for metric columns.
pub fn bar_value(t: &Theme, v: f32, label: impl Into<SharedString>, color: gpui::Hsla) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(8.))
        .child(div().w(px(48.)).child(progress(t, v, color, 4.)))
        .child(text(ty::SM, t.text_primary).tnum().child(label.into()))
}
