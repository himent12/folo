//! One module per screen, each an `impl FoloApp` block returning the page element.

pub mod bench;
pub mod datasets;
mod first_step;
pub mod models;
pub mod overview;
pub mod playground;
pub mod runs;
pub mod settings;

use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Div, FontWeight, IntoElement, ParentElement, RenderImage, SharedString, Styled,
    Window, canvas, div, px,
};

use crate::api::{ApiError, ImageInfo};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::store::ImgKind;
use crate::theme::{Theme, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, paint_image_with_boxes};

pub use first_step::FirstStep;
pub(crate) use overview::dataset_problem;

/// What a dataset tile whose image failed says (Studio's words).
pub const TILE_UNAVAILABLE: &str = "Image unavailable";

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
                .child(div().h(px(ty::XL.line)).flex().items_center().min_w_0().child(title))
                .child(div().text_color(t.text_secondary).text_size(px(ty::SM.size)).child(subtitle)),
        )
        .child(
            div()
                .min_h(px(ty::XL.line))
                .min_w_0()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_end()
                .gap(px(8.))
                .children(actions),
        )
}

impl FoloApp {
    /// A dataset image painted with optional boxes, at a fixed tile size. `labels` puts class chips
    /// on the boxes; catalogue previews leave them off (a montage of chips reads as a label wall).
    #[allow(clippy::too_many_arguments)]
    pub fn tile(
        &mut self,
        dataset: &str,
        meta: &ImageInfo,
        boxes: bool,
        labels: bool,
        fit: Fit,
        reveal: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        self.tile_with_classes(dataset, meta, None, boxes, labels, fit, reveal, cx)
    }

    /// Same as [`Self::tile`], reusing a pre-interned slice of class names when provided so visible
    /// grid tiles clone `SharedString` handles instead of allocating per box.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tile_with_classes(
        &mut self,
        dataset: &str,
        meta: &ImageInfo,
        shared_classes: Option<&[SharedString]>,
        boxes: bool,
        labels: bool,
        fit: Fit,
        reveal: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let img = self.image(dataset, &meta.id, ImgKind::Thumb, cx);
        let dims = (meta.width as f32, meta.height as f32);
        // On a dataset page with a class filter, the other classes drop to 30 % (the selection rule),
        // so the filter shows inside each tile.
        let focus = self
            .class_filter(dataset)
            .filter(|_| matches!(&self.route, crate::app::Route::Dataset(d) if d == dataset));
        let draw: Vec<DrawBox> = if boxes {
            let fallback = if shared_classes.is_none() {
                self.store.dataset_summary(dataset).map(|d| d.classes.as_slice()).unwrap_or_default()
            } else {
                &[]
            };
            // Filtered, a tile at rest names one match: the largest the painter can label (it tries
            // them largest first), so every tile reads the same way; hovered or focused, all of them.
            let one = !reveal;
            meta.boxes
                .iter()
                .map(|b| {
                    let label = match shared_classes {
                        Some(classes) => classes.get(b.class as usize).cloned(),
                        None => fallback.get(b.class as usize).map(|n| SharedString::from(n.clone())),
                    };
                    DrawBox {
                        dim: focus.is_some_and(|c| c != b.class),
                        one_of: one && focus == Some(b.class),
                        label,
                        ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let unavailable = img.is_none() && self.thumb_unavailable(dataset, &meta.id);
        // Class names on the boxes (14 px chips in small views; none on boxes under 20 px), so a
        // class never has to be matched by colour alone.
        let painted = image_canvas(&t, img, dims, draw, fit, labels, reveal);
        if !unavailable {
            return painted;
        }
        // Failed, not loading (d1-datasets, x4-soak): said on the tile, as Studio's DatasetImage
        // does. Its Retry is the inspector's, a press of the tile (Enter or a click) away; the tile
        // itself is one focus stop that opens it, so no second control is nested in it.
        unavailable_over(&t, painted, true)
    }

    /// A dataset tile's thumbnail failed (it waits for its next automatic retry, or for a Retry by
    /// hand once those are spent): the tile says "Image unavailable" instead of looking as if it
    /// still loads.
    pub(crate) fn thumb_unavailable(&self, dataset: &str, image: &str) -> bool {
        self.image_failed(dataset, image, ImgKind::Thumb)
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
        // The address answered, but not as folo serve: the health check says so, and what a page's
        // own call got back (a web page, another answer) would only say it couldn't read it.
        let e = match self.store.health.error() {
            Some(h) if h.code == "not_folo" => h,
            _ => e,
        };
        let hint = match &self.start_error {
            // Nothing answers because the local server didn't start: why, not "start it yourself" alone.
            Some(why) if e.code == "server_unreachable" => {
                format!(
                    "Couldn't start folo serve: {why}. Start one yourself, or switch to the built-in demo in Settings."
                )
            }
            _ => e.hint.clone().unwrap_or_else(|| format!("Error code {}", e.code)),
        };
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
        // A list's answer still out (the refresh's) would undo the request asked for now.
        self.restart_lists();
        // A `/system` answer still out is not waited for: the next one is asked now, and that
        // older one dropped when it lands (`fetch_system`), also before the next is asked.
        self.system_out = false;
        self.system_seq += 1;
        let s = &mut self.store;
        s.health = Load::Idle;
        s.system = Load::Idle;
        s.capabilities = None;
        s.data_dir = None;
        s.features.clear();
        s.datasets = Load::Idle;
        s.models = Load::Idle;
        s.runs = Load::Idle;
        s.dataset.clear();
        s.images.clear();
        s.previews.clear();
        s.evals.clear();
        // Per-dataset issue counts too: ids repeat across servers, and a fetch dropped by a
        // backend switch would otherwise stay Loading.
        s.issues.clear();
        s.run_data.clear();
        s.samples.clear();
        s.logs.clear();
        // Images that failed are asked for again (their few automatic retries may be spent).
        s.images_px.forget_failed();
        self.images_more.clear();
        self.failed_pages.clear();
        // The throughput table reloads too (a probe in flight keeps its own state), and the cloud.
        self.throughput = Load::Idle;
        self.clear_cloud();
        self.clear_bench();
        cx.notify();
    }
}

/// A painted image whose pixels failed to load, marked so over its placeholder: an image glyph and,
/// with `words` (a tile tall enough), "Image unavailable". Without the mark it reads as loading.
pub fn unavailable_over(t: &Theme, painted: AnyElement, words: bool) -> AnyElement {
    div().size_full().relative().child(painted).child(unavailable_mark(t, words)).into_any_element()
}

/// The mark of [`unavailable_over`] alone, laid over its positioned parent: for a frame whose size
/// comes from an aspect ratio, where a full-size wrapper inside it would have no height.
pub fn unavailable_mark(t: &Theme, words: bool) -> Div {
    div()
        .absolute()
        .inset_0()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap(px(4.))
        .px(px(6.))
        .child(icon("icons/image.svg", 16., t.text_tertiary))
        .when(words, |d| d.child(text(ty::XS, t.text_tertiary).text_center().child(TILE_UNAVAILABLE)))
}

/// Paint an image with boxes filling its parent.
pub fn image_canvas(
    t: &Theme,
    img: Option<Arc<RenderImage>>,
    dims: (f32, f32),
    boxes: Vec<DrawBox>,
    fit: Fit,
    labels: bool,
    reveal: bool,
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
                reveal,
                0.,
            );
        },
    )
    .size_full()
    .into_any_element()
}

/// Shimmering placeholder blocks shaped like a panel of rows.
pub fn loading_rows(t: &Theme, id: &str, rows: usize) -> Div {
    // Shaped like the tables they stand in for: 36 px rows on hairlines, a name at the left and a
    // right-aligned number (no leading icon, which no table has).
    div().flex().flex_col().children((0..rows).map(|i| {
        div()
            .h(px(36.))
            .px(px(16.))
            .flex()
            .gap(px(16.))
            .items_center()
            .border_b_1()
            .border_color(t.line_subtle)
            .child(
                // Relative widths so placeholders fit any panel.
                div()
                    .h(px(10.))
                    .w(folo_ui::relative(0.3 + (i % 3) as f32 * 0.08))
                    .min_w_0()
                    .child(skeleton(t, format!("{id}-b{i}"))),
            )
            .child(div().flex_1())
            .child(div().h(px(10.)).w(px(48.)).flex_none().child(skeleton(t, format!("{id}-c{i}"))))
    }))
}

/// A two-line table cell: primary text over a tertiary detail.
pub fn cell2(t: &Theme, primary: impl Into<SharedString>, secondary: impl Into<SharedString>) -> Div {
    // It takes the row's remaining width (basis 0), so its ellipses cut only where the row ends,
    // not at the text's own minimum.
    div()
        .flex_1()
        .flex_basis(px(0.))
        .flex()
        .flex_col()
        .min_w_0()
        .child(
            text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).ellipsis().child(primary.into()),
        )
        .child(text(ty::SM, t.text_tertiary).ellipsis().child(secondary.into()))
}

/// Table header cell (design §2 Tables: the head row in `sm` secondary ink).
pub fn th(t: &Theme, label: &str) -> Div {
    text(ty::SM, t.text_secondary)
        .font_weight(FontWeight::MEDIUM)
        .whitespace_nowrap()
        .child(label.to_string())
}
