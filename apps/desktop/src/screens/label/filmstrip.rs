//! Thumbnail filmstrip beneath the Agent Labeling review stage.

use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString, Styled, Window, div,
    px,
};

use crate::api::{LabelImage, LabelImageStatus};
use crate::app::{FoloApp, act};
use crate::store::ImgKind;
use crate::theme::{Theme, alpha, radius};
use crate::viz::{DrawBox, Fit};

/// Computes how many 72 px thumbnails (with 6 px gaps) fit inside `review_w` without clipping.
pub(crate) fn filmstrip_max_tiles(review_w: f32) -> usize {
    // 28 px horizontal stage padding + 2 px panel border.
    let inner = (review_w - 30.).max(0.);
    (((inner + 6.) / 78.).floor() as usize).clamp(3, 12)
}

impl FoloApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn label_filmstrip(
        &mut self,
        t: &Theme,
        job_id: &str,
        shown: &[LabelImage],
        active_id: &str,
        review_w: f32,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        // Fit the thumbnail window to the review column width so no tile or active image ever clips.
        let cur = shown.iter().position(|im| im.id == active_id).unwrap_or(0);
        let max_tiles = filmstrip_max_tiles(review_w);
        let start = cur.saturating_sub(max_tiles / 2).min(shown.len().saturating_sub(max_tiles));
        let end = (start + max_tiles).min(shown.len());
        let tiles: Vec<AnyElement> = shown[start..end]
            .iter()
            .map(|im| {
                let is_cur = im.id == active_id;
                let thumb = self.label_image_px(job_id, &im.id, ImgKind::Thumb, cx);
                let dims = (im.width.max(1) as f32, im.height.max(1) as f32);
                let boxes: Vec<DrawBox> =
                    im.boxes.iter().map(|b| DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)).collect();
                let dot_color = match (im.status, im.reviewed) {
                    (LabelImageStatus::Labeled, true) => t.success_solid,
                    (LabelImageStatus::Labeled, false) => t.accent_solid,
                    (LabelImageStatus::Failed, _) => t.danger_solid,
                    _ => t.text_disabled,
                };
                let target_id = im.id.clone();
                let key = format!("label-thumb-{}", im.id);
                let el = div()
                    .id(SharedString::from(key.clone()))
                    .w(px(72.))
                    .h(px(54.))
                    .flex_none()
                    .relative()
                    .rounded(px(radius::SM))
                    .overflow_hidden()
                    .cursor_pointer()
                    .border_2()
                    .border_color(if is_cur { t.accent_solid } else { alpha(t.line_subtle, 0.8) })
                    .child(crate::screens::image_canvas(t, thumb, dims, boxes, Fit::Cover, false, false))
                    .child(
                        div().absolute().top(px(4.)).right(px(4.)).size(px(7.)).rounded_full().bg(dot_color),
                    );
                self.bind(
                    el,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| this.request_select_label_image(target_id.clone(), cx)),
                )
                .into_any_element()
            })
            .collect();

        div().flex().items_center().gap(px(6.)).overflow_hidden().children(tiles).into_any_element()
    }
}
