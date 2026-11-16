//! An entry's images panel: sorted, paged, with what the contestant found on each.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Bench, BenchEntry};
use crate::app::{FoloApp, act};
use crate::bench::ImageSort;
use crate::controls::Btn;
use crate::screens::image_canvas;
use crate::store::Load;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

impl FoloApp {
    /// An entry's images: the truth (dashed) and its detections at its best-F1 confidence, worst first
    /// or in the benchmark's order, a page at a time.
    pub(super) fn entry_images_panel(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        best_conf: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        self.ensure_entry_images(&b.id, &e.id, cx);
        let sort = self.bench.image_sort;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let sorts = [ImageSort::Worst, ImageSort::Order];
        let seg = self.segmented(
            "entry-img-sort",
            &["Worst first", "In order"],
            sorts.iter().position(|s| *s == sort).unwrap_or(0),
            window,
            cx,
            std::rc::Rc::new(move |this, i, _, cx| {
                this.bench.image_sort = sorts[i.min(1)];
                this.ensure_entry_images(&bid, &eid, cx);
                cx.notify();
            }),
        );
        let header = panel_header(&t, "Images").child(seg);
        let state = self
            .bench
            .entry_images
            .get(&(b.id.clone(), e.id.clone()))
            .map(|x| (x.page.clone(), x.more, x.more_error.clone()));
        let (page, loading_more, more_error) = match state {
            Some((Load::Ready(p), more, err)) => (p, more, err),
            Some((Load::Failed(err), _, _)) => {
                return panel(&t)
                    .child(header)
                    .child(div().p(px(16.)).child(text(ty::SM, t.danger_text).child(err.message)))
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(header)
                    .child(div().p(px(16.)).flex().flex_wrap().gap(px(8.)).children(
                        (0..8).map(|i| div().w(px(152.)).h(px(152.)).child(skeleton(&t, format!("eimg{i}")))),
                    ))
                    .into_any_element();
            }
        };
        let tiles: Vec<AnyElement> =
            page.images
                .iter()
                .map(|im| {
                    let img = self.bench_image(&b.id, &im.image.id, None, cx);
                    let mut boxes: Vec<DrawBox> = im
                        .image
                        .boxes
                        .iter()
                        .map(|x| DrawBox { truth: true, ..DrawBox::new(x.class, x.x1, x.y1, x.x2, x.y2) })
                        .collect();
                    boxes.extend(im.detections.iter().map(|d| DrawBox {
                        label: b.classes.get(d.class as usize).cloned().map(Into::into),
                        score: e.scored.then_some(d.score),
                        ..DrawBox::new(d.class, d.x1, d.y1, d.x2, d.y2)
                    }));
                    let dims = (im.image.width as f32, im.image.height as f32);
                    // Failed, not loading: said on the frame, which is then a Retry (click, Enter, Space).
                    let failed = img.is_none() && self.bench_image_failed(&b.id, &im.image.id, None);
                    let painted = image_canvas(&t, img, dims, boxes, Fit::Contain, true, false);
                    let painted =
                        if failed { crate::screens::unavailable_over(&t, painted, true) } else { painted };
                    // The image clips inside; the control itself doesn't, so its focus ring shows whole.
                    let frame =
                        div().w(px(152.)).h(px(152.)).rounded(px(radius::MD)).bg(t.bg_inset).child(
                            div().size_full().rounded(px(radius::MD)).overflow_hidden().child(painted),
                        );
                    let key = format!("eimg-retry-{}", im.image.id);
                    let frame = if failed {
                        self.passing_stop(&key, true, cx);
                        let (bid, iid) = (b.id.clone(), im.image.id.clone());
                        let el = frame.id(SharedString::from(key.clone())).cursor_pointer();
                        self.bind(
                            el,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, _, cx| this.retry_bench_image(&bid, &iid, None, cx)),
                        )
                        .into_any_element()
                    } else {
                        match self.passing_stop(&key, false, cx) {
                            Some(fh) => frame.track_focus(&fh).into_any_element(),
                            None => frame.into_any_element(),
                        }
                    };
                    let ok = im.missed == 0 && im.wrong == 0;
                    let caption_text = if ok {
                        "all right".to_string()
                    } else {
                        [(im.missed, "missed"), (im.wrong, "wrong")]
                            .iter()
                            .filter(|(n, _)| *n > 0)
                            .map(|(n, w)| format!("{n} {w}"))
                            .collect::<Vec<_>>()
                            .join(" · ")
                    };
                    div()
                        .id(SharedString::from(format!("eimg-{}", im.image.id)))
                        .w(px(152.))
                        .flex_none()
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .tooltip(crate::controls::tip(
                            if failed {
                                format!("{} · {} · Retry", im.image.file, crate::screens::TILE_UNAVAILABLE)
                            } else {
                                im.image.file.clone()
                            },
                            None,
                        ))
                        .child(frame)
                        .child(
                            text(ty::SM, if ok { t.text_tertiary } else { t.text_secondary })
                                .tnum()
                                .child(caption_text),
                        )
                        .into_any_element()
                })
                .collect();
        let more = (page.images.len() as u64) < page.total;
        let (bid, eid) = (b.id.clone(), e.id.clone());
        let footer = div()
            .px(px(16.))
            .pb(px(12.))
            .flex()
            .items_center()
            .justify_between()
            .gap(px(12.))
            .child(text(ty::SM, t.text_tertiary).tnum().child(format!(
                "{} of {} images · dashed: the truth · solid: its detections{}",
                int(page.images.len() as u64),
                int(page.total),
                if e.scored {
                    format!(" at confidence {} and up", fmt_conf(best_conf))
                } else {
                    String::new()
                },
            )))
            .when_some(more_error, |d, err| d.child(text(ty::SM, t.danger_text).child(err.message)))
            .when(more, |d| {
                d.child(if loading_more {
                    self.button_disabled(
                        "entry-img-more",
                        "Loading…",
                        Btn::Secondary,
                        Some("icons/chevron-down.svg"),
                        "Loading the next images",
                        window,
                        cx,
                    )
                } else {
                    self.button(
                        "entry-img-more",
                        "Show more",
                        Btn::Secondary,
                        Some("icons/chevron-down.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.more_entry_images(&bid, &eid, cx)),
                    )
                })
            });
        panel(&t)
            .child(header)
            .child(div().p(px(16.)).flex().flex_wrap().gap(px(12.)).children(tiles))
            .child(footer)
            .into_any_element()
    }
}
