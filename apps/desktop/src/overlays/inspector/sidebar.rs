//! Right-hand sidebar of the image inspector: file metadata, clickable label/prediction rows
//! (click to zoom to a box), and keyboard shortcut legend.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{ImageInfo, Label, ScoredBox};
use crate::app::{FoloApp, act};
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn inspector_sidebar(
        &self,
        t: &Theme,
        meta: &ImageInfo,
        dims: (f32, f32),
        classes: &[String],
        preds: Option<&[ScoredBox]>,
        sample_line: Option<(String, String)>,
        hover: Option<usize>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let n_labels = meta.boxes.len();
        // By class, then largest first, like the PR legend; indices stay the image's.
        let mut box_order: Vec<(usize, &crate::api::BBox)> = meta.boxes.iter().enumerate().collect();
        box_order.sort_by(|(_, a), (_, b)| {
            let area = |x: &crate::api::BBox| (x.x2 - x.x1) * (x.y2 - x.y1);
            a.class.cmp(&b.class).then(area(b).total_cmp(&area(a)))
        });
        let insp_chrome = self.scroll_chrome(self.insp_list_scroll.clone(), t.bg_overlay, 0., 0., cx);
        let mut box_rows: Vec<AnyElement> = box_order
            .into_iter()
            .map(|(i, b)| {
                let hot = hover == Some(i);
                let key = format!("insp-box-{i}");
                let coords = (b.x1, b.y1, b.x2, b.y2);
                // The class column is 70 px: a name that may not fit shows in full on hover and on
                // keyboard focus (short ones need neither).
                let full_name = classes.get(b.class as usize).cloned().unwrap_or_default();
                let long = crate::ui::may_overflow(&full_name, ty::SM.size, 70.);
                let row = self
                    .disclose(
                        div().id(SharedString::from(key.clone())),
                        &key,
                        full_name,
                        long,
                        30.,
                        false,
                        window,
                        cx,
                    )
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(t)))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            if *h {
                                insp.hover_from_stage = false;
                                if insp.hover_box != Some(i) {
                                    insp.hover_box = Some(i);
                                    cx.notify();
                                }
                            } else if !insp.hover_from_stage && insp.hover_box == Some(i) {
                                insp.hover_box = None;
                                cx.notify();
                            }
                        }
                    }))
                    .child(crate::ui::class_swatch(t, b.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .ellipsis()
                            .child(classes.get(b.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(
                        text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                            .tnum()
                            .flex_1()
                            .text_right()
                            .child(format!(
                                "{:.0}, {:.0} · {:.0}×{:.0}",
                                b.x1,
                                b.y1,
                                b.x2 - b.x1,
                                b.y2 - b.y1
                            )),
                    );
                // Focusable: Tab highlights a box; clicking or pressing Enter/Space zooms to it.
                self.bind(
                    row,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| this.focus_inspector_box(i, dims, coords, cx)),
                )
                .into_any_element()
            })
            .collect();
        // The epoch's predictions, after the labels: score and box, each a Tab stop that
        // highlights and zooms to its box (as a label row does).
        if let Some(ps) = preds {
            box_rows.push(
                div()
                    .px(px(8.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(t, "Predictions"))
                    .child(text(ty::SM, t.text_tertiary).child("score · x, y · w×h"))
                    .into_any_element(),
            );
            for (j, p) in ps.iter().enumerate() {
                let idx = n_labels + j;
                let hot = hover == Some(idx);
                let key = format!("insp-pred-{j}");
                let coords = (p.x1, p.y1, p.x2, p.y2);
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .h(px(30.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(t)))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if let Some(insp) = this.inspector.as_mut() {
                            if *h {
                                insp.hover_from_stage = false;
                                if insp.hover_box != Some(idx) {
                                    insp.hover_box = Some(idx);
                                    cx.notify();
                                }
                            } else if !insp.hover_from_stage && insp.hover_box == Some(idx) {
                                insp.hover_box = None;
                                cx.notify();
                            }
                        }
                    }))
                    .child(crate::ui::class_swatch(t, p.class, 8.))
                    .child(
                        text(ty::SM, t.text_primary)
                            .w(px(70.))
                            .ellipsis()
                            .child(classes.get(p.class as usize).cloned().unwrap_or_default()),
                    )
                    .child(text(ty::SM, t.text_primary).tnum().w(px(32.)).child(format!("{:.2}", p.score)))
                    .child(
                        text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                            .tnum()
                            .flex_1()
                            .text_right()
                            .child(format!(
                                "{:.0}, {:.0} · {:.0}×{:.0}",
                                p.x1,
                                p.y1,
                                p.x2 - p.x1,
                                p.y2 - p.y1
                            )),
                    );
                box_rows.push(
                    self.bind(
                        row,
                        &key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, _, cx| this.focus_inspector_box(idx, dims, coords, cx)),
                    )
                    .into_any_element(),
                );
            }
        }
        div()
            .w(px(300.))
            .flex_none()
            .flex()
            .flex_col()
            .bg(t.bg_overlay)
            .border_l_1()
            .border_color(t.line_subtle)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .p(px(16.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child({
                        // The panel's 268 px hold ~33 monospace glyphs (0.6 em each); a longer
                        // path ends in "…" and shows in full on hover.
                        let long = meta.file.chars().count() as f32 * 0.6 * ty::BASE.size > 268.;
                        let el = div().id("insp-file");
                        self.disclose(el, "insp-file", meta.file.clone(), long, 24., true, window, cx).child(
                            text(ty::BASE, t.text_primary)
                                .font_family(crate::theme::MONO)
                                .ellipsis()
                                .child(meta.file.clone()),
                        )
                    })
                    .child(
                        div()
                            .flex()
                            .gap(px(6.))
                            .child(badge(t, meta.split.label(), Tone::Neutral))
                            .child(badge(t, format!("{} × {}", meta.width, meta.height), Tone::Neutral))
                            .child(badge(
                                t,
                                match preds {
                                    Some(_) => counted(meta.boxes.len() as u64, "label", "labels"),
                                    None => counted(meta.boxes.len() as u64, "box", "boxes"),
                                },
                                Tone::Neutral,
                            ))
                            .when_some(preds.map(<[ScoredBox]>::len), |d, m| {
                                d.child(badge(
                                    t,
                                    format!("{m} {}", if m == 1 { "prediction" } else { "predictions" }),
                                    Tone::Neutral,
                                ))
                            }),
                    )
                    .when_some(sample_line, |d, (run, how)| {
                        d.child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(2.))
                                .child(text(ty::SM, t.text_secondary).ellipsis().child(run))
                                .child(text(ty::SM, t.text_tertiary).child(how)),
                        )
                    }),
            )
            .child(
                div()
                    .px(px(16.))
                    .pt(px(12.))
                    .pb(px(4.))
                    .flex()
                    .justify_between()
                    .child(overline(t, "Labels"))
                    .child(text(ty::SM, t.text_tertiary).child("top-left x, y · w×h (px)")),
            )
            .child(
                div()
                    .relative()
                    .flex_shrink()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("insp-boxes")
                            // As tall as its rows (up to the room left), so the key hints sit right
                            // under it.
                            .flex_shrink()
                            .min_h_0()
                            .overflow_y_scroll()
                            .track_scroll(&self.insp_list_scroll)
                            .px(px(8.))
                            .children(box_rows)
                            .when(meta.boxes.is_empty(), |d| {
                                d.child(div().p(px(8.)).child(
                                    text(ty::SM, t.text_tertiary).child("No labels: a background image."),
                                ))
                            }),
                    )
                    .children(insp_chrome),
            )
            .child(inspector_key_hints(t))
            .into_any_element()
    }
}

/// Keyboard and pointer hint legend anchored beneath the inspector's label list.
fn inspector_key_hints(t: &Theme) -> AnyElement {
    let row = |keys: AnyElement, what: &'static str| {
        div()
            .flex()
            .items_center()
            .child(div().w(px(96.)).flex_none().flex().flex_wrap().items_center().gap(px(4.)).child(keys))
            // At 12 px the longest hint needs two lines in the 300 px panel: it wraps inside its
            // column rather than running past the edge.
            .child(text(ty::SM, t.text_tertiary).flex_1().min_w_0().whitespace_normal().child(what))
    };
    div()
        .p(px(12.))
        .flex()
        .flex_col()
        .gap(px(6.))
        .border_t_1()
        .border_color(t.line_subtle)
        .child(row(
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .child(shortcut(t, &["←", "→"]))
                .child(shortcut(t, &["PgUp", "PgDn"]))
                .into_any_element(),
            "previous / next",
        ))
        .child(row(
            div()
                .flex()
                .gap(px(4.))
                .child(kbd(t, "Scroll"))
                .child(shortcut(t, &["+", "−"]))
                .into_any_element(),
            "zoom",
        ))
        .child(row(kbd(t, "0").into_any_element(), "fit"))
        .child(row(shortcut(t, &["B", "L"]).into_any_element(), "boxes / labels"))
        .child(row(kbd(t, "Click").into_any_element(), "zoom to label (again to fit)"))
        .child(row(kbd(t, "Drag").into_any_element(), "pan (arrows too, when zoomed)"))
        .into_any_element()
}
