//! A dataset's stat strip (the Statistics tab's KPI strip): counts, splits and label health. Its
//! Empty images tile shows them in the grid and its Broken labels tile copies the check command.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::list::names_summary;
use crate::api::{Dataset, SplitsExt};
use crate::app::{DsTab, FoloApp, act};
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn dataset_strip_stats(
        &self,
        ds: &Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // In the Statistics tab, under the health line (it no longer sits above the image grid, so
        // it keeps its full size in short windows too).
        let (step, pad) = (ty::X2XL, 12.);
        let stat = |t: &Theme, label: &str, value: String, unit: &str, detail: Option<Div>| {
            stat_at(t, label, value, unit, detail, step)
        };
        let total = ds.splits.total().max(1) as f32;
        let s = ds.stats.as_ref();
        let split_c = [
            t.text_tertiary,
            crate::theme::alpha(t.text_tertiary, 0.6),
            crate::theme::alpha(t.text_tertiary, 0.35),
        ];
        let split_bar = div()
            .flex()
            .h(px(4.))
            .w_full()
            .rounded_full()
            .overflow_hidden()
            .gap(px(2.))
            .child(div().h_full().w(folo_ui::relative(ds.splits.train as f32 / total)).bg(split_c[0]))
            .child(div().h_full().w(folo_ui::relative(ds.splits.val as f32 / total)).bg(split_c[1]))
            .when(ds.splits.test > 0, |d| d.child(div().h_full().flex_1().bg(split_c[2])));
        let legend = div().flex().gap(px(10.)).children(
            [
                ("train", ds.splits.train, split_c[0]),
                ("val", ds.splits.val, split_c[1]),
                ("test", ds.splits.test, split_c[2]),
            ]
            .into_iter()
            .filter(|x| x.1 > 0)
            .map(|(n, c, col)| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(5.))
                    .child(swatch(col, 6.))
                    .child(kpi_note(&t).child(format!("{n} {}", int(c))))
            }),
        );
        let boxes = s.map_or(0, |s| s.boxes);
        let per = boxes as f32 / total;
        let broken = s.map_or(0, |s| s.broken_labels);
        // Size the strip by actual content width (which shrinks at 1120 px when the sidebar expands),
        // not raw viewport width, so the Broken labels tile never clips at the sidebar breakpoint.
        let cw = self.content_width(window);
        let narrow_strip = cw < 1000.;
        let tile_px = if cw < 780. { 12. } else { 16. };
        let tile = |content: folo_ui::Div| {
            div().flex_1().min_w_0().overflow_hidden().px(px(tile_px)).py(px(pad)).child(content)
        };
        let img_w = if cw < 780. {
            200.
        } else if narrow_strip {
            244.
        } else {
            300.
        };
        panel(&t)
            .flex()
            .child(div().flex_none().w(px(img_w)).px(px(tile_px)).py(px(pad)).child(
                // The split bar on the strip's first detail line (where the others' captions
                // sit), its legend on the second: the same grid as every tile beside it.
                stat(
                    &t,
                    "Images",
                    int(ds.splits.total()),
                    "",
                    Some(div().flex().flex_col().gap(px(6.)).child(detail_line(split_bar)).child(legend)),
                ),
            ))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Boxes",
                int(boxes),
                "",
                Some(kpi_note(&t).child(format!("{per:.1} per image"))),
            )))
            .child(vline(&t))
            .child(
                tile(stat(
                    &t,
                    "Classes",
                    ds.classes.len().to_string(),
                    "",
                    Some(kpi_note(&t).min_w_0().w_full().ellipsis().child(names_summary(&ds.classes, 24))),
                ))
                .id("ds-classes-tile")
                .when(!ds.classes.is_empty(), |d| {
                    d.tooltip(crate::controls::tip(ds.classes.join(", "), None))
                }),
            )
            .child(vline(&t))
            .child({
                // Defects are actionable: this tile filters the grid to the images in question.
                let empty = s.map_or(0, |s| s.empty_images);
                let on = self.empty_only(&ds.id);
                let el = div()
                    .id("ds-empty-tile")
                    .flex_1()
                    .min_w_0()
                    .px(px(tile_px))
                    .py(px(pad))
                    .when(empty > 0, |d| {
                        d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active))
                    })
                    .when(on, |d| d.bg(t.bg_selected))
                    // Clip tile contents inside the focusable wrapper so `bind`'s exterior focus ring is never clipped.
                    .child(div().min_w_0().w_full().overflow_hidden().child(stat(
                        &t,
                        "Empty images",
                        int(empty),
                        "",
                        Some(if empty > 0 {
                            if on {
                                text(ty::SM, t.accent_text).child("Showing them · clear")
                            } else {
                                inline_link(&t, "Show them")
                            }
                        } else if ds.splits.total() == 0 {
                            // Nothing to have labels: not "every image has labels" for no image.
                            kpi_note(&t).child("no images")
                        } else {
                            kpi_note(&t).child("every image has labels")
                        }),
                    )));
                if empty > 0 {
                    self.bind(
                        el,
                        "ds-empty-tile",
                        0.,
                        window,
                        cx,
                        act({
                            let id = ds.id.clone();
                            move |this, w, cx| {
                                this.ds_empty_of = (!this.empty_only(&id)).then(|| id.clone());
                                // Empty images have no class: the two filters can't both apply to this
                                // dataset (another's class filter stays with it).
                                if this.ds_empty_of.is_some()
                                    && this.ds_class.as_ref().is_some_and(|(of, _)| *of == id)
                                {
                                    this.ds_class = None;
                                }
                                this.ds_tab = DsTab::Images;
                                this.ds_hover_tile = None;
                                this.ds_scroll_anchor = None;
                                this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
                                let target =
                                    if this.ds_empty_of.is_some() { "ds-empty-chip" } else { "ds-class" };
                                let fh = this.fh(target, cx);
                                w.focus(&fh);
                                cx.notify();
                            }
                        }),
                    )
                    .into_any_element()
                } else {
                    el.into_any_element()
                }
            })
            .child(vline(&t))
            .child({
                let el = div()
                    .id("ds-broken-tile")
                    .flex_1()
                    .min_w_0()
                    .px(px(tile_px))
                    .py(px(pad))
                    .when(broken > 0, |d| {
                        d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active))
                    })
                    .when(broken > 0, |d| d.tooltip(crate::controls::tip(broken_tip(), None)))
                    // Clip tile contents inside the focusable wrapper so `bind`'s exterior focus ring is never clipped.
                    .child(div().min_w_0().w_full().overflow_hidden().child(if broken > 0 {
                        // Actionable: a warning, not one more inventory count.
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(6.))
                            .min_w_0()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(4.))
                                    .min_w_0()
                                    .child(div().flex_none().child(icon(
                                        "icons/triangle-alert.svg",
                                        12.,
                                        t.warning_solid,
                                    )))
                                    .child(
                                        text(ty::SM, t.warning_text)
                                            .font_weight(FontWeight::MEDIUM)
                                            .min_w_0()
                                            .ellipsis()
                                            .child("Broken labels"),
                                    ),
                            )
                            .child(metric(&t, broken.to_string(), "", step, ty::SM))
                            // Split by severity when the server reports it, then the copy action.
                            .when_some(
                                self.store
                                    .issues
                                    .get(&ds.id)
                                    .and_then(|l| l.ready())
                                    .filter(|c| c.error + c.warning > 0),
                                |d, c| d.child(severity_detail(&t, c)),
                            )
                            // The action, named (the tile copies `folo dataset check`).
                            .child(broken_detail(&t, narrow_strip))
                    } else {
                        stat(
                            &t,
                            "Broken labels",
                            broken.to_string(),
                            "",
                            Some(div().child(text(ty::SM, t.success_text).child("all labels valid"))),
                        )
                    }));
                if broken > 0 {
                    let (cmd, how) = check_command(&ds.path);
                    self.bind(
                        el,
                        "ds-broken-tile",
                        0.,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(cmd.clone()));
                            this.toast(
                                Tone::Neutral,
                                "Copied the check command",
                                Some(how.clone().into()),
                                cx,
                            );
                        }),
                    )
                    .into_any_element()
                } else {
                    el.into_any_element()
                }
            })
            .into_any_element()
    }
}

/// The Broken labels tile's tooltip: the health finding's words for what the count is, then the tile's action.
fn broken_tip() -> String {
    format!("{} Click to copy the terminal command that lists them.", super::health::BROKEN_LABELS_DETAIL)
}

/// The Broken labels tile's command for the dataset at `path` (the server's path, as the API gives
/// it), quoted as the CLI quotes it (`folo_api::json::quote_path`: a path with spaces in double
/// quotes), and what its toast says. A path a shell would change inside double quotes (`"`, `$`,
/// `%`, a backtick, a `\`) is left as `<dataset>` in the command, and the toast gives the path to
/// put there, quoted for the person's shell: the clipboard only ever holds a command line.
fn check_command(path: &str) -> (String, String) {
    match folo_api::json::quote_path(path) {
        Some(quoted) => {
            let cmd = format!("folo dataset check {quoted}");
            let how = format!("Run `{cmd}` in a terminal to list the broken labels.");
            (cmd, how)
        }
        None => (
            "folo dataset check <dataset>".into(),
            format!("Replace <dataset> with this path, quoted for your shell, then run it: {path}"),
        ),
    }
}

/// Under the broken-label count: the check command to copy, or that every label is valid.
fn broken_detail(t: &crate::theme::Theme, narrow: bool) -> folo_ui::Div {
    if narrow { copy_cmd(t, "Copy check") } else { copy_cmd(t, "Copy check command") }
}

/// "2 errors · 1 warning" in their severity colours.
fn severity_detail(t: &crate::theme::Theme, c: &crate::api::IssueCounts) -> folo_ui::Div {
    let n = |k: u64, one: &str, many: &str| format!("{k} {}", if k == 1 { one } else { many });
    div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap_x(px(4.))
        .gap_y(px(2.))
        .min_w_0()
        .when(c.error > 0, |d| {
            d.child(
                text(ty::SM, t.danger_text).tnum().whitespace_nowrap().child(n(c.error, "error", "errors")),
            )
        })
        .when(c.error > 0 && c.warning > 0, |d| d.child(text(ty::SM, t.text_tertiary).child("·")))
        .when(c.warning > 0, |d| {
            d.child(
                text(ty::SM, t.warning_text)
                    .tnum()
                    .whitespace_nowrap()
                    .child(n(c.warning, "warning", "warnings")),
            )
        })
}

fn copy_cmd(t: &crate::theme::Theme, label: &'static str) -> folo_ui::Div {
    div()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .child(text(ty::SM, t.text_secondary).min_w_0().ellipsis().child(label))
        .child(div().flex_none().child(icon("icons/copy.svg", 12., t.text_secondary)))
}

#[cfg(test)]
mod tests {
    /// `stats.broken_labels` counts labels, and a bad COCO category or annotation leaves no image out
    /// (docs/API.md): the tile's tooltip says what the health finding says, not "label files ... images skipped".
    #[test]
    fn the_broken_labels_tip_counts_labels_as_the_health_finding_does() {
        let tip = super::broken_tip();
        assert!(tip.starts_with(super::super::health::BROKEN_LABELS_DETAIL), "{tip}");
        assert!(tip.ends_with("Click to copy the terminal command that lists them."), "{tip}");
        assert!(!tip.contains("label files") && !tip.contains("images are skipped"), "{tip}");
    }

    /// #382: the tile wrapped the path in double quotes itself, so a quote in it ended the argument
    /// and a `$` in it was expanded by PowerShell; now it is quoted as the CLI quotes its commands.
    /// Review round 1: the path is taken as the server wrote it (a POSIX `\` kept, never made a
    /// `/`), and what is copied is a command line, the path to put in it said in the toast.
    #[test]
    fn the_check_command_quotes_its_path_as_the_cli_does() {
        let cmd = |p: &str| super::check_command(p).0;
        assert_eq!(cmd("D:/data/shapes"), "folo dataset check D:/data/shapes");
        assert_eq!(cmd("D:/my data/shapes"), "folo dataset check \"D:/my data/shapes\"");
        assert_eq!(
            super::check_command("D:/my data").1,
            "Run `folo dataset check \"D:/my data\"` in a terminal to list the broken labels."
        );
        for odd in ["D:/O\"Brien/$set", r"/data/set\blue"] {
            let (cmd, how) = super::check_command(odd);
            assert_eq!(cmd, "folo dataset check <dataset>");
            assert_eq!(
                how,
                format!("Replace <dataset> with this path, quoted for your shell, then run it: {odd}")
            );
        }
    }

    #[test]
    fn long_class_name_clips_inside_classes_tile() {
        use crate::app::{DsTab, FoloApp, Route, tests::drive};
        use crate::store::Load;
        use folo_ui::{
            AppContext, Bounds, HeadlessApp, HeadlessOptions, WindowBounds, WindowOptions, point, px, size,
        };

        let api = std::sync::Arc::new(crate::api::mock::MockBackend::new(4200, true));
        let mut app =
            HeadlessApp::new(HeadlessOptions { size: size(px(960.), px(720.)), ..Default::default() });
        let win = app.update(|cx| {
            cx.set_global(crate::theme::Theme::new(crate::theme::Mode::Dark));
            let bounds = Bounds::new(point(px(0.), px(0.)), size(px(960.), px(720.)));
            let options =
                WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
            cx.open_window(options, |w, cx| cx.new(|cx| FoloApp::new(api, true, w, cx)))
                .expect("open 960px window")
        });
        for _ in 0..3 {
            app.frame(win.into());
            app.run_until_idle();
        }
        let long_name = "circle-shaped-object-with-an-intentionally-long-class-name-for-testing";
        drive(&mut app, win, |this, w, cx| {
            let mut ds = this.store.datasets.ready().unwrap()[0].clone();
            ds.classes = vec![long_name.into(), "square".into()];
            if let Some(stats) = ds.stats.as_mut() {
                stats.empty_images = 2;
            }
            let id = ds.id.clone();
            if let Load::Ready(list) = &mut this.store.datasets {
                list[0] = ds.clone();
            }
            this.store.dataset.insert(id.clone(), Load::Ready(ds));
            this.ds_tab = DsTab::Statistics;
            this.kb = true;
            this.focus_key("ds-empty-tile", w, cx);
            this.navigate(Route::Dataset(id), w, cx);
        });
        for _ in 0..3 {
            app.frame(win.into());
            app.run_until_idle();
        }
        drive(&mut app, win, |this, w, cx| {
            this.kb = true;
            this.focus_key("ds-empty-tile", w, cx);
            cx.notify();
        });
        app.frame(win.into());
        let scene = app.scene(win.into());
        // Locate the KPI strip's 4 vertical dividers (`vline`: 1 px wide, >= 60 px tall):
        // [Images | Boxes | Classes | Empty images | Broken labels].
        let mut dividers: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                folo_ui::Primitive::Quad(q) => Some(q),
                _ => None,
            })
            .filter(|q| {
                q.bounds.size.width == px(1.).scale(1.)
                    && q.bounds.size.height >= px(60.).scale(1.)
                    && q.bounds.origin.y >= px(100.).scale(1.)
                    && q.bounds.origin.y <= px(350.).scale(1.)
            })
            .collect();
        dividers.sort_by(|a, b| a.bounds.origin.x.partial_cmp(&b.bounds.origin.x).unwrap());
        assert_eq!(dividers.len(), 4, "expected 4 KPI strip dividers, got {}", dividers.len());
        let classes_left = dividers[1].bounds.origin.x;
        let classes_right = dividers[2].bounds.origin.x;
        let strip_top = dividers[1].bounds.origin.y;
        let strip_bottom = dividers[1].bounds.bottom();

        // Every glyph inside the Classes tile must be clipped within `[classes_left, classes_right]`
        // so the 70-char class name never bleeds into Empty images or Broken labels.
        let classes_glyphs: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                folo_ui::Primitive::MonoSprite(s) => Some(s),
                _ => None,
            })
            .filter(|s| {
                s.bounds.origin.y >= strip_top
                    && s.bounds.origin.y <= strip_bottom
                    && s.bounds.origin.x >= classes_left
                    && s.bounds.origin.x < classes_right
            })
            .collect();
        assert!(!classes_glyphs.is_empty(), "Classes tile rendered text glyphs");
        // With ellipsis active, the 70-char class name is truncated to far fewer glyphs than its full length.
        assert!(
            classes_glyphs.len() < 45,
            "Classes tile should ellipsize long class name rather than painting all 70+ glyphs (got {})",
            classes_glyphs.len()
        );
        for s in &classes_glyphs {
            assert!(
                s.clip.right() <= classes_right + px(1.).scale(1.),
                "Classes tile glyph clip spills past right divider: clip={:?} divider_x={:?}",
                s.clip,
                classes_right
            );
        }

        // Actionable Empty images tile keeps its keyboard focus ring visible outside the tile bounds
        // (not clipped away by an outer `overflow_hidden`).
        let focus_color = crate::theme::Theme::new(crate::theme::Mode::Dark).focus;
        // `folo-ui` emits a hollow focus ring as 4 thin border-band quads (top, right, bottom, left).
        let ring_quads: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                folo_ui::Primitive::Quad(q)
                    if q.border_color == focus_color
                        && q.bounds.origin.y < strip_top
                        && q.bounds.bottom() > strip_bottom =>
                {
                    Some(q.clip)
                }
                _ => None,
            })
            .collect();
        assert_eq!(ring_quads.len(), 4, "ds-empty-tile focus ring should emit all 4 edge bands");
        assert!(
            ring_quads.iter().any(|c| c.origin.y < strip_top && c.size.height > px(0.).scale(1.))
                && ring_quads.iter().any(|c| c.bottom() > strip_bottom && c.size.height > px(0.).scale(1.)),
            "ds-empty-tile focus ring top and bottom edges must paint unclipped outside the tile: {ring_quads:?}"
        );
    }
}
