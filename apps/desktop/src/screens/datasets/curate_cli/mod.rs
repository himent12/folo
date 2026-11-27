//! Verified Dataset Curate & Export CLI Command Builder panel (`folo dataset convert`, `split`,
//! `filter`, `check --perceptual`, and `export --format coco`), verified against
//! `crates/folo-cli/src/cli/dataset.rs`.

pub(crate) mod command;
#[cfg(test)]
mod tests;

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Dataset, Split};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

pub use command::{
    CocoImageMode, DHASH_PRESETS, DatasetCliMode, DatasetCliState, DatasetTargetFormat, FilterClassMode,
    SPLIT_TEST_PRESETS, SPLIT_VAL_PRESETS, build_dataset_cli, split_image_count,
};

impl FoloApp {
    /// Compact verified Dataset Curate & Export CLI Builder panel on the Statistics tab.
    pub(super) fn dataset_curate_cli_panel(
        &mut self,
        ds: &Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        self.ds_cli.sync_dataset(ds);
        let t = theme(cx);
        let built = build_dataset_cli(ds, &self.ds_cli);

        let mode_labels: Vec<&str> = DatasetCliMode::ALL.iter().map(|m| m.label()).collect();
        let mode_idx = DatasetCliMode::ALL.iter().position(|m| *m == self.ds_cli.mode).unwrap_or(0);
        let mode_seg = self.segmented(
            "ds-cli-mode",
            &mode_labels,
            mode_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.ds_cli.mode = DatasetCliMode::ALL[i.min(DatasetCliMode::ALL.len() - 1)];
                cx.notify();
            }),
        );

        let controls: AnyElement = match self.ds_cli.mode {
            DatasetCliMode::Convert => {
                let chips = DatasetTargetFormat::ALL.iter().map(|&fmt| {
                    let active = self.ds_cli.convert_to == fmt;
                    self.chip(
                        &format!("ds-cli-conv-{}", fmt.cli_value()),
                        fmt.label(),
                        active,
                        None,
                        None,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.ds_cli.convert_to = fmt;
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                });
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_tertiary).child("Target format"))
                    .children(chips)
                    .into_any_element()
            }
            DatasetCliMode::Split => {
                let val_chips: Vec<AnyElement> = SPLIT_VAL_PRESETS
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let active = self.ds_cli.split_val_idx == i;
                        let label = match *p {
                            "0.1" => "10%",
                            "0.2" => "20%",
                            _ => "30%",
                        };
                        self.chip(
                            &format!("ds-cli-split-val-{i}"),
                            label,
                            active,
                            None,
                            None,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.ds_cli.split_val_idx = i;
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();
                let test_chips: Vec<AnyElement> = SPLIT_TEST_PRESETS
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let active = self.ds_cli.split_test_idx == i;
                        let label = match *p {
                            "0.0" => "None",
                            "0.1" => "10%",
                            _ => "20%",
                        };
                        self.chip(
                            &format!("ds-cli-split-test-{i}"),
                            label,
                            active,
                            None,
                            None,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.ds_cli.split_test_idx = i;
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();
                let to_chips: Vec<AnyElement> = [
                    None,
                    Some(DatasetTargetFormat::Yolo),
                    Some(DatasetTargetFormat::Coco),
                    Some(DatasetTargetFormat::Voc),
                ]
                .into_iter()
                .enumerate()
                .map(|(i, fmt)| {
                    let active = self.ds_cli.split_to == fmt;
                    let label = fmt.map_or("Same", DatasetTargetFormat::label);
                    self.chip(
                        &format!("ds-cli-split-to-{i}"),
                        label,
                        active,
                        None,
                        None,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.ds_cli.split_to = fmt;
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                })
                .collect();
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(12.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_tertiary).child("Val"))
                            .children(val_chips),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_tertiary).child("Test"))
                            .children(test_chips),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_tertiary).child("Format"))
                            .children(to_chips),
                    )
                    .into_any_element()
            }
            DatasetCliMode::Filter => {
                let split_chips: Vec<AnyElement> = [
                    (None, "All splits"),
                    (Some(Split::Train), "train"),
                    (Some(Split::Val), "val"),
                    (Some(Split::Test), "test"),
                ]
                .into_iter()
                .enumerate()
                .map(|(i, (sp, label))| {
                    let active = self.ds_cli.filter_split == sp;
                    self.chip(
                        &format!("ds-cli-flt-split-{i}"),
                        label,
                        active,
                        None,
                        None,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.ds_cli.filter_split = sp;
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                })
                .collect();
                let drop_empty = self.chip(
                    "ds-cli-flt-empty",
                    "Drop empty images",
                    self.ds_cli.filter_drop_empty,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_cli.filter_drop_empty = !this.ds_cli.filter_drop_empty;
                        cx.notify();
                    }),
                );
                let drop_tiny = self.chip(
                    "ds-cli-flt-tiny",
                    "Drop tiny boxes (<5%)",
                    self.ds_cli.filter_drop_tiny,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_cli.filter_drop_tiny = !this.ds_cli.filter_drop_tiny;
                        cx.notify();
                    }),
                );
                let class_row = (!ds.classes.is_empty()).then(|| {
                    let mode_idx = usize::from(self.ds_cli.filter_class_mode == FilterClassMode::Exclude);
                    let cls_mode = self.segmented(
                        "ds-cli-flt-clsmode",
                        &["Include", "Exclude"],
                        mode_idx,
                        window,
                        cx,
                        Rc::new(|this, i, _, cx| {
                            this.ds_cli.filter_class_mode =
                                if i == 1 { FilterClassMode::Exclude } else { FilterClassMode::Include };
                            cx.notify();
                        }),
                    );
                    let mut opts: Vec<(SharedString, Option<SharedString>)> =
                        vec![("Any class".into(), None)];
                    opts.extend(ds.classes.iter().map(|c| (SharedString::from(c.clone()), None)));
                    let sel = self.ds_cli.filter_class.map_or(0, |c| (c as usize) + 1).min(opts.len() - 1);
                    let cur_label = opts[sel].0.clone();
                    let cls_sel = self.select(
                        "ds-cli-flt-class",
                        cur_label,
                        opts,
                        sel,
                        window,
                        cx,
                        Rc::new(|this, i, _, cx| {
                            this.ds_cli.filter_class = if i == 0 { None } else { Some((i - 1) as u32) };
                            cx.notify();
                        }),
                    );
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(8.))
                        .child(text(ty::SM, t.text_tertiary).child("Class"))
                        .child(cls_mode)
                        .child(div().w(px(180.)).child(cls_sel))
                });
                let to_chips: Vec<AnyElement> = [
                    None,
                    Some(DatasetTargetFormat::Yolo),
                    Some(DatasetTargetFormat::Coco),
                    Some(DatasetTargetFormat::Voc),
                ]
                .into_iter()
                .enumerate()
                .map(|(i, fmt)| {
                    let active = self.ds_cli.filter_to == fmt;
                    let label = fmt.map_or("Same", DatasetTargetFormat::label);
                    self.chip(
                        &format!("ds-cli-flt-to-{i}"),
                        label,
                        active,
                        None,
                        None,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.ds_cli.filter_to = fmt;
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                })
                .collect();
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .children(split_chips)
                            .child(drop_empty)
                            .child(drop_tiny),
                    )
                    .children(class_row)
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_tertiary).child("Format"))
                            .children(to_chips),
                    )
                    .into_any_element()
            }
            DatasetCliMode::DedupCheck => {
                let perceptual = self.chip(
                    "ds-cli-chk-perceptual",
                    "Perceptual dHash",
                    self.ds_cli.check_perceptual,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_cli.check_perceptual = !this.ds_cli.check_perceptual;
                        cx.notify();
                    }),
                );
                let dry_run = self.chip(
                    "ds-cli-chk-dryrun",
                    "Fix dry-run",
                    self.ds_cli.check_fix_dry_run,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_cli.check_fix_dry_run = !this.ds_cli.check_fix_dry_run;
                        cx.notify();
                    }),
                );
                let dhash_group = self.ds_cli.check_perceptual.then(|| {
                    let chips = DHASH_PRESETS.iter().enumerate().map(|(i, &bits)| {
                        let active = self.ds_cli.check_dhash_idx == i;
                        self.chip(
                            &format!("ds-cli-chk-bits-{bits}"),
                            format!("{bits} bits"),
                            active,
                            None,
                            None,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.ds_cli.check_dhash_idx = i;
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    });
                    div()
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(text(ty::SM, t.text_tertiary).child("dHash max distance"))
                        .children(chips)
                });
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(8.))
                    .child(perceptual)
                    .children(dhash_group)
                    .child(dry_run)
                    .into_any_element()
            }
            DatasetCliMode::ExportCoco => {
                let split_chips: Vec<AnyElement> = [
                    (None, "All splits"),
                    (Some(Split::Train), "train"),
                    (Some(Split::Val), "val"),
                    (Some(Split::Test), "test"),
                ]
                .into_iter()
                .enumerate()
                .map(|(i, (sp, label))| {
                    let empty_split = sp.is_some_and(|s| split_image_count(ds, s) == 0);
                    if empty_split {
                        return div()
                            .id(SharedString::from(format!("ds-cli-coco-split-{i}-disabled")))
                            .h(px(26.))
                            .px(px(10.))
                            .flex()
                            .items_center()
                            .rounded(px(radius::SM))
                            .border_1()
                            .border_color(t.line_subtle)
                            .opacity(0.45)
                            .tooltip(crate::controls::tip(format!("No {label} images in this dataset"), None))
                            .child(text(ty::SM, t.text_tertiary).child(label))
                            .into_any_element();
                    }
                    let active = self.ds_cli.export_coco_split == sp;
                    self.chip(
                        &format!("ds-cli-coco-split-{i}"),
                        label,
                        active,
                        None,
                        None,
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.ds_cli.export_coco_split = sp;
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                })
                .collect();
                let img_chips: Vec<AnyElement> = CocoImageMode::ALL
                    .iter()
                    .map(|&mode| {
                        let active = self.ds_cli.export_coco_images == mode;
                        self.chip(
                            &format!("ds-cli-coco-img-{}", mode.cli_value()),
                            mode.label(),
                            active,
                            None,
                            None,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.ds_cli.export_coco_images = mode;
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();
                let pretty_chip = self.chip(
                    "ds-cli-coco-pretty",
                    "Pretty JSON",
                    self.ds_cli.export_coco_pretty,
                    None,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.ds_cli.export_coco_pretty = !this.ds_cli.export_coco_pretty;
                        cx.notify();
                    }),
                );
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_tertiary).child("Split"))
                            .children(split_chips),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .child(text(ty::SM, t.text_tertiary).child("Images"))
                            .children(img_chips)
                            .child(pretty_chip),
                    )
                    .into_any_element()
            }
        };

        let copy_disabled = built.copy_blocked.is_some();
        if copy_disabled && self.fh("ds-cli-copy", cx).is_focused(window) {
            window.focus(&self.fh("ds-cli-copy-disabled", cx));
        } else if !copy_disabled && self.fh("ds-cli-copy-disabled", cx).is_focused(window) {
            window.focus(&self.fh("ds-cli-copy", cx));
        }
        let copy_btn = if let Some(reason) = built.copy_blocked {
            self.button_disabled(
                "ds-cli-copy",
                "Copy command",
                Btn::Secondary,
                Some("icons/copy.svg"),
                reason,
                window,
                cx,
            )
            .into_any_element()
        } else {
            let copy_cmd = built.command.clone();
            self.button(
                "ds-cli-copy",
                "Copy command",
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_cmd.clone()));
                    this.toast(Tone::Success, "Copied dataset CLI command", None, cx);
                }),
            )
            .into_any_element()
        };

        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Curate & export CLI").child(mode_seg))
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .child(controls)
                    .child(
                        div()
                            .px(px(12.))
                            .py(px(10.))
                            .rounded(px(radius::MD))
                            .bg(t.bg_inset)
                            .border_1()
                            .border_color(t.line_subtle)
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_family(crate::theme::MONO)
                                    .whitespace_normal()
                                    .child(built.command),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap(px(10.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(200.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(2.))
                                    .child(
                                        text(ty::SM, t.text_tertiary).whitespace_normal().child(built.hint),
                                    )
                                    .when_some(built.quote_note, |d, note| {
                                        d.child(text(ty::SM, t.warning_text).whitespace_normal().child(note))
                                    }),
                            )
                            .child(div().flex_none().child(copy_btn)),
                    ),
            )
            .into_any_element()
    }
}
