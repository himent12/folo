//! Verified Dataset Curate & Convert CLI Command Builder (`folo dataset convert`, `split`,
//! `filter`, and `check --perceptual` — every subcommand and flag verified against
//! `crates/folo-cli/src/cli/dataset.rs`), with shell-safe quoting via `folo_api::json::quote_path`.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, SharedString, Styled, Window, div, px};

use crate::api::{Dataset, DatasetFormat, Label, Split};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Active workflow tab of the Dataset Curate / Convert CLI Builder.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DatasetCliMode {
    #[default]
    Convert,
    Split,
    Filter,
    DedupCheck,
}

impl DatasetCliMode {
    pub const ALL: [Self; 4] = [Self::Convert, Self::Split, Self::Filter, Self::DedupCheck];

    pub fn label(self) -> &'static str {
        match self {
            Self::Convert => "Convert",
            Self::Split => "Split",
            Self::Filter => "Filter",
            Self::DedupCheck => "Dedup check",
        }
    }
}

/// Target format matching `TargetFormatArg` in `crates/folo-cli/src/cli/dataset.rs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DatasetTargetFormat {
    #[default]
    Yolo,
    Coco,
    Voc,
}

impl DatasetTargetFormat {
    pub const ALL: [Self; 3] = [Self::Yolo, Self::Coco, Self::Voc];

    pub fn label(self) -> &'static str {
        match self {
            Self::Yolo => "YOLO",
            Self::Coco => "COCO",
            Self::Voc => "VOC",
        }
    }

    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Yolo => "yolo",
            Self::Coco => "coco",
            Self::Voc => "voc",
        }
    }

    fn default_convert_target(fmt: DatasetFormat) -> Self {
        match fmt {
            DatasetFormat::Yolo | DatasetFormat::Auto => Self::Coco,
            DatasetFormat::Coco => Self::Yolo,
        }
    }
}

/// Whether `folo dataset filter` keeps (`--include-class`) or drops (`--exclude-class`) the chosen class.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FilterClassMode {
    #[default]
    Include,
    Exclude,
}

/// Verified presets for `folo dataset split --val` and `--test`.
pub const SPLIT_VAL_PRESETS: [&str; 3] = ["0.1", "0.2", "0.3"];
pub const SPLIT_TEST_PRESETS: [&str; 3] = ["0.0", "0.1", "0.2"];
/// Verified presets for `folo dataset check --dhash-threshold <BITS>` (CLI default 6).
pub const DHASH_PRESETS: [u32; 3] = [4, 6, 10];
/// Minimum normalized box area `w * h` matching `health.rs`'s `TINY_SIZE = 0.05` (`0.05^2 = 0.0025`).
const TINY_BOX_MIN_AREA: &str = "0.0025";

/// Interactive state for the Dataset Curate & Convert CLI Builder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetCliState {
    pub dataset_id: Option<String>,
    pub mode: DatasetCliMode,
    pub convert_to: DatasetTargetFormat,
    pub split_val_idx: usize,
    pub split_test_idx: usize,
    pub split_to: Option<DatasetTargetFormat>,
    pub filter_split: Option<Split>,
    pub filter_drop_empty: bool,
    pub filter_drop_tiny: bool,
    pub filter_class_mode: FilterClassMode,
    pub filter_class: Option<u32>,
    pub filter_to: Option<DatasetTargetFormat>,
    pub check_perceptual: bool,
    pub check_dhash_idx: usize,
    pub check_fix_dry_run: bool,
}

impl Default for DatasetCliState {
    fn default() -> Self {
        Self {
            dataset_id: None,
            mode: DatasetCliMode::Convert,
            convert_to: DatasetTargetFormat::Coco,
            split_val_idx: 1,  // 0.2
            split_test_idx: 0, // 0.0
            split_to: None,
            filter_split: None,
            filter_drop_empty: false,
            filter_drop_tiny: false,
            filter_class_mode: FilterClassMode::Include,
            filter_class: None,
            filter_to: None,
            check_perceptual: true,
            check_dhash_idx: 1, // 6 bits (CLI default)
            check_fix_dry_run: true,
        }
    }
}

impl DatasetCliState {
    /// Resets dataset-scoped fields (such as class filter index and default convert target) when
    /// opening a different dataset.
    pub fn sync_dataset(&mut self, ds: &Dataset) {
        if self.dataset_id.as_deref() == Some(ds.id.as_str()) {
            if self.filter_class.is_some_and(|c| (c as usize) >= ds.classes.len()) {
                self.filter_class = None;
            }
            return;
        }
        self.dataset_id = Some(ds.id.clone());
        self.filter_class = None;
        self.convert_to = DatasetTargetFormat::default_convert_target(ds.format);
    }
}

/// Generated CLI command and shell-quoting / usage notes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltDatasetCli {
    pub command: String,
    pub hint: String,
    pub quote_note: Option<String>,
}

/// Derives a sibling output folder path for `convert` / `split` / `filter` from `ds.path`,
/// stripping trailing path separators and known manifest filenames (`data.yaml`, `*.json`, `*.xml`).
fn derive_output_dir(path: &str, suffix: &str) -> String {
    let trimmed = path.trim().trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        return format!("<output-{suffix}>");
    }
    let lower = trimmed.to_ascii_lowercase();
    let base = if lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".json")
        || lower.ends_with(".xml")
    {
        trimmed
            .rsplit_once(['/', '\\'])
            .map(|(parent, _)| parent.trim_end_matches(['/', '\\']))
            .filter(|p| !p.is_empty())
            .unwrap_or(trimmed)
    } else {
        trimmed
    };
    format!("{base}-{suffix}")
}

fn quote_or_placeholder(raw: &str, placeholder: &str, notes: &mut Vec<String>) -> String {
    if raw.trim().is_empty() {
        return placeholder.to_string();
    }
    match folo_api::json::quote_path(raw) {
        Some(q) => q,
        None => {
            notes.push(format!("Replace {placeholder} with this value, quoted for your shell: {raw}"));
            placeholder.to_string()
        }
    }
}

/// Builds a verified, shell-safe `folo dataset` CLI command matching `crates/folo-cli/src/cli/dataset.rs`.
pub fn build_dataset_cli(ds: &Dataset, state: &DatasetCliState) -> BuiltDatasetCli {
    let mut notes: Vec<String> = Vec::new();
    let input_arg = quote_or_placeholder(&ds.path, "<dataset>", &mut notes);

    match state.mode {
        DatasetCliMode::Convert => {
            let suffix = state.convert_to.cli_value();
            let raw_out = derive_output_dir(&ds.path, suffix);
            let out_arg = if raw_out.starts_with('<') {
                raw_out
            } else {
                quote_or_placeholder(&raw_out, "<output>", &mut notes)
            };
            let cmd =
                format!("folo dataset convert {input_arg} {out_arg} --to {}", state.convert_to.cli_value());
            let hint = format!(
                "Converts {} to {} (images are copied unchanged; output folder must be new or empty).",
                ds.format.label(),
                state.convert_to.label()
            );
            BuiltDatasetCli { command: cmd, hint, quote_note: (!notes.is_empty()).then(|| notes.join(" ")) }
        }
        DatasetCliMode::Split => {
            let val = SPLIT_VAL_PRESETS[state.split_val_idx.min(SPLIT_VAL_PRESETS.len() - 1)];
            let test = SPLIT_TEST_PRESETS[state.split_test_idx.min(SPLIT_TEST_PRESETS.len() - 1)];
            let raw_out = derive_output_dir(&ds.path, "split");
            let out_arg = if raw_out.starts_with('<') {
                raw_out
            } else {
                quote_or_placeholder(&raw_out, "<output>", &mut notes)
            };
            let mut parts = vec![
                "folo dataset split".to_string(),
                input_arg,
                "--out".to_string(),
                out_arg,
                "--val".to_string(),
                val.to_string(),
            ];
            if test != "0.0" {
                parts.push("--test".to_string());
                parts.push(test.to_string());
            }
            if let Some(to) = state.split_to {
                parts.push("--to".to_string());
                parts.push(to.cli_value().to_string());
            }
            let hint = "Deterministic multi-label stratified re-split (preserves rare-class balance across train / val / test).".to_string();
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                quote_note: (!notes.is_empty()).then(|| notes.join(" ")),
            }
        }
        DatasetCliMode::Filter => {
            let raw_out = derive_output_dir(&ds.path, "filtered");
            let out_arg = if raw_out.starts_with('<') {
                raw_out
            } else {
                quote_or_placeholder(&raw_out, "<output>", &mut notes)
            };
            let mut parts = vec!["folo dataset filter".to_string(), input_arg, "--out".to_string(), out_arg];
            if let Some(sp) = state.filter_split {
                let s = match sp {
                    Split::Train => "train",
                    Split::Val => "val",
                    Split::Test => "test",
                };
                parts.push("--split".to_string());
                parts.push(s.to_string());
            }
            if let Some(ci) = state.filter_class
                && let Some(cname) = ds.classes.get(ci as usize)
            {
                let flag = match state.filter_class_mode {
                    FilterClassMode::Include => "--include-class",
                    FilterClassMode::Exclude => "--exclude-class",
                };
                let cls_arg = quote_or_placeholder(cname, "<class>", &mut notes);
                parts.push(flag.to_string());
                parts.push(cls_arg);
            }
            if state.filter_drop_tiny {
                parts.push("--min-box-area".to_string());
                parts.push(TINY_BOX_MIN_AREA.to_string());
            }
            if state.filter_drop_empty {
                parts.push("--drop-empty-images".to_string());
            }
            if let Some(to) = state.filter_to {
                parts.push("--to".to_string());
                parts.push(to.cli_value().to_string());
            }
            let hint = "Writes a curated copy without modifying source files (`--min-box-area 0.0025` drops boxes smaller than 5% of the image side).".to_string();
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                quote_note: (!notes.is_empty()).then(|| notes.join(" ")),
            }
        }
        DatasetCliMode::DedupCheck => {
            let mut parts = vec!["folo dataset check".to_string(), input_arg];
            if state.check_perceptual {
                parts.push("--perceptual".to_string());
                let bits = DHASH_PRESETS[state.check_dhash_idx.min(DHASH_PRESETS.len() - 1)];
                if bits != 6 {
                    parts.push("--dhash-threshold".to_string());
                    parts.push(bits.to_string());
                }
            }
            if state.check_fix_dry_run {
                parts.push("--fix-dry-run".to_string());
            }
            let hint = if state.check_perceptual {
                "Checks exact duplicates, train/val split leakage, and 64-bit perceptual dHash near-duplicates (read-only)."
            } else {
                "Validates labels, corrupt images, exact duplicate files, and train/val split leakage (read-only)."
            }
            .to_string();
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                quote_note: (!notes.is_empty()).then(|| notes.join(" ")),
            }
        }
    }
}

impl FoloApp {
    /// Compact verified Dataset Curate & Convert CLI Builder panel on the Statistics tab.
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
        };

        let copy_cmd = built.command.clone();
        let copy_note = built.quote_note.clone();
        let copy_btn = self.button(
            "ds-cli-copy",
            "Copy command",
            Btn::Secondary,
            Some("icons/copy.svg"),
            window,
            cx,
            act(move |this, _, cx| {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_cmd.clone()));
                this.toast(
                    Tone::Success,
                    "Copied dataset CLI command",
                    copy_note.clone().map(Into::into),
                    cx,
                );
            }),
        );

        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Curate & convert CLI").child(mode_seg))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, mock::MockBackend};

    #[test]
    fn builds_verified_convert_split_filter_and_dedup_check_cli_commands() {
        let api = MockBackend::new(4200, true);
        let mut ds = api.dataset("ds_3b91").unwrap();
        ds.path = "/data/my shapes/data.yaml".into();
        ds.format = DatasetFormat::Yolo;
        ds.classes = vec!["circle".into(), "stop sign".into()];

        let mut st = DatasetCliState::default();
        st.sync_dataset(&ds);
        assert_eq!(st.convert_to, DatasetTargetFormat::Coco);

        // 1. Convert: strips data.yaml to derive sibling output dir and quotes spaces safely.
        let conv = build_dataset_cli(&ds, &st);
        assert_eq!(
            conv.command,
            "folo dataset convert \"/data/my shapes/data.yaml\" \"/data/my shapes-coco\" --to coco"
        );
        assert!(conv.quote_note.is_none());

        // 2. Split: includes --out, --val, optional --test and --to.
        st.mode = DatasetCliMode::Split;
        st.split_val_idx = 1; // 0.2
        st.split_test_idx = 1; // 0.1
        st.split_to = Some(DatasetTargetFormat::Voc);
        let sp = build_dataset_cli(&ds, &st);
        assert_eq!(
            sp.command,
            "folo dataset split \"/data/my shapes/data.yaml\" --out \"/data/my shapes-split\" --val 0.2 --test 0.1 --to voc"
        );

        // 3. Filter: uses exact verified flags --split, --include-class, --min-box-area, --drop-empty-images.
        st.mode = DatasetCliMode::Filter;
        st.filter_split = Some(Split::Train);
        st.filter_class_mode = FilterClassMode::Include;
        st.filter_class = Some(1);
        st.filter_drop_tiny = true;
        st.filter_drop_empty = true;
        let flt = build_dataset_cli(&ds, &st);
        assert_eq!(
            flt.command,
            "folo dataset filter \"/data/my shapes/data.yaml\" --out \"/data/my shapes-filtered\" --split train --include-class \"stop sign\" --min-box-area 0.0025 --drop-empty-images"
        );

        // 4. Dedup check: uses real `folo dataset check --perceptual [--dhash-threshold] [--fix-dry-run]`.
        st.mode = DatasetCliMode::DedupCheck;
        st.check_perceptual = true;
        st.check_dhash_idx = 0; // 4 bits (non-default -> emitted)
        st.check_fix_dry_run = true;
        let chk = build_dataset_cli(&ds, &st);
        assert_eq!(
            chk.command,
            "folo dataset check \"/data/my shapes/data.yaml\" --perceptual --dhash-threshold 4 --fix-dry-run"
        );

        // 5. Shell-unsafe path falls back to <dataset> placeholder with an explicit quote_note.
        ds.path = "/data/bad`$(rm -rf)/data.yaml".into();
        st.mode = DatasetCliMode::Convert;
        let unsafe_cmd = build_dataset_cli(&ds, &st);
        assert!(unsafe_cmd.command.contains("<dataset>"));
        assert!(unsafe_cmd.quote_note.is_some());
    }
}
