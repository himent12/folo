//! Pure command builder for `folo dataset` subcommands (`convert`, `split`, `filter`, `check`,
//! and `export --format coco`), verified against `crates/folo-cli/src/cli/dataset.rs` with
//! shell-safe quoting via `folo_api::json::quote_path`.

use crate::api::{Dataset, DatasetFormat, Label, Split};

/// Active workflow tab of the Dataset Curate & Export CLI Builder.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DatasetCliMode {
    #[default]
    Convert,
    Split,
    Filter,
    DedupCheck,
    ExportCoco,
}

impl DatasetCliMode {
    pub const ALL: [Self; 5] = [Self::Convert, Self::Split, Self::Filter, Self::DedupCheck, Self::ExportCoco];

    pub fn label(self) -> &'static str {
        match self {
            Self::Convert => "Convert",
            Self::Split => "Split",
            Self::Filter => "Filter",
            Self::DedupCheck => "Dedup check",
            Self::ExportCoco => "Export COCO",
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

/// How `folo dataset export --format coco` handles image files, matching `CocoImageModeArg` in
/// `crates/folo-cli/src/cli/dataset.rs`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CocoImageMode {
    #[default]
    Copy,
    Link,
    Reference,
}

impl CocoImageMode {
    pub const ALL: [Self; 3] = [Self::Copy, Self::Link, Self::Reference];

    pub fn label(self) -> &'static str {
        match self {
            Self::Copy => "Copy images",
            Self::Link => "Hardlink",
            Self::Reference => "Reference only",
        }
    }

    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Link => "link",
            Self::Reference => "reference",
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

/// Interactive state for the Dataset Curate & Export CLI Builder.
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
    pub export_coco_split: Option<Split>,
    pub export_coco_images: CocoImageMode,
    pub export_coco_pretty: bool,
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
            export_coco_split: None,
            export_coco_images: CocoImageMode::Copy,
            export_coco_pretty: false,
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
    /// Set when the command cannot be safely copied as-is (e.g. class name contains a comma,
    /// which `folo dataset filter` splits on via `value_delimiter = ','`, or an unquotable path
    /// still contains placeholders).
    pub copy_blocked: Option<&'static str>,
}

/// Derives a sibling output folder path for `convert` / `split` / `filter` / `export` from `ds.path`,
/// stripping trailing path separators, known manifest filenames (`data.yaml`, `*.json`, `*.xml`),
/// and a standard COCO/VOC `annotations` parent directory so the derived output never lands inside
/// the dataset root. Returns `None` when `path` is empty or reduces to a bare filesystem root.
fn derive_output_dir(path: &str, suffix: &str) -> Option<String> {
    let trimmed = path.trim().trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let is_manifest = lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".json")
        || lower.ends_with(".xml");
    let mut base = if is_manifest {
        trimmed
            .rsplit_once(['/', '\\'])
            .map(|(parent, _)| parent.trim_end_matches(['/', '\\']))
            .filter(|p| !p.is_empty())?
    } else {
        trimmed
    };
    if (lower.ends_with(".json") || lower.ends_with(".xml"))
        && let Some((parent, tail)) = base.rsplit_once(['/', '\\'])
        && tail.eq_ignore_ascii_case("annotations")
    {
        let parent = parent.trim_end_matches(['/', '\\']);
        if parent.is_empty() {
            return None;
        }
        base = parent;
    }
    // Reject bare Windows drive prefixes ("C:") so we never emit "C:-yolo".
    if base.len() == 2 && base.as_bytes()[0].is_ascii_alphabetic() && base.as_bytes()[1] == b':' {
        return None;
    }
    Some(format!("{base}-{suffix}"))
}

fn quote_or_placeholder(raw: &str, placeholder: &str, notes: &mut Vec<String>) -> (String, bool) {
    if raw.trim().is_empty() {
        notes.push(format!("Replace {placeholder} with the dataset path."));
        return (placeholder.to_string(), true);
    }
    match folo_api::json::quote_path(raw) {
        Some(q) => (q, false),
        None => {
            notes.push(format!("Replace {placeholder} with this value, quoted for your shell: {raw}"));
            (placeholder.to_string(), true)
        }
    }
}

fn resolve_output_arg(path: &str, suffix: &str, notes: &mut Vec<String>) -> (String, bool) {
    match derive_output_dir(path, suffix) {
        Some(raw_out) => quote_or_placeholder(&raw_out, "<output>", notes),
        None => {
            notes.push(format!("Replace <output-{suffix}> with a new or empty output folder path."));
            (format!("<output-{suffix}>"), true)
        }
    }
}

const BLOCKED_PLACEHOLDER_REASON: &str =
    "Replace placeholder arguments with shell-quoted paths before running";
const BLOCKED_COMMA_CLASS_REASON: &str =
    "Selected class contains a comma, which `folo dataset filter` splits into multiple names";

fn cli_source_format(fmt: DatasetFormat) -> Option<&'static str> {
    match fmt {
        DatasetFormat::Auto => None,
        DatasetFormat::Yolo => Some("yolo"),
        DatasetFormat::Coco => Some("coco"),
    }
}

fn split_cli_value(sp: Split) -> &'static str {
    match sp {
        Split::Train => "train",
        Split::Val => "val",
        Split::Test => "test",
    }
}

/// Builds a verified, shell-safe `folo dataset` CLI command matching `crates/folo-cli/src/cli/dataset.rs`.
pub fn build_dataset_cli(ds: &Dataset, state: &DatasetCliState) -> BuiltDatasetCli {
    let mut notes: Vec<String> = Vec::new();
    let (input_arg, mut needs_replace) = quote_or_placeholder(&ds.path, "<dataset>", &mut notes);
    let src_fmt = cli_source_format(ds.format);

    match state.mode {
        DatasetCliMode::Convert => {
            let suffix = state.convert_to.cli_value();
            let (out_arg, out_placeholder) = resolve_output_arg(&ds.path, suffix, &mut notes);
            needs_replace |= out_placeholder;
            let mut parts = vec!["folo dataset convert".to_string(), input_arg, out_arg];
            if let Some(from) = src_fmt {
                parts.push("--from".to_string());
                parts.push(from.to_string());
            }
            parts.push("--to".to_string());
            parts.push(state.convert_to.cli_value().to_string());
            let hint = format!(
                "Converts {} to {} (images are copied unchanged; output folder must be new or empty).",
                ds.format.label(),
                state.convert_to.label()
            );
            let quote_note = (!notes.is_empty()).then(|| notes.join(" "));
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                copy_blocked: needs_replace.then_some(BLOCKED_PLACEHOLDER_REASON),
                quote_note,
            }
        }
        DatasetCliMode::Split => {
            let val = SPLIT_VAL_PRESETS[state.split_val_idx.min(SPLIT_VAL_PRESETS.len() - 1)];
            let test = SPLIT_TEST_PRESETS[state.split_test_idx.min(SPLIT_TEST_PRESETS.len() - 1)];
            let (out_arg, out_placeholder) = resolve_output_arg(&ds.path, "split", &mut notes);
            needs_replace |= out_placeholder;
            let mut parts = vec!["folo dataset split".to_string(), input_arg, "--out".to_string(), out_arg];
            if let Some(from) = src_fmt {
                parts.push("--from".to_string());
                parts.push(from.to_string());
            }
            parts.push("--val".to_string());
            parts.push(val.to_string());
            if test != "0.0" {
                parts.push("--test".to_string());
                parts.push(test.to_string());
            }
            if let Some(to) = state.split_to {
                parts.push("--to".to_string());
                parts.push(to.cli_value().to_string());
            }
            let hint = "Deterministic multi-label stratified re-split (preserves rare-class balance across train / val / test).".to_string();
            let quote_note = (!notes.is_empty()).then(|| notes.join(" "));
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                copy_blocked: needs_replace.then_some(BLOCKED_PLACEHOLDER_REASON),
                quote_note,
            }
        }
        DatasetCliMode::Filter => {
            let (out_arg, out_placeholder) = resolve_output_arg(&ds.path, "filtered", &mut notes);
            needs_replace |= out_placeholder;
            let mut comma_block: Option<String> = None;
            let mut parts = vec!["folo dataset filter".to_string(), input_arg, "--out".to_string(), out_arg];
            if let Some(from) = src_fmt {
                parts.push("--from".to_string());
                parts.push(from.to_string());
            }
            if let Some(sp) = state.filter_split {
                parts.push("--split".to_string());
                parts.push(split_cli_value(sp).to_string());
            }
            if let Some(ci) = state.filter_class
                && let Some(cname) = ds.classes.get(ci as usize)
            {
                let flag = match state.filter_class_mode {
                    FilterClassMode::Include => "--include-class",
                    FilterClassMode::Exclude => "--exclude-class",
                };
                if cname.contains(',') {
                    comma_block = Some(format!(
                        "Class `{cname}` contains a comma, which `folo dataset filter {flag}` splits as multiple names (`value_delimiter = ','`). Pick a different class."
                    ));
                    parts.push(flag.to_string());
                    parts.push("<class>".to_string());
                } else {
                    let (cls_arg, cls_placeholder) = quote_or_placeholder(cname, "<class>", &mut notes);
                    needs_replace |= cls_placeholder;
                    parts.push(flag.to_string());
                    parts.push(cls_arg);
                }
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
            let quote_note = (!notes.is_empty()).then(|| notes.join(" "));
            let copy_blocked = comma_block
                .as_ref()
                .map(|_| BLOCKED_COMMA_CLASS_REASON)
                .or_else(|| needs_replace.then_some(BLOCKED_PLACEHOLDER_REASON));
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                quote_note: match (comma_block, quote_note) {
                    (Some(c), Some(q)) => Some(format!("{c} {q}")),
                    (Some(c), None) => Some(c),
                    (None, q) => q,
                },
                copy_blocked,
            }
        }
        DatasetCliMode::DedupCheck => {
            let mut parts = vec!["folo dataset check".to_string(), input_arg];
            if let Some(fmt) = src_fmt {
                parts.push("--format".to_string());
                parts.push(fmt.to_string());
            }
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
            let quote_note = (!notes.is_empty()).then(|| notes.join(" "));
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                copy_blocked: needs_replace.then_some(BLOCKED_PLACEHOLDER_REASON),
                quote_note,
            }
        }
        DatasetCliMode::ExportCoco => {
            let (out_arg, out_placeholder) = resolve_output_arg(&ds.path, "coco-export", &mut notes);
            needs_replace |= out_placeholder;
            let mut parts = vec![
                "folo dataset export".to_string(),
                input_arg,
                "--out".to_string(),
                out_arg,
                "--format".to_string(),
                "coco".to_string(),
            ];
            if let Some(from) = src_fmt {
                parts.push("--from".to_string());
                parts.push(from.to_string());
            }
            if let Some(sp) = state.export_coco_split {
                parts.push("--split".to_string());
                parts.push(split_cli_value(sp).to_string());
            }
            parts.push("--images".to_string());
            parts.push(state.export_coco_images.cli_value().to_string());
            if state.export_coco_pretty {
                parts.push("--pretty".to_string());
            }
            let hint = match state.export_coco_images {
                CocoImageMode::Copy => {
                    "Exports COCO `annotations/instances_<split>.json` and copies images into the output folder (output must be new or empty)."
                }
                CocoImageMode::Link => {
                    "Exports COCO `annotations/instances_<split>.json` and hardlinks images (falls back to copy across filesystems)."
                }
                CocoImageMode::Reference => {
                    "Writes COCO `annotations/instances_<split>.json` referencing the original image paths without copying files."
                }
            }
            .to_string();
            let quote_note = (!notes.is_empty()).then(|| notes.join(" "));
            BuiltDatasetCli {
                command: parts.join(" "),
                hint,
                copy_blocked: needs_replace.then_some(BLOCKED_PLACEHOLDER_REASON),
                quote_note,
            }
        }
    }
}
