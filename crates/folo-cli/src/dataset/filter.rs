//! `folo dataset filter`: subset filtering by class whitelist/blacklist, split selection, normalised box area,
//! box/image pixel dimensions, and empty-image dropping on top of `folo_data::curate::write_filtered`.

use std::collections::HashSet;
use std::time::Instant;

use folo_data::{ClassNormalize, DataError, FilterConfig, LoadedDataset, Split};
use serde_json::json;

use crate::cli::{FilterArgs, bounds};
use crate::output::{CliError, ExitCode, Out};

use super::{
    check_convert_output_outside, ensure_empty_output, format_name, format_of, open, output_data_error, rel,
    splits_json, target_format_of, unreadable_split_manifests,
};

fn invalid_filter(message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "invalid_config", message)
        .hint("see `folo describe dataset filter --json` for the accepted flags and ranges")
}

/// Validates filter bounds upfront and constructs [`FilterConfig`] before opening the dataset.
fn build_filter_config(args: &FilterArgs) -> Result<FilterConfig, CliError> {
    for (flag, val) in [("--min-box-area", args.min_box_area), ("--max-box-area", args.max_box_area)] {
        if let Some(v) = val
            && !bounds::UNIT_INTERVAL.contains_f64(f64::from(v))
        {
            return Err(invalid_filter(format!("{flag} must be in 0..=1, got {v}")));
        }
    }
    for (flag, val) in [
        ("--min-box-width", args.min_box_width),
        ("--max-box-width", args.max_box_width),
        ("--min-box-height", args.min_box_height),
        ("--max-box-height", args.max_box_height),
    ] {
        if let Some(v) = val
            && !bounds::NON_NEGATIVE_FLOAT.contains_f64(f64::from(v))
        {
            return Err(invalid_filter(format!("{flag} must be finite and >= 0, got {v}")));
        }
    }
    for (flag, val) in [
        ("--min-image-width", args.min_image_width),
        ("--max-image-width", args.max_image_width),
        ("--min-image-height", args.min_image_height),
        ("--max-image-height", args.max_image_height),
    ] {
        if let Some(v) = val
            && !bounds::POSITIVE_INT.contains_u64(u64::from(v))
        {
            return Err(invalid_filter(format!("{flag} must be at least 1, got {v}")));
        }
    }
    let include_classes = if args.include_class.is_empty() { None } else { Some(args.include_class.clone()) };
    let splits = if args.split.is_empty() {
        None
    } else {
        Some(args.split.iter().copied().map(Split::from).collect())
    };
    let cfg = FilterConfig {
        include_classes,
        exclude_classes: args.exclude_class.clone(),
        normalize: args.normalize.into(),
        compact_classes: !args.keep_all_classes,
        splits,
        min_box_area: args.min_box_area,
        max_box_area: args.max_box_area,
        min_box_width_px: args.min_box_width,
        min_box_height_px: args.min_box_height,
        max_box_width_px: args.max_box_width,
        max_box_height_px: args.max_box_height,
        min_image_width: args.min_image_width,
        min_image_height: args.min_image_height,
        max_image_width: args.max_image_width,
        max_image_height: args.max_image_height,
        probe_image_sizes: false,
        drop_empty_images: args.drop_empty_images,
    };
    cfg.validate().map_err(|e| invalid_filter(e.to_string()))?;
    Ok(cfg)
}

/// Class names requested in `--include-class` or `--exclude-class` that do not match any class in `ds_classes`
/// under `normalize` (deduplicated, preserving input order).
fn find_unmatched_classes(ds_classes: &[String], cfg: &FilterConfig) -> Vec<String> {
    let norm = |s: &str| match cfg.normalize {
        ClassNormalize::Exact => s.to_string(),
        ClassNormalize::CaseAndWhitespace => {
            s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
        }
    };
    let known: HashSet<String> = ds_classes.iter().map(|c| norm(c)).collect();
    let mut seen = HashSet::new();
    let mut unmatched = Vec::new();
    for name in cfg.include_classes.iter().flatten().chain(&cfg.exclude_classes) {
        let key = norm(name);
        if !known.contains(&key) && seen.insert(key) {
            unmatched.push(name.clone());
        }
    }
    unmatched
}

fn total_images(ds: &LoadedDataset) -> usize {
    Split::ALL.iter().map(|&s| ds.split(s).entries().len()).sum()
}

pub(super) fn filter(args: FilterArgs, out: Out) -> Result<(), CliError> {
    let cfg = build_filter_config(&args)?;
    let ds = open(&args.path, format_of(args.from))?;
    let bad_splits = unreadable_split_manifests(&ds, &args.path);
    if !bad_splits.is_empty() {
        let (first_path, first_msg) = &bad_splits[0];
        let mut msg = format!("{}: {first_msg}", rel(first_path, &ds.root));
        if bad_splits.len() > 1 {
            msg.push_str(&format!(" (+{} more unreadable split file(s))", bad_splits.len() - 1));
        }
        return Err(CliError::new(ExitCode::Runtime, "invalid_dataset", msg).hint(
            folo_api::json::cli_command(
                "fix the corrupt annotation or split file; run `folo dataset check {dataset}` to inspect all issues",
                &[("dataset", &args.path)],
            ),
        ));
    }
    let to = args.to.map_or(ds.format, target_format_of);
    check_convert_output_outside(&args.out, &args.path, &ds)?;
    ensure_empty_output(&args.out)?;
    let skipped = ds.report.skipped_images() + ds.report.skipped_label_images();
    let src_images = total_images(&ds);
    let unmatched = find_unmatched_classes(&ds.classes, &cfg);
    let t0 = Instant::now();
    let filtered = folo_data::filter_dataset(&ds, &cfg).map_err(|e| match e {
        DataError::Config(m) => invalid_filter(m),
        e => output_data_error(e, &args.out),
    })?;
    if total_images(&filtered) == 0 {
        let mut hint = "relax the split, class, box, or image dimension filters".to_string();
        if !unmatched.is_empty() {
            hint.push_str(&format!(" (unmatched class filter(s): {})", unmatched.join(", ")));
        }
        return Err(CliError::new(
            ExitCode::Runtime,
            "invalid_dataset",
            format!(
                "no images in {} matched the filter (0 of {src_images} kept)",
                folo_api::json::path(&args.path)
            ),
        )
        .hint(hint));
    }
    let out_classes = filtered.classes.clone();
    let done = folo_data::convert::convert(&filtered, &args.out, to).map_err(|e| match e {
        DataError::Config(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        e => output_data_error(e, &args.out),
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let mut rec = json!({
        "input": folo_api::json::path(&args.path),
        "output": folo_api::json::path(&done.root),
        "from": format_name(ds.format),
        "to": format_name(to),
        "classes": out_classes,
        "splits": splits_json(done.splits),
        "boxes": done.boxes,
        "skipped_images": skipped,
        "seconds": secs,
    });
    if !unmatched.is_empty() {
        rec["unmatched_classes"] = json!(unmatched);
    }
    out.record(&rec);
    let [train, val, test] = done.splits;
    out.human(format!(
        "filtered {} {} → {} at {} ({train} train / {val} val / {test} test images, {} classes, {} boxes) in \
         {secs:.1}s",
        format_name(ds.format),
        folo_api::json::path(&args.path),
        format_name(to),
        folo_api::json::path(&done.root),
        out_classes.len(),
        done.boxes,
    ));
    if !unmatched.is_empty() {
        out.human(format!(
            "  warning: unmatched class filter(s) not found in dataset: {} (available: {})",
            unmatched.join(", "),
            ds.classes.join(", "),
        ));
    }
    if skipped > 0 {
        out.human(format!(
            "  {skipped} image(s) were skipped (unreadable or broken labels); {}",
            folo_api::json::cli_command("run `folo dataset check {dataset}`", &[("dataset", &args.path)])
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn build_filter_config_validates_bounds_upfront() {
        let parse = |argv: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Command::Dataset(crate::cli::DatasetCommand::Filter(f)) = cli.command else {
                panic!()
            };
            f
        };
        let ok = build_filter_config(&parse(&[
            "folo",
            "dataset",
            "filter",
            "ds",
            "--out",
            "out",
            "--include-class",
            "circle,square",
            "--split",
            "train,val",
            "--min-box-area",
            "0.01",
            "--max-box-area",
            "0.5",
            "--drop-empty-images",
        ]))
        .unwrap();
        assert_eq!(ok.include_classes, Some(vec!["circle".into(), "square".into()]));
        assert_eq!(ok.splits, Some(vec![Split::Train, Split::Val]));
        assert!(ok.drop_empty_images);

        for argv in [
            &["folo", "dataset", "filter", "ds", "--out", "out", "--min-box-area", "1.5"][..],
            &[
                "folo",
                "dataset",
                "filter",
                "ds",
                "--out",
                "out",
                "--min-box-area",
                "0.6",
                "--max-box-area",
                "0.2",
            ][..],
            &["folo", "dataset", "filter", "ds", "--out", "out", "--min-image-width", "0"][..],
            &[
                "folo",
                "dataset",
                "filter",
                "ds",
                "--out",
                "out",
                "--min-image-width",
                "640",
                "--max-image-width",
                "320",
            ][..],
        ] {
            let err = build_filter_config(&parse(argv)).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "invalid_config"), "{argv:?}");
        }
    }
}
