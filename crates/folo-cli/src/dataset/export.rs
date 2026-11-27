//! `folo dataset export --format coco`: spec-compliant COCO instances JSON export with selectable splits and
//! `copy` / `link` / `reference` image handling on top of `folo_data::convert::export_coco`.

use std::time::Instant;

use folo_data::convert::{CocoExportOptions, CocoImageMode, export_coco};
use folo_data::{DataError, Split};
use serde_json::json;

use crate::cli::{CocoImageModeArg, ExportDatasetArgs, ExportDatasetFormatArg};
use crate::output::{CliError, ExitCode, Out};

use super::{
    check_convert_output_outside, ensure_empty_output, format_name, format_of, open, output_data_error, rel,
    splits_json, unreadable_split_manifests,
};

fn image_mode_name(mode: CocoImageMode) -> &'static str {
    match mode {
        CocoImageMode::Copy => "copy",
        CocoImageMode::Link => "link",
        CocoImageMode::Reference => "reference",
    }
}

pub(super) fn export(args: ExportDatasetArgs, out: Out) -> Result<(), CliError> {
    let ExportDatasetFormatArg::Coco = args.format;
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
    check_convert_output_outside(&args.out, &args.path, &ds)?;
    ensure_empty_output(&args.out)?;
    let skipped = ds.report.skipped_images() + ds.report.skipped_label_images();
    let image_mode: CocoImageMode = args.images.into();
    let opts = CocoExportOptions {
        splits: args.split.iter().copied().map(Split::from).collect(),
        images: image_mode,
        pretty: args.pretty,
        description: args.description.clone(),
    };
    let t0 = Instant::now();
    let report = export_coco(&ds, &args.out, &opts).map_err(|e| match e {
        DataError::Config(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        e => output_data_error(e, &args.out),
    })?;
    let seconds = t0.elapsed().as_secs_f64();
    let exported_splits: Vec<serde_json::Value> = report
        .exported_splits
        .iter()
        .map(|s| {
            json!({
                "split": s.split.as_str(),
                "annotation_file": folo_api::json::path(&s.annotation_file),
                "images": s.images,
                "annotations": s.annotations,
                "linked_images": s.linked_images,
                "copied_images": s.copied_images,
                "referenced_images": s.referenced_images,
            })
        })
        .collect();
    out.record(&json!({
        "input": folo_api::json::path(&args.path),
        "output": folo_api::json::path(&report.root),
        "from": format_name(ds.format),
        "format": "coco",
        "images": image_mode_name(report.image_mode),
        "pretty": args.pretty,
        "classes": ds.classes,
        "categories": report.categories,
        "splits": splits_json(report.splits),
        "boxes": report.boxes,
        "skipped_images": skipped,
        "exported_splits": exported_splits,
        "seconds": seconds,
    }));
    let total_images: usize = report.splits.iter().sum();
    let mode_label = match args.images {
        CocoImageModeArg::Copy => "copy",
        CocoImageModeArg::Link => "link",
        CocoImageModeArg::Reference => "reference",
    };
    let mut msg = format!(
        "Exported {} ({}) -> {} as COCO ({mode_label}): {total_images} images (train {}, val {}, test {}), {} \
         boxes, {} categories in {seconds:.2}s",
        folo_api::json::path(&args.path),
        format_name(ds.format),
        folo_api::json::path(&report.root),
        report.splits[0],
        report.splits[1],
        report.splits[2],
        report.boxes,
        report.categories,
    );
    if skipped > 0 {
        msg.push_str(&format!(" ({skipped} skipped)"));
    }
    for s in &report.exported_splits {
        msg.push_str(&format!(
            "\n  {}: {} ({} images, {} annotations)",
            s.split.as_str(),
            folo_api::json::path(&s.annotation_file),
            s.images,
            s.annotations
        ));
    }
    out.human(msg);
    Ok(())
}
