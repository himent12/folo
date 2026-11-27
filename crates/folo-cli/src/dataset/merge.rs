//! `folo dataset merge`: combine one or more datasets (YOLO, COCO, Pascal VOC) into one unified YOLO dataset
//! on top of `folo_data::curate::write_merged`.

use std::collections::HashMap;
use std::time::Instant;

use folo_data::{DataError, Format, LoadedDataset, MergeConfig, MergeSplitPolicy};
use serde_json::json;

use crate::cli::{MergeArgs, MergeSplitArg};
use crate::output::{CliError, ExitCode, Out};

use super::{
    check_convert_output_outside, ensure_empty_output, open, output_data_error, rel, split::split_config,
    splits_json, unreadable_split_manifests,
};

fn invalid_merge(message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "invalid_config", message)
        .hint("see `folo describe dataset merge --json` for the accepted flags and ranges")
}

fn normalize_key(name: &str, mode: folo_data::ClassNormalize) -> String {
    match mode {
        folo_data::ClassNormalize::Exact => name.to_string(),
        folo_data::ClassNormalize::CaseAndWhitespace => {
            name.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
        }
    }
}

/// Parses `--remap SRC=DST` and `--drop-class NAME` flags into `MergeConfig::class_map`, rejecting malformed
/// pairs and conflicting mappings (including conflicts under `normalize`) upfront.
fn parse_class_map(
    remap: &[String],
    drop_class: &[String],
    normalize: folo_data::ClassNormalize,
) -> Result<HashMap<String, Option<String>>, CliError> {
    let mut map: HashMap<String, Option<String>> = HashMap::new();
    let mut seen: HashMap<String, Option<String>> = HashMap::new();
    for spec in remap {
        let Some((src, dst)) = spec.split_once('=') else {
            return Err(invalid_merge(format!(
                "--remap `{spec}` must have the form `SOURCE=TARGET` (use --drop-class NAME to drop a class)"
            )));
        };
        if src.trim().is_empty() || dst.trim().is_empty() {
            return Err(invalid_merge(format!(
                "--remap `{spec}` must have non-empty SOURCE and TARGET class names"
            )));
        }
        let norm_src = normalize_key(src, normalize);
        let norm_dst = Some(normalize_key(dst, normalize));
        if let Some(prev) = seen.insert(norm_src, norm_dst.clone())
            && prev != norm_dst
        {
            return Err(invalid_merge(format!("--remap specifies conflicting targets for class `{src}`")));
        }
        map.insert(src.to_string(), Some(dst.to_string()));
    }
    for name in drop_class {
        if name.trim().is_empty() {
            return Err(invalid_merge("--drop-class name must not be empty"));
        }
        let norm_src = normalize_key(name, normalize);
        if let Some(Some(dst)) = seen.insert(norm_src, None) {
            return Err(invalid_merge(format!(
                "class `{name}` cannot be both remapped to `{dst}` and dropped with --drop-class"
            )));
        }
        map.insert(name.clone(), None);
    }
    Ok(map)
}

/// Validates merge arguments and constructs [`MergeConfig`] before opening any input dataset.
fn build_merge_config(args: &MergeArgs) -> Result<MergeConfig, CliError> {
    if args.inputs.is_empty() {
        return Err(invalid_merge("merge requires at least one input dataset"));
    }
    if !args.prefix.is_empty() && args.prefix.len() != args.inputs.len() {
        return Err(invalid_merge(format!(
            "--prefix count ({}) must match the number of input datasets ({})",
            args.prefix.len(),
            args.inputs.len()
        )));
    }
    let splits = match args.splits {
        MergeSplitArg::Preserve => {
            if args.val.is_some() || args.test.is_some() {
                return Err(invalid_merge("--val and --test only apply with `--splits stratified`"));
            }
            MergeSplitPolicy::Preserve
        }
        MergeSplitArg::PoolToTrain => {
            if args.val.is_some() || args.test.is_some() {
                return Err(invalid_merge("--val and --test only apply with `--splits stratified`"));
            }
            MergeSplitPolicy::PoolToTrain
        }
        MergeSplitArg::Stratified => {
            let Some(val) = args.val else {
                return Err(invalid_merge("`--splits stratified` requires `--val <RATIO>`"));
            };
            let test = args.test.unwrap_or(0.0);
            MergeSplitPolicy::Stratified(split_config(val, test, args.seed)?)
        }
    };
    let normalize: folo_data::ClassNormalize = args.normalize.into();
    let class_map = parse_class_map(&args.remap, &args.drop_class, normalize)?;
    Ok(MergeConfig {
        normalize,
        class_map,
        drop_unmapped: args.drop_unmapped,
        drop_empty_images: args.drop_empty_images,
        splits,
        collisions: args.collisions.into(),
        source_prefixes: args.prefix.clone(),
    })
}

pub(super) fn merge(args: MergeArgs, out: Out) -> Result<(), CliError> {
    let cfg = build_merge_config(&args)?;
    let mut datasets: Vec<LoadedDataset> = Vec::with_capacity(args.inputs.len());
    let mut skipped = 0usize;
    for input in &args.inputs {
        let ds = open(input, Format::Auto)?;
        let bad_splits = unreadable_split_manifests(&ds, input);
        if !bad_splits.is_empty() {
            let (first_path, first_msg) = &bad_splits[0];
            let mut msg = format!("{}: {first_msg}", rel(first_path, &ds.root));
            if bad_splits.len() > 1 {
                msg.push_str(&format!(" (+{} more unreadable split file(s))", bad_splits.len() - 1));
            }
            return Err(CliError::new(ExitCode::Runtime, "invalid_dataset", msg).hint(
                folo_api::json::cli_command(
                    "fix the corrupt annotation or split file; run `folo dataset check {dataset}` to inspect all issues",
                    &[("dataset", input)],
                ),
            ));
        }
        check_convert_output_outside(&args.out, input, &ds)?;
        skipped += ds.report.skipped_images() + ds.report.skipped_label_images();
        datasets.push(ds);
    }
    ensure_empty_output(&args.out)?;
    let refs: Vec<&LoadedDataset> = datasets.iter().collect();
    let t0 = Instant::now();
    let report = folo_data::write_merged(&refs, &args.out, &cfg).map_err(|e| match e {
        DataError::Config(m) => {
            if m.contains("nothing to merge") || m.contains("only has test images") {
                CliError::new(ExitCode::Runtime, "invalid_dataset", m)
            } else {
                invalid_merge(m)
            }
        }
        e => output_data_error(e, &args.out),
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let inputs_json: Vec<String> = args.inputs.iter().map(folo_api::json::path).collect();
    out.record(&json!({
        "inputs": inputs_json,
        "output": folo_api::json::path(&report.root),
        "format": "yolo",
        "classes": report.classes,
        "splits": splits_json(report.splits),
        "boxes": report.boxes,
        "renamed_collisions": report.renamed_collisions,
        "skipped_images": skipped,
        "seconds": secs,
    }));
    let [train, val, test] = report.splits;
    out.human(format!(
        "merged {} dataset(s) → {} ({train} train / {val} val / {test} test images, {} classes, {} boxes, {} \
         renamed collision(s)) in {secs:.1}s",
        args.inputs.len(),
        folo_api::json::path(&report.root),
        report.classes.len(),
        report.boxes,
        report.renamed_collisions,
    ));
    if skipped > 0 {
        out.human(format!(
            "  {skipped} image(s) were skipped across inputs (unreadable or broken labels); run `folo dataset \
             check` on each input"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn build_merge_config_validates_flags_upfront() {
        let parse = |argv: &[&str]| {
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Command::Dataset(crate::cli::DatasetCommand::Merge(m)) = cli.command else {
                panic!()
            };
            m
        };
        let ok = build_merge_config(&parse(&[
            "folo",
            "dataset",
            "merge",
            "a",
            "b",
            "--out",
            "out",
            "--remap",
            "auto=car",
            "--drop-class",
            "junk",
            "--splits",
            "stratified",
            "--val",
            "0.2",
            "--prefix",
            "d1,d2",
        ]))
        .unwrap();
        assert_eq!(ok.class_map.get("auto"), Some(&Some("car".to_string())));
        assert_eq!(ok.class_map.get("junk"), Some(&None));
        assert_eq!(ok.source_prefixes, vec!["d1", "d2"]);

        for argv in [
            // Missing `=` in --remap.
            &["folo", "dataset", "merge", "a", "--out", "out", "--remap", "bad"][..],
            // Conflicting remap and drop.
            &["folo", "dataset", "merge", "a", "--out", "out", "--remap", "car=v", "--drop-class", "car"][..],
            // --val without --splits stratified.
            &["folo", "dataset", "merge", "a", "--out", "out", "--val", "0.2"][..],
            // --splits stratified without --val.
            &["folo", "dataset", "merge", "a", "--out", "out", "--splits", "stratified"][..],
            // Prefix count mismatch.
            &["folo", "dataset", "merge", "a", "b", "--out", "out", "--prefix", "only_one"][..],
        ] {
            let err = build_merge_config(&parse(argv)).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "invalid_config"), "{argv:?}");
        }
    }
}
