//! `folo dataset split`: deterministic multi-label stratified splitting on top of `folo_data::split`.

use std::time::Instant;

use folo_data::{DataError, SplitConfig};
use serde_json::json;

use crate::cli::{SplitArgs, bounds};
use crate::output::{CliError, ExitCode, Out};

use super::{
    check_convert_output_outside, ensure_empty_output, format_name, format_of, open, output_data_error, rel,
    splits_json, target_format_of, unreadable_coco_splits,
};

fn invalid_split(message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "invalid_config", message)
        .hint("see `folo describe dataset split --json` for the accepted ranges")
}

/// Validates `--val` and `--test` ratios upfront before opening the input dataset.
fn split_config(val: f64, test: f64, seed: u64) -> Result<SplitConfig, CliError> {
    if !bounds::UNIT_INTERVAL.contains_f64(val) {
        return Err(invalid_split(format!("--val must be in 0..=1, got {val}")));
    }
    if !bounds::UNIT_INTERVAL.contains_f64(test) {
        return Err(invalid_split(format!("--test must be in 0..=1, got {test}")));
    }
    let train = 1.0 - val - test;
    if !train.is_finite() || train < -1e-6 {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--val ({val}) + --test ({test}) must not exceed 1.0"),
        )
        .hint("train receives the remainder `1 - val - test`"));
    }
    let cfg = SplitConfig::train_val_test(train.max(0.0), val, test, seed);
    cfg.validate().map_err(|e| invalid_split(e.to_string()))?;
    Ok(cfg)
}

pub(super) fn split(args: SplitArgs, out: Out) -> Result<(), CliError> {
    let cfg = split_config(args.val, args.test, args.seed)?;
    let ds = open(&args.path, format_of(args.from))?;
    let bad_splits = unreadable_coco_splits(&ds, &args.path);
    if !bad_splits.is_empty() {
        let (first_path, first_msg) = &bad_splits[0];
        let mut msg = format!("{}: {first_msg}", rel(first_path, &ds.root));
        if bad_splits.len() > 1 {
            msg.push_str(&format!(" (+{} more unreadable annotation file(s))", bad_splits.len() - 1));
        }
        return Err(CliError::new(ExitCode::Runtime, "invalid_dataset", msg).hint(
            folo_api::json::cli_command(
                "fix the corrupt annotation file; run `folo dataset check {dataset}` to inspect all issues",
                &[("dataset", &args.path)],
            ),
        ));
    }
    let to = args.to.map_or(ds.format, target_format_of);
    check_convert_output_outside(&args.out, &args.path, &ds)?;
    ensure_empty_output(&args.out)?;
    let skipped = ds.report.skipped_images() + ds.report.skipped_label_images();
    let t0 = Instant::now();
    let done = folo_data::write_split(&ds, &args.out, to, &cfg).map_err(|e| match e {
        DataError::Config(m) => CliError::new(ExitCode::Runtime, "invalid_dataset", m),
        e => output_data_error(e, &args.out),
    })?;
    let secs = t0.elapsed().as_secs_f64();
    out.record(&json!({
        "input": folo_api::json::path(&args.path),
        "output": folo_api::json::path(&done.root),
        "from": format_name(ds.format),
        "to": format_name(to),
        "classes": ds.classes,
        "ratios": { "train": cfg.train, "val": cfg.val, "test": cfg.test },
        "seed": cfg.seed,
        "splits": splits_json(done.splits),
        "boxes": done.boxes,
        "skipped_images": skipped,
        "seconds": secs,
    }));
    let [train, val, test] = done.splits;
    out.human(format!(
        "split {} {} → {} at {} ({train} train / {val} val / {test} test images, {} boxes, seed {}) in {secs:.1}s",
        format_name(ds.format),
        folo_api::json::path(&args.path),
        format_name(to),
        folo_api::json::path(&done.root),
        done.boxes,
        cfg.seed,
    ));
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

    #[test]
    fn split_config_validates_ratios_upfront() {
        let two = split_config(0.2, 0.0, 42).unwrap();
        assert!((two.train - 0.8).abs() < 1e-12);
        assert_eq!((two.val, two.test, two.seed), (0.2, 0.0, 42));

        let three = split_config(0.2, 0.1, 7).unwrap();
        assert!((three.train - 0.7).abs() < 1e-12);

        let all_val = split_config(1.0, 0.0, 0).unwrap();
        assert_eq!(all_val.train, 0.0);

        for (val, test) in [(-0.1, 0.0), (1.1, 0.0), (0.2, -0.1), (0.2, 1.1), (0.6, 0.5), (f64::NAN, 0.0)] {
            let err = split_config(val, test, 0).unwrap_err();
            assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "invalid_config"), "{val}/{test}");
        }
    }
}
