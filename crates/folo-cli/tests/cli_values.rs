//! Flags that take several values parse in every form their help documents, and single-value flags take exactly
//! one (DOGFOOD-3: `--hsv 0,0.5,0.3`, the documented form, was rejected).

use std::process::Command;

use clap::Parser;
use folo_cli::cli::{Cli, Command as Sub, DatasetCommand, TrainArgs};

fn train(args: &[&str]) -> Result<TrainArgs, clap::Error> {
    let argv = ["folo", "train", "--data", "d"].into_iter().chain(args.iter().copied());
    Cli::try_parse_from(argv).map(|c| match c.command {
        Sub::Train(t) => t,
        _ => unreachable!(),
    })
}

/// `--hsv`: `h,s,v` (the help's form), three separate values, `--hsv=h,s,v`, or the default.
#[test]
fn hsv_takes_the_comma_and_the_spaced_form() {
    for args in [&["--hsv", "0,0.5,0.3"][..], &["--hsv", "0", "0.5", "0.3"], &["--hsv=0,0.5,0.3"]] {
        assert_eq!(train(args).unwrap().hsv, [0.0, 0.5, 0.3], "{args:?}");
    }
    assert_eq!(train(&[]).unwrap().hsv, [0.015, 0.7, 0.4]);
    assert!(train(&["--hsv", "0,x,0.3"]).is_err(), "not numbers");
}

/// A gain count other than three parses (clap counts the comma list as one value) but `folo train` refuses it as a
/// usage error before it opens anything.
#[test]
fn hsv_needs_three_gains() {
    for hsv in ["0.1,0.2", "0.1,0.2,0.3,0.4"] {
        let out = Command::new(env!("CARGO_BIN_EXE_folo"))
            .args(["train", "--json", "--data", "does-not-exist", "--hsv", hsv])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{hsv}");
        let stdout = String::from_utf8(out.stdout).unwrap();
        assert!(stdout.contains("\"usage\"") && stdout.contains("three gains"), "{hsv}: {stdout}");
    }
}

/// Reads one numeric train setting.
type Field = fn(&TrainArgs) -> f64;

/// The other numeric train flags take one value each: no comma lists, no second value.
#[test]
fn scalar_train_flags_take_one_value() {
    let get: [(&str, Field); 8] = [
        ("--mosaic", |t| t.mosaic as f64),
        ("--mixup", |t| t.mixup as f64),
        ("--fliplr", |t| t.fliplr as f64),
        ("--scale", |t| t.scale as f64),
        ("--translate", |t| t.translate as f64),
        ("--lr0", |t| t.lr0 as f64),
        ("--lrf", |t| t.lrf),
        ("--warmup-epochs", |t| t.warmup_epochs),
    ];
    for (flag, value) in get {
        let t = train(&[flag, "0.25"]).unwrap_or_else(|e| panic!("{flag}: {e}"));
        assert_eq!(value(&t), 0.25, "{flag}");
        assert!(train(&[flag, "0.25,0.5"]).is_err(), "{flag} takes no list");
        assert!(train(&[flag, "0.25", "0.5"]).is_err(), "{flag} takes one value");
    }
}

/// `folo dataset synth --classes`: a comma list (the only multi-value flag besides `--hsv`).
#[test]
fn synth_classes_take_a_comma_list() {
    let classes = |args: &[&str]| {
        let argv = ["folo", "dataset", "synth", "out"].into_iter().chain(args.iter().copied());
        match Cli::try_parse_from(argv).unwrap().command {
            Sub::Dataset(DatasetCommand::Synth(s)) => s.classes,
            _ => unreachable!(),
        }
    };
    assert_eq!(classes(&["--classes", "a,b,c"]), ["a", "b", "c"]);
    assert_eq!(classes(&["--classes", "solo"]), ["solo"]);
    assert_eq!(classes(&[]), ["circle", "square", "triangle"]);
}
