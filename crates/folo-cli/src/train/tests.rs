//! Unit tests of `folo train`: `--set` overrides, what `--resume` refuses, distinct run folders for concurrent
//! runs, and the server's CLI commands parsing into the server's config.

use super::config::apply_sets;
use super::resume::same;
use folo_test_support::TestDir;
use folo_vision::TrainConfig;
use serde_json::json;

/// `--set KEY=VALUE`: dotted keys of the `train` object, JSON values (or strings), errors for unknown keys
/// and mistyped values.
#[test]
fn set_overrides_train_keys() {
    let sets = |s: &[&str]| s.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let c = apply_sets(
        TrainConfig::default(),
        &sets(&["momentum=0.9", "loss.tal.topk = 13", "augment.degrees=5", "auto_rule=1", "head=o2o"]),
    )
    .unwrap();
    assert_eq!((c.momentum, c.loss.tal.topk, c.augment.degrees, c.auto_rule), (0.9, 13, 5.0, 1));
    assert_eq!(json!(c.head), json!("o2o"));
    assert_eq!(apply_sets(TrainConfig::default(), &[]).unwrap(), TrainConfig::default());
    for bad in ["momentum", "nope=1", "loss.tal.nope=1", "momentum=fast", "loss.tal.topk=-1"] {
        let e = apply_sets(TrainConfig::default(), &sets(&[bad])).unwrap_err();
        assert_eq!(e.code, "invalid_config", "{bad}");
    }
}

/// An object value is merged into the key's object: a misspelt key at any depth is `invalid_config` (it was
/// dropped by the structs' `serde(default)`, keeping the default), and keys it leaves out keep their values
/// (`--mosaic 0` survived only until `--set augment={...}` reset the whole object to its defaults).
#[test]
fn set_object_values_reject_unknown_keys_and_merge() {
    let set = |cfg, s: &str| apply_sets(cfg, &[s.to_string()]);
    for bad in [
        r#"loss={"cls_gian":1.0}"#,
        r#"loss={"tal":{"topk":13,"nope":1}}"#,
        r#"augment={"degres":5}"#,
        r#"loss.tal={"top_k":13}"#,
    ] {
        let e = set(TrainConfig::default(), bad).unwrap_err();
        assert_eq!(e.code, "invalid_config", "{bad}");
    }
    let mut cfg = TrainConfig::default();
    cfg.augment.mosaic = 0.0;
    let c = set(cfg, r#"augment={"degrees":5}"#).unwrap();
    assert_eq!((c.augment.mosaic, c.augment.degrees), (0.0, 5.0));
    let c = set(TrainConfig::default(), r#"loss={"cls_gain":1.0,"tal":{"topk":13}}"#).unwrap();
    assert_eq!((c.loss.cls_gain, c.loss.tal.topk, c.loss.box_gain), (1.0, 13, 7.5));
}

/// Keys whose flags have side effects are refused with the flag to use: `--set name=X` was ignored, `dataset`
/// and `data_format` were written to config.json while training read `--data` (a later resume reopened the
/// wrong path), `amp` skipped asking for tensor cores and `device` skipped creating the device.
#[test]
fn set_refuses_keys_with_their_own_flags() {
    for (key, flag) in [
        ("name=x", "--name"),
        ("dataset=D:/elsewhere", "--data"),
        ("data_format=coco", "--format"),
        ("arch=folo-s", "--model"),
        ("device=gpu:0", "--device"),
        (" amp =true", "--amp"),
    ] {
        let e = apply_sets(TrainConfig::default(), &[key.to_string()]).unwrap_err();
        assert!(e.code == "invalid_config" && matches!(e.exit, crate::output::ExitCode::Usage), "{key}");
        assert!(e.message.contains(flag), "{key}: {}", e.message);
    }
}

/// `--weights` with `--resume` is refused: the run continues from its checkpoint, so the weights were silently
/// ignored (the flag is not one a resume may change, and the stored config has nothing to compare it with).
#[test]
fn resume_refuses_weights() {
    use clap::Parser;
    let cli = crate::cli::Cli::try_parse_from([
        "folo",
        "train",
        "--resume",
        "r/weights/last.folo",
        "--weights",
        "other.folo",
    ])
    .unwrap();
    let crate::cli::Command::Train(mut args) = cli.command else { panic!("not train") };
    args.explicit = vec!["resume".into(), "weights".into()];
    let e = super::resume::check_resume_flags(&args, &TrainConfig::default()).unwrap_err();
    assert_eq!((e.code.as_str(), e.exit), ("invalid_config", crate::output::ExitCode::Usage));
    assert!(e.message.contains("--weights"), "{}", e.message);
}

/// `--teacher` parses; it stores a file as an absolute path and a published name as given, and `--set distill.*`
/// changes the starting values only once a teacher is set.
#[test]
fn teacher_flag_and_distill_keys() {
    use clap::Parser;
    let parse = |extra: &[&str]| {
        let argv = ["folo", "train", "--data", "d"].iter().chain(extra).copied().collect::<Vec<_>>();
        let crate::cli::Command::Train(args) = crate::cli::Cli::try_parse_from(argv).unwrap().command else {
            panic!("not train")
        };
        args
    };
    assert_eq!(parse(&["--teacher", "folo-s-coco"]).teacher.as_deref(), Some("folo-s-coco"));
    assert_eq!(parse(&[]).teacher, None);
    assert_eq!(super::teacher_setting("folo-s-coco"), "folo-s-coco");
    let dir = TestDir::new("teacher-setting");
    std::fs::write(dir.join("t.folo"), b"x").unwrap();
    let file = folo_api::json::path(dir.join("t.folo"));
    assert_eq!(super::teacher_setting(&file), file, "an existing file, absolute");

    let with = TrainConfig {
        distill: Some(folo_vision::DistillConfig::new("folo-s-coco")),
        ..TrainConfig::default()
    };
    let sets = |s: &[&str]| s.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let c = apply_sets(with.clone(), &sets(&["distill.temperature=2", "distill.cls_gain=0.25"])).unwrap();
    let d = c.distill.unwrap();
    assert_eq!((d.temperature, d.cls_gain, d.dfl_gain), (2.0, 0.25, 1.5));
    // The rank term's keys (docs/MODEL.md §8.9), off until rank_gain is set.
    assert_eq!(with.distill.as_ref().unwrap().rank_gain, 0.0);
    let c = apply_sets(with, &sets(&["distill.rank_gain=2", "distill.rank_power=1", "distill.rank_min=12"]))
        .unwrap();
    let d = c.distill.unwrap();
    assert_eq!((d.rank_gain, d.rank_power, d.rank_min, d.rank_floor), (2.0, 1.0, 12, 0.005));
    let e = apply_sets(TrainConfig::default(), &sets(&["distill.cls_gain=1"])).unwrap_err();
    assert_eq!(e.code, "invalid_config");
}

/// The resume override: `--resume` with `--teacher` turns distillation on (starting values, or the run's stored
/// gains with the new teacher), `--set distill.*` changes the gains, and without either the stored config stays.
#[test]
fn resume_overrides_the_teacher() {
    use super::resume::{check_resume_flags, resume_distill};
    use clap::Parser;
    let dir = TestDir::new("resume-teacher");
    std::fs::write(dir.join("t.folo"), b"x").unwrap();
    let file = folo_api::json::path(dir.join("t.folo"));
    let args = |extra: &[&str]| {
        let mut argv = vec!["folo", "train", "--resume", "r/weights/last.folo"];
        argv.extend_from_slice(extra);
        let crate::cli::Command::Train(mut a) = crate::cli::Cli::try_parse_from(argv).unwrap().command else {
            panic!("not train")
        };
        a.explicit = vec!["resume".into()];
        if a.teacher.is_some() {
            a.explicit.push("teacher".into());
        }
        a
    };
    let plain = TrainConfig::default();
    let mut tuned = folo_vision::DistillConfig::new("folo-s-coco");
    tuned.temperature = 3.0;
    let distilled = TrainConfig { distill: Some(tuned), ..TrainConfig::default() };

    // Without --teacher and --set: the stored config stays (no override).
    assert_eq!(resume_distill(&args(&[]), &distilled).unwrap(), None);
    // --teacher on a run without distillation: on, at the starting values.
    let a = args(&["--teacher", &file]);
    assert!(check_resume_flags(&a, &plain).is_ok(), "--teacher may change on a resume");
    assert_eq!(resume_distill(&a, &plain).unwrap(), Some(folo_vision::DistillConfig::new(file.clone())));
    // --teacher on a distillation run: the new teacher with the run's gains; --set changes them.
    let d = resume_distill(&args(&["--teacher", &file, "--set", "distill.cls_gain=0.25"]), &distilled)
        .unwrap()
        .unwrap();
    assert_eq!((d.teacher.as_str(), d.cls_gain, d.temperature), (file.as_str(), 0.25, 3.0));
    // --set distill.* alone changes a distillation run's gains and needs a teacher otherwise.
    let d = resume_distill(&args(&["--set", "distill.dfl_gain=0"]), &distilled).unwrap().unwrap();
    assert_eq!((d.teacher.as_str(), d.dfl_gain), ("folo-s-coco", 0.0));
    let e = resume_distill(&args(&["--set", "distill.dfl_gain=1"]), &plain).unwrap_err();
    assert!(e.message.contains("--teacher"), "{}", e.message);
    // Invalid values are errors; a missing teacher file is not found.
    let e =
        resume_distill(&args(&["--teacher", &file, "--set", "distill.temperature=0"]), &plain).unwrap_err();
    assert!(e.message.contains("distill.temperature"), "{}", e.message);
    assert!(resume_distill(&args(&["--teacher", "no/such/teacher.folo"]), &plain).is_err());
}

#[test]
fn resume_settings_compare_integers_exactly() {
    let f32s = |a, b| same(&a, &b, true);
    assert!(!f32s(json!(20_261_108u64), json!(20_261_109u64)));
    assert!(!f32s(json!(u64::MAX), json!(u64::MAX - 1)));
    assert!(f32s(json!(16), json!(16u32)));
    // f32 settings widened to f64 still match what the flag parses to.
    assert!(f32s(json!(0.015f32 as f64), json!(0.015f32)));
    assert!(f32s(json!([0.015f32 as f64, 0.7]), json!([0.015f32, 0.7f32])));
    assert!(!f32s(json!(0.5), json!(0.6)));
    // Small learning rates are told apart.
    assert!(!f32s(json!(1e-7), json!(1e-6)));
    assert!(f32s(json!(folo_vision::train::f32_decimal(1e-7)), json!(1e-7f32)));
    // f64 settings compare exactly (1e-50 is 0 as an f32).
    assert!(!same(&json!(1e-50), &json!(0.0), false));
    assert!(same(&json!(0.01), &json!(0.01), false));
}

#[test]
fn concurrent_runs_reserve_distinct_folders() {
    let project = TestDir::new("reserve");
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let project = project.to_path_buf();
            std::thread::spawn(move || super::reserve_dir(&project, "shapes-n").unwrap())
        })
        .collect();
    let mut dirs: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    dirs.sort();
    dirs.dedup();
    assert_eq!(dirs.len(), 8);
    assert!(dirs.contains(&project.join("shapes-n")) && dirs.contains(&project.join("shapes-n8")));
    let _ = std::fs::remove_dir_all(&project);
}

/// The server's `folo train` command (`RunEstimate.cli`, `GET /api/runs/{id}/cli`) parses with the real
/// arguments, and maps to the trainer config the server builds from the same `TrainConfig` (`from_api`).
#[test]
fn server_cli_commands_train_the_servers_config() {
    use clap::Parser;
    use folo_vision::model::HeadMode;
    let base: folo_api::TrainConfig =
        serde_json::from_value(json!({ "dataset_id": "ds_1", "name": "fasteners-n" })).unwrap();
    let mut all = base.clone();
    all.name = "my run 'two'".into();
    all.arch = folo_api::Arch::S;
    (all.epochs, all.batch, all.imgsz, all.seed) = (37, 8, 320, 7);
    all.device = "gpu:0".into();
    all.optimizer = folo_api::Optimizer::Sgd;
    all.lr0 = 0.0123;
    all.amp = true;
    all.head = folo_api::Head::O2o;
    all.augment = folo_api::Augment {
        mosaic: 0.5,
        mixup: 0.15,
        hsv: [0.0, 0.25, 0.333],
        fliplr: 0.0,
        scale: 0.9,
        translate: 0.05,
        close_mosaic: 3,
    };
    let mut adamw = base.clone();
    adamw.optimizer = folo_api::Optimizer::Adamw;
    // A name that looks like a flag.
    let mut flaglike = base.clone();
    flaglike.name = "--amp".into();
    let data = "C:/data sets/fasteners";
    let (auto, coco) = (folo_api::DatasetFormat::Auto, folo_api::DatasetFormat::Coco);
    // (config, registered format, --weights, the head of the weights file: the CLI's default without --head)
    let cases = [
        (base.clone(), auto, None, HeadMode::O2m),
        (all.clone(), coco, None, HeadMode::O2m),
        (adamw, folo_api::DatasetFormat::Yolo, None, HeadMode::O2m),
        (flaglike, auto, None, HeadMode::O2m),
        (base.clone(), auto, Some("folo-n-coco"), HeadMode::O2o),
        (all.clone(), coco, Some("D:/folo/runs/run_1a2b/weights/best.folo"), HeadMode::O2o),
    ];
    for (api, format, weights, file_head) in cases {
        let cmd = folo_server::cli::train_command(&api, data, format, weights);
        let cli = crate::cli::Cli::try_parse_from(&cmd.argv).unwrap_or_else(|e| panic!("{}: {e}", cmd.cli));
        let crate::cli::Command::Train(args) = cli.command else { panic!("not train: {}", cmd.cli) };
        assert_eq!(args.data.as_deref(), Some(data), "{}", cmd.cli);
        assert_eq!(args.weights.as_deref(), weights, "{}", cmd.cli);
        // Read as the server reads the registered dataset (runs/mod.rs `data_format`).
        let want_format = match format {
            folo_api::DatasetFormat::Auto => folo_data::Format::Auto,
            folo_api::DatasetFormat::Yolo => folo_data::Format::Yolo,
            folo_api::DatasetFormat::Coco => folo_data::Format::Coco,
        };
        assert_eq!(crate::dataset::format_of(args.format), want_format, "{}", cmd.cli);
        // With --weights the CLI takes the architecture from the file; the server already set it to the weights'.
        let arch = if weights.is_some() { api.arch.as_str().to_string() } else { args.model.clone() };
        let got = super::config::config_of(&args, args.name.clone().unwrap(), data.into(), arch, file_head);
        let mut want = folo_vision::TrainConfig::from_api(&api).unwrap();
        (want.dataset, want.data_format, want.workers) = (data.into(), want_format, got.workers);
        assert_eq!(got, want, "{}", cmd.cli);
    }
}
