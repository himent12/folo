//! Unit tests of `folo train`: `--set` overrides, what `--resume` refuses, distinct run folders for concurrent
//! runs, what `folo stop` asks and refuses, and the server's CLI commands parsing into the server's config.

use super::RunClaim;
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
    let c = apply_sets(with, &sets(&["distill.temperature=2", "distill.cls_gain=0.25"])).unwrap();
    let d = c.distill.unwrap();
    assert_eq!((d.temperature, d.cls_gain, d.dfl_gain), (2.0, 0.25, 1.5));

    // Wave-7 keys (including skip_serializing_if defaults `wd_eff`, `distill.bckd_gamma`, `distill.gt_verify`)
    // must be reachable through `--set`.
    let with_w7 = TrainConfig {
        distill: Some(folo_vision::DistillConfig::new("folo-s-coco")),
        ..TrainConfig::default()
    };
    let c = apply_sets(
        with_w7,
        &sets(&[
            "wd_eff=0.004",
            "decoupled_wd=true",
            "cautious_wd=true",
            "gauge_norm=true",
            "bn_freeze_close_mosaic=true",
            "cls_lr_close_scale=0.333",
            "loss.mal_gamma=1.5",
            "loss.nwd_weight=0.5",
            "loss.tal.nwd_weight=0.5",
            "loss.tal.eps_free=true",
            "distill.bckd_gamma=1.5",
            "distill.gt_verify=true",
        ]),
    )
    .unwrap();
    assert_eq!(c.wd_eff, Some(0.004));
    assert!(c.decoupled_wd && c.cautious_wd && c.gauge_norm && c.bn_freeze_close_mosaic);
    assert!((c.cls_lr_close_scale - 0.333).abs() < 1e-9);
    assert_eq!((c.loss.mal_gamma, c.loss.nwd_weight, c.loss.tal.nwd_weight), (1.5, 0.5, 0.5));
    assert!(c.loss.tal.eps_free);
    let d = c.distill.unwrap();
    assert_eq!((d.bckd_gamma, d.gt_verify), (1.5, true));
    let e = apply_sets(TrainConfig::default(), &sets(&["distill.cls_gain=1"])).unwrap_err();
    assert_eq!(e.code, "invalid_config");

    // Empirically validated math winners must parse cleanly via `--set`.
    let w = apply_sets(
        TrainConfig::default(),
        &sets(&[
            "musgd.ortho=polar3_cs9",
            "ema_mode=power",
            "loss.box_loss=saeiou",
            "loss.dfl_crps_weight=0.25",
            "loss.cls_loss=vfl",
        ]),
    )
    .unwrap();
    assert_eq!(w.musgd.ortho, folo_nn::optim::MuonOrtho::Polar3Cs9);
    assert_eq!(w.ema_mode, folo_nn::EmaMode::Power);
    assert_eq!(w.loss.box_loss, folo_vision::loss::BoxLossKind::Saeiou);
    assert_eq!(w.loss.dfl_crps_weight, 0.25);
    assert_eq!(w.loss.cls_loss, folo_vision::loss::ClsLossKind::Vfl);
    assert!(w.loss.validate_params().is_ok());

    let qfl = apply_sets(TrainConfig::default(), &sets(&["loss.cls_loss=qfl"])).unwrap();
    assert_eq!(qfl.loss.cls_loss, folo_vision::loss::ClsLossKind::Qfl);
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
            std::thread::spawn(move || super::reserve_dir(&project, "shapes-n").unwrap().0)
        })
        .collect();
    let mut dirs: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    dirs.sort();
    dirs.dedup();
    assert_eq!(dirs.len(), 8);
    assert!(dirs.contains(&project.join("shapes-n")) && dirs.contains(&project.join("shapes-n8")));
    let _ = std::fs::remove_dir_all(&project);
}

/// `folo stop` code for `dir` (`None`: the request was written).
fn stop_code(dir: &std::path::Path) -> Option<String> {
    super::request_stop(dir, crate::output::Out { json: false }).err().map(|e| e.code)
}

/// A folder `folo train` reserved, with the files a run in training has (`metrics.jsonl` holding `lines`).
fn run_folder(root: &TestDir, name: &str, lines: &str) -> (std::path::PathBuf, RunClaim) {
    let (dir, claim) = super::reserve_dir(root, name).unwrap();
    std::fs::create_dir(dir.join("weights")).unwrap();
    std::fs::write(dir.join("metrics.jsonl"), lines).unwrap();
    (dir, claim)
}

const STEP: &str = "{\"event\":\"step\",\"step\":1,\"loss\":0.5}\n";
const COMPLETED: &str = "{\"event\":\"status\",\"status\":\"completed\",\"error\":null}\n";

/// #311: a folder `folo train` did not make keeps every byte, whatever project files it has (review round 1,
/// finding 1: `weights/` and a `metrics.jsonl` of another trainer are no proof).
#[test]
fn stop_refuses_folders_folo_train_did_not_make() {
    let root = TestDir::new("stop-other");
    let other = root.join("other");
    std::fs::create_dir_all(other.join("weights")).unwrap();
    std::fs::write(other.join("config.json"), r#"{"app":"unrelated settings"}"#).unwrap();
    std::fs::write(other.join("metrics.jsonl"), "{\"step\":1,\"loss\":0.5}\n").unwrap();
    std::fs::write(other.join("stop"), b"keep these bytes").unwrap();
    assert_eq!(stop_code(&other).as_deref(), Some("run_not_found"));
    assert_eq!(std::fs::read(other.join("stop")).unwrap(), b"keep these bytes");
    std::fs::remove_file(other.join("stop")).unwrap();
    assert_eq!(stop_code(&other).as_deref(), Some("run_not_found"));
    assert_eq!(std::fs::read_dir(&other).unwrap().count(), 3, "nothing written");
}

/// Finding 2: whether a run is live never depends on reading its `metrics.jsonl` (held open without sharing, torn
/// after its final record, or a final record over 64 KiB).
#[test]
fn stop_of_an_ended_run_does_not_read_its_log() {
    let root = TestDir::new("stop-ended");
    let big =
        format!("{{\"event\":\"status\",\"status\":\"completed\",\"error\":\"{}\"}}\n", "x".repeat(70_000));
    for (name, log) in
        [("torn", format!("{STEP}{COMPLETED}{{\"event\":\"st")), ("big", format!("{STEP}{big}"))]
    {
        let (dir, claim) = run_folder(&root, name, &log);
        drop(claim);
        assert_eq!(stop_code(&dir).as_deref(), Some("run_not_active"), "{name}");
        assert!(!dir.join("stop").exists(), "{name}");
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        let (dir, claim) = run_folder(&root, "held", &format!("{STEP}{COMPLETED}"));
        drop(claim);
        let _held =
            std::fs::OpenOptions::new().read(true).share_mode(0).open(dir.join("metrics.jsonl")).unwrap();
        assert_eq!(stop_code(&dir).as_deref(), Some("run_not_active"));
        assert!(!dir.join("stop").exists());
    }
}

/// Finding 3: a run is live exactly while its `folo train` holds the lock, from the moment it reserved the folder
/// (setup: no `metrics.jsonl` yet) until its process ends, also when it is killed (the OS lets go of the lock; a
/// request left behind is removed, and a resume takes it too).
#[test]
fn stop_follows_the_trainer_process() {
    let root = TestDir::new("stop-live");
    let (dir, claim) = super::reserve_dir(&root, "setup").unwrap();
    assert_eq!(stop_code(&dir), None, "a run in setup is asked");
    assert!(dir.join("stop").is_file());
    assert!(RunClaim::take(&dir).unwrap().is_none(), "one trainer per run folder");
    // Killed: the lock is gone with the process, its request is still there.
    drop(claim);
    std::fs::write(dir.join("stop"), b"stop\n").unwrap();
    assert_eq!(stop_code(&dir).as_deref(), Some("run_not_active"));
    assert!(!dir.join("stop").exists());
    // A resume of a run killed with a request pending does not stop at once.
    std::fs::write(dir.join("stop"), b"stop\n").unwrap();
    let resumed = RunClaim::take(&dir).unwrap().expect("free");
    assert!(!dir.join("stop").exists());
    assert_eq!(stop_code(&dir), None);
    drop(resumed);
}

/// Finding 4: a request stays until the run ends, so asking again is the same answer and leaves nothing behind
/// once the trainer is gone.
#[test]
fn stop_twice_is_the_same_answer() {
    let root = TestDir::new("stop-twice");
    let (dir, claim) = run_folder(&root, "run", STEP);
    assert_eq!(stop_code(&dir), None);
    std::fs::write(dir.join("stop"), b"pending").unwrap();
    assert_eq!(stop_code(&dir), None);
    assert_eq!(std::fs::read(dir.join("stop")).unwrap(), b"pending", "never overwritten");
    drop(claim);
    assert!(!dir.join("stop").exists(), "removed when the run ends");
    assert_eq!(stop_code(&dir).as_deref(), Some("run_not_active"));
    assert!(!dir.join("stop").exists());
}

/// Finding 5: a FOLO server's run (its `run.json`) is stopped through the server, never by a file.
#[test]
fn stop_refuses_server_runs() {
    let root = TestDir::new("stop-server");
    let dir = root.join("runs").join("run_a1");
    std::fs::create_dir_all(dir.join("weights")).unwrap();
    std::fs::write(dir.join("metrics.jsonl"), STEP).unwrap();
    std::fs::write(dir.join("run.json"), r#"{"run":{"id":"run_a1","status":"running"},"dataset":{}}"#)
        .unwrap();
    let err = super::request_stop(&dir, crate::output::Out { json: false }).unwrap_err();
    assert_eq!(err.code, "run_owned_by_server");
    assert!(err.hint.as_deref().is_some_and(|h| h.contains("/api/runs/run_a1/stop")), "{:?}", err.hint);
    assert!(!dir.join("stop").exists());
}

/// Review round 2, finding 4: a server run resumed in place with `folo train --resume` is the CLI's to stop while
/// that `folo train` holds it, and the server's again once it ends.
#[test]
fn stop_of_a_server_run_a_folo_train_holds_is_the_clis() {
    let root = TestDir::new("stop-server-resumed");
    let dir = root.join("runs").join("run_a1");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("run.json"), r#"{"run":{"id":"run_a1","status":"stopped"},"dataset":{}}"#)
        .unwrap();
    let claim = RunClaim::take(&dir).unwrap().expect("free");
    assert_eq!(stop_code(&dir), None);
    assert!(dir.join("stop").is_file());
    drop(claim);
    assert_eq!(stop_code(&dir).as_deref(), Some("run_owned_by_server"));
}

/// Finding 1a: the trainer ended between `folo stop`'s look at the lock and its write, so the request it wrote is
/// taken back (the two halves of `request_stop`, with the trainer's lock dropped between them).
#[test]
fn stop_after_the_trainer_ended_takes_its_request_back() {
    let root = TestDir::new("stop-late");
    let (dir, claim) = run_folder(&root, "run", STEP);
    drop(claim);
    super::stop::write_request(&dir).unwrap();
    assert!(dir.join("stop").is_file());
    let err = super::stop::confirm_request(&dir, crate::output::Out { json: false }).unwrap_err();
    assert_eq!(err.code, "run_not_active");
    assert!(!dir.join("stop").exists());
    // With the trainer still there, the request stays.
    let claim = RunClaim::take(&dir).unwrap().expect("free");
    super::stop::write_request(&dir).unwrap();
    super::stop::confirm_request(&dir, crate::output::Out { json: false }).unwrap();
    assert!(dir.join("stop").is_file());
    drop(claim);
}

/// Finding 3: claiming a folder keeps a `folo-train.json` FOLO wrote, creates a missing one, and refuses one with
/// other contents without changing anything.
#[test]
fn claim_never_overwrites_a_marker_folo_did_not_write() {
    let root = TestDir::new("stop-marker");
    let ours = root.join("ours");
    std::fs::create_dir(&ours).unwrap();
    let kept = br#"{"owner":"folo train","note":"kept"}"#;
    std::fs::write(ours.join("folo-train.json"), kept).unwrap();
    drop(RunClaim::take(&ours).unwrap().expect("free"));
    assert_eq!(std::fs::read(ours.join("folo-train.json")).unwrap(), kept);
    // A run of an older FOLO gets its marker.
    let old = root.join("old");
    std::fs::create_dir_all(old.join("weights")).unwrap();
    drop(RunClaim::take(&old).unwrap().expect("free"));
    assert_eq!(std::fs::read(old.join("folo-train.json")).unwrap(), b"{\"owner\":\"folo train\"}\n");
    // Someone's own file of that name.
    let user = root.join("user");
    std::fs::create_dir(&user).unwrap();
    std::fs::write(user.join("folo-train.json"), b"{\"mine\":true}").unwrap();
    std::fs::write(user.join("stop"), b"keep").unwrap();
    let err = RunClaim::take(&user).err().expect("refused");
    assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read(user.join("folo-train.json")).unwrap(), b"{\"mine\":true}");
    assert_eq!(std::fs::read(user.join("stop")).unwrap(), b"keep");
    assert_eq!(std::fs::read_dir(&user).unwrap().count(), 2, "nothing written");
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

/// `--name` rejects blank values, control characters, path separators, `.`/`..`, and `:` before creating any
/// directory under `--project`, while keeping ordinary names (including spaces, quotes, and leading dashes).
#[test]
fn check_run_name_blocks_traversal_and_blank_names() {
    for ok in ["shapes-n", "  shapes-n  ", "my run 'two'", "--amp", "v1.0"] {
        assert_eq!(super::check_run_name(ok).unwrap(), ok.trim(), "{ok}");
    }
    for bad in ["", "   ", ".", "..", "../escape", "sub/run", r"sub\run", "a:stream", "a\nb", "a\0b"] {
        let err = super::check_run_name(bad).unwrap_err();
        assert_eq!(
            (err.exit, err.code.as_str()),
            (crate::output::ExitCode::Usage, "invalid_config"),
            "{bad:?}"
        );
    }
    #[cfg(windows)]
    for bad in ["bad<name>", "trailing.", "NUL"] {
        let err = super::check_run_name(bad).unwrap_err();
        assert_eq!(
            (err.exit, err.code.as_str()),
            (crate::output::ExitCode::Usage, "invalid_config"),
            "{bad:?}"
        );
    }
}
