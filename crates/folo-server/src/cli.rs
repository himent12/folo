//! The `folo train` command of a [`TrainConfig`] (`RunEstimate.cli`, `GET /api/runs/{id}/cli`): the flags the CLI
//! maps back to the same training settings, so a run set up in Studio or over the API can be repeated from a
//! terminal on the server's machine. `crates/folo-cli`'s tests parse it with the real `folo` arguments and compare the
//! resulting config with the server's.

use folo_api::{CliCommand, DatasetFormat, Head, Optimizer, TrainConfig};

/// The command that trains `config` on the dataset at `data` (a path on the server's machine) read as `format`
/// (the registered format: `--format` unless it is `auto`), starting from `weights` (a `.folo` path there, or a
/// published name; `None` trains from scratch).
///
/// Settings equal to the CLI's defaults are left out (they are the API's defaults too). With `weights` the
/// architecture is the weights' (the CLI refuses `--model` with `--weights`), and the head is always given: the CLI
/// defaults to the weights file's head, the API to `o2m`. A value that starts with `-` (a run named `--amp`) is
/// attached as `--flag=value`, so it is never read as a flag.
pub fn train_command(
    config: &TrainConfig,
    data: &str,
    format: DatasetFormat,
    weights: Option<&str>,
) -> CliCommand {
    let d: TrainConfig =
        serde_json::from_value(serde_json::json!({ "dataset_id": "" })).expect("TrainConfig has defaults");
    let mut argv: Vec<String> = ["folo", "train"].map(String::from).to_vec();
    let mut arg = |flag: &str, value: String| {
        if value.starts_with('-') {
            argv.push(format!("--{flag}={value}"));
        } else {
            argv.push(format!("--{flag}"));
            argv.push(value);
        }
    };
    arg("data", data.to_string());
    match format {
        DatasetFormat::Auto => {}
        DatasetFormat::Yolo => arg("format", "yolo".into()),
        DatasetFormat::Coco => arg("format", "coco".into()),
    }
    match weights {
        Some(w) => arg("weights", w.to_string()),
        None if config.arch != d.arch => arg("model", config.arch.as_str().to_string()),
        None => {}
    }
    if !config.name.trim().is_empty() {
        arg("name", config.name.clone());
    }
    if config.epochs != d.epochs {
        arg("epochs", config.epochs.to_string());
    }
    if config.batch != d.batch {
        arg("batch", config.batch.to_string());
    }
    if config.imgsz != d.imgsz {
        arg("imgsz", config.imgsz.to_string());
    }
    if config.device != d.device {
        arg("device", config.device.clone());
    }
    if config.optimizer != d.optimizer {
        let o = match config.optimizer {
            Optimizer::Auto => Some("auto"),
            Optimizer::Sgd => Some("sgd"),
            Optimizer::Adamw => Some("adamw"),
            Optimizer::Musgd => Some("musgd"),
            // Refused by `check_optimizer` before a command is built for a run.
            Optimizer::Unknown => None,
        };
        if let Some(o) = o {
            arg("optimizer", o.to_string());
        }
    }
    if config.lr0 != d.lr0 {
        arg("lr0", config.lr0.to_string());
    }
    if config.seed != d.seed {
        arg("seed", config.seed.to_string());
    }
    if weights.is_some() || config.head != Head::O2m {
        arg("head", config.head.as_str().to_string());
    }
    let (a, da) = (&config.augment, &d.augment);
    for (flag, v, dv) in [
        ("mosaic", a.mosaic, da.mosaic),
        ("mixup", a.mixup, da.mixup),
        ("fliplr", a.fliplr, da.fliplr),
        ("scale", a.scale, da.scale),
        ("translate", a.translate, da.translate),
    ] {
        if v != dv {
            arg(flag, v.to_string());
        }
    }
    if a.hsv != da.hsv {
        arg("hsv", a.hsv.map(|v| v.to_string()).join(","));
    }
    if a.close_mosaic != da.close_mosaic {
        arg("close-mosaic", a.close_mosaic.to_string());
    }
    if config.amp {
        argv.push("--amp".into());
    }
    CliCommand { cli: shell_join(&argv), argv }
}

/// Refuses (`invalid_config`) the command of a config whose optimizer this build does not know
/// ([`Optimizer::Unknown`]: a run made by a newer FOLO): no `--optimizer` value here says what it trained with, and
/// leaving the flag out would train `auto` instead.
pub fn check_optimizer(config: &TrainConfig) -> crate::error::Result<()> {
    if config.optimizer != Optimizer::Unknown {
        return Ok(());
    }
    Err(crate::error::Error::new(
        axum::http::StatusCode::BAD_REQUEST,
        "invalid_config",
        "this run's optimizer is unknown to this server (made by a newer FOLO): no folo train command here repeats it",
    )
    .hint("use the CLI of the FOLO version that made the run, or start a new run with --optimizer auto, sgd, adamw or musgd"))
}

/// The trainer config a run trained with (the `train` object of the `config.json` the trainer wrote into its
/// folder), or, when the folder has no `config.json`, the `folo.train` of its checkpoint (`weights/last.folo`, else
/// `weights/best.folo`; only the file's header is read). A cloud run's folder has no `config.json`: the server
/// mirrors its history and copies its weights (docs/CLOUD.md). `None` when the run has neither (it never started,
/// and starts with the current defaults).
fn recorded_train(run_dir: &std::path::Path) -> Option<serde_json::Value> {
    if let Ok(bytes) = std::fs::read(run_dir.join("config.json")) {
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        return Some(v.get_mut("train")?.take());
    }
    ["last.folo", "best.folo"].iter().find_map(|f| {
        let header =
            folo_nn::io::safetensors::Container::read_header(&run_dir.join("weights").join(f)).ok()?;
        serde_json::from_str(header.metadata?.get("folo.train")?).ok()
    })
}

/// The `auto` rule (`TrainConfig.auto_rule`, docs/MODEL.md §8.2) a run trained with ([`recorded_train`]): its
/// `auto_rule`, or rule 1 when the key is absent (written before it existed). `None` when nothing records the run's
/// config (it never started, and starts with the current rule).
pub fn recorded_auto_rule(run_dir: &std::path::Path) -> Option<u32> {
    let train = recorded_train(run_dir)?;
    Some(train.get("auto_rule").and_then(serde_json::Value::as_u64).map_or(1, |r| r as u32))
}

/// `cmd` (a [`train_command`] of `config`) with `--set auto_rule=N` when `config` asks for the `auto` optimizer and
/// the run trained under `rule`, an older rule than a new run would take: the command then trains what the run did
/// (rule 1 or 2: SGD for long runs, where rule 3 picks MuSGD).
pub fn with_auto_rule(mut cmd: CliCommand, config: &TrainConfig, rule: Option<u32>) -> CliCommand {
    let current = folo_vision::TrainConfig::default().auto_rule;
    if let Some(r) = rule
        && config.optimizer == Optimizer::Auto
        && r != current
    {
        cmd.argv.extend(["--set".to_string(), format!("auto_rule={r}")]);
        cmd.cli = shell_join(&cmd.argv);
    }
    cmd
}

/// The warmup floor (`TrainConfig.warmup_floor`, docs/MODEL.md §8.4) a run trained with ([`recorded_train`]): its
/// `warmup_floor`, or 100 when the key is absent (written before it existed). `None` when nothing records the run's
/// config (it starts with the current default).
pub fn recorded_warmup_floor(run_dir: &std::path::Path) -> Option<u64> {
    let train = recorded_train(run_dir)?;
    Some(train.get("warmup_floor").and_then(serde_json::Value::as_u64).unwrap_or(folo_nn::WARMUP_FLOOR))
}

/// `cmd` with `--set warmup_floor=N` when the run trained with another floor than a new run takes, so the command
/// warms up as the run did (runs made before the key: 100 iterations).
pub fn with_warmup_floor(mut cmd: CliCommand, floor: Option<u64>) -> CliCommand {
    if let Some(f) = floor
        && f != folo_vision::TrainConfig::default().warmup_floor
    {
        cmd.argv.extend(["--set".to_string(), format!("warmup_floor={f}")]);
        cmd.cli = shell_join(&cmd.argv);
    }
    cmd
}

/// The EMA window fraction (`TrainConfig.ema_run_fraction`, docs/MODEL.md §8.5) a run trained with
/// ([`recorded_train`]): its `ema_run_fraction`, or 0 (the default window) when the key is absent (written before it
/// existed). `None` when nothing records the run's config (it starts with the current default).
pub fn recorded_ema_run_fraction(run_dir: &std::path::Path) -> Option<f64> {
    let train = recorded_train(run_dir)?;
    Some(train.get("ema_run_fraction").and_then(serde_json::Value::as_f64).unwrap_or(0.0))
}

/// `cmd` with `--set ema_run_fraction=F` when a run from pretrained weights (`transfer`; the fraction changes no other
/// run) trained with another fraction than a new run takes, so the command averages as the run did (fine-tunes made
/// before the key: the default window, 0).
pub fn with_ema_run_fraction(
    mut cmd: CliCommand,
    fraction: Option<f64>,
    transfer: Option<bool>,
) -> CliCommand {
    if let Some(f) = fraction
        && transfer == Some(true)
        && f != folo_vision::TrainConfig::default().ema_run_fraction
    {
        cmd.argv.extend(["--set".to_string(), format!("ema_run_fraction={f}")]);
        cmd.cli = shell_join(&cmd.argv);
    }
    cmd
}

/// Whether a run started from pretrained weights (`TrainConfig.transfer`, docs/MODEL.md §8.2) ([`recorded_train`]):
/// `false` when the key is absent (written before it existed). `None` when nothing records the run's config.
pub fn recorded_transfer(run_dir: &std::path::Path) -> Option<bool> {
    let train = recorded_train(run_dir)?;
    Some(train.get("transfer").and_then(serde_json::Value::as_bool).unwrap_or(false))
}

/// The MuSGD step weights a run's config records (`musgd.muon`, `musgd.sgd`; [`recorded_train`]), the defaults
/// 0.2 / 1.0 when absent. `None` when nothing records the run's config.
pub fn recorded_musgd_weights(run_dir: &std::path::Path) -> Option<folo_nn::optim::MuSgdWeights> {
    let train = recorded_train(run_dir)?;
    let d = folo_nn::optim::MuSgdWeights::default();
    let w = |k: &str, d: f64| {
        train.get("musgd").and_then(|m| m.get(k)).and_then(serde_json::Value::as_f64).unwrap_or(d)
    };
    Some(folo_nn::optim::MuSgdWeights { muon: w("muon", d.muon), sgd: w("sgd", d.sgd) })
}

/// `cmd` with `--set auto_rule=4` for an `auto` run under the current rule that trained as a transfer run
/// (`transfer`) while the command has no `--weights` (e.g. a run continued from an imported checkpoint): the command
/// would train from scratch, which rule 5 resolves to other MuSGD step weights; rule 4 resolves the same optimizer
/// with `musgd.*`, so the run's own `musgd` weights (`weights`) follow when they are not the defaults. Runs under
/// older rules already carry their rule.
pub fn with_transfer(
    mut cmd: CliCommand,
    config: &TrainConfig,
    rule: Option<u32>,
    transfer: Option<bool>,
    weights: Option<folo_nn::optim::MuSgdWeights>,
) -> CliCommand {
    let current = folo_vision::TrainConfig::default().auto_rule;
    if config.optimizer == Optimizer::Auto
        && transfer == Some(true)
        && rule.unwrap_or(current) == current
        && current >= 5
        && !cmd.argv.iter().any(|a| a == "--weights" || a.starts_with("--weights="))
    {
        cmd.argv.extend(["--set".to_string(), "auto_rule=4".to_string()]);
        if let Some(w) = weights.filter(|w| *w != folo_nn::optim::MuSgdWeights::default()) {
            for (k, v) in [("muon", w.muon), ("sgd", w.sgd)] {
                cmd.argv.extend(["--set".to_string(), format!("musgd.{k}={v}")]);
            }
        }
        cmd.cli = shell_join(&cmd.argv);
    }
    cmd
}

/// The distillation settings a run trained with (`distill`, docs/MODEL.md §8.9; [`recorded_train`]), `None` when it
/// had no teacher or nothing records its config.
pub fn recorded_distill(run_dir: &std::path::Path) -> Option<folo_vision::DistillConfig> {
    serde_json::from_value(recorded_train(run_dir)?.get("distill")?.clone()).ok()
}

/// `cmd` with `--teacher` and `--set distill.<key>=<value>` for each setting that differs from the starting values,
/// when the run distilled from a teacher (a run continued from a CLI run's checkpoint: the API itself takes none).
pub fn with_distill(mut cmd: CliCommand, distill: Option<folo_vision::DistillConfig>) -> CliCommand {
    let Some(d) = distill else { return cmd };
    let start = folo_vision::DistillConfig::new(d.teacher.clone());
    cmd.argv.extend(["--teacher".to_string(), d.teacher.clone()]);
    let mut set = |k: &str, v: String| cmd.argv.extend(["--set".to_string(), format!("distill.{k}={v}")]);
    for (k, v, s) in [
        ("cls_gain", d.cls_gain, start.cls_gain),
        ("dfl_gain", d.dfl_gain, start.dfl_gain),
        ("temperature", d.temperature, start.temperature),
    ] {
        if v != s {
            set(k, v.to_string());
        }
    }
    cmd.cli = shell_join(&cmd.argv);
    cmd
}

/// `argv` as one command line, quoted for PowerShell (as Studio shows commands): an argument with characters a shell
/// treats specially is single-quoted, with each quote inside doubled (`'O''Brien'`; PowerShell also reads the
/// typographic quotes U+2018 to U+201B as single quotes, so they are doubled too). POSIX shells read the same words
/// unless an argument holds a quote; `argv` is exact either way.
pub fn shell_join(argv: &[String]) -> String {
    argv.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ")
}

/// A single quote to PowerShell.
fn is_quote(c: char) -> bool {
    matches!(c, '\'' | '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}')
}

fn quote(a: &str) -> String {
    let plain = !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:=".contains(c));
    if plain {
        return a.to_string();
    }
    let mut q = String::with_capacity(a.len() + 2);
    q.push('\'');
    for c in a.chars() {
        q.push(c);
        if is_quote(c) {
            q.push(c);
        }
    }
    q.push('\'');
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_left_out_and_arguments_quoted() {
        let mut c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        let cmd = train_command(&c, "D:/data/fasteners", DatasetFormat::Auto, None);
        assert_eq!(cmd.cli, "folo train --data D:/data/fasteners");
        c.name = "fasteners n".into();
        c.amp = true;
        c.augment.hsv = [0.0, 0.5, 0.25];
        let cmd = train_command(&c, "C:/Users/a b/data", DatasetFormat::Coco, Some("folo-n-coco"));
        assert_eq!(
            cmd.cli,
            "folo train --data 'C:/Users/a b/data' --format coco --weights folo-n-coco --name 'fasteners n' --head o2m \
             --hsv '0,0.5,0.25' --amp"
        );
        assert_eq!(cmd.argv[3], "C:/Users/a b/data");
        // A value that looks like a flag is attached.
        c.name = "--amp".into();
        let cmd = train_command(&c, "d", DatasetFormat::Auto, None);
        assert!(cmd.argv.contains(&"--name=--amp".to_string()), "{:?}", cmd.argv);
    }

    /// A run made before `auto` rule 3 repeats with its own rule (rule 1 when its config.json predates the key), so
    /// the command trains SGD for a long run as it did, not MuSGD; current-rule runs, explicit optimizers and runs
    /// that never started get no `--set`.
    #[test]
    fn auto_runs_repeat_with_their_rule() {
        use folo_test_support::TestDir;
        let dir = TestDir::new("cli-auto-rule");
        let write = |v: serde_json::Value| std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        assert_eq!(recorded_auto_rule(&dir), None, "no config.json");
        write(serde_json::json!({ "train": { "epochs": 300 } }));
        assert_eq!(recorded_auto_rule(&dir), Some(1), "written before the key");
        write(serde_json::json!({ "train": { "epochs": 300, "auto_rule": 2 } }));
        assert_eq!(recorded_auto_rule(&dir), Some(2));
        let mut c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        let base = || train_command(&c, "D:/coco", DatasetFormat::Coco, None);
        let cmd = with_auto_rule(base(), &c, Some(1));
        assert_eq!(cmd.cli, "folo train --data D:/coco --format coco --set auto_rule=1");
        assert_eq!(cmd.argv[cmd.argv.len() - 2..], ["--set", "auto_rule=1"]);
        let current = folo_vision::TrainConfig::default().auto_rule;
        assert_eq!(with_auto_rule(base(), &c, Some(current)).cli, base().cli);
        assert_eq!(with_auto_rule(base(), &c, None).cli, base().cli);
        c.optimizer = Optimizer::Sgd;
        let sgd = train_command(&c, "D:/coco", DatasetFormat::Coco, None);
        assert_eq!(with_auto_rule(sgd.clone(), &c, Some(1)).cli, sgd.cli, "an explicit optimizer");
    }

    /// A run made before `warmup_floor` (100 iterations then) repeats with `--set warmup_floor=100`; a run with
    /// the current default, or that never started, gets no `--set`.
    #[test]
    fn runs_repeat_with_their_warmup_floor() {
        use folo_test_support::TestDir;
        let dir = TestDir::new("cli-warmup-floor");
        let write = |v: serde_json::Value| std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        assert_eq!(recorded_warmup_floor(&dir), None, "no config.json");
        write(serde_json::json!({ "train": { "epochs": 100 } }));
        assert_eq!(recorded_warmup_floor(&dir), Some(100), "written before the key");
        let c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        let base = || train_command(&c, "D:/bccd", DatasetFormat::Yolo, None);
        let cmd = with_warmup_floor(base(), Some(100));
        assert_eq!(cmd.argv[cmd.argv.len() - 2..], ["--set", "warmup_floor=100"]);
        let current = folo_vision::TrainConfig::default().warmup_floor;
        assert_eq!(with_warmup_floor(base(), Some(current)).cli, base().cli);
        assert_eq!(with_warmup_floor(base(), None).cli, base().cli);
    }

    /// A fine-tune made before `ema_run_fraction` (the default window then) repeats with `--set ema_run_fraction=0`;
    /// a fine-tune with the current default, a run from scratch and a run that never started get no `--set`.
    #[test]
    fn fine_tunes_repeat_with_their_ema_window() {
        use folo_test_support::TestDir;
        let dir = TestDir::new("cli-ema-window");
        let write = |v: serde_json::Value| std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        assert_eq!(recorded_ema_run_fraction(&dir), None, "no config.json");
        write(serde_json::json!({ "train": { "epochs": 100, "transfer": true } }));
        assert_eq!(recorded_ema_run_fraction(&dir), Some(0.0), "written before the key");
        write(serde_json::json!({ "train": { "ema_run_fraction": 0.1 } }));
        assert_eq!(recorded_ema_run_fraction(&dir), Some(0.1));
        let c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        let base = || train_command(&c, "D:/bccd", DatasetFormat::Yolo, None);
        let cmd = with_ema_run_fraction(base(), Some(0.0), Some(true));
        assert_eq!(cmd.argv[cmd.argv.len() - 2..], ["--set", "ema_run_fraction=0"]);
        let current = folo_vision::TrainConfig::default().ema_run_fraction;
        assert_eq!(with_ema_run_fraction(base(), Some(current), Some(true)).cli, base().cli);
        assert_eq!(with_ema_run_fraction(base(), Some(0.0), Some(false)).cli, base().cli, "from scratch");
        assert_eq!(with_ema_run_fraction(base(), None, None).cli, base().cli);
    }

    /// A run folder without `config.json` (a cloud run's) reads the recipe from `weights/last.folo`, else
    /// `weights/best.folo`; a `config.json` wins over both.
    #[test]
    fn recipe_falls_back_to_the_checkpoint() {
        use folo_test_support::TestDir;
        let dir = TestDir::new("cli-recipe-checkpoint");
        std::fs::create_dir_all(dir.join("weights")).unwrap();
        let checkpoint = |file: &str, train: serde_json::Value| {
            let meta = std::collections::BTreeMap::from([("folo.train".to_string(), train.to_string())]);
            let t = folo_core::Tensor::from_vec(vec![0f32], [1]).unwrap();
            let path = dir.join("weights").join(file);
            folo_nn::io::safetensors::write_file(&path, Some(&meta), &[("w".into(), t)]).unwrap();
        };
        checkpoint("best.folo", serde_json::json!({ "auto_rule": 2 }));
        assert_eq!((recorded_auto_rule(&dir), recorded_warmup_floor(&dir)), (Some(2), Some(100)));
        checkpoint("last.folo", serde_json::json!({ "auto_rule": 4, "warmup_floor": 0 }));
        assert_eq!((recorded_auto_rule(&dir), recorded_warmup_floor(&dir)), (Some(4), Some(0)));
        let config = serde_json::json!({ "train": { "auto_rule": 3, "warmup_floor": 50 } });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        assert_eq!((recorded_auto_rule(&dir), recorded_warmup_floor(&dir)), (Some(3), Some(50)));
    }

    /// A run whose optimizer this build does not know gets no command (one without `--optimizer` would train
    /// `auto`): `invalid_config`, naming the unknown optimizer; known ones pass.
    #[test]
    fn unknown_optimizer_has_no_command() {
        let mut c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        for o in [Optimizer::Auto, Optimizer::Sgd, Optimizer::Adamw, Optimizer::Musgd] {
            c.optimizer = o;
            assert!(check_optimizer(&c).is_ok(), "{o:?}");
        }
        c.optimizer = serde_json::from_value(serde_json::json!("lion")).unwrap();
        let e = check_optimizer(&c).unwrap_err().body;
        assert_eq!(e.code, "invalid_config");
        assert!(e.message.contains("optimizer is unknown"), "{}", e.message);
    }

    /// A run that distilled from a teacher (`distill` in its config) repeats with `--teacher` and the settings that
    /// differ from the starting values; a run without one gets nothing.
    #[test]
    fn distillation_runs_repeat_with_their_teacher() {
        use folo_test_support::TestDir;
        let dir = TestDir::new("cli-distill");
        let c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        let base = || train_command(&c, "D:/coco", DatasetFormat::Coco, None);
        assert_eq!(with_distill(base(), recorded_distill(&dir)).cli, base().cli, "no config.json");
        let write = |v: serde_json::Value| std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        write(serde_json::json!({ "train": { "epochs": 3 } }));
        assert_eq!(recorded_distill(&dir), None);
        let distill = serde_json::json!({
            "teacher": "folo-s-coco", "cls_gain": 0.5, "dfl_gain": 1.5, "temperature": 2.0
        });
        write(serde_json::json!({ "train": { "distill": distill } }));
        let cmd = with_distill(base(), recorded_distill(&dir));
        assert_eq!(
            cmd.cli,
            "folo train --data D:/coco --format coco --teacher folo-s-coco --set distill.temperature=2"
        );
    }

    #[test]
    fn quoting_is_powershells() {
        assert_eq!(quote("D:/a-b_c.folo"), "D:/a-b_c.folo");
        assert_eq!(quote("a b"), "'a b'");
        assert_eq!(quote("0,0.5"), "'0,0.5'", "a bare comma list is an array in PowerShell");
        assert_eq!(quote("O'Brien $HOME"), "'O''Brien $HOME'");
        assert_eq!(quote("it\u{2019}s"), "'it\u{2019}\u{2019}s'");
        assert_eq!(quote(""), "''");
    }
}
