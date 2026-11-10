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
            Optimizer::Auto => "auto",
            Optimizer::Sgd => "sgd",
            Optimizer::Adamw => "adamw",
            Optimizer::Musgd => "musgd",
        };
        arg("optimizer", o.to_string());
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
        let cmd = train_command(&c, "D:/data/r6", DatasetFormat::Auto, None);
        assert_eq!(cmd.cli, "folo train --data D:/data/r6");
        c.name = "r6 n".into();
        c.amp = true;
        c.augment.hsv = [0.0, 0.5, 0.25];
        let cmd = train_command(&c, "C:/Users/a b/data", DatasetFormat::Coco, Some("folo-n-coco"));
        assert_eq!(
            cmd.cli,
            "folo train --data 'C:/Users/a b/data' --format coco --weights folo-n-coco --name 'r6 n' --head o2m \
             --hsv '0,0.5,0.25' --amp"
        );
        assert_eq!(cmd.argv[3], "C:/Users/a b/data");
        // A value that looks like a flag is attached.
        c.name = "--amp".into();
        let cmd = train_command(&c, "d", DatasetFormat::Auto, None);
        assert!(cmd.argv.contains(&"--name=--amp".to_string()), "{:?}", cmd.argv);
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
