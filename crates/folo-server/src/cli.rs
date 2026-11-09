//! The `folo train` command of a [`TrainConfig`] (`RunEstimate.cli`, `GET /api/runs/{id}/cli`): the flags the CLI
//! maps back to the same training settings, so a run set up in Studio or over the API can be repeated from a
//! terminal on the server's machine. `crates/folo-cli`'s tests parse it with the real `folo` arguments and compare the
//! resulting config with the server's.

use folo_api::{CliCommand, Head, Optimizer, TrainConfig};

/// The command that trains `config` on the dataset at `data` (a path on the server's machine), starting from
/// `weights` (a `.folo` path there, or a published name; `None` trains from scratch).
///
/// Settings equal to the CLI's defaults are left out (they are the API's defaults too). With `weights` the
/// architecture is the weights' (the CLI refuses `--model` with `--weights`), and the head is always given: the CLI
/// defaults to the weights file's head, the API to `o2m`.
pub fn train_command(config: &TrainConfig, data: &str, weights: Option<&str>) -> CliCommand {
    let d: TrainConfig =
        serde_json::from_value(serde_json::json!({ "dataset_id": "" })).expect("TrainConfig has defaults");
    let mut argv: Vec<String> = ["folo", "train", "--data", data].map(String::from).to_vec();
    let mut arg = |flag: &str, value: String| {
        argv.push(format!("--{flag}"));
        argv.push(value);
    };
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

/// `argv` as one command line: arguments with characters a shell treats specially are single-quoted (`'` inside
/// as `'\''`), which POSIX shells and PowerShell both read, unless the argument itself holds a `'`.
pub fn shell_join(argv: &[String]) -> String {
    argv.iter().map(|a| quote(a)).collect::<Vec<_>>().join(" ")
}

fn quote(a: &str) -> String {
    let plain = !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:=,+".contains(c));
    if plain { a.to_string() } else { format!("'{}'", a.replace('\'', "'\\''")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_left_out_and_paths_quoted() {
        let mut c: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
        let cmd = train_command(&c, "D:/data/r6", None);
        assert_eq!(cmd.cli, "folo train --data D:/data/r6");
        c.name = "r6 n".into();
        c.amp = true;
        c.augment.hsv = [0.0, 0.5, 0.25];
        let cmd = train_command(&c, "C:/Users/a b/data", Some("folo-n-coco"));
        assert_eq!(
            cmd.cli,
            "folo train --data 'C:/Users/a b/data' --weights folo-n-coco --name 'r6 n' --head o2m --hsv 0,0.5,0.25 \
             --amp"
        );
        assert_eq!(cmd.argv[3], "C:/Users/a b/data");
        assert_eq!(quote("it's"), r"'it'\''s'");
    }
}
