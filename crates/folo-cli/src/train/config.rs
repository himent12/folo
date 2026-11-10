//! The trainer config of a new run: built from the flags, then changed by `--set KEY=VALUE`.

use folo_data::AugmentConfig;
use folo_vision::TrainConfig;
use folo_vision::train::OptimizerChoice;
use serde_json::json;

use crate::cli::{OptimizerArg, TrainArgs};
use crate::output::{CliError, ExitCode};

/// The trainer's config of `folo train` with `args`, the run `name`, the `dataset` path, the architecture `arch`
/// (the weights' when fine-tuning) and the head a run without `--head` gets (the weights file's, else o2m).
pub(super) fn config_of(
    args: &TrainArgs,
    name: String,
    dataset: String,
    arch: String,
    default_head: folo_vision::model::HeadMode,
) -> TrainConfig {
    TrainConfig {
        name,
        dataset,
        data_format: crate::dataset::format_of(args.format),
        arch,
        epochs: args.epochs,
        batch: args.batch as usize,
        imgsz: args.imgsz,
        device: args.device.clone(),
        optimizer: match args.optimizer {
            OptimizerArg::Auto => OptimizerChoice::Auto,
            OptimizerArg::Sgd => OptimizerChoice::Sgd,
            OptimizerArg::Adamw => OptimizerChoice::Adamw,
            OptimizerArg::Musgd => OptimizerChoice::Musgd,
        },
        lr0: folo_vision::train::f32_decimal(args.lr0),
        lrf: args.lrf,
        cos_lr: args.cos_lr,
        amp: args.amp,
        head: args.head.map_or(default_head, Into::into),
        warmup_epochs: args.warmup_epochs,
        patience: args.patience,
        seed: args.seed,
        workers: args.workers.unwrap_or_else(folo_vision::train::default_workers),
        augment: AugmentConfig {
            mosaic: args.mosaic,
            mixup: args.mixup,
            hsv: [args.hsv[0], args.hsv[1], args.hsv[2]],
            fliplr: args.fliplr,
            scale: args.scale,
            translate: args.translate,
            close_mosaic: args.close_mosaic,
            ..AugmentConfig::default()
        },
        ..TrainConfig::default()
    }
}

/// Keys of the `train` object whose flags do more than set the key (pick the run folder, open the dataset, create
/// the device, ask for tensor cores), with that flag: `--set` refuses them.
const FLAG_ONLY_KEYS: [(&str, &str); 6] = [
    ("name", "--name"),
    ("dataset", "--data"),
    ("data_format", "--format"),
    ("arch", "--model"),
    ("device", "--device"),
    ("amp", "--amp"),
];

/// `--set KEY=VALUE`: `cfg` with each dotted key of its JSON form replaced by the value (JSON when it parses as
/// JSON, else a string; an object is merged into the key's object). A key the config does not have (at any depth), a
/// key of [`FLAG_ONLY_KEYS`], or a value its field cannot take, is `invalid_config`.
pub(super) fn apply_sets(cfg: TrainConfig, sets: &[String]) -> Result<TrainConfig, CliError> {
    if sets.is_empty() {
        return Ok(cfg);
    }
    let bad = |msg: String| {
        CliError::new(ExitCode::Usage, "invalid_config", msg)
            .hint("--set takes KEY=VALUE with a key of the run's config.json `train` object, e.g. --set momentum=0.9")
    };
    let mut v = serde_json::to_value(&cfg).expect("configs serialize");
    for s in sets {
        let Some((key, raw)) = s.split_once('=') else {
            return Err(bad(format!("--set {s}: expected KEY=VALUE")));
        };
        if let Some((k, flag)) = FLAG_ONLY_KEYS.iter().find(|(k, _)| *k == key.trim()) {
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!("--set {k}: use {flag} (it does more than set the key: --set cannot change `{k}`)"),
            )
            .hint(format!("use {flag} instead of --set {k}=...")));
        }
        let mut slot = &mut v;
        for part in key.trim().split('.') {
            let keys = slot.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "));
            slot = match slot.get_mut(part) {
                Some(next) => next,
                None => {
                    return Err(bad(format!(
                        "--set {key}: no training key `{part}`{}",
                        keys.map_or(String::new(), |k| format!(" (there are: {k})"))
                    )));
                }
            };
        }
        let value = serde_json::from_str(raw.trim()).unwrap_or_else(|_| json!(raw.trim()));
        merge(slot, value).map_err(|path| bad(format!("--set {key}: no training key `{path}`")))?;
    }
    serde_json::from_value(v).map_err(|e| bad(format!("--set: {e}")))
}

/// `value` written into `slot`: an object into an object key by key (keys it leaves out keep their values), anything
/// else replacing it. A key the slot's object does not have is `Err` with its dotted path (the config's structs would
/// drop it silently).
fn merge(slot: &mut serde_json::Value, value: serde_json::Value) -> Result<(), String> {
    match (slot, value) {
        (serde_json::Value::Object(have), serde_json::Value::Object(given)) => {
            for (k, v) in given {
                let next = have.get_mut(&k).ok_or_else(|| k.clone())?;
                merge(next, v).map_err(|path| format!("{k}.{path}"))?;
            }
            Ok(())
        }
        (slot, value) => {
            *slot = value;
            Ok(())
        }
    }
}
