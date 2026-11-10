//! FOLO Studio's "CLI command" (New run) parses with the real `folo` command tree: every command in
//! `web/src/lib/trainCli.fixtures.json` (written by the web tests from representative run settings) goes through
//! clap's `Cli::try_parse_from`, and the parsed `train` arguments hold the run's values (DOGFOOD-3 S1: a fine-tune
//! must not pass `--model` with `--weights`). Studio shows the command for PowerShell, so it is split by PowerShell's
//! rules.

use clap::Parser;
use serde_json::Value;

use folo_cli::cli::{Cli, Command, HeadArg, OptimizerArg};

/// Whether PowerShell reads `c` as a single quote: `'` and the typographic ones (U+2018 to U+201B).
fn is_quote(c: char) -> bool {
    matches!(c, '\'' | '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}')
}

/// A command line as PowerShell splits it, for the words Studio writes: bare words and single-quoted ones, where a
/// doubled quote stands for one (`'O''Brien'`).
fn split(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if is_quote(c) => {
                in_word = true;
                while let Some(q) = chars.next() {
                    if !is_quote(q) {
                        word.push(q);
                    } else if let Some(n) = chars.next_if(|&n| is_quote(n)) {
                        word.push(n);
                    } else {
                        break;
                    }
                }
            }
            c if c.is_whitespace() => {
                if in_word {
                    out.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            c => {
                in_word = true;
                word.push(c);
            }
        }
    }
    if in_word {
        out.push(word);
    }
    out
}

fn f32_of(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

#[test]
fn studio_train_commands_parse() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/src/lib/trainCli.fixtures.json");
    let fixture: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert!(fixture.len() >= 10, "the fixture lists representative runs");
    for case in &fixture {
        let name = case["name"].as_str().unwrap();
        let command = case["command"].as_str().unwrap();
        let argv: Vec<String> =
            case["argv"].as_array().unwrap().iter().map(|a| a.as_str().unwrap().into()).collect();
        // The shown command is the argv, quoted.
        let words = split(command);
        assert_eq!(words[0], "folo", "{name}");
        assert_eq!(words[1..], argv[..], "{name}: {command}");

        let cli = Cli::try_parse_from(std::iter::once("folo".to_string()).chain(argv))
            .unwrap_or_else(|e| panic!("{name}: `{command}` does not parse: {e}"));
        let Command::Train(t) = cli.command else { panic!("{name}: not `train`") };
        let x = &case["expect"];
        assert_eq!(t.data.as_deref(), x["data"].as_str(), "{name}");
        match (x["weights"].as_str(), x["model"].as_str()) {
            (Some(w), _) => assert_eq!(t.weights.as_deref(), Some(w), "{name}"),
            (None, Some(m)) => {
                assert_eq!(t.model, m, "{name}");
                assert!(t.weights.is_none(), "{name}");
            }
            (None, None) => panic!("{name}: neither model nor weights"),
        }
        assert_eq!(t.epochs as u64, x["epochs"].as_u64().unwrap(), "{name}");
        assert_eq!(t.batch as u64, x["batch"].as_u64().unwrap(), "{name}");
        assert_eq!(t.imgsz as u64, x["imgsz"].as_u64().unwrap(), "{name}");
        assert_eq!(t.device, x["device"].as_str().unwrap(), "{name}");
        let optimizer = match t.optimizer {
            OptimizerArg::Auto => "auto",
            OptimizerArg::Sgd => "sgd",
            OptimizerArg::Adamw => "adamw",
            OptimizerArg::Musgd => "musgd",
        };
        assert_eq!(optimizer, x["optimizer"].as_str().unwrap(), "{name}");
        assert_eq!(t.lr0, f32_of(&x["lr0"]), "{name}");
        assert_eq!(t.amp, x["amp"].as_bool().unwrap(), "{name}");
        let head = t.head.map(|h| match h {
            HeadArg::O2m => "o2m",
            HeadArg::O2o => "o2o",
        });
        assert_eq!(head, x["head"].as_str(), "{name}");
        assert_eq!(t.mosaic, f32_of(&x["mosaic"]), "{name}");
        assert_eq!(t.mixup, f32_of(&x["mixup"]), "{name}");
        let hsv: Vec<f32> = x["hsv"].as_array().unwrap().iter().map(f32_of).collect();
        assert_eq!(t.hsv, hsv, "{name}");
        assert_eq!(t.fliplr, f32_of(&x["fliplr"]), "{name}");
        assert_eq!(t.scale, f32_of(&x["scale"]), "{name}");
        assert_eq!(t.translate, f32_of(&x["translate"]), "{name}");
        assert_eq!(t.close_mosaic as u64, x["close_mosaic"].as_u64().unwrap(), "{name}");
        assert_eq!(t.seed, x["seed"].as_u64().unwrap(), "{name}");
        assert_eq!(t.name.as_deref(), x["name"].as_str().filter(|n| !n.is_empty()), "{name}");
    }
}

/// The server's command (`RunEstimate.cli`, `GET /api/runs/{id}/cli`) splits by the same PowerShell rules into its
/// `argv`, including names with straight and typographic quotes, and parses.
#[test]
fn server_commands_split_like_studios() {
    let mut c: folo_api::TrainConfig =
        serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1" })).unwrap();
    c.augment.hsv = [0.0, 0.5, 0.3];
    for name in ["r6 n", "O'Brien's run", "it\u{2019}s", "--amp", "plain"] {
        c.name = name.into();
        for weights in [None, Some("folo-n-coco"), Some("C:/Users/Ann O'Neil/best.folo")] {
            let cmd = folo_server::cli::train_command(
                &c,
                "C:/data sets/r6",
                folo_api::DatasetFormat::Coco,
                weights,
            );
            assert_eq!(split(&cmd.cli), cmd.argv, "{}", cmd.cli);
            let cli = Cli::try_parse_from(&cmd.argv).unwrap_or_else(|e| panic!("`{}`: {e}", cmd.cli));
            let Command::Train(t) = cli.command else { panic!("not train") };
            assert_eq!((t.name.as_deref(), t.weights.as_deref()), (Some(name), weights), "{}", cmd.cli);
            assert_eq!(t.hsv, [0.0, 0.5, 0.3], "{}", cmd.cli);
        }
    }
}
