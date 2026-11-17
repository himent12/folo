//! The `folo` command-line tool. See `docs/CLI.md`.
//!
//! This file parses the command line, dispatches each subcommand (`run`), turns on the shared caches and resolves
//! the data dir. The command tree is `folo_cli::cli` (src/cli.rs). The modules:
//!
//! - `serve`: `folo serve`, plus the tracing setup and data-dir lock errors `mcp` and `doctor` share.
//! - `mcp`: `folo mcp` (the folo-mcp server on stdio).
//! - `doctor`: `folo doctor`.
//! - `describe`: `folo describe` (the clap tree as JSON).
//! - `train`: `folo train` and `folo stop` (folder: config, progress, resume, stop).
//! - `infer`: `folo val`, `folo predict` and `folo export`; `coco` (`val --coco` / `--save-json`) and `draw`
//!   (`predict --save` images) serve it.
//! - `quantize`: `folo quantize` (int8 calibration stored with the weights).
//! - `dataset`: `folo dataset check | stats | synth | convert`.
//! - `weights`: `folo weights list | pull` and published-name resolution (folder: download).
//! - `cloud`: `folo cloud ...` through `folo_server::cloud`.
//! - `engine`: opening datasets, registering the GPU backend, engine errors as CLI errors.
//! - `output`: `--json` records, the error envelope and the exit codes.
//! - `serve_lock`: one owner per data dir (`serve.lock`, `serve.json`).
//!
//! `folo bench` answers `not_implemented` (exit 4).

mod cloud;
mod coco;
mod dataset;
mod describe;
mod doctor;
mod draw;
mod engine;
mod infer;
mod mcp;
mod output;
mod quantize;
mod serve;
mod serve_lock;
mod train;
mod weights;

use std::path::PathBuf;

use clap::CommandFactory;

// The command tree is the library's (src/lib.rs), so tests parse commands with it.
use folo_cli::cli;

use cli::{Cli, Command};
use output::{CliError, ExitCode, Out};

fn main() -> std::process::ExitCode {
    let json = std::env::args_os().skip(1).any(|a| a == "--json");
    let out = Out { json };
    let cli = match parse() {
        Ok(cli) => cli,
        Err(e) => return usage_error(out, e),
    };
    // `folo mcp`'s stdout belongs to the protocol, also with --json: its errors go to stderr only.
    let out = Out { json: cli.json && !matches!(cli.command, Command::Mcp(_)) };
    init_caches();
    let result = run(cli, out);
    // The process-wide GPU context is never dropped: write the pipelines it compiled before exiting.
    folo_gpu::save_pipeline_caches();
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            out.error(&e);
            std::process::ExitCode::from(e.exit as u8)
        }
    }
}

/// Parses the command line. `folo train` also learns which flags were given (not defaulted), so `--resume` can
/// refuse a changed setting.
fn parse() -> Result<Cli, clap::Error> {
    use clap::FromArgMatches;
    let matches = Cli::command().try_get_matches()?;
    let mut cli = Cli::from_arg_matches(&matches)?;
    if let (Command::Train(t), Some(("train", m))) = (&mut cli.command, matches.subcommand()) {
        t.explicit = m
            .ids()
            .filter(|id| m.value_source(id.as_str()) == Some(clap::parser::ValueSource::CommandLine))
            .map(|id| id.to_string())
            .collect();
    }
    Ok(cli)
}

/// Help and version go to stdout with exit 0; real parse errors are usage errors (exit 2).
fn usage_error(out: Out, e: clap::Error) -> std::process::ExitCode {
    use clap::error::ErrorKind;
    if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
        let _ = e.print();
        return std::process::ExitCode::SUCCESS;
    }
    let rendered = e.render().to_string();
    // The first paragraph, on one line: a missing argument's names follow "were not provided:" on their own lines.
    let message: Vec<&str> =
        rendered.lines().map(str::trim).take_while(|l| !l.is_empty()).filter(|l| !l.is_empty()).collect();
    let message = match message.join(" ").trim_start_matches("error: ") {
        "" => "invalid usage".to_string(),
        m => m.to_string(),
    };
    let err =
        CliError::new(ExitCode::Usage, "usage", message).hint("run `folo --help` or `folo describe --json`");
    if out.json {
        out.record(&err.to_json());
    }
    eprint!("{rendered}");
    std::process::ExitCode::from(ExitCode::Usage as u8)
}

fn run(cli: Cli, out: Out) -> Result<(), CliError> {
    match cli.command {
        Command::Serve(args) => serve::serve(args, out),
        Command::Describe(args) => {
            let surface = describe::describe(&Cli::command(), &args.command, !args.no_mcp_tools)?;
            if out.json {
                out.record(&surface);
            } else {
                println!("{}", serde_json::to_string_pretty(&surface).expect("JSON value serializes"));
            }
            Ok(())
        }
        Command::Doctor(args) => doctor::doctor(args, out),
        Command::Train(args) => train::train(args, out),
        Command::Stop(args) => train::request_stop(&args.run_dir, out),
        Command::Val(args) => infer::val(args, out),
        Command::Predict(args) => infer::predict(args, out),
        Command::Export(args) => infer::export(args, out),
        Command::Quantize(args) => quantize::quantize(args, out),
        Command::Bench(_) => Err(not_implemented("bench", "the engine")),
        Command::Dataset(cmd) => dataset::run(cmd, out),
        Command::Weights(cmd) => weights::run(cmd, out),
        Command::Cloud(cmd) => cloud::run(cmd, out),
        Command::Mcp(args) => mcp::mcp(args),
    }
}

fn not_implemented(cmd: &str, waits_on: &str) -> CliError {
    CliError::new(ExitCode::NotImplemented, "not_implemented", format!("`folo {cmd}` is not implemented yet"))
        .hint(format!(
            "it waits on {waits_on}; see docs/ROADMAP.md. `folo describe --json` shows its planned arguments, and \
             `folo serve --mock` serves mock datasets, runs and predictions"
        ))
}

/// Turns on the on-disk caches every command shares: compiled GPU pipelines (`gpu/`) and dataset scans (`scan/`)
/// under `$FOLO_CACHE_DIR`, else `<FOLO_HOME or ~/.folo>/cache`. `FOLO_NO_CACHE=1` leaves them off (docs/CLI.md,
/// Caches).
fn init_caches() {
    if std::env::var_os("FOLO_NO_CACHE").is_some_and(|v| !v.is_empty() && v != "0") {
        return;
    }
    let dir = match std::env::var_os("FOLO_CACHE_DIR") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => folo_api::weights::home_dir().join("cache"),
    };
    folo_gpu::set_pipeline_cache_dir(Some(dir.join("gpu")));
    folo_data::set_scan_cache_dir(Some(dir.join("scan")));
}

/// `$FOLO_HOME`, else `~/.folo`, else `./.folo`.
pub(crate) fn default_data_dir() -> PathBuf {
    folo_api::weights::home_dir()
}

/// `--data-dir` (else [`default_data_dir`]), refused when it is something other than a folder: the commands create
/// it when it does not exist, but a file there failed every write later, as `internal`.
pub(crate) fn data_dir(given: Option<PathBuf>) -> Result<PathBuf, CliError> {
    let dir = given.unwrap_or_else(default_data_dir);
    if dir.exists() && !dir.is_dir() {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--data-dir {} is not a folder", folo_api::json::path(&dir)),
        )
        .hint("pass a folder (it is created when it does not exist)"));
    }
    // Absolute, so what a server reports (`/api/system`, `serve.json`) means the same from any working directory.
    Ok(std::path::absolute(&dir).unwrap_or(dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    #[test]
    fn clap_tree_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn train_defaults_match_the_api() {
        let cli = Cli::try_parse_from(["folo", "train", "--data", "d"]).unwrap();
        let Command::Train(t) = cli.command else { panic!() };
        let c: folo_server::api::TrainConfig = serde_json::from_str(r#"{"dataset_id":"d"}"#).unwrap();
        assert_eq!(t.model, c.arch.as_str());
        assert_eq!((t.epochs, t.batch, t.imgsz, t.seed), (c.epochs, c.batch, c.imgsz, c.seed));
        assert_eq!((t.device.as_str(), t.lr0), (c.device.as_str(), c.lr0));
        let a = c.augment;
        assert_eq!(
            (t.mosaic, t.mixup, t.fliplr, t.scale, t.translate),
            (a.mosaic, a.mixup, a.fliplr, a.scale, a.translate)
        );
        assert_eq!((t.hsv.as_slice(), t.close_mosaic), (a.hsv.as_slice(), a.close_mosaic));
    }

    /// Help strings that state a computed default state the code's: `--workers` is
    /// `folo_vision::train::default_workers` (at most 16, not 8), and `export --output` names every default file.
    #[test]
    fn help_states_the_computed_defaults() {
        let d = describe::describe(&Cli::command(), &[], false).unwrap();
        let help = |cmd: &str, arg: &str| {
            let c = d["commands"].as_array().unwrap().iter().find(|c| c["name"] == cmd).unwrap();
            let a = c["args"].as_array().unwrap().iter().find(|a| a["name"] == arg).unwrap();
            a["help"].as_str().unwrap().to_string()
        };
        for cmd in ["train", "val"] {
            assert!(help(cmd, "workers").contains("min(cores, 16)"), "{cmd}: {}", help(cmd, "workers"));
        }
        let output = help("export", "output");
        for name in
            ["-f16.folo", "-fused.folo", "-weights.folo", ".onnx", "-decoded.onnx", "-decoded-f16.onnx"]
        {
            assert!(output.contains(name), "{name}: {output}");
        }
    }

    #[test]
    fn describe_covers_every_command() {
        let d = describe::describe(&Cli::command(), &[], true).unwrap();
        let names: Vec<&str> =
            d["commands"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap()).collect();
        for want in [
            "serve", "describe", "doctor", "train", "stop", "val", "predict", "export", "quantize", "bench",
            "dataset", "mcp",
        ] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        assert_eq!(d["global_args"][0]["long"], "--json");
        let serve = d["commands"].as_array().unwrap().iter().find(|c| c["name"] == "serve").unwrap();
        let port = serve["args"].as_array().unwrap().iter().find(|a| a["name"] == "port").unwrap();
        assert_eq!(port["type"], "integer");
        assert_eq!(port["default"], "7878");
        let mock = serve["args"].as_array().unwrap().iter().find(|a| a["name"] == "mock").unwrap();
        assert_eq!(mock["type"], "boolean");
        let dataset = d["commands"].as_array().unwrap().iter().find(|c| c["name"] == "dataset").unwrap();
        let synth = dataset["subcommands"].as_array().unwrap().iter().find(|c| c["name"] == "synth").unwrap();
        assert_eq!(synth["path"], "folo dataset synth");
        let classes = synth["args"].as_array().unwrap().iter().find(|a| a["name"] == "classes").unwrap();
        assert_eq!(classes["default"], json!(["circle", "square", "triangle"]));
        let train = d["commands"].as_array().unwrap().iter().find(|c| c["name"] == "train").unwrap();
        let opt = train["args"].as_array().unwrap().iter().find(|a| a["name"] == "optimizer").unwrap();
        assert_eq!(opt["type"], "enum");
        // auto, sgd, adamw, musgd.
        assert_eq!(opt["possible_values"].as_array().unwrap().len(), 4);
        assert!(opt["possible_values"].to_string().contains("musgd"), "{}", opt["possible_values"]);
    }
}
