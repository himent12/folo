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
//! - `label`: `folo label ...`, agent labeling on folo-mcp's labeling operations (folder: print).
//! - `bench`: `folo bench ...`, vision benchmarks on folo-mcp's benchmark operations (folder: print).
//! - `engine`: opening datasets, registering the GPU backend, engine errors as CLI errors.
//! - `output`: `--json` records, the error envelope and the exit codes.
//! - `serve_lock`: one owner per data dir (`serve.lock`, `serve.json`).

mod bench;
mod cloud;
mod coco;
mod dataset;
mod describe;
mod doctor;
mod draw;
mod engine;
mod infer;
mod label;
mod mcp;
mod output;
mod output_file;
mod quantize;
mod serve;
mod serve_lock;
mod train;
mod weights;

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use clap::CommandFactory;

// The command tree is the library's (src/lib.rs), so tests parse commands with it.
use folo_cli::cli;

use cli::{Cli, Command};
use output::{CliError, ExitCode, Out};

fn main() -> std::process::ExitCode {
    let cli = match parse() {
        Ok(cli) => cli,
        Err(e) => {
            // `folo mcp`'s stdout is the protocol's also when its arguments do not parse.
            let args: Vec<_> = std::env::args_os().collect();
            let json = args.iter().skip(1).any(|a| a == "--json") && !names_mcp(&args);
            return usage_error(Out { json }, &args, e);
        }
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

/// Whether a command line that did not parse names `folo mcp`: its first subcommand token is exactly `mcp`
/// ([`subcommand_token`]). Not clap's own recovery (`ignore_errors`), which stops at the first root-level error.
fn names_mcp(args: &[OsString]) -> bool {
    subcommand_token(&Cli::command(), args).is_some_and(|t| t == "mcp")
}

/// The argument that names the subcommand (`args[0]` is the program), whatever errors come before it: the first that
/// is neither a root flag nor a root option's value. A root option (as `root` defines it) that takes a value and is
/// not given one in its token (`--opt=v`, `-ov`) takes the next argument unless that starts with `-`, as clap does;
/// anything else starting with `-` (a repeated, unknown or malformed flag) is skipped alone. `--` ends the scan with
/// no subcommand: what follows it is a positional, and the root has none. A lone `-` is a value, as clap takes it. An
/// argument that is not Unicode is read lossily: a flag still starts with `-` and matches no root flag, so it is
/// skipped as an unknown one; anything else is a subcommand token that is not `mcp`.
fn subcommand_token<'a>(root: &clap::Command, args: &'a [OsString]) -> Option<&'a OsStr> {
    let rest = args.get(1..)?;
    next_subcommand_index(root, rest).map(|i| rest[i].as_os_str())
}

fn next_subcommand_index(cmd: &clap::Command, args: &[OsString]) -> Option<usize> {
    let takes_value = |arg: Option<&clap::Arg>| arg.is_some_and(|a| a.get_action().takes_values());
    let mut i = 0;
    while i < args.len() {
        let s = args[i].to_string_lossy();
        let wants_next = if s == "--" {
            return None;
        } else if let Some(long) = s.strip_prefix("--") {
            !long.contains('=') && takes_value(cmd.get_arguments().find(|a| a.get_long() == Some(long)))
        } else if let Some(shorts) = s.strip_prefix('-').filter(|s| !s.is_empty()) {
            // A cluster (`-ab`): the first short that takes a value takes the rest of the token, if any.
            let mut chars = shorts.chars();
            loop {
                match chars.next() {
                    None => break false,
                    Some(c) if takes_value(cmd.get_arguments().find(|a| a.get_short() == Some(c))) => {
                        break chars.as_str().is_empty();
                    }
                    Some(_) => {}
                }
            }
        } else {
            return Some(i);
        };
        i += 1;
        if wants_next
            && args.get(i).is_some_and(|n| n.as_os_str() == "-" || !n.to_string_lossy().starts_with('-'))
        {
            i += 1;
        }
    }
    None
}

/// Matched subcommand path (e.g. `["dataset", "stats"]`) extracted from clap's `Usage:` line.
fn command_path_from_usage(root: &clap::Command, usage: &str) -> Vec<String> {
    let line = usage.lines().next().unwrap_or("");
    let rest = line.trim().strip_prefix("Usage:").unwrap_or(line).trim();
    let mut cmd = root;
    let mut path = Vec::new();
    for token in rest.split_whitespace().skip(1) {
        let Some(sub) = cmd.get_subcommands().filter(|c| !c.is_hide_set()).find(|c| c.get_name() == token)
        else {
            break;
        };
        path.push(sub.get_name().to_string());
        cmd = sub;
    }
    path
}

/// Matched subcommand path walked from raw `args` when clap's error omits a `Usage:` line (e.g. invalid enum value).
fn subcommand_path(root: &clap::Command, args: &[OsString]) -> Vec<String> {
    let mut cmd = root;
    let mut rest = args.get(1..).unwrap_or(&[]);
    let mut path = Vec::new();
    while let Some(idx) = next_subcommand_index(cmd, rest) {
        let Some(name) = rest[idx].to_str() else { break };
        let Some(sub) = cmd.get_subcommands().filter(|c| !c.is_hide_set()).find(|c| c.get_name() == name)
        else {
            break;
        };
        path.push(sub.get_name().to_string());
        cmd = sub;
        rest = &rest[idx + 1..];
    }
    path
}

/// Builds the structured `--json` usage error for a clap parse error, preserving clap's argument /
/// subcommand / value suggestions and scoping the help hint to the matched subcommand.
fn usage_cli_error(root: &clap::Command, args: &[OsString], e: &clap::Error) -> CliError {
    use clap::error::{ContextKind, ContextValue};
    let rendered = e.render().to_string();
    // The first paragraph, on one line: a missing argument's names follow "were not provided:" on their own lines.
    let message: Vec<&str> =
        rendered.lines().map(str::trim).take_while(|l| !l.is_empty()).filter(|l| !l.is_empty()).collect();
    let message = match message.join(" ").trim_start_matches("error: ") {
        "" => "invalid usage".to_string(),
        m => m.to_string(),
    };

    let mut tips = Vec::new();
    let mut usage_line: Option<String> = None;
    for (kind, value) in e.context() {
        match (kind, value) {
            (ContextKind::SuggestedArg, ContextValue::String(s))
            | (ContextKind::SuggestedSubcommand, ContextValue::String(s))
            | (ContextKind::SuggestedValue, ContextValue::String(s)) => {
                let tip = format!("did you mean `{s}`?");
                if !tips.contains(&tip) {
                    tips.push(tip);
                }
            }
            (ContextKind::SuggestedArg, ContextValue::Strings(vs))
            | (ContextKind::SuggestedSubcommand, ContextValue::Strings(vs))
            | (ContextKind::SuggestedValue, ContextValue::Strings(vs))
                if !vs.is_empty() =>
            {
                let formatted: Vec<String> = vs.iter().map(|s| format!("`{s}`")).collect();
                let tip = format!("did you mean {}?", formatted.join(", "));
                if !tips.contains(&tip) {
                    tips.push(tip);
                }
            }
            (ContextKind::Suggested, ContextValue::StyledStrs(vs)) => {
                for s in vs {
                    let text = s.to_string();
                    let text = text.trim();
                    if !text.is_empty() && !tips.iter().any(|t| t == text) {
                        tips.push(text.to_string());
                    }
                }
            }
            (ContextKind::Usage, ContextValue::StyledStr(u)) => {
                usage_line = Some(u.to_string());
            }
            _ => {}
        }
    }

    let path = match usage_line.as_deref() {
        Some(u) => command_path_from_usage(root, u),
        None => subcommand_path(root, args),
    };
    let help = if path.is_empty() {
        "run `folo --help` or `folo describe --json`".to_string()
    } else {
        let joined = path.join(" ");
        format!("run `folo {joined} --help` or `folo describe {joined} --json`")
    };
    let hint = if tips.is_empty() { help } else { format!("{}; {help}", tips.join("; ")) };
    CliError::new(ExitCode::Usage, "usage", message).hint(hint)
}

/// Help and version go to stdout with exit 0; real parse errors are usage errors (exit 2).
fn usage_error(out: Out, args: &[OsString], e: clap::Error) -> std::process::ExitCode {
    use clap::error::ErrorKind;
    if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
        let _ = e.print();
        return std::process::ExitCode::SUCCESS;
    }
    let rendered = e.render().to_string();
    if out.json {
        let err = usage_cli_error(&Cli::command(), args, &e);
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
        Command::Bench(cmd) => bench::run(cmd, out),
        Command::Dataset(cmd) => dataset::run(cmd, out),
        Command::Weights(cmd) => weights::run(cmd, out),
        Command::Cloud(cmd) => cloud::run(cmd, out),
        Command::Label(cmd) => label::run(cmd, out),
        Command::Mcp(args) => mcp::mcp(args),
    }
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

    /// `names_mcp` over raw command lines: the first subcommand token decides, whatever error comes before it; a
    /// line whose first subcommand token is not exactly `mcp` keeps the JSON usage record.
    #[test]
    fn names_mcp_reads_the_first_subcommand_token() {
        let names = |root: &clap::Command, line: &str| {
            let args: Vec<OsString> = line.split(' ').map(OsString::from).collect();
            subcommand_token(root, &args).is_some_and(|t| t == "mcp")
        };
        for (line, want) in [
            ("folo mcp --json --extra", true),
            ("folo --json mcp --extra", true),
            // Root-level errors before `mcp`: a repeated flag, a value for a flag, an unknown flag.
            ("folo --json --json mcp", true),
            ("folo --json=true mcp --json", true),
            ("folo --bogus mcp --json", true),
            ("folo -x mcp --json", true),
            // `--` ends the options: `mcp` after it is a stray positional (clap's "unexpected argument").
            ("folo --json -- mcp", false),
            ("folo --json", false),
            ("folo mcpp --json", false),
            ("folo predict mcp --json --bad", false),
            ("folo predict --conf mcp --json", false),
            ("folo - mcp --json", false),
        ] {
            assert_eq!(names(&Cli::command(), line), want, "{line}");
        }
        // A root option that takes a value (the root has none today) is read from clap's definition of it.
        let root = Cli::command().arg(clap::Arg::new("dir").long("dir").short('d'));
        for (line, want) in [
            ("folo --dir x mcp --json", true),
            ("folo --dir=x mcp --json", true),
            ("folo -dx mcp --json", true),
            ("folo -d x mcp --json", true),
            // Missing its value: clap does not take a flag as one.
            ("folo --dir --json mcp", true),
            ("folo -d --json mcp", true),
            // A value equal to `mcp` is a value.
            ("folo --dir mcp --json", false),
            ("folo -d mcp --json", false),
            // A lone `-` is a value.
            ("folo --dir - mcp --json", true),
        ] {
            assert_eq!(names(&root, line), want, "{line}");
        }
        // An argument that is not Unicode: a flag is skipped as an unknown one, anything else names a subcommand.
        let not_unicode = |prefix: &str| -> OsString {
            #[cfg(windows)]
            {
                use std::os::windows::ffi::OsStringExt;
                OsString::from_wide(&prefix.encode_utf16().chain([0xD800]).collect::<Vec<_>>())
            }
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                OsString::from_vec([prefix.as_bytes(), &[0xFF]].concat())
            }
        };
        for (first, want) in [("--x", true), ("-x", true), ("x", false)] {
            let args = ["folo".into(), not_unicode(first), "mcp".into(), "--json".into()];
            assert_eq!(subcommand_token(&Cli::command(), &args).is_some_and(|t| t == "mcp"), want, "{first}");
        }
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

    #[test]
    fn usage_cli_error_preserves_suggestions_and_scopes_help() {
        let parse_err = |argv: &[&str]| {
            let args: Vec<OsString> = argv.iter().map(OsString::from).collect();
            let err = Cli::command().try_get_matches_from(&args).unwrap_err();
            usage_cli_error(&Cli::command(), &args, &err)
        };

        // Flag typo on a subcommand: keeps clap's `--batch` suggestion and scopes help to `folo train`.
        let e = parse_err(&["folo", "train", "--batc", "2", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert!(hint.contains("`--batch`"), "{hint}");
        assert!(hint.contains("`folo train --help`"), "{hint}");
        assert!(hint.contains("`folo describe train --json`"), "{hint}");

        // Nested subcommand typo: keeps clap's `stats` suggestion and scopes help to `folo dataset`.
        let e = parse_err(&["folo", "dataset", "stat", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert!(hint.contains("`stats`"), "{hint}");
        assert!(hint.contains("`folo dataset --help`"), "{hint}");
        assert!(hint.contains("`folo describe dataset --json`"), "{hint}");

        // Invalid enum value (no `Usage:` line from clap): falls back to raw args for subcommand path.
        let e = parse_err(&["folo", "train", "--data", "d", "--optimizer", "adam", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert!(hint.contains("`adamw`"), "{hint}");
        assert!(hint.contains("`folo train --help`"), "{hint}");

        // Nested subcommand missing required arg: scoped to `folo dataset stats`.
        let e = parse_err(&["folo", "dataset", "stats", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert_eq!(hint, "run `folo dataset stats --help` or `folo describe dataset stats --json`");

        // Top-level unknown subcommand typo: keeps `train` suggestion and top-level help.
        let e = parse_err(&["folo", "trani", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert!(hint.contains("`train`"), "{hint}");
        assert!(hint.contains("`folo --help`"), "{hint}");

        // Root-level flag errors before a valid subcommand stay scoped to root, not the subcommand.
        let e = parse_err(&["folo", "--bogus", "dataset", "stats", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert_eq!(hint, "run `folo --help` or `folo describe --json`");

        let e = parse_err(&["folo", "--jso", "train", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert!(hint.contains("`--json`"), "{hint}");
        assert!(hint.contains("`folo --help`"), "{hint}");
        assert!(!hint.contains("folo train"), "{hint}");

        // Command group missing its subcommand: scoped to that command group.
        let e = parse_err(&["folo", "dataset", "--json"]);
        let hint = e.hint.as_deref().unwrap();
        assert_eq!(hint, "run `folo dataset --help` or `folo describe dataset --json`");
    }
}
