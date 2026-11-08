//! The `folo` command-line tool. See `docs/CLI.md`.

mod cli;
mod describe;
mod output;

use std::path::PathBuf;
use std::sync::Arc;

use clap::{CommandFactory, Parser};
use folo_server::{Cors, MockStore, Options, RealStore, Server, Store};
use serde_json::json;

use cli::{Cli, Command, DatasetCommand, DoctorArgs, ServeArgs};
use output::{CliError, ExitCode, Out};

fn main() -> std::process::ExitCode {
    let json = std::env::args_os().skip(1).any(|a| a == "--json");
    let out = Out { json };
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => return usage_error(out, e),
    };
    let out = Out { json: cli.json };
    match run(cli, out) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            out.error(&e);
            std::process::ExitCode::from(e.exit as u8)
        }
    }
}

/// Help and version go to stdout with exit 0; real parse errors are usage errors (exit 2).
fn usage_error(out: Out, e: clap::Error) -> std::process::ExitCode {
    use clap::error::ErrorKind;
    if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
        let _ = e.print();
        return std::process::ExitCode::SUCCESS;
    }
    let rendered = e.render().to_string();
    let message =
        rendered.lines().next().unwrap_or("invalid usage").trim_start_matches("error: ").to_string();
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
        Command::Serve(args) => serve(args, out),
        Command::Describe => {
            let surface = describe::describe(&Cli::command());
            if out.json {
                out.record(&surface);
            } else {
                println!("{}", serde_json::to_string_pretty(&surface).expect("JSON value serializes"));
            }
            Ok(())
        }
        Command::Doctor(args) => doctor(args, out),
        Command::Train(_) => Err(not_implemented("train", "the trainer (folo-vision)")),
        Command::Val(_) => Err(not_implemented("val", "evaluation (folo-vision)")),
        Command::Predict(_) => Err(not_implemented("predict", "the predictor (folo-vision)")),
        Command::Export(_) => Err(not_implemented("export", "weight export (folo-nn / folo-vision)")),
        Command::Bench(_) => Err(not_implemented("bench", "the engine")),
        Command::Dataset(DatasetCommand::Check(_)) => Err(not_implemented("dataset check", "folo-data")),
        Command::Dataset(DatasetCommand::Stats(_)) => Err(not_implemented("dataset stats", "folo-data")),
        Command::Dataset(DatasetCommand::Synth(_)) => Err(not_implemented("dataset synth", "folo-data")),
        Command::Mcp(_) => Err(not_implemented("mcp", "the MCP server")),
    }
}

fn not_implemented(cmd: &str, waits_on: &str) -> CliError {
    CliError::new(ExitCode::NotImplemented, "not_implemented", format!("`folo {cmd}` is not implemented yet"))
        .hint(format!(
            "it waits on {waits_on}; see docs/ROADMAP.md. `folo describe --json` shows its planned arguments, and \
             `folo serve --mock` serves mock datasets, runs and predictions"
        ))
}

/// `$FOLO_HOME`, else `~/.folo`, else `./.folo`.
fn default_data_dir() -> PathBuf {
    if let Some(h) = std::env::var_os("FOLO_HOME") {
        return PathBuf::from(h);
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    home.map_or_else(|| PathBuf::from(".folo"), |h| PathBuf::from(h).join(".folo"))
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Logs always go to stderr so stdout stays clean for `--json` records.
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
}

fn serve(args: ServeArgs, out: Out) -> Result<(), CliError> {
    init_tracing();
    let data_dir = args.data_dir.clone().unwrap_or_else(default_data_dir);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::new(ExitCode::Runtime, "runtime", format!("could not start tokio: {e}")))?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await.map_err(|e| {
            CliError::new(
                ExitCode::Runtime,
                "bind_failed",
                format!("could not bind {}:{}: {e}", args.host, args.port),
            )
            .hint("is another `folo serve` running? pick another --port (0 chooses a free one)")
        })?;
        let addr = listener
            .local_addr()
            .map_err(|e| CliError::new(ExitCode::Runtime, "bind_failed", e.to_string()))?;
        let store: Arc<dyn Store> =
            if args.mock { Arc::new(MockStore::new()) } else { Arc::new(RealStore::open(data_dir.clone())) };
        let web_ready = folo_server::web_dir_ready(&args.web_dir);
        let cors = if args.cors_any { Cors::Any } else { Cors::Loopback };
        let server = Server::new(store, Options { web_dir: Some(args.web_dir.clone()), cors });
        let url = format!("http://{addr}");
        out.human(format!(
            "FOLO {} serving {} on {url}",
            env!("CARGO_PKG_VERSION"),
            if args.mock { "mock data" } else { "the real store" }
        ));
        out.human(format!("  API:    {url}/api/health"));
        if web_ready {
            out.human(format!("  Studio: {url}/  (from {})", args.web_dir.display()));
        } else {
            out.human(format!(
                "  Studio: not built ({} has no index.html); build web/ or use the Vite dev server on :5173",
                args.web_dir.display()
            ));
        }
        if !args.mock {
            out.human(format!("  Data:   {}", data_dir.display()));
        }
        out.human("Press Ctrl+C to stop.");
        out.record(&json!({
            "event": "listening",
            "url": url,
            "api": format!("{url}/api"),
            "mock": args.mock,
            "web_dir": args.web_dir,
            "web_ready": web_ready,
            "data_dir": if args.mock { None } else { Some(&data_dir) },
        }));
        server
            .serve(listener, folo_server::shutdown_signal())
            .await
            .map_err(|e| CliError::new(ExitCode::Runtime, "serve_failed", e.to_string()))?;
        out.record(&json!({ "event": "stopped" }));
        out.human("Stopped.");
        Ok(())
    })
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn doctor(args: DoctorArgs, out: Out) -> Result<(), CliError> {
    let host = folo_server::system::host_info();
    let web_exists = args.web_dir.is_dir();
    let web_ready = folo_server::web_dir_ready(&args.web_dir);
    // TODO(gpu): enumerate wgpu adapters (name, backend, memory, shader-f16) once folo-gpu exists.
    let report = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "os": host.os,
        "arch": host.arch,
        "cpu": { "name": host.cpu_name, "threads": host.cpu_threads },
        "memory": { "total_bytes": host.memory_total_bytes, "available_bytes": host.memory_available_bytes },
        "gpus": [],
        "gpu_detection": "not_implemented",
        "web_dist": { "path": args.web_dir, "exists": web_exists, "index_html": web_ready },
    });
    out.record(&report);
    out.human(format!("folo      {}", env!("CARGO_PKG_VERSION")));
    out.human(format!("os        {} ({})", host.os, host.arch));
    out.human(format!("cpu       {} ({} threads)", host.cpu_name, host.cpu_threads));
    out.human(format!(
        "memory    {} total, {} available",
        gib(host.memory_total_bytes),
        gib(host.memory_available_bytes)
    ));
    out.human("gpu       detection not implemented yet (waits on folo-gpu)");
    let web = if web_ready {
        "built".to_string()
    } else if web_exists {
        "folder exists but has no index.html".to_string()
    } else {
        "not built (run `npm ci && npm run build` in web/)".to_string()
    };
    out.human(format!("web/dist  {} — {web}", args.web_dir.display()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn describe_covers_every_command() {
        let d = describe::describe(&Cli::command());
        let names: Vec<&str> =
            d["commands"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap()).collect();
        for want in
            ["serve", "describe", "doctor", "train", "val", "predict", "export", "bench", "dataset", "mcp"]
        {
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
        assert_eq!(opt["possible_values"].as_array().unwrap().len(), 3);
    }
}
