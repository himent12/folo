//! The `folo` command-line tool. See `docs/CLI.md`.

mod cloud;
mod coco;
mod dataset;
mod describe;
mod draw;
mod engine;
mod infer;
mod output;
mod serve_lock;
mod train;
mod weights;

use std::path::PathBuf;
use std::sync::Arc;

use clap::CommandFactory;
use folo_server::{Cors, MockStore, Options, RealStore, Server, Store, WebSource};
use serde_json::json;

// The command tree is the library's (src/lib.rs), so tests parse commands with it.
use folo_cli::{VERSION, cli};

use cli::{Cli, Command, DoctorArgs, McpArgs, ServeArgs};
use output::{CliError, ExitCode, Out};

/// The web app folder used when `--web-dir` is not given and nothing is embedded.
const DEFAULT_WEB_DIR: &str = "web/dist";

/// What `folo serve` serves outside `/api`: an explicit `--web-dir`, else the embedded Studio (`embed-web`
/// builds), else `web/dist`.
fn resolve_web(web_dir: Option<PathBuf>) -> WebSource {
    match web_dir {
        Some(dir) => WebSource::Dir(dir),
        None if folo_server::web::embedded().is_some() => WebSource::Embedded,
        None => WebSource::Dir(PathBuf::from(DEFAULT_WEB_DIR)),
    }
}

fn main() -> std::process::ExitCode {
    let json = std::env::args_os().skip(1).any(|a| a == "--json");
    let out = Out { json };
    let cli = match parse() {
        Ok(cli) => cli,
        Err(e) => return usage_error(out, e),
    };
    // `folo mcp`'s stdout belongs to the protocol, also with --json: its errors go to stderr only.
    let out = Out { json: cli.json && !matches!(cli.command, Command::Mcp(_)) };
    match run(cli, out) {
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
        Command::Serve(args) => serve(args, out),
        Command::Describe(args) => {
            let surface = describe::describe(&Cli::command(), &args.command, !args.no_mcp_tools)?;
            if out.json {
                out.record(&surface);
            } else {
                println!("{}", serde_json::to_string_pretty(&surface).expect("JSON value serializes"));
            }
            Ok(())
        }
        Command::Doctor(args) => doctor(args, out),
        Command::Train(args) => train::train(args, out),
        Command::Stop(args) => train::request_stop(&args.run_dir, out),
        Command::Val(args) => infer::val(args, out),
        Command::Predict(args) => infer::predict(args, out),
        Command::Export(args) => infer::export(args, out),
        Command::Bench(_) => Err(not_implemented("bench", "the engine")),
        Command::Dataset(cmd) => dataset::run(cmd, out),
        Command::Weights(cmd) => weights::run(cmd, out),
        Command::Cloud(cmd) => cloud::run(cmd, out),
        Command::Mcp(args) => mcp(args),
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
    Ok(dir)
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Logs always go to stderr so stdout stays clean for `--json` records.
    let ansi = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(ansi)
        .try_init();
}

fn serve(args: ServeArgs, out: Out) -> Result<(), CliError> {
    init_tracing();
    let precision: folo_vision::Precision = args
        .precision
        .parse()
        .map_err(|e: folo_vision::Error| CliError::new(ExitCode::Usage, "usage", e.to_string()))?;
    if args.tensor_cores {
        // Before anything creates the GPU context: the device has to be created with them.
        folo_gpu::context::request_tensor_cores();
    }
    let data_dir = if args.mock {
        args.data_dir.clone().unwrap_or_else(default_data_dir)
    } else {
        data_dir(args.data_dir.clone())?
    };
    // One server per data dir (the mock writes nothing): a second would stop the first one's active runs.
    let serve_lock = if args.mock {
        None
    } else {
        match serve_lock::acquire(&data_dir) {
            Ok(serve_lock::Acquire::Held(lock)) => Some(lock),
            Ok(serve_lock::Acquire::InUse { url }) => {
                let dir = folo_api::json::path(&data_dir);
                out.record(&json!({ "event": "data_dir_in_use", "data_dir": dir, "url": url }));
                let hint = match &url {
                    Some(url) => format!("it listens on {url}; use that one, or pass another --data-dir"),
                    None => "stop it first, or pass another --data-dir".into(),
                };
                return Err(CliError::new(
                    ExitCode::Runtime,
                    "data_dir_in_use",
                    format!("another folo serve is using {dir}"),
                )
                .hint(hint));
            }
            Err(e) => {
                tracing::warn!("could not lock {}: {e}", folo_api::json::path(data_dir.join("serve.lock")));
                None
            }
        }
    };
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
        let store: Arc<dyn Store> = if args.mock {
            Arc::new(MockStore::new())
        } else {
            Arc::new(
                RealStore::open(data_dir.clone())
                    .device(args.device.clone())
                    .precision(precision)
                    .published_weights(folo_api::weights::cache_dir()),
            )
        };
        let web = resolve_web(args.web_dir.clone());
        let (web_dir, web_ready) = match &web {
            WebSource::Dir(dir) => (Some(dir.clone()), folo_server::web_dir_ready(dir)),
            _ => (None, true),
        };
        let cors = if args.cors_any { Cors::Any } else { Cors::Loopback };
        let token = args.token.clone().and_then(folo_server::Token::new);
        let token_set = token.is_some();
        let server = Server::new(
            store,
            Options {
                web: web.clone(),
                cors,
                token,
                bind: Some(addr.ip()),
                allow_hosts: args.allow_host.clone(),
            },
        );
        let url = format!("http://{addr}");
        if let Some(lock) = &serve_lock
            && let Err(e) = lock.publish(&url)
        {
            tracing::warn!("could not record the address in the data dir: {e}");
        }
        out.human(format!(
            "FOLO {VERSION} serving {} on {url}",
            if args.mock { "mock data" } else { "the real store" }
        ));
        out.human(format!("  API:    {url}/api/health"));
        match &web_dir {
            None => out.human(format!("  Studio: {url}/  (embedded)")),
            Some(dir) if web_ready => {
                out.human(format!("  Studio: {url}/  (from {})", folo_api::json::path(dir)))
            }
            Some(dir) => out.human(format!(
                "  Studio: not built ({} has no index.html); build web/ or use the Vite dev server on :5173",
                folo_api::json::path(dir)
            )),
        }
        if !args.mock {
            out.human(format!("  Data:   {}", folo_api::json::path(&data_dir)));
        }
        if token_set {
            out.human("  Token:  required on /api (Authorization: Bearer)");
        }
        out.human("Press Ctrl+C to stop.");
        out.record(&json!({
            "event": "listening",
            "url": url,
            "api": format!("{url}/api"),
            "mock": args.mock,
            "web_source": if web_dir.is_some() { "dir" } else { "embedded" },
            "web_dir": web_dir,
            "web_ready": web_ready,
            "data_dir": if args.mock { None } else { Some(&data_dir) },
            "token": token_set,
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

/// `folo mcp`: MCP over stdio. stdout belongs to the protocol, so nothing else is printed there (not even with
/// `--json`); logs go to stderr.
fn mcp(args: McpArgs) -> Result<(), CliError> {
    init_tracing();
    if args.tensor_cores && args.server.is_none() {
        // As `folo serve --tensor-cores`: before anything creates the GPU context.
        folo_gpu::context::request_tensor_cores();
    }
    // Embedded mode writes there; a mock or proxying server does not.
    let data_dir = if args.server.is_none() && !args.mock {
        data_dir(args.data_dir.clone())?
    } else {
        args.data_dir.clone().unwrap_or_else(default_data_dir)
    };
    let backend = match (&args.server, args.mock) {
        (Some(url), _) => folo_mcp::Backend::Proxy { url: url.clone(), token: args.token.clone() },
        (None, true) => folo_mcp::Backend::Mock,
        (None, false) => folo_mcp::Backend::Real,
    };
    let opts = folo_mcp::Options {
        data_dir: data_dir.clone(),
        read_only: args.read_only,
        confine: args.confine,
        device: args.device.clone(),
        published_weights: Some(folo_api::weights::cache_dir()),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::new(ExitCode::Runtime, "runtime", format!("could not start tokio: {e}")))?;
    let result = runtime.block_on(async move {
        let server = folo_mcp::build(backend.clone(), opts)
            .map_err(|e| CliError::new(ExitCode::Usage, "usage", e).hint("see `folo mcp --help`"))?;
        tracing::info!(?backend, data_dir = %folo_api::json::path(&data_dir), read_only = args.read_only, confine = args.confine, "folo mcp on stdio");
        folo_mcp::serve_stdio(server, folo_server::shutdown_signal())
            .await
            .map_err(|e| CliError::new(ExitCode::Runtime, "mcp_failed", e))
    });
    // tokio's stdin reader sits on a blocking thread that cannot be interrupted; do not wait for it.
    runtime.shutdown_background();
    result
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn doctor(args: DoctorArgs, out: Out) -> Result<(), CliError> {
    let host = folo_server::system::host_info();
    let web_dir = args.web_dir.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_WEB_DIR));
    let web_exists = web_dir.is_dir();
    let web_ready = folo_server::web_dir_ready(&web_dir);
    let web_wasm = folo_server::web::dir_has_folo_wasm(&web_dir);
    let embedded = folo_server::web::embedded();
    let embedded_wasm = embedded.is_some_and(folo_server::web::has_folo_wasm);
    let embedded = embedded.map(<[_]>::len);
    let web_source = match resolve_web(args.web_dir.clone()) {
        WebSource::Embedded => "embedded",
        _ => "dir",
    };
    // Whether the Studio `folo serve` would serve can run models in the browser (its Playground's WebGPU / CPU).
    // (`folo serve` does not serve a folder without index.html at all.)
    let served_wasm = if web_source == "embedded" { embedded_wasm } else { web_ready && web_wasm };
    if args.tensor_cores {
        folo_gpu::context::request_tensor_cores();
    }
    let options = folo_gpu::GpuOptions::from_env();
    let adapters = folo_gpu::GpuContext::enumerate(&options);
    // Creating the device (the shared one, as `folo serve` does for `device: auto`) proves the default adapter
    // works and shows which one FOLO picks.
    let capabilities = folo_server::system::capabilities();
    let selected = folo_gpu::GpuContext::shared().map(|c| (c.summary().clone(), c.tensor_cores_status()));
    let gpus: Vec<_> = adapters
        .iter()
        .map(|a| {
            json!({
                "index": a.index,
                "name": a.name,
                "backend": a.backend.to_ascii_lowercase(),
                "device_type": a.device_type,
                "vendor": a.vendor,
                "device": a.device,
                "driver": a.driver,
                "driver_info": a.driver_info,
                "shader_f16": a.shader_f16,
                "tensor_cores": a.tensor_cores,
                "max_buffer_bytes": a.max_buffer_size,
                "max_storage_binding_bytes": a.max_storage_buffer_binding_size,
                "workgroup_storage_bytes": a.max_compute_workgroup_storage_size,
                "max_invocations_per_workgroup": a.max_compute_invocations_per_workgroup,
                "subgroup_size": [a.subgroup_size.0, a.subgroup_size.1],
            })
        })
        .collect();
    let selected_json = match &selected {
        Ok((s, tc)) => json!({
            "index": s.index,
            "name": s.name,
            "backend": s.backend.to_ascii_lowercase(),
            "tensor_cores": tc,
        }),
        Err(e) => json!({ "error": e.to_string() }),
    };
    let report = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "git_sha": env!("FOLO_GIT_SHA"),
        "build_date": env!("FOLO_BUILD_DATE"),
        "os": host.os,
        "arch": host.arch,
        "cpu": { "name": host.cpu_name, "threads": host.cpu_threads },
        "memory": { "total_bytes": host.memory_total_bytes, "available_bytes": host.memory_available_bytes },
        "gpus": gpus,
        "gpu_detection": "wgpu",
        "gpu_selected": selected_json,
        "capabilities": capabilities,
        "web_dist": { "path": web_dir, "exists": web_exists, "index_html": web_ready, "folo_wasm": web_wasm },
        "web_embedded": { "present": embedded.is_some(), "files": embedded.unwrap_or(0), "folo_wasm": embedded_wasm },
        "web_source": web_source,
        "web_folo_wasm": served_wasm,
    });
    out.record(&report);
    out.human(format!("folo      {VERSION}"));
    out.human(format!("os        {} ({})", host.os, host.arch));
    out.human(format!("cpu       {} ({} threads)", host.cpu_name, host.cpu_threads));
    out.human(format!(
        "memory    {} total, {} available",
        gib(host.memory_total_bytes),
        gib(host.memory_available_bytes)
    ));
    if adapters.is_empty() {
        out.human("gpu       no adapters found (Vulkan / DX12 / Metal); FOLO runs on the CPU");
    }
    for a in &adapters {
        out.human(format!("gpu       {a}"));
    }
    match &selected {
        Ok((s, tc)) => {
            out.human(format!(
                "gpu       FOLO uses [{}] {} ({}); set {} to pick another",
                s.index,
                s.name,
                s.backend,
                folo_gpu::context::ENV_ADAPTER
            ));
            out.human(format!("gpu       tensor cores (f16): {tc}"));
        }
        Err(e) => out.human(format!("gpu       no usable adapter: {e}")),
    }
    let tc = &capabilities.tensor_cores;
    let amp = match &capabilities.amp.reason {
        None => format!("mixed-precision training (amp) available on {}", capabilities.device),
        // The adapter has them, only this process did not ask: `folo train --amp` does.
        Some(_)
            if tc.available
                && tc.reason.as_deref() == Some(folo_gpu::context::TENSOR_CORES_NOT_REQUESTED) =>
        {
            format!(
                "available on {}: `folo train --amp` asks for tensor cores itself; `folo serve` and `folo mcp` need \
                 `--tensor-cores` for amp runs",
                capabilities.device
            )
        }
        Some(why) => format!("not available: {why}"),
    };
    out.human(format!("amp       {amp}"));
    let web = if web_ready {
        "built".to_string()
    } else if web_exists {
        "folder exists but has no index.html".to_string()
    } else {
        "not built (run `npm ci && npm run build` in web/)".to_string()
    };
    out.human(format!("web/dist  {} — {web}", folo_api::json::path(&web_dir)));
    let studio = match embedded {
        Some(n) if web_source == "embedded" => format!("embedded ({n} files); `folo serve` uses it"),
        Some(n) => format!("embedded ({n} files); `folo serve --web-dir` overrides it"),
        None => "not embedded (build with --features embed-web)".to_string(),
    };
    out.human(format!("studio    {studio}"));
    let source = if web_source == "embedded" {
        "the embedded Studio".to_string()
    } else {
        folo_api::json::path(&web_dir)
    };
    let wasm = if served_wasm {
        format!("folo-wasm in {source}; the Playground can run models in the browser (WebGPU or CPU)")
    } else {
        format!(
            "no folo-wasm in {source}; the Playground runs models on the server only (build it with scripts/build-wasm before `npm run build`)"
        )
    };
    out.human(format!("browser   {wasm}"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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
            "serve", "describe", "doctor", "train", "stop", "val", "predict", "export", "bench", "dataset",
            "mcp",
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
