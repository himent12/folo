//! `folo serve`: the HTTP + SSE API (folo-server) and FOLO Studio, on the real store or the mock. Also the helpers
//! `folo mcp` and `folo doctor` share with it: tracing to stderr, the data dir's lock errors, where the Studio comes
//! from.

use std::path::PathBuf;
use std::sync::Arc;

use folo_cli::VERSION;
use folo_server::{Cors, MockStore, Options, RealStore, Server, Store, WebSource};
use serde_json::json;

use crate::cli::ServeArgs;
use crate::output::{CliError, ExitCode, Out};
use crate::{data_dir, default_data_dir, serve_lock};

/// The web app folder used when `--web-dir` is not given and nothing is embedded.
pub(crate) const DEFAULT_WEB_DIR: &str = "web/dist";

/// What `folo serve` serves outside `/api`: an explicit `--web-dir`, else the embedded Studio (`embed-web`
/// builds), else `web/dist`.
pub(crate) fn resolve_web(web_dir: Option<PathBuf>) -> WebSource {
    match web_dir {
        Some(dir) => WebSource::Dir(dir),
        None if folo_server::web::embedded().is_some() => WebSource::Embedded,
        None => WebSource::Dir(PathBuf::from(DEFAULT_WEB_DIR)),
    }
}

pub(crate) fn init_tracing() {
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

pub(crate) fn serve(args: ServeArgs, out: Out) -> Result<(), CliError> {
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
    // One owner per data dir (the mock writes nothing): a second would stop the first one's active runs.
    let serve_lock = if args.mock {
        None
    } else {
        match serve_lock::acquire(&data_dir) {
            Ok(serve_lock::Acquire::Held(lock)) => Some(lock),
            Ok(serve_lock::Acquire::InUse { url }) => {
                out.record(&json!({
                    "event": "data_dir_in_use",
                    "data_dir": folo_api::json::path(&data_dir),
                    "url": url,
                }));
                return Err(data_dir_in_use(&data_dir, url.as_deref()));
            }
            Err(e) => return Err(data_dir_lock_failed(&data_dir, e)),
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
        let url = format!("http://{addr}");
        // Before the store opens (it can take a while): a second start can say where this one is.
        if let Some(lock) = &serve_lock
            && let Err(e) = lock.publish(&url)
        {
            tracing::warn!("could not record the address in the data dir: {e}");
        }
        // A file in place of a folder the store keeps (runs/, ...) would be listed as empty with nothing said.
        if !args.mock {
            RealStore::check_layout(&data_dir)
                .map_err(|m| CliError::new(ExitCode::Runtime, "serve_failed", m))?;
        }
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
            "web_dir": web_dir.as_ref().map(folo_api::json::path),
            "web_ready": web_ready,
            "data_dir": (!args.mock).then(|| folo_api::json::path(&data_dir)),
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

/// Another process (`folo serve`, or an embedded `folo mcp`) owns `dir`; `url` is where it listens, when it says.
pub(crate) fn data_dir_in_use(dir: &std::path::Path, url: Option<&str>) -> CliError {
    let dir = folo_api::json::path(dir);
    let hint = match url {
        Some(url) => format!("a folo serve listens on {url}; use that one, or pass another --data-dir"),
        None => "stop the other folo serve or folo mcp first, or pass another --data-dir".into(),
    };
    CliError::new(ExitCode::Runtime, "data_dir_in_use", format!("another FOLO process is using {dir}"))
        .hint(hint)
}

/// The data dir's lock could not be taken (not merely held elsewhere): refuse rather than share the store.
pub(crate) fn data_dir_lock_failed(dir: &std::path::Path, e: std::io::Error) -> CliError {
    let lock = folo_api::json::path(dir.join("serve.lock"));
    CliError::new(ExitCode::Runtime, "data_dir_lock_failed", format!("could not lock {lock}: {e}"))
        .hint("the data dir must be writable and support file locks; pass another --data-dir")
}
