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
                    "data_dir": crate::output::record_path(&data_dir),
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
        let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((args.host.as_str(), args.port))
            .await
            .map_err(|e| host_lookup_failed(&args.host, args.port, e))?
            .collect();
        if addrs.is_empty() {
            return Err(host_lookup_failed(
                &args.host,
                args.port,
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "no socket addresses returned"),
            ));
        }
        let listener = tokio::net::TcpListener::bind(addrs.as_slice())
            .await
            .map_err(|e| bind_failed(&args.host, args.port, e))?;
        let addr = listener
            .local_addr()
            .map_err(|e| CliError::new(ExitCode::Runtime, "bind_failed", e.to_string()))?;
        let listen = ListenUrls::from_addr(addr);
        // Before the store opens (it can take a while): a second start can say where this one is.
        if let Some(lock) = &serve_lock
            && let Err(e) = lock.publish(&listen.url)
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
        out.human(listen.banner_line(args.mock));
        out.human(format!("  API:    {}/api/health", listen.url));
        match &web_dir {
            None => out.human(format!("  Studio: {}/  (embedded)", listen.url)),
            Some(dir) if web_ready => {
                out.human(format!("  Studio: {}/  (from {})", listen.url, folo_api::json::path(dir)))
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
        out.record(&listen.listening_record(args.mock, web_dir.as_deref(), web_ready, &data_dir, token_set));
        server
            .serve(listener, folo_server::shutdown_signal())
            .await
            .map_err(|e| CliError::new(ExitCode::Runtime, "serve_failed", e.to_string()))?;
        out.record(&json!({ "event": "stopped" }));
        out.human("Stopped.");
        Ok(())
    })
}

/// Connectable URL and bound socket address derived from a bound listener.
///
/// When `folo serve` binds a wildcard address (`0.0.0.0` or `::`), `url` uses the matching loopback address
/// (`http://127.0.0.1:<port>` or `http://[::1]:<port>`) so humans and local agents immediately get a directly
/// clickable, connectable URL (connecting to a wildcard address fails on Windows), while `bind` preserves the
/// exact socket address the listener bound.
#[derive(Debug, PartialEq, Eq)]
struct ListenUrls {
    url: String,
    bind: String,
    wildcard: bool,
}

impl ListenUrls {
    fn from_addr(addr: std::net::SocketAddr) -> Self {
        Self {
            url: serve_lock::local_url(addr),
            bind: addr.to_string(),
            wildcard: addr.ip().is_unspecified(),
        }
    }

    fn banner_line(&self, mock: bool) -> String {
        let mode = if mock { "mock data" } else { "the real store" };
        if self.wildcard {
            format!("FOLO {VERSION} serving {mode} on {} (bound to {})", self.url, self.bind)
        } else {
            format!("FOLO {VERSION} serving {mode} on {}", self.url)
        }
    }

    fn listening_record(
        &self,
        mock: bool,
        web_dir: Option<&std::path::Path>,
        web_ready: bool,
        data_dir: &std::path::Path,
        token: bool,
    ) -> serde_json::Value {
        json!({
            "event": "listening",
            "url": self.url,
            "bind": self.bind,
            "api": format!("{}/api", self.url),
            "mock": mock,
            "web_source": if web_dir.is_some() { "dir" } else { "embedded" },
            "web_dir": web_dir.map(crate::output::record_path),
            "web_ready": web_ready,
            "data_dir": (!mock).then(|| crate::output::record_path(data_dir)),
            "token": token,
        })
    }
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

/// Maps a hostname-resolution failure before `bind()` to `bind_failed` with `--host` guidance.
fn host_lookup_failed(host: &str, port: u16, e: std::io::Error) -> CliError {
    CliError::new(ExitCode::Runtime, "bind_failed", format!("could not bind {host}:{port}: {e}")).hint(
        "could not resolve --host; pass an IP address or resolvable hostname (e.g. 127.0.0.1 or 0.0.0.0)",
    )
}

/// Maps a `TcpListener::bind` failure to `bind_failed` with a hint that matches the failure kind: occupied ports
/// point to `--port`, non-local addresses point to `--host`, and permission errors distinguish privileged ports.
fn bind_failed(host: &str, port: u16, e: std::io::Error) -> CliError {
    use std::io::ErrorKind;
    let hint = match e.kind() {
        ErrorKind::AddrInUse => "is another `folo serve` running? pick another --port (0 chooses a free one)",
        ErrorKind::AddrNotAvailable => {
            "that --host address is not assigned to this machine; use 127.0.0.1 (local) or 0.0.0.0 (all interfaces)"
        }
        ErrorKind::PermissionDenied if (1..1024).contains(&port) => {
            "permission denied for this privileged port; pick a port >= 1024 (0 chooses a free one) or check --host"
        }
        ErrorKind::PermissionDenied => {
            "permission denied (the port may be reserved or exclusively bound, or the address disallowed); pick another --port (0 chooses a free one) or check --host"
        }
        _ => "check --host and --port (e.g. --host 127.0.0.1 --port 0 chooses a free local port)",
    };
    CliError::new(ExitCode::Runtime, "bind_failed", format!("could not bind {host}:{port}: {e}")).hint(hint)
}

#[cfg(test)]
mod tests {
    use std::io::{Error, ErrorKind};

    use super::*;

    #[test]
    fn bind_failed_distinguishes_port_and_host_errors() {
        let in_use = bind_failed("127.0.0.1", 7878, Error::new(ErrorKind::AddrInUse, "in use"));
        assert_eq!(in_use.code, "bind_failed");
        let h = in_use.hint.as_deref().unwrap();
        assert!(h.contains("--port") && !h.contains("--host"), "{h}");

        let not_avail = bind_failed("203.0.113.1", 0, Error::new(ErrorKind::AddrNotAvailable, "not local"));
        let h = not_avail.hint.as_deref().unwrap();
        assert!(h.contains("--host") && !h.contains("--port"), "{h}");

        let dns = host_lookup_failed("no-such-host.invalid", 7878, Error::other("failed to lookup address"));
        let h = dns.hint.as_deref().unwrap();
        assert!(h.contains("resolve --host") && !h.contains("--port"), "{h}");

        let priv_denied = bind_failed("127.0.0.1", 80, Error::new(ErrorKind::PermissionDenied, "denied"));
        let h = priv_denied.hint.as_deref().unwrap();
        assert!(h.contains("privileged port") && h.contains(">= 1024"), "{h}");

        let high_denied = bind_failed("127.0.0.1", 7878, Error::new(ErrorKind::PermissionDenied, "denied"));
        let h = high_denied.hint.as_deref().unwrap();
        assert!(h.contains("exclusively bound") && !h.contains(">= 1024"), "{h}");

        // A non-lookup socket error on a hostname does not claim DNS resolution failed.
        let other = bind_failed("localhost", 7878, Error::other("too many open files"));
        let h = other.hint.as_deref().unwrap();
        assert!(!h.contains("resolve --host") && h.contains("--host") && h.contains("--port"), "{h}");
    }

    #[test]
    fn listen_urls_expose_loopback_for_wildcards_and_preserve_explicit_hosts() {
        let data = std::path::Path::new("data");

        // IPv4 wildcard -> clickable IPv4 loopback URL + explicit bind address.
        let v4_wild = ListenUrls::from_addr("0.0.0.0:7878".parse().unwrap());
        assert!(v4_wild.wildcard);
        assert_eq!(v4_wild.url, "http://127.0.0.1:7878");
        assert_eq!(v4_wild.bind, "0.0.0.0:7878");
        assert_eq!(
            v4_wild.banner_line(false),
            format!("FOLO {VERSION} serving the real store on http://127.0.0.1:7878 (bound to 0.0.0.0:7878)")
        );
        let rec = v4_wild.listening_record(true, None, true, data, false);
        assert_eq!(rec["url"], "http://127.0.0.1:7878");
        assert_eq!(rec["api"], "http://127.0.0.1:7878/api");
        assert_eq!(rec["bind"], "0.0.0.0:7878");

        // IPv6 wildcard -> clickable IPv6 loopback URL + explicit bind address.
        let v6_wild = ListenUrls::from_addr("[::]:7878".parse().unwrap());
        assert!(v6_wild.wildcard);
        assert_eq!(v6_wild.url, "http://[::1]:7878");
        assert_eq!(v6_wild.bind, "[::]:7878");
        assert_eq!(
            v6_wild.banner_line(true),
            format!("FOLO {VERSION} serving mock data on http://[::1]:7878 (bound to [::]:7878)")
        );
        let rec = v6_wild.listening_record(false, None, true, data, true);
        assert_eq!(rec["url"], "http://[::1]:7878");
        assert_eq!(rec["api"], "http://[::1]:7878/api");
        assert_eq!(rec["bind"], "[::]:7878");

        // Explicit loopback and LAN hosts stay as bound without a redundant "(bound to ...)" suffix.
        for (raw, want_url, want_bind) in [
            ("127.0.0.1:7878", "http://127.0.0.1:7878", "127.0.0.1:7878"),
            ("[::1]:7878", "http://[::1]:7878", "[::1]:7878"),
            ("192.168.1.5:7878", "http://192.168.1.5:7878", "192.168.1.5:7878"),
        ] {
            let l = ListenUrls::from_addr(raw.parse().unwrap());
            assert!(!l.wildcard, "{raw}");
            assert_eq!(l.url, want_url, "{raw}");
            assert_eq!(l.bind, want_bind, "{raw}");
            assert_eq!(l.banner_line(false), format!("FOLO {VERSION} serving the real store on {want_url}"));
            let rec = l.listening_record(false, None, true, data, false);
            assert_eq!(rec["url"], want_url);
            assert_eq!(rec["api"], format!("{want_url}/api"));
            assert_eq!(rec["bind"], want_bind);
        }

        // Scoped IPv6 host is preserved verbatim.
        let scoped =
            std::net::SocketAddr::V6(std::net::SocketAddrV6::new("fe80::1234".parse().unwrap(), 7878, 0, 7));
        let l = ListenUrls::from_addr(scoped);
        assert!(!l.wildcard);
        assert_eq!(l.url, "http://[fe80::1234%7]:7878");
        assert_eq!(l.bind, "[fe80::1234%7]:7878");
        assert!(!l.banner_line(false).contains("bound to"));
    }
}
