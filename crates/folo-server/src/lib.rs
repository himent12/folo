//! FOLO Studio server: HTTP + SSE API for datasets, training runs and inference.
//!
//! Implements the contract in `docs/API.md` on axum. Handlers sit on a [`Store`]: [`MockStore`] serves deterministic
//! synthetic data (`folo serve --mock`), [`RealStore`] serves datasets from disk (read with folo-data, indexed and
//! cached under the data dir), trains runs with folo-vision's trainer on a background worker, lists their models,
//! runs inference, has coding agents label images ([`label`]) and runs vision benchmarks ([`bench`]). The request / response types live in the `folo-api` crate (re-exported as [`api`]) so the CLI and
//! FOLO Desktop share them.
//!
//! See `docs/ARCHITECTURE.md` for how this crate fits into FOLO.

pub mod bench;
pub mod cli;
pub mod cloud;
mod engine;
pub mod error;
mod index;
pub mod label;
pub mod mock;
mod models;
mod probe;
mod raster;
pub mod real;
mod routes;
mod runs;
mod stats;
pub mod store;
pub mod system;
pub mod throughput;
mod uploads;
mod util;
pub mod web;

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tower_http::compression::CompressionLayer;
use tower_http::compression::predicate::{DefaultPredicate, NotForContentType, Predicate};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::Level;

pub use folo_api as api;
pub use mock::MockStore;
pub use real::RealStore;
pub use store::Store;

/// Which browser origins may call the API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cors {
    /// Any `http(s)://localhost:*`, `127.0.0.1:*` or `[::1]:*` origin (the Vite dev server on :5173, Studio served
    /// from another local port). Other websites cannot drive the local server from a visitor's browser.
    #[default]
    Loopback,
    /// Any origin (fully permissive).
    Any,
}

/// Where FOLO Studio's static files come from. Either way they are served with an SPA fallback to `index.html`;
/// when there is nothing to serve, non-API routes answer 404 with a hint.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum WebSource {
    /// No web app: only the API.
    #[default]
    None,
    /// A built web app on disk (`web/dist`); used when it holds an `index.html`.
    Dir(PathBuf),
    /// The web app compiled into the binary (the `embed-web` feature, see [`web::embedded`]).
    Embedded,
}

/// A secret every `/api` request must carry as `Authorization: Bearer <token>` (`folo serve --token`; docs/CLOUD.md,
/// Securing a server). It never reaches a log: `Debug` hides it, and the server compares it in constant time.
#[derive(Clone, PartialEq, Eq)]
pub struct Token(String);

impl Token {
    /// The token `secret`; `None` when it is empty (an empty token would protect nothing).
    pub fn new(secret: impl Into<String>) -> Option<Token> {
        let secret = secret.into();
        (!secret.is_empty()).then_some(Token(secret))
    }

    /// Whether `presented` is the token. The time taken depends on the token's length only, not on where the two
    /// differ.
    fn matches(&self, presented: &[u8]) -> bool {
        let token = self.0.as_bytes();
        let mut diff = token.len() ^ presented.len();
        for (i, &t) in token.iter().enumerate() {
            diff |= usize::from(t ^ presented.get(i).copied().unwrap_or(0));
        }
        std::hint::black_box(diff) == 0
    }
}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(…)")
    }
}

/// Server options.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The web app to serve outside `/api`.
    pub web: WebSource,
    /// CORS policy.
    pub cors: Cors,
    /// The token `/api` requests must carry (`None`: no token needed, the default for a server on loopback).
    pub token: Option<Token>,
    /// The address the server listens on (`folo serve --host`). On a loopback address, and when `None` (a router
    /// used in process), a request whose `Host` names anything but a loopback address or one of
    /// [`Options::allow_hosts`] is refused before any handler runs (`403 forbidden_host`, the guard against DNS
    /// rebinding; docs/API.md, Conventions). Beyond loopback every `Host` is served.
    pub bind: Option<IpAddr>,
    /// More `Host` names a server on loopback answers (`folo serve --allow-host`): a tunnel or reverse proxy that
    /// keeps its own `Host`. A port in a name is ignored.
    pub allow_hosts: Vec<String>,
}

/// Shared handler state.
#[derive(Clone)]
pub struct AppState {
    store: Arc<dyn Store>,
    probe: Arc<system::SystemProbe>,
    shutdown: Arc<watch::Sender<bool>>,
    /// The served Studio includes the in-browser runtime (`/api/system`'s `web_wasm` feature).
    web_wasm: bool,
}

/// A configured server: the router plus the shutdown switch that ends open SSE streams.
pub struct Server {
    router: Router,
    shutdown: Arc<watch::Sender<bool>>,
    store: Arc<dyn Store>,
    /// `/api` needs a token.
    token: bool,
}

/// True when the web app folder exists and has an `index.html`.
pub fn web_dir_ready(dir: &Path) -> bool {
    dir.join("index.html").is_file()
}

fn is_loopback_origin(origin: &HeaderValue) -> bool {
    let Ok(o) = origin.to_str() else { return false };
    let Some(rest) = o.strip_prefix("http://").or_else(|| o.strip_prefix("https://")) else { return false };
    let host = if rest.starts_with('[') {
        rest.split(']').next().map(|h| &h[1..]).unwrap_or("")
    } else {
        rest.split(':').next().unwrap_or("")
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// True when `origin` names the same host and port the request was sent to (a page served by this server).
fn is_same_origin(origin: &HeaderValue, host: Option<&HeaderValue>) -> bool {
    let (Ok(o), Some(Ok(h))) = (origin.to_str(), host.map(|h| h.to_str())) else { return false };
    let authority = o.strip_prefix("http://").or_else(|| o.strip_prefix("https://")).unwrap_or("");
    !authority.is_empty() && authority.eq_ignore_ascii_case(h)
}

/// The name in a `Host` value (`name`, `name:port`, `[v6]`, `[v6]:port`), lower case and without a trailing dot;
/// `None` when the value is malformed.
fn host_name(host: &str) -> Option<String> {
    let (name, port) = if host.starts_with('[') {
        let end = host.find(']')? + 1;
        let rest = &host[end..];
        (&host[..end], if rest.is_empty() { None } else { Some(rest.strip_prefix(':')?) })
    } else {
        match host.split_once(':') {
            Some((name, port)) => (name, Some(port)),
            None => (host, None),
        }
    };
    if name.is_empty() || port.is_some_and(|p| p.parse::<u16>().is_err()) {
        return None;
    }
    let name = name.to_ascii_lowercase();
    Some(name.strip_suffix('.').map(str::to_owned).unwrap_or(name))
}

/// `localhost`, `127.x.y.z` or `[::1]` (a name from [`host_name`]).
fn is_loopback_name(name: &str) -> bool {
    name == "localhost"
        || name.parse::<Ipv4Addr>().is_ok_and(|ip| ip.is_loopback())
        || name
            .strip_prefix('[')
            .and_then(|n| n.strip_suffix(']'))
            .and_then(|n| n.parse::<Ipv6Addr>().ok())
            .is_some_and(|ip| ip.is_loopback())
}

/// DNS rebinding: a website whose name re-resolves to 127.0.0.1 makes the browser send its requests to a server
/// on loopback, with `Host: <its name>` and a matching `Origin`, so CORS and [`guard_origin`] let them through.
/// A server on loopback therefore answers only requests whose `Host` is a loopback name or an `--allow-host` one
/// (`allowed`, already in [`host_name`] form). A request without `Host` (in-process tests, raw clients; a
/// browser always sends one) is served.
async fn guard_host(State(allowed): State<Arc<[String]>>, req: Request, next: Next) -> Response {
    let host = match req.headers().get(header::HOST) {
        Some(h) => Some(h.to_str().unwrap_or("")),
        None => req.uri().authority().map(|a| a.as_str()),
    };
    let Some(host) = host else { return next.run(req).await };
    if host_name(host).is_some_and(|n| is_loopback_name(&n) || allowed.contains(&n)) {
        return next.run(req).await;
    }
    error::Error::new(
        StatusCode::FORBIDDEN,
        "forbidden_host",
        format!("this server listens on loopback and does not answer requests for host '{host}'"),
    )
    .hint(
        "reach it as localhost or 127.0.0.1; for a tunnel or reverse proxy that keeps its own Host, start the \
         server with --allow-host <name> (and --token when others can reach that name)",
    )
    .into_response()
}

/// CORS only hides responses; a browser still *sends* "simple" cross-origin POSTs (form posts) without a
/// preflight. So with [`Cors::Loopback`], state-changing requests whose `Origin` is neither loopback nor this
/// server itself are refused before they reach a handler. Requests without `Origin` (curl, agents, the desktop
/// app) are allowed.
/// One log line per request: at INFO for requests that change something (not GET / HEAD / OPTIONS) and for
/// failures, at DEBUG for successful reads, so Studio's polling (`GET /api/runs` every 2 s) stays out of the
/// default log (`RUST_LOG=folo_server=debug` shows it).
async fn log_request(req: Request, next: Next) -> Response {
    let (method, uri) = (req.method().clone(), req.uri().path().to_owned());
    let t0 = std::time::Instant::now();
    let res = next.run(req).await;
    let (status, ms) = (res.status().as_u16(), t0.elapsed().as_secs_f64() * 1e3);
    let read = matches!(method, Method::GET | Method::HEAD | Method::OPTIONS);
    if read && !res.status().is_client_error() && !res.status().is_server_error() {
        tracing::debug!(%method, uri, status, ms, "request");
    } else {
        tracing::info!(%method, uri, status, ms, "request");
    }
    res
}

/// With a [`Token`], every `/api` request needs `Authorization: Bearer <token>`, except `GET /api/health` (so a
/// client can tell a server that is up from one it may not use) and CORS preflights; anything else is `401
/// unauthorized` with `WWW-Authenticate: Bearer`. The header is never logged (the trace layer leaves headers out).
async fn require_token(State(token): State<Token>, req: Request, next: Next) -> Response {
    let open = match *req.method() {
        Method::OPTIONS => true,
        Method::GET | Method::HEAD => req.uri().path() == "/health",
        _ => false,
    };
    let presented = req.headers().get(header::AUTHORIZATION).map(|v| {
        let v = v.as_bytes();
        // The scheme is case-insensitive (RFC 9110 §11.1).
        match v.get(..7) {
            Some(scheme) if scheme.eq_ignore_ascii_case(b"bearer ") => &v[7..],
            _ => &[][..],
        }
    });
    if open || presented.is_some_and(|p| token.matches(p)) {
        return next.run(req).await;
    }
    let message = match presented {
        None => "this server needs a token: send `Authorization: Bearer <token>`",
        Some(_) => "the token is not this server's",
    };
    let mut res = error::Error::new(StatusCode::UNAUTHORIZED, "unauthorized", message)
        .hint(
            "send the token the server was started with (`folo serve --token` or FOLO_TOKEN) as \
             `Authorization: Bearer <token>`",
        )
        .into_response();
    res.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    res
}

async fn guard_origin(State(cors): State<Cors>, req: Request, next: Next) -> Response {
    let safe = matches!(*req.method(), Method::GET | Method::HEAD | Method::OPTIONS);
    if cors == Cors::Loopback
        && !safe
        && let Some(origin) = req.headers().get(header::ORIGIN)
        && !is_loopback_origin(origin)
        && !is_same_origin(origin, req.headers().get(header::HOST))
    {
        let o = origin.to_str().unwrap_or("?");
        return error::Error::new(
            StatusCode::FORBIDDEN,
            "forbidden_origin",
            format!("requests from origin '{o}' are not allowed"),
        )
        .hint("only localhost pages may change state; start the server with --cors-any to allow every origin")
        .into_response();
    }
    next.run(req).await
}

async fn web_not_built() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        "FOLO Studio is not built. Run `npm ci && npm run build` in web/ (or pass --web-dir, or build folo with \
         --features embed-web), or use the Vite dev server on :5173. The API is under /api.\n",
    )
}

impl Server {
    /// Builds the router around `store`.
    pub fn new(store: Arc<dyn Store>, opts: Options) -> Server {
        let embedded = if opts.web == WebSource::Embedded { web::embedded() } else { None };
        Server::with_embedded(store, opts, embedded)
    }

    /// [`Server::new`] with the embedded web app passed in (tests use a fixture table).
    fn with_embedded(
        store: Arc<dyn Store>,
        opts: Options,
        embedded: Option<&'static [web::EmbeddedFile]>,
    ) -> Server {
        let (tx, _) = watch::channel(false);
        let shutdown = Arc::new(tx);
        // What the fallback below serves: the folder when it is a built app, else the embedded files.
        let web_wasm = match (&opts.web, embedded) {
            (WebSource::Dir(dir), _) if web_dir_ready(dir) => web::dir_has_folo_wasm(dir),
            (_, Some(files)) => web::has_folo_wasm(files),
            _ => false,
        };
        let state = AppState {
            store: store.clone(),
            probe: Arc::new(system::SystemProbe::default()),
            shutdown: shutdown.clone(),
            web_wasm,
        };
        let token = opts.token.is_some();
        let check_host = opts.bind.is_none_or(|ip| ip.to_canonical().is_loopback());
        let allow_hosts: Arc<[String]> =
            opts.allow_hosts.iter().filter_map(|h| host_name(h.trim())).collect();
        let api = match opts.token {
            Some(t) => routes::api().layer(axum::middleware::from_fn_with_state(t, require_token)),
            None => routes::api(),
        };
        let mut router = Router::new().nest("/api", api).with_state(state);
        router = match (opts.web, embedded) {
            (WebSource::Dir(dir), _) if web_dir_ready(&dir) => {
                let spa = ServeDir::new(&dir).fallback(ServeFile::new(dir.join("index.html")));
                router.fallback_service(spa)
            }
            (_, Some(files)) => {
                router.fallback(move |method: Method, uri: Uri, headers: HeaderMap| async move {
                    web::respond(files, &method, &uri, &headers)
                })
            }
            _ => router.fallback(web_not_built),
        };
        let cors = match opts.cors {
            Cors::Any => CorsLayer::permissive(),
            Cors::Loopback => CorsLayer::new()
                .allow_origin(AllowOrigin::predicate(|o, _| is_loopback_origin(o)))
                .allow_methods([
                    Method::GET,
                    Method::HEAD,
                    Method::POST,
                    Method::PUT,
                    Method::DELETE,
                    Method::OPTIONS,
                ])
                .allow_headers(tower_http::cors::Any)
                // Readable by page scripts (Studio caches downloaded weights by ETag).
                .expose_headers([header::ETAG, header::CONTENT_DISPOSITION]),
        };
        let mut router = router
            .layer(axum::middleware::from_fn_with_state(opts.cors, guard_origin))
            // Weights files (`application/octet-stream`) are sent as they are: f32 / f16 tensors barely compress,
            // and their Content-Length and strong ETag describe the file's own bytes.
            .layer(CompressionLayer::new().compress_when(
                DefaultPredicate::new().and(NotForContentType::const_new("application/octet-stream")),
            ))
            .layer(cors);
        if check_host {
            // Outside the routes, the web app, CORS (which answers preflights itself) and the origin guard:
            // nothing runs for a foreign Host.
            router = router.layer(axum::middleware::from_fn_with_state(allow_hosts, guard_host));
        }
        let router = router.layer(axum::middleware::from_fn(log_request)).layer(
            TraceLayer::new_for_http()
                // Without headers (the default), so a token never reaches the log.
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO).include_headers(false))
                .on_response(DefaultOnResponse::new().level(Level::DEBUG)),
        );
        Server { router, shutdown, store, token }
    }

    /// The router (cheap clone), e.g. for `tower::ServiceExt::oneshot` in tests.
    pub fn router(&self) -> Router {
        self.router.clone()
    }

    /// Ends every open SSE stream (called on shutdown so graceful shutdown does not wait on them forever).
    pub fn close_streams(&self) {
        self.shutdown.send_replace(true);
    }

    /// Serves until `signal` resolves, then ends open SSE streams, waits for in-flight requests and shuts the
    /// store down ([`Store::shutdown`]: the real store stops an active training run at its next batch, keeping the
    /// last checkpoint).
    pub async fn serve(
        self,
        listener: TcpListener,
        signal: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        if !self.token
            && let Ok(addr) = listener.local_addr()
            && !addr.ip().is_loopback()
        {
            tracing::warn!(
                %addr,
                "the API has no token and listens beyond loopback: anyone who can reach this port can use it \
                 (start the server with --token)"
            );
        }
        let shutdown = self.shutdown.clone();
        let served = axum::serve(listener, self.router)
            .with_graceful_shutdown(async move {
                signal.await;
                tracing::info!("shutting down");
                shutdown.send_replace(true);
            })
            .await;
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || store.shutdown()).await;
        served
    }
}

/// Resolves on Ctrl+C (and SIGTERM on Unix).
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::ServiceExt;

    static WEB: &[web::EmbeddedFile] = &[web::EmbeddedFile {
        path: "index.html",
        bytes: b"<!doctype html><html><head><title>FOLO Studio</title></head><body><div id=app></div></body></html>",
        etag: "W/\"i1\"",
    }];

    async fn web_request(method: Method, headers: &[(header::HeaderName, &str)]) -> Response {
        let server = Server::with_embedded(Arc::new(MockStore::new()), Options::default(), Some(WEB));
        let mut req = Request::builder().method(method).uri("/runs/x");
        for (k, v) in headers {
            req = req.header(k, *v);
        }
        server.router().oneshot(req.body(Body::empty()).unwrap()).await.unwrap()
    }

    #[tokio::test]
    async fn embedded_head_matches_compressed_get() {
        let gzip = [(header::ACCEPT_ENCODING, "gzip")];
        let get = web_request(Method::GET, &gzip).await;
        let head = web_request(Method::HEAD, &gzip).await;
        for res in [&get, &head] {
            assert_eq!(res.status(), StatusCode::OK);
            assert_eq!(res.headers()[header::CONTENT_ENCODING], "gzip");
            assert_eq!(res.headers()[header::ETAG], "W/\"i1\"");
            assert!(res.headers().get_all(header::VARY).iter().any(|v| v == "accept-encoding"));
        }
        let plain = web_request(Method::HEAD, &[]).await;
        assert!(plain.headers().get(header::CONTENT_ENCODING).is_none());
        assert_eq!(plain.headers()[header::CONTENT_LENGTH], WEB[0].bytes.len().to_string());

        let not_modified = web_request(
            Method::GET,
            &[(header::IF_NONE_MATCH, "W/\"i1\""), (header::ACCEPT_ENCODING, "gzip")],
        )
        .await;
        assert_eq!(not_modified.status(), StatusCode::NOT_MODIFIED);
        assert!(not_modified.headers().get_all(header::VARY).iter().any(|v| v == "accept-encoding"));
    }

    #[test]
    fn loopback_origins() {
        let ok = |s: &str| is_loopback_origin(&HeaderValue::from_str(s).unwrap());
        assert!(ok("http://localhost:5173"));
        assert!(ok("http://127.0.0.1:7878"));
        assert!(ok("http://[::1]:5173"));
        assert!(ok("https://localhost"));
        assert!(!ok("http://localhost.evil.com"));
        assert!(!ok("https://example.com"));
        assert!(!ok("null"));
    }
}
