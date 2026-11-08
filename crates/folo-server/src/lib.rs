//! FOLO Studio server: HTTP + SSE API for datasets, training runs and inference.
//!
//! Implements the contract in `docs/API.md` on axum. Handlers sit on a [`Store`]: [`MockStore`] serves deterministic
//! synthetic data (`folo serve --mock`), [`RealStore`] serves datasets from disk (read with folo-data, indexed and
//! cached under the data dir), trains runs with folo-vision's trainer on a background worker, lists their models and
//! runs inference. The request / response types live in the `folo-api` crate (re-exported as [`api`]) so the CLI and
//! FOLO Desktop share them.
//!
//! See `docs/ARCHITECTURE.md` for how this crate fits into FOLO.

mod engine;
pub mod error;
mod index;
pub mod mock;
mod models;
mod raster;
pub mod real;
mod routes;
mod runs;
mod stats;
pub mod store;
pub mod system;
mod util;
pub mod web;

use std::future::Future;
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

/// Server options.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The web app to serve outside `/api`.
    pub web: WebSource,
    /// CORS policy.
    pub cors: Cors,
}

/// Shared handler state.
#[derive(Clone)]
pub struct AppState {
    store: Arc<dyn Store>,
    probe: Arc<system::SystemProbe>,
    shutdown: Arc<watch::Sender<bool>>,
}

/// A configured server: the router plus the shutdown switch that ends open SSE streams.
pub struct Server {
    router: Router,
    shutdown: Arc<watch::Sender<bool>>,
    store: Arc<dyn Store>,
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

/// CORS only hides responses; a browser still *sends* "simple" cross-origin POSTs (form posts) without a
/// preflight. So with [`Cors::Loopback`], state-changing requests whose `Origin` is neither loopback nor this
/// server itself are refused before they reach a handler. Requests without `Origin` (curl, agents, the desktop
/// app) are allowed.
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
        let state = AppState {
            store: store.clone(),
            probe: Arc::new(system::SystemProbe::default()),
            shutdown: shutdown.clone(),
        };
        let mut router = Router::new().nest("/api", routes::api()).with_state(state);
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
                .allow_methods([Method::GET, Method::HEAD, Method::POST, Method::DELETE, Method::OPTIONS])
                .allow_headers(tower_http::cors::Any)
                // Readable by page scripts (Studio caches downloaded weights by ETag).
                .expose_headers([header::ETAG, header::CONTENT_DISPOSITION]),
        };
        let router = router
            .layer(axum::middleware::from_fn_with_state(opts.cors, guard_origin))
            // Weights files (`application/octet-stream`) are sent as they are: f32 / f16 tensors barely compress,
            // and their Content-Length and strong ETag describe the file's own bytes.
            .layer(CompressionLayer::new().compress_when(
                DefaultPredicate::new().and(NotForContentType::const_new("application/octet-stream")),
            ))
            .layer(cors)
            .layer(
                TraceLayer::new_for_http()
                    .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                    .on_response(DefaultOnResponse::new().level(Level::INFO)),
            );
        Server { router, shutdown, store }
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
