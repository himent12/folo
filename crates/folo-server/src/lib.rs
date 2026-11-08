//! FOLO Studio server: HTTP + SSE API for datasets, training runs and inference.
//!
//! Implements the contract in `docs/API.md` on axum. Handlers sit on a [`Store`]: [`MockStore`] serves deterministic
//! synthetic data (`folo serve --mock`), [`RealStore`] serves datasets from disk and answers `501 not_implemented`
//! for training and inference until the engine is wired in. The request / response types live in the `folo-api`
//! crate (re-exported as [`api`]) so the CLI and FOLO Desktop share them.
//!
//! See `docs/ARCHITECTURE.md` for how this crate fits into FOLO.

mod disk;
pub mod error;
pub mod mock;
mod raster;
pub mod real;
mod routes;
mod stats;
pub mod store;
pub mod system;
mod util;

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::IntoResponse;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tower_http::compression::CompressionLayer;
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

/// Server options.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Folder with the built web app (`web/dist`). Served with an SPA fallback to `index.html` when it holds an
    /// `index.html`; otherwise non-API routes answer 404 with a hint.
    pub web_dir: Option<PathBuf>,
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

async fn web_not_built() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        "FOLO Studio is not built. Run `npm ci && npm run build` in web/ (or pass --web-dir), \
         or use the Vite dev server on :5173. The API is under /api.\n",
    )
}

impl Server {
    /// Builds the router around `store`.
    pub fn new(store: Arc<dyn Store>, opts: Options) -> Server {
        let (tx, _) = watch::channel(false);
        let shutdown = Arc::new(tx);
        let state =
            AppState { store, probe: Arc::new(system::SystemProbe::default()), shutdown: shutdown.clone() };
        let mut router = Router::new().nest("/api", routes::api()).with_state(state);
        router = match opts.web_dir.as_deref().filter(|d| web_dir_ready(d)) {
            Some(dir) => {
                let spa = ServeDir::new(dir).fallback(ServeFile::new(dir.join("index.html")));
                router.fallback_service(spa)
            }
            None => router.fallback(web_not_built),
        };
        let cors = match opts.cors {
            Cors::Any => CorsLayer::permissive(),
            Cors::Loopback => CorsLayer::new()
                .allow_origin(AllowOrigin::predicate(|o, _| is_loopback_origin(o)))
                .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
                .allow_headers(tower_http::cors::Any),
        };
        let router = router.layer(CompressionLayer::new()).layer(cors).layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        );
        Server { router, shutdown }
    }

    /// The router (cheap clone), e.g. for `tower::ServiceExt::oneshot` in tests.
    pub fn router(&self) -> Router {
        self.router.clone()
    }

    /// Ends every open SSE stream (called on shutdown so graceful shutdown does not wait on them forever).
    pub fn close_streams(&self) {
        self.shutdown.send_replace(true);
    }

    /// Serves until `signal` resolves, then ends open SSE streams and waits for in-flight requests.
    pub async fn serve(
        self,
        listener: TcpListener,
        signal: impl Future<Output = ()> + Send + 'static,
    ) -> std::io::Result<()> {
        let shutdown = self.shutdown.clone();
        axum::serve(listener, self.router)
            .with_graceful_shutdown(async move {
                signal.await;
                tracing::info!("shutting down");
                shutdown.send_replace(true);
            })
            .await
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
