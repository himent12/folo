//! HTTP around the API: static files, CORS and compression, origins and hosts, malformed requests.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{Cors, MockStore, Options, Server, WebSource};
use folo_test_support::TestDir;
use serde_json::json;
use tower::ServiceExt;

use super::harness::{err, get, mock, ok, send};

#[tokio::test]
async fn static_files_spa_fallback() {
    let dir = TestDir::new("web");
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    std::fs::write(dir.join("index.html"), "<!doctype html><title>Studio</title>").unwrap();
    std::fs::write(dir.join("assets/app.js"), "console.log(1)").unwrap();
    let server = Server::new(
        Arc::new(MockStore::new()),
        Options {
            web: WebSource::Dir(dir.to_path_buf()),
            cors: Cors::Loopback,
            token: None,
            ..Options::default()
        },
    );
    let app = server.router();
    let (s, _, b) = send(&app, get("/")).await;
    assert_eq!(s, StatusCode::OK);
    assert!(String::from_utf8_lossy(&b).contains("Studio"));
    let (s, h, b) = send(&app, get("/assets/app.js")).await;
    assert_eq!((s, b.as_slice()), (StatusCode::OK, b"console.log(1)".as_slice()));
    assert!(h[header::CONTENT_TYPE].to_str().unwrap().contains("javascript"));
    let (s, _, b) = send(&app, get("/runs/run_1a01")).await;
    assert_eq!(s, StatusCode::OK);
    assert!(String::from_utf8_lossy(&b).contains("Studio"), "SPA fallback");
    // `web_wasm` only when the served build has the in-browser runtime.
    let wasm = |app: &axum::Router| {
        let app = app.clone();
        async move {
            let s: SystemResponse = ok(&app, get("/api/system")).await;
            s.features.iter().any(|f| f == FEATURE_WEB_WASM)
        }
    };
    assert!(!wasm(&app).await);
    std::fs::create_dir_all(dir.join("folo-wasm")).unwrap();
    for f in ["folo-wasm/folo_wasm.js", "folo-wasm/folo_wasm_bg.wasm"] {
        std::fs::write(dir.join(f), "x").unwrap();
    }
    let with = Server::new(
        Arc::new(MockStore::new()),
        Options {
            web: WebSource::Dir(dir.to_path_buf()),
            cors: Cors::Loopback,
            token: None,
            ..Options::default()
        },
    );
    assert!(wasm(&with.router()).await);
    err(&app, get("/api/not-a-route"), StatusCode::NOT_FOUND, "route_not_found").await;

    // Without a built web app: a plain 404 with a hint, API unaffected.
    let bare = Server::new(Arc::new(MockStore::new()), Options::default()).router();
    let (s, _, b) = send(&bare, get("/")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert!(String::from_utf8_lossy(&b).contains("not built"));
    std::fs::remove_dir_all(&dir).unwrap();

    // The embedded web app (only present in `embed-web` builds) serves the same way.
    let embedded = Server::new(
        Arc::new(MockStore::new()),
        Options { web: WebSource::Embedded, cors: Cors::Loopback, token: None, ..Options::default() },
    )
    .router();
    let (s, h, b) = send(&embedded, get("/runs/run_1a01")).await;
    if folo_server::web::embedded().is_some() {
        assert_eq!(s, StatusCode::OK);
        assert!(h[header::CONTENT_TYPE].to_str().unwrap().starts_with("text/html"));
        assert!(String::from_utf8_lossy(&b).to_ascii_lowercase().contains("<html"), "SPA fallback");
    } else {
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert!(String::from_utf8_lossy(&b).contains("embed-web"));
    }
    err(&embedded, get("/api/not-a-route"), StatusCode::NOT_FOUND, "route_not_found").await;
}

/// `folo serve --web-dir` caches as the embedded app does (issue #309): Vite's hashed `assets/` for a year, everything
/// else revalidated (`no-cache` with `Last-Modified`), and an asset name that is no file falls back to `index.html`
/// without being kept.
#[tokio::test]
async fn web_dir_cache_headers() {
    let dir = TestDir::new("web-cache");
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    std::fs::write(dir.join("index.html"), "<!doctype html><title>Studio</title>").unwrap();
    std::fs::write(dir.join("assets/app-1a2b3c.js"), "console.log(1)").unwrap();
    std::fs::write(dir.join("favicon.svg"), "<svg/>").unwrap();
    let app = Server::new(
        Arc::new(MockStore::new()),
        Options { web: WebSource::Dir(dir.to_path_buf()), ..Options::default() },
    )
    .router();
    let cache = |h: &axum::http::HeaderMap| h[header::CACHE_CONTROL].to_str().unwrap().to_string();
    let immutable = "public, max-age=31536000, immutable";

    let (s, h, b) = send(&app, get("/assets/app-1a2b3c.js")).await;
    assert_eq!((s, b.as_slice()), (StatusCode::OK, b"console.log(1)".as_slice()));
    assert_eq!(cache(&h), immutable);
    for path in [
        "/",
        "/index.html",
        "/runs/run_1a01",
        "/favicon.svg",
        "/assets/app-old999.js",
        "/assets/../index.html",
    ] {
        let (s, h, _) = send(&app, get(path)).await;
        assert_eq!(s, StatusCode::OK, "{path}");
        assert_eq!(cache(&h), "no-cache", "{path}");
        assert!(h.contains_key(header::LAST_MODIFIED), "{path} has a validator");
    }

    // Revalidated: an unchanged file answers 304, still `no-cache`; the fallback for a gone asset never turns immutable.
    let (_, h, _) = send(&app, get("/")).await;
    let since = h[header::LAST_MODIFIED].to_str().unwrap().to_string();
    for path in ["/", "/assets/app-old999.js"] {
        let req = Request::builder()
            .uri(path)
            .header(header::IF_MODIFIED_SINCE, &since)
            .body(Body::empty())
            .unwrap();
        let (s, h, b) = send(&app, req).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED, "{path}");
        assert_eq!(cache(&h), "no-cache", "{path}");
        assert!(b.is_empty());
    }
    // An API answer is not the web app's to cache.
    let (_, h, _) = send(&app, get("/api/health")).await;
    assert_ne!(h.get(header::CACHE_CONTROL).map(|v| v.to_str().unwrap()), Some(immutable));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn cors_and_gzip() {
    let (server, _) = mock();
    let app = server.router();
    let preflight = |origin: &str| {
        Request::builder()
            .method("OPTIONS")
            .uri("/api/runs")
            .header(header::ORIGIN, origin)
            .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
            .header(header::ACCESS_CONTROL_REQUEST_HEADERS, "content-type")
            .body(Body::empty())
            .unwrap()
    };
    let (_, h, _) = send(&app, preflight("http://localhost:5173")).await;
    assert_eq!(h[header::ACCESS_CONTROL_ALLOW_ORIGIN], "http://localhost:5173");
    let (_, h, _) = send(&app, preflight("https://evil.example")).await;
    assert!(h.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none());

    let any = Server::new(
        Arc::new(MockStore::new()),
        Options { web: WebSource::None, cors: Cors::Any, token: None, ..Options::default() },
    )
    .router();
    let (_, h, _) = send(&any, preflight("https://elsewhere.example")).await;
    assert_eq!(h[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");

    let req = Request::get("/api/runs").header(header::ACCEPT_ENCODING, "gzip").body(Body::empty()).unwrap();
    let (s, h, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h[header::CONTENT_ENCODING], "gzip");

    // SSE is never compressed (it would buffer events).
    let list: RunList = ok(&app, get("/api/runs")).await;
    let req = Request::get(format!("/api/runs/{}/events", list.runs[0].id))
        .header(header::ACCEPT_ENCODING, "gzip")
        .body(Body::empty())
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert!(res.headers().get(header::CONTENT_ENCODING).is_none());
}

#[tokio::test]
async fn cross_origin_mutations_are_refused() {
    let (server, _) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap().id.clone();
    // A "simple" form POST needs no preflight, so CORS alone would let it through.
    let form_post = |origin: &str, host: &str| {
        Request::post(format!("/api/runs/{live}/stop"))
            .header(header::ORIGIN, origin)
            .header(header::HOST, host)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from("a=b"))
            .unwrap()
    };
    err(&app, form_post("https://evil.example", "127.0.0.1:7878"), StatusCode::FORBIDDEN, "forbidden_origin")
        .await;
    err(&app, form_post("null", "127.0.0.1:7878"), StatusCode::FORBIDDEN, "forbidden_origin").await;
    let r: Run = ok(&app, get(&format!("/api/runs/{live}"))).await;
    assert_eq!(r.status, RunStatus::Running, "the refused request had no effect");
    // Reads are fine from anywhere (CORS hides the response).
    let req = Request::get("/api/health")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK);
    // Same origin (Studio served by this server on a LAN address) and loopback origins may mutate.
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let create = |origin: &str, host: &str| {
        Request::post("/api/runs")
            .header(header::ORIGIN, origin)
            .header(header::HOST, host)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "dataset_id": ds.datasets[0].id }).to_string()))
            .unwrap()
    };
    let lan = Server::new(
        Arc::new(MockStore::new()),
        Options { bind: Some("0.0.0.0".parse().unwrap()), ..Options::default() },
    )
    .router();
    let (s, _, _) = send(&lan, create("http://192.168.1.5:7878", "192.168.1.5:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
    let (s, _, _) = send(&app, create("http://localhost:7878", "localhost:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
    let (s, _, _) = send(&app, create("http://localhost:5173", "127.0.0.1:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
    // --cors-any turns the guard off.
    let any = Server::new(
        Arc::new(MockStore::new()),
        Options { web: WebSource::None, cors: Cors::Any, token: None, ..Options::default() },
    )
    .router();
    let (s, _, _) = send(&any, create("https://elsewhere.example", "127.0.0.1:7878")).await;
    assert_eq!(s, StatusCode::CREATED);
}

/// DNS rebinding: a page on `evil.example` whose name re-resolves to 127.0.0.1 sends `Host: evil.example:7878`
/// (and a matching `Origin`). A server on loopback refuses every such request before a handler runs: reads,
/// changes, SSE and static files alike.
#[tokio::test]
async fn foreign_host_is_refused_on_a_loopback_server() {
    let (server, _) = mock();
    let app = server.router();
    let list: RunList = ok(&app, get("/api/runs")).await;
    let live = list.runs.iter().find(|r| r.status == RunStatus::Running).unwrap().id.clone();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let foreign = |req: axum::http::request::Builder| {
        req.header(header::HOST, "evil.example:7878").header(header::ORIGIN, "http://evil.example:7878")
    };
    for uri in ["/api/datasets", "/api/health", "/", "/runs/x"] {
        let e = err(
            &app,
            foreign(Request::get(uri)).body(Body::empty()).unwrap(),
            StatusCode::FORBIDDEN,
            "forbidden_host",
        )
        .await;
        let hint = e.hint.unwrap_or_default();
        assert!(hint.contains("--allow-host") && hint.contains("--token"), "{hint}");
    }
    let create = foreign(Request::post("/api/runs"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({ "dataset_id": ds.datasets[0].id }).to_string()))
        .unwrap();
    err(&app, create, StatusCode::FORBIDDEN, "forbidden_host").await;
    let sse = foreign(Request::get(format!("/api/runs/{live}/events"))).body(Body::empty()).unwrap();
    let res = app.clone().oneshot(sse).await.unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let after: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(after.runs.len(), list.runs.len(), "the refused POST created nothing");
    // CORS answers preflights itself: the host guard runs before it.
    let preflight = Request::builder()
        .method("OPTIONS")
        .uri("/api/runs")
        .header(header::HOST, "evil.example:7878")
        .header(header::ORIGIN, "http://localhost:5173")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .body(Body::empty())
        .unwrap();
    err(&app, preflight, StatusCode::FORBIDDEN, "forbidden_host").await;
    // Other spellings of a foreign name.
    for host in [
        "evil.example",
        "localhost.evil.example",
        "127.0.0.1.evil.example:7878",
        "[::2]:7878",
        "0.0.0.0:7878",
    ] {
        let req = Request::get("/api/health").header(header::HOST, host).body(Body::empty()).unwrap();
        err(&app, req, StatusCode::FORBIDDEN, "forbidden_host").await;
    }
}

/// What a server on loopback still answers: every spelling of a loopback name, no `Host` at all, and the
/// `--allow-host` names; a server bound beyond loopback (`--host 0.0.0.0`, a cloud worker) checks no `Host`. The
/// origin guard still applies to a request the host guard let through.
#[tokio::test]
async fn loopback_and_allowed_hosts_are_served() {
    let health = |host: Option<&str>| {
        let mut req = Request::get("/api/health");
        if let Some(h) = host {
            req = req.header(header::HOST, h);
        }
        req.body(Body::empty()).unwrap()
    };
    let loopback = [
        "localhost",
        "localhost:7878",
        "LOCALHOST:7878",
        "LocalHost",
        "localhost.",
        "localhost.:7878",
        "127.0.0.1",
        "127.0.0.1:7878",
        "127.1.2.3:80",
        "[::1]",
        "[::1]:7878",
    ];
    let (server, _) = mock();
    let app = server.router();
    for host in loopback {
        let (s, _, body) = send(&app, health(Some(host))).await;
        assert_eq!(s, StatusCode::OK, "{host}: {}", String::from_utf8_lossy(&body));
    }
    let (s, _, _) = send(&app, health(None)).await;
    assert_eq!(s, StatusCode::OK, "no Host");
    for host in ["localhost:x", "[::1", "[::1]x", ":7878", "localhost..", "127.0.0.1:99999"] {
        let req = health(Some(host));
        err(&app, req, StatusCode::FORBIDDEN, "forbidden_host").await;
    }
    for bind in ["127.0.0.1", "::1", "127.0.0.2"] {
        let app = Server::new(
            Arc::new(MockStore::new()),
            Options { bind: Some(bind.parse().unwrap()), ..Options::default() },
        )
        .router();
        err(&app, health(Some("evil.example")), StatusCode::FORBIDDEN, "forbidden_host").await;
        let (s, _, _) = send(&app, health(Some("127.0.0.1:7878"))).await;
        assert_eq!(s, StatusCode::OK, "bind {bind}");
    }

    // Beyond loopback: no Host check (the user exposed the server; docs/CLOUD.md says to use --token).
    for bind in ["0.0.0.0", "::", "192.168.1.5"] {
        let app = Server::new(
            Arc::new(MockStore::new()),
            Options { bind: Some(bind.parse().unwrap()), ..Options::default() },
        )
        .router();
        let (s, _, _) = send(&app, health(Some("evil.example:7878"))).await;
        assert_eq!(s, StatusCode::OK, "bind {bind}");
    }

    // --allow-host: a tunnel that keeps its own Host. Case, a trailing dot and a port do not matter.
    let tunnel = Server::new(
        Arc::new(MockStore::new()),
        Options {
            bind: Some("127.0.0.1".parse().unwrap()),
            allow_hosts: vec!["Folo.Tunnel.example".into(), "proxy.example:8443".into()],
            ..Options::default()
        },
    )
    .router();
    for host in [
        "folo.tunnel.example",
        "FOLO.TUNNEL.EXAMPLE:443",
        "folo.tunnel.example.",
        "proxy.example",
        "localhost",
    ] {
        let (s, _, _) = send(&tunnel, health(Some(host))).await;
        assert_eq!(s, StatusCode::OK, "{host}");
    }
    err(&tunnel, health(Some("other.tunnel.example")), StatusCode::FORBIDDEN, "forbidden_host").await;
    // ...and the origin guard runs after it: an allowed Host does not let a foreign page change state.
    let stop = |origin: &str| {
        Request::post("/api/runs/r_0001/stop")
            .header(header::HOST, "folo.tunnel.example")
            .header(header::ORIGIN, origin)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from("a=b"))
            .unwrap()
    };
    err(&tunnel, stop("https://evil.example"), StatusCode::FORBIDDEN, "forbidden_origin").await;
}

#[tokio::test]
async fn malformed_paths_and_oversized_uploads_use_the_envelope() {
    let (server, _) = mock();
    let app = server.router();
    err(&app, get("/api/datasets/%FF"), StatusCode::BAD_REQUEST, "invalid_path").await;
    err(&app, get("/api/runs/%FF/metrics"), StatusCode::BAD_REQUEST, "invalid_path").await;

    let boundary = "XfoloBoundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\nm_31aa\r\n--{boundary}\r\n\
         Content-Disposition: form-data; name=\"image\"; filename=\"big.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
    )
    .into_bytes();
    body.resize(body.len() + 65 * 1024 * 1024, 0);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let req = Request::post("/api/predict")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    err(&app, req, StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large").await;
}
