//! FOLO Studio compiled into the binary (the `embed-web` feature), served like `web/dist` on disk: an SPA fallback
//! to `index.html`, long-lived caching for Vite's hashed `assets/`, `no-cache` plus an ETag for everything else.

use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};

/// One embedded file of the built web app.
#[derive(Debug)]
pub struct EmbeddedFile {
    /// Path relative to the dist folder, `/`-separated (e.g. `assets/index-3f2a.js`).
    pub path: &'static str,
    /// File contents.
    pub bytes: &'static [u8],
    /// Weak ETag derived from the contents (`W/"..."`): weak because the compression layer may re-encode the body,
    /// and a strong validator must differ between encodings.
    pub etag: &'static str,
}

#[cfg(feature = "embed-web")]
static FILES: &[EmbeddedFile] = include!(concat!(env!("OUT_DIR"), "/web_files.rs"));

/// The embedded web app (sorted by path), or `None` when built without the `embed-web` feature.
pub fn embedded() -> Option<&'static [EmbeddedFile]> {
    #[cfg(feature = "embed-web")]
    return Some(FILES);
    #[cfg(not(feature = "embed-web"))]
    None
}

/// `Content-Type` for the file types a Vite build emits.
pub(crate) fn content_type(path: &str) -> &'static str {
    let ext = path.rsplit_once('.').map_or("", |(_, e)| e);
    match ext.to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        "xml" => "application/xml",
        _ => "application/octet-stream",
    }
}

/// Whether the request's `If-None-Match` matches `etag` (`None` without the header). It uses the weak comparison
/// (RFC 9110 §13.1.2): opaque tags match regardless of `W/`, and `*` matches anything.
pub(crate) fn if_none_match(headers: &HeaderMap, etag: &str) -> Option<bool> {
    let opaque = |t: &str| t.trim().trim_start_matches("W/").to_string();
    let mut values = headers.get_all(header::IF_NONE_MATCH).iter().peekable();
    values.peek()?;
    Some(
        values
            .filter_map(|v| v.to_str().ok())
            .any(|v| v.split(',').any(|t| t.trim() == "*" || opaque(t) == opaque(etag))),
    )
}

fn find(files: &'static [EmbeddedFile], path: &str) -> Option<&'static EmbeddedFile> {
    files.binary_search_by(|f| f.path.cmp(path)).ok().map(|i| &files[i])
}

/// Answers a non-API request from `files` (sorted by path). HEAD gets the full GET response so the compression
/// layer picks the same encoding and headers; hyper drops the body on the wire.
pub(crate) fn respond(
    files: &'static [EmbeddedFile],
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "GET, HEAD")]).into_response();
    }
    let rel = uri.path().trim_start_matches('/');
    let rel = if rel.is_empty() || rel.ends_with('/') { format!("{rel}index.html") } else { rel.to_string() };
    let Some(file) = find(files, &rel).or_else(|| find(files, "index.html")) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    // Vite content-hashes everything under assets/, so those never change under the same name.
    let cache =
        if file.path.starts_with("assets/") { "public, max-age=31536000, immutable" } else { "no-cache" };
    let common = [
        (header::ETAG, HeaderValue::from_static(file.etag)),
        (header::CACHE_CONTROL, HeaderValue::from_static(cache)),
        // The body may be gzip-encoded by the compression layer; caches must key on the request's encoding.
        (header::VARY, HeaderValue::from_static("accept-encoding")),
    ];
    if if_none_match(headers, file.etag) == Some(true) {
        return (StatusCode::NOT_MODIFIED, common).into_response();
    }
    (StatusCode::OK, common, [(header::CONTENT_TYPE, content_type(file.path))], file.bytes).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    static FILES: &[EmbeddedFile] = &[
        EmbeddedFile { path: "assets/app-1a2b.js", bytes: b"console.log(1)", etag: "W/\"a1\"" },
        EmbeddedFile { path: "favicon.svg", bytes: b"<svg/>", etag: "W/\"f1\"" },
        EmbeddedFile { path: "index.html", bytes: b"<!doctype html>Studio", etag: "W/\"i1\"" },
    ];

    fn get(path: &str, headers: &[(header::HeaderName, &str)]) -> Response {
        let mut h = HeaderMap::new();
        for (k, v) in headers {
            h.insert(k, HeaderValue::from_str(v).unwrap());
        }
        respond(FILES, &Method::GET, &path.parse().unwrap(), &h)
    }

    async fn body(res: Response) -> Vec<u8> {
        axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap().to_vec()
    }

    #[test]
    fn table_lookup_matches_a_sorted_slice() {
        assert!(FILES.windows(2).all(|w| w[0].path < w[1].path));
        assert_eq!(find(FILES, "favicon.svg").unwrap().etag, "W/\"f1\"");
        assert!(find(FILES, "missing").is_none());
    }

    #[tokio::test]
    async fn serves_files_with_types_and_cache_headers() {
        let res = get("/assets/app-1a2b.js?v=1", &[]);
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "text/javascript; charset=utf-8");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "public, max-age=31536000, immutable");
        assert_eq!(res.headers()[header::VARY], "accept-encoding");
        assert_eq!(body(res).await, b"console.log(1)");

        let res = get("/", &[]);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "text/html; charset=utf-8");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
        assert_eq!(res.headers()[header::ETAG], "W/\"i1\"");
        assert_eq!(body(res).await, b"<!doctype html>Studio");
    }

    #[tokio::test]
    async fn spa_fallback_etag_head_and_methods() {
        let res = get("/runs/run_1a01", &[]);
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body(res).await, b"<!doctype html>Studio", "unknown paths get index.html");

        for tag in ["\"x\", W/\"f1\"", "\"f1\"", "*"] {
            assert_eq!(
                get("/favicon.svg", &[(header::IF_NONE_MATCH, tag)]).status(),
                StatusCode::NOT_MODIFIED,
                "{tag}"
            );
        }
        let res = get("/favicon.svg", &[(header::IF_NONE_MATCH, "W/\"f1\"")]);
        assert_eq!(res.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(res.headers()[header::ETAG], "W/\"f1\"");
        assert_eq!(res.headers()[header::VARY], "accept-encoding");
        assert!(body(res).await.is_empty());
        assert_eq!(get("/favicon.svg", &[(header::IF_NONE_MATCH, "\"old\"")]).status(), StatusCode::OK);

        let uri: Uri = "/favicon.svg".parse().unwrap();
        // HEAD answers like GET (hyper strips the body), so compression sees the same representation.
        let res = respond(FILES, &Method::HEAD, &uri, &HeaderMap::new());
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "image/svg+xml");
        assert_eq!(body(res).await, b"<svg/>");

        let res = respond(FILES, &Method::POST, &uri, &HeaderMap::new());
        assert_eq!(res.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(res.headers()[header::ALLOW], "GET, HEAD");
    }

    #[test]
    fn content_types() {
        assert_eq!(content_type("a/b.WOFF2"), "font/woff2");
        assert_eq!(content_type("x.css"), "text/css; charset=utf-8");
        assert_eq!(content_type("noext"), "application/octet-stream");
    }

    #[cfg(feature = "embed-web")]
    #[test]
    fn embedded_build_has_an_index() {
        let files = embedded().unwrap();
        assert!(find(files, "index.html").is_some());
        assert!(files.windows(2).all(|w| w[0].path < w[1].path), "build.rs sorts the table");
    }
}
