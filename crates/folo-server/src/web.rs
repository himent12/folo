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
    /// Quoted strong ETag derived from the contents.
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

fn find(files: &'static [EmbeddedFile], path: &str) -> Option<&'static EmbeddedFile> {
    files.binary_search_by(|f| f.path.cmp(path)).ok().map(|i| &files[i])
}

/// Answers a non-API request from `files` (sorted by path).
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
    ];
    let fresh = headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()).is_some_and(|v| {
        v.split(',').map(|t| t.trim()).any(|t| t == "*" || t.trim_start_matches("W/") == file.etag)
    });
    if fresh {
        return (StatusCode::NOT_MODIFIED, common).into_response();
    }
    let body = if method == Method::HEAD { &[][..] } else { file.bytes };
    let mut res =
        (StatusCode::OK, common, [(header::CONTENT_TYPE, content_type(file.path))], body).into_response();
    res.headers_mut().insert(header::CONTENT_LENGTH, HeaderValue::from(file.bytes.len()));
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    static FILES: &[EmbeddedFile] = &[
        EmbeddedFile { path: "assets/app-1a2b.js", bytes: b"console.log(1)", etag: "\"a1\"" },
        EmbeddedFile { path: "favicon.svg", bytes: b"<svg/>", etag: "\"f1\"" },
        EmbeddedFile { path: "index.html", bytes: b"<!doctype html>Studio", etag: "\"i1\"" },
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
        assert_eq!(find(FILES, "favicon.svg").unwrap().etag, "\"f1\"");
        assert!(find(FILES, "missing").is_none());
    }

    #[tokio::test]
    async fn serves_files_with_types_and_cache_headers() {
        let res = get("/assets/app-1a2b.js?v=1", &[]);
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "text/javascript; charset=utf-8");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "public, max-age=31536000, immutable");
        assert_eq!(res.headers()[header::CONTENT_LENGTH], "14");
        assert_eq!(body(res).await, b"console.log(1)");

        let res = get("/", &[]);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "text/html; charset=utf-8");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
        assert_eq!(res.headers()[header::ETAG], "\"i1\"");
        assert_eq!(body(res).await, b"<!doctype html>Studio");
    }

    #[tokio::test]
    async fn spa_fallback_etag_head_and_methods() {
        let res = get("/runs/run_1a01", &[]);
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body(res).await, b"<!doctype html>Studio", "unknown paths get index.html");

        let res = get("/favicon.svg", &[(header::IF_NONE_MATCH, "\"x\", W/\"f1\"")]);
        assert_eq!(res.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(res.headers()[header::ETAG], "\"f1\"");
        assert!(body(res).await.is_empty());
        assert_eq!(get("/favicon.svg", &[(header::IF_NONE_MATCH, "\"old\"")]).status(), StatusCode::OK);

        let uri: Uri = "/favicon.svg".parse().unwrap();
        let res = respond(FILES, &Method::HEAD, &uri, &HeaderMap::new());
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CONTENT_LENGTH], "6");
        assert!(body(res).await.is_empty());

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
