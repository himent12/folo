//! FOLO Studio compiled into the binary (the `embed-web` feature), served like `web/dist` on disk: an SPA fallback
//! to `index.html`, long-lived caching for Vite's hashed `assets/`, `no-cache` plus an ETag for everything else.
//!
//! A folder (`--web-dir`) keeps a file for a year only under Vite's hashed name while its `index.html` is a Vite page
//! ([`hashed_name`], [`dir_is_vite_build`]). A name merely shaped like a hash, in a folder whose page loads such a
//! module, is trusted.

use std::path::Path;

use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use tokio::io::AsyncReadExt;

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

/// The files of folo-wasm (FOLO in the browser, docs/WASM.md) inside a Studio build, relative to the dist folder:
/// scripts/build-wasm writes them to `web/public/folo-wasm/` and `vite build` copies them to `dist/folo-wasm/`.
/// Without them the Playground can only run models on the server.
pub const FOLO_WASM_FILES: [&str; 2] = ["folo-wasm/folo_wasm.js", "folo-wasm/folo_wasm_bg.wasm"];

/// Whether `files` (sorted by path, like [`embedded`]) include folo-wasm.
pub fn has_folo_wasm(files: &[EmbeddedFile]) -> bool {
    FOLO_WASM_FILES.iter().all(|p| find(files, p).is_some())
}

/// Whether the web app folder `dir` (e.g. `web/dist`) includes folo-wasm.
pub fn dir_has_folo_wasm(dir: &Path) -> bool {
    FOLO_WASM_FILES.iter().all(|p| dir.join(p).is_file())
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

/// The `Cache-Control` of the web app's file at `path` (relative to the dist folder, `/`-separated), embedded or on
/// disk (`--web-dir`): Vite content-hashes everything under `assets/`, so those never change under the same name and
/// are cached for a year; everything else (`index.html`, the SPA fallback, the unhashed files of `public/`) is
/// revalidated on every use, so a new build is picked up at once. A source map is named after its chunk's hash, not
/// its own, so a change to a comment changes the map under the same name: it is revalidated too, whatever the case of
/// its extension (a case-insensitive volume serves `.MAP` from `.map`). The rule assumes `assets/` holds only Vite's
/// hashed output: `web/public` has no `assets/` folder, whose files Vite would copy there under their own names.
pub(crate) fn cache_control(path: &str) -> &'static str {
    let map = path.get(path.len().saturating_sub(4)..).is_some_and(|e| e.eq_ignore_ascii_case(".map"));
    if path.starts_with("assets/") && !map { "public, max-age=31536000, immutable" } else { "no-cache" }
}

/// How much of a folder's `index.html` is read to tell a Vite page. Vite puts the entry module in `<head>`: in Studio's
/// build its tag ends at byte 6,757, after the inline first-paint script, so this leaves about nine times that. A page
/// whose tag ends later is not taken for a Vite page, so its files are only `no-cache`.
pub const VITE_HEAD_BYTES: usize = 64 * 1024;

/// Whether `path` (relative to the folder) is Vite's hashed form: `assets/<name>-<8 base64url characters>.<ext>`.
pub(crate) fn hashed_name(path: &str) -> bool {
    let Some(file) = path.strip_prefix("assets/") else { return false };
    let Some((stem, ext)) = file.rsplit_once('.') else { return false };
    if file.contains('/') || ext.is_empty() || !stem.is_ascii() || stem.len() < 10 {
        return false;
    }
    let (name, hash) = stem.split_at(stem.len() - 8);
    name.ends_with('-') && hash.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// Whether `html` loads a module script with a hashed name, in a whole tag (one cut by the read's bound does not count).
fn is_vite_page(html: &str) -> bool {
    html.split("<script").skip(1).any(|rest| {
        let Some((tag, _)) = rest.split_once('>') else { return false };
        let src = tag.split_once(r#"src=""#).and_then(|(_, s)| s.split_once('"')).map(|(src, _)| src);
        tag.contains(r#"type="module""#)
            && src.is_some_and(|s| hashed_name(s.trim_start_matches("./").trim_start_matches('/')))
    })
}

/// Whether the folder's current `index.html` is a Vite page, from its first [`VITE_HEAD_BYTES`] bytes.
pub(crate) async fn dir_is_vite_build(dir: &Path) -> bool {
    let Ok(file) = tokio::fs::File::open(dir.join("index.html")).await else { return false };
    let mut head = Vec::new();
    if file.take(VITE_HEAD_BYTES as u64).read_to_end(&mut head).await.is_err() {
        return false;
    }
    is_vite_page(&String::from_utf8_lossy(&head))
}

fn trim_http_ws(mut s: &[u8]) -> &[u8] {
    while let Some((&b, rest)) = s.split_first() {
        if b == b' ' || b == b'\t' {
            s = rest;
        } else {
            break;
        }
    }
    while let Some((&b, rest)) = s.split_last() {
        if b == b' ' || b == b'\t' {
            s = rest;
        } else {
            break;
        }
    }
    s
}

/// Splits an HTTP entity-tag header (`If-None-Match`, `If-Match`) on commas outside quotes so a quoted opaque tag
/// containing `,` (RFC 9110 §8.8.3) stays whole.
fn etag_list(raw: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    for (i, &b) in raw.iter().enumerate() {
        match b {
            b'"' => in_quotes = !in_quotes,
            b',' if !in_quotes => {
                let item = trim_http_ws(&raw[start..i]);
                if !item.is_empty() {
                    out.push(item);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = trim_http_ws(&raw[start..]);
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn is_quoted_tag(t: &[u8]) -> bool {
    t.len() >= 2 && t.starts_with(b"\"") && t.ends_with(b"\"")
}

fn opaque_tag(t: &[u8]) -> &[u8] {
    t.strip_prefix(b"W/").unwrap_or(t)
}

/// Whether the request's `If-None-Match` matches `etag` (`None` without the header). Uses the weak comparison
/// (RFC 9110 §13.1.2): opaque tags match regardless of `W/`, and `*` matches only as the sole field value.
pub(crate) fn if_none_match(headers: &HeaderMap, etag: &str) -> Option<bool> {
    let target = opaque_tag(trim_http_ws(etag.as_bytes()));
    let mut values = headers.get_all(header::IF_NONE_MATCH).iter().peekable();
    values.peek()?;
    Some(values.any(|v| {
        let raw = trim_http_ws(v.as_bytes());
        raw == b"*"
            || etag_list(raw).into_iter().any(|t| {
                let o = opaque_tag(t);
                is_quoted_tag(o) && o == target
            })
    }))
}

/// False when the request has `If-Match` and it names neither `etag` nor `*` (strong comparison, RFC 9110 §13.1.1:
/// a weak tag never matches).
pub(crate) fn if_match(headers: &HeaderMap, etag: &str) -> bool {
    let target = trim_http_ws(etag.as_bytes());
    let mut values = headers.get_all(header::IF_MATCH).iter().peekable();
    if values.peek().is_none() {
        return true;
    }
    let strong = is_quoted_tag(target);
    values.any(|v| {
        let raw = trim_http_ws(v.as_bytes());
        raw == b"*" || (strong && etag_list(raw).into_iter().any(|t| is_quoted_tag(t) && t == target))
    })
}

fn find<'a>(files: &'a [EmbeddedFile], path: &str) -> Option<&'a EmbeddedFile> {
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
    let cache = cache_control(file.path);
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
    use folo_test_support::TestDir;

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
    fn etag_matching_respects_quotes_wildcards_and_non_ascii() {
        let mut h = HeaderMap::new();
        h.insert(header::IF_NONE_MATCH, HeaderValue::from_static("\"old,*,tag\""));
        assert_eq!(if_none_match(&h, "\"f1\""), Some(false), "comma-quoted * is not a wildcard");

        let mut h = HeaderMap::new();
        h.insert(header::IF_NONE_MATCH, HeaderValue::from_bytes(b"\"non-\xff-ascii\", \"f1\"").unwrap());
        assert_eq!(if_none_match(&h, "\"f1\""), Some(true), "non-ASCII sibling tag does not hide a match");

        let mut h = HeaderMap::new();
        h.insert(header::IF_NONE_MATCH, HeaderValue::from_static("W/W/\"f1\""));
        assert_eq!(if_none_match(&h, "\"f1\""), Some(false), "repeated W/ prefix is malformed");

        let mut h = HeaderMap::new();
        h.insert(header::IF_MATCH, HeaderValue::from_static("W/\"f1\""));
        assert!(!if_match(&h, "\"f1\""), "If-Match uses strong comparison");
        h.insert(header::IF_MATCH, HeaderValue::from_static("\"a,b\", \"f1\""));
        assert!(if_match(&h, "\"f1\""));
        h.insert(header::IF_MATCH, HeaderValue::from_static("\"old,*,tag\""));
        assert!(!if_match(&h, "\"f1\""));
    }

    #[test]
    fn folo_wasm_presence() {
        assert!(!has_folo_wasm(FILES));
        let with: &[EmbeddedFile] = &[
            EmbeddedFile { path: "folo-wasm/folo_wasm.js", bytes: b"", etag: "W/\"j\"" },
            EmbeddedFile { path: "folo-wasm/folo_wasm_bg.wasm", bytes: b"", etag: "W/\"w\"" },
            EmbeddedFile { path: "index.html", bytes: b"", etag: "W/\"i\"" },
        ];
        assert!(has_folo_wasm(with));
        assert!(!has_folo_wasm(&with[1..]), "the module needs its JS glue");

        let dir = TestDir::new("web-wasm");
        std::fs::create_dir_all(dir.join("folo-wasm")).unwrap();
        assert!(!dir_has_folo_wasm(&dir));
        std::fs::write(dir.join(FOLO_WASM_FILES[0]), b"").unwrap();
        assert!(!dir_has_folo_wasm(&dir));
        std::fs::write(dir.join(FOLO_WASM_FILES[1]), b"").unwrap();
        assert!(dir_has_folo_wasm(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn vite_hashed_names() {
        for p in ["assets/index-xjBsy1zZ.js", "assets/a-b-1a2b3c4d.css", "assets/app-Ab_c-123.woff2"] {
            assert!(hashed_name(p), "{p}");
        }
        for p in [
            "index.html",
            "assets/app.js",
            "assets/logo.svg",
            "assets/app-1a2b3c4.js",
            "assets/app1a2b3c4d.js",
            "assets/x/app-1a2b3c4d.js",
            "assets/app-1a2b3c4d",
            "assets/app-1a2b3c4d.js.map",
        ] {
            assert!(!hashed_name(p), "{p}");
        }
    }

    #[test]
    fn a_vite_page_loads_a_hashed_module() {
        for page in [
            r#"<script type="module" crossorigin src="/assets/app-1a2b3c4d.js"></script>"#,
            r#"<script src="./assets/app-1a2b3c4d.js" type="module"></script>"#,
        ] {
            assert!(is_vite_page(page), "{page}");
        }
        for page in [
            r#"<script src="/assets/app-1a2b3c4d.js"></script>"#,
            r#"<script type="module" src="/assets/app.js"></script>"#,
            r#"<script type="module">import "/assets/app-1a2b3c4d.js"</script>"#,
            // Cut by the read's bound: inside the tag, and inside its `src`.
            r#"<script type="module" src="/assets/app-1a2b3c4d.js""#,
            r#"<script type="module" src="/assets/app-1a2b3c4d.js"#,
        ] {
            assert!(!is_vite_page(page), "{page}");
        }
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
