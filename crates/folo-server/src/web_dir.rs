//! FOLO Studio from a folder (`folo serve --web-dir`, else `web/dist`), with an SPA fallback to `index.html`. Unlike the
//! embedded copy ([`crate::web`]), a folder can change while the server runs and need not be a Vite build, so:
//! - a file is kept for a year only when its name is Vite's content-hashed one ([`hashed_name`]) **and** the folder's
//!   `index.html` is a Vite build's ([`is_vite_page`]): a hand-made folder's `assets/` can change under their names;
//! - everything else (`index.html`, the fallback, source maps, any other file) is `no-cache`, and revalidates by an
//!   `ETag` of the file's content: a rebuild shows on the next plain reload, and an unchanged file costs a `304`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use tower_http::services::{ServeDir, ServeFile};

use crate::web;

/// The largest file given an `ETag`: the tag is a hash of the whole file, made on every request. A larger one is sent
/// in full each time.
const MAX_HASHED: u64 = 64 << 20;

/// The web app in the folder `dir`, which holds an `index.html`.
pub(crate) fn router(dir: &Path) -> Router {
    let index = dir.join("index.html");
    let fallback = axum::routing::any_service(ServeFile::new(&index))
        .layer(from_fn_with_state(Arc::new(index), fallback_validators));
    let spa = ServeDir::new(dir).fallback(fallback);
    Router::new()
        .fallback_service(spa)
        .layer(from_fn_with_state(Arc::new(dir.to_path_buf()), file_validators))
}

/// Whether `path` (relative to the folder) is named as Vite names a content-hashed file by default:
/// `assets/<name>-<8 characters of base64url>.<ext>`, directly in `assets/`.
fn hashed_name(path: &str) -> bool {
    let Some(file) = path.strip_prefix("assets/") else { return false };
    let Some((stem, ext)) = file.rsplit_once('.') else { return false };
    if file.contains('/') || ext.is_empty() || !stem.is_ascii() || stem.len() < 10 {
        return false;
    }
    let (name, hash) = stem.split_at(stem.len() - 8);
    name.ends_with('-') && hash.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// Whether `html` is the page of a Vite build: a module script loaded from a hashed name under `assets/`.
fn is_vite_page(html: &str) -> bool {
    html.split("<script").skip(1).any(|tag| {
        let tag = tag.split('>').next().unwrap_or("");
        tag.contains(r#"type="module""#)
            && tag
                .split(r#"src=""#)
                .nth(1)
                .and_then(|s| s.split('"').next())
                .is_some_and(|src| hashed_name(src.trim_start_matches("./").trim_start_matches('/')))
    })
}

/// A path that names exactly one file, also on a case-insensitive or Windows volume: only ASCII letters, digits, `.`,
/// `_`, `-` and `/`, no empty, `.` or `..` segment (so no trailing `/`, which `ServeDir` answers with a folder's
/// `index.html`) and no segment ending in `.` (no percent-encoding, `\`, `:` or `~` alias).
fn is_plain(path: &str) -> bool {
    path.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
        && path.split('/').all(|s| !s.is_empty() && !s.ends_with('.'))
}

/// 64-bit FNV-1a, as the embedded files' ETags (`build.rs`), continued from `hash` over `bytes`.
fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// The weak `ETag` of `file`'s content (weak because the compression layer may re-encode the body), `None` when it
/// cannot be read or is over [`MAX_HASHED`]. From the content, not the date and size: a build restored or unpacked with
/// fixed dates can change a file and keep both. Read in pieces, so a large file is not held in memory per request.
async fn etag_of(file: PathBuf) -> Option<HeaderValue> {
    tokio::task::spawn_blocking(move || {
        let mut f = std::fs::File::open(&file).ok()?;
        if f.metadata().ok()?.len() > MAX_HASHED {
            return None;
        }
        let (mut hash, mut len, mut piece) = (0xcbf2_9ce4_8422_2325, 0u64, vec![0; 64 << 10]);
        loop {
            let n = f.read(&mut piece).ok()?;
            if n == 0 {
                break;
            }
            hash = fnv1a(hash, &piece[..n]);
            len += n as u64;
        }
        HeaderValue::from_str(&format!("W/\"{len:x}-{hash:016x}\"")).ok()
    })
    .await
    .ok()
    .flatten()
}

fn stamp(headers: &mut HeaderMap, cache: &'static str, tag: Option<&HeaderValue>) {
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    // The body may be gzip-encoded by the compression layer; caches must key on the request's encoding.
    headers.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    if let Some(tag) = tag {
        headers.insert(header::ETAG, tag.clone());
    }
}

/// Answers `304` when the request holds `file`'s current tag; else what is inside, given `cache` and the tag when it is
/// a `200` or `206` no inner layer has marked (the fallback marks its own).
async fn validated(file: Option<PathBuf>, cache: &'static str, req: Request, next: Next) -> Response {
    let tag = match file {
        Some(file) if matches!(*req.method(), Method::GET | Method::HEAD) => etag_of(file).await,
        _ => None,
    };
    if let Some(t) = &tag
        && t.to_str().is_ok_and(|t| web::if_none_match(req.headers(), t) == Some(true))
    {
        let mut res = StatusCode::NOT_MODIFIED.into_response();
        stamp(res.headers_mut(), cache, tag.as_ref());
        return res;
    }
    let mut res = next.run(req).await;
    if matches!(res.status(), StatusCode::OK | StatusCode::PARTIAL_CONTENT)
        && !res.headers().contains_key(header::CACHE_CONTROL)
    {
        stamp(res.headers_mut(), cache, tag.as_ref());
    }
    res
}

/// The SPA fallback (`index.html` for a path that is no file): stored, and revalidated before every use.
async fn fallback_validators(State(index): State<Arc<PathBuf>>, req: Request, next: Next) -> Response {
    validated(Some(index.to_path_buf()), "no-cache", req, next).await
}

/// `ServeDir`'s own answers. A path that is not plain ([`is_plain`]) is sent `no-cache` without a tag, since which file
/// answers it is `ServeDir`'s to say; a redirect to a folder's `/` or an error is sent as it is.
///
/// `If-Modified-Since` is dropped: `ServeDir` compares dates only, so a date from another answer (the fallback's, cached
/// under an asset's name before that asset existed; a newer `index.html` before a rollback) would get a `304` for a
/// different file. `ServeDir` ignores `If-Range`, so a range asked for with one is answered whole: its bytes could
/// otherwise come from a file that changed since the part the client holds.
async fn file_validators(State(dir): State<Arc<PathBuf>>, mut req: Request, next: Next) -> Response {
    let headers = req.headers_mut();
    headers.remove(header::IF_MODIFIED_SINCE);
    if headers.remove(header::IF_RANGE).is_some() {
        headers.remove(header::RANGE);
    }
    let path = req.uri().path().trim_start_matches('/').to_string();
    let plain = is_plain(&path);
    let named = if path.is_empty() {
        Some(dir.join("index.html"))
    } else if plain {
        Some(dir.join(&path))
    } else {
        None
    };
    let file = match named {
        Some(f) if tokio::fs::metadata(&f).await.is_ok_and(|m| m.is_file()) => Some(f),
        _ => None,
    };
    let keep = file.is_some()
        && plain
        && hashed_name(&path)
        && tokio::fs::read_to_string(dir.join("index.html")).await.is_ok_and(|html| is_vite_page(&html));
    validated(file, if keep { web::cache_control(&path) } else { "no-cache" }, req, next).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vite_hashed_names() {
        for p in ["assets/index-OV1-8lMu.js", "assets/BenchEntry-XUJ_B9O-.css", "assets/a-b-12345678.woff2"] {
            assert!(hashed_name(p), "{p}");
        }
        for p in [
            "index-OV1-8lMu.js",
            "assets/app.js",
            "assets/app-1234567.js",
            "assets/app-123456789.js",
            "assets/-12345678.js",
            "assets/sub/app-12345678.js",
            "assets/app-12345678.js.map",
            "assets/app-12345678",
            "assets/app-1234567é.js",
        ] {
            assert!(!hashed_name(p), "{p}");
        }
    }

    #[test]
    fn a_vite_page_loads_a_hashed_module() {
        let page = |s: &str| format!("<!doctype html><head>{s}</head>");
        for s in [
            r#"<script type="module" crossorigin src="/assets/index-OV1-8lMu.js"></script>"#,
            r#"<script type="module" src="./assets/index-OV1-8lMu.js"></script>"#,
        ] {
            assert!(is_vite_page(&page(s)), "{s}");
        }
        for s in [
            "",
            r#"<script src="/assets/index-OV1-8lMu.js"></script>"#,
            r#"<script type="module" src="/assets/app.js"></script>"#,
            r#"<script type="module" src="/index-OV1-8lMu.js"></script>"#,
            r#"<script type="module">import "/assets/index-OV1-8lMu.js"</script>"#,
        ] {
            assert!(!is_vite_page(&page(s)), "{s}");
        }
    }
}
