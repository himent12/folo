//! Wire helpers for [`super::HttpStore`]: multipart encoding, content-type mapping, and safetensors header
//! probing against older servers.

use folo_client::{Client, encode};
use folo_server::store::check_arch_override;
use http::StatusCode;

use crate::errors::Error;

/// Reads the canonical detector architecture (`folo-*` or `folo2-*`) from the safetensors header of
/// `GET /api/models/{id}/weights` via bounded `Range` requests. Used only against older real servers that do
/// not advertise `folo2_arch` or emit `x-folo-arch`, so a weights-only fine-tune of a `folo2-*` file never
/// silently runs as v1.
pub(super) fn probe_weights_arch(client: &Client, base: &str, id: &str) -> Result<String, Error> {
    const INITIAL_BYTES: u64 = 64 * 1024;
    const MAX_HEADER_BYTES: u64 = 4 * 1024 * 1024;
    let path = format!("/models/{}/weights", encode(id));
    let fail_unverified = |detail: &str| {
        Error::new(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            format!(
                "the folo serve at {base} is too old to report `x-folo-arch` for weights '{id}' ({detail})"
            ),
        )
        .hint("update folo serve so it advertises `folo2_arch` and `x-folo-arch`")
    };
    let first =
        client.get_range(&path, 0..INITIAL_BYTES, None).map_err(|e| fail_unverified(&e.error.message))?;
    let mut buf = first.bytes;
    if buf.len() < 8 {
        return Err(fail_unverified("weights header truncated"));
    }
    let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap());
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(fail_unverified("invalid weights header length"));
    }
    let need = 8u64.saturating_add(header_len);
    if (buf.len() as u64) < need {
        let rest = client
            .get_range(&path, (buf.len() as u64)..need, (!first.etag.is_empty()).then_some(&first.etag))
            .map_err(|e| fail_unverified(&e.error.message))?;
        buf.extend_from_slice(&rest.bytes);
        if (buf.len() as u64) < need {
            return Err(fail_unverified("weights header truncated"));
        }
    }
    let header: serde_json::Value = serde_json::from_slice(&buf[8..(need as usize)])
        .map_err(|_| fail_unverified("unreadable weights header"))?;
    let cfg_str = header
        .get("__metadata__")
        .and_then(|m| m.get("folo.config"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| fail_unverified("missing folo.config metadata"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(cfg_str).map_err(|_| fail_unverified("invalid folo.config metadata"))?;
    let family = cfg
        .get("family")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| fail_unverified("missing family in folo.config"))?;
    let scale = cfg
        .get("scale")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| fail_unverified("missing scale in folo.config"))?;
    let canonical = match family {
        "folo2" => format!("folo2-{scale}"),
        "folo" => format!("folo-{scale}"),
        _ => return Err(fail_unverified("unknown model family in folo.config")),
    };
    check_arch_override(Some(&canonical))?;
    Ok(canonical)
}

pub(super) fn static_content_type(ct: &str) -> &'static str {
    match ct.split(';').next().unwrap_or("").trim() {
        "image/png" => "image/png",
        "image/bmp" => "image/bmp",
        "image/webp" => "image/webp",
        "image/jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    }
}

/// Picks a multipart boundary that does not appear anywhere in `fields` or `image`.
fn choose_boundary(fields: &[(&str, String)], image: &[u8]) -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let nanos =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0u128, |d| d.as_nanos());
    loop {
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = format!("folo-mcp-{nanos:x}-{seq:x}");
        let needle = candidate.as_bytes();
        let clashes = image.windows(needle.len()).any(|w| w == needle)
            || fields.iter().any(|(_, v)| v.contains(&candidate));
        if !clashes {
            return candidate;
        }
    }
}

/// A `multipart/form-data` body for `POST /api/predict`.
pub(super) fn multipart(fields: &[(&str, String)], image: &[u8]) -> Result<(String, Vec<u8>), Error> {
    for (name, value) in fields {
        if value.contains(['\r', '\n', '\0']) {
            return Err(crate::errors::invalid_arguments(format!(
                "{name} must not contain line breaks or NUL bytes"
            )));
        }
    }
    let boundary = choose_boundary(fields, image);
    let mut body = Vec::with_capacity(image.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"image\"\r\nContent-Type: \
             application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(image);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    Ok((format!("multipart/form-data; boundary={boundary}"), body))
}
