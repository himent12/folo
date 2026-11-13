//! The transport: HTTP agents, path encoding, and the JSON requests and error decoding every call
//! goes through.

use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::api::*;

use super::HttpBackend;

pub(super) fn agent(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_global(timeout)
        .build()
        .into()
}

pub(super) fn stream_agent(headers: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_recv_response(Some(headers))
        .build()
        .into()
}

/// A server message or hint as a sentence, as Studio shows them (`web/src/lib/format.ts`,
/// `sentence`): a capital first letter and a closing full stop ("the dataset has no training
/// images" → "The dataset has no training images."). One that starts with an identifier in
/// backticks keeps its case: it is a name, not a word.
pub(super) fn sentence(s: &str) -> String {
    let t = s.trim();
    let mut chars = t.chars();
    let mut out: String = match chars.next() {
        None => return String::new(),
        Some('`') => t.to_string(),
        Some(first) => first.to_uppercase().chain(chars).collect(),
    };
    if !out.ends_with(['.', '!', '?']) {
        out.push('.');
    }
    out
}

/// Percent-encode one path segment (ids are opaque strings).
pub(super) fn seg(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

impl HttpBackend {
    pub(super) fn url(&self, path: &str) -> String {
        format!("{}/api{path}", self.base)
    }

    pub(super) fn unreachable(&self, e: &ureq::Error) -> ApiError {
        ApiError::new("server_unreachable", format!("Could not reach folo serve at {}", self.host()))
            .with_hint(format!(
                "Start it with `folo serve`, or switch to the built-in demo in Settings. ({e})"
            ))
    }

    pub(super) fn host(&self) -> String {
        self.base.trim_start_matches("http://").trim_start_matches("https://").to_string()
    }

    /// Turn a response into `T`, or the API's error body into an [`ApiError`].
    pub(super) fn decode<T: DeserializeOwned>(
        &self,
        mut resp: ureq::http::Response<ureq::Body>,
    ) -> ApiResult<T> {
        let status = resp.status().as_u16();
        if status >= 400 {
            let body: Option<folo_api::ErrorBody> = resp.body_mut().read_json().ok();
            return Err(match body {
                Some(b) => ApiError {
                    code: b.error.code,
                    message: sentence(&b.error.message),
                    hint: b.error.hint.as_deref().map(sentence),
                },
                None => ApiError::new("http_error", format!("folo serve answered HTTP {status}")),
            });
        }
        if status == 204 {
            return serde_json::from_str("null").map_err(|e| ApiError::new("bad_response", e.to_string()));
        }
        resp.body_mut()
            .with_config()
            .limit(256 << 20)
            .read_json()
            .map_err(|e| ApiError::new("bad_response", format!("Unexpected response from folo serve: {e}")))
    }

    pub(super) fn get<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self.agent.get(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    pub(super) fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = self.agent.post(&self.url(path)).send_json(body).map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    /// A POST that may legitimately take minutes (indexing a large folder, writing images).
    pub(super) fn post_slow<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = agent(Some(Duration::from_secs(600)))
            .post(&self.url(path))
            .send_json(body)
            .map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    pub(super) fn post_empty<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self
            .agent
            .post(&self.url(path))
            .header("Content-Type", "application/json")
            .send("{}")
            .map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    pub(super) fn put<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = self.agent.put(&self.url(path)).send_json(body).map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    /// A DELETE: `204` decodes as `()`, a body as `T`.
    pub(super) fn delete<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self.agent.delete(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(seg("ds_8f2c"), "ds_8f2c");
        assert_eq!(seg("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn server_messages_read_as_sentences() {
        assert_eq!(sentence("the dataset has no training images"), "The dataset has no training images.");
        assert_eq!(sentence("  'D:/x' does not exist. "), "'D:/x' does not exist.");
        assert_eq!(sentence("`claude` printed no result"), "`claude` printed no result.");
        assert_eq!(sentence("étiquettes manquantes!"), "Étiquettes manquantes!");
        assert_eq!(sentence(""), "");
    }
}
