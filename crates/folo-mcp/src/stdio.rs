//! Stdio framing adapter for `folo mcp`.
//!
//! `rmcp` (pinned `=3.9.0`) has two JSON-RPC 2.0 quirks on malformed client frames:
//! 1. Valid JSON that is not a valid JSON-RPC message (such as `{}`) produces a `-32600 Invalid request` error
//!    whose `id` is `None`, which serde omits on the wire instead of writing `"id":null`.
//! 2. A recognized MCP method with malformed `params` (such as `tools/call` with `"arguments":[]`) fails
//!    untagged request enum deserialization and falls through to `CustomRequest`, which returns `-32601`
//!    (`Method not found`) with `message` set to the method name instead of `-32602` (`Invalid params`).
//!
//! [`ErrorIdWriter`] wraps the outgoing stdio stream and normalizes both error shapes while copying every other
//! message byte-for-byte without parsing JSON.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use serde_json::Value;
use tokio::io::AsyncWrite;

/// MCP request methods FOLO implements across every supported protocol revision. A `-32601` whose message is one
/// of these came from malformed `params` falling through `rmcp`'s request enum rather than an unknown method, so
/// it is normalized to `-32602`.
const COMMON_METHODS: &[&str] = &[
    "initialize",
    "ping",
    "logging/setLevel",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/templates/list",
    "resources/read",
    "prompts/list",
    "prompts/get",
];

/// Methods implemented only in the `2025-11-25` revision (absent in `2026-07-28`, where `-32601` is required).
const LEGACY_2025_METHODS: &[&str] = &["resources/subscribe", "resources/unsubscribe"];

/// An [`AsyncWrite`] wrapper over newline-delimited JSON-RPC frames that guarantees error responses carry
/// `"id":null` when omitted and report `-32602` (`Invalid params`) when a recognized MCP method receives
/// malformed parameters.
pub(crate) struct ErrorIdWriter<W> {
    inner: W,
    /// Incomplete current line (never contains `b'\n'`).
    line: Vec<u8>,
    /// Normalized bytes queued to be flushed to `inner`, starting at `pos`.
    out: Vec<u8>,
    pos: usize,
    /// Set once the outgoing `initialize` result negotiates the `2025-11-25` revision.
    legacy_2025: bool,
}

impl<W> ErrorIdWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self { inner, line: Vec::new(), out: Vec::new(), pos: 0, legacy_2025: false }
    }

    fn is_recognized_method(&self, method: &str) -> bool {
        COMMON_METHODS.contains(&method) || (self.legacy_2025 && LEGACY_2025_METHODS.contains(&method))
    }

    /// Rewrites `line` into `out` when it is either (a) a JSON-RPC 2.0 error object missing `id` (injecting
    /// `"id":null`) or (b) a `-32601` error whose message names a recognized FOLO MCP method for the session's
    /// protocol revision (normalizing it to `-32602` Invalid params). Every other message is copied byte-for-byte
    /// without parsing JSON.
    fn normalize_line(&mut self, line: &[u8]) {
        if !self.legacy_2025
            && line.windows(29).any(|w| w == b"\"protocolVersion\":\"2025-11-25\"")
            && line.windows(9).any(|w| w == b"\"result\":")
        {
            self.legacy_2025 = true;
        }
        // Fast path: skip JSON parsing for notifications and normal responses. Only lines that either lack an
        // `"id":` prefix while containing `"error":`, or carry `"code":-32601`, need JSON inspection.
        let is_notification = line.starts_with(b"{\"jsonrpc\":\"2.0\",\"method\":");
        let maybe_idless_error = !is_notification
            && !line.starts_with(b"{\"jsonrpc\":\"2.0\",\"id\":")
            && line.windows(8).any(|w| w == b"\"error\":");
        let maybe_known_method_not_found =
            !is_notification && line.windows(13).any(|w| w == b"\"code\":-32601");
        if (maybe_idless_error || maybe_known_method_not_found)
            && let Ok(mut val) = serde_json::from_slice::<Value>(line)
            && let Some(obj) = val.as_object_mut()
            && obj.get("jsonrpc").and_then(Value::as_str) == Some("2.0")
            && obj.contains_key("error")
            && !obj.contains_key("method")
        {
            let mut changed = false;
            if !obj.contains_key("id") {
                obj.insert("id".into(), Value::Null);
                changed = true;
            }
            if let Some(err) = obj.get_mut("error").and_then(Value::as_object_mut)
                && err.get("code").and_then(Value::as_i64) == Some(-32601)
                && let Some(method) = err.get("message").and_then(Value::as_str).map(str::to_string)
                && self.is_recognized_method(&method)
            {
                err.insert("code".into(), Value::from(-32602));
                err.insert("message".into(), Value::String(format!("invalid params for {method}")));
                changed = true;
            }
            if changed && serde_json::to_writer(&mut self.out, &val).is_ok() {
                return;
            }
        }
        self.out.extend_from_slice(line);
    }
}

impl<W: AsyncWrite + Unpin> ErrorIdWriter<W> {
    fn poll_drain(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        while self.pos < self.out.len() {
            let n = ready!(Pin::new(&mut self.inner).poll_write(cx, &self.out[self.pos..]))?;
            if n == 0 {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write MCP stdio frame",
                )));
            }
            self.pos += n;
        }
        self.out.clear();
        self.pos = 0;
        Poll::Ready(Ok(()))
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for ErrorIdWriter<W> {
    fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        if self.pos < self.out.len() {
            ready!(self.poll_drain(cx))?;
        }
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let mut start = 0;
        for (i, &b) in buf.iter().enumerate() {
            if b == b'\n' {
                let chunk = &buf[start..i];
                if self.line.is_empty() {
                    self.normalize_line(chunk);
                } else {
                    self.line.extend_from_slice(chunk);
                    let mut line = std::mem::take(&mut self.line);
                    self.normalize_line(&line);
                    line.clear();
                    self.line = line;
                }
                self.out.push(b'\n');
                start = i + 1;
            }
        }
        if start < buf.len() {
            self.line.extend_from_slice(&buf[start..]);
        }
        // Best-effort eager drain: `buf` is already buffered so return `Ok(buf.len())` even if the inner sink
        // returns `Pending`; the next `poll_write` or `poll_flush` finishes draining.
        if let Poll::Ready(Err(e)) = self.poll_drain(cx) {
            return Poll::Ready(Err(e));
        }
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        ready!(self.poll_drain(cx))?;
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if !self.line.is_empty() {
            let mut line = std::mem::take(&mut self.line);
            self.normalize_line(&line);
            line.clear();
            self.line = line;
        }
        ready!(self.poll_drain(cx))?;
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    /// An `AsyncWrite` sink that writes at most `max_chunk` bytes per poll and yields `Pending` every other poll,
    /// exercising partial writes and backpressure draining.
    struct ChunkedSink {
        bytes: Vec<u8>,
        max_chunk: usize,
        yield_next: bool,
    }

    impl AsyncWrite for ChunkedSink {
        fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
            if self.yield_next {
                self.yield_next = false;
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            self.yield_next = true;
            let n = buf.len().min(self.max_chunk);
            self.bytes.extend_from_slice(&buf[..n]);
            Poll::Ready(Ok(n))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn injects_null_id_only_on_idless_jsonrpc_errors() {
        let mut w = ErrorIdWriter::new(ChunkedSink { bytes: Vec::new(), max_chunk: 7, yield_next: true });
        let idless = "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32600,\"message\":\"Invalid request\"}}\n";
        let with_id = "{\"jsonrpc\":\"2.0\",\"id\":7,\"error\":{\"code\":-32601,\"message\":\"nope\"}}\n";
        let normal = "{\"jsonrpc\":\"2.0\",\"id\":8,\"result\":{\"ok\":true}}\n";
        let note = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n";

        // Write byte-by-byte across frame boundaries to exercise chunked input + chunked output.
        for b in format!("{idless}{with_id}{normal}{note}").as_bytes().chunks(5) {
            w.write_all(b).await.unwrap();
        }
        w.flush().await.unwrap();
        w.shutdown().await.unwrap();

        let text = String::from_utf8(w.inner.bytes).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");

        let v0: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v0["jsonrpc"], "2.0");
        assert_eq!(v0.get("id"), Some(&Value::Null), "{v0}");
        assert_eq!(v0["error"]["code"], -32600);
        assert_eq!(v0["error"]["message"], "Invalid request");

        // All other frames pass through byte-for-byte untouched.
        assert_eq!(lines[1], with_id.trim_end());
        assert_eq!(lines[2], normal.trim_end());
        assert_eq!(lines[3], note.trim_end());
    }

    #[tokio::test]
    async fn malformed_tool_params_return_invalid_params_over_real_rmcp_session() {
        use rmcp::ServiceExt;
        use std::sync::Arc;
        use tokio::io::{AsyncBufReadExt, BufReader};

        let home = folo_test_support::TestDir::new("mcp-stdio-invalid-params");
        let opts = crate::Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        };
        let server = crate::FoloMcp::new(Arc::new(folo_server::MockStore::new()), None, opts);
        let (mut client_w, server_r) = tokio::io::duplex(64 * 1024);
        let (server_w, client_r) = tokio::io::duplex(64 * 1024);
        let serve = tokio::spawn(async move {
            let running = server.serve((server_r, ErrorIdWriter::new(server_w))).await.unwrap();
            let _ = running.waiting().await;
        });

        let mut reader = BufReader::new(client_r);
        let send_and_read =
            async |w: &mut tokio::io::DuplexStream, r: &mut BufReader<tokio::io::DuplexStream>, msg: &str| {
                w.write_all(msg.as_bytes()).await.unwrap();
                w.write_all(b"\n").await.unwrap();
                w.flush().await.unwrap();
                let mut line = String::new();
                r.read_line(&mut line).await.unwrap();
                serde_json::from_str::<Value>(&line).unwrap()
            };

        let init = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#,
        )
        .await;
        assert!(init.get("result").is_some(), "{init}");
        client_w
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await
            .unwrap();
        client_w.flush().await.unwrap();

        // Malformed params on recognized `tools/call` must report -32602 (Invalid params), not -32601.
        let bad_params = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"folo_info","arguments":[]}}"#,
        )
        .await;
        assert_eq!(bad_params["id"], 2);
        assert_eq!(bad_params["error"]["code"], -32602, "{bad_params}");
        assert!(
            bad_params["error"]["message"].as_str().unwrap_or_default().contains("tools/call"),
            "{bad_params}"
        );

        // Truly unknown method still reports -32601 (Method not found).
        let unknown = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":3,"method":"no/such_method","params":{}}"#,
        )
        .await;
        assert_eq!(unknown["id"], 3);
        assert_eq!(unknown["error"]["code"], -32601, "{unknown}");

        // Valid `tools/call` still succeeds.
        let ok = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"folo_info","arguments":{}}}"#,
        )
        .await;
        assert_eq!(ok["id"], 4);
        assert!(ok.get("result").is_some(), "{ok}");

        drop(client_w);
        serve.await.unwrap();
    }
}
