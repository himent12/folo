//! Stdio framing adapter for `folo mcp`.
//!
//! `rmcp` (pinned `=3.9.0`) responds to valid JSON that is not a valid JSON-RPC message (such as `{}`) with a
//! `-32600 Invalid request` error whose `id` is `None`, which serde omits on the wire. FOLO's MCP contract
//! (`docs/MCP.md`) and JSON-RPC 2.0 require `"id":null` on such error responses. [`ErrorIdWriter`] wraps the
//! outgoing stdio stream and injects `"id":null` on any newline-delimited JSON-RPC 2.0 error object that omits
//! `id`, leaving every other message untouched.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use serde_json::Value;
use tokio::io::AsyncWrite;

/// An [`AsyncWrite`] wrapper over newline-delimited JSON-RPC frames that guarantees error responses carry
/// `"id":null` when the underlying SDK omits `id`.
pub(crate) struct ErrorIdWriter<W> {
    inner: W,
    /// Incomplete current line (never contains `b'\n'`).
    line: Vec<u8>,
    /// Normalized bytes queued to be flushed to `inner`, starting at `pos`.
    out: Vec<u8>,
    pos: usize,
}

impl<W> ErrorIdWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self { inner, line: Vec::new(), out: Vec::new(), pos: 0 }
    }

    /// Rewrites `line` into `out`, injecting `"id":null` iff `line` is a JSON-RPC 2.0 error object without `id` or
    /// `method`. Every other message is copied byte-for-byte without parsing JSON.
    fn normalize_line(&mut self, line: &[u8]) {
        // Fast path: rmcp serializes normal responses/requests starting with `{"jsonrpc":"2.0","id":` and
        // notifications starting with `{"jsonrpc":"2.0","method":`. Only lines lacking those prefixes and
        // containing `"error":` need JSON inspection.
        let maybe_idless_error = !line.starts_with(b"{\"jsonrpc\":\"2.0\",\"id\":")
            && !line.starts_with(b"{\"jsonrpc\":\"2.0\",\"method\":")
            && line.windows(8).any(|w| w == b"\"error\":");
        if maybe_idless_error
            && let Ok(mut val) = serde_json::from_slice::<Value>(line)
            && let Some(obj) = val.as_object_mut()
            && obj.get("jsonrpc").and_then(Value::as_str) == Some("2.0")
            && obj.contains_key("error")
            && !obj.contains_key("id")
            && !obj.contains_key("method")
        {
            obj.insert("id".into(), Value::Null);
            if serde_json::to_writer(&mut self.out, &val).is_ok() {
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

        // Malformed params on recognized `tools/call` and `resources/subscribe` report -32602 in 2025-11-25.
        let bad_params = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"folo_info","arguments":[]}}"#,
        )
        .await;
        assert_eq!(bad_params["id"], 2);
        assert_eq!(bad_params["error"]["code"], -32602, "{bad_params}");

        let bad_sub = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":3,"method":"resources/subscribe","params":{"uri":123}}"#,
        )
        .await;
        assert_eq!(bad_sub["id"], 3);
        assert_eq!(bad_sub["error"]["code"], -32602, "{bad_sub}");

        // 2026-only method `subscriptions/listen` and unknown methods still report -32601 in 2025-11-25.
        let listen_2025 = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":4,"method":"subscriptions/listen","params":{"notifications":"bad"}}"#,
        )
        .await;
        assert_eq!(listen_2025["error"]["code"], -32601, "{listen_2025}");

        let unknown = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":5,"method":"no/such_method","params":{}}"#,
        )
        .await;
        assert_eq!(unknown["error"]["code"], -32601, "{unknown}");

        // Valid `tools/call` still succeeds.
        let ok = send_and_read(
            &mut client_w,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"folo_info","arguments":{}}}"#,
        )
        .await;
        assert!(ok.get("result").is_some(), "{ok}");

        drop(client_w);
        serve.await.unwrap();

        // 2026-07-28 inline session: `ping` and `resources/subscribe` return -32601, while malformed
        // `tools/call`, `subscriptions/listen` and `server/discover` return -32602.
        let opts2 = crate::Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        };
        let server2 = crate::FoloMcp::new(Arc::new(folo_server::MockStore::new()), None, opts2);
        let (mut client_w, server_r) = tokio::io::duplex(64 * 1024);
        let (server_w, client_r) = tokio::io::duplex(64 * 1024);
        let serve2 = tokio::spawn(async move {
            let running = server2.serve((server_r, ErrorIdWriter::new(server_w))).await.unwrap();
            let _ = running.waiting().await;
        });
        let mut reader = BufReader::new(client_r);
        let meta = r#""_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientInfo":{"name":"test","version":"1"},"io.modelcontextprotocol/clientCapabilities":{}}"#;

        // Open the 2026-07-28 inline session with a valid request first.
        let open_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(r#"{{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{{{meta},"name":"folo_info","arguments":{{}}}}}}"#),
        )
        .await;
        assert!(open_2026.get("result").is_some(), "{open_2026}");

        let ping_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(r#"{{"jsonrpc":"2.0","id":10,"method":"ping","params":{{{meta}}}}}"#),
        )
        .await;
        assert_eq!(ping_2026["error"]["code"], -32601, "{ping_2026}");

        let sub_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(
                r#"{{"jsonrpc":"2.0","id":11,"method":"resources/subscribe","params":{{{meta},"uri":123}}}}"#
            ),
        )
        .await;
        assert_eq!(sub_2026["error"]["code"], -32601, "{sub_2026}");

        let bad_call_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(r#"{{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{{{meta},"name":"folo_info","arguments":[]}}}}"#),
        )
        .await;
        assert_eq!(bad_call_2026["error"]["code"], -32602, "{bad_call_2026}");

        let bad_listen_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(r#"{{"jsonrpc":"2.0","id":13,"method":"subscriptions/listen","params":{{{meta},"notifications":"bad"}}}}"#),
        )
        .await;
        assert_eq!(bad_listen_2026["error"]["code"], -32602, "{bad_listen_2026}");

        let discover_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(r#"{{"jsonrpc":"2.0","id":14,"method":"server/discover","params":{{{meta}}}}}"#),
        )
        .await;
        assert!(discover_2026.get("result").is_some(), "{discover_2026}");

        let bad_read_2026 = send_and_read(
            &mut client_w,
            &mut reader,
            &format!(
                r#"{{"jsonrpc":"2.0","id":15,"method":"resources/read","params":{{{meta},"uri":123}}}}"#
            ),
        )
        .await;
        assert_eq!(bad_read_2026["error"]["code"], -32602, "{bad_read_2026}");

        drop(client_w);
        serve2.await.unwrap();
    }
}
