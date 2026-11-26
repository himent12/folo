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
    fn normalize_line(line: &[u8], out: &mut Vec<u8>) {
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
            if serde_json::to_writer(&mut *out, &val).is_ok() {
                return;
            }
        }
        out.extend_from_slice(line);
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
                    Self::normalize_line(chunk, &mut self.out);
                } else {
                    self.line.extend_from_slice(chunk);
                    let mut line = std::mem::take(&mut self.line);
                    Self::normalize_line(&line, &mut self.out);
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
            Self::normalize_line(&line, &mut self.out);
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
}
