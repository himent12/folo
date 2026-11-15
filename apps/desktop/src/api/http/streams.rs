//! Server-sent events: the SSE reader, and the run and benchmark event streams on top of it.

use std::io::{BufRead, BufReader};
use std::sync::mpsc::channel;

use crate::api::*;

use super::HttpBackend;
use super::transport::seg;

/// Split an SSE stream into `(event, data)` pairs; comments and unknown fields are skipped.
#[cfg(test)]
pub fn parse_sse(reader: impl BufRead, on_event: impl FnMut(&str, &str) -> bool) {
    parse_sse_while(reader, || true, on_event)
}

/// [`parse_sse`] that also stops as soon as `go` turns false, checked on every line received
/// (keep-alive comments included), so a quiet stream still notices a cancelled subscriber.
pub fn parse_sse_while(
    reader: impl BufRead,
    go: impl Fn() -> bool,
    mut on_event: impl FnMut(&str, &str) -> bool,
) {
    let mut event = String::new();
    let mut data = String::new();
    for line in reader.lines() {
        let Ok(line) = line else { return };
        if !go() {
            return;
        }
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            if !data.is_empty() {
                let name = if event.is_empty() { "message" } else { event.as_str() };
                if !on_event(name, &data) {
                    return;
                }
            }
            event.clear();
            data.clear();
        } else if let Some(v) = line.strip_prefix("event:") {
            event = v.trim_start().to_string();
        } else if let Some(v) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(v.strip_prefix(' ').unwrap_or(v));
        }
    }
}

impl HttpBackend {
    /// [`Backend::events`]'s answer.
    pub(super) fn events_for(&self, id: &str) -> ApiResult<Subscription> {
        let url = self.url(&format!("/runs/{}/events", seg(id)));
        let resp = self
            .stream_agent
            .get(&url)
            .header("Accept", "text/event-stream")
            .call()
            .map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "event stream failed")));
        }
        let (tx, rx) = channel();
        let (sub, cancel) = Subscription::new(rx);
        let reader = BufReader::new(resp.into_body().into_reader());
        std::thread::spawn(move || {
            // Ends when the server closes the stream or the subscription is dropped; the reader and
            // its socket go with it.
            parse_sse_while(
                reader,
                || !cancel.load(std::sync::atomic::Ordering::Relaxed),
                |name, data| match RunEvent::from_sse(name, data) {
                    // A finished run's stream stays open on the server; once its final status is
                    // delivered there's nothing more to read.
                    Some(Ok(RunEvent::Status(s))) if !s.status.is_active() => {
                        let _ = tx.send(RunEvent::Status(s));
                        false
                    }
                    Some(Ok(ev)) => tx.send(ev).is_ok(),
                    // Unknown event types are skipped; malformed data too (the stream goes on).
                    _ => true,
                },
            );
        });
        Ok(sub)
    }

    /// [`Backend::bench_events`]'s answer.
    pub(super) fn bench_events_for(&self, id: &str) -> ApiResult<BenchSubscription> {
        let url = self.url(&format!("/bench/benchmarks/{}/events", seg(id)));
        let resp = self
            .stream_agent
            .get(&url)
            .header("Accept", "text/event-stream")
            .call()
            .map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "benchmark event stream failed")));
        }
        let (tx, rx) = channel();
        let (sub, cancel) = BenchSubscription::new(rx);
        let reader = BufReader::new(resp.into_body().into_reader());
        std::thread::spawn(move || {
            parse_sse_while(
                reader,
                || !cancel.load(std::sync::atomic::Ordering::Relaxed),
                |name, data| match BenchEvent::from_sse(name, data) {
                    Some(Ok(ev)) => tx.send(ev).is_ok(),
                    _ => true,
                },
            );
        });
        Ok(sub)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A server that accepts the event stream request, sends `prelude`, then only keep-alives.
    fn keepalive_server(prelude: &'static [u8], write_failed: std::sync::mpsc::Sender<()>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
            );
            if !prelude.is_empty() {
                let frame = [format!("{:x}\r\n", prelude.len()).as_bytes(), prelude, b"\r\n"].concat();
                let _ = sock.write_all(&frame);
            }
            loop {
                std::thread::sleep(Duration::from_millis(10));
                let chunk = b": keep-alive\n\n";
                let frame = [format!("{:x}\r\n", chunk.len()).as_bytes(), chunk, b"\r\n"].concat();
                if sock.write_all(&frame).and_then(|_| sock.flush()).is_err() {
                    let _ = write_failed.send(());
                    return;
                }
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn dropping_a_subscription_closes_a_quiet_stream() {
        let (tx, failed) = std::sync::mpsc::channel();
        let base = keepalive_server(b"", tx);
        let sub = HttpBackend::new(&base).events("run_x").expect("stream opens");
        // Still open while subscribed, through several keep-alives.
        assert!(failed.recv_timeout(Duration::from_millis(60)).is_err());
        drop(sub);
        // The reader sees the next keep-alive, stops, and closes the socket: the server's writes fail.
        assert!(
            failed.recv_timeout(Duration::from_secs(5)).is_ok(),
            "reader thread outlived its subscription"
        );
    }

    #[test]
    fn a_finished_run_closes_its_stream() {
        let (tx, failed) = std::sync::mpsc::channel();
        let base =
            keepalive_server(b"event: status\ndata: {\"status\":\"completed\",\"error\":null}\n\n", tx);
        let sub = HttpBackend::new(&base).events("run_x").expect("stream opens");
        let first = sub.recv_timeout(Duration::from_secs(2)).expect("status delivered");
        assert!(matches!(first, RunEvent::Status(s) if s.status == RunStatus::Completed));
        // Still subscribed, yet the reader is done: the socket closes and the server's writes fail.
        assert!(failed.recv_timeout(Duration::from_secs(5)).is_ok(), "reader kept a finished stream open");
        drop(sub);
    }

    #[test]
    fn a_silent_server_cannot_hang_the_stream_handshake() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept and never answer.
        std::thread::spawn(move || {
            let _held = listener.accept();
            std::thread::sleep(Duration::from_secs(10));
        });
        let t0 = std::time::Instant::now();
        let base = format!("http://{addr}");
        let r = HttpBackend::with_stream_limits(&base, Duration::from_millis(50), Duration::from_secs(45))
            .events("run_x");
        assert!(r.is_err());
        assert!(t0.elapsed() < Duration::from_secs(3), "handshake took {:?}", t0.elapsed());
    }

    /// A link that stops passing bytes without closing (a stuck proxy, a sleeping laptop): the stream
    /// ends once it has been silent for longer than the keep-alives allow, so the live pump notices,
    /// drops it and reads the run again. Before, the reader blocked in `read` for good and the run's
    /// page froze, also after the run ended.
    #[test]
    fn a_stream_that_goes_silent_ends() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Holds the socket open until the test has its answer (or panics): the end seen must be the
        // reader's own timeout, not the server closing.
        let (release, held) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
            );
            let chunk = b"event: status\ndata: {\"status\":\"running\",\"error\":null}\n\n";
            let frame = [format!("{:x}\r\n", chunk.len()).as_bytes(), chunk, b"\r\n"].concat();
            let _ = sock.write_all(&frame).and_then(|_| sock.flush());
            // Then nothing, the socket held open.
            let _ = held.recv();
            drop(sock);
        });
        let base = format!("http://{addr}");
        let sub = HttpBackend::with_stream_limits(&base, Duration::from_secs(5), Duration::from_millis(300))
            .events("run_x")
            .expect("stream opens");
        let first = sub.recv_timeout(Duration::from_secs(5)).expect("status delivered");
        assert!(matches!(first, RunEvent::Status(s) if s.status == RunStatus::Running));
        let t0 = std::time::Instant::now();
        let next = sub.recv_timeout(Duration::from_secs(10));
        assert!(
            matches!(next, Err(std::sync::mpsc::RecvTimeoutError::Disconnected)),
            "a silent stream stays open: {:?} after {:?}",
            next.map(|_| "an event"),
            t0.elapsed()
        );
        drop(release);
        server.join().unwrap();
    }

    #[test]
    fn sse_frames_split_into_events() {
        let stream = "event: status\ndata: {\"status\":\"running\",\"error\":null}\n\n: keep-alive\n\nevent: log\ndata: {\"t\":\"2026-11-08T07:00:00Z\",\"level\":\"info\",\"message\":\"hi\"}\n\nevent: step\ndata: {\"epoch\":12,\"step\":340,\"lr\":0.0098,\"loss\":{\"box\":1.21,\"cls\":0.83,\"dfl\":1.02,\"total\":3.06},\"imgs_per_s\":412.5,\"eta_s\":1840}\n\nevent: future\ndata: {}\n\n";
        let mut seen = Vec::new();
        parse_sse(std::io::Cursor::new(stream), |name, data| {
            seen.push((name.to_string(), RunEvent::from_sse(name, data)));
            true
        });
        let names: Vec<&str> = seen.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["status", "log", "step", "future"]);
        assert!(matches!(seen[0].1, Some(Ok(RunEvent::Status(_)))));
        assert!(matches!(seen[1].1, Some(Ok(RunEvent::Log(_)))));
        // The loss's `box` field (a Rust keyword) decodes.
        assert!(matches!(&seen[2].1, Some(Ok(RunEvent::Step(s))) if s.loss.box_ == 1.21));
        assert!(seen[3].1.is_none(), "an unknown event is skipped");
    }
}
