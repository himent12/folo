use super::*;

#[test]
fn encodes_segments() {
    assert_eq!(encode("ds_1"), "ds_1");
    assert_eq!(encode("a/b c"), "a%2Fb%20c");
}

#[test]
fn check_base_url_accepts_mixed_case_schemes_and_rejects_bad_ports() {
    for good in [
        "http://127.0.0.1:7878",
        "HTTP://127.0.0.1:7878/Folo/",
        "HtTpS://gpu-box.example.com:7878/SubPath",
        "HTTPS://[::1]:7878",
    ] {
        assert!(check_base_url(good).is_ok(), "{good} should be accepted");
    }
    for bad in [
        "ftp://127.0.0.1:7878",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:0",
        "http://127.0.0.1:",
        "http://[::1]:abc",
    ] {
        assert!(check_base_url(bad).is_err(), "{bad} should be rejected");
    }
}

#[test]
fn an_unreachable_server_is_transient_and_keeps_the_token_out_of_debug() {
    // Port 9 (discard) is essentially never served on loopback. Windows retries a refused loopback connection
    // for about 2 s, so the connection times out first: refused and timed out are both unreachable.
    let c =
        Client::with_timeouts("http://127.0.0.1:9/", Duration::from_millis(100), Duration::from_millis(100))
            .token("s3cret");
    assert_eq!(c.base(), "http://127.0.0.1:9");
    assert!(!format!("{c:?}").contains("s3cret"));
    let e = c.get::<serde_json::Value>("/health").unwrap_err();
    assert_eq!((e.code(), e.status), (UNREACHABLE, None));
    assert!(e.is_unreachable() && e.is_transient());
    assert!(!format!("{e}").contains("s3cret"));
}

#[test]
fn local_request_errors_are_non_transient_and_skip_transfer_retries() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    const BAD_SECRET: &str = "bad\ntoken_secret_123";
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();

    let bad_tok = Client::new(&format!("http://{addr}")).token(BAD_SECRET);
    let e = bad_tok.get::<serde_json::Value>("/health").unwrap_err();
    assert_eq!((e.status, e.code()), (Some(StatusCode::BAD_REQUEST), "bad_request"));
    assert!(!e.is_transient() && !e.is_unreachable(), "{e:?}");
    assert!(!e.error.message.contains(BAD_SECRET), "{e:?}");

    let mut sink = Vec::new();
    let se = bad_tok.get_range_to("/m/weights", 0..4, None, &mut sink, || false).unwrap_err();
    assert!(
        matches!(se, StreamError::Http(ref err) if err.status == Some(StatusCode::BAD_REQUEST) && !err.is_transient()),
        "{se:?}"
    );

    // Transfer upload fails immediately without sleeping/retrying or touching the socket.
    let stop_calls = AtomicUsize::new(0);
    let stop = || {
        stop_calls.fetch_add(1, Ordering::SeqCst);
        false
    };
    let mut ctl = transfer::Control { progress: &mut |_| {}, stop: &stop, backoff: Duration::from_secs(5) };
    let req = folo_api::CreateUpload {
        kind: folo_api::UploadKind::Model,
        name: "m".into(),
        format: folo_api::DatasetFormat::Auto,
        fingerprint: None,
        bytes: Some(4),
    };
    let t0 = Instant::now();
    let te = transfer::upload(&bad_tok, &req, |w| w.write_all(b"abcd"), &mut ctl).unwrap_err();
    assert!(t0.elapsed() < Duration::from_secs(1), "must not sleep/retry local request errors");
    assert!(
        matches!(te, transfer::TransferError::Failed(ref err) if err.status == Some(StatusCode::BAD_REQUEST) && !err.is_transient()),
        "{te:?}"
    );
    assert!(listener.accept().is_err(), "local request error must not open a TCP connection");

    // Malformed URI is also non-transient 400 bad_request.
    let bad_uri = Client::new("http://bad host:7878").get::<serde_json::Value>("/health").unwrap_err();
    assert_eq!((bad_uri.status, bad_uri.code()), (Some(StatusCode::BAD_REQUEST), "bad_request"));
    assert!(!bad_uri.is_transient() && !bad_uri.is_unreachable(), "{bad_uri:?}");
}

#[test]
fn a_deadline_ends_a_call_the_server_never_answers() {
    // Connections wait in the listener's backlog: accepted by the OS, never answered.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let c = Client::new(&format!("http://{}", listener.local_addr().unwrap()));
    // On a thread: without the deadline the call holds for the 10-minute timeout, the watchdog fails the test,
    // and closing the listener ends the call, so no thread outlives the test.
    let r = std::thread::scope(|scope| {
        let (done, answer) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let r = c.until(Instant::now() + Duration::from_millis(50)).get::<serde_json::Value>("/health");
            let _ = done.send(r);
        });
        let got = answer.recv_timeout(Duration::from_secs(60));
        drop(listener);
        got.expect("the deadline did not end the call")
    });
    assert!(r.unwrap_err().is_unreachable());
}

#[test]
fn only_some_statuses_are_transient() {
    let c = Client::new("http://127.0.0.1:9");
    let body = br#"{"error":{"code":"run_not_found","message":"no run 'x'","hint":null}}"#;
    let e = c.failure(StatusCode::NOT_FOUND, body);
    assert_eq!((e.code(), e.is_transient()), ("run_not_found", false));
    // A proxy's own page, not FOLO's envelope.
    let e = c.failure(StatusCode::BAD_GATEWAY, b"<html>502 Bad Gateway</html>");
    assert_eq!((e.code(), e.is_transient()), ("bad_gateway", true));
    assert!(e.error.message.contains("502 Bad Gateway"), "{}", e.error.message);
    // A proxy's 404 for a server it cannot reach at the moment (RunPod's, for a pod's port) passes; so do
    // Cloudflare's 52x. FOLO's own 404 and a plain 400 or 500 do not.
    let e = c.failure(StatusCode::NOT_FOUND, b"404 page not found");
    assert_eq!((e.code(), e.is_transient()), ("bad_gateway", true));
    for (status, transient) in
        [(400, false), (500, false), (520, true), (522, true), (524, true), (525, false)]
    {
        let e = c.failure(StatusCode::from_u16(status).unwrap(), b"");
        assert_eq!(e.is_transient(), transient, "{status}");
    }
}

#[test]
fn get_range_to_streams_validates_headers_and_checks_stop() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn serve_raw(responses: Vec<Vec<u8>>) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            for resp in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req_buf = [0u8; 1024];
                let _ = stream.read(&mut req_buf);
                let _ = stream.write_all(&resp);
                let _ = stream.flush();
            }
        });
        (format!("http://{addr}"), handle)
    }

    let payload: Vec<u8> = (0..(STREAM_CHUNK * 2 + 17)).map(|i| (i % 251) as u8).collect();
    let total = payload.len();

    let mut ok_206 = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-{}/{total}\r\nETag: \"v1\"\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n",
        total - 1
    )
    .into_bytes();
    ok_206.extend_from_slice(&payload);

    let missing_cr =
        b"HTTP/1.1 206 Partial Content\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd"
            .to_vec();
    let mismatched_offset =
        b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 4-7/8\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd"
            .to_vec();
    let mismatched_len =
        b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-5/8\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd"
            .to_vec();
    let ignored_range =
        b"HTTP/1.1 200 OK\r\nETag: \"v1\"\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd".to_vec();
    let huge_403_len = (MAX_ERROR_BODY as usize) + 64 * 1024;
    let mut huge_403 =
        format!("HTTP/1.1 403 Forbidden\r\nContent-Length: {huge_403_len}\r\nConnection: close\r\n\r\n")
            .into_bytes();
    huge_403.resize(huge_403.len() + huge_403_len, b'e');
    let truncated = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-99/100\r\nETag: \"v1\"\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{}",
        "x".repeat(20)
    )
    .into_bytes();
    let stop_body = ok_206.clone();

    let (base, handle) = serve_raw(vec![
        ok_206,
        missing_cr,
        mismatched_offset,
        mismatched_len,
        ignored_range,
        huge_403,
        truncated,
        stop_body,
    ]);
    let client = Client::new(&base);

    // 1. Streams full multi-chunk 206 response directly to writer.
    let mut out = Vec::new();
    let meta = client.get_range_to("/m/weights", 0..total as u64, None, &mut out, || false).unwrap();
    assert_eq!(meta, StreamedPart { written: total as u64, total: total as u64, etag: "\"v1\"".into() });
    assert_eq!(out, payload);

    // 2. Rejects 206 without Content-Range before writing anything to `out`.
    let mut out = Vec::new();
    let err = client.get_range_to("/m/weights", 0..4, None, &mut out, || false).unwrap_err();
    assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));
    assert!(out.is_empty());

    // 3. Rejects 206 whose Content-Range start offset does not match requested start.
    let mut out = Vec::new();
    let err = client.get_range_to("/m/weights", 0..4, None, &mut out, || false).unwrap_err();
    assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));
    assert!(out.is_empty());

    // 4. Rejects 206 whose body length does not match the advertised Content-Range span.
    let mut out = Vec::new();
    let err = client.get_range_to("/m/weights", 0..6, None, &mut out, || false).unwrap_err();
    assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));

    // 5. Rejects 200 for non-zero start before writing anything to `out`.
    let mut out = Vec::new();
    let err = client.get_range_to("/m/weights", 10..14, None, &mut out, || false).unwrap_err();
    assert!(matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_GATEWAY)));
    assert!(out.is_empty());

    // 6. Preserves HTTP status (and non-transient classification) even when an error body exceeds MAX_ERROR_BODY.
    let mut out = Vec::new();
    let err = client.get_range_to("/m/weights", 0..4, None, &mut out, || false).unwrap_err();
    assert!(
        matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::FORBIDDEN) && !e.is_transient())
    );
    assert!(out.is_empty());

    // 7. Mid-stream truncated TCP body returns a transient transport error.
    let mut out = Vec::new();
    let err = client.get_range_to("/m/weights", 0..100, None, &mut out, || false).unwrap_err();
    assert!(matches!(err, StreamError::Http(e) if e.is_transient()));

    // 8. Stopping mid-stream aborts between chunks without reading the entire body.
    let stopped = Arc::new(AtomicBool::new(false));
    struct StopAfterFirstChunk {
        stopped: Arc<AtomicBool>,
        bytes: usize,
    }
    impl std::io::Write for StopAfterFirstChunk {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.bytes += buf.len();
            self.stopped.store(true, Ordering::SeqCst);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut sink = StopAfterFirstChunk { stopped: stopped.clone(), bytes: 0 };
    let err = client
        .get_range_to("/m/weights", 0..total as u64, None, &mut sink, || stopped.load(Ordering::SeqCst))
        .unwrap_err();
    assert!(matches!(err, StreamError::Stopped));
    assert!(sink.bytes > 0 && sink.bytes < total, "stopped mid-stream at {} / {total}", sink.bytes);

    handle.join().unwrap();
}

#[test]
fn redirects_are_not_followed() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct StopOnDrop {
        flag: Arc<AtomicBool>,
        addr: std::net::SocketAddr,
    }
    impl Drop for StopOnDrop {
        fn drop(&mut self) {
            self.flag.store(true, Ordering::Relaxed);
            let _ = std::net::TcpStream::connect_timeout(&self.addr, Duration::from_secs(1));
        }
    }

    let io_timeout = Some(Duration::from_secs(5));
    // A second server that must never be contacted if redirects are disabled.
    let target_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let target_addr = target_listener.local_addr().unwrap();
    let target_hit = Arc::new(AtomicBool::new(false));
    let stop_target = Arc::new(AtomicBool::new(false));

    let redirect_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let redirect_addr = redirect_listener.local_addr().unwrap();
    let stop_redirect = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        let hit_clone = target_hit.clone();
        let stop_t = stop_target.clone();
        scope.spawn(move || {
            for stream in target_listener.incoming() {
                if stop_t.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(mut s) = stream {
                    let _ = s.set_read_timeout(io_timeout);
                    let _ = s.set_write_timeout(io_timeout);
                    hit_clone.store(true, Ordering::Relaxed);
                    let mut buf = [0u8; 512];
                    let _ = s.read(&mut buf);
                    let _ = s.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                    );
                }
            }
        });
        let _stop_t = StopOnDrop { flag: stop_target.clone(), addr: target_addr };

        let stop_r = stop_redirect.clone();
        scope.spawn(move || {
            if let Ok((mut s, _)) = redirect_listener.accept()
                && !stop_r.load(Ordering::Relaxed)
            {
                let _ = s.set_read_timeout(io_timeout);
                let _ = s.set_write_timeout(io_timeout);
                let mut buf = [0u8; 1024];
                let _ = s.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://{target_addr}/api/health\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = s.write_all(resp.as_bytes());
            }
        });
        let _stop_r = StopOnDrop { flag: stop_redirect.clone(), addr: redirect_addr };

        let client = Client::with_timeouts(
            &format!("http://{redirect_addr}"),
            Duration::from_secs(2),
            Duration::from_secs(5),
        )
        .token("s3cret");
        let err = client.get::<serde_json::Value>("/health").unwrap_err();
        assert_eq!(err.status, Some(StatusCode::FOUND), "{err:?}");
        assert_eq!(err.code(), NOT_FOLO, "{err:?}");
        assert!(!err.is_transient(), "{err:?}");
        assert!(!target_hit.load(Ordering::Relaxed), "redirect target must not be contacted");
    });
}

#[test]
fn non_2xx_error_bodies_are_capped_tokens_scrubbed_and_expired_deadlines_never_send() {
    use std::io::{Read, Write};

    const SECRET: &str = "tok_secret_1234567890";
    let c = Client::new("http://127.0.0.1:9").token(SECRET);

    // Echoed bearer token is scrubbed from FOLO error envelopes (including arbitrary `error.code`) and raw
    // proxy bodies (even across the 200-char truncation cut).
    let env = format!(
        r#"{{"error":{{"code":"err_{SECRET}","message":"bad token {SECRET}","hint":"check {SECRET}"}}}}"#
    );
    let e = c.failure(StatusCode::UNAUTHORIZED, env.as_bytes());
    assert_eq!(e.code(), NOT_FOLO, "{e:?}");
    assert!(!e.error.message.contains(SECRET) && e.error.message.contains("[redacted]"), "{e:?}");
    assert!(
        e.error.hint.as_deref().is_some_and(|h| !h.contains(SECRET) && h.contains("[redacted]")),
        "{e:?}"
    );
    let raw = format!("{} {SECRET}", "x".repeat(195));
    let e = c.failure(StatusCode::BAD_GATEWAY, raw.as_bytes());
    assert!(!e.error.message.contains(&SECRET[..8]), "{e:?}");

    // Empty byte range is rejected locally without touching the network.
    let mut out = Vec::new();
    let err = c.get_range_to("/m/weights", 5..5, None, &mut out, || false).unwrap_err();
    assert!(
        matches!(err, StreamError::Http(e) if e.status == Some(StatusCode::BAD_REQUEST) && !e.is_transient())
    );

    // Already-expired deadline fails immediately without contacting the server.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap_or_else(Instant::now);
    let expired = Client::new(&format!("http://{addr}")).until(past);
    let e = expired.post_empty::<serde_json::Value>("/runs").unwrap_err();
    assert!(e.is_unreachable());
    assert!(listener.accept().is_err(), "expired deadline must not connect");

    // Huge non-2xx body on normal JSON call is bounded and preserves the real non-transient HTTP status.
    listener.set_nonblocking(false).unwrap();
    let huge_len = (MAX_ERROR_BODY as usize) + 64 * 1024;
    let srv = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut buf = [0u8; 512];
        let _ = s.read(&mut buf);
        let mut resp =
            format!("HTTP/1.1 403 Forbidden\r\nContent-Length: {huge_len}\r\nConnection: close\r\n\r\n")
                .into_bytes();
        resp.resize(resp.len() + huge_len, b'e');
        let _ = s.write_all(&resp);
    });
    let e = Client::new(&format!("http://{addr}")).get::<serde_json::Value>("/health").unwrap_err();
    srv.join().unwrap();
    assert_eq!(e.status, Some(StatusCode::FORBIDDEN));
    assert!(!e.is_transient());
}

#[test]
fn check_base_url_validates_host_port_and_forbids_userinfo_or_query() {
    for ok in [
        "http://127.0.0.1:7878",
        "http://127.0.0.1:7878/",
        "https://folo.example.com",
        "https://folo.example.com:8443/prefix/",
        "http://localhost:7878",
        "http://[::1]:7878",
    ] {
        assert!(check_base_url(ok).is_ok(), "{ok:?} should be valid");
    }
    for bad in [
        "",
        "127.0.0.1:7878",
        "ftp://127.0.0.1:7878",
        "http://",
        "https://",
        "http:///",
        "http:///api",
        "http://:7878",
        "http://127.0.0.1:",
        "http://127.0.0.1:0",
        "http://127.0.0.1:70000",
        "http://127.0.0.1:abc",
        "http://[::1",
        "http://[not-ipv6]:7878",
        "http://[::1]junk:7878",
        "http://user:pass@127.0.0.1:7878",
        "http://127.0.0.1:7878?token=abc",
        "http://127.0.0.1:7878#frag",
        "http://127.0.0.1:7878 bad",
    ] {
        assert!(check_base_url(bad).is_err(), "{bad:?} should be rejected");
    }
}

#[test]
fn tls_ca_rejects_missing_empty_and_malformed_pem() {
    let dir = folo_test_support::TestDir::new("client-tls-ca");
    let t = Duration::from_secs(2);
    let err = Client::with_tls_ca_pem("https://127.0.0.1:7878", t, t, b"not a pem").unwrap_err();
    assert!(err.contains("no PEM certificates"), "{err}");

    let bad_pem = b"-----BEGIN CERTIFICATE-----\n!!!not-base64!!!\n-----END CERTIFICATE-----\n";
    let err = Client::with_tls_ca_pem("https://127.0.0.1:7878", t, t, bad_pem).unwrap_err();
    assert!(err.contains("invalid PEM certificate"), "{err}");

    let err = Client::with_tls_ca_file("https://127.0.0.1:7878", t, t, &dir.join("missing.pem")).unwrap_err();
    assert!(err.contains("does not exist"), "{err}");

    let empty = dir.join("empty.pem");
    std::fs::write(&empty, b"").unwrap();
    let err = Client::with_tls_ca_file("https://127.0.0.1:7878", t, t, &empty).unwrap_err();
    assert!(err.contains("no PEM certificates"), "{err}");

    // Valid PEM framing containing bogus DER (`AA==`) must be rejected upfront as an invalid X.509 anchor.
    let bad_der = b"-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n";
    let err = Client::with_tls_ca_pem("https://127.0.0.1:7878", t, t, bad_der).unwrap_err();
    assert!(err.contains("not a valid X.509 trust anchor"), "{err}");

    // Malformed PEM header must never echo input bytes (e.g. if a combined cert+key file is mangled).
    let sentinel = "SECRETKEYBYTES9999";
    let mangled = format!("-----BEGIN PRIVATE KEY-----{sentinel}-----END PRIVATE KEY-----");
    let err = Client::with_tls_ca_pem("https://127.0.0.1:7878", t, t, mangled.as_bytes()).unwrap_err();
    assert!(!err.contains(sentinel) && !err.contains("83, 69, 67"), "{err}");
}
