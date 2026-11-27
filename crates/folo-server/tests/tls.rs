//! End-to-end native TLS tests for `Server::serve_tls`: HTTPS API calls with `--token`, untrusted CA rejection,
//! plaintext client isolation, stalled handshake isolation, SSE streaming over HTTPS, and graceful shutdown.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use folo_server::api::{DatasetList, Health, SystemResponse};
use folo_server::{MockStore, Options, Server, TlsConfig, Token};
use folo_test_support::TestDir;
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, ServerName};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

const SECRET: &str = "tls-worker-secret-99";

fn make_cert(sans: &[&str]) -> (String, String) {
    let params = rcgen::CertificateParams::new(sans.iter().map(|s| (*s).to_string()).collect::<Vec<_>>())
        .expect("cert params");
    let key = rcgen::KeyPair::generate().expect("key pair");
    let cert = params.self_signed(&key).expect("self-signed cert");
    (cert.pem(), key.serialize_pem())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_tls_handles_https_auth_plaintext_clients_sse_and_graceful_shutdown() {
    let dir = TestDir::new("server-tls-e2e");
    let (cert_pem, key_pem) = make_cert(&["localhost", "127.0.0.1"]);
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    std::fs::write(&cert_path, &cert_pem).unwrap();
    std::fs::write(&key_path, &key_pem).unwrap();

    let tls = TlsConfig::from_pem_files(&cert_path, &key_path).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base_https = format!("https://{addr}");

    let server = Server::new(
        Arc::new(MockStore::new()),
        Options { token: Token::new(SECRET), bind: Some(addr.ip()), ..Options::default() },
    );
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let serve_task = tokio::spawn(async move {
        server
            .serve_tls(listener, tls, async move {
                let _ = stop_rx.await;
            })
            .await
    });

    // 1. Authenticated and unauthenticated HTTPS requests through `folo-client` with the custom CA.
    let base_for_blocking = base_https.clone();
    let cert_for_blocking = cert_pem.clone();
    tokio::task::spawn_blocking(move || {
        // Without the custom CA, certificate verification rejects the self-signed cert.
        let untrusted = folo_client::Client::with_timeouts(
            &base_for_blocking,
            Duration::from_secs(5),
            Duration::from_secs(10),
        )
        .token(SECRET);
        let err = untrusted.get::<Health>("/health").unwrap_err();
        assert_eq!(err.code(), folo_client::UNREACHABLE, "{err:?}");

        // With the custom CA, `/api/health` is open while `/api/system` and `/api/datasets` need the token.
        let anon = folo_client::Client::with_tls_ca_pem(
            &base_for_blocking,
            Duration::from_secs(5),
            Duration::from_secs(10),
            cert_for_blocking.as_bytes(),
        )
        .unwrap();
        let health: Health = anon.get("/health").unwrap();
        assert_eq!(health.status, "ok");
        assert!(health.mock);

        let unauth = anon.get::<SystemResponse>("/system").unwrap_err();
        assert_eq!(unauth.status, Some(axum::http::StatusCode::UNAUTHORIZED));
        assert_eq!(unauth.code(), "unauthorized");

        let authed = anon.token(SECRET);
        let sys: SystemResponse = authed.get("/system").unwrap();
        assert!(sys.system.cpu.threads >= 1);
        let ds: DatasetList = authed.get("/datasets").unwrap();
        assert!(!ds.datasets.is_empty());

        // 2. A plaintext HTTP client hitting the TLS listener fails cleanly without crashing the server.
        let mut plain = TcpStream::connect(addr).unwrap();
        plain.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let _ = plain.write_all(b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
        let mut plain_resp = Vec::new();
        let _ = plain.read_to_end(&mut plain_resp);
        assert!(
            !plain_resp.starts_with(b"HTTP/1.1 200"),
            "TLS server must never answer plaintext HTTP with 200"
        );

        // 3. A stalled TCP connection (opens socket, sends nothing) must not block another HTTPS client.
        let _stalled = TcpStream::connect(addr).unwrap();
        let health_after: Health = authed.get("/health").unwrap();
        assert_eq!(health_after.status, "ok");
    })
    .await
    .unwrap();

    // 4. Live SSE stream over HTTPS (`GET /api/runs/run_1a01/events`) streams initial snapshot and closes cleanly
    //    when shutdown fires.
    let mut roots = rustls::RootCertStore::empty();
    for cert in CertificateDer::pem_slice_iter(cert_pem.as_bytes()) {
        roots.add(cert.unwrap()).unwrap();
    }
    let client_cfg =
        rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client_cfg));
    let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let server_name = ServerName::from(addr.ip());
    let mut tls_stream = connector.connect(server_name, tcp).await.unwrap();
    let req = format!(
        "GET /api/runs/run_1a01/events HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {SECRET}\r\nAccept: \
         text/event-stream\r\n\r\n"
    );
    tls_stream.write_all(req.as_bytes()).await.unwrap();
    tls_stream.flush().await.unwrap();

    let mut reader = BufReader::new(tls_stream);
    let mut status_line = String::new();
    tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut status_line)).await.unwrap().unwrap();
    assert!(status_line.starts_with("HTTP/1.1 200"), "{status_line:?}");

    let mut got_sse_content_type = false;
    let mut got_status_event = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline && !got_status_event {
        let mut line = String::new();
        let n =
            tokio::time::timeout(Duration::from_secs(2), reader.read_line(&mut line)).await.unwrap().unwrap();
        assert!(n > 0, "SSE stream closed before initial status event");
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("content-type:") && lower.contains("text/event-stream") {
            got_sse_content_type = true;
        }
        if line.trim() == "event: status" {
            got_status_event = true;
        }
    }
    assert!(got_sse_content_type, "expected text/event-stream response header");
    assert!(got_status_event, "expected initial SSE `event: status` frame over HTTPS");

    // 5. Trigger graceful shutdown while the SSE stream is open: the stream ends cleanly and `serve_tls` returns.
    let _ = stop_tx.send(());
    let mut drain = String::new();
    let eof = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            drain.clear();
            let n = reader.read_line(&mut drain).await.unwrap_or(0);
            if n == 0 {
                break;
            }
        }
    })
    .await;
    assert!(eof.is_ok(), "SSE stream over HTTPS did not close on shutdown");

    let serve_res = tokio::time::timeout(Duration::from_secs(5), serve_task)
        .await
        .expect("serve_tls shut down promptly")
        .unwrap();
    serve_res.unwrap();
}
