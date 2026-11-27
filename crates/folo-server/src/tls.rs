//! Native TLS configuration and HTTPS serving for `folo serve` (pure-Rust `rustls` + `ring`, no OpenSSL).
//!
//! [`TlsConfig::from_pem_files`] loads and validates a PEM certificate chain and matching private key upfront so
//! missing files, corrupt PEM, empty chains, or cert/key mismatches fail before the server binds or opens its
//! store. [`serve_tls`] runs the TLS accept loop with per-connection handshakes (bounded by [`HANDSHAKE_TIMEOUT`]
//! and cancelled immediately on shutdown) and `hyper-util` graceful connection draining so plaintext or stalled
//! clients never block other connections and open SSE streams close cleanly on shutdown.

use std::fmt;
use std::future::Future;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_rustls::TlsAcceptor;

/// Maximum time allowed for a single client TLS handshake before the connection is dropped.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Why loading or validating a TLS certificate / key pair failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsErrorKind {
    /// The certificate file does not exist.
    CertNotFound,
    /// The private key file does not exist.
    KeyNotFound,
    /// The certificate file could not be read.
    CertRead,
    /// The private key file could not be read.
    KeyRead,
    /// The certificate PEM is malformed, empty, or contains an invalid certificate.
    InvalidCert,
    /// The private key PEM is malformed, missing, ambiguous, or unsupported.
    InvalidKey,
    /// The private key does not match the end-entity certificate's public key.
    CertKeyMismatch,
    /// Building the TLS server configuration failed for another reason.
    Config,
}

/// Error returned when loading or validating TLS credentials.
///
/// Error messages never include private key material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlsError {
    kind: TlsErrorKind,
    message: String,
    hint: &'static str,
}

impl TlsError {
    fn new(kind: TlsErrorKind, message: impl Into<String>, hint: &'static str) -> Self {
        Self { kind, message: message.into(), hint }
    }

    /// The failure category.
    pub fn kind(&self) -> TlsErrorKind {
        self.kind
    }

    /// Actionable recovery guidance.
    pub fn hint(&self) -> &'static str {
        self.hint
    }

    /// True when the certificate or key file does not exist.
    pub fn is_not_found(&self) -> bool {
        matches!(self.kind, TlsErrorKind::CertNotFound | TlsErrorKind::KeyNotFound)
    }
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TlsError {}

/// Validated TLS server configuration for [`crate::Server::serve_tls`].
#[derive(Clone)]
pub struct TlsConfig {
    server: Arc<rustls::ServerConfig>,
}

impl fmt::Debug for TlsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TlsConfig(..)")
    }
}

impl TlsConfig {
    /// Reads and validates a PEM certificate chain (`cert_path`) and matching private key (`key_path`).
    pub fn from_pem_files(cert_path: &Path, key_path: &Path) -> Result<Self, TlsError> {
        let cert_label = folo_api::json::path(cert_path);
        let key_label = folo_api::json::path(key_path);
        let cert_pem = std::fs::read(cert_path).map_err(|e| read_error(e, &cert_label, true))?;
        let key_pem = std::fs::read(key_path).map_err(|e| read_error(e, &key_label, false))?;
        Self::from_pem_labeled(&cert_pem, &key_pem, &cert_label, &key_label)
    }

    /// Parses and validates a PEM certificate chain and matching private key from memory.
    pub fn from_pem(cert_pem: &[u8], key_pem: &[u8]) -> Result<Self, TlsError> {
        Self::from_pem_labeled(cert_pem, key_pem, "TLS certificate", "TLS private key")
    }

    fn from_pem_labeled(
        cert_pem: &[u8],
        key_pem: &[u8],
        cert_label: &str,
        key_label: &str,
    ) -> Result<Self, TlsError> {
        let certs = parse_certs(cert_pem, cert_label)?;
        let key = parse_key(key_pem, key_label)?;
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| {
                TlsError::new(
                    TlsErrorKind::Config,
                    format!("could not initialize TLS protocol versions: {e}"),
                    "check your TLS certificate and key configuration",
                )
            })?
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .map_err(|e| map_rustls_error(e, cert_label, key_label))?;
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(Self { server: Arc::new(config) })
    }

    /// The underlying rustls server configuration.
    pub(crate) fn server_config(&self) -> Arc<rustls::ServerConfig> {
        Arc::clone(&self.server)
    }
}

fn read_error(e: std::io::Error, label: &str, is_cert: bool) -> TlsError {
    let what = if is_cert { "certificate" } else { "private key" };
    if e.kind() == ErrorKind::NotFound {
        let kind = if is_cert { TlsErrorKind::CertNotFound } else { TlsErrorKind::KeyNotFound };
        TlsError::new(
            kind,
            format!("TLS {what} file '{label}' does not exist"),
            if is_cert {
                "pass an existing PEM certificate chain file to --tls-cert (or FOLO_TLS_CERT)"
            } else {
                "pass an existing PEM private key file to --tls-key (or FOLO_TLS_KEY)"
            },
        )
    } else {
        let kind = if is_cert { TlsErrorKind::CertRead } else { TlsErrorKind::KeyRead };
        TlsError::new(
            kind,
            format!("could not read TLS {what} file '{label}': {e}"),
            if is_cert {
                "check that --tls-cert points to a readable PEM file"
            } else {
                "check that --tls-key points to a readable PEM file"
            },
        )
    }
}

fn parse_certs(pem: &[u8], label: &str) -> Result<Vec<CertificateDer<'static>>, TlsError> {
    let mut certs = Vec::new();
    for item in CertificateDer::pem_slice_iter(pem) {
        let cert = item.map_err(|e| {
            TlsError::new(
                TlsErrorKind::InvalidCert,
                format!("invalid PEM certificate in '{label}': {e}"),
                "provide a valid PEM-encoded X.509 certificate chain (-----BEGIN CERTIFICATE-----) to --tls-cert",
            )
        })?;
        certs.push(cert);
    }
    if certs.is_empty() {
        return Err(TlsError::new(
            TlsErrorKind::InvalidCert,
            format!("no PEM certificates found in '{label}'"),
            "provide a PEM-encoded X.509 certificate chain (-----BEGIN CERTIFICATE-----) to --tls-cert",
        ));
    }
    Ok(certs)
}

fn parse_key(pem: &[u8], label: &str) -> Result<PrivateKeyDer<'static>, TlsError> {
    let mut keys = Vec::new();
    for item in PrivateKeyDer::pem_slice_iter(pem) {
        let key = item.map_err(|e| {
            TlsError::new(
                TlsErrorKind::InvalidKey,
                format!("invalid PEM private key in '{label}': {e}"),
                "provide a valid unencrypted PKCS#8, PKCS#1, or SEC1 PEM private key to --tls-key",
            )
        })?;
        keys.push(key);
    }
    match keys.len() {
        0 => Err(TlsError::new(
            TlsErrorKind::InvalidKey,
            format!("no PEM private key found in '{label}'"),
            "provide an unencrypted PEM private key (PRIVATE KEY, RSA PRIVATE KEY, or EC PRIVATE KEY) to --tls-key",
        )),
        1 => Ok(keys.pop().expect("len == 1")),
        n => Err(TlsError::new(
            TlsErrorKind::InvalidKey,
            format!("expected a single PEM private key in '{label}', found {n}"),
            "keep only one private key in the file passed to --tls-key",
        )),
    }
}

fn map_rustls_error(e: rustls::Error, cert_label: &str, key_label: &str) -> TlsError {
    match e {
        rustls::Error::InconsistentKeys(_) => TlsError::new(
            TlsErrorKind::CertKeyMismatch,
            format!("TLS certificate '{cert_label}' and private key '{key_label}' do not match"),
            "verify that --tls-cert and --tls-key come from the same key pair",
        ),
        rustls::Error::InvalidCertificate(err) => TlsError::new(
            TlsErrorKind::InvalidCert,
            format!("TLS certificate '{cert_label}' is invalid: {err}"),
            "provide a valid X.509 certificate chain in PEM format to --tls-cert",
        ),
        other => TlsError::new(
            TlsErrorKind::InvalidKey,
            format!("could not use TLS certificate '{cert_label}' with private key '{key_label}': {other}"),
            "verify that --tls-cert is a valid X.509 certificate and --tls-key is a supported unencrypted RSA, ECDSA, or Ed25519 key",
        ),
    }
}

/// Resolves once `rx` observes `true` (or its sender is dropped).
async fn wait_shutdown(rx: &mut watch::Receiver<bool>) {
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            return;
        }
    }
}

fn is_connection_error(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::ConnectionRefused | ErrorKind::ConnectionAborted | ErrorKind::ConnectionReset
    )
}

/// Serves `router` over TLS on `listener` until `signal` resolves, then closes open SSE streams via `shutdown` and
/// drains in-flight HTTPS connections gracefully.
pub(crate) async fn serve_tls(
    listener: TcpListener,
    router: Router,
    tls: TlsConfig,
    shutdown: Arc<watch::Sender<bool>>,
    signal: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let acceptor = TlsAcceptor::from(tls.server_config());
    let graceful = GracefulShutdown::new();
    tokio::pin!(signal);

    loop {
        let (tcp, peer) = tokio::select! {
            biased;
            _ = &mut signal => {
                tracing::info!("shutting down");
                shutdown.send_replace(true);
                break;
            }
            accepted = listener.accept() => match accepted {
                Ok(conn) => conn,
                Err(e) if is_connection_error(&e) => continue,
                Err(e) => {
                    tracing::error!(error = %e, "TLS accept error; retrying");
                    tokio::select! {
                        biased;
                        _ = &mut signal => {
                            tracing::info!("shutting down");
                            shutdown.send_replace(true);
                            break;
                        }
                        _ = tokio::time::sleep(Duration::from_secs(1)) => continue,
                    }
                }
            }
        };

        let _ = tcp.set_nodelay(true);
        let acceptor = acceptor.clone();
        let router = router.clone();
        let watcher = graceful.watcher();
        let mut conn_shutdown = shutdown.subscribe();

        tokio::spawn(async move {
            let tls_stream = tokio::select! {
                biased;
                _ = wait_shutdown(&mut conn_shutdown) => return,
                res = tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(tcp)) => match res {
                    Ok(Ok(stream)) => stream,
                    Ok(Err(e)) => {
                        tracing::debug!(%peer, error = %e, "TLS handshake failed");
                        return;
                    }
                    Err(_) => {
                        tracing::debug!(%peer, "TLS handshake timed out");
                        return;
                    }
                },
            };

            let io = TokioIo::new(tls_stream);
            let service = TowerToHyperService::new(router);
            let conn = hyper::server::conn::http1::Builder::new()
                .timer(TokioTimer::new())
                .serve_connection(io, service);
            let conn = watcher.watch(conn);
            if let Err(e) = conn.await {
                tracing::debug!(%peer, error = %e, "HTTPS connection ended");
            }
        });
    }

    graceful.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::*;

    fn generate_self_signed(host: &str) -> (String, String) {
        let cert = rcgen::generate_simple_self_signed(vec![host.to_string()]).unwrap();
        (cert.cert.pem(), cert.signing_key.serialize_pem())
    }

    #[test]
    fn loads_valid_pem_pair_and_redacts_debug() {
        let dir = TestDir::new("tls-valid");
        let (cert_pem, key_pem) = generate_self_signed("localhost");
        let cert_path = dir.join("cert.pem");
        let key_path = dir.join("key.pem");
        std::fs::write(&cert_path, &cert_pem).unwrap();
        std::fs::write(&key_path, &key_pem).unwrap();

        let cfg = TlsConfig::from_pem_files(&cert_path, &key_path).unwrap();
        let dbg = format!("{cfg:?}");
        assert_eq!(dbg, "TlsConfig(..)");
        assert!(!dbg.contains(&key_pem));

        // A combined PEM file containing both certificate and private key also works when passed to both flags.
        let combined = dir.join("combined.pem");
        std::fs::write(&combined, format!("{cert_pem}\n{key_pem}")).unwrap();
        TlsConfig::from_pem_files(&combined, &combined).unwrap();
    }

    #[test]
    fn rejects_missing_empty_malformed_and_mismatched_credentials() {
        let dir = TestDir::new("tls-errors");
        let (cert_a, key_a) = generate_self_signed("localhost");
        let (cert_b, key_b) = generate_self_signed("localhost");
        let cert_a_path = dir.join("cert-a.pem");
        let key_a_path = dir.join("key-a.pem");
        let key_b_path = dir.join("key-b.pem");
        std::fs::write(&cert_a_path, &cert_a).unwrap();
        std::fs::write(&key_a_path, &key_a).unwrap();
        std::fs::write(&key_b_path, &key_b).unwrap();

        // Missing cert / key files.
        let err = TlsConfig::from_pem_files(&dir.join("missing.pem"), &key_a_path).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::CertNotFound);
        assert!(err.is_not_found());
        let err = TlsConfig::from_pem_files(&cert_a_path, &dir.join("missing-key.pem")).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::KeyNotFound);
        assert!(err.is_not_found());

        // Directory instead of file -> read error.
        let subdir = dir.join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        let err = TlsConfig::from_pem_files(&subdir, &key_a_path).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::CertRead);
        assert!(!err.is_not_found());

        // Empty / non-PEM cert or key.
        let err = TlsConfig::from_pem(b"not a pem", key_a.as_bytes()).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::InvalidCert);
        let err = TlsConfig::from_pem(cert_a.as_bytes(), b"not a key").unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::InvalidKey);

        // Corrupt PEM block.
        let bad_pem = b"-----BEGIN CERTIFICATE-----\n!!!not-base64!!!\n-----END CERTIFICATE-----\n";
        let err = TlsConfig::from_pem(bad_pem, key_a.as_bytes()).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::InvalidCert);
        let bad_key = b"-----BEGIN PRIVATE KEY-----\n!!!not-base64!!!\n-----END PRIVATE KEY-----\n";
        let err = TlsConfig::from_pem(cert_a.as_bytes(), bad_key).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::InvalidKey);

        // Multiple private keys.
        let two_keys = format!("{key_a}\n{key_b}");
        let err = TlsConfig::from_pem(cert_a.as_bytes(), two_keys.as_bytes()).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::InvalidKey);
        assert!(err.to_string().contains("found 2"), "{err}");

        // Mismatched cert and key.
        let err = TlsConfig::from_pem_files(&cert_a_path, &key_b_path).unwrap_err();
        assert_eq!(err.kind(), TlsErrorKind::CertKeyMismatch);
        assert!(err.to_string().contains("do not match"), "{err}");
        assert!(!err.to_string().contains(&key_b));

        // Valid cert_b + key_b still works.
        TlsConfig::from_pem(cert_b.as_bytes(), key_b.as_bytes()).unwrap();
    }
}
