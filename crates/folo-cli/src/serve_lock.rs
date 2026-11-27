//! One owner per data dir. Two stores on one data dir would each treat the other's active runs as interrupted
//! (the store stops them when it opens) and write the same registries, so `folo serve` (real store) and an
//! embedded `folo mcp` hold an exclusive OS lock on `<data dir>/serve.lock` while they run. A server also writes
//! its address to `<data dir>/serve.json`, so a second start can say where it is (FOLO Desktop connects there,
//! `folo mcp` goes through it). The address is a hint: a reader checks that the server there serves this data dir.

use std::fs::{File, OpenOptions, TryLockError};
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

/// The local URL another process on this machine can connect to for a listener bound to `addr`: unspecified
/// wildcard hosts (`0.0.0.0`, `::`) become loopback (`127.0.0.1`, `[::1]`) because connecting to a wildcard
/// address fails on Windows (`WSAEADDRNOTAVAIL`). Specific binds, including scoped IPv6 addresses, are
/// formatted as bound. Uses `https://` when `tls` is true, else `http://`.
pub fn local_url(addr: SocketAddr, tls: bool) -> String {
    let scheme = if tls { "https" } else { "http" };
    match addr.ip() {
        IpAddr::V4(v4) if v4.is_unspecified() => format!("{scheme}://127.0.0.1:{}", addr.port()),
        IpAddr::V6(v6) if v6.is_unspecified() => format!("{scheme}://[::1]:{}", addr.port()),
        _ => format!("{scheme}://{addr}"),
    }
}

/// Normalizes a `serve.json` URL read from an existing server: an older server bound to a wildcard host wrote
/// `http://0.0.0.0:<port>` or `http://[::]:<port>`, which fails to connect on Windows. Specific binds are kept
/// as written.
fn normalize_local_url(url: String) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url;
    };
    if let Some(port) = rest.strip_prefix("0.0.0.0:") {
        return format!("{scheme}://127.0.0.1:{port}");
    }
    if let Some(port) = rest.strip_prefix("[::]:") {
        return format!("{scheme}://[::1]:{port}");
    }
    url
}

/// The data dir's serve lock, held until dropped; dropping it also removes `serve.json`.
pub struct ServeLock {
    /// Locked; dropping it unlocks (the OS also releases it when the process dies).
    _file: File,
    info: PathBuf,
}

/// What [`acquire`] found.
pub enum Acquire {
    /// This process owns the data dir now.
    Held(ServeLock),
    /// Another process holds the lock; `url` is where it listens, when it has said so.
    InUse { url: Option<String> },
}

/// Takes `dir`'s serve lock (creating `dir` when missing) without waiting.
pub fn acquire(dir: &Path) -> std::io::Result<Acquire> {
    std::fs::create_dir_all(dir)?;
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("serve.lock"))?;
    let info = dir.join("serve.json");
    match file.try_lock() {
        Ok(()) => {
            // A killed owner's address is stale: nobody listens there any more.
            let _ = std::fs::remove_file(&info);
            Ok(Acquire::Held(ServeLock { _file: file, info }))
        }
        Err(TryLockError::WouldBlock) => {
            let url = std::fs::read(&info)
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| v["url"].as_str().map(str::to_string))
                .map(normalize_local_url);
            Ok(Acquire::InUse { url })
        }
        Err(TryLockError::Error(e)) => Err(e),
    }
}

impl ServeLock {
    /// Records where this server listens, for a second `folo serve` on the same data dir to report.
    pub fn publish(&self, url: &str) -> std::io::Result<()> {
        let body = serde_json::json!({ "url": url, "pid": std::process::id() });
        // Whole or not at all for a reader: written beside it, then renamed over it.
        let tmp = self.info.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&body).expect("json"))?;
        folo_nn::io::rename_over(&tmp, &self.info)
    }
}

/// How long [`serves`] waits for `/api/system`. A server's first answer samples the hardware, which took 9 s on a
/// loaded machine: with 3 s, a `folo label` right after `folo serve` started took the server for another process.
const SERVES_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Whether the `folo serve` at `url` serves `dir` (its `/api/system` `data_dir`): `serve.json` can be stale, and a
/// new process may listen on a dead server's port. Connecting is quick or nothing listens there.
pub fn serves(url: &str, dir: &Path, token: Option<&str>, tls_ca: Option<&Path>) -> bool {
    serves_within(url, dir, token, tls_ca, SERVES_TIMEOUT)
}

/// [`serves`] with `/api/system` given at most `timeout`.
fn serves_within(
    url: &str,
    dir: &Path,
    token: Option<&str>,
    tls_ca: Option<&Path>,
    timeout: std::time::Duration,
) -> bool {
    let connect = std::time::Duration::from_secs(3);
    let mut client = match tls_ca {
        Some(ca) => match folo_client::Client::with_tls_ca_file(url, connect, timeout, ca) {
            Ok(c) => c,
            Err(_) => return false,
        },
        None => folo_client::Client::with_timeouts(url, connect, timeout),
    };
    if let Some(token) = token {
        client = client.token(token);
    }
    client
        .get::<folo_api::SystemResponse>("/system")
        .ok()
        .and_then(|s| s.data_dir)
        .is_some_and(|d| same_dir(Path::new(&d), dir))
}

/// The same folder, however it was spelled (relative, `\` or `/`, letter case on Windows).
pub fn same_dir(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

impl Drop for ServeLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.info);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    #[test]
    fn a_second_server_sees_the_first_and_its_address() {
        let dir = TestDir::new("serve-lock");
        let Acquire::Held(first) = acquire(dir.path()).unwrap() else { panic!("the first takes the lock") };
        assert!(matches!(acquire(dir.path()).unwrap(), Acquire::InUse { url: None }));
        first.publish("http://127.0.0.1:4242").unwrap();
        match acquire(dir.path()).unwrap() {
            Acquire::InUse { url } => assert_eq!(url.as_deref(), Some("http://127.0.0.1:4242")),
            Acquire::Held(_) => panic!("the lock is held"),
        }
        drop(first);
        assert!(!dir.path().join("serve.json").exists());
        assert!(matches!(acquire(dir.path()).unwrap(), Acquire::Held(_)));
    }

    #[test]
    fn a_stale_address_is_dropped_by_the_next_owner() {
        let dir = TestDir::new("serve-lock-stale");
        std::fs::write(dir.path().join("serve.json"), r#"{"url":"http://127.0.0.1:1"}"#).unwrap();
        let Acquire::Held(_lock) = acquire(dir.path()).unwrap() else { panic!("free") };
        assert!(!dir.path().join("serve.json").exists());
    }

    /// A server whose `/api/system` answers after `delay` (a first answer samples the hardware), on its own
    /// thread; its URL.
    fn slow_server(dir: &Path, delay: std::time::Duration) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let body = serde_json::json!({
            "cpu": { "name": "c", "threads": 1, "util": 0 }, "memory": { "used_bytes": 1, "total_bytes": 2 },
            "gpus": [], "data_dir": folo_api::json::path(dir),
        })
        .to_string();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            std::thread::sleep(delay);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
        });
        url
    }

    /// A server whose answer takes longer than a short wait but not the wait given is still the data dir's: a
    /// 3 s wait took a server busy with its first hardware sample (9 s) for another process.
    #[test]
    fn a_slow_answer_still_serves_within_the_wait() {
        use std::time::Duration;
        let dir = TestDir::new("serve-lock-slow");
        let ms = Duration::from_millis;
        assert!(!serves_within(&slow_server(dir.path(), ms(300)), dir.path(), None, None, ms(50)));
        assert!(serves_within(&slow_server(dir.path(), ms(300)), dir.path(), None, None, ms(5000)));
    }

    #[test]
    fn wildcard_bind_addresses_publish_loopback_urls() {
        assert_eq!(local_url("0.0.0.0:8080".parse().unwrap(), false), "http://127.0.0.1:8080");
        assert_eq!(local_url("[::]:8080".parse().unwrap(), false), "http://[::1]:8080");
        assert_eq!(local_url("127.0.0.1:8080".parse().unwrap(), false), "http://127.0.0.1:8080");
        assert_eq!(local_url("[::1]:8080".parse().unwrap(), false), "http://[::1]:8080");
        assert_eq!(local_url("192.168.1.5:8080".parse().unwrap(), false), "http://192.168.1.5:8080");
        let scoped = SocketAddr::V6(std::net::SocketAddrV6::new("fe80::1234".parse().unwrap(), 8080, 0, 7));
        assert_eq!(local_url(scoped, false), "http://[fe80::1234%7]:8080");

        assert_eq!(local_url("0.0.0.0:8080".parse().unwrap(), true), "https://127.0.0.1:8080");
        assert_eq!(local_url("[::]:8080".parse().unwrap(), true), "https://[::1]:8080");
        assert_eq!(local_url("127.0.0.1:8080".parse().unwrap(), true), "https://127.0.0.1:8080");
        assert_eq!(local_url(scoped, true), "https://[fe80::1234%7]:8080");
    }

    #[test]
    fn acquire_normalizes_wildcard_urls_from_older_servers() {
        let dir = TestDir::new("serve-lock-wildcard-read");
        let Acquire::Held(_first) = acquire(dir.path()).unwrap() else { panic!("the first takes the lock") };
        for (stored, want) in [
            ("http://0.0.0.0:4242", "http://127.0.0.1:4242"),
            ("https://0.0.0.0:4242", "https://127.0.0.1:4242"),
            ("http://[::]:4242", "http://[::1]:4242"),
            ("https://[::]:4242", "https://[::1]:4242"),
            ("http://127.0.0.1:4242", "http://127.0.0.1:4242"),
            ("http://192.168.1.5:4242", "http://192.168.1.5:4242"),
            ("http://[fe80::1234%7]:4242", "http://[fe80::1234%7]:4242"),
        ] {
            std::fs::write(
                dir.path().join("serve.json"),
                serde_json::to_vec(&serde_json::json!({ "url": stored, "pid": 1 })).unwrap(),
            )
            .unwrap();
            match acquire(dir.path()).unwrap() {
                Acquire::InUse { url } => assert_eq!(url.as_deref(), Some(want), "stored={stored}"),
                Acquire::Held(_) => panic!("the lock is held"),
            }
        }
    }
}
