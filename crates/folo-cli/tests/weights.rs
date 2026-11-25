//! `folo weights pull` against a release server on loopback (`FOLO_WEIGHTS_BASE_URL`): the plain HTTPS download
//! with its redirect, what is installed and what is not (a bad checksum, a cut body, a server error, no answer),
//! and the fallback to the GitHub CLI when the download is refused (`FOLO_GH` names a missing or failing `gh`).

use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use folo_nn::io::ModelConfig;
use folo_test_support::TestDir;
use folo_vision::FoloModel;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The published name every test pulls, and where its files are on a release server.
const NAME: &str = "folo-n-coco-v2";
const ASSET: &str = "/weights-v2/folo-n-coco-v2.folo";
const SUMS: &str = "/weights-v2/SHA256SUMS";
/// A program that does not exist, as `FOLO_GH`.
const NO_GH: &str = "folo-test-no-such-gh";

/// A small real weights file (folo-n, two classes) and its SHA-256, made once.
fn model() -> &'static (Arc<Vec<u8>>, String) {
    static MODEL: OnceLock<(Arc<Vec<u8>>, String)> = OnceLock::new();
    MODEL.get_or_init(|| {
        let dir = TestDir::new("weights-model");
        let path = dir.join("m.folo");
        let mut m = FoloModel::with_rng(ModelConfig::new("n", 2).unwrap(), &mut folo_core::rng::Rng::seed(7))
            .unwrap();
        m.set_classes(vec!["cat".into(), "dog".into()]).unwrap();
        m.save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let sha = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        (Arc::new(bytes), sha)
    })
}

fn sums_of(sha: &str) -> Answer {
    Answer::Ok(Arc::new(format!("{sha}  folo-n-coco-v2.folo\n0000  other.folo\n").into_bytes()))
}

/// What the server answers for a path.
#[derive(Clone)]
enum Answer {
    Ok(Arc<Vec<u8>>),
    /// `302` to this URL.
    Redirect(String),
    Status(u16),
    /// `200` announcing more bytes than it sends, then the connection closes.
    Cut(Arc<Vec<u8>>),
}

/// A release server on loopback: answers by path (else `404`), one connection at a time, and keeps every request's
/// head.
struct Server {
    base: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    fn start(routes: impl FnOnce(&str) -> Vec<(String, Answer)>) -> Server {
        Self::start_seq(|base| routes(base).into_iter().map(|(p, a)| (p, vec![a])).collect())
    }

    /// Serves a queue of answers per path: each request pops the front until one answer remains, which then repeats.
    fn start_seq(routes: impl FnOnce(&str) -> Vec<(String, Vec<Answer>)>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let mut queues: HashMap<String, VecDeque<Answer>> =
            routes(&base).into_iter().map(|(p, answers)| (p, VecDeque::from(answers))).collect();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let head = read_head(&mut s);
                let path = head.split_whitespace().nth(1).unwrap_or_default().to_string();
                log.lock().unwrap().push(head);
                let answer = queues
                    .get_mut(&path)
                    .and_then(|q| if q.len() > 1 { q.pop_front() } else { q.front().cloned() })
                    .unwrap_or(Answer::Status(404));
                let _ = respond(&mut s, &answer);
            }
        });
        Server { base, requests }
    }

    fn paths(&self) -> Vec<String> {
        let requests = self.requests.lock().unwrap();
        requests.iter().map(|h| h.split_whitespace().nth(1).unwrap_or_default().to_string()).collect()
    }
}

/// A server on loopback that closes every connection at once without answering, so the client gets a transport
/// error, never an HTTP status. It stays bound until the test process ends: no other server can take its port
/// (which a port let go of before the client connects could not promise).
fn hang_up() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            drop(stream);
        }
    });
    base
}

fn read_head(s: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut b = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match s.read(&mut b) {
            Ok(1) => head.push(b[0]),
            _ => break,
        }
    }
    String::from_utf8_lossy(&head).into_owned()
}

fn respond(s: &mut TcpStream, answer: &Answer) -> std::io::Result<()> {
    let (status, extra, body, announced): (&str, String, &[u8], usize) = match answer {
        Answer::Ok(b) => ("200 OK", String::new(), b, b.len()),
        Answer::Redirect(to) => ("302 Found", format!("Location: {to}\r\n"), b"", 0),
        Answer::Status(404) => ("404 Not Found", String::new(), b"Not Found", 9),
        Answer::Status(code) => {
            let reason = match *code {
                400 => "400 Bad Request",
                429 => "429 Too Many Requests",
                502 => "502 Bad Gateway",
                503 => "503 Service Unavailable",
                _ => "500 Error",
            };
            (reason, String::new(), b"", 0)
        }
        Answer::Cut(b) => ("200 OK", String::new(), &b[..b.len() / 2], b.len()),
    };
    write!(s, "HTTP/1.1 {status}\r\n{extra}Content-Length: {announced}\r\nConnection: close\r\n\r\n")?;
    s.write_all(body)?;
    s.flush()
}

/// `folo weights pull <NAME> --json` (and `extra`) into the cache `cache`, from `base`, with `gh` as `FOLO_GH`.
fn pull(cache: &Path, base: &str, gh: &str, extra: &[&str]) -> (i32, Vec<Value>, String) {
    let out = folo_cmd()
        .args(["weights", "pull", NAME, "--json"])
        .args(extra)
        .env("FOLO_WEIGHTS_DIR", cache)
        .env("FOLO_WEIGHTS_BASE_URL", base)
        .env("FOLO_GH", gh)
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines = stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON: {l}: {e}")))
        .collect();
    (out.status.code().unwrap(), lines, String::from_utf8(out.stderr).unwrap())
}

/// The names in the cache folder (none when it does not exist): no download folder is ever left behind.
fn cached(cache: &Path) -> Vec<String> {
    let Ok(dir) = std::fs::read_dir(cache) else { return Vec::new() };
    let mut names: Vec<String> =
        dir.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}

/// The error of a failed pull: its exit code is 1 and nothing is installed.
fn failed(cache: &Path, (code, lines, stderr): (i32, Vec<Value>, String)) -> Value {
    assert_eq!(code, 1, "{stderr}");
    assert_eq!(cached(cache), Vec::<String>::new(), "nothing installed");
    lines.last().unwrap_or_else(|| panic!("no error record: {stderr}"))["error"].clone()
}

#[test]
fn a_public_release_downloads_over_https_with_its_redirect() {
    let (bytes, sha) = model();
    let server = Server::start(|base| {
        vec![
            (SUMS.into(), sums_of(sha)),
            // GitHub sends release downloads on to its CDN.
            (ASSET.into(), Answer::Redirect(format!("{base}/cdn/blob?sig=1"))),
            ("/cdn/blob?sig=1".into(), Answer::Ok(Arc::clone(bytes))),
        ]
    });
    let root = TestDir::new("weights-https");
    let cache = root.join("cache");

    let (code, lines, stderr) = pull(&cache, &server.base, NO_GH, &[]);
    assert_eq!(code, 0, "{stderr}");
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["downloaded"].as_bool()), (Some("done"), Some(true)), "{done}");
    assert_eq!(
        (done["via"].as_str(), done["sha256"].as_str()),
        (Some("https"), Some(sha.as_str())),
        "{done}"
    );
    assert_eq!(done["bytes"].as_u64(), Some(bytes.len() as u64));
    assert_eq!(done["path"].as_str().unwrap(), folo_api::json::path(cache.join("folo-n-coco-v2.folo")));
    assert_eq!(cached(&cache), ["folo-n-coco-v2.folo"]);
    assert_eq!(std::fs::read(cache.join("folo-n-coco-v2.folo")).unwrap(), **bytes);
    // The checksums first, then the file through its redirect; never with credentials.
    assert_eq!(server.paths(), [SUMS, ASSET, "/cdn/blob?sig=1"]);
    assert!(
        server.requests.lock().unwrap().iter().all(|h| !h.to_ascii_lowercase().contains("authorization:"))
    );

    // Cached: nothing is asked for; --force downloads it again.
    let (code, lines, stderr) = pull(&cache, &server.base, NO_GH, &[]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((lines[0]["downloaded"].as_bool(), &lines[0]["via"]), (Some(false), &Value::Null));
    assert_eq!(server.paths().len(), 3);
    let (code, lines, stderr) = pull(&cache, &server.base, NO_GH, &["--force"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["downloaded"].as_bool(), Some(true));
    assert_eq!(server.paths().len(), 6);
    assert_eq!(cached(&cache), ["folo-n-coco-v2.folo"]);
}

#[test]
fn a_checksum_mismatch_installs_nothing() {
    let (bytes, _) = model();
    let wrong = "ab".repeat(32);
    let server = Server::start(|_| {
        vec![(SUMS.into(), sums_of(&wrong)), (ASSET.into(), Answer::Ok(Arc::clone(bytes)))]
    });
    let root = TestDir::new("weights-mismatch");
    let cache = root.join("cache");
    let e = failed(&cache, pull(&cache, &server.base, NO_GH, &[]));
    assert_eq!(e["code"], "checksum_mismatch", "{e}");
    assert!(e["message"].as_str().unwrap().contains(&wrong), "{e}");
    assert!(e["hint"].as_str().unwrap().contains("nothing was installed"), "{e}");
}

#[test]
fn a_refused_download_falls_back_to_gh_and_says_both_when_it_fails() {
    // Every path answers 404, as GitHub does for a private repository's release or a draft.
    let server = Server::start(|_| Vec::new());
    let root = TestDir::new("weights-refused");
    let cache = root.join("cache");

    let e = failed(&cache, pull(&cache, &server.base, NO_GH, &[]));
    assert_eq!(e["code"], "gh_failed", "{e}");
    let message = e["message"].as_str().unwrap();
    assert!(message.contains(&format!("{}{SUMS} answered 404", server.base)), "{e}");
    assert!(message.contains("not public") && message.contains(&format!("`{NO_GH}` could not run")), "{e}");
    let hint = e["hint"].as_str().unwrap();
    assert!(hint.contains("gh auth login") && hint.contains("https://cli.github.com"), "{e}");
    // The checksums were refused, so the weights were never asked for.
    assert_eq!(server.paths(), [SUMS]);

    // A `gh` that runs and fails (here the folo binary, which has no `release` command).
    let e =
        failed(&cache, pull(&cache, &server.base, env!("CARGO_BIN_EXE_folo"), &["--repo", "someone/else"]));
    assert_eq!(e["code"], "gh_failed", "{e}");
    let message = e["message"].as_str().unwrap();
    assert!(message.contains("answered 404"), "{e}");
    assert!(message.contains("`gh release download weights-v2 --repo someone/else` failed"), "{e}");
    assert!(e["hint"].as_str().unwrap().contains("gh auth status"), "{e}");
}

#[test]
fn a_cut_body_a_server_error_or_no_answer_is_a_download_error() {
    let (bytes, sha) = model();
    let cut =
        Server::start(|_| vec![(SUMS.into(), sums_of(sha)), (ASSET.into(), Answer::Cut(Arc::clone(bytes)))]);
    let root = TestDir::new("weights-broken");
    let cache = root.join("cache");
    let e = failed(&cache, pull(&cache, &cut.base, NO_GH, &[]));
    assert_eq!(e["code"], "download_failed", "{e}");
    assert!(e["message"].as_str().unwrap().contains("broke off"), "{e}");
    assert_eq!(cut.paths(), [SUMS, ASSET, ASSET, ASSET, ASSET]);

    // A server error is retried and never falls back to `gh`.
    let busy = Server::start(|_| vec![(SUMS.into(), Answer::Status(503))]);
    let e = failed(&cache, pull(&cache, &busy.base, NO_GH, &[]));
    assert_eq!(e["code"], "download_failed", "{e}");
    assert!(e["message"].as_str().unwrap().ends_with("answered 503"), "{e}");
    assert_eq!(busy.paths(), [SUMS, SUMS, SUMS, SUMS]);

    // A connection closed without an answer (as when nothing answers): no HTTP status at all.
    let e = failed(&cache, pull(&cache, &hang_up(), NO_GH, &[]));
    assert_eq!(e["code"], "download_failed", "{e}");
    assert!(e["hint"].as_str().unwrap().contains("network"), "{e}");
}

#[test]
fn transient_errors_and_cut_bodies_recover_on_retry() {
    let (bytes, sha) = model();
    let server = Server::start_seq(|_| {
        vec![
            (SUMS.into(), vec![Answer::Status(503), Answer::Status(429), sums_of(sha)]),
            (
                ASSET.into(),
                vec![Answer::Cut(Arc::clone(bytes)), Answer::Status(502), Answer::Ok(Arc::clone(bytes))],
            ),
        ]
    });
    let root = TestDir::new("weights-retry-ok");
    let cache = root.join("cache");

    let (code, lines, stderr) = pull(&cache, &server.base, NO_GH, &[]);
    assert_eq!(code, 0, "{stderr}");
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["downloaded"].as_bool()), (Some("done"), Some(true)), "{done}");
    assert_eq!(done["via"].as_str(), Some("https"));
    assert_eq!(done["sha256"].as_str(), Some(sha.as_str()));
    assert_eq!(cached(&cache), ["folo-n-coco-v2.folo"]);
    assert_eq!(std::fs::read(cache.join("folo-n-coco-v2.folo")).unwrap(), **bytes);
    assert_eq!(server.paths(), [SUMS, SUMS, SUMS, ASSET, ASSET, ASSET]);
}

#[test]
fn non_retryable_client_error_fails_immediately_without_retries() {
    let server = Server::start(|_| vec![(SUMS.into(), Answer::Status(400))]);
    let root = TestDir::new("weights-400");
    let cache = root.join("cache");

    let e = failed(&cache, pull(&cache, &server.base, NO_GH, &[]));
    assert_eq!(e["code"], "download_failed", "{e}");
    assert!(e["message"].as_str().unwrap().ends_with("answered 400"), "{e}");
    assert_eq!(server.paths(), [SUMS]);
}

#[test]
fn malformed_mirror_url_fails_immediately_without_retries() {
    let root = TestDir::new("weights-bad-url");
    let cache = root.join("cache");

    // Human mode prints retry notices to stderr; a malformed URL must emit none.
    let out = folo_cmd()
        .args(["weights", "pull", NAME])
        .env("FOLO_WEIGHTS_DIR", &cache)
        .env("FOLO_WEIGHTS_BASE_URL", "http://127.0.0.1/bad mirror")
        .env("FOLO_GH", NO_GH)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!stderr.contains("retrying"), "unexpected retry notice: {stderr}");
    assert_eq!(cached(&cache), Vec::<String>::new());
}

#[test]
fn failed_force_repull_preserves_existing_cached_weights() {
    let (bytes, sha) = model();
    let ok =
        Server::start(|_| vec![(SUMS.into(), sums_of(sha)), (ASSET.into(), Answer::Ok(Arc::clone(bytes)))]);
    let root = TestDir::new("weights-atomic-force");
    let cache = root.join("cache");

    let (code, _, stderr) = pull(&cache, &ok.base, NO_GH, &[]);
    assert_eq!(code, 0, "{stderr}");

    let broken = Server::start(|_| vec![(SUMS.into(), Answer::Status(503))]);
    let (code, lines, stderr) = pull(&cache, &broken.base, NO_GH, &["--force"]);
    assert_eq!(code, 1, "{stderr}");
    assert_eq!(lines.last().unwrap()["error"]["code"], "download_failed");
    assert_eq!(cached(&cache), ["folo-n-coco-v2.folo"]);
    assert_eq!(std::fs::read(cache.join("folo-n-coco-v2.folo")).unwrap(), **bytes);
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1");
    c.env("FOLO_WEIGHTS_RETRY_DELAY_MS", "0");
    c
}
