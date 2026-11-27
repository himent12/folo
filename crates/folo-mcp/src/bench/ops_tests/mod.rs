//! Tests of the benchmark operations: shared test helpers (`bench_store`, `with_server`, `Forwarder`),
//! JSONL/part parsing (`parsing`), and multipart upload / recovery (`upload`).

mod parsing;
mod upload;

use super::*;

pub(super) const THREE: PartLimits = PartLimits { records: 3, boxes: 10, bytes: usize::MAX };

pub(super) fn record(image: &str, boxes: usize) -> BenchPrediction {
    BenchPrediction {
        image: image.into(),
        condition: None,
        severity: None,
        boxes: vec![
            folo_api::BenchDetection { class: 0, score: 1.0, x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
            boxes
        ],
    }
}

/// A store with a benchmark of 4 synthetic images and haze 2 in its plan; its id.
pub(super) fn bench_store(home: &Path) -> (std::sync::Arc<folo_server::RealStore>, String) {
    use folo_server::Store;
    let store = std::sync::Arc::new(folo_server::RealStore::open(home.to_path_buf()).device("cpu"));
    let req = folo_api::SyntheticDataset {
        name: "shapes".into(),
        images: 20,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 3,
    };
    let ds = store.create_synthetic(req).unwrap();
    let plan = BenchPlan {
        conditions: vec![folo_api::BenchPlanCondition {
            condition: "haze".into(),
            severities: vec![2],
            weight: 1.0,
            why: None,
        }],
        rationale: None,
        designed_by: None,
    };
    let req = CreateBench {
        name: "shapes".into(),
        description: String::new(),
        source: folo_api::BenchSource::Dataset { dataset_id: ds.id, split: None },
        max_images: None,
        seed: 0,
        plan: Some(plan),
    };
    let bench = Benchmarks::Store(store.clone()).create_bench(req).unwrap();
    assert_eq!(bench.images, 4);
    (store, bench.id)
}

/// A real server on a loopback port, for the time of `f(url, benchmark id, folder)`. Server and store shutdown
/// always run on drop even if `f` panics.
pub(super) fn with_server(name: &str, f: impl FnOnce(&str, &str, &Path)) {
    struct Guard(Option<Box<dyn FnOnce()>>);
    impl Drop for Guard {
        fn drop(&mut self) {
            if let Some(d) = self.0.take() {
                d();
            }
        }
    }
    let home = folo_test_support::TestDir::new(name);
    let (store, id) = bench_store(&home.join("server"));
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = folo_server::Server::new(store.clone(), folo_server::Options::default());
    let serving = rt.spawn(server.serve(listener, async move {
        let _ = stopped.await;
    }));
    let _guard = Guard(Some(Box::new(move || {
        let _ = stop.send(());
        let _ = rt.block_on(serving);
        folo_server::Store::shutdown(&*store);
    })));
    f(&url, &id, home.path());
}

pub(super) fn http(url: &str) -> Benchmarks {
    Benchmarks::Http(std::sync::Arc::new(crate::proxy::HttpStore::new(url, None)))
}

/// What becomes of a request through a [`Forwarder`].
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Fate {
    /// It reaches the server and its answer comes back.
    Pass,
    /// It reaches the server; its answer is dropped.
    LoseAnswer,
    /// It never reaches the server.
    Never,
    /// It never reaches the server, and this status and FOLO error code come back.
    Answer(u16, &'static str),
    /// It never reaches the server, and this status with a non-FOLO plain-text body comes back.
    Raw(u16, &'static str),
}

impl From<bool> for Fate {
    fn from(lose: bool) -> Fate {
        if lose { Fate::LoseAnswer } else { Fate::Pass }
    }
}

/// True for a request that opens or starts a benchmark entry.
pub(super) fn starts_entry(line: &str) -> bool {
    line.starts_with("POST") && (line.contains("/entries ") || line.contains("/entries?"))
}

/// True for a request that stops a benchmark entry.
pub(super) fn stops_entry(line: &str) -> bool {
    line.starts_with("POST") && (line.contains("/stop ") || line.contains("/stop?"))
}

/// True for a request that stops or deletes something.
pub(super) fn removes(line: &str) -> bool {
    line.starts_with("DELETE") || stops_entry(line)
}

/// A request that reads one entry.
pub(super) fn reads_entry(line: &str) -> bool {
    line.starts_with("GET") && line.contains("/entries/")
}

/// Forwards requests to a server, one per connection. `hook` sees each request line and says its [`Fate`] (`true`:
/// the answer is lost).
pub(super) struct Forwarder {
    pub(super) url: String,
    done: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Forwarder {
    pub(super) fn new<R: Into<Fate>>(upstream: &str, hook: impl Fn(&str) -> R + Send + 'static) -> Forwarder {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let upstream = upstream.trim_start_matches("http://").to_string();
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let ended = done.clone();
        let thread = std::thread::spawn(move || {
            for client in listener.incoming() {
                if ended.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(client) = client {
                    forward(client, &upstream, &|line| hook(line).into());
                }
            }
        });
        Forwarder { url, done, thread: Some(thread) }
    }
}

impl Drop for Forwarder {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(self.url.trim_start_matches("http://"));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// One request through: its head and body to `upstream` with `Connection: close` and the whole answer back, as its
/// [`Fate`] says (the client sees its connection closed with no answer otherwise). Every socket gets a read/write
/// deadline so a stalled peer cannot hang the forwarder or its drop join.
fn forward(mut client: std::net::TcpStream, upstream: &str, hook: &dyn Fn(&str) -> Fate) {
    use std::io::Write;
    let io_timeout = Some(Duration::from_secs(10));
    let _ = client.set_read_timeout(io_timeout);
    let _ = client.set_write_timeout(io_timeout);
    let mut reader = BufReader::new(client.try_clone().unwrap());
    let (mut head, mut length) = (Vec::new(), 0usize);
    let mut first = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        if first.is_empty() {
            first = line.clone();
        }
        if line == "\r\n" {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse().unwrap();
        }
        if !lower.starts_with("connection:") {
            head.extend_from_slice(line.as_bytes());
        }
    }
    head.extend_from_slice(b"Connection: close\r\n\r\n");
    let mut body = vec![0; length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    let fate = hook(first.trim());
    if let Fate::Answer(status, code) = fate {
        let json = format!(r#"{{"error":{{"code":"{code}","message":"answered by the test","hint":null}}}}"#);
        let _ = write!(
            client,
            "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
            json.len()
        );
        let _ = client.shutdown(std::net::Shutdown::Both);
        return;
    }
    if let Fate::Raw(status, text) = fate {
        let _ = write!(
            client,
            "HTTP/1.1 {status} Test\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
            text.len()
        );
        let _ = client.shutdown(std::net::Shutdown::Both);
        return;
    }
    if fate == Fate::Never {
        let _ = client.shutdown(std::net::Shutdown::Both);
        return;
    }
    let Ok(mut server) = std::net::TcpStream::connect(upstream) else {
        let _ = client.shutdown(std::net::Shutdown::Both);
        return;
    };
    let _ = server.set_read_timeout(io_timeout);
    let _ = server.set_write_timeout(io_timeout);
    if server.write_all(&head).is_err() || server.write_all(&body).is_err() {
        let _ = client.shutdown(std::net::Shutdown::Both);
        return;
    }
    let mut answer = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut chunk = [0u8; 4096];
    let ok = loop {
        let Some(rem) = deadline.checked_duration_since(std::time::Instant::now()).filter(|d| !d.is_zero())
        else {
            break false;
        };
        let _ = server.set_read_timeout(Some(rem));
        match server.read(&mut chunk) {
            Ok(0) => break true,
            Ok(n) => answer.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break false,
        }
    };
    if ok && fate == Fate::Pass {
        let _ = client.write_all(&answer);
    }
    let _ = client.shutdown(std::net::Shutdown::Both);
}

/// Four records of the unchanged images and four of haze 2, nothing found: a whole entry of the test benchmark.
pub(super) fn whole_file(dir: &Path) -> PathBuf {
    let file = dir.join("whole.jsonl");
    let mut lines: Vec<String> = (1..=4).map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
    lines.extend((1..=4).map(|i| format!(r#"{{"image":"{i}","condition":"haze","severity":2,"boxes":[]}}"#)));
    std::fs::write(&file, lines.join("\n")).unwrap();
    file
}

/// How many requests that stop or delete went through `via`, for a hook that counts them with `count`.
pub(super) fn counter() -> (std::sync::Arc<std::sync::atomic::AtomicUsize>, impl Fn(&str) + Send + 'static) {
    let n = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = n.clone();
    (n, move |line: &str| {
        if removes(line) {
            seen.fetch_add(1, Ordering::Relaxed);
        }
    })
}
