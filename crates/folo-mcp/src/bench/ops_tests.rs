//! Tests of the benchmark operations: reading a file of predictions, its parts, and an upload against a real
//! store and a loopback server (with a forwarder that can lose an answer).

use super::*;
use folo_test_support::TestFile;

fn records(text: &str) -> Result<Vec<BenchPrediction>, Error> {
    let f = TestFile::new("bench-predictions", "p.jsonl");
    std::fs::write(&f, text).unwrap();
    read_predictions(&f)?.collect()
}

#[test]
fn a_file_of_predictions_is_json_lines_of_records() {
    let r =
        |image: &str| format!(r#"{{"image":"{image}","boxes":[{{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}}]}}"#);
    let lines = format!("{}\n{}\n", r("1"), r("2"));
    let got: Vec<String> = records(&lines).unwrap().into_iter().map(|p| p.image).collect();
    assert_eq!(got, ["1", "2"]);
    assert_eq!(records(&r("9")).unwrap()[0].boxes[0].score, 1.0);
    assert!(records("").unwrap().is_empty());
    // Blank lines are skipped, and line numbers still count them.
    let e = records(&format!("{}\n\n{{\"image\":\"3\"}}\n", r("1"))).unwrap_err();
    assert!(e.body.message.contains("line 3") && e.body.message.contains("boxes"), "{e:?}");
    // Arrays and API bodies are not records.
    for doc in [format!("[{}]", r("1")), format!("{{\"predictions\": [{}]}}", r("1"))] {
        let e = records(&format!("{}\n{doc}\n", r("1"))).unwrap_err();
        assert!(e.body.message.contains("line 2 is not a record"), "{e:?}");
    }
    // Bad JSON says where; a bad record says which.
    let e = records(&format!("{}\n{{\"image\": ", r("1"))).unwrap_err();
    assert!(e.body.code == "invalid_arguments" && e.body.message.contains("line 2"), "{e:?}");
    let e = records(&format!("{}\n{{\"image\":\"2\"}}\n", r("1"))).unwrap_err();
    assert!(e.body.message.contains("line 2") && e.body.message.contains("boxes"), "{e:?}");
    let missing = read_predictions(Path::new("no/such/predictions.jsonl")).err().unwrap();
    assert_eq!(missing.body.code, "path_not_found");
}

/// A line is judged after at most [`MAX_RECORD_BYTES`] and its line ending: no more of it is read, what is left of
/// it does not pass as a line of its own, and a record of exactly the limit is one with either line ending.
#[test]
fn a_long_line_is_refused_before_it_is_read_whole() {
    let first = |text: &str| {
        let mut reader = std::io::Cursor::new(text.as_bytes());
        let r = records_of(&mut reader, "f".into()).next().unwrap();
        (r, reader.position() as usize)
    };
    let array = format!("[{}]", "0,".repeat(MAX_RECORD_BYTES));
    let (r, read) = first(&array);
    assert!(r.unwrap_err().body.message.contains("line 1 is not a record"));
    assert!(read <= MAX_RECORD_BYTES + 2, "{read} bytes read");
    let object = format!("{{\"image\":\"{}\",\"boxes\":[]}}", "x".repeat(2 * MAX_RECORD_BYTES));
    let (r, read) = first(&object);
    assert!(r.unwrap_err().body.message.contains("line 1 is longer"));
    assert!(read <= MAX_RECORD_BYTES + 2, "{read} bytes read");
    // Blanks past the limit, then a record: one line, too long, not a blank line and a record.
    let padded = format!("{}{{\"image\":\"1\",\"boxes\":[]}}\n", " ".repeat(MAX_RECORD_BYTES + 2));
    assert!(records(&padded).is_err());
    let head = "{\"image\":\"";
    let tail = "\",\"boxes\":[]}";
    let exact = format!("{head}{}{tail}", "x".repeat(MAX_RECORD_BYTES - head.len() - tail.len()));
    assert_eq!(exact.len(), MAX_RECORD_BYTES);
    for ending in ["", "\n", "\r\n"] {
        assert_eq!(records(&format!("{exact}{ending}")).unwrap().len(), 1, "{ending:?}");
    }
    let over = format!("{head}x{}{tail}", "x".repeat(MAX_RECORD_BYTES - head.len() - tail.len()));
    assert!(records(&format!("{over}\r\n")).unwrap_err().body.message.contains("longer"));
}

/// A record over the server's limit per record is refused where it is read, so no part can pass a part's limit.
#[test]
fn a_record_over_the_box_limit_is_refused() {
    let boxes = |n: usize| vec![r#"{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}"#; n].join(",");
    let line = |n: usize| format!("{{\"image\":\"1\",\"boxes\":[{}]}}\n", boxes(n));
    assert_eq!(records(&line(MAX_BOXES)).unwrap()[0].boxes.len(), MAX_BOXES);
    let e = records(&line(MAX_BOXES + 1)).unwrap_err();
    assert!(e.body.message.contains("has 1001 boxes"), "{e:?}");
}

fn record(image: &str, boxes: usize) -> BenchPrediction {
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

#[test]
fn parts_hold_whole_records_within_the_limits() {
    let sizes = |p: Vec<BenchPrediction>| p.iter().map(|r| r.boxes.len()).collect::<Vec<_>>();
    // Boxes: a record that would cross the limit starts the next part.
    let limits = PartLimits { records: 100, boxes: 10, bytes: usize::MAX };
    let mut it = [4, 5, 2, 10, 0].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert_eq!(sizes(next_part(&mut it, limits).unwrap()), [4, 5]);
    assert_eq!(sizes(next_part(&mut it, limits).unwrap()), [2]);
    assert_eq!(sizes(next_part(&mut it, limits).unwrap()), [10, 0]);
    assert!(next_part(&mut it, limits).unwrap().is_empty());
    // A record that fits no part is an error, not an empty part (which would end the file there).
    let mut it = [11].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert!(next_part(&mut it, limits).unwrap_err().body.message.contains("11 boxes"));
    // Bytes: a part stops before its limit, and a record that alone is over it is an error.
    let one = serde_json::to_vec(&record("1", 1)).unwrap().len() + 1;
    let limits = PartLimits { records: 100, boxes: 100, bytes: 2 * one };
    let mut it = [1, 1, 1].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert_eq!(next_part(&mut it, limits).unwrap().len(), 2);
    assert_eq!(next_part(&mut it, limits).unwrap().len(), 1);
    let mut it = [3].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert!(next_part(&mut it, limits).unwrap_err().body.message.contains("bytes"));
    // Records, empty ones too.
    let limits = PartLimits { records: 2, boxes: 10, bytes: usize::MAX };
    let mut it = [0, 0, 0].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert_eq!(next_part(&mut it, limits).unwrap().len(), 2);
    assert_eq!(next_part(&mut it, limits).unwrap().len(), 1);
    // The real limits: 199,999 boxes and then 1,000 make two parts.
    let mut it = [MAX_PART_BOXES - 1, 1000].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert_eq!(sizes(next_part(&mut it, PARTS).unwrap()), [MAX_PART_BOXES - 1]);
    assert_eq!(sizes(next_part(&mut it, PARTS).unwrap()), [1000]);
}

/// A store with a benchmark of 4 synthetic images and haze 2 in its plan; its id.
fn bench_store(home: &Path) -> (std::sync::Arc<folo_server::RealStore>, String) {
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

/// A file of predictions in several parts: the parts after the first (the condition's records here) are sent
/// and scored; a later part refused removes the entry.
fn parts(b: &Benchmarks, id: &str, dir: &Path) {
    // Nothing found anywhere: 4 records of the unchanged images, then 4 of haze 2.
    let file = dir.join("p.jsonl");
    let write = |haze_image: &str| {
        let mut lines: Vec<String> = (1..=4).map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
        lines.extend(
            ["1", "2", "3", haze_image]
                .map(|i| format!(r#"{{"image":"{i}","condition":"haze","severity":2,"boxes":[]}}"#)),
        );
        std::fs::write(&file, lines.join("\n")).unwrap();
    };
    let three = PartLimits { records: 3, boxes: 10, bytes: usize::MAX };
    write("4");
    let go = AtomicBool::new(false);
    let e = b.submit_in(id, "parts".into(), true, &file, three, &go).unwrap();
    let state = b.state_of(e).unwrap();
    assert_eq!((state.status, state.done), (folo_api::BenchEntryStatus::Done, 8), "{state:?}");
    let r = state.result.unwrap();
    assert_eq!((r.conditions.len(), r.conditions[0].points.len()), (1, 1), "{:?}", r.conditions);
    // The third part names an image the benchmark does not have: refused, and the entry is gone.
    write("99");
    let e = b.submit_in(id, "refused".into(), true, &file, three, &go).unwrap_err();
    let hint = e.body.hint.clone().unwrap_or_default();
    assert!(e.body.message.contains("99") && hint.contains("stopped and deleted"), "{e:?}");
    assert_eq!(b.entries(id).unwrap().len(), 1);
    // Ctrl+C before anything was sent: no entry at all.
    write("4");
    let e = b.submit_in(id, "early".into(), true, &file, three, &AtomicBool::new(true)).unwrap_err();
    assert_eq!(e.body.code, "interrupted", "{e:?}");
    assert_eq!(b.entries(id).unwrap().len(), 1);
}

/// A finish whose answer was lost: what the server did decides, and a score it saved is never deleted.
#[test]
fn a_lost_finish_answer_keeps_what_the_server_did() {
    let home = folo_test_support::TestDir::new("mcp-bench-lost-finish");
    let (store, id) = bench_store(home.path());
    let b = Benchmarks::Store(store.clone());
    let lost = || Error::new(StatusCode::BAD_GATEWAY, "server_unreachable", "no answer");
    let open = |name: &str| {
        let predictions = (1..=4).map(|i| record(&i.to_string(), 0)).collect();
        let contestant = BenchContestantSpec::External { scored: true, open: true, predictions };
        let req = StartBenchEntry { contestant, name: Some(name.into()), max_images: None, conditions: None };
        b.start_entry(&id, req).unwrap().id
    };
    // Scored before the answer was lost: the result, and the entry stays.
    let scored = open("scored");
    b.finish(&id, &scored).unwrap();
    let got = b.after_lost_finish(&id, &scored, lost()).unwrap();
    assert_eq!(got.status, BenchEntryStatus::Done);
    assert_eq!(b.entry(&id, &scored).unwrap().status, BenchEntryStatus::Done);
    // Still open: kept, and the error names it.
    let kept = open("kept");
    let e = b.after_lost_finish(&id, &kept, lost()).unwrap_err();
    assert!(e.body.hint.as_deref().unwrap_or_default().contains(&kept), "{e:?}");
    assert!(b.entry(&id, &kept).unwrap().status.is_active());
    // Stopped from elsewhere: removed as any failed upload is.
    b.stop_entry(&id, &kept).unwrap();
    let e = b.after_lost_finish(&id, &kept, lost()).unwrap_err();
    assert!(e.body.hint.as_deref().unwrap_or_default().contains("stopped and deleted"), "{e:?}");
    assert!(b.entry(&id, &kept).is_err());
    folo_server::Store::shutdown(&*store);
}

#[test]
fn predictions_go_in_parts_in_process() {
    let home = folo_test_support::TestDir::new("mcp-bench-parts");
    let (store, id) = bench_store(home.path());
    parts(&Benchmarks::Store(store.clone()), &id, home.path());
    folo_server::Store::shutdown(&*store);
}

/// A real server on a loopback port, for the time of `f(url, benchmark id, folder)`.
fn with_server(name: &str, f: impl FnOnce(&str, &str, &Path)) {
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
    f(&url, &id, home.path());
    let _ = stop.send(());
    rt.block_on(serving).unwrap().unwrap();
    folo_server::Store::shutdown(&*store);
}

fn http(url: &str) -> Benchmarks {
    Benchmarks::Http(std::sync::Arc::new(crate::proxy::HttpStore::new(url, None)))
}

/// Forwards requests to a server, one per connection. `hook` sees each request line before it is forwarded; when it
/// answers true, the request still reaches the server but its answer is dropped.
struct Forwarder {
    url: String,
    done: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Forwarder {
    fn new(upstream: &str, hook: impl Fn(&str) -> bool + Send + 'static) -> Forwarder {
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
                    forward(client, &upstream, &hook);
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

/// One request through: its head and body to `upstream` with `Connection: close`, the whole answer back unless
/// `hook` says to drop it (the client then sees its connection closed with no answer).
fn forward(mut client: std::net::TcpStream, upstream: &str, hook: &dyn Fn(&str) -> bool) {
    use std::io::Write;
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
    reader.read_exact(&mut body).unwrap();
    let lose = hook(first.trim());
    let mut server = std::net::TcpStream::connect(upstream).unwrap();
    server.write_all(&head).unwrap();
    server.write_all(&body).unwrap();
    let mut answer = Vec::new();
    server.read_to_end(&mut answer).unwrap();
    if !lose {
        client.write_all(&answer).unwrap();
    }
    let _ = client.shutdown(std::net::Shutdown::Both);
}

/// Four records of the unchanged images and four of haze 2, nothing found: a whole entry of the test benchmark.
fn whole_file(dir: &Path) -> PathBuf {
    let file = dir.join("whole.jsonl");
    let mut lines: Vec<String> = (1..=4).map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
    lines.extend((1..=4).map(|i| format!(r#"{{"image":"{i}","condition":"haze","severity":2,"boxes":[]}}"#)));
    std::fs::write(&file, lines.join("\n")).unwrap();
    file
}

const THREE: PartLimits = PartLimits { records: 3, boxes: 10, bytes: usize::MAX };

/// The server scores the entry and its answer to the finish never arrives: the upload reports the score, and the
/// entry stays on the leaderboard.
#[test]
fn a_finish_answer_lost_on_the_wire_keeps_the_score() {
    with_server("mcp-bench-lost-wire", |url, id, dir| {
        let via = Forwarder::new(url, |line| line.starts_with("POST") && line.contains("/finish "));
        let e = http(&via.url)
            .submit_in(id, "lost".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(e.status, BenchEntryStatus::Done);
        let kept = http(url).entry(id, &e.id).unwrap();
        assert_eq!((kept.status, kept.name.as_str()), (BenchEntryStatus::Done, "lost"));
    });
}

/// Another client finishes the upload first: this one's finish is refused, and the score it finds is not deleted.
#[test]
fn a_finish_refused_because_another_client_finished_keeps_the_score() {
    with_server("mcp-bench-other-finish", |url, id, dir| {
        let other = http(url);
        let bench = id.to_string();
        let via = Forwarder::new(url, move |line| {
            if line.starts_with("POST") && line.contains("/finish ") {
                let path = line.split(' ').nth(1).unwrap();
                let entry = path.rsplit('/').nth(1).unwrap();
                other.finish(&bench, entry).unwrap();
            }
            false
        });
        let e = http(&via.url)
            .submit_in(id, "raced".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(e.status, BenchEntryStatus::Done);
        assert_eq!(http(url).entry(id, &e.id).unwrap().status, BenchEntryStatus::Done);
    });
}

/// Ctrl+C lands while the first part is on its way: the part is accepted, the next is not sent, and the entry is
/// stopped and deleted.
#[test]
fn an_interrupt_during_an_upload_removes_its_entry() {
    with_server("mcp-bench-interrupt", |url, id, dir| {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let parts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = parts.clone();
        let via = Forwarder::new(url, move |line| {
            if line.starts_with("POST") && line.contains("/entries ") {
                flag.store(true, Ordering::Relaxed);
            }
            if line.contains("/predictions ") {
                seen.fetch_add(1, Ordering::Relaxed);
            }
            false
        });
        let e =
            http(&via.url).submit_in(id, "stopped".into(), true, &whole_file(dir), THREE, &stop).unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(e.body.code == "interrupted" && hint.contains("stopped and deleted"), "{e:?}");
        assert_eq!(parts.load(Ordering::Relaxed), 0, "no part after the interrupt");
        assert!(http(url).entries(id).unwrap().is_empty());
    });
}

/// The answer to the request that opens the entry is lost: the error says an entry may exist under its name.
#[test]
fn a_lost_answer_to_the_first_part_says_an_entry_may_exist() {
    with_server("mcp-bench-lost-start", |url, id, dir| {
        let via = Forwarder::new(url, |line| line.starts_with("POST") && line.contains("/entries "));
        let e = http(&via.url)
            .submit_in(id, "opened".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("may have opened an entry named 'opened'"), "{e:?}");
    });
}

#[test]
fn predictions_go_in_parts_through_a_server() {
    with_server("mcp-bench-parts-http", |url, id, dir| parts(&http(url), id, dir));
}
