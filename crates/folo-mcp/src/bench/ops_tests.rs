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
    // UTF-8 BOM on line 1 (common from Windows editors/PowerShell) is accepted cleanly.
    let bom = format!("\u{FEFF}{}\n{}\n", r("1"), r("2"));
    assert_eq!(records(&bom).unwrap().len(), 2);
    // Misspelled keys in records or boxes fail loudly instead of silently falling back to defaults.
    let typo_box = r#"{"image":"1","boxes":[{"class":0,"confidence":0.4,"x1":1,"y1":2,"x2":3,"y2":4}]}"#;
    let e = records(typo_box).unwrap_err();
    assert!(e.body.message.contains("line 1") && e.body.message.contains("confidence"), "{e:?}");
    let typo_rec = r#"{"image":"1","cond":"haze","boxes":[]}"#;
    let e = records(typo_rec).unwrap_err();
    assert!(e.body.message.contains("line 1") && e.body.message.contains("cond"), "{e:?}");
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
    // A `\r` with no `\n` after it is a byte of the line: a record of the limit plus one at the end of the file is over.
    assert!(records(&format!("{exact}\r")).unwrap_err().body.message.contains("longer"));
    assert_eq!(records(&format!("{}\r", r#"{"image":"1","boxes":[]}"#)).unwrap().len(), 1);
    // BOM-prefixed line 1 obeys the exact same `MAX_RECORD_BYTES` boundary for EOF, LF, CRLF, and overlong lines.
    for ending in ["", "\n", "\r\n"] {
        assert_eq!(records(&format!("\u{FEFF}{exact}{ending}")).unwrap().len(), 1, "bom {ending:?}");
    }
    for ending in ["", "\n", "\r\n"] {
        assert!(
            records(&format!("\u{FEFF}{over}{ending}")).unwrap_err().body.message.contains("longer"),
            "bom over {ending:?}"
        );
    }
    assert!(records(&format!("\u{FEFF}{exact}\r")).unwrap_err().body.message.contains("longer"));
    let bom_padded = format!("\u{FEFF}{}{{\"image\":\"1\",\"boxes\":[]}}\n", " ".repeat(MAX_RECORD_BYTES));
    assert!(records(&bom_padded).is_err(), "overlong BOM-prefixed whitespace line must not split");
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
    // Bytes, at the exact boundary: the limit counts the records' JSON, so two records take exactly twice one; the
    // commas and brackets of the array that is sent are on top. A byte less holds one record.
    let one = serde_json::to_vec(&record("1", 1)).unwrap().len();
    let limits = PartLimits { records: 100, boxes: 100, bytes: 2 * one };
    let mut it = [1, 1, 1].map(|b| Ok(record("1", b))).into_iter().peekable();
    let two = next_part(&mut it, limits).unwrap();
    assert_eq!(two.len(), 2);
    assert_eq!(serde_json::to_vec(&two).unwrap().len(), 2 * one + 3, "the records, a comma and two brackets");
    assert_eq!(next_part(&mut it, limits).unwrap().len(), 1);
    let tight = PartLimits { bytes: 2 * one - 1, ..limits };
    let mut it = [1, 1].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert_eq!(next_part(&mut it, tight).unwrap().len(), 1);
    let alone = PartLimits { bytes: one, ..limits };
    let mut it = [1].map(|b| Ok(record("1", b))).into_iter().peekable();
    assert_eq!(next_part(&mut it, alone).unwrap().len(), 1);
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
    assert!(
        e.body.message.contains("99") && hint.contains("'refused'") && hint.contains("was removed"),
        "{e:?}"
    );
    assert_eq!(b.entries(id).unwrap().len(), 1);
    // Ctrl+C before anything was sent: no entry at all.
    write("4");
    let e = b.submit_in(id, "early".into(), true, &file, three, &AtomicBool::new(true)).unwrap_err();
    assert_eq!(e.body.code, "interrupted", "{e:?}");
    assert_eq!(b.entries(id).unwrap().len(), 1);
}

/// A failure before any finish: an entry still open is removed; any other is left and named.
#[test]
fn the_cleanup_removes_only_an_entry_still_open() {
    let home = folo_test_support::TestDir::new("mcp-bench-clean-up");
    let (store, id) = bench_store(home.path());
    let b = Benchmarks::Store(store.clone());
    let open = |name: &str| {
        let predictions = (1..=4).map(|i| record(&i.to_string(), 0)).collect();
        let contestant = BenchContestantSpec::External { scored: true, open: true, predictions };
        let req = StartBenchEntry { contestant, name: Some(name.into()), max_images: None, conditions: None };
        b.start_entry(&id, req).unwrap()
    };
    let scored = open("scored");
    b.finish(&id, &scored.id).unwrap();
    let what = b.clean_up(&id, &scored);
    assert!(what.contains("'scored'") && what.contains(&scored.id) && what.contains("it is done"), "{what}");
    assert_eq!(b.entry(&id, &scored.id).unwrap().status, BenchEntryStatus::Done);

    let elsewhere = open("elsewhere");
    b.stop_entry(&id, &elsewhere.id).unwrap();
    let what = b.clean_up(&id, &elsewhere);
    assert!(what.contains("it is stopped"), "{what}");
    assert!(b.entry(&id, &elsewhere.id).is_ok());

    let fresh = open("fresh");
    let what = b.clean_up(&id, &fresh);
    assert!(what.contains("was removed"), "{what}");
    assert!(b.entry(&id, &fresh.id).is_err());
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

/// What becomes of a request through a [`Forwarder`].
#[derive(Clone, Copy, PartialEq)]
enum Fate {
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

/// True for a request that stops or deletes something.
fn removes(line: &str) -> bool {
    line.starts_with("DELETE") || (line.starts_with("POST") && line.contains("/stop "))
}

/// A request that reads one entry.
fn reads_entry(line: &str) -> bool {
    line.starts_with("GET") && line.contains("/entries/")
}

/// Forwards requests to a server, one per connection. `hook` sees each request line and says its [`Fate`] (`true`:
/// the answer is lost).
struct Forwarder {
    url: String,
    done: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Forwarder {
    fn new<R: Into<Fate>>(upstream: &str, hook: impl Fn(&str) -> R + Send + 'static) -> Forwarder {
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
    if server.read_to_end(&mut answer).is_ok() && fate == Fate::Pass {
        let _ = client.write_all(&answer);
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

/// How many requests that stop or delete went through `via`, for a hook that counts them with `count`.
fn counter() -> (std::sync::Arc<std::sync::atomic::AtomicUsize>, impl Fn(&str) + Send + 'static) {
    let n = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = n.clone();
    (n, move |line: &str| {
        if removes(line) {
            seen.fetch_add(1, Ordering::Relaxed);
        }
    })
}

/// The server scores the entry and its answer to the finish never arrives: the upload reports the score, the entry
/// stays on the leaderboard, and nothing was stopped.
#[test]
fn a_finish_answer_lost_on_the_wire_keeps_the_score() {
    with_server("mcp-bench-lost-wire", |url, id, dir| {
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            line.starts_with("POST") && line.contains("/finish ")
        });
        let e = http(&via.url)
            .submit_in(id, "lost".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap();
        assert!(e.status == BenchEntryStatus::Done && e.score.is_some(), "{e:?}");
        let kept = http(url).entry(id, &e.id).unwrap();
        assert_eq!((kept.status, kept.name.as_str()), (BenchEntryStatus::Done, "lost"));
        assert_eq!(removals.load(Ordering::Relaxed), 0, "no stop or delete after a finish");
    });
}

/// The finish never reached the server, so the entry is still open when it is read: it is left and named, not stopped.
#[test]
fn a_finish_that_never_arrived_leaves_the_entry_open() {
    with_server("mcp-bench-finish-never", |url, id, dir| {
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            if line.starts_with("POST") && line.contains("/finish ") { Fate::Never } else { Fate::Pass }
        });
        let e = http(&via.url)
            .submit_in(id, "unsent".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        let listed = http(url).entries(id).unwrap();
        assert_eq!((listed.len(), listed[0].status), (1, BenchEntryStatus::Running));
        assert!(
            hint.contains("'unsent'") && hint.contains(&listed[0].id) && hint.contains("was left"),
            "{e:?}"
        );
        assert_eq!(removals.load(Ordering::Relaxed), 0, "no stop or delete after a finish");
    });
}

/// The finish's answer and the state read are both lost: nothing is stopped, the score stays, and the entry is named.
#[test]
fn a_lost_finish_with_no_state_read_stops_nothing() {
    with_server("mcp-bench-finish-unread", |url, id, dir| {
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            (line.starts_with("POST") && line.contains("/finish ")) || reads_entry(line)
        });
        let e = http(&via.url)
            .submit_in(id, "unread".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        let listed = http(url).entries(id).unwrap();
        assert_eq!((listed.len(), listed[0].status), (1, BenchEntryStatus::Done));
        assert!(hint.contains("'unread'") && hint.contains(&listed[0].id), "{e:?}");
        assert!(hint.contains("could not be read"), "{e:?}");
        assert_eq!(removals.load(Ordering::Relaxed), 0, "no stop or delete after a finish");
    });
}

/// A finish answered `409` or `5xx` on an entry that still reads `running` (as one being scored by another request
/// does): nothing is stopped, the entry stays, and the hint names it.
#[test]
fn a_finish_answered_409_or_5xx_stops_nothing() {
    with_server("mcp-bench-finish-answered", |url, id, dir| {
        for (status, code, name) in
            [(409, "bench_entry_not_open", "conflict"), (500, "internal", "broken"), (503, "busy", "busy")]
        {
            let (removals, count) = counter();
            let via = Forwarder::new(url, move |line| {
                count(line);
                if line.starts_with("POST") && line.contains("/finish ") {
                    Fate::Answer(status, code)
                } else {
                    Fate::Pass
                }
            });
            let e = http(&via.url)
                .submit_in(id, name.into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
                .unwrap_err();
            assert_eq!((e.status.as_u16(), e.body.code.as_str()), (status, code), "{e:?}");
            let listed = http(url).entries(id).unwrap();
            let kept = listed.iter().find(|l| l.name == name).expect("the entry stays");
            assert_eq!(kept.status, BenchEntryStatus::Running);
            let hint = e.body.hint.clone().unwrap_or_default();
            assert!(hint.contains(&kept.id) && hint.contains("was left"), "{e:?}");
            assert_eq!(removals.load(Ordering::Relaxed), 0, "no stop or delete after {status}");
        }
    });
}

/// The server refuses the finish (an image is unanswered): it did not take the entry, so the entry is removed and the
/// refusal is the error.
#[test]
fn a_refused_finish_removes_the_open_entry() {
    with_server("mcp-bench-finish-refused", |url, id, dir| {
        let file = dir.join("short.jsonl");
        let lines: Vec<String> = (1..=3).map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
        std::fs::write(&file, lines.join("\n")).unwrap();
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            false
        });
        let e = http(&via.url)
            .submit_in(id, "short".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert_eq!(e.body.code, "invalid_body", "{e:?}");
        assert!(e.body.message.contains("'4'") && hint.contains("was removed"), "{e:?}");
        assert!(http(url).entries(id).unwrap().is_empty(), "no entry is left");
        assert_eq!(removals.load(Ordering::Relaxed), 2, "one stop and one delete");
    });
}

/// A part is refused and the entry's state cannot be read, on a server that would accept a stop: none is sent, and
/// the entry is named.
#[test]
fn a_cleanup_that_cannot_read_the_entry_stops_nothing() {
    with_server("mcp-bench-unread", |url, id, dir| {
        let file = dir.join("bad.jsonl");
        let lines: Vec<String> =
            ["1", "2", "3", "99"].iter().map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
        std::fs::write(&file, lines.join("\n")).unwrap();
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            reads_entry(line)
        });
        let e = http(&via.url)
            .submit_in(id, "bad".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        let listed = http(url).entries(id).unwrap();
        assert_eq!((listed.len(), listed[0].status), (1, BenchEntryStatus::Running));
        assert!(hint.contains("'bad'") && hint.contains(&listed[0].id), "{e:?}");
        assert!(hint.contains("could not be read"), "{e:?}");
        assert_eq!(removals.load(Ordering::Relaxed), 0, "no stop without reading the entry");
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

/// The records of `whole_file`; `during` runs as each is read, with how many have been.
fn counted(dir: &Path, during: impl Fn(usize)) -> impl Iterator<Item = Result<BenchPrediction, Error>> {
    let all: Vec<_> = read_predictions(&whole_file(dir)).unwrap().collect();
    let mut n = 0;
    all.into_iter().inspect(move |_| {
        n += 1;
        during(n);
    })
}

/// Ctrl+C lands during the request that opens the entry: the next part is not even read, and the entry is removed.
#[test]
fn an_interrupt_during_a_request_stops_before_the_next_read() {
    use std::sync::{Arc, atomic::AtomicUsize};
    with_server("mcp-bench-interrupt", |url, id, dir| {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let via = Forwarder::new(url, move |line| {
            if line.starts_with("POST") && line.contains("/entries ") {
                flag.store(true, Ordering::Relaxed);
            }
            false
        });
        let read = Arc::new(AtomicUsize::new(0));
        let count = read.clone();
        let records = counted(dir, move |n| count.store(n, Ordering::Relaxed));
        let e = http(&via.url)
            .submit_records(id, "stopped".into(), true, "f", records, THREE, &stop)
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(e.body.code == "interrupted" && hint.contains("was removed"), "{e:?}");
        assert_eq!(read.load(Ordering::Relaxed), 3, "only the first part was read");
        assert!(http(url).entries(id).unwrap().is_empty());
    });
}

/// Ctrl+C lands during a read: what was read is not sent. During the first read no entry is made at all.
#[test]
fn an_interrupt_during_a_read_stops_before_the_next_request() {
    use std::sync::{Arc, atomic::AtomicUsize};
    with_server("mcp-bench-interrupt-read", |url, id, dir| {
        let (parts, opens) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (seen, opened) = (parts.clone(), opens.clone());
        let via = Forwarder::new(url, move |line| {
            if line.contains("/predictions ") {
                seen.fetch_add(1, Ordering::Relaxed);
            }
            if line.starts_with("POST") && line.contains("/entries ") {
                opened.fetch_add(1, Ordering::Relaxed);
            }
            false
        });
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let read = Arc::new(AtomicUsize::new(0));
        let count = read.clone();
        // The fourth record is the first of the second part.
        let records = counted(dir, move |n| {
            count.store(n, Ordering::Relaxed);
            if n == 4 {
                flag.store(true, Ordering::Relaxed);
            }
        });
        let e = http(&via.url)
            .submit_records(id, "stopped".into(), true, "f", records, THREE, &stop)
            .unwrap_err();
        assert_eq!(e.body.code, "interrupted", "{e:?}");
        assert_eq!(parts.load(Ordering::Relaxed), 0, "no part after the interrupt");
        assert_eq!(
            read.load(Ordering::Relaxed),
            4,
            "no record is read after the one the interrupt came with"
        );
        assert!(http(url).entries(id).unwrap().is_empty());

        let first = Arc::new(AtomicBool::new(false));
        let flag = first.clone();
        // With the last record of the first part: no further read would see it, so the check after the part must.
        let records = counted(dir, move |n| {
            if n == 3 {
                flag.store(true, Ordering::Relaxed);
            }
        });
        let e =
            http(&via.url).submit_records(id, "early".into(), true, "f", records, THREE, &first).unwrap_err();
        assert_eq!(e.body.code, "interrupted", "{e:?}");
        assert_eq!(opens.load(Ordering::Relaxed), 1, "only the first upload opened an entry");
        assert!(http(url).entries(id).unwrap().is_empty());
    });
}

/// The answer to the request that opens the entry is lost: the server did make it, and the error names it as the
/// server lists it (trimmed).
#[test]
fn a_lost_answer_to_the_first_part_names_the_entry_as_listed() {
    with_server("mcp-bench-lost-start", |url, id, dir| {
        let via = Forwarder::new(url, |line| line.starts_with("POST") && line.contains("/entries "));
        let e = http(&via.url)
            .submit_in(id, " opened ".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("may have opened an entry named 'opened'"), "{e:?}");
        let listed = http(url).entries(id).unwrap();
        assert_eq!(listed.len(), 1, "the server made the entry");
        assert_eq!((listed[0].name.as_str(), listed[0].status), ("opened", BenchEntryStatus::Running));
    });
}

/// A later part is refused because another client finished the entry: no stop is sent, and its score stays.
#[test]
fn a_part_refused_after_another_client_scored_keeps_the_score() {
    use std::sync::{Arc, atomic::AtomicUsize};
    with_server("mcp-bench-refused-scored", |url, id, dir| {
        let rest: Vec<BenchPrediction> =
            read_predictions(&whole_file(dir)).unwrap().skip(3).map(Result::unwrap).collect();
        let (other, bench) = (http(url), id.to_string());
        let stops = Arc::new(AtomicUsize::new(0));
        let seen = stops.clone();
        let via = Forwarder::new(url, move |line| {
            if line.contains("/stop ") {
                seen.fetch_add(1, Ordering::Relaxed);
            }
            if line.starts_with("POST") && line.contains("/predictions ") {
                let path = line.split(' ').nth(1).unwrap();
                let entry = path.rsplit('/').nth(1).unwrap();
                other.add_predictions(&bench, entry, rest.clone()).unwrap();
                other.finish(&bench, entry).unwrap();
            }
            false
        });
        let e = http(&via.url)
            .submit_in(id, "raced".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(e.body.code == "bench_entry_not_open" && hint.contains("it is done"), "{e:?}");
        assert_eq!(stops.load(Ordering::Relaxed), 0, "no stop without reading the entry");
        let kept = http(url).entries(id).unwrap();
        assert_eq!((kept.len(), kept[0].status), (1, BenchEntryStatus::Done));
        assert!(kept[0].score.is_some());
    });
}

/// The server stops the entry and the answer to the stop is lost: the entry's state says so, and it is removed.
#[test]
fn a_lost_answer_to_the_stop_still_removes_the_entry() {
    with_server("mcp-bench-lost-stop", |url, id, dir| {
        let file = dir.join("bad.jsonl");
        let lines: Vec<String> =
            ["1", "2", "3", "99"].iter().map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
        std::fs::write(&file, lines.join("\n")).unwrap();
        let via = Forwarder::new(url, |line| line.starts_with("POST") && line.contains("/stop "));
        let e = http(&via.url)
            .submit_in(id, "bad".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(e.body.message.contains("99") && hint.contains("was removed"), "{e:?}");
        assert!(http(url).entries(id).unwrap().is_empty());
    });
}

#[test]
fn predictions_go_in_parts_through_a_server() {
    with_server("mcp-bench-parts-http", |url, id, dir| parts(&http(url), id, dir));
}

/// Only parsed FOLO 4xx answers (excluding `409 bench_entry_not_open`) count as finish refusals: an unparsed 4xx
/// from a proxy stops nothing, while a FOLO 409 with another code still cleans up the open entry.
#[test]
fn refused_distinguishes_unparsed_4xx_and_other_409_codes() {
    with_server("mcp-bench-refused-rules", |url, id, dir| {
        // 1. Proxy 404 HTML/plain body (unparsed -> bad_gateway): NOT treated as a FOLO refusal; stops nothing.
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            if line.starts_with("POST") && line.contains("/finish ") {
                Fate::Raw(404, "<html>not found by proxy</html>")
            } else {
                Fate::Pass
            }
        });
        let e = http(&via.url)
            .submit_in(id, "proxy404".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!((e.status.as_u16(), e.body.code.as_str()), (404, folo_client::NOT_FOLO), "{e:?}");
        assert_eq!(removals.load(Ordering::Relaxed), 0, "unparsed 4xx must not stop or delete");
        let kept = http(url).entries(id).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].status, BenchEntryStatus::Running);
        // Clean up the left-open entry before the next subcase.
        http(url).stop_entry(id, &kept[0].id).unwrap();
        http(url).delete_entry(id, &kept[0].id).unwrap();

        // 2. Parsed FOLO 409 with a code other than `bench_entry_not_open`: IS a refusal; stops & removes entry.
        let (removals, count) = counter();
        let via = Forwarder::new(url, move |line| {
            count(line);
            if line.starts_with("POST") && line.contains("/finish ") {
                Fate::Answer(409, "bench_frozen")
            } else {
                Fate::Pass
            }
        });
        let e = http(&via.url)
            .submit_in(id, "other409".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "bench_frozen"), "{e:?}");
        assert!(e.body.hint.as_deref().unwrap_or_default().contains("was removed"), "{e:?}");
        assert_eq!(removals.load(Ordering::Relaxed), 2, "one stop and one delete");
        assert!(http(url).entries(id).unwrap().is_empty());
    });
}

/// After a lost finish, if recovery reads the entry `done` and the follow-up `/result` request fails, the upload
/// still succeeds with `status: done` and a hint noting the score instead of failing the whole command.
#[test]
fn lost_finish_recovered_done_survives_failed_result_fetch() {
    with_server("mcp-bench-lost-finish-result", |url, id, dir| {
        let via = Forwarder::new(url, |line| {
            if line.starts_with("POST") && line.contains("/finish ") {
                Fate::LoseAnswer
            } else if line.starts_with("GET") && line.contains("/result ") {
                Fate::Never
            } else {
                Fate::Pass
            }
        });
        let who = Contestant::Predictions { path: whole_file(dir), scored: true };
        let state =
            http(&via.url).start_unless(id, who, Some("recovered".into()), &AtomicBool::new(false)).unwrap();
        assert_eq!(state.status, BenchEntryStatus::Done, "{state:?}");
        assert!(state.result.is_none(), "detailed result was unavailable");
        let hint = state.hint.as_deref().unwrap_or_default();
        assert!(hint.contains("was scored") && hint.contains("score "), "{hint}");
    });
}

/// Ctrl+C set while a blocked reader returns a blank line stops before starting another read.
#[test]
fn interrupt_during_blank_line_stops_before_next_read() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    struct BlankPipe {
        reads: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
    }

    impl Read for BlankPipe {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.reads.fetch_add(1, Ordering::Relaxed);
            assert_eq!(n, 0, "must not start a second read after Ctrl+C on a blank line");
            self.stop.store(true, Ordering::Relaxed);
            buf[0] = b'\n';
            Ok(1)
        }
    }

    let reads = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let reader = BufReader::new(BlankPipe { reads: reads.clone(), stop: stop.clone() });
    let mut it = records_of_unless(reader, "pipe".into(), Some(&stop));
    let err = it.next().unwrap().unwrap_err();
    assert_eq!(err.body.code, "interrupted", "{err:?}");
    assert_eq!(reads.load(Ordering::Relaxed), 1);
}

/// `left()` names `queued` and `stopping` accurately instead of calling them "open or being scored".
#[test]
fn left_hint_names_each_status_accurately() {
    let mut entry = BenchEntry {
        id: "be_1".into(),
        bench_id: "bn_1".into(),
        name: "m".into(),
        contestant: BenchContestant::External,
        status: BenchEntryStatus::Running,
        created: String::new(),
        finished: None,
        images: 4,
        conditions: false,
        scored: true,
        fingerprint: String::new(),
        progress: folo_api::BenchProgress { done: 0, total: 4, condition: None, severity: None },
        score: None,
        map50_95: None,
        ci95: None,
        conditions_run: 0,
        speed: None,
        usage: None,
        missing_classes: vec![],
        error: None,
        hint: None,
    };
    for (status, expected) in [
        (BenchEntryStatus::Running, "it is open or being scored"),
        (BenchEntryStatus::Queued, "it is queued"),
        (BenchEntryStatus::Stopping, "it is stopping"),
        (BenchEntryStatus::Done, "it is done"),
        (BenchEntryStatus::Failed, "it is failed"),
        (BenchEntryStatus::Stopped, "it is stopped"),
    ] {
        entry.status = status;
        let text = left(&named(&entry), &entry);
        assert!(text.contains(expected), "{status:?} -> {text}");
    }
}
