//! Unit tests for predictions JSONL parsing, part chunking, line-number error rewriting, and status hints.

use folo_test_support::TestFile;

use super::*;

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
    let mut it = records_of_unless(reader, "pipe".into(), true, Some(&stop));
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

#[test]
fn rewrite_part_error_maps_part_indices_to_file_line_numbers() {
    let mk = |msg: &str| Error::new(StatusCode::BAD_REQUEST, "invalid_body", msg);
    // Cross-part conflict on line 20001 (1st record of part 2).
    let err = rewrite_part_error(
        mk(
            "prediction 1 (image '1', unchanged) was received in an earlier part with other boxes: answer each image and condition once",
        ),
        "preds.jsonl",
        &[20001],
    );
    assert_eq!(
        err.body.message,
        "preds.jsonl: line 20001 (image '1', unchanged) was received in an earlier part with other boxes: answer each image and condition once"
    );

    // Intra-part duplicate translates both record indices to exact file lines (accounting for blank lines).
    let err = rewrite_part_error(
        mk(
            "prediction 2 (image '1', unchanged) repeats prediction 1 with other boxes: answer each image and condition once",
        ),
        "preds.jsonl",
        &[10, 15],
    );
    assert_eq!(
        err.body.message,
        "preds.jsonl: line 15 (image '1', unchanged) repeats line 10 with other boxes: answer each image and condition once"
    );

    // An image id containing a similar substring does not hijack the suffix rewrite, and an unknown-image error
    // with that image id leaves the image id completely intact.
    let tricky = rewrite_part_error(
        mk(
            "prediction 2 (image 'x repeats prediction 1 with other boxes:', unchanged) repeats prediction 1 with other boxes: answer each image and condition once",
        ),
        "preds.jsonl",
        &[10, 15],
    );
    assert_eq!(
        tricky.body.message,
        "preds.jsonl: line 15 (image 'x repeats prediction 1 with other boxes:', unchanged) repeats line 10 with other boxes: answer each image and condition once"
    );
    let unknown_img = rewrite_part_error(
        mk(
            "prediction 2 (image 'x repeats prediction 1 with other boxes:', unchanged): the benchmark has no image 'x repeats prediction 1 with other boxes:'",
        ),
        "preds.jsonl",
        &[10, 15],
    );
    assert_eq!(
        unknown_img.body.message,
        "preds.jsonl: line 15 (image 'x repeats prediction 1 with other boxes:', unchanged): the benchmark has no image 'x repeats prediction 1 with other boxes:'"
    );

    // Non-record server errors stay untouched.
    let other = rewrite_part_error(mk("benchmark 'bn_1' does not exist"), "preds.jsonl", &[1]);
    assert_eq!(other.body.message, "benchmark 'bn_1' does not exist");
}
