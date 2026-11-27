//! Multipart upload, cleanup, error-line reporting, and lost-answer recovery tests against a real store and
//! loopback server.

use super::*;

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
    // The third part names an image the benchmark does not have: refused with exact file line 8, and entry removed.
    write("99");
    let e = b.submit_in(id, "refused".into(), true, &file, three, &go).unwrap_err();
    let hint = e.body.hint.clone().unwrap_or_default();
    assert!(
        e.body.message.contains("line 8")
            && e.body.message.contains("99")
            && hint.contains("'refused'")
            && hint.contains("was removed"),
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
            if starts_entry(line) {
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
            if starts_entry(line) {
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

/// Lost answers to the request that opens an external entry:
/// 1. Against a current server (`bench_upload_safe`), the client chooses `entry_id` upfront, finds the entry it
///    opened, stops it conditionally and removes it (zero orphaned entries).
/// 2. If the follow-up state read also fails, the hint names the exact `entry_id` matching the server's entry.
/// 3. If the open request never reached the server, the follow-up read sees 404 and names the chosen `entry_id`.
/// 4. Against a legacy server without `bench_upload_safe`, it falls back to naming the trimmed entry name.
#[test]
fn a_lost_answer_to_the_first_part_names_the_entry_as_listed() {
    with_server("mcp-bench-lost-start", |url, id, dir| {
        // 1. Current server: answer to first part lost -> client knows `entry_id`, stops and removes it.
        let via = Forwarder::new(url, starts_entry);
        let e = http(&via.url)
            .submit_in(id, " opened ".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("entry 'opened' (be_") && hint.contains("was removed"), "{e:?}");
        assert!(http(url).entries(id).unwrap().is_empty(), "lost-start entry must be cleaned up");

        // 2. Both the open answer and follow-up GET are lost -> hint names the exact id created on the server.
        let via_unread = Forwarder::new(url, |line| starts_entry(line) || reads_entry(line));
        let e = http(&via_unread.url)
            .submit_in(id, " stranded ".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let listed = http(url).entries(id).unwrap();
        assert_eq!(listed.len(), 1, "server created the entry");
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(
            hint.contains(&format!("entry 'stranded' ({})", listed[0].id))
                && hint.contains("its state could not be read"),
            "{e:?}"
        );
        http(url).stop_entry(id, &listed[0].id).unwrap();
        http(url).delete_entry(id, &listed[0].id).unwrap();

        // 3. Open request never reached the server -> follow-up GET sees 404, reports not found with id.
        let via_never = Forwarder::new(url, |line| if starts_entry(line) { Fate::Never } else { Fate::Pass });
        let e = http(&via_never.url)
            .submit_in(id, " never ".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("entry 'never' (be_") && hint.contains("was not found on the server"), "{e:?}");
        assert!(http(url).entries(id).unwrap().is_empty());

        // 4. Legacy server without `bench_upload_safe`: falls back to naming the trimmed entry name.
        let legacy = Benchmarks::Http(std::sync::Arc::new(
            crate::proxy::HttpStore::new(&via.url, None)
                .assume_features(vec![folo_api::FEATURE_VISION_BENCH.into()]),
        ));
        let e = legacy
            .submit_in(id, " legacy ".into(), true, &whole_file(dir), THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("may have opened an entry named 'legacy'"), "{e:?}");
        let listed = http(url).entries(id).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!((listed[0].name.as_str(), listed[0].status), ("legacy", BenchEntryStatus::Running));
        http(url).stop_entry(id, &listed[0].id).unwrap();
        http(url).delete_entry(id, &listed[0].id).unwrap();
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
            if stops_entry(line) {
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
        let via = Forwarder::new(url, stops_entry);
        let e = http(&via.url)
            .submit_in(id, "bad".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(
            e.body.message.contains("line 4")
                && e.body.message.contains("99")
                && hint.contains("was removed"),
            "{e:?}"
        );
        assert!(http(url).entries(id).unwrap().is_empty());
    });
}

/// Between cleanup's state read (seeing `running`) and its conditional `POST .../stop?only_if_open=true`:
/// 1. Another client finishes the entry -> conditional stop returns `409 bench_entry_not_open`, score stays intact.
/// 2. Another client stops the entry -> conditional stop returns `409 bench_entry_not_open`, entry is NOT deleted.
#[test]
fn conditional_stop_protects_entry_finished_or_stopped_between_read_and_stop() {
    with_server("mcp-bench-conditional-stop-race", |url, id, dir| {
        let file = dir.join("bad.jsonl");
        let lines: Vec<String> =
            ["1", "2", "3", "99"].iter().map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
        std::fs::write(&file, lines.join("\n")).unwrap();

        // 1. Another client completes and finishes the entry right before cleanup's conditional stop.
        let (other, bench) = (http(url), id.to_string());
        let via_finish = Forwarder::new(url, move |line| {
            if stops_entry(line) {
                let path = line.split(' ').nth(1).unwrap();
                let entry_id = path.split('?').next().unwrap().rsplit('/').nth(1).unwrap();
                other.add_predictions(&bench, entry_id, vec![record("4", 0)]).unwrap();
                other.finish(&bench, entry_id).unwrap();
            }
            Fate::Pass
        });
        let e = http(&via_finish.url)
            .submit_in(id, "raced-finish".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("'raced-finish'") && hint.contains("it is done"), "{e:?}");
        let listed = http(url).entries(id).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].status, BenchEntryStatus::Done);
        assert!(listed[0].score.is_some(), "other client's score must be preserved");
        http(url).delete_entry(id, &listed[0].id).unwrap();

        // 2. Another client stops the entry right before cleanup's conditional stop -> not deleted!
        let (other, bench) = (http(url), id.to_string());
        let via_stop = Forwarder::new(url, move |line| {
            if stops_entry(line) {
                let path = line.split(' ').nth(1).unwrap();
                let entry_id = path.split('?').next().unwrap().rsplit('/').nth(1).unwrap();
                other.stop_entry(&bench, entry_id).unwrap();
            }
            Fate::Pass
        });
        let e = http(&via_stop.url)
            .submit_in(id, "raced-stop".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(hint.contains("'raced-stop'") && hint.contains("it is stopped"), "{e:?}");
        let listed = http(url).entries(id).unwrap();
        assert_eq!(listed.len(), 1, "entry stopped by another client must not be deleted");
        assert_eq!(listed[0].status, BenchEntryStatus::Stopped);
        http(url).delete_entry(id, &listed[0].id).unwrap();
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

/// Non-finite box coordinates (such as `1e100` overflowing `f32`) are caught before HTTP serialization so direct
/// and proxied runs report the same actionable error naming the line, image and box (never `invalid type: null`).
#[test]
fn non_finite_coordinates_report_same_actionable_error_in_direct_and_proxy_modes() {
    with_server("mcp-bench-non-finite-parity", |url, id, dir| {
        let file = dir.join("overflow.jsonl");
        std::fs::write(
            &file,
            "{\"image\":\"1\",\"boxes\":[{\"class\":0,\"score\":0.5,\"x1\":1e100,\"y1\":0,\"x2\":3,\"y2\":3}]}\n\
             {\"image\":\"2\",\"boxes\":[]}\n",
        )
        .unwrap();
        let via_http = http(url)
            .submit_in(id, "overflow".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!(via_http.body.code, "invalid_arguments", "{via_http:?}");
        assert!(
            via_http.body.message.contains("line 1")
                && via_http.body.message.contains("image '1'")
                && via_http.body.message.contains("box 1")
                && via_http.body.message.contains("not a finite number"),
            "{via_http:?}"
        );
        // No orphaned entry was created on the server.
        assert!(http(url).entries(id).unwrap().is_empty());

        // Unscored submissions normalize ignored scores (including non-finite `1e100`) to `1.0` over HTTP.
        let unscored = dir.join("unscored_score_overflow.jsonl");
        let mut lines = vec![
            r#"{"image":"1","boxes":[{"class":0,"score":1e100,"x1":0,"y1":0,"x2":3,"y2":3}]}"#.to_string(),
        ];
        lines.extend((2..=4).map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)));
        std::fs::write(&unscored, lines.join("\n")).unwrap();
        let entry = http(url)
            .submit_in(id, "unscored".into(), false, &unscored, THREE, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(entry.status, BenchEntryStatus::Done);
    });
}

/// A multipart conflict on a later part reports the exact 1-based file line number (including blank lines)
/// and replaces the server's open-entry HTTP retry hint once the entry is removed.
#[test]
fn multipart_conflict_reports_exact_file_line_and_non_contradictory_cleanup_hint() {
    with_server("mcp-bench-multipart-line", |url, id, dir| {
        let file = dir.join("conflict.jsonl");
        // Lines 1..=3 fill part 1 (THREE.records == 3); line 4 is blank; line 5 is part 2 record 1 and conflicts
        // with line 1's empty boxes for image 1.
        std::fs::write(
            &file,
            "{\"image\":\"1\",\"boxes\":[]}\n\
             {\"image\":\"2\",\"boxes\":[]}\n\
             {\"image\":\"3\",\"boxes\":[]}\n\
             \n\
             {\"image\":\"1\",\"boxes\":[{\"class\":0,\"score\":0.5,\"x1\":5,\"y1\":5,\"x2\":15,\"y2\":15}]}\n",
        )
        .unwrap();
        let err = http(url)
            .submit_in(id, "wrong-row".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        assert_eq!(err.body.code, "invalid_body", "{err:?}");
        assert!(
            err.body.message.contains("line 5 (image '1', unchanged)")
                && !err.body.message.contains("prediction 1"),
            "expected exact file line 5: {}",
            err.body.message
        );
        let hint = err.body.hint.as_deref().unwrap_or_default();
        assert!(
            hint.contains("'wrong-row'")
                && hint.contains("was removed")
                && hint.contains("fix the predictions file and run again")
                && !hint.contains(RETRY_PART_HINT),
            "stale open-entry retry hint must be replaced after removal: {hint}"
        );
        assert!(http(url).entries(id).unwrap().is_empty(), "removed entry must not remain on server");

        // Intra-part conflict in part 1 (with a blank line between lines 1 and 3) maps both indices to file lines.
        let intra = dir.join("intra.jsonl");
        std::fs::write(
            &intra,
            "{\"image\":\"1\",\"boxes\":[]}\n\
             \n\
             {\"image\":\"1\",\"boxes\":[{\"class\":0,\"score\":0.5,\"x1\":5,\"y1\":5,\"x2\":15,\"y2\":15}]}\n",
        )
        .unwrap();
        let err = http(url)
            .submit_in(id, "intra".into(), true, &intra, THREE, &AtomicBool::new(false))
            .unwrap_err();
        assert!(
            err.body.message.contains("line 3 (image '1', unchanged) repeats line 1 with other boxes"),
            "{}",
            err.body.message
        );
    });
}

/// When cleanup reads `404 bench_entry_not_found` (another client already removed the entry), cleanup classifies
/// it as closed and replaces `RETRY_PART_HINT`; a transport failure leaves the state unknown and keeps the hint.
#[test]
fn cleanup_replaces_retry_hint_when_entry_already_deleted_but_keeps_it_when_unread() {
    with_server("mcp-bench-cleanup-gone", |url, id, dir| {
        let file = dir.join("conflict.jsonl");
        std::fs::write(
            &file,
            "{\"image\":\"1\",\"boxes\":[]}\n\
             {\"image\":\"2\",\"boxes\":[]}\n\
             {\"image\":\"3\",\"boxes\":[]}\n\
             \n\
             {\"image\":\"1\",\"boxes\":[{\"class\":0,\"score\":0.5,\"x1\":5,\"y1\":5,\"x2\":15,\"y2\":15}]}\n",
        )
        .unwrap();

        // 1. Another client stops and deletes the entry right before cleanup's GET -> `404 bench_entry_not_found`.
        let (other, bench) = (http(url), id.to_string());
        let via = Forwarder::new(url, move |line| {
            if line.starts_with("GET") && line.contains("/entries/") {
                let path = line.split(' ').nth(1).unwrap();
                let entry_id = path.rsplit('/').next().unwrap();
                let _ = other.stop_entry(&bench, entry_id);
                let _ = other.delete_entry(&bench, entry_id);
            }
            Fate::Pass
        });
        let err = http(&via.url)
            .submit_in(id, "already-gone".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = err.body.hint.as_deref().unwrap_or_default();
        assert!(
            hint.contains("was already removed")
                && hint.contains("fix the predictions file and run again")
                && !hint.contains(RETRY_PART_HINT),
            "expected replacement hint when entry is 404 bench_entry_not_found: {hint}"
        );

        // 2. Cleanup's GET fails with a transport error -> state is unknown, so `RETRY_PART_HINT` stays.
        let via_unread = Forwarder::new(url, |line| {
            if line.starts_with("GET") && line.contains("/entries/") { Fate::Never } else { Fate::Pass }
        });
        let err = http(&via_unread.url)
            .submit_in(id, "unread".into(), true, &file, THREE, &AtomicBool::new(false))
            .unwrap_err();
        let hint = err.body.hint.as_deref().unwrap_or_default();
        assert!(
            hint.contains("its state could not be read") && hint.contains(RETRY_PART_HINT),
            "expected retry hint preserved when cleanup cannot read entry state: {hint}"
        );
        // Clean up the left-open entry.
        for e in http(url).entries(id).unwrap() {
            let _ = http(url).stop_entry(id, &e.id);
            let _ = http(url).delete_entry(id, &e.id);
        }
    });
}
