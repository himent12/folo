//! Unit tests of the real store's runs: metrics and step cursors, the queue (positions, shutdown, early stops),
//! log lines, throughput samples, model ids, resumes and runs left behind by a previous process.

use super::*;
use axum::http::StatusCode;
use folo_api::{Epoch, LogLevel, LogLine, StatusEvent, TrainConfig};
use folo_test_support::TestDir;

use super::history::read_metrics;
use crate::store::{LogsQuery, MetricsQuery, QueueMove};

#[test]
fn resumed_metrics_replace_repeated_steps() {
    let dir = TestDir::new("runs-metrics");
    let p = dir.join("metrics.jsonl");
    let loss = r#""loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"imgs_per_s":1,"eta_s":1"#;
    let val = r#""val":{"map50":0.1,"map50_95":0.05,"precision":0.1,"recall":0.1,"per_class":[]}"#;
    let step = |e: u32, s: u64| format!(r#"{{"event":"step","epoch":{e},"step":{s},{loss}}}"#);
    let epoch = |e: u32| {
        format!(
            r#"{{"event":"epoch","epoch":{e},"train_loss":{{"box":1,"cls":1,"dfl":1,"total":3}},{val},"duration_s":1}}"#
        )
    };
    // Epoch 1 (steps 1-2), epoch 2 interrupted after step 3, resumed: steps 3-4 again, epoch 2.
    let lines = [step(1, 1), step(1, 2), epoch(1), step(2, 3), step(2, 3), step(2, 4), epoch(2)];
    std::fs::write(&p, lines.join("\n")).unwrap();
    let m = read_metrics(&p);
    assert_eq!(m.steps.iter().map(|s| s.step).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert_eq!(m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>(), [1, 2]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Mid-epoch, `/metrics` adds the steps the trainer sent live (some skipped) to the flushed file: all of them for
/// a chart, but with a step cursor only as far as they follow on without a gap, so a poller misses none.
#[test]
fn a_step_cursor_never_skips_a_step() {
    let dir = TestDir::new("runs-cursor");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"c"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let e = runs.register(config, dataset, None, None, 0, "cpu".into(), None).unwrap();
    let step = |s: u64| -> Step {
        let loss = r#""loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"imgs_per_s":1,"eta_s":1"#;
        serde_json::from_str(&format!(r#"{{"epoch":1,"step":{s},{loss}}}"#)).unwrap()
    };
    let line = |s: u64| folo_vision::train::event_record(&RunEvent::Step(step(s))).to_string();
    std::fs::write(e.metrics_path(), [line(1), line(2)].join("\n")).unwrap();
    for s in [2, 3, 5] {
        e.on_event(&RunEvent::Step(step(s)));
    }
    let steps =
        |q: MetricsQuery| runs.metrics(e.id(), &q).unwrap().steps.iter().map(|s| s.step).collect::<Vec<_>>();
    assert_eq!(steps(MetricsQuery::default()), [1, 2, 3, 5]);
    assert_eq!(steps(MetricsQuery { after_step: Some(0), ..MetricsQuery::default() }), [1, 2, 3]);
    assert_eq!(steps(MetricsQuery { after_step: Some(2), ..MetricsQuery::default() }), [3]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The record of run `id` queued with `config` that never started.
fn queued_file(id: &str, config: TrainConfig) -> RunFile {
    let run = Run {
        id: id.into(),
        name: config.name.clone(),
        status: RunStatus::Queued,
        epochs: config.epochs,
        config,
        created: now(),
        started: None,
        finished: None,
        paused_s: None,
        epoch: 0,
        step: 0,
        steps_per_epoch: 0,
        eta_s: None,
        best: None,
        device: "cpu".into(),
        model_id: None,
        error: None,
        error_code: None,
        resolved: None,
        queue_position: None,
        cloud: None,
        dataset: None,
    };
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    RunFile {
        run,
        dataset,
        weights: None,
        val_offset: 0,
        epochs_done: 0,
        last_metrics: None,
        resumed_at: None,
        alive_at: None,
    }
}

/// A create or resume that passed `check_open` before the shutdown and reaches `enqueue` after it: the run is
/// marked stopped instead of queued, and no worker starts.
#[test]
fn enqueue_after_shutdown_refuses_and_stops_the_run() {
    let dir = TestDir::new("runs-late");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"late"}"#).unwrap();
    let file = queued_file("run_late", config);
    std::fs::create_dir_all(dir.join("run_late")).unwrap();
    let entry = Arc::new(RunEntry::new("run_late".into(), dir.join("run_late"), file));
    runs.list.write().unwrap().push(entry.clone());

    runs.shutdown(None);
    let e = runs.enqueue(&entry).unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::SERVICE_UNAVAILABLE, "shutting_down"));
    assert_eq!(entry.run().status, RunStatus::Stopped);
    assert!(runs.queue.lock().unwrap().runs.is_empty());
    assert!(!*runs.worker.lock().unwrap(), "no worker for a refused run");
    let logs = runs.logs("run_late", &LogsQuery::last(10)).unwrap();
    assert!(logs.lines.iter().any(|l| l.message.contains("not queued")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A run's log has writers on several threads (the trainer on the worker, a stop or a queue move from a
/// request): every line they append reads back whole.
#[test]
fn lines_appended_at_once_stay_whole() {
    let dir = TestDir::new("runs-append");
    let path = dir.join("log.jsonl");
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let path = path.clone();
            std::thread::spawn(move || {
                for i in 0..500 {
                    append_line(&path, &format!(r#"{{"thread":{t},"line":{i},"pad":"{}"}}"#, "x".repeat(64)));
                }
            })
        })
        .collect();
    threads.into_iter().for_each(|t| t.join().unwrap());
    let text = std::fs::read_to_string(&path).unwrap();
    let whole = text.lines().filter(|l| serde_json::from_str::<serde_json::Value>(l).is_ok()).count();
    assert_eq!((whole, text.lines().count()), (2000, 2000));
}

/// A stop that lands after the trainer's last look at its flag (it reported `completed`, the run is not published
/// yet) is accepted (`stopping`) but has nothing left to stop: the run is published `completed`, and the log's
/// acknowledgment, its last line, says that this can happen (#333).
#[test]
fn a_stop_after_the_trainers_last_look_completes_the_run() {
    let dir = TestDir::new("runs-stop-late");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"r"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let e = runs.register(config, dataset, None, None, 0, "cpu".into(), None).unwrap();
    assert!(e.claim());
    e.on_event(&RunEvent::Status(StatusEvent::new(RunStatus::Running, None)));
    e.execute(|| {
        e.on_event(&RunEvent::Status(StatusEvent::new(RunStatus::Completed, None)));
        assert_eq!(runs.stop(e.id()).unwrap().status, RunStatus::Stopping);
        Ended::Finished
    });
    assert_eq!(runs.snapshot(e.id()).unwrap().status, RunStatus::Completed);
    let logs = runs.logs(e.id(), &LogsQuery::last(10)).unwrap();
    assert_eq!(logs.lines.last().unwrap().message, STOP_REQUESTED, "{:?}", logs.lines);
    runs.shutdown(None);
}

/// A stop that lands between a resume's restart and its enqueue (the run is `queued` but not in the queue yet,
/// so it stops at once): the run stays stopped and out of the queue, and no worker starts for it.
#[test]
fn a_run_stopped_before_it_is_queued_stays_out_of_the_queue() {
    let dir = TestDir::new("runs-stop-resume");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"r"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let e = runs.register(config.clone(), dataset, None, None, 0, "cpu".into(), None).unwrap();
    runs.stop(e.id()).unwrap();
    // `resume`: restart, then (after the interleaved stop) enqueue.
    runs.restart(&e, config, None).unwrap();
    assert_eq!(runs.stop(e.id()).unwrap().status, RunStatus::Stopped);
    runs.enqueue(&e).unwrap();
    let run = runs.snapshot(e.id()).unwrap();
    assert_eq!((run.status, run.queue_position), (RunStatus::Stopped, None));
    assert!(!runs.busy() && !*runs.worker.lock().unwrap(), "a stopped run waits in the queue");
    let logs = runs.logs(e.id(), &LogsQuery::last(10)).unwrap();
    assert!(logs.lines.last().unwrap().message.contains("stopped before it started"), "{:?}", logs.lines);
    runs.shutdown(None);
}

/// `list` reads one queue: positions stay unique and gapless while another thread keeps moving runs.
#[test]
fn listed_queue_positions_are_one_snapshot() {
    let dir = TestDir::new("runs-list");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"q"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let ids: Vec<String> = (0..4).map(|i| format!("run_q{i}")).collect();
    for id in &ids {
        let run = Run {
            id: id.clone(),
            name: "q".into(),
            status: RunStatus::Queued,
            epochs: config.epochs,
            config: config.clone(),
            created: now(),
            started: None,
            finished: None,
            paused_s: None,
            epoch: 0,
            step: 0,
            steps_per_epoch: 0,
            eta_s: None,
            best: None,
            device: "cpu".into(),
            model_id: None,
            error: None,
            error_code: None,
            resolved: None,
            queue_position: None,
            cloud: None,
            dataset: None,
        };
        let file = RunFile {
            run,
            dataset: dataset.clone(),
            weights: None,
            val_offset: 0,
            epochs_done: 0,
            last_metrics: None,
            resumed_at: None,
            alive_at: None,
        };
        std::fs::create_dir_all(dir.join(id)).unwrap();
        let entry = Arc::new(RunEntry::new(id.clone(), dir.join(id), file));
        runs.list.write().unwrap().push(entry.clone());
        // Queued without a worker, so nothing leaves the queue.
        runs.queue.lock().unwrap().runs.push_back(entry);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let mover = {
        let (runs, ids, stop) = (runs.clone(), ids.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                let mv = if i % 2 == 0 { QueueMove::Top } else { QueueMove::Bottom };
                runs.move_queued(&ids[i % ids.len()], mv).unwrap();
                i += 1;
            }
        })
    };
    for _ in 0..300 {
        let mut places: Vec<u32> = runs.list().iter().filter_map(|r| r.queue_position).collect();
        places.sort_unstable();
        assert_eq!(places, [1, 2, 3, 4]);
    }
    stop.store(true, Ordering::Relaxed);
    mover.join().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The throughput samples of a run: the session's first epoch is warm-up, an epoch whose last step was not
/// sent (a non-finite final batch) is no sample, and the rest go to their phase with a validation rate from the
/// remainder of the epoch.
#[test]
fn only_complete_epochs_after_the_first_are_samples() {
    let tp = Arc::new(Throughput::in_memory(Default::default()));
    let device = crate::throughput::DeviceId { device: "cpu".into(), device_name: "a cpu".into() };
    let key = TrainKey { device, arch: folo_api::Arch::N, imgsz: 64, batch: 4, amp: false };
    let mut m = Measure {
        throughput: tp.clone(),
        key,
        train_images: 10,
        val_images: 2,
        epochs: 4,
        mosaic: true,
        close_mosaic_epochs: 1,
        device: folo_core::Device::Cpu,
        first: None,
        last_step: None,
    };
    let loss = r#""loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"eta_s":1"#;
    let step = |e: u32, s: u64, ips: f32| -> Step {
        serde_json::from_str(&format!(
            r#"{{"epoch":{e},"step":{s},"imgs_per_s":{ips},"steps_per_epoch":3,"epochs":4,{loss}}}"#
        ))
        .unwrap()
    };
    let val = r#""val":{"map50":0.1,"map50_95":0.05,"precision":0.1,"recall":0.1,"per_class":[]}"#;
    let epoch = |e: u32, d: f64| -> Epoch {
        serde_json::from_str(&format!(
            r#"{{"epoch":{e},"train_loss":{{"box":1,"cls":1,"dfl":1,"total":3}},{val},"duration_s":{d}}}"#
        ))
        .unwrap()
    };
    for (e, s, ips, d) in [(1, 3, 5.0, 4.0), (2, 5, 7.0, 1.5), (3, 9, 10.0, 1.2), (4, 12, 20.0, 0.7)] {
        m.last_step = Some(step(e, s, ips));
        m.epoch(&epoch(e, d));
    }
    let rows = tp.file().training;
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    // Epoch 1: warm-up; epoch 2 ended on step 5 of 6: skipped; epoch 3 mosaic, epoch 4 close mosaic.
    assert_eq!((r.mosaic.values.as_slice(), r.close_mosaic.values.as_slice()), (&[10.0][..], &[20.0][..]));
    // Validation: 2 images in 1.2 - 10 / 10 = 0.2 s and 0.7 - 10 / 20 = 0.2 s.
    assert_eq!(r.val.values.len(), 2);
    assert!(r.val.values.iter().all(|v| (v - 10.0).abs() < 1e-6), "{:?}", r.val.values);
    assert!(r.startup.is_empty(), "epoch 2 was no sample, so no startup");
}

#[test]
fn run_model_ids() {
    assert_eq!(run_model_id("run_7d1e", false), "m_7d1e");
    assert_eq!(run_model_id("run_7d1e", true), "m_7d1e-last");
}

/// A failed run names the epoch it failed in (its steps were in epoch 2, one epoch had finished); a stopped one the
/// last epoch it finished (its checkpoint's).
#[test]
fn a_failed_run_names_the_epoch_it_failed_in() {
    let dir = TestDir::new("runs-failed-epoch");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"f"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let e = runs.register(config, dataset, None, None, 0, "cpu".into(), None).unwrap();
    let loss = r#""loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"imgs_per_s":1,"eta_s":1"#;
    let step: Step = serde_json::from_str(&format!(r#"{{"epoch":2,"step":25,{loss}}}"#)).unwrap();
    e.on_event(&RunEvent::Step(step));
    for (status, epoch) in [(RunStatus::Failed, 2), (RunStatus::Stopped, 1)] {
        let mut st = e.state.lock().unwrap();
        st.file.epochs_done = 1;
        st.file.run.epoch = 2;
        e.finish(&mut st, status, None);
        assert_eq!(st.file.run.epoch, epoch, "{status:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_epoch_is_one_log_line() {
    let loss = r#"{"box":1,"cls":1,"dfl":1,"total":3.21449}"#;
    let val = r#"{"map50":0.1,"map50_95":0.0734,"precision":0.1,"recall":0.1,"per_class":[]}"#;
    let e: Epoch =
        serde_json::from_str(&format!(r#"{{"epoch":3,"train_loss":{loss},"val":{val},"duration_s":15.24}}"#))
            .unwrap();
    assert_eq!(super::entry::epoch_line(&e, 5), "epoch 3/5: loss 3.214, mAP50-95 0.073, 15.2 s");
}

/// A resumed run's `paused_s`: the time from its end to the resume, then the wait in the queue once it trains again;
/// a stop while it waited counts from that resume. A run resumed before it ever started has none.
#[test]
fn a_resumed_run_records_the_time_it_was_not_training() {
    let dir = TestDir::new("runs-paused");
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"p"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let e = runs.register(config.clone(), dataset, None, None, 0, "cpu".into(), None).unwrap();
    let t = unix_now().floor();
    let ended = |started: Option<f64>, finished: f64| {
        let mut st = e.state.lock().unwrap();
        st.file.run.status = RunStatus::Stopped;
        st.file.run.started = started.map(rfc3339);
        st.file.run.finished = Some(rfc3339(finished));
    };
    let paused = || e.run().paused_s;
    // Each of the `adds` additions so far was measured against a "now" at or after `t`, so each may be late by the
    // time this test has taken (whole seconds, as the server counts), and never early.
    let near = |got: Option<f64>, want: f64, adds: f64| {
        let got = got.expect("paused_s set");
        let late = adds * (unix_now().floor() - t);
        assert!((want..=want + late).contains(&got), "paused_s {got}, want {want} to {}", want + late);
    };

    // Stopped before it ever trained: resuming it starts it over, with nothing to leave out.
    ended(None, t - 100.0);
    runs.restart(&e, config.clone(), None).unwrap();
    assert_eq!(paused(), None);
    assert_eq!(serde_json::to_value(e.run()).unwrap().get("paused_s"), None, "absent, not null");

    // Trained, stopped overnight (10 h ago), resumed now: the night counts, and the wait in the queue once it trains.
    ended(Some(t - 40_000.0), t - 36_000.0);
    runs.restart(&e, config.clone(), None).unwrap();
    near(paused(), 36_000.0, 1.0);
    e.state.lock().unwrap().file.resumed_at = Some(t - 60.0);
    e.on_event(&RunEvent::Status(StatusEvent::new(RunStatus::Running, None)));
    near(paused(), 36_060.0, 2.0);
    assert_eq!(e.state.lock().unwrap().file.resumed_at, None);

    // Resumed, stopped while it waited (it never trained in between), resumed again: from the first resume on (each
    // resume is "now"; the first is moved 500 s back by hand).
    ended(Some(t - 40_000.0), t - 50.0);
    runs.restart(&e, config.clone(), None).unwrap();
    e.state.lock().unwrap().file.resumed_at = Some(t - 500.0);
    ended(Some(t - 40_000.0), t - 100.0);
    runs.restart(&e, config.clone(), None).unwrap();
    near(paused(), 36_060.0 + 50.0 + 500.0, 4.0);

    // The reviewer's case: trained 60 s, stopped, resumed an hour later behind a long run (moved back by hand), and
    // stopped while it still waited. Training time, as the API defines it, is 60 s while it waits, once it is
    // stopped there (the wait is settled into `paused_s`), and after the next resume until it trains.
    let training = || {
        let r = e.run();
        let secs = |s: &Option<String>| s.as_deref().and_then(crate::util::parse_rfc3339);
        secs(&r.finished).unwrap_or_else(|| unix_now().floor())
            - secs(&r.started).unwrap()
            - r.paused_s.unwrap()
    };
    let sixty = |got: f64| {
        let late = 2.0 * (unix_now().floor() - t) + 1.0;
        assert!((60.0 - late..=60.0 + late).contains(&got), "trained {got} s, want 60");
    };
    ended(Some(t - 7200.0), t - 7140.0);
    e.state.lock().unwrap().file.run.paused_s = None;
    runs.restart(&e, config.clone(), None).unwrap();
    {
        let mut st = e.state.lock().unwrap();
        st.file.run.paused_s = Some(3540.0);
        st.file.resumed_at = Some(t - 3600.0);
    }
    sixty(training());
    runs.stop(e.id()).unwrap();
    assert_eq!(e.run().status, RunStatus::Stopped);
    assert_eq!(e.state.lock().unwrap().file.resumed_at, None, "the wait is settled");
    sixty(training());
    near(paused(), 7140.0, 2.0);
    runs.restart(&e, config, None).unwrap();
    sixty(training());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A cloud run another process follows (`queued`, never started, so not claimed here): deleting it is `409
/// run_active` and its folder stays; only a local run that never started may be deleted while it waits.
#[test]
fn a_cloud_run_followed_elsewhere_is_not_deleted() {
    let dir = TestDir::new("runs-delete-cloud");
    let config: TrainConfig = serde_json::from_str(
        r#"{"dataset_id":"ds_1","name":"c","compute":{"connection_id":"cc_1","offer_id":"o"}}"#,
    )
    .unwrap();
    let run_dir = dir.join("run_c");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("run.json"), serde_json::to_vec(&queued_file("run_c", config)).unwrap())
        .unwrap();
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let e = runs.delete("run_c").unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (StatusCode::CONFLICT, "run_active"));
    assert!(e.body.message.contains("another FOLO process"), "{}", e.body.message);
    assert!(run_dir.join("run.json").is_file() && runs.get("run_c").is_ok());
    runs.shutdown(None);
}

/// A copy of a run folder next to it (a backup, `run_a - Copy`) has the same id: one run is loaded, from the folder
/// named after the id, not two rows under one id (x3-server-9).
#[test]
fn a_copied_run_folder_is_loaded_once() {
    let dir = TestDir::new("runs-copied");
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"orig"}"#).unwrap();
    let bytes = serde_json::to_vec(&queued_file("run_a", config)).unwrap();
    for folder in ["run_0", "run_a", "run_a - Copy"] {
        std::fs::create_dir_all(dir.join(folder)).unwrap();
        std::fs::write(dir.join(folder).join("run.json"), &bytes).unwrap();
    }
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let list = runs.entries();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].dir(), dir.join("run_a"), "the folder named after the id is the run");
    // The skipped copies are left as they were (the loaded run was marked stopped, they were not touched).
    assert_eq!(std::fs::read(dir.join("run_0/run.json")).unwrap(), bytes);
    runs.shutdown(None);
}

/// A run whose `run.json` has a `resolved` object from before its EMA pair (with `close_mosaic`, or without) gets the
/// pair from its config.json on load; the fields it already has are kept.
#[test]
fn an_older_resolved_gets_its_ema_pair_from_config_json() {
    let dir = TestDir::new("runs-resolved-ema");
    let tuned =
        folo_vision::TrainConfig { transfer: true, epochs: 100, ..folo_vision::TrainConfig::default() };
    let mut old_resolved = serde_json::to_value(tuned.resolved_train(205, 3)).unwrap();
    old_resolved.as_object_mut().unwrap().retain(|k, _| !k.starts_with("ema_"));
    let mut old_train = serde_json::to_value(&tuned).unwrap();
    old_train.as_object_mut().unwrap().remove("ema_run_fraction");
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1"}"#).unwrap();
    for (id, close_mosaic) in [("run_cm", Some(7)), ("run_nocm", None)] {
        let mut file = queued_file(id, config.clone());
        let mut resolved = old_resolved.clone();
        resolved["close_mosaic"] = serde_json::json!(close_mosaic);
        file.run.resolved = Some(serde_json::from_value(resolved).unwrap());
        let run_dir = dir.join(id);
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(run_dir.join("run.json"), serde_json::to_vec(&file).unwrap()).unwrap();
        let cfg =
            serde_json::json!({ "resolved": old_resolved, "train": old_train, "close_mosaic_epochs": 10 });
        std::fs::write(run_dir.join("config.json"), cfg.to_string()).unwrap();
    }
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    for (id, close_mosaic) in [("run_cm", 7), ("run_nocm", 10)] {
        let r = runs.get(id).unwrap().run().resolved.unwrap();
        // A fine-tune from before `ema_run_fraction` trained with the default EMA.
        assert_eq!((r.ema_decay, r.ema_tau), (Some(folo_nn::EMA_DECAY), Some(folo_nn::EMA_TAU)), "{id}");
        assert_eq!(r.close_mosaic, Some(close_mosaic), "{id}");
    }
    runs.shutdown(None);
}

/// A run named past today's 64-character limit by an older version still resumes, its config sent back as it is.
#[test]
fn a_run_with_a_legacy_long_name_resumes() {
    let dir = TestDir::new("runs-long-name");
    let name = "n".repeat(80);
    let config: TrainConfig =
        serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1", "name": name })).unwrap();
    let mut file = queued_file("run_long", config.clone());
    file.dataset.path = dir.to_string_lossy().into_owned();
    let run_dir = dir.join("run_long");
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("run.json"), serde_json::to_vec(&file).unwrap()).unwrap();
    let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let fields = serde_json::to_value(&config).unwrap().as_object().unwrap().clone();
    let (e, wanted) = runs.resume_config("run_long", &fields).unwrap();
    assert_eq!(wanted.name, name);
    assert_eq!(runs.restart(&e, wanted, None).unwrap().name, name);
    runs.shutdown(None);
}

/// A run the server was killed under ends when it last trained, not when the server starts again: at `alive_at`,
/// which a training run keeps up to date; in a `run.json` from before it, at the newest of its files; at a resume it
/// waited after without training again. Its resume then leaves the time the server was down out of training time.
#[test]
fn a_run_left_training_ends_when_it_last_trained() {
    let dir = TestDir::new("runs-killed");
    let open = || Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
    let runs = open();
    let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"k"}"#).unwrap();
    let dataset = DatasetRef {
        id: "ds_1".into(),
        name: "d".into(),
        path: "nowhere".into(),
        format: DatasetFormat::Yolo,
        classes: vec![],
    };
    let e = runs.register(config.clone(), dataset, None, None, 0, "cpu".into(), None).unwrap();
    let t = unix_now().floor();
    let running = |alive: Option<f64>, resumed: Option<f64>| {
        let mut st = e.state.lock().unwrap();
        st.file.run.status = RunStatus::Running;
        st.file.run.started = Some(rfc3339(t - 1000.0));
        st.file.run.finished = None;
        st.file.alive_at = alive;
        st.file.resumed_at = resumed;
        e.persist(&st);
    };
    let after_restart = || {
        let run = open().get(e.id()).unwrap().run();
        assert_eq!((run.status, run.error_code.as_deref()), (RunStatus::Stopped, Some("interrupted")));
        run
    };

    // A step keeps `alive_at` (written at most every ALIVE_EVERY) for a kill to find.
    running(None, None);
    let step: Step = serde_json::from_str(
        r#"{"epoch":1,"step":1,"loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"imgs_per_s":1,"eta_s":1}"#,
    )
    .unwrap();
    e.on_event(&RunEvent::Step(step));
    let alive = e.state.lock().unwrap().file.alive_at.expect("alive_at set by a step");
    assert!((t..=unix_now()).contains(&alive), "alive_at {alive}, now {t}");

    // So does a phase that sends no steps, only log lines: written once the last write is ALIVE_EVERY old by the
    // monotonic clock, also after the wall clock was set back (`alive_at` an hour ahead of now); sooner than that, a
    // report sets it in memory only.
    let on_disk = || {
        let f: RunFile = serde_json::from_slice(&std::fs::read(e.dir.join("run.json")).unwrap()).unwrap();
        f.alive_at.unwrap()
    };
    // Between the start of this test and now, however long the machine took over it.
    let near_now = |at: f64| assert!((t..=unix_now()).contains(&at), "alive_at {at}, now {t}");
    let log = RunEvent::Log(LogLine { t: now(), level: LogLevel::Info, message: "validating".into() });
    {
        let mut st = e.state.lock().unwrap();
        st.file.alive_at = Some(t + 3600.0);
        e.persist(&st);
        st.alive_written = std::time::Instant::now().checked_sub(ALIVE_EVERY);
    }
    let before = std::time::Instant::now();
    e.on_event(&log);
    near_now(on_disk());
    {
        let mut st = e.state.lock().unwrap();
        // The write is recorded when it happens. Moved an hour ahead, it stays recent however long this machine
        // takes to reach the next report (a busy CI runner has taken longer than ALIVE_EVERY).
        let written = st.alive_written.expect("the write is recorded");
        assert!(written >= before, "alive_written is the time of the write");
        st.alive_written = Some(written + std::time::Duration::from_secs(3600));
        st.file.alive_at = Some(t - 50.0);
        e.persist(&st);
    }
    e.on_event(&log);
    assert_eq!(on_disk(), t - 50.0);
    near_now(e.state.lock().unwrap().file.alive_at.unwrap());

    // Killed 900 s after it started, down until now: it ended then, and trained 100 s.
    running(Some(t - 900.0), None);
    assert_eq!(after_restart().finished.as_deref(), Some(rfc3339(t - 900.0).as_str()));

    // A run.json from before `alive_at`: the newest of its files (all written 700 s ago, here).
    running(None, None);
    let then = std::time::UNIX_EPOCH + std::time::Duration::from_secs_f64(t - 700.0);
    for f in ["run.json", "log.jsonl"] {
        std::fs::File::options().write(true).open(e.dir.join(f)).unwrap().set_modified(then).unwrap();
    }
    assert_eq!(after_restart().finished.as_deref(), Some(rfc3339(t - 700.0).as_str()));

    // Resumed 300 s ago and killed while it waited: it ended at that resume, and resuming it now leaves the 300 s
    // out (with the 100 s from its first end to that resume, recorded then).
    running(Some(t - 400.0), Some(t - 300.0));
    e.state.lock().unwrap().file.run.paused_s = Some(100.0);
    e.persist(&e.state.lock().unwrap());
    let runs = open();
    let e2 = runs.get(e.id()).unwrap();
    assert_eq!(e2.run().finished.as_deref(), Some(rfc3339(t - 300.0).as_str()));
    runs.restart(&e2, config, None).unwrap();
    let paused = e2.run().paused_s.unwrap();
    let late = unix_now().floor() - t;
    assert!((400.0..=400.0 + late).contains(&paused), "paused_s {paused}, want 400 to {}", 400.0 + late);
    let _ = std::fs::remove_dir_all(&dir);
}
