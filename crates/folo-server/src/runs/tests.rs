use super::*;
use axum::http::StatusCode;
use folo_api::{Epoch, TrainConfig};
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
    RunFile { run, dataset, weights: None, val_offset: 0, epochs_done: 0, last_metrics: None }
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
