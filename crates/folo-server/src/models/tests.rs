use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use folo_api::{ExportFormat, ExportSpec, ModelEval, ModelPrecision, TrainedWith};

use super::meta::{fused, precision, trained_with};
use super::weights::DerivedLocks;
use super::*;

#[test]
fn trained_with_reads_folo_train_leniently() {
    let t = |s: &str| trained_with(Some(&s.to_string()));
    // The trainer's resolved config: the three fields, other keys ignored.
    let full = r#"{"name":"a","epochs":80,"imgsz":512,"amp":true,"head":"o2o","augment":{"mosaic":1.0}}"#;
    assert_eq!(t(full), Some(TrainedWith { amp: true, imgsz: 512, epochs: 80 }));
    // Written before mixed precision: no `amp`.
    assert_eq!(t(r#"{"epochs":3,"imgsz":64}"#), Some(TrainedWith { amp: false, imgsz: 64, epochs: 3 }));
    // Missing, unparsable or without the fields: none, never an error.
    assert_eq!(trained_with(None), None);
    for bad in [
        "",
        "not json",
        "[]",
        r#"{"epochs":3}"#,
        r#"{"epochs":-1,"imgsz":64}"#,
        r#"{"amp":"yes","epochs":3,"imgsz":64}"#,
    ] {
        assert_eq!(t(bad), None, "{bad}");
    }
}

#[test]
fn precision_and_fused_read_their_keys_leniently() {
    assert_eq!(precision(Some("F32")), Some(ModelPrecision::F32));
    assert_eq!(precision(Some("F16")), Some(ModelPrecision::F16));
    for unknown in [None, Some("BF16"), Some("f16"), Some("")] {
        assert_eq!(precision(unknown), None, "{unknown:?}");
    }
    assert_eq!(
        (fused(None), fused(Some("false")), fused(Some("true"))),
        (Some(false), Some(false), Some(true))
    );
    assert_eq!((fused(Some("yes")), fused(Some("TRUE"))), (None, None));
}

/// A run model is named after its run, and a run's name is any text: the export still gets a file.
#[test]
fn models_with_any_name_export() {
    let dir = folo_test_support::TestDir::new("models-export-names");
    let path = dir.join("w.folo");
    let mut m = folo_vision::FoloModel::with_rng(
        folo_nn::io::ModelConfig::new("n", 1).unwrap(),
        &mut folo_core::rng::Rng::seed(1),
    )
    .unwrap();
    m.set_classes(vec!["cat".into()]).unwrap();
    m.set_imgsz(64);
    m.save(&path).unwrap();
    let models = Models::new(dir.join("home"));
    let meta = read_meta(&path).unwrap();
    let spec = ExportSpec { format: ExportFormat::FoloF16, onnx: Default::default() };
    for name in ["cats <v2>? \"best\" | *-best", "tab\there-best", &"long".repeat(80)] {
        let m = ModelRef::new(model_of("m_1".into(), name.into(), &meta, None, None), path.clone(), None);
        let out = models.export(&m, spec).unwrap_or_else(|e| panic!("{name:?}: {e}"));
        let file = Path::new(&out.path).file_name().unwrap().to_string_lossy().into_owned();
        assert!(file.ends_with("-f16.folo") && file.len() <= 255, "{file}");
        assert!(!file.contains(['<', '>', '?', '"', '|', '*', '\t']), "{file}");
    }
}

/// A damaged weights file is said once per version of the file, not on every listing (x3-server-8).
#[test]
fn a_damaged_weights_file_is_warned_about_once() {
    let dir = folo_test_support::TestDir::new("models-skipped");
    let models = Models::new(dir.join("home"));
    std::fs::create_dir_all(models.models_dir()).unwrap();
    let path = models.models_dir().join("broken.folo");
    std::fs::write(&path, b"not weights").unwrap();
    // What the listings log, caught on this thread.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let log = Captured::default();
    let sink = log.clone();
    let subscriber = tracing_subscriber::fmt().with_ansi(false).with_writer(move || sink.clone()).finish();
    let warnings = || String::from_utf8_lossy(&log.0.lock().unwrap()).matches("skipping").count();
    tracing::subscriber::with_default(subscriber, || {
        for _ in 0..3 {
            assert!(models.scan().is_empty());
        }
        assert_eq!(warnings(), 1, "three listings say it once");
        // Another version of the file (another size) is said again.
        std::fs::write(&path, b"still not weights").unwrap();
        assert!(models.scan().is_empty());
        assert!(models.scan().is_empty());
        assert_eq!(warnings(), 2, "a changed file is said again, once");
    });
    let err = models.meta(&path).unwrap_err();
    assert!(!models.warn_skipped(dir.join("gone.folo").as_path(), &err), "a missing file is not said");
}

fn sample_eval(images: u32) -> ModelEval {
    ModelEval {
        pr: Vec::new(),
        confusion: Vec::new(),
        per_class: Vec::new(),
        images,
        best_f1_conf: None,
        dataset_id: None,
        dataset_name: None,
        split: None,
        split_fallback: None,
        confusion_conf: None,
        confusion_iou: None,
    }
}

/// Concurrent `/eval` misses for the same `(model, key)` run one evaluation; distinct models evaluate in parallel;
/// panics wake waiters to retry cleanly.
#[test]
fn eval_cached_singleflight_deduplicates_and_runs_distinct_models_in_parallel() {
    let dir = folo_test_support::TestDir::new("models-eval-sf");
    let models = Models::new(dir.join("home"));
    let calls = AtomicUsize::new(0);
    let entered = Barrier::new(2);
    let release = Barrier::new(2);

    let (e1, e2) = std::thread::scope(|s| {
        let h1 = s.spawn(|| {
            models
                .eval_cached_with("m_1", "k1", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    entered.wait();
                    release.wait();
                    Ok(sample_eval(42))
                })
                .unwrap()
        });
        entered.wait();
        let h2 = s.spawn(|| {
            models
                .eval_cached_with("m_1", "k1", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(sample_eval(99))
                })
                .unwrap()
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let fk = ("m_1".to_string(), "k1".to_string());
        let mut parked = false;
        while Instant::now() < deadline {
            if models.evals.lock().unwrap().in_flight[&fk].waiters() >= 1 {
                parked = true;
                break;
            }
            std::thread::yield_now();
        }
        release.wait();
        let res = (h1.join().unwrap(), h2.join().unwrap());
        assert!(parked, "waiter did not park in Flight::wait");
        res
    });
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!((e1.images, e2.images), (42, 42));

    // Distinct models run concurrently (bounded 2-thread rendezvous inside both closures).
    let both = crate::util::Rendezvous::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = [("m_a", 10u32), ("m_b", 20u32)]
            .into_iter()
            .map(|(id, n)| {
                let (models, both) = (&models, &both);
                s.spawn(move || {
                    let mut met = false;
                    let got = models
                        .eval_cached_with(id, "k", || {
                            met = both.arrive(Duration::from_secs(5));
                            Ok(sample_eval(n))
                        })
                        .unwrap();
                    assert!(met, "distinct model evals did not run concurrently");
                    assert_eq!(got.images, n);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    // Leader panic wakes waiters to retry cleanly.
    let entered = Barrier::new(2);
    let release = Barrier::new(2);
    let fail = AtomicBool::new(true);
    std::thread::scope(|s| {
        let h1 = s.spawn(|| {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = models.eval_cached_with("m_p", "kp", || {
                    entered.wait();
                    release.wait();
                    if fail.swap(false, Ordering::SeqCst) {
                        panic!("eval panic");
                    }
                    Ok(sample_eval(1))
                });
            }));
        });
        entered.wait();
        let h2 = s.spawn(|| models.eval_cached_with("m_p", "kp", || Ok(sample_eval(7))).unwrap());
        let deadline = Instant::now() + Duration::from_secs(5);
        let fk = ("m_p".to_string(), "kp".to_string());
        let mut parked = false;
        while Instant::now() < deadline {
            if models.evals.lock().unwrap().in_flight[&fk].waiters() >= 1 {
                parked = true;
                break;
            }
            std::thread::yield_now();
        }
        release.wait();
        h1.join().unwrap();
        let got = h2.join().unwrap();
        assert!(parked, "waiter did not park in Flight::wait");
        assert_eq!(got.images, 7);
    });
}

/// Per-`(model, variant)` derived locks let distinct models and distinct variants of one model proceed in
/// parallel, serialize the same `(model, variant)`, and let `forget_weights(id)` wait only for `id`.
#[test]
fn derived_locks_allow_parallel_models_and_variants_while_serializing_same_pair() {
    let locks = Mutex::new(DerivedLocks::default());

    // 1. Two distinct variants of the same model run in parallel (bounded rendezvous).
    let rendezvous = crate::util::Rendezvous::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = ["f16.folo", "onnx"]
            .into_iter()
            .map(|var| {
                let (locks, rendezvous) = (&locks, &rendezvous);
                s.spawn(move || {
                    let (m, v) = locks.lock().unwrap().pair("m_1", var);
                    let _mg = m.read().unwrap();
                    let _vg = v.lock().unwrap();
                    assert!(
                        rendezvous.arrive(Duration::from_secs(5)),
                        "distinct variants did not run concurrently"
                    );
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    // 2. Same `(model, variant)` shares one mutex (serializes); `m_1.write()` blocks `m_1.read()` but not `m_2`.
    {
        let (m1a, v1a) = locks.lock().unwrap().pair("m_1", "f16.folo");
        let (m1b, v1b) = locks.lock().unwrap().pair("m_1", "f16.folo");
        assert!(Arc::ptr_eq(&m1a, &m1b) && Arc::ptr_eq(&v1a, &v1b), "same pair shares one lock");
        let _write_m1 = m1a.write().unwrap();
        assert!(m1b.try_read().is_err(), "model write lock blocks concurrent exports of the same model");
        let (m2, v2) = locks.lock().unwrap().pair("m_2", "f16.folo");
        let _read_m2 = m2.read().unwrap();
        let _guard_v2 = v2.lock().unwrap();
    }

    // Weak entries are pruned once no caller holds them.
    let _held = locks.lock().unwrap().pair("m_3", "onnx");
    let guard = locks.lock().unwrap();
    assert_eq!(guard.models.len(), 1);
    assert_eq!(guard.variants.len(), 1);
}

/// `has_stale_weights` detects files from older versions of the same model while ignoring current-version
/// variants and other models; `remove_weights` prunes stale files (and `forget_weights` removes every file of
/// that model) without touching other models.
#[test]
fn stale_weight_detection_and_pruning_scope_to_model_and_version() {
    let dir = folo_test_support::TestDir::new("models-prune-stale");
    let models = Models::new(dir.join("home"));
    let cache = models.weights_cache();
    std::fs::create_dir_all(&cache).unwrap();

    let cur_f16 = cache.join("m_1.0000000000000002.f16.folo");
    let cur_onnx = cache.join("m_1.0000000000000002.onnx");
    let other_model = cache.join("m_2.0000000000000001.f16.folo");
    for p in [&cur_f16, &cur_onnx, &other_model] {
        std::fs::write(p, b"x").unwrap();
    }
    assert!(
        !models.has_stale_weights("m_1", "m_1.0000000000000002."),
        "no write-lock prune needed when only current-version variants and other models exist"
    );

    let old_onnx = cache.join("m_1.0000000000000001.onnx");
    std::fs::write(&old_onnx, b"x").unwrap();
    assert!(models.has_stale_weights("m_1", "m_1.0000000000000002."));

    models.remove_weights("m_1", Some("m_1.0000000000000002."));
    assert!(!old_onnx.exists(), "stale export of older version is removed");
    assert!(cur_f16.exists() && cur_onnx.exists(), "all current-version variants are kept");
    assert!(other_model.exists(), "other model's export is untouched");

    models.forget_weights("m_1");
    assert!(!cur_f16.exists() && !cur_onnx.exists(), "forget_weights removes every file of m_1");
    assert!(other_model.exists(), "forget_weights leaves other models untouched");
}

#[test]
fn bounded_cache_evicts_least_recently_used() {
    let mut cache = BoundedCache::new(2);
    cache.insert("a", 1);
    cache.insert("b", 2);
    // Touch "a" so "b" becomes least recently used.
    assert_eq!(cache.get("a"), Some(&1));
    cache.insert("c", 3);
    assert_eq!(cache.len(), 2);
    assert!(cache.contains_key("a") && cache.contains_key("c") && !cache.contains_key("b"));

    // Updating an existing key keeps capacity intact and refreshes recency.
    assert_eq!(cache.insert("a", 10), Some(1));
    cache.insert("d", 4);
    assert!(cache.contains_key("a") && cache.contains_key("d") && !cache.contains_key("c"));
}

fn write_tiny_model(path: &Path, seed: u64) {
    let mut m = folo_vision::FoloModel::with_rng(
        folo_nn::io::ModelConfig::new("n", 1).unwrap(),
        &mut folo_core::rng::Rng::seed(seed),
    )
    .unwrap();
    m.set_classes(vec!["cat".into()]).unwrap();
    m.set_imgsz(64);
    m.save(path).unwrap();
}

/// Model metadata, skipped-file, file-id and eval caches stay bounded under churn, and `file_model` resolves from
/// the id index and clears stale entries when a file is deleted.
#[test]
fn models_caches_stay_bounded_and_file_model_tracks_disk_changes() {
    let dir = folo_test_support::TestDir::new("models-bounded-cache");
    let models = Models::with_caps(dir.join("home"), 2, 2, 2, 2);
    std::fs::create_dir_all(models.models_dir()).unwrap();

    let mut paths = Vec::new();
    for i in 0..3 {
        let p = models.models_dir().join(format!("m{i}.folo"));
        write_tiny_model(&p, i + 1);
        paths.push(p);
    }

    let scanned = models.scan();
    assert_eq!(scanned.len(), 3);
    assert!(models.meta.lock().unwrap().len() <= 2, "meta cache bounded");
    assert!(models.by_id.lock().unwrap().len() <= 2, "by_id cache bounded");

    // Consecutive scans over a catalog larger than the cache keep the first `cap` entries warm (no LRU thrash).
    let id0 = file_model_id(&paths[0]);
    let id1 = file_model_id(&paths[1]);
    let id2 = file_model_id(&paths[2]);
    assert!(models.meta.lock().unwrap().contains_key(&paths[0]));
    assert!(models.meta.lock().unwrap().contains_key(&paths[1]));
    assert!(models.by_id.lock().unwrap().contains_key(&id0));
    assert!(models.by_id.lock().unwrap().contains_key(&id1));
    assert_eq!(models.scan().len(), 3);
    assert!(models.meta.lock().unwrap().contains_key(&paths[0]), "second scan did not thrash m0");
    assert!(models.meta.lock().unwrap().contains_key(&paths[1]), "second scan did not thrash m1");

    // Cold lookup of an uncached ID warms only that file (does not read headers of unrelated broken files).
    let broken = models.models_dir().join("z_broken.folo");
    std::fs::write(&broken, b"not weights").unwrap();
    let found2 = models.file_model(&id2).unwrap();
    assert_eq!(found2.path, paths[2]);
    assert!(models.by_id.lock().unwrap().contains_key(&id2), "cold lookup warmed requested ID");
    assert!(models.meta.lock().unwrap().contains_key(&paths[2]), "cold lookup warmed requested meta");
    assert_eq!(models.skipped.lock().unwrap().len(), 0, "cold lookup did not read sibling headers");
    std::fs::remove_file(&broken).unwrap();

    // Warm hit via by_id index, then deletion invalidates cleanly.
    let found = models.file_model(&id2).unwrap();
    assert_eq!(found.path, paths[2]);
    std::fs::remove_file(&paths[2]).unwrap();
    assert!(models.file_model(&id2).is_none(), "deleted model is never served stale");
    assert!(!models.by_id.lock().unwrap().contains_key(&id2));

    // Warm lookup refuses paths outside `models_dir` or replaced with symlinks.
    let outside = dir.join("outside.folo");
    write_tiny_model(&outside, 99);
    assert!(!is_regular_model_file(&models.models_dir(), &outside));
    assert!(!is_regular_model_file(&models.models_dir(), &models.models_dir().join("not_a_model.txt")));
    let link_path = models.models_dir().join("link.folo");
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(&outside, &link_path).is_ok();
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(&outside, &link_path).is_ok();
    if linked {
        let link_id = file_model_id(&link_path);
        models.by_id.lock().unwrap().insert(link_id.clone(), link_path.clone());
        assert!(
            models.file_model(&link_id).is_none(),
            "symlinked model file must not resolve on warm lookup"
        );
        assert!(!models.by_id.lock().unwrap().contains_key(&link_id));
        let _ = std::fs::remove_file(&link_path);
    }

    // Skipped-file cache stays bounded too.
    let err = Error::internal("corrupt");
    for i in 0..3 {
        let bad = models.models_dir().join(format!("bad{i}.folo"));
        std::fs::write(&bad, b"junk").unwrap();
        models.warn_skipped(&bad, &err);
    }
    assert!(models.skipped.lock().unwrap().len() <= 2, "skipped cache bounded");

    // Eval memory cache stays bounded too.
    for i in 0..3 {
        let mid = format!("m_{i}");
        models.eval_cached_with(&mid, "k", || Ok(sample_eval(i))).unwrap();
    }
    assert!(models.evals.lock().unwrap().done.len() <= 2, "eval done cache bounded");
}

#[test]
fn published_model_reads_only_requested_asset() {
    let dir = folo_test_support::TestDir::new("models-published-single");
    let pub_dir = dir.join("published");
    std::fs::create_dir_all(&pub_dir).unwrap();
    let p0 = &folo_api::weights::PUBLISHED[0];
    let p1 = &folo_api::weights::PUBLISHED[1];
    write_tiny_model(&pub_dir.join(p0.asset), 10);
    // Deliberately place a broken file at a sibling published asset: asking for p0 must not touch p1.
    std::fs::write(pub_dir.join(p1.asset), b"corrupt").unwrap();

    let mut models = Models::new(dir.join("home"));
    models.set_published(Some(pub_dir));

    let found = models.published_model(p0.name).unwrap();
    assert_eq!(found.model.id, p0.name);
    assert_eq!(models.skipped.lock().unwrap().len(), 0, "sibling asset was not touched");
    assert!(models.published_model("does-not-exist").is_none());
}
