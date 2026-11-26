use super::*;

#[test]
fn labels_follow_yolo_beside_images_and_mirror_rules() {
    let dir = folo_test_support::TestDir::new("mcp-confine-label-paths");
    // Dataset below an `images` ancestor with NO `labels` mirror: labels sit beside images.
    let ds = dir.join("images").join("birds");
    std::fs::create_dir_all(&ds).unwrap();
    let lp = LabelPaths::new(&ds);
    assert_eq!(lp.of(&ds.join("a.jpg")), ds.join("a.txt"));
    // An `images/` subfolder inside the dataset still maps to `labels/` inside the dataset.
    assert_eq!(
        lp.of(&ds.join("images").join("train").join("a.jpg")),
        ds.join("labels").join("train").join("a.txt")
    );

    // Once the mirror `<dir>/labels/birds` exists, labels resolve through the mirror instead of beside.
    let mirror = dir.join("labels").join("birds");
    std::fs::create_dir_all(&mirror).unwrap();
    let lp_mirror = LabelPaths::new(&ds);
    assert_eq!(lp_mirror.of(&ds.join("a.jpg")), mirror.join("a.txt"));

    // A root spelled with `..` through a nested `images` segment (`images/images/../birds`, or
    // `images\images\..\birds` on Windows) resolves both its mirror probe and label lookup consistently.
    let nested_images = dir.join("images").join("images");
    std::fs::create_dir_all(&nested_images).unwrap();
    let dotdot_root = nested_images.join("..").join("birds");
    let lp_dotdot = LabelPaths::new(&dotdot_root);
    assert_eq!(lp_dotdot.of(&dotdot_root.join("a.jpg")), mirror.join("a.txt"));
    std::fs::remove_dir(&mirror).unwrap();
    let lp_dotdot_beside = LabelPaths::new(&dotdot_root);
    assert_eq!(lp_dotdot_beside.of(&dotdot_root.join("a.jpg")), ds.join("a.txt"));

    // On Unix, a `data.yaml` folder reached through a symlink (`work/current -> releases/v7/meta`) with
    // `../images/birds` resolves its mirror in the link target's parent (`releases/v7/labels/birds`).
    #[cfg(unix)]
    {
        let release = dir.join("releases/v7");
        std::fs::create_dir_all(release.join("images/birds")).unwrap();
        std::fs::create_dir_all(release.join("meta")).unwrap();
        std::fs::create_dir_all(dir.join("work")).unwrap();
        std::os::unix::fs::symlink(release.join("meta"), dir.join("work/current")).unwrap();
        let linked_root = dir.join("work/current/../images/birds");
        let linked_img = linked_root.join("a.jpg");
        let canon_img = std::fs::canonicalize(release.join("images/birds")).unwrap().join("a.jpg");
        let canon_mirror =
            std::fs::canonicalize(&release).unwrap().join("labels").join("birds").join("a.txt");
        assert_eq!(LabelPaths::new(&linked_root).of(&linked_img), canon_img.with_extension("txt"));
        std::fs::create_dir_all(release.join("labels/birds")).unwrap();
        assert_eq!(LabelPaths::new(&linked_root).of(&linked_img), canon_mirror);
    }
}

#[test]
fn local_url_detection_covers_loopback_unspecified_and_remote_hosts() {
    for local in [
        "http://127.0.0.1:7878",
        "HTTP://127.0.0.1:7878",
        "http://localhost:7878/api",
        "http://LOCALHOST.:7878",
        "http://foo.localhost:7878",
        "https://[::1]:7878",
        "http://[::1%25lo]:7878",
        "http://0.0.0.0:7878",
        "http://[::]:7878",
        "http://[::ffff:127.0.0.1]:7878",
        // Empty hosts and non-canonical IPv4 loopback/unspecified literals fail closed as local.
        "",
        "http://:7878",
        "http:///api",
        "http://127.1:7878",
        "http://0:7878",
        "http://2130706433:7878",
        "http://0x7f000001:7878",
        "http://0177.0.0.1:7878",
    ] {
        assert!(is_local_url(local), "expected local: {local}");
    }
    for remote in [
        "http://192.168.1.5:7878",
        "https://gpu.example.com:7878",
        "https://[2001:db8::1]:7878",
        "http://notlocalhost.example.com:7878",
        // Multi-byte UTF-8 hostname near the `.localhost` suffix length must never panic.
        "http://éééééa:7878",
    ] {
        assert!(!is_local_url(remote), "expected remote: {remote}");
    }
}

#[test]
fn push_parent_preserves_root_and_leading_relative_parents() {
    let mut at_root = PathBuf::from(if cfg!(windows) { r"C:\" } else { "/" });
    push_parent(&mut at_root);
    assert_eq!(at_root, PathBuf::from(if cfg!(windows) { r"C:\" } else { "/" }));

    let mut rel = PathBuf::from("..");
    push_parent(&mut rel);
    assert_eq!(rel, PathBuf::from("../.."));
    rel.push("a");
    assert_eq!(rel, PathBuf::from("../../a"));
}

#[test]
fn denied_reads_become_path_outside_data_dir() {
    let e = Error::internal(format!("reading x: x {}", folo_data::guard::DENIED));
    assert_eq!(map_denied(e).body.code, "path_outside_data_dir");
    assert_eq!(map_denied(Error::internal("other")).body.code, "internal");
}

#[test]
fn dataset_does_not_cache_transient_store_errors() {
    use folo_api::{
        Dataset, ExportResult, ExportSpec, GpuInfo, ImagePage, IssuePage, Model, ModelEval, PredictResponse,
        RegisterDataset, Run, RunLogs, RunMetrics, RunSamples, SyntheticDataset, TrainConfig,
    };
    use folo_server::store::{ImageData, LogsQuery, MetricsQuery, PredictInput, Subscription};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FlakyStore {
        inner: folo_server::MockStore,
        calls: AtomicUsize,
    }
    impl Store for FlakyStore {
        fn is_mock(&self) -> bool {
            self.inner.is_mock()
        }
        fn gpus(&self) -> Vec<GpuInfo> {
            self.inner.gpus()
        }
        fn list_datasets(&self) -> Result<Vec<Dataset>, Error> {
            self.inner.list_datasets()
        }
        fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset, Error> {
            self.inner.register_dataset(req)
        }
        fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset, Error> {
            self.inner.create_synthetic(req)
        }
        fn get_dataset(&self, id: &str) -> Result<Dataset, Error> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(Error::new(StatusCode::BAD_GATEWAY, "server_unreachable", "temporary outage"));
            }
            self.inner.get_dataset(id)
        }
        fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage, Error> {
            self.inner.list_images(id, q)
        }
        fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<IssuePage, Error> {
            self.inner.list_issues(id, q)
        }
        fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData, Error> {
            self.inner.image(id, image_id, thumb)
        }
        fn delete_dataset(&self, id: &str) -> Result<(), Error> {
            self.inner.delete_dataset(id)
        }
        fn list_models(&self) -> Result<Vec<Model>, Error> {
            self.inner.list_models()
        }
        fn get_model(&self, id: &str) -> Result<Model, Error> {
            self.inner.get_model(id)
        }
        fn model_eval(&self, id: &str) -> Result<ModelEval, Error> {
            self.inner.model_eval(id)
        }
        fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult, Error> {
            self.inner.export_model(id, spec)
        }
        fn list_runs(&self) -> Result<Vec<Run>, Error> {
            self.inner.list_runs()
        }
        fn create_run(&self, config: TrainConfig) -> Result<Run, Error> {
            self.inner.create_run(config)
        }
        fn get_run(&self, id: &str) -> Result<Run, Error> {
            self.inner.get_run(id)
        }
        fn stop_run(&self, id: &str) -> Result<Run, Error> {
            self.inner.stop_run(id)
        }
        fn delete_run(&self, id: &str) -> Result<(), Error> {
            self.inner.delete_run(id)
        }
        fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics, Error> {
            self.inner.run_metrics(id, q)
        }
        fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples, Error> {
            self.inner.run_samples(id, epoch)
        }
        fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs, Error> {
            self.inner.run_logs(id, q)
        }
        fn subscribe(&self, id: &str) -> Result<Subscription, Error> {
            self.inner.subscribe(id)
        }
        fn predict(&self, input: PredictInput) -> Result<PredictResponse, Error> {
            self.inner.predict(input)
        }
    }

    let dir = folo_test_support::TestDir::new("mcp-confine-transient");
    let c = Confinement::new(PathPolicy { data_dir: dir.to_path_buf(), confine: true }, None);
    let store = FlakyStore { inner: folo_server::MockStore::new(), calls: AtomicUsize::new(0) };
    let id = store.list_datasets().unwrap().remove(0).id;

    let first = c.dataset(&store, &id).unwrap_err();
    assert_eq!(first.body.code, "server_unreachable");
    assert!(c.dataset(&store, &id).is_ok(), "transient error must not be permanently cached");
    assert_eq!(store.calls.load(Ordering::SeqCst), 2);
    // Now that Ok(()) was cached, a third call uses the cache without hitting the store again.
    assert!(c.dataset(&store, &id).is_ok());
    assert_eq!(store.calls.load(Ordering::SeqCst), 2);
}

#[test]
fn local_proxy_confinement_validates_before_get_dataset_and_rejects_invalid_local_format() {
    use folo_api::{
        Dataset, DatasetFormat, ExportResult, ExportSpec, GpuInfo, ImagePage, IssuePage, Model, ModelEval,
        PredictResponse, RegisterDataset, Run, RunLogs, RunMetrics, RunSamples, Splits, SyntheticDataset,
        TrainConfig,
    };
    use folo_server::store::{ImageData, LogsQuery, MetricsQuery, PredictInput, Subscription};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let root = folo_test_support::TestDir::new("mcp-confine-precheck");
    let data_dir = root.join("data");
    let outside = root.join("outside");
    let ds_dir = data_dir.join("ds");
    let empty_dir = data_dir.join("not_a_dataset");
    std::fs::create_dir_all(outside.join("images/train")).unwrap();
    std::fs::create_dir_all(&ds_dir).unwrap();
    std::fs::create_dir_all(&empty_dir).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3]))
        .save(outside.join("images/train/a.png"))
        .unwrap();
    std::fs::write(
        ds_dir.join("data.yaml"),
        format!("path: {}\ntrain: images/train\nnames: [box]\n", folo_api::json::path(&outside)),
    )
    .unwrap();

    // `pre_register_local` rejects an unrecognized folder locally (`unknown_format`) without forwarding to server.
    let proxy = Arc::new(HttpStore::new("http://127.0.0.1:9", None));
    let c = Confinement::new(PathPolicy { data_dir: data_dir.clone(), confine: true }, Some(proxy));
    let fmt_err = c.pre_register_local(&empty_dir, DatasetFormat::Auto).unwrap_err();
    assert_eq!(fmt_err.body.code, "unknown_format");

    // In local proxy mode, `Confinement::dataset` inspects via registry-only `list_datasets` and validates
    // locally BEFORE calling `get_dataset` (which would run `fresh_index` on unguarded `folo serve`).
    struct SpyStore {
        ds: Dataset,
        get_calls: AtomicUsize,
    }
    impl Store for SpyStore {
        fn is_mock(&self) -> bool {
            false
        }
        fn gpus(&self) -> Vec<GpuInfo> {
            vec![]
        }
        fn list_datasets(&self) -> Result<Vec<Dataset>, Error> {
            Ok(vec![self.ds.clone()])
        }
        fn register_dataset(&self, _: RegisterDataset) -> Result<Dataset, Error> {
            unreachable!()
        }
        fn create_synthetic(&self, _: SyntheticDataset) -> Result<Dataset, Error> {
            unreachable!()
        }
        fn get_dataset(&self, _: &str) -> Result<Dataset, Error> {
            self.get_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.ds.clone())
        }
        fn list_images(&self, _: &str, _: &ImageQuery) -> Result<ImagePage, Error> {
            unreachable!("outside dataset must fail before list_images")
        }
        fn list_issues(&self, _: &str, _: &IssueQuery) -> Result<IssuePage, Error> {
            unreachable!("outside dataset must fail before list_issues")
        }
        fn image(&self, _: &str, _: &str, _: Option<u32>) -> Result<ImageData, Error> {
            unreachable!()
        }
        fn delete_dataset(&self, _: &str) -> Result<(), Error> {
            unreachable!()
        }
        fn list_models(&self) -> Result<Vec<Model>, Error> {
            unreachable!()
        }
        fn get_model(&self, _: &str) -> Result<Model, Error> {
            unreachable!()
        }
        fn model_eval(&self, _: &str) -> Result<ModelEval, Error> {
            unreachable!()
        }
        fn export_model(&self, _: &str, _: ExportSpec) -> Result<ExportResult, Error> {
            unreachable!()
        }
        fn list_runs(&self) -> Result<Vec<Run>, Error> {
            unreachable!()
        }
        fn create_run(&self, _: TrainConfig) -> Result<Run, Error> {
            unreachable!()
        }
        fn get_run(&self, _: &str) -> Result<Run, Error> {
            unreachable!()
        }
        fn stop_run(&self, _: &str) -> Result<Run, Error> {
            unreachable!()
        }
        fn delete_run(&self, _: &str) -> Result<(), Error> {
            unreachable!()
        }
        fn run_metrics(&self, _: &str, _: &MetricsQuery) -> Result<RunMetrics, Error> {
            unreachable!()
        }
        fn run_samples(&self, _: &str, _: Option<u32>) -> Result<RunSamples, Error> {
            unreachable!()
        }
        fn run_logs(&self, _: &str, _: &LogsQuery) -> Result<RunLogs, Error> {
            unreachable!()
        }
        fn subscribe(&self, _: &str) -> Result<Subscription, Error> {
            unreachable!()
        }
        fn predict(&self, _: PredictInput) -> Result<PredictResponse, Error> {
            unreachable!()
        }
    }

    let spy = SpyStore {
        ds: Dataset {
            id: "ds_evil".into(),
            name: "evil".into(),
            path: folo_api::json::path(&ds_dir),
            format: DatasetFormat::Yolo,
            classes: vec!["box".into()],
            splits: Splits { train: 1, val: 0, test: 0 },
            created: "2026-01-01T00:00:00Z".into(),
            stats: None,
            revision: None,
        },
        get_calls: AtomicUsize::new(0),
    };
    let err = c.dataset(&spy, "ds_evil").unwrap_err();
    assert_eq!(err.body.code, "path_outside_data_dir");
    assert_eq!(
        spy.get_calls.load(Ordering::SeqCst),
        0,
        "must reject outside dataset locally before calling get_dataset/fresh_index"
    );
}

#[cfg(unix)]
#[test]
fn discovery_candidates_and_canonical_signature_catch_symlinks_escaping_data_dir() {
    let root = folo_test_support::TestDir::new("mcp-confine-symlinks");
    let data_dir = root.join("data");
    let outside = root.join("outside");
    let ds = data_dir.join("decoy");
    std::fs::create_dir_all(ds.join("images/train")).unwrap();
    std::fs::create_dir_all(ds.join("labels/train")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])).save(ds.join("images/train/a.png")).unwrap();
    std::fs::write(ds.join("labels/train/a.txt"), "0 0.5 0.5 0.4 0.4\n").unwrap();
    std::fs::write(outside.join("secret.yaml"), "names: [leaked]\n").unwrap();

    let proxy = Arc::new(HttpStore::new("http://127.0.0.1:9", None));
    let c = Confinement::new(PathPolicy { data_dir, confine: true }, Some(proxy));

    // 1. Custom `*.yaml` symlink pointing outside `--data-dir` is caught even when a valid YOLO decoy exists.
    std::os::unix::fs::symlink(outside.join("secret.yaml"), ds.join("custom.yaml")).unwrap();
    let err = c.pre_register_local(&ds, folo_api::DatasetFormat::Auto).unwrap_err();
    assert_eq!(err.body.code, "path_outside_data_dir");
    std::fs::remove_file(ds.join("custom.yaml")).unwrap();

    // 2. Replacing a split folder with a symlink pointing outside `--data-dir` invalidates `canonical_signature`
    //    even if filenames, sizes, and mtimes stay identical.
    assert!(c.canonical_signature(std::slice::from_ref(&ds)).is_some());
    let moved = outside.join("train_copy");
    std::fs::rename(ds.join("images/train"), &moved).unwrap();
    std::os::unix::fs::symlink(&moved, ds.join("images/train")).unwrap();
    assert!(
        c.canonical_signature(&[ds]).is_none(),
        "canonical_signature must reject symlinked split folder pointing outside data_dir"
    );
}
