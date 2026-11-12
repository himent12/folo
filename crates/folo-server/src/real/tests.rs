//! Dataset ids across unregistering (tombstones) and deleting a dataset's files while something reads them.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use folo_api::{Dataset, DatasetFormat, RegisterDataset, SyntheticDataset, TrainConfig};
use folo_test_support::TestDir;

use super::RealStore;
use crate::store::Store;

fn register(store: &RealStore, path: &Path) -> Dataset {
    let req = RegisterDataset {
        name: None,
        path: path.to_string_lossy().into_owned(),
        format: DatasetFormat::Auto,
    };
    store.register_dataset(req).unwrap()
}

fn synthetic(store: &RealStore, name: &str) -> Dataset {
    let req: SyntheticDataset =
        serde_json::from_value(serde_json::json!({ "name": name, "images": 6, "imgsz": 64 })).unwrap();
    store.create_synthetic(req).unwrap()
}

/// A run of dataset `id` (queued, without a worker to take it), named after `name`.
fn run_on(store: &RealStore, id: &str, name: &str) -> String {
    let e = store.entry(id).unwrap();
    let config: TrainConfig =
        serde_json::from_value(serde_json::json!({ "dataset_id": id, "name": name })).unwrap();
    let entry =
        store.runs.register(config, store.dataset_ref(&e), None, None, 0, "cpu".into(), None).unwrap();
    entry.id().to_string()
}

#[test]
fn an_unregistered_dataset_gets_its_id_back_and_never_another_ones() {
    let t = TestDir::new("real-tombstones");
    let store = RealStore::open(t.join("home"));
    let (a, b) = (synthetic(&store, "a"), synthetic(&store, "b"));
    let (run_a, _) = (run_on(&store, &a.id, "on-a"), run_on(&store, &b.id, "on-b"));
    // Both unregistered, then registered again (in the other order, by another spelling of the path): each gets its
    // own id back.
    store.delete_dataset(&a.id).unwrap();
    store.delete_dataset(&b.id).unwrap();
    let path = |d: &Dataset| Path::new(&d.path).join("..").join(Path::new(&d.path).file_name().unwrap());
    assert_eq!(register(&store, &path(&b)).id, b.id);
    assert_eq!(register(&store, Path::new(&a.path)).id, a.id);
    // Unregistered while its run exists, another dataset never gets the id; the tombstone stays until the run goes.
    store.delete_dataset(&a.id).unwrap();
    for i in 0..20 {
        let other = synthetic(&store, &format!("other-{i}"));
        assert_ne!(other.id, a.id);
        store.delete_dataset(&other.id).unwrap();
    }
    let registry = super::registry::read_registry(&store.data_dir).unwrap();
    assert_eq!(registry.tombstones.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), [a.id.as_str()]);
    store.runs.delete(&run_a).unwrap();
    store.delete_dataset(&b.id).unwrap();
    let registry = super::registry::read_registry(&store.data_dir).unwrap();
    assert_eq!(registry.tombstones.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), [b.id.as_str()]);
}

/// The reviewer's case: a description outside the folder whose `path:` points into it reads its files, so the
/// folder is not deleted, nor while a queued run or a run of an unregistered dataset reads them (by its dataset
/// path, whatever id it names).
#[test]
fn files_are_not_deleted_while_something_reads_them() {
    let t = TestDir::new("real-delete-readers");
    let store = RealStore::open(t.join("home"));
    let a = synthetic(&store, "a");
    let folder = Path::new(&a.path).to_path_buf();
    let yaml = std::fs::read_to_string(folder.join("data.yaml")).unwrap();
    let outside = t.join("elsewhere");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("data.yaml"), format!("path: '{}'\n{yaml}", a.path)).unwrap();
    let b = register(&store, &outside.join("data.yaml"));
    assert_ne!(a.id, b.id);
    let e = store.delete_dataset_files(&a.id).unwrap_err();
    assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "dataset_in_use"), "{}", e.body.message);
    assert!(folder.join("images").is_dir() && store.entry(&a.id).is_ok());
    store.delete_dataset(&b.id).unwrap();

    // A queued run on the folder, its dataset registered again by its data.yaml (another id).
    let run = run_on(&store, &a.id, "queued");
    store.delete_dataset(&a.id).unwrap();
    let again = register(&store, &folder.join("data.yaml"));
    assert_ne!(again.id, a.id);
    let e = store.delete_dataset_files(&again.id).unwrap_err();
    assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "run_active"), "{}", e.body.message);
    // Stopped, it is a run of an unregistered dataset that keeps its id: still refused.
    store.runs.stop(&run).unwrap();
    let e = store.delete_dataset_files(&again.id).unwrap_err();
    assert_eq!(e.body.code, "dataset_in_use", "{}", e.body.message);
    assert!(folder.join("images").is_dir());
    // Once that run is gone, nothing reads the folder.
    store.runs.delete(&run).unwrap();
    store.delete_dataset_files(&again.id).unwrap();
    assert!(!folder.exists());
}

/// Registration and run admission wait while a deletion of dataset files holds the lease.
#[test]
fn registering_and_creating_runs_wait_for_a_deletion() {
    let t = TestDir::new("real-delete-lease");
    let store = Arc::new(RealStore::open(t.join("home")));
    let data = t.join("shapes");
    let cfg = folo_data::synthetic::SyntheticConfig { images: 4, imgsz: 64, seed: 5, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, &data, folo_data::ImageFormat::Png).unwrap();
    let lease = store.files.write().unwrap();
    let (s1, s2) = (store.clone(), store.clone());
    let registering = std::thread::spawn(move || register(&s1, &data).id);
    let creating = std::thread::spawn(move || {
        let config: TrainConfig =
            serde_json::from_value(serde_json::json!({ "dataset_id": "ds_none" })).unwrap();
        s2.create_run(config).unwrap_err().body.code
    });
    std::thread::sleep(Duration::from_millis(200));
    assert!(!registering.is_finished() && !creating.is_finished());
    drop(lease);
    assert!(registering.join().unwrap().starts_with("ds_"));
    assert_eq!(creating.join().unwrap(), "dataset_not_found");
}
