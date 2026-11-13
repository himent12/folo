//! Dataset ids across unregistering (tombstones) and deleting a dataset's files while something reads them.

use std::path::Path;
use std::sync::{Arc, Barrier};
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

/// studio-data F7: two sample sets generated with one name, also at once (two tabs), are told apart by their names
/// as by their folders; a registration's own name is kept as asked.
#[test]
fn sample_sets_of_one_name_are_numbered_also_when_generated_at_once() {
    let t = TestDir::new("real-synthetic-names");
    let store = Arc::new(RealStore::open(t.join("home")));
    let go = Arc::new(Barrier::new(2));
    let both: Vec<Dataset> = (0..2)
        .map(|_| {
            let (store, go) = (store.clone(), go.clone());
            std::thread::spawn(move || {
                go.wait();
                synthetic(&store, "Big Set")
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let mut names: Vec<&str> = both.iter().map(|d| d.name.as_str()).collect();
    names.sort();
    assert_eq!(names, ["Big Set", "Big Set-2"]);
    // A third later, whatever the case it is asked in.
    assert_eq!(synthetic(&store, "big set").name, "big set-3");
    // A folder registered under a name a dataset has keeps that name (Studio warns before it asks).
    let second = both.iter().find(|d| d.name == "Big Set-2").unwrap();
    store.delete_dataset(&second.id).unwrap();
    let req = RegisterDataset {
        name: Some("Big Set".into()),
        path: second.path.clone(),
        format: DatasetFormat::Auto,
    };
    assert_eq!(store.register_dataset(req).unwrap().name, "Big Set");
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

/// The reviewer's second case: a dataset whose description pointed elsewhere when it was indexed, then was changed
/// to point into the folder, reads it now (its cached index still says otherwise); and any active run, on whatever
/// dataset, keeps every folder.
#[test]
fn files_are_judged_by_descriptions_as_they_read_now() {
    let t = TestDir::new("real-delete-changed");
    let store = RealStore::open(t.join("home"));
    let a = synthetic(&store, "a");
    let yaml = std::fs::read_to_string(Path::new(&a.path).join("data.yaml")).unwrap();
    let c = t.join("c");
    let cfg = folo_data::synthetic::SyntheticConfig { images: 4, imgsz: 64, seed: 3, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, &c, folo_data::ImageFormat::Png).unwrap();
    let b = t.join("b");
    std::fs::create_dir_all(&b).unwrap();
    let points_to = |p: &Path| {
        std::fs::write(b.join("data.yaml"), format!("path: '{}'\n{yaml}", p.display())).unwrap();
    };
    points_to(&c);
    let on_c = register(&store, &b.join("data.yaml"));
    points_to(Path::new(&a.path));
    let e = store.delete_dataset_files(&a.id).unwrap_err();
    assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "dataset_in_use"), "{}", e.body.message);
    assert!(e.body.message.contains(&on_c.name), "{}", e.body.message);
    points_to(&c);

    // A queued run on another dataset altogether.
    let run = run_on(&store, &on_c.id, "elsewhere");
    let e = store.delete_dataset_files(&a.id).unwrap_err();
    assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "run_active"), "{}", e.body.message);
    assert!(Path::new(&a.path).join("images").is_dir() && store.entry(&a.id).is_ok());
    store.runs.stop(&run).unwrap();
    store.delete_dataset_files(&a.id).unwrap();
    assert!(!Path::new(&a.path).exists());
}

/// Review R1: dataset B's `images` is a junction (a directory symlink off Windows) into generated A's, so B reads A's
/// images through it, and A's files are kept.
#[test]
fn files_are_kept_while_a_link_in_another_dataset_leads_into_them() {
    let t = TestDir::new("real-delete-link");
    let store = RealStore::open(t.join("home"));
    let a = synthetic(&store, "a");
    let folder = Path::new(&a.path).to_path_buf();
    let b = t.join("b");
    std::fs::create_dir_all(b.join("labels")).unwrap();
    std::fs::copy(folder.join("data.yaml"), b.join("data.yaml")).unwrap();
    for split in std::fs::read_dir(folder.join("labels")).unwrap() {
        let split = split.unwrap().path();
        let to = b.join("labels").join(split.file_name().unwrap());
        std::fs::create_dir_all(&to).unwrap();
        for f in std::fs::read_dir(&split).unwrap() {
            let f = f.unwrap().path();
            std::fs::copy(&f, to.join(f.file_name().unwrap())).unwrap();
        }
    }
    if let Err(why) = dir_link(&folder.join("images"), &b.join("images")) {
        eprintln!("skipped: could not link a folder ({why})");
        return;
    }
    let on_a = register(&store, &b.join("data.yaml"));
    assert!(on_a.splits.train > 0, "B reads A's images through its link");
    let e = store.delete_dataset_files(&a.id).unwrap_err();
    assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "dataset_in_use"), "{}", e.body.message);
    assert!(e.body.message.contains(&on_a.name), "{}", e.body.message);
    assert!(folder.join("images").is_dir() && store.entry(&a.id).is_ok());
}

/// A link at `link` to the folder `target`: a junction on Windows (`mklink /J`, which needs no privilege), a
/// symlink elsewhere.
fn dir_link(target: &Path, link: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        let win = |p: &Path| folo_data::paths::shown(p).replace('/', "\\");
        let out = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J", &win(link), &win(target)])
            .output()
            .map_err(|e| e.to_string())?;
        out.status
            .success()
            .then_some(())
            .ok_or_else(|| String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
    #[cfg(not(windows))]
    std::os::unix::fs::symlink(target, link).map_err(|e| e.to_string())
}

/// Review R2: another dataset whose description cannot be read now (it no longer parses; on Windows, also held open
/// by another program) might read the folder, so the files are kept and nothing is unregistered. Unregistering that
/// dataset still works, and then the files go.
#[test]
fn files_are_kept_while_another_dataset_cannot_be_checked() {
    let t = TestDir::new("real-delete-unchecked");
    let store = RealStore::open(t.join("home"));
    let a = synthetic(&store, "a");
    let c = t.join("c");
    let cfg = folo_data::synthetic::SyntheticConfig { images: 4, imgsz: 64, seed: 4, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, &c, folo_data::ImageFormat::Png).unwrap();
    let yaml = c.join("data.yaml");
    let other = register(&store, &yaml);
    let refused = || {
        let e = store.delete_dataset_files(&a.id).unwrap_err();
        assert_eq!((e.status.as_u16(), e.body.code.as_str()), (409, "dataset_in_use"), "{}", e.body.message);
        assert!(e.body.message.contains(&other.name), "{}", e.body.message);
        assert!(e.body.message.contains("could not be checked"), "{}", e.body.message);
        assert!(Path::new(&a.path).join("images").is_dir() && store.entry(&a.id).is_ok());
    };
    #[cfg(windows)]
    let good = std::fs::read_to_string(&yaml).unwrap();
    std::fs::write(&yaml, "train: [images/train\n").unwrap();
    refused();
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        std::fs::write(&yaml, &good).unwrap();
        let held = std::fs::OpenOptions::new().read(true).share_mode(0).open(&yaml).unwrap();
        refused();
        drop(held);
    }
    store.delete_dataset(&other.id).unwrap();
    store.delete_dataset_files(&a.id).unwrap();
    assert!(!Path::new(&a.path).exists());
}

/// A dataset that is not there at all reads nothing: one deleted with its files whose run keeps its id, and a
/// registered one whose folder was moved away, do not keep the next generated set's files for good.
#[test]
fn a_dataset_that_is_gone_does_not_keep_other_files() {
    let t = TestDir::new("real-delete-gone");
    let store = RealStore::open(t.join("home"));
    let a = synthetic(&store, "a");
    let run = run_on(&store, &a.id, "on-a");
    store.runs.stop(&run).unwrap();
    store.delete_dataset_files(&a.id).unwrap();
    assert!(!Path::new(&a.path).exists());
    let registry = super::registry::read_registry(&store.data_dir).unwrap();
    assert_eq!(registry.tombstones.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), [a.id.as_str()]);

    let c = t.join("c");
    let cfg = folo_data::synthetic::SyntheticConfig { images: 4, imgsz: 64, seed: 4, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, &c, folo_data::ImageFormat::Png).unwrap();
    let moved = register(&store, &c);
    std::fs::rename(&c, t.join("c-moved")).unwrap();

    let b = synthetic(&store, "b");
    store.delete_dataset_files(&b.id).unwrap();
    assert!(!Path::new(&b.path).exists());
    assert!(store.entry(&moved.id).is_ok(), "the moved dataset stays registered");
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

/// The reviewer's R2 interleaving: a run admitted (under the shared lease, as `start_run` holds it) while the dataset
/// is unregistered. The unregister waits for the admission, so the run is in the registry when the tombstones are
/// pruned, and the dataset registered again gets its id back.
#[test]
fn unregistering_waits_for_a_run_being_admitted() {
    let t = TestDir::new("real-unregister-admission");
    let store = Arc::new(RealStore::open(t.join("home")));
    let a = synthetic(&store, "a");
    let admitting = store.files.read().unwrap();
    let e = store.entry(&a.id).unwrap();
    let started = Arc::new(Barrier::new(2));
    let (s, b, id) = (store.clone(), started.clone(), a.id.clone());
    let unregistering = std::thread::spawn(move || {
        b.wait();
        s.delete_dataset(&id).unwrap();
    });
    started.wait();
    std::thread::sleep(Duration::from_millis(200));
    assert!(!unregistering.is_finished(), "unregistering did not wait for the admission");
    let config: TrainConfig =
        serde_json::from_value(serde_json::json!({ "dataset_id": a.id, "name": "admitted" })).unwrap();
    store.runs.register(config, store.dataset_ref(&e), None, None, 0, "cpu".into(), None).unwrap();
    drop(admitting);
    unregistering.join().unwrap();
    assert!(store.entry(&a.id).is_err());
    assert_eq!(register(&store, Path::new(&a.path)).id, a.id);
}

/// `datasets.json` stays the plain list every version reads (an older build sees the datasets this one wrote), the
/// tombstones live in `datasets.removed.json` (none when it is missing), and the object form this version's
/// pre-release builds wrote is read, then written as a list at the next write.
#[test]
fn the_registry_stays_the_list_older_builds_read() {
    let t = TestDir::new("real-registry-format");
    let home = t.join("home");
    let data = t.join("shapes");
    let cfg = folo_data::synthetic::SyntheticConfig { images: 4, imgsz: 64, seed: 9, ..Default::default() };
    folo_data::synthetic::write_yolo(&cfg, &data, folo_data::ImageFormat::Png).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    let old = serde_json::json!([{
        "id": "ds_main", "name": "shapes", "path": data, "format": "yolo", "created": "2026-11-01T00:00:00Z"
    }]);
    let (list_path, removed_path) = (home.join("datasets.json"), home.join("datasets.removed.json"));
    std::fs::write(&list_path, old.to_string()).unwrap();
    let ids =
        |store: &RealStore| store.list_datasets().unwrap().into_iter().map(|d| d.id).collect::<Vec<_>>();
    // An older build's data dir (no tombstone file) opens with its dataset.
    let store = RealStore::open(home.clone());
    assert_eq!(ids(&store), ["ds_main"]);
    // This build's writes: still a list, read as an older build reads it; the tombstone in its own file.
    let b = synthetic(&store, "b");
    run_on(&store, &b.id, "on-b");
    store.delete_dataset(&b.id).unwrap();
    let as_older: Vec<super::Registration> =
        serde_json::from_str(&std::fs::read_to_string(&list_path).unwrap()).unwrap();
    assert_eq!(as_older.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["ds_main"]);
    let removed: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&removed_path).unwrap()).unwrap();
    assert_eq!(removed.iter().map(|t| t["id"].as_str().unwrap()).collect::<Vec<_>>(), [b.id.as_str()]);
    drop(store);

    // The object form, its tombstone inside: read, the id kept for its dataset, and rewritten as a list.
    std::fs::remove_file(&removed_path).unwrap();
    let object = serde_json::json!({ "datasets": old, "tombstones": removed });
    std::fs::write(&list_path, object.to_string()).unwrap();
    let store = RealStore::open(home.clone());
    assert_eq!(ids(&store), ["ds_main"]);
    assert_eq!(register(&store, Path::new(&b.path)).id, b.id);
    let as_older: Vec<super::Registration> =
        serde_json::from_str(&std::fs::read_to_string(&list_path).unwrap()).unwrap();
    assert_eq!(as_older.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["ds_main", b.id.as_str()]);
    assert_eq!(std::fs::read_to_string(&removed_path).unwrap().trim(), "[]");
}

/// Review R3: the object form's tombstones have their only copy in the `datasets.json` the list replaces, so its
/// first write back saves them to `datasets.removed.json` before the list, also for a write that gains a
/// registration: an unrelated registration keeps the tombstone a run references, also when the list write fails.
#[test]
fn the_object_registrys_tombstones_are_saved_before_its_list() {
    use super::registry::{Gains, read_registry, write_with};
    let t = TestDir::new("real-registry-object");
    let home = t.join("home");
    let store = RealStore::open(home.clone());
    let b = synthetic(&store, "b");
    run_on(&store, &b.id, "on-b");
    store.delete_dataset(&b.id).unwrap();
    drop(store);
    let (list_path, removed_path) = (home.join("datasets.json"), home.join("datasets.removed.json"));
    let removed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&removed_path).unwrap()).unwrap();
    let object = serde_json::json!({ "datasets": [], "tombstones": removed }).to_string();
    let write_object = || {
        std::fs::write(&list_path, &object).unwrap();
        let _ = std::fs::remove_file(&removed_path);
    };
    let kept = || {
        let text = std::fs::read_to_string(&removed_path).unwrap_or_default();
        let removed: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap_or_default();
        removed.iter().map(|t| t["id"].as_str().unwrap().to_string()).collect::<Vec<_>>()
    };

    // The list write fails: the tombstone is in its own file already, the object unchanged.
    write_object();
    let mut file = read_registry(&home).unwrap();
    let c = serde_json::json!({
        "id": "ds_c", "name": "c", "path": t.join("c"), "format": "yolo", "created": "2026-11-11T00:00:00Z"
    });
    file.datasets.push(serde_json::from_value(c).unwrap());
    let mut write = |p: &Path, bytes: &[u8]| match p == list_path {
        true => Err(std::io::Error::other("refused")),
        false => crate::index::write_atomic(p, bytes),
    };
    write_with(&home, &file, Gains::Registration, &mut write).unwrap_err();
    assert_eq!(kept(), [b.id.as_str()]);
    assert_eq!(std::fs::read_to_string(&list_path).unwrap(), object);

    // An unrelated registration through the store: the list written, the tombstone kept, its id still B's.
    write_object();
    let store = RealStore::open(home.clone());
    let c = synthetic(&store, "c");
    assert_eq!(kept(), [b.id.as_str()]);
    let list: Vec<super::Registration> =
        serde_json::from_str(&std::fs::read_to_string(&list_path).unwrap()).unwrap();
    assert_eq!(list.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), [c.id.as_str()]);
    assert_eq!(register(&store, Path::new(&b.path)).id, b.id);
}
