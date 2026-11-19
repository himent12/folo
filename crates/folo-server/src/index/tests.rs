use super::*;
use folo_test_support::TestDir;

fn tmp(tag: &str) -> TestDir {
    TestDir::new(&format!("index-{tag}"))
}

/// Store writes (run under locks) wait briefly for a held file: checked on the policy, not timed.
#[test]
fn store_writes_wait_briefly_for_a_held_file() {
    assert_eq!(STORE_RETRY, folo_nn::io::RenameRetry::BRIEF);
    assert!(STORE_RETRY.budget <= std::time::Duration::from_millis(250), "{STORE_RETRY:?}");
}

/// A store write over a file held open (as Python's `open()` does) gives up, with a watchdog far above any
/// budget, leaving the old content and no temporary file.
#[cfg(windows)]
#[test]
fn a_store_write_gives_up_on_a_file_another_program_holds() {
    use std::os::windows::fs::OpenOptionsExt;
    let dir = tmp("held");
    let path = dir.join("run.json");
    write_atomic(&path, b"old").unwrap();
    let held = std::fs::OpenOptions::new().read(true).share_mode(1 | 2).open(&path).unwrap();
    let (done, result) = std::sync::mpsc::channel();
    let p = path.clone();
    let _ = std::thread::spawn(move || done.send(write_atomic(&p, b"new").is_err()));
    let failed = result.recv_timeout(std::time::Duration::from_secs(60)).expect("gave up within 60 s");
    assert!(failed, "the write over a held file succeeded");
    drop(held);
    assert_eq!(std::fs::read(&path).unwrap(), b"old");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "no temporary file stays");
}

#[test]
fn default_names_come_from_the_folder_and_are_numbered_when_taken() {
    let dir = tmp("names");
    let bccd = dir.join("bccd");
    std::fs::create_dir_all(bccd.join("annotations")).unwrap();
    for f in ["annotations/instances_test.json", "data.yaml", "loose.json"] {
        std::fs::write(bccd.join(f), "{}").unwrap();
    }
    let free = |_: &str| false;
    assert_eq!(default_name(&bccd, free), "bccd");
    assert_eq!(default_name(&bccd.join("annotations/instances_test.json"), free), "bccd");
    assert_eq!(default_name(&bccd.join("data.yaml"), free), "bccd");
    assert_eq!(default_name(&bccd.join("loose.json"), free), "loose");
    let taken = |n: &str| ["bccd", "bccd-2"].contains(&n.to_lowercase().as_str());
    assert_eq!(default_name(&bccd, taken), "bccd-3");
}

/// The diagnostics of a folder that is not a dataset stop after their budget of entries, whatever they find
/// (x3-server-1: they used to count up to 10,000 *images*, so a wide tree without images was walked whole).
#[test]
fn diagnostics_of_a_non_dataset_read_a_bounded_number_of_entries() {
    let dir = tmp("diagnose");
    for i in 0..40 {
        std::fs::create_dir_all(dir.join(format!("sub{i:02}"))).unwrap();
        std::fs::write(dir.join(format!("{i:02}.jpg")), b"x").unwrap();
    }
    std::fs::create_dir_all(dir.join("sub39/set/images")).unwrap();
    std::fs::write(dir.join("sub39/set/data.yaml"), "names: [x]\n").unwrap();
    assert_eq!(images_under(&dir, 1000), 40);
    // The folder itself is the first entry walked: at most 9 of the 40 images are seen.
    assert!(images_under(&dir, 10) < 10, "the walk stops after 10 entries");
    assert!(dataset_below(&dir, 1000).is_some());
    assert_eq!(dataset_below(&dir, 10), None, "the folder search stops after 10 entries");
}

/// Looking for a dataset in a folder below counts what it reads there too: a folder of many files before the
/// dataset uses the budget up, where `detect_format` read it whole (review round 2).
#[test]
fn looking_inside_a_folder_below_counts_against_the_budget() {
    let dir = tmp("diagnose-inside");
    std::fs::create_dir_all(dir.join("a-junk")).unwrap();
    for i in 0..3000 {
        std::fs::write(dir.join(format!("a-junk/{i:04}.txt")), b"").unwrap();
    }
    let set = dir.join("b-set");
    std::fs::create_dir_all(set.join("images")).unwrap();
    std::fs::write(set.join("coco128.yaml"), "path: .\ntrain: images\nnames: [x]\n").unwrap();
    assert_eq!(dataset_below(&dir, DIAGNOSE_ENTRIES), Some(set.clone()), "a YAML with a train key");
    assert_eq!(dataset_below(&dir, 1000), None, "the 3000 files of a-junk use 1000 entries up");
    let mut budget = 1000;
    assert!(!looks_like_dataset(&dir.join("a-junk"), &mut budget));
    assert_eq!(budget, 0, "it stopped at the budget");
    // What detection finds without reading the folder, and COCO.
    std::fs::remove_file(set.join("coco128.yaml")).unwrap();
    std::fs::create_dir_all(set.join("labels")).unwrap();
    assert!(looks_like_dataset(&set, &mut 100), "images/ + labels/");
    let coco = dir.join("c-coco/annotations");
    std::fs::create_dir_all(&coco).unwrap();
    std::fs::write(coco.join("captions_val.json"), "{}").unwrap();
    assert!(!looks_like_dataset(coco.parent().unwrap(), &mut 100), "captions only");
    std::fs::write(coco.join("instances_val.json"), "{}").unwrap();
    assert!(looks_like_dataset(coco.parent().unwrap(), &mut 100));
}

/// A FIFO named like a YAML, with no writer, is not opened by the probe of a folder below, and opening one
/// (swapped in after the check) does not wait for a writer: registering its parent answers (review round 3).
#[cfg(unix)]
#[test]
fn a_fifo_named_like_a_yaml_never_holds_the_diagnostics_up() {
    let dir = tmp("diagnose-fifo");
    std::fs::create_dir_all(dir.join("child")).unwrap();
    let fifo = dir.join("child/probe.yaml");
    let made = std::process::Command::new("mkfifo").arg(&fifo).status().expect("mkfifo runs");
    assert!(made.success(), "mkfifo {}", fifo.display());
    let (tx, rx) = std::sync::mpsc::channel();
    let (root, pipe) = (dir.to_path_buf(), fifo.clone());
    std::thread::spawn(move || {
        let below = dataset_below(&root, DIAGNOSE_ENTRIES);
        let opened = open_without_waiting(&pipe).is_ok();
        let _ = tx.send((below, yaml_has_train(&pipe), opened));
    });
    let (below, train, opened) =
        rx.recv_timeout(std::time::Duration::from_secs(30)).expect("the FIFO held the probe up");
    assert_eq!((below, train), (None, false));
    // Where O_NONBLOCK is not known nothing is opened, so nothing waits either (review round 4).
    assert_eq!(opened, O_NONBLOCK.is_some(), "opened without waiting for a writer");
}

#[test]
fn requested_paths_lose_quotes_and_must_be_absolute() {
    let dir = tmp("requested");
    let p = dir.to_str().unwrap();
    for raw in [
        p.to_string(),
        format!("\"{p}\""),
        format!("  '{p}' "),
        format!(
            "\" {p}\"
"
        ),
    ] {
        assert_eq!(requested_path(&raw).unwrap(), p, "{raw:?}");
    }
    for raw in ["bccd", "\"data/x\"", "  ", "\"\""] {
        let e = requested_path(raw).unwrap_err();
        assert_eq!((e.status, e.body.code.as_str()), (StatusCode::BAD_REQUEST, "invalid_body"), "{raw:?}");
    }
    assert_eq!(requested_path("bccd").unwrap_err().body.message, "'bccd' is not a full path");
}

#[test]
fn build_list_and_fingerprint() {
    let dir = tmp("build");
    for s in ["train", "val"] {
        std::fs::create_dir_all(dir.join("images").join(s)).unwrap();
        std::fs::create_dir_all(dir.join("labels").join(s)).unwrap();
    }
    std::fs::write(dir.join("data.yaml"), "train: images/train\nval: images/val\nnames: [cat, dog]\n")
        .unwrap();
    let img = image::RgbImage::from_pixel(40, 20, image::Rgb([200, 10, 10]));
    img.save(dir.join("images/train/a.png")).unwrap();
    img.save(dir.join("images/val/b.png")).unwrap();
    std::fs::write(dir.join("labels/train/a.txt"), "1 0.5 0.5 0.5 0.5\n0 0.995 0.5 0.02 0.2\n").unwrap();
    let idx = Index::build(dir.to_str().unwrap(), DatasetFormat::Auto, &[]).unwrap();
    assert_eq!(idx.classes, vec!["cat", "dog"]);
    assert_eq!(idx.splits, Splits { train: 1, val: 1, test: 0 });
    assert_eq!(idx.stats.class_counts, vec![1, 1]);
    assert_eq!(idx.stats.empty_images, 1);
    let page = idx.list_images(&ImageQuery { limit: 10, ..Default::default() });
    assert_eq!(page.total, 2);
    assert_eq!(page.images[0].file, "images/train/a.png");
    assert_eq!((page.images[0].width, page.images[0].boxes[0].x1), (40, 10.0));
    // A box poking out of the right edge is clipped, never inverted.
    let edge = page.images[0].boxes[1];
    assert!(edge.x1 < edge.x2 && edge.x2 == 40.0, "{edge:?}");
    let dog = idx.list_images(&ImageQuery { limit: 10, class: Some(1), ..Default::default() });
    assert_eq!(dog.total, 1);
    assert!(idx.is_fresh());
    assert!(idx.at("01").is_err() && idx.at("2").is_err());

    let cache = dir.join("cache");
    let path = idx.image_path("0").unwrap();
    let th = read_image(&path, Some(10), Some(&cache)).unwrap();
    let decoded = image::load_from_memory(&th.bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (10, 5));
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1, "thumbnail cached");
    assert_eq!(read_image(&path, Some(10), Some(&cache)).unwrap().bytes, th.bytes);
    assert!(!read_image(&path, Some(4096), None).unwrap().bytes.is_empty());

    std::fs::write(dir.join("labels/train/a.txt"), "1 0.5 0.5 0.5 0.5\n").unwrap();
    assert!(!idx.is_fresh(), "editing a label invalidates the index");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn errors_map_to_api_codes() {
    let dir = tmp("errors");
    let code = |p: &Path| Index::build(p.to_str().unwrap(), DatasetFormat::Auto, &[]).unwrap_err().body.code;
    assert_eq!(code(&dir.join("nope")), "path_not_found");
    assert_eq!(code(&dir), "unknown_format");
    std::fs::write(dir.join("data.yaml"), "train: images\nnc: 2\nnames: [a]\n").unwrap();
    assert_eq!(code(&dir), "invalid_dataset");
    std::fs::remove_dir_all(&dir).unwrap();
}
