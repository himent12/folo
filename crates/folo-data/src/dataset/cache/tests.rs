//! The scan cache: opens with and without it give the same dataset and see changes; file stamps and recently
//! modified files; the read guard; pruning old cache files; corrupt or foreign cache files; small folders; COCO.

use super::*;
use crate::dataset::{LoadedDataset, OpenOptions, Split, open_with};
use folo_test_support::TestDir;
use std::time::Duration;

fn settings(cache: &Path) -> Settings {
    Settings { dir: Some(cache.to_path_buf()), min_files: 1, list: true, now: None }
}

fn uncached() -> Settings {
    Settings { dir: None, min_files: usize::MAX, list: false, now: None }
}

/// Moves a file's modification time an hour back, so its stamp is settled and cacheable.
fn age(path: &Path) {
    let f = std::fs::File::options().write(true).open(path).unwrap();
    f.set_modified(SystemTime::now() - Duration::from_secs(3600)).unwrap();
}

fn png(path: &Path, w: u32, h: u32) {
    image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 30])).save(path).unwrap();
    age(path);
}

fn write(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
    age(path);
}

/// Everything an open returns, for comparisons.
fn view(d: &LoadedDataset) -> String {
    format!(
        "{:?} {:?} {:?} {:?} {:?} {:?} {:?}",
        d.format,
        d.classes,
        d.train.entries(),
        d.val.entries(),
        d.test.entries(),
        d.report,
        d.sources
    )
}

fn open(path: &Path, s: Settings) -> String {
    with_settings(s, || view(&open_with(path, &OpenOptions::default()).unwrap()))
}

/// A YOLO set: good images with labels, a background image, a corrupt image, a broken and a warned label.
fn yolo(root: &Path) {
    for split in ["train", "val"] {
        std::fs::create_dir_all(root.join("images").join(split)).unwrap();
        std::fs::create_dir_all(root.join("labels").join(split)).unwrap();
    }
    write(&root.join("data.yaml"), "path: .\ntrain: images/train\nval: images/val\nnames: [a, b]\n");
    for i in 0..4 {
        png(&root.join(format!("images/train/{i}.png")), 20 + i, 10);
        write(&root.join(format!("labels/train/{i}.txt")), &format!("{} 0.5 0.5 0.2 0.{}\n", i % 2, i + 1));
    }
    png(&root.join("images/train/background.png"), 8, 8);
    write(&root.join("images/train/corrupt.png"), "not a png");
    png(&root.join("images/train/broken.png"), 8, 8);
    write(&root.join("labels/train/broken.txt"), "7 0.5 0.5 0.2 0.2\n");
    png(&root.join("images/train/warned.png"), 8, 8);
    write(&root.join("labels/train/warned.txt"), "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n");
    png(&root.join("images/val/v.png"), 30, 12);
    write(&root.join("labels/val/v.txt"), "1 0.4 0.4 0.1 0.1\n");
}

#[test]
fn yolo_opens_the_same_with_and_without_the_cache_and_sees_changes() {
    let data = TestDir::new("scan-yolo");
    let cache = TestDir::new("scan-yolo-cache");
    yolo(&data);
    let plain = open(&data, uncached());
    assert_eq!(open(&data, settings(&cache)), plain, "cold cache");
    assert!(std::fs::read_dir(&*cache).unwrap().count() >= 2, "a cache file per folder");
    assert_eq!(open(&data, settings(&cache)), plain, "warm cache");

    // The cache is what answers: a label rewritten with its old length and modification time is not re-read.
    let label = data.join("labels/train/0.txt");
    let stamp = Stamp::of(&std::fs::metadata(&label).unwrap()).unwrap();
    let mtime = std::fs::metadata(&label).unwrap().modified().unwrap();
    std::fs::write(&label, "1 0.5 0.5 0.2 0.1\n").unwrap();
    std::fs::File::options().write(true).open(&label).unwrap().set_modified(mtime).unwrap();
    assert_eq!(Stamp::of(&std::fs::metadata(&label).unwrap()).unwrap(), stamp);
    assert_eq!(open(&data, settings(&cache)), plain, "same stamp: cached");
    // A new modification time (or length) is seen.
    age(&label);
    let changed = open(&data, uncached());
    assert_ne!(changed, plain);
    assert_eq!(open(&data, settings(&cache)), changed, "new stamp: re-read");

    // A fixed corrupt image and a new image are seen too.
    png(&data.join("images/train/corrupt.png"), 9, 9);
    png(&data.join("images/train/new.png"), 9, 9);
    std::fs::remove_file(data.join("images/train/1.png")).unwrap();
    let changed = open(&data, uncached());
    assert_eq!(open(&data, settings(&cache)), changed);
}

/// Sets `path`'s modification time to `t`.
fn set_mtime(path: &Path, t: SystemTime) {
    std::fs::File::options().write(true).open(path).unwrap().set_modified(t).unwrap();
}

/// A PNG of `w` x `h` padded with zeros after its end to `len` bytes (decoders stop at the end chunk), so two images
/// of different sizes can share a length.
fn padded_png(path: &Path, w: u32, h: u32, len: usize) {
    let mut bytes = Vec::new();
    image::RgbImage::from_pixel(w, h, image::Rgb([90, 120, 30]))
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    assert!(bytes.len() <= len, "{} bytes", bytes.len());
    bytes.resize(len, 0);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn image_probes_are_reused_only_while_the_stamp_and_class_count_match() {
    let data = TestDir::new("scan-probe");
    let cache = TestDir::new("scan-probe-cache");
    yolo(&data);
    let img = data.join("images/train/0.png");
    padded_png(&img, 20, 10, 4096);
    age(&img);
    let plain = open(&data, uncached());
    assert_eq!(open(&data, settings(&cache)), plain, "cold cache");

    // Another valid image of another size under the old length and modification time: the cached size answers.
    let mtime = std::fs::metadata(&img).unwrap().modified().unwrap();
    padded_png(&img, 33, 17, 4096);
    set_mtime(&img, mtime);
    assert_ne!(open(&data, uncached()), plain, "the new image has another size");
    assert_eq!(open(&data, settings(&cache)), plain, "same stamp: cached size");
    // Only the length changes (same modification time): probed again.
    padded_png(&img, 33, 17, 4097);
    set_mtime(&img, mtime);
    let changed = open(&data, uncached());
    assert_ne!(changed, plain);
    assert_eq!(open(&data, settings(&cache)), changed, "new length: re-probed");

    // Labels parsed against two classes are parsed again against eight (`7 ...` in broken.txt is then valid).
    write(
        &data.join("data.yaml"),
        "path: .\ntrain: images/train\nval: images/val\nnames: [a, b, c, d, e, f, g, h]\n",
    );
    let eight = open(&data, uncached());
    assert_ne!(eight, changed);
    assert_eq!(open(&data, settings(&cache)), eight, "another class count: re-parsed");
}

#[test]
fn files_modified_in_the_last_seconds_are_not_cached() {
    // The rule, on a clock the test sets: a stamp less than `SETTLE` old, or ahead of the clock, is not settled.
    let now = SystemTime::now();
    let stamp =
        |t: SystemTime| Stamp { len: 1, mtime: t.duration_since(UNIX_EPOCH).unwrap().as_nanos() as i64 };
    assert!(!stamp(now - Duration::from_millis(500)).settled(now));
    assert!(!stamp(now + Duration::from_secs(60)).settled(now));
    assert!(stamp(now - SETTLE).settled(now));

    // An open follows it, on the clock the test gives it (so how long the open takes does not matter).
    let data = TestDir::new("scan-recent");
    let cache = TestDir::new("scan-recent-cache");
    yolo(&data);
    let img = data.join("images/train/0.png");
    let recent = now - Duration::from_millis(500);
    let at = |now: SystemTime| Settings { now: Some(now), ..settings(&cache) };
    let rewrite = |w, h| {
        padded_png(&img, w, h, 4096);
        set_mtime(&img, recent);
    };
    rewrite(20, 10);
    open(&data, at(now));
    // Modified 500 ms before the clock: not cached, so a same-stamp rewrite is probed again.
    rewrite(33, 17);
    let big = open(&data, uncached());
    assert_eq!(open(&data, at(now)), big, "recent: probed again");
    // The same file once the clock is past the window: cached, so a same-stamp rewrite is answered from the cache.
    let later = now + SETTLE;
    assert_eq!(open(&data, at(later)), big);
    rewrite(20, 10);
    assert_ne!(open(&data, uncached()), big, "the rewrite has another size");
    assert_eq!(open(&data, at(later)), big, "settled: cached");
}

#[test]
fn nothing_is_cached_or_listed_with_a_read_guard() {
    // The guard is process-wide and cannot be uninstalled, so this checks the settings an installed guard gives.
    let data = TestDir::new("scan-guard");
    let cache = TestDir::new("scan-guard-cache");
    yolo(&data);
    let guarded = Settings::new(Some(cache.to_path_buf()), true);
    assert!(guarded.dir.is_none() && !guarded.list, "{guarded:?}");
    assert!(Settings::new(Some(cache.to_path_buf()), false).dir.is_some());
    assert_eq!(open(&data, Settings { min_files: 1, ..guarded }), open(&data, uncached()));
    assert_eq!(std::fs::read_dir(&*cache).unwrap().count(), 0);
}

#[test]
fn prune_deletes_only_old_cache_files() {
    let cache = TestDir::new("scan-prune");
    let names = [
        "dir-0123456789abcdef.bin",
        "coco-0123456789abcdef.bin",
        ".dir-0123456789abcdef.bin.42-7.tmp",
        "notes.bin",
        "coco-0123456789abcdeg.bin",
        "dir-0123.bin",
        "other-0123456789abcdef.bin",
        "dir-0123456789abcdef.bin.tmp",
        ".notes.bin.42-7.tmp",
    ];
    for n in names {
        let p = cache.join(n);
        std::fs::write(&p, "x").unwrap();
        set_mtime(&p, SystemTime::now() - MAX_AGE - Duration::from_secs(60));
    }
    std::fs::write(cache.join("dir-fedcba9876543210.bin"), "new").unwrap();
    prune(&cache, &cache.join("dir-fedcba9876543210.bin"));
    let mut left: Vec<String> =
        std::fs::read_dir(&*cache).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    left.sort();
    let mut want: Vec<&str> = names[3..].to_vec();
    want.push("dir-fedcba9876543210.bin");
    want.sort();
    assert_eq!(left, want);
}

#[test]
fn corrupt_or_foreign_cache_files_fall_back_to_a_normal_open() {
    for (name, make) in [("yolo", yolo as fn(&Path)), ("coco", coco)] {
        corrupt_files_fall_back(name, make);
    }
}

/// Every cache file of the dataset `make` writes, truncated or with a flipped byte: the open is still exact.
fn corrupt_files_fall_back(name: &str, make: fn(&Path)) {
    let data = TestDir::new(&format!("scan-corrupt-{name}"));
    let cache = TestDir::new(&format!("scan-corrupt-{name}-cache"));
    make(&data);
    let plain = open(&data, uncached());
    open(&data, settings(&cache));
    let files: Vec<PathBuf> = std::fs::read_dir(&*cache).unwrap().map(|e| e.unwrap().path()).collect();
    assert!(!files.is_empty());
    if name == "coco" {
        assert!(
            files.iter().any(|f| f.file_name().unwrap().to_str().unwrap().starts_with("coco-")),
            "{files:?}"
        );
    }
    for f in &files {
        let good = std::fs::read(f).unwrap();
        // Every truncation and a flipped byte anywhere: the open is still exact.
        for cut in [7, good.len() / 2, good.len() - 1] {
            std::fs::write(f, &good[..cut]).unwrap();
            assert_eq!(open(&data, settings(&cache)), plain, "{} cut to {cut}", f.display());
        }
        for i in [9, good.len() / 3, good.len() - 2] {
            let mut bad = good.clone();
            bad[i] ^= 0x40;
            std::fs::write(f, &bad).unwrap();
            assert_eq!(open(&data, settings(&cache)), plain, "{} byte {i} flipped", f.display());
        }
    }
}

#[test]
fn small_folders_are_not_cached() {
    let data = TestDir::new("scan-small");
    let cache = TestDir::new("scan-small-cache");
    yolo(&data);
    let s = Settings { min_files: 1000, ..settings(&cache) };
    assert_eq!(open(&data, s), open(&data, uncached()));
    assert_eq!(std::fs::read_dir(&*cache).map(|r| r.count()).unwrap_or(0), 0);
}

fn coco(root: &Path) {
    std::fs::create_dir_all(root.join("annotations")).unwrap();
    std::fs::create_dir_all(root.join("train2017")).unwrap();
    std::fs::create_dir_all(root.join("val2017")).unwrap();
    for (dir, n) in [("train2017", 5), ("val2017", 2)] {
        for i in 0..n {
            png(&root.join(dir).join(format!("{i}.png")), 40, 30);
        }
    }
    write(&root.join("train2017/bad.png"), "x");
    let train = r#"{"images": [
        {"id": 1, "file_name": "0.png", "width": 40, "height": 30},
        {"id": 2, "file_name": "1.png", "width": 41, "height": 30},
        {"id": 3, "file_name": "2.png"},
        {"id": 3, "file_name": "3.png"},
        {"id": 4, "file_name": "4.png", "width": 40, "height": 30},
        {"id": 5, "file_name": "bad.png", "width": 4, "height": 3},
        {"id": 6, "file_name": "gone.png", "width": 4, "height": 3}],
      "annotations": [
        {"image_id": 1, "category_id": 1, "bbox": [1.25, 2, 10.1, 5]},
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 39, 29]},
        {"image_id": 2, "category_id": 9, "bbox": [0, 0, 1, 1]},
        {"image_id": 3, "category_id": 1, "bbox": [0, 0, 0, 1]},
        {"image_id": 4, "category_id": 1, "bbox": "x"},
        {"image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "iscrowd": 1}],
      "categories": [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}]}"#;
    write(&root.join("annotations/instances_train2017.json"), train);
    let val = r#"{"images": [{"id": 1, "file_name": "0.png", "width": 40, "height": 30},
        {"id": 2, "file_name": "1.png", "width": 40, "height": 30}],
      "annotations": [{"image_id": 2, "category_id": 3, "bbox": [3, 4, 5, 6]}],
      "categories": [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}, {"id": 4, "name": "eel"}]}"#;
    write(&root.join("annotations/instances_val2017.json"), val);
}

#[test]
fn coco_opens_the_same_with_and_without_the_cache_and_sees_changes() {
    let data = TestDir::new("scan-coco");
    let cache = TestDir::new("scan-coco-cache");
    coco(&data);
    let plain = open(&data, uncached());
    assert_eq!(open(&data, settings(&cache)), plain, "cold cache");
    let names: Vec<String> =
        std::fs::read_dir(&*cache).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    assert!(names.iter().any(|n| n.starts_with("coco-")), "{names:?}");
    assert_eq!(open(&data, settings(&cache)), plain, "warm cache");
    // One split only: its own cache entry, exactly what an uncached open of that split gives.
    let val_only = |s| {
        with_settings(s, || {
            view(&crate::dataset::open_splits(&*data, &OpenOptions::default(), &[Split::Val]).unwrap())
        })
    };
    assert_eq!(val_only(settings(&cache)), val_only(uncached()));
    assert_eq!(val_only(settings(&cache)), val_only(uncached()));

    // The cache answers while the JSON keeps its length and modification time...
    let json = data.join("annotations/instances_train2017.json");
    let mtime = std::fs::metadata(&json).unwrap().modified().unwrap();
    let text = std::fs::read_to_string(&json).unwrap().replace("[1.25, 2, 10.1, 5]", "[2.25, 2, 10.1, 5]");
    std::fs::write(&json, &text).unwrap();
    std::fs::File::options().write(true).open(&json).unwrap().set_modified(mtime).unwrap();
    assert_eq!(open(&data, settings(&cache)), plain, "same stamp: cached");
    // ...and a changed JSON is read again.
    age(&json);
    let changed = open(&data, uncached());
    assert_ne!(changed, plain);
    assert_eq!(open(&data, settings(&cache)), changed);
}

#[test]
fn annotations_round_trip_exactly() {
    let data = TestDir::new("scan-coco-codec");
    coco(&data);
    let jsons = crate::dataset::coco::find_jsons(&data);
    let read = || crate::dataset::coco::read_annotations(&data, &jsons, &Split::ALL).unwrap();
    let a = read();
    let stamps = [Stamp { len: 1, mtime: -5 }, Stamp { len: 2, mtime: 7 }];
    let bytes = coco::encode(&a, &stamps).unwrap();
    assert_eq!(coco::decode(&bytes, &stamps), Some(a));
    assert_eq!(coco::decode(&bytes, &stamps[..1]), None);
    assert_eq!(coco::decode(&bytes, &[stamps[0], Stamp { len: 2, mtime: 8 }]), None);
    for cut in 0..bytes.len() {
        assert_eq!(coco::decode(&bytes[..cut], &stamps), None, "cut to {cut}");
    }
}
