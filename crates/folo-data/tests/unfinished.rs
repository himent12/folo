//! A dataset folder being written stays recognisably unfinished until complete (issue #316): readers refuse a
//! marked folder however they reach it, and a writer never writes into, clears or deletes one.

mod common;

use common::{TempDir, write};
use folo_data::check::{CheckOptions, check};
use folo_data::synthetic::{SyntheticConfig, write_yolo, write_yolo_with_progress};
use folo_data::unfinished::MARKER;
use folo_data::{DataError, Format, ImageFormat, detect_format, open};
use std::path::Path;

fn small(images: usize) -> SyntheticConfig {
    SyntheticConfig { images, imgsz: 64, seed: 3, ..Default::default() }
}

/// Files under `dir`, relative, sorted.
fn files(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = walkdir::WalkDir::new(dir)
        .into_iter()
        .map(Result::unwrap)
        .filter(|e| !e.file_type().is_dir())
        .map(|e| e.path().strip_prefix(dir).unwrap().to_string_lossy().replace('\\', "/"))
        .collect();
    out.sort();
    out
}

fn assert_refused(err: DataError) {
    assert!(matches!(&err, DataError::Unfinished(_)), "{err}");
    assert!(err.to_string().contains(MARKER), "{err}");
}

/// `open`, `detect_format` and `check` all refuse `path`.
fn assert_unfinished(path: &Path) {
    assert_refused(open(path).unwrap_err());
    assert_refused(detect_format(path, Format::Auto).unwrap_err());
    assert_refused(check(path, &CheckOptions::default()).unwrap_err());
}

fn is_already_exists(err: &DataError) -> bool {
    matches!(err, DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::AlreadyExists)
}

/// Stopped part way through the images (here: a panic from the progress callback), the folder keeps the marker and
/// every reader refuses it; a write into it again is refused and deletes nothing. A finished write leaves no marker.
#[test]
fn a_write_stopped_part_way_is_refused_and_kept() {
    let t = TempDir::new("unfinished-midway");
    let dir = t.join("shapes");
    let stop = std::panic::catch_unwind(|| {
        write_yolo_with_progress(&small(40), &dir, ImageFormat::Png, |done| assert!(done < 20, "stopped"))
    });
    assert!(stop.is_err());
    let left = files(&dir);
    assert!(left.contains(&MARKER.to_string()) && left.len() > 20);
    assert_unfinished(&dir);

    let err = write_yolo(&small(6), &dir, ImageFormat::Png).unwrap_err();
    assert!(is_already_exists(&err), "{err}");
    assert_eq!(files(&dir), left);

    let done = t.join("done");
    write_yolo(&small(6), &done, ImageFormat::Png).unwrap();
    assert!(!done.join(MARKER).exists());
    assert_eq!(open(&done).unwrap().split_sizes(), [5, 1, 0]);
}

/// A second writer (the marker already there, as a running writer leaves it) is refused and changes nothing:
/// `write_yolo` and `convert`, into an otherwise empty folder.
#[test]
fn a_second_writer_is_refused_and_changes_nothing() {
    let t = TempDir::new("unfinished-second");
    write_yolo(&small(6), &t.join("src"), ImageFormat::Png).unwrap();
    let ds = open(t.join("src")).unwrap();
    let dir = t.join("busy");
    write(&dir.join(MARKER), "theirs");
    let err = write_yolo(&small(6), &dir, ImageFormat::Png).unwrap_err();
    assert!(is_already_exists(&err), "{err}");
    for to in [Format::Yolo, Format::Coco] {
        assert!(folo_data::convert::convert(&ds, &dir, to).is_err());
    }
    assert_eq!(files(&dir), [MARKER]);
    assert_eq!(std::fs::read_to_string(dir.join(MARKER)).unwrap(), "theirs");
}

/// A `data.yaml` outside a marked folder whose `path:` points into it is refused (registering it reads it the same
/// way), as is the folder's own `data.yaml` (all files written, the marker not removed).
#[test]
fn a_yaml_pointing_into_a_marked_folder_is_refused() {
    let t = TempDir::new("unfinished-yaml");
    let dir = t.join("shapes");
    write_yolo(&small(6), &dir, ImageFormat::Png).unwrap();
    write(&dir.join(MARKER), "");
    let outside = t.join("outside.yaml");
    let yaml = format!("path: {}\ntrain: images/train\nval: images/val\nnames: [a]\n", dir.display());
    write(&outside, yaml.replace('\\', "/"));
    assert_refused(open(&outside).unwrap_err());
    assert_refused(check(&outside, &CheckOptions::default()).unwrap_err());
    assert_unfinished(&dir.join("data.yaml"));
}

/// A marked COCO output is refused through its annotation file, also spelled in another letter case on Windows.
#[test]
fn a_coco_annotation_file_of_a_marked_folder_is_refused() {
    let t = TempDir::new("unfinished-coco");
    write_yolo(&small(6), &t.join("src"), ImageFormat::Png).unwrap();
    let out = t.join("coco");
    folo_data::convert::convert(&open(t.join("src")).unwrap(), &out, Format::Coco).unwrap();
    assert!(!out.join(MARKER).exists());
    write(&out.join(MARKER), "");
    assert_unfinished(&out);
    assert_unfinished(&out.join("annotations/instances_train.json"));
    #[cfg(windows)]
    assert_unfinished(&out.join("ANNOTATIONS/instances_train.json"));
}
