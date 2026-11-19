//! A dataset folder written in place stays recognisably unfinished until complete (issue #316): readers refuse a
//! stopped write, the next write clears it and writes the folder whole, and a folder being written or holding
//! anything else is left alone.

mod common;

use common::{TempDir, write};
use folo_data::check::{CheckOptions, check};
use folo_data::synthetic::{SyntheticConfig, write_yolo, write_yolo_with_progress};
use folo_data::unfinished::{MARKER, Writing};
use folo_data::{DataError, Format, ImageFormat, detect_format, open};
use std::path::Path;

fn small(images: usize) -> SyntheticConfig {
    SyntheticConfig { images, imgsz: 64, seed: 3, ..Default::default() }
}

/// A write of `dir` stopped right after it took the folder (the marker holds the writer's note).
fn stopped(dir: &Path) {
    drop(Writing::begin(dir).unwrap());
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

fn assert_unfinished(dir: &Path) {
    for err in [
        open(dir).unwrap_err(),
        detect_format(dir, Format::Auto).unwrap_err(),
        check(dir, &CheckOptions::default()).unwrap_err(),
    ] {
        assert!(matches!(&err, DataError::Unfinished(_)), "{err}");
        assert!(err.to_string().contains("run that command again"), "{err}");
    }
}

fn is_already_exists(err: &DataError) -> bool {
    matches!(err, DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::AlreadyExists)
}

/// Killed part way through the images (here: a panic from the progress callback), the folder is refused by every
/// reader; the next write, for fewer images than the stopped one wrote, leaves exactly its own files.
#[test]
fn a_write_stopped_part_way_is_refused_then_written_whole() {
    let t = TempDir::new("unfinished-midway");
    let dir = t.join("shapes");
    let stop = std::panic::catch_unwind(|| {
        write_yolo_with_progress(&small(40), &dir, ImageFormat::Png, |done| assert!(done < 20, "stopped"))
    });
    assert!(stop.is_err());
    assert!(dir.join(MARKER).is_file() && files(&dir).len() > 20);
    assert_unfinished(&dir);

    write_yolo(&small(6), &dir, ImageFormat::Png).unwrap();
    assert_eq!(open(&dir).unwrap().split_sizes(), [5, 1, 0]);
    assert_eq!(files(&dir).len(), 6 * 2 + 1, "{:?}", files(&dir));
    assert!(!dir.join(MARKER).exists());
}

/// Every other moment a write can stop at: the marker made but not written, the folder taken, every file written
/// but the marker not removed. Each is refused by readers and written whole by the next write.
#[test]
fn every_stopping_point_is_refused_then_written_whole() {
    let t = TempDir::new("unfinished-stages");
    let made = t.join("made");
    write(&made.join(MARKER), "");
    let taken = t.join("taken");
    stopped(&taken);
    let written = t.join("written");
    write_yolo(&small(10), &written, ImageFormat::Png).unwrap();
    stopped(&t.join("note"));
    std::fs::copy(t.join("note").join(MARKER), written.join(MARKER)).unwrap();
    for dir in [made, taken, written] {
        assert_unfinished(&dir);
        if dir.join("data.yaml").exists() {
            assert_unfinished(&dir.join("data.yaml"));
        }
        write_yolo(&small(6), &dir, ImageFormat::Png).unwrap();
        assert_eq!(open(&dir).unwrap().split_sizes(), [5, 1, 0], "{}", dir.display());
        assert_eq!(files(&dir).len(), 6 * 2 + 1);
    }
}

/// Stopped after emptying the marker, before removing it, the dataset is complete: the next write removes the
/// marker and refuses the folder as not empty, which then reads as the dataset it holds.
#[test]
fn a_complete_dataset_beside_an_empty_marker_is_kept() {
    let t = TempDir::new("unfinished-emptied");
    let dir = t.join("shapes");
    write_yolo(&small(10), &dir, ImageFormat::Png).unwrap();
    write(&dir.join(MARKER), "");
    assert_unfinished(&dir);
    let err = write_yolo(&small(6), &dir, ImageFormat::Png).unwrap_err();
    assert!(is_already_exists(&err), "{err}");
    assert_eq!(open(&dir).unwrap().split_sizes(), [8, 2, 0]);
}

/// While one process writes a folder (it holds the marker's lock), another write of it is refused and changes
/// nothing; a folder with anything else in it, or with a marker that is not FOLO's, is refused untouched.
#[test]
fn a_folder_being_written_or_holding_other_files_is_left_alone() {
    let t = TempDir::new("unfinished-busy");
    let dir = t.join("shapes");
    let first = Writing::begin(&dir).unwrap();
    let note = std::fs::metadata(dir.join(MARKER)).unwrap().len();
    let err = write_yolo(&small(6), &dir, ImageFormat::Png).unwrap_err();
    assert!(is_already_exists(&err) && err.to_string().contains("another FOLO command is writing"), "{err}");
    assert_eq!(files(&dir), [MARKER]);
    assert_eq!(std::fs::metadata(dir.join(MARKER)).unwrap().len(), note);
    first.finish().unwrap();
    assert!(files(&dir).is_empty());

    let theirs = t.join("theirs");
    write(&theirs.join("notes.txt"), "mine");
    let foreign = t.join("foreign");
    write(&foreign.join(MARKER), "not FOLO's");
    write(&foreign.join("notes.txt"), "mine");
    for dir in [&theirs, &foreign] {
        let before = files(dir);
        let err = write_yolo(&small(6), dir, ImageFormat::Png).unwrap_err();
        assert!(is_already_exists(&err), "{err}");
        assert_eq!(files(dir), before);
        assert_eq!(std::fs::read_to_string(dir.join("notes.txt")).unwrap(), "mine");
    }
    assert_eq!(std::fs::read_to_string(foreign.join(MARKER)).unwrap(), "not FOLO's");
}

/// A folder link left in a stopped write (a junction on Windows, a symlink elsewhere) is removed as a link: what it
/// points to keeps its files.
#[test]
fn a_link_in_a_stopped_write_is_removed_as_a_link() {
    let t = TempDir::new("unfinished-link");
    let outside = t.join("outside");
    write(&outside.join("keep.txt"), "keep");
    let dir = t.join("shapes");
    stopped(&dir);
    let link = dir.join("images");
    #[cfg(windows)]
    let linked = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(link.to_string_lossy().replace('/', "\\"))
        .arg(outside.to_string_lossy().replace('/', "\\"))
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap()
        .success();
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(&outside, &link).is_ok();
    assert!(linked && link.join("keep.txt").is_file());

    write_yolo(&small(6), &dir, ImageFormat::Png).unwrap();
    assert_eq!(std::fs::read_to_string(outside.join("keep.txt")).unwrap(), "keep");
    assert_eq!(files(&outside), ["keep.txt"]);
    assert_eq!(open(&dir).unwrap().split_sizes(), [5, 1, 0]);
}

/// `convert` writes the same way: into a stopped write it writes whole, into a folder being written it is refused.
#[test]
fn convert_writes_over_a_stopped_write_only() {
    let t = TempDir::new("unfinished-convert");
    write_yolo(&small(6), &t.join("src"), ImageFormat::Png).unwrap();
    let ds = open(t.join("src")).unwrap();
    let out = t.join("coco");
    stopped(&out);
    write(&out.join("annotations/instances_train.json"), "half");
    assert_unfinished(&out);
    assert_unfinished(&out.join("annotations/instances_train.json"));
    folo_data::convert::convert(&ds, &out, Format::Coco).unwrap();
    assert_eq!(open(&out).unwrap().split_sizes(), [5, 1, 0]);
    assert!(!out.join(MARKER).exists());

    let busy = t.join("busy");
    let _writer = Writing::begin(&busy).unwrap();
    let err = folo_data::convert::convert(&ds, &busy, Format::Yolo).unwrap_err();
    assert!(is_already_exists(&err), "{err}");
    assert_eq!(files(&busy), [MARKER]);
}
