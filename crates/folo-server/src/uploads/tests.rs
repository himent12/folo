use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::http::StatusCode;
use folo_api::{CreateUpload, DatasetFormat, UploadKind};
use folo_test_support::TestDir;

use super::fingerprints::Fingerprints;
use super::pipe::{Finish, Pipe, UnpackGuard};
use super::{MAX_PART_BYTES, Uploads};

fn create_req(kind: UploadKind, name: &str) -> CreateUpload {
    CreateUpload { kind, name: name.into(), format: DatasetFormat::Auto, fingerprint: None, bytes: None }
}

fn raw_tar_entry(name: &str, data: &[u8]) -> Vec<u8> {
    let mut h = tar::Header::new_gnu();
    h.as_mut_bytes()[..name.len()].copy_from_slice(name.as_bytes());
    h.set_entry_type(tar::EntryType::Regular);
    h.set_mode(0o644);
    h.set_size(data.len() as u64);
    h.set_cksum();
    let mut out = h.as_bytes().to_vec();
    out.extend_from_slice(data);
    out.resize(out.len().div_ceil(512) * 512 + 1024, 0);
    out
}

fn staging_entries(data_dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(data_dir.join("uploads")) else { return Vec::new() };
    rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()
}

/// `Fingerprints` reads `uploads.json` at most once and serves subsequent lookups and updates from memory.
#[test]
fn fingerprints_cache_reads_uploads_json_once() {
    let dir = TestDir::new("uploads-fp-cache");
    let uploads = Uploads::open(dir.path());
    assert_eq!(uploads.known(UploadKind::Model, "fp1"), None);
    assert_eq!(uploads.known.disk_reads.load(Ordering::Relaxed), 1);

    uploads.remember(UploadKind::Model, "fp1", "m_1");
    assert_eq!(uploads.known(UploadKind::Model, "fp1").as_deref(), Some("m_1"));
    uploads.remember(UploadKind::Model, "fp1", "m_2");
    assert_eq!(uploads.known(UploadKind::Model, "fp1").as_deref(), Some("m_2"));
    assert_eq!(uploads.known.disk_reads.load(Ordering::Relaxed), 1, "no repeat disk reads after warmup");

    // Fresh instance loads persisted state on first lookup.
    uploads.shutdown();
    let reopened = Uploads::open(dir.path());
    assert_eq!(reopened.known(UploadKind::Model, "fp1").as_deref(), Some("m_2"));
    assert_eq!(reopened.known.disk_reads.load(Ordering::Relaxed), 1);
    reopened.shutdown();

    // A transient non-NotFound read error (e.g. a directory in place of `uploads.json`) is not cached.
    let bad = TestDir::new("uploads-fp-ioerr");
    std::fs::create_dir(bad.join("uploads.json")).unwrap();
    let fp = Fingerprints::new(bad.join("uploads.json"));
    assert_eq!(fp.find(UploadKind::Model, "fp1"), None);
    assert_eq!(fp.find(UploadKind::Model, "fp1"), None);
    assert_eq!(fp.disk_reads.load(Ordering::Relaxed), 2, "transient I/O error is retried, not latched");
}

/// `Uploads::open` removes leftover session directories and stale `.uploads.json.*.tmp` files.
#[test]
fn open_cleans_leftover_sessions_and_stale_temp_files() {
    let dir = TestDir::new("uploads-open-cleanup");
    std::fs::create_dir_all(dir.join("uploads").join("up_leftover")).unwrap();
    std::fs::write(dir.join("uploads").join("up_leftover").join("partial.bin"), b"abc").unwrap();
    let stale_tmp = dir.join(".uploads.json.99999.0.tmp");
    std::fs::write(&stale_tmp, b"partial").unwrap();
    let unrelated = dir.join("keep.txt");
    std::fs::write(&unrelated, b"keep").unwrap();

    let uploads = Uploads::open(dir.path());
    assert!(staging_entries(dir.path()).is_empty());
    assert!(!stale_tmp.exists(), "stale atomic temp file should be cleaned on open");
    assert!(unrelated.exists(), "unrelated files must remain untouched");
    uploads.shutdown();
}

/// `Uploads::shutdown` wakes the condvar sweeper and joins it immediately (no 100 ms sleep-loop lag).
#[test]
fn upload_sweeper_wakes_and_shuts_down_promptly() {
    let dir = TestDir::new("uploads-sweeper-wake");
    let uploads = Uploads::open(dir.path());
    let t0 = Instant::now();
    uploads.shutdown();
    assert!(t0.elapsed() < Duration::from_secs(2), "shutdown stalled: {:?}", t0.elapsed());
    assert!(uploads.sweeper_thread.lock().unwrap().is_none());
}

/// Direct `Uploads::put` rejects parts over `MAX_PART_BYTES` with `413 payload_too_large` and keeps the session open.
#[test]
fn put_over_max_part_bytes_returns_413_with_hint_and_preserves_session() {
    let dir = TestDir::new("uploads-size-limit");
    let uploads = Uploads::open(dir.path());
    let u = uploads.create(&create_req(UploadKind::Model, "weights")).unwrap();

    let oversized = Bytes::from(vec![0u8; MAX_PART_BYTES + 1]);
    let err = uploads.put(&u.id, 0, oversized).unwrap_err();
    assert_eq!(err.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(err.body.code, "payload_too_large");
    assert!(err.body.hint.as_deref().is_some_and(|h| h.contains("max_part_bytes")));

    // Session remains usable for a properly sized part and cleans up when deleted.
    let updated = uploads.put(&u.id, 0, Bytes::from_static(b"hello")).unwrap();
    assert_eq!((updated.part, updated.received_bytes), (1, 5));
    uploads.delete(&u.id).unwrap();
    assert!(staging_entries(dir.path()).is_empty());
    uploads.shutdown();
}

/// Path-traversal attempts (including backslash traversals on non-Windows) fail with `422 invalid_upload` and clean
/// up both any earlier extracted files and the session folder.
#[test]
fn path_traversal_and_midstream_failure_clean_staging_dir() {
    let dir = TestDir::new("uploads-traversal-cleanup");
    let uploads = Uploads::open(dir.path());

    for bad_name in ["../escape.txt", "a/../../escape.txt", "/etc/passwd", r"..\escape.txt", "a:stream"] {
        let u = uploads.create(&create_req(UploadKind::Dataset, "bad")).unwrap();
        // Prepend a valid entry so disk files already exist when the hostile second entry arrives.
        let mut archive = raw_tar_entry("data.yaml", b"names: [cat]\n");
        archive.truncate(archive.len() - 1024);
        archive.extend_from_slice(&raw_tar_entry(bad_name, b"evil"));

        let err = uploads.put(&u.id, 0, Bytes::from(archive)).unwrap_err();
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY, "{bad_name}: {err}");
        assert_eq!(err.body.code, "invalid_upload", "{bad_name}: {err}");
        assert!(staging_entries(dir.path()).is_empty(), "{bad_name}: staging dir not cleaned");
        assert_eq!(uploads.get(&u.id).unwrap_err().status, StatusCode::NOT_FOUND);
    }

    uploads.shutdown();
}

/// Aborting a dataset upload mid-file or dropping a finished model cleans up its staging folder completely.
#[test]
fn aborted_mid_file_and_dropped_finished_clean_up_temp_files() {
    let dir = TestDir::new("uploads-abort-cleanup");
    let uploads = Uploads::open(dir.path());

    // Dataset upload aborted while unpacker is blocked midway through a file entry.
    let u = uploads.create(&create_req(UploadKind::Dataset, "halfway")).unwrap();
    let mut header = tar::Header::new_gnu();
    header.set_path("images/train/a.jpg").unwrap();
    header.set_entry_type(tar::EntryType::Regular);
    header.set_mode(0o644);
    header.set_size(4096);
    header.set_cksum();
    let mut partial = header.as_bytes().to_vec();
    partial.extend_from_slice(&[0xAB; 256]);
    uploads.put(&u.id, 0, Bytes::from(partial)).unwrap();
    assert_eq!(staging_entries(dir.path()), vec![u.id.clone()]);

    uploads.delete(&u.id).unwrap();
    assert!(staging_entries(dir.path()).is_empty(), "aborted dataset staging folder was not removed");

    // Model upload completed and then dropped (simulating failed adoption) cleans up `model.folo` and folder.
    let m = uploads.create(&create_req(UploadKind::Model, "aborted-model")).unwrap();
    uploads.put(&m.id, 0, Bytes::from_static(b"not-a-folo-file")).unwrap();
    let finished = uploads.finish(&m.id).unwrap();
    assert!(finished.file().exists());
    drop(finished);
    assert!(staging_entries(dir.path()).is_empty(), "dropped Finished did not remove staging folder");

    uploads.shutdown();
}

/// If an unpacker thread unwinds unexpectedly, `UnpackGuard` wakes waiters with an error instead of deadlocking.
#[test]
fn unpack_guard_prevents_pipe_deadlock_on_panic() {
    let pipe = Arc::new(Pipe::default());
    let worker_pipe = pipe.clone();
    let handle = std::thread::spawn(move || {
        let _guard = UnpackGuard::new(worker_pipe);
        panic!("simulated unpacker panic");
    });
    let _ = handle.join();

    let feed_err = pipe.feed(Bytes::from_static(b"data")).unwrap_err();
    assert_eq!(feed_err.status, StatusCode::INTERNAL_SERVER_ERROR);
    match pipe.finish() {
        Finish::Failed(err) => assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR),
        _ => panic!("expected Finish::Failed after unpacker panic"),
    }
}
