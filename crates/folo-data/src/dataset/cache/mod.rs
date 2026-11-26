//! The dataset scan cache: what opening a large dataset learnt about its files, kept on disk so the next open
//! skips the work (docs/DATA.md, Scan cache). Opening COCO's 118 287 training images otherwise reads every image
//! header (30-80 s on a cold disk) and parses a 448 MB JSON file (5-8 s) at every `folo train` start.
//!
//! Two kinds of entries, both off until [`set_scan_cache_dir`] names a folder (`folo` sets `<FOLO_HOME>/cache/scan`):
//!
//! - **Per folder** ([`probe`]): for every image in a folder, the size its header gives (or why it cannot be
//!   decoded), and for every YOLO label file, its parsed boxes and warnings (or why it is broken), each stored
//!   with the file's length and modification time. A file is reused only while both are unchanged, and the
//!   folder's listing gives them without opening any file (one directory read instead of a stat per file).
//! - **Per COCO annotation set** ([`coco`]): everything derived from the JSON files alone (classes, every image's
//!   boxes, the issues found), stored with every file's path, length and modification time and reused only
//!   while all of them are unchanged.
//!
//! A cache file is checked as a whole (its length, a magic, a format version, its key and a checksum of its
//! contents) and ignored when anything is off, so a torn, corrupt or stale file costs one normal open, never a
//! wrong result. Results come out exactly as without the cache (same entries in the same order, same issues and
//! messages) as long as every change to a file changes its length or modification time; the folder listings are a
//! snapshot taken when the open starts. Only folders with at least [`MIN_FILES`] files of a dataset are listed and
//! cached (small ones gain nothing). With a read guard installed ([`crate::guard`]) nothing is cached or listed:
//! every file is read and checked again.
//! Bump [`FORMAT`] when the scan's rules change (what a probe or a label parse returns, the JSON rules).
//!
//! Files: [`probe`], [`coco`], [`listing`] (folder listings and file stamps) and `tests.rs`.

pub(crate) mod coco;
pub(crate) mod listing;
pub(crate) mod probe;

use std::cell::RefCell;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The folder of [`set_scan_cache_dir`].
static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Files of a dataset in one folder below which the folder is neither listed nor cached.
pub const MIN_FILES: usize = 256;

/// Version of the cached data; part of every file's header.
pub(crate) const FORMAT: u32 = 4;
const MAGIC: &[u8; 8] = b"FOLOSCAN";
/// A cache file larger than this is ignored (COCO train2017's entry is 24 MB).
const MAX_CACHE_FILE: u64 = 1 << 30;
/// The kinds of cache file, the first part of their names.
const KINDS: [&str; 2] = [coco::KIND, probe::KIND];
/// Numbers this process's temporary files, so two writers in one process never share one.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);
/// Cache files not written for this long are deleted when the next one is written.
const MAX_AGE: Duration = Duration::from_secs(60 * 86_400);
/// The most the cache folder's files may take together: past it, the least recently written are deleted when the
/// next one is written (COCO train + val take 30 MB).
pub const MAX_TOTAL: u64 = 2 << 30;
/// A file modified less than this long before it is scanned is not cached: a write within the same timestamp
/// tick could change it without changing its length or modification time.
const SETTLE: Duration = Duration::from_secs(2);

/// Caches dataset scans in `dir` from now on (`None`, the default, turns the cache off). The folder is created
/// when the first entry is written.
pub fn set_scan_cache_dir(dir: Option<PathBuf>) {
    *DIR.lock().unwrap_or_else(|p| p.into_inner()) = dir;
}

/// The folder set by [`set_scan_cache_dir`].
pub fn scan_cache_dir() -> Option<PathBuf> {
    DIR.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// How one open uses the cache, read once on the opening thread.
#[derive(Clone, Debug)]
pub(crate) struct Settings {
    /// The cache folder; `None` when caching is off (no folder set, or a read guard installed).
    pub dir: Option<PathBuf>,
    /// Folders with fewer files of the dataset are not listed or cached ([`MIN_FILES`]).
    pub min_files: usize,
    /// Whether folders may be listed at all (not with a read guard: every file is then checked on its own).
    pub list: bool,
    /// The clock file stamps are checked against ([`Stamp::settled`]): `None` (outside tests) is the system clock
    /// when the check is made.
    pub now: Option<SystemTime>,
}

thread_local! {
    /// Tests' settings for opens on this thread (the process-wide folder is shared by tests running in parallel).
    static OVERRIDE: RefCell<Option<Settings>> = const { RefCell::new(None) };
}

impl Settings {
    /// This thread's settings: the process-wide folder and [`MIN_FILES`], or a test's override.
    pub fn current() -> Self {
        if let Some(s) = OVERRIDE.with(|o| o.borrow().clone()) {
            return s;
        }
        Self::new(scan_cache_dir(), crate::guard::installed())
    }

    /// The settings with the cache folder `dir`, and with a read guard installed when `guarded` (then nothing is
    /// cached or listed).
    fn new(dir: Option<PathBuf>, guarded: bool) -> Self {
        Settings { dir: if guarded { None } else { dir }, min_files: MIN_FILES, list: !guarded, now: None }
    }
}

/// Runs `f` with `settings` for opens on this thread (tests).
#[cfg(test)]
pub(crate) fn with_settings<T>(settings: Settings, f: impl FnOnce() -> T) -> T {
    let prev = OVERRIDE.with(|o| o.replace(Some(settings)));
    let out = f();
    OVERRIDE.with(|o| *o.borrow_mut() = prev);
    out
}

/// A file's length and modification time (nanoseconds since the Unix epoch), as a cache entry is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stamp {
    pub len: u64,
    pub mtime: i64,
}

impl Stamp {
    pub fn of(m: &std::fs::Metadata) -> Option<Self> {
        let t = m.modified().ok()?;
        let mtime = match t.duration_since(UNIX_EPOCH) {
            Ok(d) => i64::try_from(d.as_nanos()).ok()?,
            Err(e) => -i64::try_from(e.duration().as_nanos()).ok()?,
        };
        Some(Stamp { len: m.len(), mtime })
    }

    /// Whether the file was last modified long enough ago (and not in the future) for its stamp to identify it.
    pub fn settled(&self, now: SystemTime) -> bool {
        let Ok(now) = now.duration_since(UNIX_EPOCH) else { return false };
        let now = i64::try_from(now.as_nanos()).unwrap_or(i64::MAX);
        self.mtime <= now.saturating_sub(SETTLE.as_nanos() as i64)
    }
}

/// 64-bit FNV-1a: cache file names and checksums.
pub(crate) fn fnv(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let (chunks, rest) = data.as_chunks::<8>();
    for c in chunks {
        h = (h ^ u64::from_le_bytes(*c)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &b in rest {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Little-endian encoder of cache contents.
#[derive(Default)]
pub(crate) struct Enc(pub Vec<u8>);

impl Enc {
    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    /// An `f32` by bit pattern, so it comes back exactly.
    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    pub fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
    pub fn opt_str(&mut self, s: Option<&str>) {
        match s {
            Some(s) => {
                self.u8(1);
                self.str(s);
            }
            None => self.u8(0),
        }
    }
    pub fn stamp(&mut self, s: Stamp) {
        self.u64(s.len);
        self.i64(s.mtime);
    }
    /// A count, as `u64`.
    pub fn len(&mut self, n: usize) {
        self.u64(n as u64);
    }
}

/// Decoder matching [`Enc`]; every read is checked (`None` on a short or malformed buffer).
pub(crate) struct Dec<'a>(pub &'a [u8]);

impl<'a> Dec<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (a, b) = self.0.split_at_checked(n)?;
        self.0 = b;
        Some(a)
    }
    pub fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    pub fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    pub fn f32(&mut self) -> Option<f32> {
        Some(f32::from_bits(self.u32()?))
    }
    pub fn str(&mut self) -> Option<&'a str> {
        let n = usize::try_from(self.u64()?).ok()?;
        std::str::from_utf8(self.take(n)?).ok()
    }
    pub fn opt_str(&mut self) -> Option<Option<&'a str>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(self.str()?)),
            _ => None,
        }
    }
    pub fn stamp(&mut self) -> Option<Stamp> {
        Some(Stamp { len: self.u64()?, mtime: self.i64()? })
    }
    /// A count written by [`Enc::len`], refused when it could not fit in what is left (each item at least
    /// `min_item` bytes), so a corrupt count cannot ask for a huge allocation.
    pub fn len(&mut self, min_item: usize) -> Option<usize> {
        let n = usize::try_from(self.u64()?).ok()?;
        (n.checked_mul(min_item.max(1))? <= self.0.len()).then_some(n)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The cache file of `kind` for `key`.
fn file_for(dir: &Path, kind: &str, key: &str) -> PathBuf {
    dir.join(format!("{kind}-{:016x}.bin", fnv(key.as_bytes())))
}

/// The contents of the cache file of `kind` for `key`, when it exists and its header, key and checksum match. The
/// file's length is checked (at most [`MAX_CACHE_FILE`], exactly its header plus the payload length the header
/// gives) before the payload is read.
pub(crate) fn read(dir: &Path, kind: &str, key: &str) -> Option<Vec<u8>> {
    let mut f = std::fs::File::open(file_for(dir, kind, key)).ok()?;
    let file_len = f.metadata().ok()?.len();
    if file_len > MAX_CACHE_FILE {
        return None;
    }
    // Magic, format, kind and key (each with its length), checksum, payload length.
    let head_len = 8 + 4 + 8 + kind.len() + 8 + key.len() + 8 + 8;
    let mut head = vec![0; head_len];
    f.read_exact(&mut head).ok()?;
    let mut d = Dec(&head);
    if d.take(8)? != MAGIC || d.u32()? != FORMAT || d.str()? != kind || d.str()? != key {
        return None;
    }
    let sum = d.u64()?;
    let len = d.u64()?;
    if (head_len as u64).checked_add(len)? != file_len {
        return None;
    }
    let mut payload = Vec::with_capacity(usize::try_from(len).ok()?);
    f.take(len + 1).read_to_end(&mut payload).ok()?;
    (payload.len() as u64 == len && fnv(&payload) == sum).then_some(payload)
}

/// Writes `payload` as the cache file of `kind` for `key` (atomically: a temporary file renamed over the old one)
/// and prunes the folder ([`prune`]). A file that would be larger than [`MAX_CACHE_FILE`] (it could never be read)
/// is not written. Errors are ignored: a cache that cannot be written only means the next open does the work
/// again.
pub(crate) fn write(dir: &Path, kind: &str, key: &str, payload: &[u8]) {
    if payload.len() as u64 + (key.len() + kind.len() + 64) as u64 > MAX_CACHE_FILE {
        return;
    }
    let mut e = Enc::default();
    e.0.extend_from_slice(MAGIC);
    e.u32(FORMAT);
    e.str(kind);
    e.str(key);
    e.u64(fnv(payload));
    e.u64(payload.len() as u64);
    e.0.extend_from_slice(payload);
    let path = file_for(dir, kind, key);
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = dir.join(format!(
        ".{}.{}-{}.tmp",
        path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    if std::fs::write(&tmp, &e.0).is_err() || std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    prune(dir, &path);
}

/// Deletes cache files (and temporary files left by a killed writer) older than [`MAX_AGE`], then, while the rest
/// take more than [`MAX_TOTAL`] together, the least recently written ones (never `keep`, the file just written).
/// Only files named as this module names them ([`is_cache_name`]) are ever deleted or counted, so a folder shared
/// with other files keeps them.
fn prune(dir: &Path, keep: &Path) {
    prune_to(dir, keep, MAX_TOTAL);
}

fn prune_to(dir: &Path, keep: &Path, max_total: u64) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let now = SystemTime::now();
    let mut files = Vec::new();
    for e in rd.flatten() {
        if !e.file_name().to_str().is_some_and(is_cache_name) {
            continue;
        }
        let Some(m) = e.metadata().ok().filter(|m| m.is_file()) else { continue };
        let modified = m.modified().unwrap_or(now);
        if now.duration_since(modified).is_ok_and(|age| age > MAX_AGE) {
            let _ = std::fs::remove_file(e.path());
        } else {
            files.push((modified, m.len(), e.path()));
        }
    }
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, path) in files {
        if total <= max_total {
            break;
        }
        if path == keep {
            continue;
        }
        match std::fs::remove_file(&path) {
            // Gone either way: another process pruning at the same time may have deleted it first.
            Ok(()) => total -= len,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => total -= len,
            Err(_) => {}
        }
    }
}

/// Whether `name` is a cache file (`<kind>-<16 hex digits>.bin`) or one of [`write`]'s temporary files
/// (`.<cache file name>.<pid>-<n>.tmp`).
fn is_cache_name(name: &str) -> bool {
    let cache_file = |n: &str| {
        KINDS.iter().any(|k| {
            n.strip_prefix(k)
                .and_then(|r| r.strip_prefix('-'))
                .and_then(|r| r.strip_suffix(".bin"))
                .is_some_and(|h| h.len() == 16 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        })
    };
    let digits = |n: &str| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit());
    if let Some(t) = name.strip_prefix('.').and_then(|n| n.strip_suffix(".tmp"))
        && let Some((file, ids)) = t.rsplit_once('.')
        && let Some((pid, n)) = ids.split_once('-')
    {
        return cache_file(file) && digits(pid) && digits(n);
    }
    cache_file(name)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod prune_tests {
    use super::*;
    use folo_test_support::TestDir;

    #[test]
    fn the_least_recently_written_go_first_past_the_total() {
        let dir = TestDir::new("scan-prune-total");
        let name = |i: u64| dir.join(format!("dir-{i:016x}.bin"));
        let now = SystemTime::now();
        for i in 0..4u64 {
            std::fs::write(name(i), vec![0u8; 100]).unwrap();
            let f = std::fs::File::options().write(true).open(name(i)).unwrap();
            f.set_modified(now - Duration::from_secs(3600 * (4 - i))).unwrap();
        }
        std::fs::write(dir.join("other.bin"), vec![0u8; 1000]).unwrap();
        // 400 bytes of cache files, at most 300: one goes, the oldest but `keep`, never another tool's file.
        prune_to(&dir, &name(0), 300);
        let left = || (0..4).map(|i| name(i).exists()).collect::<Vec<_>>();
        assert_eq!(left(), [true, false, true, true]);
        // At most 150: oldest first until under it.
        prune_to(&dir, &name(3), 150);
        assert_eq!(left(), [false, false, false, true]);
        assert!(dir.join("other.bin").exists());
    }
}
