//! The persistent pipeline cache: compiled pipelines kept on disk between processes, so a run after the first
//! skips most of the driver's shader compilation (docs/GPU.md, Pipeline cache).
//!
//! It is wgpu's [`wgpu::PipelineCache`] (a `VkPipelineCache` on Vulkan; other backends do not offer one, and then
//! nothing here runs) written to `<dir>/<wgpu key>.bin` once [`set_pipeline_cache_dir`] has named a folder. The
//! file starts with FOLO's own header: a magic, a format version, the folo-gpu version, the adapter's name and
//! driver, the payload length and a checksum of the payload. A file whose header does not match this process (a
//! new driver, another adapter, another folo-gpu version), whose checksum is wrong (a torn or corrupt file) or that
//! is larger than [`MAX_FILE`] is ignored and the cache starts empty; wgpu and the driver validate the payload once
//! more (`fallback: true`). A changed shader simply misses: pipelines are looked up by their compiled code.
//!
//! Writes are atomic (a temporary file renamed over the old one) and happen off the compiling thread: a saver
//! thread writes the cache about a second after the last new pipeline was compiled, dropping the context writes
//! whatever is still unsaved, and so does [`save_pipeline_caches`] (for the process-wide context, which is never
//! dropped). A cache that has grown past [`MAX_FILE`] is not written (the previous file stays). Temporary files
//! that a killed writer left behind are deleted when a cache opens, once they are an hour old. Every failure (no
//! folder, a read-only disk) is silent: the cache only ever makes compilation faster, never makes a run fail.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime};

/// The folder of [`set_pipeline_cache_dir`]; `None` (the default) keeps the cache off.
static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
/// Every open cache, for [`save_pipeline_caches`].
static OPEN: Mutex<Vec<Weak<DiskCache>>> = Mutex::new(Vec::new());

/// A cache file larger than this is discarded instead of loaded, and a cache that would be written larger is not
/// written (pipelines of old shaders are never evicted otherwise).
pub const MAX_FILE: u64 = 256 << 20;
/// Temporary files of [`write_atomic`] older than this are a killed writer's and deleted when a cache opens.
const STALE_TMP: Duration = Duration::from_secs(3600);
/// Numbers this process's temporary files, so two writers in one process never share one.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

const MAGIC: &[u8; 8] = b"FOLOPIPE";
const FORMAT: u32 = 1;
/// How long the saver waits after the last new pipeline before it writes the cache.
#[cfg(not(target_arch = "wasm32"))]
const QUIET: Duration = Duration::from_millis(1000);

/// Keeps compiled pipelines in `dir` for contexts created afterwards (`None` turns the cache off, the default).
/// `folo` sets it to `<FOLO_HOME>/cache/gpu` (docs/GPU.md, Pipeline cache).
pub fn set_pipeline_cache_dir(dir: Option<PathBuf>) {
    *DIR.lock().unwrap_or_else(|p| p.into_inner()) = dir;
}

/// The folder set by [`set_pipeline_cache_dir`].
pub fn pipeline_cache_dir() -> Option<PathBuf> {
    DIR.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Writes every context's pipeline cache that has pipelines not on disk yet, now (a process about to exit calls
/// this: the saver thread waits for a quiet second first). `folo` calls it when a command ends. With
/// `FOLO_GPU_TRACE_COMPILE=1` it also prints the compile timer's totals (docs/GPU.md, Pipeline cache).
pub fn save_pipeline_caches() {
    #[cfg(not(target_arch = "wasm32"))]
    crate::pipeline::trace::report();
    #[cfg(not(target_arch = "wasm32"))]
    crate::kernel_list::save_all();
    let open: Vec<Arc<DiskCache>> = {
        let mut v = OPEN.lock().unwrap_or_else(|p| p.into_inner());
        v.retain(|w| w.strong_count() > 0);
        v.iter().filter_map(Weak::upgrade).collect()
    };
    for d in open {
        d.save();
    }
}

/// The cache file for `info` in `dir`, when the backend has pipeline caches.
pub(crate) fn file_for(dir: &Path, info: &wgpu::AdapterInfo) -> Option<PathBuf> {
    wgpu::util::pipeline_cache_key(info).map(|k| dir.join(format!("{k}.bin")))
}

/// The cache file a context on `adapter` should use with the folder `dir`, when the adapter has
/// [`wgpu::Features::PIPELINE_CACHE`].
pub(crate) fn wanted(adapter: &wgpu::Adapter, dir: Option<&Path>) -> Option<PathBuf> {
    if !adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    file_for(dir?, &adapter.get_info())
}

/// What identifies the process that may reuse a file: folo-gpu's version and the adapter and driver.
fn identity(info: &wgpu::AdapterInfo) -> String {
    format!(
        "folo-gpu {}; {:?} {:04x}:{:04x} {}; driver {} {}",
        env!("CARGO_PKG_VERSION"),
        info.backend,
        info.vendor,
        info.device,
        info.name,
        info.driver,
        info.driver_info
    )
}

/// 64-bit FNV-1a, eight bytes at a time for speed (a checksum, not a hash table key).
pub(crate) fn checksum(data: &[u8]) -> u64 {
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

/// The file's payload when its header matches `identity` and its checksum is right.
pub(crate) fn decode<'a>(file: &'a [u8], identity: &str) -> Option<&'a [u8]> {
    decode_as(MAGIC, FORMAT, file, identity)
}

/// [`decode`] for a file of another kind (`magic`, `format`): the kernel lists of `crate::kernel_list` use the
/// same header.
pub(crate) fn decode_as<'a>(
    magic: &[u8; 8],
    format: u32,
    file: &'a [u8],
    identity: &str,
) -> Option<&'a [u8]> {
    let mut r = file;
    let mut take = |n: usize| -> Option<&'a [u8]> {
        let (a, b) = r.split_at_checked(n)?;
        r = b;
        Some(a)
    };
    if take(8)? != magic || u32::from_le_bytes(take(4)?.try_into().ok()?) != format {
        return None;
    }
    let id_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if take(id_len)? != identity.as_bytes() {
        return None;
    }
    let len = u64::from_le_bytes(take(8)?.try_into().ok()?);
    let sum = u64::from_le_bytes(take(8)?.try_into().ok()?);
    let payload = take(usize::try_from(len).ok()?)?;
    (r.is_empty() && checksum(payload) == sum).then_some(payload)
}

/// Whether `file` is a whole, valid cache file for some identity (tests).
#[cfg(test)]
pub(crate) fn decode_for_test(file: &[u8]) -> bool {
    let Some(id_len) = file.get(12..16).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
    else {
        return false;
    };
    let Some(id) = file.get(16..16 + id_len).and_then(|b| std::str::from_utf8(b).ok()) else { return false };
    decode(file, id).is_some()
}

/// A cache file holding `payload`.
pub(crate) fn encode(payload: &[u8], identity: &str) -> Vec<u8> {
    encode_as(MAGIC, FORMAT, payload, identity)
}

/// [`encode`] for a file of another kind (see [`decode_as`]).
pub(crate) fn encode_as(magic: &[u8; 8], format: u32, payload: &[u8], identity: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + identity.len() + 32);
    out.extend_from_slice(magic);
    out.extend_from_slice(&format.to_le_bytes());
    out.extend_from_slice(&(identity.len() as u32).to_le_bytes());
    out.extend_from_slice(identity.as_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Writes `bytes` to `path` through a temporary file in the same folder (`.<name>.<pid>-<n>.tmp`), so a reader
/// never sees half a file. The temporary file is removed when the write or the rename fails.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".{name}.{}-{n}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, path)).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Whether `name` is one of [`write_atomic`]'s temporary files for the cache file `file`.
fn is_tmp_of(name: &str, file: &str) -> bool {
    let digits = |n: &str| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit());
    name.strip_prefix('.')
        .and_then(|n| n.strip_prefix(file))
        .and_then(|n| n.strip_prefix('.'))
        .and_then(|n| n.strip_suffix(".tmp"))
        .and_then(|n| n.split_once('-'))
        .is_some_and(|(pid, n)| digits(pid) && digits(n))
}

/// Deletes the temporary files of `path` older than [`STALE_TMP`] (a killed writer's).
pub(crate) fn remove_stale_tmp(path: &Path) {
    let (Some(dir), Some(file)) = (path.parent(), path.file_name().and_then(|n| n.to_str())) else { return };
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let now = SystemTime::now();
    for e in rd.flatten() {
        if !e.file_name().to_str().is_some_and(|n| is_tmp_of(n, file)) {
            continue;
        }
        let stale = e
            .metadata()
            .ok()
            .filter(|m| m.is_file())
            .and_then(|m| m.modified().ok())
            .and_then(|t| now.duration_since(t).ok())
            .is_some_and(|age| age > STALE_TMP);
        if stale {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// A context's on-disk pipeline cache.
pub(crate) struct DiskCache {
    pub cache: wgpu::PipelineCache,
    path: PathBuf,
    identity: String,
    /// Whether a valid file was loaded when the cache opened (else it started empty).
    pub loaded: bool,
    /// Pipelines compiled through the cache, and how many of them the last write covered.
    compiled: AtomicU64,
    saved: AtomicU64,
    /// Serialises writes (the saver thread and drop).
    writing: Mutex<()>,
}

impl DiskCache {
    /// Opens the cache at `path` on `device` (created with [`wgpu::Features::PIPELINE_CACHE`]), loading the file
    /// when it is valid for this adapter and starting empty otherwise, and starts its saver thread.
    pub fn open(device: &wgpu::Device, info: &wgpu::AdapterInfo, path: PathBuf) -> Arc<Self> {
        remove_stale_tmp(&path);
        let identity = identity(info);
        let file = std::fs::metadata(&path)
            .ok()
            .filter(|m| m.len() <= MAX_FILE)
            .and_then(|_| std::fs::read(&path).ok());
        let data = file.as_deref().and_then(|f| decode(f, &identity));
        // SAFETY: `data` is what `PipelineCache::get_data` returned for an adapter with the same wgpu cache key,
        // backend, name and driver (FOLO's header says so and its checksum matches); wgpu validates its own header
        // again and `fallback: true` starts an empty cache when the data is not usable.
        let cache = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("folo-gpu pipeline cache"),
                data,
                fallback: true,
            })
        };
        let disk = Arc::new(Self {
            cache,
            path,
            identity,
            loaded: data.is_some(),
            compiled: AtomicU64::new(0),
            saved: AtomicU64::new(0),
            writing: Mutex::new(()),
        });
        {
            let mut open = OPEN.lock().unwrap_or_else(|p| p.into_inner());
            open.retain(|w| w.strong_count() > 0);
            open.push(Arc::downgrade(&disk));
        }
        #[cfg(not(target_arch = "wasm32"))]
        spawn_saver(Arc::downgrade(&disk));
        disk
    }

    /// Notes one more pipeline compiled through the cache.
    pub fn compiled(&self) {
        self.compiled.fetch_add(1, Ordering::Relaxed);
    }

    /// Writes the cache when pipelines were compiled since the last write, unless the file would be larger than
    /// [`MAX_FILE`] (the previous file is then kept). Errors are ignored.
    pub fn save(&self) {
        let _w = self.writing.lock().unwrap_or_else(|p| p.into_inner());
        let n = self.compiled.load(Ordering::Relaxed);
        if n == self.saved.load(Ordering::Relaxed) {
            return;
        }
        let Some(data) = self.cache.get_data() else { return };
        let file = encode(&data, &self.identity);
        // Too large: not written, and not tried again until more pipelines are compiled.
        if file.len() as u64 > MAX_FILE || write_atomic(&self.path, &file).is_ok() {
            self.saved.store(n, Ordering::Relaxed);
        }
    }

    /// The file this cache is written to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        self.save();
    }
}

/// Writes the cache once compilation has gone quiet for [`QUIET`]; ends when the cache is dropped.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_saver(disk: Weak<DiskCache>) {
    let _ = std::thread::Builder::new().name("folo-gpu-pipeline-cache".into()).spawn(move || {
        let mut seen = 0;
        loop {
            std::thread::sleep(QUIET);
            let Some(d) = disk.upgrade() else { return };
            let n = d.compiled.load(Ordering::Relaxed);
            // Write only after a quiet interval: no new pipeline since the last look.
            if n == seen && n != d.saved.load(Ordering::Relaxed) {
                d.save();
            }
            seen = n;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_and_rejects_changes() {
        let payload: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let file = encode(&payload, "id a");
        assert_eq!(decode(&file, "id a"), Some(&payload[..]));
        // Another adapter, driver or version.
        assert_eq!(decode(&file, "id b"), None);
        // Every truncation, and one flipped bit anywhere, is rejected.
        for n in 0..file.len() {
            assert_eq!(decode(&file[..n], "id a"), None, "truncated to {n}");
        }
        for i in [0, 9, 13, 20, 30, file.len() / 2, file.len() - 1] {
            let mut bad = file.clone();
            bad[i] ^= 1;
            assert_eq!(decode(&bad, "id a"), None, "bit flipped at {i}");
        }
        let mut longer = file.clone();
        longer.push(0);
        assert_eq!(decode(&longer, "id a"), None);
    }

    #[test]
    fn temporary_file_names() {
        let file = "wgpu_pipeline_cache_vulkan_10de_2702.bin";
        assert!(is_tmp_of(&format!(".{file}.1234-0.tmp"), file));
        assert!(is_tmp_of(&format!(".{file}.1-56.tmp"), file));
        for other in [
            format!("{file}.1234-0.tmp"),
            format!(".{file}.1234.tmp"),
            format!(".{file}.x-0.tmp"),
            format!(".{file}.1-0.tmp.keep"),
            ".other.bin.1-0.tmp".to_string(),
            "notes.tmp".to_string(),
        ] {
            assert!(!is_tmp_of(&other, file), "{other}");
        }
    }
}
