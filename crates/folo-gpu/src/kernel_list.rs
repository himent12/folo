//! Kernel lists: the kernels (name and specialisation constants, in order of first use) one kind of run compiled,
//! kept on disk beside the pipeline cache so the next run of that kind compiles them all up front on every core
//! instead of one by one as its first step meets them ([`crate::GpuContext::precompile`], docs/GPU.md, Pipeline
//! cache).
//!
//! A list lives in the pipeline cache folder ([`crate::set_pipeline_cache_dir`]) as
//! `kernels-<backend>-<vendor>-<device>-<kind hash>-<shape hash>.list`. The kind names what decides most kernels
//! (`folo train` passes the command, model, precision, head, optimizer and class count), the shape what decides the
//! rest (image size and batch: tile sizes and bounds checks follow the tensor shapes). With no list for its shape, a
//! run uses the most recently written list of its kind, which shares most kernels. A list has the pipeline cache's
//! header: a magic, a format version, an identity (folo-gpu's version, the adapter and the kind, but not the
//! driver), the payload length and a checksum; a file
//! whose header does not match, whose checksum is wrong or that is larger than [`MAX_FILE`] is ignored. The payload
//! is text, one kernel per line (`name flags NAME=<type><value> ...`); a line this build cannot use (a malformed
//! one, or a kernel this build does not have) is skipped. A list is only ever a hint: a missing, stale or corrupt
//! one means compiling lazily, as without it.
//!
//! Writes are atomic (`pipeline_cache::write_atomic`). A folder holding more than [`MAX_LISTS`] lists loses the
//! least recently written ones when a list is written.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use crate::pipeline::{KernelKey, Spec};
use crate::pipeline_cache::{checksum, decode_as, encode_as, remove_stale_tmp, write_atomic};

/// A list file larger than this is ignored, and a list that would be larger is cut at the last whole line.
pub const MAX_FILE: u64 = 1 << 20;
/// Kernels kept in one list (the first ones used).
pub const MAX_ENTRIES: usize = 4096;
/// Lists kept in one folder; writing one deletes the least recently written beyond this.
pub const MAX_LISTS: usize = 64;
/// Distinct specialisation-constant names read from lists in one process (each is kept for the process).
const MAX_NAMES: usize = 1024;

const MAGIC: &[u8; 8] = b"FOLOKLST";
const FORMAT: u32 = 1;
const PREFIX: &str = "kernels-";
const SUFFIX: &str = ".list";

/// The file name of `kind`'s lists on the adapter `info`, up to the shape hash.
fn kind_prefix(info: &wgpu::AdapterInfo, kind: &str) -> String {
    let backend = format!("{:?}", info.backend).to_ascii_lowercase();
    format!("{PREFIX}{backend}-{:04x}-{:04x}-{:016x}-", info.vendor, info.device, checksum(kind.as_bytes()))
}

/// The list file for runs of `kind` at `shape` on the adapter `info`, in `dir`.
pub(crate) fn path_for(dir: &Path, info: &wgpu::AdapterInfo, kind: &str, shape: &str) -> PathBuf {
    dir.join(format!("{}{:016x}{SUFFIX}", kind_prefix(info, kind), checksum(shape.as_bytes())))
}

/// The other lists of `kind` in `dir` (other shapes), most recently written first.
pub(crate) fn neighbours(dir: &Path, info: &wgpu::AdapterInfo, kind: &str, exact: &Path) -> Vec<PathBuf> {
    let prefix = kind_prefix(info, kind);
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with(&prefix) && n.ends_with(SUFFIX)))
        .filter(|e| e.path() != exact)
        .filter_map(|e| Some((e.metadata().ok().filter(|m| m.is_file())?.modified().ok()?, e.path())))
        .collect();
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    found.into_iter().map(|(_, p)| p).collect()
}

/// What a list must have been written for: folo-gpu's version, the adapter and the kind. Unlike the pipeline
/// cache, not the driver: a list names kernels, not compiled code, so it stays useful after a driver update, which
/// is when the driver's own cache is cold and compiling takes longest.
pub(crate) fn identity_for(info: &wgpu::AdapterInfo, kind: &str) -> String {
    format!(
        "folo-gpu {}; {:?} {:04x}:{:04x} {}; kernels {kind}",
        env!("CARGO_PKG_VERSION"),
        info.backend,
        info.vendor,
        info.device,
        info.name
    )
}

/// `name` as a `&'static str`: specialisation-constant names are `'static` in [`Spec`]. Each distinct name read from
/// a list is kept once for the process, at most [`MAX_NAMES`] of them (then `None`).
fn intern(name: &str) -> Option<&'static str> {
    static NAMES: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = NAMES.lock().unwrap_or_else(|p| p.into_inner());
    let names = guard.get_or_insert_with(HashSet::new);
    if let Some(n) = names.get(name) {
        return Some(n);
    }
    if names.len() >= MAX_NAMES {
        return None;
    }
    let n: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(n);
    Some(n)
}

/// One list line for `key`.
fn encode_line(key: &KernelKey, out: &mut String) {
    out.push_str(key.name);
    out.push(' ');
    key.spec.encode_fields(out);
    out.push('\n');
}

/// The kernel of one list line, when the line is well formed and this build has the kernel.
fn decode_line(line: &str) -> Option<KernelKey> {
    let mut fields = line.split(' ');
    let name = fields.next()?;
    let spec = Spec::decode_fields(fields, intern)?;
    let def = crate::kernels::find_def(name, &spec)?;
    Some(KernelKey { name: def.name, spec })
}

/// The list file's text for `keys` (at most [`MAX_ENTRIES`], and cut so the file stays under [`MAX_FILE`]).
pub(crate) fn encode(keys: &[KernelKey], identity: &str) -> Vec<u8> {
    let budget = MAX_FILE as usize - identity.len() - 64;
    let mut text = String::new();
    let mut line = String::new();
    for k in keys.iter().take(MAX_ENTRIES) {
        line.clear();
        encode_line(k, &mut line);
        if text.len() + line.len() > budget {
            break;
        }
        text.push_str(&line);
    }
    encode_as(MAGIC, FORMAT, text.as_bytes(), identity)
}

/// The kernels of a list file written for `identity`, in order, without duplicates and lines this build cannot
/// use; `None` when the file is not a valid list for `identity`.
pub(crate) fn decode(file: &[u8], identity: &str) -> Option<Vec<KernelKey>> {
    let text = std::str::from_utf8(decode_as(MAGIC, FORMAT, file, identity)?).ok()?;
    let mut seen = HashSet::new();
    Some(text.lines().take(MAX_ENTRIES).filter_map(decode_line).filter(|k| seen.insert(k.clone())).collect())
}

/// The payload of a valid list file (tests).
#[cfg(test)]
pub(crate) fn decode_payload_for_test(file: &[u8], identity: &str) -> Vec<u8> {
    decode_as(MAGIC, FORMAT, file, identity).expect("a valid list").to_vec()
}

/// A list file holding `text` as it is (tests).
#[cfg(test)]
pub(crate) fn encode_text_for_test(text: &str, identity: &str) -> Vec<u8> {
    encode_as(MAGIC, FORMAT, text.as_bytes(), identity)
}

/// The list at `path` (empty when it is missing, too large or not valid for `identity`).
pub(crate) fn read(path: &Path, identity: &str) -> Vec<KernelKey> {
    remove_stale_tmp(path);
    std::fs::metadata(path)
        .ok()
        .filter(|m| m.is_file() && m.len() <= MAX_FILE)
        .and_then(|_| std::fs::read(path).ok())
        .and_then(|f| decode(&f, identity))
        .unwrap_or_default()
}

/// Writes the list for `keys` to `path` atomically, then deletes the least recently written lists of the folder
/// beyond [`MAX_LISTS`].
pub(crate) fn write(path: &Path, identity: &str, keys: &[KernelKey]) -> std::io::Result<()> {
    write_atomic(path, &encode(keys, identity))?;
    if let Some(dir) = path.parent() {
        prune(dir, MAX_LISTS);
    }
    Ok(())
}

/// Deletes the least recently written list files in `dir` beyond `keep`.
fn prune(dir: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut lists: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with(PREFIX) && n.ends_with(SUFFIX)))
        .filter_map(|e| {
            let m = e.metadata().ok().filter(|m| m.is_file())?;
            Some((m.modified().ok()?, e.path()))
        })
        .collect();
    if lists.len() <= keep {
        return;
    }
    lists.sort();
    for (_, p) in &lists[..lists.len() - keep] {
        let _ = std::fs::remove_file(p);
    }
}

/// One recording in a context (a kind and shape): the kernels first used since [`crate::GpuContext::precompile`]
/// started it, in order, and the set its file holds. It is written when it holds a kernel the file does not have (so a run that stopped
/// early, using fewer kernels than the list, keeps the longer list), by a saver thread once compilation has been
/// quiet for a second, by [`crate::save_pipeline_caches`], and when the context is dropped.
pub(crate) struct Recording {
    pub tag: String,
    path: PathBuf,
    identity: String,
    state: Mutex<RecState>,
    /// Serialises writes (the saver thread, [`save_all`] and drop).
    writing: Mutex<()>,
}

#[derive(Default)]
struct RecState {
    used: Vec<KernelKey>,
    seen: HashSet<KernelKey>,
    on_disk: HashSet<KernelKey>,
    /// Kernels added since the list was read or last written that the file does not have.
    new: usize,
    /// Kernels added in all (the saver writes once this stops changing).
    changes: u64,
}

/// Every live recording, for [`save_all`].
static OPEN: Mutex<Vec<Weak<Recording>>> = Mutex::new(Vec::new());

impl Recording {
    /// A recording for `tag` written to `path`, starting from the kernels `on_disk` that file holds.
    pub fn open(tag: &str, path: PathBuf, identity: String, on_disk: &[KernelKey]) -> Arc<Self> {
        let rec = Arc::new(Self {
            tag: tag.to_owned(),
            path,
            identity,
            state: Mutex::new(RecState { on_disk: on_disk.iter().cloned().collect(), ..RecState::default() }),
            writing: Mutex::new(()),
        });
        let mut open = OPEN.lock().unwrap_or_else(|p| p.into_inner());
        open.retain(|w| w.strong_count() > 0);
        open.push(Arc::downgrade(&rec));
        drop(open);
        #[cfg(not(target_arch = "wasm32"))]
        spawn_saver(Arc::downgrade(&rec));
        rec
    }

    /// Notes the first use of `key` in the context.
    pub fn used(&self, key: &KernelKey) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if s.used.len() >= MAX_ENTRIES || !s.seen.insert(key.clone()) {
            return;
        }
        s.used.push(key.clone());
        if !s.on_disk.contains(key) {
            s.new += 1;
        }
        s.changes += 1;
    }

    fn changes(&self) -> u64 {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).changes
    }

    /// Writes the list when it holds a kernel the file does not have. Errors are ignored (the list is a hint).
    pub fn save(&self) {
        let _w = self.writing.lock().unwrap_or_else(|p| p.into_inner());
        let keys = {
            let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if s.new == 0 {
                return;
            }
            s.used.clone()
        };
        if write(&self.path, &self.identity, &keys).is_ok() {
            let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
            // Kernels first used while the file was written are not in it.
            s.new = s.used.len() - keys.len();
            s.on_disk = keys.into_iter().collect();
        }
    }

    /// The list file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.save();
    }
}

/// Writes every live recording that has kernels its file does not have (`folo` calls this through
/// [`crate::save_pipeline_caches`] when a command ends).
pub(crate) fn save_all() {
    let open: Vec<Arc<Recording>> = {
        let mut v = OPEN.lock().unwrap_or_else(|p| p.into_inner());
        v.retain(|w| w.strong_count() > 0);
        v.iter().filter_map(Weak::upgrade).collect()
    };
    for r in open {
        r.save();
    }
}

/// Writes the recording once no kernel was added for a second; ends when the recording is dropped.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_saver(rec: Weak<Recording>) {
    let _ = std::thread::Builder::new().name("folo-gpu-kernel-list".into()).spawn(move || {
        let mut seen = 0;
        loop {
            std::thread::sleep(std::time::Duration::from_millis(1000));
            let Some(r) = rec.upgrade() else { return };
            let n = r.changes();
            if n == seen {
                r.save();
            }
            seen = n;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &'static str, spec: Spec) -> KernelKey {
        KernelKey { name, spec }
    }

    #[test]
    fn lists_round_trip_and_reject_changes() {
        let keys = vec![
            key("gemm", Spec::new().u32("BM", 64).bool("ACC", true).f16(true)),
            key("unary", Spec::new().i32("OP", -3).f32("ALPHA", 0.25)),
            key("bn_stats", Spec::new()),
            key("optim_many", Spec::new().u32("MODE", 1).u32("J", 7).u32("NS", 1)),
            key("coop_gemm", Spec::new().u32("TILE", 2).f16(true).coop(true)),
        ];
        let file = encode(&keys, "id a");
        assert_eq!(decode(&file, "id a").as_deref(), Some(&keys[..]));
        assert_eq!(decode(&file, "id b"), None, "another adapter, driver, version or tag");
        let mut bad = file.clone();
        let mid = bad.len() / 2;
        bad[mid] ^= 1;
        assert_eq!(decode(&bad, "id a"), None);
    }

    #[test]
    fn unusable_lines_are_skipped() {
        let text = "gemm - BM=u64\nno_such_kernel - X=u1\ngemm - 1BAD=u1\ngemm - BM=q3\ngemm x\nunary - OP=i-2\n\
                    gemm - BM=u64\noptim_many - J=u99\noptim_many - J=u3 NS=u7\noptim_many - MODE=u1 J=u3 NS=u2\n";
        let file = encode_as(MAGIC, FORMAT, text.as_bytes(), "id");
        let got = decode(&file, "id").unwrap();
        let names: Vec<String> = got.iter().map(|k| format!("{} {}", k.name, k.spec.describe())).collect();
        assert_eq!(names, ["gemm BM=64", "unary OP=-2", "optim_many MODE=1 J=3 NS=2"]);
    }

    #[test]
    fn every_kernel_name_is_unique() {
        let mut names = HashSet::new();
        for d in crate::kernels::all_defs() {
            assert!(names.insert(d.name), "two kernels named {}", d.name);
        }
        for many in ["optim_many", "unscale_check_many", "norm_partials_many", "scale_many"] {
            assert!(!names.contains(many), "{many} is built per layout");
        }
    }

    #[test]
    fn prune_keeps_the_newest_lists() {
        let dir = folo_test_support::TestDir::new("gpu-kernel-lists");
        for i in 0..5 {
            let p = dir.join(format!("{PREFIX}{i}{SUFFIX}"));
            std::fs::write(&p, b"x").unwrap();
            let t = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000 + i);
            std::fs::File::options().write(true).open(&p).unwrap().set_modified(t).unwrap();
        }
        std::fs::write(dir.join("other.bin"), b"x").unwrap();
        prune(&dir, 3);
        let mut left: Vec<String> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        left.sort();
        assert_eq!(left, ["kernels-2.list", "kernels-3.list", "kernels-4.list", "other.bin"]);
    }
}
