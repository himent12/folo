//! `--cache ram`: every training image decoded once and kept resized (long side `imgsz`, as the transform resizes
//! it), so later epochs skip the JPEG decode and the resize (about 89% of a mosaic sample for 1080p JPEGs).
//!
//! The cache holds exactly what [`crate::augment::resize_long_side`] makes of the decoded image, plus the decoded
//! size. The transform's own resize of such an image is the identity (rows are copied), and its letterbox
//! geometry is computed from the original size ([`crate::augment::Fetched::orig_size`]), so a batch is the same, pixel
//! for pixel and box for box, with or without the cache. Images are cached the first time a worker loads them
//! (any order: the result does not depend on it); an image that cannot be read is not cached.

use crate::augment::resize_long_side;
use crate::dataset::Dataset;
use crate::error::{DataError, Result};
use crate::image::Sample;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// How [`crate::DataLoader`] keeps images between epochs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheMode {
    /// Decode every image every time.
    #[default]
    None,
    /// Keep every image decoded and resized in memory (`--cache ram`).
    Ram,
}

/// Part of the available memory a RAM cache may take.
pub const RAM_CACHE_SHARE: f64 = 0.8;

/// A cached image: the sample resized, and its decoded size.
type Entry = Arc<(Sample, [u32; 2])>;

/// Bytes the live RAM caches of this process reserved (each its [`ram_cache_bytes`], from creation to drop), so
/// two loaders created before either fills are not both admitted against the same free memory. A cache that has
/// filled counts twice (in its reservation and in the memory it took): a further cache is refused rather than
/// risked.
static RESERVED: AtomicU64 = AtomicU64::new(0);

/// Bytes reserved in a pool ([`RESERVED`]), given back on drop.
struct Reservation {
    pool: &'static AtomicU64,
    bytes: u64,
}

impl Reservation {
    /// Reserves `need` bytes of `pool` if they fit, with what is reserved already, in [`RAM_CACHE_SHARE`] of
    /// `free`; else the bytes reserved already.
    fn take(pool: &'static AtomicU64, need: u64, free: u64) -> std::result::Result<Self, u64> {
        let cap = (RAM_CACHE_SHARE * free as f64) as u64;
        let mut reserved = pool.load(Ordering::SeqCst);
        loop {
            let total = reserved.checked_add(need).filter(|&t| t <= cap).ok_or(reserved)?;
            match pool.compare_exchange_weak(reserved, total, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Ok(Reservation { pool, bytes: need }),
                Err(now) => reserved = now,
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.pool.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

/// Images decoded and resized once, by dataset index.
pub(crate) struct RamCache {
    imgsz: u32,
    slots: Vec<OnceLock<Entry>>,
    _reserved: Reservation,
}

/// The size of an image after [`resize_long_side`] to `long`.
fn resized(w: u32, h: u32, long: u32) -> (u64, u64) {
    let r = long as f64 / w.max(h) as f64;
    (((w as f64 * r).round() as u64).max(1), ((h as f64 * r).round() as u64).max(1))
}

/// Bytes a cached image takes besides its pixels and boxes: its slot, the shared entry and the entry's
/// reference counts.
const ENTRY_OVERHEAD: u64 = (size_of::<OnceLock<Entry>>()
    + size_of::<(Sample, [u32; 2])>()
    + 2 * size_of::<std::sync::atomic::AtomicUsize>()) as u64;

/// Bytes a RAM cache of `ds` at `imgsz` holds: every image resized (its size from the dataset when known, else a
/// full `imgsz x imgsz`), 3 bytes a pixel, plus its boxes and its entry.
pub fn ram_cache_bytes(ds: &dyn Dataset, imgsz: u32) -> u64 {
    (0..ds.len())
        .map(|i| {
            let (w, h) = ds.image_size(i).filter(|&(w, h)| w > 0 && h > 0).unwrap_or((imgsz, imgsz));
            let (rw, rh) = resized(w, h, imgsz);
            rw * rh * 3 + size_of_val(ds.boxes(i)) as u64 + ENTRY_OVERHEAD
        })
        .sum()
}

/// Memory this process can still take, in bytes: what the system has available, and on Linux no more than the
/// tightest memory limit of its cgroups ([`cgroup_free`]: a container, a systemd scope, any ancestor).
fn available_memory() -> u64 {
    let host = host_available();
    #[cfg(target_os = "linux")]
    let host =
        match (std::fs::read_to_string("/proc/self/cgroup"), std::fs::read_to_string("/proc/self/mountinfo"))
        {
            (Ok(cgroup), Ok(mountinfo)) => cgroup_free(&cgroup, &mountinfo).map_or(host, |cg| host.min(cg)),
            _ => host,
        };
    host
}

/// The memory the system has available, in bytes (0 when it can't say), as sysinfo's `available_memory` gives it.
/// Windows and Linux read it themselves so the crate does not build sysinfo (and the second `windows` crate it
/// brings) there: Windows' `GlobalMemoryStatusEx` `ullAvailPhys`, Linux's `/proc/meminfo` ([`meminfo_available`]).
#[cfg(windows)]
fn host_available() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: MEMORYSTATUSEX is plain integers, for which all zeroes is a valid value.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is a writable MEMORYSTATUSEX whose `dwLength` is set, as the call requires.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return 0;
    }
    status.ullAvailPhys
}

#[cfg(target_os = "linux")]
fn host_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").map_or(0, |text| meminfo_available(&text))
}

/// Elsewhere (macOS and the rest) sysinfo's figure.
#[cfg(not(any(windows, target_os = "linux")))]
fn host_available() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Available bytes from `/proc/meminfo`'s text: `MemAvailable`, or on kernels before 3.14, which lack it,
/// `MemFree + Buffers + Cached + SReclaimable - Shmem` (sysinfo's rule). Values are KiB though written "kB".
#[cfg(any(target_os = "linux", test))]
fn meminfo_available(meminfo: &str) -> u64 {
    let field = |key: &str| {
        meminfo.lines().find_map(|line| {
            let (k, v) = line.split_once(':')?;
            (k == key).then(|| v.trim_start().split(' ').next()?.parse::<u64>().ok())?
        })
    };
    if let Some(kib) = field("MemAvailable") {
        return kib.saturating_mul(1024);
    }
    let kib = |key| field(key).unwrap_or(0).saturating_mul(1024);
    kib("MemFree")
        .saturating_add(kib("Buffers"))
        .saturating_add(kib("Cached"))
        .saturating_add(kib("SReclaimable"))
        .saturating_sub(kib("Shmem"))
}

/// A mounted cgroup hierarchy that limits memory: cgroup v2, or v1 with the `memory` controller (alone or
/// co-mounted with others).
struct CgroupMount {
    /// The cgroup path the mount shows at its mount point (`/` unless a subtree is mounted).
    root: String,
    point: std::path::PathBuf,
    v2: bool,
}

/// `/proc/self/mountinfo`'s octal escapes (`\040` for a space) undone.
fn unescape(field: &str) -> String {
    let b = field.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\'
            && i + 3 < b.len()
            && (b'0'..=b'3').contains(&b[i + 1])
            && b[i + 2..i + 4].iter().all(|c| (b'0'..=b'7').contains(c))
        {
            out.push((b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0'));
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The memory-limiting cgroup hierarchies in `/proc/self/mountinfo`'s text.
fn cgroup_mounts(mountinfo: &str) -> Vec<CgroupMount> {
    mountinfo
        .lines()
        .filter_map(|line| {
            let (pre, post) = line.split_once(" - ")?;
            let pre: Vec<&str> = pre.split(' ').collect();
            let mut post = post.split(' ');
            let (fstype, _source, options) = (post.next()?, post.next()?, post.next().unwrap_or(""));
            let v2 = match fstype {
                "cgroup2" => true,
                "cgroup" if options.split(',').any(|o| o == "memory") => false,
                _ => return None,
            };
            Some(CgroupMount { root: unescape(pre.get(3)?), point: unescape(pre.get(4)?).into(), v2 })
        })
        .collect()
}

/// What the memory limits of the process's cgroups still allow, the tightest of them, given the text of
/// `/proc/self/cgroup` and `/proc/self/mountinfo`: for the process's cgroup in every memory-limiting hierarchy
/// (cgroup v2 `0::path`, v1 lines naming the `memory` controller), translated through each mount of that hierarchy
/// (its root and mount point, so a co-mounted controller or a mounted subtree resolves), the cgroup's directory
/// and each ancestor up to the mount point: the limit less the usage, the reclaimable inactive file cache not
/// counted as used (as `docker stats` counts it). v2: `memory.max` / `memory.current` / `inactive_file`; v1:
/// `memory.limit_in_bytes` / `memory.usage_in_bytes` / `total_inactive_file`. `None` when no level has a limit.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_free(proc_cgroup: &str, mountinfo: &str) -> Option<u64> {
    let mounts = cgroup_mounts(mountinfo);
    let mut tightest: Option<u64> = None;
    for line in proc_cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(_), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let v2 = controllers.is_empty();
        if !v2 && !controllers.split(',').any(|c| c == "memory") {
            continue;
        }
        for m in mounts.iter().filter(|m| m.v2 == v2) {
            // The cgroup's path below the mount's root; a cgroup outside the mounted subtree is not visible there.
            let rel = if m.root == "/" {
                path
            } else {
                match path.strip_prefix(m.root.as_str()) {
                    Some(r) if r.is_empty() || r.starts_with('/') => r,
                    _ => continue,
                }
            };
            let mut dir = m.point.join(rel.trim_start_matches('/'));
            loop {
                if let Some(free) = cgroup_free_at(&dir, v2) {
                    tightest = Some(tightest.map_or(free, |t| t.min(free)));
                }
                if dir == m.point || !dir.pop() || !dir.starts_with(&m.point) {
                    break;
                }
            }
        }
    }
    tightest
}

/// What the memory limit of the cgroup at `dir` still allows (see [`cgroup_free`]); `None` without a limit.
fn cgroup_free_at(dir: &std::path::Path, v2: bool) -> Option<u64> {
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
    let number = |name: &str| read(name)?.trim().parse::<u64>().ok();
    let (limit, usage, inactive) = if v2 {
        (number("memory.max")?, number("memory.current")?, "inactive_file")
    } else {
        // v1 reports "no limit" as a huge number (the largest page-aligned i64).
        let limit = number("memory.limit_in_bytes").filter(|&l| l < 1 << 62)?;
        (limit, number("memory.usage_in_bytes")?, "total_inactive_file")
    };
    let reclaimable = read("memory.stat").and_then(|s| stat_value(&s, inactive)).unwrap_or(0);
    Some(limit.saturating_sub(usage.saturating_sub(reclaimable)))
}

/// The value of `key` in a cgroup `memory.stat` (`key value` lines).
fn stat_value(stat: &str, key: &str) -> Option<u64> {
    stat.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix(' ')?.trim().parse().ok())
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

impl RamCache {
    /// A cache for `ds` at `imgsz`, after checking that it fits, with what the other live caches of this process
    /// reserved, in [`RAM_CACHE_SHARE`] of the memory available to this process (an error saying how much it
    /// needs otherwise). Its size stays reserved until it is dropped.
    pub fn new(ds: &dyn Dataset, imgsz: u32) -> Result<Self> {
        let need = ram_cache_bytes(ds, imgsz);
        let free = available_memory();
        let reserved = Reservation::take(&RESERVED, need, free).map_err(|others| {
            let held = if others > 0 {
                format!(" (other RAM caches of this process hold {})", gib(others))
            } else {
                String::new()
            };
            DataError::Config(format!(
                "--cache ram needs {} for {} images at imgsz {imgsz}, more than {:.0}% of the {} of memory available \
                 to this process{held}; train without --cache, with a smaller --imgsz, or on fewer images",
                gib(need),
                ds.len(),
                RAM_CACHE_SHARE * 100.0,
                gib(free)
            ))
        })?;
        Ok(RamCache { imgsz, slots: (0..ds.len()).map(|_| OnceLock::new()).collect(), _reserved: reserved })
    }

    /// Sample `i` resized, and its decoded size: from the cache, else loaded with `load` and kept.
    pub fn get(&self, i: usize, load: impl FnOnce() -> Result<Sample>) -> Result<(Sample, [u32; 2])> {
        let slot = &self.slots[i];
        let entry = match slot.get() {
            Some(e) => e.clone(),
            None => {
                let s = load()?;
                let orig = [s.image.width, s.image.height];
                // Workers racing on one image may each decode it; only the one whose `get_or_init` wins resizes
                // it, and the others take the kept entry (the same pixels) and drop their decode.
                slot.get_or_init(|| Arc::new((resize_long_side(s, self.imgsz), orig))).clone()
            }
        };
        Ok((entry.0.clone(), entry.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::BBox;
    use crate::dataset::{InMemoryDataset, Split};
    use crate::image::Image;

    fn sample(w: u32, h: u32) -> Sample {
        let data = (0..w * h * 3).map(|k| (k * 7 % 251) as u8).collect();
        Sample { image: Image { width: w, height: h, data }, boxes: vec![BBox::new(0, 0.5, 0.5, 0.2, 0.3)] }
    }

    /// The cache keeps what `resize_long_side` makes of the decoded image (long side `imgsz`, for images larger
    /// and smaller than it) with the decoded size, and serves it again without loading.
    #[test]
    fn keeps_the_resized_sample_and_its_decoded_size() {
        let ds = InMemoryDataset {
            samples: vec![sample(200, 120), sample(30, 50)],
            classes: vec!["a".into()],
            split: Split::Train,
        };
        let cache = RamCache::new(&ds, 64).unwrap();
        for (i, size, orig) in [(0, [64, 38], [200, 120]), (1, [38, 64], [30, 50])] {
            let (s, o) = cache.get(i, || ds.get(i)).unwrap();
            assert_eq!(([s.image.width, s.image.height], o), (size, orig), "image {i}");
            assert_eq!(s, resize_long_side(ds.get(i).unwrap(), 64));
            // What the slot holds, not only what `get` returns: the resized sample (the memory estimate's size).
            let kept = cache.slots[i].get().expect("filled");
            assert_eq!((&kept.0, kept.1), (&s, orig), "image {i} is stored resized");
            let (again, o) = cache.get(i, || panic!("image {i} loaded twice")).unwrap();
            assert_eq!((again, o), (s, orig));
        }
    }

    #[test]
    fn reservations_add_up_until_dropped() {
        static POOL: AtomicU64 = AtomicU64::new(0);
        let free = 1000;
        let a = Reservation::take(&POOL, 500, free).unwrap();
        assert_eq!(Reservation::take(&POOL, 500, free).err(), Some(500), "1000 bytes exceed 80% of 1000");
        let b = Reservation::take(&POOL, 300, free).unwrap();
        drop(a);
        let c = Reservation::take(&POOL, 500, free).unwrap();
        drop((b, c));
        assert_eq!(POOL.load(Ordering::SeqCst), 0);
    }

    /// The tightest limit of the process's cgroup and its ancestors, found through the mounts: v2, v1 with memory
    /// alone or co-mounted with cpu, a mounted subtree (mount root not `/`); unlimited levels skipped.
    #[test]
    fn cgroup_limits_of_the_cgroup_and_its_ancestors() {
        let root = folo_test_support::TestDir::new("data-cgroup");
        let write = |rel: &str, name: &str, text: &str| {
            let d = root.join(rel);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(name), text).unwrap();
        };
        let at = |rel: &str| root.join(rel).to_string_lossy().replace('\\', "/").replace(' ', "\\040");
        let mount = |root_in: &str, point: &str, fs: &str, opts: &str| {
            format!("30 23 0:26 {root_in} {} rw,nosuid shared:4 - {fs} cgroup {opts}\n", at(point))
        };
        // v2 at "v2": the scope is unlimited, its slice allows 1000 - (900 - 300 inactive) = 400.
        let v2 = mount("/", "v2", "cgroup2", "rw,nsdelegate");
        write("v2/user.slice/run.scope", "memory.max", "max\n");
        write("v2/user.slice/run.scope", "memory.current", "10\n");
        write("v2/user.slice", "memory.max", "1000\n");
        write("v2/user.slice", "memory.current", "900\n");
        write("v2/user.slice", "memory.stat", "active_file 5\ninactive_file 300\n");
        assert_eq!(cgroup_free("0::/user.slice/run.scope\n", &v2), Some(400));
        // A tighter scope wins.
        write("v2/user.slice/run.scope", "memory.max", "50\n");
        assert_eq!(cgroup_free("0::/user.slice/run.scope\n", &v2), Some(40));
        // No mount, no limit; a cgroup line without a mount of its kind is skipped.
        assert_eq!(cgroup_free("0::/user.slice/run.scope\n", ""), None);
        assert_eq!(cgroup_free("4:memory:/user.slice\n", &v2), None);
        // v1 memory co-mounted with cpu at "cpu,memory", no `memory` link; "no limit" is a huge number.
        let v1 = mount("/", "cpu,memory", "cgroup", "rw,cpu,memory");
        write("cpu,memory/docker/abc", "memory.limit_in_bytes", "2000\n");
        write("cpu,memory/docker/abc", "memory.usage_in_bytes", "500\n");
        write("cpu,memory/docker", "memory.limit_in_bytes", "9223372036854771712\n");
        write("cpu,memory/docker", "memory.usage_in_bytes", "1\n");
        assert_eq!(cgroup_free("12:cpu,memory:/docker/abc\n", &v1), Some(1500));
        // Hybrid: both hierarchies count, the tightest wins.
        assert_eq!(
            cgroup_free("12:cpu,memory:/docker/abc\n0::/user.slice/run.scope\n", &(v1.clone() + &v2)),
            Some(40)
        );
        // A subtree mounted with root /tenant at "sub": /tenant/job is "sub/job", and the mount point bounds the walk.
        let sub = mount("/tenant", "sub dir", "cgroup2", "rw");
        write("sub dir/job", "memory.max", "300\n");
        write("sub dir/job", "memory.current", "100\n");
        write("sub dir", "memory.max", "5000\n");
        write("sub dir", "memory.current", "0\n");
        assert_eq!(cgroup_free("0::/tenant/job\n", &sub), Some(200));
        assert_eq!(cgroup_free("0::/elsewhere/job\n", &sub), None, "outside the mounted subtree");
        assert_eq!(cgroup_free("", &sub), None);
    }

    /// Escapes up to `\377` decode, anything else (`\400`, a short or non-octal one) stays as written.
    #[test]
    fn unescapes_mountinfo_fields() {
        assert_eq!(unescape(r"/sys/fs/cgroup/a\040b"), "/sys/fs/cgroup/a b");
        assert_eq!(unescape(r"x\134y\011"), "x\u{5c}y\t");
        assert_eq!(unescape(r"\400\777\08\12"), r"\400\777\08\12");
    }

    /// `MemAvailable` when the kernel writes it, else the pre-3.14 estimate; KiB to bytes; nothing readable is 0.
    #[test]
    fn reads_meminfo() {
        let new = "MemTotal:       32000000 kB\nMemFree:          100 kB\nMemAvailable:   2000000 kB\nCached: 5 kB\n";
        assert_eq!(meminfo_available(new), 2_000_000 * 1024);
        let old = "MemFree: 100 kB\nBuffers: 20 kB\nCached: 300 kB\nShmem: 50 kB\nSReclaimable: 30 kB\n";
        assert_eq!(meminfo_available(old), (100 + 20 + 300 + 30 - 50) * 1024);
        assert_eq!(meminfo_available("MemAvailable: x kB\nMemFree: 7 kB\n"), 7 * 1024);
        assert_eq!(meminfo_available(""), 0);
    }

    /// The host figure is there on the platforms the crate reads it on itself.
    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn host_has_memory_available() {
        assert!(host_available() > 0);
    }

    #[test]
    fn reads_memory_stat() {
        let stat = "active_file 7\ninactive_file_x 1\ninactive_file 4096\n";
        assert_eq!(stat_value(stat, "inactive_file"), Some(4096));
        assert_eq!(stat_value(stat, "total_inactive_file"), None);
    }
}
