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
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let host = sys.available_memory();
    #[cfg(target_os = "linux")]
    let host = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|text| cgroup_free(&text, std::path::Path::new("/sys/fs/cgroup")))
        .map_or(host, |cg| host.min(cg));
    host
}

/// What the memory limits of the process's cgroups still allow, the tightest of them, given `/proc/self/cgroup`'s
/// text and the cgroup mount `root`: for the process's own cgroup and every ancestor up to the root, of both
/// hierarchies, the limit less the usage, the reclaimable inactive file cache not counted as used (as
/// `docker stats` counts it). cgroup v2: `memory.max` / `memory.current` / `inactive_file`; v1 (the `memory`
/// controller under `root/memory`): `memory.limit_in_bytes` / `memory.usage_in_bytes` / `total_inactive_file`.
/// `None` when no cgroup on the way has a limit.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_free(proc_cgroup: &str, root: &std::path::Path) -> Option<u64> {
    let mut tightest: Option<u64> = None;
    for line in proc_cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(_), Some(controllers), Some(path)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let (base, v2) = if controllers.is_empty() {
            (root.to_path_buf(), true)
        } else if controllers.split(',').any(|c| c == "memory") {
            (root.join("memory"), false)
        } else {
            continue;
        };
        let mut dir = base.join(path.trim_start_matches('/'));
        loop {
            if let Some(free) = cgroup_free_at(&dir, v2) {
                tightest = Some(tightest.map_or(free, |t| t.min(free)));
            }
            if dir == base || !dir.pop() || !dir.starts_with(&base) {
                break;
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
        (
            number("memory.limit_in_bytes").filter(|&l| l < 1 << 62)?,
            number("memory.usage_in_bytes")?,
            "total_inactive_file",
        )
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

    /// The tightest limit of the process's cgroup and its ancestors, v2 and v1, unlimited levels skipped.
    #[test]
    fn cgroup_limits_of_the_cgroup_and_its_ancestors() {
        let root = folo_test_support::TestDir::new("data-cgroup");
        let write = |rel: &str, name: &str, text: &str| {
            let d = root.join(rel);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(name), text).unwrap();
        };
        // v2: the scope is unlimited, its slice allows 1000 - (900 - 300 inactive) = 400, the root has no files.
        write("user.slice/run.scope", "memory.max", "max\n");
        write("user.slice/run.scope", "memory.current", "10\n");
        write("user.slice", "memory.max", "1000\n");
        write("user.slice", "memory.current", "900\n");
        write("user.slice", "memory.stat", "active_file 5\ninactive_file 300\n");
        assert_eq!(cgroup_free("0::/user.slice/run.scope\n", &root), Some(400));
        // A tighter scope wins.
        write("user.slice/run.scope", "memory.max", "50\n");
        assert_eq!(cgroup_free("0::/user.slice/run.scope\n", &root), Some(40));
        // v1 memory controller; "no limit" is a huge number.
        write("memory/docker/abc", "memory.limit_in_bytes", "2000\n");
        write("memory/docker/abc", "memory.usage_in_bytes", "500\n");
        write("memory/docker", "memory.limit_in_bytes", "9223372036854771712\n");
        write("memory/docker", "memory.usage_in_bytes", "1\n");
        assert_eq!(cgroup_free("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n", &root), Some(1500));
        // No limit anywhere.
        assert_eq!(cgroup_free("0::/\n", &root), None);
        assert_eq!(cgroup_free("", &root), None);
    }

    #[test]
    fn reads_memory_stat() {
        let stat = "active_file 7\ninactive_file_x 1\ninactive_file 4096\n";
        assert_eq!(stat_value(stat, "inactive_file"), Some(4096));
        assert_eq!(stat_value(stat, "total_inactive_file"), None);
    }
}
