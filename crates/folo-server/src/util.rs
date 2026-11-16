//! Small helpers: hashing, a seeded RNG, RFC 3339 formatting, folder names.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// FNV-1a 64-bit hash of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 finaliser: a good 64-bit mix of `x`.
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Deterministic noise in `[-1, 1)` for a tuple of keys.
pub fn noise(keys: &[u64]) -> f32 {
    let mut h = 0x5eed_u64;
    for &k in keys {
        h = mix64(h ^ k);
    }
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// A tiny seeded RNG (SplitMix64). Deterministic across platforms.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Rng {
        Rng(mix64(seed))
    }

    /// Next raw 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in `[0, n)`; `n` must be positive.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }
}

/// Seconds since the Unix epoch now.
pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Formats Unix seconds as RFC 3339 UTC with second precision, e.g. `2026-11-08T06:00:00Z`.
pub fn rfc3339(unix_secs: f64) -> String {
    let secs = unix_secs.floor() as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", tod / 3600, (tod / 60) % 60, tod % 60)
}

/// Formats Unix milliseconds as RFC 3339 UTC to the millisecond (`2026-11-08T12:00:00.250Z`).
pub fn rfc3339_millis(unix_ms: i64) -> String {
    let secs = rfc3339(unix_ms.div_euclid(1000) as f64);
    format!("{}.{:03}Z", &secs[..19], unix_ms.rem_euclid(1000))
}

/// Parses what [`rfc3339`] or [`rfc3339_millis`] wrote back to Unix milliseconds.
pub fn parse_rfc3339_millis(s: &str) -> Option<i64> {
    match s.len() {
        20 => Some(parse_rfc3339(s)? as i64 * 1000),
        24 if s.as_bytes()[19] == b'.' && s.ends_with('Z') => {
            let ms = s.get(20..23)?.parse::<i64>().ok()?;
            Some(parse_rfc3339(&format!("{}Z", &s[..19]))? as i64 * 1000 + ms)
        }
        _ => None,
    }
}

/// Parses `YYYY-MM-DDTHH:MM:SSZ` (as produced by [`rfc3339`]) back to Unix seconds.
pub fn parse_rfc3339(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[19] != b'Z' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, se) = (num(11..13)?, num(14..16)?, num(17..19)?);
    Some((days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se) as f64)
}

// Howard Hinnant's civil calendar algorithms.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The first [`short_id`] derived from `seed` for which `taken` is false. Ids have only 16 bits, so callers must
/// check every id that exists (and decide under the same lock that inserts the new one).
pub fn unique_id(prefix: &str, seed: u64, taken: impl Fn(&str) -> bool) -> String {
    let mut h = seed;
    loop {
        let id = short_id(prefix, h);
        if !taken(&id) {
            return id;
        }
        h = mix64(h);
    }
}

/// Short lowercase hex id with a prefix, e.g. `id("ds", h) == "ds_8f2c"`.
pub fn short_id(prefix: &str, h: u64) -> String {
    format!("{prefix}_{:04x}", (mix64(h) >> 48) as u16)
}

/// A folder name for a display name: lowercase ASCII letters, digits and `-` (at most 48 characters), or
/// `fallback` when nothing of the name is left. A name Windows reserves for a device (`nul`, `con`, `com1`, ...)
/// gets `-set` appended: a folder `nul` is the null device there, so a sample set named "NUL" was written into
/// nothing and its registration then failed with "does not exist" (stress s1-datasets-5).
pub fn slug(name: &str, fallback: &str) -> String {
    let mut s = String::new();
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
    }
    let s = s.chars().take(48).collect::<String>();
    let s = s.trim_end_matches('-');
    if s.is_empty() {
        fallback.into()
    } else if is_device_name(s) {
        format!("{s}-set")
    } else {
        s.into()
    }
}

/// `con`, `prn`, `aux`, `nul`, `com0`-`com9`, `lpt0`-`lpt9`: names Windows maps to devices in any folder.
fn is_device_name(s: &str) -> bool {
    match s.as_bytes() {
        b"con" | b"prn" | b"aux" | b"nul" => true,
        [b'c', b'o', b'm', d] | [b'l', b'p', b't', d] => d.is_ascii_digit(),
        _ => false,
    }
}

/// Creates a new folder `parent/base`, or `parent/base-2`, `-3`, ... when that is taken, and returns it.
/// `create_dir` claims the folder atomically, so concurrent callers with the same name never share one.
pub fn claim_dir(parent: &Path, base: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(parent)?;
    let mut dir = parent.join(base);
    let mut n = 1;
    loop {
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                n += 1;
                dir = parent.join(format!("{base}-{n}"));
            }
            Err(e) => return Err(e),
        }
    }
}

/// One record per id, of records loaded from folders as `(folder, id, record)`. A copy of a folder (a backup made
/// beside it) has its original's id: the record from the folder [`choose`] ranked highest is kept, else the one from
/// the folder named after its id, else the first by folder name, and the copies are logged as `what` and skipped, not
/// listed twice under one id (x3-server-9). `unknown`: the folders whose record could not be read, which may be copies
/// of any id. A folder [`choose`] cannot mark gives way to the next copy (f3-server-ci review round 2).
pub fn one_per_id<T>(
    what: &str,
    mut loaded: Vec<(PathBuf, String, T)>,
    unknown: &[PathBuf],
) -> Vec<(PathBuf, T)> {
    loaded.sort_by_cached_key(|(dir, id, _)| (std::cmp::Reverse(chosen_rank(dir)), copy_order(dir, id)));
    let mut at = std::collections::HashMap::new();
    let mut groups: Vec<(String, Vec<(PathBuf, T)>)> = Vec::new();
    for (dir, id, record) in loaded {
        let i = *at.entry(id.clone()).or_insert_with(|| {
            groups.push((id, Vec::new()));
            groups.len() - 1
        });
        groups[i].1.push((dir, record));
    }
    let mut out = Vec::new();
    for (id, group) in groups {
        let dirs: Vec<PathBuf> =
            group.iter().map(|(d, _)| d.clone()).chain(unknown.iter().cloned()).collect();
        let mut kept: Option<PathBuf> = None;
        for (dir, record) in group {
            if let Some(first) = &kept {
                tracing::warn!(
                    "skipping {}: {what} {id} is loaded from {} already",
                    dir.display(),
                    first.display()
                );
                continue;
            }
            match choose(&dir, &dirs) {
                Ok(()) => {
                    kept = Some(dir.clone());
                    out.push((dir, record));
                }
                Err(e) => tracing::warn!(
                    "skipping {what} {}: cannot mark it as the copy loaded: {e}",
                    dir.display()
                ),
            }
        }
    }
    out
}

/// The order copies of one id are taken in: the folder named after the id first, then by folder name.
fn copy_order(dir: &Path, id: &str) -> (bool, PathBuf) {
    (dir.file_name() != Some(std::ffi::OsStr::new(id)), dir.to_path_buf())
}

/// The file in a record's folder that ranks it among the copies of its record (see [`choose`]).
pub const CHOSEN: &str = ".folo-chosen";

/// The rank [`choose`] gave the folder `dir`, 0 without one.
pub fn chosen_rank(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join(CHOSEN)).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

/// Makes `dir`, the copy of a record about to be loaded, the one the next start takes first: when `others` (the other
/// copies of its id, and the folders whose record could not be read, which may be copies of it) hold one ranked as
/// high, it gets the rank above them all ([`CHOSEN`], written atomically). The work saved to a copy loaded because
/// the original did not load (it was damaged, unreadable or could not write its recovery back) is not shadowed by the
/// original at the next start, also when the original reads again (f3-server-ci review round 2). A record without
/// copies or unreadable neighbours writes nothing.
pub fn choose(dir: &Path, others: &[PathBuf]) -> std::io::Result<()> {
    match others.iter().filter(|o| o.as_path() != dir).map(|o| chosen_rank(o)).max() {
        Some(top) if chosen_rank(dir) <= top => {
            crate::index::write_atomic(&dir.join(CHOSEN), format!("{}\n", top + 1).as_bytes())
        }
        _ => Ok(()),
    }
}

/// The folders under `root` holding the JSON record `file`, grouped by id, in the order to try loading them, picked
/// before anything loads them, so the recovery a load does (closing what a previous process left active, written
/// back) never rewrites a backup (review round 2). `readable` says, without writing anything, whether a record's
/// bytes are one its loader takes (it parses, its version is known): in each group the readable records come first,
/// so a damaged original does not hide an intact backup (review round 3), then the copy [`choose`] ranked highest (the
/// one loaded last time a copy had to be picked), then the record written most often (a record's `saves`, 0 without
/// one), then in [`one_per_id`]'s order; [`load_first`] loads the first of a group that loads (review round 4). A
/// folder whose id cannot be read is a group of its own and is listed in [`Folders::unknown`]. A copy loaded in place
/// of one written more often counts its saves on from the group's [`most_saves`] (f3-server-ci review round 1).
pub fn folders_by_id(root: &Path, file: &str, readable: impl Fn(&[u8]) -> bool) -> Folders {
    let mut found: Vec<(PathBuf, String, bool, u64)> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|dir| dir.join(file).is_file())
        .map(|dir| {
            let bytes = std::fs::read(dir.join(file)).ok();
            let head = bytes.as_deref().and_then(|b| serde_json::from_slice::<Head>(b).ok());
            // No id: a key no id has (ids have no NUL), so it is kept and its loader logs it.
            let saves = head.as_ref().map_or(0, |h| h.saves);
            let id = head.map_or_else(|| format!("\0{}", dir.display()), |h| h.id);
            let ok = bytes.as_deref().is_some_and(&readable);
            (dir, id, ok, saves)
        })
        .collect();
    found.sort_by_cached_key(|(dir, id, ok, saves)| {
        (!*ok, std::cmp::Reverse(chosen_rank(dir)), std::cmp::Reverse(*saves), copy_order(dir, id))
    });
    let unknown =
        found.iter().filter(|(_, id, _, _)| id.starts_with('\0')).map(|(d, ..)| d.clone()).collect();
    let (mut at, mut groups) = (std::collections::HashMap::new(), Vec::<Vec<PathBuf>>::new());
    for (dir, id, _, _) in found {
        let i = *at.entry(id).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[i].push(dir);
    }
    Folders { groups, unknown }
}

/// What [`folders_by_id`] found.
pub struct Folders {
    /// The folders of each id, in the order to try loading them.
    pub groups: Vec<Vec<PathBuf>>,
    /// The folders whose record has no id that can be read (each also a group of its own, so its loader logs it):
    /// one may be a copy of any id, so [`choose`] ranks the copy it loads above them.
    pub unknown: Vec<PathBuf>,
}

/// The id and write count a record of [`folders_by_id`] starts with.
#[derive(serde::Deserialize)]
struct Head {
    id: String,
    #[serde(default)]
    saves: u64,
}

/// The most `saves` of the records `file` in `dirs` (a group of [`folders_by_id`]), 0 without any: where the write
/// count of the copy that is loaded goes on from, so whatever it saves next outranks every copy of the group, also one
/// that was written more often but did not load this time.
pub fn most_saves(dirs: &[PathBuf], file: &str) -> u64 {
    dirs.iter()
        .filter_map(|dir| std::fs::read(dir.join(file)).ok())
        .filter_map(|bytes| serde_json::from_slice::<Head>(&bytes).ok())
        .map(|h| h.saves)
        .max()
        .unwrap_or(0)
}

/// Of each group of [`folders_by_id`], the record of the first folder `load` loads (given the folder and its whole
/// group); why one does not is logged as `what`. A record that reads but does not load, such as one whose recovery
/// cannot be written back to a folder the server may not write, gives way to the next copy (review round 4), and so
/// does one [`choose`] cannot mark before it is loaded (f3-server-ci review round 2). The copies after the loaded one
/// are logged and left alone.
pub fn load_first<T, E: std::fmt::Display>(
    what: &str,
    folders: Folders,
    mut load: impl FnMut(&Path, &[PathBuf]) -> std::result::Result<T, E>,
) -> Vec<T> {
    let mut out = Vec::new();
    let Folders { groups, unknown } = folders;
    for dirs in groups {
        let mut loaded: Option<&PathBuf> = None;
        for dir in &dirs {
            if let Some(first) = loaded {
                tracing::warn!(
                    "skipping {}: {what} is loaded from {} already",
                    dir.display(),
                    first.display()
                );
                continue;
            }
            if !unknown.contains(dir) {
                let others: Vec<PathBuf> = dirs.iter().chain(&unknown).cloned().collect();
                if let Err(e) = choose(dir, &others) {
                    tracing::warn!(
                        "skipping {what} {}: cannot mark it as the copy loaded: {e}",
                        dir.display()
                    );
                    continue;
                }
            }
            match load(dir, &dirs) {
                Ok(record) => {
                    out.push(record);
                    loaded = Some(dir);
                }
                Err(err) => tracing::warn!("skipping {what} {}: {err}", dir.display()),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Copies are told apart by the id in their record, before any is loaded; a readable record goes before the
    /// original's damaged one (review round 3); a record without an id is a group of its own. The first copy of a
    /// group that loads is kept: one that reads but does not load gives way to the next (review round 4).
    #[test]
    fn folders_are_picked_by_id_before_loading() {
        let t = folo_test_support::TestDir::new("util-folders-by-id");
        let root = t.join("label");
        for (dir, body) in [
            ("lj_1", r#"{"id":"lj_1","version":1}"#),
            ("lj_1 - Copy", r#"{"id":"lj_1","version":1}"#),
            ("lj_2", r#"{"id":"lj_2","version":999}"#),
            ("lj_2 - Copy", r#"{"id":"lj_2","version":1}"#),
            ("lj_3", r#"{"id":"lj_3"}"#),
            ("lj_3 - Copy", r#"{"id":"lj_3"}"#),
            ("broken", "{"),
        ] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            std::fs::write(root.join(dir).join("job.json"), body).unwrap();
        }
        std::fs::create_dir_all(root.join("empty")).unwrap();
        let readable =
            |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).is_ok_and(|v| v["version"] == 1);
        let mut folders = folders_by_id(&root, "job.json", readable);
        folders.groups.sort();
        let names = |dirs: &[&str]| dirs.iter().map(|d| root.join(d)).collect::<Vec<_>>();
        let want = [
            names(&["broken"]),
            names(&["lj_1", "lj_1 - Copy"]),
            names(&["lj_2 - Copy", "lj_2"]),
            names(&["lj_3", "lj_3 - Copy"]),
        ];
        assert_eq!(folders.groups, want, "the readable records first, the original before its copies");
        assert_eq!(folders.unknown, names(&["broken"]));
        // lj_1 reads but does not load (as when its recovery cannot be written back): its copy is loaded instead.
        let load = |dir: &Path, _: &[PathBuf]| {
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            let fails = name == "lj_1" || name == "broken" || name.starts_with("lj_3");
            if fails { Err("cannot load") } else { Ok(name) }
        };
        assert_eq!(load_first("labeling job", folders, load), ["lj_1 - Copy", "lj_2 - Copy"]);
        // The copy loaded is the one the next start tries first, also when the original would load then.
        let mut again = folders_by_id(&root, "job.json", readable).groups;
        again.sort();
        assert_eq!(again[1], names(&["lj_1 - Copy", "lj_1"]));
        assert_eq!(
            load_first("labeling job", folders_by_id(&root, "job.json", readable), |d, _| load(d, &[])).len(),
            2
        );
    }

    /// A copy loaded because the original's record could not be read at all (no id to group it by) stays the one
    /// loaded once the original reads again, so the work saved to it is not shadowed; a record without copies or
    /// unreadable neighbours gets no rank written (f3-server-ci review round 2).
    #[test]
    fn the_copy_loaded_in_place_of_an_unreadable_original_stays_loaded() {
        let t = folo_test_support::TestDir::new("util-chosen");
        let root = t.join("bench");
        for (dir, id) in [("b_1", "b_1"), ("b_1 - Copy", "b_1"), ("b_2", "b_2")] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            std::fs::write(root.join(dir).join("bench.json"), format!(r#"{{"id":"{id}"}}"#)).unwrap();
        }
        let readable = |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).is_ok();
        let start = || {
            let mut loaded =
                load_first("benchmark", folders_by_id(&root, "bench.json", readable), |dir, _| {
                    let bytes = std::fs::read(dir.join("bench.json")).map_err(|e| e.to_string())?;
                    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| e.to_string())?;
                    Ok::<_, String>(dir.file_name().unwrap().to_string_lossy().into_owned())
                });
            loaded.sort();
            loaded
        };
        assert_eq!(start(), ["b_1", "b_2"]);
        assert!(!root.join("b_2").join(CHOSEN).exists(), "nothing to choose between: nothing written");
        let original = std::fs::read(root.join("b_1/bench.json")).unwrap();
        std::fs::write(root.join("b_1/bench.json"), "{").unwrap();
        assert_eq!(start(), ["b_1 - Copy", "b_2"], "the original does not read: its copy is loaded");
        std::fs::write(root.join("b_1/bench.json"), &original).unwrap();
        for _ in 0..2 {
            assert_eq!(start(), ["b_1 - Copy", "b_2"], "the original reads again: the copy stays");
        }
    }

    #[test]
    fn copies_of_a_folder_load_once() {
        let t = folo_test_support::TestDir::new("util-one-per-id");
        let p = |s: &str| {
            std::fs::create_dir_all(t.join(s)).unwrap();
            t.join(s)
        };
        let loaded = || {
            vec![
                (p("lj_1 - Copy"), "lj_1".to_string(), 1),
                (p("lj_2"), "lj_2".to_string(), 2),
                (p("lj_1"), "lj_1".to_string(), 3),
                (p("backup-b"), "lj_9".to_string(), 4),
                (p("backup-a"), "lj_9".to_string(), 5),
            ]
        };
        let kept = |unknown: &[PathBuf]| -> Vec<i32> {
            let mut r: Vec<i32> = one_per_id("job", loaded(), unknown).into_iter().map(|(_, r)| r).collect();
            r.sort();
            r
        };
        assert_eq!(kept(&[]), [2, 3, 5], "the folder named after the id, else the first by name");
        assert!(!t.join("lj_2").join(CHOSEN).exists(), "a run without copies is not marked");
        // A copy chosen once (its original did not read) stays chosen.
        std::fs::write(t.join("lj_1 - Copy").join(CHOSEN), "9\n").unwrap();
        assert_eq!(kept(&[p("unreadable")]), [1, 2, 5]);
        assert_eq!(chosen_rank(&t.join("lj_1 - Copy")), 9, "already above its copies: not rewritten");
        assert_eq!(chosen_rank(&t.join("lj_2")), 1, "ranked above the unreadable folder");
    }

    #[test]
    fn millisecond_times_round_trip_and_sort_as_time() {
        let t = 1_794_300_000_250;
        let s = rfc3339_millis(t);
        assert_eq!(s.len(), 24);
        assert!(s.ends_with(".250Z") && s.starts_with(&rfc3339(t as f64 / 1000.0)[..19]), "{s}");
        assert_eq!(parse_rfc3339_millis(&s), Some(t));
        assert_eq!(parse_rfc3339_millis(&rfc3339(1_794_300_000.0)), Some(1_794_300_000_000));
        assert!(
            rfc3339_millis(t) < rfc3339_millis(t + 1) && rfc3339_millis(t + 1) < rfc3339_millis(t + 1000)
        );
        assert_eq!(parse_rfc3339_millis("2026-11-10"), None);
        assert_eq!(parse_rfc3339_millis("2026-11-10T07:22:50,123Z"), None);
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("Shapes 1k!", "synthetic"), "shapes-1k");
        assert_eq!(slug("  --  ", "synthetic"), "synthetic");
        assert_eq!(slug("a__b", "x"), "a-b");
        // Cut at 48 characters, never ending on a dash.
        assert_eq!(slug(&format!("{} b", "a".repeat(47)), "x"), "a".repeat(47));
    }

    #[test]
    fn slugs_never_name_a_windows_device() {
        for (name, folder) in
            [("NUL", "nul-set"), ("con", "con-set"), ("Aux.txt", "aux-txt"), ("COM1", "com1-set")]
        {
            assert_eq!(slug(name, "synthetic"), folder, "{name}");
        }
        assert_eq!(slug("lpt9", "x"), "lpt9-set");
        assert_eq!(slug("console", "x"), "console");
        assert_eq!(slug("com10", "x"), "com10");
    }

    #[test]
    fn claimed_folders_get_a_suffix_when_taken() {
        let parent = folo_test_support::TestDir::new("util-claim-dir");
        let names = ["set", "set-2", "set-3"];
        for name in names {
            let dir = claim_dir(&parent.join("datasets"), "set").unwrap();
            assert!(dir.is_dir());
            assert_eq!(dir.file_name().unwrap(), name);
        }
    }

    #[test]
    fn rfc3339_known_values() {
        assert_eq!(rfc3339(0.0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_794_117_600.0), "2026-11-08T06:00:00Z");
        assert_eq!(parse_rfc3339("2026-11-08T06:00:00Z"), Some(1_794_117_600.0));
        assert_eq!(
            parse_rfc3339("2024-02-29T23:59:59Z").map(rfc3339).as_deref(),
            Some("2024-02-29T23:59:59Z")
        );
        assert_eq!(parse_rfc3339("nope"), None);
    }

    #[test]
    fn unique_id_skips_taken_ids() {
        let first = unique_id("m", 42, |_| false);
        let second = unique_id("m", 42, |id| id == first);
        assert_ne!(first, second);
        assert!(second.starts_with("m_") && second.len() == 6);
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.f32();
            assert_eq!(x, b.f32());
            assert!((0.0..1.0).contains(&x));
        }
        for k in 0..1000u64 {
            let n = noise(&[k, 3]);
            assert!((-1.0..1.0).contains(&n));
        }
    }
}
