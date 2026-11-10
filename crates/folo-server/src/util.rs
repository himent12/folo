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
/// `fallback` when nothing of the name is left.
pub fn slug(name: &str, fallback: &str) -> String {
    let mut s = String::new();
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
    }
    let s = s.trim_end_matches('-').chars().take(48).collect::<String>();
    if s.is_empty() { fallback.into() } else { s }
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

#[cfg(test)]
mod tests {
    use super::*;

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
