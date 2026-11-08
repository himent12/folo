//! Bakes build metadata into `folo --version`: `FOLO_GIT_SHA` (12-char commit, from `$FOLO_GIT_SHA` or
//! `git rev-parse`, else `unknown`) and `FOLO_BUILD_DATE` (UTC `YYYY-MM-DD`, from `$SOURCE_DATE_EPOCH` or the clock).

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=FOLO_GIT_SHA");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let sha = std::env::var("FOLO_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(12).collect())
        .or_else(|| git(&["rev-parse", "--short=12", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=FOLO_GIT_SHA={sha}");
    watch_git_head();

    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64));
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    println!("cargo:rustc-env=FOLO_BUILD_DATE={y:04}-{m:02}-{d:02}");
}

/// Runs git in the crate directory; `None` when git is missing, this is not a checkout, or the output is empty.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

/// Re-runs this script when HEAD moves: HEAD itself, the branch ref it points to and packed-refs. Only paths that
/// exist are emitted (a missing one would make cargo re-run the script on every build).
fn watch_git_head() {
    let mut paths = vec!["HEAD".to_string(), "packed-refs".to_string()];
    paths.extend(git(&["symbolic-ref", "-q", "HEAD"]));
    for p in paths {
        if let Some(path) = git(&["rev-parse", "--git-path", &p])
            && Path::new(&path).exists()
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}

/// Days since 1970-01-01 to a proleptic Gregorian (year, month, day) (Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}
