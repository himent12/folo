//! `folo weights`: published pretrained weights, downloaded from a GitHub release into a local cache, and the
//! name → path resolution behind `folo train --weights <name>` and `--model <name>` (docs/CLI.md, Weights).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::cli::{WeightsCommand, WeightsPullArgs};
use crate::output::{CliError, ExitCode, Out};

pub use folo_api::weights::{PUBLISHED, Published, SUMS, cache_dir, find};

/// Whether `name` is a published weights name.
pub fn is_published(name: &str) -> bool {
    find(name).is_some()
}

fn not_downloaded(p: &Published, path: &Path) -> CliError {
    CliError::new(
        ExitCode::NotFound,
        "weights_not_found",
        format!("weights `{}` are not downloaded ({} does not exist)", p.name, folo_api::json::path(path)),
    )
    .hint(format!("run `folo weights pull {}`", p.name))
}

/// A `.folo` path as given, or the cached file of a published name. An existing file always wins, so a local
/// file named like a published model is still used.
pub fn resolve(arg: &Path) -> Result<PathBuf, CliError> {
    if arg.is_file() {
        return Ok(arg.to_path_buf());
    }
    let Some(p) = arg.to_str().and_then(find) else {
        return Err(CliError::new(
            ExitCode::NotFound,
            "model_not_found",
            format!("{} does not exist", folo_api::json::path(arg)),
        )
        .hint(format!(
            "pass a .folo file, e.g. runs/<name>/weights/best.folo from `folo train`, or a published name ({})",
            folo_api::weights::names()
        )));
    };
    let path = cache_dir().join(p.asset);
    if path.is_file() { Ok(path) } else { Err(not_downloaded(p, &path)) }
}

pub fn run(cmd: WeightsCommand, out: Out) -> Result<(), CliError> {
    match cmd {
        WeightsCommand::List => {
            list(out);
            Ok(())
        }
        WeightsCommand::Pull(args) => pull(args, out),
    }
}

fn list(out: Out) {
    let dir = cache_dir();
    for p in PUBLISHED {
        let path = dir.join(p.asset);
        let cached = path.is_file();
        out.record(&json!({
            "name": p.name, "release": p.release, "asset": p.asset, "path": folo_api::json::path(&path),
            "cached": cached, "about": p.about,
        }));
        out.human(format!(
            "{:<18} {:<11} {}\n{:<18} {}",
            p.name,
            if cached { "downloaded" } else { "-" },
            p.about,
            "",
            folo_api::json::path(&path)
        ));
    }
}

/// Lowercase hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The hash `sums` (`sha256sum` output) lists for `file`.
pub fn expected_sha256(sums: &str, file: &str) -> Option<String> {
    sums.lines().find_map(|l| {
        let (hash, name) = l.trim().split_once(char::is_whitespace)?;
        (name.trim().trim_start_matches('*') == file).then(|| hash.to_ascii_lowercase())
    })
}

fn pull(args: WeightsPullArgs, out: Out) -> Result<(), CliError> {
    let Some(p) = find(&args.name) else {
        return Err(CliError::new(
            ExitCode::NotFound,
            "weights_not_found",
            format!("no published weights `{}`", args.name),
        )
        .hint("`folo weights list` shows the published names"));
    };
    let dir = cache_dir();
    let dest = dir.join(p.asset);
    let io =
        |what: &str, e: std::io::Error| CliError::new(ExitCode::Runtime, "io_error", format!("{what}: {e}"));
    if dest.is_file() && !args.force {
        let sha =
            sha256_file(&dest).map_err(|e| io(&format!("cannot read {}", folo_api::json::path(&dest)), e))?;
        let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        out.record(&json!({
            "event": "done", "name": p.name, "path": folo_api::json::path(&dest), "downloaded": false,
            "sha256": sha, "bytes": bytes,
        }));
        out.human(format!(
            "{} is already downloaded: {} (--force downloads it again)",
            p.name,
            folo_api::json::path(&dest)
        ));
        return Ok(());
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| io(&format!("cannot create {}", folo_api::json::path(&dir)), e))?;
    // Download into a fresh folder next to the cache, verify, then move into place.
    let tmp = dir.join(format!(".pull-{}-{}", p.name, std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)
        .map_err(|e| io(&format!("cannot create {}", folo_api::json::path(&tmp)), e))?;
    let result = (|| {
        out.human(format!("downloading {} from {} release {} ...", p.asset, args.repo, p.release));
        let status = Command::new("gh")
            .args(["release", "download", p.release, "--repo", &args.repo, "--pattern", p.asset, "--pattern", SUMS])
            .arg("--dir")
            .arg(&tmp)
            .status()
            .map_err(|e| {
                CliError::new(ExitCode::Runtime, "gh_failed", format!("could not run the GitHub CLI `gh`: {e}"))
                    .hint("install it from https://cli.github.com and run `gh auth login` (the repository is private)")
            })?;
        if !status.success() {
            return Err(CliError::new(
                ExitCode::Runtime,
                "gh_failed",
                format!("`gh release download {} --repo {}` failed ({status})", p.release, args.repo),
            )
            .hint("check `gh auth status` and that you can see the repository's releases"));
        }
        let file = tmp.join(p.asset);
        let sums =
            std::fs::read_to_string(tmp.join(SUMS)).map_err(|e| io(&format!("release has no {SUMS}"), e))?;
        let want = expected_sha256(&sums, p.asset).ok_or_else(|| {
            CliError::new(ExitCode::Runtime, "checksum_missing", format!("{SUMS} does not list {}", p.asset))
        })?;
        let got =
            sha256_file(&file).map_err(|e| io(&format!("cannot read {}", folo_api::json::path(&file)), e))?;
        if got != want {
            return Err(CliError::new(
                ExitCode::Runtime,
                "checksum_mismatch",
                format!("{}: sha256 {got}, but {SUMS} says {want}", p.asset),
            )
            .hint("download again; nothing was installed"));
        }
        // Check it is a FOLO file before installing it.
        folo_nn::io::ModelFile::open(&file)
            .map_err(|e| CliError::new(ExitCode::Runtime, "invalid_weights", format!("{}: {e}", p.asset)))?;
        let _ = std::fs::remove_file(&dest);
        std::fs::rename(&file, &dest)
            .map_err(|e| io(&format!("cannot move into {}", folo_api::json::path(&dest)), e))?;
        Ok(got)
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    let sha = result?;
    let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    out.record(&json!({
        "event": "done", "name": p.name, "path": folo_api::json::path(&dest), "downloaded": true, "sha256": sha,
        "bytes": bytes,
    }));
    out.human(format!(
        "{} → {} ({:.1} MB, sha256 {sha} verified)",
        p.name,
        folo_api::json::path(&dest),
        bytes as f64 / 1e6
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_lookup() {
        let sums = "abc123  folo-n-coco.folo\nDEF456 *folo-n-coco-f16.folo\n";
        assert_eq!(expected_sha256(sums, "folo-n-coco.folo").as_deref(), Some("abc123"));
        assert_eq!(expected_sha256(sums, "folo-n-coco-f16.folo").as_deref(), Some("def456"));
        assert_eq!(expected_sha256(sums, "folo-n.folo"), None);
    }

    #[test]
    fn sha256_known_value() {
        let dir = std::env::temp_dir().join(format!("folo-sha-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("abc.txt");
        std::fs::write(&f, b"abc").unwrap();
        assert_eq!(
            sha256_file(&f).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
