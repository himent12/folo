//! `folo dataset synth`: synthetic shapes dataset generation with atomic sibling-staging for new output folders
//! and clean-on-failure rollback for pre-existing empty folders.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use folo_data::synthetic::{SyntheticConfig, WrittenDataset, shapes_for, write_yolo_with_progress};
use folo_data::{DataError, ImageFormat};
use serde_json::json;

use crate::cli::{SynthArgs, SynthImageFormat};
use crate::output::{CliError, ExitCode, Out};

use super::{data_error, ensure_empty_output, output_data_error, output_exists, splits_json};

/// Throttled progress output: `--json` records, or a `\r` line on an interactive stderr.
struct Progress {
    out: Out,
    total: usize,
    tty: bool,
    last: Mutex<Option<Instant>>,
}

impl Progress {
    fn update(&self, done: usize) {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let due = last.is_none_or(|t| now.duration_since(t) >= Duration::from_millis(200));
        if !(due || done == self.total) {
            return;
        }
        *last = Some(now);
        self.out.record(&json!({ "event": "progress", "done": done, "total": self.total }));
        if !self.out.json && self.tty {
            let pct = 100.0 * done as f64 / self.total.max(1) as f64;
            let mut e = std::io::stderr().lock();
            let _ = write!(e, "\rgenerating {done}/{} images ({pct:.0}%)", self.total);
            if done == self.total {
                let _ = write!(e, "\r{:60}\r", "");
            }
            let _ = e.flush();
        }
    }
}

/// Stages a new dataset in a short sibling directory (`.folo-synth-<pid>-<n>.tmp`) inside the output's parent and
/// atomically moves it to `target` once complete without replacing any existing directory or file. On any error
/// before [`SynthStage::publish`], `Drop` deletes the staging folder and any empty parent folders this call created,
/// leaving no partial output behind.
struct SynthStage {
    /// Target path with any trailing `/` or `/.` normalized (`parent.join(file_name)`), for reliable rename.
    target: PathBuf,
    stage: PathBuf,
    created_parents: Vec<PathBuf>,
    published: bool,
}

impl SynthStage {
    fn new(raw_target: &Path) -> Result<Self, CliError> {
        let parent = raw_target.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let name = raw_target.file_name().ok_or_else(|| {
            CliError::new(ExitCode::Usage, "usage", "output directory path must name a folder")
        })?;
        // Normalizes trailing `/` and `/.` so POSIX rename targets the folder name itself.
        let target = if raw_target.parent().is_some_and(|p| !p.as_os_str().is_empty()) {
            parent.join(name)
        } else {
            PathBuf::from(name)
        };
        let created_parents = create_parents_tracked(parent, &target)?;
        let target_name = name.to_string_lossy().to_lowercase();
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stage = loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let candidate_name = format!(".folo-synth-{}-{n}.tmp", std::process::id());
            if candidate_name.to_lowercase() == target_name {
                continue;
            }
            let candidate = parent.join(&candidate_name);
            match std::fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    remove_empty_parents(&created_parents);
                    return Err(output_data_error(
                        DataError::Io { path: target.clone(), source: e },
                        &target,
                    ));
                }
            }
        };
        Ok(SynthStage { target, stage, created_parents, published: false })
    }

    fn path(&self) -> &Path {
        &self.stage
    }

    /// Atomically publishes the finished staging directory at `self.target`, refusing to overwrite any file or
    /// directory (including an empty directory created concurrently).
    fn publish(mut self) -> Result<PathBuf, CliError> {
        if std::fs::symlink_metadata(&self.target).is_ok() {
            return Err(output_exists(
                &self.target,
                self.target.join(folo_data::unfinished::MARKER).exists(),
            ));
        }
        let mut tries = 0;
        let res = loop {
            match rename_dir_noreplace(&self.stage, &self.target) {
                Ok(()) => break Ok(()),
                // On Windows, a search indexer or antivirus scanner can briefly hold a freshly written file open,
                // causing directory rename to return PermissionDenied or ERROR_SHARING_VIOLATION (32).
                Err(e)
                    if tries < 10
                        && std::fs::symlink_metadata(&self.target).is_err()
                        && (e.kind() == std::io::ErrorKind::PermissionDenied
                            || (cfg!(windows) && e.raw_os_error() == Some(32))) =>
                {
                    tries += 1;
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => break Err(e),
            }
        };
        match res {
            Ok(()) => {
                self.published = true;
                Ok(self.target.clone())
            }
            Err(e)
                if std::fs::symlink_metadata(&self.target).is_ok()
                    || matches!(
                        e.kind(),
                        std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::DirectoryNotEmpty
                    ) =>
            {
                Err(output_exists(&self.target, self.target.join(folo_data::unfinished::MARKER).exists()))
            }
            Err(e) => {
                Err(output_data_error(DataError::Io { path: self.target.clone(), source: e }, &self.target))
            }
        }
    }
}

impl Drop for SynthStage {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_dir_all(&self.stage);
            remove_empty_parents(&self.created_parents);
        }
    }
}

/// Renames directory `from` to `to` without replacing an existing `to` (even an empty directory).
///
/// On Linux, uses `renameat2(..., RENAME_NOREPLACE)` so a concurrently created empty directory at `to` is
/// atomically rejected with `EEXIST` instead of being silently replaced; on filesystems without `RENAME_NOREPLACE`
/// (`EINVAL` / `ENOSYS`), falls back to checking `symlink_metadata` + `std::fs::rename`. On Windows,
/// `folo_nn::io::publish_new` calls `MoveFileExW(..., 0)` without `MOVEFILE_REPLACE_EXISTING`, which atomically
/// rejects any existing file or directory at `to` (and supports long `\\?\` paths).
fn rename_dir_noreplace(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        folo_nn::io::publish_new(from, to)
    }
    #[cfg(target_os = "linux")]
    {
        use std::ffi::{CString, c_char, c_int, c_uint};
        use std::os::unix::ffi::OsStrExt;
        unsafe extern "C" {
            fn renameat2(
                olddirfd: c_int,
                oldpath: *const c_char,
                newdirfd: c_int,
                newpath: *const c_char,
                flags: c_uint,
            ) -> c_int;
        }
        const AT_FDCWD: c_int = -100;
        const RENAME_NOREPLACE: c_uint = 1;
        let to_c = |p: &Path| CString::new(p.as_os_str().as_bytes()).map_err(std::io::Error::from);
        let (old, new) = (to_c(from)?, to_c(to)?);
        // SAFETY: both pointers are NUL-terminated C strings alive across the call; return value is checked.
        if unsafe { renameat2(AT_FDCWD, old.as_ptr(), AT_FDCWD, new.as_ptr(), RENAME_NOREPLACE) } == 0 {
            return Ok(());
        }
        let err = std::io::Error::last_os_error();
        if matches!(err.kind(), std::io::ErrorKind::Unsupported | std::io::ErrorKind::InvalidInput) {
            if std::fs::symlink_metadata(to).is_ok() {
                return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
            }
            return std::fs::rename(from, to);
        }
        Err(err)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        if std::fs::symlink_metadata(to).is_ok() {
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }
        std::fs::rename(from, to)
    }
}

/// Cleans up partial output written into a pre-existing empty directory (such as `.`) if `synth` fails.
///
/// MUST only be constructed *after* [`folo_data::unfinished::claim`] succeeds, so a losing writer never deletes
/// another writer's marker or dataset files. Removes generated contents before removing `.folo-unfinished` so no
/// other writer can claim the directory mid-rollback.
struct CleanEmptyDirGuard {
    dir: PathBuf,
    finished: bool,
}

impl CleanEmptyDirGuard {
    fn new(dir: &Path) -> Self {
        CleanEmptyDirGuard { dir: dir.to_path_buf(), finished: false }
    }

    fn finish(mut self) {
        self.finished = true;
    }
}

impl Drop for CleanEmptyDirGuard {
    fn drop(&mut self) {
        if !self.finished {
            for sub in ["images", "labels"] {
                let _ = std::fs::remove_dir_all(self.dir.join(sub));
            }
            let _ = std::fs::remove_file(self.dir.join("data.yaml"));
            // Remove the marker last so the folder stays marked unfinished until cleanup finishes.
            let _ = std::fs::remove_file(self.dir.join(folo_data::unfinished::MARKER));
        }
    }
}

/// Creates missing ancestor directories of `parent` and returns the ones created (shallowest first) so they can be
/// rolled back if empty on failure.
fn create_parents_tracked(parent: &Path, target: &Path) -> Result<Vec<PathBuf>, CliError> {
    if parent.is_dir() {
        return Ok(Vec::new());
    }
    let mut missing = Vec::new();
    let mut cur = parent;
    while !cur.as_os_str().is_empty() && !cur.exists() {
        missing.push(cur.to_path_buf());
        match cur.parent() {
            Some(p) if p != cur => cur = p,
            _ => break,
        }
    }
    missing.reverse();
    let mut created = Vec::new();
    for dir in missing {
        match std::fs::create_dir(&dir) {
            Ok(()) => created.push(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && dir.is_dir() => {}
            Err(e) => {
                remove_empty_parents(&created);
                return Err(output_data_error(
                    DataError::Io { path: target.to_path_buf(), source: e },
                    target,
                ));
            }
        }
    }
    Ok(created)
}

fn remove_empty_parents(created: &[PathBuf]) {
    for dir in created.iter().rev() {
        let _ = std::fs::remove_dir(dir);
    }
}

/// Rewrites any `DataError::Io` path rooted inside `stage` to point at `out_dir`, so internal staging directory
/// names never leak into error messages or confuse `output_data_error`'s output-vs-input hint.
fn remap_stage_error(e: DataError, stage: &Path, out_dir: &Path) -> DataError {
    match e {
        DataError::Io { path, source } => {
            let mapped = match path.strip_prefix(stage) {
                Ok(rel) if rel.as_os_str().is_empty() => out_dir.to_path_buf(),
                Ok(rel) => out_dir.join(rel),
                Err(_) => path,
            };
            DataError::Io { path: mapped, source }
        }
        other => other,
    }
}

/// Runs `writer` against `out_dir`: stages into a sibling directory and atomically renames when `out_dir` does not
/// yet exist, or writes directly into a pre-existing empty directory (like `.`) with rollback armed *only after*
/// `claim` succeeds.
fn run_synth_into(
    out_dir: &Path,
    writer: impl FnOnce(folo_data::unfinished::Claim) -> Result<WrittenDataset, DataError>,
) -> Result<WrittenDataset, CliError> {
    let pre_existing_dir = out_dir.is_dir();
    let can_stage = !pre_existing_dir
        && out_dir.file_name().is_some()
        && !matches!(
            out_dir.components().next_back(),
            Some(
                std::path::Component::CurDir
                    | std::path::Component::ParentDir
                    | std::path::Component::RootDir
            )
        );
    if can_stage {
        let stage = SynthStage::new(out_dir)?;
        let stage_path = stage.path().to_path_buf();
        let claim = folo_data::unfinished::claim(&stage_path)
            .map_err(|e| output_data_error(remap_stage_error(e, &stage_path, out_dir), out_dir))?;
        let mut written = writer(claim)
            .map_err(|e| output_data_error(remap_stage_error(e, &stage_path, out_dir), out_dir))?;
        let final_root = stage.publish()?;
        written.yaml = final_root.join("data.yaml");
        written.root = final_root;
        Ok(written)
    } else {
        // Claim FIRST: if another writer already claimed `out_dir`, return immediately without arming rollback
        // so we never touch the winner's marker or files.
        let claim = folo_data::unfinished::claim(out_dir).map_err(|e| output_data_error(e, out_dir))?;
        let guard = CleanEmptyDirGuard::new(out_dir);
        let written = writer(claim).map_err(|e| output_data_error(e, out_dir))?;
        guard.finish();
        Ok(written)
    }
}

pub(super) fn synth(args: SynthArgs, out: Out) -> Result<(), CliError> {
    if args.out_dir.as_os_str().is_empty() {
        return Err(CliError::new(ExitCode::Usage, "usage", "output directory path must not be empty")
            .hint("pass a target folder path, e.g. `folo dataset synth ./shapes`"));
    }
    if !crate::cli::bounds::SYNTH_IMAGES.contains_u64(args.images) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--images must be in 1..=1000000"));
    }
    if !crate::cli::bounds::SYNTH_IMGSZ.contains_u64(u64::from(args.imgsz)) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--imgsz must be in 32..=8192"));
    }
    let names: Vec<String> = args.classes.iter().map(|c| c.trim().to_string()).collect();
    let classes = shapes_for(&names).map_err(|e| data_error(e, &args.out_dir))?;
    let cfg = SyntheticConfig {
        images: args.images as usize,
        classes: classes.clone(),
        names: names.clone(),
        imgsz: args.imgsz,
        seed: args.seed,
        val_fraction: args.val_frac,
        vary_aspect: args.vary_aspect,
        ..Default::default()
    };
    cfg.check().map_err(|e| data_error(e, &args.out_dir))?;
    ensure_empty_output(&args.out_dir)?;
    let format = match args.image_format {
        SynthImageFormat::Jpeg => ImageFormat::Jpeg(90),
        SynthImageFormat::Png => ImageFormat::Png,
    };
    let t0 = Instant::now();
    let progress =
        Progress { out, total: cfg.images, tty: std::io::stderr().is_terminal(), last: Mutex::new(None) };
    let written = run_synth_into(&args.out_dir, |claim| {
        write_yolo_with_progress(&cfg, claim, format, |d| progress.update(d))
    })?;
    let secs = t0.elapsed().as_secs_f64();
    let [train, val] = written.splits;
    out.record(&json!({
        "event": "done",
        "path": folo_api::json::path(&written.root),
        "yaml": folo_api::json::path(&written.yaml),
        "images": cfg.images,
        "splits": splits_json([train, val, 0]),
        "boxes": written.boxes,
        "classes": names,
        "shapes": classes,
        "imgsz": cfg.imgsz,
        "seed": cfg.seed,
        "seconds": secs,
    }));
    out.human(format!(
        "wrote {} images ({train} train / {val} val, {} boxes) to {} in {secs:.1}s",
        cfg.images,
        written.boxes,
        folo_api::json::path(&written.root)
    ));
    let mapping: Vec<String> = names
        .iter()
        .zip(&classes)
        .map(|(n, s)| if n.eq_ignore_ascii_case(s) { n.clone() } else { format!("{n} ({s})") })
        .collect();
    out.human(format!("  classes  {}", mapping.join(", ")));
    out.human(format!("  yaml     {}", folo_api::json::path(&written.yaml)));
    out.human(folo_api::json::cli_command(
        "next: folo dataset stats {dataset}",
        &[("dataset", &args.out_dir)],
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::*;

    fn dir_entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn failed_synth_into_new_dir_leaves_no_partial_output_or_staging_dir() {
        let root = TestDir::new("cli-synth-rollback-new");
        let target = root.join("nested/child/shapes");
        let err = run_synth_into(&target, |claim| {
            std::fs::create_dir_all(claim.dir().join("images/train")).unwrap();
            std::fs::write(claim.dir().join("images/train/000000.png"), b"half").unwrap();
            Err(DataError::Io {
                path: claim.dir().to_path_buf(),
                source: std::io::Error::other("simulated disk full"),
            })
        })
        .unwrap_err();
        assert_eq!(err.code, "io_error");
        assert!(!err.message.contains(".folo-synth-"), "staging path leaked: {}", err.message);
        assert!(!target.exists(), "target must not exist after failure");
        assert!(!root.join("nested").exists(), "newly created empty parents must be cleaned up");
        assert!(dir_entries(root.path()).is_empty(), "leftovers: {:?}", dir_entries(root.path()));
    }

    #[test]
    fn failed_synth_into_existing_empty_dir_restores_empty_dir() {
        let root = TestDir::new("cli-synth-rollback-empty");
        let target = root.join("existing_empty");
        std::fs::create_dir(&target).unwrap();
        let err = run_synth_into(&target, |claim| {
            std::fs::create_dir_all(claim.dir().join("images/train")).unwrap();
            std::fs::create_dir_all(claim.dir().join("labels/train")).unwrap();
            std::fs::write(claim.dir().join("images/train/000000.png"), b"half").unwrap();
            std::fs::write(claim.dir().join("data.yaml"), "partial").unwrap();
            Err(DataError::Io {
                path: claim.dir().to_path_buf(),
                source: std::io::Error::other("simulated disk full"),
            })
        })
        .unwrap_err();
        assert_eq!(err.code, "io_error");
        assert!(target.is_dir(), "pre-existing folder itself stays");
        assert!(
            dir_entries(&target).is_empty(),
            "pre-existing folder must be empty after rollback, got: {:?}",
            dir_entries(&target)
        );
    }

    #[test]
    fn failed_claim_on_claimed_dir_never_deletes_winners_dataset() {
        let root = TestDir::new("cli-synth-claim-race");
        let target = root.join("shared");
        std::fs::create_dir(&target).unwrap();
        // Simulate another writer that already won `claim(&target)` and is actively writing files.
        let _winner_claim = folo_data::unfinished::claim(&target).unwrap();
        std::fs::create_dir_all(target.join("images/train")).unwrap();
        std::fs::write(target.join("images/train/000000.png"), b"winner").unwrap();
        std::fs::write(target.join("data.yaml"), "winner: true\n").unwrap();

        let err = run_synth_into(&target, |_claim| unreachable!("loser must fail at claim")).unwrap_err();
        assert_eq!(err.code, "output_exists");
        assert!(target.join(folo_data::unfinished::MARKER).is_file(), "winner's marker must stay");
        assert_eq!(std::fs::read(target.join("images/train/000000.png")).unwrap(), b"winner");
        assert_eq!(std::fs::read_to_string(target.join("data.yaml")).unwrap(), "winner: true\n");
    }

    #[test]
    fn concurrent_target_creation_refuses_publish_and_cleans_stage() {
        let root = TestDir::new("cli-synth-race");
        // Non-empty target created concurrently.
        let target = root.join("raced");
        let stage = SynthStage::new(&target).unwrap();
        std::fs::write(stage.path().join("data.yaml"), "ok").unwrap();
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("keep.txt"), b"mine").unwrap();
        let err = stage.publish().unwrap_err();
        assert_eq!(err.code, "output_exists");
        assert_eq!(std::fs::read(target.join("keep.txt")).unwrap(), b"mine");
        assert_eq!(dir_entries(root.path()), ["raced"], "staging dir must be removed on publish error");

        // Empty directory created concurrently right before atomic no-replace rename must NOT be replaced.
        let empty_target = root.join("raced_empty");
        let stage2 = SynthStage::new(&empty_target).unwrap();
        std::fs::write(stage2.path().join("data.yaml"), "ok").unwrap();
        std::fs::create_dir(&empty_target).unwrap();
        let rename_err = rename_dir_noreplace(stage2.path(), &empty_target).unwrap_err();
        assert!(
            matches!(
                rename_err.kind(),
                std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
            ),
            "expected AlreadyExists/PermissionDenied, got {rename_err:?}"
        );
        assert!(dir_entries(&empty_target).is_empty(), "empty target must stay untouched");
        drop(stage2);
        assert_eq!(dir_entries(root.path()), ["raced", "raced_empty"]);
    }
}
