//! The file a command (export, quantize) writes: built under a temporary name in the output's folder and given the
//! output's name only once complete ([`OutputFile::publish`], `folo_nn::io::publish_new`: one move that fails when
//! the name exists). An export that fails or is killed therefore leaves the name free for a retry (#312), concurrent
//! commands never share or replace a file, and an existing file is never overwritten.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::output::{CliError, ExitCode};

/// A command's output being written: write directly into the reserved temporary file via [`OutputFile::write_with`]
/// or [`OutputFile::write_bytes`] (or [`OutputFile::tmp`] for writers that take a path), then [`OutputFile::publish`]
/// it. Dropped unpublished (an error), the temporary file is removed; a killed process leaves at most that temporary
/// file (`.folo-<pid>-<n>.tmp`), which no later run reads or blocks on.
pub(crate) struct OutputFile {
    command: &'static str,
    /// `--output`, or `None` for a free name next to the model.
    given: Option<PathBuf>,
    /// The free name's parts: the model and the name's suffix and extension.
    model: PathBuf,
    suffix: &'static str,
    ext: &'static str,
    tmp: PathBuf,
}

impl OutputFile {
    /// The output of `command`: `output`, or a free name next to `model` (`<stem><suffix>.<ext>`, then `-2`,
    /// `-3`, ...). Checked before any work: an existing `output` is `output_exists` (and again at
    /// [`OutputFile::publish`] when another command took the name meanwhile), an existing folder is what `create_new`
    /// says of it, and a folder that cannot take the file is `io_error`, all naming the output.
    pub(crate) fn new(
        command: &'static str,
        model: &Path,
        output: Option<PathBuf>,
        suffix: &'static str,
        ext: &'static str,
    ) -> Result<OutputFile, CliError> {
        if let Some(o) = &output {
            let same = |a: &Path, b: &Path| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
                (Ok(a), Ok(b)) => a == b,
                _ => a == b,
            };
            if same(o, model) {
                return Err(CliError::new(ExitCode::Usage, "usage", "--output must differ from --model"));
            }
            match std::fs::symlink_metadata(o) {
                // A folder: what main's `create_new` said (`output_exists` on Unix, access denied on Windows).
                Ok(m) if m.is_dir() => {
                    return Err(match std::fs::OpenOptions::new().write(true).create_new(true).open(o) {
                        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => exists(command, o),
                        Err(e) => io_error(o, e),
                        Ok(_) => io_error(o, std::io::ErrorKind::IsADirectory.into()),
                    });
                }
                Ok(_) => return Err(exists(command, o)),
                Err(_) => {}
            }
        }
        let mut file = OutputFile {
            command,
            given: output,
            model: model.to_path_buf(),
            suffix,
            ext,
            tmp: PathBuf::new(),
        };
        let first = file.name(1);
        let dir = first.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let first_name = first.file_name().map(|n| n.to_string_lossy().to_lowercase());
        loop {
            let tmp = folo_nn::io::temp_path(dir).map_err(|e| io_error(&first, e))?;
            // `--output .folo-<pid>-<n>.tmp` is not the temporary file too: the next temporary name is taken.
            if tmp.file_name().map(|n| n.to_string_lossy().to_lowercase()) != first_name {
                file.tmp = tmp;
                return Ok(file);
            }
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// The temporary file to write the output to (in the output's folder; created empty, the writer writes over it).
    pub(crate) fn tmp(&self) -> &Path {
        &self.tmp
    }

    /// Writes directly into the reserved temporary file through a 1 MiB buffer, flushes and `sync_all`s, then
    /// closes the handle before returning so [`OutputFile::publish`] can rename it cleanly on Windows without
    /// creating a second nested temporary file.
    pub(crate) fn write_with(
        &self,
        write: impl FnOnce(&mut std::io::BufWriter<std::fs::File>) -> Result<(), CliError>,
    ) -> Result<(), CliError> {
        let target = self.name(1);
        let f = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.tmp)
            .map_err(|e| io_error(&target, e))?;
        let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
        write(&mut w)?;
        let f = w.into_inner().map_err(|e| io_error(&target, e.into_error()))?;
        f.sync_all().map_err(|e| io_error(&target, e))?;
        Ok(())
    }

    /// Writes `bytes` directly into the reserved temporary file, syncs it to disk, and closes it.
    pub(crate) fn write_bytes(&self, bytes: &[u8]) -> Result<(), CliError> {
        let target = self.name(1);
        self.write_with(|w| w.write_all(bytes).map_err(|e| io_error(&target, e)))
    }

    /// Gives the written temporary file its name and returns it.
    pub(crate) fn publish(self) -> Result<PathBuf, CliError> {
        let published = match &self.given {
            Some(o) => folo_nn::io::publish_new(&self.tmp, o).map(|()| o.clone()),
            None => folo_nn::io::publish_free(&self.tmp, |n| self.name(n)),
        };
        published.map_err(|e| match &self.given {
            Some(o) if e.kind() == std::io::ErrorKind::AlreadyExists => exists(self.command, o),
            _ => io_error(&self.name(1), e),
        })
    }

    /// The `n`th name tried: `--output`, or `<stem><suffix>.<ext>` next to the model (`-n` before the extension
    /// from the second on).
    fn name(&self, n: u32) -> PathBuf {
        if let Some(o) = &self.given {
            return o.clone();
        }
        let stem =
            self.model.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
        let tag = if n == 1 { String::new() } else { format!("-{n}") };
        self.model.with_file_name(format!("{stem}{}{tag}.{}", self.suffix, self.ext))
    }
}

impl Drop for OutputFile {
    fn drop(&mut self) {
        // Gone already once published (moved to its name), or never written.
        let _ = std::fs::remove_file(&self.tmp);
    }
}

/// `output_exists` for `o`.
fn exists(command: &str, o: &Path) -> CliError {
    CliError::new(
        ExitCode::Runtime,
        "output_exists",
        format!("{} exists; {command} never overwrites", folo_api::json::path(o)),
    )
    .hint("pass another --output, or leave it out for a free name next to --model")
}

/// `io_error` for the output `o`.
fn io_error(o: &Path, e: std::io::Error) -> CliError {
    CliError::new(ExitCode::Runtime, "io_error", format!("cannot create {}: {e}", folo_api::json::path(o)))
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    use super::OutputFile;

    /// The names in `dir`, sorted.
    fn names(dir: &TestDir) -> Vec<String> {
        let mut n: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        n.sort();
        n
    }

    #[test]
    fn a_killed_export_leaves_its_output_absent_or_complete_never_empty() {
        let dir = TestDir::new("cli-output-killed");
        let (model, out) = (dir.join("best.folo"), dir.join("new.onnx"));
        std::fs::write(&model, b"model").unwrap();
        // Killed after each step: created, half written, written, published. A killed process runs no destructor.
        for step in 0..4 {
            let killed = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
            if step >= 1 {
                killed.write_bytes(b"ha").unwrap();
            }
            if step >= 2 {
                killed.write_bytes(b"done").unwrap();
            }
            if step == 3 {
                assert_eq!(killed.publish().unwrap(), out);
                assert_eq!(std::fs::read(&out).unwrap(), b"done");
                continue;
            }
            std::mem::forget(killed);
            assert!(!out.exists(), "step {step}: the output exists before the export finished");
            let stale = names(&dir);
            assert!(
                stale.iter().all(|n| n == "best.folo" || (n.starts_with(".folo-") && n.ends_with(".tmp")))
            );
        }
        // What stays is only temporary files: at most one per killed run (the first one's empty, its name reserved).
        assert_eq!(names(&dir).iter().filter(|n| n.ends_with(".tmp")).count(), 3, "{:?}", names(&dir));
        // A retry of a killed export is not blocked by its leftovers.
        std::fs::remove_file(&out).unwrap();
        let retry = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        retry.write_bytes(b"again").unwrap();
        assert_eq!(retry.publish().unwrap(), out);
        assert_eq!(std::fs::read(&out).unwrap(), b"again");
    }

    #[test]
    fn failed_write_cleans_up_temp_file_and_leaves_no_output() {
        let dir = TestDir::new("cli-output-write-fail");
        let (model, out) = (dir.join("best.folo"), dir.join("failed.onnx"));
        std::fs::write(&model, b"model").unwrap();
        {
            let file = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
            let err = file
                .write_with(|w| {
                    use std::io::Write;
                    w.write_all(b"partial").map_err(|e| {
                        crate::output::CliError::new(
                            crate::output::ExitCode::Runtime,
                            "io_error",
                            e.to_string(),
                        )
                    })?;
                    Err(crate::output::CliError::new(
                        crate::output::ExitCode::Runtime,
                        "io_error",
                        "simulated write error",
                    ))
                })
                .unwrap_err();
            assert_eq!(err.code, "io_error");
        }
        assert!(!out.exists(), "failed write must not publish output");
        assert_eq!(names(&dir), ["best.folo"], "temp file must be removed on failure");
    }

    #[test]
    fn concurrent_exports_never_replace_a_finished_file() {
        let dir = TestDir::new("cli-output-race");
        let model = dir.join("best.folo");
        std::fs::write(&model, b"model").unwrap();
        // Two exports to one --output: the first to finish wins, the other is refused and leaves nothing.
        let out = dir.join("new.onnx");
        let a = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        let b = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        a.write_bytes(b"a").unwrap();
        b.write_bytes(b"b").unwrap();
        a.publish().unwrap();
        assert_eq!(b.publish().unwrap_err().code, "output_exists");
        assert_eq!(std::fs::read(&out).unwrap(), b"a");
        // Two exports without --output take distinct free names.
        let c = OutputFile::new("export", &model, None, "-f16", "folo").unwrap();
        let d = OutputFile::new("export", &model, None, "-f16", "folo").unwrap();
        c.write_bytes(b"c").unwrap();
        d.write_bytes(b"d").unwrap();
        assert_eq!(d.publish().unwrap(), dir.join("best-f16.folo"));
        assert_eq!(c.publish().unwrap(), dir.join("best-f16-2.folo"));
        assert_eq!(names(&dir), ["best-f16-2.folo", "best-f16.folo", "best.folo", "new.onnx"]);
    }

    #[test]
    fn an_output_name_as_long_as_the_file_system_allows_is_written() {
        let dir = TestDir::new("cli-output-long");
        let model = dir.join("best.folo");
        std::fs::write(&model, b"model").unwrap();
        // 255 characters, the limit of one path component (main took it; a temporary name built on it did not).
        let out = dir.join(format!("{}.onnx", "q".repeat(250)));
        let file = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        file.write_bytes(b"done").unwrap();
        assert_eq!(file.publish().unwrap(), out);
        assert_eq!(std::fs::read(&out).unwrap(), b"done");
    }

    #[test]
    fn an_output_named_like_the_next_temporary_file_is_not_written_through_it() {
        let dir = TestDir::new("cli-output-tmp-name");
        let model = dir.join("best.folo");
        std::fs::write(&model, b"model").unwrap();
        // `--output` is the temporary name this process takes next (unless another test takes it first).
        let probe = folo_nn::io::temp_path(dir.path()).unwrap();
        std::fs::remove_file(&probe).unwrap();
        let n: u64 =
            probe.to_string_lossy().rsplit('-').next().unwrap().trim_end_matches(".tmp").parse().unwrap();
        let out = dir.join(format!(".folo-{}-{}.tmp", std::process::id(), n + 1));
        let file = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        assert_ne!(file.tmp(), out);
        file.write_bytes(b"done").unwrap();
        assert_eq!(file.publish().unwrap(), out);
        assert_eq!(std::fs::read(&out).unwrap(), b"done");
        assert_eq!(names(&dir).len(), 2, "{:?}", names(&dir));
    }

    #[test]
    fn errors_name_the_output_as_before() {
        let dir = TestDir::new("cli-output-errors");
        let model = dir.join("best.folo");
        std::fs::write(&model, b"model").unwrap();
        // An existing folder is what main's `create_new` said: EEXIST (output_exists) on Unix, access denied
        // (io_error) on Windows.
        let folder = dir.join("folder");
        std::fs::create_dir(&folder).unwrap();
        let e = OutputFile::new("export", &model, Some(folder.clone()), "", "onnx").err().unwrap();
        let p = folo_api::json::path(&folder);
        let (code, start) = if cfg!(windows) {
            ("io_error", format!("cannot create {p}: "))
        } else {
            ("output_exists", format!("{p} exists"))
        };
        assert_eq!(e.code, code);
        assert!(e.message.starts_with(&start), "{}", e.message);
        // A missing folder: io_error naming the output, not a temporary file.
        let missing = dir.join("nope").join("x.onnx");
        let e = OutputFile::new("export", &model, Some(missing.clone()), "", "onnx").err().unwrap();
        let p = folo_api::json::path(&missing);
        assert_eq!(e.code, "io_error");
        assert!(
            e.message.starts_with(&format!("cannot create {p}: ")) && !e.message.contains(".tmp"),
            "{}",
            e.message
        );
        // An existing file is output_exists.
        std::fs::write(dir.join("taken.onnx"), b"old").unwrap();
        let e = OutputFile::new("export", &model, Some(dir.join("taken.onnx")), "", "onnx").err().unwrap();
        assert_eq!(e.code, "output_exists");
        assert_eq!(std::fs::read(dir.join("taken.onnx")).unwrap(), b"old");
    }
}
