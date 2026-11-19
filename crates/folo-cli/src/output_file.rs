//! The file a command (export, quantize) writes: built under a temporary name in the output's folder and given the
//! output's name only once complete ([`OutputFile::publish`], `folo_nn::io::publish_new`: one move that fails when
//! the name exists). An export that fails or is killed therefore leaves the name free for a retry (#312), concurrent
//! commands never share or replace a file, and an existing file is never overwritten.

use std::path::{Path, PathBuf};

use crate::output::{CliError, ExitCode};

/// A command's output being written: write [`OutputFile::tmp`], then [`OutputFile::publish`] it. Dropped
/// unpublished (an error), the temporary file is removed; a killed process leaves at most that temporary file
/// (`.folo-<pid>-<n>.tmp`, or the writer's own temporary file named after it), which no later run reads or blocks on.
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
    /// [`OutputFile::publish`] when another command took the name meanwhile), and a folder that cannot take the
    /// file is `io_error`, both naming the output.
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
                // A folder is not a file that exists: what opening it for writing says (access denied, is a
                // directory).
                Ok(m) if m.is_dir() => {
                    let e = std::fs::OpenOptions::new().write(true).open(o).err();
                    return Err(io_error(o, e.unwrap_or_else(|| std::io::ErrorKind::IsADirectory.into())));
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
        file.tmp = folo_nn::io::temp_path(dir).map_err(|e| io_error(&first, e))?;
        Ok(file)
    }

    /// The temporary file to write the output to (in the output's folder; the writer creates it).
    pub(crate) fn tmp(&self) -> &Path {
        &self.tmp
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
                std::fs::write(killed.tmp(), b"ha").unwrap();
            }
            if step >= 2 {
                std::fs::write(killed.tmp(), b"done").unwrap();
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
        // What stays is only temporary files: none at all from the first step, then at most one per run.
        assert_eq!(names(&dir).iter().filter(|n| n.ends_with(".tmp")).count(), 2, "{:?}", names(&dir));
        // A retry of a killed export is not blocked by its leftovers.
        std::fs::remove_file(&out).unwrap();
        let retry = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        std::fs::write(retry.tmp(), b"again").unwrap();
        assert_eq!(retry.publish().unwrap(), out);
        assert_eq!(std::fs::read(&out).unwrap(), b"again");
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
        std::fs::write(a.tmp(), b"a").unwrap();
        std::fs::write(b.tmp(), b"b").unwrap();
        a.publish().unwrap();
        assert_eq!(b.publish().unwrap_err().code, "output_exists");
        assert_eq!(std::fs::read(&out).unwrap(), b"a");
        // Two exports without --output take distinct free names.
        let c = OutputFile::new("export", &model, None, "-f16", "folo").unwrap();
        let d = OutputFile::new("export", &model, None, "-f16", "folo").unwrap();
        std::fs::write(c.tmp(), b"c").unwrap();
        std::fs::write(d.tmp(), b"d").unwrap();
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
        std::fs::write(file.tmp(), b"done").unwrap();
        assert_eq!(file.publish().unwrap(), out);
        assert_eq!(std::fs::read(&out).unwrap(), b"done");
    }

    #[test]
    fn errors_name_the_output_as_before() {
        let dir = TestDir::new("cli-output-errors");
        let model = dir.join("best.folo");
        std::fs::write(&model, b"model").unwrap();
        // An existing folder is io_error, as main's `create_new` said (not output_exists).
        let folder = dir.join("folder");
        std::fs::create_dir(&folder).unwrap();
        let e = OutputFile::new("export", &model, Some(folder.clone()), "", "onnx").err().unwrap();
        let p = folo_api::json::path(&folder);
        assert_eq!(e.code, "io_error");
        assert!(e.message.starts_with(&format!("cannot create {p}: ")), "{}", e.message);
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
