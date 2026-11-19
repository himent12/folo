//! The file a command (export, quantize) writes: built under a temporary name next to its output and given the
//! output's name only once complete ([`OutputFile::publish`]), atomically and only while that name is free. An
//! export that fails or is killed therefore leaves the name free for a retry (#312), concurrent commands never
//! share or replace a file, and an existing file is never overwritten.

use std::path::{Path, PathBuf};

use crate::output::{CliError, ExitCode};

/// A command's output being written: write [`OutputFile::tmp`], then [`OutputFile::publish`] it. Dropped
/// unpublished (an error), the temporary file is removed; a killed process leaves only that temporary file
/// (`<output>.<pid>.<n>.tmp`), which no later run reads, claims or blocks on.
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
    /// `-3`, ...). An existing `output` is `output_exists` here already (before any work), and again at
    /// [`OutputFile::publish`] when another command took the name meanwhile.
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
            if std::fs::symlink_metadata(o).is_ok() {
                return Err(exists(command, o));
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
        let (tmp, _) = folo_nn::io::safetensors::create_temp_beside(&first).map_err(|e| {
            CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!("cannot create {}: {e}", folo_api::json::path(&first)),
            )
        })?;
        file.tmp = tmp;
        Ok(file)
    }

    /// The temporary file to write the output to (it exists, empty).
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
            _ => CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!(
                    "cannot write {}: {e}",
                    folo_api::json::path(self.given.as_ref().unwrap_or(&self.tmp))
                ),
            ),
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
        // Gone already once published (the hard link's source is removed, or it was renamed).
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
    fn a_killed_export_leaves_its_output_free_for_a_retry() {
        let dir = TestDir::new("cli-output-killed");
        let (model, out) = (dir.join("best.folo"), dir.join("new.onnx"));
        std::fs::write(&model, b"model").unwrap();
        let killed = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        std::fs::write(killed.tmp(), b"half").unwrap();
        // A killed process runs no destructor: only its temporary file stays, and the output's name is free.
        std::mem::forget(killed);
        assert!(!out.exists(), "the output exists before the export finished");
        let retry = OutputFile::new("export", &model, Some(out.clone()), "", "onnx").unwrap();
        std::fs::write(retry.tmp(), b"done").unwrap();
        assert_eq!(retry.publish().unwrap(), out);
        assert_eq!(std::fs::read(&out).unwrap(), b"done");
        let stale = names(&dir).into_iter().filter(|n| n.ends_with(".tmp")).count();
        assert_eq!(stale, 1, "{:?}", names(&dir));
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
}
