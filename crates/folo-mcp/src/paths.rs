//! Input paths: relative ones resolve against `--data-dir`, and `--confine` keeps reads inside it, comparing
//! canonical paths (symlinks and junctions resolved) so `..`, links and junctions cannot escape.

use std::path::{Path, PathBuf};

use http::StatusCode;

use crate::errors::{Error, invalid_arguments};

/// Largest number of images `predict_batch` accepts (after glob expansion).
pub const MAX_BATCH: usize = 256;

/// How input paths are resolved and checked.
#[derive(Clone, Debug)]
pub struct PathPolicy {
    /// Base for relative paths (and the sandbox with `confine`).
    pub data_dir: PathBuf,
    /// Reject paths outside `data_dir`.
    pub confine: bool,
}

impl PathPolicy {
    /// Resolves `input` for reading: relative to `data_dir`, and with `confine` canonicalised and checked. The
    /// result is absolute when `data_dir` is.
    pub fn resolve(&self, input: &str) -> Result<PathBuf, Error> {
        if input.trim().is_empty() {
            return Err(invalid_arguments("path must not be empty"));
        }
        let p = Path::new(input);
        let joined = if p.is_absolute() { p.to_path_buf() } else { self.data_dir.join(p) };
        if !self.confine {
            return Ok(joined);
        }
        let canon = std::fs::canonicalize(&joined).map_err(|_| {
            Error::new(
                StatusCode::NOT_FOUND,
                "path_not_found",
                format!("{} does not exist", joined.display()),
            )
        })?;
        self.check_inside(&canon)?;
        Ok(canon)
    }

    /// With `confine`, fails with `path_outside_data_dir` unless the canonical path `canon` is inside the canonical
    /// data dir.
    fn check_inside(&self, canon: &Path) -> Result<(), Error> {
        outside_error(&self.canonical_root()?, canon)
    }

    /// The canonical data dir (created if missing).
    pub(crate) fn canonical_root(&self) -> Result<PathBuf, Error> {
        std::fs::create_dir_all(&self.data_dir)
            .and_then(|_| std::fs::canonicalize(&self.data_dir))
            .map_err(|e| Error::internal(format!("data dir {}: {e}", self.data_dir.display())))
    }

    /// With `confine`, checks that every existing path in `files` resolves (symlinks and junctions followed) inside
    /// the data dir. Paths that cannot be resolved (missing, or not a valid name on this OS) are skipped: they cannot
    /// be read either.
    pub fn check_files<'a>(&self, files: impl IntoIterator<Item = &'a Path>) -> Result<(), Error> {
        if !self.confine {
            return Ok(());
        }
        let root = self.canonical_root()?;
        for f in files {
            if let Ok(canon) = std::fs::canonicalize(f) {
                outside_error(&root, &canon)?;
            }
        }
        Ok(())
    }

    /// Expands a glob into at most [`MAX_BATCH`] files (sorted), each checked like [`PathPolicy::resolve`].
    pub fn expand_glob(&self, pattern: &str) -> Result<Vec<PathBuf>, Error> {
        if pattern.trim().is_empty() {
            return Err(invalid_arguments("glob must not be empty"));
        }
        let full = if Path::new(pattern).is_absolute() {
            pattern.to_string()
        } else {
            self.data_dir.join(pattern).to_string_lossy().into_owned()
        };
        let entries =
            glob::glob(&full).map_err(|e| invalid_arguments(format!("glob `{pattern}` is invalid: {e}")))?;
        let mut out = Vec::new();
        for entry in entries {
            let Ok(path) = entry else { continue };
            if !path.is_file() {
                continue;
            }
            if out.len() == MAX_BATCH {
                return Err(too_many_inputs(format!("glob `{pattern}` matches more than {MAX_BATCH} files")));
            }
            let path = if self.confine {
                let canon = std::fs::canonicalize(&path)
                    .map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
                self.check_inside(&canon)?;
                canon
            } else {
                path
            };
            out.push(path);
        }
        out.sort();
        Ok(out)
    }
}

fn outside_error(root: &Path, canon: &Path) -> Result<(), Error> {
    if canon.starts_with(root) {
        return Ok(());
    }
    Err(Error::new(
        StatusCode::FORBIDDEN,
        "path_outside_data_dir",
        format!(
            "{} is outside the data dir {}",
            folo_api::json::path(canon),
            folo_api::json::path(root)
        ),
    )
    .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine"))
}

/// `too_many_inputs`.
pub fn too_many_inputs(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "too_many_inputs", message)
        .hint(format!("split the images into batches of at most {MAX_BATCH}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("folo-mcp-paths-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn relative_paths_resolve_against_the_data_dir() {
        let d = tmp("rel");
        let p = PathPolicy { data_dir: d.clone(), confine: false };
        assert_eq!(p.resolve("a/b.jpg").unwrap(), d.join("a/b.jpg"));
        assert!(p.resolve("  ").is_err());
    }

    #[test]
    fn confine_rejects_escapes() {
        let root = tmp("confine");
        let data = root.join("data");
        std::fs::create_dir_all(data.join("in")).unwrap();
        std::fs::write(data.join("in/x.jpg"), b"x").unwrap();
        std::fs::write(root.join("secret.jpg"), b"s").unwrap();
        let p = PathPolicy { data_dir: data.clone(), confine: true };
        assert!(p.resolve("in/x.jpg").is_ok());
        let e = p.resolve("../secret.jpg").unwrap_err();
        assert_eq!(e.body.code, "path_outside_data_dir");
        let e = p.resolve(root.join("secret.jpg").to_str().unwrap()).unwrap_err();
        assert_eq!(e.body.code, "path_outside_data_dir");
        assert_eq!(p.resolve("missing.jpg").unwrap_err().body.code, "path_not_found");
        // A glob that reaches outside is rejected as a whole.
        assert_eq!(p.expand_glob("../*.jpg").unwrap_err().body.code, "path_outside_data_dir");
        assert_eq!(p.expand_glob("in/*.jpg").unwrap().len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn confine_resolves_symlinks() {
        let root = tmp("link");
        let data = root.join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(root.join("secret.jpg"), b"s").unwrap();
        std::os::unix::fs::symlink(root.join("secret.jpg"), data.join("link.jpg")).unwrap();
        let p = PathPolicy { data_dir: data, confine: true };
        assert_eq!(p.resolve("link.jpg").unwrap_err().body.code, "path_outside_data_dir");
    }

    #[test]
    fn glob_limit() {
        let d = tmp("glob");
        for i in 0..=MAX_BATCH {
            std::fs::write(d.join(format!("{i:04}.jpg")), b"x").unwrap();
        }
        let p = PathPolicy { data_dir: d, confine: false };
        assert_eq!(p.expand_glob("*.jpg").unwrap_err().body.code, "too_many_inputs");
        assert_eq!(p.expand_glob("000*.jpg").unwrap().len(), 10);
    }
}
