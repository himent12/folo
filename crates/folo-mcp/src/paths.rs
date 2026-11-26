//! Input paths: relative ones resolve against `--data-dir`, and `--confine` keeps reads inside it, comparing
//! canonical paths (symlinks and junctions resolved) so `..`, links and junctions cannot escape.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use http::StatusCode;

use crate::errors::{Error, invalid_arguments};

/// Largest number of images `predict_batch` accepts (after glob expansion).
pub const MAX_BATCH: usize = 256;

/// Maximum symlink hops followed while checking confinement of a missing/broken path.
const MAX_SYMLINK_HOPS: u8 = 40;

/// Outcome of resolving one `predict_batch` path entry under [`PathPolicy`].
pub enum BatchPath {
    /// Path exists (or confinement is off) and is ready to read.
    Ready(PathBuf),
    /// Path stays inside `--data-dir` under `--confine` but does not exist; record in `failed[]`.
    Missing(Error),
}

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
                format!("{} does not exist", folo_api::json::path(&joined)),
            )
        })?;
        self.check_inside(&canon)?;
        Ok(canon)
    }

    /// Resolves one `predict_batch` path entry: empty paths and paths escaping `--data-dir` (existing or missing,
    /// including broken symlinks pointing outside) fail the whole call; missing paths inside `--data-dir` return
    /// [`BatchPath::Missing`] so they are recorded in `failed[]` rather than aborting the batch.
    pub fn resolve_for_batch(&self, input: &str) -> Result<BatchPath, Error> {
        if input.trim().is_empty() {
            return Err(invalid_arguments("path must not be empty"));
        }
        let p = Path::new(input);
        let joined = if p.is_absolute() { p.to_path_buf() } else { self.data_dir.join(p) };
        if !self.confine {
            return Ok(BatchPath::Ready(joined));
        }
        let root = self.canonical_root()?;
        if let Ok(canon) = std::fs::canonicalize(&joined) {
            outside_error(&root, &canon)?;
            return Ok(BatchPath::Ready(canon));
        }
        match resolve_nonexistent(&joined, 0) {
            Some(target) => outside_error(&root, &target)?,
            None => return Err(outside_error(&root, &joined).unwrap_err()),
        }
        Ok(BatchPath::Missing(Error::new(
            StatusCode::NOT_FOUND,
            "path_not_found",
            format!("{} does not exist", folo_api::json::path(&joined)),
        )))
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
            .map_err(|e| Error::internal(format!("data dir {}: {e}", folo_api::json::path(&self.data_dir))))
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
    /// Relative patterns resolve `self.data_dir` natively and apply the caller's glob below it, so metacharacters
    /// or differing letter case in `data_dir`'s name and traverse-only parent folders never break matching.
    pub fn expand_glob(&self, pattern: &str) -> Result<Vec<PathBuf>, Error> {
        if pattern.trim().is_empty() {
            return Err(invalid_arguments("glob must not be empty"));
        }
        glob::Pattern::new(pattern)
            .map_err(|e| invalid_arguments(format!("glob `{pattern}` is invalid: {e}")))?;
        if Path::new(pattern).is_absolute() {
            let entries = glob::glob(pattern)
                .map_err(|e| invalid_arguments(format!("glob `{pattern}` is invalid: {e}")))?;
            let mut out = Vec::new();
            for entry in entries {
                let Ok(path) = entry else { continue };
                if !path.is_file() {
                    continue;
                }
                if out.len() == MAX_BATCH {
                    return Err(too_many_inputs(format!(
                        "glob `{pattern}` matches more than {MAX_BATCH} files"
                    )));
                }
                let path = if self.confine {
                    let canon = std::fs::canonicalize(&path)
                        .map_err(|e| Error::internal(format!("{}: {e}", folo_api::json::path(&path))))?;
                    self.check_inside(&canon)?;
                    canon
                } else {
                    path
                };
                out.push(path);
            }
            out.sort();
            return Ok(out);
        }
        if pattern.ends_with('/') || (cfg!(windows) && pattern.ends_with('\\')) {
            return Ok(Vec::new());
        }
        let segs = parse_relative_glob(pattern)?;
        if segs.is_empty() {
            return Ok(Vec::new());
        }
        let mut walker = GlobWalker {
            policy: self,
            pattern,
            opts: glob::MatchOptions::new(),
            seen: HashSet::new(),
            out: Vec::new(),
        };
        let mut ancestors: Vec<PathBuf> = std::fs::canonicalize(&self.data_dir).into_iter().collect();
        walker.walk(&self.data_dir, &segs, &mut ancestors)?;
        walker.out.sort();
        Ok(walker.out)
    }
}

/// Resolves a path that failed `std::fs::canonicalize` left-to-right so symlinks (including broken ones) are
/// followed before `..` and confinement checks never leak existence of files outside `--data-dir`.
fn resolve_nonexistent(p: &Path, depth: u8) -> Option<PathBuf> {
    if depth >= MAX_SYMLINK_HOPS || p.as_os_str().is_empty() {
        return None;
    }
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut cur = PathBuf::new();
    let mut missing = false;
    for c in abs.components() {
        match c {
            Component::CurDir => {}
            Component::Prefix(prefix) => cur.push(prefix.as_os_str()),
            Component::RootDir => {
                cur.push(Component::RootDir);
                if let Ok(canon) = std::fs::canonicalize(&cur) {
                    cur = canon;
                    missing = false;
                } else {
                    missing = true;
                }
            }
            Component::ParentDir => {
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(cur.join("..")) {
                        cur = canon;
                    } else {
                        let _ = cur.pop();
                        missing = true;
                    }
                } else {
                    let _ = cur.pop();
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                        missing = false;
                    }
                }
            }
            Component::Normal(part) => {
                cur.push(part);
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                    } else if std::fs::symlink_metadata(&cur).is_ok_and(|m| m.file_type().is_symlink()) {
                        let target = std::fs::read_link(&cur).ok()?;
                        let next = if target.is_absolute() {
                            target
                        } else {
                            cur.parent().unwrap_or(Path::new("")).join(target)
                        };
                        cur = resolve_nonexistent(&next, depth + 1)?;
                        missing = !cur.exists();
                    } else {
                        missing = true;
                    }
                }
            }
        }
    }
    if cur.as_os_str().is_empty() {
        return None;
    }
    if missing && !cur.ancestors().skip(1).any(Path::exists) {
        return None;
    }
    Some(cur)
}

enum Seg {
    Parent,
    GlobStar,
    Literal(String),
    Pattern(glob::Pattern),
}

fn parse_relative_glob(pattern: &str) -> Result<Vec<Seg>, Error> {
    let mut segs = Vec::new();
    for raw in pattern.split(|c| c == '/' || (cfg!(windows) && c == '\\')) {
        if raw.is_empty() || raw == "." {
            continue;
        }
        let seg = if raw == ".." {
            Seg::Parent
        } else if raw == "**" {
            if matches!(segs.last(), Some(Seg::GlobStar)) {
                continue;
            }
            Seg::GlobStar
        } else if raw.contains(['*', '?', '[']) {
            Seg::Pattern(
                glob::Pattern::new(raw)
                    .map_err(|e| invalid_arguments(format!("glob `{pattern}` is invalid: {e}")))?,
            )
        } else {
            Seg::Literal(raw.to_string())
        };
        segs.push(seg);
    }
    Ok(segs)
}

struct GlobWalker<'a> {
    policy: &'a PathPolicy,
    pattern: &'a str,
    opts: glob::MatchOptions,
    seen: HashSet<PathBuf>,
    out: Vec<PathBuf>,
}

impl GlobWalker<'_> {
    fn push_file(&mut self, path: PathBuf) -> Result<(), Error> {
        if !path.is_file() {
            return Ok(());
        }
        let resolved = if self.policy.confine {
            let canon = std::fs::canonicalize(&path)
                .map_err(|e| Error::internal(format!("{}: {e}", folo_api::json::path(&path))))?;
            self.policy.check_inside(&canon)?;
            canon
        } else {
            path
        };
        if self.seen.insert(resolved.clone()) {
            if self.out.len() == MAX_BATCH {
                return Err(too_many_inputs(format!(
                    "glob `{}` matches more than {MAX_BATCH} files",
                    self.pattern
                )));
            }
            self.out.push(resolved);
        }
        Ok(())
    }

    fn walk(&mut self, dir: &Path, segs: &[Seg], ancestors: &mut Vec<PathBuf>) -> Result<(), Error> {
        let Some((first, rest)) = segs.split_first() else {
            return self.push_file(dir.to_path_buf());
        };
        match first {
            Seg::Literal(lit) => {
                let next = dir.join(lit);
                if rest.is_empty() {
                    self.push_file(next)
                } else if next.is_dir() {
                    self.walk(&next, rest, ancestors)
                } else {
                    Ok(())
                }
            }
            Seg::Parent => {
                if rest.is_empty() {
                    Ok(())
                } else {
                    self.walk(&dir.join(".."), rest, ancestors)
                }
            }
            Seg::Pattern(pat) => {
                let Ok(rd) = std::fs::read_dir(dir) else { return Ok(()) };
                let mut names: Vec<std::ffi::OsString> =
                    rd.filter_map(|e| e.ok().map(|e| e.file_name())).collect();
                names.sort();
                for name in names {
                    let s = name.to_string_lossy();
                    if !pat.matches_with(&s, self.opts) {
                        continue;
                    }
                    let next = dir.join(&name);
                    if rest.is_empty() {
                        self.push_file(next)?;
                    } else if next.is_dir() {
                        self.walk(&next, rest, ancestors)?;
                    }
                }
                Ok(())
            }
            Seg::GlobStar => {
                if rest.is_empty() {
                    self.walk_all_files(dir, ancestors)
                } else {
                    self.walk(dir, rest, ancestors)?;
                    self.for_each_subdir(dir, ancestors, |this, sub, ancestors| {
                        this.walk(sub, segs, ancestors)
                    })
                }
            }
        }
    }

    fn walk_all_files(&mut self, dir: &Path, ancestors: &mut Vec<PathBuf>) -> Result<(), Error> {
        let Ok(rd) = std::fs::read_dir(dir) else { return Ok(()) };
        let mut names: Vec<std::ffi::OsString> = rd.filter_map(|e| e.ok().map(|e| e.file_name())).collect();
        names.sort();
        for name in names {
            let next = dir.join(&name);
            if next.is_dir() {
                if let Ok(canon) = std::fs::canonicalize(&next) {
                    if ancestors.contains(&canon) {
                        continue;
                    }
                    ancestors.push(canon);
                    let res = self.walk_all_files(&next, ancestors);
                    ancestors.pop();
                    res?;
                }
            } else {
                self.push_file(next)?;
            }
        }
        Ok(())
    }

    fn for_each_subdir(
        &mut self,
        dir: &Path,
        ancestors: &mut Vec<PathBuf>,
        mut f: impl FnMut(&mut Self, &Path, &mut Vec<PathBuf>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let Ok(rd) = std::fs::read_dir(dir) else { return Ok(()) };
        let mut names: Vec<std::ffi::OsString> = rd.filter_map(|e| e.ok().map(|e| e.file_name())).collect();
        names.sort();
        for name in names {
            let next = dir.join(&name);
            if !next.is_dir() {
                continue;
            }
            if let Ok(canon) = std::fs::canonicalize(&next) {
                if ancestors.contains(&canon) {
                    continue;
                }
                ancestors.push(canon);
                let res = f(self, &next, ancestors);
                ancestors.pop();
                res?;
            }
        }
        Ok(())
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
    use folo_test_support::TestDir;

    fn tmp(name: &str) -> TestDir {
        TestDir::new(&format!("mcp-paths-{name}"))
    }

    #[test]
    fn relative_paths_resolve_against_the_data_dir() {
        let d = tmp("rel");
        let p = PathPolicy { data_dir: d.to_path_buf(), confine: false };
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
        let p = PathPolicy { data_dir: d.to_path_buf(), confine: false };
        assert_eq!(p.expand_glob("*.jpg").unwrap_err().body.code, "too_many_inputs");
        assert_eq!(p.expand_glob("000*.jpg").unwrap().len(), 10);
    }

    #[test]
    fn relative_glob_escapes_metacharacters_in_data_dir() {
        let root = tmp("glob-meta");
        // A sibling folder that an unescaped `data [v2]` character class would match.
        let sibling = root.join("data 2");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("secret.png"), b"s").unwrap();
        for name in ["data [v2]", "data [1"] {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("a.png"), b"x").unwrap();
            for confine in [false, true] {
                let p = PathPolicy { data_dir: dir.clone(), confine };
                let matched = p.expand_glob("*.png").unwrap();
                assert_eq!(matched.len(), 1, "{name} confine={confine}: {matched:?}");
                assert_eq!(matched[0].file_name().and_then(|s| s.to_str()), Some("a.png"));
                assert_eq!(
                    std::fs::canonicalize(&matched[0]).unwrap(),
                    std::fs::canonicalize(dir.join("a.png")).unwrap(),
                    "{name} confine={confine}: {matched:?}"
                );
            }
        }
        // Plain data dirs keep native path separators when `--confine` is off (no mixed `/` and `\` on Windows).
        let plain = root.join("plain");
        std::fs::create_dir_all(plain.join("sub")).unwrap();
        let target = plain.join("sub").join("a.png");
        std::fs::write(&target, b"x").unwrap();
        let p = PathPolicy { data_dir: plain.clone(), confine: false };
        let matched = p.expand_glob("sub/*.png").unwrap();
        assert_eq!(matched, vec![target.clone()]);
        assert_eq!(matched[0].to_string_lossy(), target.to_string_lossy());

        // On Windows, a verbatim `\\?\` data dir keeps its prefix verbatim while still escaping metacharacters after it.
        #[cfg(windows)]
        {
            let canon_plain = std::fs::canonicalize(&plain).unwrap();
            assert!(canon_plain.to_string_lossy().starts_with(r"\\?\"), "{canon_plain:?}");
            let want_plain = canon_plain.join("sub").join("a.png");
            let p = PathPolicy { data_dir: canon_plain, confine: false };
            let matched = p.expand_glob("sub/*.png").unwrap();
            assert_eq!(matched, vec![want_plain.clone()]);
            assert_eq!(matched[0].to_string_lossy(), want_plain.to_string_lossy());

            let canon_meta = std::fs::canonicalize(root.join("data [v2]")).unwrap();
            assert!(canon_meta.to_string_lossy().starts_with(r"\\?\"), "{canon_meta:?}");
            let want_meta = canon_meta.join("a.png");
            let p = PathPolicy { data_dir: canon_meta, confine: false };
            let matched = p.expand_glob("*.png").unwrap();
            assert_eq!(matched, vec![want_meta.clone()]);
            assert_eq!(matched[0].to_string_lossy(), want_meta.to_string_lossy());

            // Differing letter case in a data dir containing glob metacharacters still matches natively on Windows.
            let cased = root.join("MixedCase [v3]");
            std::fs::create_dir_all(cased.join("sub")).unwrap();
            std::fs::write(cased.join("sub/pic.png"), b"x").unwrap();
            let wrong_case = root.join("mixedcase [v3]");
            for confine in [false, true] {
                let p = PathPolicy { data_dir: wrong_case.clone(), confine };
                assert_eq!(p.expand_glob("sub/*.png").unwrap().len(), 1, "confine={confine}");
                assert_eq!(p.expand_glob("**/*.png").unwrap().len(), 1, "confine={confine}");
            }
        }
    }

    #[test]
    fn resolve_for_batch_records_missing_inside_and_rejects_missing_outside() {
        let root = tmp("batch-resolve");
        let data = root.join("data");
        std::fs::create_dir_all(data.join("sub")).unwrap();
        std::fs::write(data.join("ok.png"), b"x").unwrap();
        let p = PathPolicy { data_dir: data.clone(), confine: true };

        assert!(matches!(p.resolve_for_batch("ok.png").unwrap(), BatchPath::Ready(_)));
        match p.resolve_for_batch("missing.png").unwrap() {
            BatchPath::Missing(e) => assert_eq!(e.body.code, "path_not_found"),
            BatchPath::Ready(path) => panic!("expected Missing, got Ready({path:?})"),
        }
        match p.resolve_for_batch("sub/deep/missing.png").unwrap() {
            BatchPath::Missing(e) => assert_eq!(e.body.code, "path_not_found"),
            BatchPath::Ready(path) => panic!("expected Missing, got Ready({path:?})"),
        }
        assert_eq!(
            p.resolve_for_batch("../outside_missing.png").err().unwrap().body.code,
            "path_outside_data_dir"
        );
        assert_eq!(
            p.resolve_for_batch(root.join("outside_missing.png").to_str().unwrap()).err().unwrap().body.code,
            "path_outside_data_dir"
        );
        assert_eq!(p.resolve_for_batch("  ").err().unwrap().body.code, "invalid_arguments");

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("outside_missing.png"), data.join("broken_out.png"))
                .unwrap();
            assert_eq!(
                p.resolve_for_batch("broken_out.png").err().unwrap().body.code,
                "path_outside_data_dir"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn relative_glob_works_under_traverse_only_parent_folder() {
        use std::os::unix::fs::PermissionsExt;
        let root = tmp("traverse-only");
        let parent = root.join("no_list");
        let data = parent.join("data [v1]");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("a.png"), b"x").unwrap();
        // Execute-only parent (0111): cannot be listed, only traversed.
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o111)).unwrap();
        let p = PathPolicy { data_dir: data.clone(), confine: true };
        let res = p.expand_glob("*.png");
        // Restore permissions before asserting so TestDir cleanup always succeeds.
        let _ = std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755));
        let matched = res.unwrap();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].file_name().and_then(|s| s.to_str()), Some("a.png"));
    }
}
