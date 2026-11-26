//! Input paths: relative ones resolve against `--data-dir`, and `--confine` keeps reads inside it, comparing
//! canonical paths (symlinks and junctions resolved) so `..`, links and junctions cannot escape.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use http::StatusCode;

use crate::errors::{Error, invalid_arguments};

/// Largest number of images `predict_batch` accepts (after glob expansion).
pub const MAX_BATCH: usize = 256;

/// Maximum symlink hops followed while checking confinement of a missing/broken path.
const MAX_SYMLINK_HOPS: u8 = 40;

/// Largest number of path segments accepted in a glob pattern.
const MAX_GLOB_SEGMENTS: usize = 256;

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
            None => return Err(path_outside_error(&root, &joined)),
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
    ///
    /// Relative patterns resolve `self.data_dir` natively and apply the caller's glob below it, so metacharacters
    /// or differing letter case in `data_dir`'s name and traverse-only parent folders never break matching.
    /// With `confine`, directory boundaries are enforced before enumerating any directory for both relative and
    /// absolute patterns, so an outside pattern fails with `path_outside_data_dir` whether or not anything matches.
    /// `**` matches zero or more directories and a terminal `**` matches all files beneath it, identically for
    /// relative and absolute patterns. Duplicate symlink aliases are preserved and counted toward [`MAX_BATCH`].
    pub fn expand_glob(&self, pattern: &str) -> Result<Vec<PathBuf>, Error> {
        if pattern.trim().is_empty() {
            return Err(invalid_arguments("glob must not be empty"));
        }
        glob::Pattern::new(pattern)
            .map_err(|e| invalid_arguments(format!("glob `{pattern}` is invalid: {e}")))?;
        let parsed = parse_glob(&self.data_dir, pattern)?;
        let root = if self.confine { Some(self.canonical_root()?) } else { None };
        let first_wild = parsed
            .segs
            .iter()
            .position(|s| matches!(s, Seg::Pattern(_) | Seg::GlobStar))
            .unwrap_or(parsed.segs.len());
        let static_segs = &parsed.segs[..first_wild];
        let wild_segs = &parsed.segs[first_wild..];

        let base = resolve_static_prefix(&parsed, static_segs, root.as_deref())?;
        if base.crossed_missing_dir || !base.exists {
            return Ok(Vec::new());
        }
        if parsed.trailing_sep && root.is_none() {
            return Ok(Vec::new());
        }
        if wild_segs.is_empty() {
            if parsed.trailing_sep {
                return Ok(Vec::new());
            }
            return if base.dir.is_file() { Ok(vec![base.dir]) } else { Ok(Vec::new()) };
        }
        if !base.dir.is_dir() {
            return Ok(Vec::new());
        }

        let mut walker = GlobWalker {
            root: root.as_deref(),
            pattern,
            trailing_sep: parsed.trailing_sep,
            opts: glob::MatchOptions::new(),
            ancestors: Vec::new(),
            seen: HashSet::new(),
            out: Vec::new(),
        };
        walker.run(base.dir, wild_segs)?;
        walker.out.sort();
        Ok(walker.out)
    }
}

/// Resolves a path that failed `std::fs::canonicalize` left-to-right so symlinks and Windows junctions
/// (including broken ones) are followed before `..` and confinement checks never leak existence of files
/// outside `--data-dir`.
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
                // When `!missing`, `cur` is already canonical (no symlinks remain), so popping avoids Win32
                // rejecting `..` after a `\\?\` verbatim prefix while preserving symlink-before-`..` semantics.
                let _ = cur.pop();
                if let Ok(canon) = std::fs::canonicalize(&cur) {
                    cur = canon;
                    missing = false;
                } else {
                    missing = true;
                }
            }
            Component::Normal(part) => {
                cur.push(part);
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                    } else if let Ok(target) = std::fs::read_link(&cur) {
                        // `read_link` resolves both symlinks and Windows directory junctions.
                        let next = if target.is_absolute() {
                            target
                        } else {
                            cur.parent().unwrap_or(Path::new("")).join(target)
                        };
                        cur = resolve_nonexistent(&next, depth + 1)?;
                        missing = !cur.exists();
                    } else if std::fs::symlink_metadata(&cur).is_ok_and(|m| m.file_type().is_symlink()) {
                        return None;
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
    Literal(OsString),
    Pattern(glob::Pattern),
}

struct ParsedGlob {
    start_dir: PathBuf,
    has_prefix_or_root: bool,
    trailing_sep: bool,
    segs: Vec<Seg>,
}

fn parse_glob(data_dir: &Path, pattern: &str) -> Result<ParsedGlob, Error> {
    let trailing_sep = pattern.ends_with('/') || (cfg!(windows) && pattern.ends_with('\\'));
    let mut comps = Path::new(pattern).components().peekable();
    let mut prefix_path = PathBuf::new();
    let mut has_prefix_or_root = false;
    if let Some(Component::Prefix(p)) = comps.peek().copied() {
        prefix_path.push(p.as_os_str());
        comps.next();
        has_prefix_or_root = true;
    }
    if let Some(Component::RootDir) = comps.peek().copied() {
        prefix_path.push(Component::RootDir);
        comps.next();
        has_prefix_or_root = true;
    }
    let start_dir = if has_prefix_or_root {
        std::path::absolute(&prefix_path).unwrap_or(prefix_path)
    } else {
        data_dir.to_path_buf()
    };

    let mut segs = Vec::new();
    let mut raw_count = 0usize;
    for comp in comps {
        match comp {
            Component::CurDir => continue,
            Component::ParentDir => {
                raw_count += 1;
                if raw_count > MAX_GLOB_SEGMENTS {
                    return Err(invalid_arguments(format!(
                        "glob `{pattern}` has too many path segments (max {MAX_GLOB_SEGMENTS})"
                    )));
                }
                segs.push(Seg::Parent);
            }
            Component::Normal(os) => {
                raw_count += 1;
                if raw_count > MAX_GLOB_SEGMENTS {
                    return Err(invalid_arguments(format!(
                        "glob `{pattern}` has too many path segments (max {MAX_GLOB_SEGMENTS})"
                    )));
                }
                let raw = os.to_string_lossy();
                if raw == ".." {
                    segs.push(Seg::Parent);
                } else if raw == "**" {
                    if !matches!(segs.last(), Some(Seg::GlobStar)) {
                        segs.push(Seg::GlobStar);
                    }
                } else if raw.contains("**") {
                    return Err(invalid_arguments(format!(
                        "glob `{pattern}` is invalid: recursive wildcards must form a single path component"
                    )));
                } else if raw.contains(['*', '?', '[']) {
                    segs.push(Seg::Pattern(
                        glob::Pattern::new(&raw)
                            .map_err(|e| invalid_arguments(format!("glob `{pattern}` is invalid: {e}")))?,
                    ));
                } else {
                    segs.push(Seg::Literal(os.to_os_string()));
                }
            }
            Component::Prefix(_) | Component::RootDir => {}
        }
    }
    Ok(ParsedGlob { start_dir, has_prefix_or_root, trailing_sep, segs })
}

struct StaticBase {
    dir: PathBuf,
    exists: bool,
    crossed_missing_dir: bool,
}

fn check_prefix_boundary(root: &Path, cur: &Path, has_prefix_or_root: bool) -> Result<(), Error> {
    if !has_prefix_or_root {
        outside_error(root, cur)
    } else if cur.starts_with(root) || root.starts_with(cur) {
        Ok(())
    } else {
        Err(path_outside_error(root, cur))
    }
}

fn resolve_static_prefix(
    parsed: &ParsedGlob,
    static_segs: &[Seg],
    root: Option<&Path>,
) -> Result<StaticBase, Error> {
    let Some(root) = root else {
        let mut cur = parsed.start_dir.clone();
        let mut crossed_missing_dir = false;
        for seg in static_segs {
            match seg {
                Seg::Literal(lit) => cur.push(lit),
                Seg::Parent => {
                    if !cur.exists() {
                        crossed_missing_dir = true;
                    }
                    step_unconfined_parent(&mut cur);
                }
                Seg::GlobStar | Seg::Pattern(_) => unreachable!("static prefix has no wildcards"),
            }
        }
        let exists = cur.exists();
        return Ok(StaticBase { dir: cur, exists, crossed_missing_dir });
    };

    let (mut cur, mut missing) = if !parsed.has_prefix_or_root {
        (root.to_path_buf(), false)
    } else if let Ok(canon) = std::fs::canonicalize(&parsed.start_dir) {
        (canon, false)
    } else {
        let resolved = resolve_nonexistent(&parsed.start_dir, 0)
            .ok_or_else(|| path_outside_error(root, &parsed.start_dir))?;
        let miss = !resolved.exists();
        (resolved, miss)
    };
    check_prefix_boundary(root, &cur, parsed.has_prefix_or_root)?;

    let mut crossed_missing_dir = false;
    for seg in static_segs {
        match seg {
            Seg::Literal(lit) => {
                cur.push(lit);
                if !missing {
                    if let Ok(canon) = std::fs::canonicalize(&cur) {
                        cur = canon;
                    } else if let Ok(target) = std::fs::read_link(&cur) {
                        let next = if target.is_absolute() {
                            target
                        } else {
                            cur.parent().unwrap_or(Path::new("")).join(target)
                        };
                        cur = resolve_nonexistent(&next, 1).ok_or_else(|| path_outside_error(root, &cur))?;
                        missing = !cur.exists();
                    } else if std::fs::symlink_metadata(&cur).is_ok_and(|m| m.file_type().is_symlink()) {
                        return Err(path_outside_error(root, &cur));
                    } else {
                        missing = true;
                    }
                }
                check_prefix_boundary(root, &cur, parsed.has_prefix_or_root)?;
            }
            Seg::Parent => {
                if missing {
                    crossed_missing_dir = true;
                }
                let _ = cur.pop();
                if let Ok(canon) = std::fs::canonicalize(&cur) {
                    cur = canon;
                    missing = false;
                } else {
                    missing = true;
                }
                check_prefix_boundary(root, &cur, parsed.has_prefix_or_root)?;
            }
            Seg::GlobStar | Seg::Pattern(_) => unreachable!("static prefix has no wildcards"),
        }
    }
    outside_error(root, &cur)?;
    Ok(StaticBase { dir: cur, exists: !missing, crossed_missing_dir })
}

fn is_verbatim(path: &Path) -> bool {
    cfg!(windows) && path.as_os_str().to_string_lossy().starts_with(r"\\?\")
}

fn step_unconfined_parent(cur: &mut PathBuf) {
    if is_verbatim(cur) {
        let _ = cur.pop();
    } else {
        cur.push("..");
    }
}

struct AncestorNode {
    canon: PathBuf,
    parent: Option<usize>,
}

struct Frame {
    dir: PathBuf,
    seg_idx: usize,
    ancestor: Option<usize>,
}

struct GlobWalker<'a> {
    root: Option<&'a Path>,
    pattern: &'a str,
    trailing_sep: bool,
    opts: glob::MatchOptions,
    ancestors: Vec<AncestorNode>,
    seen: HashSet<PathBuf>,
    out: Vec<PathBuf>,
}

impl GlobWalker<'_> {
    fn ancestors_contains(&self, mut idx: Option<usize>, canon: &Path) -> bool {
        while let Some(i) = idx {
            let node = &self.ancestors[i];
            if node.canon == canon {
                return true;
            }
            idx = node.parent;
        }
        false
    }

    fn push_ancestor(&mut self, canon: PathBuf, parent: Option<usize>) -> usize {
        let idx = self.ancestors.len();
        self.ancestors.push(AncestorNode { canon, parent });
        idx
    }

    /// Checks confinement (before any directory enumeration) and optional symlink-cycle protection when entering a
    /// directory. Keeps the traversed path `next` in the returned frame so duplicate symlink aliases remain distinct
    /// in `seen` and count toward [`MAX_BATCH`].
    fn enter_dir(
        &mut self,
        next: PathBuf,
        parent_anc: Option<usize>,
        check_cycle: bool,
    ) -> Result<Option<(PathBuf, Option<usize>)>, Error> {
        if let Some(root) = self.root {
            match std::fs::canonicalize(&next) {
                Ok(canon) => {
                    outside_error(root, &canon)?;
                    if !canon.is_dir() {
                        return Ok(None);
                    }
                    if check_cycle && self.ancestors_contains(parent_anc, &canon) {
                        return Ok(None);
                    }
                    let anc = self.push_ancestor(canon, parent_anc);
                    Ok(Some((next, Some(anc))))
                }
                Err(_) => {
                    match resolve_nonexistent(&next, 0) {
                        Some(target) => outside_error(root, &target)?,
                        None => return Err(path_outside_error(root, &next)),
                    }
                    Ok(None)
                }
            }
        } else {
            if !next.is_dir() {
                return Ok(None);
            }
            let Ok(canon) = std::fs::canonicalize(&next) else { return Ok(None) };
            if check_cycle && self.ancestors_contains(parent_anc, &canon) {
                return Ok(None);
            }
            let anc = self.push_ancestor(canon, parent_anc);
            Ok(Some((next, Some(anc))))
        }
    }

    fn enter_parent(
        &mut self,
        dir: &Path,
        parent_anc: Option<usize>,
    ) -> Result<Option<(PathBuf, Option<usize>)>, Error> {
        if let Some(root) = self.root {
            let mut canon = match std::fs::canonicalize(dir) {
                Ok(c) => c,
                Err(_) => resolve_nonexistent(dir, 0).ok_or_else(|| path_outside_error(root, dir))?,
            };
            let _ = canon.pop();
            outside_error(root, &canon)?;
            if !canon.is_dir() {
                return Ok(None);
            }
            let anc = self.push_ancestor(canon.clone(), parent_anc);
            Ok(Some((canon, Some(anc))))
        } else {
            let mut next = dir.to_path_buf();
            step_unconfined_parent(&mut next);
            self.enter_dir(next, parent_anc, false)
        }
    }

    fn push_file(&mut self, path: PathBuf) -> Result<(), Error> {
        if let Some(root) = self.root {
            match std::fs::canonicalize(&path) {
                Ok(canon) => {
                    outside_error(root, &canon)?;
                    if !canon.is_file() {
                        return Ok(());
                    }
                    if !self.seen.insert(path) {
                        return Ok(());
                    }
                    if self.out.len() == MAX_BATCH {
                        return Err(too_many_inputs(format!(
                            "glob `{}` matches more than {MAX_BATCH} files",
                            self.pattern
                        )));
                    }
                    self.out.push(canon);
                }
                Err(_) => {
                    // Reject broken symlinks/junctions pointing outside `data_dir` without creating an oracle.
                    if std::fs::read_link(&path).is_ok()
                        || std::fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink())
                    {
                        match resolve_nonexistent(&path, 0) {
                            Some(target) => outside_error(root, &target)?,
                            None => return Err(path_outside_error(root, &path)),
                        }
                    }
                }
            }
        } else {
            if !path.is_file() {
                return Ok(());
            }
            if !self.seen.insert(path.clone()) {
                return Ok(());
            }
            if self.out.len() == MAX_BATCH {
                return Err(too_many_inputs(format!(
                    "glob `{}` matches more than {MAX_BATCH} files",
                    self.pattern
                )));
            }
            self.out.push(path);
        }
        Ok(())
    }

    fn run(&mut self, start_dir: PathBuf, segs: &[Seg]) -> Result<(), Error> {
        let initial_anc = std::fs::canonicalize(&start_dir).ok().map(|c| self.push_ancestor(c, None));
        let mut stack = vec![Frame { dir: start_dir, seg_idx: 0, ancestor: initial_anc }];
        while let Some(frame) = stack.pop() {
            if frame.seg_idx == segs.len() {
                if !self.trailing_sep {
                    self.push_file(frame.dir)?;
                }
                continue;
            }
            let is_last = frame.seg_idx + 1 == segs.len();
            match &segs[frame.seg_idx] {
                Seg::Literal(lit) => {
                    let next = frame.dir.join(lit);
                    if is_last && !self.trailing_sep {
                        self.push_file(next)?;
                    } else if let Some((dir, anc)) = self.enter_dir(next, frame.ancestor, false)? {
                        stack.push(Frame { dir, seg_idx: frame.seg_idx + 1, ancestor: anc });
                    }
                }
                Seg::Parent => {
                    if let Some((dir, anc)) = self.enter_parent(&frame.dir, frame.ancestor)?
                        && !is_last
                    {
                        stack.push(Frame { dir, seg_idx: frame.seg_idx + 1, ancestor: anc });
                    }
                }
                Seg::Pattern(pat) => {
                    let Ok(rd) = std::fs::read_dir(&frame.dir) else { continue };
                    let mut names: Vec<OsString> = rd.filter_map(|e| e.ok().map(|e| e.file_name())).collect();
                    names.sort();
                    if is_last && !self.trailing_sep {
                        for name in names {
                            if pat.matches_with(&name.to_string_lossy(), self.opts) {
                                self.push_file(frame.dir.join(name))?;
                            }
                        }
                    } else {
                        let mut next_frames = Vec::new();
                        for name in names {
                            if !pat.matches_with(&name.to_string_lossy(), self.opts) {
                                continue;
                            }
                            let next = frame.dir.join(name);
                            if let Some((dir, anc)) = self.enter_dir(next, frame.ancestor, false)? {
                                next_frames.push(Frame { dir, seg_idx: frame.seg_idx + 1, ancestor: anc });
                            }
                        }
                        for f in next_frames.into_iter().rev() {
                            stack.push(f);
                        }
                    }
                }
                Seg::GlobStar => {
                    let Ok(rd) = std::fs::read_dir(&frame.dir) else {
                        if !is_last {
                            stack.push(Frame {
                                dir: frame.dir,
                                seg_idx: frame.seg_idx + 1,
                                ancestor: frame.ancestor,
                            });
                        }
                        continue;
                    };
                    let mut names: Vec<OsString> = rd.filter_map(|e| e.ok().map(|e| e.file_name())).collect();
                    names.sort();
                    let mut next_frames = Vec::new();
                    for name in names {
                        let next = frame.dir.join(name);
                        if next.is_dir() {
                            if let Some((dir, anc)) = self.enter_dir(next, frame.ancestor, true)? {
                                next_frames.push(Frame { dir, seg_idx: frame.seg_idx, ancestor: anc });
                            }
                        } else if is_last && !self.trailing_sep {
                            self.push_file(next)?;
                        }
                    }
                    for f in next_frames.into_iter().rev() {
                        stack.push(f);
                    }
                    if !is_last {
                        stack.push(Frame {
                            dir: frame.dir,
                            seg_idx: frame.seg_idx + 1,
                            ancestor: frame.ancestor,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

fn path_outside_error(root: &Path, path: &Path) -> Error {
    Error::new(
        StatusCode::FORBIDDEN,
        "path_outside_data_dir",
        format!(
            "{} is outside the data dir {}",
            folo_api::json::path(path),
            folo_api::json::path(root)
        ),
    )
    .hint("`folo mcp --confine` only reads inside --data-dir; copy the files there or restart without --confine")
}

fn outside_error(root: &Path, canon: &Path) -> Result<(), Error> {
    if canon.starts_with(root) {
        return Ok(());
    }
    Err(path_outside_error(root, canon))
}

/// `too_many_inputs`.
pub fn too_many_inputs(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "too_many_inputs", message)
        .hint(format!("split the images into batches of at most {MAX_BATCH}"))
}

#[cfg(test)]
#[path = "paths_tests.rs"]
mod tests;
