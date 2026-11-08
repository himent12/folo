//! Path helpers shared by `folo dataset check --json` and the server, so both name files the same way.

use std::path::{Component, Path, PathBuf};

/// `p` with `.` and `..` components resolved lexically (no file system access, links are not followed).
pub fn lexical(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// `p` made absolute against the working directory and [`lexical`]ly normalised.
pub fn normalize(p: &Path) -> PathBuf {
    lexical(&std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()))
}

/// A path as FOLO shows it to people and agents (messages, hints, logs): `/` separators on Windows (which accepts
/// them) and no `\\?\` verbatim prefix. The same rule as `folo_api::json::path`.
pub fn shown(p: impl AsRef<Path>) -> String {
    let s = p.as_ref().to_string_lossy();
    let s = s
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .unwrap_or_else(|| s.strip_prefix(r"\\?\").unwrap_or(&s).to_string());
    if cfg!(windows) { s.replace('\\', "/") } else { s }
}

/// How a dataset file is reported: relative to the dataset `root` with `/` separators, or the [`normalize`]d
/// absolute path when it lies outside the root.
pub fn relative_to(file: &Path, root: &Path) -> String {
    let (file, root) = (normalize(file), normalize(root));
    file.strip_prefix(&root).unwrap_or(&file).to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shown_paths_use_forward_slashes() {
        assert_eq!(shown("runs/a/best.folo"), "runs/a/best.folo");
        if cfg!(windows) {
            assert_eq!(shown(r"D:\data\images/x.jpg"), "D:/data/images/x.jpg");
            assert_eq!(shown(r"\\?\D:\data"), "D:/data");
            assert_eq!(shown(r"\\?\UNC\server\share\x"), "//server/share/x");
        }
    }

    #[test]
    fn lexical_and_relative() {
        assert_eq!(lexical(Path::new("a/./b/../c")), PathBuf::from("a/c"));
        assert_eq!(lexical(Path::new("../x")), PathBuf::from("../x"));
        assert_eq!(relative_to(Path::new("d/images/../images/a.jpg"), Path::new("./d")), "images/a.jpg");
        let outside = relative_to(Path::new("elsewhere/a.jpg"), Path::new("d"));
        assert!(Path::new(&outside).is_absolute(), "{outside}");
    }
}
