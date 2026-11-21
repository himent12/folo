//! Regression tests for `folo predict`, `folo export` and `folo val --coco`.

use std::path::{Path, PathBuf};

use folo_test_support::TestDir;

use super::{folo, s, tiny_run};

/// A directory link (junction on Windows, symlink elsewhere) removed on drop so `TestDir` cleans up even on failure.
struct DirLink(PathBuf);

impl DirLink {
    fn new(target: &Path, link: PathBuf) -> Self {
        #[cfg(windows)]
        {
            let native = |p: &Path| p.to_string_lossy().replace('/', "\\");
            let ok = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(native(&link))
                .arg(native(target))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            assert!(ok, "mklink /J {} -> {}", link.display(), target.display());
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, &link).unwrap();
        }
        DirLink(link)
    }
}

impl Drop for DirLink {
    fn drop(&mut self) {
        if std::fs::remove_dir(&self.0).is_err() {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// `folo predict` on a folder containing a self-referencing directory link visits each real directory once
/// (instead of looping until the OS symlink/reparse limit and emitting duplicate predictions) while still following
/// a non-cyclic link to an external folder.
#[test]
fn predict_skips_directory_link_cycles() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let src_img = tiny.data.join("images/val").read_dir().unwrap().next().unwrap().unwrap().path();

    let root = TestDir::new("cli-predict-cycle");
    let input = root.join("input");
    let external = root.join("external");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::create_dir_all(&external).unwrap();
    std::fs::copy(&src_img, input.join("a.png")).unwrap();
    std::fs::copy(&src_img, external.join("b.png")).unwrap();

    // Self-loop plus a legitimate non-cyclic link to an external folder.
    let _self_loop = DirLink::new(&input, input.join("loop"));
    let _ext_link = DirLink::new(&external, input.join("extra"));

    let (code, lines, stderr) = folo(&["predict", "--json", "--model", s(&best), s(&input)]);
    assert_eq!(code, 0, "{stderr}");
    let preds: Vec<&str> = lines
        .iter()
        .filter(|l| l["event"] == "prediction")
        .map(|l| l["file"].as_str().unwrap())
        .collect();
    assert_eq!(preds.len(), 2, "expected 2 images (a.png and extra/b.png), got {preds:?}");
    assert!(preds.iter().any(|p| p.ends_with("/a.png")), "{preds:?}");
    assert!(preds.iter().any(|p| p.ends_with("/extra/b.png")), "{preds:?}");
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["images"].as_u64()), (Some("done"), Some(2)), "{done}");
}
