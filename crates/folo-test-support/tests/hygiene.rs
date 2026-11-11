//! Temp-dir hygiene across the workspace: tests and test helpers make their scratch folders with
//! `folo_test_support::TestDir` (removed on drop, under `FOLO_TEST_TMP`), never with `std::env::temp_dir()` directly,
//! which is how thousands of `folo-*` folders once piled up in %TEMP%. The binaries do not write to the temp dir
//! either. FOLO Desktop's sources (`apps/desktop/src`, and folo-ui's `src`, `tests` and `examples`) are scanned too.
//! Benchmarks (`benches/`) are not run by `cargo test` and are not checked.

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn nothing_but_test_support_uses_the_system_temp_dir() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates = manifest.parent().unwrap();
    let mut files = Vec::new();
    for c in std::fs::read_dir(crates).unwrap().flatten() {
        if c.path() == manifest {
            continue;
        }
        for sub in ["src", "tests"] {
            rust_files(&c.path().join(sub), &mut files);
        }
    }
    rust_files(&crates.join("../apps/desktop/src"), &mut files);
    // folo-ui, FOLO Desktop's UI framework: its tests and examples write PNGs too.
    for sub in ["src", "tests", "examples"] {
        rust_files(&crates.join("../apps/desktop/ui").join(sub), &mut files);
    }
    assert!(files.len() > 100, "{} files", files.len());
    let mut bad = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        for (i, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            if code.contains("env::temp_dir") || code.contains("temp_dir()") {
                bad.push(format!("{}:{}: {}", f.display(), i + 1, code));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "use folo_test_support::TestDir (removed on drop, under FOLO_TEST_TMP) instead of the system temp dir:\n{}",
        bad.join("\n")
    );
}
