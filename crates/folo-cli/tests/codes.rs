//! One error-code vocabulary (`folo_api::codes`): every code the CLI, the server and the MCP server construct is
//! listed there, and the CLI exits with the listed exit code for it. Scans the sources, so a new code cannot be
//! added without documenting it.

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `(prefix variant, code)` for every `<prefix>::<Variant>, "<code>"` in `text` (a status or exit code followed by
/// a string literal, as in `Error::new(StatusCode::NOT_FOUND, "run_not_found", ...)`).
fn literal_codes<'a>(text: &'a str, prefix: &str) -> Vec<(&'a str, &'a str)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(prefix) {
        rest = &rest[i + prefix.len()..];
        let variant_end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len());
        let variant = &rest[..variant_end];
        let after = rest[variant_end..].trim_start();
        let Some(after) = after.strip_prefix(',') else { continue };
        let Some(lit) = after.trim_start().strip_prefix('"') else { continue };
        let Some(end) = lit.find('"') else { continue };
        let code = &lit[..end];
        if !code.is_empty() && code.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            out.push((variant, code));
        }
    }
    out
}

#[test]
fn every_error_code_is_in_the_shared_vocabulary() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files = Vec::new();
    for c in ["folo-cli", "folo-server", "folo-mcp"] {
        rust_files(&crates.join(c).join("src"), &mut files);
    }
    assert!(files.len() > 20);
    let mut seen = 0;
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        for (variant, code) in literal_codes(&text, "ExitCode::") {
            let entry = folo_api::codes::find(code)
                .unwrap_or_else(|| panic!("{}: `{code}` is not in folo_api::codes::CODES", f.display()));
            let exit = match variant {
                "Runtime" => 1,
                "Usage" => 2,
                "NotFound" => 3,
                "NotImplemented" => 4,
                other => panic!("{}: unexpected ExitCode::{other}", f.display()),
            };
            assert_eq!(
                exit,
                entry.exit,
                "{}: `{code}` exits {exit}, the vocabulary says {}",
                f.display(),
                entry.exit
            );
            seen += 1;
        }
        for (_, code) in literal_codes(&text, "StatusCode::") {
            assert!(
                folo_api::codes::find(code).is_some(),
                "{}: `{code}` is not in folo_api::codes",
                f.display()
            );
            seen += 1;
        }
    }
    assert!(seen > 50, "the scan found {seen} codes");
    // `Error::not_found(what, id)` builds `<what>_not_found`.
    for what in ["dataset", "image", "model", "run", "path", "device"] {
        assert_eq!(folo_api::codes::exit_code(&format!("{what}_not_found")), 3, "{what}");
    }
    // The engine's codes (folo_vision::Error::code) are in the vocabulary too.
    for code in [
        "invalid_config",
        "invalid_dataset",
        "data_error",
        "io_error",
        "invalid_weights",
        "internal",
        "not_implemented",
        "device_not_found",
        "training_failed",
        "gpu_out_of_memory",
    ] {
        assert!(folo_api::codes::find(code).is_some(), "{code}");
    }
}
