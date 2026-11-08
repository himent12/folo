//! With the `embed-web` feature, compiles the built FOLO Studio (`web/dist`, or `$FOLO_WEB_DIST`) into the binary:
//! writes `$OUT_DIR/web_files.rs`, a path-sorted table of `include_bytes!` entries with precomputed weak ETags.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=FOLO_WEB_DIST");
    if std::env::var_os("CARGO_FEATURE_EMBED_WEB").is_none() {
        return;
    }
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let repo = manifest.join("../..");
    // Relative `FOLO_WEB_DIST` paths are resolved against the repository root.
    let dist = std::env::var_os("FOLO_WEB_DIST").map_or_else(|| repo.join("web/dist"), |d| repo.join(d));
    let dist = std::path::absolute(&dist).expect("absolute path");
    if !dist.join("index.html").is_file() {
        panic!(
            "the `embed-web` feature needs a built FOLO Studio, but {} has no index.html. Run `npm ci && npm run \
             build` in web/, or point FOLO_WEB_DIST at a built dist folder.",
            dist.display()
        );
    }
    // The Studio still works without folo-wasm, but its Playground can then only run models on the server.
    if !["folo_wasm.js", "folo_wasm_bg.wasm"].iter().all(|f| dist.join("folo-wasm").join(f).is_file()) {
        println!(
            "cargo:warning={} has no folo-wasm/: the embedded Studio cannot run models in the browser. Run              scripts/build-wasm.sh (or .ps1) before `npm run build`.",
            dist.display()
        );
    }
    // A directory makes cargo rescan every file under it for changes.
    println!("cargo:rerun-if-changed={}", dist.display());

    let mut files = Vec::new();
    collect(&dist, &dist, &mut files);
    files.sort();
    let mut out = String::from("&[\n");
    for (rel, abs) in &files {
        let bytes = std::fs::read(abs).unwrap_or_else(|e| panic!("reading {}: {e}", abs.display()));
        let abs = abs.to_str().unwrap_or_else(|| panic!("non UTF-8 path {}", abs.display()));
        writeln!(
            out,
            "    EmbeddedFile {{ path: {rel:?}, bytes: include_bytes!({abs:?}), etag: \"W/\\\"{:016x}\\\"\" }},",
            fnv1a(&bytes)
        )
        .unwrap();
    }
    out.push(']');
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("set by cargo"));
    std::fs::write(out_dir.join("web_files.rs"), out).expect("write web_files.rs");
}

/// Every file under `dir`, as (`/`-separated path relative to `root`, absolute path).
fn collect(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(root, &path, files);
        } else {
            let rel = path.strip_prefix(root).expect("under root");
            let rel: Vec<&str> = rel
                .components()
                .map(|c| {
                    c.as_os_str().to_str().unwrap_or_else(|| panic!("non UTF-8 path {}", path.display()))
                })
                .collect();
            files.push((rel.join("/"), path));
        }
    }
}

/// 64-bit FNV-1a: a cheap content hash for ETags.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}
