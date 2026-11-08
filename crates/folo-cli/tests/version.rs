//! The workspace version (`[workspace.package] version` in the root Cargo.toml) is the single source of truth:
//! everything else that carries a version must match it. The release workflow also checks the tag against it.

use std::path::Path;

fn repo() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// The first `"version": "..."` (package.json) or `version = "..."` (Cargo.toml `[package]`) value.
fn first_version(text: &str) -> Option<&str> {
    text.lines().map(str::trim).find_map(|l| {
        let rest = l.strip_prefix("\"version\"").or_else(|| l.strip_prefix("version"))?;
        let rest = rest.trim_start().strip_prefix([':', '='])?.trim_start().strip_prefix('"')?;
        rest.split_once('"').map(|(v, _)| v)
    })
}

#[test]
fn versions_match_the_workspace() {
    let want = env!("CARGO_PKG_VERSION");
    let web = std::fs::read_to_string(repo().join("web/package.json")).unwrap();
    assert_eq!(first_version(&web), Some(want), "web/package.json: bump it with the workspace version");
    // FOLO Desktop is its own cargo workspace, so it cannot inherit the version.
    if let Ok(desktop) = std::fs::read_to_string(repo().join("apps/desktop/Cargo.toml")) {
        let package =
            desktop.split_once("[package]").map(|(_, p)| p).expect("[package] in apps/desktop/Cargo.toml");
        assert_eq!(
            first_version(package),
            Some(want),
            "apps/desktop/Cargo.toml: bump it with the workspace version"
        );
    }
}

#[test]
fn version_parser() {
    assert_eq!(first_version("{\n  \"name\": \"x\",\n  \"version\": \"1.2.3\",\n}"), Some("1.2.3"));
    assert_eq!(first_version("name = \"x\"\nversion = \"0.4.0\"\n"), Some("0.4.0"));
    assert_eq!(first_version("version.workspace = true\n"), None);
}
