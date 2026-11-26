use super::*;
use folo_test_support::TestDir;

fn tmp(name: &str) -> TestDir {
    TestDir::new(&format!("mcp-paths-{name}"))
}

#[cfg(windows)]
fn make_junction(link: &Path, target: &Path) {
    let status = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(link)
        .arg(target)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("run mklink /J");
    assert!(status.success(), "mklink /J {} {} failed", link.display(), target.display());
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
        let p = PathPolicy { data_dir: canon_plain.clone(), confine: false };
        let matched = p.expand_glob("sub/*.png").unwrap();
        assert_eq!(matched, vec![want_plain.clone()]);
        assert_eq!(matched[0].to_string_lossy(), want_plain.to_string_lossy());

        // `..` under a verbatim `\\?\` data dir works both unconfined and confined.
        std::fs::write(plain.join("top.png"), b"x").unwrap();
        let want_top = canon_plain.join("top.png");
        assert_eq!(p.expand_glob("sub/../top.png").unwrap(), vec![want_top]);
        let confined_verbatim = PathPolicy { data_dir: canon_plain, confine: true };
        assert_eq!(confined_verbatim.expand_glob("../*.png").unwrap_err().body.code, "path_outside_data_dir");

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
        std::os::unix::fs::symlink(root.join("outside_missing.png"), data.join("broken_out.png")).unwrap();
        assert_eq!(p.resolve_for_batch("broken_out.png").err().unwrap().body.code, "path_outside_data_dir");

        // An in-tree symlink loop must return a clean error rather than panicking when `data_dir` is canonical.
        let canon_p = PathPolicy { data_dir: std::fs::canonicalize(&data).unwrap(), confine: true };
        std::os::unix::fs::symlink(data.join("loop_b"), data.join("loop_a")).unwrap();
        std::os::unix::fs::symlink(data.join("loop_a"), data.join("loop_b")).unwrap();
        assert_eq!(canon_p.resolve_for_batch("loop_a").err().unwrap().body.code, "path_outside_data_dir");
    }

    #[cfg(windows)]
    {
        // `..` on Windows must keep resolving case and directory junctions for subsequent components.
        let cased = root.join("MixedData");
        std::fs::create_dir_all(cased.join("sub")).unwrap();
        let outside = root.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        make_junction(&cased.join("escape"), &outside);
        // Broken junction pointing to an outside folder that is then removed.
        let gone = root.join("gone_outside");
        std::fs::create_dir_all(&gone).unwrap();
        make_junction(&cased.join("broken_escape"), &gone);
        std::fs::remove_dir(&gone).unwrap();

        let wp = PathPolicy { data_dir: cased.clone(), confine: true };
        assert!(matches!(wp.resolve_for_batch("../mixeddata/missing.png").unwrap(), BatchPath::Missing(_)));
        assert_eq!(
            wp.resolve_for_batch("sub/../escape/missing.png").err().unwrap().body.code,
            "path_outside_data_dir"
        );
        assert_eq!(
            wp.resolve_for_batch("broken_escape/missing.png").err().unwrap().body.code,
            "path_outside_data_dir"
        );
        // Cleanup junctions before TestDir drops.
        let _ = std::fs::remove_dir(cased.join("escape"));
        let _ = std::fs::remove_dir(cased.join("broken_escape"));
    }
}

#[test]
fn confined_glob_enforces_boundaries_before_enumeration() {
    let root = tmp("glob-boundary");
    let data = root.join("data");
    let outside = root.join("outside");
    std::fs::create_dir_all(data.join("sub")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(data.join("sub/a.png"), b"x").unwrap();
    let p = PathPolicy { data_dir: data.clone(), confine: true };

    // Outside globs fail with `path_outside_data_dir` even when no files match or the directory is missing.
    assert_eq!(p.expand_glob("../*.nope").unwrap_err().body.code, "path_outside_data_dir");
    assert_eq!(p.expand_glob("../outside/*.nope").unwrap_err().body.code, "path_outside_data_dir");
    assert_eq!(p.expand_glob("../nonexistent/*.png").unwrap_err().body.code, "path_outside_data_dir");
    assert_eq!(p.expand_glob("sub/../../outside/*.nope").unwrap_err().body.code, "path_outside_data_dir");
    assert_eq!(p.expand_glob("*/../../outside/*.nope").unwrap_err().body.code, "path_outside_data_dir");
    let abs_out = format!("{}/*.nope", folo_api::json::path(&outside));
    assert_eq!(p.expand_glob(&abs_out).unwrap_err().body.code, "path_outside_data_dir");
    let abs_missing = format!("{}/nonexistent/*.png", folo_api::json::path(&outside));
    assert_eq!(p.expand_glob(&abs_missing).unwrap_err().body.code, "path_outside_data_dir");

    // Directory link pointing outside is rejected before enumerating, even with a non-matching pattern.
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&outside, data.join("escape_dir")).unwrap();
        assert_eq!(p.expand_glob("**/*.nope").unwrap_err().body.code, "path_outside_data_dir");
        assert_eq!(p.expand_glob("*/*.nope").unwrap_err().body.code, "path_outside_data_dir");
    }
    #[cfg(windows)]
    {
        make_junction(&data.join("escape_dir"), &outside);
        assert_eq!(p.expand_glob("**/*.nope").unwrap_err().body.code, "path_outside_data_dir");
        assert_eq!(p.expand_glob("*/*.nope").unwrap_err().body.code, "path_outside_data_dir");
        let _ = std::fs::remove_dir(data.join("escape_dir"));

        // Root-relative Windows patterns are not silently treated as relative to `data_dir`.
        assert_eq!(p.expand_glob("/sub/*.png").unwrap_err().body.code, "path_outside_data_dir");
        assert_eq!(p.expand_glob(r"\sub\*.png").unwrap_err().body.code, "path_outside_data_dir");
    }
}

#[test]
fn glob_rejects_over_deep_patterns_and_embedded_globstar() {
    let root = tmp("glob-limits");
    let p = PathPolicy { data_dir: root.to_path_buf(), confine: false };
    let deep_parent = format!("{}*.png", "../".repeat(MAX_GLOB_SEGMENTS + 10));
    assert_eq!(p.expand_glob(&deep_parent).unwrap_err().body.code, "invalid_arguments");
    let deep_sub = format!("{}*.png", "a/".repeat(MAX_GLOB_SEGMENTS + 10));
    assert_eq!(p.expand_glob(&deep_sub).unwrap_err().body.code, "invalid_arguments");
    assert_eq!(p.expand_glob("**.png").unwrap_err().body.code, "invalid_arguments");
    assert_eq!(p.expand_glob("a**b/*.png").unwrap_err().body.code, "invalid_arguments");
}

#[test]
fn terminal_globstar_matches_files_identically_for_relative_and_absolute() {
    let root = tmp("glob-star");
    std::fs::create_dir_all(root.join("sub/nested")).unwrap();
    std::fs::write(root.join("sub/a.png"), b"x").unwrap();
    std::fs::write(root.join("sub/nested/b.png"), b"x").unwrap();
    let abs_pat = format!("{}/sub/**", folo_api::json::path(root.path()));
    for confine in [false, true] {
        let p = PathPolicy { data_dir: root.to_path_buf(), confine };
        let rel = p.expand_glob("sub/**").unwrap();
        let abs = p.expand_glob(&abs_pat).unwrap();
        assert_eq!(rel.len(), 2, "confine={confine}: {rel:?}");
        assert_eq!(abs.len(), 2, "confine={confine}: {abs:?}");
        let rel_canon: Vec<_> = rel.iter().map(|x| std::fs::canonicalize(x).unwrap()).collect();
        let abs_canon: Vec<_> = abs.iter().map(|x| std::fs::canonicalize(x).unwrap()).collect();
        assert_eq!(rel_canon, abs_canon, "confine={confine}");
    }
}

#[cfg(unix)]
#[test]
fn glob_cycle_and_alias_semantics() {
    let root = tmp("glob-cycle-alias");
    std::fs::create_dir_all(root.join("a/sub")).unwrap();
    std::fs::write(root.join("a/x.png"), b"x").unwrap();
    // Symlink inside `a/sub` pointing back to `a`: unconfined `a/**/*.png` must not duplicate `a/x.png`.
    std::os::unix::fs::symlink(root.join("a"), root.join("a/sub/loop")).unwrap();
    let unconfined = PathPolicy { data_dir: root.to_path_buf(), confine: false };
    let matched = unconfined.expand_glob("a/**/*.png").unwrap();
    assert_eq!(matched.len(), 1, "cycle should not duplicate matches: {matched:?}");

    // Symlink alias inside `data_dir` stays distinct for both confined and unconfined globs.
    let alias_dir = root.join("aliases");
    std::fs::create_dir_all(&alias_dir).unwrap();
    std::fs::write(alias_dir.join("orig.png"), b"x").unwrap();
    std::os::unix::fs::symlink(alias_dir.join("orig.png"), alias_dir.join("dup.png")).unwrap();
    for confine in [false, true] {
        let p = PathPolicy { data_dir: alias_dir.clone(), confine };
        assert_eq!(p.expand_glob("*.png").unwrap().len(), 2, "confine={confine}");
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
