//! The hints of a dataset registration that failed honour the process-wide read guard (`folo mcp --confine`): a
//! folder or YAML it refuses is not looked into to phrase the error (review round 3). A separate test binary,
//! because a guard stays installed for the whole process: here it refuses any `forbidden` folder and any
//! `custom.yaml`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::{Options, RealStore, Server};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

#[tokio::test]
async fn a_failed_registration_does_not_read_what_the_guard_refuses() {
    assert!(folo_data::guard::install(|p: &Path| {
        let in_confined_root = p.components().any(|c| c.as_os_str() == "confined_root");
        let in_confined_parent = p.components().any(|c| c.as_os_str() == "confined_parent");
        (!in_confined_parent || in_confined_root)
            && !p.components().any(|c| c.as_os_str() == "forbidden")
            && p.file_name().is_none_or(|n| n != "custom.yaml")
    }));
    let t = TestDir::new("confined-diagnostics");
    let home = t.join("home");
    let yaml = "path: .\ntrain: images\nnames: [x]\n";
    for (dir, file) in [("a/child", "custom.yaml"), ("b/forbidden", "data.yaml"), ("c/open", "other.yaml")] {
        std::fs::create_dir_all(t.join(dir).join("images")).unwrap();
        std::fs::write(t.join(dir).join(file), yaml).unwrap();
    }
    let store = RealStore::open(home).recheck_every(Duration::ZERO);
    let app = Server::new(Arc::new(store), Options::default()).router();
    let mut said = Vec::new();
    for parent in ["a", "b", "c"] {
        let req = Request::post("/api/datasets")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "path": t.join(parent) }).to_string()))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST, "{parent}");
        let body: Value =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        said.push(body["error"]["message"].as_str().unwrap().replace('\\', "/"));
    }
    let names_inside = |m: &str| m.ends_with("inside it holds a dataset");
    assert!(!names_inside(&said[0]), "a refused YAML is not read: {}", said[0]);
    assert!(!names_inside(&said[1]), "a refused folder is not looked into: {}", said[1]);
    assert!(names_inside(&said[2]) && said[2].contains("c/open"), "an allowed one is: {}", said[2]);

    // Dangling-link probing must also honour the read guard (including broken links inside a refused folder)
    // and never overwrite a guard-denied issue.
    struct LinkGuard(std::path::PathBuf);
    impl Drop for LinkGuard {
        fn drop(&mut self) {
            #[cfg(windows)]
            let _ = std::fs::remove_dir(&self.0);
            #[cfg(unix)]
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let forbidden_link = t.join("b").join("forbidden").join("dead");
    // Construct the cleanup guard before creating the junction/symlink so unwinding kills/reaps `mklink` first
    // and then removes the junction before `TestDir` drops.
    let link_guard = LinkGuard(forbidden_link.clone());
    #[cfg(windows)]
    {
        let target = t.join("dead-target");
        std::fs::create_dir_all(&target).unwrap();
        let win = |p: &Path| p.to_string_lossy().replace('/', "\\");
        let mut cmd = std::process::Command::new("cmd");
        cmd.args(["/d", "/c", "mklink", "/j"])
            .arg(win(&forbidden_link))
            .arg(win(&target))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut child = folo_test_support::ChildGuard::new(cmd.spawn().expect("spawn mklink /j"));
        let start = std::time::Instant::now();
        let status = loop {
            if let Some(st) = child.try_wait().unwrap() {
                break st;
            }
            assert!(start.elapsed() < Duration::from_secs(15), "mklink /j timed out");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success());
        std::fs::remove_dir(&target).unwrap();
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(t.join("dead-target"), &forbidden_link).unwrap();

    assert_eq!(folo_server::dangling_link_message(&forbidden_link), None);
    let fp_with_link = folo_server::Fingerprint::of(std::slice::from_ref(&forbidden_link), &[]);
    assert_eq!((fp_with_link.files, fp_with_link.bytes), (0, 0));
    let mut report = folo_data::check::CheckReport {
        root: t.join("c").join("open"),
        format: folo_data::Format::Yolo,
        classes: vec!["x".into()],
        splits: [0, 0, 0],
        issues: vec![folo_data::check::CheckIssue {
            kind: folo_data::check::CheckKind::MissingImage,
            file: forbidden_link.clone(),
            line: None,
            message: format!("cannot read: {}", folo_data::guard::DENIED),
            fix: None,
        }],
    };
    folo_server::enrich_dangling_symlinks(&t.join("c").join("open"), &mut report);
    assert!(report.issues[0].message.contains(folo_data::guard::DENIED));

    // End-to-end confined registration whose manifest references a refused dangling split: `Index::build`
    // registers cleanly without probing or emitting a dangling-link diagnostic for the refused folder.
    let allowed_ds = t.join("d").join("allowed");
    std::fs::create_dir_all(allowed_ds.join("images").join("train")).unwrap();
    std::fs::create_dir_all(allowed_ds.join("labels").join("train")).unwrap();
    image::RgbImage::from_pixel(16, 16, image::Rgb([10, 20, 30]))
        .save(allowed_ds.join("images").join("train").join("a.png"))
        .unwrap();
    std::fs::write(allowed_ds.join("labels").join("train").join("a.txt"), "0 0.5 0.5 0.4 0.4\n").unwrap();
    std::fs::write(
        allowed_ds.join("data.yaml"),
        "path: .\ntrain: images/train\nval: ../../b/forbidden/dead\nnames: [x]\n",
    )
    .unwrap();
    let req = Request::post("/api/datasets")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({ "path": allowed_ds }).to_string()))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let ds: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let issues_req = Request::get(format!("/api/datasets/{}/issues?limit=100", ds["id"].as_str().unwrap()))
        .body(Body::empty())
        .unwrap();
    let issues_res = app.clone().oneshot(issues_req).await.unwrap();
    assert_eq!(issues_res.status(), StatusCode::OK);
    let issues_body: Value =
        serde_json::from_slice(&issues_res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert!(
        issues_body["issues"]
            .as_array()
            .unwrap()
            .iter()
            .all(|i| !i["message"].as_str().unwrap_or("").contains("dangling symlink")),
        "refused forbidden link must not be probed during registration: {issues_body}"
    );

    // A denied descendant under an allowed watch root is never enumerated or hashed by `Fingerprint::of`.
    let denied_sub = allowed_ds.join("forbidden");
    std::fs::create_dir_all(&denied_sub).unwrap();
    let fp_before_denied = folo_server::Fingerprint::of(std::slice::from_ref(&allowed_ds), &[]);
    std::fs::write(denied_sub.join("secret.txt"), "top-secret-contents").unwrap();
    let fp_after_denied = folo_server::Fingerprint::of(std::slice::from_ref(&allowed_ds), &[]);
    assert_eq!(
        fp_before_denied, fp_after_denied,
        "denied descendant must not affect allowed root fingerprint"
    );

    // Multi-level absent path under a refused folder is also rejected without probing.
    let deep_missing = t.join("b").join("forbidden").join("missing_a").join("missing_b");
    let fp_deep = folo_server::Fingerprint::of(&[deep_missing], &[]);
    assert_eq!((fp_deep.files, fp_deep.bytes, fp_deep.hash.as_str()), (0, 0, "cbf29ce484222325"));

    // A dataset registered directly at a confinement root (allowed itself while its parent `confined_parent` is
    // refused) still fingerprints its files and refreshes when labels change.
    let confined_root = t.join("confined_parent").join("confined_root");
    std::fs::create_dir_all(confined_root.join("images").join("train")).unwrap();
    std::fs::create_dir_all(confined_root.join("labels").join("train")).unwrap();
    image::RgbImage::from_pixel(16, 16, image::Rgb([40, 50, 60]))
        .save(confined_root.join("images").join("train").join("a.png"))
        .unwrap();
    let lbl = confined_root.join("labels").join("train").join("a.txt");
    std::fs::write(&lbl, "0 0.5 0.5 0.4 0.4\n").unwrap();
    std::fs::write(confined_root.join("data.yaml"), "path: .\ntrain: images/train\nnames: [x]\n").unwrap();
    let fp_root = folo_server::Fingerprint::of(std::slice::from_ref(&confined_root), &[]);
    assert!(fp_root.files >= 3, "confinement root must be fingerprinted: {fp_root:?}");
    let req = Request::post("/api/datasets")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({ "path": confined_root }).to_string()))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::CREATED);
    let root_ds: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let root_issues_uri = format!("/api/datasets/{}/issues?limit=100", root_ds["id"].as_str().unwrap());
    let before_res =
        app.clone().oneshot(Request::get(&root_issues_uri).body(Body::empty()).unwrap()).await.unwrap();
    let before_body: Value =
        serde_json::from_slice(&before_res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(before_body["total"], 0, "{before_body}");
    std::fs::write(&lbl, "99 0.5 0.5 0.4 0.4\n").unwrap();
    let after_res =
        app.clone().oneshot(Request::get(&root_issues_uri).body(Body::empty()).unwrap()).await.unwrap();
    let after_body: Value =
        serde_json::from_slice(&after_res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert!(
        after_body["issues"].as_array().unwrap().iter().any(|i| i["kind"] == "class_overflow"),
        "editing a dataset at the confinement root must invalidate its fingerprint: {after_body}"
    );

    // Fingerprinting a refused path never probes it: removing the forbidden link leaves the fingerprint identical.
    drop(link_guard);
    let fp_without_link = folo_server::Fingerprint::of(&[forbidden_link], &[]);
    assert_eq!(fp_with_link, fp_without_link);
}
