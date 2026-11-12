//! The process-wide read guard (`folo_data::guard`). A separate test binary, because a guard stays installed for the
//! whole process: here it refuses any path with a `forbidden` component and any `classes.txt`.

use std::path::Path;

use folo_data::{Image, ImageFormat, open};
use folo_test_support::TestDir;

fn tmp(name: &str) -> TestDir {
    TestDir::new(&format!("guard-{name}"))
}

fn image(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    folo_data::write_image(&Image::new(16, 12, [200, 10, 10]), path, ImageFormat::Png).unwrap();
}

#[test]
fn refused_reads_are_reported_and_not_descended() {
    assert!(folo_data::guard::install(|p: &Path| {
        !p.components().any(|c| c.as_os_str() == "forbidden")
            && p.file_name().is_none_or(|n| n != "classes.txt")
    }));
    assert!(folo_data::guard::installed());
    let root = tmp("walk");
    image(&root.join("images/train/ok.png"));
    std::fs::create_dir_all(root.join("labels/train")).unwrap();
    std::fs::write(root.join("labels/train/ok.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
    for i in 0..5 {
        image(&root.join(format!("images/train/forbidden/deep/{i}.png")));
    }
    std::fs::write(root.join("classes.txt"), "secret\n").unwrap();

    let ds = open(&root).unwrap();
    assert_eq!(ds.train.entries().len(), 1, "only the allowed image");
    let msgs: Vec<&str> = ds.report.issues.iter().map(|i| i.message.as_str()).collect();
    let folders = msgs.iter().filter(|m| m.starts_with("cannot read folder")).count();
    assert_eq!(folders, 1, "the refused folder is reported once: {msgs:?}");
    assert!(!msgs.iter().any(|m| m.starts_with("cannot read image:")), "not descended: {msgs:?}");
    assert!(msgs.iter().any(|m| m.starts_with("cannot read class names")), "{msgs:?}");
    assert!(msgs.iter().all(|m| !m.contains("secret")));
    assert!(!ds.classes.contains(&"secret".to_string()), "{:?}", ds.classes);
    assert!(msgs.iter().filter(|m| m.starts_with("cannot")).all(|m| m.contains(folo_data::guard::DENIED)));
    // A refused file fails like an unreadable one.
    let e = folo_data::read_image(&root.join("images/train/forbidden/deep/0.png")).unwrap_err();
    assert!(e.to_string().contains(folo_data::guard::DENIED), "{e}");

    // With a guard installed the scan cache is neither read nor written, even for a folder large enough to cache.
    let big = tmp("cache");
    let cache = tmp("cache-dir");
    for i in 0..folo_data::SCAN_CACHE_MIN_FILES + 4 {
        image(&big.join(format!("images/train/{i}.png")));
    }
    std::fs::create_dir_all(big.join("labels/train")).unwrap();
    folo_data::set_scan_cache_dir(Some(cache.to_path_buf()));
    let ds = open(&big).unwrap();
    folo_data::set_scan_cache_dir(None);
    assert_eq!(ds.train.entries().len(), folo_data::SCAN_CACHE_MIN_FILES + 4);
    assert_eq!(std::fs::read_dir(&*cache).map(|r| r.count()).unwrap_or(0), 0, "nothing cached under a guard");
}
