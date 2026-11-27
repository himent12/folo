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
        let p = big.join(format!("images/train/{i}.png"));
        image(&p);
        // An hour old: old enough to cache, so only the guard keeps the cache out.
        let f = std::fs::File::options().write(true).open(&p).unwrap();
        f.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(3600)).unwrap();
    }
    std::fs::create_dir_all(big.join("labels/train")).unwrap();
    folo_data::set_scan_cache_dir(Some(cache.to_path_buf()));
    let ds = open(&big).unwrap();
    folo_data::set_scan_cache_dir(None);
    assert_eq!(ds.train.entries().len(), folo_data::SCAN_CACHE_MIN_FILES + 4);
    assert_eq!(std::fs::read_dir(&*cache).map(|r| r.count()).unwrap_or(0), 0, "nothing cached under a guard");

    // Pascal VOC also obeys the read guard under both verify_images=true and verify_images=false.
    let voc = tmp("voc");
    image(&voc.join("JPEGImages/ok.png"));
    image(&voc.join("forbidden/secret.png"));
    std::fs::create_dir_all(voc.join("Annotations")).unwrap();
    std::fs::write(
        voc.join("Annotations/ok.xml"),
        "<annotation><filename>ok.png</filename><size><width>16</width><height>12</height></size>\
         <object><name>box</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>",
    )
    .unwrap();
    std::fs::write(
        voc.join("Annotations/bad.xml"),
        "<annotation><folder>forbidden</folder><filename>secret.png</filename><size><width>16</width><height>12</height></size>\
         <object><name>box</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>",
    )
    .unwrap();
    for verify in [true, false] {
        let ds = folo_data::open_with(
            &voc,
            &folo_data::OpenOptions { format: folo_data::Format::Voc, verify_images: verify },
        )
        .unwrap();
        assert_eq!(ds.train.entries().len(), 1, "forbidden VOC image must not load (verify={verify})");
    }
    // `convert` also refuses to copy a forbidden source file.
    let forbidden_ds = folo_data::LoadedDataset {
        format: folo_data::Format::Yolo,
        root: voc.to_path_buf(),
        classes: vec!["x".into()],
        train: folo_data::FileDataset::new(
            folo_data::Split::Train,
            vec!["x".into()],
            vec![folo_data::ImageEntry {
                image: voc.join("forbidden/secret.png"),
                label: None,
                size: Some((16, 12)),
                boxes: vec![],
            }],
        ),
        val: folo_data::FileDataset::new(folo_data::Split::Val, vec!["x".into()], vec![]),
        test: folo_data::FileDataset::new(folo_data::Split::Test, vec!["x".into()], vec![]),
        report: Default::default(),
        sources: vec![],
    };
    let out = tmp("voc-out");
    let err = folo_data::convert::convert(&forbidden_ds, out.path(), folo_data::Format::Yolo).unwrap_err();
    assert!(err.to_string().contains(folo_data::guard::DENIED), "{err}");
}
