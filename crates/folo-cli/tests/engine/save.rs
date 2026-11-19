//! `predict --save`: annotated copies never replace or re-read a source image.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{folo, s, tiny_run};

#[test]
fn predict_save_keeps_the_sources() {
    let tiny = tiny_run();
    let (code, _, stderr) = &tiny.train;
    assert_eq!(*code, 0, "{stderr}");
    let root = &tiny.root;
    let best = tiny.run.join("weights/best.folo");
    let images = tiny.data.join("images/val");
    // --save never overwrites a source, and same-named sources get distinct outputs.
    let (code, lines, _) =
        folo(&["predict", "--json", "--model", s(&best), s(&images), "--save", s(&images)]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");
    let other = root.join("other");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::copy(images.join("000008.png"), other.join("000008.png")).unwrap();
    let both = root.join("pred2");
    let (code, _, stderr) = folo(&[
        "predict",
        "--json",
        "--model",
        s(&best),
        s(&images.join("000008.png")),
        s(&other),
        "--save",
        s(&both),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(both.join("000008.png").is_file() && both.join("000008_2.png").is_file());
    // A --save folder inside a source folder is never read as input: run twice, it reads and writes what a --save
    // folder outside does (`..` spells the same folder).
    let nested = root.join("nested");
    std::fs::create_dir_all(nested.join("sub")).unwrap();
    std::fs::copy(images.join("000008.png"), nested.join("a.png")).unwrap();
    std::fs::copy(images.join("000009.png"), nested.join("sub/b.png")).unwrap();
    let twice = |save: &Path, again: &Path| {
        [save, again].map(|dir| {
            let (code, lines, stderr) =
                folo(&["predict", "--json", "--model", s(&best), s(&nested), "--save", s(dir)]);
            assert_eq!(code, 0, "{stderr}");
            let rel = |l: &Value, k: &str, base: &Path| {
                let p = PathBuf::from(l[k].as_str().unwrap());
                p.strip_prefix(base).unwrap_or(&p).to_string_lossy().replace('\\', "/")
            };
            let preds = lines.iter().filter(|l| l["event"] == "prediction");
            preds.map(|l| (rel(l, "file", &nested), rel(l, "saved", dir))).collect::<Vec<_>>()
        })
    };
    let outside = twice(&root.join("pred4"), &root.join("pred4"));
    let inside = twice(&nested.join("out"), &nested.join("sub/../out"));
    let want = vec![("a.png".to_string(), "a.png".to_string()), ("sub/b.png".into(), "sub/b.png".into())];
    assert_eq!(outside, [want.clone(), want]);
    assert_eq!(inside, outside);
    // A --save destination hard-linked to a source is replaced, never written through.
    let linked = root.join("linked");
    std::fs::create_dir_all(&linked).unwrap();
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::copy(images.join("000009.png"), src.join("h.png")).unwrap();
    std::fs::hard_link(src.join("h.png"), linked.join("h.png")).unwrap();
    let before = std::fs::read(src.join("h.png")).unwrap();
    let (code, _, stderr) = folo(&["predict", "--json", "--model", s(&best), s(&src), "--save", s(&linked)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(std::fs::read(src.join("h.png")).unwrap(), before, "the source stays byte-identical");
    // Same-named sources that differ only in case get distinct outputs too.
    let upper = root.join("upper");
    std::fs::create_dir_all(&upper).unwrap();
    std::fs::copy(images.join("000009.png"), upper.join("A.png")).unwrap();
    let lower = root.join("lower");
    std::fs::create_dir_all(&lower).unwrap();
    std::fs::copy(images.join("000009.png"), lower.join("a.png")).unwrap();
    let cased = root.join("pred3");
    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&best), s(&upper), s(&lower), "--save", s(&cased)]);
    assert_eq!(code, 0, "{stderr}");
    let saved: Vec<String> =
        lines.iter().filter_map(|l| l["saved"].as_str().map(str::to_lowercase)).collect();
    assert_eq!(saved.len(), 2);
    assert_ne!(saved[0], saved[1]);
}
