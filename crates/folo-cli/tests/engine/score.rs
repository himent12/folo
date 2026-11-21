//! `--score consensus` through the CLI: `val` records the rule, `predict` keeps the boxes and only lowers scores.

use super::{folo, s, tiny_run};

#[test]
fn score_consensus_on_val_and_predict() {
    let tiny = tiny_run();
    let (data, best) = (&tiny.data, tiny.run.join("weights/best.folo"));
    let val =
        ["val", "--json", "--model", s(&best), "--data", s(data), "--workers", "1", "--score", "consensus"];
    let (code, lines, stderr) = folo(&val);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["score"], "consensus");

    // The same boxes as without it, none with a higher score, and at least one lower. A low `conf`, so that this
    // one-epoch model has detections to compare.
    let images = data.join("images/val");
    let boxes_of = |extra: &[&str]| {
        let mut args = vec!["predict", "--json", "--model", s(&best), s(&images), "--conf", "0.001"];
        args.extend_from_slice(extra);
        let (code, lines, stderr) = folo(&args);
        assert_eq!(code, 0, "{stderr}");
        let mut out: Vec<(String, f64)> = lines
            .iter()
            .filter(|l| l["event"] == "prediction")
            .flat_map(|p| {
                let file = p["file"].to_string();
                p["boxes"].as_array().unwrap().iter().map(move |b| {
                    let edges = ["x1", "y1", "x2", "y2", "class"].map(|k| b[k].to_string()).join(",");
                    (format!("{file} {edges}"), b["score"].as_f64().unwrap())
                })
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    };
    let (plain, consensus) = (boxes_of(&[]), boxes_of(&["--score", "consensus"]));
    assert!(!plain.is_empty(), "no detections to compare");
    assert_eq!(plain.len(), consensus.len());
    for (p, c) in plain.iter().zip(&consensus) {
        assert_eq!(p.0, c.0, "the same box");
        assert!(c.1 <= p.1, "{c:?} against {p:?}");
    }
    assert!(plain.iter().zip(&consensus).any(|(p, c)| c.1 < p.1), "the rule changed no score");
}
