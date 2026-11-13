//! DFL evidence fusion against the Python post-process that chose it (`bench/fusion/post.py`), on three COCO
//! val2017 images of folo-n-coco's pre-NMS candidates in `tests/fixtures/fusion/dfl.json` (written by
//! `bench/fusion/make_fixture.py`, which also explains the rounding that makes both sides start from the same f32
//! inputs). Python computes the side statistics in f64, Rust in f32: the fused edges agree to a few f32 ulps.

use folo_vision::decode::side_precisions;
use folo_vision::fusion::BoxFusion;
use folo_vision::nms::{NmsConfig, non_max_suppression, non_max_suppression_with};
use folo_vision::{Detection, Xyxy};
use serde_json::Value;
use std::path::Path;

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn dfl_fusion_matches_the_python_reference() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fusion/dfl.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (rm, nc) = (fx["reg_max"].as_u64().unwrap() as usize, fx["num_classes"].as_u64().unwrap() as usize);
    let c = &fx["cfg"];
    let f = &c["fuse"];
    assert_eq!((f["kind"].as_str(), f["a"].as_f64(), f["b"].as_f64()), (Some("var"), Some(1.0), Some(1.0)));
    assert_eq!(
        (f["eps"].as_f64(), f["excess"].as_bool()),
        (Some(0.1), Some(true)),
        "the shipped DFL weights"
    );
    let cfg = NmsConfig {
        conf: c["conf"].as_f64().unwrap() as f32,
        iou: c["iou"].as_f64().unwrap() as f32,
        max_det: c["max_det"].as_u64().unwrap() as usize,
        multi_label: true,
        fusion: BoxFusion::Dfl,
        fusion_iou: f["tau"].as_f64().unwrap() as f32,
        ..NmsConfig::default()
    };
    let (mut moved, mut kept) = (0, 0);
    for im in fx["images"].as_array().unwrap() {
        let file = im["file"].as_str().unwrap();
        let anchors: Vec<Vec<f32>> = im["anchors"].as_array().unwrap().iter().map(f32s).collect();
        let boxes: Vec<Xyxy> = im["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let b = f32s(b);
                Xyxy::new(b[0], b[1], b[2], b[3])
            })
            .collect();
        let prec: Vec<[f32; 4]> = im["logits"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&anchors)
            .map(|(l, a)| side_precisions(&f32s(l), rm, a[2]))
            .collect();
        let mut scores = vec![0f32; boxes.len() * nc];
        for p in im["pairs"].as_array().unwrap() {
            let p = p.as_array().unwrap();
            let (a, k) = (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize);
            scores[a * nc + k] = p[2].as_f64().unwrap() as f32;
        }
        let want: Vec<Detection> = im["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                let e = f32s(e);
                Detection { bbox: Xyxy::new(e[0], e[1], e[2], e[3]), score: e[4], class: e[5] as u32 }
            })
            .collect();
        let got = non_max_suppression_with(&boxes, &scores, Some(&prec), nc, &cfg);
        let plain =
            non_max_suppression(&boxes, &scores, nc, &NmsConfig { fusion: BoxFusion::None, ..cfg.clone() });
        assert_eq!(got.len(), want.len(), "{file}: detections");
        for ((g, w), p) in got.iter().zip(&want).zip(&plain) {
            assert_eq!(
                (g.score, g.class),
                (w.score, w.class),
                "{file}: NMS keeps the same candidates in order"
            );
            for (x, y) in g.bbox.to_array().into_iter().zip(w.bbox.to_array()) {
                assert!(
                    (x - y).abs() <= 1e-4 * (1.0 + y.abs()),
                    "{file}: {:?} vs Python {:?}",
                    g.bbox,
                    w.bbox
                );
            }
            moved += usize::from(g.bbox != p.bbox);
        }
        kept += got.len();
    }
    // Most kept boxes have other members in their cluster and move; the rest keep their box exactly.
    assert!(kept >= 40 && moved * 2 > kept, "{moved} of {kept} kept boxes moved");
}
