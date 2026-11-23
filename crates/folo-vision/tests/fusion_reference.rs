//! DFL evidence fusion against the Python post-process that chose it (`bench/fusion/post.py`), on three COCO
//! val2017 images of folo-n-coco's pre-NMS candidates in `tests/fixtures/fusion/dfl.json` (written by
//! `bench/fusion/make_fixture.py`, which also explains the rounding that makes both sides start from the same f32
//! inputs). Python computes the side statistics in f64, Rust in f32: the fused edges agree to a few f32 ulps.

use folo_vision::decode::{dfl_entropy, side_precisions};
use folo_vision::fusion::{BoxFusion, ScoreRule};
use folo_vision::nms::{
    DflInputs, NmsConfig, non_max_suppression, non_max_suppression_with, non_max_suppression_with_inputs,
};
use folo_vision::{Detection, Xyxy};
use serde_json::Value;
use std::path::Path;

/// Largest edge difference allowed, in pixels: a few f32 ulps at 640 px (one ulp there is 6.1e-5); measured
/// 1.2e-4 on this fixture. Switching fusion to voting or dropping the excess-variance term moves edges by far more.
const TOL_PX: f32 = 5e-4;

/// Largest calibrated score difference allowed against the Python `dfl-repulse` reference.
const TOL_SCORE: f32 = 1e-6;

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
    let (mut moved, mut kept, mut worst) = (0, 0, 0f32);
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
                worst = worst.max((x - y).abs());
                assert!((x - y).abs() <= TOL_PX, "{file}: {:?} vs Python {:?}", g.bbox, w.bbox);
            }
            moved += usize::from(g.bbox != p.bbox);
        }
        kept += got.len();
    }
    eprintln!("largest edge difference to Python: {worst} px");
    // Most kept boxes have other members in their cluster and move; the rest keep their box exactly.
    assert!(kept >= 40 && moved * 2 > kept, "{moved} of {kept} kept boxes moved");
}

/// Exact Rust-vs-Python parity for `--fusion dfl --score dfl-repulse --iou 0.75 --max-det 1000` on the same
/// three COCO val2017 images of `tests/fixtures/fusion/dfl.json`: every kept detection's class, box (`<= 5e-4 px`)
/// and calibrated score (`<= 1e-6`) matches the frozen Python M5 reference.
#[test]
fn dfl_repulse_matches_the_python_reference() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fusion/dfl.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (rm, nc) = (fx["reg_max"].as_u64().unwrap() as usize, fx["num_classes"].as_u64().unwrap() as usize);
    let c = &fx["cfg"];
    let cfg = NmsConfig {
        conf: c["conf"].as_f64().unwrap() as f32,
        iou: 0.75,
        max_det: 1000,
        multi_label: true,
        fusion: BoxFusion::Dfl,
        fusion_iou: 0.75,
        score: ScoreRule::DflRepulse,
        ..NmsConfig::default()
    };

    // Exact Python M5 outputs for the three fixture images (x1, y1, x2, y2, score, class).
    type RefDet = (f32, f32, f32, f32, f32, u32);
    #[allow(clippy::excessive_precision)]
    let expected: [&[RefDet]; 3] = [
        &[
            (508.3872375, 276.6311951, 632.9579468, 492.5585327, 0.973019898, 0),
            (4.3451824, 273.4728699, 132.7720947, 503.7139893, 0.972018600, 0),
            (324.9312744, 277.0073547, 396.3174133, 478.7844543, 0.970328271, 0),
            (195.9366913, 341.5190430, 248.8914948, 378.5188293, 0.440317512, 24),
            (560.6228638, 192.6216431, 640.0706787, 258.1475525, 0.340157747, 25),
            (197.0288086, 333.9904785, 331.4683533, 476.6983643, 0.314795107, 13),
            (614.0337524, 339.3630981, 639.9287109, 450.1457825, 0.248473451, 0),
            (201.5565796, 337.2600708, 329.2350464, 478.5894775, 0.219694510, 56),
            (195.8702545, 341.3739014, 248.8583984, 378.0798950, 0.182494536, 26),
            (195.7952118, 341.4703979, 250.0895386, 403.4356689, 0.108378761, 24),
            (405.4588318, 406.9896545, 477.8724670, 511.6605835, 0.094018266, 28),
            (613.3322754, 355.4914246, 638.5066528, 424.1752014, 0.083486676, 26),
            (340.8660278, 153.5316620, 399.3879089, 211.7525024, 0.063934259, 74),
            (341.9975281, 153.6253052, 399.0078430, 212.4410553, 0.058084648, 62),
            (244.1990051, 332.4130554, 315.7249756, 403.5101929, 0.050210450, 24),
            (610.2299805, 253.9573669, 639.8767700, 450.0840454, 0.039270408, 0),
            (204.3683014, 334.6593323, 335.2193909, 477.6380005, 0.029215721, 0),
            (208.4700012, 445.0100098, 321.2600098, 477.1600037, 0.022601116, 36),
            (613.2037354, 355.3684998, 638.2828979, 425.8721924, 0.022448346, 24),
            (195.7438202, 341.4328003, 250.0107117, 406.2144775, 0.019684166, 26),
            (560.1832275, 192.2819366, 639.9777832, 295.4413147, 0.017866151, 25),
            (612.4833984, 342.8845825, 639.6206665, 448.6596069, 0.015973492, 26),
            (239.1100006, 332.7300110, 329.8999939, 410.8500061, 0.008051805, 24),
            (611.0300293, 338.8200073, 636.7600098, 423.6700134, 0.005387030, 26),
            (609.9498901, 213.7965088, 639.9055786, 448.4272461, 0.005015565, 0),
            (195.3192902, 341.1908569, 250.8391418, 412.6004944, 0.002850828, 24),
            (204.9102478, 347.5767517, 325.6566162, 478.4989929, 0.002102235, 56),
            (342.4243774, 154.3614044, 399.2331848, 210.4182587, 0.000492255, 74),
        ],
        &[
            (5.5376320, 377.0771790, 146.4838867, 458.7733459, 0.915931642, 71),
            (412.4145203, 335.0265198, 494.6354675, 544.2676392, 0.887071609, 61),
            (412.0383911, 276.4487000, 434.3331604, 302.6594543, 0.257374287, 41),
            (410.8568420, 302.4238586, 517.3116455, 544.2573242, 0.204996228, 61),
            (434.2158508, 275.6415710, 459.9141541, 302.8966980, 0.064020969, 41),
            (170.6542358, 417.9700928, 297.7208252, 516.1129150, 0.056798011, 71),
            (3.9337161, 376.4734192, 150.3435059, 566.6389771, 0.052918296, 71),
            (246.6100006, 64.7651749, 266.1454773, 88.3590393, 0.027687110, 39),
            (600.7199707, 410.1199951, 621.5000000, 432.9599915, 0.015118836, 41),
            (573.8900146, 353.6199951, 616.8099976, 427.6099854, 0.013223280, 41),
            (409.5605164, 243.0462036, 527.7831421, 545.2243042, 0.008149276, 61),
        ],
        &[
            (293.2990417, 472.8362732, 396.9448547, 522.9143677, 0.417168260, 61),
            (452.1244202, 417.5357971, 531.7926636, 584.2854614, 0.396679282, 61),
            (449.1851501, 417.4895020, 531.1779785, 513.5562134, 0.205618501, 61),
            (461.3136597, 479.7970276, 532.1689453, 584.9437256, 0.202997029, 61),
            (189.2500153, 216.5764465, 304.1752014, 568.0764160, 0.158786088, 61),
            (189.6020355, 217.7227936, 396.4281616, 562.2718506, 0.107072569, 61),
            (171.1305542, 24.4086647, 292.6241150, 579.2158813, 0.097487710, 61),
            (173.6510773, 24.9437504, 262.7403259, 291.4940186, 0.049355850, 61),
            (188.5467377, 242.1905975, 263.7703552, 290.2644348, 0.029484289, 45),
            (294.8999939, 471.8699951, 396.8999939, 523.4799805, 0.025859917, 71),
            (244.6435394, 223.5466156, 396.6704102, 530.2377319, 0.023838885, 61),
            (251.8600006, 212.4100037, 308.2999878, 479.0700073, 0.012472454, 61),
            (172.9654999, 25.9724083, 308.0517578, 573.0411377, 0.002671542, 61),
        ],
    ];

    let (mut worst_px, mut worst_score) = (0f32, 0f32);
    for (im, want_list) in fx["images"].as_array().unwrap().iter().zip(expected) {
        let file = im["file"].as_str().unwrap();
        let anchors: Vec<Vec<f32>> = im["anchors"].as_array().unwrap().iter().map(f32s).collect();
        let strides: Vec<f32> = anchors.iter().map(|a| a[2]).collect();
        let boxes: Vec<Xyxy> = im["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let b = f32s(b);
                Xyxy::new(b[0], b[1], b[2], b[3])
            })
            .collect();
        let logits: Vec<Vec<f32>> = im["logits"].as_array().unwrap().iter().map(f32s).collect();
        let prec: Vec<[f32; 4]> =
            logits.iter().zip(&strides).map(|(l, &st)| side_precisions(l, rm, st)).collect();
        let ent: Vec<f32> = logits.iter().map(|l| dfl_entropy(l, rm)).collect();
        let mut scores = vec![0f32; boxes.len() * nc];
        for p in im["pairs"].as_array().unwrap() {
            let p = p.as_array().unwrap();
            let (a, k) = (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize);
            scores[a * nc + k] = p[2].as_f64().unwrap() as f32;
        }
        let dfl = DflInputs {
            precision: Some(&prec),
            entropy: Some(&ent),
            strides: Some(&strides),
            ..Default::default()
        };
        let got = non_max_suppression_with_inputs(&boxes, &scores, dfl, nc, &cfg);
        assert_eq!(got.len(), want_list.len(), "{file}: detection count");
        for (i, (g, &(x1, y1, x2, y2, sc, cls))) in got.iter().zip(want_list).enumerate() {
            assert_eq!(g.class, cls, "{file}[{i}]: class");
            let ds = (g.score - sc).abs();
            worst_score = worst_score.max(ds);
            assert!(ds <= TOL_SCORE, "{file}[{i}]: score {} vs Python {sc} (diff {ds:e})", g.score);
            for (x, y) in g.bbox.to_array().into_iter().zip([x1, y1, x2, y2]) {
                let dp = (x - y).abs();
                worst_px = worst_px.max(dp);
                assert!(dp <= TOL_PX, "{file}[{i}]: box {:?} vs Python [{x1},{y1},{x2},{y2}]", g.bbox);
            }
        }
    }
    eprintln!("dfl-repulse max diff vs Python: score {worst_score:e}, box {worst_px:e} px");
}
