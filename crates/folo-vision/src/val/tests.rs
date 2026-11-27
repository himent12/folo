use super::*;
use folo_data::Split;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_nn::io::ModelConfig;

/// The evaluator's boxes are clipped to the letterboxed image, not to the input: a 640×480 image at 640 has 80
/// pixels of padding above and below, and a box reaching into it ends at the image's edge.
#[test]
fn evaluator_boxes_are_clipped_to_the_image() {
    let meta = folo_data::ImageMeta { index: 0, orig_size: [320, 240], ratio: [2.0, 2.0], pad: [0.0, 80.0] };
    let d = |x1, y1, x2, y2| Detection { bbox: Xyxy::new(x1, y1, x2, y2), score: 0.5, class: 1 };
    let got = clipped_to_image(
        &[d(-5.0, 70.0, 100.0, 200.0), d(10.0, 90.0, 650.0, 600.0), d(1.0, 81.0, 2.0, 82.0)],
        &meta,
    );
    assert_eq!(got, [d(0.0, 80.0, 100.0, 200.0), d(10.0, 90.0, 640.0, 560.0), d(1.0, 81.0, 2.0, 82.0)]);
    // As the COCO path clips: the clipped box maps back to the clipped original-image box.
    for (g, o) in got.iter().zip([d(-5.0, 70.0, 100.0, 200.0), d(10.0, 90.0, 650.0, 600.0)]) {
        let back = |b: &Xyxy| meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
        assert_eq!(back(&g.bbox), back(&o.bbox));
    }
}

/// Validation's metrics score the clipped box: a prediction that matches its truth inside the image but reaches
/// 20 pixels into the padding is a perfect detection (unclipped, its IoU would be 0.86 and mAP50-95 0.8).
#[test]
fn validation_scores_the_clipped_box() {
    let (_, ds) = untrained(Device::Cpu);
    let most = AtomicUsize::new(0);
    let mut post = Post {
        ev: Evaluator::new(1),
        samples: Vec::new(),
        kept: Vec::new(),
        index: 0,
        most: &most,
        from_pairs: 0,
    };
    let meta = folo_data::ImageMeta { index: 0, orig_size: [320, 240], ratio: [2.0, 2.0], pad: [0.0, 80.0] };
    let truth = Truth { bbox: Xyxy::new(0.0, 80.0, 100.0, 200.0), class: 0 };
    let images = Outputs::Host(vec![(vec![Xyxy::new(0.0, 60.0, 100.0, 200.0)], vec![0.9])]);
    let d = Decoded {
        images,
        aux: DflAuxOutputs::Host(DflAux::default()),
        nc: 1,
        mode: HeadMode::O2m,
        truths: vec![vec![truth]],
        meta: vec![meta],
    };
    let cfg = ValConfig { samples: 0, ..ValConfig::new(640, 1) };
    post.add(d, &*ds, &cfg).unwrap();
    assert_eq!(post.ev.compute().map50_95, 1.0);
}

/// The pipelined [`validate`] (post-processing on a second thread) gives exactly what the serial loop gives:
/// predict, then score each image in order. With conf 0 an untrained model keeps a box at every anchor, so NMS,
/// the metrics, the samples and the kept detections all have work to do.
#[test]
fn validate_matches_the_serial_loop() {
    let (model, ds) = untrained(Device::Cpu);
    check_against_serial(&model, &ds, 0.0, BoxFusion::None, crate::fusion::ScoreRule::None);
    // DFL fusion + DFL-repulse: stats read from logits on the host, as `predict_raw`'s full decode reads them.
    check_against_serial(&model, &ds, 0.0, BoxFusion::Dfl, crate::fusion::ScoreRule::None);
    check_against_serial(&model, &ds, 0.0, BoxFusion::Dfl, crate::fusion::ScoreRule::DflRepulse);
}

/// The same on a GPU, where the other thread also downloads the decoded outputs and DFL stats: at conf 0 the
/// whole decode ([`Outputs::Device`]); at validation's conf 0.001, the device candidate pairs
/// ([`Outputs::Pairs`]), in f32 and f16, with DFL fusion and DFL-repulse scoring.
#[test]
fn validate_matches_the_serial_loop_on_gpu() {
    let Some(device) = gpu() else { return };
    let (mut model, ds) = untrained(device);
    let conf = NmsConfig::val().conf;
    for (fusion, score) in [
        (BoxFusion::None, crate::fusion::ScoreRule::None),
        (BoxFusion::Dfl, crate::fusion::ScoreRule::None),
        (BoxFusion::Dfl, crate::fusion::ScoreRule::DflRepulse),
    ] {
        assert_eq!(
            check_against_serial(&model, &ds, 0.0, fusion, score),
            0,
            "conf 0 downloads the whole decode"
        );
        assert_eq!(
            check_against_serial(&model, &ds, conf, fusion, score),
            10,
            "every image from its device pairs"
        );
    }
    if folo_core::backend::computes_f16(device) {
        model.to_dtype(DType::F16).unwrap();
        assert_eq!(
            check_against_serial(&model, &ds, conf, BoxFusion::None, crate::fusion::ScoreRule::None),
            10,
            "f16: every image from its pairs"
        );
        assert_eq!(
            check_against_serial(&model, &ds, conf, BoxFusion::Dfl, crate::fusion::ScoreRule::DflRepulse),
            10,
            "f16, DFL fusion + DFL-repulse"
        );
    } else {
        eprintln!("note: skipping the f16 case, the GPU does not compute in f16");
    }
}

/// A threshold outside 0..=1 (NaN included) is a config error before any image is read, the fusion cluster IoU
/// too: NaN would put every same-class candidate in one cluster, a value above 1 would switch fusion off.
#[test]
fn validate_rejects_thresholds_outside_the_unit_range() {
    let (model, ds) = untrained(Device::Cpu);
    // (conf, iou, fusion_iou)
    for (conf, iou, fusion_iou) in
        [(f32::NAN, 0.7, 0.75), (0.001, 1.5, 0.75), (0.001, 0.7, f32::NAN), (0.001, 0.7, 1.5)]
    {
        let mut vc = ValConfig::new(64, 4);
        vc.nms.fusion = BoxFusion::Dfl;
        (vc.nms.conf, vc.nms.iou, vc.nms.fusion_iou) = (conf, iou, fusion_iou);
        let got = validate(&model, ds.clone(), &vc);
        assert!(matches!(got, Err(Error::Config(_))), "conf {conf} iou {iou} fusion_iou {fusion_iou}");
    }
}

/// The pairs path's capacity: none once the pairs would not be clearly smaller than the decode, else at least
/// [`MIN_PAIRS`], KiB-rounded, and never a region larger than half the decode.
#[test]
fn pair_capacity_is_bounded_by_half_the_decode() {
    // COCO at 640: 8400 anchors, 80 classes, half the decode is 352_800 words.
    assert_eq!(pair_capacity(0, 8400, 80), Some(MIN_PAIRS));
    assert_eq!(pair_capacity(20_000, 8400, 80), Some(25 * 1024));
    assert_eq!(pair_capacity(94_000, 8400, 80), Some(117_599)); // 1 + 3 · 117_599 = 352_798: the limit
    assert_eq!(pair_capacity(94_100, 8400, 80), None);
    // 400 classes at a low conf: a capacity for every pair would be three times the decode.
    assert_eq!(pair_capacity(8400 * 400, 8400, 400), None);
    // Tiny heads: the limit, and none when not even one record fits.
    assert_eq!(pair_capacity(0, 84, 2), Some(83));
    assert_eq!(pair_capacity(0, 1, 1), None);
    for (most, a, nc) in [(0, 84, 2), (60, 84, 2), (1000, 8400, 400), (300_000, 8400, 80)] {
        if let Some(cap) = pair_capacity(most, a, nc) {
            assert!(cap >= 1 && 2 * (1 + 3 * cap) <= (4 + nc) * a, "{most} {a} {nc}: {cap}");
        }
    }
}

/// The device candidate pairs give exactly the full-row NMS: random decodes with scores at and around `conf`,
/// NaNs and many overlapping boxes, with room for every pair and with too little room (the full fallback).
#[test]
fn device_pairs_match_full_rows_nms() {
    let Some(device) = gpu() else { return };
    let (b, nc, a) = (3usize, 5usize, 700usize);
    let conf = 0.25f32;
    let mut rng = folo_data::Rng::new(11);
    let mut host = vec![0f32; b * (4 + nc) * a];
    for i in 0..b {
        for j in 0..a {
            let at = |r: usize| (i * (4 + nc) + r) * a + j;
            let (x, y) = (rng.uniform(0.0, 60.0), rng.uniform(0.0, 60.0));
            let (w, h) = (rng.uniform(1.0, 30.0), rng.uniform(1.0, 30.0));
            host[at(0)] = x;
            host[at(1)] = y;
            host[at(2)] = x + w;
            host[at(3)] = y + h;
            for c in 0..nc {
                host[at(4 + c)] = match (j + c) % 23 {
                    0 => conf,
                    1 => f32::NAN,
                    2 => f32::from_bits(conf.to_bits() + 1),
                    _ => rng.uniform(0.0, 1.0),
                };
            }
        }
    }
    // Image 1 has few pairs, so a capacity between its count and the others' overflows images 0 and 2 only.
    for j in 0..a {
        for c in 0..nc {
            let v = &mut host[(4 + nc + 4 + c) * a + j];
            if j % 9 != 0 {
                *v = 0.0;
            }
        }
    }
    let decoded = Tensor::from_vec(host.clone(), [b, 4 + nc, a]).unwrap().to_device(device).unwrap();
    let nms = NmsConfig { conf, max_det: 40, max_candidates: 900, ..NmsConfig::val() };
    let want: Vec<Vec<Detection>> = (0..b)
        .map(|i| {
            let mut rows = Vec::with_capacity(a * (4 + nc));
            for j in 0..a {
                rows.extend((0..4 + nc).map(|r| host[(i * (4 + nc) + r) * a + j]));
            }
            crate::nms::non_max_suppression_rows(&rows, nc, &nms)
        })
        .collect();
    assert!(want.iter().all(|d| !d.is_empty()));
    let counts: Vec<usize> = (0..b)
        .map(|i| host[(i * (4 + nc) + 4) * a..][..nc * a].iter().filter(|&&s| s > conf).count())
        .collect();
    assert!(counts[1] < 1024 && counts[0] > 1024 && counts[2] > 1024, "{counts:?}");
    let most = AtomicUsize::new(0);
    let boxes = decoded.narrow(1, 0, 4).unwrap();
    // DFL fusion with arbitrary side precisions: the pairs path still gives the full-row result.
    let dfl = NmsConfig { fusion: BoxFusion::Dfl, fusion_iou: 0.5, ..nms.clone() };
    let prec: Vec<Vec<[f32; 4]>> =
        (0..b).map(|_| (0..a).map(|_| std::array::from_fn(|_| rng.uniform(0.01, 4.0))).collect()).collect();
    let want_dfl: Vec<Vec<Detection>> = (0..b)
        .map(|i| {
            let mut rows = Vec::with_capacity(a * (4 + nc));
            for j in 0..a {
                rows.extend((0..4 + nc).map(|r| host[(i * (4 + nc) + r) * a + j]));
            }
            crate::nms::non_max_suppression_rows_with(&rows, Some(&prec[i]), nc, &dfl)
        })
        .collect();
    assert!(want_dfl.iter().zip(&want).any(|(f, w)| f.iter().zip(w).any(|(f, w)| f.bbox != w.bbox)));
    // The consensus score: the pairs path gives the full-row result, which is not the plain one.
    let consensus = NmsConfig { score: crate::fusion::ScoreRule::Consensus, ..nms.clone() };
    let want_consensus: Vec<Vec<Detection>> = (0..b)
        .map(|i| {
            let mut rows = Vec::with_capacity(a * (4 + nc));
            for j in 0..a {
                rows.extend((0..4 + nc).map(|r| host[(i * (4 + nc) + r) * a + j]));
            }
            crate::nms::non_max_suppression_rows(&rows, nc, &consensus)
        })
        .collect();
    assert_ne!(want_consensus, want, "the rule moved no score");
    let empty_aux = DflAux::default();
    let dfl_aux = DflAux { precision: Some(prec), ..DflAux::default() };
    // Room for every pair, for image 1's only, for none: how many images are served from their pairs.
    for (cap, served) in [(a * nc, 3), (1024, 1), (200, 0)] {
        let mut post = Post {
            ev: Evaluator::new(nc),
            samples: Vec::new(),
            kept: Vec::new(),
            index: 0,
            most: &most,
            from_pairs: 0,
        };
        let pairs = folo_gpu::detect::multi_label_candidates(&decoded, conf, cap).unwrap();
        let words = pairs.to_vec::<u32>().unwrap();
        let region = folo_gpu::detect::candidate_region_words(cap);
        assert_eq!((0..b).map(|i| words[i * region] as usize).collect::<Vec<_>>(), counts, "cap {cap}");
        let got = post.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &nms, &empty_aux).unwrap();
        assert_eq!(got, want, "cap {cap}");
        assert_eq!(post.from_pairs, served, "cap {cap}");
        let got = post.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &dfl, &dfl_aux).unwrap();
        assert_eq!(got, want_dfl, "cap {cap}, DFL fusion");
        let got = post.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &consensus, &empty_aux).unwrap();
        assert_eq!(got, want_consensus, "cap {cap}, consensus");
    }
    assert_eq!(most.load(Ordering::Relaxed), *counts.iter().max().unwrap());
}

/// The registered GPU, or `None` (with a note) when there is none or only a software adapter.
fn gpu() -> Option<Device> {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    Some(folo_gpu::init().unwrap())
}

/// An untrained folo-n in eval mode on `device` and 10 synthetic images of its 2 classes.
fn untrained(device: Device) -> (FoloModel, Arc<dyn Dataset>) {
    let cfg = SyntheticConfig {
        images: 10,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 5,
        val_fraction: 0.0,
        ..SyntheticConfig::default()
    };
    let ds: Arc<dyn Dataset> = Arc::new(SyntheticDataset::new(cfg, Split::Train).unwrap());
    let model = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
    let model = crate::train::place_model(model, device).unwrap();
    model.eval();
    (model, ds)
}

/// [`validate`] of `model` on `ds` at `conf` (the sample conf too) with `fusion` and `score` against the serial
/// loop; returns how many of the images it post-processed from device candidate pairs.
fn check_against_serial(
    model: &FoloModel,
    ds: &Arc<dyn Dataset>,
    conf: f32,
    fusion: BoxFusion,
    score: crate::fusion::ScoreRule,
) -> usize {
    let (device, dtype) = (model.device(), model.dtype());
    let mut vc = ValConfig { samples: 5, keep_detections: true, workers: 2, ..ValConfig::new(64, 4) };
    vc.nms.fusion = fusion;
    vc.nms.score = score;
    // Untrained scores sit near the class prior: conf 0 keeps everything, so NMS has every anchor to sort out.
    (vc.nms.conf, vc.sample_conf) = (conf, conf);
    let (got, from_pairs) = validate_counted(model, ds.clone(), &vc).unwrap();

    let lc = LoaderConfig { workers: 0, u8_images: !device.is_cpu(), ..LoaderConfig::val(4, 64) };
    let loader = DataLoader::new(ds.clone(), lc).unwrap();
    let mut ev = Evaluator::new(2);
    let (mut samples, mut kept, mut index) = (Vec::new(), Vec::new(), 0);
    for batch in loader.epoch(0) {
        let mut batch = batch.unwrap();
        let truths = batch_truths(&batch);
        let x = batch_tensor(&mut batch, device, dtype).unwrap();
        let pred = model.predict_raw(&x, &vc.nms).unwrap();
        for (i, img) in pred.images.iter().enumerate() {
            let meta = &batch.meta[i];
            ev.add_image(&clipped_to_image(&img.detections, meta), &truths[i]);
            let unmap = |b: &Xyxy| {
                let [x1, y1, x2, y2] = meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
                Xyxy::new(x1, y1, x2, y2)
            };
            if index < vc.samples {
                samples.push(ValSample {
                    index: meta.index,
                    file: None,
                    width: meta.orig_size[0],
                    height: meta.orig_size[1],
                    truth: truths[i].iter().map(|t| Truth { bbox: unmap(&t.bbox), class: t.class }).collect(),
                    pred: img
                        .detections
                        .iter()
                        .filter(|d| d.score >= vc.sample_conf)
                        .map(|d| Detection { bbox: unmap(&d.bbox), ..*d })
                        .collect(),
                });
            }
            kept.push((
                meta.index,
                img.detections.iter().map(|d| Detection { bbox: unmap(&d.bbox), ..*d }).collect(),
            ));
            index += 1;
        }
    }
    let boxes = kept.iter().map(|k: &(usize, Vec<Detection>)| k.1.len()).sum::<usize>();
    assert!(boxes > if conf == 0.0 { 100 } else { 20 }, "too few boxes to test: {boxes}");
    assert_eq!(got.report, ev.compute());
    assert_eq!(got.samples, samples);
    assert_eq!(got.detections, kept);
    from_pairs
}

#[test]
fn check_config_rejects_invalid_soft_nms_settings() {
    let model = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
    model.eval();
    for bad_sigma in [0.0f32, -0.5, f32::NAN, f32::INFINITY] {
        for kind in [
            crate::nms::NmsKind::SoftGaussian { sigma: bad_sigma },
            crate::nms::NmsKind::MatrixGaussian { sigma: bad_sigma },
        ] {
            let mut vc = ValConfig::new(64, 2);
            vc.nms.nms_kind = kind;
            assert!(
                matches!(check_config(&model, 2, &vc), Err(Error::Config(_))),
                "{kind:?} should fail check_config"
            );
        }
    }
    for kind in [crate::nms::NmsKind::SoftLinear, crate::nms::NmsKind::MatrixLinear] {
        let mut vc = ValConfig::new(64, 2);
        vc.nms.nms_kind = kind;
        vc.nms.score = crate::fusion::ScoreRule::Elc;
        assert!(matches!(check_config(&model, 2, &vc), Err(Error::Config(_))));
        vc.nms.score = crate::fusion::ScoreRule::None;
        vc.nms.fusion = BoxFusion::Sebf;
        assert!(matches!(check_config(&model, 2, &vc), Err(Error::Config(_))));
    }
}
