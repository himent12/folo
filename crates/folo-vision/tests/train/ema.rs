//! EMA contract, window scaling, and resume bit-exactness tests (legacy and `PC-PEMA`).

use super::*;

/// A weights file to fine-tune from: a fresh folo-n's checkpoint (no training needed).
fn start_weights(dir: &str) -> TestFile {
    let start = TestFile::new(dir, "start.folo");
    Trainer::new(micro_config(1), data(&shapes(2, 1, 64)), options(None).0)
        .unwrap()
        .save_checkpoint(&start)
        .unwrap();
    start
}

/// docs/MODEL.md §8.5: a fine-tune's EMA averages over a quarter of its optimizer steps, at most the default
/// window of 10,000: 400 steps get decay 0.99 and tau 20, a COCO-length fine-tune (40,000 steps) and a run from
/// scratch of 400 steps the default 0.9999 and 2000 bit for bit, in the checkpoint's `folo.ema` and in `resolved`.
/// No training needed: 16 images in batches of 4 without accumulation are 4 optimizer steps an epoch.
#[test]
fn fine_tune_ema_window_follows_the_run_length() {
    let start = start_weights("train-ema-window");
    let ckpt = TestFile::new("train-ema-window", "c.folo");
    let saved = |t: &Trainer| -> (u64, u64) {
        t.save_checkpoint(&ckpt).unwrap();
        let c = ModelFile::open(&ckpt).unwrap().checkpoint().unwrap().ema.unwrap();
        (c.decay.to_bits(), c.tau.to_bits())
    };
    let build = |epochs: u32, weights: bool| {
        let cfg = TrainConfig { epochs, nbs: 4, ..micro_config(1) };
        let opts = TrainOptions { weights: weights.then(|| start.path().to_path_buf()), ..options(None).0 };
        Trainer::new(cfg, data(&shapes(16, 1, 64)), opts).unwrap()
    };
    let bits = |(d, t): (f64, f64)| (d.to_bits(), t.to_bits());
    let default = bits((folo_nn::EMA_DECAY, folo_nn::EMA_TAU));
    let short = build(100, true);
    assert_eq!(short.schedule().total_iters(), 400);
    assert_eq!(saved(&short), bits((0.99, 20.0)));
    let r = short.resolved_train();
    assert_eq!((r.ema_decay, r.ema_tau), (Some(0.99), Some(20.0)));
    let coco = build(10_000, true);
    assert_eq!(coco.schedule().total_iters(), 40_000);
    assert_eq!(saved(&coco), default, "a COCO-length fine-tune keeps the default EMA");
    assert_eq!(saved(&build(100, false)), default, "a run from scratch keeps the default EMA");
}

/// A fine-tune stopped after its first epoch and resumed ends bit for bit as the uninterrupted run, EMA included:
/// the resume continues the run's own (scaled) EMA. `ema_run_fraction` 4 over 4 optimizer steps gives a window of
/// 16 (decay 0.9375), so the average is not just the last step. On the CPU: the EMA update is the same code on
/// either device, and a GPU shared with training runs makes the test slow.
#[test]
fn a_resumed_fine_tune_continues_its_ema() {
    let start = start_weights("train-ema-resume");
    let ds = shapes(4, 1, 64);
    let cfg =
        TrainConfig { batch: 2, nbs: 2, ema_run_fraction: 4.0, device: "cpu".into(), ..micro_config(2) };
    let opts =
        |dir: &Path| TrainOptions { weights: Some(start.path().to_path_buf()), ..options(Some(dir)).0 };
    let (dir_a, dir_b) = (run_dir("ema-a"), run_dir("ema-b"));
    let sum = Trainer::new(cfg.clone(), data(&ds), opts(&dir_a)).unwrap().run().unwrap();
    assert_eq!(sum.status, RunStatus::Completed);
    let stopped = TrainOptions { after_epoch: Some(Box::new(StopAfter(1))), ..opts(&dir_b) };
    let sum = Trainer::new(cfg, data(&ds), stopped).unwrap().run().unwrap();
    assert_eq!((sum.status, sum.epochs_completed), (RunStatus::Stopped, 1));
    // A checkpoint whose recipe does not give its stored pair (one made with an environment knob, its `folo.train`
    // without the key): the run reports the pair it averages with, the stored one.
    let knob = dir_b.join("knob.folo");
    {
        let file = ModelFile::open(dir_b.join("weights/last.folo")).unwrap();
        let raw = std::fs::read(dir_b.join("weights/last.folo")).unwrap();
        let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + header_len]).unwrap();
        let mut map = std::collections::BTreeMap::new();
        for (k, v) in header["__metadata__"].as_object().unwrap() {
            map.insert(k.clone(), v.as_str().unwrap().to_string());
        }
        let mut train: serde_json::Value = serde_json::from_str(&map["folo.train"]).unwrap();
        train.as_object_mut().unwrap().remove("ema_run_fraction");
        map.insert("folo.train".into(), train.to_string());
        let tensors: Vec<(String, folo_core::Tensor)> =
            file.tensor_infos().iter().map(|i| (i.name.clone(), file.tensor(&i.name).unwrap())).collect();
        folo_nn::io::safetensors::write_file(&knob, Some(&map), &tensors).unwrap();
    }
    let k = Trainer::resume(&knob, data(&ds), options(None).0, ResumeOverrides::default()).unwrap();
    let r = k.resolved_train();
    assert_eq!((r.ema_decay, r.ema_tau), (Some(0.9375), Some(3.2)));
    drop(k);
    let mut b = Trainer::resume(
        dir_b.join("weights/last.folo"),
        data(&ds),
        options(Some(&dir_b)).0,
        ResumeOverrides::default(),
    )
    .unwrap();
    assert_eq!(b.run().unwrap().status, RunStatus::Completed);
    let last = |dir: &Path| dir.join("weights/last.folo");
    let c = ModelFile::open(last(&dir_a)).unwrap().checkpoint().unwrap().ema.unwrap();
    assert_eq!((c.decay, c.tau), (0.9375, 3.2));
    let (ta, tb) = (tensors(&last(&dir_a)), tensors(&last(&dir_b)));
    assert_eq!(ta.len(), tb.len());
    assert!(ta.iter().any(|(n, _)| n.starts_with("ema.")));
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }
    assert_eq!(ModelFile::open(last(&dir_b)).unwrap().checkpoint().unwrap().ema.unwrap(), c);
}

/// End-to-end `PC-PEMA` (`ema_mode = Power`) run stopped after epoch 1 and resumed ends bit-exact to uninterrupted.
#[test]
fn pc_pema_resumes_bit_exactly() {
    let ds = shapes(4, 1, 64);
    let cfg = TrainConfig {
        batch: 2,
        nbs: 2,
        ema_mode: folo_nn::EmaMode::Power,
        device: "cpu".into(),
        ..micro_config(2)
    };
    let (dir_a, dir_b) = (run_dir("pema-a"), run_dir("pema-b"));
    let (opts_a, log_a) = options(Some(&dir_a));
    assert_eq!(
        Trainer::new(cfg.clone(), data(&ds), opts_a).unwrap().run().unwrap().status,
        RunStatus::Completed
    );
    let logs: Vec<String> = log_a
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Log(l) = e { Some(l.message.clone()) } else { None })
        .collect();
    assert!(logs.iter().any(|m| m.contains("EMA PC-PEMA")), "{logs:?}");
    let tr = Trainer::new(cfg.clone(), data(&ds), options(None).0).unwrap();
    let res_tr = tr.resolved_train();
    assert_eq!((res_tr.ema_decay, res_tr.ema_tau), (None, None));
    let api_res = cfg.resolved_train(4, 1);
    assert_eq!((api_res.ema_decay, api_res.ema_tau), (None, None));
    let mut cfg_json =
        serde_json::json!({ "resolved": api_res, "train": serde_json::to_value(&cfg).unwrap() });
    let back = folo_vision::train::ResolvedOptim::from_run_config(&cfg_json).unwrap();
    assert_eq!((back.ema_decay, back.ema_tau), (None, None));
    cfg_json["resolved"].as_object_mut().unwrap().retain(|k, _| !k.starts_with("ema_"));
    let back2 = folo_vision::train::ResolvedOptim::from_run_config(&cfg_json).unwrap();
    assert_eq!((back2.ema_decay, back2.ema_tau), (None, None));
    let stopped = TrainOptions { after_epoch: Some(Box::new(StopAfter(1))), ..options(Some(&dir_b)).0 };
    assert_eq!(Trainer::new(cfg, data(&ds), stopped).unwrap().run().unwrap().status, RunStatus::Stopped);
    let mut b = Trainer::resume(
        dir_b.join("weights/last.folo"),
        data(&ds),
        options(Some(&dir_b)).0,
        ResumeOverrides::default(),
    )
    .unwrap();
    assert_eq!(b.run().unwrap().status, RunStatus::Completed);
    let last = |d: &Path| d.join("weights/last.folo");
    let ca = ModelFile::open(last(&dir_a)).unwrap().checkpoint().unwrap().ema.unwrap();
    assert!(ca.power.is_some());
    assert_eq!(ModelFile::open(last(&dir_b)).unwrap().checkpoint().unwrap().ema.unwrap(), ca);
    assert_eq!(tensors(&last(&dir_a)), tensors(&last(&dir_b)));
}
