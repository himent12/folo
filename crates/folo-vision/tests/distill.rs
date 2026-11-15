//! Dense distillation in the trainer (docs/MODEL.md §8.9) on a micro run.

use std::path::Path;
use std::sync::{Arc, Mutex};

use folo_api::RunEvent;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, Split};
use folo_nn::Module;
use folo_test_support::TestDir;
use folo_vision::train::{OptimizerChoice, ResumeOverrides};
use folo_vision::{DistillConfig, TrainConfig, TrainData, TrainOptions, Trainer};

fn data() -> TrainData {
    let cfg = SyntheticConfig {
        images: 10,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 3,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 0.2,
        ..SyntheticConfig::default()
    };
    let split = |s| Arc::new(SyntheticDataset::new(cfg.clone(), s).unwrap());
    TrainData { train: split(Split::Train), val: split(Split::Val), classes: cfg.classes.clone() }
}

/// 8 training images at 64 px in batches of 4 (one optimizer step per batch), AdamW, no augmentation.
fn micro(epochs: u32) -> TrainConfig {
    TrainConfig {
        epochs,
        batch: 4,
        nbs: 4,
        imgsz: 64,
        device: "cpu".into(),
        optimizer: OptimizerChoice::Adamw,
        lr0: 0.002,
        momentum: 0.9,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        ..TrainConfig::default()
    }
}

/// FNV-1a over every tensor of the live model, in canonical order.
fn weights_hash(t: &Trainer) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for nt in t.model().named_tensors() {
        let bytes: Vec<u8> = match nt.tensor.dtype() {
            folo_core::DType::I64 => {
                nt.tensor.to_vec::<i64>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            _ => nt.tensor.to_vec::<f32>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        for b in nt.name.bytes().chain(bytes) {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// The weights hash of one epoch of [`micro`] without a teacher on origin/main at 1436ad68 (before distillation).
const NO_TEACHER_HASH: u64 = 0x1851_7ce4_1137_dca4;

/// Whether this machine takes the CPU kernels' AVX2 + FMA path, the one [`NO_TEACHER_HASH`] was pinned on.
fn pinned_kernels() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// A teacher file: a folo-n of `nc` classes with its own random weights.
fn teacher_file(dir: &TestDir, nc: usize, seed: u64) -> String {
    let config = folo_nn::io::ModelConfig::new("n", nc).unwrap();
    let mut m = folo_vision::FoloModel::with_rng(config, &mut folo_core::rng::Rng::seed(seed)).unwrap();
    m.set_classes((0..nc).map(|c| ["circle", "square", "star"][c].to_string()).collect()).unwrap();
    let path = dir.join(format!("teacher-{nc}-{seed}.folo"));
    m.save(&path).unwrap();
    path.to_string_lossy().into_owned()
}

fn with_teacher(cfg: TrainConfig, teacher: &str) -> TrainConfig {
    TrainConfig { distill: Some(DistillConfig::new(teacher)), ..cfg }
}

type Log = Arc<Mutex<Vec<RunEvent>>>;

fn options(dir: Option<&Path>) -> (TrainOptions, Log) {
    let log: Log = Arc::default();
    let sink = log.clone();
    let opts = TrainOptions {
        run_dir: dir.map(Path::to_path_buf),
        events: Some(Box::new(move |e| sink.lock().unwrap().push(e.clone()))),
        step_interval: std::time::Duration::ZERO,
        ..TrainOptions::default()
    };
    (opts, log)
}

fn logs(log: &Log) -> Vec<String> {
    log.lock()
        .unwrap()
        .iter()
        .filter_map(|e| if let RunEvent::Log(l) = e { Some(l.message.clone()) } else { None })
        .collect()
}

/// Runs `cfg` to the end and returns the hash of its weights and its log lines.
fn train(cfg: TrainConfig) -> (u64, Vec<String>) {
    let (opts, log) = options(None);
    let mut t = Trainer::new(cfg, data(), opts).unwrap();
    t.run().unwrap();
    (weights_hash(&t), logs(&log))
}

/// A run without `distill` trains exactly as before distillation existed: the weights after one epoch hash to the
/// value origin/main gave, and its config has no `distill` key. CPU kernels pick an AVX2 + FMA path at run time, so
/// the pin holds on x86-64 machines that have both; elsewhere the test only prints the hash.
#[test]
fn a_run_without_a_teacher_is_unchanged() {
    let mut t = Trainer::new(micro(1), data(), TrainOptions::default()).unwrap();
    t.run().unwrap();
    let h = weights_hash(&t);
    if pinned_kernels() {
        assert_eq!(h, NO_TEACHER_HASH, "{h:#018x}: a run without a teacher no longer trains as before");
    } else {
        eprintln!("note: weights hash {h:#018x} not compared (no AVX2 + FMA)");
    }
    let cfg = serde_json::to_value(t.config()).unwrap();
    assert!(cfg.get("distill").is_none(), "{cfg}");
    assert!(t.run_config()["train"].get("distill").is_none());
}

/// A teacher changes the weights, deterministically (two runs give the same weights), also for an `o2o` student;
/// the epoch's log reports both terms and the teacher it loaded.
#[test]
fn a_teacher_trains_deterministically() {
    let dir = TestDir::new("distill-det");
    let teacher = teacher_file(&dir, 2, 7);
    let (plain, _) = train(micro(1));
    let with = with_teacher(micro(1), &teacher);
    let (a, log) = train(with.clone());
    assert_eq!(a, train(with).0, "deterministic");
    assert_ne!(a, plain, "the teacher changes the weights");
    assert!(log.iter().any(|l| l.contains("distilling from") && l.contains("temperature 4")), "{log:?}");
    let line = log.iter().find(|l| l.starts_with("epoch 1: distillation cls")).expect("the epoch's terms");
    assert!(line.contains("over 2 batches"), "{line}");
    let o2o = with_teacher(TrainConfig { head: folo_vision::model::HeadMode::O2o, ..micro(1) }, &teacher);
    let (o, _) = train(o2o.clone());
    assert_eq!(o, train(o2o).0, "o2o student: deterministic");
    assert!(o != a && o != plain, "the o2o student trains differently");
}

/// Stops the run at the boundary after epoch `.0`.
struct StopAfter(u32);

impl folo_vision::train::EpochHook for StopAfter {
    fn pauses(&mut self, epoch: u32) -> bool {
        epoch == self.0
    }

    fn pause(&mut self, _: u32, stop: &std::sync::atomic::AtomicBool) {
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Every tensor of a `.folo` file as raw little-endian bytes, by name.
fn file_tensors(path: &Path) -> Vec<(String, Vec<u8>)> {
    let f = folo_nn::io::ModelFile::open(path).unwrap();
    let mut names: Vec<String> = f.tensor_infos().iter().map(|i| i.name.clone()).collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let t = f.tensor(&n).unwrap();
            let bytes = match t.dtype() {
                folo_core::DType::I64 => {
                    t.to_vec::<i64>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                _ => t.to_vec::<f32>().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect(),
            };
            (n, bytes)
        })
        .collect()
}

/// The teacher is stored with the run: a resume reloads it and continues bit for bit, and a resume whose teacher
/// is gone fails with a message naming it.
#[test]
fn a_resume_reloads_the_teacher_and_continues_bit_exactly() {
    let dir = TestDir::new("distill-resume");
    let teacher = teacher_file(&dir, 2, 8);
    let cfg = with_teacher(micro(2), &teacher);
    let (dir_a, dir_b) = (dir.join("a"), dir.join("b"));
    let (opts, _) = options(Some(&dir_a));
    Trainer::new(cfg.clone(), data(), opts).unwrap().run().unwrap();
    let (opts, _) = options(Some(&dir_b));
    let opts = TrainOptions { after_epoch: Some(Box::new(StopAfter(1))), ..opts };
    let s = Trainer::new(cfg, data(), opts).unwrap().run().unwrap();
    assert_eq!(s.epochs_completed, 1);
    let ckpt = dir_b.join("weights/last.folo");
    let stored = folo_nn::io::ModelFile::open(&ckpt).unwrap().checkpoint().unwrap().train.clone();
    assert_eq!(stored["distill"]["teacher"], teacher.as_str());
    assert_eq!(stored["distill"]["dfl_gain"], 1.5);
    let (opts, log) = options(Some(&dir_b));
    let mut b = Trainer::resume(&ckpt, data(), opts, ResumeOverrides::default()).unwrap();
    b.run().unwrap();
    assert!(logs(&log).iter().any(|l| l.contains("distilling from")));
    let (ta, tb) = (file_tensors(&dir_a.join("weights/last.folo")), file_tensors(&ckpt));
    assert_eq!(ta.len(), tb.len());
    for ((na, va), (nb, vb)) in ta.iter().zip(&tb) {
        assert_eq!(na, nb);
        assert!(va == vb, "tensor {na} differs after resume");
    }
    // Without the teacher the run cannot resume.
    std::fs::remove_file(&teacher).unwrap();
    let (opts, _) = options(None);
    let e = Trainer::resume(dir_a.join("weights/last.folo"), data(), opts, ResumeOverrides::default())
        .err()
        .expect("the teacher is gone");
    assert!(e.to_string().contains("distill.teacher") && e.to_string().contains("does not exist"), "{e}");
}

/// The resume override (`folo train --resume --teacher`): a run that started without a teacher distils from the
/// resumed epoch on, its `config.json` and later checkpoints record it, and the first epoch matches nothing changed
/// (the weights differ from the same resume without the override).
#[test]
fn a_resume_can_turn_distillation_on() {
    let dir = TestDir::new("distill-resume-on");
    let teacher = teacher_file(&dir, 2, 10);
    let run = dir.join("r");
    let (opts, _) = options(Some(&run));
    let opts = TrainOptions { after_epoch: Some(Box::new(StopAfter(1))), ..opts };
    Trainer::new(micro(2), data(), opts).unwrap().run().unwrap();
    let ckpt = run.join("weights/last.folo");
    let saved = std::fs::read(&ckpt).unwrap();
    let (opts, _) = options(None);
    let mut plain = Trainer::resume(&ckpt, data(), opts, ResumeOverrides::default()).unwrap();
    plain.run().unwrap();
    assert!(plain.config().distill.is_none(), "no override: the stored config");
    std::fs::write(&ckpt, &saved).unwrap();
    let (opts, log) = options(Some(&run));
    let over = ResumeOverrides { distill: Some(DistillConfig::new(&teacher)), ..ResumeOverrides::default() };
    let mut on = Trainer::resume(&ckpt, data(), opts, over).unwrap();
    on.run().unwrap();
    assert!(logs(&log).iter().any(|l| l.starts_with("epoch 2: distillation cls")), "{:?}", logs(&log));
    assert_ne!(weights_hash(&on), weights_hash(&plain), "the teacher teaches from the resumed epoch on");
    let stored = folo_nn::io::ModelFile::open(&ckpt).unwrap().checkpoint().unwrap().train.clone();
    assert_eq!(stored["distill"]["teacher"], teacher.as_str());
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["train"]["distill"]["teacher"], teacher.as_str(), "{config}");
}

/// A teacher that cannot teach this student, or settings out of range, are errors before anything trains.
#[test]
fn incompatible_teachers_and_bad_settings_are_errors() {
    let dir = TestDir::new("distill-bad");
    let three = teacher_file(&dir, 3, 9);
    let err = |cfg: TrainConfig| {
        Trainer::new(cfg, data(), TrainOptions::default()).err().expect("refused").to_string()
    };
    let e = err(with_teacher(micro(1), &three));
    assert!(e.contains("nc 3") && e.contains("student 2"), "{e}");
    let missing = dir.join("nowhere.folo").to_string_lossy().into_owned();
    assert!(err(with_teacher(micro(1), &missing)).contains("does not exist"));
    let two = teacher_file(&dir, 2, 9);
    let set = |d: DistillConfig| TrainConfig { distill: Some(d), ..micro(1) };
    let e = err(set(DistillConfig { temperature: 0.0, ..DistillConfig::new(&two) }));
    assert!(e.contains("distill.temperature"), "{e}");
    let e = err(set(DistillConfig { dfl_gain: -1.0, ..DistillConfig::new(&two) }));
    assert!(e.contains("distill.dfl_gain"), "{e}");
}

/// `distill` in a stored config: absent (every config before it) is `None`; present, it round-trips.
#[test]
fn stored_configs_with_and_without_distill_load() {
    let mut old = serde_json::to_value(TrainConfig::default()).unwrap();
    old.as_object_mut().unwrap().remove("distill");
    let cfg: TrainConfig = serde_json::from_value(old).unwrap();
    assert_eq!(cfg.distill, None);
    let with = with_teacher(TrainConfig::default(), "folo-s-coco");
    let v = serde_json::to_value(&with).unwrap();
    assert_eq!(
        v["distill"],
        serde_json::json!({ "teacher": "folo-s-coco", "cls_gain": 0.5, "dfl_gain": 1.5, "temperature": 4.0 })
    );
    assert_eq!(serde_json::from_value::<TrainConfig>(v).unwrap(), with);
}
