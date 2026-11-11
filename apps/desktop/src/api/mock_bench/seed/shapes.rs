//! The demo's Shapes val benchmark (`bn_5e12`).

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;

use crate::api::mock_bench::benches::fingerprint_of;
use crate::api::mock_bench::images::{count_class_boxes, resolve_source, sample_source};
use crate::api::mock_bench::scoring::make_done_entry;
use crate::api::mock_bench::{BenchImgSource, EntryFixture, EntrySkill};

/// Shapes val, as [`seed_shapes`] makes it for `seed_data` to assemble.
pub(super) struct ShapesSeed {
    pub(super) b2: Bench,
    pub(super) imgs_2: Vec<BenchImgSource>,
    pub(super) n0: EntryFixture,
    pub(super) n1: EntryFixture,
    pub(super) n2: EntryFixture,
    pub(super) n3: EntryFixture,
    pub(super) s0: EntryFixture,
    pub(super) s1: EntryFixture,
    pub(super) weak: EntryFixture,
    pub(super) loops_2: Vec<BenchLoop>,
}

/// Shapes val (`bn_5e12`): its images, finished entries, and three loops (improved, held, failed).
pub(super) fn seed_shapes(
    datasets: &[DatasetFixture],
    jobs: &[LabelJob],
    h: &dyn Fn(f64) -> i64,
    model_spec: &dyn Fn(&str, &str, u64) -> BenchContestant,
) -> ShapesSeed {
    // 2. Shapes val (`bn_5e12`): 3 loops (Improved, Held, Failed) + finished entries.
    let (_, src_name_2, classes_2, pool_2) = resolve_source(
        &BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Val) },
        datasets,
        jobs,
        Split::Val,
    )
    .unwrap();
    let imgs_2 = sample_source(&pool_2, 0, 200);
    let boxes_2 = count_class_boxes(&imgs_2, classes_2.len());
    let plan_2 = folo_bench::default_plan();
    let fp_2 = fingerprint_of(0, &classes_2, imgs_2.len(), &plan_2);
    let b2 = Bench {
        id: "bn_5e12".into(),
        name: "Shapes val".into(),
        description: "Coloured shapes on textured backgrounds: held-out validation split grading clean accuracy, robustness and multi-generation self-labeling.".into(),
        classes: classes_2,
        created: time::format(h(160.0)),
        source: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Val) },
        source_name: src_name_2,
        images: imgs_2.len() as u64,
        boxes: boxes_2.iter().sum(),
        class_boxes: boxes_2,
        plan: plan_2,
        seed: 0,
        fingerprint: fp_2,
        entries: 0,
        best: None,
        active: false,
    };
    let n0 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a10",
        "shapes-n-best",
        model_spec("m_31aa", "folo-n", 2_630_000),
        h(140.0),
        24,
        EntrySkill { map: 0.781, fragility: 0.88, ece: 0.035, iou: 0.83, self_recall: 0.69 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.3 }),
        None,
        None,
    );
    let n1 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a11",
        "shapes-n-best loop gen 1-best",
        BenchContestant::Model {
            model_id: "m_5a11".into(),
            arch: "folo-s".into(),
            params: 9_420_000,
            imgsz: 640,
        },
        h(138.0),
        32,
        EntrySkill { map: 0.808, fragility: 0.86, ece: 0.033, iou: 0.84, self_recall: 0.72 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let n2 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a12",
        "shapes-n-best loop gen 2-best",
        BenchContestant::Model {
            model_id: "m_5a12".into(),
            arch: "folo-s".into(),
            params: 9_420_000,
            imgsz: 640,
        },
        h(136.0),
        32,
        EntrySkill { map: 0.829, fragility: 0.85, ece: 0.031, iou: 0.85, self_recall: 0.75 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let n3 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a13",
        "shapes-n-best loop gen 3-best",
        BenchContestant::Model {
            model_id: "m_5a13".into(),
            arch: "folo-s".into(),
            params: 9_420_000,
            imgsz: 640,
        },
        h(134.0),
        32,
        EntrySkill { map: 0.846, fragility: 0.84, ece: 0.030, iou: 0.86, self_recall: 0.77 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let s0 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a20",
        "shapes-n-f16",
        model_spec("m_31ab", "folo-n", 2_630_000),
        h(100.0),
        19,
        EntrySkill { map: 0.780, fragility: 0.88, ece: 0.036, iou: 0.83, self_recall: 0.68 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 1.7 }),
        None,
        None,
    );
    let s1 = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a21",
        "shapes-n-f16 loop gen 1-best",
        BenchContestant::Model {
            model_id: "m_5a21".into(),
            arch: "folo-n".into(),
            params: 2_630_000,
            imgsz: 640,
        },
        h(98.0),
        24,
        EntrySkill { map: 0.786, fragility: 0.88, ece: 0.035, iou: 0.83, self_recall: 0.69 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.3 }),
        None,
        None,
    );
    let weak = make_done_entry(
        &b2,
        &imgs_2,
        "be_5a30",
        "shapes-mini-n-last",
        model_spec("m_90d2", "folo-n", 2_630_000),
        h(60.0),
        18,
        EntrySkill { map: 0.242, fragility: 1.18, ece: 0.14, iou: 0.70, self_recall: 0.0 },
        imgs_2.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 1.4 }),
        None,
        None,
    );

    let gen_from = |gen_idx: u32,
                    ef: &EntryFixture,
                    labels: Option<BenchLabels>,
                    run_id: Option<&str>,
                    model_id: &str| {
        BenchGeneration {
            generation: gen_idx,
            labels,
            run_id: run_id.map(Into::into),
            model_id: Some(model_id.into()),
            model_name: Some(ef.entry.name.clone()),
            entry_id: Some(ef.entry.id.clone()),
            score: ef.entry.score.as_ref().map(|s| s.total),
            map50_95: ef.entry.map50_95,
            ci95: ef.entry.ci95,
        }
    };
    let mk_labels = |th: f32, labeled: u64, boxes: u64| BenchLabels {
        threshold: th,
        images: 800,
        labeled_images: labeled,
        boxes,
        excluded: 0,
        dataset_id: "ds_8f2c".into(),
    };
    let loops_2 = vec![
        BenchLoop {
            id: "bl_5l01".into(),
            bench_id: b2.id.clone(),
            name: "shapes-n-best loop".into(),
            status: BenchLoopStatus::Done,
            phase: None,
            created: time::format(h(139.5)),
            finished: Some(time::format(h(133.8))),
            config: StartBenchLoop {
                name: Some("shapes-n-best loop".into()),
                teacher: "m_31aa".into(),
                pool: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Train) },
                generations: Some(3),
                min_confidence: None,
                student: BenchStudent {
                    arch: Some("folo-s".into()),
                    epochs: Some(30),
                    imgsz: Some(640),
                    batch: Some(16),
                    device: Some("auto".into()),
                    weights: None,
                    seed: 0,
                },
            },
            generations: vec![
                gen_from(0, &n0, None, Some("run_7d1e"), "m_31aa"),
                gen_from(1, &n1, Some(mk_labels(0.53, 756, 2_890)), None, "m_5a11"),
                gen_from(2, &n2, Some(mk_labels(0.51, 772, 3_010)), None, "m_5a12"),
                gen_from(3, &n3, Some(mk_labels(0.50, 781, 3_094)), None, "m_5a13"),
            ],
            verdict: Some(BenchLoopVerdict::Improved),
            error: None,
            hint: None,
        },
        BenchLoop {
            id: "bl_5l02".into(),
            bench_id: b2.id.clone(),
            name: "shapes-n-f16 loop".into(),
            status: BenchLoopStatus::Done,
            phase: None,
            created: time::format(h(99.5)),
            finished: Some(time::format(h(97.8))),
            config: StartBenchLoop {
                name: Some("shapes-n-f16 loop".into()),
                teacher: "m_31ab".into(),
                pool: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Train) },
                generations: Some(1),
                min_confidence: None,
                student: BenchStudent {
                    arch: Some("folo-n".into()),
                    epochs: Some(30),
                    imgsz: Some(640),
                    batch: Some(16),
                    device: Some("auto".into()),
                    weights: None,
                    seed: 0,
                },
            },
            generations: vec![
                gen_from(0, &s0, None, Some("run_7d1e"), "m_31ab"),
                gen_from(1, &s1, Some(mk_labels(0.53, 752, 2_860)), None, "m_5a21"),
            ],
            verdict: Some(BenchLoopVerdict::Held),
            error: None,
            hint: None,
        },
        BenchLoop {
            id: "bl_5l03".into(),
            bench_id: b2.id.clone(),
            name: "shapes-mini-n-last loop".into(),
            status: BenchLoopStatus::Failed,
            phase: None,
            created: time::format(h(58.0)),
            finished: Some(time::format(h(57.95))),
            config: StartBenchLoop {
                name: Some("shapes-mini-n-last loop".into()),
                teacher: "m_90d2".into(),
                pool: BenchSource::Dataset { dataset_id: "ds_8f2c".into(), split: Some(Split::Train) },
                generations: Some(1),
                min_confidence: None,
                student: BenchStudent {
                    arch: Some("folo-n".into()),
                    epochs: Some(30),
                    imgsz: Some(320),
                    batch: Some(16),
                    device: Some("auto".into()),
                    weights: None,
                    seed: 0,
                },
            },
            generations: vec![
                gen_from(0, &weak, None, Some("run_1a2b"), "m_90d2"),
                BenchGeneration {
                    generation: 1,
                    labels: None,
                    run_id: None,
                    model_id: None,
                    model_name: None,
                    entry_id: None,
                    score: None,
                    map50_95: None,
                    ci95: None,
                },
            ],
            verdict: None,
            error: Some(
                "shapes-mini-n-last never reaches 90% precision on at least 10 labels on this benchmark, so its entry found no threshold."
                    .into(),
            ),
            hint: Some(
                "Start a loop with a stronger teacher, or set a minimum confidence explicitly.".into(),
            ),
        },
    ];
    ShapesSeed { b2, imgs_2, n0, n1, n2, n3, s0, s1, weak, loops_2 }
}
