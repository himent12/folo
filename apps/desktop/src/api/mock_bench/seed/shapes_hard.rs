//! The demo's Shapes-hard val benchmark (`bn_3c90`).

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;

use crate::api::mock_bench::benches::{fingerprint_of, plan_passes, propose_plan};
use crate::api::mock_bench::images::{count_class_boxes, resolve_source, sample_source};
use crate::api::mock_bench::live::pass_at;
use crate::api::mock_bench::scoring::make_done_entry;
use crate::api::mock_bench::{BenchImgSource, EntryFixture, EntrySkill};

/// Shapes-hard val, as [`seed_shapes_hard`] makes it for `seed_data` to assemble.
pub(super) struct ShapesHardSeed {
    pub(super) b1: Bench,
    pub(super) imgs_1: Vec<BenchImgSource>,
    pub(super) e1_lead: EntryFixture,
    pub(super) e1_onnx: EntryFixture,
    pub(super) e1_crowd: EntryFixture,
    pub(super) e1_partial_cls: EntryFixture,
    pub(super) e1_agent: EntryFixture,
    pub(super) e1_failed: EntryFixture,
    pub(super) e1_running: EntryFixture,
    pub(super) loop_1: BenchLoop,
}

/// Shapes-hard val (`bn_3c90`): its images, entries of every kind and state, and an active loop.
pub(super) fn seed_shapes_hard(
    datasets: &[DatasetFixture],
    jobs: &[LabelJob],
    h: &dyn Fn(f64) -> i64,
    model_spec: &dyn Fn(&str, &str, u64) -> BenchContestant,
) -> ShapesHardSeed {
    // 1. Shapes-hard val (`bn_3c90`): agent-designed plan, entries of every kind & state, 1 active loop.
    let (_, src_name_1, classes_1, pool_1) = resolve_source(
        &BenchSource::Dataset { dataset_id: "ds_3b91".into(), split: Some(Split::Val) },
        datasets,
        jobs,
        Split::Val,
    )
    .unwrap();
    let imgs_1 = sample_source(&pool_1, 0, 500);
    let boxes_1 = count_class_boxes(&imgs_1, classes_1.len());
    let mut b1 = Bench {
        id: "bn_3c90".into(),
        name: "Shapes-hard val".into(),
        description: "Overhead inspection camera over mixed shapes: clutter, overlapping parts, motion smear on fast belts and dim corners at night shift.".into(),
        classes: classes_1.clone(),
        created: time::format(h(26.0)),
        source: BenchSource::Dataset { dataset_id: "ds_3b91".into(), split: Some(Split::Val) },
        source_name: src_name_1,
        images: imgs_1.len() as u64,
        boxes: boxes_1.iter().sum(),
        class_boxes: boxes_1,
        plan: folo_bench::default_plan(),
        seed: 0,
        fingerprint: String::new(),
        entries: 0,
        best: None,
        active: true,
    };
    b1.plan = propose_plan(
        &b1,
        &DesignBench {
            agent: LabelAgentId::Claude,
            model: Some("sonnet".into()),
            effort: Some(LabelEffort::Medium),
            note: "night shift".into(),
        },
    );
    b1.fingerprint = fingerprint_of(b1.seed, &b1.classes, imgs_1.len(), &b1.plan);

    let e1_lead = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c01",
        "shapes-hard-s-best",
        model_spec("m_44c0", "folo-s", 9_420_000),
        h(22.0),
        54,
        EntrySkill { map: 0.624, fragility: 0.86, ece: 0.038, iou: 0.81, self_recall: 0.58 },
        imgs_1.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.1 }),
        None,
        None,
    );
    let e1_onnx = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c02",
        "shapes-hard-s.onnx",
        BenchContestant::External,
        h(20.0),
        40,
        EntrySkill { map: 0.622, fragility: 0.87, ece: 0.040, iou: 0.81, self_recall: 0.57 },
        imgs_1.len() as u64,
        true,
        true,
        Vec::new(),
        None,
        None,
        None,
    );
    let e1_crowd = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c03",
        "shapes-crowded-s-best",
        model_spec("m_7c11", "folo-s", 9_420_000),
        h(18.0),
        52,
        EntrySkill { map: 0.553, fragility: 0.94, ece: 0.052, iou: 0.78, self_recall: 0.46 },
        imgs_1.len() as u64,
        true,
        true,
        Vec::new(),
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 4.2 }),
        None,
        None,
    );
    let e1_partial_cls = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c04",
        "shapes-n-best",
        model_spec("m_31aa", "folo-n", 2_630_000),
        h(15.0),
        28,
        EntrySkill { map: 0.432, fragility: 0.98, ece: 0.061, iou: 0.76, self_recall: 0.31 },
        imgs_1.len() as u64,
        true,
        true,
        vec!["star".into(), "hexagon".into()],
        Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.3 }),
        None,
        None,
    );
    let e1_agent = make_done_entry(
        &b1,
        &imgs_1,
        "be_3c05",
        "Claude Code (sonnet)",
        BenchContestant::Agent {
            agent: LabelAgentId::Claude,
            model: Some("sonnet".into()),
            effort: LabelEffort::Medium,
        },
        h(12.0),
        95,
        EntrySkill { map: 0.492, fragility: 1.0, ece: 0.0, iou: 0.75, self_recall: 0.34 },
        48,
        false,
        false,
        Vec::new(),
        None,
        Some(LabelUsage {
            calls: 12,
            input_tokens: 19_400,
            cached_input_tokens: 0,
            output_tokens: 3_120,
            cost_usd: Some(0.031),
        }),
        None,
    );
    let e1_failed = EntryFixture {
        entry: BenchEntry {
            id: "be_3c06".into(),
            bench_id: b1.id.clone(),
            name: "Codex (o3)".into(),
            contestant: BenchContestant::Agent {
                agent: LabelAgentId::Codex,
                model: Some("o3".into()),
                effort: LabelEffort::Medium,
            },
            status: BenchEntryStatus::Failed,
            created: time::format(h(6.0)),
            finished: Some(time::format(h(5.98))),
            progress: BenchProgress { done: 0, total: 48, condition: None, severity: None },
            images: 48,
            conditions: false,
            scored: false,
            fingerprint: b1.fingerprint.clone(),
            missing_classes: Vec::new(),
            score: None,
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: None,
            usage: None,
            error: Some("codex exited with status 1: not signed in.".into()),
            hint: Some("Run `codex login` on the server's machine, then add the entry again.".into()),
        },
        skill: EntrySkill { map: 0.4, fragility: 1.0, ece: 0.0, iou: 0.7, self_recall: 0.0 },
        open_records: None,
        cached_result: None,
        live: None,
    };
    let run_total = imgs_1.len() as u64 * (1 + plan_passes(&b1.plan) as u64);
    let run_done = (run_total as f64 * 0.56) as u64;
    let (r_cond, r_sev) = pass_at(&b1.plan, imgs_1.len() as u64, true, run_done);
    let e1_running = EntryFixture {
        entry: BenchEntry {
            id: "be_3c07".into(),
            bench_id: b1.id.clone(),
            name: "shapes-hard-s-f16".into(),
            contestant: BenchContestant::Model {
                model_id: "m_44c0".into(),
                arch: "folo-s".into(),
                params: 9_420_000,
                imgsz: 640,
            },
            status: BenchEntryStatus::Running,
            created: time::format(h(0.02)),
            finished: None,
            progress: BenchProgress { done: run_done, total: run_total, condition: r_cond, severity: r_sev },
            images: imgs_1.len() as u64,
            conditions: true,
            scored: true,
            fingerprint: b1.fingerprint.clone(),
            missing_classes: Vec::new(),
            score: None,
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: Some(BenchSpeed { device: "gpu:0".into(), ms_per_image: 2.9 }),
            usage: None,
            error: None,
            hint: None,
        },
        skill: EntrySkill { map: 0.623, fragility: 0.86, ece: 0.039, iou: 0.81, self_recall: 0.58 },
        open_records: None,
        cached_result: None,
        live: Some((run_done, 0.0)),
    };
    let loop_1 = BenchLoop {
        id: "bl_3c10".into(),
        bench_id: b1.id.clone(),
        name: "shapes-hard-s-best loop".into(),
        status: BenchLoopStatus::Running,
        phase: Some(BenchLoopPhase::Training),
        created: time::format(h(0.6)),
        finished: None,
        config: StartBenchLoop {
            name: Some("shapes-hard-s-best loop".into()),
            teacher: "m_44c0".into(),
            pool: BenchSource::Dataset { dataset_id: "ds_3b91".into(), split: Some(Split::Train) },
            generations: Some(2),
            min_confidence: None,
            student: BenchStudent {
                arch: Some("folo-m".into()),
                epochs: Some(40),
                imgsz: Some(640),
                batch: Some(16),
                device: Some("auto".into()),
                weights: None,
                seed: 0,
            },
        },
        generations: vec![
            BenchGeneration {
                generation: 0,
                labels: None,
                run_id: Some("run_52e8".into()),
                model_id: Some("m_44c0".into()),
                model_name: Some("shapes-hard-s-best".into()),
                entry_id: Some("be_3c01".into()),
                score: e1_lead.entry.score.as_ref().map(|s| s.total),
                map50_95: e1_lead.entry.map50_95,
                ci95: e1_lead.entry.ci95,
            },
            BenchGeneration {
                generation: 1,
                labels: Some(BenchLabels {
                    threshold: 0.56,
                    images: 3200,
                    labeled_images: 2940,
                    boxes: 14_820,
                    excluded: 0,
                    dataset_id: "ds_3b91".into(),
                }),
                run_id: Some("run_a19f".into()),
                model_id: None,
                model_name: None,
                entry_id: None,
                score: None,
                map50_95: None,
                ci95: None,
            },
        ],
        verdict: None,
        error: None,
        hint: None,
    };
    ShapesHardSeed {
        b1,
        imgs_1,
        e1_lead,
        e1_onnx,
        e1_crowd,
        e1_partial_cls,
        e1_agent,
        e1_failed,
        e1_running,
        loop_1,
    }
}
