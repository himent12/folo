//! The demo's benchmarks, labeling agents and jobs, made once on first use.

mod shapes;
mod shapes_hard;

use std::collections::HashMap;

use crate::api::mock::{DatasetFixture, MOCK_NOW};
use crate::api::time;
use crate::api::*;

use crate::api::mock_bench::benches::fingerprint_of;
use crate::api::mock_bench::images::{count_class_boxes, resolve_source, sample_source};
use crate::api::mock_bench::{BenchFixture, MockBenchData};

use shapes::{ShapesSeed, seed_shapes};
use shapes_hard::{ShapesHardSeed, seed_shapes_hard};

pub(super) fn seed_data(datasets: &[DatasetFixture], models: &[Model]) -> MockBenchData {
    let h = |hours: f64| MOCK_NOW - (hours * 3600.0) as i64;
    let agents = vec![
        LabelAgent {
            id: LabelAgentId::Claude,
            name: "Claude Code".into(),
            available: true,
            version: Some("2.1.30".into()),
            models: vec![
                LabelModel { id: "sonnet".into(), label: "Sonnet".into(), fast: false },
                LabelModel { id: "haiku".into(), label: "Haiku".into(), fast: false },
                LabelModel { id: "opus".into(), label: "Opus".into(), fast: false },
            ],
            default_model: Some("sonnet".into()),
            default_effort: LabelEffort::Medium,
            fast: false,
            hint: None,
        },
        LabelAgent {
            id: LabelAgentId::Codex,
            name: "Codex".into(),
            available: true,
            version: Some("0.42.0".into()),
            models: vec![
                LabelModel { id: "o3".into(), label: "o3".into(), fast: true },
                LabelModel { id: "codex-mini".into(), label: "codex-mini".into(), fast: true },
            ],
            default_model: Some("o3".into()),
            default_effort: LabelEffort::Medium,
            fast: true,
            hint: None,
        },
        LabelAgent {
            id: LabelAgentId::Antigravity,
            name: "Antigravity".into(),
            available: true,
            version: Some("1.5.13".into()),
            models: vec![
                LabelModel { id: "gemini-4-neon".into(), label: "Gemini 4 Neon".into(), fast: false },
                LabelModel { id: "gemini-4-hydrogen".into(), label: "Gemini 4 Hydrogen".into(), fast: false },
                LabelModel { id: "gemini-4-carbon".into(), label: "Gemini 4 Carbon".into(), fast: false },
                LabelModel { id: "gemini-4-argon".into(), label: "Gemini 4 Argon".into(), fast: false },
                LabelModel { id: "gemini-3.8-flash".into(), label: "Gemini 3.8 Flash".into(), fast: false },
            ],
            default_model: None,
            default_effort: LabelEffort::Low,
            fast: false,
            hint: None,
        },
    ];
    let jobs = vec![LabelJob {
        id: "lb_4f10".into(),
        name: "line-rejects".into(),
        classes: vec!["circle".into(), "square".into(), "triangle".into()],
        instructions: "Shapes on the reject chute under uneven LEDs.".into(),
        created: time::format(h(48.0)),
        status: LabelJobStatus::Idle,
        counts: LabelCounts { images: 140, pending: 20, labeled: 120, failed: 0, boxes: 492 },
        run: None,
        usage: LabelUsage {
            calls: 30,
            input_tokens: 48_000,
            cached_input_tokens: 0,
            output_tokens: 7_200,
            cost_usd: Some(0.084),
        },
        dataset_id: None,
    }];

    let model_spec = |mid: &str, fallback_arch: &str, fallback_params: u64| -> BenchContestant {
        match models.iter().find(|m| m.id == mid) {
            Some(m) => BenchContestant::Model {
                model_id: m.id.clone(),
                arch: m.arch.as_str().to_string(),
                params: m.params,
                imgsz: m.imgsz,
            },
            None => BenchContestant::Model {
                model_id: mid.into(),
                arch: fallback_arch.into(),
                params: fallback_params,
                imgsz: 640,
            },
        }
    };

    let ShapesHardSeed {
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
    } = seed_shapes_hard(datasets, &jobs, &h, &model_spec);
    let ShapesSeed { b2, imgs_2, n0, n1, n2, n3, s0, s1, weak, loops_2 } =
        seed_shapes(datasets, &jobs, &h, &model_spec);

    // 3. Night-shift tray (`bn_7a04`): fresh benchmark with 0 entries (editable plan!).
    let (_, src_name_3, classes_3, pool_3) = resolve_source(
        &BenchSource::Dataset { dataset_id: "ds_c07a".into(), split: Some(Split::Val) },
        datasets,
        &jobs,
        Split::Val,
    )
    .unwrap();
    let imgs_3 = sample_source(&pool_3, 0, 60);
    let boxes_3 = count_class_boxes(&imgs_3, classes_3.len());
    let plan_3 = folo_bench::default_plan();
    let fp_3 = fingerprint_of(0, &classes_3, imgs_3.len(), &plan_3);
    let b3 = Bench {
        id: "bn_7a04".into(),
        name: "Night-shift tray".into(),
        description: "Sorting tray camera under warm low-wattage LEDs: shadows at the rim and occasional motion blur when the tray indexes.".into(),
        classes: classes_3,
        created: time::format(h(1.5)),
        source: BenchSource::Dataset { dataset_id: "ds_c07a".into(), split: Some(Split::Val) },
        source_name: src_name_3,
        images: imgs_3.len() as u64,
        boxes: boxes_3.iter().sum(),
        class_boxes: boxes_3,
        plan: plan_3,
        seed: 0,
        fingerprint: fp_3,
        entries: 0,
        best: None,
        active: false,
    };

    MockBenchData {
        seq: 100,
        benches: vec![
            BenchFixture {
                bench: b3,
                images: imgs_3,
                entries: Vec::new(),
                loops: Vec::new(),
                loop_live: HashMap::new(),
            },
            BenchFixture {
                bench: b1,
                images: imgs_1,
                entries: vec![e1_lead, e1_onnx, e1_crowd, e1_partial_cls, e1_agent, e1_failed, e1_running],
                loops: vec![loop_1],
                loop_live: HashMap::new(),
            },
            BenchFixture {
                bench: b2,
                images: imgs_2,
                entries: vec![n3, n2, n1, n0, s1, s0, weak],
                loops: loops_2,
                loop_live: HashMap::new(),
            },
        ],
        agents,
        jobs,
    }
}
