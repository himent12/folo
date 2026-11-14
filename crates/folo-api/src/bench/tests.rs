//! Unit tests of the benchmark types: axis weights and names, request defaults and events by name.

use super::*;
use crate::LabelAgentId;

#[test]
fn axis_weights_sum_to_one_and_names_round_trip() {
    let sum: f32 = BenchAxis::ALL.iter().map(|a| a.weight()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "{sum}");
    for a in BenchAxis::ALL {
        assert_eq!(serde_json::to_value(a).unwrap(), a.as_str());
    }
}

#[test]
fn requests_take_their_defaults() {
    let c: CreateBench =
        serde_json::from_str(r#"{"name":"pond","source":{"kind":"dataset","dataset_id":"ds_1"}}"#).unwrap();
    assert_eq!(c.source, BenchSource::Dataset { dataset_id: "ds_1".into(), split: None });
    assert!(c.plan.is_none() && c.max_images.is_none() && c.description.is_empty() && c.seed == 0);

    let p: BenchPlanCondition = serde_json::from_str(r#"{"condition":"haze","severities":[1,3]}"#).unwrap();
    assert_eq!((p.weight, p.why), (1.0, None));

    let s: StartBenchEntry = serde_json::from_str(
        r#"{"contestant":{"kind":"external","predictions":[{"image":"1","boxes":[{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}]}]},"name":"yolo"}"#,
    )
    .unwrap();
    let BenchContestantSpec::External { scored, open, predictions } = s.contestant else {
        panic!("external")
    };
    assert!(scored && !open && predictions[0].condition.is_none() && predictions[0].boxes[0].score == 1.0);
    let s: StartBenchEntry =
        serde_json::from_str(r#"{"contestant":{"kind":"external","open":true},"name":"yolo"}"#).unwrap();
    assert!(
        matches!(s.contestant, BenchContestantSpec::External { open: true, ref predictions, .. } if predictions.is_empty())
    );
    let s: StartBenchEntry =
        serde_json::from_str(r#"{"contestant":{"kind":"agent","agent":"codex"}}"#).unwrap();
    assert!(matches!(s.contestant, BenchContestantSpec::Agent { agent: LabelAgentId::Codex, .. }));
    assert!(serde_json::from_str::<StartBenchEntry>(r#"{"contestant":{"kind":"robot"}}"#).is_err());

    let l: StartBenchLoop =
        serde_json::from_str(r#"{"teacher":"m1","pool":{"kind":"label_job","job_id":"lb_1"}}"#).unwrap();
    assert_eq!(l.student, BenchStudent::default());
    assert!(l.generations.is_none() && l.min_confidence.is_none());
}

#[test]
fn events_round_trip_by_name() {
    let ev = BenchEvent::Removed(BenchRemoved { id: "be_1".into() });
    assert_eq!(ev.name(), "removed");
    assert_eq!(BenchEvent::from_sse(ev.name(), &ev.data_json()).unwrap().unwrap(), ev);
    assert!(BenchEvent::from_sse("later", "{}").is_none());
    assert!(BenchEvent::from_sse("entry", "{").unwrap().is_err());
}
