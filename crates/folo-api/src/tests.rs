//! Unit tests of the wire format: defaults, round trips and compatibility (older servers' responses still parse,
//! unknown enum values from newer ones decode as `Unknown`, the schema leaves `Unknown` out).

use super::*;

/// An optimizer value from a newer build decodes as `Unknown` (a client built before it keeps listing runs);
/// the known ones still decode as themselves.
#[test]
fn unknown_optimizers_decode() {
    let opt = |s: &str| serde_json::from_value::<Optimizer>(serde_json::json!(s)).unwrap();
    let res = |s: &str| serde_json::from_value::<ResolvedOptimizer>(serde_json::json!(s)).unwrap();
    assert_eq!(
        [opt("auto"), opt("sgd"), opt("adamw"), opt("musgd")],
        [Optimizer::Auto, Optimizer::Sgd, Optimizer::Adamw, Optimizer::Musgd]
    );
    assert_eq!(opt("lion"), Optimizer::Unknown);
    assert_eq!(
        [res("sgd"), res("adamw"), res("musgd")],
        [ResolvedOptimizer::Sgd, ResolvedOptimizer::Adamw, ResolvedOptimizer::Musgd]
    );
    assert_eq!(res("lion"), ResolvedOptimizer::Unknown);
    // Inside a config, as a newer server's run carries it.
    let c: TrainConfig =
        serde_json::from_value(serde_json::json!({"dataset_id": "ds_1", "optimizer": "lion"})).unwrap();
    assert_eq!(c.optimizer, Optimizer::Unknown);
}

/// The schema (MCP inputs, `folo describe`) offers only the values a client may send.
#[cfg(feature = "schema")]
#[test]
fn schema_leaves_unknown_out() {
    for s in [schemars::schema_for!(Optimizer), schemars::schema_for!(ResolvedOptimizer)] {
        let v = serde_json::to_value(&s).unwrap().to_string();
        assert!(v.contains("musgd") && !v.contains("unknown"), "{v}");
    }
}

#[test]
fn system_capabilities_are_additive() {
    // A server older than `capabilities` (and `features`, `data_dir`).
    let old =
        r#"{"cpu":{"name":"c","threads":2,"util":0},"memory":{"used_bytes":1,"total_bytes":2},"gpus":[]}"#;
    let s: SystemResponse = serde_json::from_str(old).unwrap();
    assert_eq!((s.capabilities, s.features.len(), s.data_dir), (None, 0, None));
    // A partial object (a field this version lacks, one a newer server adds) parses with defaults.
    let v = serde_json::json!({
        "cpu": {"name": "c", "threads": 2, "util": 0}, "memory": {"used_bytes": 1, "total_bytes": 2}, "gpus": [],
        "capabilities": {"device": "gpu:0", "tensor_cores": {"available": true}, "future": 1},
    });
    let c = serde_json::from_value::<SystemResponse>(v).unwrap().capabilities.unwrap();
    assert_eq!(c.device, "gpu:0");
    assert!(c.tensor_cores.available && !c.tensor_cores.enabled && c.tensor_cores.reason.is_none());
    assert_eq!(c.amp, AmpSupport::default());
    // `reason` is sent as `null`, not left out.
    let v = serde_json::to_value(Capabilities::default()).unwrap();
    assert!(v["amp"]["reason"].is_null() && v["amp"].as_object().unwrap().contains_key("reason"), "{v}");
    assert!(v["tensor_cores"].as_object().unwrap().contains_key("reason"), "{v}");
    // Older clients (without the field) still parse the new response: SystemInfo ignores it.
    let new = SystemResponse {
        system: serde_json::from_str(old).unwrap(),
        data_dir: None,
        features: vec![],
        capabilities: Some(Capabilities::default()),
        gpu_pool: Some(GpuPool {
            device: "gpu:0".into(),
            live_bytes: 1,
            cached_bytes: 2,
            peak_live_bytes: 3,
        }),
    };
    let back: SystemInfo = serde_json::from_value(serde_json::to_value(&new).unwrap()).unwrap();
    assert_eq!(back, new.system);
}

#[test]
fn model_head_and_trained_with_are_additive() {
    // A server older than `head` and `trained_with`.
    let old = r#"{"id":"m_1","name":"a","arch":"folo-n","imgsz":640,"classes":["x"],"params":1,"flops_g":1.0,
        "bytes":1,"created":"2026-01-01T00:00:00Z","run_id":null,"metrics":null}"#;
    let m: Model = serde_json::from_str(old).unwrap();
    assert_eq!((m.head, m.trained_with), (Head::O2m, None));
    // `trained_with` is sent as `null`, not left out; both round-trip.
    let v = serde_json::to_value(&m).unwrap();
    assert_eq!(v["head"], "o2m");
    assert!(v.as_object().unwrap().contains_key("trained_with") && v["trained_with"].is_null(), "{v}");
    let m =
        Model { head: Head::O2o, trained_with: Some(TrainedWith { amp: true, imgsz: 512, epochs: 80 }), ..m };
    let v = serde_json::to_value(&m).unwrap();
    assert_eq!(v["head"], "o2o");
    assert_eq!(v["trained_with"], serde_json::json!({ "amp": true, "imgsz": 512, "epochs": 80 }));
    assert_eq!(serde_json::from_value::<Model>(v).unwrap(), m);
}

#[test]
fn model_precision_and_fused_are_additive() {
    // A server older than `precision` and `fused`.
    let old = r#"{"id":"m_1","name":"a","arch":"folo-n","imgsz":640,"classes":["x"],"params":1,"flops_g":1.0,
        "bytes":1,"created":"2026-01-01T00:00:00Z","run_id":null,"metrics":null,"head":"o2m","trained_with":null}"#;
    let m: Model = serde_json::from_str(old).unwrap();
    assert_eq!((m.precision, m.fused), (None, None));
    // Unknown stays absent on the wire (not `null`).
    let v = serde_json::to_value(&m).unwrap();
    let o = v.as_object().unwrap();
    assert!(!o.contains_key("precision") && !o.contains_key("fused"), "{v}");
    // Both round-trip.
    for (p, s) in [(ModelPrecision::F32, "f32"), (ModelPrecision::F16, "f16")] {
        assert_eq!((p.as_str(), ModelPrecision::parse(s)), (s, Some(p)));
        let m = Model { precision: Some(p), fused: Some(p == ModelPrecision::F16), ..m.clone() };
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(
            (&v["precision"], &v["fused"]),
            (&serde_json::json!(s), &serde_json::json!(p == ModelPrecision::F16))
        );
        assert_eq!(serde_json::from_value::<Model>(v).unwrap(), m);
    }
    // A dtype this version has no name for (a newer server's), `null` or a non-string reads as unknown,
    // never as an error.
    for p in [r#""bf16""#, r#""F16""#, "null", "16"] {
        let json = old.replace(r#""trained_with":null"#, &format!(r#""trained_with":null,"precision":{p}"#));
        assert_eq!(serde_json::from_str::<Model>(&json).unwrap().precision, None, "{p}");
    }
}

#[test]
fn train_config_defaults() {
    let c: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1"}"#).unwrap();
    assert_eq!(c.epochs, 100);
    assert_eq!(c.arch, Arch::N);
    assert_eq!(c.device, "auto");
    assert_eq!(c.augment, Augment::default());
    let v = serde_json::to_value(&c).unwrap();
    assert_eq!(v["arch"], "folo-n");
    assert_eq!(v["optimizer"], "auto");
    assert_eq!(v["weights"], serde_json::Value::Null);
}

#[test]
fn error_hint_is_always_present() {
    let e = ErrorBody { error: ApiError { code: "x".into(), message: "y".into(), hint: None } };
    assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"error":{"code":"x","message":"y","hint":null}}"#);
}

#[test]
fn run_event_roundtrip() {
    let ev = RunEvent::Status(StatusEvent::new(RunStatus::Running, None).with_length(100, 50));
    let back = RunEvent::from_sse(ev.name(), &ev.data_json()).unwrap().unwrap();
    assert_eq!(back, ev);
    assert!(RunEvent::from_sse("future", "{}").is_none());
}

#[test]
fn run_length_fields_default_for_older_servers() {
    let s: StatusEvent = serde_json::from_str(r#"{"status":"running","error":null}"#).unwrap();
    assert_eq!(s, StatusEvent::new(RunStatus::Running, None));
    let step: Step = serde_json::from_str(
        r#"{"epoch":1,"step":1,"lr":0.01,"loss":{"box":1,"cls":1,"dfl":1,"total":3},"imgs_per_s":1,"eta_s":5}"#,
    )
    .unwrap();
    assert_eq!((step.steps_per_epoch, step.epochs), (0, 0));
    let v = serde_json::to_value(StatusEvent::new(RunStatus::Queued, None).with_length(3, 7)).unwrap();
    assert_eq!((v["epochs"].as_u64(), v["steps_per_epoch"].as_u64()), (Some(3), Some(7)));
}

#[test]
fn eval_provenance_is_additive() {
    // A server older than the provenance fields (and `best_f1_conf`).
    let old = r#"{"pr":[],"confusion":[[0]],"per_class":[],"images":3}"#;
    let e: ModelEval = serde_json::from_str(old).unwrap();
    assert_eq!((e.dataset_id.as_deref(), e.dataset_name.as_deref(), e.split), (None, None, None));
    assert_eq!((e.split_fallback.as_ref(), e.confusion_conf, e.confusion_iou), (None, None, None));
    // Unknown fields are sent as `null`, never left out.
    let v = serde_json::to_value(&e).unwrap();
    for k in ["dataset_id", "dataset_name", "split", "split_fallback", "confusion_conf", "confusion_iou"] {
        assert!(v.as_object().unwrap().contains_key(k) && v[k].is_null(), "{k}: {v}");
    }
    let fb = SplitFallback::empty_split(Split::Val, Split::Train, "shapes", 80);
    assert_eq!(fb.reason, "empty_split");
    assert!(fb.message.contains("'shapes'") && fb.message.contains("no val images"), "{}", fb.message);
    assert!(fb.message.contains("train split (80 images)") && fb.message.contains("training-set"));
    let e = ModelEval {
        dataset_id: Some("ds_1".into()),
        dataset_name: Some("shapes".into()),
        split: Some(Split::Train),
        split_fallback: Some(fb),
        confusion_conf: Some(0.25),
        confusion_iou: Some(0.45),
        ..e
    };
    // Through text, as on the wire: f32 thresholds print as written.
    let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
    assert_eq!(
        (v["split"].as_str(), v["split_fallback"]["requested"].as_str()),
        (Some("train"), Some("val"))
    );
    assert_eq!((v["confusion_conf"].as_f64(), v["confusion_iou"].as_f64()), (Some(0.25), Some(0.45)));
    assert_eq!(serde_json::from_value::<ModelEval>(v).unwrap(), e);
    // `ValReport.dataset_name` too.
    let old = r#"{"model_id":"m","dataset_id":"d","split":"test","imgsz":64,"images":1,
        "val":{"map50":0,"map50_95":0,"precision":0,"recall":0,"per_class":[]},
        "timing":{"total_s":0,"per_image_ms":0}}"#;
    assert_eq!(serde_json::from_str::<ValReport>(old).unwrap().dataset_name, None);
}
