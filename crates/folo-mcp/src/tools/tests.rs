//! Unit tests of the tools: the tool count and list the docs and llms.txt state, schema memoization and unique
//! names, unknown arguments, paging and size limits, run warnings and config diffs, issue filtering for an older
//! server, and capped run names.

use super::*;
use folo_api::{IssuePage, IssueSeverity, LogLevel, LogLine};
use folo_server::store::IssueQuery;

use super::arguments::unknown_argument;
use super::datasets::list_issues_compat;
use super::models::decode_image_base64;
use super::results::{MAX_RESULT_BYTES, MAX_SUMMARY_CHARS, cursor_at, json_len, out, page_of};
use super::runs::{MAX_WARNINGS, config_diff, log_warnings};
use crate::errors::Error;

/// The tool count the docs state, and llms.txt's tool list, follow [`TOOLS`] (DOGFOOD-1, -2 and -3 each found
/// a stale count).
#[test]
fn docs_name_every_tool() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let count = format!("{} tools", TOOLS.len());
    for doc in ["README.md", "docs/QUICKSTART.md", "docs/MCP.md", "llms.txt"] {
        let text = std::fs::read_to_string(repo.join(doc)).unwrap();
        assert!(text.contains(&count), "{doc} does not say \"{count}\"");
    }
    let llms = std::fs::read_to_string(repo.join("llms.txt")).unwrap();
    let list = llms.split_once(&count).unwrap().1.split_once("- **CLI:**").unwrap().0;
    for t in TOOLS {
        assert!(list.contains(t.name), "llms.txt's tool list leaves out {}", t.name);
    }
}

/// An oversized `image_base64` answers as an oversized image file does (docs/MCP.md). The limit is a parameter, so
/// the boundary is tested on a few bytes rather than 64 MiB: padding makes 4, 5 and 6 bytes the same 8 characters.
#[test]
fn an_oversized_base64_image_is_payload_too_large() {
    const MAX: u64 = 4;
    assert_eq!(decode_image_base64("AAAAAA==", MAX).unwrap().len(), 4);
    assert_eq!(
        decode_image_base64("  AAAAAA==\r\n", MAX).unwrap().len(),
        4,
        "whitespace around it is not counted"
    );
    for over in ["AAAAAAA=", "AAAAAAAA", " AAAAAAA= ", "AAAAAAAAAAAA"] {
        let e = decode_image_base64(over, MAX).unwrap_err();
        assert_eq!((e.status.as_u16(), e.body.code.as_str()), (413, "payload_too_large"), "{over}");
        assert_eq!(e.body.message, "image_base64 is larger than 64 MiB");
    }
    // Not base64 is still its own error, and an over-long string is refused before it is decoded.
    assert_eq!(decode_image_base64("not base64!", 64).unwrap_err().body.code, "invalid_arguments");
    assert_eq!(decode_image_base64(&"!".repeat(13), MAX).unwrap_err().body.code, "payload_too_large");
}

#[test]
fn run_status_lists_distinct_warnings() {
    let line = |level, m: &str| LogLine { t: String::new(), level, message: m.into() };
    let mut log = vec![
        line(LogLevel::Info, "folo-n on gpu:0 …"),
        line(LogLevel::Warn, "this run takes only 36 optimizer steps"),
        line(LogLevel::Warn, "this run takes only 36 optimizer steps"),
        line(LogLevel::Error, "run failed"),
    ];
    log.extend((0..10).map(|i| line(LogLevel::Warn, &format!("epoch 1 batch {i}: non-finite loss"))));
    let w = log_warnings(&log);
    assert_eq!(w.len(), MAX_WARNINGS);
    assert_eq!(w[0], "this run takes only 36 optimizer steps");
    assert_eq!(w[1], "epoch 1 batch 0: non-finite loss");
    assert!(log_warnings(&log[..1]).is_empty());
}

#[test]
fn unknown_arguments_are_rejected_with_a_suggestion() {
    let args = |v: Value| v.as_object().unwrap().clone();
    let train = find("train").unwrap();
    let schema = train.input_schema();
    assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
    let ok = args(json!({ "dataset_id": "ds_1", "epochs": 1, "lr0": 0.1, "augment": { "mosaic": 0.5 } }));
    assert!(unknown_argument("train", schema, &ok).is_none());
    assert!(unknown_argument("train", schema, &args(json!({ "resume_run_id": "run_1" }))).is_none());
    let e = unknown_argument("train", schema, &args(json!({ "dataset_id": "d", "lr": 0.1 }))).unwrap();
    assert_eq!(e.body.code, "invalid_arguments");
    assert_eq!(e.body.message, "unknown argument `lr`; did you mean `lr0`?");
    assert!(e.body.hint.as_deref().unwrap().contains("dataset_id"));
    let e =
        unknown_argument("train", schema, &args(json!({ "dataset_id": "d", "model": "folo-s" }))).unwrap();
    assert!(e.body.message.contains("did you mean `arch`?"), "{}", e.body.message);
    let nested = args(json!({ "dataset_id": "d", "augment": { "mosiac": 0.5 } }));
    let e = unknown_argument("train", schema, &nested).unwrap();
    assert_eq!(e.body.message, "unknown argument `augment.mosiac`; did you mean `mosaic`?");
    for t in TOOLS {
        let s = t.input_schema();
        assert_eq!(s.get("additionalProperties"), Some(&json!(false)), "{}", t.name);
        assert!(unknown_argument(t.name, s, &args(json!({ "zzz_bogus": 1 }))).is_some(), "{}", t.name);
    }
}

#[test]
fn tool_schemas_are_memoized_and_names_are_unique() {
    let mut seen = std::collections::HashSet::new();
    for t in TOOLS {
        assert!(seen.insert(t.name), "duplicate tool name `{}`", t.name);
        let in_a = t.input_schema();
        let in_b = t.input_schema();
        assert!(Arc::ptr_eq(in_a, in_b), "{}: input schema not memoized", t.name);
        let out_a = t.output_schema();
        let out_b = t.output_schema();
        assert!(Arc::ptr_eq(out_a, out_b), "{}: output schema not memoized", t.name);
        let tool_a = t.tool();
        let tool_b = t.tool();
        assert!(Arc::ptr_eq(&tool_a.input_schema, in_a), "{}: tool() rebuilt input schema", t.name);
        assert!(Arc::ptr_eq(&tool_a.input_schema, &tool_b.input_schema), "{}", t.name);
        assert!(
            Arc::ptr_eq(tool_a.output_schema.as_ref().unwrap(), out_a),
            "{}: tool() rebuilt output schema",
            t.name
        );
        assert!(
            Arc::ptr_eq(tool_a.output_schema.as_ref().unwrap(), tool_b.output_schema.as_ref().unwrap()),
            "{}",
            t.name
        );
    }
}

#[test]
fn issues_from_an_older_server_are_filtered_and_counted_here() {
    use folo_api::{DatasetIssue, IssueCounts};
    let issue = |kind: &str, severity| DatasetIssue {
        kind: kind.into(),
        severity,
        file: "f".into(),
        line: None,
        message: "m".into(),
        fix: None,
    };
    // 60 background images, then one error: an old server ignores `severity` and sends no counts.
    let mut all: Vec<DatasetIssue> = (0..60).map(|_| issue("empty_image", IssueSeverity::Info)).collect();
    all.push(issue("class_overflow", IssueSeverity::Error));
    let old = |q: &IssueQuery| -> Result<IssuePage, Error> {
        let page = all.iter().skip(q.offset as usize).take(q.limit as usize).cloned().collect();
        Ok(IssuePage { total: all.len() as u64, issues: page, counts: IssueCounts::default() })
    };
    let q = IssueQuery { severity: Some(IssueSeverity::Error), limit: 50, ..IssueQuery::default() };
    let page = list_issues_compat(old, &q).unwrap();
    assert_eq!((page.total, page.issues.len()), (1, 1));
    assert_eq!(page.issues[0].kind, "class_overflow");
    assert_eq!((page.counts.error, page.counts.info), (1, 60));
    // A new server's answer is used as is, also an empty one.
    let new = |q: &IssueQuery| Ok(q.page(all.iter()));
    assert_eq!(list_issues_compat(new, &q).unwrap(), q.page(all.iter()));
    let none = |_: &IssueQuery| Ok(IssuePage { total: 0, issues: vec![], counts: IssueCounts::default() });
    assert_eq!(list_issues_compat(none, &q).unwrap().total, 0);
    // An old server that honours `kind`: a kind without matches still gets the dataset-wide counts.
    let old_kind = |q: &IssueQuery| -> Result<IssuePage, Error> {
        let m: Vec<_> =
            all.iter().filter(|i| q.kind.as_ref().is_none_or(|k| *k == i.kind)).cloned().collect();
        let page = m.iter().skip(q.offset as usize).take(q.limit as usize).cloned().collect();
        Ok(IssuePage { total: m.len() as u64, issues: page, counts: IssueCounts::default() })
    };
    let q = IssueQuery { kind: Some("clipped_box".into()), limit: 50, ..IssueQuery::default() };
    let page = list_issues_compat(old_kind, &q).unwrap();
    assert_eq!((page.total, page.counts.error, page.counts.info), (0, 1, 60));
}

#[test]
fn pages_fit_the_budget() {
    let items: Vec<String> = (0..500).map(|i| format!("{i:0>200}")).collect();
    let p = page_of(items, None, Some(200)).unwrap();
    assert!(p.items.len() < 200);
    assert!(json_len(&p) <= MAX_RESULT_BYTES);
    let next = p.next_cursor.clone().unwrap();
    assert_eq!(next, cursor_at(p.items.len() as u64));
    let items: Vec<u32> = (0..10).collect();
    let p = page_of(items.clone(), Some("o8"), Some(5)).unwrap();
    assert_eq!(p.items, vec![8, 9]);
    assert!(p.next_cursor.is_none());
    assert!(page_of(items.clone(), Some("bogus"), None).is_err());
    assert!(page_of(items, None, Some(201)).is_err());
}

#[test]
fn oversized_results_are_tool_errors() {
    let big = vec!["z".repeat(1000); 40];
    let e = out(&big, "s").err().unwrap();
    assert_eq!(e.body.code, "result_too_large");
    let ok = out(&json!({ "a": 1 }), "w".repeat(1000)).ok().unwrap();
    assert_eq!(ok.summary.chars().count(), MAX_SUMMARY_CHARS + 1);
    assert_eq!(short("héllo", 2), "hé…");
    assert_eq!(short("hi", 2), "hi");
}

#[test]
fn diff_lists_differing_fields() {
    let a = json!({ "epochs": 10, "augment": { "mosaic": 1.0, "fliplr": 0.5 }, "name": "a" });
    let b = json!({ "epochs": 20, "augment": { "mosaic": 1.0, "fliplr": 0.0 }, "name": "b" });
    let d = config_diff(&[a, b]);
    assert_eq!(d.keys().collect::<Vec<_>>(), ["augment.fliplr", "epochs", "name"]);
    assert_eq!(d["epochs"], vec![json!(10), json!(20)]);
}

/// `train` takes a run name as long as the server does (`POST /api/runs`: 64 characters), and its inputSchema says so.
#[tokio::test]
async fn train_names_are_capped_as_the_server_caps_them() {
    use folo_server::store::{MAX_RUN_NAME_CHARS, Store};
    let schema = find("train").unwrap().input_schema();
    assert_eq!(schema["properties"]["name"]["maxLength"], json!(MAX_RUN_NAME_CHARS));
    let store = std::sync::Arc::new(folo_server::MockStore::new());
    let ds = store.list_datasets().unwrap().remove(0).id;
    let home = folo_test_support::TestDir::new("mcp-train-name");
    let opts = crate::Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    };
    let server = crate::server::FoloMcp::new(store, None, opts);
    let args = |n: usize| {
        json!({ "dataset_id": ds, "name": "é".repeat(n), "epochs": 1 }).as_object().unwrap().clone()
    };
    let e = server.inner.train(args(MAX_RUN_NAME_CHARS + 1)).await.err().unwrap();
    assert_eq!(
        (e.body.code.as_str(), e.body.message.as_str()),
        ("invalid_arguments", "name must be at most 64 characters")
    );
    assert!(server.inner.train(args(MAX_RUN_NAME_CHARS)).await.is_ok());
}

#[test]
fn tool_annotations_match_tool_semantics() {
    let mut destructive = Vec::new();
    let mut idempotent_mutating = Vec::new();
    let mut open_world = Vec::new();
    for t in TOOLS {
        let ann = t.tool().annotations.expect("every tool carries annotations");
        assert_eq!(ann.read_only_hint, Some(!t.mutating), "{}: read_only_hint", t.name);
        if !t.mutating {
            assert_eq!(ann.idempotent_hint, Some(true), "{}: read-only tool must be idempotent", t.name);
            assert_eq!(
                ann.destructive_hint,
                Some(false),
                "{}: read-only tool must not be destructive",
                t.name
            );
        } else if ann.idempotent_hint == Some(true) {
            idempotent_mutating.push(t.name);
        }
        if ann.destructive_hint == Some(true) {
            destructive.push(t.name);
        }
        if ann.open_world_hint == Some(true) {
            open_world.push(t.name);
        }
    }
    assert_eq!(destructive, ["cloud_terminate", "delete_benchmark"]);
    assert_eq!(
        idempotent_mutating,
        ["register_dataset", "stop_run", "stop_labeling", "review_label_images", "stop_bench_entry"]
    );
    assert_eq!(
        open_world,
        ["cloud_connections", "cloud_offers", "cloud_instances", "cloud_terminate", "start_labeling"]
    );
}

#[test]
fn closest_match_and_closest_named_suggest_accurately() {
    use super::arguments::{MAX_SUGGEST_CHARS, closest_match, closest_named};

    // Case-insensitive exact match, while exact matches and unrelated 3-char strings return None.
    assert_eq!(closest_match("api", &["API", "CLI", "MCP"]), Some("API"));
    assert_eq!(closest_match("agent_cookbook", &["API", "AGENT_COOKBOOK"]), Some("AGENT_COOKBOOK"));
    assert_eq!(closest_match("API", &["API", "CLI"]), None);
    assert_eq!(closest_match("CLI", &["API", "MCP"]), None);
    assert_eq!(closest_match("MPC", &["API", "MCP"]), Some("MCP"));

    // Adjacent transposition (OSA distance 1) and 1-2 edit typos.
    assert_eq!(closest_match("predcit", &["predict", "predict_batch", "train"]), Some("predict"));
    assert_eq!(closest_match("empty_iamge", &folo_api::ISSUE_KINDS), Some("empty_image"));
    assert_eq!(closest_match("list_runss", &["list_runs", "list_models"]), Some("list_runs"));

    // Unambiguous prefix and nested prefix hierarchy work, while ambiguous siblings do not guess.
    assert_eq!(closest_match("ds_5a", &["ds_5a1e", "ds_7c2f"]), Some("ds_5a1e"));
    assert_eq!(closest_match("pred", &["predict", "predict_batch", "train"]), Some("predict"));
    assert_eq!(closest_match("ds_", &["ds_5a1e", "ds_7c2f"]), None);
    assert_eq!(closest_match("list_", &["list_runs", "list_models", "list_datasets"]), None);

    // Distant or oversized needles return None cleanly.
    assert_eq!(closest_match("typo", &folo_api::ISSUE_KINDS), None);
    assert_eq!(closest_match("no_such_tool", &["train", "val", "predict"]), None);
    assert_eq!(closest_match(&"a".repeat(MAX_SUGGEST_CHARS + 1), &["a", "aa"]), None);
    assert_eq!(closest_match("", &["train"]), None);

    // Named entities: ID typo, unique display name, display name typo, and duplicate-name ambiguity.
    let items = vec![
        ("ds_5a1e".into(), "shapes".into()),
        ("ds_7c2f".into(), "traffic-cams".into()),
        ("ds_9001".into(), "dup".into()),
        ("ds_9002".into(), "dup".into()),
        ("ds_9003".into(), "dups".into()),
    ];
    assert_eq!(closest_named("ds_5a1f", &items), Some(("ds_5a1e", "shapes")));
    assert_eq!(closest_named("shapes", &items), Some(("ds_5a1e", "shapes")));
    assert_eq!(closest_named("shapess", &items), Some(("ds_5a1e", "shapes")));
    // Ambiguous duplicate names must not fall through to a neighboring name like `dups`.
    assert_eq!(closest_named("dup", &items), None);
    assert_eq!(closest_named("du", &items), None);
    assert_eq!(closest_named("dupx", &items), None);

    // Case-insensitive alias that is absent from a tool's schema still falls through to closest_match.
    let register = find("register_dataset").unwrap();
    let e = unknown_argument(
        "register_dataset",
        register.input_schema(),
        json!({ "PATH": "." }).as_object().unwrap(),
    )
    .unwrap();
    assert_eq!(e.body.message, "unknown argument `PATH`; did you mean `path`?");
}

#[tokio::test]
async fn not_found_tool_errors_and_issue_kinds_include_suggestions() {
    let store = std::sync::Arc::new(folo_server::MockStore::new());
    let home = folo_test_support::TestDir::new("mcp-suggest");
    let opts = crate::Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    };
    let server = crate::server::FoloMcp::new(store, None, opts);
    let call = |tool: &str, v: Value| {
        let def = find(tool).unwrap();
        let args = v.as_object().unwrap().clone();
        let inner = server.inner.clone();
        let label = format!("{tool}({v})");
        async move {
            let mut p = Progress::none();
            inner
                .call(def, args, &mut p)
                .await
                .err()
                .unwrap_or_else(|| panic!("{label} unexpectedly succeeded"))
        }
    };

    // Dataset ID typo & display name suggestion preserve existing hint.
    let e = call("get_dataset", json!({ "dataset_id": "ds_5a1f" })).await;
    assert_eq!(e.body.code, "dataset_not_found");
    let hint = e.body.hint.as_deref().unwrap();
    assert!(hint.contains("list_datasets"), "{hint}");
    assert!(hint.contains("did you mean `ds_5a1e`"), "{hint}");

    let e = call("get_dataset", json!({ "dataset_id": "shapes" })).await;
    assert_eq!(e.body.code, "dataset_not_found");
    assert!(e.body.hint.as_deref().unwrap().contains("did you mean `ds_5a1e`"), "{:?}", e.body.hint);

    // Model ID typo (via predict and via train.weights).
    let e = call("export", json!({ "model_id": "m_53d9", "format": "onnx" })).await;
    assert_eq!(e.body.code, "model_not_found");
    assert!(e.body.hint.as_deref().unwrap().contains("did you mean `m_53d1`"), "{:?}", e.body.hint);

    let e = call("train", json!({ "dataset_id": "ds_5a1e", "arch": "folo-n", "weights": "m_53d9" })).await;
    assert_eq!(e.body.code, "model_not_found");
    assert!(e.body.hint.as_deref().unwrap().contains("did you mean `m_53d1`"), "{:?}", e.body.hint);

    // Run ID typo (run_status and compare_runs).
    let e = call("run_status", json!({ "run_id": "run_2b20" })).await;
    assert_eq!(e.body.code, "run_not_found");
    assert!(e.body.hint.as_deref().unwrap().contains("did you mean `run_2b02`"), "{:?}", e.body.hint);

    let e = call("compare_runs", json!({ "run_ids": ["run_1a01", "run_2b20"] })).await;
    assert_eq!(e.body.code, "run_not_found");
    assert!(e.body.hint.as_deref().unwrap().contains("did you mean `run_2b02`"), "{:?}", e.body.hint);

    // When compare_runs fails on an empty/overlong first ID, do not suggest for a later ID.
    for first in [String::new(), "x".repeat(200)] {
        let e = call("compare_runs", json!({ "run_ids": [first, "run_2b20"] })).await;
        assert_eq!(e.body.code, "run_not_found");
        assert!(
            !e.body.hint.as_deref().unwrap_or("").contains("did you mean"),
            "must not suggest for unreached index 1: {:?}",
            e.body.hint
        );
    }

    // Unknown issue kind typo.
    let e = call("list_dataset_issues", json!({ "dataset_id": "ds_7c2f", "kind": "empty_iamge" })).await;
    assert_eq!(e.body.code, "invalid_arguments");
    assert!(e.body.message.contains("did you mean `empty_image`?"), "{}", e.body.message);
}

#[tokio::test]
async fn predict_batch_evicts_oldest_results_past_lru_cap() {
    use crate::batch_cache::{BatchLookup, MAX_BATCHES};

    let store = std::sync::Arc::new(folo_server::MockStore::new());
    let home = folo_test_support::TestDir::new("mcp-batch-lru");
    let img_path = home.join("a.jpg");
    image::RgbImage::from_pixel(16, 16, image::Rgb([10, 20, 30])).save(&img_path).unwrap();
    let opts = crate::Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    };
    let server = crate::server::FoloMcp::new(store, None, opts);
    let def = find("predict_batch").unwrap();
    for i in 0..=MAX_BATCHES {
        let mut p = Progress::none();
        let out = server
            .inner
            .call(
                def,
                json!({ "model_id": "m_31aa", "paths": ["a.jpg"] }).as_object().unwrap().clone(),
                &mut p,
            )
            .await
            .unwrap();
        assert_eq!(out.value["batch_id"], format!("pb_{}", i + 1));
        // Touch pb_1 right after the first two batches so pb_2 becomes the LRU entry.
        if i == 1 {
            let mut cache = server.inner.batches.lock().unwrap();
            assert!(matches!(cache.lookup("pb_1"), BatchLookup::Found(_)));
        }
    }
    {
        let mut cache = server.inner.batches.lock().unwrap();
        assert_eq!(cache.len(), MAX_BATCHES);
        assert!(matches!(cache.lookup("pb_1"), BatchLookup::Found(_)), "promoted pb_1 must stay cached");
        assert!(matches!(cache.lookup("pb_2"), BatchLookup::Evicted), "LRU pb_2 must be evicted");
        assert!(matches!(cache.lookup(&format!("pb_{}", MAX_BATCHES + 1)), BatchLookup::Found(_)));

        // An oversized batch exceeding the cache byte budget returns `payload_too_large` without evicting
        // existing cached entries.
        *cache = crate::batch_cache::BatchCache::with_limits(MAX_BATCHES, 16);
        assert!(cache.try_insert("pb_keep".into(), std::sync::Arc::new(json!({}))));
    }
    let mut p = Progress::none();
    let e = server
        .inner
        .call(def, json!({ "model_id": "m_31aa", "paths": ["a.jpg"] }).as_object().unwrap().clone(), &mut p)
        .await
        .err()
        .unwrap();
    assert_eq!(e.body.code, "payload_too_large");
    let cache = server.inner.batches.lock().unwrap();
    assert!(cache.contains("pb_keep"), "oversized batch must not evict valid cached entries");
}
