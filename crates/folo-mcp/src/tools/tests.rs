use super::*;
use folo_api::{IssuePage, IssueSeverity, LogLevel, LogLine};
use folo_server::store::IssueQuery;

use super::arguments::unknown_argument;
use super::datasets::list_issues_compat;
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
    let schema = (train.input)();
    assert_eq!(schema.get("additionalProperties"), Some(&json!(false)));
    let ok = args(json!({ "dataset_id": "ds_1", "epochs": 1, "lr0": 0.1, "augment": { "mosaic": 0.5 } }));
    assert!(unknown_argument("train", &schema, &ok).is_none());
    assert!(unknown_argument("train", &schema, &args(json!({ "resume_run_id": "run_1" }))).is_none());
    let e = unknown_argument("train", &schema, &args(json!({ "dataset_id": "d", "lr": 0.1 }))).unwrap();
    assert_eq!(e.body.code, "invalid_arguments");
    assert_eq!(e.body.message, "unknown argument `lr`; did you mean `lr0`?");
    assert!(e.body.hint.as_deref().unwrap().contains("dataset_id"));
    let e =
        unknown_argument("train", &schema, &args(json!({ "dataset_id": "d", "model": "folo-s" }))).unwrap();
    assert!(e.body.message.contains("did you mean `arch`?"), "{}", e.body.message);
    let nested = args(json!({ "dataset_id": "d", "augment": { "mosiac": 0.5 } }));
    let e = unknown_argument("train", &schema, &nested).unwrap();
    assert_eq!(e.body.message, "unknown argument `augment.mosiac`; did you mean `mosaic`?");
    for t in TOOLS {
        let s = (t.input)();
        assert_eq!(s.get("additionalProperties"), Some(&json!(false)), "{}", t.name);
        assert!(unknown_argument(t.name, &s, &args(json!({ "zzz_bogus": 1 }))).is_some(), "{}", t.name);
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
    let schema = (find("train").unwrap().input)();
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
