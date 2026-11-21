//! The protocol: the full flow under each supported revision, an unsupported version, and the size
//! limit of results.

use std::io::Write;
use std::time::Duration;

use base64::Engine as _;
use serde_json::{Value, json};

use super::harness::{
    ALL_TOOLS, Client, NEW, OLD, check_tools, checked, init_old, spawn, tmp_dir, validate, yolo_files,
};

/// Everything after the handshake / discovery, for one revision.
fn full_flow(c: &mut Client, data_dir: &std::path::Path) {
    // Subscribed first: the first change notice comes within the watch interval (1 s), while the rest runs. It
    // follows the mock's scripted live run, which changes on every tick (the mock's own runs take seconds per
    // epoch, so the models below come from a seeded completed run too).
    let run_uri = "folo://runs/run_3c02".to_string();
    let subscribed = if c.inline.is_some() {
        let listen = c.start(
            "subscriptions/listen",
            json!({ "notifications": { "resourceSubscriptions": [run_uri, "folo://nope"] } }),
            json!({}),
        );
        let ack = c.notification(Duration::from_secs(10), |n| {
            n["method"] == "notifications/subscriptions/acknowledged"
        });
        assert_eq!(ack["params"]["notifications"]["resourceSubscriptions"], json!([run_uri]), "{ack}");
        Some(listen)
    } else {
        let r = c.result("resources/subscribe", json!({ "uri": run_uri }));
        assert!(r.is_object());
        let bad = c.request("resources/subscribe", json!({ "uri": "folo://runs/nope" }));
        assert!(bad.get("error").is_some(), "{bad}");
        None
    };
    let schemas = check_tools(c, &ALL_TOOLS);
    let info = checked(c, &schemas, "folo_info", json!({}));
    assert_eq!(info["mode"], "mock");
    assert_eq!(info["scales"].as_array().unwrap().len(), 5);

    // Resources: listing, templates, docs and an image blob.
    let res = c.result("resources/list", json!({}))["resources"].as_array().unwrap().clone();
    let uris: Vec<&str> = res.iter().map(|r| r["uri"].as_str().unwrap()).collect();
    for want in
        ["folo://datasets/ds_5a1e", "folo://runs/run_3c02", "folo://runs/run_3c02/metrics", "folo://docs/MCP"]
    {
        assert!(uris.contains(&want), "{want} missing from {uris:?}");
    }
    let templates =
        c.result("resources/templates/list", json!({}))["resourceTemplates"].as_array().unwrap().clone();
    assert!(templates.iter().any(|t| t["uriTemplate"] == "folo://datasets/{id}/images/{image_id}{?size}"));
    assert!(
        templates
            .iter()
            .any(|t| t["uriTemplate"] == "folo://estimate/{dataset_id}{?arch,epochs,batch,imgsz,device,amp}")
    );
    assert!(uris.contains(&"folo://system/throughput"));
    let doc = c.result("resources/read", json!({ "uri": "folo://docs/API" }));
    assert_eq!(doc["contents"][0]["mimeType"], "text/markdown");
    assert!(doc["contents"][0]["text"].as_str().unwrap().contains("/api/datasets"));
    let page = checked(
        c,
        &schemas,
        "list_dataset_images",
        json!({ "dataset_id": "ds_5a1e", "split": "val", "limit": 3, "include_boxes": true }),
    );
    assert_eq!(page["items"].as_array().unwrap().len(), 3);
    assert_eq!(page["total"], 200);
    assert_eq!(page["next_cursor"], "o3");
    let next = checked(
        c,
        &schemas,
        "list_dataset_images",
        json!({ "dataset_id": "ds_5a1e", "cursor": "o3", "limit": 1 }),
    );
    assert!(next["items"][0].get("boxes").is_none());
    let image_id = page["items"][0]["image_id"].as_str().unwrap().to_string();
    let img =
        c.result("resources/read", json!({ "uri": format!("folo://datasets/ds_5a1e/images/{image_id}") }));
    let blob = &img["contents"][0];
    assert_eq!(blob["mimeType"], "image/jpeg");
    let jpeg = base64::engine::general_purpose::STANDARD.decode(blob["blob"].as_str().unwrap()).unwrap();
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    // The default is a thumbnail of at most 512 px; `?size` picks another size, `?size=original` the file.
    let side = |bytes: &[u8]| {
        let im = image::load_from_memory(bytes).unwrap();
        im.width().max(im.height())
    };
    assert!(side(&jpeg) <= 512);
    let mut read = |q: &str| {
        let r = c.result(
            "resources/read",
            json!({ "uri": format!("folo://datasets/ds_5a1e/images/{image_id}{q}") }),
        );
        base64::engine::general_purpose::STANDARD.decode(r["contents"][0]["blob"].as_str().unwrap()).unwrap()
    };
    let (small, original) = (read("?size=64"), read("?size=original"));
    assert!(side(&small) <= 64, "{}", side(&small));
    assert!(side(&original) >= side(&small));
    let bad = c.request(
        "resources/read",
        json!({ "uri": format!("folo://datasets/ds_5a1e/images/{image_id}?size=big") }),
    );
    assert!(bad["error"]["code"].is_i64(), "{bad}");
    let missing = c.request("resources/read", json!({ "uri": "folo://datasets/nope" }));
    assert!(missing["error"]["code"].is_i64(), "{missing}");
    let samples = c.result("resources/read", json!({ "uri": "folo://runs/run_2b02/samples/1" }));
    assert_eq!(samples["contents"][0]["mimeType"], "image/jpeg");
    let mut grid = |uri: &str| {
        let r = c.result("resources/read", json!({ "uri": uri }));
        let b = base64::engine::general_purpose::STANDARD
            .decode(r["contents"][0]["blob"].as_str().unwrap())
            .unwrap();
        image::load_from_memory(&b).unwrap().width()
    };
    assert_eq!(grid("folo://runs/run_2b02/samples/1"), 4 * 160);
    assert_eq!(grid("folo://runs/run_2b02/samples/1?tile=256"), 4 * 256);

    // Prompts: retrieval returns messages only.
    let prompts = c.result("prompts/list", json!({}))["prompts"].as_array().unwrap().clone();
    let names: Vec<&str> = prompts.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["train_on_folder", "label_folder", "diagnose_run"]);
    let p = c.result("prompts/get", json!({ "name": "diagnose_run", "arguments": { "run_id": "run_2b02" } }));
    let text = p["messages"][0]["content"]["text"].as_str().unwrap();
    assert!(text.contains("run_2b02") && text.contains("TrainConfig"));
    let p = c.result(
        "prompts/get",
        json!({ "name": "train_on_folder", "arguments": { "folder_path": "D:/data/cats" } }),
    );
    assert!(p["messages"][0]["content"]["text"].as_str().unwrap().contains("register_dataset"));
    let bad = c.request("prompts/get", json!({ "name": "train_on_folder", "arguments": {} }));
    assert_eq!(bad["error"]["code"], -32602);
    let p = c.result(
        "prompts/get",
        json!({ "name": "label_folder", "arguments": { "folder_path": "D:/photos/street", "classes": "car, person" } }),
    );
    let text = p["messages"][0]["content"]["text"].as_str().unwrap();
    assert!(text.contains("create_label_job") && text.contains(r#""classes":["car","person"]"#), "{text}");
    assert!(text.contains(r#""name":"street""#) && text.contains("view_label_image"), "{text}");
    let bad =
        c.request("prompts/get", json!({ "name": "label_folder", "arguments": { "folder_path": "x" } }));
    assert_eq!(bad["error"]["code"], -32602);

    // create_synthetic_dataset with progress → train → subscribe / run_status → predict.
    let id = c.start(
        "tools/call",
        json!({ "name": "create_synthetic_dataset", "arguments": { "name": "tiny", "images": 20, "imgsz": 64 } }),
        json!({ "progressToken": "synth-1" }),
    );
    let r = c.response(id);
    let progress: Vec<Value> = c
        .notes
        .iter()
        .filter(|n| n["method"] == "notifications/progress" && n["params"]["progressToken"] == "synth-1")
        .cloned()
        .collect();
    assert!(!progress.is_empty(), "no progress for a request with a progressToken");
    let values: Vec<f64> = progress.iter().map(|n| n["params"]["progress"].as_f64().unwrap()).collect();
    assert!(values.windows(2).all(|w| w[0] < w[1]), "progress must increase: {values:?}");
    assert_eq!(*values.last().unwrap(), 20.0);
    c.notes.retain(|n| n["method"] != "notifications/progress");
    let ds = r["result"]["structuredContent"].clone();
    validate(&schemas["create_synthetic_dataset"], &ds, "create_synthetic_dataset");
    let ds_id = ds["id"].as_str().unwrap().to_string();
    assert_eq!(ds["splits"]["train"], 16);
    // No progressToken, no progress.
    c.ok("create_synthetic_dataset", json!({ "name": "tiny2", "images": 5, "imgsz": 64 }));
    assert!(c.notes.iter().all(|n| n["method"] != "notifications/progress"));

    // Registering a folder again (any spelling of its path) returns the same dataset.
    let folder = data_dir.join("reg");
    yolo_files(&folder);
    let first = checked(c, &schemas, "register_dataset", json!({ "path": folder, "name": "reg" }));
    let again = c.call("register_dataset", json!({ "path": folder.join("..").join("reg"), "name": "reg2" }));
    assert_eq!(again["structuredContent"]["id"], first["id"], "{again}");
    assert!(again["content"][0]["text"].as_str().unwrap().contains("already registered"), "{again}");

    let detail = checked(c, &schemas, "get_dataset", json!({ "dataset_id": ds_id }));
    assert!(detail["stats"]["class_counts"].is_array());
    assert_eq!(detail["issues"], json!([]));
    // traffic-cams has broken labels, which get_dataset lists as issues.
    let cams = checked(c, &schemas, "get_dataset", json!({ "dataset_id": "ds_7c2f" }));
    assert!(cams["issues_total"].as_u64().unwrap() >= 1, "{cams}");
    assert!(cams["issues"][0]["kind"].is_string());
    // Errors first, then warnings; infos (empty images) are only counted.
    let counts = &cams["issue_counts"];
    assert_eq!(counts["error"], 3, "{cams}");
    assert_eq!(counts["by_kind"].get("empty_image").cloned().unwrap_or(json!(0)), counts["info"], "{counts}");
    let issues = cams["issues"].as_array().unwrap();
    let ranks: Vec<u8> = issues.iter().map(|i| if i["severity"] == "error" { 0 } else { 1 }).collect();
    assert!(issues.iter().all(|i| i["severity"] != "info") && ranks.is_sorted(), "{issues:?}");
    assert!(issues[0]["fix"].is_string() && issues[0]["line"].is_u64(), "{}", issues[0]);
    let errors =
        checked(c, &schemas, "list_dataset_issues", json!({ "dataset_id": "ds_7c2f", "severity": "error" }));
    assert_eq!((errors["total"].as_u64(), errors["items"].as_array().unwrap().len()), (Some(3), 3));
    assert_eq!(errors["counts"], cams["issue_counts"]);
    let page = checked(
        c,
        &schemas,
        "list_dataset_issues",
        json!({ "dataset_id": "ds_7c2f", "severity": "error", "limit": 2 }),
    );
    assert_eq!((page["items"].as_array().unwrap().len(), &page["next_cursor"]), (2, &json!("o2")));
    let rest = checked(
        c,
        &schemas,
        "list_dataset_issues",
        json!({ "dataset_id": "ds_7c2f", "severity": "error", "cursor": "o2" }),
    );
    assert_eq!(rest["items"][0], errors["items"][2]);
    let kind = checked(
        c,
        &schemas,
        "list_dataset_issues",
        json!({ "dataset_id": "ds_7c2f", "kind": "empty_image" }),
    );
    assert_eq!(kind["total"], counts["info"]);
    c.tool_error(
        "list_dataset_issues",
        json!({ "dataset_id": "ds_7c2f", "kind": "typo" }),
        "invalid_arguments",
    );
    c.tool_error(
        "list_dataset_issues",
        json!({ "dataset_id": "ds_7c2f", "severity": "fatal" }),
        "invalid_arguments",
    );

    let t = checked(
        c,
        &schemas,
        "train",
        json!({ "dataset_id": ds_id, "epochs": 2, "batch": 4, "imgsz": 64, "name": "tiny-n" }),
    );
    let run_id = t["run_id"].as_str().unwrap().to_string();
    assert_eq!(t["status"], "queued");
    let queued = checked(c, &schemas, "run_status", json!({ "run_id": run_id, "log_lines": 5 }));
    assert!(matches!(queued["status"].as_str(), Some("queued" | "running")), "{queued}");
    assert!(queued["log"].as_array().unwrap().len() <= 5);

    let status = checked(c, &schemas, "run_status", json!({ "run_id": "run_1a01", "log_lines": 5 }));
    assert_eq!(status["status"], "completed", "{status}");
    assert!(status["last_epoch"]["epoch"].as_u64().unwrap() >= 1, "{status}");
    let model_id = status["model_id"].as_str().unwrap().to_string();

    let pred = checked(
        c,
        &schemas,
        "predict",
        json!({ "model_id": model_id, "dataset_id": ds_id, "image_id": "000017", "conf": 0.1 }),
    );
    assert!(pred["boxes"].is_array());
    assert_eq!(pred["image"]["width"], 64);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
    checked(c, &schemas, "predict", json!({ "model_id": model_id, "image_base64": b64, "max_det": 3 }));

    // The rest of the table.
    std::fs::write(data_dir.join("a.jpg"), &jpeg).unwrap();
    std::fs::write(data_dir.join("b.jpg"), &jpeg).unwrap();
    std::fs::write(data_dir.join("c.jpg"), b"not an image").unwrap();
    checked(c, &schemas, "predict", json!({ "model_id": model_id, "image_path": "a.jpg" }));
    let id = c.start(
        "tools/call",
        json!({ "name": "predict_batch", "arguments": { "model_id": model_id, "glob": "*.jpg" } }),
        json!({ "progressToken": 7 }),
    );
    let r = c.response(id)["result"].clone();
    assert_ne!(r["isError"], json!(true), "{r}");
    let batch = r["structuredContent"].clone();
    validate(&schemas["predict_batch"], &batch, "predict_batch");
    assert_eq!(batch["processed"], 2);
    assert_eq!(batch["failed"][0]["code"], "image_decode_failed");
    let link =
        r["content"].as_array().unwrap().iter().find(|c| c["type"] == "resource_link").unwrap().clone();
    assert_eq!(link["uri"], batch["details_uri"]);
    assert!(
        c.notes.iter().any(|n| n["method"] == "notifications/progress" && n["params"]["progressToken"] == 7)
    );
    c.notes.retain(|n| n["method"] != "notifications/progress");
    let details = c.result("resources/read", json!({ "uri": batch["details_uri"] }));
    let details: Value = serde_json::from_str(details["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(details["images"].as_array().unwrap().len(), 3);

    let val = checked(c, &schemas, "val", json!({ "model_id": model_id, "dataset_id": ds_id }));
    assert_eq!(val["images"], 4);
    assert!(val["dataset_name"].is_string(), "{val}");
    checked(c, &schemas, "export", json!({ "model_id": model_id, "format": "onnx" }));
    let models = checked(c, &schemas, "list_models", json!({ "limit": 2 }));
    assert_eq!(models["items"].as_array().unwrap().len(), 2);
    // Head and training settings, as on the API's Model (mock: shapes-s is o2o, fasteners-n trained with AMP).
    let models = checked(c, &schemas, "list_models", json!({ "limit": 200 }));
    let item = |id: &str| models["items"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone();
    assert_eq!(
        (item("m_31ab")["head"].clone(), item("m_31aa")["head"].clone()),
        (json!("o2o"), json!("o2m"))
    );
    assert_eq!(item("m_53d1")["trained_with"], json!({ "amp": true, "imgsz": 640, "epochs": 80 }));
    assert!(item("m_53d2")["trained_with"].is_null());
    // Stored dtype and fusing (mock: m_53d2 is the fused f16 export of m_53d1's checkpoint).
    assert_eq!((&item("m_53d2")["precision"], &item("m_53d2")["fused"]), (&json!("f16"), &json!(true)));
    assert_eq!((&item("m_53d1")["precision"], &item("m_53d1")["fused"]), (&json!("f32"), &json!(false)));
    let runs = checked(c, &schemas, "list_runs", json!({ "status": "completed" }));
    assert!(runs["items"].as_array().unwrap().iter().all(|r| r["status"] == "completed"));
    let cmp = checked(c, &schemas, "compare_runs", json!({ "run_ids": [run_id, "run_2b02"] }));
    assert!(cmp["config_diff"]["dataset_id"].is_array());
    // Throughput and estimates are resources: the mock measured folo-n at 640 on its GPU; the CPU has only the
    // reference.
    let mut json_resource = |uri: &str| -> Value {
        let r = c.result("resources/read", json!({ "uri": uri }));
        assert_eq!(r["contents"][0]["mimeType"], "application/json", "{uri}");
        serde_json::from_str(r["contents"][0]["text"].as_str().unwrap()).unwrap()
    };
    let t = json_resource("folo://system/throughput");
    assert_eq!(t["training"][0]["source"], "measured", "{t}");
    let est = json_resource("folo://estimate/ds_5a1e?epochs=50");
    assert_eq!(
        (est["source"].as_str(), est["confidence"].as_str()),
        (Some("measured"), Some("high")),
        "{est}"
    );
    assert!(est["est_total_s_low"].as_f64() < est["est_total_s"].as_f64(), "{est}");
    let est = json_resource("folo://estimate/ds_5a1e?device=cpu&imgsz=320");
    assert_eq!(est["source"], "reference", "{est}");
    for bad in
        ["folo://estimate/nope", "folo://estimate/ds_5a1e?lr0=1", "folo://estimate/ds_5a1e?epochs=many"]
    {
        let e = c.request("resources/read", json!({ "uri": bad }));
        assert!(e["error"]["code"].is_i64(), "{bad}: {e}");
    }
    checked(c, &schemas, "list_datasets", json!({}));
    let live = checked(c, &schemas, "list_runs", json!({ "status": "running", "limit": 1 }));
    if let Some(r) = live["items"].as_array().unwrap().first() {
        checked(c, &schemas, "stop_run", json!({ "run_id": r["id"] }));
    }

    // Tool errors (isError) vs protocol errors (JSON-RPC).
    let e = c.tool_error("get_dataset", json!({ "dataset_id": "nope" }), "dataset_not_found");
    assert!(e["hint"].as_str().unwrap().contains("list_datasets"), "{e}");
    c.tool_error("list_datasets", json!({ "limit": "many" }), "invalid_arguments");
    c.tool_error("get_dataset", json!({}), "invalid_arguments");
    c.tool_error("list_datasets", json!({ "limit": 500 }), "invalid_arguments");
    c.tool_error("predict", json!({ "model_id": model_id }), "invalid_arguments");
    c.tool_error("compare_runs", json!({ "run_ids": [run_id] }), "invalid_arguments");
    c.tool_error("reorder_run", json!({ "run_id": "run_2b02", "position": 1 }), "run_not_queued");
    c.tool_error(
        "reorder_run",
        json!({ "run_id": run_id, "action": "up", "position": 1 }),
        "invalid_arguments",
    );
    c.tool_error("reorder_run", json!({ "run_id": run_id, "position": 0 }), "invalid_arguments");
    c.tool_error("reorder_run", json!({ "run_id": "run_none", "action": "top" }), "run_not_found");
    // The mock store does not resume runs (the real one does, see `real_store_end_to_end`).
    c.tool_error("train", json!({ "dataset_id": ds_id, "resume_run_id": run_id }), "not_implemented");
    c.tool_error("train", json!({ "resume_run_id": 7 }), "invalid_arguments");
    // Unknown arguments are rejected, with the likely name (`lr` for `lr0`), also inside `augment`.
    let e =
        c.tool_error("train", json!({ "dataset_id": ds_id, "epochs": 1, "lr": 0.1 }), "invalid_arguments");
    assert_eq!(e["message"], "unknown argument `lr`; did you mean `lr0`?");
    let e = c.tool_error(
        "train",
        json!({ "dataset_id": ds_id, "augment": { "mosiac": 0 } }),
        "invalid_arguments",
    );
    assert!(e["message"].as_str().unwrap().contains("augment.mosiac"), "{e}");
    c.tool_error("list_runs", json!({ "state": "running" }), "invalid_arguments");
    // One vocabulary with the CLI: a bad TrainConfig value is invalid_config, an empty split invalid_dataset.
    c.tool_error("train", json!({ "dataset_id": ds_id, "arch": "folo-q" }), "invalid_config");
    c.tool_error(
        "val",
        json!({ "model_id": model_id, "dataset_id": ds_id, "split": "test" }),
        "invalid_dataset",
    );
    let unknown = c.request("tools/call", json!({ "name": "no_such_tool", "arguments": {} }));
    assert_eq!(unknown["error"]["code"], -32602, "{unknown}");
    let method = c.request("no/such/method", json!({}));
    assert_eq!(method["error"]["code"], -32601, "{method}");
    // Cloud: the mock's connection, its offers, and run_3c01's instance (terminating one that is gone answers it).
    let conns = checked(c, &schemas, "cloud_connections", json!({}));
    assert_eq!(conns["connections"][0]["id"], "cc_4d2a");
    let offers = checked(c, &schemas, "cloud_offers", json!({ "connection_id": "cc_4d2a" }));
    assert!(offers["offers"].as_array().unwrap().len() > 1);
    c.tool_error("cloud_offers", json!({ "connection_id": "cc_none" }), "connection_not_found");
    let instances = checked(c, &schemas, "cloud_instances", json!({}));
    let seeded = instances["instances"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["run_id"] == "run_3c01")
        .unwrap()
        .clone();
    let ended = checked(c, &schemas, "cloud_terminate", json!({ "instance_id": seeded["id"] }));
    assert_eq!((ended["status"].as_str(), &ended["ended"]), (Some("terminated"), &seeded["ended"]));
    c.tool_error("cloud_terminate", json!({ "instance_id": "ci_none" }), "instance_not_found");
    // The live run's change notice, which came in meanwhile.
    let update =
        c.notification(Duration::from_secs(20), |n| n["method"] == "notifications/resources/updated");
    assert_eq!(update["params"]["uri"], run_uri);
    if let Some(listen) = subscribed {
        assert_eq!(
            update["params"]["_meta"]["io.modelcontextprotocol/subscriptionId"],
            json!(listen),
            "{update}"
        );
        c.send(&json!({ "jsonrpc": "2.0", "method": "notifications/cancelled", "params": { "requestId": listen } }));
    } else {
        c.result("resources/unsubscribe", json!({ "uri": run_uri }));
    }
    let snap = c.result("resources/read", json!({ "uri": format!("{run_uri}/epochs") }));
    let epochs: Value = serde_json::from_str(snap["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert!(epochs["epochs"].is_array() && epochs.get("steps").is_none(), "{epochs}");
    let snap = c.result("resources/read", json!({ "uri": run_uri }));
    let snap: Value = serde_json::from_str(snap["contents"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(snap["run"]["id"], "run_3c02");
    assert!(snap["revision"].as_u64().unwrap() >= 1);
    // Unparsable input is ignored (as by every official SDK); the server keeps working.
    writeln!(c.stdin, "this is not json").unwrap();
    c.ok("folo_info", json!({}));
}

#[test]
fn revision_2025_11_25() {
    let dir = tmp_dir("old");
    let mut c = spawn(&["--mock", "--data-dir", dir.to_str().unwrap()], None);
    let init = c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "folo-test", "version": "1" } }),
    );
    assert_eq!(init["protocolVersion"], OLD);
    assert_eq!(init["serverInfo"]["name"], "folo");
    let caps = &init["capabilities"];
    assert_eq!(caps["tools"]["listChanged"], false);
    assert_eq!(caps["resources"]["subscribe"], true);
    assert!(caps["prompts"].is_object() && caps["logging"].is_object());
    c.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
    // A 2026-only method is not part of this revision.
    let listen = c.request("subscriptions/listen", json!({ "notifications": {} }));
    assert_eq!(listen["error"]["code"], -32601, "{listen}");
    full_flow(&mut c, &dir);
    c.finish();
}

#[test]
fn revision_2026_07_28() {
    let dir = tmp_dir("new");
    let mut c = spawn(&["--mock", "--data-dir", dir.to_str().unwrap()], Some(NEW));
    // A direct tool call, without discovery or a handshake.
    let info = c.ok("folo_info", json!({}));
    assert_eq!(info["mode"], "mock");
    let disc = c.result("server/discover", json!({}));
    assert!(disc["supportedVersions"].as_array().unwrap().contains(&json!(NEW)), "{disc}");
    assert!(disc["supportedVersions"].as_array().unwrap().contains(&json!(OLD)), "{disc}");
    assert_eq!(disc["capabilities"]["resources"]["subscribe"], true);
    // Unsupported version in a request's metadata.
    let id = c.start(
        "tools/call",
        json!({ "name": "folo_info", "arguments": {} }),
        json!({ "io.modelcontextprotocol/protocolVersion": "2099-01-01" }),
    );
    let bad = c.response(id);
    assert!(bad["error"]["code"].is_i64(), "{bad}");
    // The legacy subscription method is not part of this revision.
    let legacy = c.request("resources/subscribe", json!({ "uri": "folo://runs/run_3c02" }));
    assert_eq!(legacy["error"]["code"], -32601, "{legacy}");
    full_flow(&mut c, &dir);
    c.finish();
}

#[test]
fn unsupported_version_before_anything_else() {
    let mut c = spawn(&["--mock"], Some("2099-01-01"));
    let r = c.request("tools/call", json!({ "name": "folo_info", "arguments": {} }));
    assert!(r["error"]["code"].is_i64(), "{r}");
    assert!(r["error"]["data"].to_string().contains(NEW), "the error lists the supported versions: {r}");
    c.finish();
}

#[test]
fn results_stay_under_the_size_limit() {
    let mut c = spawn(&["--mock"], None);
    init_old(&mut c);
    let long = "x".repeat(200);
    c.tool_error("create_synthetic_dataset", json!({ "name": long, "images": 5 }), "invalid_arguments");
    let classes: Vec<String> = (0..81).map(|i| format!("c{i}")).collect();
    c.tool_error(
        "create_synthetic_dataset",
        json!({ "name": "n", "images": 5, "classes": classes }),
        "invalid_arguments",
    );
    c.tool_error("train", json!({ "dataset_id": "ds_5a1e", "name": long }), "invalid_arguments");
    // Error results are bounded too, even when the arguments are huge.
    let huge = "q".repeat(40_000);
    let r = c.call("list_datasets", json!({ "cursor": huge }));
    assert_eq!(r["isError"], true);
    assert!(serde_json::to_vec(&r).unwrap().len() < 4096, "{}", serde_json::to_vec(&r).unwrap().len());
    let r = c.call("get_dataset", json!({ "dataset_id": huge }));
    assert!(serde_json::to_vec(&r).unwrap().len() < 4096);
    // The largest allowed dataset still fits, inline and in a page of 200.
    let classes: Vec<String> = (0..80).map(|i| format!("{i:0>64}")).collect();
    let name = "y".repeat(128);
    for _ in 0..3 {
        c.ok(
            "create_synthetic_dataset",
            json!({ "name": name, "images": 5, "imgsz": 64, "classes": classes }),
        );
    }
    let page = c.ok("list_datasets", json!({ "limit": 200 }));
    assert_eq!(page["total"], 6);
    let mut seen = page["items"].as_array().unwrap().len();
    let mut cursor = page["next_cursor"].clone();
    while let Some(cur) = cursor.as_str().map(str::to_string) {
        let p = c.ok("list_datasets", json!({ "cursor": cur, "limit": 200 }));
        seen += p["items"].as_array().unwrap().len();
        cursor = p["next_cursor"].clone();
    }
    assert_eq!(seen, 6);
    c.finish();
}

#[test]
fn resources_list_handles_large_cursors_without_overflow() {
    let mut c = spawn(&["--mock"], None);
    init_old(&mut c);
    let first = c.result("resources/list", json!({}));
    assert!(!first["resources"].as_array().unwrap().is_empty(), "{first}");
    for cursor in [format!("o{}", usize::MAX - 95), format!("o{}", usize::MAX)] {
        let r = c.result("resources/list", json!({ "cursor": cursor }));
        assert_eq!(r["resources"], json!([]), "{r}");
        assert!(r.get("nextCursor").is_none_or(Value::is_null), "{r}");
    }
    c.finish();
}
