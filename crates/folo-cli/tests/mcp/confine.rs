//! `--read-only` and `--confine`: hidden and refused tools, paths outside the data dir, and what a
//! dataset reads.

use base64::Engine as _;
use serde_json::{Value, json};

use super::harness::{
    OLD, check_tools, init_old, make_dir_link, remove_dir_link, spawn, tmp_dir, yolo_files,
};

#[test]
fn read_only_hides_and_rejects_mutating_tools() {
    let mut c = spawn(&["--mock", "--read-only"], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    let listed = [
        "folo_info",
        "list_datasets",
        "get_dataset",
        "list_dataset_issues",
        "list_dataset_images",
        "list_models",
        "run_status",
        "list_runs",
        "val",
        "predict",
        "predict_batch",
        "compare_runs",
        "cloud_connections",
        "cloud_offers",
        "cloud_instances",
        "list_label_jobs",
        "label_status",
        "list_label_images",
        "view_label_image",
        "bench_catalog",
        "list_benchmarks",
        "get_benchmark",
        "bench_entry",
    ];
    check_tools(&mut c, &listed);
    for (tool, args) in [
        ("train", json!({ "dataset_id": "ds_5a1e" })),
        ("create_synthetic_dataset", json!({ "name": "x" })),
        ("register_dataset", json!({ "path": "." })),
        ("stop_run", json!({ "run_id": "run_3c02" })),
        ("reorder_run", json!({ "run_id": "run_3c02", "action": "top" })),
        ("export", json!({ "model_id": "m_31aa", "format": "onnx" })),
        ("cloud_terminate", json!({ "instance_id": "ci_3c01" })),
        ("create_label_job", json!({ "name": "x", "classes": ["a"] })),
        ("start_labeling", json!({ "job_id": "lb_1", "agent": "claude" })),
        ("set_label_boxes", json!({ "job_id": "lb_1", "image_id": "1", "boxes": [] })),
        ("export_label_dataset", json!({ "job_id": "lb_1" })),
        ("create_benchmark", json!({ "name": "x", "dataset_id": "ds_5a1e" })),
        ("delete_benchmark", json!({ "benchmark_id": "bn_1" })),
        ("start_bench_entry", json!({ "benchmark_id": "bn_1", "model_id": "m_31aa" })),
        ("stop_bench_entry", json!({ "benchmark_id": "bn_1", "entry_id": "be_1" })),
    ] {
        c.tool_error(tool, args, "read_only");
    }
    assert_eq!(c.ok("folo_info", json!({}))["read_only"], true);
    let prompts = c.result("prompts/list", json!({}))["prompts"].as_array().unwrap().clone();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0]["name"], "diagnose_run");
    // Unknown tool/prompt suggestions never reveal hidden mutating tools or prompts in --read-only mode.
    let hidden_tool = c.request("tools/call", json!({ "name": "trani", "arguments": {} }));
    assert_eq!(hidden_tool["error"]["code"], -32602, "{hidden_tool}");
    assert!(!hidden_tool["error"]["message"].as_str().unwrap().contains("train"), "{hidden_tool}");
    let visible_tool = c.request("tools/call", json!({ "name": "predcit", "arguments": {} }));
    assert_eq!(visible_tool["error"]["code"], -32602, "{visible_tool}");
    assert!(
        visible_tool["error"]["message"].as_str().unwrap().contains("did you mean `predict`?"),
        "{visible_tool}"
    );
    let hidden_prompt = c.request("prompts/get", json!({ "name": "label_folders", "arguments": {} }));
    assert_eq!(hidden_prompt["error"]["code"], -32602, "{hidden_prompt}");
    assert!(!hidden_prompt["error"]["message"].as_str().unwrap().contains("did you mean"), "{hidden_prompt}");
    c.finish();
}

#[test]
fn confine_rejects_paths_outside_the_data_dir() {
    let root = tmp_dir("confine");
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(root.join("outside.jpg"), b"x").unwrap();
    let mut c = spawn(&["--mock", "--confine", "--data-dir", data.to_str().unwrap()], None);
    c.result(
        "initialize",
        json!({ "protocolVersion": OLD, "capabilities": {}, "clientInfo": { "name": "t", "version": "1" } }),
    );
    let outside = root.join("outside.jpg");
    c.tool_error("predict", json!({ "model_id": "m_31aa", "image_path": outside }), "path_outside_data_dir");
    c.tool_error(
        "predict",
        json!({ "model_id": "m_31aa", "image_path": "../outside.jpg" }),
        "path_outside_data_dir",
    );
    c.tool_error(
        "predict_batch",
        json!({ "model_id": "m_31aa", "glob": "../*.jpg" }),
        "path_outside_data_dir",
    );
    c.tool_error("register_dataset", json!({ "path": root.path() }), "path_outside_data_dir");
    // Labeling reads images in place: folders and files to label are confined the same way.
    let job = json!({ "name": "x", "classes": ["a"], "paths": [root.path()] });
    c.tool_error("create_label_job", job, "path_outside_data_dir");
    let add = json!({ "job_id": "lb_1", "paths": ["../outside.jpg"] });
    c.tool_error("create_label_job", add, "path_outside_data_dir");
    // A benchmark's files are not confined, so its tools are refused.
    c.tool_error("create_benchmark", json!({ "name": "x", "dataset_id": "ds_5a1e" }), "not_supported");
    c.tool_error("bench_catalog", json!({}), "not_supported");
    c.finish();
}

#[test]
fn confine_checks_what_a_dataset_reads() {
    let root = tmp_dir("confine-ds");
    let data = root.join("data");
    let outside = root.join("outside");
    yolo_files(&outside);
    // Inside the data dir, but its data.yaml reads the images from outside.
    let sneaky = data.join("sneaky");
    std::fs::create_dir_all(&sneaky).unwrap();
    let yaml =
        format!("path: {}\ntrain: images/train\nval: images/train\nnames: [thing]\n", outside.display());
    std::fs::write(sneaky.join("data.yaml"), yaml.replace('\\', "/")).unwrap();
    let honest = data.join("honest");
    yolo_files(&honest);

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error("register_dataset", json!({ "path": "sneaky" }), "path_outside_data_dir");
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 0, "the rejected dataset was unregistered again");
    // An image list inside the root naming outside images, and a list file outside the root.
    let listed = data.join("listed");
    std::fs::create_dir_all(&listed).unwrap();
    let outside_img = outside.join("images/train/a.jpg").to_string_lossy().replace('\\', "/");
    std::fs::write(listed.join("train.txt"), format!("{outside_img}\n")).unwrap();
    std::fs::write(listed.join("data.yaml"), "train: train.txt\nval: train.txt\nnames: [thing]\n").unwrap();
    c.tool_error("register_dataset", json!({ "path": "listed" }), "path_outside_data_dir");
    let far = data.join("far");
    std::fs::create_dir_all(&far).unwrap();
    std::fs::write(root.join("list.txt"), "images/train/a.jpg\n").unwrap();
    let list = root.join("list.txt").to_string_lossy().replace('\\', "/");
    std::fs::write(far.join("data.yaml"), format!("train: {list}\nnames: [thing]\n")).unwrap();
    c.tool_error("register_dataset", json!({ "path": "far" }), "path_outside_data_dir");
    // A list outside whose name contains a comma, naming an image inside: the whole name must be checked.
    let comma = data.join("comma");
    std::fs::create_dir_all(&comma).unwrap();
    std::fs::write(root.join("secret,list.txt"), "../honest/images/train/a.jpg\n").unwrap();
    std::fs::write(comma.join("data.yaml"), "train: ../../secret,list.txt\nnames: [thing]\n").unwrap();
    c.tool_error("register_dataset", json!({ "path": "comma" }), "path_outside_data_dir");
    // An indented key (a YAML form the precheck's parser need not understand) naming an outside list.
    let indented = data.join("indented");
    std::fs::create_dir_all(&indented).unwrap();
    std::fs::write(indented.join("data.yaml"), "  train: ../../list.txt\nnames: [thing]\n").unwrap();
    // (For the reader this one is not valid YAML at all; either way nothing outside is read.)
    c.tool_error("register_dataset", json!({ "path": "indented" }), "invalid_dataset");
    // A folder inside the root that is a link (a junction on Windows) to an outside folder.
    let linked = data.join("linked");
    std::fs::create_dir_all(linked.join("labels/train")).unwrap();
    assert!(
        make_dir_link(&outside.join("images"), &linked.join("images")),
        "could not create a directory link"
    );
    c.tool_error("register_dataset", json!({ "path": "linked" }), "path_outside_data_dir");
    remove_dir_link(&linked.join("images"));
    // YAML forms a hand-written parser might miss: a tab after the colon, an escaped scalar, another manifest name.
    // The reader itself refuses the outside list, so the exact syntax does not matter.
    for (name, yaml) in [
        ("tabbed", "train:\t../../list.txt\nnames: [thing]\n"),
        ("escaped", "train: \"../../l\\u0069st.txt\"\nnames: [thing]\n"),
    ] {
        let d = data.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("data.yaml"), yaml).unwrap();
        c.tool_error("register_dataset", json!({ "path": name }), "path_outside_data_dir");
    }
    let alt = data.join("alt");
    std::fs::create_dir_all(&alt).unwrap();
    std::fs::write(alt.join("dataset.yml"), "train: ../../list.txt\nnames: [thing]\n").unwrap();
    c.tool_error("register_dataset", json!({ "path": "alt" }), "path_outside_data_dir");
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 0);
    // Class names as a flow map are not paths (and are not valid file names on Windows).
    let mapped = data.join("mapped");
    yolo_files(&mapped);
    std::fs::write(
        mapped.join("data.yaml"),
        "train: images/train\nval: images/train\nnames: {0: cat, 1: dog}\n",
    )
    .unwrap();
    let m = c.ok("register_dataset", json!({ "path": "mapped" }));
    assert_eq!(m["classes"], json!(["cat", "dog"]));
    // Edited after registration to read outside: the re-index on the next request cannot read it, although this
    // session has checked the dataset before.
    let mapped_id = m["id"].as_str().unwrap().to_string();
    let outside_dir = outside.join("images/train").to_string_lossy().replace('\\', "/");
    std::fs::write(
        mapped.join("data.yaml"),
        format!("train: {outside_dir}\nval: {outside_dir}\nnames: [cat]\n"),
    )
    .unwrap();
    let after = c.call("list_dataset_images", json!({ "dataset_id": mapped_id }));
    if after["isError"] == true {
        // The re-index refused the outside folder and failed.
        let e: Value = serde_json::from_str(after["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(e["code"], "path_outside_data_dir", "{e}");
    } else {
        let outside_bytes = std::fs::read(outside.join("images/train/a.jpg")).unwrap();
        for im in after["structuredContent"]["items"].as_array().unwrap() {
            let uri = format!(
                "folo://datasets/{mapped_id}/images/{}?size=original",
                im["image_id"].as_str().unwrap()
            );
            let r = c.request("resources/read", json!({ "uri": uri }));
            if let Some(blob) = r["result"]["contents"][0]["blob"].as_str() {
                let bytes = base64::engine::general_purpose::STANDARD.decode(blob).unwrap();
                assert_ne!(bytes, outside_bytes, "an outside image was served: {im}");
            }
        }
    }
    // Even though `mapped` was approved earlier in this same session, once its manifest points outside
    // neither tool nor resource fuzzy suggestions may disclose its ID or name.
    let mapped_typo = format!("{}_x", &mapped_id[..mapped_id.len().saturating_sub(1)]);
    for probe in [mapped_typo.as_str(), "mapped"] {
        let e = c.tool_error("get_dataset", json!({ "dataset_id": probe }), "dataset_not_found");
        assert!(!e.to_string().contains(&mapped_id), "unconfined dataset leaked via tool: {e}");
        assert!(!e.to_string().contains("did you mean"), "unconfined dataset suggested via tool: {e}");
    }
    let res_typo = c.request("resources/read", json!({ "uri": format!("folo://datasets/{mapped_typo}") }));
    assert!(res_typo["error"]["code"].is_i64(), "{res_typo}");
    assert!(!res_typo.to_string().contains(&mapped_id), "unconfined dataset leaked via resource: {res_typo}");
    assert!(
        !res_typo.to_string().contains("did you mean"),
        "unconfined dataset suggested via resource: {res_typo}"
    );
    let ds = c.ok("register_dataset", json!({ "path": "honest" }));
    let id = ds["id"].as_str().unwrap().to_string();
    let imgs = c.ok("list_dataset_images", json!({ "dataset_id": id }));
    let image_id = imgs["items"][0]["image_id"].as_str().unwrap().to_string();
    let img = c.result("resources/read", json!({ "uri": format!("folo://datasets/{id}/images/{image_id}") }));
    assert_eq!(img["contents"][0]["mimeType"], "image/jpeg");
    c.finish();

    // A dataset registered without --confine (persisted by the real store) is invisible with it.
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let out_ds = c.ok("register_dataset", json!({ "path": outside }));
    let out_id = out_ds["id"].as_str().unwrap().to_string();
    c.finish();
    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let list = c.ok("list_datasets", json!({}));
    let ids: Vec<&str> =
        list["items"].as_array().unwrap().iter().map(|d| d["id"].as_str().unwrap()).collect();
    // `mapped` now reads outside (edited above) and is hidden too.
    assert_eq!(ids, [id.as_str()]);
    c.tool_error("get_dataset", json!({ "dataset_id": out_id }), "path_outside_data_dir");
    // Neither an ID typo nor the display name of a confined dataset may leak it in a suggestion.
    let out_typo = format!("{}_x", &out_id[..out_id.len().saturating_sub(1)]);
    for probe in [out_typo.as_str(), "outside"] {
        let missing_out = c.tool_error("get_dataset", json!({ "dataset_id": probe }), "dataset_not_found");
        assert!(
            !missing_out.to_string().contains(&out_id),
            "confined dataset leaked in suggestion: {missing_out}"
        );
        assert!(
            !missing_out.to_string().contains("did you mean"),
            "confined dataset suggested for `{probe}`: {missing_out}"
        );
    }
    c.tool_error("list_dataset_images", json!({ "dataset_id": out_id }), "path_outside_data_dir");
    let read = c.request("resources/read", json!({ "uri": format!("folo://datasets/{out_id}/images/0") }));
    assert!(read.get("error").is_some(), "{read}");
    let res = c.result("resources/list", json!({}))["resources"].clone();
    assert!(!res.to_string().contains(&out_id));
    c.finish();
}

#[test]
fn confine_rebuilds_persisted_indexes() {
    // A COCO dataset whose images are inside but whose annotations folder links outside, registered without
    // --confine (so its index is cached), must not be trusted from that cache once --confine is on.
    let root = tmp_dir("confine-coco");
    let data = root.join("data");
    let ds = data.join("coco");
    std::fs::create_dir_all(ds.join("images")).unwrap();
    image::RgbImage::from_pixel(32, 24, image::Rgb([9, 9, 9])).save(ds.join("images/a.jpg")).unwrap();
    let ann = root.join("outside-ann");
    std::fs::create_dir_all(&ann).unwrap();
    let coco = json!({
        "images": [{ "id": 1, "file_name": "a.jpg", "width": 32, "height": 24 }],
        "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0 }],
        "categories": [{ "id": 1, "name": "secretcat" }],
    });
    std::fs::write(ann.join("instances_train.json"), coco.to_string()).unwrap();
    assert!(make_dir_link(&ann, &ds.join("annotations")), "could not create a directory link");
    // And a YOLO dataset whose labels folder links outside (it still indexes under the guard, minus the labels).
    let y = data.join("yolo");
    std::fs::create_dir_all(y.join("images/train")).unwrap();
    image::RgbImage::from_pixel(32, 24, image::Rgb([9, 9, 9])).save(y.join("images/train/a.jpg")).unwrap();
    let out_labels = root.join("outside-labels");
    std::fs::create_dir_all(out_labels.join("train")).unwrap();
    std::fs::write(
        out_labels.join("train/a.txt"),
        "0 0.5 0.5 0.25 0.25
",
    )
    .unwrap();
    assert!(make_dir_link(&out_labels, &y.join("labels")));
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let reg = c.ok("register_dataset", json!({ "path": "coco" }));
    assert_eq!(reg["classes"], json!(["secretcat"]));
    let id = reg["id"].as_str().unwrap().to_string();
    let yreg = c.ok("register_dataset", json!({ "path": "yolo" }));
    assert_eq!(yreg["stats"]["boxes"], 1);
    let yid = yreg["id"].as_str().unwrap().to_string();
    c.finish();

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let list = c.ok("list_datasets", json!({}));
    assert!(!list.to_string().contains("secretcat"), "{list}");
    assert_eq!(list["total"], 0, "{list}");
    c.tool_error("get_dataset", json!({ "dataset_id": yid }), "path_outside_data_dir");
    let r = c.call("get_dataset", json!({ "dataset_id": id }));
    assert_eq!(r["isError"], true);
    assert!(!r.to_string().contains("secretcat"), "{r}");
    c.finish();

    // The confined session's partial index was not cached: without --confine the dataset is whole again.
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let d = c.ok("get_dataset", json!({ "dataset_id": yid }));
    assert_eq!(d["stats"]["boxes"], 1, "{d}");
    let d = c.ok("get_dataset", json!({ "dataset_id": id }));
    assert_eq!(d["classes"], json!(["secretcat"]), "{d}");
    assert_eq!(d["splits"]["train"].as_u64().unwrap() + d["splits"]["val"].as_u64().unwrap(), 1, "{d}");
    c.finish();
}

#[test]
fn confine_keeps_yolo_labels_inside() {
    // With the data dir at <tmp>/images, the label of <tmp>/images/ds/a.jpg would be <tmp>/labels/ds/a.txt.
    let root = tmp_dir("confine-labels");
    let data = root.join("images");
    let ds = data.join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])).save(ds.join("a.jpg")).unwrap();
    std::fs::write(ds.join("list.txt"), "a.jpg\n").unwrap();
    std::fs::write(ds.join("data.yaml"), "train: list.txt\nnames: [secret]\n").unwrap();
    std::fs::create_dir_all(root.join("labels/ds")).unwrap();
    std::fs::write(root.join("labels/ds/a.txt"), "0 0.5 0.5 0.9 0.9\n").unwrap();

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error("register_dataset", json!({ "path": "ds" }), "path_outside_data_dir");
    c.finish();

    // Registered without --confine (persisted), the dataset is rejected with it: its labels are outside.
    let mut c = spawn(&["--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let id = c.ok("register_dataset", json!({ "path": "ds" }))["id"].as_str().unwrap().to_string();
    c.finish();
    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error(
        "list_dataset_images",
        json!({ "dataset_id": id, "include_boxes": true }),
        "path_outside_data_dir",
    );
    assert_eq!(c.ok("list_datasets", json!({}))["total"], 0);
    c.finish();
}

/// With `--confine`, `folo mcp` refuses every benchmark tool and does not read or clean up `<data_dir>/bench/` at
/// startup.
#[test]
fn confined_mcp_does_not_open_bench_folder_at_startup() {
    let data = folo_test_support::TestDir::new("mcp-confine-no-bench-open");
    // An incomplete benchmark folder (missing `bench.json`) would be deleted by `Benches::open`.
    let canary = data.join("bench").join("bn_unfinished");
    std::fs::create_dir_all(&canary).unwrap();

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    c.tool_error("list_benchmarks", json!({}), "not_supported");
    c.finish();

    assert!(canary.is_dir(), "`folo mcp --confine` must not touch `<data_dir>/bench/` at startup");
}

/// When `--data-dir` (or an ancestor) is named `images` and no outside `labels/` mirror exists, `folo-data` reads
/// YOLO labels beside the images inside the dataset; `--confine` must follow the same rule and allow the dataset.
#[test]
fn confined_mcp_allows_beside_image_labels_under_images_data_dir() {
    let root = folo_test_support::TestDir::new("mcp-confine-beside-labels");
    let data = root.join("images");
    let ds = data.join("ds");
    std::fs::create_dir_all(&ds).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])).save(ds.join("a.jpg")).unwrap();
    std::fs::write(ds.join("a.txt"), "0 0.5 0.5 0.4 0.4\n").unwrap();
    std::fs::write(ds.join("list.txt"), "a.jpg\n").unwrap();
    std::fs::write(ds.join("data.yaml"), "train: list.txt\nnames: [box]\n").unwrap();

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);
    let reg = c.ok("register_dataset", json!({ "path": "ds" }));
    let id = reg["id"].as_str().unwrap().to_string();
    let page = c.ok("list_dataset_images", json!({ "dataset_id": id, "include_boxes": true }));
    assert_eq!(page["total"], 1);
    assert_eq!(page["items"][0]["boxes"].as_array().unwrap().len(), 1);
    c.finish();
}

/// Direct `data.yaml` registration with a `path:` override resolves images and beside-image labels from the
/// effective dataset root (`Index.root`), not the registration folder: an unrelated `labels/ds/` mirror outside
/// `--data-dir` must not cause the valid `path: nested` dataset to be rejected.
#[test]
fn confined_mcp_uses_effective_yaml_root_and_ignores_coco_sidecars() {
    let root = folo_test_support::TestDir::new("mcp-confine-yaml-path-and-coco");
    let data = root.join("images");
    let ds = data.join("ds");
    let nested = ds.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])).save(nested.join("a.jpg")).unwrap();
    std::fs::write(nested.join("a.txt"), "0 0.5 0.5 0.4 0.4\n").unwrap();
    std::fs::write(nested.join("list.txt"), "a.jpg\n").unwrap();
    std::fs::write(ds.join("data.yaml"), "path: nested\ntrain: list.txt\nnames: [box]\n").unwrap();
    // An unrelated outside mirror for `images/ds` (NOT for `images/ds/nested`).
    std::fs::create_dir_all(root.join("labels/ds")).unwrap();
    std::fs::write(root.join("labels/ds/a.txt"), "0 0.5 0.5 0.9 0.9\n").unwrap();

    // A COCO dataset under `<root>/images/coco` with an unrelated outside `labels/coco` mirror (and, on Unix, an
    // unused `.txt` symlink beside the image pointing outside `--data-dir`). COCO reads annotations JSON, not
    // YOLO `.txt` sidecars.
    let coco = data.join("coco");
    std::fs::create_dir_all(coco.join("annotations")).unwrap();
    std::fs::create_dir_all(coco.join("train")).unwrap();
    image::RgbImage::from_pixel(8, 8, image::Rgb([2, 3, 4])).save(coco.join("train/a.jpg")).unwrap();
    std::fs::write(
        coco.join("annotations/instances_train.json"),
        json!({
            "images": [{ "id": 1, "file_name": "a.jpg", "width": 8, "height": 8 }],
            "annotations": [{ "id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0 }],
            "categories": [{ "id": 1, "name": "box" }]
        })
        .to_string(),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("labels/coco/train")).unwrap();
    std::fs::write(root.join("labels/coco/train/a.txt"), "0 0.5 0.5 0.9 0.9\n").unwrap();
    #[cfg(unix)]
    {
        std::fs::write(root.join("outside.txt"), "0 0.5 0.5 0.9 0.9\n").unwrap();
        std::os::unix::fs::symlink(root.join("outside.txt"), coco.join("train/a.txt")).unwrap();
    }

    let mut c = spawn(&["--confine", "--data-dir", data.to_str().unwrap()], None);
    init_old(&mut c);

    let yolo_reg = c.ok("register_dataset", json!({ "path": "ds/data.yaml" }));
    let yolo_id = yolo_reg["id"].as_str().unwrap().to_string();
    let yolo_page = c.ok("list_dataset_images", json!({ "dataset_id": yolo_id, "include_boxes": true }));
    assert_eq!(yolo_page["total"], 1);
    assert_eq!(yolo_page["items"][0]["boxes"].as_array().unwrap().len(), 1);

    let coco_reg = c.ok("register_dataset", json!({ "path": "coco" }));
    let coco_id = coco_reg["id"].as_str().unwrap().to_string();
    let coco_page = c.ok("list_dataset_images", json!({ "dataset_id": coco_id, "include_boxes": true }));
    assert_eq!(coco_page["total"], 1);
    assert_eq!(coco_page["items"][0]["boxes"].as_array().unwrap().len(), 1);
    c.finish();
}

/// A confined MCP proxy whose `--data-dir` differs from the proxied `folo serve`'s must refuse
/// `create_synthetic_dataset` before generating or registering anything outside its sandbox; when the two share
/// `--data-dir`, generation succeeds and the returned dataset is immediately usable.
#[test]
fn confined_proxy_refuses_synthetic_dataset_outside_data_dir() {
    use super::harness::folo_cmd;
    use folo_test_support::ChildGuard;
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;

    let root = folo_test_support::TestDir::new("mcp-confine-proxy-synth");
    let server_dir = root.join("server");
    let other_dir = root.join("other");
    std::fs::create_dir_all(&server_dir).unwrap();
    std::fs::create_dir_all(&other_dir).unwrap();

    let mut serve: ChildGuard = folo_cmd()
        .args(["serve", "--device", "cpu", "--port", "0", "--json", "--data-dir"])
        .arg(&server_dir)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut first = String::new();
    BufReader::new(serve.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let url = serde_json::from_str::<Value>(&first).unwrap()["url"].as_str().unwrap().to_string();

    // Different --data-dir: refused before the server writes or registers anything.
    let mut c = spawn(&["--server", &url, "--confine", "--data-dir", other_dir.to_str().unwrap()], None);
    init_old(&mut c);
    let e = c.tool_error(
        "create_synthetic_dataset",
        json!({ "name": "blocked", "images": 4, "imgsz": 64 }),
        "path_outside_data_dir",
    );
    assert!(e["message"].as_str().unwrap().contains("outside"), "{e}");
    c.finish();

    let client = folo_client::Client::new(&url);
    let ds_list: Value = client.get("/datasets").unwrap();
    assert_eq!(ds_list["datasets"].as_array().unwrap().len(), 0, "server must not register blocked dataset");
    assert!(!server_dir.join("datasets").exists(), "server must not write files for blocked dataset");

    // Same --data-dir: succeeds and the returned dataset is immediately visible and readable.
    let mut c = spawn(&["--server", &url, "--confine", "--data-dir", server_dir.to_str().unwrap()], None);
    init_old(&mut c);
    let created = c.ok("create_synthetic_dataset", json!({ "name": "allowed", "images": 4, "imgsz": 64 }));
    let id = created["id"].as_str().unwrap().to_string();
    let got = c.ok("get_dataset", json!({ "dataset_id": id }));
    assert_eq!(got["id"], id);
    let listed = c.ok("list_datasets", json!({}));
    assert_eq!(listed["total"], 1);
    c.finish();

    let _ = serve.kill();
    let _ = serve.wait();
}
