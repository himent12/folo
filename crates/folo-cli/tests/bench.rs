//! `folo bench` with `--json`: the flags, the JSON it prints and its exit codes, on a benchmark of a synthetic
//! dataset run in this process (an untrained folo-n at 64 px on the CPU) and through a `folo serve` child, found by
//! its data dir and named with `--server`. The operations themselves are tested in folo-mcp. Every child has a
//! deadline.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use folo_test_support::{ChildGuard, TestDir};
use serde_json::{Value, json};

/// Longest a `folo bench` command may take (an entry of four 64 px images on the CPU takes about a second).
const DEADLINE: Duration = Duration::from_secs(120);

/// A one-condition plan, so an entry is two passes.
const PLAN: &str = r#"{"conditions":[{"condition":"haze","severities":[2]}]}"#;

/// Runs `cmd` to its end within [`DEADLINE`] (killed and the test failed after it): exit code, stdout, stderr.
fn run(mut cmd: Command) -> (i32, String, String) {
    let child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut child = ChildGuard::new(child);
    let pipe = |mut r: Box<dyn Read + Send>| {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut s = String::new();
            let _ = r.read_to_string(&mut s);
            let _ = tx.send(s);
        });
        rx
    };
    let out = pipe(Box::new(child.stdout.take().unwrap()));
    let err = pipe(Box::new(child.stderr.take().unwrap()));
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() >= DEADLINE {
            let _ = child.kill();
            panic!("{cmd:?} still runs after {DEADLINE:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let read =
        |rx: mpsc::Receiver<String>| rx.recv_timeout(DEADLINE).expect("the command's output never closed");
    (status.code().unwrap_or(-1), read(out), read(err))
}

fn folo() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1")
        .env("FOLO_GPU_ADAPTER", "folo-cli-tests-no-such-adapter")
        .env_remove("FOLO_DATA_DIR")
        .env_remove("FOLO_DEVICE")
        .env_remove("FOLO_SERVER")
        .env_remove("FOLO_TOKEN");
    c
}

/// `folo bench <args> --json` with `conn` after them: exit code, stdout records, stderr.
fn bench(args: &[&str], conn: &[&str]) -> (i32, Vec<Value>, String) {
    let mut c = folo();
    c.arg("bench").args(args).args(conn).arg("--json");
    let (code, stdout, stderr) = run(c);
    let records =
        stdout.lines().map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}"))).collect();
    (code, records, stderr)
}

/// The one record of a command that succeeds.
fn ok(args: &[&str], conn: &[&str]) -> Value {
    let (code, mut records, err) = bench(args, conn);
    assert_eq!(code, 0, "{args:?}: {err}");
    assert_eq!(records.len(), 1, "{args:?}: {records:?}");
    records.remove(0)
}

/// The error of a command that fails with exit code `exit`.
fn error(args: &[&str], conn: &[&str], exit: i32) -> Value {
    let (code, records, err) = bench(args, conn);
    assert_eq!(code, exit, "{args:?}: {err}");
    records.last().unwrap()["error"].clone()
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| panic!("not a string: {v}"))
}

/// A synthetic dataset of 20 images of circles and squares (16 train, 4 val) in `dir`, and an untrained folo-n at
/// 64 px with those classes, `<home>/models/shapes.folo` (a model named `shapes`).
fn fixture(dir: &Path, home: &Path) {
    let mut c = folo();
    c.args(["dataset", "synth"]).arg(dir).args([
        "--images",
        "20",
        "--imgsz",
        "64",
        "--classes",
        "circle,square",
    ]);
    let (code, _, err) = run(c);
    assert_eq!(code, 0, "{err}");
    let mut m = folo_vision::FoloModel::with_rng(
        folo_nn::io::ModelConfig::new("n", 2).unwrap(),
        &mut folo_core::rng::Rng::seed(1),
    )
    .unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m.set_imgsz(64);
    std::fs::create_dir_all(home.join("models")).unwrap();
    m.save(home.join("models").join("shapes.folo")).unwrap();
}

/// A file of predictions in `dir`: `records`, one per line.
fn predictions(dir: &Path, name: &str, records: &[Value]) -> String {
    let path = dir.join(name);
    let lines: Vec<String> = records.iter().map(Value::to_string).collect();
    std::fs::write(&path, lines.join("\n")).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn bench_in_this_process() {
    let t = TestDir::new("cli-bench");
    let home = t.join("home");
    fixture(&t.join("shapes"), &home);
    let d = ["--data-dir", home.to_str().unwrap()];

    let catalog = ok(&["catalog"], &d);
    assert!(catalog["conditions"].as_array().unwrap().iter().any(|c| c["id"] == "haze"), "{catalog}");
    assert_eq!(catalog["axes"].as_array().unwrap().len(), 5);
    assert_eq!(ok(&["list"], &d)["benchmarks"], json!([]));

    // A folder is registered first; the plan is checked.
    let bad = r#"{"conditions":[{"condition":"glare","severities":[1]}]}"#;
    let shapes = t.join("shapes");
    let e = error(&["create", "x", "--data", shapes.to_str().unwrap(), "--plan", bad], &d, 2);
    assert!(s(&e["code"]) == "usage" && s(&e["message"]).contains("glare"), "{e}");
    let made = ok(&["create", "shapes", "--data", shapes.to_str().unwrap(), "--plan", PLAN], &d);
    let b = s(&made["id"]).to_string();
    assert_eq!((made["images"].as_u64(), s(&made["plan"][0]["condition"])), (Some(4), "haze"));

    // No server owns the data dir: the entry runs in this process, which waits for it. The model is named.
    let (code, events, err) = bench(&["run", &b, "--model", "shapes", "--device", "cpu"], &d);
    assert_eq!(code, 0, "{err}");
    assert_eq!((s(&events[0]["event"]), events[0]["total"].as_u64()), ("started", Some(8)));
    let finished = events.last().unwrap();
    assert_eq!((s(&finished["event"]), s(&finished["status"])), ("finished", "done"), "{finished}");
    let result = &finished["result"];
    assert_eq!(result["axes"].as_array().unwrap().len(), 5, "{result}");
    assert!(result["score"].as_f64().unwrap() >= 0.0);
    assert_eq!(
        (s(&result["conditions"][0]["condition"]), result["conditions"][0]["points"][0][0].as_f64()),
        ("haze", Some(2.0))
    );
    let e1 = s(&finished["entry_id"]).to_string();
    assert_eq!(ok(&["status", &b, &e1], &d)["result"], *result);
    assert_eq!(s(&error(&["stop", &b, &e1], &d, 1)["code"]), "bench_entry_not_active");
    let e = error(&["run", &b, "--model", "circles"], &d, 3);
    assert!(s(&e["code"]) == "model_not_found" && s(&e["hint"]).contains("shapes"), "{e}");

    // Predictions from a file: every unchanged image answered (finding nothing), the condition left out.
    let empty: Vec<Value> = (1..=4).map(|i| json!({ "image": i.to_string(), "boxes": [] })).collect();
    let file = predictions(&t, "empty.jsonl", &empty);
    assert_eq!(s(&error(&["run", &b, "--predictions", &file], &d, 2)["code"]), "usage", "a name is required");
    let done = ok(&["run", &b, "--predictions", &file, "--name", "nothing"], &d);
    assert_eq!((s(&done["status"]), done["result"]["partial"].as_bool()), ("done", Some(true)), "{done}");
    // A record the server refuses makes no entry. A file it cannot finish (image 4 is missing) had a finish sent, so
    // its entry is left and named; this process ends with it, and an open entry ends `failed` with its process.
    let unknown = predictions(&t, "unknown.jsonl", &[json!({ "image": "99", "boxes": [] })]);
    let e = error(&["run", &b, "--predictions", &unknown, "--name", "bad"], &d, 2);
    assert!(s(&e["message"]).contains("'99'"), "{e}");
    let short = predictions(&t, "short.jsonl", &empty[..3]);
    let e = error(&["run", &b, "--predictions", &short, "--name", "short"], &d, 2);
    assert!(
        s(&e["message"]).contains("'4'")
            && s(&e["hint"]).contains("'short'")
            && s(&e["hint"]).contains("was removed"),
        "{e}"
    );

    let shown = ok(&["show", &b], &d);
    let board: Vec<&str> = shown["leaderboard"].as_array().unwrap().iter().map(|e| s(&e["name"])).collect();
    assert_eq!(board.len(), 2, "{shown}");
    assert!(board.contains(&"shapes") && board.contains(&"nothing"), "{board:?}");
    assert_eq!(ok(&["delete", &b], &d)["deleted"], true);
    assert_eq!(s(&error(&["show", &b], &d, 3)["code"]), "bench_not_found");
}

/// A `folo serve` child on `home` (with `extra` flags) on a free port: the child, its URL and a client.
fn serve(home: &Path, extra: &[&str]) -> (ChildGuard, String, folo_client::Client) {
    let mut serve: ChildGuard = folo()
        .args(["serve", "--port", "0", "--json", "--device", "cpu", "--data-dir"])
        .arg(home)
        .args(extra)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let (tx, rx) = mpsc::channel();
    let stdout = serve.stdout.take().unwrap();
    std::thread::spawn(move || {
        let mut first = String::new();
        let _ = BufReader::new(stdout).read_line(&mut first);
        let _ = tx.send(first);
    });
    let first = rx.recv_timeout(DEADLINE).expect("folo serve never said where it listens");
    let url = serde_json::from_str::<Value>(&first).unwrap()["url"].as_str().unwrap().to_string();
    let client = folo_client::Client::with_timeouts(&url, Duration::from_secs(5), DEADLINE);
    client.get::<Value>("/system").expect("folo serve answers /api/system");
    (serve, url, client)
}

#[test]
fn bench_through_a_folo_serve() {
    let t = TestDir::new("cli-bench-serve");
    let home = t.join("home");
    fixture(&t.join("shapes"), &home);
    let (_serve, url, client) = serve(&home, &[]);

    // Invalid create arguments fail before registering `--data`, and a source-level refusal rolls back a
    // newly registered dataset instead of leaving it behind.
    let shapes = t.join("shapes");
    let srv = ["--server", url.as_str()];
    for (label, extra, exit, code) in [
        ("max-images 0", vec!["--max-images", "0"], 2, "usage"),
        (
            "bad plan severity",
            vec!["--plan", r#"{"conditions":[{"condition":"haze","severities":[0]}]}"#],
            2,
            "usage",
        ),
        ("empty test split", vec!["--split", "test"], 1, "invalid_dataset"),
    ] {
        let mut args = vec!["create", "invalid", "--data", shapes.to_str().unwrap()];
        args.extend(extra);
        let e = error(&args, &srv, exit);
        assert_eq!(s(&e["code"]), code, "{label}: {e}");
        let listed: Value = client.get("/datasets").unwrap();
        assert_eq!(
            listed["datasets"].as_array().unwrap().len(),
            0,
            "{label} must not leave a registered dataset behind: {listed}"
        );
    }

    // The data dir's server is found through its lock.
    let d = ["--data-dir", home.to_str().unwrap()];
    let made = ok(&["create", "shapes", "--data", shapes.to_str().unwrap(), "--plan", PLAN], &d);
    let b = s(&made["id"]).to_string();

    // The entry lives in the server: `run` returns at once, `status --wait` follows it by id.
    let e = error(&["run", &b, "--model", "shapes", "--device", "cpu"], &srv, 2);
    assert!(s(&e["message"]).contains("--device"), "{e}");
    let started = ok(&["run", &b, "--model", "shapes"], &srv);
    let entry = s(&started["entry_id"]).to_string();
    let (code, events, err) = bench(&["status", &b, &entry, "--wait"], &srv);
    assert_eq!(code, 0, "{err}");
    let finished = events.last().unwrap();
    assert_eq!((s(&finished["event"]), s(&finished["status"])), ("finished", "done"), "{finished}");

    // The truth as predictions, under the plan's condition too: a perfect accuracy.
    let page: Value = client.get(&format!("/bench/benchmarks/{b}/images?limit=100")).unwrap();
    let mut records = Vec::new();
    for variant in [None, Some(("haze", 2))] {
        let q = variant.map_or(String::new(), |(c, v)| format!("&condition={c}&severity={v}"));
        let p: Value = client.get(&format!("/bench/benchmarks/{b}/images?limit=100{q}")).unwrap();
        for im in p["images"].as_array().unwrap() {
            let boxes: Vec<Value> = im["boxes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| json!({ "class": x["class"], "score": 0.9, "x1": x["x1"], "y1": x["y1"], "x2": x["x2"], "y2": x["y2"] }))
                .collect();
            let mut r = json!({ "image": im["id"], "boxes": boxes });
            if let Some((c, v)) = variant {
                r["condition"] = json!(c);
                r["severity"] = json!(v);
            }
            records.push(r);
        }
    }
    assert_eq!(records.len(), 2 * page["total"].as_u64().unwrap() as usize);
    let file = predictions(&t, "truth.jsonl", &records);
    let truth = ok(&["run", &b, "--predictions", &file, "--name", "truth"], &srv);
    assert_eq!(truth["result"]["accuracy"]["map50_95"].as_f64(), Some(1.0), "{truth}");
    let shown = ok(&["show", &b], &srv);
    assert_eq!(s(&shown["leaderboard"][0]["name"]), "truth", "{shown}");
}

/// A server without benchmarks (`--mock`) refuses `create --data` before it registers the folder.
#[test]
fn a_server_without_benchmarks_registers_nothing() {
    let t = TestDir::new("cli-bench-mock");
    let shapes = t.join("shapes");
    fixture(&shapes, &t.join("unused"));
    let (_serve, url, client) = serve(&t.join("home"), &["--mock"]);
    let before: Value = client.get("/datasets").unwrap();
    let e = error(&["create", "x", "--data", shapes.to_str().unwrap()], &["--server", url.as_str()], 4);
    assert_eq!(s(&e["code"]), "not_implemented", "{e}");
    let after: Value = client.get("/datasets").unwrap();
    assert_eq!(after, before);
}

/// Concurrent `folo bench create --data` calls where one fails on an empty split and another succeeds on a valid
/// split must never unregister the successful benchmark's source dataset.
#[test]
fn concurrent_bench_create_keeps_successful_source_registered() {
    let t = TestDir::new("cli-bench-concurrent");
    let home = t.join("home");
    let (_serve, url, client) = serve(&home, &[]);

    for round in 0..4 {
        let data = t.join(format!("data_{round}"));
        let mut synth = folo();
        synth.args(["dataset", "synth"]).arg(&data).args([
            "--images",
            "8",
            "--imgsz",
            "64",
            "--val-frac",
            "0.5",
            "--seed",
            "13",
        ]);
        let (code, _, err) = run(synth);
        assert_eq!(code, 0, "round {round} synth: {err}");

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let spawn_create = |name: String, split: &'static str, b: std::sync::Arc<std::sync::Barrier>| {
            let data = data.clone();
            let url = url.clone();
            std::thread::spawn(move || {
                b.wait();
                bench(
                    &["create", &name, "--data", data.to_str().unwrap(), "--split", split, "--plan", PLAN],
                    &["--server", url.as_str()],
                )
            })
        };

        let h_test = spawn_create(format!("Round{round} test"), "test", barrier.clone());
        let h_val = spawn_create(format!("Round{round} val"), "val", barrier);
        let (test_code, test_rec, test_err) = h_test.join().unwrap();
        let (val_code, val_rec, val_err) = h_val.join().unwrap();

        assert_eq!(test_code, 1, "round {round} test should fail: {test_rec:?} {test_err}");
        assert_eq!(s(&test_rec.last().unwrap()["error"]["code"]), "invalid_dataset", "{test_rec:?}");
        assert_eq!(val_code, 0, "round {round} val should succeed: {val_rec:?} {val_err}");

        let bench_rec = val_rec.last().unwrap();
        let ds_id = s(&bench_rec["source"]["dataset_id"]);
        let ds: Value = client
            .get(&format!("/datasets/{ds_id}"))
            .unwrap_or_else(|e| panic!("round {round}: source dataset {ds_id} disappeared: {e}"));
        assert_eq!(s(&ds["id"]), ds_id);
    }
}

/// Malformed `--server` URLs (including scheme-only `http://`) fail upfront with exit 2 (`usage`), not a
/// `server_unreachable` network error, across `bench`, `label`, and `mcp`, and never echo rejected credentials.
#[test]
fn invalid_server_urls_fail_with_usage_error() {
    for (bad, secret) in [
        ("http://", None),
        ("https://", None),
        ("http:///", None),
        ("http://:7878", None),
        ("http://127.0.0.1:0", None),
        ("http://127.0.0.1:99999", None),
        ("http://[::1", None),
        ("http://user:s3cretpass@127.0.0.1:7878", Some("s3cretpass")),
        ("http://127.0.0.1:7878?token=s3crettok", Some("s3crettok")),
    ] {
        let (b_code, b_rec, b_err) = bench(&["list"], &["--server", bad]);
        assert_eq!(b_code, 2, "bench --server {bad:?}: {b_rec:?} {b_err}");
        let e = &b_rec.last().unwrap()["error"];
        assert_eq!(s(&e["code"]), "usage", "bench --server {bad:?}: {e}");
        if let Some(sec) = secret {
            assert!(!e.to_string().contains(sec) && !b_err.contains(sec), "bench leaked secret: {e} {b_err}");
        }

        let mut lbl = folo();
        lbl.args(["label", "jobs", "--server", bad, "--json"]);
        let (code, stdout, stderr) = run(lbl);
        assert_eq!(code, 2, "label --server {bad:?}: {stdout} {stderr}");
        let lbl_err: Value = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
        assert_eq!(s(&lbl_err["error"]["code"]), "usage", "label --server {bad:?}: {lbl_err}");
        if let Some(sec) = secret {
            assert!(!stdout.contains(sec) && !stderr.contains(sec), "label leaked secret: {stdout} {stderr}");
        }

        let mut mcp = folo();
        mcp.args(["mcp", "--server", bad]);
        let (code, stdout, stderr) = run(mcp);
        assert_eq!(code, 2, "mcp --server {bad:?}: stdout={stdout} stderr={stderr}");
        assert!(stdout.is_empty(), "mcp stdout must stay empty on error: {stdout}");
        assert!(stderr.contains("--server"), "mcp stderr: {stderr}");
        if let Some(sec) = secret {
            assert!(!stderr.contains(sec), "mcp leaked secret: {stderr}");
        }
    }
}

/// `folo bench create --generated` (and `--synthetic` alias) builds a deterministic benchmark directly from
/// `folo-data`'s procedural shapes generator without registering any dataset, rejects `--split` and conflicting
/// `--images`/`--max-images`, and runs a model entry end to end with stress conditions.
#[test]
fn bench_create_generated_is_deterministic_and_runs_model() {
    let t = TestDir::new("cli-bench-generated");
    let home = t.join("home");
    let mut m = folo_vision::FoloModel::with_rng(
        folo_nn::io::ModelConfig::new("n", 2).unwrap(),
        &mut folo_core::rng::Rng::seed(2),
    )
    .unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m.set_imgsz(64);
    std::fs::create_dir_all(home.join("models")).unwrap();
    m.save(home.join("models").join("shapes.folo")).unwrap();
    let d = ["--data-dir", home.to_str().unwrap()];

    // `--split` and conflicting `--images` / `--max-images` are rejected upfront.
    let e = error(&["create", "bad-split", "--generated", "--split", "val"], &d, 2);
    assert_eq!(s(&e["code"]), "usage");
    let e = error(&["create", "bad-count", "--generated", "--images", "4", "--max-images", "6"], &d, 2);
    assert_eq!(s(&e["code"]), "usage");

    // Create two generated benchmarks with identical spec & seed (`--generated` and `--synthetic` alias).
    let b1 = ok(
        &[
            "create",
            "synth-a",
            "--generated",
            "--images",
            "4",
            "--imgsz",
            "64",
            "--classes",
            "circle,square",
            "--seed",
            "9",
            "--plan",
            PLAN,
        ],
        &d,
    );
    let b2 = ok(
        &[
            "create",
            "synth-b",
            "--synthetic",
            "--images",
            "4",
            "--imgsz",
            "64",
            "--classes",
            "circle,square",
            "--seed",
            "9",
            "--plan",
            PLAN,
        ],
        &d,
    );
    assert_eq!(b1["images"], 4);
    assert_eq!(b1["classes"], json!(["circle", "square"]));
    assert_eq!(b1["source"]["dataset_id"], "generated");
    assert_eq!(b1["fingerprint"], b2["fingerprint"], "same spec & seed must yield identical fingerprint");

    // Run a model entry against the generated benchmark (unchanged + haze severity 2 = 8 total passes).
    let (code, events, err) = bench(&["run", s(&b1["id"]), "--model", "shapes", "--device", "cpu"], &d);
    assert_eq!(code, 0, "{err}");
    assert_eq!((s(&events[0]["event"]), events[0]["total"].as_u64()), ("started", Some(8)));
    let finished = events.last().unwrap();
    assert_eq!(s(&finished["event"]), "finished");
    assert_eq!(s(&finished["status"]), "done");
    assert!(finished["result"]["score"].is_number(), "{finished}");
}
