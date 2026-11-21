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
            && s(&e["hint"]).contains("was left"),
        "{e}"
    );

    let shown = ok(&["show", &b], &d);
    let board: Vec<&str> = shown["leaderboard"].as_array().unwrap().iter().map(|e| s(&e["name"])).collect();
    assert_eq!(board.len(), 3, "{shown}");
    assert!(board.contains(&"shapes") && board.contains(&"nothing") && board.contains(&"short"), "{board:?}");
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

    // The data dir's server is found through its lock.
    let d = ["--data-dir", home.to_str().unwrap()];
    let shapes = t.join("shapes");
    let made = ok(&["create", "shapes", "--data", shapes.to_str().unwrap(), "--plan", PLAN], &d);
    let b = s(&made["id"]).to_string();

    // The entry lives in the server: `run` returns at once, `status --wait` follows it by id.
    let srv = ["--server", url.as_str()];
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
