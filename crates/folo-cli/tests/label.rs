//! `folo label` with `--json`: a job labeled end to end in this process (a fake `claude` that always answers one
//! car), its review and export, the error codes, and the same commands through a `folo serve` child, found by its
//! data dir and named with `--server`. No real agent CLI is called.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use folo_test_support::{ChildGuard, TestDir};
use serde_json::Value;

/// What the fake `claude` prints for every request: one `result` event whose reply puts a car on image 1, so
/// runs use `--batch 1`.
const REPLY: &str = r#"{"type":"result","subtype":"success","is_error":false,"result":"{\"images\":[{\"image\":1,\"boxes\":[{\"label\":\"car\",\"box\":[250,100,750,600]}]}]}","usage":{"input_tokens":600,"output_tokens":40},"total_cost_usd":0.001}"#;

/// Writes the fake `claude` into `dir`: `--version` prints a version, anything else [`REPLY`].
fn fake_claude(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    if cfg!(windows) {
        let p = dir.join("claude.cmd");
        let script = format!(
            "@echo off\r\nif \"%~1\"==\"--version\" (\r\n  echo 9.9.9 Claude Code\r\n  exit /b 0\r\n)\r\necho {REPLY}\r\n"
        );
        std::fs::write(&p, script).unwrap();
        p
    } else {
        let p = dir.join("claude");
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo '9.9.9 (Claude Code)'; exit 0; fi\nprintf '%s\\n' '{REPLY}'\n"
        );
        std::fs::write(&p, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        p
    }
}

/// Five 64×48 PNGs in `dir`.
fn photos(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    for i in 0..5u8 {
        let img = image::RgbImage::from_fn(64, 48, |x, y| image::Rgb([x as u8 * 4, y as u8 * 5, i * 40]));
        img.save(dir.join(format!("p{i}.png"))).unwrap();
    }
    dir.to_path_buf()
}

struct Env {
    claude: PathBuf,
    no_codex: PathBuf,
    /// Arguments every command gets.
    after: Vec<String>,
}

impl Env {
    fn new(root: &Path) -> Env {
        Env {
            claude: fake_claude(&root.join("bin")),
            no_codex: root.join("bin").join("no-codex"),
            after: vec![],
        }
    }

    /// This environment with `args` after every command's own.
    fn after(self, args: &[&str]) -> Env {
        Env { after: args.iter().map(|a| a.to_string()).collect(), ..self }
    }

    fn cmd(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
        c.env("FOLO_NO_CACHE", "1")
            .env("FOLO_CLAUDE_BIN", &self.claude)
            .env("FOLO_CODEX_BIN", &self.no_codex)
            .env("FOLO_GPU_ADAPTER", "folo-cli-tests-no-such-adapter")
            .env_remove("FOLO_DATA_DIR")
            .env_remove("FOLO_TOKEN");
        c
    }

    /// `folo label <args> --json`: exit code, stdout records, stderr.
    fn label(&self, args: &[&str]) -> (i32, Vec<Value>, String) {
        let out = self
            .cmd()
            .arg("label")
            .args(args)
            .args(&self.after)
            .arg("--json")
            .stdin(Stdio::null())
            .output()
            .unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let records =
            stdout.lines().map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}"))).collect();
        (out.status.code().unwrap(), records, String::from_utf8(out.stderr).unwrap())
    }

    /// The one record of a command that succeeds.
    fn ok(&self, args: &[&str]) -> Value {
        let (code, mut records, err) = self.label(args);
        assert_eq!(code, 0, "{args:?}: {err}");
        assert_eq!(records.len(), 1, "{args:?}: {records:?}");
        records.remove(0)
    }

    /// The error code of a command that fails with exit code `exit`.
    fn fails(&self, args: &[&str], exit: i32) -> String {
        let (code, records, err) = self.label(args);
        assert_eq!(code, exit, "{args:?}: {err}");
        records.last().unwrap()["error"]["code"].as_str().unwrap().to_string()
    }
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap()
}

#[test]
fn label_review_and_export_in_this_process() {
    let t = TestDir::new("cli-label");
    let env = Env::new(&t);
    let home = t.join("home");
    let dir = photos(&t.join("photos"));
    let env = env.after(&["--data-dir", home.to_str().unwrap()]);

    let agents = env.ok(&["agents"]);
    assert_eq!(
        (s(&agents["agents"][0]["id"]), &agents["agents"][0]["available"]),
        ("claude", &Value::Bool(true))
    );
    assert_eq!(s(&agents["agents"][0]["version"]), "9.9.9");
    assert_eq!(agents["agents"][1]["available"], false);
    assert!(s(&agents["agents"][1]["hint"]).contains("FOLO_CODEX_BIN"));

    let made = env.ok(&["create", "cars", "--classes", "car,person", "--path", dir.to_str().unwrap()]);
    let job = s(&made["job_id"]).to_string();
    assert_eq!((made["added"].as_u64(), made["images"].as_u64()), (Some(5), Some(5)));

    // No server owns the data dir: the run works in this process, which waits for it.
    let (code, events, err) = env.label(&["start", &job, "--agent", "claude", "--batch", "1"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(s(&events[0]["event"]), "started");
    assert_eq!(events[0]["run"]["total"], 5);
    let finished = events.last().unwrap();
    assert_eq!((s(&finished["event"]), finished["counts"]["labeled"].as_u64()), ("finished", Some(5)));
    assert_eq!((s(&finished["run"]["outcome"]), finished["usage"]["calls"].as_u64()), ("completed", Some(5)));

    let page = env.ok(&["images", &job, "--boxes", "--limit", "2"]);
    assert_eq!((page["total"].as_u64(), page["items"].as_array().unwrap().len()), (Some(5), 2));
    assert_eq!(s(&page["next_cursor"]), "o2");
    let first = &page["items"][0];
    assert_eq!(first["boxes"], serde_json::json!([[0, 250, 100, 750, 600]]));
    let (a, b) = (s(&first["id"]).to_string(), s(&page["items"][1]["id"]).to_string());

    // The next image to review, drawn to a file.
    let out = t.join("view.jpg");
    let view = env.ok(&["view", &job, "-o", out.to_str().unwrap(), "--size", "256"]);
    assert_eq!((s(&view["image"]["id"]), view["to_review"].as_u64()), (a.as_str(), Some(5)));
    assert_eq!(Path::new(s(&view["path"])), out.as_path());
    let jpeg = image::open(&out).unwrap();
    assert_eq!((jpeg.width(), jpeg.height()), (64, 48), "never upscaled");

    let set = env.ok(&["set", &job, &a, "--relabel", "1=person", "--add", "car,0,0,500,500"]);
    assert_eq!(set["boxes"], serde_json::json!([[1, 250, 100, 750, 600], [0, 0, 0, 500, 500]]));
    assert_eq!((set["reviewed"].as_bool(), s(&set["source"])), (Some(true), "manual"));
    assert_eq!(env.fails(&["set", &job, &a, "--add", "car,1,2"], 2), "usage");
    assert_eq!(env.fails(&["set", &job, &a, "--add", "truck,0,0,9,9"], 2), "usage");

    let reviewed = env.ok(&["review", &job, &b, "nope"]);
    assert_eq!((reviewed["updated"].as_u64(), reviewed["to_review"].as_u64()), (Some(1), Some(3)));
    assert_eq!(s(&reviewed["failed"][0]["id"]), "nope");
    let left = env.ok(&["images", &job, "--status", "labeled", "--reviewed", "false"]);
    assert_eq!(left["total"], 3);
    let status = env.ok(&["status", &job, "--log", "0"]);
    assert_eq!((status["reviewed"].as_u64(), status["log"].as_array().unwrap().len()), (Some(2), 0));

    let ds = env.ok(&["export", &job, "--name", "cars-v1", "--val-fraction", "0.2"]);
    assert_eq!((s(&ds["name"]), ds["train"].as_u64(), ds["val"].as_u64()), ("cars-v1", Some(4), Some(1)));
    assert!(Path::new(s(&ds["path"])).join("data.yaml").is_file());
    let jobs = env.ok(&["jobs"]);
    assert_eq!(jobs["jobs"][0]["dataset_id"], ds["dataset_id"]);

    // The codes of `folo_api::codes`, with their exit codes.
    assert_eq!(env.fails(&["status", "lb_nope"], 3), "label_job_not_found");
    assert_eq!(env.fails(&["start", &job, "--agent", "codex", "--all"], 4), "not_supported");
    assert_eq!(env.fails(&["view", &job, "nope"], 3), "image_not_found");
    assert_eq!(env.fails(&["images", &job, "--limit", "0"], 2), "usage");
}

#[test]
fn label_through_a_folo_serve() {
    let t = TestDir::new("cli-label-serve");
    let env = Env::new(&t);
    let home = t.join("home");
    let dir = photos(&t.join("photos"));
    let mut serve: ChildGuard = env
        .cmd()
        .args(["serve", "--port", "0", "--json", "--data-dir"])
        .arg(&home)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
        .into();
    let mut first = String::new();
    BufReader::new(serve.stdout.take().unwrap()).read_line(&mut first).unwrap();
    let url = serde_json::from_str::<Value>(&first).unwrap()["url"].as_str().unwrap().to_string();

    // The data dir's server is found through its lock; the paths are made absolute for it.
    let d = ["--data-dir", home.to_str().unwrap()];
    let made =
        env.ok(&[&["create", "cars", "--classes", "car", "--path", dir.to_str().unwrap()], &d[..]].concat());
    let job = s(&made["job_id"]).to_string();

    // `--server` without a data dir; the run lives in the server, and `--wait` follows it.
    let srv = ["--server", url.as_str()];
    let (code, events, err) =
        env.label(&[&["start", &job, "--agent", "claude", "--batch", "1", "--wait"], &srv[..]].concat());
    assert_eq!(code, 0, "{err}");
    let finished = events.last().unwrap();
    assert_eq!((s(&finished["event"]), finished["counts"]["labeled"].as_u64()), ("finished", Some(5)));

    let page = env.ok(&[&["images", &job], &srv[..]].concat());
    let a = s(&page["items"][0]["id"]).to_string();
    let out = t.join("view.jpg");
    let view = env.ok(&[&["view", &job, &a, "-o", out.to_str().unwrap()], &srv[..]].concat());
    assert_eq!(s(&view["image"]["id"]), a);
    assert!(out.is_file());
    let set = env.ok(&[&["set", &job, &a, "--clear"], &srv[..]].concat());
    assert_eq!((set["boxes"].as_array().unwrap().len(), set["reviewed"].as_bool()), (0, Some(true)));
    let reviewed = env.ok(&[&["review", &job, &a, "--undo"], &d[..]].concat());
    assert_eq!(reviewed["updated"], 1);
    assert_eq!(env.fails(&[&["status", "lb_nope"], &srv[..]].concat(), 3), "label_job_not_found");

    // Without `--wait` the command returns at once; the server's run ends on its own.
    let again =
        [&["start", &job, "--agent", "claude", "--images", &a, "--note", "look again"], &srv[..]].concat();
    let started = env.ok(&again);
    assert_eq!((s(&started["job_id"]), started["run"]["total"].as_u64()), (job.as_str(), Some(1)));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let status = env.ok(&[&["status", &job], &srv[..]].concat());
        if s(&status["status"]) == "idle" {
            assert_eq!(s(&status["run"]["outcome"]), "completed");
            break;
        }
        assert!(std::time::Instant::now() < deadline, "{status}");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
