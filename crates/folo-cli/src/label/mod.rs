//! `folo label` (docs/CLI.md, Labeling): agent labeling from the command line, on the operations the MCP
//! labeling tools use (`folo_mcp::label::Labels`), so `--json` prints what those tools return.
//!
//! The jobs are reached through the `folo serve` that owns the data dir (or `--server`), else in this process on
//! the data dir, whose lock it then holds as `folo mcp` does. A run lives in the process that started it, so
//! `start` without a server waits for the run to end. `print` holds the human output.

mod print;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_api::{LabelAgentId, LabelDatasetRequest, LabelEffort, LabelImageStatus, LabelScope, StartLabeling};
use folo_mcp::label::{BoxEdit, Labels};
use folo_mcp::proxy::HttpStore;
use folo_mcp::types::{LabelImageViewResult, Page, RelBox};
use folo_server::{RealStore, Store};
use serde_json::json;

use crate::cli::*;
use crate::output::{CliError, ExitCode, Out};
use crate::serve::{data_dir_in_use, data_dir_lock_failed};
use crate::serve_lock::{self, Acquire, ServeLock};

/// How often `start --wait` looks at the run.
const POLL: Duration = Duration::from_millis(500);

pub fn run(cmd: LabelCommand, out: Out) -> Result<(), CliError> {
    match cmd {
        LabelCommand::Agents(conn) => {
            let r = open(&conn)?.labels.agent_list().map_err(api_error)?;
            out.record(&r);
            print::agents(&r, out);
        }
        LabelCommand::Jobs(conn) => {
            let jobs = open(&conn)?.labels.job_list().map_err(api_error)?;
            out.record(&json!({ "jobs": jobs }));
            print::jobs(&jobs, out);
        }
        LabelCommand::Create(a) => {
            let s = open(&a.conn)?;
            let paths = s.paths(&a.paths)?;
            let r = s.labels.create_job(a.name, a.classes, a.instructions, &paths).map_err(api_error)?;
            out.record(&r);
            print::added(&r, out);
        }
        LabelCommand::Add(a) => {
            let s = open(&a.conn)?;
            let paths = s.paths(&a.paths)?;
            let r = s.labels.add_paths(&a.job, &paths).map_err(api_error)?;
            out.record(&r);
            print::added(&r, out);
        }
        LabelCommand::Start(a) => start(a, out)?,
        LabelCommand::Stop(a) => {
            let r = open(&a.conn)?.labels.stop_run(&a.job).map_err(api_error)?;
            out.record(&r);
            out.human(format!("{}: {}", r.job_id, print::status_name(r.status)));
        }
        LabelCommand::Status(a) => {
            if a.log > 50 {
                return Err(usage("--log must be in 0..=50"));
            }
            let r = open(&a.conn)?.labels.status(&a.job, a.log as usize).map_err(api_error)?;
            out.record(&r);
            print::status(&r, out);
        }
        LabelCommand::Images(a) => {
            if !(1..=1000).contains(&a.limit) {
                return Err(usage("--limit must be in 1..=1000"));
            }
            let status = a.status.map(image_status);
            let (items, total) = open(&a.conn)?
                .labels
                .image_page(&a.job, status, a.reviewed, a.boxes, a.offset, a.limit)
                .map_err(api_error)?;
            let end = a.offset + items.len() as u64;
            let page = Page { next_cursor: (end < total).then(|| format!("o{end}")), total, items };
            out.record(&page);
            print::images(&page, a.offset, out);
        }
        LabelCommand::View(a) => view(a, out)?,
        LabelCommand::Set(a) => {
            let s = open(&a.conn)?;
            let edit = box_edit(&s.labels, &a)?;
            let v = s.labels.set_boxes(&a.job, &a.image, &edit).map_err(api_error)?;
            out.record(&v);
            out.human(format!("image {}: {} boxes saved (by hand, reviewed)", v.id, v.boxes.len()));
        }
        LabelCommand::Review(a) => {
            let r = open(&a.conn)?.labels.review(&a.job, &a.images, !a.undo).map_err(api_error)?;
            out.record(&r);
            print::reviewed(&r, !a.undo, out);
        }
        LabelCommand::Export(a) => {
            let s = open(&a.conn)?;
            let req = LabelDatasetRequest { name: a.name, val_fraction: a.val_fraction, seed: a.seed };
            let r = s.labels.export(&a.job, req).map_err(api_error)?;
            out.record(&r);
            out.human(format!(
                "dataset {} '{}' in {}: {} train, {} val images\ntrain it: folo train --data {}",
                r.dataset_id, r.name, r.path, r.train, r.val, r.path
            ));
        }
    }
    Ok(())
}

/// The jobs of a data dir: through its server, or in this process.
struct Session {
    labels: Labels,
    /// In this process: the store (shut down when the command ends) and the data dir's lock.
    local: Option<(Arc<RealStore>, ServeLock)>,
    data_dir: PathBuf,
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some((store, _)) = &self.local {
            store.shutdown();
        }
    }
}

impl Session {
    /// Input paths as the server takes them: absolute, relative ones against the current folder (a server never
    /// reads a relative path, which would mean its own working folder).
    fn paths(&self, paths: &[String]) -> Result<Vec<String>, CliError> {
        paths
            .iter()
            .map(|p| {
                let p = Path::new(p.trim());
                let abs = std::path::absolute(p).map_err(|e| usage(format!("{}: {e}", p.display())))?;
                Ok(folo_api::json::path(abs))
            })
            .collect()
    }
}

fn open(conn: &LabelServerArgs) -> Result<Session, CliError> {
    let http = |url: &str| Labels::Http(Arc::new(HttpStore::new(url, conn.token.as_deref())));
    if let Some(url) = &conn.server {
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(usage(format!(
                "--server must be the http:// or https:// URL of a `folo serve`, got `{url}`"
            )));
        }
        let data_dir = conn.data_dir.clone().unwrap_or_else(crate::default_data_dir);
        return Ok(Session { labels: http(url), local: None, data_dir });
    }
    let data_dir = crate::data_dir(conn.data_dir.clone())?;
    match serve_lock::acquire(&data_dir) {
        Ok(Acquire::Held(lock)) => {
            let store = Arc::new(RealStore::open(data_dir.clone()));
            let labels = Labels::Store(store.clone() as Arc<dyn Store>);
            Ok(Session { labels, local: Some((store, lock)), data_dir })
        }
        Ok(Acquire::InUse { url: Some(url) })
            if serve_lock::serves(&url, &data_dir, conn.token.as_deref()) =>
        {
            Ok(Session { labels: http(&url), local: None, data_dir })
        }
        Ok(Acquire::InUse { url }) => Err(data_dir_in_use(&data_dir, url.as_deref())),
        Err(e) => Err(data_dir_lock_failed(&data_dir, e)),
    }
}

fn start(a: LabelStartArgs, out: Out) -> Result<(), CliError> {
    let s = open(&a.conn)?;
    let req = StartLabeling {
        agent: match a.agent {
            AgentArg::Claude => LabelAgentId::Claude,
            AgentArg::Codex => LabelAgentId::Codex,
        },
        model: a.model,
        effort: a.effort.map(|e| match e {
            EffortArg::Low => LabelEffort::Low,
            EffortArg::Medium => LabelEffort::Medium,
            EffortArg::High => LabelEffort::High,
        }),
        scope: if a.all { LabelScope::All } else { LabelScope::Unlabeled },
        image_ids: (!a.images.is_empty()).then_some(a.images),
        note: a.note,
        batch: a.batch,
        parallel: a.parallel,
    };
    let started = s.labels.start_run(&a.job, req).map_err(api_error)?;
    let in_process = s.local.is_some();
    let wait = a.wait || in_process;
    if !wait {
        out.record(&started);
        print::started(&started, out);
        return Ok(());
    }
    out.record(&json!({ "event": "started", "job_id": started.job_id, "run": started.run }));
    print::started(&started, out);
    if in_process {
        out.human(format!(
            "no folo serve owns {}: the run works in this process until it ends (Ctrl+C stops it)",
            folo_api::json::path(&s.data_dir)
        ));
    }
    let stop = Arc::new(AtomicBool::new(false));
    if in_process {
        crate::train::watch_interrupt(stop.clone(), "stopping the labeling run (interrupt again to abort)…");
    }
    let mut asked_stop = false;
    let mut last = None;
    loop {
        std::thread::sleep(POLL);
        if stop.load(Ordering::Relaxed) && !asked_stop {
            asked_stop = true;
            // Already over is fine: the loop sees it next.
            let _ = s.labels.stop(&a.job);
        }
        let job = s.labels.job(&a.job).map_err(api_error)?;
        let progress = job.run.as_ref().map(|r| (r.done, r.failed, r.total));
        if progress != last
            && let Some((done, failed, total)) = progress
        {
            out.record(&json!({ "event": "progress", "done": done, "failed": failed, "total": total }));
            out.human(format!("{done}/{total} labeled, {failed} failed"));
            last = progress;
        }
        if !job.status.is_active() {
            break;
        }
    }
    let status = s.labels.status(&a.job, 5).map_err(api_error)?;
    let mut record = folo_api::json::to_value(&status).unwrap_or_default();
    record["event"] = json!("finished");
    out.record(&record);
    print::status(&status, out);
    match status.run.as_ref().and_then(|r| r.outcome) {
        Some(folo_api::LabelOutcome::Failed) => {
            let run = status.run.as_ref().expect("an outcome has a run");
            let err = CliError::new(
                ExitCode::Runtime,
                "label_run_failed",
                run.error.clone().unwrap_or_else(|| "the labeling run failed".into()),
            );
            Err(match &run.hint {
                Some(h) => err.hint(h.clone()),
                None => err.hint("see `folo label status` for its log"),
            })
        }
        _ => Ok(()),
    }
}

fn view(a: LabelViewArgs, out: Out) -> Result<(), CliError> {
    if !(256..=1568).contains(&a.size) {
        return Err(usage("--size must be in 256..=1568"));
    }
    let s = open(&a.conn)?;
    let (r, jpeg): (LabelImageViewResult, _) =
        s.labels.view(&a.job, a.image.as_deref(), a.size).map_err(api_error)?;
    let (Some(image), Some(jpeg)) = (&r.image, jpeg) else {
        out.record(
            &json!({ "job_id": r.job_id, "classes": r.classes, "to_review": r.to_review, "image": null }),
        );
        out.human(format!("{}: nothing left to review", r.job_id));
        return Ok(());
    };
    let path =
        a.output.unwrap_or_else(|| PathBuf::from(format!("{}-{}.jpg", safe(&r.job_id), safe(&image.id))));
    std::fs::write(&path, &jpeg).map_err(|e| {
        CliError::new(ExitCode::Runtime, "io_error", format!("writing {}: {e}", folo_api::json::path(&path)))
    })?;
    let mut record = folo_api::json::to_value(&r).unwrap_or_default();
    record["path"] = json!(folo_api::json::path(&path));
    out.record(&record);
    out.human(folo_mcp::label::view_text(&r.job_id, image, &r.classes, r.to_review));
    out.human(format!("wrote {}", folo_api::json::path(&path)));
    Ok(())
}

/// `s` with only `A-Z a-z 0-9 - _` (a file name made from ids).
fn safe(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// `folo label set`'s flags as a [`BoxEdit`]; classes may be given by name.
fn box_edit(labels: &Labels, a: &LabelSetArgs) -> Result<BoxEdit, CliError> {
    let names = if a.boxes.iter().chain(&a.add).chain(&a.relabel).next().is_some() {
        labels.job(&a.job).map_err(api_error)?.classes
    } else {
        Vec::new()
    };
    let class = |s: &str| -> Result<u32, CliError> {
        let s = s.trim();
        if let Ok(i) = s.parse::<u32>() {
            return Ok(i);
        }
        names.iter().position(|n| n.eq_ignore_ascii_case(s)).map(|i| i as u32).ok_or_else(|| {
            usage(format!("`{s}` is not a class of the job (its classes: {})", names.join(", ")))
        })
    };
    let rel = |s: &String| -> Result<RelBox, CliError> {
        let parts: Vec<&str> = s.split(',').collect();
        let [c, x1, y1, x2, y2] = parts[..] else {
            return Err(usage(format!("`{s}` is not CLASS,X1,Y1,X2,Y2")));
        };
        let n = |v: &str| {
            v.trim().parse::<u32>().map_err(|_| usage(format!("`{s}`: `{v}` is not a whole number")))
        };
        Ok([class(c)?, n(x1)?, n(y1)?, n(x2)?, n(y2)?])
    };
    let boxes = if a.clear {
        Some(Vec::new())
    } else if a.boxes.is_empty() {
        None
    } else {
        Some(a.boxes.iter().map(rel).collect::<Result<_, _>>()?)
    };
    let relabel = a
        .relabel
        .iter()
        .map(|r| {
            let (n, c) =
                r.split_once('=').ok_or_else(|| usage(format!("--relabel takes N=CLASS, got `{r}`")))?;
            let n = n
                .trim()
                .parse::<u32>()
                .map_err(|_| usage(format!("--relabel {r}: `{n}` is not a number")))?;
            Ok([n, class(c)?])
        })
        .collect::<Result<_, CliError>>()?;
    let add = a.add.iter().map(rel).collect::<Result<_, _>>()?;
    let mut seen = HashSet::new();
    let remove = a.remove.iter().copied().filter(|n| seen.insert(*n)).collect();
    Ok(BoxEdit { boxes, remove, relabel, add })
}

fn image_status(s: ImageStatusArg) -> LabelImageStatus {
    match s {
        ImageStatusArg::Pending => LabelImageStatus::Pending,
        ImageStatusArg::Labeling => LabelImageStatus::Labeling,
        ImageStatusArg::Labeled => LabelImageStatus::Labeled,
        ImageStatusArg::Failed => LabelImageStatus::Failed,
    }
}

/// A store or server error as the CLI reports it: the same code, its exit code.
fn api_error(e: folo_server::error::Error) -> CliError {
    let err = CliError::new(ExitCode::of(&e.body.code), &e.body.code, e.body.message);
    match e.body.hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

fn usage(message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "usage", message).hint("see `folo label --help`")
}
