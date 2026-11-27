//! `folo bench` (docs/CLI.md, Vision benchmarks): benchmarks from the command line, on the operations the MCP
//! benchmark tools use (`folo_mcp::bench::Benchmarks`), so `--json` prints what those tools return.
//!
//! The benchmarks are reached through the `folo serve` that owns the data dir (or `--server`), else in this process
//! on the data dir, whose lock it then holds as `folo mcp` does. A model's entry runs in the process that started it,
//! so `run --model` without a server waits for the entry to end. `print` holds the human output.

mod print;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_api::{
    BenchEntry, BenchEntryStatus, BenchPlan, BenchSource, CreateBench, DatasetFormat, RegisterDataset,
};
use folo_mcp::bench::hints::{self, Front};
use folo_mcp::bench::{Benchmarks, Contestant};
use folo_mcp::proxy::HttpStore;
use folo_mcp::types::BenchEntryState;
use folo_server::{RealStore, Store};
use serde_json::json;

use crate::cli::*;
use crate::output::{CliError, ExitCode, Out};
use crate::serve::{data_dir_in_use, data_dir_lock_failed};
use crate::serve_lock::{self, Acquire, ServeLock};

/// How often a wait reads the entry: in this process, and through a server.
const POLL_HERE: Duration = Duration::from_millis(100);
const POLL_SERVER: Duration = Duration::from_millis(500);

pub fn run(cmd: BenchCommand, out: Out) -> Result<(), CliError> {
    match cmd {
        BenchCommand::Catalog(conn) => {
            let r = open(&conn, None)?.benchmarks.catalog_result().map_err(api_error)?;
            out.record(&r);
            print::catalog(&r, out);
        }
        BenchCommand::List(conn) => {
            let list = open(&conn, None)?.benchmarks.summaries().map_err(api_error)?;
            out.record(&json!({ "benchmarks": list }));
            print::list(&list, out);
        }
        BenchCommand::Create(a) => create(a, out)?,
        BenchCommand::Show(a) => {
            let r = open(&a.conn, None)?.benchmarks.detail(&a.bench).map_err(api_error)?;
            out.record(&r);
            print::detail(&r, out);
        }
        BenchCommand::Delete(a) => {
            open(&a.conn, None)?.benchmarks.delete(&a.bench).map_err(api_error)?;
            out.record(&json!({ "benchmark_id": a.bench, "deleted": true }));
            out.human(format!("deleted benchmark {}", a.bench));
        }
        BenchCommand::Run(a) => run_entry(a, out)?,
        BenchCommand::Status(a) => {
            let s = open(&a.conn, None)?;
            if !a.wait {
                let r = s.benchmarks.state(&a.bench, &a.entry).map_err(api_error)?;
                out.record(&r);
                print::state(&r, out);
                return Ok(());
            }
            let first = s.benchmarks.entry(&a.bench, &a.entry).map_err(api_error)?;
            follow(&s, first, None, out)?;
        }
        BenchCommand::Stop(a) => {
            let r = open(&a.conn, None)?.benchmarks.stop(&a.bench, &a.entry).map_err(api_error)?;
            out.record(&r);
            print::state(&r, out);
        }
    }
    Ok(())
}

/// The benchmarks of a data dir: through its server, or in this process.
struct Session {
    benchmarks: Benchmarks,
    /// The same store (for registering a dataset folder).
    store: Arc<dyn Store>,
    /// In this process: the data dir's lock, held until the command ends.
    local: Option<ServeLock>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // Ends an entry of this process (it fails, "the server stopped") before the lock goes.
        if self.local.is_some() {
            self.store.shutdown();
        }
    }
}

/// The benchmarks of `conn`. `device`: in this process, where model entries run (`run --device`).
fn open(conn: &BenchServerArgs, device: Option<&str>) -> Result<Session, CliError> {
    let tls_ca = serve_lock::load_tls_ca(conn.tls_ca.as_deref())?;
    let http = |url: &str| {
        let h = Arc::new(HttpStore::with_tls_ca(url, conn.token.as_deref(), tls_ca.as_ref()));
        Session { benchmarks: Benchmarks::Http(h.clone()), store: h, local: None }
    };
    if let Some(url) = &conn.server {
        if let Err(why) = folo_client::check_base_url(url) {
            return Err(usage(format!(
                "--server (or FOLO_SERVER) must be a valid http:// or https:// URL of a `folo serve` ({why})"
            )));
        }
        return Ok(http(url));
    }
    let data_dir = crate::data_dir(conn.data_dir.clone())?;
    match serve_lock::acquire(&data_dir) {
        Ok(Acquire::Held(lock)) => {
            let device = device
                .map(str::to_string)
                .or_else(|| std::env::var("FOLO_DEVICE").ok())
                .unwrap_or("auto".into());
            let store =
                RealStore::open(data_dir).device(device).published_weights(folo_api::weights::cache_dir());
            let store: Arc<dyn Store> = Arc::new(store);
            Ok(Session { benchmarks: Benchmarks::Store(store.clone()), store, local: Some(lock) })
        }
        Ok(Acquire::InUse { url: Some(url) })
            if serve_lock::serves(&url, &data_dir, conn.token.as_deref(), tls_ca.as_ref()) =>
        {
            Ok(http(&url))
        }
        Ok(Acquire::InUse { url }) => Err(data_dir_in_use(&data_dir, url.as_deref())),
        Err(e) => Err(data_dir_lock_failed(&data_dir, e)),
    }
}

fn create(a: BenchCreateArgs, out: Out) -> Result<(), CliError> {
    if a.split.is_some() && a.label_job.is_some() {
        return Err(usage("--split is for a dataset (--dataset or --data), not for --label-job"));
    }
    if a.split.is_some() && a.generated {
        return Err(usage("--split is for a dataset (--dataset or --data), not for --generated"));
    }
    let plan = a.plan.as_deref().map(plan).transpose()?;
    if a.generated {
        let images = match (a.images, a.max_images) {
            (Some(i), Some(m)) if i != m => {
                return Err(usage(format!(
                    "--images ({i}) and --max-images ({m}) disagree: pass one or the same count"
                )));
            }
            (Some(n), _) | (_, Some(n)) => n,
            (None, None) => folo_server::bench::GENERATED_DEFAULT_IMAGES,
        };
        let spec = folo_server::bench::GeneratedSourceSpec {
            images,
            classes: a.classes.unwrap_or_else(|| {
                folo_server::bench::GENERATED_DEFAULT_CLASSES.iter().map(|s| (*s).to_string()).collect()
            }),
            imgsz: a.imgsz.unwrap_or(folo_server::bench::GENERATED_DEFAULT_IMGSZ),
            seed: a.seed,
            vary_aspect: a.vary_aspect,
        };
        spec.to_synthetic_config().map_err(usage)?;
        folo_server::bench::check_create(&a.name, &a.description, Some(images), a.seed, plan.as_ref())
            .map_err(api_error)?;
        let s = open(&a.conn, None)?;
        let req = folo_server::bench::CreateGeneratedBench {
            name: a.name,
            description: a.description,
            source: spec,
            plan,
        };
        let r = s.benchmarks.create_generated_bench(req).map_err(api_error)?;
        out.record(&r);
        print::detail(&r, out);
        return Ok(());
    }
    // Validate benchmark arguments before opening the store or registering `--data`, so invalid flags or plans
    // never leave a newly registered dataset behind.
    folo_server::bench::check_create(&a.name, &a.description, a.max_images, a.seed, plan.as_ref())
        .map_err(api_error)?;
    let s = open(&a.conn, None)?;
    let split = a.split.map(|s| match s {
        SplitArg::Train => folo_api::Split::Train,
        SplitArg::Val => folo_api::Split::Val,
        SplitArg::Test => folo_api::Split::Test,
    });
    let mut registered_ds = None;
    let source = match (a.dataset, a.data, a.label_job) {
        (Some(dataset_id), _, _) => BenchSource::Dataset { dataset_id, split },
        (_, Some(path), _) => {
            let path = crate::label::paths(&[path], a.conn.server.is_some())?.remove(0);
            // Not a dataset registered for a benchmark that cannot be made.
            s.benchmarks.check().map_err(api_error)?;
            let req = RegisterDataset { name: None, path, format: DatasetFormat::Auto };
            let (ds, rollback) = s.store.register_dataset_tracked(req).map_err(api_error)?;
            let id = ds.id.clone();
            registered_ds = Some((ds, rollback));
            BenchSource::Dataset { dataset_id: id, split }
        }
        (_, _, Some(job_id)) => BenchSource::LabelJob { job_id },
        _ => unreachable!("clap requires a source"),
    };
    let req = CreateBench {
        name: a.name,
        description: a.description,
        source,
        max_images: a.max_images,
        seed: a.seed,
        plan,
    };
    let r = match s.benchmarks.create_bench(req) {
        Ok(r) => r,
        Err(e) => {
            // If `--data` just registered a dataset and the server definitively refused to create the benchmark
            // (e.g. an empty requested split), unregister it only if this exact registration is still exclusive
            // and unreferenced (never deleting files).
            if e.status.is_client_error()
                && let Some((ds, Some(token))) = &registered_ds
            {
                let _ = s.store.delete_dataset_if_unused(&ds.id, token);
            }
            return Err(api_error(e));
        }
    };
    if let Some((ds, _)) = &registered_ds {
        out.human(format!("dataset {} '{}' ({})", ds.id, ds.name, ds.path));
    }
    out.record(&r);
    print::detail(&r, out);
    Ok(())
}

/// Strict CLI input for `--plan`, rejecting misspelled keys (e.g. `wieght`) instead of silently falling back to
/// defaults.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchPlanInput {
    conditions: Vec<BenchPlanConditionInput>,
    #[serde(default)]
    rationale: Option<String>,
    #[serde(default)]
    designed_by: Option<BenchDesignerInput>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchPlanConditionInput {
    condition: String,
    severities: Vec<u8>,
    #[serde(default = "default_weight")]
    weight: f32,
    #[serde(default)]
    why: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchDesignerInput {
    agent: folo_api::LabelAgentId,
    #[serde(default)]
    model: Option<String>,
}

fn default_weight() -> f32 {
    1.0
}

/// `--plan`: JSON when it starts with `{`, else a file of it.
fn plan(arg: &str) -> Result<BenchPlan, CliError> {
    let text = if arg.trim_start().starts_with('{') {
        arg.to_string()
    } else {
        std::fs::read_to_string(arg).map_err(|e| usage(format!("--plan {arg}: {e}")))?
    };
    let parsed: BenchPlanInput =
        serde_json::from_str(&text).map_err(|e| usage(format!("--plan is not a plan: {e}")))?;
    Ok(BenchPlan {
        conditions: parsed
            .conditions
            .into_iter()
            .map(|c| folo_api::BenchPlanCondition {
                condition: c.condition,
                severities: c.severities,
                weight: c.weight,
                why: c.why,
            })
            .collect(),
        rationale: parsed.rationale,
        designed_by: parsed.designed_by.map(|d| folo_api::BenchDesigner { agent: d.agent, model: d.model }),
    })
}

fn run_entry(a: BenchRunArgs, out: Out) -> Result<(), CliError> {
    let remote = a.conn.server.is_some();
    let who = match (a.model, a.predictions) {
        (Some(model_id), _) => {
            if a.unscored {
                return Err(usage("--unscored is for --predictions: a model gives its confidences"));
            }
            if let Some(n) = a.max_images
                && !crate::cli::bounds::POSITIVE_INT.contains_u64(u64::from(n))
            {
                return Err(usage("--max-images must be at least 1"));
            }
            Contestant::Model {
                model_id,
                max_images: a.max_images,
                conditions: a.no_conditions.then_some(false),
            }
        }
        (_, Some(path)) => {
            if a.max_images.is_some() || a.no_conditions || a.device.is_some() {
                return Err(usage(
                    "--max-images, --no-conditions and --device are for --model: predictions say what they ran",
                ));
            }
            Contestant::Predictions { path, scored: !a.unscored }
        }
        _ => unreachable!("clap requires a contestant"),
    };
    let model = matches!(who, Contestant::Model { .. });
    let s = open(&a.conn, a.device.as_deref())?;
    if a.device.is_some() && s.local.is_none() {
        let why = if remote { "--server / FOLO_SERVER" } else { "the `folo serve` that owns the data dir" };
        return Err(usage(format!(
            "--device is for an entry run in this process; {why} runs it on its own device"
        )));
    }
    let in_process = s.local.is_some();
    // Armed before the entry starts, so an early Ctrl+C stops it too.
    let stop = Arc::new(AtomicBool::new(false));
    if model && in_process {
        crate::train::watch_interrupt(
            stop.clone(),
            "stopping the benchmark entry (interrupt again to abort)…",
        );
    } else if !model {
        // Here or through a server: an entry left half-filled would block deleting the benchmark.
        crate::train::watch_interrupt(stop.clone(), "stopping the upload (interrupt again to abort)…");
    }
    let who = match who {
        Contestant::Model { model_id, max_images, conditions } => {
            Contestant::Model { model_id: model_id_of(&*s.store, &model_id)?, max_images, conditions }
        }
        other => other,
    };
    let state = s.benchmarks.start_unless(&a.bench, who, a.name, &stop).map_err(api_error)?;
    if !model || !(a.wait || in_process) {
        out.record(&state);
        print::state(&state, out);
        // A file of predictions is scored at once, and scoring can fail.
        return failure(state.status, state.error.clone(), state.hint.clone());
    }
    let mut record = folo_api::json::to_value(&state).unwrap_or_default();
    record["event"] = json!("started");
    out.record(&record);
    print::state(&state, out);
    if in_process {
        out.human(
            "no folo serve owns the data dir: the entry runs in this process until it ends (Ctrl+C stops it)",
        );
    }
    let first = s.benchmarks.entry(&state.benchmark_id, &state.entry_id).map_err(api_error)?;
    follow(&s, first, Some(&stop), out)
}

/// The id of the model `arg` names: its id, else its name when one model has it (the CLI lists no models, so a
/// person knows them by name). Not found: `model_not_found` with the names there are.
fn model_id_of(store: &dyn Store, arg: &str) -> Result<String, CliError> {
    let models = store.list_models().map_err(api_error)?;
    if models.iter().any(|m| m.id == arg) {
        return Ok(arg.to_string());
    }
    let named: Vec<&folo_api::Model> = models.iter().filter(|m| m.name == arg).collect();
    match named[..] {
        [m] => Ok(m.id.clone()),
        [] => {
            let names: Vec<&str> = models.iter().take(20).map(|m| m.name.as_str()).collect();
            let hint = match names.len() {
                0 => {
                    "there are no models: train one, or `folo weights pull NAME` for published weights".into()
                }
                n if n < models.len() => format!("models: {}, …", names.join(", ")),
                _ => format!("models: {}", names.join(", ")),
            };
            Err(CliError::new(
                ExitCode::NotFound,
                "model_not_found",
                format!("no model has the id or name '{arg}'"),
            )
            .hint(hint))
        }
        _ => {
            let ids: Vec<&str> = named.iter().map(|m| m.id.as_str()).collect();
            Err(usage(format!("{} models are named '{arg}': pass one's id ({})", ids.len(), ids.join(", "))))
        }
    }
}

/// Follows entry `first` to its end, printing `progress` records when it moves and a `finished` record with its
/// state; an entry that ends `failed` is the error `bench_entry_failed`. `stop` (Ctrl+C) stops the entry.
fn follow(s: &Session, first: BenchEntry, stop: Option<&AtomicBool>, out: Out) -> Result<(), CliError> {
    let poll = if s.local.is_some() { POLL_HERE } else { POLL_SERVER };
    let (id, entry_id) = (first.bench_id.clone(), first.id.clone());
    let mut last = None;
    let mut asked_stop = false;
    let mut progress = |e: &BenchEntry| {
        let p = (e.progress.done, e.progress.total, e.progress.condition.clone(), e.progress.severity);
        if last.as_ref() != Some(&p) {
            out.record(&json!({
                "event": "progress", "done": p.0, "total": p.1, "condition": p.2, "severity": p.3
            }));
            out.human(print::progress(e));
            last = Some(p);
        }
        if stop.is_some_and(|f| f.load(Ordering::Relaxed)) && !asked_stop {
            asked_stop = true;
            // Already over is fine: the next read sees it.
            let _ = s.benchmarks.stop_entry(&id, &entry_id);
        }
        true
    };
    let end = if first.status.is_active() {
        progress(&first);
        s.benchmarks.follow(&first.bench_id, &first.id, poll, progress).map_err(api_error)?
    } else {
        first
    };
    // The record and the exit code both come from the read that showed the entry over.
    let (status, error, hint) = (end.status, end.error.clone(), end.hint.clone());
    let state: BenchEntryState = s.benchmarks.state_of(end).map_err(api_error)?;
    let mut record = folo_api::json::to_value(&state).unwrap_or_default();
    record["event"] = json!("finished");
    out.record(&record);
    print::state(&state, out);
    failure(status, error, hint)
}

/// An entry that ended `failed` is the error `bench_entry_failed`, with the entry's error and hint.
fn failure(status: BenchEntryStatus, error: Option<String>, hint: Option<String>) -> Result<(), CliError> {
    if status != BenchEntryStatus::Failed {
        return Ok(());
    }
    let err = CliError::new(
        ExitCode::Runtime,
        "bench_entry_failed",
        error.unwrap_or_else(|| "the benchmark entry failed".into()),
    );
    Err(err.hint(hint.unwrap_or_else(|| "see `folo bench status BENCH ENTRY`".into())))
}

/// A store or server error as the CLI reports it, with a hint in its words: the same code (the API's and the
/// tools' names for bad input are the CLI's `usage`) and its exit code.
fn api_error(e: folo_server::error::Error) -> CliError {
    let mut e = hints::decorate(e, Front::Cli);
    if matches!(
        e.body.code.as_str(),
        "invalid_arguments" | "invalid_body" | "invalid_query" | "invalid_path" | "bad_request"
    ) {
        e.body.code = "usage".into();
        e.body.hint.get_or_insert_with(|| "see `folo bench <command> --help`".into());
    }
    let err = CliError::new(ExitCode::of(&e.body.code), &e.body.code, e.body.message);
    match e.body.hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

fn usage(message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "usage", message).hint("see `folo bench --help`")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_is_json_or_a_file_of_it() {
        let p = plan(r#"{"conditions":[{"condition":"haze","severities":[1,3]}]}"#).unwrap();
        assert_eq!((p.conditions[0].condition.as_str(), p.conditions[0].weight), ("haze", 1.0));
        let f = folo_test_support::TestFile::new("bench-plan", "plan.json");
        std::fs::write(&f, r#"{"conditions":[]}"#).unwrap();
        assert!(plan(f.to_str().unwrap()).unwrap().conditions.is_empty());
        assert_eq!(plan("{oops").unwrap_err().code, "usage");
        assert_eq!(
            plan(&std::path::Path::new("no").join("plan.json").to_string_lossy()).unwrap_err().code,
            "usage"
        );
        // Misspelled keys (e.g. `wieght` instead of `weight`) must fail loudly instead of silently defaulting.
        let typo = plan(r#"{"conditions":[{"condition":"haze","severities":[2],"wieght":3}]}"#).unwrap_err();
        assert_eq!(typo.code, "usage");
        assert!(typo.message.contains("wieght"), "{}", typo.message);
        let top_typo = plan(r#"{"condtions":[]}"#).unwrap_err();
        assert_eq!(top_typo.code, "usage");
        assert!(top_typo.message.contains("condtions"), "{}", top_typo.message);
        let designer_typo =
            plan(r#"{"conditions":[],"designed_by":{"agent":"claude","modle":"opus"}}"#).unwrap_err();
        assert_eq!(designer_typo.code, "usage");
        assert!(designer_typo.message.contains("modle"), "{}", designer_typo.message);
    }

    #[test]
    fn run_entry_rejects_zero_max_images_before_opening_store_or_network() {
        let err = run(
            BenchCommand::Run(BenchRunArgs {
                bench: "bn_1".into(),
                model: Some("mdl_1".into()),
                predictions: None,
                name: None,
                max_images: Some(0),
                no_conditions: false,
                unscored: false,
                wait: false,
                device: None,
                conn: BenchServerArgs {
                    data_dir: None,
                    server: Some("http://127.0.0.1:9".into()),
                    token: None,
                    tls_ca: None,
                },
            }),
            Out { json: true },
        )
        .unwrap_err();
        assert_eq!((err.exit, err.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(err.message.contains("--max-images"), "{}", err.message);
    }
}
