//! The cloud control plane end to end (docs/CLOUD.md) against in-process workers: a test provider whose `launch`
//! starts a real store behind the launch's token on a loopback port, with every provider call recorded and any of
//! them scriptable (fail, panic, lose its answer, hold until released). Tiny synthetic datasets, one- to three-epoch
//! runs on the CPU, and short timings, so every lifecycle path runs in seconds: whole runs, stops in every phase,
//! resumes (on a new instance and here), refused and lost launches, boot failures, a worker that dies, terminations
//! the provider refuses, crash recovery, the deadline, a `remote` connection, and `/api/cloud` over HTTP.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use folo_cloud::{Account, ErrorKind, InstanceState, LaunchSpec, Provider, Values};
use folo_server::api::*;
use folo_server::cloud::Tuning;
use folo_server::store::{LogsQuery, MetricsQuery};
use folo_server::{Options, RealStore, Server, Store, Token};
use folo_test_support::TestDir;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Longest wait for a run or an instance to reach a state.
const TIMEOUT: Duration = Duration::from_secs(120);
/// The test connection's API key: it must never leave `connections.json`.
const SECRET: &str = "test-secret-key-0123456789";

/// Every test here trains on the CPU, whatever GPU the machine has: no adapter matches this filter, so `device:
/// auto` resolves to the CPU in the control plane and in every worker (they share the process).
fn no_gpu() {
    static ONCE: Once = Once::new();
    // SAFETY: every test calls this first, before it starts a thread, and the other tests wait for the `Once`, so
    // nothing reads the environment while it changes.
    ONCE.call_once(|| unsafe { std::env::set_var("FOLO_GPU_ADAPTER", "folo-cloud-tests-no-such-adapter") });
}

// ---- the in-process cloud -------------------------------------------------------------------------------------------

/// A provider call, as recorded and scripted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Call {
    Check,
    Offers,
    Launch,
    State,
    Find,
    Terminate,
}

/// What a scripted call does instead of its job.
#[derive(Clone, Debug)]
enum Fault {
    /// Fails with this error.
    Fail(ErrorKind),
    /// `launch` only: creates the instance, then fails as if its answer was lost (`Unavailable`).
    LostAnswer,
    /// Panics, like a provider with a bug.
    Panic,
}

/// A router served on a loopback port by its own runtime, until dropped.
struct Live {
    url: String,
    rt: Option<tokio::runtime::Runtime>,
}

impl Live {
    fn start(router: Router) -> Live {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        rt.spawn(async move { axum::serve(listener, router).await.unwrap() });
        Live { url, rt: Some(rt) }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            rt.shutdown_background();
        }
    }
}

/// What the workers' fronts do to requests (set by a test).
#[derive(Default)]
struct Gates {
    /// Uploads wait while set.
    uploads: AtomicBool,
    /// `/api/system` answers like a FOLO from before cloud workers.
    old: AtomicBool,
}

async fn gate(State(g): State<Arc<Gates>>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    while path.starts_with("/api/uploads") && g.uploads.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    if path == "/api/system" && g.old.load(Ordering::SeqCst) {
        let system = json!({ "cpu": { "name": "c", "threads": 1, "util": 0.0 },
                             "memory": { "used_bytes": 0, "total_bytes": 0 }, "gpus": [],
                             "features": ["queue_control"] });
        return axum::Json(system).into_response();
    }
    next.run(req).await
}

/// A launched instance: a real store and, while it runs, its server.
struct Instance {
    name: String,
    store: Arc<RealStore>,
    url: String,
    live: Option<Live>,
    /// `state` answers `starting` this many more times.
    starting: u32,
}

#[derive(Default)]
struct Script {
    calls: Vec<(Call, String)>,
    faults: Vec<(Call, Fault, usize)>,
    held: HashSet<Call>,
    instances: BTreeMap<String, Instance>,
    launched: u32,
}

/// The test provider (see the module docs).
struct TestCloud {
    root: PathBuf,
    script: Mutex<Script>,
    released: Condvar,
    gates: Arc<Gates>,
    hold_epochs: AtomicBool,
    starting: AtomicU32,
}

fn offer(id: &str, availability: CloudAvailability) -> CloudOffer {
    CloudOffer {
        id: id.into(),
        gpu: "Test GPU".into(),
        gpu_count: 1,
        vram_bytes: 8 << 30,
        price_per_hour: Some(0.36),
        region: Some("loopback".into()),
        detail: None,
        availability,
    }
}

impl TestCloud {
    fn new(root: &Path) -> Arc<TestCloud> {
        Arc::new(TestCloud {
            root: root.to_path_buf(),
            script: Mutex::default(),
            released: Condvar::new(),
            gates: Arc::default(),
            hold_epochs: AtomicBool::new(false),
            starting: AtomicU32::new(1),
        })
    }

    /// The next `times` calls of `call` do `fault`.
    fn fail(&self, call: Call, fault: Fault, times: usize) {
        self.script.lock().unwrap().faults.push((call, fault, times));
    }

    /// `call` succeeds again.
    fn clear(&self, call: Call) {
        self.script.lock().unwrap().faults.retain(|f| f.0 != call);
    }

    /// Calls of `call` wait until [`TestCloud::release`].
    fn hold(&self, call: Call) {
        self.script.lock().unwrap().held.insert(call);
    }

    fn release(&self, call: Call) {
        self.script.lock().unwrap().held.remove(&call);
        self.released.notify_all();
    }

    /// Workers (also those running) hold their runs after each epoch while `on`.
    fn hold_epochs(&self, on: bool) {
        self.hold_epochs.store(on, Ordering::SeqCst);
        for i in self.script.lock().unwrap().instances.values() {
            i.store.hold_runs_after_epochs(on);
        }
    }

    fn calls(&self, call: Call) -> usize {
        self.script.lock().unwrap().calls.iter().filter(|c| c.0 == call).count()
    }

    /// The instances whose server runs.
    fn live(&self) -> usize {
        self.script.lock().unwrap().instances.values().filter(|i| i.live.is_some()).count()
    }

    /// The store of the `n`-th instance launched (0-based).
    fn worker(&self, n: usize) -> Arc<RealStore> {
        self.script.lock().unwrap().instances.values().nth(n).expect("launched").store.clone()
    }

    /// Records `call` with `arg`, waits while it is held, and plays its next fault: `Ok(true)` for a lost answer.
    fn call(&self, call: Call, arg: &str) -> folo_cloud::Result<bool> {
        let mut s = self.script.lock().unwrap();
        s.calls.push((call, arg.to_string()));
        while s.held.contains(&call) {
            s = self.released.wait(s).unwrap();
        }
        let Some(i) = s.faults.iter().position(|f| f.0 == call) else { return Ok(false) };
        let fault = s.faults[i].1.clone();
        s.faults[i].2 -= 1;
        if s.faults[i].2 == 0 {
            s.faults.remove(i);
        }
        drop(s);
        match fault {
            Fault::Fail(kind) => {
                Err(folo_cloud::Error::new(kind, format!("the test cloud refuses {call:?}")))
            }
            Fault::LostAnswer => Ok(true),
            Fault::Panic => panic!("the test cloud panics in {call:?}"),
        }
    }

    /// Shuts instance `r`'s worker down (its server, then its store): terminated, or died.
    fn stop_instance(&self, r: &str) {
        let (live, store) = {
            let mut s = self.script.lock().unwrap();
            let Some(i) = s.instances.get_mut(r) else { return };
            (i.live.take(), i.store.clone())
        };
        drop(live);
        store.shutdown();
    }

    /// The `n`-th instance dies: its server and store stop, and the provider says it is gone.
    fn kill(&self, n: usize) {
        let r = self.script.lock().unwrap().instances.keys().nth(n).expect("launched").clone();
        self.stop_instance(&r);
    }
}

impl Drop for TestCloud {
    fn drop(&mut self) {
        let refs: Vec<String> = self.script.lock().unwrap().instances.keys().cloned().collect();
        for r in refs {
            self.stop_instance(&r);
        }
    }
}

impl Provider for TestCloud {
    fn info(&self) -> CloudProvider {
        CloudProvider {
            id: "test".into(),
            name: "Test cloud".into(),
            about: "In-process workers for the control plane's tests.".into(),
            docs_url: None,
            fields: vec![CloudField {
                key: "api_key".into(),
                label: "API key".into(),
                secret: true,
                required: true,
                placeholder: None,
                help: None,
            }],
            provisions: true,
            self_terminates: false,
        }
    }

    fn check(&self, values: &Values) -> folo_cloud::Result<Account> {
        self.call(Call::Check, "")?;
        if values.require("api_key")?.contains("bad") {
            return Err(folo_cloud::Error::new(ErrorKind::Auth, "the test cloud rejects this key"));
        }
        Ok(Account { label: Some("tester@example.com".into()), balance_usd: Some(12.5), note: None })
    }

    fn offers(&self, _: &Values) -> folo_cloud::Result<Vec<CloudOffer>> {
        self.call(Call::Offers, "")?;
        Ok(vec![offer("cpu@test", CloudAvailability::High), offer("none@test", CloudAvailability::None)])
    }

    fn launch(&self, _: &Values, spec: &LaunchSpec) -> folo_cloud::Result<String> {
        let lost = self.call(Call::Launch, &spec.name)?;
        if spec.offer_id != "cpu@test" {
            return Err(folo_cloud::Error::new(ErrorKind::Refused, "no capacity for this offer"));
        }
        let mut s = self.script.lock().unwrap();
        s.launched += 1;
        let r = format!("pod-{}", s.launched);
        let store = Arc::new(RealStore::open(self.root.join(&r)).recheck_every(Duration::ZERO).device("cpu"));
        store.hold_runs_after_epochs(self.hold_epochs.load(Ordering::SeqCst));
        let options = Options { token: Token::new(spec.token.clone()), ..Options::default() };
        let router = Server::new(store.clone(), options)
            .router()
            .layer(axum::middleware::from_fn_with_state(self.gates.clone(), gate));
        let live = Live::start(router);
        let instance = Instance {
            name: spec.name.clone(),
            store,
            url: live.url.clone(),
            live: Some(live),
            starting: self.starting.load(Ordering::SeqCst),
        };
        s.instances.insert(r.clone(), instance);
        if lost {
            return Err(folo_cloud::Error::new(ErrorKind::Unavailable, "the launch answer was lost"));
        }
        Ok(r)
    }

    fn state(&self, _: &Values, r: &str) -> folo_cloud::Result<InstanceState> {
        self.call(Call::State, r)?;
        let mut s = self.script.lock().unwrap();
        Ok(match s.instances.get_mut(r) {
            Some(i) if i.live.is_some() && i.starting > 0 => {
                i.starting -= 1;
                InstanceState::Starting
            }
            Some(i) if i.live.is_some() => InstanceState::Running { url: i.url.clone() },
            _ => InstanceState::Gone,
        })
    }

    fn find(&self, _: &Values, name: &str) -> folo_cloud::Result<Vec<String>> {
        self.call(Call::Find, name)?;
        let s = self.script.lock().unwrap();
        Ok(s.instances
            .iter()
            .filter(|(_, i)| i.name == name && i.live.is_some())
            .map(|(r, _)| r.clone())
            .collect())
    }

    fn terminate(&self, _: &Values, r: &str) -> folo_cloud::Result<()> {
        self.call(Call::Terminate, r)?;
        self.stop_instance(r);
        Ok(())
    }
}

// ---- helpers --------------------------------------------------------------------------------------------------------

/// Timings short enough for tests (a run copies its checkpoint after every epoch).
fn fast() -> Tuning {
    Tuning {
        poll: Duration::from_millis(20),
        poll_max: Duration::from_millis(100),
        boot_timeout: Duration::from_secs(30),
        remote_boot_timeout: Duration::from_secs(10),
        checkpoint_every: Duration::ZERO,
        reaper_every: Duration::from_millis(100),
        lost_grace: Duration::from_millis(400),
        release_wait: Duration::from_millis(500),
        require_gpu: false,
        max_run: None,
    }
}

/// A control plane on `home` with the test cloud.
fn plane(home: &Path, cloud: &Arc<TestCloud>, tuning: Tuning) -> Arc<RealStore> {
    let providers: Vec<Arc<dyn Provider>> = vec![cloud.clone()];
    Arc::new(RealStore::open_with_cloud(home.to_path_buf(), providers, tuning).recheck_every(Duration::ZERO))
}

/// A 12-image synthetic dataset (64 px, two classes).
fn dataset(store: &RealStore) -> Dataset {
    let req = SyntheticDataset {
        name: "tiny".into(),
        images: 12,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 5,
    };
    store.create_synthetic(req).unwrap()
}

/// A connection to the test cloud.
fn connect(store: &RealStore) -> CloudConnection {
    let values = BTreeMap::from([("api_key".to_string(), SECRET.to_string())]);
    store.connect_cloud(ConnectCloud { provider: "test".into(), name: String::new(), values }).unwrap()
}

/// A tiny run of `epochs` on `connection`'s CPU offer.
fn cloud_run(ds: &Dataset, connection: &str, epochs: u32) -> TrainConfig {
    serde_json::from_value(json!({
        "dataset_id": ds.id, "epochs": epochs, "batch": 4, "imgsz": 64, "augment": { "close_mosaic": 1 },
        "compute": { "connection_id": connection, "offer_id": "cpu@test" }
    }))
    .unwrap()
}

/// Polls `f` until it gives a value.
fn until<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(t0.elapsed() < TIMEOUT, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn run_of(store: &RealStore, id: &str) -> Run {
    store.get_run(id).unwrap()
}

/// Waits until run `id` ended (its final status is published after the release).
fn finished(store: &RealStore, id: &str) -> Run {
    until(&format!("{id} to end"), || Some(run_of(store, id)).filter(|r| !r.status.is_active()))
}

fn phase(r: &Run) -> CloudPhase {
    r.cloud.as_ref().expect("a cloud run").phase
}

fn log_of(store: &RealStore, id: &str) -> Vec<LogLine> {
    store.run_logs(id, &LogsQuery::last(5000)).unwrap().lines
}

fn logged(store: &RealStore, id: &str, part: &str) -> bool {
    log_of(store, id).iter().any(|l| l.message.contains(part))
}

/// Collects run `id`'s events until its final status.
fn record(store: &RealStore, id: &str) -> std::thread::JoinHandle<Vec<RunEvent>> {
    let mut events = store.subscribe(id).unwrap().events;
    std::thread::spawn(move || {
        let mut seen = Vec::new();
        while let Ok(ev) = events.blocking_recv() {
            let end = matches!(&ev, RunEvent::Status(s) if !s.status.is_active());
            seen.push(ev);
            if end {
                break;
            }
        }
        seen
    })
}

fn steps(m: &RunMetrics) -> Vec<u64> {
    m.steps.iter().map(|s| s.step).collect()
}

fn epochs(m: &RunMetrics) -> Vec<u32> {
    m.epochs.iter().map(|e| e.epoch).collect()
}

fn resume(store: &RealStore, id: &str, extra: Value) -> Run {
    let fields = extra.as_object().unwrap().clone();
    store.resume_run(id, &fields).unwrap()
}

// ---- tests ----------------------------------------------------------------------------------------------------------

#[test]
fn a_cloud_run_trains_on_an_instance_mirrors_it_and_releases_it() {
    no_gpu();
    let t = TestDir::new("cloud-run");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    assert_eq!((conn.account.as_deref(), conn.balance_usd), (Some("tester@example.com"), Some(12.5)));

    cloud.hold(Call::Launch);
    let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
    assert_eq!((run.status, run.queue_position), (RunStatus::Queued, None));
    let c = run.cloud.clone().unwrap();
    assert_eq!(
        (c.phase, c.provider.as_str(), c.offer.id.as_str()),
        (CloudPhase::Provisioning, "test", "cpu@test")
    );
    let events = record(&store, &run.id);
    cloud.release(Call::Launch);
    let done = finished(&store, &run.id);
    let events = events.join().unwrap();
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));

    // Phases in order, in the events and in the run; the status stays `queued` until the worker trains.
    let mut phases: Vec<CloudPhase> = Vec::new();
    let mut running_at = None;
    for ev in &events {
        match ev {
            RunEvent::Cloud(c) if phases.last() != Some(&c.phase) => phases.push(c.phase),
            RunEvent::Status(s) if s.status == RunStatus::Running => running_at = Some(phases.clone()),
            _ => {}
        }
    }
    assert_eq!(phases, CloudPhase::ALL);
    assert_eq!(running_at.unwrap().last(), Some(&CloudPhase::Training));
    let c = done.cloud.clone().unwrap();
    assert_eq!(c.phase, CloudPhase::Released);
    assert!(c.instance_id.is_some() && c.deadline.is_some() && c.synced.is_some() && c.cost_usd.is_some());

    // The worker trained the run without `compute`, on its own device, with the dataset sent as FOLO reads it.
    let worker = cloud.worker(0);
    let w = worker.list_runs().unwrap().remove(0);
    assert!(w.config.compute.is_none() && w.config.device == "auto" && w.name.contains(&run.id), "{w:?}");
    assert_eq!((done.device.as_str(), done.steps_per_epoch), (w.device.as_str(), w.steps_per_epoch));
    assert_eq!(done.resolved, w.resolved);
    // Its history, here: every step and epoch, every log line, every sample (with this dataset's ids).
    let all = MetricsQuery::default();
    let (local, remote) =
        (store.run_metrics(&run.id, &all).unwrap(), worker.run_metrics(&w.id, &all).unwrap());
    assert_eq!((steps(&local), epochs(&local)), ((1..=6).collect::<Vec<_>>(), vec![1, 2]));
    assert_eq!(local, remote);
    let here = log_of(&store, &run.id);
    let mut rest = here.iter();
    for line in log_of(&worker, &w.id) {
        assert!(rest.any(|l| *l == line), "the worker's line `{}` is not in the run's log", line.message);
    }
    for phase_line in
        ["launching Test GPU at Test cloud", "uploading the dataset tiny", "copied weights/last.folo"]
    {
        assert!(here.iter().any(|l| l.message.contains(phase_line)), "no `{phase_line}` in {here:?}");
    }
    assert!(
        here.iter().any(|l| l.message.starts_with("terminated the instance; it cost about $")),
        "{here:?}"
    );
    for epoch in [1, 2] {
        let (a, b) = (
            store.run_samples(&run.id, Some(epoch)).unwrap(),
            worker.run_samples(&w.id, Some(epoch)).unwrap(),
        );
        assert_eq!(a.samples.len(), b.samples.len());
        for (x, y) in a.samples.iter().zip(&b.samples) {
            assert_eq!((&x.image_id, &x.truth, &x.pred), (&y.image_id, &y.truth, &y.pred));
            assert_eq!(x.dataset_id, ds.id);
            store.image(&ds.id, &x.image_id, None).unwrap();
        }
    }
    // The weights are the run's models.
    let model = done.model_id.clone().unwrap();
    for id in [model.clone(), format!("{model}-last")] {
        assert_eq!(store.get_model(&id).unwrap().run_id.as_deref(), Some(run.id.as_str()));
    }
    // Nothing is left at the provider; the ledger says what it cost.
    assert_eq!(cloud.live(), 0);
    let instances = store.cloud_instances().unwrap();
    assert_eq!(instances.len(), 1);
    let i = &instances[0];
    assert_eq!(
        (i.status, i.run_id.as_deref(), i.provider_ref.as_deref()),
        (CloudInstanceStatus::Terminated, Some(run.id.as_str()), Some("pod-1"))
    );
    assert!(i.cost_usd.is_some() && i.ended.is_some() && i.name.starts_with(&format!("folo-{}-", run.id)));
    assert_eq!(cloud.calls(Call::Terminate), 1);
}

#[test]
fn a_stopped_run_resumes_on_a_new_instance_and_then_here() {
    no_gpu();
    let t = TestDir::new("cloud-resume");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("epoch 1 here", || run_of(&store, &run.id).best);
    store.stop_run(&run.id).unwrap();
    let stopped = finished(&store, &run.id);
    assert_eq!(
        (stopped.status, stopped.epoch, phase(&stopped)),
        (RunStatus::Stopped, 1, CloudPhase::Released)
    );
    let dir = t.join("home").join("runs").join(&run.id);
    assert!(dir.join("weights").join("last.folo").is_file(), "the checkpoint was collected");
    assert_eq!(cloud.live(), 0);

    // On a new instance, from the checkpoint: the step numbers go on.
    let resumed = resume(&store, &run.id, json!({}));
    assert_eq!((resumed.status, phase(&resumed)), (RunStatus::Queued, CloudPhase::Provisioning));
    until("epoch 2 here", || {
        Some(run_of(&store, &run.id)).filter(|r| {
            r.best.is_some_and(|_| r.epoch >= 2)
                && store.run_metrics(&run.id, &MetricsQuery::default()).unwrap().epochs.len() == 2
        })
    });
    store.stop_run(&run.id).unwrap();
    let again = finished(&store, &run.id);
    assert_eq!((again.status, again.epoch), (RunStatus::Stopped, 2));
    assert_eq!(store.cloud_instances().unwrap().len(), 2);
    assert!(logged(&store, &run.id, "uploading the checkpoint weights/last.folo"));

    // Here: `compute: null` resumes on this machine; the run keeps its last cloud state.
    cloud.hold_epochs(false);
    let here = resume(&store, &run.id, json!({ "compute": null }));
    assert!(here.config.compute.is_none() && phase(&here) == CloudPhase::Released);
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.epoch), (RunStatus::Completed, 3));
    let m = store.run_metrics(&run.id, &MetricsQuery::default()).unwrap();
    assert_eq!((steps(&m), epochs(&m)), ((1..=9).collect::<Vec<_>>(), vec![1, 2, 3]));
    assert_eq!(cloud.live(), 0);
}

#[test]
fn a_stop_before_training_releases_at_once() {
    no_gpu();
    let t = TestDir::new("cloud-early-stop");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    let stopped = |phase_now: CloudPhase, hold: &dyn Fn(), go: &dyn Fn()| {
        hold();
        let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
        until(&format!("{phase_now:?}"), || (phase(&run_of(&store, &run.id)) == phase_now).then_some(()));
        let r = store.stop_run(&run.id).unwrap();
        assert_eq!(r.status, RunStatus::Stopping);
        go();
        let done = finished(&store, &run.id);
        assert_eq!((done.status, done.model_id.as_deref()), (RunStatus::Stopped, None));
        assert!(logged(&store, &run.id, "stopped before the worker trained"));
        assert_eq!(cloud.live(), 0, "released while {phase_now:?}");
    };
    stopped(CloudPhase::Provisioning, &|| cloud.hold(Call::Launch), &|| cloud.release(Call::Launch));
    stopped(CloudPhase::Booting, &|| cloud.hold(Call::State), &|| cloud.release(Call::State));
    let g = cloud.gates.clone();
    stopped(CloudPhase::Syncing, &|| g.uploads.store(true, Ordering::SeqCst), &|| {
        g.uploads.store(false, Ordering::SeqCst)
    });
    let all = store.cloud_instances().unwrap();
    assert_eq!(all.len(), 3);
    assert!(all.iter().all(|i| i.status == CloudInstanceStatus::Terminated), "{all:?}");
}

#[test]
fn refused_and_lost_launches() {
    no_gpu();
    let t = TestDir::new("cloud-launch");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    // Refused: no instance, and no record of one.
    cloud.fail(Call::Launch, Fault::Fail(ErrorKind::Refused), 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.error_code.as_deref()), (RunStatus::Failed, Some("cloud_refused")));
    assert!(done.cloud.unwrap().instance_id.is_none());
    assert!(store.cloud_instances().unwrap().is_empty());

    // Created, but the answer was lost: found by its name and used.
    cloud.fail(Call::Launch, Fault::LostAnswer, 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    assert!(logged(&store, &run.id, "but the instance exists; using it"));
    assert_eq!(cloud.calls(Call::Launch), 2, "one instance, not a second launch");
    assert_eq!(cloud.live(), 0);

    // Lost, and the provider cannot say whether it exists: the run fails, and the release looks it up again.
    cloud.fail(Call::Launch, Fault::LostAnswer, 1);
    cloud.fail(Call::Find, Fault::Fail(ErrorKind::Unavailable), 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.error_code.as_deref(), Some("cloud_unavailable"));
    until("the lost launch released", || (cloud.live() == 0).then_some(()));
}

#[test]
fn boot_failures_name_the_check_and_release() {
    no_gpu();
    let t = TestDir::new("cloud-boot");
    let cloud = TestCloud::new(&t.join("workers"));
    let boot = |name: &str, tuning: Tuning| {
        let store = plane(&t.join(name), &cloud, tuning);
        let ds = dataset(&store);
        let conn = connect(&store);
        let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
        let done = finished(&store, &run.id);
        assert_eq!(done.error_code.as_deref(), Some("cloud_boot_failed"), "{name}: {:?}", done.error);
        assert_eq!(cloud.live(), 0, "{name}: released");
        done.error.unwrap()
    };
    cloud.starting.store(u32::MAX, Ordering::SeqCst);
    let e = boot("timeout", Tuning { boot_timeout: Duration::from_millis(300), ..fast() });
    assert!(e.contains("did not start within"), "{e}");
    cloud.starting.store(1, Ordering::SeqCst);
    cloud.gates.old.store(true, Ordering::SeqCst);
    let e = boot("old", fast());
    assert!(
        e.contains("cannot train cloud runs") && e.contains("uploads, run_cursors, run_checkpoint"),
        "{e}"
    );
    cloud.gates.old.store(false, Ordering::SeqCst);
    let e = boot("no-gpu", Tuning { require_gpu: true, ..fast() });
    assert!(e.contains("no GPU FOLO can use"), "{e}");
}

#[test]
fn a_worker_that_dies_loses_the_run_but_not_its_checkpoint() {
    no_gpu();
    let t = TestDir::new("cloud-lost");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
    until("the first checkpoint copied", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    cloud.kill(0);
    let lost = finished(&store, &run.id);
    assert_eq!((lost.status, lost.error_code.as_deref()), (RunStatus::Failed, Some("cloud_instance_lost")));
    assert!(t.join("home").join("runs").join(&run.id).join("weights").join("last.folo").is_file());
    cloud.hold_epochs(false);
    resume(&store, &run.id, json!({}));
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.epoch), (RunStatus::Completed, 2));
}

#[test]
fn a_panic_still_releases_the_instance() {
    no_gpu();
    let t = TestDir::new("cloud-panic");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.fail(Call::State, Fault::Panic, 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.error_code.as_deref()), (RunStatus::Failed, Some("internal")));
    assert!(done.error.as_deref().unwrap().contains("panicked"));
    assert_eq!((cloud.live(), phase(&done)), (0, CloudPhase::Released));
}

#[test]
fn a_refused_termination_is_retried_until_it_succeeds() {
    no_gpu();
    let t = TestDir::new("cloud-terminate");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.fail(Call::Terminate, Fault::Fail(ErrorKind::Unavailable), usize::MAX);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "the run does not wait for its instance");
    assert!(logged(&store, &run.id, "the instance is not terminated yet"));
    let i = store.cloud_instances().unwrap().remove(0);
    assert_eq!(i.status, CloudInstanceStatus::Terminating);
    assert!(i.error.unwrap().contains("refuses Terminate"));
    assert!(
        cloud.live() == 1 && store.disconnect_cloud(&conn.id).unwrap_err().body.code == "connection_in_use"
    );
    cloud.clear(Call::Terminate);
    until("the reaper's termination", || {
        (store.cloud_instances().unwrap()[0].status == CloudInstanceStatus::Terminated).then_some(())
    });
    assert_eq!(cloud.live(), 0);
    assert!(store.cloud_instances().unwrap()[0].error.is_none());
}

#[test]
fn a_restart_releases_what_a_crash_left_and_follows_a_training_run_again() {
    no_gpu();
    let t = TestDir::new("cloud-restart");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    // A run that trains when this server goes away: the next one follows it to the end.
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("epoch 1 here", || run_of(&store, &run.id).best);
    drop(store);
    // Meanwhile a crash left two `launching` records: one whose launch created an instance, one whose did not.
    let values = Values::from(BTreeMap::from([("api_key".to_string(), SECRET.to_string())]));
    let spec = |name: &str| LaunchSpec {
        name: name.into(),
        offer_id: "cpu@test".into(),
        disk_gb: 20,
        token: "t".into(),
        lifetime_s: 60,
        worker_base_url: "http://x".into(),
    };
    cloud.launch(&values, &spec("folo-run_gone-a1b2")).unwrap();
    let record = |id: &str, name: &str| {
        let i = json!({ "id": id, "connection_id": conn.id, "provider": "test", "provider_ref": null, "name": name,
                        "run_id": "run_gone", "offer": offer("cpu@test", CloudAvailability::High),
                        "status": "launching", "created": "2026-11-09T00:00:00Z" });
        std::fs::write(home.join("cloud").join("instances").join(format!("{id}.json")), i.to_string())
            .unwrap();
    };
    record("ci_dead", "folo-run_gone-a1b2");
    record("ci_none", "folo-run_gone-c3d4");
    assert_eq!(cloud.live(), 2, "the training run's worker and the orphan");

    let store = plane(&home, &cloud, fast());
    until("the run followed again", || {
        logged(&store, &run.id, "following the run on its worker again").then_some(())
    });
    until("the orphan terminated", || {
        let all = store.cloud_instances().unwrap();
        let orphan = all.iter().find(|i| i.id == "ci_dead").map(|i| i.status);
        (orphan == Some(CloudInstanceStatus::Terminated) && !all.iter().any(|i| i.id == "ci_none"))
            .then_some(())
    });
    cloud.hold_epochs(false);
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    let m = store.run_metrics(&run.id, &MetricsQuery::default()).unwrap();
    assert_eq!((steps(&m), epochs(&m)), ((1..=9).collect::<Vec<_>>(), vec![1, 2, 3]));
    let worker = cloud.worker(0);
    let w = worker.list_runs().unwrap().remove(0);
    assert_eq!(m, worker.run_metrics(&w.id, &MetricsQuery::default()).unwrap(), "a complete history");
    assert_eq!(cloud.live(), 0);
}

#[test]
fn a_shutdown_before_training_releases_the_instance() {
    no_gpu();
    let t = TestDir::new("cloud-shutdown");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    // The instance never finishes starting.
    cloud.starting.store(u32::MAX, Ordering::SeqCst);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    until("the instance launched", || (cloud.live() == 1).then_some(()));
    store.shutdown();
    assert_eq!(cloud.live(), 0, "released at the shutdown, not at the next start");
    let stopped = run_of(&store, &run.id);
    assert_eq!((stopped.status, phase(&stopped)), (RunStatus::Stopped, CloudPhase::Released));
    assert!(
        logged(&store, &run.id, "this server shuts down while the run is"),
        "{:?}",
        log_of(&store, &run.id)
    );
    assert_eq!(store.cloud_instances().unwrap()[0].status, CloudInstanceStatus::Terminated);
}

#[test]
fn the_deadline_stops_the_run_and_releases_its_instance() {
    no_gpu();
    let t = TestDir::new("cloud-deadline");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, Tuning { max_run: Some(Duration::from_secs(3)), ..fast() });
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Stopped, "{:?}", log_of(&store, &run.id));
    assert!(logged(&store, &run.id, "the time limit of 3.0 s"), "{:?}", log_of(&store, &run.id));
    assert_eq!(cloud.live(), 0);
}

#[test]
fn a_remote_server_is_used_as_it_is_and_keeps_the_dataset() {
    no_gpu();
    let t = TestDir::new("cloud-remote");
    let worker = Arc::new(RealStore::open(t.join("worker")).recheck_every(Duration::ZERO).device("cpu"));
    let options = Options { token: Token::new("remote-token-0123456789"), ..Options::default() };
    let live = Live::start(Server::new(worker.clone(), options).router());
    let store = Arc::new(
        RealStore::open_with_cloud(t.join("home"), folo_cloud::providers(), fast())
            .recheck_every(Duration::ZERO),
    );
    let ds = dataset(&store);
    let values = BTreeMap::from([
        ("url".to_string(), live.url.clone()),
        ("token".to_string(), "remote-token-0123456789".to_string()),
    ]);
    let conn =
        store.connect_cloud(ConnectCloud { provider: "remote".into(), name: String::new(), values }).unwrap();
    assert_eq!(conn.values["token"], "••••6789");
    let offers = store.cloud_offers(&conn.id, false).unwrap().offers;
    assert_eq!((offers.len(), offers[0].id.as_str(), offers[0].price_per_hour), (1, "cpu", None));
    let config: TrainConfig = {
        serde_json::from_value(json!({ "dataset_id": ds.id, "epochs": 1, "batch": 4, "imgsz": 64,
                                       "compute": { "connection_id": conn.id, "offer_id": "cpu" } }))
        .unwrap()
    };
    for n in 0..2 {
        let run = store.create_run(config.clone()).unwrap();
        assert_eq!(phase(&run), CloudPhase::Booting, "nothing to provision");
        let done = finished(&store, &run.id);
        assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
        let c = done.cloud.unwrap();
        assert!(c.instance_id.is_none() && c.cost_usd.is_none());
        assert_eq!(logged(&store, &run.id, "the worker already has this dataset"), n == 1);
    }
    assert!(store.cloud_instances().unwrap().is_empty(), "no ledger record");
    assert_eq!((worker.list_datasets().unwrap().len(), worker.list_runs().unwrap().len()), (1, 2));
    drop(store);
    drop(live);
}

// ---- /api/cloud ----------------------------------------------------------------------------------------------------

async fn send(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let req = Request::builder().method(method).uri(uri);
    let req = match body {
        Some(b) => req.header("content-type", "application/json").body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    };
    let res = app.clone().oneshot(req.unwrap()).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn code(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    status: StatusCode,
    want: &str,
) -> Value {
    let (got, v) = send(app, method, uri, body).await;
    assert_eq!((got, v["error"]["code"].as_str()), (status, Some(want)), "{method} {uri}: {v}");
    v
}

/// Every file under `dir` that holds `needle`.
fn files_with(dir: &Path, needle: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in walk(dir) {
        if std::fs::read(&e).is_ok_and(|b| b.windows(needle.len()).any(|w| w == needle.as_bytes())) {
            out.push(e);
        }
    }
    out
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() { out.extend(walk(&p)) } else { out.push(p) }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn the_cloud_routes_check_redact_cache_and_refuse() {
    no_gpu();
    let t = TestDir::new("cloud-routes");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let app = Server::new(store.clone(), Options::default()).router();

    let (_, system) = send(&app, "GET", "/api/system", None).await;
    assert!(system["features"].as_array().unwrap().iter().any(|f| f == "cloud"));
    let (status, p) = send(&app, "GET", "/api/cloud/providers", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(p["providers"][0]["id"], "test");

    // Connecting checks everything before anything is stored.
    let connect = |values: Value| json!({ "provider": "test", "values": values });
    code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(json!({ "provider": "nope", "values": {} })),
        StatusCode::NOT_FOUND,
        "provider_not_found",
    )
    .await;
    let e = code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(connect(json!({}))),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e["error"]["message"].as_str().unwrap().contains("api_key"));
    let e = code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(connect(json!({ "api_key": SECRET, "region": "x" }))),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    assert!(e["error"]["message"].as_str().unwrap().contains("region"));
    code(
        &app,
        "POST",
        "/api/cloud/connections",
        Some(connect(json!({ "api_key": "bad-key-0123" }))),
        StatusCode::UNPROCESSABLE_ENTITY,
        "cloud_auth_failed",
    )
    .await;
    assert!(!home.join("cloud").join("connections.json").exists(), "nothing stored");
    let (status, c) =
        send(&app, "POST", "/api/cloud/connections", Some(connect(json!({ "api_key": SECRET })))).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!((c["name"].as_str(), c["values"]["api_key"].as_str()), (Some("Test cloud"), Some("••••6789")));
    let id = c["id"].as_str().unwrap().to_string();
    let (_, c2) =
        send(&app, "POST", "/api/cloud/connections", Some(connect(json!({ "api_key": "short" })))).await;
    assert_eq!((c2["name"].as_str(), c2["values"]["api_key"].as_str()), (Some("Test cloud 2"), Some("••••")));

    // A failed check is an answer, not an error.
    cloud.fail(Call::Check, Fault::Fail(ErrorKind::Auth), 1);
    let (status, checked) = send(&app, "POST", &format!("/api/cloud/connections/{id}/check"), None).await;
    assert_eq!((status, checked["status"].as_str()), (StatusCode::OK, Some("error")));
    assert!(checked["error"].as_str().unwrap().contains("refuses Check"));
    let (_, checked) = send(&app, "POST", &format!("/api/cloud/connections/{id}/check"), None).await;
    assert_eq!((checked["status"].as_str(), checked["error"].as_str()), (Some("ok"), None));
    code(
        &app,
        "POST",
        "/api/cloud/connections/cc_none/check",
        None,
        StatusCode::NOT_FOUND,
        "connection_not_found",
    )
    .await;

    // Offers are cached for a minute, unless refreshed.
    let offers = format!("/api/cloud/connections/{id}/offers");
    let (_, first) = send(&app, "GET", &offers, None).await;
    let (_, second) = send(&app, "GET", &offers, None).await;
    assert_eq!(
        (first["offers"][0]["id"].as_str(), &first["fetched"]),
        (Some("cpu@test"), &second["fetched"])
    );
    assert_eq!(cloud.calls(Call::Offers), 1);
    send(&app, "GET", &format!("{offers}?refresh=true"), None).await;
    assert_eq!(cloud.calls(Call::Offers), 2);

    // Cloud runs are checked like any run, plus compute.
    let ds = dataset(&store);
    let run = |compute: Value, extra: Value| {
        let mut body =
            json!({ "dataset_id": ds.id, "epochs": 1, "batch": 4, "imgsz": 64, "compute": compute });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        Some(body)
    };
    let compute = |offer: &str| json!({ "connection_id": id, "offer_id": offer });
    code(
        &app,
        "POST",
        "/api/runs",
        run(json!({ "connection_id": "cc_none", "offer_id": "cpu@test" }), json!({})),
        StatusCode::NOT_FOUND,
        "connection_not_found",
    )
    .await;
    code(
        &app,
        "POST",
        "/api/runs",
        run(compute("gone@test"), json!({})),
        StatusCode::NOT_FOUND,
        "offer_not_found",
    )
    .await;
    code(
        &app,
        "POST",
        "/api/runs",
        run(compute("cpu@test"), json!({ "device": "cpu" })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    let long = json!({ "connection_id": id, "offer_id": "cpu@test", "max_hours": 1000 });
    code(&app, "POST", "/api/runs", run(long, json!({})), StatusCode::BAD_REQUEST, "invalid_config").await;
    let small = json!({ "connection_id": id, "offer_id": "cpu@test", "disk_gb": 5 });
    code(&app, "POST", "/api/runs", run(small, json!({})), StatusCode::BAD_REQUEST, "invalid_config").await;
    assert!(store.list_runs().unwrap().is_empty(), "refused runs leave nothing");

    // The estimate is the reference's, with what it would cost.
    let (status, est) = send(&app, "POST", "/api/runs/estimate", run(compute("cpu@test"), json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{est}");
    assert_eq!(
        (est["device"].as_str(), est["source"].as_str(), est["confidence"].as_str()),
        (Some("cloud"), Some("reference"), Some("low"))
    );
    assert_eq!(est["cost"]["max_usd"], 4.32);
    assert!(est["cost"]["est_usd"].as_f64().unwrap() <= est["cost"]["est_usd_high"].as_f64().unwrap());

    // A connection with an active run cannot go; one launched instance is listed, and terminated by hand.
    cloud.hold(Call::State);
    let (status, r) = send(&app, "POST", "/api/runs", run(compute("cpu@test"), json!({}))).await;
    assert_eq!(status, StatusCode::CREATED, "{r}");
    let run_id = r["id"].as_str().unwrap().to_string();
    let e = code(
        &app,
        "DELETE",
        &format!("/api/cloud/connections/{id}"),
        None,
        StatusCode::CONFLICT,
        "connection_in_use",
    )
    .await;
    assert!(e["error"]["message"].as_str().unwrap().contains(&run_id));
    let instance = until("the instance", || {
        store.cloud_instances().unwrap().into_iter().find(|i| i.provider_ref.is_some())
    });
    cloud.release(Call::State);
    let (status, i) = send(&app, "DELETE", &format!("/api/cloud/instances/{}", instance.id), None).await;
    assert_eq!((status, i["status"].as_str()), (StatusCode::OK, Some("terminated")));
    let (_, again) = send(&app, "DELETE", &format!("/api/cloud/instances/{}", instance.id), None).await;
    assert_eq!(again["ended"], i["ended"], "terminating one that is gone answers its record");
    code(&app, "DELETE", "/api/cloud/instances/ci_none", None, StatusCode::NOT_FOUND, "instance_not_found")
        .await;
    let lost = finished(&store, &run_id);
    assert_eq!(lost.error_code.as_deref(), Some("cloud_instance_lost"));
    let (_, listed) = send(&app, "GET", "/api/cloud/instances", None).await;
    assert_eq!(listed["instances"].as_array().unwrap().len(), 1);
    let (status, _) =
        send(&app, "DELETE", &format!("/api/cloud/connections/{}", c2["id"].as_str().unwrap()), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, list) = send(&app, "GET", "/api/cloud/connections", None).await;
    assert_eq!(list["connections"].as_array().unwrap().len(), 1);

    // The secret is nowhere but in connections.json.
    let (_, everything) = send(&app, "GET", &format!("/api/runs/{run_id}/logs?limit=5000"), None).await;
    assert!(!everything.to_string().contains(SECRET) && !list.to_string().contains(SECRET));
    assert_eq!(files_with(&home, SECRET), [home.join("cloud").join("connections.json")]);
    let shown = format!("{:?}", store.cloud_connections().unwrap());
    assert!(!shown.contains(SECRET));
    drop(app);
}
