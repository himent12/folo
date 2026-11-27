//! The harness: timings, the CPU-only setup, the in-process cloud (a test provider whose `launch` starts a
//! real store behind the launch's token on a loopback port, every call recorded and any of them
//! scriptable), and helpers that make a control plane, a dataset, a connection and a run, and wait for them.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use folo_cloud::{Account, ErrorKind, InstanceState, LaunchSpec, Provider, Values};
use folo_server::api::*;
use folo_server::cloud::Tuning;
use folo_server::store::LogsQuery;
use folo_server::{Options, RealStore, Server, Store, Token};
use serde_json::{Value, json};

/// Longest wait for a run or an instance to reach a state: only a hang gets here. A wait takes seconds on a quiet
/// machine, but the waits for a first checkpoint copy or for every instance terminated took over two minutes with
/// this binary's 26 tests training at once on a busy runner.
const TIMEOUT: Duration = Duration::from_secs(600);
/// The test connection's API key: it must never leave `connections.json`.
pub(super) const SECRET: &str = "test-secret-key-0123456789";

/// Every test here trains on the CPU, whatever GPU the machine has: no adapter matches this filter, so `device:
/// auto` resolves to the CPU in the control plane and in every worker (they share the process).
pub(super) fn no_gpu() {
    static ONCE: Once = Once::new();
    // SAFETY: every test calls this first, before it starts a thread, and the other tests wait for the `Once`, so
    // nothing reads the environment while it changes.
    ONCE.call_once(|| unsafe { std::env::set_var("FOLO_GPU_ADAPTER", "folo-cloud-tests-no-such-adapter") });
}

// ---- the in-process cloud -------------------------------------------------------------------------------------------

/// A provider call, as recorded and scripted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Call {
    Check,
    Offers,
    Launch,
    State,
    Find,
    Terminate,
}

/// What a scripted call does instead of its job.
#[derive(Clone, Debug)]
pub(super) enum Fault {
    /// Fails with this error.
    Fail(ErrorKind),
    /// `launch` only: creates the instance, then fails as if its answer was lost (`Unavailable`).
    LostAnswer,
    /// `find` only: answers that there is no such instance, as a provider that does not list a new one yet.
    Hidden,
    /// Panics, like a provider with a bug.
    Panic,
}

/// A router served on a loopback port by its own runtime, until dropped.
pub(super) struct Live {
    pub(super) url: String,
    rt: Option<tokio::runtime::Runtime>,
}

impl Live {
    pub(super) fn start(router: Router) -> Live {
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
pub(super) struct Gates {
    /// Uploads wait while set.
    pub(super) uploads: AtomicBool,
    /// `/api/system` answers like a FOLO from before cloud workers.
    pub(super) old: AtomicBool,
    /// Nothing answers yet, as a provider's proxy does for an instance whose worker has not started: `404`.
    pub(super) down: AtomicBool,
    /// While set, the first request for weights cuts the worker off: from then on only weights are served, every
    /// other request is a proxy's `502` (`cut`).
    pub(super) cut_after_weights: AtomicBool,
    /// The worker is cut off (set by `cut_after_weights`; cleared, it answers again).
    pub(super) cut: AtomicBool,
    /// Requests for weights wait while set, and then find the instance as it is by then (gone, or not).
    pub(super) weights: AtomicBool,
    /// Requests for weights that waited at `weights` so far.
    pub(super) weights_held: AtomicU32,
}

async fn gone(State((gone, g)): State<(Arc<AtomicBool>, Arc<Gates>)>, req: Request, next: Next) -> Response {
    if req.uri().path().ends_with("/weights") && g.weights.load(Ordering::SeqCst) {
        g.weights_held.fetch_add(1, Ordering::SeqCst);
        while g.weights.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    if gone.load(Ordering::SeqCst) {
        return (StatusCode::BAD_GATEWAY, "proxy: the instance is gone").into_response();
    }
    next.run(req).await
}

/// While set, answers the next `POST /api/runs` with a proxy's `503` (the request never reaches the server), and
/// clears.
pub(super) async fn busy_once(State(busy): State<Arc<AtomicBool>>, req: Request, next: Next) -> Response {
    if req.method() == axum::http::Method::POST
        && req.uri().path() == "/api/runs"
        && busy.swap(false, Ordering::SeqCst)
    {
        return (StatusCode::SERVICE_UNAVAILABLE, "proxy: busy").into_response();
    }
    next.run(req).await
}

/// While its flag is set, the next `POST /api/runs` reaches the worker, and its answer is lost: the proxy answers
/// `502` once the run it created has ended there (the flag clears).
pub(super) async fn lose_answer(
    State((lose, worker)): State<(Arc<AtomicBool>, Arc<RealStore>)>,
    req: Request,
    next: Next,
) -> Response {
    let post = req.method() == axum::http::Method::POST && req.uri().path() == "/api/runs";
    if !(post && lose.swap(false, Ordering::SeqCst)) {
        return next.run(req).await;
    }
    let _created = next.run(req).await;
    while worker.list_runs().unwrap().iter().any(|r| r.status.is_active()) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    (StatusCode::BAD_GATEWAY, "proxy: the answer was lost").into_response()
}

async fn gate(State(g): State<Arc<Gates>>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    if g.down.load(Ordering::SeqCst) {
        return axum::http::StatusCode::NOT_FOUND.into_response();
    }
    if g.cut_after_weights.load(Ordering::SeqCst) {
        if path.ends_with("/weights") {
            g.cut.store(true, Ordering::SeqCst);
        } else if g.cut.load(Ordering::SeqCst) {
            return (StatusCode::BAD_GATEWAY, "proxy: no route to the instance").into_response();
        }
    }
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
    /// Set when the instance stops: its server answers `502`, as a provider's proxy does for an instance that is
    /// gone. (A closed port would do too, but Windows takes 2 s to refuse each connection to one.)
    gone: Arc<AtomicBool>,
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
    /// The servers of stopped instances, until the test cloud goes.
    stopped: Vec<Live>,
    /// Names `find` does not list (yet), as a provider that lists a new instance late.
    hidden: HashSet<String>,
    /// By name: the calls of `find` (1-based) that fail, and those that list nothing.
    find_fails: BTreeMap<String, Vec<usize>>,
    find_hides: BTreeMap<String, Vec<usize>>,
}

/// The test provider (see the module docs).
pub(super) struct TestCloud {
    root: PathBuf,
    script: Mutex<Script>,
    released: Condvar,
    pub(super) gates: Arc<Gates>,
    hold_epochs: AtomicBool,
    pub(super) starting: AtomicU32,
}

pub(super) fn offer(id: &str, availability: CloudAvailability) -> CloudOffer {
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
    pub(super) fn new(root: &Path) -> Arc<TestCloud> {
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
    pub(super) fn fail(&self, call: Call, fault: Fault, times: usize) {
        self.script.lock().unwrap().faults.push((call, fault, times));
    }

    /// `call` succeeds again.
    pub(super) fn clear(&self, call: Call) {
        self.script.lock().unwrap().faults.retain(|f| f.0 != call);
    }

    /// Calls of `call` wait until [`TestCloud::release`].
    pub(super) fn hold(&self, call: Call) {
        self.script.lock().unwrap().held.insert(call);
    }

    pub(super) fn release(&self, call: Call) {
        self.script.lock().unwrap().held.remove(&call);
        self.released.notify_all();
    }

    /// Workers (also those running) hold their runs after each epoch while `on`.
    pub(super) fn hold_epochs(&self, on: bool) {
        self.hold_epochs.store(on, Ordering::SeqCst);
        for i in self.script.lock().unwrap().instances.values() {
            i.store.hold_runs_after_epochs(on);
        }
    }

    pub(super) fn calls(&self, call: Call) -> usize {
        self.script.lock().unwrap().calls.iter().filter(|c| c.0 == call).count()
    }

    /// The calls of `call` with `arg`.
    pub(super) fn calls_with(&self, call: Call, arg: &str) -> usize {
        self.script.lock().unwrap().calls.iter().filter(|c| c.0 == call && c.1 == arg).count()
    }

    /// The `fails`-th and `hides`-th calls of `find` for `name` (1-based) fail, or list nothing.
    pub(super) fn script_find(&self, name: &str, fails: &[usize], hides: &[usize]) {
        let mut s = self.script.lock().unwrap();
        s.find_fails.insert(name.to_string(), fails.to_vec());
        s.find_hides.insert(name.to_string(), hides.to_vec());
    }

    /// `find` does not list instances named `name` while `hidden`.
    pub(super) fn hide(&self, name: &str, hidden: bool) {
        let mut s = self.script.lock().unwrap();
        if hidden {
            s.hidden.insert(name.to_string())
        } else {
            s.hidden.remove(name)
        };
    }

    /// The instances whose server runs.
    pub(super) fn live(&self) -> usize {
        self.script.lock().unwrap().instances.values().filter(|i| i.live.is_some()).count()
    }

    /// The store of the `n`-th instance launched (0-based).
    pub(super) fn worker(&self, n: usize) -> Arc<RealStore> {
        self.script.lock().unwrap().instances.values().nth(n).expect("launched").store.clone()
    }

    /// Records `call` with `arg`, waits while it is held, and plays its next fault: `Ok(true)` for a lost answer
    /// (or a hidden instance).
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
            Fault::LostAnswer | Fault::Hidden => Ok(true),
            Fault::Panic => panic!("the test cloud panics in {call:?}"),
        }
    }

    /// Shuts instance `r`'s worker down (its server answers `502` from now on, its store stops): terminated, or
    /// died.
    fn stop_instance(&self, r: &str) {
        let store = {
            let mut s = self.script.lock().unwrap();
            let Some(i) = s.instances.get_mut(r) else { return };
            i.gone.store(true, Ordering::SeqCst);
            let (live, store) = (i.live.take(), i.store.clone());
            s.stopped.extend(live);
            store
        };
        store.shutdown();
    }

    /// The `n`-th instance dies: its server and store stop, and the provider says it is gone.
    pub(super) fn kill(&self, n: usize) {
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
        let gone_flag = Arc::new(AtomicBool::new(false));
        let router = Server::new(store.clone(), options)
            .router()
            .layer(axum::middleware::from_fn_with_state(self.gates.clone(), gate))
            .layer(axum::middleware::from_fn_with_state((gone_flag.clone(), self.gates.clone()), gone));
        let live = Live::start(router);
        let instance = Instance {
            name: spec.name.clone(),
            store,
            url: live.url.clone(),
            live: Some(live),
            gone: gone_flag,
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
        if self.call(Call::Find, name)? {
            return Ok(Vec::new());
        }
        let s = self.script.lock().unwrap();
        let n = s.calls.iter().filter(|c| c.0 == Call::Find && c.1 == name).count();
        if s.find_fails.get(name).is_some_and(|f| f.contains(&n)) {
            return Err(folo_cloud::Error::new(ErrorKind::Unavailable, "the test cloud does not answer"));
        }
        let hidden = s.hidden.contains(name) || s.find_hides.get(name).is_some_and(|h| h.contains(&n));
        Ok(s.instances
            .iter()
            .filter(|(_, i)| i.name == name && i.live.is_some() && !hidden)
            .map(|(r, _)| r.clone())
            .collect())
    }

    fn terminate(&self, _: &Values, r: &str) -> folo_cloud::Result<()> {
        self.call(Call::Terminate, r)?;
        self.stop_instance(r);
        Ok(())
    }
}

/// Timings short enough for tests (a run copies its checkpoint after every epoch).
pub(super) fn fast() -> Tuning {
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
        max_preemption_retries: 3,
        preemption_backoff: Duration::from_millis(50),
        ..Tuning::default()
    }
}

/// A control plane on `home` with the test cloud.
pub(super) fn plane(home: &Path, cloud: &Arc<TestCloud>, tuning: Tuning) -> Arc<RealStore> {
    let providers: Vec<Arc<dyn Provider>> = vec![cloud.clone()];
    Arc::new(RealStore::open_with_cloud(home.to_path_buf(), providers, tuning).recheck_every(Duration::ZERO))
}

/// A 5-image synthetic dataset (64 px, two classes): 4 training images, 1 to validate on.
pub(super) fn dataset(store: &RealStore) -> Dataset {
    let req = SyntheticDataset {
        name: "tiny".into(),
        images: 5,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 5,
    };
    store.create_synthetic(req).unwrap()
}

/// A connection to the test cloud.
pub(super) fn connect(store: &RealStore) -> CloudConnection {
    let values = BTreeMap::from([("api_key".to_string(), SECRET.to_string())]);
    store.connect_cloud(ConnectCloud { provider: "test".into(), name: String::new(), values }).unwrap()
}

/// A tiny run of `epochs` on `connection`'s CPU offer (2 steps of 2 images an epoch).
pub(super) fn cloud_run(ds: &Dataset, connection: &str, epochs: u32) -> TrainConfig {
    serde_json::from_value(json!({
        "dataset_id": ds.id, "epochs": epochs, "batch": 2, "imgsz": 64, "augment": { "close_mosaic": 1 },
        "compute": { "connection_id": connection, "offer_id": "cpu@test" }
    }))
    .unwrap()
}

/// Polls `f` until it gives a value.
pub(super) fn until<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(t0.elapsed() < TIMEOUT, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

pub(super) fn run_of(store: &RealStore, id: &str) -> Run {
    store.get_run(id).unwrap()
}

/// Waits until run `id` ended (its final status is published after the release). A run that moves (its status,
/// step, epoch, queue place, phase, transfer, checkpoint copy or log) is waited for however long a loaded runner
/// takes: only one that stands still for [`TIMEOUT`] is a hang. (A fixed bound on the whole run failed eight
/// tests of one job on a busy runner, each of a run that was still training.)
pub(super) fn finished(store: &RealStore, id: &str) -> Run {
    let mut seen = None;
    let mut since = Instant::now();
    loop {
        let r = run_of(store, id);
        if !r.status.is_active() {
            return r;
        }
        let logged = store.run_logs(id, &LogsQuery::last(1)).unwrap().total;
        let cloud = r.cloud.as_ref().map(|c| (c.phase, c.progress, c.synced.clone()));
        let now = Some((r.status, r.step, r.epoch, r.queue_position, cloud, logged));
        if now != seen {
            (seen, since) = (now, Instant::now());
        }
        assert!(since.elapsed() < TIMEOUT, "timed out waiting for {id} to end: it stood still as {seen:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

pub(super) fn phase(r: &Run) -> CloudPhase {
    r.cloud.as_ref().expect("a cloud run").phase
}

pub(super) fn log_of(store: &RealStore, id: &str) -> Vec<LogLine> {
    store.run_logs(id, &LogsQuery::last(5000)).unwrap().lines
}

pub(super) fn logged(store: &RealStore, id: &str, part: &str) -> bool {
    log_of(store, id).iter().any(|l| l.message.contains(part))
}

/// Collects run `id`'s events until its final status.
pub(super) fn record(store: &RealStore, id: &str) -> std::thread::JoinHandle<Vec<RunEvent>> {
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

pub(super) fn steps(m: &RunMetrics) -> Vec<u64> {
    m.steps.iter().map(|s| s.step).collect()
}

pub(super) fn epochs(m: &RunMetrics) -> Vec<u32> {
    m.epochs.iter().map(|e| e.epoch).collect()
}

pub(super) fn resume(store: &RealStore, id: &str, extra: Value) -> Run {
    let fields = extra.as_object().unwrap().clone();
    store.resume_run(id, &fields).unwrap()
}
