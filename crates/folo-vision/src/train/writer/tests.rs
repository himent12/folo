//! The checkpoint writer: files on disk before their event, events in the order of a run that writes inline,
//! failed writes, a trainer dropped mid-write, pauses at the boundary, and the snapshot the writer is handed.
//!
//! A [`Gate`] holds a chosen epoch's write pending until the training thread has found it pending a number of times
//! (so the next epoch's steps provably ran during the write), and [`inline`] makes a run write on the training
//! thread, as main did, for the reference event stream.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use folo_api::RunEvent;
use folo_test_support::TestDir;
use serde_json::Value;

use super::super::tests::{micro, micro_on};
use super::super::{EpochHook, TrainOptions, TrainSummary, event_record};
use super::EpochFiles;
use crate::error::Result;

/// The thread that wrote each `last.folo`.
static WRITERS: Mutex<Vec<(PathBuf, ThreadId)>> = Mutex::new(Vec::new());

/// Run folders written inline, and the gates of chosen writes: `(run folder, epoch, gate)`.
static INLINE: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
static GATES: Mutex<Vec<(PathBuf, u32, Arc<Gate>)>> = Mutex::new(Vec::new());

/// The longest a [`Gate`] holds a write (it then lets it go and says so): only reached when the test fails.
const GATE_TIMEOUT: Duration = Duration::from_secs(10);

/// Holds a write pending until the training thread found it pending `polls` times or the trainer was dropped.
pub(super) struct Gate {
    polls: u32,
    state: Mutex<GateState>,
    cv: Condvar,
}

#[derive(Clone, Copy, Debug, Default)]
struct GateState {
    /// Steps that found the write pending.
    polled: u32,
    /// The trainer was dropped with the write in flight.
    dropped: bool,
    /// The gate let the write go at [`GATE_TIMEOUT`].
    timed_out: bool,
}

impl Gate {
    pub(super) fn polled(&self) {
        self.state.lock().unwrap().polled += 1;
        self.cv.notify_all();
    }

    pub(super) fn dropped(&self) {
        self.state.lock().unwrap().dropped = true;
        self.cv.notify_all();
    }

    fn wait(&self) {
        let deadline = Instant::now() + GATE_TIMEOUT;
        let mut s = self.state.lock().unwrap();
        while s.polled < self.polls && !s.dropped {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                s.timed_out = true;
                break;
            }
            s = self.cv.wait_timeout(s, left).unwrap().0;
        }
    }

    fn state(&self) -> GateState {
        *self.state.lock().unwrap()
    }
}

/// Holds epoch `epoch`'s write in the run folder `dir` until `polls` steps found it pending.
fn gate(dir: &Path, epoch: u32, polls: u32) -> Arc<Gate> {
    let g = Arc::new(Gate { polls, state: Mutex::default(), cv: Condvar::new() });
    GATES.lock().unwrap().push((dir.to_path_buf(), epoch, g.clone()));
    g
}

/// Every write in the run folder `dir` happens on the training thread.
fn inline(dir: &Path) {
    INLINE.lock().unwrap().push(dir.to_path_buf());
}

/// The knobs of epoch `epoch`'s write of `last` (`<run folder>/weights/last.folo`): inline, and its gate.
pub(super) fn knobs(last: &Path, epoch: u32) -> (bool, Option<Arc<Gate>>) {
    let Some(dir) = last.parent().and_then(Path::parent) else { return (false, None) };
    let inline = INLINE.lock().unwrap().iter().any(|d| d == dir);
    let mut gates = GATES.lock().unwrap();
    let gate = gates.iter().position(|(d, e, _)| d == dir && *e == epoch).map(|i| gates.remove(i).2);
    (inline, gate)
}

/// The writer's start: records its thread and waits at the write's gate.
pub(super) fn wrote(f: &EpochFiles) {
    WRITERS.lock().unwrap().push((f.last.clone(), std::thread::current().id()));
    if let Some(g) = &f.gate {
        g.wait();
    }
}

/// An event record as compared between runs: without its times, and the numbers of log messages masked (they
/// carry durations).
fn canonical(mut v: Value) -> Value {
    let o = v.as_object_mut().unwrap();
    for k in ["imgs_per_s", "eta_s", "duration_s", "t"] {
        o.remove(k);
    }
    if let Some(Value::String(m)) = o.get_mut("message") {
        let mut masked = String::with_capacity(m.len());
        for c in m.chars() {
            match (c.is_ascii_digit(), masked.ends_with('#')) {
                (true, true) => {}
                (true, false) => masked.push('#'),
                (false, _) => masked.push(c),
            }
        }
        *m = masked;
    }
    v
}

/// `step <epoch>/<step>`, `epoch <n>`, `status <s>` or `log`.
fn label(v: &Value) -> String {
    match v["event"].as_str().unwrap() {
        "step" => format!("step {}/{}", v["epoch"], v["step"]),
        "epoch" => format!("epoch {}", v["epoch"]),
        "status" => format!("status {}", v["status"].as_str().unwrap()),
        other => other.to_string(),
    }
}

/// Called with each event before it is recorded.
type Probe = Box<dyn FnMut(&RunEvent) + Send>;

/// A 3-epoch (2 steps each) micro run in `dir` with every step delivered, its events ([`canonical`]) in order.
/// At every `epoch` event it checks the promise of docs/API.md: that epoch's files are on disk.
fn run_in(
    dir: &Path,
    hook: Option<Box<dyn EpochHook>>,
    mut probe: Option<Probe>,
) -> (Result<TrainSummary>, Vec<Value>) {
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let (log, rd) = (seen.clone(), dir.to_path_buf());
    let sink = move |e: &RunEvent| {
        if let RunEvent::Epoch(ep) = e {
            let f = folo_nn::io::ModelFile::open(rd.join("weights/last.folo")).unwrap();
            assert_eq!(f.checkpoint().unwrap().epoch, u64::from(ep.epoch));
            assert!(rd.join(format!("samples/epoch_{:03}.json", ep.epoch)).is_file());
        }
        if let Some(p) = &mut probe {
            p(e);
        }
        log.lock().unwrap().push(canonical(event_record(e)));
    };
    let opts = TrainOptions {
        run_dir: Some(dir.to_path_buf()),
        events: Some(Box::new(sink)),
        step_interval: Duration::ZERO,
        after_epoch: hook,
        ..TrainOptions::default()
    };
    let r = micro(3, 100, opts).run();
    let seen = seen.lock().unwrap().clone();
    (r, seen)
}

/// The events of [`run_in`] with every write on the training thread (main's behaviour), run once.
fn reference() -> &'static [Value] {
    static EVENTS: OnceLock<Vec<Value>> = OnceLock::new();
    EVENTS.get_or_init(|| {
        let dir = TestDir::new("train-writer-inline");
        inline(&dir);
        let (r, seen) = run_in(&dir, None, None);
        assert_eq!(r.unwrap().epochs_completed, 3);
        let writers = WRITERS.lock().unwrap();
        let mine = writers.iter().filter(|(p, _)| p.starts_with(&dir));
        assert!(
            mine.clone().count() == 3 && mine.into_iter().all(|(_, t)| *t == std::thread::current().id())
        );
        seen
    })
}

/// The reference events before the one labelled `upto`.
fn reference_before(upto: &str) -> &'static [Value] {
    let r = reference();
    &r[..r.iter().position(|v| label(v) == upto).unwrap()]
}

/// A run's failure as its last two events: the error log and the `failed` status.
fn assert_failed(seen: &[Value], code: &str) {
    let [log, status] = &seen[seen.len() - 2..] else { unreachable!() };
    assert_eq!((label(log), label(status)), ("log".to_string(), "status failed".to_string()), "{seen:#?}");
    let msg = log["message"].as_str().unwrap();
    assert!(msg.starts_with(&format!("run failed ({code})")), "{msg}");
}

/// The epoch's files are written on another thread while the next epoch trains (epoch 2's steps ran during epoch
/// 1's write), are on disk when the epoch event arrives, and the events (also in `metrics.jsonl`) are those of a
/// run that writes inline, in the same order.
#[test]
fn epoch_files_are_written_while_the_next_epoch_trains() {
    let want = reference();
    let dir = TestDir::new("train-writer");
    let g = gate(&dir, 1, 2);
    let (probe_gate, polled_at_epoch_1) = (g.clone(), Arc::new(AtomicU32::new(0)));
    let at = polled_at_epoch_1.clone();
    let probe: Probe = Box::new(move |e| {
        if matches!(e, RunEvent::Epoch(ep) if ep.epoch == 1) {
            at.store(probe_gate.state().polled, Ordering::SeqCst);
        }
    });
    let (r, seen) = run_in(&dir, None, Some(probe));
    assert_eq!(r.unwrap().epochs_completed, 3);
    let s = g.state();
    assert!(!s.timed_out, "{s:?}");
    assert_eq!(polled_at_epoch_1.load(Ordering::SeqCst), 2, "both epoch-2 steps ran during epoch 1's write");
    assert_eq!(seen, want, "the events differ from those of a run that writes inline");
    let labels: Vec<String> = seen.iter().map(label).filter(|l| l != "log").collect();
    let order = [
        "status running",
        "step 1/1",
        "step 1/2",
        "epoch 1",
        "step 2/3",
        "step 2/4",
        "epoch 2",
        "step 3/5",
        "step 3/6",
        "epoch 3",
        "status completed",
    ];
    assert_eq!(labels, order);
    let text = std::fs::read_to_string(dir.join("metrics.jsonl")).unwrap();
    let file: Vec<Value> = text.lines().map(|l| canonical(serde_json::from_str(l).unwrap())).collect();
    assert_eq!(file, seen, "metrics.jsonl holds the delivered events, in their order");
    let writers = WRITERS.lock().unwrap();
    let mine: Vec<_> = writers.iter().filter(|(p, _)| p.starts_with(&dir)).collect();
    assert_eq!(mine.len(), 3);
    assert!(mine.iter().all(|(_, t)| *t != std::thread::current().id()), "written on a writer thread");
}

/// A write that fails while the next epoch's events are held fails the run as if at the epoch's end: the events
/// up to that epoch's, then the failure; no epoch event, the held steps dropped.
#[test]
fn a_failed_write_fails_the_run_without_its_epoch_event() {
    let dir = TestDir::new("train-writer-fail");
    // `last.folo` is a non-empty folder: renaming the written file over it fails.
    std::fs::create_dir_all(dir.join("weights/last.folo/x")).unwrap();
    let g = gate(&dir, 1, 2);
    let (r, seen) = run_in(&dir, None, None);
    assert_eq!(r.unwrap_err().code(), "io_error");
    let s = g.state();
    assert!(s.polled >= 2 && !s.timed_out, "the write failed with epoch 2's steps held: {s:?}");
    assert_eq!(&seen[..seen.len() - 2], reference_before("epoch 1"));
    assert_failed(&seen, "io_error");
}

/// After an epoch written in full, a write whose samples fail (its `last.folo` written) fails the run at that
/// epoch: epoch 1's event and epoch 2's steps arrive, epoch 2's event and epoch 3's steps do not. The samples file
/// is written atomically: no temporary file is left.
#[test]
fn a_failed_samples_write_after_a_good_epoch_fails_the_run_at_its_epoch() {
    let dir = TestDir::new("train-writer-fail-samples");
    std::fs::create_dir_all(dir.join("samples/epoch_002.json/x")).unwrap();
    let g = gate(&dir, 2, 2);
    let (r, seen) = run_in(&dir, None, None);
    assert_eq!(r.unwrap_err().code(), "io_error");
    let s = g.state();
    assert!(s.polled >= 2 && !s.timed_out, "{s:?}");
    assert_eq!(&seen[..seen.len() - 2], reference_before("epoch 2"));
    assert_failed(&seen, "io_error");
    let f = folo_nn::io::ModelFile::open(dir.join("weights/last.folo")).unwrap();
    assert_eq!(f.checkpoint().unwrap().epoch, 2, "last.folo was written before the samples");
    let mut names: Vec<String> = std::fs::read_dir(dir.join("samples"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["epoch_001.json", "epoch_002.json"]);
}

/// A trainer dropped with a write in flight (here a panic at the epoch boundary unwinds through `run`) waits for
/// the writer, so nothing writes into the run folder afterwards, and delivers no event.
#[test]
fn a_dropped_trainer_joins_its_writer() {
    struct Panics;
    impl EpochHook for Panics {
        fn pauses(&mut self, _: u32) -> bool {
            panic!("test: unwinding with a checkpoint write in flight")
        }
        fn pause(&mut self, _: u32, _: &AtomicBool) {}
    }
    let dir = TestDir::new("train-writer-drop");
    // Only the drop releases it.
    let g = gate(&dir, 1, u32::MAX);
    let epochs = Arc::new(AtomicU32::new(0));
    let n = epochs.clone();
    let probe: Probe = Box::new(move |e| {
        if matches!(e, RunEvent::Epoch(_)) {
            n.fetch_add(1, Ordering::SeqCst);
        }
    });
    let unwound = catch_unwind(AssertUnwindSafe(|| run_in(&dir, Some(Box::new(Panics)), Some(probe))));
    assert!(unwound.is_err());
    let s = g.state();
    assert!(s.dropped && !s.timed_out, "{s:?}");
    // Written by the time the drop returned.
    let f = folo_nn::io::ModelFile::open(dir.join("weights/last.folo")).unwrap();
    assert_eq!(f.checkpoint().unwrap().epoch, 1);
    assert!(dir.join("samples/epoch_001.json").is_file());
    assert_eq!(epochs.load(Ordering::SeqCst), 0, "no epoch event from a dropped trainer");
}

/// Records each boundary it is asked about; pauses after `pause_at`, checks there that the epoch's event was
/// delivered, and stops the run.
struct Hook {
    asked: Arc<Mutex<Vec<u32>>>,
    pause_at: Option<u32>,
    /// The last epoch event delivered (set by the run's probe).
    delivered: Arc<AtomicU32>,
    /// `(epoch, last epoch event delivered)` at each pause.
    paused: Arc<Mutex<Vec<(u32, u32)>>>,
}

impl EpochHook for Hook {
    fn pauses(&mut self, epoch: u32) -> bool {
        self.asked.lock().unwrap().push(epoch);
        self.pause_at == Some(epoch)
    }

    fn pause(&mut self, epoch: u32, stop: &AtomicBool) {
        self.paused.lock().unwrap().push((epoch, self.delivered.load(Ordering::SeqCst)));
        stop.store(true, Ordering::SeqCst);
    }
}

/// A boundary where `after_epoch` does not pause does not wait for the write (epoch 2's steps run during epoch 1's
/// write); one where it pauses waits, and the pause comes after the epoch's event.
#[test]
fn only_a_pause_waits_for_the_write() {
    let run = |pause_at: Option<u32>, dir: &Path| {
        let delivered = Arc::new(AtomicU32::new(0));
        let hook =
            Hook { asked: Arc::default(), pause_at, delivered: delivered.clone(), paused: Arc::default() };
        let (asked, paused) = (hook.asked.clone(), hook.paused.clone());
        let probe: Probe = Box::new(move |e| {
            if let RunEvent::Epoch(ep) = e {
                delivered.store(ep.epoch, Ordering::SeqCst);
            }
        });
        let (r, seen) = run_in(dir, Some(Box::new(hook)), Some(probe));
        let (asked, paused) = (asked.lock().unwrap().clone(), paused.lock().unwrap().clone());
        (r.unwrap(), seen, asked, paused)
    };
    let dir = TestDir::new("train-writer-hook");
    let g = gate(&dir, 1, 2);
    let (s, seen, asked, paused) = run(None, &dir);
    assert_eq!(s.epochs_completed, 3);
    assert!(!g.state().timed_out, "the boundary waited for the write: {:?}", g.state());
    assert_eq!((asked, paused), (vec![1, 2, 3], vec![]));
    assert_eq!(seen, reference());

    let dir = TestDir::new("train-writer-pause");
    let (s, seen, asked, paused) = run(Some(1), &dir);
    assert_eq!((s.status, s.epochs_completed), (folo_api::RunStatus::Stopped, 1));
    assert_eq!((asked, paused), (vec![1], vec![(1, 1)]));
    assert_eq!(&seen[..seen.len() - 2], reference_before("step 2/3"));
}

/// The snapshot owns its tensors: training on (which updates the live weights, BatchNorm statistics and optimizer
/// state in place) leaves it as it was, so the writer thread writes the boundary's state. Before the first epoch
/// and after one (with optimizer state), on the CPU and, when there is one, on the GPU.
fn the_snapshot_does_not_change_when_training_goes_on(device: &str) {
    let bytes = |t: &[(String, folo_core::Tensor)]| -> Vec<(String, Vec<u8>)> {
        t.iter().map(|(n, x)| (n.clone(), x.to_cpu_storage().unwrap().as_bytes().to_vec())).collect()
    };
    let mut t = micro_on(device, 1, 100, TrainOptions::default());
    let (_, at_start) = t.checkpoint_snapshot().unwrap();
    let start = bytes(&at_start);
    let (_, fresh) = micro_on(device, 1, 100, TrainOptions::default()).checkpoint_snapshot().unwrap();
    assert_eq!(bytes(&fresh), start);
    t.run().unwrap();
    let (_, after_one) = t.checkpoint_snapshot().unwrap();
    let one = bytes(&after_one);
    let optim = |t: &[(String, Vec<u8>)]| -> Vec<Vec<u8>> {
        t.iter().filter(|(n, _)| n.starts_with("optim.")).map(|(_, b)| b.clone()).collect()
    };
    assert!(!optim(&one).is_empty() && optim(&one) != optim(&start), "optimizer state after an epoch");
    t.cfg.epochs = 2;
    t.run().unwrap();
    assert_eq!(t.epoch, 2);
    assert_eq!(bytes(&at_start), start, "training changed the snapshot taken before it");
    assert_eq!(bytes(&after_one), one, "training changed the snapshot taken after epoch 1");
    let (_, now) = t.checkpoint_snapshot().unwrap();
    assert_ne!(bytes(&now), one, "the run trained");
}

#[test]
fn the_snapshot_does_not_change_when_training_goes_on_cpu() {
    the_snapshot_does_not_change_when_training_goes_on("cpu");
}

#[test]
fn the_snapshot_does_not_change_when_training_goes_on_gpu() {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping GPU test, no adapter: {e}"),
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        return eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
    }
    folo_gpu::init().unwrap();
    the_snapshot_does_not_change_when_training_goes_on("gpu:0");
}

/// The epoch's `last.folo` holds the same bytes as `save_checkpoint` at that boundary (but the creation time).
#[test]
fn epoch_checkpoint_matches_save_checkpoint() {
    let dir = TestDir::new("train-writer-bytes");
    let opts = TrainOptions { run_dir: Some(dir.to_path_buf()), ..TrainOptions::default() };
    let mut t = micro(2, 100, opts);
    t.run().unwrap();
    let other = dir.join("again.folo");
    t.save_checkpoint(&other).unwrap();
    let blank = |mut b: Vec<u8>| {
        let key = b"folo.created";
        if let Some(i) = b.windows(key.len()).position(|w| w == key) {
            b[i..i + key.len() + 26].fill(0);
        }
        b
    };
    let (a, b) = (std::fs::read(dir.join("weights/last.folo")).unwrap(), std::fs::read(&other).unwrap());
    assert_eq!(a.len(), b.len());
    assert!(blank(a) == blank(b), "the epoch checkpoint differs from save_checkpoint's");
}

/// O(1) hardlink promotion of `best.folo`: an earlier best epoch stays intact after a later epoch overwrites
/// `last.folo`, a new best atomically replaces `best.folo`, and no temporary link files are left behind.
#[test]
fn promote_best_preserves_prior_best_when_last_overwritten() {
    let dir = TestDir::new("train-writer-hardlink");
    let weights = dir.join("weights");
    std::fs::create_dir_all(&weights).unwrap();
    let last = weights.join("last.folo");
    let best = weights.join("best.folo");

    let write_last = |val: f32| {
        let t = folo_core::Tensor::from_slice(&[val, val + 1.0], [2]).unwrap();
        folo_nn::io::safetensors::write_file(&last, None, &[("w".into(), t)]).unwrap();
    };
    let read_w = |p: &Path| {
        let bytes = std::fs::read(p).unwrap();
        let c = folo_nn::io::safetensors::Container::parse(&bytes).unwrap();
        folo_nn::io::safetensors::decode(&c.tensors[0], c.data(&bytes, &c.tensors[0]))
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    };

    // Epoch 1: improved -> best promoted from last (val = 1.0).
    write_last(1.0);
    super::promote_best(&last, &best).unwrap();
    assert_eq!(read_w(&last), vec![1.0, 2.0]);
    assert_eq!(read_w(&best), vec![1.0, 2.0]);

    // Epoch 2: not improved -> last overwritten via atomic rename, best must retain epoch 1's exact bytes.
    write_last(2.0);
    assert_eq!(read_w(&last), vec![2.0, 3.0]);
    assert_eq!(read_w(&best), vec![1.0, 2.0]);

    // Epoch 3: improved -> best atomically updated to epoch 3, no leftover `.tmp` entries.
    write_last(3.0);
    super::promote_best(&last, &best).unwrap();
    assert_eq!(read_w(&best), vec![3.0, 4.0]);
    let temps: Vec<_> = std::fs::read_dir(&weights)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
        .collect();
    assert!(temps.is_empty(), "leftover temp files in weights/: {temps:?}");
}
