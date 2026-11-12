//! Parallel precompile ([`GpuContext::precompile`]): the kernels an earlier run of the same kind recorded
//! (`crate::kernel_list`) are compiled on worker threads, in the order that run first used them, while the caller
//! goes on (opening a dataset, building the model, filling the loader). A thread that needs a kernel takes it
//! ready, waits for the worker compiling it, or compiles it itself when no worker has started it yet, so a
//! missing, stale or corrupt list only ever means compiling lazily, as without it.
// Precompile and kernel lists are native only (no threads in the browser); in wasm32 builds most of this is unused.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports, unused_variables))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

use super::{GpuContext, Inner};
use crate::error::Result;
use crate::kernel_list::{self, Recording};
use crate::kernels::bounds::precompilable;
use crate::pipeline::{Kernel, KernelDef, KernelKey};

/// Recordings a context keeps (the most recently asked for; an older one is written and dropped). A training run
/// has one; `folo serve` one per model, precision and class count.
const MAX_RECORDINGS: usize = 16;

/// A context's precompile state: the kernels workers were asked to compile that no thread has taken yet, and the
/// recordings of the tags [`GpuContext::precompile`] started.
#[derive(Default)]
pub(crate) struct Precompile {
    slots: Mutex<HashMap<KernelKey, Arc<Slot>>>,
    /// The live recordings, least recently asked for first.
    recordings: Mutex<Vec<Arc<Recording>>>,
    /// Bumped (under the `recordings` lock) whenever a recording opens: a kernel whose `Kernel::recorded` is not
    /// this was not yet offered to every live recording, so its next use is ([`GpuContext::record_use`]).
    generation: AtomicU64,
    /// First uses served by a worker's pipeline.
    taken: AtomicU64,
}

/// One kernel handed to the workers.
#[derive(Default)]
struct Slot {
    state: Mutex<SlotState>,
    /// Signalled on every change of `state` that ends a wait (to `Claimed` or `Done`).
    done: Condvar,
    /// Threads waiting on `done` (tests).
    #[cfg(test)]
    waiters: AtomicUsize,
}

#[derive(Default)]
enum SlotState {
    /// No worker has started it.
    #[default]
    Queued,
    /// A worker is compiling it.
    Running,
    /// A requesting thread took it before any worker started it (and compiles it itself).
    Claimed,
    /// Compiled (`None`: it failed; the requesting thread compiles it again and reports the error).
    Done(Option<Arc<Kernel>>),
}

impl Slot {
    /// For a worker: whether it may compile the kernel (no requesting thread claimed it).
    fn start(&self) -> bool {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if matches!(*s, SlotState::Queued) {
            *s = SlotState::Running;
            true
        } else {
            false
        }
    }

    fn finish(&self, k: Option<Arc<Kernel>>) {
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = SlotState::Done(k);
        self.done.notify_all();
    }

    /// For the requesting thread: the kernel once the worker has it (waiting while it compiles), `None` when it
    /// failed or no worker had started it (then nobody will). The second value says whether it waited.
    fn take(&self) -> (Option<Arc<Kernel>>, bool) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let mut waited = false;
        loop {
            match &*s {
                SlotState::Queued => {
                    *s = SlotState::Claimed;
                    drop(s);
                    // Ends a `wait_settled` too: no worker will touch the slot now.
                    self.done.notify_all();
                    return (None, false);
                }
                SlotState::Running => {
                    waited = true;
                    s = self.wait(s);
                }
                SlotState::Claimed => return (None, waited),
                SlotState::Done(k) => return (k.clone(), waited),
            }
        }
    }

    /// Waits until no worker will touch the slot any more: it is done, or a requesting thread claimed it.
    fn wait_settled(&self) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        while matches!(*s, SlotState::Queued | SlotState::Running) {
            s = self.wait(s);
        }
    }

    fn wait<'a>(&self, s: MutexGuard<'a, SlotState>) -> MutexGuard<'a, SlotState> {
        #[cfg(test)]
        self.waiters.fetch_add(1, Ordering::SeqCst);
        let s = self.done.wait(s).unwrap_or_else(|p| p.into_inner());
        #[cfg(test)]
        self.waiters.fetch_sub(1, Ordering::SeqCst);
        s
    }

    /// Whether a worker finished it.
    fn is_done(&self) -> bool {
        matches!(*self.state.lock().unwrap_or_else(|p| p.into_inner()), SlotState::Done(_))
    }
}

impl GpuContext {
    /// Starts recording the kernels this context uses from now on as a run of `kind` at `shape`, and compiles on
    /// worker threads (one per core) the kernels an earlier process recorded for the same kind and shape on this
    /// adapter, in the order that process first used them, so the first steps of a run find their pipelines ready.
    /// With no list for `shape`, the most recently written list of `kind` is compiled instead (most kernels are
    /// shared). `kind` names what decides most kernels (`folo train` passes the command, model, precision, head,
    /// optimizer and class count), `shape` what decides the rest (image size and batch); `folo serve` passes an
    /// empty shape, so one list per kind serves every image size, batch and head it is asked for. Returns how many
    /// kernels went to the workers.
    ///
    /// A recording holds every kernel the context uses while it is open, in order of first use in it, also
    /// kernels the context compiled before it opened (a dispatch pays one relaxed load and compare for that). A
    /// context keeps the 16 most recently asked for recordings; an older one is written and dropped (asking for it
    /// again starts it again from its file).
    ///
    /// Needs the pipeline cache folder ([`crate::set_pipeline_cache_dir`]): without one nothing is recorded or
    /// compiled ahead. A list is a hint: a kernel no worker has started when it is needed is compiled by the
    /// thread that needs it, and one a worker failed to build is compiled again there (which reports the error).
    /// A list entry whose constants no call site sets is skipped (`kernels::bounds`). A second call with the same
    /// kind and shape (while its recording is kept) does nothing. Not available in the browser (no threads).
    pub fn precompile(&self, kind: &str, shape: &str) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (kind, shape);
            0
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.precompile_native(kind, shape)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn precompile_native(&self, kind: &str, shape: &str) -> usize {
        let Some(dir) = self.inner.cache_dir.as_deref() else { return 0 };
        let info = self.inner.adapter.get_info();
        let tag = format!("{kind} | {shape}");
        let path = kernel_list::path_for(dir, &info, kind, shape);
        let identity = kernel_list::identity_for(&info, kind);
        let p = &self.inner.precompile;
        let list = {
            let mut recs = p.recordings.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = recs.iter().position(|r| r.tag == tag) {
                let r = recs.remove(i);
                recs.push(r);
                return 0;
            }
            let own = kernel_list::read(&path, &identity);
            recs.push(Recording::open(&tag, path.clone(), identity.clone(), &own));
            p.generation.fetch_add(1, Ordering::Relaxed);
            // Recordings beyond `MAX_RECORDINGS` are written one last time and retired under this lock, so a
            // reopening of the same list (which takes it) reads their file and is never overwritten by them.
            let over = recs.len().saturating_sub(MAX_RECORDINGS);
            for r in recs.drain(..over) {
                r.retire();
            }
            if own.is_empty() {
                kernel_list::neighbours(dir, &info, kind, &path)
                    .iter()
                    .map(|n| kernel_list::read(n, &identity))
                    .find(|l| !l.is_empty())
                    .unwrap_or_default()
            } else {
                own
            }
        };
        let jobs: Vec<(KernelKey, &'static KernelDef, Arc<Slot>)> = {
            let compiled = self.inner.kernels.lock().expect("kernel lock");
            let mut slots = p.slots.lock().unwrap_or_else(|e| e.into_inner());
            let mut jobs = Vec::new();
            for k in list {
                if !self.can_build(&k) || compiled.contains_key(&k) || slots.contains_key(&k) {
                    continue;
                }
                let Some(def) = crate::kernels::find_def(k.name, &k.spec) else { continue };
                if !precompilable(def.name, &k.spec) {
                    continue;
                }
                let slot = Arc::new(Slot::default());
                slots.insert(k.clone(), slot.clone());
                jobs.push((k, def, slot));
            }
            jobs
        };
        let n = jobs.len();
        if n > 0 {
            spawn_workers(Arc::downgrade(&self.inner), Arc::new(jobs));
        }
        n
    }

    /// Whether this context can build `key` at all (f16 and tensor-core kernels need their device features).
    fn can_build(&self, key: &KernelKey) -> bool {
        (!key.spec.is_f16() || self.inner.caps.shader_f16) && (!key.spec.is_coop() || self.inner.experimental)
    }

    /// The first use of `key` in this context (its pipeline was not in the context's map): the precompiled
    /// pipeline, else one compiled here; then it joins the map and the live recordings.
    pub(super) fn first_use(&self, def: &'static KernelDef, key: KernelKey) -> Result<Arc<Kernel>> {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = crate::pipeline::trace::start();
        let slot = self.inner.precompile.slots.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
        let (ready, waited) = match slot.map(|s| s.take()) {
            // The same layout as `def` (a kernel found by name is the same definition; this only guards that).
            Some((Some(k), w)) if k.storage == def.storage => (Some(k), w),
            Some((_, w)) => (None, w),
            None => (None, false),
        };
        let precompiled = ready.is_some();
        if precompiled {
            self.inner.precompile.taken.fetch_add(1, Ordering::Relaxed);
        }
        let kernel = match ready {
            Some(k) => k,
            None => {
                let disk = self.inner.pipeline_cache.as_deref();
                let k =
                    Arc::new(Kernel::compile(&self.inner.device, def, &key.spec, disk.map(|d| &d.cache))?);
                if let Some(d) = disk {
                    d.compiled();
                }
                k
            }
        };
        let kernel =
            self.inner.kernels.lock().expect("kernel lock").entry(key.clone()).or_insert(kernel).clone();
        if !self.recorded(&kernel) {
            self.record_use(&key, &kernel);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t0) = t0 {
            use crate::pipeline::trace::FirstUse;
            let how = match (precompiled, waited) {
                (true, false) => FirstUse::Ready,
                (true, true) => FirstUse::Waited,
                (false, _) => FirstUse::Compiled,
            };
            crate::pipeline::trace::first_use(t0, how);
        }
        Ok(kernel)
    }

    /// Whether `kernel` was offered to the recordings at the current generation (the dispatch fast path: two
    /// relaxed loads).
    #[inline]
    pub(super) fn recorded(&self, kernel: &Kernel) -> bool {
        kernel.recorded.load(Ordering::Relaxed) == self.inner.precompile.generation.load(Ordering::Relaxed)
    }

    /// A use of `kernel` (as `key`) at another recording generation than the one it was last offered to the
    /// recordings at: every live recording adds it (one that holds it already skips it), and the kernel takes the
    /// generation, so its next uses stay on the fast path until another recording opens.
    pub(super) fn record_use(&self, key: &KernelKey, kernel: &Kernel) {
        let p = &self.inner.precompile;
        let recs = p.recordings.lock().unwrap_or_else(|e| e.into_inner());
        for r in recs.iter() {
            r.used(key);
        }
        kernel.recorded.store(p.generation.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Drops the pipelines precompile workers finished that no thread has taken (a list of another shape names
    /// kernels this run never uses); kernels still queued or compiling stay, and whoever needs one later
    /// compiles it ([`GpuContext::release_memory`]).
    pub(super) fn drop_finished_precompiles(&self) {
        self.inner.precompile.slots.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, s| !s.is_done());
    }

    /// The kernels the live recording for `kind` and `shape` holds, by name, in order (tests).
    #[cfg(test)]
    pub(crate) fn recorded_kernels(&self, kind: &str, shape: &str) -> Vec<&'static str> {
        let tag = format!("{kind} | {shape}");
        let recs = self.inner.precompile.recordings.lock().unwrap_or_else(|e| e.into_inner());
        recs.iter()
            .find(|r| r.tag == tag)
            .map_or_else(Vec::new, |r| r.keys().iter().map(|k| k.name).collect())
    }

    /// Precompiled pipelines no thread has taken yet, finished or not (tests).
    #[cfg(test)]
    pub(crate) fn precompile_slots(&self) -> usize {
        self.inner.precompile.slots.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// The list file of the recording for `kind` and `shape`, once [`GpuContext::precompile`] started one (tests).
    #[doc(hidden)]
    pub fn kernel_list_file(&self, kind: &str, shape: &str) -> Option<std::path::PathBuf> {
        let tag = format!("{kind} | {shape}");
        let recs = self.inner.precompile.recordings.lock().unwrap_or_else(|e| e.into_inner());
        recs.iter().find(|r| r.tag == tag).map(|r| r.path().to_path_buf())
    }

    /// Waits until the precompile workers have finished every kernel no thread has taken yet (tests).
    #[doc(hidden)]
    pub fn precompile_wait(&self) {
        let slots: Vec<Arc<Slot>> =
            self.inner.precompile.slots.lock().unwrap_or_else(|e| e.into_inner()).values().cloned().collect();
        for slot in slots {
            slot.wait_settled();
        }
    }

    /// How many first uses of a kernel in this context took a precompile worker's pipeline (tests).
    #[doc(hidden)]
    pub fn precompiled_taken(&self) -> u64 {
        self.inner.precompile.taken.load(Ordering::Relaxed)
    }

    /// Writes the recordings that hold kernels their files do not have, now (tests; `folo` calls
    /// [`crate::save_pipeline_caches`]).
    #[doc(hidden)]
    pub fn save_kernel_lists(&self) {
        for r in self.inner.precompile.recordings.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            r.save();
        }
    }

    /// The names of kernels first used in this context that a list could not name again (no entry in
    /// `kernels::find_def`, another definition under that name, or constants outside `kernels::bounds`), for tests
    /// that every kernel can be precompiled.
    #[doc(hidden)]
    pub fn kernels_without_registry(&self) -> Vec<String> {
        let map = self.inner.kernels.lock().expect("kernel lock");
        let mut out: Vec<String> = map
            .iter()
            .filter(|(k, kernel)| {
                crate::kernels::find_def(k.name, &k.spec).is_none_or(|d| {
                    d.name != k.name || d.storage != kernel.storage || !precompilable(d.name, &k.spec)
                })
            })
            .map(|(k, _)| k.name.to_string())
            .collect();
        out.sort();
        out.dedup();
        out
    }
}

/// Compiles `jobs` in order on one thread per core (fewer for a short list). A worker stops when the context is
/// gone; a kernel that panics while it is built (a list from another build asking for constants the generator
/// does not expect) counts as failed.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_workers(inner: Weak<Inner>, jobs: Arc<Vec<(KernelKey, &'static KernelDef, Arc<Slot>)>>) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(jobs.len());
    let next = Arc::new(AtomicUsize::new(0));
    for i in 0..threads {
        let (inner, jobs2, next) = (inner.clone(), jobs.clone(), next.clone());
        let work = move || {
            let jobs = jobs2;
            loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                let Some((key, def, slot)) = jobs.get(j) else { return };
                if !slot.start() {
                    continue;
                }
                let Some(inner) = inner.upgrade() else {
                    slot.finish(None);
                    // Release the slots no worker will reach.
                    for (_, _, s) in &jobs[j + 1..] {
                        if s.start() {
                            s.finish(None);
                        }
                    }
                    return;
                };
                let disk = inner.pipeline_cache.as_deref();
                let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Kernel::compile(&inner.device, def, &key.spec, disk.map(|d| &d.cache))
                }));
                let k = match built {
                    Ok(Ok(k)) => {
                        if let Some(d) = disk {
                            d.compiled();
                        }
                        Some(Arc::new(k))
                    }
                    _ => None,
                };
                slot.finish(k);
            }
        };
        let spawned = std::thread::Builder::new().name(format!("folo-gpu-precompile-{i}")).spawn(work);
        if spawned.is_err() && i == 0 {
            // No thread at all: nothing will compile ahead, so release every slot.
            for (_, _, s) in jobs.iter() {
                if s.start() {
                    s.finish(None);
                }
            }
            return;
        }
    }
}

/// The slot hand-off between a worker, the thread that needs the kernel and `precompile_wait`, step by step: each
/// test parks a thread in a wait, then makes the transition that must wake it.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Runs `f` on a new thread once it is parked on `slot`'s condition variable, then `wake`, and returns what
    /// `f` returned (failing instead of hanging when `wake` does not wake it).
    fn parked<T: Send + 'static>(
        slot: &Arc<Slot>,
        f: impl FnOnce(&Slot) -> T + Send + 'static,
        wake: impl FnOnce(&Slot),
    ) -> T {
        let (tx, rx) = mpsc::channel();
        let s = slot.clone();
        let t = std::thread::spawn(move || tx.send(f(&s)).unwrap());
        while slot.waiters.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        wake(slot);
        let got = rx.recv_timeout(Duration::from_secs(30)).expect("the waiting thread was woken");
        t.join().unwrap();
        got
    }

    #[test]
    fn a_taker_waits_for_the_running_worker() {
        let slot = Arc::new(Slot::default());
        assert!(slot.start(), "a worker starts a queued slot");
        let (k, waited) = parked(&slot, Slot::take, |s| s.finish(None));
        assert!(k.is_none() && waited, "the worker's (failed) result, after waiting");
        assert!(!slot.start(), "a finished slot does not start again");
    }

    #[test]
    fn a_claim_wakes_precompile_wait() {
        let slot = Arc::new(Slot::default());
        parked(&slot, Slot::wait_settled, |s| assert!(!s.take().1, "claimed without waiting"));
        assert!(!slot.start(), "no worker compiles a claimed slot");
        assert!(matches!(slot.take(), (None, false)), "a claimed slot stays the claimer's");
    }

    #[test]
    fn precompile_wait_waits_for_the_worker() {
        let slot = Arc::new(Slot::default());
        assert!(slot.start());
        parked(&slot, Slot::wait_settled, |s| s.finish(None));
        assert!(slot.is_done());
    }
}
