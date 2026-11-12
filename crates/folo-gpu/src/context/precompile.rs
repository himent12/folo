//! Parallel precompile ([`GpuContext::precompile`]): the kernels an earlier run of the same kind recorded
//! (`crate::kernel_list`) are compiled on worker threads, in the order that run first used them, while the caller
//! goes on (opening a dataset, building the model, filling the loader). A thread that needs a kernel takes it
//! ready, waits for the worker compiling it, or compiles it itself when no worker has started it yet, so a
//! missing, stale or corrupt list only ever means compiling lazily, as without it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

use super::{GpuContext, Inner};
use crate::error::Result;
use crate::kernel_list::{self, Recording};
use crate::pipeline::{Kernel, KernelDef, KernelKey};

/// A context's precompile state: the kernels workers were asked to compile that no thread has taken yet, and the
/// recordings of the tags [`GpuContext::precompile`] started.
#[derive(Default)]
pub(crate) struct Precompile {
    slots: Mutex<HashMap<KernelKey, Arc<Slot>>>,
    recordings: Mutex<Vec<Arc<Recording>>>,
    /// First uses served by a worker's pipeline.
    taken: AtomicU64,
}

/// One kernel handed to the workers.
#[derive(Default)]
struct Slot {
    state: Mutex<SlotState>,
    done: Condvar,
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
                    return (None, false);
                }
                SlotState::Running => {
                    waited = true;
                    s = self.done.wait(s).unwrap_or_else(|p| p.into_inner());
                }
                SlotState::Claimed => return (None, waited),
                SlotState::Done(k) => return (k.clone(), waited),
            }
        }
    }
}

impl GpuContext {
    /// Starts recording the kernels this context compiles from now on as a run of `kind` at `shape`, and compiles
    /// on worker threads (one per core) the kernels an earlier process recorded for the same kind and shape on this
    /// adapter, in the order that process first used them, so the first steps of a run find their pipelines ready.
    /// With no list for `shape`, the most recently written list of `kind` is compiled instead (most kernels are
    /// shared). `kind` names what decides most kernels (`folo train` passes the command, model, precision, head,
    /// optimizer and class count), `shape` what decides the rest (image size and batch). Returns how many kernels
    /// went to the workers.
    ///
    /// Needs the pipeline cache folder ([`crate::set_pipeline_cache_dir`]): without one nothing is recorded or
    /// compiled ahead. A list is a hint: a kernel no worker has started when it is needed is compiled by the
    /// thread that needs it, and one a worker failed to build is compiled again there (which reports the error).
    /// A second call with the same kind and shape does nothing. Not available in the browser (no threads).
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
            if recs.iter().any(|r| r.tag == tag) {
                return 0;
            }
            let own = kernel_list::read(&path, &identity);
            recs.push(Recording::open(&tag, path.clone(), identity.clone(), &own));
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
    /// pipeline, else one compiled here; then it joins the map and every recording.
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
        let (kernel, new) = {
            let mut map = self.inner.kernels.lock().expect("kernel lock");
            match map.get(&key) {
                Some(k) => (k.clone(), false),
                None => {
                    map.insert(key.clone(), kernel.clone());
                    (kernel, true)
                }
            }
        };
        if new {
            for r in self.inner.precompile.recordings.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                r.used(&key);
            }
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
            let mut s = slot.state.lock().unwrap_or_else(|e| e.into_inner());
            while matches!(*s, SlotState::Queued | SlotState::Running) {
                s = slot.done.wait(s).unwrap_or_else(|e| e.into_inner());
            }
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
    /// `kernels::find_def`, or another definition under that name), for tests that every kernel can be precompiled.
    #[doc(hidden)]
    pub fn kernels_without_registry(&self) -> Vec<String> {
        let map = self.inner.kernels.lock().expect("kernel lock");
        let mut out: Vec<String> = map
            .iter()
            .filter(|(k, kernel)| {
                crate::kernels::find_def(k.name, &k.spec)
                    .is_none_or(|d| d.name != k.name || d.storage != kernel.storage)
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
