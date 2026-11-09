//! The fork-join pool of the CPU kernels: `for_each(n, threads, f)` runs `f(0) .. f(n - 1)` on the calling
//! thread and up to `threads - 1` workers.
//!
//! A forward pass is a few hundred parallel regions of 20 µs to a millisecond with a few microseconds of serial
//! work between them. A work-stealing pool whose idle workers go to sleep (rayon's do after about thirty
//! `yield`s) pays a wake-up per worker per region, one after another as the work is split, which is as long as
//! the region itself for the small layers. This pool is built for that pattern, like an OpenMP runtime:
//!
//! - idle workers spin on one shared word for [`spin_window`] after their last task before they park, so a
//!   region starts on every thread within a fraction of a microsecond while a pass is running;
//! - a region is one shared counter: every participant claims the next task index until none is left, so a
//!   faster core (or one that was not interrupted) simply takes more tasks, and there is no splitting, queueing
//!   or per-task allocation;
//! - a region for `threads` threads uses the calling thread and workers `0 .. threads - 1`, always the same
//!   ones, so a small layer keeps running on the same few (warm) cores.
//!
//! The pool runs one region at a time. A region started from inside a task runs on that thread alone, and one
//! started while another thread's region is running falls back to rayon, so every call completes whatever
//! else is going on. Which thread runs which task is not deterministic; kernels give each task its own outputs,
//! so results never depend on it.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;

/// Most workers the pool ever spawns.
const MAX_WORKERS: usize = 255;

/// A value on its own cache line, so the words every thread polls do not share one with the words they write.
#[repr(align(128))]
struct Padded<T>(T);

/// What a worker shares with the thread that starts regions.
struct Slot {
    /// True from just before the worker parks until it runs again.
    sleeping: AtomicBool,
    thread: OnceLock<Thread>,
}

/// The task of the open region, with its lifetime erased.
type Task = *const (dyn Fn(usize) + Sync);

struct Pool {
    /// Taken by the thread that runs a region, from before it writes the job until every worker has left it.
    busy: Padded<AtomicBool>,
    /// Odd while a region is open. Workers poll it.
    epoch: Padded<AtomicU64>,
    /// Workers that may be reading the job (see [`worker`]).
    active: Padded<AtomicUsize>,
    /// The next unclaimed task index of the open region.
    next: Padded<AtomicUsize>,
    /// The open region: its task, the number of tasks and the number of workers that take part. Written only
    /// by the holder of `busy`, while the region is closed and `active` is zero.
    job: UnsafeCell<Option<Task>>,
    n: AtomicUsize,
    helpers: AtomicUsize,
    /// Parked workers.
    sleepers: AtomicUsize,
    /// The first panic of a worker's task in the open region.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    /// The spawned workers, by index. Grown only by the holder of `busy`.
    slots: UnsafeCell<Vec<Arc<Slot>>>,
    /// False once a worker thread could not be spawned (no threads on this target): regions then run serially.
    can_spawn: AtomicBool,
}

// SAFETY: `job` and `slots` are only written by the holder of `busy` under the protocol described on the fields,
// and the task behind `job` is `Sync`.
unsafe impl Sync for Pool {}

static POOL: Pool = Pool {
    busy: Padded(AtomicBool::new(false)),
    epoch: Padded(AtomicU64::new(0)),
    active: Padded(AtomicUsize::new(0)),
    next: Padded(AtomicUsize::new(0)),
    job: UnsafeCell::new(None),
    n: AtomicUsize::new(0),
    helpers: AtomicUsize::new(0),
    sleepers: AtomicUsize::new(0),
    panic: Mutex::new(None),
    slots: UnsafeCell::new(Vec::new()),
    can_spawn: AtomicBool::new(true),
};

thread_local! {
    /// True on a thread that is running a task of a region (always, on a worker).
    static IN_REGION: Cell<bool> = const { Cell::new(false) };
}

/// How long an idle worker spins before it parks: `FOLO_SPIN_US` microseconds (default 1000). Within a pass the
/// gaps between regions are far shorter, so its workers never park; a caller that comes back every few
/// milliseconds (a camera) pays one wake-up per call.
fn spin_window() -> Duration {
    static D: OnceLock<Duration> = OnceLock::new();
    *D.get_or_init(|| {
        let us = std::env::var("FOLO_SPIN_US").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(1000u64);
        Duration::from_micros(us)
    })
}

/// The thread budget of a kernel started on this thread: the size of the rayon pool it runs in (the global one
/// outside any pool, so `RAYON_NUM_THREADS` and `ThreadPool::install` both set it), or one inside a task of a
/// region (which would run its own regions serially anyway).
pub(crate) fn threads() -> usize {
    if IN_REGION.get() { 1 } else { rayon::current_num_threads().min(MAX_WORKERS + 1) }
}

/// Run `f(i)` for every `i` in `0..n`, each exactly once, on the calling thread and up to `threads - 1` others,
/// and return when all are done. Tasks must be independent: any thread may run any of them, in any order. A
/// panic in a task is rethrown here once the others have stopped (tasks not yet started are dropped).
pub(crate) fn for_each<F: Fn(usize) + Sync>(n: usize, threads: usize, f: F) {
    let helpers = threads.min(n).saturating_sub(1).min(MAX_WORKERS);
    if helpers == 0 || IN_REGION.get() {
        return (0..n).for_each(f);
    }
    let pool = &POOL;
    if pool.busy.0.compare_exchange(false, true, SeqCst, SeqCst).is_err() {
        // Another thread's region is running: rayon shares its workers between callers.
        let f = &f;
        return (0..n).into_par_iter().for_each(|i| f(i));
    }
    let helpers = pool.grow(helpers);
    if helpers == 0 {
        pool.busy.0.store(false, SeqCst);
        return (0..n).for_each(f);
    }
    let task: &(dyn Fn(usize) + Sync) = &f;
    // SAFETY: `busy` is held, the region is closed and no worker is active, so nobody reads the job. The
    // lifetime of `task` is erased; `Region` closes the region and waits for every worker to leave it before
    // this function returns (or unwinds), so no worker calls `f` after that.
    unsafe {
        *pool.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Task>(task));
    }
    pool.n.store(n, Relaxed);
    pool.helpers.store(helpers, Relaxed);
    pool.next.0.store(0, Relaxed);
    let region = Region { pool };
    pool.epoch.0.fetch_add(1, SeqCst);
    if pool.sleepers.load(SeqCst) > 0 {
        // SAFETY: `slots` is only grown by the holder of `busy` (this thread).
        let slots: &Vec<Arc<Slot>> = unsafe { &*pool.slots.get() };
        for s in &slots[..helpers] {
            if s.sleeping.load(SeqCst)
                && let Some(t) = s.thread.get()
            {
                t.unpark();
            }
        }
    }
    IN_REGION.set(true);
    let mine = catch_unwind(AssertUnwindSafe(|| claim(pool, n, &f)));
    IN_REGION.set(false);
    drop(region);
    let theirs = pool.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
    pool.busy.0.store(false, SeqCst);
    if let Err(p) = mine {
        resume_unwind(p);
    }
    if let Some(p) = theirs {
        resume_unwind(p);
    }
}

/// [`for_each`] over the chunks of `data` (`chunk` elements each, the last one shorter): `f(index, chunk)`.
pub(crate) fn for_each_chunk_mut<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    data: &mut [T],
    chunk: usize,
    threads: usize,
    f: F,
) {
    assert!(chunk > 0, "for_each_chunk_mut: empty chunks");
    let len = data.len();
    let base = super::elementwise::SendPtr(data.as_mut_ptr());
    for_each(len.div_ceil(chunk), threads, |i| {
        let base = &base;
        let (s, e) = (i * chunk, ((i + 1) * chunk).min(len));
        // SAFETY: chunk `i` is `data[s..e]`, inside `data`, and every index is claimed by one task only, so
        // the chunks are disjoint and each is borrowed once.
        f(i, unsafe { std::slice::from_raw_parts_mut(base.0.add(s), e - s) });
    });
}

/// Closes the open region and waits until no worker can be inside it, also when the caller unwinds.
struct Region {
    pool: &'static Pool,
}

impl Drop for Region {
    fn drop(&mut self) {
        let p = self.pool;
        // No more tasks for anyone (a panic may have left some unclaimed).
        p.next.0.store(usize::MAX / 2, SeqCst);
        p.epoch.0.fetch_add(1, SeqCst);
        let mut spins = 0u32;
        while p.active.0.load(SeqCst) != 0 {
            spins += 1;
            if spins.is_multiple_of(1 << 14) {
                // A worker was descheduled inside its task: give its core a chance.
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: `busy` is held, the region is closed and no worker is active.
        unsafe { *p.job.get() = None };
    }
}

/// Claim and run tasks of the open region until none is left.
fn claim(pool: &Pool, n: usize, f: &(dyn Fn(usize) + Sync)) {
    loop {
        let i = pool.next.0.fetch_add(1, Relaxed);
        if i >= n {
            return;
        }
        f(i);
    }
}

impl Pool {
    /// Make sure workers `0..helpers` exist and return how many of them do. Called by the holder of `busy`.
    fn grow(&'static self, helpers: usize) -> usize {
        // SAFETY: only the holder of `busy` touches `slots`.
        let slots = unsafe { &mut *self.slots.get() };
        while slots.len() < helpers && self.can_spawn.load(Relaxed) {
            let w = slots.len();
            let slot = Arc::new(Slot { sleeping: AtomicBool::new(false), thread: OnceLock::new() });
            let theirs = slot.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("folo-par-{w}"))
                .spawn(move || worker(self, w, &theirs));
            match spawned {
                Ok(h) => {
                    let _ = slot.thread.set(h.thread().clone());
                    slots.push(slot);
                }
                Err(_) => self.can_spawn.store(false, Relaxed),
            }
        }
        helpers.min(slots.len())
    }
}

/// The loop of worker `w`: wait for a region to open (spinning, then parked), take part if it wants more than
/// `w` workers, repeat.
///
/// A worker reads the job only between raising `active` and lowering it again, and only if it saw the region
/// open (an odd epoch) after raising it. The starting thread closes the region (an even epoch) and then waits
/// for `active` to reach zero before it touches the job again. Both sides use sequentially consistent
/// operations, so either the worker's raise comes before the closing thread's read of `active` (which then
/// waits for it), or the worker's read of the epoch comes after the close (and it backs off).
fn worker(pool: &'static Pool, w: usize, slot: &Slot) {
    IN_REGION.set(true);
    let window = spin_window();
    let mut last = 0u64;
    let mut idle_since = Instant::now();
    let mut spins = 0u32;
    loop {
        let e = pool.epoch.0.load(SeqCst);
        if e != last && e & 1 == 1 {
            pool.active.0.fetch_add(1, SeqCst);
            let e = pool.epoch.0.load(SeqCst);
            if e & 1 == 1 {
                last = e;
                if w < pool.helpers.load(Relaxed) {
                    // SAFETY: the region is open and this worker is counted in `active`, so the job is set
                    // and stays alive until `active` is lowered below.
                    let f = unsafe { &*(*pool.job.get()).expect("an open region has a job") };
                    let n = pool.n.load(Relaxed);
                    if let Err(p) = catch_unwind(AssertUnwindSafe(|| claim(pool, n, f))) {
                        pool.next.0.store(usize::MAX / 2, SeqCst);
                        pool.panic.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(p);
                    }
                    idle_since = Instant::now();
                    spins = 0;
                }
            }
            pool.active.0.fetch_sub(1, SeqCst);
            continue;
        }
        std::hint::spin_loop();
        spins += 1;
        if spins.is_multiple_of(128) && idle_since.elapsed() >= window {
            // Park until a region that wants this worker opens. The starting thread raises the epoch, then
            // looks at `sleeping`; this side raises `sleeping`, then looks at the epoch: one of them sees the
            // other, and an `unpark` before the `park` makes it return at once.
            slot.sleeping.store(true, SeqCst);
            pool.sleepers.fetch_add(1, SeqCst);
            if pool.epoch.0.load(SeqCst) == e {
                std::thread::park();
            }
            pool.sleepers.fetch_sub(1, SeqCst);
            slot.sleeping.store(false, SeqCst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_runs_once() {
        for threads in [1, 2, 3, 8] {
            for n in [0usize, 1, 2, 7, 64, 1000] {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                for_each(n, threads, |i| {
                    hits[i].fetch_add(1, SeqCst);
                });
                assert!(hits.iter().all(|h| h.load(SeqCst) == 1), "threads {threads} n {n}");
            }
        }
    }

    #[test]
    fn chunks_cover_the_slice() {
        let mut v = vec![0u32; 1003];
        for_each_chunk_mut(&mut v, 10, 4, |i, c| c.iter_mut().for_each(|x| *x += 1 + i as u32));
        assert!(v.iter().enumerate().all(|(j, &x)| x == 1 + (j / 10) as u32));
    }

    /// Thousands of tiny regions back to back, with a pause now and then so workers park and are woken.
    #[test]
    fn many_small_regions() {
        let sum = AtomicUsize::new(0);
        for r in 0..4000 {
            for_each(5, 4, |i| {
                sum.fetch_add(i + 1, Relaxed);
            });
            if r % 1000 == 999 {
                std::thread::sleep(Duration::from_millis(3));
            }
        }
        assert_eq!(sum.load(SeqCst), 4000 * 15);
    }

    /// A region inside a task runs on that thread; regions from several threads at once all complete.
    #[test]
    fn nested_and_concurrent_regions() {
        let sum = AtomicUsize::new(0);
        for_each(6, 3, |_| {
            assert_eq!(threads(), 1);
            for_each(10, 4, |j| {
                sum.fetch_add(j, Relaxed);
            });
        });
        assert_eq!(sum.load(SeqCst), 6 * 45);
        let total = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..200 {
                        for_each(16, 3, |i| {
                            total.fetch_add(i, Relaxed);
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(SeqCst), 4 * 200 * 120);
    }

    #[test]
    fn a_panicking_task_is_rethrown_and_the_pool_survives() {
        for bad in [0usize, 37] {
            let r = catch_unwind(|| {
                for_each(64, 4, |i| {
                    if i == bad {
                        panic!("task {i}");
                    }
                })
            });
            assert!(r.is_err());
        }
        let n = AtomicUsize::new(0);
        for_each(100, 4, |_| {
            n.fetch_add(1, Relaxed);
        });
        assert_eq!(n.load(SeqCst), 100);
    }
}
