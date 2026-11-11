//! The renderer's own thread pool (not rayon's global pool, which belongs to the engine): a few
//! threads parked on a condition variable between frames, all running one job at a time.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// A job: every participant calls it once; it pulls work items from a shared counter itself.
type Job = dyn Fn() + Sync;

/// A lifetime-erased pointer to the running job.
#[derive(Clone, Copy)]
struct JobPtr(*const Job);

// SAFETY: the pointee is `Sync`, and `Pool::run` keeps it alive until every worker is done with it.
unsafe impl Send for JobPtr {}

struct State {
    generation: u64,
    job: Option<JobPtr>,
    /// Workers still running the current job.
    active: usize,
    panic: Option<Box<dyn std::any::Any + Send>>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
}

/// A fixed set of parked worker threads.
pub(crate) struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pool with `workers` threads (the caller of `run` takes part too).
    pub fn new(workers: usize) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { generation: 0, job: None, active: 0, panic: None, shutdown: false }),
            start: Condvar::new(),
            done: Condvar::new(),
        });
        let workers = (0..workers)
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("folo-ui-raster-{i}"))
                    .spawn(move || worker(&shared))
                    .expect("spawn a raster thread")
            })
            .collect();
        Pool { shared, workers }
    }

    /// How many threads take part in a job, the caller included.
    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `job` on every worker and on the calling thread, and returns when all are done.
    pub fn run<'a>(&self, job: &'a (dyn Fn() + Sync + 'a)) {
        let raw: *const (dyn Fn() + Sync + 'a) = job;
        // SAFETY: only the lifetime is erased; `run` does not return before `active` is back to 0,
        // that is before every worker has stopped using the pointer (it is cleared below as well).
        let ptr = JobPtr(unsafe {
            std::mem::transmute::<*const (dyn Fn() + Sync + 'a), *const (dyn Fn() + Sync + 'static)>(raw)
        });
        {
            let mut s = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            s.job = Some(ptr);
            s.active = self.workers.len();
            s.generation += 1;
        }
        self.shared.start.notify_all();
        let mine = catch_unwind(AssertUnwindSafe(job));
        let mut s = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.active > 0 {
            s = self.shared.done.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.job = None;
        let theirs = s.panic.take();
        drop(s);
        if let Err(p) = mine {
            resume_unwind(p);
        }
        if let Some(p) = theirs {
            resume_unwind(p);
        }
    }
}

/// Runs `f` on every task: on `pool`'s threads (and the caller) when given, else on the caller.
pub(crate) fn for_each<T: Send>(pool: Option<&Pool>, tasks: Vec<T>, f: &(dyn Fn(T) + Sync)) {
    match pool {
        Some(pool) if tasks.len() > 1 => {
            let queue = Mutex::new(tasks);
            pool.run(&|| {
                loop {
                    let task = queue.lock().unwrap_or_else(|e| e.into_inner()).pop();
                    match task {
                        Some(t) => f(t),
                        None => break,
                    }
                }
            });
        }
        _ => tasks.into_iter().for_each(f),
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        let job = {
            let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while s.generation == seen && !s.shutdown {
                s = shared.start.wait(s).unwrap_or_else(|e| e.into_inner());
            }
            if s.shutdown {
                return;
            }
            seen = s.generation;
            s.job
        };
        let result = match job {
            // SAFETY: `Pool::run` keeps the job alive until this worker decrements `active` below.
            Some(JobPtr(p)) => catch_unwind(AssertUnwindSafe(|| unsafe { (*p)() })),
            None => Ok(()),
        };
        let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(p) = result {
            s.panic.get_or_insert(p);
        }
        s.active -= 1;
        if s.active == 0 {
            shared.done.notify_one();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner()).shutdown = true;
        self.shared.start.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_item_runs_once_and_panics_propagate() {
        let pool = Pool::new(3);
        for _ in 0..50 {
            let next = AtomicUsize::new(0);
            let sum = AtomicUsize::new(0);
            pool.run(&|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= 100 {
                        break;
                    }
                    sum.fetch_add(i, Ordering::Relaxed);
                }
            });
            assert_eq!(sum.load(Ordering::Relaxed), 4950);
        }
        let r = catch_unwind(AssertUnwindSafe(|| pool.run(&|| panic!("boom"))));
        assert!(r.is_err());
        // Still usable afterwards.
        let n = AtomicUsize::new(0);
        pool.run(&|| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), pool.threads());
    }
}
