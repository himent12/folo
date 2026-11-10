//! Releasing inference memory once the store is idle: the GPU pool's cached buffers and the cached predictors.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use folo_core::Device;

use crate::engine::{self, Predictors};
use crate::runs::Runs;

use super::{GPU_IDLE_RELEASE, PREDICTOR_IDLE};

/// When the store last used the GPU for inference, and how many such calls are in flight; read by the idle sweeper
/// ([`spawn_idle_sweeper`]).
pub(super) struct GpuIdle {
    started: Instant,
    /// Milliseconds after `started` at which the last inference call ended.
    last_ms: AtomicU64,
    active: AtomicUsize,
    /// This store's inference used the GPU since the pool was last released (so a store that runs on the CPU never
    /// touches the process's GPU pool, which another store in the same process may be using).
    gpu_used: AtomicBool,
    /// The ordinal of the GPU this store's inference ran on (the store's `--device`, resolved; the same for every
    /// call), whose pool the sweeper releases.
    gpu: AtomicUsize,
    pub(super) release_after_ms: AtomicU64,
    pub(super) predictors_after_ms: AtomicU64,
}

impl GpuIdle {
    pub(super) fn new() -> GpuIdle {
        GpuIdle {
            started: Instant::now(),
            last_ms: AtomicU64::new(0),
            active: AtomicUsize::new(0),
            gpu_used: AtomicBool::new(false),
            gpu: AtomicUsize::new(0),
            release_after_ms: AtomicU64::new(GPU_IDLE_RELEASE.as_millis() as u64),
            predictors_after_ms: AtomicU64::new(PREDICTOR_IDLE.as_millis() as u64),
        }
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// The GPU whose pool the sweeper releases.
    fn release_device(&self) -> Device {
        Device::Gpu(self.gpu.load(Ordering::SeqCst))
    }

    /// Marks an inference call in flight until the guard drops (its end is then the last use).
    pub(super) fn using(self: &Arc<Self>) -> GpuUse {
        self.active.fetch_add(1, Ordering::SeqCst);
        GpuUse { idle: self.clone(), gpu: AtomicBool::new(false) }
    }
}

impl GpuUse {
    /// The call runs on `device`: a GPU's pool is released once inference is idle.
    pub(super) fn on(&self, device: Device) {
        if let Device::Gpu(n) = device {
            self.idle.gpu.store(n, Ordering::SeqCst);
            self.gpu.store(true, Ordering::SeqCst);
            self.idle.gpu_used.store(true, Ordering::SeqCst);
        }
    }
}

/// An inference call in flight ([`GpuIdle::using`]).
pub(super) struct GpuUse {
    idle: Arc<GpuIdle>,
    gpu: AtomicBool,
}

impl Drop for GpuUse {
    fn drop(&mut self) {
        // Marked again at the end (the guard is the call's first local, so it drops after the call's predictor and
        // tensors): a call that overlapped a sweep, whose predictor the sweep evicted or whose buffers came back to
        // the pool after the release, gets the next idle stretch's release.
        if self.gpu.load(Ordering::SeqCst) {
            self.idle.gpu_used.store(true, Ordering::SeqCst);
        }
        self.idle.last_ms.store(self.idle.now_ms(), Ordering::SeqCst);
        self.idle.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Releases inference memory once the store has been idle: the GPU pool's cached buffers after
/// [`GPU_IDLE_RELEASE`], the cached predictors after [`PREDICTOR_IDLE`]; never while an inference call is in flight
/// or a run is queued or training, and only after this store's inference used the GPU. The thread ends with the
/// store.
pub(super) fn spawn_idle_sweeper(idle: Weak<GpuIdle>, predictors: Weak<Predictors>, runs: Weak<Runs>) {
    let spawned = std::thread::Builder::new().name("folo-gpu-idle".into()).spawn(move || {
        // The last use the predictors were dropped after.
        let mut dropped = None;
        loop {
            let tick = match idle.upgrade() {
                Some(g) => g.release_after_ms.load(Ordering::Relaxed).clamp(50, 10_000) / 2,
                None => return,
            };
            std::thread::sleep(Duration::from_millis(tick));
            let (Some(g), Some(predictors), Some(runs)) =
                (idle.upgrade(), predictors.upgrade(), runs.upgrade())
            else {
                return;
            };
            if g.active.load(Ordering::SeqCst) > 0 || runs.busy() {
                continue;
            }
            let last = g.last_ms.load(Ordering::SeqCst);
            let since = g.now_ms().saturating_sub(last);
            if since >= g.predictors_after_ms.load(Ordering::Relaxed) && dropped != Some(last) {
                dropped = Some(last);
                if predictors.clear() {
                    // GPU models' buffers are in the pool now: release them too.
                    g.gpu_used.store(true, Ordering::SeqCst);
                }
            }
            if since >= g.release_after_ms.load(Ordering::Relaxed) && g.gpu_used.swap(false, Ordering::SeqCst)
            {
                engine::release_gpu_memory(g.release_device());
            }
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("could not start the GPU idle sweeper: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The idle sweeper releases the pool of the GPU inference ran on (`--device gpu:1`), not always `gpu:0`.
    #[test]
    fn idle_release_targets_the_gpu_inference_used() {
        for n in [0, 1, 3] {
            let idle = Arc::new(GpuIdle::new());
            idle.using().on(Device::Gpu(n));
            assert!(idle.gpu_used.load(Ordering::SeqCst));
            assert_eq!(idle.release_device(), Device::Gpu(n));
            // A later call on the CPU does not change it.
            idle.using().on(Device::Cpu);
            assert_eq!(idle.release_device(), Device::Gpu(n));
        }
    }
}
