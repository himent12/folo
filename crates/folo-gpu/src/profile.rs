//! Per-kernel GPU timings for profiling (`FOLO_GPU_PROFILE=1`).
//!
//! With the variable set (and an adapter that has timestamp queries inside passes and encoders), a context writes
//! a timestamp before and after every dispatch, buffer copy and clear, resolves them when the batch is submitted
//! (waiting for the GPU there, so a profiled run is slower end to end) and adds the differences up per label:
//! the caller's [`scope`] tag, if one is set on the encoding thread, then the kernel name. Read the totals with
//! [`crate::GpuContext::profile_take`]. Without the variable none of this exists: no feature is requested, no
//! query is written and [`scope`] does not run its closure.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Environment variable that turns per-kernel timing on for contexts created afterwards (`FOLO_GPU_PROFILE=1`).
pub const ENV_PROFILE: &str = "FOLO_GPU_PROFILE";

/// Timed commands per submission: two queries each, and wgpu allows 4096 queries per set.
pub(crate) const MAX_TIMED: u32 = 2048;

/// Set once a profiling context exists, so [`scope`] costs one load otherwise.
static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TAG: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Clears the thread's tag when dropped, if this scope set it.
pub struct Scope(bool);

impl Drop for Scope {
    fn drop(&mut self) {
        if self.0 {
            TAG.with(|t| *t.borrow_mut() = None);
        }
    }
}

/// Tags the dispatches this thread encodes until the guard is dropped (the outermost scope wins). `label` only
/// runs while a profiling context exists.
pub fn scope(label: impl FnOnce() -> String) -> Scope {
    if !ENABLED.load(Ordering::Relaxed) {
        return Scope(false);
    }
    TAG.with(|t| {
        let mut t = t.borrow_mut();
        if t.is_some() {
            return Scope(false);
        }
        *t = Some(label());
        Scope(true)
    })
}

/// One row of [`crate::GpuContext::profile_take`].
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileRow {
    /// `tag/kernel` (or the kernel name alone, `copy`, `clear`).
    pub label: String,
    /// Commands timed under this label.
    pub count: u64,
    /// Their GPU time in milliseconds.
    pub ms: f64,
}

pub(crate) struct Profiler {
    pub set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period: f64,
    labels: Vec<String>,
    totals: HashMap<String, (u64, f64)>,
}

impl Profiler {
    /// The features a profiling device needs.
    pub fn features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
    }

    /// Whether profiling was asked for and `adapter` can do it.
    pub fn wanted(adapter: &wgpu::Adapter) -> bool {
        std::env::var(ENV_PROFILE).is_ok_and(|v| v != "0" && !v.is_empty())
            && adapter.features().contains(Self::features())
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let bytes = 8 * 2 * MAX_TIMED as u64;
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        ENABLED.store(true, Ordering::Relaxed);
        Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("folo-gpu profile"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * MAX_TIMED,
            }),
            resolve: buffer(
                "folo-gpu profile resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            read: buffer(
                "folo-gpu profile read",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            period: queue.get_timestamp_period() as f64,
            labels: Vec::new(),
            totals: HashMap::new(),
        }
    }

    /// Registers the next timed command and returns its first query index (the second is the next one).
    pub fn begin(&mut self, kernel: &str) -> u32 {
        let label = TAG.with(|t| match t.borrow().as_deref() {
            Some(tag) => format!("{tag}/{kernel}"),
            None => kernel.to_string(),
        });
        self.labels.push(label);
        2 * (self.labels.len() as u32 - 1)
    }

    /// Encodes the resolve of this batch's queries; `None` when nothing was timed.
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) -> Option<u64> {
        let n = self.labels.len() as u32;
        if n == 0 {
            return None;
        }
        enc.resolve_query_set(&self.set, 0..2 * n, &self.resolve, 0);
        let bytes = 16 * n as u64;
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, bytes);
        Some(bytes)
    }

    /// Reads the resolved timestamps of the batch just submitted (waits for the GPU) and adds them to the totals.
    pub fn collect(&mut self, device: &wgpu::Device, bytes: u64) {
        let slice = self.read.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(view) = slice.get_mapped_range() {
            let ticks: Vec<u64> = view.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
            for (label, t) in self.labels.drain(..).zip(ticks.as_chunks::<2>().0) {
                let ms = t[1].saturating_sub(t[0]) as f64 * self.period * 1e-6;
                let e = self.totals.entry(label).or_default();
                e.0 += 1;
                e.1 += ms;
            }
        }
        self.labels.clear();
        self.read.unmap();
    }

    /// The totals so far, largest first, and a reset.
    pub fn take(&mut self) -> Vec<ProfileRow> {
        let mut rows: Vec<ProfileRow> =
            self.totals.drain().map(|(label, (count, ms))| ProfileRow { label, count, ms }).collect();
        rows.sort_by(|a, b| b.ms.total_cmp(&a.ms).then_with(|| a.label.cmp(&b.label)));
        rows
    }
}
