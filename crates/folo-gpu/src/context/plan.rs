//! Frozen dispatch lists and forward plans.
//!
//! An inference forward of one model on one input shape encodes the same operations every call: the same pipelines
//! on the same buffers with the same parameters. A **plan** records those operations once (a thread-local recorder
//! fed by the dispatch, copy and clear paths) and encodes them again with nothing but the wgpu calls: no tensor ops,
//! no pool, no bind-group lookups, no uniform arena. Every bind group is made once and every parameter block is
//! written once, into the plan's own uniform buffer.
//!
//! The buffers a recorded call touched fall in four groups, and the plan keeps each safe to write or read again:
//! - **the input**: bound where the call bound it, per replay (the dispatches that read it get a new bind group
//!   only when the input buffer differs from the last replay's, which a predictor's pooled input rarely does);
//! - **intermediates** (handed out by the pool during the call): the plan owns one buffer per pool buffer the call
//!   used, so the replay reproduces the call's own reuse of them and nothing else can be given them. It takes the
//!   very buffers out of the pool when they are free again at the end of the call
//!   ([`crate::pool::BufferPool::adopt`]; another thread may hold one by then) and makes new ones otherwise;
//! - **outputs** (handed out during the call and still held by its results): bound per replay like the input, to
//!   fresh pool buffers of the same sizes that become the caller's results, so the next replay never overwrites a
//!   result someone holds;
//! - **external tensors** (the model's weights): bound as they are. The caller lists them; the tensor layer
//!   ([`crate::ForwardPlans`]) checks before every replay that they are the same tensors at the same storage
//!   versions.
//!
//! Anything else (a buffer from outside the call that is not listed, an upload or a readback inside it) makes the
//! call unplannable. A replay is encoded under one hold of the batch lock, so replays from several threads never
//! interleave. Plans are keyed by the tensor-core switch; a tensor-core kernel that fails to build (tensor cores
//! then go off for good) bumps [`GpuContext`]'s plan epoch, which retires every plan, and so does
//! [`GpuContext::release_memory`], which also frees them.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use folo_core::{DType, Shape, TensorId};

use super::binds::{FxMap, FxSet, id_of};
use super::{GpuContext, TracedKind, TracedOp};
use crate::error::Result;
use crate::pipeline::UNIFORM_SLOT;

/// Plans a context keeps at most (least recently used first out).
pub const MAX_PLANS: usize = 8;
/// Bytes the plans of a context may pin in all (least recently used first out).
pub const MAX_PLAN_BYTES: u64 = 1 << 30;
/// Bytes one plan may pin; a call that would need more is not planned. Plans pay where the host is the bottleneck,
/// at small batches (folo-n and folo-s at batch 1: 23–102 MiB); at batch 16 they tied the normal call while pinning
/// 316–617 MiB (docs/notes/hill/r3-plan.md).
pub const MAX_PLAN_BYTES_EACH: u64 = MAX_PLAN_BYTES / 4;

/// Threads recording now, on any context: the dispatch path looks at its thread's recorder only when this is not 0.
static RECORDING: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static REC: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

/// What one thread's call encoded on one context (see [`GpuContext::record_begin`]).
pub(crate) struct Recorder {
    ctx: usize,
    ops: Vec<TracedOp>,
    /// Ids of the pool buffers handed out to this thread during the call.
    taken: FxSet,
    /// Why the call cannot be replayed (an upload or a readback inside it).
    tainted: Option<&'static str>,
    /// [`GpuContext::plan_epoch`] when the recording began: a `release_memory` during the call retires its plan.
    epoch: u64,
}

/// Recording on this thread; ends (and is discarded) on drop, also in a panic.
pub(crate) struct RecordGuard(());

impl RecordGuard {
    /// Stops recording and returns what was recorded.
    pub(crate) fn finish(self) -> Recorder {
        let rec = REC.with(|r| r.borrow_mut().take()).expect("recording");
        RECORDING.fetch_sub(1, Ordering::Relaxed);
        std::mem::forget(self);
        rec
    }
}

impl Drop for RecordGuard {
    fn drop(&mut self) {
        if REC.with(|r| r.borrow_mut().take()).is_some() {
            RECORDING.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Runs `f` on this thread's recorder when it records for `ctx`.
fn with_recorder(ctx: &GpuContext, f: impl FnOnce(&mut Recorder)) {
    if RECORDING.load(Ordering::Relaxed) == 0 {
        return;
    }
    let key = ctx.key();
    REC.with(|r| {
        if let Some(rec) = r.borrow_mut().as_mut().filter(|rec| rec.ctx == key) {
            f(rec);
        }
    });
}

/// A buffer a step uses: a fixed one, or the call's input or one of its results (bound per replay).
#[derive(Clone)]
enum Buf {
    Fixed(wgpu::Buffer),
    Input,
    Output(usize),
}

/// How a dispatch gets its bind group: made once, or made per replay from [`Frozen`]'s `dynamic` entry (it binds the
/// input or a result) and kept while the same buffers come back.
enum Bind {
    Static(wgpu::BindGroup),
    Dynamic(usize),
}

enum Step {
    Dispatch { pipeline_id: u64, pipeline: wgpu::ComputePipeline, bind: Bind, offset: u32, groups: [u32; 3] },
    Copy { src: Buf, src_off: u64, dst: Buf, dst_off: u64, len: u64 },
    Clear { buf: Buf, bytes: u64 },
}

/// A dispatch whose bind group depends on the replay's input or results.
struct DynamicBind {
    layout: wgpu::BindGroupLayout,
    buffers: Vec<Buf>,
}

fn resolve_buf<'a>(
    buf: &'a Buf,
    input: Option<&'a wgpu::Buffer>,
    results: &[&'a wgpu::Buffer],
) -> &'a wgpu::Buffer {
    match buf {
        Buf::Fixed(b) => b,
        Buf::Input => input.expect("a replay of a plan has its input"),
        Buf::Output(k) => results[*k],
    }
}

/// Creates a dynamic replay bind group without allocating a temporary `Vec` for up to 24 storage buffers.
fn create_dynamic_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    buffers: &[Buf],
    input: Option<&wgpu::Buffer>,
    results: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    const INLINE_REFS: usize = 24;
    if buffers.is_empty() {
        return super::binds::create(device, layout, uniform, &[]);
    }
    if buffers.len() <= INLINE_REFS {
        let first = resolve_buf(&buffers[0], input, results);
        let mut refs = [first; INLINE_REFS];
        for (slot, buf) in refs.iter_mut().zip(buffers.iter()).skip(1) {
            *slot = resolve_buf(buf, input, results);
        }
        super::binds::create(device, layout, uniform, &refs[..buffers.len()])
    } else {
        let refs: Vec<&wgpu::Buffer> = buffers.iter().map(|buf| resolve_buf(buf, input, results)).collect();
        super::binds::create(device, layout, uniform, &refs)
    }
}

/// Traced operations made ready to encode again without the tensor-op layer, the pool or the bind-group cache
/// ([`GpuContext::freeze`]), on the buffers they ran on.
pub struct Frozen {
    steps: Vec<Step>,
    dynamic: Vec<DynamicBind>,
    /// Every dispatch's parameter block at its own slot (also held by the bind groups).
    uniform: Option<wgpu::Buffer>,
}

impl Frozen {
    /// Operations in the list.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// One result of a planned call: its dtype and shape (dense from byte 0 of its own buffer).
pub(crate) struct PlanOutput {
    pub dtype: DType,
    pub shape: Shape,
}

/// The bind groups of a plan's dynamic dispatches made at its last replay, with the buffers they bind.
type LastBinds = Vec<Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>>;

/// A recorded call, ready to replay ([`GpuContext::replay_plan`]).
pub(crate) struct Plan {
    frozen: Frozen,
    /// What the plan's key was hashed from (the caller's key, the input's dtype, buffer size and dims, the
    /// tensor-core switch): a replay needs all of it equal, so two keys whose hashes collide never share a plan.
    pub signature: Vec<u64>,
    /// The sizes of the result buffers (a replay allocates them from the pool).
    pub output_sizes: Vec<u64>,
    pub results: Vec<PlanOutput>,
    /// The external tensors and their storage versions when the call was recorded.
    pub state: Vec<(TensorId, u64)>,
    /// Bytes of device memory the plan holds: intermediates, parameters, and the sizes of its input and results (the
    /// last ones it was replayed with stay referenced). The listed tensors it binds are not counted: their owner
    /// holds them.
    pub bytes: u64,
    /// [`GpuContext::plan_epoch`] when the recording began.
    pub epoch: u64,
    last: std::sync::Mutex<LastBinds>,
    /// The intermediates (the bind groups hold them too).
    _internal: Vec<wgpu::Buffer>,
}

/// The role of a recorded buffer in a plan.
enum Role {
    Keep,
    Input,
    Output(usize),
    /// Index into the plan's intermediates.
    Internal(usize),
}

/// The plans of a context, the keys seen once (a key is recorded on its second call) and the keys whose recording
/// did not become a plan (not recorded again until the store is cleared).
#[derive(Default)]
pub(crate) struct PlanStore {
    entries: Vec<StoreEntry>,
    seen: FxSet,
    refused: FxSet,
    tick: u64,
}

struct StoreEntry {
    owner: u64,
    key: u64,
    plan: Arc<Plan>,
    used: u64,
}

impl PlanStore {
    pub(crate) fn get(&mut self, owner: u64, key: u64) -> Option<Arc<Plan>> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.iter_mut().find(|e| e.owner == owner && e.key == key).map(|e| {
            e.used = tick;
            e.plan.clone()
        })
    }

    /// Notes a call of `(owner, key)` without a plan; true when one was seen before (record this one).
    pub(crate) fn second_sighting(&mut self, owner: u64, key: u64) -> bool {
        let k = id_of(&(owner, key));
        if self.refused.contains(&k) {
            return false;
        }
        if self.seen.remove(&k) {
            return true;
        }
        if self.seen.len() >= 1024 {
            self.seen.clear();
        }
        self.seen.insert(k);
        false
    }

    /// Keeps `plan` for `(owner, key)`, dropping the least recently used plans past [`MAX_PLANS`] or
    /// [`MAX_PLAN_BYTES`]; false (and nothing kept) when the plan is larger than [`MAX_PLAN_BYTES_EACH`].
    pub(crate) fn insert(&mut self, owner: u64, key: u64, plan: Plan) -> bool {
        self.remove(owner, key);
        if plan.bytes > MAX_PLAN_BYTES_EACH {
            return false;
        }
        while self.entries.len() >= MAX_PLANS || self.bytes() + plan.bytes > MAX_PLAN_BYTES {
            let lru = (0..self.entries.len()).min_by_key(|&i| self.entries[i].used).expect("not empty");
            self.entries.swap_remove(lru);
        }
        self.tick += 1;
        self.entries.push(StoreEntry { owner, key, plan: Arc::new(plan), used: self.tick });
        true
    }

    /// Stops recording `(owner, key)`.
    pub(crate) fn refuse(&mut self, owner: u64, key: u64) {
        if self.refused.len() >= 1024 {
            self.refused.clear();
        }
        self.refused.insert(id_of(&(owner, key)));
    }

    pub(crate) fn remove(&mut self, owner: u64, key: u64) {
        self.entries.retain(|e| e.owner != owner || e.key != key);
    }

    pub(crate) fn remove_owner(&mut self, owner: u64) {
        self.entries.retain(|e| e.owner != owner);
    }

    /// Drops every plan and returns the bytes they held.
    pub(crate) fn clear(&mut self) -> u64 {
        let bytes = self.bytes();
        self.entries.clear();
        self.seen.clear();
        self.refused.clear();
        bytes
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.plan.bytes).sum()
    }
}

impl GpuContext {
    /// Identifies this context in thread-local recorders.
    fn key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    /// Starts recording what this thread encodes on this context; `None` when it records already.
    pub(crate) fn record_begin(&self) -> Option<RecordGuard> {
        REC.with(|r| {
            let mut r = r.borrow_mut();
            if r.is_some() {
                return None;
            }
            *r = Some(Recorder {
                ctx: self.key(),
                ops: Vec::new(),
                taken: FxSet::default(),
                tainted: None,
                epoch: self.plan_epoch(),
            });
            RECORDING.fetch_add(1, Ordering::Relaxed);
            Some(RecordGuard(()))
        })
    }

    /// Whether this thread records for this context (the dispatch path then hands it its operations).
    pub(crate) fn recording(&self) -> bool {
        let mut on = false;
        with_recorder(self, |_| on = true);
        on
    }

    pub(super) fn record_op(&self, op: TracedOp) {
        with_recorder(self, |rec| rec.ops.push(op));
    }

    /// Notes a pool buffer handed out to this thread.
    pub(super) fn record_taken(&self, buf: &wgpu::Buffer) {
        with_recorder(self, |rec| {
            rec.taken.insert(id_of(buf));
        });
    }

    /// Marks this thread's recording as not replayable.
    pub(super) fn record_taint(&self, why: &'static str) {
        with_recorder(self, |rec| {
            rec.tainted.get_or_insert(why);
        });
    }

    /// Bumped by every change that can make a recorded plan pick other kernels than a new call would.
    pub(crate) fn plan_epoch(&self) -> u64 {
        self.inner.plan_epoch.load(Ordering::Relaxed)
    }

    pub(super) fn bump_plan_epoch(&self) {
        self.inner.plan_epoch.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn plans(&self) -> std::sync::MutexGuard<'_, PlanStore> {
        self.inner.plans.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Whether forward plans are on for this context ([`GpuContext::set_forward_plans`]).
    pub fn forward_plans(&self) -> bool {
        self.inner.plans_on.load(Ordering::Relaxed)
    }

    /// Turns forward plans ([`crate::ForwardPlans`]) on or off for later calls, for A/B comparisons in one process
    /// (the default is on, or off with [`super::ENV_NO_PLAN`]). Turning them off drops every plan.
    pub fn set_forward_plans(&self, on: bool) {
        self.inner.plans_on.store(on, Ordering::Relaxed);
        if !on {
            self.plans().clear();
        }
    }

    /// Plans kept and the device bytes they hold.
    pub fn plan_stats(&self) -> (usize, u64) {
        let s = self.plans();
        (s.len(), s.bytes())
    }

    /// Makes `ops` ready for [`GpuContext::replay_frozen`]: one bind group per dispatch, every parameter block in
    /// one uniform buffer written now. The operations keep running on the buffers they were traced on.
    pub fn freeze(&self, ops: &[TracedOp]) -> Result<Frozen> {
        Ok(self.freeze_mapped(ops, |b| Buf::Fixed(b.clone())))
    }

    /// The size of the uniform buffer [`GpuContext::freeze_mapped`] makes for `ops`: a slot per dispatch.
    fn uniform_bytes(&self, ops: &[TracedOp]) -> u64 {
        let align = UNIFORM_SLOT.max(self.inner.caps.uniform_align as u64);
        ops.iter().filter(|o| matches!(o.kind, TracedKind::Dispatch { .. })).count() as u64 * align
    }

    fn freeze_mapped(&self, ops: &[TracedOp], map: impl Fn(&wgpu::Buffer) -> Buf) -> Frozen {
        let align = UNIFORM_SLOT.max(self.inner.caps.uniform_align as u64);
        let size = self.uniform_bytes(ops);
        let uniform = (size > 0).then(|| {
            self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("folo-gpu plan params"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let mut bytes = vec![0u8; size as usize];
        let mut steps = Vec::with_capacity(ops.len());
        let mut dynamic = Vec::new();
        let mut slot = 0u64;
        for op in ops {
            steps.push(match &op.kind {
                TracedKind::Dispatch { kernel, buffers, params } => {
                    let u = uniform.as_ref().expect("a dispatch makes the buffer");
                    let mapped: Vec<Buf> = buffers.iter().map(&map).collect();
                    let fixed: Vec<&wgpu::Buffer> = mapped
                        .iter()
                        .filter_map(|b| match b {
                            Buf::Fixed(b) => Some(b),
                            _ => None,
                        })
                        .collect();
                    let bind = if fixed.len() == mapped.len() {
                        self.inner.bind_groups.fetch_add(1, Ordering::Relaxed);
                        Bind::Static(super::binds::create(&self.inner.device, &kernel.layout, u, &fixed))
                    } else {
                        dynamic.push(DynamicBind { layout: kernel.layout.clone(), buffers: mapped });
                        Bind::Dynamic(dynamic.len() - 1)
                    };
                    bytes[slot as usize..(slot + UNIFORM_SLOT) as usize].copy_from_slice(params.bytes());
                    let offset = slot as u32;
                    slot += align;
                    Step::Dispatch {
                        pipeline_id: kernel.pipeline_id,
                        pipeline: kernel.pipeline.clone(),
                        bind,
                        offset,
                        groups: op.groups,
                    }
                }
                TracedKind::Copy { src, src_off, dst, dst_off, len } => Step::Copy {
                    src: map(src),
                    src_off: *src_off,
                    dst: map(dst),
                    dst_off: *dst_off,
                    len: *len,
                },
                TracedKind::Clear { buf, bytes } => Step::Clear { buf: map(buf), bytes: *bytes },
            });
        }
        if let Some(u) = &uniform {
            self.inner.queue.write_buffer(u, 0, &bytes);
        }
        Frozen { steps, dynamic, uniform }
    }

    /// Encodes a [`Frozen`] list into the open batch, submitting on the schedule of the calls it came from.
    pub fn replay_frozen(&self, f: &Frozen) {
        let mut b = self.inner.batch.lock().expect("batch lock");
        self.encode_steps(&mut b, f, None, &[], &mut []);
    }

    /// Encodes the steps of `f`, binding `input` and `results` where the call bound its input and results (a frozen
    /// trace has neither), and reusing the dynamic bind groups of `last` that bind the same buffers.
    fn encode_steps(
        &self,
        b: &mut super::Batch,
        f: &Frozen,
        input: Option<&wgpu::Buffer>,
        results: &[&wgpu::Buffer],
        last: &mut [Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>],
    ) {
        for step in &f.steps {
            match step {
                Step::Dispatch { pipeline_id, pipeline, bind, offset, groups } => {
                    let bind: &wgpu::BindGroup = match bind {
                        Bind::Static(g) => g,
                        Bind::Dynamic(i) => {
                            let d = &f.dynamic[*i];
                            let hit = matches!(
                                &last[*i],
                                Some((held, _))
                                    if held.len() == d.buffers.len()
                                        && held
                                            .iter()
                                            .zip(&d.buffers)
                                            .all(|(h, buf)| h == resolve_buf(buf, input, results))
                            );
                            if !hit {
                                let u = f.uniform.as_ref().expect("a dispatch makes the buffer");
                                let g = create_dynamic_bind(
                                    &self.inner.device,
                                    &d.layout,
                                    u,
                                    &d.buffers,
                                    input,
                                    results,
                                );
                                self.inner.bind_groups.fetch_add(1, Ordering::Relaxed);
                                let bufs: Vec<wgpu::Buffer> = d
                                    .buffers
                                    .iter()
                                    .map(|buf| resolve_buf(buf, input, results).clone())
                                    .collect();
                                last[*i] = Some((bufs, g));
                            }
                            &last[*i].as_ref().expect("dynamic bind populated").1
                        }
                    };
                    if b.pass.is_none() {
                        let enc = self.encoder(b);
                        let pass = enc
                            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            })
                            .forget_lifetime();
                        b.pass = Some(pass);
                    }
                    let pass = b.pass.as_mut().expect("pass");
                    if b.last_pipeline != Some(*pipeline_id) {
                        pass.set_pipeline(pipeline);
                        b.last_pipeline = Some(*pipeline_id);
                        #[cfg(test)]
                        self.inner.pipeline_sets.fetch_add(1, Ordering::Relaxed);
                    }
                    pass.set_bind_group(0, bind, &[*offset]);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                    self.inner.dispatches.fetch_add(1, Ordering::Relaxed);
                }
                Step::Copy { src, src_off, dst, dst_off, len } => self.encoder(b).copy_buffer_to_buffer(
                    resolve_buf(src, input, results),
                    *src_off,
                    resolve_buf(dst, input, results),
                    *dst_off,
                    *len,
                ),
                Step::Clear { buf, bytes } => {
                    self.encoder(b).clear_buffer(resolve_buf(buf, input, results), 0, Some(*bytes))
                }
            }
            self.bump(b);
        }
    }

    /// Turns a recording into a plan, or says why it cannot be one. `input` is the call's input buffer, `outputs`
    /// the buffers of its results, `externals` the buffers of the tensors the caller vouches for
    /// ([`crate::ForwardPlans`] checks them before each replay).
    pub(crate) fn build_plan(
        &self,
        rec: Recorder,
        input: &wgpu::Buffer,
        outputs: &[wgpu::Buffer],
        externals: &[wgpu::Buffer],
    ) -> std::result::Result<Plan, String> {
        if let Some(why) = rec.tainted {
            return Err(format!("the call made {why}"));
        }
        let ext: FxMap<&wgpu::Buffer> = externals.iter().map(|b| (id_of(b), b)).collect();
        let mut roles: FxMap<(wgpu::Buffer, Role)> = FxMap::default();
        let mut internal = Vec::new();
        for op in &rec.ops {
            let bufs: Vec<&wgpu::Buffer> = match &op.kind {
                TracedKind::Dispatch { buffers, .. } => buffers.iter().collect(),
                TracedKind::Copy { src, dst, .. } => vec![src, dst],
                TracedKind::Clear { buf, .. } => vec![buf],
            };
            for buf in bufs {
                let id = id_of(buf);
                if let Some((known, _)) = roles.get(&id) {
                    if known != buf {
                        return Err("two buffers share an id".into());
                    }
                    continue;
                }
                let role = if buf == input {
                    Role::Input
                } else if buf == self.dummy() || ext.get(&id).is_some_and(|e| *e == buf) {
                    Role::Keep
                } else if !rec.taken.contains(&id) {
                    return Err(format!(
                        "{} reads a buffer from outside the call that is not listed",
                        op.name
                    ));
                } else if let Some(k) = outputs.iter().position(|o| o == buf) {
                    Role::Output(k)
                } else {
                    internal.push(buf.clone());
                    Role::Internal(internal.len() - 1)
                };
                roles.insert(id, (buf.clone(), role));
            }
        }
        if let Some(k) = (0..outputs.len())
            .find(|&k| !roles.values().any(|(_, r)| matches!(r, Role::Output(i) if *i == k)))
        {
            return Err(format!("result {k} was not written by the call"));
        }
        // Before any buffer leaves the pool: a plan too large to keep would only cost the pool its buffers. Besides
        // its intermediates, a plan keeps the last input and results it was replayed with (in the bind groups of
        // its dynamic dispatches), so those count too, as does the uniform buffer it will make.
        let held = input.size() + outputs.iter().map(wgpu::Buffer::size).sum::<u64>();
        let pinned: u64 =
            internal.iter().map(wgpu::Buffer::size).sum::<u64>() + held + self.uniform_bytes(&rec.ops);
        if pinned > MAX_PLAN_BYTES_EACH {
            return Err(format!("the plan would hold {pinned} bytes, more than {MAX_PLAN_BYTES_EACH}"));
        }
        let internal: Vec<wgpu::Buffer> = {
            let mut pool = self.inner.pool.lock().expect("pool lock");
            internal
                .into_iter()
                .map(|b| {
                    if pool.adopt(&b) {
                        return b;
                    }
                    self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("folo-gpu plan intermediate"),
                        size: b.size(),
                        usage: b.usage(),
                        mapped_at_creation: false,
                    })
                })
                .collect()
        };
        let frozen = self.freeze_mapped(&rec.ops, |b| match roles.get(&id_of(b)).map(|(_, r)| r) {
            Some(Role::Input) => Buf::Input,
            Some(Role::Output(k)) => Buf::Output(*k),
            Some(Role::Internal(k)) => Buf::Fixed(internal[*k].clone()),
            _ => Buf::Fixed(b.clone()),
        });
        let bytes = pinned;
        let last = std::sync::Mutex::new((0..frozen.dynamic.len()).map(|_| None).collect());
        Ok(Plan {
            frozen,
            signature: Vec::new(),
            output_sizes: outputs.iter().map(wgpu::Buffer::size).collect(),
            results: Vec::new(),
            state: Vec::new(),
            bytes,
            epoch: rec.epoch,
            last,
            _internal: internal,
        })
    }

    /// Encodes a replay of `plan` reading `input` (a buffer of the size it was recorded on) and writing its results
    /// into `results` (fresh buffers of [`Plan::output_sizes`]), under one hold of the batch lock.
    pub(crate) fn replay_plan(&self, plan: &Plan, input: &wgpu::Buffer, results: &[&wgpu::Buffer]) {
        let mut b = self.inner.batch.lock().expect("batch lock");
        let mut last = plan.last.lock().unwrap_or_else(|p| p.into_inner());
        self.encode_steps(&mut b, &plan.frozen, Some(input), results, &mut last);
    }
}
