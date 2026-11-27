//! Reusable C4 `Vec<u32>` activation scratch pool for [`super::exec::Int8`].
//!
//! During a single int8 forward pass, at most ~6–8 activation buffers are live at once (the current tensor plus
//! skip connections and narrow slices). Pooling buffers directly in a per-forward `ScratchPool` (checked out once
//! from [`super::Int8Plan`] and returned at the end of `features`) eliminates per-layer `Arc` heap allocations,
//! atomic refcount traffic, global buffer-pool lookups, and `malloc`/`free` churn for small `< 4096`-word tensors
//! (`20×20` / `12×20` maps below `folo_core`'s `MIN_POOLED` threshold).

use std::cell::RefCell;
use std::rc::Rc;

use folo_core::cpu::q8;

/// Maximum number of free `Vec<u32>` buffers retained inside one [`ScratchPool`] across calls.
const MAX_FREE_BUFS: usize = 20;
/// Maximum total words (`u32`, 4 bytes each) retained in `free_bufs` across calls (~32 MiB per pool).
const MAX_RETAINED_WORDS: usize = 8 * 1024 * 1024;
/// Maximum oversizing factor (`cap <= len * MAX_OVER_RATIO`) when reusing a larger buffer for a smaller layer,
/// so tiny `20×20` layers do not steal large `160×160` or `320×320` buffers needed by subsequent layers.
const MAX_OVER_RATIO: usize = 2;

struct Slot {
    buf: Vec<u32>,
    refs: u32,
}

/// Allocation counters tracked per [`ScratchPool`] (exposed for tests and profiling).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScratchStats {
    /// Buffers satisfied from the pool's free list without allocating.
    pub reuses: u64,
    /// Fresh buffers allocated (via [`q8::alloc`]) because no free buffer fit.
    pub allocs: u64,
}

/// Single-threaded reusable C4 buffer slab used by one active int8 forward pass.
#[derive(Default)]
pub(crate) struct ScratchPool {
    slots: Vec<Slot>,
    free_slots: Vec<u16>,
    free_bufs: Vec<Vec<u32>>,
    free_words: usize,
    stats: ScratchStats,
}

impl ScratchPool {
    /// Take a `Vec<u32>` with `buf.len() >= need` (exact/best-fit reuse from `free_bufs`, else fresh `q8::alloc`).
    pub(crate) fn take_buf(&mut self, n: usize, c: usize, hw: usize) -> Vec<u32> {
        let need = n
            .checked_mul(c / 4)
            .and_then(|v| v.checked_mul(hw))
            .and_then(|v| v.checked_add(q8::SLACK))
            .expect("ScratchPool::take_buf: size overflow");
        let max_len = need.saturating_mul(MAX_OVER_RATIO).max(4096);
        let mut best: Option<(usize, usize)> = None; // (index, buf.len())
        for (i, b) in self.free_bufs.iter().enumerate() {
            let blen = b.len();
            if blen >= need && blen <= max_len {
                if blen == need {
                    best = Some((i, blen));
                    break;
                }
                if best.is_none_or(|(_, cur)| blen < cur) {
                    best = Some((i, blen));
                }
            }
        }
        if let Some((idx, blen)) = best {
            let buf = self.free_bufs.swap_remove(idx);
            self.free_words = self.free_words.saturating_sub(blen);
            self.stats.reuses += 1;
            return buf;
        }
        self.stats.allocs += 1;
        q8::alloc(n, c, hw)
    }

    /// Return an uncommitted buffer (e.g. on an error path) back to the free list.
    pub(crate) fn return_buf(&mut self, buf: Vec<u32>) {
        if buf.is_empty() {
            return;
        }
        let blen = buf.len();
        if self.free_bufs.len() < MAX_FREE_BUFS && self.free_words.saturating_add(blen) <= MAX_RETAINED_WORDS
        {
            self.free_words += blen;
            self.free_bufs.push(buf);
        } else {
            q8::recycle(buf);
        }
    }

    /// Store `buf` in a slot with `refs = 1` and return its slot index.
    pub(crate) fn insert_slot(&mut self, buf: Vec<u32>) -> u16 {
        if let Some(idx) = self.free_slots.pop() {
            let slot = &mut self.slots[idx as usize];
            debug_assert!(slot.refs == 0 && slot.buf.is_empty());
            slot.buf = buf;
            slot.refs = 1;
            idx
        } else {
            let idx = u16::try_from(self.slots.len()).expect("ScratchPool: too many simultaneous live slots");
            self.slots.push(Slot { buf, refs: 1 });
            idx
        }
    }

    #[inline]
    pub(crate) fn slice(&self, slot: u16) -> &[u32] {
        &self.slots[slot as usize].buf
    }

    #[inline]
    fn inc_ref(&mut self, slot: u16) {
        self.slots[slot as usize].refs += 1;
    }

    #[inline]
    fn dec_ref(&mut self, slot: u16) {
        let s = &mut self.slots[slot as usize];
        debug_assert!(s.refs > 0);
        s.refs -= 1;
        if s.refs == 0 {
            let buf = std::mem::take(&mut s.buf);
            self.free_slots.push(slot);
            self.return_buf(buf);
        }
    }

    pub(crate) fn stats(&self) -> ScratchStats {
        self.stats
    }

    /// Trim excess retained buffers when returning a pool to [`super::Int8Plan`] after a huge batch.
    pub(crate) fn trim_for_checkin(&mut self) {
        while self.free_bufs.len() > MAX_FREE_BUFS || self.free_words > MAX_RETAINED_WORDS {
            let Some(buf) = self.free_bufs.pop() else { break };
            self.free_words = self.free_words.saturating_sub(buf.len());
            q8::recycle(buf);
        }
    }
}

impl Drop for ScratchPool {
    fn drop(&mut self) {
        for buf in self.free_bufs.drain(..) {
            q8::recycle(buf);
        }
        for slot in self.slots.drain(..) {
            if !slot.buf.is_empty() {
                q8::recycle(slot.buf);
            }
        }
    }
}

/// Ref-counted handle to one C4 activation buffer inside a [`ScratchPool`] (non-atomic, zero heap allocs).
pub(crate) struct SlotHandle {
    pool: Rc<RefCell<ScratchPool>>,
    slot: u16,
}

impl SlotHandle {
    pub(crate) fn new(pool: &Rc<RefCell<ScratchPool>>, buf: Vec<u32>) -> Self {
        let slot = pool.borrow_mut().insert_slot(buf);
        SlotHandle { pool: Rc::clone(pool), slot }
    }

    #[inline]
    pub(crate) fn slot(&self) -> u16 {
        self.slot
    }

    #[inline]
    pub(crate) fn pool(&self) -> &Rc<RefCell<ScratchPool>> {
        &self.pool
    }
}

impl Clone for SlotHandle {
    #[inline]
    fn clone(&self) -> Self {
        self.pool.borrow_mut().inc_ref(self.slot);
        SlotHandle { pool: Rc::clone(&self.pool), slot: self.slot }
    }
}

impl Drop for SlotHandle {
    #[inline]
    fn drop(&mut self) {
        self.pool.borrow_mut().dec_ref(self.slot);
    }
}
