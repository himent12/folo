//! Reuse of image-sized byte buffers on the data loader's threads.
//!
//! A sample allocates a few image-sized buffers (decoded images, resized images, the warped output) and drops them
//! a few milliseconds later. A fresh buffer of a megabyte or more is fresh memory from the system on Windows, and
//! its first writes fault every page in; with 8 to 16 loader threads doing that at once the faults serialise in
//! the kernel (measured on our test machine: 0.5-1 ms for a 1.2 MB buffer at 8-16 threads against 0.05-0.1 ms for
//! the same buffer reused; docs/notes/hill/r6-loader.md). Inside [`pooled`] (the loader's workers and collator),
//! [`zeroed`] and [`scratch`] take a buffer from a process-wide pool and [`recycle`] gives one back; elsewhere they
//! are a plain allocation and a drop. The pool holds at most [`MAX_BYTES`] and is emptied when the last pooled
//! thread leaves [`pooled`], so it never outlives the loader's threads.
//!
//! Only where a buffer comes from changes: [`zeroed`] is all zeros as `vec![0; len]` is, and [`scratch`] is used
//! only for buffers that are overwritten whole, so the bytes a kernel writes do not depend on the pool.

use std::cell::{Cell, RefCell};
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The most bytes of free buffers the global pool keeps.
pub const MAX_BYTES: usize = 512 << 20;

/// Per-thread fast-path byte budget inside [`pooled`] before spilling to the global pool.
const TLS_POOL_MAX_BYTES: usize = 32 << 20;

/// Buffers below this size are allocated and dropped as usual (the allocator reuses small blocks cheaply).
const MIN_LEN: usize = 64 << 10;

/// Small reusable row/plan scratch limits per thread (kept warm across calls with zero mutex contention).
const TLS_SCRATCH_U8_SLOTS: usize = 4;
const TLS_SCRATCH_U8_MAX_BYTES: usize = 256 << 10;
const TLS_SCRATCH_U32_SLOTS: usize = 8;
const TLS_SCRATCH_U32_MAX_ELEMS: usize = 32 << 10;

struct Pool {
    free: Vec<Vec<u8>>,
    bytes: usize,
    /// Threads inside [`pooled`].
    users: usize,
}

#[derive(Default)]
struct TlsPool {
    free: Vec<Vec<u8>>,
    bytes: usize,
}

/// Maximum capacity of the per-thread reusable file-read buffer (keeps typical JPEGs warm with zero allocs).
const TLS_FILE_BUF_MAX_BYTES: usize = 4 << 20;

#[derive(Default)]
struct TlsScratch {
    u8s: Vec<Vec<u8>>,
    u32s: Vec<Vec<u32>>,
    file_buf: Vec<u8>,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { free: Vec::new(), bytes: 0, users: 0 });
/// Total bytes retained across the global pool and every thread's `TLS_POOL`, enforcing [`MAX_BYTES`] process-wide.
static TOTAL_POOLED_BYTES: AtomicUsize = AtomicUsize::new(0);

fn try_reserve_pooled(cap: usize) -> bool {
    let mut cur = TOTAL_POOLED_BYTES.load(Ordering::Relaxed);
    loop {
        let Some(next) = cur.checked_add(cap).filter(|&n| n <= MAX_BYTES) else {
            return false;
        };
        match TOTAL_POOLED_BYTES.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(now) => cur = now,
        }
    }
}

fn release_pooled(cap: usize) {
    if cap > 0 {
        TOTAL_POOLED_BYTES.fetch_sub(cap, Ordering::AcqRel);
    }
}

thread_local! {
    static POOLED: Cell<bool> = const { Cell::new(false) };
    static TLS_POOL: RefCell<TlsPool> = const { RefCell::new(TlsPool { free: Vec::new(), bytes: 0 }) };
    static TLS_SCRATCH: RefCell<TlsScratch> = const { RefCell::new(TlsScratch { u8s: Vec::new(), u32s: Vec::new(), file_buf: Vec::new() }) };
}

fn lock() -> std::sync::MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|p| p.into_inner())
}

fn take_from(free: &mut Vec<Vec<u8>>, bytes: &mut usize, len: usize) -> Option<Vec<u8>> {
    // The smallest free buffer that holds `len` without wasting more than half of it, preferring buffers whose
    // initialised length already covers `len` so reuse needs zero zeroing.
    let best = free
        .iter()
        .enumerate()
        .filter(|(_, b)| b.capacity() >= len && b.capacity() <= len.saturating_mul(2))
        .min_by_key(|(_, b)| (b.capacity(), usize::from(b.len() < len)))
        .map(|(i, _)| i)?;
    let b = free.swap_remove(best);
    *bytes -= b.capacity();
    release_pooled(b.capacity());
    Some(b)
}

/// Runs `f` with buffer reuse on for the current thread.
pub(crate) fn pooled<R>(f: impl FnOnce() -> R) -> R {
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            POOLED.with(|p| p.set(self.0));
            if !self.0 {
                TLS_POOL.with(|tp| {
                    let mut tp = tp.borrow_mut();
                    let released = std::mem::take(&mut tp.bytes);
                    tp.free = Vec::new();
                    release_pooled(released);
                });
                TLS_SCRATCH.with(|ts| {
                    let mut ts = ts.borrow_mut();
                    ts.u8s = Vec::new();
                    ts.u32s = Vec::new();
                    ts.file_buf = Vec::new();
                });
                let mut pool = lock();
                pool.users -= 1;
                if pool.users == 0 {
                    let released = std::mem::take(&mut pool.bytes);
                    pool.free = Vec::new();
                    release_pooled(released);
                }
            }
        }
    }
    let was = POOLED.with(|p| p.replace(true));
    if !was {
        lock().users += 1;
    }
    let _leave = Leave(was);
    f()
}

/// The capacity a new buffer of `len` bytes gets: `len` rounded up to `2^k · {1, 1.25, 1.5, 1.75}`, so images of
/// nearby sizes can use each other's buffers.
fn class(len: usize) -> usize {
    let p = len.next_power_of_two();
    let quarter = (p / 8).max(1);
    len.div_ceil(quarter) * quarter
}

/// A buffer of `len` bytes whose contents are unspecified (left over from an earlier use, or zeros): for a caller
/// that overwrites every byte.
pub(crate) fn scratch(len: usize) -> Vec<u8> {
    if len < MIN_LEN || !POOLED.with(|p| p.get()) {
        return vec![0; len];
    }
    let found = TLS_POOL.with(|tp| {
        let mut tp = tp.borrow_mut();
        let tp = &mut *tp;
        take_from(&mut tp.free, &mut tp.bytes, len)
    });
    let found = found.or_else(|| {
        let mut pool = lock();
        let pool = &mut *pool;
        take_from(&mut pool.free, &mut pool.bytes, len)
    });
    match found {
        Some(mut b) => {
            b.truncate(len);
            if b.len() < len {
                b.resize(len, 0);
            }
            b
        }
        None => {
            let mut b = Vec::with_capacity(class(len));
            b.resize(len, 0);
            b
        }
    }
}

/// A buffer of `len` zero bytes, as `vec![0; len]`.
pub(crate) fn zeroed(len: usize) -> Vec<u8> {
    let mut b = scratch(len);
    b.fill(0);
    b
}

/// Gives `buf` back for reuse (on a pooled thread; elsewhere it is dropped).
pub(crate) fn recycle(buf: Vec<u8>) {
    let cap = buf.capacity();
    if cap < MIN_LEN || !POOLED.with(|p| p.get()) {
        return;
    }
    if !try_reserve_pooled(cap) {
        return;
    }
    let spilled = TLS_POOL.with(|tp| {
        let mut tp = tp.borrow_mut();
        if tp.bytes.checked_add(cap).is_some_and(|t| t <= TLS_POOL_MAX_BYTES) {
            tp.bytes += cap;
            tp.free.push(buf);
            None
        } else {
            Some(buf)
        }
    });
    if let Some(buf) = spilled {
        let mut pool = lock();
        if pool.users > 0 && pool.bytes.checked_add(cap).is_some_and(|t| t <= MAX_BYTES) {
            pool.bytes += cap;
            pool.free.push(buf);
        } else {
            release_pooled(cap);
        }
    }
}

/// Reusable per-thread `u8` scratch buffer checked out from TLS and returned on drop without truncating its
/// initialised length, so steady-state row/band scratch incurs zero allocations and zero zeroing.
pub(crate) struct ScratchU8 {
    buf: Vec<u8>,
    len: usize,
}

impl ScratchU8 {
    /// Empty handle that performs no TLS checkout until grown via [`ScratchU8::ensure`].
    #[inline]
    pub(crate) const fn empty() -> Self {
        ScratchU8 { buf: Vec::new(), len: 0 }
    }

    /// Checks out a buffer with at least `len` accessible bytes (contents unspecified).
    #[inline]
    pub(crate) fn new(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        if len > TLS_SCRATCH_U8_MAX_BYTES {
            return ScratchU8 { buf: vec![0; len], len };
        }
        let mut buf = TLS_SCRATCH
            .try_with(|ts| {
                let mut ts = ts.borrow_mut();
                let best = ts
                    .u8s
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, v)| (usize::from(v.len() < len), v.capacity().abs_diff(len)))
                    .map(|(i, _)| i)?;
                Some(ts.u8s.swap_remove(best))
            })
            .ok()
            .flatten()
            .unwrap_or_default();
        if buf.len() < len {
            buf.resize(len, 0);
        }
        ScratchU8 { buf, len }
    }

    /// Ensures at least `len` bytes are accessible (contents unspecified) and returns `&mut [u8]` of length `len`.
    #[inline]
    pub(crate) fn ensure(&mut self, len: usize) -> &mut [u8] {
        if self.buf.is_empty() && len > 0 {
            *self = Self::new(len);
        } else {
            if self.buf.len() < len {
                self.buf.resize(len, 0);
            }
            self.len = len;
        }
        &mut self.buf[..len]
    }
}

impl Deref for ScratchU8 {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

impl DerefMut for ScratchU8 {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.buf[..self.len]
    }
}

impl Drop for ScratchU8 {
    fn drop(&mut self) {
        let cap = self.buf.capacity();
        if cap == 0 || cap > TLS_SCRATCH_U8_MAX_BYTES {
            return;
        }
        let buf = std::mem::take(&mut self.buf);
        let _ = TLS_SCRATCH.try_with(|ts| {
            let mut ts = ts.borrow_mut();
            if ts.u8s.len() < TLS_SCRATCH_U8_SLOTS {
                ts.u8s.push(buf);
            } else if let Some((idx, smallest)) = ts.u8s.iter().enumerate().min_by_key(|(_, v)| v.capacity())
                && smallest.capacity() < cap
            {
                ts.u8s[idx] = buf;
            }
        });
    }
}

/// Reusable per-thread `u32` scratch buffer (vertical blend rows, resize taps, box-filter accumulators) checked out
/// from TLS and returned on drop with its initialised length intact.
pub(crate) struct ScratchU32 {
    buf: Vec<u32>,
    len: usize,
}

impl ScratchU32 {
    /// Empty handle that performs no TLS checkout until grown via [`ScratchU32::ensure`].
    #[inline]
    pub(crate) const fn empty() -> Self {
        ScratchU32 { buf: Vec::new(), len: 0 }
    }

    /// Checks out a buffer with at least `len` accessible `u32`s (contents unspecified).
    #[inline]
    pub(crate) fn new(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        if len > TLS_SCRATCH_U32_MAX_ELEMS {
            return ScratchU32 { buf: vec![0; len], len };
        }
        let mut buf = TLS_SCRATCH
            .try_with(|ts| {
                let mut ts = ts.borrow_mut();
                let best = ts
                    .u32s
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, v)| (usize::from(v.len() < len), v.capacity().abs_diff(len)))
                    .map(|(i, _)| i)?;
                Some(ts.u32s.swap_remove(best))
            })
            .ok()
            .flatten()
            .unwrap_or_default();
        if buf.len() < len {
            buf.resize(len, 0);
        }
        ScratchU32 { buf, len }
    }

    /// Ensures at least `len` `u32`s are accessible (contents unspecified) and returns `&mut [u32]` of length `len`.
    #[inline]
    pub(crate) fn ensure(&mut self, len: usize) -> &mut [u32] {
        if self.buf.is_empty() && len > 0 {
            *self = Self::new(len);
        } else {
            if self.buf.len() < len {
                self.buf.resize(len, 0);
            }
            self.len = len;
        }
        &mut self.buf[..len]
    }
}

impl Deref for ScratchU32 {
    type Target = [u32];
    #[inline]
    fn deref(&self) -> &[u32] {
        &self.buf[..self.len]
    }
}

impl DerefMut for ScratchU32 {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u32] {
        &mut self.buf[..self.len]
    }
}

impl Drop for ScratchU32 {
    fn drop(&mut self) {
        let cap = self.buf.capacity();
        if cap == 0 || cap > TLS_SCRATCH_U32_MAX_ELEMS {
            return;
        }
        let buf = std::mem::take(&mut self.buf);
        let _ = TLS_SCRATCH.try_with(|ts| {
            let mut ts = ts.borrow_mut();
            if ts.u32s.len() < TLS_SCRATCH_U32_SLOTS {
                ts.u32s.push(buf);
            } else if let Some((idx, smallest)) = ts.u32s.iter().enumerate().min_by_key(|(_, v)| v.capacity())
                && smallest.capacity() < cap
            {
                ts.u32s[idx] = buf;
            }
        });
    }
}

/// Checks out a reusable per-thread `u8` scratch buffer of `len` bytes.
#[inline]
pub(crate) fn scratch_u8(len: usize) -> ScratchU8 {
    ScratchU8::new(len)
}

/// Checks out a reusable per-thread `u32` scratch buffer of `len` elements.
#[inline]
pub(crate) fn scratch_u32(len: usize) -> ScratchU32 {
    ScratchU32::new(len)
}

/// Reads `path` into a reusable per-thread byte buffer and passes the bytes to `f`, avoiding a fresh heap
/// allocation per image file on worker threads.
pub(crate) fn read_file_bytes<R>(
    path: &std::path::Path,
    f: impl FnOnce(&[u8]) -> crate::error::Result<R>,
) -> crate::error::Result<R> {
    use std::io::Read;
    let mut file = crate::guard::check(path)
        .and_then(|_| std::fs::File::open(path))
        .map_err(|e| crate::error::DataError::io(path, e))?;
    let mut buf =
        TLS_SCRATCH.try_with(|ts| std::mem::take(&mut ts.borrow_mut().file_buf)).unwrap_or_default();
    buf.clear();
    // `File::read_to_end` queries file size once and uses fallible `try_reserve(len + 1)`, so steady-state
    // reused buffers incur zero allocations and zero extra metadata syscalls while huge files fail cleanly.
    let read_res = file.read_to_end(&mut buf).map_err(|e| crate::error::DataError::io(path, e));
    let out = match read_res {
        Ok(_) => f(&buf),
        Err(e) => Err(e),
    };
    if buf.capacity() <= TLS_FILE_BUF_MAX_BYTES {
        buf.clear();
        let _ = TLS_SCRATCH.try_with(|ts| {
            let mut ts = ts.borrow_mut();
            if ts.file_buf.capacity() < buf.capacity() {
                ts.file_buf = buf;
            }
        });
    }
    out
}

#[cfg(test)]
pub(crate) static TEST_POOL_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_round_up_by_quarters() {
        assert_eq!(class(1 << 20), 1 << 20);
        assert_eq!(class((1 << 20) + 1), 5 << 18);
        assert_eq!(class(640 * 480 * 3), 1 << 20);
        assert_eq!(class(640 * 640 * 3), 5 << 18);
        for len in [MIN_LEN, 100_000, 921_600, 3_000_001] {
            assert!(class(len) >= len && class(len) < len + len / 4 + 1, "{len}");
        }
    }

    #[test]
    fn buffers_are_reused_only_inside_pooled() {
        let _guard = TEST_POOL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let len = 300_000;
        // Outside: plain allocations, nothing kept.
        let a = scratch(len);
        recycle(a);
        pooled(|| {
            let mut a = zeroed(len);
            a.fill(7);
            let ptr = a.as_ptr();
            recycle(a);
            // The same buffer comes back, zeroed by `zeroed`, and with its old bytes from `scratch`.
            let b = zeroed(len - 1000);
            assert_eq!((b.as_ptr(), b.len()), (ptr, len - 1000));
            assert!(b.iter().all(|&v| v == 0));
            recycle(b);
            let c = scratch(len);
            assert_eq!(c.as_ptr(), ptr);
            assert!(c[..len - 1000].iter().all(|&v| v == 0) && c.len() == len);
            // A buffer more than twice the size asked for is not handed out.
            recycle(c);
            assert!(TOTAL_POOLED_BYTES.load(Ordering::Acquire) >= len);
            let small = scratch(len / 3);
            assert_ne!(small.as_ptr(), ptr);
        });
        assert_eq!(TOTAL_POOLED_BYTES.load(Ordering::Acquire), 0);
    }

    #[test]
    fn thread_local_scratch_reuses_without_reallocation() {
        let p_u8 = {
            let mut s = scratch_u8(1920);
            s.fill(0xAB);
            s.as_ptr()
        };
        {
            let s = scratch_u8(1200);
            assert_eq!(s.as_ptr(), p_u8, "smaller checkout reuses existing ScratchU8 allocation");
            assert!(s.iter().all(|&b| b == 0xAB), "ScratchU8 preserves initialised bytes without re-zeroing");
        }
        {
            let s = scratch_u8(1920);
            assert_eq!(s.as_ptr(), p_u8, "return to prior len reuses without re-zeroing");
            assert!(s.iter().all(|&b| b == 0xAB));
        }
        let p_u32 = {
            let mut s = scratch_u32(4096);
            s.fill(0xCAFE_BABE);
            s.as_ptr()
        };
        {
            let s = scratch_u32(2048);
            assert_eq!(s.as_ptr(), p_u32, "smaller checkout reuses existing ScratchU32 allocation");
            assert!(s.iter().all(|&v| v == 0xCAFE_BABE));
        }
        {
            let s = scratch_u32(4096);
            assert_eq!(s.as_ptr(), p_u32);
            assert!(s.iter().all(|&v| v == 0xCAFE_BABE));
        }
    }

    #[test]
    fn read_file_bytes_reuses_tls_buffer_and_clears_on_leave_pooled() {
        let dir = folo_test_support::TestDir::new("folo-data-read-file-bytes");
        let big = dir.path().join("big.bin");
        let small = dir.path().join("small.bin");
        let big_data: Vec<u8> = (0..12_000usize).map(|i| (i.wrapping_mul(31) & 0xFF) as u8).collect();
        let small_data: Vec<u8> = (0..1_024usize).map(|i| (i.wrapping_mul(17) & 0xFF) as u8).collect();
        std::fs::write(&big, &big_data).unwrap();
        std::fs::write(&small, &small_data).unwrap();

        let _guard = TEST_POOL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        pooled(|| {
            let first_ptr = read_file_bytes(&big, |bytes| {
                assert_eq!(bytes, big_data.as_slice());
                Ok(bytes.as_ptr())
            })
            .unwrap();
            let second_ptr = read_file_bytes(&small, |bytes| {
                assert_eq!(bytes, small_data.as_slice());
                Ok(bytes.as_ptr())
            })
            .unwrap();
            assert_eq!(second_ptr, first_ptr, "smaller file read reuses TLS file_buf without reallocating");
            let third_ptr = read_file_bytes(&big, |bytes| {
                assert_eq!(bytes, big_data.as_slice());
                Ok(bytes.as_ptr())
            })
            .unwrap();
            assert_eq!(
                third_ptr, first_ptr,
                "equal-sized file read reuses TLS file_buf without reallocating"
            );
        });
        // Leaving `pooled` frees the thread-local file buffer.
        let cap_after = TLS_SCRATCH.with(|ts| ts.borrow().file_buf.capacity());
        assert_eq!(cap_after, 0);
    }
}
