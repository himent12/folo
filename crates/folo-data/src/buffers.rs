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

use std::cell::Cell;
use std::sync::Mutex;

/// The most bytes of free buffers the pool keeps.
pub const MAX_BYTES: usize = 512 << 20;

/// Buffers below this size are allocated and dropped as usual (the allocator reuses small blocks cheaply).
const MIN_LEN: usize = 64 << 10;

struct Pool {
    free: Vec<Vec<u8>>,
    bytes: usize,
    /// Threads inside [`pooled`].
    users: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { free: Vec::new(), bytes: 0, users: 0 });

thread_local! {
    static POOLED: Cell<bool> = const { Cell::new(false) };
}

fn lock() -> std::sync::MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|p| p.into_inner())
}

/// Runs `f` with buffer reuse on for the current thread.
pub(crate) fn pooled<R>(f: impl FnOnce() -> R) -> R {
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            POOLED.with(|p| p.set(self.0));
            if !self.0 {
                let mut pool = lock();
                pool.users -= 1;
                if pool.users == 0 {
                    pool.free = Vec::new();
                    pool.bytes = 0;
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
    let found = {
        let mut pool = lock();
        // The smallest free buffer that holds `len` without wasting more than half of it.
        let best = pool
            .free
            .iter()
            .enumerate()
            .filter(|(_, b)| b.capacity() >= len && b.capacity() <= 2 * len)
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i);
        best.map(|i| {
            let b = pool.free.swap_remove(i);
            pool.bytes -= b.capacity();
            b
        })
    };
    match found {
        Some(mut b) => {
            b.truncate(len);
            b.resize(len, 0);
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
    if buf.capacity() < MIN_LEN || !POOLED.with(|p| p.get()) {
        return;
    }
    let mut pool = lock();
    if pool.users > 0 && pool.bytes + buf.capacity() <= MAX_BYTES {
        pool.bytes += buf.capacity();
        pool.free.push(buf);
    }
}

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
            let small = scratch(len / 3);
            assert_ne!(small.as_ptr(), ptr);
        });
    }
}
