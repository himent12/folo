//! Row parallelism that can be switched off per thread.
//!
//! Image kernels (resize, warp, HSV, collation) split work by rows with rayon. Inside the [`crate::DataLoader`] the
//! worker threads already provide the parallelism, so they mark themselves serial and the kernels run single-threaded
//! there instead of oversubscribing the machine.

use rayon::prelude::*;
use std::cell::Cell;

thread_local! {
    static SERIAL: Cell<bool> = const { Cell::new(false) };
}

/// Work (in output bytes) below which kernels never bother with rayon.
const PAR_THRESHOLD_BYTES: usize = 64 * 1024;

/// Runs `f` with row parallelism disabled on the current thread.
pub fn serial<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SERIAL.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SERIAL.with(|s| s.replace(true)));
    f()
}

/// Whether the current thread is in [`serial`] mode.
pub fn is_serial() -> bool {
    SERIAL.with(|s| s.get())
}

/// Calls `f(y, row)` for each `row_len`-sized chunk of `buf`, in parallel when it is worth it.
pub(crate) fn for_rows<T, F>(buf: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if row_len == 0 {
        return;
    }
    if std::mem::size_of_val(buf) < PAR_THRESHOLD_BYTES || is_serial() {
        buf.chunks_mut(row_len).enumerate().for_each(|(y, r)| f(y, r));
    } else {
        // Group rows so each task does a reasonable amount of work.
        let rows_per_task = (16 * 1024 / (row_len * std::mem::size_of::<T>())).max(1);
        buf.par_chunks_mut(row_len * rows_per_task).enumerate().for_each(|(t, chunk)| {
            for (i, r) in chunk.chunks_mut(row_len).enumerate() {
                f(t * rows_per_task + i, r);
            }
        });
    }
}
