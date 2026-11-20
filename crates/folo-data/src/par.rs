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
    for_bands(buf, row_len, |y0, band| {
        for (i, r) in band.chunks_mut(row_len).enumerate() {
            f(y0 + i, r);
        }
    });
}

/// Calls `f(y0, band)` for bands of whole `row_len`-sized rows of `buf` (`y0` is the band's first row), in parallel
/// when it is worth it: one band holding every row on a [`serial`] thread or for a small buffer, else bands of
/// about 16 KiB. For kernels that keep scratch buffers across the rows of a band.
pub(crate) fn for_bands<T, F>(buf: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if row_len == 0 {
        return;
    }
    if std::mem::size_of_val(buf) < PAR_THRESHOLD_BYTES || is_serial() {
        f(0, buf);
    } else {
        // Group rows so each task does a reasonable amount of work.
        let rows_per_task = (16 * 1024 / (row_len * std::mem::size_of::<T>())).max(1);
        buf.par_chunks_mut(row_len * rows_per_task)
            .enumerate()
            .for_each(|(t, chunk)| f(t * rows_per_task, chunk));
    }
}
