//! The `folo-desktop` binary: the app is the `folo_desktop` library (`lib.rs`); this file holds only what must live
//! in the executable.

/// Allocation counts for `--bench-frames` (`cargo build --release --features bench-alloc`).
#[cfg(feature = "bench-alloc")]
#[global_allocator]
static ALLOC: folo_ui::CountingAlloc = folo_ui::CountingAlloc;

fn main() {
    folo_desktop::run();
}
