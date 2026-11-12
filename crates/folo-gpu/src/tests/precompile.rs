//! Parallel precompile on a real device: a context records the kernels it used, the next one compiles them on
//! worker threads before they are asked for, falls back to the newest valid list of another shape, and a corrupt,
//! stale or broken list only means compiling lazily.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::gpu;
use crate::kernel_list;
use crate::pipeline::Spec;
use crate::{BinaryOp, GpuContext, GpuOptions, GpuTensor, UnaryOp};
use folo_test_support::TestDir;

fn context(dir: &Path) -> GpuContext {
    GpuContext::with_options(&GpuOptions {
        pipeline_cache_dir: Some(dir.to_path_buf()),
        ..GpuOptions::from_env()
    })
    .unwrap()
}

/// Three kernels (unary, binary, GEMM) with `extra` adding a fourth; checks the results.
fn work(ctx: &GpuContext, extra: bool) {
    let x = GpuTensor::from_slice(ctx, &[1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
    let y = x.unary(UnaryOp::Abs).unwrap().binary(BinaryOp::Add, &x).unwrap();
    assert_eq!(y.to_vec().unwrap(), [2.0, 0.0, 6.0, 0.0]);
    assert_eq!(y.matmul(&x).unwrap().to_vec().unwrap(), [2.0, -4.0, 6.0, -12.0]);
    if extra {
        assert_eq!(x.unary(UnaryOp::Neg).unwrap().to_vec().unwrap(), [-1.0, 2.0, -3.0, 4.0]);
    }
}

/// The kernels of `ctx`'s list for `kind` / `shape`, by name.
fn listed(ctx: &GpuContext, kind: &str, shape: &str) -> Vec<&'static str> {
    read(ctx, &ctx.kernel_list_file(kind, shape).unwrap(), kind)
}

/// The kernels of the list file at `path` for `kind`, by name.
fn read(ctx: &GpuContext, path: &Path, kind: &str) -> Vec<&'static str> {
    let id = kernel_list::identity_for(&ctx.adapter().get_info(), kind);
    kernel_list::read(path, &id).iter().map(|k| k.name).collect()
}

/// The list file of `kind` at `shape` in `dir`.
fn list_path(ctx: &GpuContext, dir: &Path, kind: &str, shape: &str) -> PathBuf {
    kernel_list::path_for(dir, &ctx.adapter().get_info(), kind, shape)
}

/// Sets `path`'s modification time to `secs` after an arbitrary fixed point.
fn set_mtime(path: &Path, secs: u64) -> SystemTime {
    let t = SystemTime::UNIX_EPOCH + Duration::from_secs(1_600_000_000 + secs);
    std::fs::File::options().write(true).open(path).unwrap().set_modified(t).unwrap();
    t
}

fn mtime(path: &Path) -> SystemTime {
    std::fs::metadata(path).unwrap().modified().unwrap()
}

#[test]
fn kernels_are_recorded_and_precompiled_next_time() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile");
    // No list yet: nothing to compile ahead; the run's kernels are recorded in order of first use.
    let a = context(&dir);
    assert_eq!(a.precompile("test", "s1"), 0);
    assert_eq!(a.precompile("test", "s1"), 0, "a second call does nothing");
    work(&a, false);
    a.save_kernel_lists();
    let first = listed(&a, "test", "s1");
    assert!(first.len() >= 3, "{first:?}");
    assert!(first.contains(&"gemm"), "{first:?}");
    assert!(a.kernels_without_registry().is_empty(), "{:?}", a.kernels_without_registry());
    drop(a);

    // The next context compiles them on the workers, and every first use takes the worker's pipeline.
    let b = context(&dir);
    let n = b.precompile("test", "s1");
    assert_eq!(n, first.len());
    b.precompile_wait();
    work(&b, false);
    assert_eq!(b.precompiled_taken(), n as u64);
    let file = b.kernel_list_file("test", "s1").unwrap();
    let written = set_mtime(&file, 1);
    b.save_kernel_lists();
    drop(b);
    assert_eq!(mtime(&file), written, "nothing new: the list is not rewritten");

    // A shorter run keeps the longer list; a run with a new kernel rewrites it with what it used.
    let c = context(&dir);
    c.precompile("test", "s1");
    let x = GpuTensor::from_slice(&c, &[1.0, -2.0, 3.0, -4.0], &[2, 2]).unwrap();
    x.unary(UnaryOp::Abs).unwrap().to_vec().unwrap();
    c.save_kernel_lists();
    assert_eq!(mtime(&file), written, "a run that used fewer kernels keeps the list");
    work(&c, true);
    c.save_kernel_lists();
    assert_ne!(mtime(&file), written);
    assert_eq!(listed(&c, "test", "s1").len(), first.len() + 1);
    drop(c);

    // Another shape of the same kind starts from that list; another kind has none.
    let d = context(&dir);
    assert_eq!(d.precompile("test", "s2"), first.len() + 1);
    assert_eq!(d.precompile("other", "s1"), 0);
    d.precompile_wait();
    work(&d, false);
    d.save_kernel_lists();
    assert_eq!(listed(&d, "test", "s2").len(), first.len(), "its own list holds what it used");
    // The neighbour's kernel this run never used stays finished in its slot until memory is released.
    assert_eq!(d.precompile_slots(), 1);
    d.release_memory().unwrap();
    assert_eq!(d.precompile_slots(), 0);
}

/// One context running two shapes: the second shape's recording holds the kernels the first compiled, and a kernel
/// first used after both opened joins both.
#[test]
fn shared_kernels_join_every_recording() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile-shared");
    let ctx = context(&dir);
    ctx.precompile("test", "a");
    work(&ctx, false);
    let a = ctx.recorded_kernels("test", "a");
    assert!(a.len() >= 3, "{a:?}");
    ctx.precompile("test", "b");
    assert!(ctx.recorded_kernels("test", "b").is_empty(), "nothing used yet");
    work(&ctx, false);
    assert_eq!(ctx.recorded_kernels("test", "b"), a, "cache hits are recorded, in order of use");
    work(&ctx, true);
    let both = [ctx.recorded_kernels("test", "a"), ctx.recorded_kernels("test", "b")];
    assert_eq!(both[0].len(), a.len() + 1);
    assert_eq!(both[0], both[1]);
    ctx.save_kernel_lists();
    assert_eq!(listed(&ctx, "test", "b"), both[1]);
}

/// A context keeps the most recently asked for recordings; an older one is written and dropped.
#[test]
fn old_recordings_are_written_and_dropped() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile-cap");
    let ctx = context(&dir);
    ctx.precompile("test", "s0");
    work(&ctx, false);
    for i in 1..16 {
        ctx.precompile("test", &format!("s{i}"));
    }
    assert!(ctx.kernel_list_file("test", "s0").is_some(), "sixteen are kept");
    ctx.precompile("test", "s1");
    ctx.precompile("test", "s16");
    assert!(ctx.kernel_list_file("test", "s0").is_none(), "the oldest is dropped");
    ctx.precompile("test", "s17");
    assert!(ctx.kernel_list_file("test", "s1").is_some(), "asked for again: kept as the most recent");
    assert!(ctx.kernel_list_file("test", "s2").is_none());
    let s0 = list_path(&ctx, &dir, "test", "s0");
    assert!(read(&ctx, &s0, "test").len() >= 3, "a dropped recording is written");
}

/// With no list for its shape, a run takes the most recently written valid list of its kind (a corrupt newer one
/// is passed over).
#[test]
fn the_newest_valid_neighbour_is_used() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile-neighbours");
    let mut lens = Vec::new();
    for (shape, extra) in [("short", false), ("long", true)] {
        let ctx = context(&dir);
        ctx.precompile("test", shape);
        work(&ctx, extra);
        ctx.save_kernel_lists();
        lens.push(listed(&ctx, "test", shape).len());
    }
    let (short, long) = (lens[0], lens[1]);
    assert_eq!(long, short + 1, "Neg");
    let ctx = context(&dir);
    let corrupt = list_path(&ctx, &dir, "test", "corrupt");
    std::fs::write(&corrupt, b"FOLOKLST not a list").unwrap();
    let (p_short, p_long) = (list_path(&ctx, &dir, "test", "short"), list_path(&ctx, &dir, "test", "long"));
    drop(ctx);

    for (newer, n) in [(&p_long, long), (&p_short, short)] {
        let older = if newer == &p_long { &p_short } else { &p_long };
        set_mtime(older, 1);
        set_mtime(newer, 2);
        set_mtime(&corrupt, 3);
        let ctx = context(&dir);
        assert_eq!(ctx.precompile("test", "new"), n, "{newer:?}");
        ctx.precompile_wait();
    }
}

#[test]
fn bad_lists_only_mean_compiling_lazily() {
    let _shared = gpu!();
    let dir = TestDir::new("gpu-precompile-bad");
    let a = context(&dir);
    a.precompile("test", "s");
    work(&a, false);
    a.save_kernel_lists();
    let file = a.kernel_list_file("test", "s").unwrap();
    let id = kernel_list::identity_for(&a.adapter().get_info(), "test");
    let n = kernel_list::read(&file, &id).len();
    drop(a);

    // Corrupt: ignored, and rewritten valid by the run.
    let good = std::fs::read(&file).unwrap();
    let mut bad = good.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x55;
    std::fs::write(&file, &bad).unwrap();
    let b = context(&dir);
    assert_eq!(b.precompile("test", "s"), 0);
    work(&b, false);
    b.save_kernel_lists();
    assert_eq!(kernel_list::read(&file, &id).len(), n);
    drop(b);

    // Stale and broken lines: an unknown kernel and constants no call site sets (a generator that would emit
    // billions of statements) are skipped; one whose constants do not build (a constant named like the kernel's
    // storage array) fails on a worker without failing anything else.
    let mut text = String::from_utf8(kernel_list::decode_payload_for_test(&good, &id)).unwrap();
    text.push_str("no_such_kernel - X=u1\nconv_split_finish - S=u4294967295\nconv_split_finish - S=u65536\nfill - Y=u1\n");
    std::fs::write(&file, kernel_list::encode_text_for_test(&text, &id)).unwrap();
    let c = context(&dir);
    assert_eq!(c.precompile("test", "s"), n + 1);
    c.precompile_wait();
    work(&c, false);
    assert_eq!(c.precompiled_taken(), n as u64);
    // The kernel that failed on the worker is compiled again where it is asked for, which reports the error.
    let spec = Spec::new().u32("Y", 1);
    let def = crate::kernels::find_def("fill", &spec).unwrap();
    let err = c.kernel(def, spec).err().expect("the broken kernel fails to compile");
    assert!(err.to_string().contains("compiling kernel fill"), "{err}");
    assert_eq!(c.precompiled_taken(), n as u64);
}
