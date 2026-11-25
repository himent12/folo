//! Context, buffer pool, pipeline cache and batching behaviour.

use super::{Rng, gpu};
use std::sync::Arc;

use crate::context::parse_backends;
use crate::pipeline::{Access, KernelDef, Params, Source, Spec};
use crate::{DType, GpuTensor, UnaryOp};

#[test]
fn backend_parsing() {
    assert_eq!(parse_backends("vulkan"), Some(wgpu::Backends::VULKAN));
    assert_eq!(parse_backends(" DX12 , metal"), Some(wgpu::Backends::DX12 | wgpu::Backends::METAL));
    assert_eq!(parse_backends("nonsense"), None);
    assert_eq!(parse_backends(""), None);
}

#[test]
fn vulkan_adapters_report_their_memory() {
    // `/api/system` sizes runs against it (New run's GPU memory meter and its "won't fit" check).
    let Some(c) = super::ctx() else { return };
    let s = c.summary();
    if s.backend != "Vulkan" {
        eprintln!("note: {} is not driven through Vulkan, whose memory size wgpu cannot read", s.name);
        return;
    }
    assert!(s.memory_bytes >= 256 << 20, "{s}: {} bytes", s.memory_bytes);
}

#[test]
fn pool_reuses_freed_buffers() {
    let ctx = gpu!();
    // A size no other test uses, so concurrent tests do not steal the freed buffer.
    let bytes = 3 * 1024 * 1024 + 4 * 977;
    let a = ctx.alloc(bytes).unwrap();
    let size = a.size();
    assert!(size >= bytes);
    drop(a);
    let before = ctx.pool_stats();
    let b = ctx.alloc(bytes).unwrap();
    let after = ctx.pool_stats();
    assert_eq!(b.size(), size);
    assert!(after.hits > before.hits, "{before:?} -> {after:?}");
}

#[test]
fn release_memory_empties_the_pool_and_lowers_its_cap() {
    // A context of its own: releasing the shared one would empty the pool under concurrent tests that count on
    // reusing a buffer they just freed.
    let _skip_without_adapter = gpu!();
    let ctx = match crate::GpuContext::new() {
        Ok(c) => c,
        Err(e) => return eprintln!("note: skipping, no second context: {e}"),
    };
    let bytes = 5 * 1024 * 1024 + 4 * 1013;
    let held: Vec<_> = (0..4).map(|_| ctx.alloc(bytes).unwrap()).collect();
    let size = held[0].size();
    drop(held);
    assert!(ctx.pool_stats().cached_bytes >= 4 * size);
    let released = ctx.release_memory().unwrap();
    assert!(released >= 4 * size, "released {released}, cached {} before", 4 * size);
    // The buffers are gone: the same size is allocated afresh.
    let before = ctx.pool_stats();
    let again = ctx.alloc(bytes).unwrap();
    let after = ctx.pool_stats();
    assert!(after.allocated_bytes >= before.allocated_bytes + size, "{before:?} -> {after:?}");
    drop(again);
}

#[test]
fn a_cleared_pool_forgets_its_old_high_water_mark() {
    let ctx = gpu!();
    let usage = wgpu::BufferUsages::STORAGE;
    let mut pool = crate::pool::BufferPool::new(1024);
    let held: Vec<_> = (0..4).map(|_| pool.take(ctx.wgpu_device(), 2048, usage)).collect();
    for b in held {
        pool.give(b, 2048);
    }
    // A training step's worth (8 KiB live at once) stays cached although the cap is 1 KiB.
    assert_eq!((pool.stats().cached_bytes, pool.stats().need_bytes), (8192, 8192));
    pool.clear();
    // The clear forgets the need with the cached buffers.
    assert_eq!((pool.stats().cached_bytes, pool.stats().need_bytes), (0, 0));
    // After the clear the cap follows the new high-water mark (2 KiB), not the old 8 KiB: once the 2 KiB class has
    // left the need (two windows of 1 KiB requests), a 1 KiB buffer freed after a cached 2 KiB one evicts it (with the
    // old mark both would stay, 3 KiB).
    let b = pool.take(ctx.wgpu_device(), 2048, usage);
    pool.give(b, 2048);
    for _ in 0..2 * crate::pool::NEED_WINDOW {
        let b = pool.take(ctx.wgpu_device(), 1024, usage);
        pool.give(b, 1024);
    }
    assert_eq!(pool.stats().cached_bytes, 1024, "{:?}", pool.stats());
    // A buffer still lent out at the clear stays in the need, and only it.
    let held = pool.take(ctx.wgpu_device(), 4096, usage);
    assert_eq!(pool.stats().need_bytes, 1024 + 4096);
    pool.clear();
    assert_eq!((pool.stats().cached_bytes, pool.stats().need_bytes), (0, 4096), "{:?}", pool.stats());
    pool.give(held, 4096);
}

#[test]
fn pool_evicts_least_recently_freed_and_caches_up_to_the_high_water_mark() {
    let ctx = gpu!();
    let usage = wgpu::BufferUsages::STORAGE;
    let mut pool = crate::pool::BufferPool::new(4096);
    let (a, b, c) = (
        pool.take(ctx.wgpu_device(), 1024, usage),
        pool.take(ctx.wgpu_device(), 2048, usage),
        pool.take(ctx.wgpu_device(), 1536, usage),
    );
    // 4.5 KiB were live at once, so the cache may hold that much even though the cap is 4 KiB.
    pool.give(a, 1024);
    pool.give(b, 2048);
    pool.give(c, 1536);
    assert_eq!(pool.stats().cached_bytes, 4608);
    // Once the first three classes have left the need (two windows of 3 KiB requests), a 3 KiB buffer pushes out the
    // two least recently freed (1 KiB, then 2 KiB).
    for _ in 0..2 * crate::pool::NEED_WINDOW + 1 {
        let d = pool.take(ctx.wgpu_device(), 3072, usage);
        pool.give(d, 3072);
    }
    assert_eq!(pool.stats().cached_bytes, 1536 + 3072);
    let before = pool.stats();
    let _ = pool.take(ctx.wgpu_device(), 1536, usage);
    let _ = pool.take(ctx.wgpu_device(), 3072, usage);
    let _ = pool.take(ctx.wgpu_device(), 1024, usage);
    let after = pool.stats();
    assert_eq!((after.hits - before.hits, after.misses - before.misses), (2, 1));
    pool.clear();
    assert_eq!(pool.stats().cached_bytes, 0);
}

#[test]
fn upload_after_reuse_does_not_clobber_pending_reads() {
    let ctx = gpu!();
    let n = 4096 + 17;
    let x1 = vec![1.5f32; n];
    let a = GpuTensor::from_slice(&ctx, &x1, &[n]).unwrap();
    let y = a.unary(UnaryOp::Square).unwrap(); // encoded, not yet submitted
    drop(a); // its buffer goes back to the pool while `y` still depends on it
    let b = GpuTensor::from_slice(&ctx, &vec![-7.0f32; n], &[n]).unwrap(); // may reuse that buffer
    assert!(y.to_vec().unwrap().iter().all(|&v| v == 2.25));
    assert!(b.to_vec().unwrap().iter().all(|&v| v == -7.0));
}

#[test]
fn long_chains_span_several_batches() {
    let ctx = gpu!();
    let mut rng = Rng::new(40);
    let x = rng.vec(1000, -1.0, 1.0);
    let one = GpuTensor::from_slice(&ctx, &[1.0], &[1]).unwrap();
    let mut t = GpuTensor::from_slice(&ctx, &x, &[1000]).unwrap();
    // More dispatches than one batch holds, and more than one uniform chunk.
    for _ in 0..2500 {
        t = t.add(&one).unwrap();
    }
    ctx.sync().unwrap();
    let got = t.to_vec().unwrap();
    for (g, x) in got.iter().zip(&x) {
        assert!((g - (x + 2500.0)).abs() < 1e-3, "{g} vs {}", x + 2500.0);
    }
}

#[test]
fn pipeline_cache_hits() {
    static ADD_ONE: KernelDef = KernelDef {
        name: "test_add_one",
        storage: &[Access::ReadWrite],
        source: Source::Static(concat!(
            "@group(0) @binding(1) var<storage, read_write> X: array<f32>;\n",
            "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {\n",
            "    if (g.x < pu(0u)) { X[g.x] += f32(N); }\n",
            "}\n",
        )),
    };
    let ctx = gpu!();
    let k1 = ctx.kernel(&ADD_ONE, Spec::new().u32("N", 1)).unwrap();
    let k2 = ctx.kernel(&ADD_ONE, Spec::new().u32("N", 1)).unwrap();
    let k3 = ctx.kernel(&ADD_ONE, Spec::new().u32("N", 2)).unwrap();
    assert!(Arc::ptr_eq(&k1, &k2), "same key, same pipeline");
    assert!(!Arc::ptr_eq(&k1, &k3), "different constants, different pipeline");
    // The specialisation constant really changes the code.
    let t = GpuTensor::zeros(&ctx, &[100], DType::F32).unwrap();
    let p = Params::new().u(100);
    ctx.dispatch(&k1, &[t.raw()], &p, [2, 1, 1]).unwrap();
    ctx.dispatch(&k3, &[t.raw()], &p, [2, 1, 1]).unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![3.0; 100]);
    // A shader that fails to compile is an error, not a panic.
    static BROKEN: KernelDef =
        KernelDef { name: "test_broken", storage: &[], source: Source::Static("fn main() { nope }") };
    assert!(ctx.kernel(&BROKEN, Spec::new()).is_err());
}

#[test]
fn shape_and_dtype_errors() {
    let ctx = gpu!();
    assert!(GpuTensor::from_slice(&ctx, &[1.0, 2.0], &[3]).is_err());
    let t = GpuTensor::zeros(&ctx, &[2, 3], DType::F32).unwrap();
    assert!(t.reshape(&[4, 2]).is_err());
    assert_eq!(t.reshape(&[3, usize::MAX]).unwrap().shape(), &[3, 2]);
    assert_eq!(t.to_vec().unwrap(), vec![0.0; 6]);
    let h = t.to_dtype(DType::F16).unwrap();
    assert!(h.unary(UnaryOp::Relu).is_err());
    assert!(t.to_f16_vec().is_err());
    // Shapes whose element count (or any sub-product) overflows are rejected before allocating.
    assert!(GpuTensor::empty(&ctx, &[1 << 32, 1 << 32], DType::F32).is_err());
    assert!(GpuTensor::empty(&ctx, &[0, 1 << 40, 1 << 40], DType::F32).is_err());
    assert!(GpuTensor::empty(&ctx, &[1 << 40], DType::F32).is_err());
    assert!(GpuTensor::empty(&ctx, &[usize::MAX], DType::F32).is_err());
    assert!(ctx.alloc(u64::MAX - 3).is_err());
    assert!(t.reshape(&[0, 1 << 40, 1 << 40]).is_err());
    let empty = GpuTensor::zeros(&ctx, &[0, 4], DType::F32).unwrap();
    assert_eq!(empty.unary(UnaryOp::Exp).unwrap().to_vec().unwrap(), Vec::<f32>::new());
    assert_eq!(empty.sum(1, false).unwrap().shape(), &[0]);
}

/// `stage` on another thread, then `upload_staged`: the bytes arrive in stream order (after work encoded earlier
/// into the same pooled buffer), including a length that is not a multiple of 4.
#[test]
fn staged_upload_from_another_thread() {
    let ctx = gpu!();
    for len in [1usize, 7, 4096, 300_001] {
        let data: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        let staged = {
            let (c, d) = (ctx.clone(), data.clone());
            std::thread::spawn(move || c.stage(&d).unwrap()).join().unwrap()
        };
        assert_eq!(staged.len(), len as u64);
        let buf = ctx.upload_staged(&staged).unwrap();
        drop(staged);
        let back = ctx.read(buf.raw(), 0, len as u64).unwrap();
        assert_eq!(back, data, "length {len}");
    }
}

/// `upload_with`: bytes written in pieces into the mapped staging memory arrive, for lengths that are not a
/// multiple of 4; the second round reuses the staging buffers the first left mapped (one size class has two
/// lengths), and two uploads in a row without a wait each keep their own bytes.
#[test]
fn upload_with_reuses_mapped_staging() {
    let ctx = gpu!();
    for round in 0..2 {
        for len in [1usize, 7, 4096, 4093, 300_001] {
            let data: Vec<u8> = (0..len).map(|i| ((i + round) * 31 % 251) as u8).collect();
            let (buf, n) = ctx
                .upload_with(len as u64, |dst| {
                    let (mut a, mut b) = dst.split_at(len / 2);
                    a.copy_from_slice(&data[..len / 2]);
                    b.copy_from_slice(&data[len / 2..]);
                    len
                })
                .unwrap();
            assert_eq!(n, len);
            let back = ctx.read(buf.raw(), 0, len as u64).unwrap();
            assert_eq!(back, data, "round {round}, length {len}");
        }
    }
    // Two uploads back to back without a wait between them: each lands in its own buffer.
    let (a, ()) = ctx.upload_with(8, |mut d| d.copy_from_slice(&[1; 8])).unwrap();
    let (b, ()) = ctx.upload_with(8, |mut d| d.copy_from_slice(&[2; 8])).unwrap();
    assert_eq!(ctx.read(a.raw(), 0, 8).unwrap(), [1; 8]);
    assert_eq!(ctx.read(b.raw(), 0, 8).unwrap(), [2; 8]);
    assert!(ctx.release_memory().is_ok());
}

/// `release_memory` right after an upload (its staging buffer still being mapped again) keeps that buffer too:
/// it is dropped when the map completes, not put back on the idle list.
#[test]
fn release_memory_drops_staging_still_being_mapped() {
    let _shared = gpu!();
    let ctx = crate::GpuContext::new().unwrap();
    let (a, ()) = ctx.upload_with(64, |mut d| d.copy_from_slice(&[3; 64])).unwrap();
    ctx.sync().unwrap();
    assert_eq!(ctx.idle_staging(), 1, "an upload leaves its staging buffer mapped for reuse");
    let (b, ()) = ctx.upload_with(64, |mut d| d.copy_from_slice(&[4; 64])).unwrap();
    let (c, ()) = ctx.upload_with(1 << 20, |mut d| d.slice(..4).copy_from_slice(&[5; 4])).unwrap();
    ctx.release_memory().unwrap();
    ctx.sync().unwrap();
    assert_eq!(ctx.idle_staging(), 0, "no staging buffer survives release_memory");
    assert_eq!(ctx.read(a.raw(), 0, 64).unwrap(), [3; 64]);
    assert_eq!(ctx.read(b.raw(), 0, 64).unwrap(), [4; 64]);
    assert_eq!(ctx.read(c.raw(), 0, 4).unwrap(), [5; 4]);
    // Uploads after the release keep their staging buffers again.
    let _ = ctx.upload_with(64, |mut d| d.copy_from_slice(&[6; 64])).unwrap();
    ctx.sync().unwrap();
    assert_eq!(ctx.idle_staging(), 1);
}

/// A lost device (here: destroyed on purpose) makes waits and readbacks return `DeviceLost` instead of panicking
/// inside wgpu, and stays lost. Uses a private context so the shared one other tests use is untouched.
#[test]
fn lost_device_is_an_error_not_a_panic() {
    let _shared = gpu!();
    let is_lost = |r: crate::Result<()>| matches!(r, Err(crate::GpuError::DeviceLost(_)));
    // Each first call after the loss reports it (a fresh context per case, so no earlier call found it first).
    let fresh = || {
        let ctx = crate::GpuContext::new().unwrap();
        let buf = ctx.upload(&[1, 2, 3, 4]).unwrap();
        ctx.sync().unwrap();
        ctx.destroy_device();
        (ctx, buf)
    };
    let (ctx, _buf) = fresh();
    assert!(is_lost(ctx.sync()), "first sync");
    assert!(ctx.lost_reason().is_some());
    assert!(is_lost(ctx.sync()), "and every later one");
    let (ctx, buf) = fresh();
    assert!(is_lost(ctx.read(buf.raw(), 0, 4).map(|_| ())), "first readback");
    let (ctx, _buf) = fresh();
    assert!(is_lost(ctx.stage(&[7; 64]).map(|_| ())), "staging (the trainer's feeder thread)");
    // As folo-core sees it through the backend: recognisable, unlike other backend errors.
    let err = ctx.sync().unwrap_err();
    assert!(crate::is_device_lost(&folo_core::Error::from(err)));
    assert!(!crate::is_device_lost(&folo_core::Error::from(crate::GpuError::Wgpu("validation".into()))));
}

/// A readback's map outcome is loss-aware (the check the browser's async readback relies on, since it cannot
/// poll): once the loss is recorded, a failed and a succeeded map both report `DeviceLost`; on a healthy device a
/// failed map stays a `Map` error.
#[test]
fn map_outcome_prefers_a_recorded_loss() {
    let _shared = gpu!();
    let ctx = crate::GpuContext::new().unwrap();
    assert!(matches!(ctx.map_outcome(Err("rejected".into())), Err(crate::GpuError::Map(_))));
    assert!(ctx.map_outcome(Ok(())).is_ok());
    ctx.destroy_device();
    assert!(ctx.sync().is_err());
    assert!(matches!(ctx.map_outcome(Err("rejected".into())), Err(crate::GpuError::DeviceLost(_))));
    assert!(matches!(ctx.map_outcome(Ok(())), Err(crate::GpuError::DeviceLost(_))));
}

/// A profiling context (`FOLO_GPU_PROFILE=1`) whose device is lost: the timestamp readback at submission waits
/// through the loss-aware poll, so the wait reports `DeviceLost` instead of panicking inside wgpu with the batch
/// lock held, and the context stays usable (no poisoned lock).
#[test]
fn profiling_context_survives_device_loss() {
    let _shared = gpu!();
    let ctx = crate::GpuContext::with_options(&crate::GpuOptions {
        profile: true,
        ..crate::GpuOptions::from_env()
    })
    .unwrap();
    if !ctx.profiling() {
        eprintln!("adapter has no timestamp queries inside passes; skipping");
        return;
    }
    let a = ctx.upload(&[1u8; 256]).unwrap();
    let b = ctx.alloc(256).unwrap();
    ctx.copy(a.raw(), 0, b.raw(), 0, 256);
    ctx.sync().unwrap();
    assert!(ctx.profile_take().iter().any(|r| r.label == "copy"));
    ctx.copy(a.raw(), 0, b.raw(), 0, 256);
    ctx.destroy_device();
    let is_lost = |r: crate::Result<()>| matches!(r, Err(crate::GpuError::DeviceLost(_)));
    assert!(is_lost(ctx.sync()), "first sync");
    ctx.copy(a.raw(), 0, b.raw(), 0, 256);
    assert!(is_lost(ctx.sync()), "and a later one, on the same batch lock");
    let _ = ctx.profile_take();
}

/// Waits with every spin budget give the right results and take the path the budget calls for, counted in
/// [`crate::WorkStats`]: none blocks at once (also on an idle queue, which a poll would find empty), one far shorter
/// than milliseconds of GPU work runs out and falls back to blocking, and one far longer than the work ends by polling
/// as soon as the queue is empty, not when the budget runs out.
#[test]
fn waits_spin_then_block() {
    use std::time::Duration;
    let _shared = gpu!();
    let ctx = crate::GpuContext::new().unwrap();
    let expected = std::env::var(crate::context::ENV_WAIT_SPIN_US)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|us| Duration::from_nanos(us.saturating_mul(1000)))
        .unwrap_or(crate::context::DEFAULT_WAIT_SPIN);
    assert_eq!(ctx.wait_spin(), expected);
    let one = GpuTensor::from_slice(&ctx, &[1.0], &[1]).unwrap();
    // `adds` additions over `len` floats, then one `sync`: how it waited (spin waits, blocked waits), how long, and
    // whether the result is exact.
    let job = |spin_us: u64, len: usize, adds: usize| {
        ctx.set_wait_spin(Duration::from_micros(spin_us));
        let mut t = GpuTensor::from_slice(&ctx, &vec![0.5f32; len], &[len]).unwrap();
        for _ in 0..adds {
            t = t.add(&one).unwrap();
        }
        let before = ctx.work_stats();
        let start = std::time::Instant::now();
        ctx.sync().unwrap();
        let waited = start.elapsed();
        let after = ctx.work_stats();
        assert!(t.to_vec().unwrap().iter().all(|&v| v == 0.5 + adds as f32), "spin {spin_us} us");
        (after.spin_waits - before.spin_waits, after.blocked_waits - before.blocked_waits, waited)
    };

    let (spun, blocked, _) = job(0, 4096, 300);
    assert_eq!((spun, blocked), (0, 1), "no spin blocks at once");
    let before = ctx.work_stats();
    ctx.sync().unwrap();
    let after = ctx.work_stats();
    assert_eq!(
        (after.spin_waits - before.spin_waits, after.blocked_waits - before.blocked_waits),
        (0, 1),
        "no spin blocks at once on an idle queue too"
    );

    // 200 passes over 16 MiB read and write 6.4 GB: milliseconds of GPU work, far longer than the spin.
    let (spun, blocked, _) = job(1, 4 << 20, 200);
    assert_eq!((spun, blocked), (0, 1), "a 1 us spin runs out while the GPU works, then the wait blocks");

    let (spun, blocked, waited) = job(5_000_000, 4096, 50);
    assert_eq!((spun, blocked), (1, 0), "a 5 s spin ends by polling");
    assert!(waited < Duration::from_secs(2), "a 5 s spin ended when the queue emptied: {waited:?}");

    ctx.destroy_device();
    assert!(matches!(ctx.sync(), Err(crate::GpuError::DeviceLost(_))), "a lost device ends the spin");
}

/// `letterbox.wgsl` single-pass `vec3<u32>` RGB prefilter matches the exact integer CPU formula across all
/// 4 byte-alignment phases (`src_offset` 0..=3, including the final pixel of the buffer) and `fx > 1, fy > 1`.
#[test]
fn letterbox_rgb8_vec3_prefilter_matches_cpu_exact() {
    use crate::kernels::vision::{LetterboxDesc, letterbox_rgb8};

    let ctx = gpu!();
    let cpu_ref = |src: &[u8], d: &LetterboxDesc, tx: &[u32], ty: &[u32]| -> Vec<f32> {
        let [ow, oh] = d.out;
        let [sw, sh] = d.src;
        let [px, py] = d.pad;
        let [rw, rh] = d.resized;
        let [fx, fy] = d.prefilter;
        let src_px = |qx: usize, qy: usize, c: usize| -> u32 {
            if fx == 1 && fy == 1 {
                return u32::from(src[d.src_offset + (qy * sw + qx) * 3 + c]);
            }
            let (x0, x1) = (qx * fx, (qx * fx + fx).min(sw));
            let (y0, y1) = (qy * fy, (qy * fy + fy).min(sh));
            let mut s = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    s += u32::from(src[d.src_offset + (y * sw + x) * 3 + c]);
                }
            }
            let n = ((x1 - x0) * (y1 - y0)) as u32;
            (s + n / 2) / n
        };
        let plane = ow * oh;
        let mut out = vec![0.0f32; 3 * plane];
        for y in 0..oh {
            for x in 0..ow {
                let idx = y * ow + x;
                if x < px || x >= px + rw || y < py || y >= py + rh {
                    let v = f32::from(d.pad_value) * d.scale;
                    out[idx] = v;
                    out[plane + idx] = v;
                    out[2 * plane + idx] = v;
                    continue;
                }
                let (dx, dy) = (x - px, y - py);
                let (a, b, wx) = (tx[dx] as usize, tx[rw + dx] as usize, tx[2 * rw + dx]);
                let (ya, yb, wy) = (ty[dy] as usize, ty[rh + dy] as usize, ty[2 * rh + dy]);
                let (iwx, iwy) = (2048 - wx, 2048 - wy);
                for c in 0..3 {
                    let t = src_px(a, ya, c) * iwx + src_px(b, ya, c) * wx;
                    let u = src_px(a, yb, c) * iwx + src_px(b, yb, c) * wx;
                    let v = (t * iwy + u * wy + (1 << 21)) >> 22;
                    out[c * plane + idx] = v as f32 * d.scale;
                }
            }
        }
        out
    };

    // Case 1: fx = 1, fy = 1 across all 4 source byte alignments (src_offset = 0, 1, 2, 3), exact tight buffer
    // size so `rgb_at` is proven never to read past the final word even on the last pixel.
    // Case 2: fx = 3, fy = 2 box prefilter with non-multiple source dims (11 x 9) and padding.
    let configs = [
        ([5usize, 3], [8usize, 6], [1usize, 1], [6usize, 4], [1usize, 1], 0usize),
        ([5, 3], [8, 6], [1, 1], [6, 4], [1, 1], 1),
        ([5, 3], [8, 6], [1, 1], [6, 4], [1, 1], 2),
        ([5, 3], [8, 6], [1, 1], [6, 4], [1, 1], 3),
        ([11, 9], [8, 8], [1, 2], [6, 5], [3, 2], 2),
    ];
    for (ci, &(src_wh, out_wh, pad, resized, prefilter, src_offset)) in configs.iter().enumerate() {
        let [sw, sh] = src_wh;
        let [rw, rh] = resized;
        let [fx, fy] = prefilter;
        let pw = sw.div_ceil(fx);
        let ph = sh.div_ceil(fy);
        let raw_bytes = src_offset + sw * sh * 3;
        let padded_bytes = raw_bytes.next_multiple_of(4);
        let src_bytes: Vec<u8> = (0..padded_bytes).map(|i| ((i * 37 + ci * 53 + 11) % 256) as u8).collect();
        let make_taps = |n: usize, lim: usize| -> Vec<u32> {
            let mut a = Vec::with_capacity(n);
            let mut b = Vec::with_capacity(n);
            let mut w = Vec::with_capacity(n);
            for i in 0..n {
                let p0 = (i * (lim - 1)) / n.max(1);
                let p1 = (p0 + 1).min(lim - 1);
                a.push(p0 as u32);
                b.push(p1 as u32);
                w.push(((i * 571 + 123) % 2049) as u32);
            }
            [a, b, w].concat()
        };
        let tx = make_taps(rw, pw);
        let ty = make_taps(rh, ph);
        let taps: Vec<u32> = [&tx[..], &ty[..]].concat();
        let desc = LetterboxDesc {
            out: out_wh,
            src: src_wh,
            src_offset,
            pad,
            resized,
            prefilter,
            pad_value: 114,
            scale: 1.0 / 255.0,
            out_offset: 0,
            taps: 0,
        };
        let src_buf = ctx.upload(&src_bytes).unwrap();
        let taps_buf = ctx.upload(bytemuck::cast_slice(&taps)).unwrap();
        let out_elems = 3 * out_wh[0] * out_wh[1];
        let out_buf = ctx.alloc((out_elems * 4) as u64).unwrap();
        letterbox_rgb8(&ctx, src_buf.raw(), taps_buf.raw(), out_buf.raw(), &desc, DType::F32).unwrap();
        let got: Vec<f32> =
            bytemuck::cast_slice(&ctx.read(out_buf.raw(), 0, (out_elems * 4) as u64).unwrap()).to_vec();
        let want = cpu_ref(&src_bytes, &desc, &tx, &ty);
        assert_eq!(got, want, "letterbox case {ci} ({src_wh:?} prefilter {prefilter:?} off {src_offset})");
    }
}
