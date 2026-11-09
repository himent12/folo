//! `cpu::in_pool`: the caller's grad mode and autocast state go with the closure, are restored after a panic, and
//! never reach jobs of the global rayon pool.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use folo_core::cpu::in_pool;
use folo_core::{autocast, autocast_dtype, is_grad_enabled, no_grad};
use rayon::prelude::*;

#[test]
fn carries_grad_mode_and_autocast() {
    assert!(in_pool(is_grad_enabled));
    {
        let _g = no_grad();
        assert!(!in_pool(is_grad_enabled));
        // Nested calls (now on a worker) run in place, under the same state.
        assert!(!in_pool(|| in_pool(is_grad_enabled)));
    }
    assert!(in_pool(is_grad_enabled));
    assert_eq!(in_pool(autocast_dtype), None);
    {
        let _c = autocast();
        assert_eq!(in_pool(autocast_dtype), autocast_dtype());
    }
    assert_eq!(in_pool(autocast_dtype), None);
}

#[test]
fn state_is_restored_after_a_panic() {
    for _ in 0..8 {
        let r = std::panic::catch_unwind(|| {
            let _g = no_grad();
            in_pool::<()>(|| panic!("in the pool"))
        });
        assert!(r.is_err());
        assert!(is_grad_enabled());
        // Whichever worker ran the panicking body, the next body sees its own caller's state.
        assert!(in_pool(is_grad_enabled));
    }
}

#[test]
fn global_pool_jobs_never_run_on_its_workers() {
    if rayon::current_num_threads() < 2 {
        return;
    }
    let stop = AtomicBool::new(false);
    let seen = AtomicUsize::new(0);
    std::thread::scope(|s| {
        // Inference under no_grad, waiting on its parallel work again and again (each wait lets its worker steal).
        s.spawn(|| {
            let _g = no_grad();
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_millis(300) {
                in_pool(|| {
                    (0..64).into_par_iter().for_each(|_| {
                        let t = Instant::now();
                        while t.elapsed() < Duration::from_micros(20) {}
                    })
                });
            }
            stop.store(true, Ordering::Relaxed);
        });
        // Meanwhile the global pool runs jobs that rely on the default state.
        while !stop.load(Ordering::Relaxed) {
            (0..256).into_par_iter().for_each(|_| {
                let name = std::thread::current().name().unwrap_or_default().to_string();
                assert!(!name.starts_with("folo-cpu"), "a global-pool job ran on {name}");
                assert!(is_grad_enabled(), "a global-pool job saw another job's grad mode");
                seen.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    assert!(seen.load(Ordering::Relaxed) > 0);
}
