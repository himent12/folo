use super::*;

const N: usize = MIN_POOLED;

fn put(p: &mut Pool, n: usize, limit: usize) -> usize {
    p.put(vec![0.0; n], limit).len()
}

/// `calls` calls that each take `k` buffers of length `n` and return them: the misses of each call.
fn workload(p: &mut Pool, n: usize, k: usize, calls: usize, limit: usize) -> Vec<usize> {
    (0..calls)
        .map(|_| {
            let bufs: Vec<Option<Vec<f32>>> = (0..k).map(|_| p.take(n)).collect();
            let misses = bufs.iter().filter(|b| b.is_none()).count();
            for b in bufs {
                p.put(b.unwrap_or_else(|| vec![0.0; n]), limit);
            }
            misses
        })
        .collect()
}

fn cached(p: &Pool) -> Vec<usize> {
    let mut v: Vec<usize> = p.free.iter().filter(|(_, c)| !c.bufs.is_empty()).map(|(&n, _)| n).collect();
    v.sort();
    v
}

/// A cache filled with one workload's lengths admits the next workload's once the old ones went unused for
/// `STALE_LIMITS` limits of traffic, and keeps them from then on: the next workload's calls stop allocating.
/// Three workloads in turn, each filling the cache: the second evicts every length the cache held.
#[test]
fn a_full_cache_admits_each_new_working_set() {
    let limit = 8 * N * 4;
    let mut p = Pool::new();
    assert_eq!(workload(&mut p, N, 8, 1, limit), [8]);
    assert_eq!(p.bytes, limit);
    for (n, k) in [(2 * N, 4), (4 * N, 2)] {
        let misses = workload(&mut p, n, k, 40, limit);
        assert_eq!(misses[0], k, "admitted at once into a full cache of recent lengths: {misses:?}");
        // Two limits of traffic: two calls of refused returns, then the old lengths are evicted.
        assert!(misses[..4].iter().sum::<usize>() < 4 * k, "never admitted: {misses:?}");
        assert!(misses[4..].iter().all(|&m| m == 0), "{misses:?}");
        assert_eq!(p.bytes, limit);
        assert_eq!(cached(&p), [n], "a stale length is still cached");
    }
}

/// A workload cycling through more lengths than the cache holds keeps a cache-sized part of them, also when a
/// cycle is far longer than the stale age (1.5 to 12 limits of traffic): a length that comes back after going
/// unused for a cycle is not evicted again for going unused that long. No buffer is evicted once settled.
#[test]
fn a_working_set_larger_than_the_cache_keeps_its_cached_part() {
    for m in [6, 12, 24, 48] {
        // Any 4 of the lengths fit, never 5.
        let limit = 4 * (N + m) * 4;
        let mut p = Pool::new();
        let (mut hits, mut gone) = (0, 0);
        for round in 0..40 {
            for n in N..N + m {
                let hit = p.take(n).is_some();
                let out = put(&mut p, n, limit);
                if round >= 30 {
                    hits += hit as usize;
                    gone += out;
                }
            }
        }
        // Only the missed lengths' returns leave the cache: nothing is evicted.
        assert_eq!((hits, gone), (10 * 4, 10 * (m - 4)), "{m} lengths: the last 10 cycles");
        assert_eq!(cached(&p).len(), 4);
    }
}

/// Returns that can never fit do not age the cache: they would evict the lengths the workload keeps reusing.
#[test]
fn a_return_larger_than_the_limit_ages_nothing() {
    let limit = 4 * N * 4;
    let mut p = Pool::new();
    let mut hits = 0;
    for _ in 0..100 {
        let small = p.take(N);
        hits += small.is_some() as usize;
        put(&mut p, N, limit);
        assert!(p.take(9 * N).is_none());
        assert_eq!(put(&mut p, 9 * N, limit), 1);
    }
    assert_eq!((hits, p.traffic), (99, 100 * N as u64 * 4));
}

/// The scan for stale lengths is skipped until the oldest kept length can be stale.
#[test]
fn a_full_cache_with_nothing_stale_does_not_rescan() {
    let limit = 2 * N * 4;
    let mut p = Pool::new();
    put(&mut p, N, limit);
    put(&mut p, N, limit);
    assert_eq!(put(&mut p, N + 1, limit), 1);
    assert_eq!(
        (p.scans, p.next_stale),
        (1, 2 * limit as u64 + 2 * N as u64 * 4),
        "N last returned at 8N bytes"
    );
    assert_eq!(put(&mut p, N + 1, limit), 1);
    assert_eq!(p.scans, 1, "rescanned before anything could be stale");
    // The return that brings the traffic to `next_stale` scans again, evicts both N buffers and is kept.
    while p.traffic + (N as u64 + 1) * 4 < p.next_stale {
        assert_eq!(put(&mut p, N + 1, limit), 1);
    }
    assert_eq!(p.scans, 1);
    assert_eq!(put(&mut p, N + 1, limit), 2, "the two N buffers leave");
    assert_eq!((p.scans, p.bytes), (2, (N + 1) * 4));
}

/// A smaller limit ages the lengths by the new limit at once, not after a scan deadline set from the old one.
#[test]
fn a_smaller_limit_takes_effect_on_the_stale_age() {
    let mut p = Pool::new();
    let big = 8 * N * 4;
    for _ in 0..8 {
        put(&mut p, N, big);
    }
    assert_eq!(put(&mut p, N + 1, big), 1);
    let small = 2 * N * 4;
    p.set_limit(small);
    put(&mut p, N, small);
    put(&mut p, N, small);
    let last = p.traffic;
    while put(&mut p, 2 * N, small) != 2 {
        assert!(p.traffic < last + 2 * small as u64 + small as u64, "still refused at {}", p.traffic);
    }
    assert_eq!(cached(&p), [2 * N]);
}

/// A limit of `usize::MAX` (no limit) caches everything and never overflows the traffic arithmetic.
#[test]
fn an_unbounded_limit_caches_everything() {
    let mut p = Pool::new();
    for i in 0..64 {
        assert_eq!(put(&mut p, N + i, usize::MAX), 0);
    }
    assert!(p.take(N).is_some());
    p.forget(usize::MAX);
    let mut out = Vec::new();
    p.evict_stale(usize::MAX, &mut out);
    assert!(out.is_empty());
    assert_eq!((p.bytes, p.next_forget, p.next_stale), ((63 * N + 64 * 63 / 2) * 4, u64::MAX, u64::MAX));
}

/// Records of lengths that come and go (each new input shape) are dropped once forgotten.
#[test]
fn lengths_used_once_are_forgotten() {
    let limit = 4 * N * 4;
    let mut p = Pool::new();
    for i in 0..2000 {
        put(&mut p, N + i, limit);
    }
    // `FORGET_LIMITS` limits of traffic at 4 returns per limit, plus a sweep's slack.
    assert!(p.free.len() <= FORGET_LIMITS as usize * 4 + 8, "{} records", p.free.len());
}

/// Thread-local magazine hits without touching the shared pool, spills oldest buffers to the shared pool
/// once full, and `clear()` / `set_limit(0)` immediately releases buffers held on other threads.
#[test]
fn tls_magazine_hits_spills_clears_and_reclaims_on_thread_exit() {
    static TEST_CACHE: Cache = Cache::new(1 << 24);
    let slot = TEST_CACHE.register_slot();

    // 1. Recycle & take on the same slot hits TLS without populating the shared pool.
    let mut v = vec![1.25f32; N];
    let ptr = v.as_ptr();
    TEST_CACHE.recycle_with_slot(Some(&slot), v);
    assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), N * 4);
    assert_eq!(TEST_CACHE.bytes(), N * 4);
    v = TEST_CACHE.take_with_slot(Some(&slot), N).expect("TLS hit");
    assert_eq!(v.as_ptr(), ptr, "reused exact buffer from TLS magazine");
    assert_eq!(TEST_CACHE.bytes(), 0);

    // 2. Filling past `TLS_MAX_BUFS` spills the oldest buffers into the shared pool, so another thread (or
    //    slot-less take) can still reuse them while the newest stay in TLS.
    for i in 0..TLS_MAX_BUFS + 3 {
        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N + i]);
    }
    let expected_tls: usize = (3..TLS_MAX_BUFS + 3).map(|i| (N + i) * 4).sum();
    let expected_pool: usize = (0..3).map(|i| (N + i) * 4).sum();
    assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), expected_tls);
    assert_eq!(TEST_CACHE.bytes() - expected_tls, expected_pool);
    assert!(TEST_CACHE.take_with_slot(None, N).is_some(), "spilled oldest length found in shared pool");
    assert!(TEST_CACHE.take_with_slot(Some(&slot), N + TLS_MAX_BUFS + 2).is_some(), "newest length in TLS");

    // 3. `clear()` releases buffers cached on a separate live worker thread immediately.
    let (tx_ready, rx_ready) = std::sync::mpsc::channel();
    let (tx_exit, rx_exit) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let wslot = TEST_CACHE.register_slot();
        TEST_CACHE.recycle_with_slot(Some(&wslot), vec![0.0; N * 2]);
        tx_ready.send(()).unwrap();
        rx_exit.recv().unwrap();
        // Recycle one more buffer, then exit without clearing so `Drop for Slot` must reclaim it.
        TEST_CACHE.recycle_with_slot(Some(&wslot), vec![0.0; N * 3]);
    });
    rx_ready.recv().unwrap();
    assert!(TEST_CACHE.bytes() >= N * 2 * 4);
    TEST_CACHE.clear();
    assert_eq!(TEST_CACHE.bytes(), 0, "clear() must drain live worker TLS slots immediately");
    tx_exit.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(TEST_CACHE.bytes(), 0, "exiting thread must release its TLS bytes on Slot::drop");

    // 4. `set_limit(0)` shrinks immediately and refuses subsequent recycles; restoring limit re-enables caching.
    TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
    assert_eq!(TEST_CACHE.bytes(), N * 4);
    TEST_CACHE.set_limit(0);
    assert_eq!(TEST_CACHE.bytes(), 0);
    TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
    assert_eq!(TEST_CACHE.bytes(), 0);

    // 5. Small custom limit: both TLS (`N`) and pool (`3 * N`) can coexist up to `4 * N`, and when a `3 * N`
    //    buffer arrives while TLS holds `2 * N` (leaving only `2 * N` for the pool), TLS is drained once so
    //    the `3 * N` buffer still fits in the shared pool.
    let small_limit = 4 * N * 4;
    TEST_CACHE.set_limit(small_limit);
    TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
    assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), N * 4);
    TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 3 * N]);
    assert_eq!(TEST_CACHE.bytes(), 4 * N * 4);
    assert!(TEST_CACHE.take_with_slot(Some(&slot), 3 * N).is_some());
    assert!(TEST_CACHE.take_with_slot(Some(&slot), N).is_some());
    assert_eq!(TEST_CACHE.bytes(), 0);

    TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 2 * N]);
    assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), 2 * N * 4);
    TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 3 * N]);
    assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), 0);
    assert_eq!(TEST_CACHE.bytes(), 3 * N * 4);
    assert!(TEST_CACHE.take_with_slot(Some(&slot), 3 * N).is_some());
    TEST_CACHE.clear();
}

/// Concurrent TLS + pool returns never exceed `limit`, `set_limit` reductions drain in-flight admissions
/// across both levels, pre-clear spills cannot evict post-clear TLS entries, and `clear()`'s epoch bump
/// invalidates both levels atomically.
#[test]
fn cache_budget_limit_reduction_and_clear_are_atomic_across_levels() {
    // 1. Atomic clear across levels: bumping `epoch` before draining must miss BOTH TLS and Pool lookups.
    static CLEAR_CACHE: Cache = Cache::new(8 * N * 4);
    let slot = CLEAR_CACHE.register_slot();
    CLEAR_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
    CLEAR_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 5 * N]);
    assert_eq!((CLEAR_CACHE.tls_bytes.load(Ordering::Acquire), CLEAR_CACHE.bytes()), (N * 4, 6 * N * 4));
    CLEAR_CACHE.epoch.fetch_add(1, Ordering::AcqRel);
    assert!(CLEAR_CACHE.take_with_slot(Some(&slot), N).is_none());
    assert!(CLEAR_CACHE.take_with_slot(Some(&slot), 5 * N).is_none());
    assert_eq!(CLEAR_CACHE.bytes(), 0);

    // 2. Pre-clear `put_global` with an old epoch must NOT drain buffers cached in TLS after `clear()`.
    let old_epoch = CLEAR_CACHE.epoch.load(Ordering::Acquire);
    CLEAR_CACHE.clear();
    CLEAR_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 2 * N]);
    assert_eq!(CLEAR_CACHE.tls_bytes.load(Ordering::Acquire), 2 * N * 4);
    CLEAR_CACHE.put_global([vec![0.0; 7 * N]], old_epoch);
    assert_eq!(
        CLEAR_CACHE.tls_bytes.load(Ordering::Acquire),
        2 * N * 4,
        "stale spill drained post-clear TLS"
    );
    assert!(CLEAR_CACHE.take_with_slot(Some(&slot), 2 * N).is_some());
    assert_eq!(CLEAR_CACHE.bytes(), 0);

    // 3. Concurrent TLS (`4 * N`) + Pool (`6 * N`) returns never exceed `limit = 8 * N`.
    const LIMIT: usize = 8 * N * 4;
    static RACE_CACHE: Cache = Cache::new(LIMIT);
    let threads: Vec<_> = (0..4)
        .map(|tid| {
            std::thread::spawn(move || {
                let s = RACE_CACHE.register_slot();
                for step in 0..200 {
                    let len = if (tid + step) % 2 == 0 { 4 * N } else { 6 * N };
                    RACE_CACHE.recycle_with_slot(Some(&s), vec![0.0; len]);
                    assert!(RACE_CACHE.bytes() <= LIMIT, "{} > {LIMIT}", RACE_CACHE.bytes());
                    let _ = RACE_CACHE.take_with_slot(Some(&s), len);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    RACE_CACHE.clear();
    assert_eq!(RACE_CACHE.bytes(), 0);

    // 4. `set_limit` reduction serializes on every slot lock and drains an in-flight old-limit TLS admission.
    static LIMIT_CACHE: Cache = Cache::new(8 * N * 4);
    let wslot = LIMIT_CACHE.register_slot();
    let wslot_bg = wslot.clone();
    let (tx_locked, rx_locked) = std::sync::mpsc::channel();
    let (tx_go, rx_go) = std::sync::mpsc::channel();
    let t_in_flight = std::thread::spawn(move || {
        let mut s = wslot_bg.lock().unwrap();
        let (cur_epoch, old_limit) =
            (LIMIT_CACHE.epoch.load(Ordering::Acquire), LIMIT_CACHE.limit.load(Ordering::Acquire));
        tx_locked.send(()).unwrap();
        rx_go.recv().unwrap();
        let (mut spill, mut stale) = (Vec::new(), Vec::new());
        s.sync_epoch(cur_epoch, &mut stale);
        assert!(s.put(vec![0.0; 3 * N], old_limit, &mut spill).is_none());
    });
    rx_locked.recv().unwrap();
    let t_shrink = std::thread::spawn(|| LIMIT_CACHE.set_limit(2 * N * 4));
    while LIMIT_CACHE.limit.load(Ordering::Acquire) != 2 * N * 4 {
        std::thread::yield_now();
    }
    tx_go.send(()).unwrap();
    t_in_flight.join().unwrap();
    t_shrink.join().unwrap();
    assert!(LIMIT_CACHE.bytes() <= 2 * N * 4, "survived shrink: {}", LIMIT_CACHE.bytes());
    LIMIT_CACHE.clear();
    drop(wslot);

    // 5. `set_limit` enforces combined `total_bytes <= bytes` under `pool.lock()` even when an in-flight
    //    `put_global` admits a pool buffer `<= new_limit` concurrently with a post-sweep TLS admission.
    static COMBO_CACHE: Cache = Cache::new(8 * N * 4);
    let cslot = COMBO_CACHE.register_slot();
    let (tx_pool_held, rx_pool_held) = std::sync::mpsc::channel();
    let (tx_pool_go, rx_pool_go) = std::sync::mpsc::channel();
    let t_pool = std::thread::spawn(move || {
        let mut g = COMBO_CACHE.pool.lock().unwrap();
        let old_limit = COMBO_CACHE.limit.load(Ordering::Acquire);
        let cur_epoch = COMBO_CACHE.epoch.load(Ordering::Acquire);
        g.sync_epoch(cur_epoch, &COMBO_CACHE.total_bytes);
        tx_pool_held.send(()).unwrap();
        rx_pool_go.recv().unwrap();
        let p = g.pool.get_or_insert_with(Pool::new);
        assert!(p.put_with_budget(vec![0.0; 5 * N], old_limit, Some(&COMBO_CACHE.total_bytes)).is_empty());
    });
    rx_pool_held.recv().unwrap();
    let t_shrink2 = std::thread::spawn(|| COMBO_CACHE.set_limit(5 * N * 4));
    while COMBO_CACHE.limit.load(Ordering::Acquire) != 5 * N * 4 {
        std::thread::yield_now();
    }
    // Admit `2 * N` into TLS under the new `5 * N` limit while `t_pool` still holds `pool.lock()`.
    COMBO_CACHE.recycle_with_slot(Some(&cslot), vec![0.0; 2 * N]);
    tx_pool_go.send(()).unwrap();
    t_pool.join().unwrap();
    t_shrink2.join().unwrap();
    assert!(COMBO_CACHE.bytes() <= 5 * N * 4, "combined TLS + pool exceeded shrink: {}", COMBO_CACHE.bytes());
    COMBO_CACHE.clear();
    drop(cslot);
}
