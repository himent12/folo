//! Image fetches when a grid is flung through a big dataset (stress x4-soak-2): every tile passed
//! asks for its image; the ones left in view must not wait for all the others.

use std::sync::{Arc, Mutex};

use folo_ui::Context;

use super::FoloApp;
use super::tests::{drive, open};
use crate::api::ImageData;
use crate::store::{ImageCache, ImgKey, ImgKind, ImgSlot, LoadQueue};

/// The demo dataset: its images (`img_00001` on) load from the mock backend when drawn.
const DS: &str = "ds_3b91";

fn key(i: usize) -> ImgKey {
    ImgKey { dataset: DS.into(), image: format!("img_{:05}", i + 1), kind: ImgKind::Thumb }
}

const MAX: usize = LoadQueue::<()>::MAX_IN_FLIGHT;

/// Asks for tiles `0..tiles` in order, as a grid flung through them does; each fetch logs its tile.
fn ask_for(this: &mut FoloApp, tiles: usize, fetched: &Arc<Mutex<Vec<usize>>>, cx: &mut Context<FoloApp>) {
    for i in 0..tiles {
        let k = key(i);
        assert!(this.store.images_px.begin(&k));
        let log = fetched.clone();
        this.load_cached_image(
            k,
            move |_| {
                log.lock().unwrap().push(i);
                Ok(ImageData::Rgba { width: 1, height: 1, pixels: vec![0; 4] })
            },
            cx,
        );
    }
}

/// The keys of tiles `ids`, in that order.
fn keys(ids: impl IntoIterator<Item = usize>) -> Vec<ImgKey> {
    ids.into_iter().map(key).collect()
}

// Fetches start on the UI thread, in the order `LoadQueue::started` records: the first ones the moment
// they are asked for (nothing is on its way yet), every later one when an earlier fetch answers.
// Nothing answers before the closure that asked returns, so those orders do not depend on threads.

#[test]
fn a_fling_through_thousands_of_tiles_fetches_the_ones_in_view_first_and_skips_the_evicted() {
    let (mut app, window) = open();
    const TILES: usize = 3000;
    let fetched = Arc::new(Mutex::new(Vec::new()));
    let (most_at_once, started) = drive(&mut app, window, |this, _, cx| {
        ask_for(this, TILES, &fetched, cx);
        (this.store.image_queue.in_flight(), this.store.image_queue.started.clone())
    });
    assert_eq!(most_at_once, MAX, "a few fetches at a time, not one per tile");
    // The first tiles asked for started at once, before the rest of the fling was asked for.
    assert_eq!(started, keys(0..MAX));
    drive(&mut app, window, |this, _, _| {
        let started = &this.store.image_queue.started;
        // Then, as those answered, the tiles asked for last (in view when the fling stopped).
        assert_eq!(started[MAX..2 * MAX], keys((TILES - MAX..TILES).rev())[..]);
        // The ones the cache gave up on while they waited (scrolled far past) were never fetched.
        assert!(started.len() <= ImageCache::BUDGET + MAX, "{} fetches", started.len());
        assert_eq!(fetched.lock().unwrap().len(), started.len(), "each started fetch ran once");
        assert!(matches!(this.store.images_px.get(&key(TILES - 1)), Some(ImgSlot::Ready(_))));
        assert!(this.store.images_px.get(&key(MAX)).is_none());
        assert_eq!((this.store.image_queue.in_flight(), this.store.image_queue.waiting()), (0, 0));
    });
}

#[test]
fn tiles_scrolled_back_to_are_fetched_before_the_offscreen_ones_queued_after_them() {
    let (mut app, window) = open();
    const TILES: usize = 500; // within the cache's budget: nothing is evicted, all stay wanted
    let fetched = Arc::new(Mutex::new(Vec::new()));
    drive(&mut app, window, |this, _, cx| {
        ask_for(this, TILES, &fetched, cx);
        // Back to tiles 20 and 8, still loading: drawn again, they are asked for again.
        for i in [20, 8, 0] {
            assert!(this.image(DS, &key(i).image, ImgKind::Thumb, cx).is_none());
        }
    });
    drive(&mut app, window, |this, _, _| {
        let started = &this.store.image_queue.started;
        assert_eq!(started[..MAX], keys(0..MAX)[..]);
        // Tile 0 was on its way already: not fetched twice. 8 was asked for last, then 20, then the rest.
        assert_eq!(started[MAX..MAX + 3], keys([8, 20, TILES - 1])[..]);
        assert_eq!(started.len(), TILES, "every tile fetched once");
        assert_eq!(fetched.lock().unwrap().len(), TILES);
        assert!((0..TILES).all(|i| matches!(this.store.images_px.get(&key(i)), Some(ImgSlot::Ready(_)))));
    });
}

// Failed images follow lane d1-datasets' model (`ImageCache::retry_delay`: asked for again 2, 8 and
// 32 s after a failure, a reconnect or a Retry forgets them), with every fetch through the queue.

fn down() -> crate::api::ApiError {
    crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7926")
}

/// Starts a fetch of tile `i` that fails with `e`.
fn fail(this: &mut FoloApp, i: usize, e: crate::api::ApiError, cx: &mut Context<FoloApp>) {
    assert!(this.store.images_px.begin(&key(i)));
    this.load_cached_image(key(i), move |_| Err(e), cx);
}

/// Draws tiles `ids` (fetched from the demo backend when needed) and returns how many new fetches
/// that started.
fn draw(this: &mut FoloApp, ids: &[usize], cx: &mut Context<FoloApp>) -> usize {
    let before = this.store.image_queue.started.len();
    for &i in ids {
        assert!(this.image(DS, &key(i).image, ImgKind::Thumb, cx).is_none());
    }
    this.store.image_queue.started.len() - before
}

fn slot(this: &FoloApp, i: usize) -> &'static str {
    match this.store.images_px.get(&key(i)) {
        None => "none",
        Some(ImgSlot::Loading) => "loading",
        Some(ImgSlot::Ready(_)) => "ready",
        Some(ImgSlot::Failed) => "failed",
    }
}

fn wait(app: &mut folo_ui::HeadlessApp, ms: u64) {
    app.advance_clock(std::time::Duration::from_millis(ms));
}

#[test]
fn a_thumbnail_that_failed_in_an_outage_the_refresh_never_saw_loads_again_by_its_retry_timer() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        // The server drops out between two refreshes: only the thumbnail sees it, and the `/system`
        // probe its failure asks for finds the server back.
        fail(this, 0, down(), cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.store.unreachable(), "no outage was seen");
        assert_eq!(slot(this, 0), "failed");
        assert_eq!(draw(this, &[0], cx), 0, "not asked for again on every frame");
    });
    wait(&mut app, 1999);
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, 0)), "failed");
    wait(&mut app, 1);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(slot(this, 0), "none", "its 2 s retry forgets it");
        assert_eq!(draw(this, &[0], cx), 1, "drawn again, it is fetched again");
    });
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, 0)), "ready");
}

#[test]
fn a_thumbnail_failing_after_reconnected_ran_loads_again_by_its_retry_timer() {
    use crate::store::Load;
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        // An outage the refresh saw, and the refresh that finds the server back.
        this.store.health = Load::Failed(down());
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.store.unreachable(), "reconnected");
        // A thumbnail asked for during the outage fails only now: reconnected() has already run.
        fail(this, 0, down(), cx);
    });
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, 0)), "failed");
    wait(&mut app, 2000);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(slot(this, 0), "none", "its 2 s retry forgets it");
        assert_eq!(draw(this, &[0], cx), 1);
    });
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, 0)), "ready");
}

#[test]
fn retry_or_a_reconnection_forgets_every_failed_image() {
    let (mut app, window) = open();
    let broken = || crate::api::ApiError::new("decode_failed", "not an image");
    let slots = |this: &FoloApp, ids: [usize; 2]| (slot(this, ids[0]), slot(this, ids[1]));
    // Each recovery against failures of its own: a reconnection...
    drive(&mut app, window, |this, _, cx| {
        fail(this, 0, broken(), cx);
        fail(this, 1, broken(), cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(slots(this, [0, 1]), ("failed", "failed"));
        this.reconnected(cx);
        assert_eq!(slots(this, [0, 1]), ("none", "none"), "reconnected() forgets them all");
    });
    // ...then Retry, with two that failed after the reconnection.
    drive(&mut app, window, |this, _, cx| {
        fail(this, 2, broken(), cx);
        fail(this, 3, broken(), cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(slots(this, [2, 3]), ("failed", "failed"), "failed right before the Retry");
        this.retry_all(cx);
        assert_eq!(slots(this, [2, 3]), ("none", "none"), "Retry forgets them all");
        assert_eq!(draw(this, &[0, 1, 2, 3], cx), 4);
    });
    let ready = drive(&mut app, window, |this, _, _| (0..4).all(|i| slot(this, i) == "ready"));
    assert!(ready, "all four load again");
}

#[test]
fn a_fetch_for_a_tile_evicted_and_drawn_again_does_not_undo_the_newer_fetch() {
    // Tile A's fetch is on its way when the cache drops A (a fling far past); A is drawn again and
    // fetched again, and the first fetch fails before the second answers.
    let (mut app, window) = open();
    let a = 0;
    let (release, held) = std::sync::mpsc::channel::<()>();
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                // Its failure lands only after this closure returns.
                fail(this, a, crate::api::ApiError::new("bad_response", "cut off"), cx);
                for i in 1..=ImageCache::BUDGET {
                    assert!(this.store.images_px.begin(&key(i)));
                    this.store.images_px.finish(key(i), None);
                }
                assert_eq!(slot(this, a), "none", "dropped while its fetch is on its way");
                assert!(this.store.images_px.begin(&key(a)));
                this.load_cached_image(
                    key(a),
                    move |_| {
                        held.recv().ok();
                        Ok(ImageData::Rgba { width: 1, height: 1, pixels: vec![0; 4] })
                    },
                    cx,
                );
                assert_eq!(this.store.image_queue.in_flight(), 2, "both fetches of A on their way");
            })
            .expect("the window is open")
    });
    let in_flight = |app: &mut folo_ui::HeadlessApp| {
        app.update(|cx| window.update(cx, |this, _, _| this.store.image_queue.in_flight()).unwrap())
    };
    // The first fetch's failure lands; the second is held.
    for _ in 0..1000 {
        if in_flight(&mut app) == 1 {
            break;
        }
        app.run_until_held(1);
    }
    assert_eq!(in_flight(&mut app), 1, "the first fetch answered");
    app.update(|cx| {
        window
            .update(cx, |this, _, _| {
                assert_eq!(slot(this, a), "loading", "the old fetch's failure is not the new one's");
                assert!(this.store.images_px.retry_delay(&key(a)).is_none(), "nor counted against it");
            })
            .unwrap()
    });
    release.send(()).unwrap();
    app.run_until_idle();
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, a)), "ready", "the newer fetch's pixels show");
    wait(&mut app, 3_600_000);
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, a)), "ready");
}

// The queue's seams with those retries.

#[test]
fn a_load_the_queue_skipped_or_the_cache_dropped_is_no_failure_and_leaves_nothing_behind() {
    let (mut app, window) = open();
    const TILES: usize = 3000;
    let fetched = Arc::new(Mutex::new(Vec::new()));
    drive(&mut app, window, |this, _, cx| {
        // One on its way whose slot the cache drops (scrolled far past) before it fails.
        let gone = TILES;
        fail(this, gone, crate::api::ApiError::new("bad_response", "cut off"), cx);
        assert_eq!(this.store.image_queue.in_flight(), 1);
        for i in TILES + 1..TILES + 1 + ImageCache::BUDGET {
            assert!(this.store.images_px.begin(&key(i)));
            this.store.images_px.finish(key(i), None);
        }
        assert_eq!(slot(this, gone), "none");
        // Then a fling: the tiles the cache drops while they wait are skipped.
        ask_for(this, TILES, &fetched, cx);
    });
    let skipped = drive(&mut app, window, |this, _, _| {
        let q = &this.store.image_queue;
        assert_eq!((q.in_flight(), q.waiting()), (0, 0));
        let loading = this.store.images_px.slots.values().filter(|s| matches!(s, ImgSlot::Loading)).count();
        assert_eq!(loading, 0, "no slot is left loading");
        let skipped: Vec<usize> = (0..=TILES).filter(|&i| slot(this, i) == "none").collect();
        assert!(skipped.contains(&MAX) && skipped.contains(&TILES), "{} skipped", skipped.len());
        for &i in &skipped {
            assert!(this.store.images_px.retry_delay(&key(i)).is_none(), "tile {i} counted as a failure");
            assert!(!this.image_failed(DS, &key(i).image, ImgKind::Thumb));
        }
        (skipped, this.store.image_queue.started.len())
    });
    // No retry timer was left for them: an hour on, nothing was fetched or asked for.
    wait(&mut app, 3_600_000);
    drive(&mut app, window, |this, _, cx| {
        let (skipped, started) = &skipped;
        assert_eq!(this.store.image_queue.started.len(), *started);
        assert!(skipped.iter().all(|&i| slot(this, i) == "none"));
        // Scrolled back to, a skipped tile is fetched, once.
        assert_eq!(draw(this, &[MAX, MAX], cx), 1);
    });
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, MAX)), "ready");
}

#[test]
fn retry_timers_start_no_fetch_and_the_tiles_drawn_again_wait_their_turn() {
    const TILES: usize = 30;
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        for i in 0..TILES {
            fail(this, i, crate::api::ApiError::new("bad_response", "cut off"), cx);
        }
        assert_eq!(this.store.image_queue.in_flight(), MAX);
    });
    let started = drive(&mut app, window, |this, _, _| {
        assert!((0..TILES).all(|i| slot(this, i) == "failed"));
        this.store.image_queue.started.len()
    });
    assert_eq!(started, TILES);
    wait(&mut app, 2000);
    drive(&mut app, window, |this, _, cx| {
        // The timers forget the failures; none of them fetches (the tiles are not drawn).
        assert!((0..TILES).all(|i| slot(this, i) == "none"));
        assert_eq!(this.store.image_queue.started.len(), TILES);
        // Drawn again all at once, they go through the queue's bound.
        let ids: Vec<usize> = (0..TILES).collect();
        assert_eq!(draw(this, &ids, cx), MAX);
        let q = &this.store.image_queue;
        assert_eq!((q.in_flight(), q.waiting()), (MAX, TILES - MAX));
    });
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.store.image_queue.started.len(), 2 * TILES);
        assert!((0..TILES).all(|i| slot(this, i) == "ready"));
    });
}

#[test]
fn a_queued_load_answered_after_a_retry_and_a_reconnect_is_not_undone_by_an_old_timer() {
    let (mut app, window) = open();
    let k = 40;
    drive(&mut app, window, |this, _, cx| {
        fail(this, k, crate::api::ApiError::new("bad_response", "cut"), cx)
    });
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, k)), "failed");
    wait(&mut app, 1000);
    let fetched = Arc::new(Mutex::new(Vec::new()));
    drive(&mut app, window, |this, _, cx| {
        ask_for(this, MAX, &fetched, cx); // the queue is full
        this.retry_failed_image(DS, &key(k).image, cx);
        assert_eq!(draw(this, &[k], cx), 0, "queued behind the ones on their way");
        assert_eq!(slot(this, k), "loading");
        this.reconnected(cx);
        assert_eq!(slot(this, k), "loading", "a reconnect leaves a queued load alone");
    });
    drive(&mut app, window, |this, _, _| {
        assert_eq!(slot(this, k), "ready");
        assert_eq!(this.store.image_queue.started.iter().filter(|s| **s == key(k)).count(), 2);
    });
    // The first failure's timer comes due: the newer result stays.
    wait(&mut app, 1000);
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, k)), "ready");
    wait(&mut app, 3_600_000);
    assert_eq!(drive(&mut app, window, |this, _, _| slot(this, k)), "ready");
}

#[test]
fn the_byte_budget_evicts_with_loads_queued_and_the_queue_skips_what_it_dropped() {
    const TILES: usize = 30;
    const SIDE: u32 = 10; // 400 decoded bytes each: five fit the budget
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.store.images_px = ImageCache::with_byte_budget(5 * 400);
        for i in 0..TILES {
            assert!(this.store.images_px.begin(&key(i)));
            let px = vec![7; (SIDE * SIDE * 4) as usize];
            this.load_cached_image(
                key(i),
                move |_| Ok(ImageData::Rgba { width: SIDE, height: SIDE, pixels: px }),
                cx,
            );
        }
    });
    drive(&mut app, window, |this, _, _| {
        let q = &this.store.image_queue;
        assert_eq!((q.in_flight(), q.waiting()), (0, 0));
        let ready: Vec<usize> = (0..TILES).filter(|&i| slot(this, i) == "ready").collect();
        assert_eq!(ready.len(), 5, "within the budget: {ready:?}");
        assert!(ready.contains(&(TILES - 1)), "the newest asked for stays: {ready:?}");
        assert!((0..TILES).all(|i| matches!(slot(this, i), "ready" | "none")), "nothing left loading");
        assert!(q.started.len() < TILES, "loads dropped while queued were not fetched: {}", q.started.len());
        assert!((0..TILES).all(|i| this.store.images_px.retry_delay(&key(i)).is_none()));
    });
}
