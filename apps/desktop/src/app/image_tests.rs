//! Image fetches when a grid is flung through a big dataset (stress x4-soak-2): every tile passed
//! asks for its image; the ones left in view must not wait for all the others.

use std::sync::{Arc, Mutex};

use folo_ui::Context;

use super::FoloApp;
use super::tests::{drive, open};
use crate::api::ImageData;
use crate::store::{ImageCache, ImgKey, ImgKind, ImgSlot, LoadQueue};

fn key(i: usize) -> ImgKey {
    ImgKey { dataset: "ds".into(), image: format!("img_{i:05}"), kind: ImgKind::Thumb }
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
            assert!(this.image("ds", &key(i).image, ImgKind::Thumb, cx).is_none());
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
