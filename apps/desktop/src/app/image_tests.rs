//! Image fetches when a grid is flung through a big dataset (stress x4-soak-2): every tile passed
//! asks for its image; the ones left in view must not wait for all the others.

use std::sync::{Arc, Mutex};

use super::tests::{drive, open};
use crate::api::ImageData;
use crate::store::{ImageCache, ImgKey, ImgKind, ImgSlot, LoadQueue};

fn key(i: usize) -> ImgKey {
    ImgKey { dataset: "ds".into(), image: format!("img_{i:05}"), kind: ImgKind::Thumb }
}

#[test]
fn a_fling_through_thousands_of_tiles_fetches_the_ones_in_view_first_and_skips_the_evicted() {
    let (mut app, window) = open();
    const TILES: usize = 3000;
    let fetched = Arc::new(Mutex::new(Vec::new()));
    let most_at_once = drive(&mut app, window, |this, _, cx| {
        for i in 0..TILES {
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
        this.store.image_queue.in_flight()
    });
    assert_eq!(most_at_once, LoadQueue::<()>::MAX_IN_FLIGHT, "a few fetches at a time, not one per tile");
    let fetched = fetched.lock().unwrap().clone();
    // The tiles asked for last (the ones in view when the fling stopped) were fetched first.
    let newest = TILES - LoadQueue::<()>::MAX_IN_FLIGHT..TILES;
    assert!(newest.contains(&fetched[0]), "first fetched: {}", fetched[0]);
    assert!(newest.clone().all(|i| fetched.contains(&i)));
    // The ones the cache gave up on while they waited (scrolled far past) were never fetched.
    assert!(
        fetched.len() <= ImageCache::BUDGET + LoadQueue::<()>::MAX_IN_FLIGHT,
        "{} fetches",
        fetched.len()
    );
    drive(&mut app, window, |this, _, _| {
        assert!(matches!(this.store.images_px.get(&key(TILES - 1)), Some(ImgSlot::Ready(_))));
        assert!(this.store.images_px.get(&key(0)).is_none());
        assert_eq!((this.store.image_queue.in_flight(), this.store.image_queue.waiting()), (0, 0));
    });
}
