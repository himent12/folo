//! `Tensor::derived_f32`: values kept with a storage until its data is next written.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use folo_core::Tensor;

#[test]
fn derived_values_are_kept_per_key_and_layout_and_dropped_by_a_write() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let made = AtomicUsize::new(0);
    let copy = |x: &Tensor, key: u64| {
        x.derived_f32(key, |v| {
            made.fetch_add(1, Ordering::Relaxed);
            v.to_vec()
        })
        .unwrap()
    };
    let a = copy(&t, 1);
    assert_eq!(*a, [1.0, 2.0, 3.0, 4.0]);
    assert!(Arc::ptr_eq(&a, &copy(&t, 1)), "kept under its key");
    assert_eq!(made.load(Ordering::Relaxed), 1);
    // A view through another layout, under the same key and type, reads other elements: another value.
    assert_eq!(*copy(&t.transpose(0, 1).unwrap(), 1), [1.0, 3.0, 2.0, 4.0]);
    assert_eq!(made.load(Ordering::Relaxed), 2);
    // The first value is still kept right before the write, and the write drops it.
    assert!(Arc::ptr_eq(&a, &copy(&t, 1)));
    t.fill_assign(0.5).unwrap();
    assert_eq!(*copy(&t, 1), [0.5; 4]);
    assert_eq!(made.load(Ordering::Relaxed), 3);
}

#[test]
fn a_write_queued_behind_a_build_drops_the_value() {
    // A reader builds a value from the old data under the read lock while a writer bumps the version and waits
    // for the write lock: the reader gets the old data's value, and nothing keeps it past the write.
    let t = Tensor::from_vec(vec![1.0f32; 4], [4]).unwrap();
    let (started, wait) = mpsc::channel();
    std::thread::scope(|s| {
        let reader = s.spawn(|| {
            t.derived_f32(7, |v| {
                started.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(100));
                v.to_vec()
            })
            .unwrap()
        });
        wait.recv().unwrap();
        let writer = s.spawn(|| t.fill_assign(2.0).unwrap());
        assert_eq!(*reader.join().unwrap(), [1.0; 4]);
        writer.join().unwrap();
    });
    assert_eq!(*t.derived_f32(7, |v| v.to_vec()).unwrap(), [2.0; 4]);
}
