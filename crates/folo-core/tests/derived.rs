//! `Tensor::derived_f32`: values kept with a storage until its data is next written.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use folo_core::Tensor;

#[test]
fn derived_values_are_kept_per_key_and_dropped_by_a_write() {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let made = AtomicUsize::new(0);
    let sum = |key: u64| {
        t.derived_f32(key, |v| {
            made.fetch_add(1, Ordering::Relaxed);
            v.iter().sum::<f32>()
        })
        .unwrap()
    };
    let a = sum(1);
    assert_eq!(*a, 10.0);
    assert!(Arc::ptr_eq(&a, &sum(1)), "kept under its key");
    assert_eq!(made.load(Ordering::Relaxed), 1);
    sum(2);
    assert_eq!(made.load(Ordering::Relaxed), 2, "another key is another value");
    // A view through another layout reads other elements, so it is another value.
    let col: Vec<f32> = t.transpose(0, 1).unwrap().derived_f32(1, |v| v.to_vec()).unwrap().to_vec();
    assert_eq!(col, [1.0, 3.0, 2.0, 4.0]);
    // A write drops every value: the next one is made from the new data.
    t.fill_assign(0.5).unwrap();
    assert_eq!(*sum(1), 2.0);
    assert_eq!(made.load(Ordering::Relaxed), 3);
}
