use std::collections::BTreeMap;

use super::*;
use crate::model::blocks::Init;
use crate::model::quant::cache::BoundaryGrids;
use crate::model::quant::{Calibration, Method};
use folo_core::rng::Rng;

/// Weight and bias views sharing the same underlying storage `RwLock` must never deadlock during `qconv` or
/// `qstem` packing (cache miss) or lookup (cache hit).
#[test]
fn aliased_weight_and_bias_storage_does_not_deadlock() {
    let mut rng = Rng::seed(99);
    let cal = Calibration {
        version: VERSION_V2,
        method: Method::MinMax,
        images: 1,
        ranges: BTreeMap::new(),
        channel_ranges: BTreeMap::new(),
        u8_stem: true,
        f32_convs: Vec::new(),
        weights: None,
    };
    let plan = Int8Plan::new(cal);
    let pool = Rc::new(RefCell::new(ScratchPool::default()));
    let exec = Int8 {
        plan: &plan,
        rgb0_input: None,
        pool: Rc::clone(&pool),
        conv_step: Cell::new(0),
        boundary_step: Cell::new(0),
    };

    // Aliased conv (`bias` is a view into `weight`'s own storage).
    let mut conv = Init::new(&mut rng).conv(8, 8, 3, 1);
    conv.fuse_().unwrap();
    conv.conv.bias = Some(conv.conv.weight.reshape([8 * 8 * 9]).unwrap().narrow(0, 0, 8).unwrap());
    let g = Grid::from_range(-1.0, 1.0);
    let bg = Arc::new(BoundaryGrids::new(vec![g; 8]));
    let buf = pool.borrow_mut().take_buf(1, 8, 16);
    let act = QActU8::new(&pool, buf, [1, 8, 4, 4], GridOwner::Owned(bg));
    let first = exec.qconv(&conv, &[&act]).unwrap();
    let second = exec.qconv(&conv, &[&act]).unwrap();
    assert!(Arc::ptr_eq(&first, &second), "second qconv call must hit cache");

    // Aliased stem (`bias` is a view into `weight`'s own storage).
    let mut stem = Init::new(&mut rng).conv(3, 16, 3, 2);
    stem.fuse_().unwrap();
    stem.conv.bias = Some(stem.conv.weight.reshape([16 * 3 * 9]).unwrap().narrow(0, 0, 16).unwrap());
    let (s1, l1) = exec.qstem(&stem).unwrap();
    let (s2, l2) = exec.qstem(&stem).unwrap();
    assert!(Arc::ptr_eq(&s1, &s2), "second qstem call must hit cache");
    assert!(l1.is_some() && l2.is_some());
}

/// Simulates an overlapping write completing between weight packing and `ScratchPool` step-cache insertion:
/// because liveness tracks `Weak` refs to the derived cache entries owned by each tensor's storage lock,
/// storing a pre-write handle after a concurrent weight or bias write must immediately miss on lookup.
#[test]
fn scratch_pool_step_cache_rejects_overlapping_write_race() {
    let mut rng = Rng::seed(2025);
    let cal = Calibration {
        version: VERSION_V2,
        method: Method::MinMax,
        images: 1,
        ranges: BTreeMap::new(),
        channel_ranges: BTreeMap::new(),
        u8_stem: true,
        f32_convs: Vec::new(),
        weights: None,
    };
    let plan = Int8Plan::new(cal);
    let pool = Rc::new(RefCell::new(ScratchPool::default()));
    let exec = Int8 {
        plan: &plan,
        rgb0_input: None,
        pool: Rc::clone(&pool),
        conv_step: Cell::new(0),
        boundary_step: Cell::new(0),
    };

    // 1. Overlapping weight write and overlapping bias write for a conv entry.
    let mut conv = Init::new(&mut rng).conv(8, 8, 1, 1);
    conv.fuse_().unwrap();
    let g = Grid::from_range(-1.0, 1.0);
    let bg = Arc::new(BoundaryGrids::new(vec![g; 8]));
    let buf = pool.borrow_mut().take_buf(1, 8, 16);
    let act = QActU8::new(&pool, buf, [1, 8, 4, 4], GridOwner::Owned(bg));
    let cv = ConvView::from_cba(&conv);
    let parts = [&act];
    let part_specs = [(&act.bg.as_ref().grids, act.grid_offset, act.dims[1])];

    let (qc_old, live_old) = exec.qconv_view(cv, &parts).unwrap();
    let live_old = live_old.expect("fresh conv pack yields liveness");
    // Overlapping write mutates weight AFTER pack but BEFORE store_conv.
    conv.conv.weight.copy_from(&Tensor::full([8, 8, 1, 1], 0.25)).unwrap();
    pool.borrow_mut().store_conv(0, cv, &part_specs, true, live_old, qc_old);
    assert!(
        !pool.borrow().conv_hit(0, cv, &part_specs, true),
        "stale conv packed before overlapping weight write must not hit"
    );

    let (qc_mid, live_mid) = exec.qconv_view(cv, &parts).unwrap();
    let live_mid = live_mid.expect("re-pack after weight write yields liveness");
    if let Some(bias) = &conv.conv.bias {
        bias.copy_from(&Tensor::full([8], 1.5)).unwrap();
        pool.borrow_mut().store_conv(0, cv, &part_specs, true, live_mid, qc_mid);
        assert!(
            !pool.borrow().conv_hit(0, cv, &part_specs, true),
            "stale conv packed before overlapping bias write must not hit"
        );
    }

    // 2. Overlapping weight write and overlapping bias write for a stem entry.
    let mut stem = Init::new(&mut rng).conv(3, 16, 3, 2);
    stem.fuse_().unwrap();
    let (qs_old, slive_old) = exec.qstem(&stem).unwrap();
    let slive_old = slive_old.expect("fresh stem pack yields liveness");
    stem.conv.weight.copy_from(&Tensor::full([16, 3, 3, 3], 0.125)).unwrap();
    pool.borrow_mut().store_stem(&stem, slive_old, qs_old);
    assert!(!pool.borrow().stem_hit(&stem), "stale stem packed before overlapping weight write must not hit");

    let (qs_mid, slive_mid) = exec.qstem(&stem).unwrap();
    let slive_mid = slive_mid.expect("re-pack after weight write yields liveness");
    if let Some(bias) = &stem.conv.bias {
        bias.copy_from(&Tensor::full([16], -0.5)).unwrap();
        pool.borrow_mut().store_stem(&stem, slive_mid, qs_mid);
        assert!(
            !pool.borrow().stem_hit(&stem),
            "stale stem packed before overlapping bias write must not hit"
        );
    }
}

/// Per-step `ScratchPool` conv and stem caches must invalidate immediately when a weight or bias tensor is
/// mutated in place, never returning stale packed outputs.
#[test]
fn scratch_pool_step_cache_invalidates_on_in_place_weight_mutation() {
    let mut rng = Rng::seed(123);
    let cal = Calibration {
        version: VERSION_V2,
        method: Method::MinMax,
        images: 1,
        ranges: BTreeMap::new(),
        channel_ranges: BTreeMap::new(),
        u8_stem: true,
        f32_convs: Vec::new(),
        weights: None,
    };
    let plan = Int8Plan::new(cal);
    let pool = Rc::new(RefCell::new(ScratchPool::default()));
    let mut conv = Init::new(&mut rng).conv(8, 8, 1, 1);
    conv.fuse_().unwrap();
    let g = Grid::from_range(-1.0, 1.0);
    let bg = Arc::new(BoundaryGrids::new(vec![g; 8]));

    let run_once = || -> Vec<f32> {
        let exec = Int8 {
            plan: &plan,
            rgb0_input: None,
            pool: Rc::clone(&pool),
            conv_step: Cell::new(0),
            boundary_step: Cell::new(0),
        };
        let x = Tensor::full([1, 8, 4, 4], 0.5);
        let act =
            QActU8::from_f32(&pool, &x, GridOwner::Owned(Arc::clone(&bg)), 0, bg.hash, bg.max_scale).unwrap();
        let mut out = vec![0.0f32; 8 * 16];
        exec.run_view("c", ConvView::from_cba(&conv), &[&act], None, QOut::F32 { data: &mut out }).unwrap();
        out
    };

    let y1 = run_once();
    let y2 = run_once(); // hits ScratchPool step cache
    assert_eq!(y1, y2);

    // Mutate weight in place: derived cache is cleared under storage write lock, so the next forward re-packs.
    conv.conv.weight.copy_from(&Tensor::full([8, 8, 1, 1], 0.0)).unwrap();
    let y3 = run_once();
    assert_ne!(y1, y3, "in-place weight mutation must invalidate per-step QConv cache");

    // Mutate bias in place: derived bias cache is cleared under storage write lock.
    if let Some(bias) = &conv.conv.bias {
        bias.copy_from(&Tensor::full([8], 2.0)).unwrap();
        let y4 = run_once();
        assert_ne!(y3, y4, "in-place bias mutation must invalidate per-step QConv cache");
    }
}
