//! [`crate::ForwardPlans`] on small calls: what makes a call unplannable, a call inside a recording, a panic while
//! recording, a lost device.
//! The model-level tests (bit-identity, invalidation, threads) are folo-vision's `tests/gpu_plan.rs`.

use folo_core::{Device, Tensor, no_grad};

use super::gpu;
use crate::ForwardPlans;

/// `relu(x * w + b)`, a pure function of `x` and the listed `w`, `b`.
fn call(x: &Tensor, w: &Tensor, b: &Tensor) -> folo_core::Result<Vec<Tensor>> {
    Ok(vec![x.mul(w)?.add(b)?.relu()?])
}

fn vals(t: &[Tensor]) -> Vec<f32> {
    t[0].to_vec::<f32>().unwrap()
}

fn setup(d: Device) -> (Tensor, Tensor, Tensor, Vec<f32>) {
    let x = Tensor::from_slice(&[-2.0f32, -1.0, 0.5, 3.0], [4]).unwrap().to_device(d).unwrap();
    let w = Tensor::from_slice(&[1.0f32, -2.0, 4.0, 0.5], [4]).unwrap().to_device(d).unwrap();
    let b = Tensor::from_slice(&[0.5f32, 0.25, -1.0, 1.0], [4]).unwrap().to_device(d).unwrap();
    (x, w, b, vec![0.0, 2.25, 1.0, 2.5])
}

#[test]
fn a_pure_call_replays_and_its_results_are_its_own() {
    let _ctx = gpu!();
    let d = crate::init().unwrap();
    let _g = no_grad();
    let (x, w, b, want) = setup(d);
    let plans = ForwardPlans::new();
    let tensors = |f: &mut dyn FnMut(&Tensor)| {
        f(&w);
        f(&b);
    };
    let mut held = Vec::new();
    for _ in 0..5 {
        let out = plans.run(&[0], &x, &tensors, |x| call(x, &w, &b)).unwrap();
        held.push(out);
    }
    assert_eq!(plans.replays(), 3, "{:?}", plans.refusal());
    for out in &held {
        assert_eq!(vals(out), want);
    }
    // Another input of the same shape, through the same plan.
    let y = x.mul_scalar(-1.0).unwrap();
    let out = plans.run(&[0], &y, &tensors, |y| call(y, &w, &b)).unwrap();
    assert_eq!(plans.replays(), 4);
    assert_eq!(vals(&out), vec![2.5, 0.0, 0.0, 0.0]);
    // The replay for y wrote none of the results held from x.
    for out in &held {
        assert_eq!(vals(out), want);
    }
}

#[test]
fn a_call_inside_a_recording_runs_as_part_of_it() {
    let _ctx = gpu!();
    let d = crate::init().unwrap();
    let _g = no_grad();
    let (x, w, b, want) = setup(d);
    let tensors = |f: &mut dyn FnMut(&Tensor)| {
        f(&w);
        f(&b);
    };
    // The inner owner has a plan before the outer one records.
    let inner = ForwardPlans::new();
    for _ in 0..3 {
        inner.run(&[0], &x, &tensors, |x| call(x, &w, &b)).unwrap();
    }
    assert_eq!(inner.replays(), 1, "{:?}", inner.refusal());
    let outer = ForwardPlans::new();
    let twice = |x: &Tensor| -> folo_core::Result<Vec<Tensor>> {
        let y = inner.run(&[0], x, &tensors, |x| call(x, &w, &b))?;
        Ok(vec![y[0].mul_scalar(2.0)?])
    };
    for _ in 0..4 {
        assert_eq!(
            vals(&outer.run(&[0], &x, &tensors, twice).unwrap()),
            want.iter().map(|v| v * 2.0).collect::<Vec<_>>()
        );
    }
    assert_eq!(outer.replays(), 2, "{:?}", outer.refusal());
    // The outer replay carries the inner work: another input gives its own results.
    let y = x.mul_scalar(-1.0).unwrap();
    assert_eq!(vals(&outer.run(&[0], &y, &tensors, twice).unwrap()), vec![5.0, 0.0, 0.0, 0.0]);
    assert_eq!(outer.replays(), 3);
}

#[test]
fn uploads_and_unlisted_tensors_make_a_call_unplannable() {
    let _ctx = gpu!();
    let d = crate::init().unwrap();
    let _g = no_grad();
    let (x, w, b, want) = setup(d);
    let both = |f: &mut dyn FnMut(&Tensor)| {
        f(&w);
        f(&b);
    };
    // An upload inside the call: its data comes from the host on every call.
    let plans = ForwardPlans::new();
    for _ in 0..4 {
        let out = plans
            .run(&[0], &x, &both, |x| {
                let b = Tensor::from_slice(&[0.5f32, 0.25, -1.0, 1.0], [4])?.to_device(x.device())?;
                call(x, &w, &b)
            })
            .unwrap();
        assert_eq!(vals(&out), want);
    }
    assert_eq!(plans.replays(), 0);
    assert!(plans.refusal().is_some_and(|r| r.contains("upload")), "{:?}", plans.refusal());

    // A tensor the call reads but the caller does not list: it could change without the plan noticing.
    let plans = ForwardPlans::new();
    let only_w = |f: &mut dyn FnMut(&Tensor)| f(&w);
    for _ in 0..4 {
        assert_eq!(vals(&plans.run(&[0], &x, &only_w, |x| call(x, &w, &b)).unwrap()), want);
    }
    assert_eq!(plans.replays(), 0);
    assert!(plans.refusal().is_some_and(|r| r.contains("not listed")), "{:?}", plans.refusal());

    // A result that is a view of a larger buffer.
    let plans = ForwardPlans::new();
    for _ in 0..4 {
        let out = plans
            .run(&[0], &x, &both, |x| Ok::<_, folo_core::Error>(vec![call(x, &w, &b)?[0].narrow(0, 1, 2)?]));
        assert_eq!(out.unwrap()[0].to_vec::<f32>().unwrap(), want[1..3]);
    }
    assert_eq!(plans.replays(), 0);

    // A readback inside the call: the host sees data the replay would not produce again.
    let plans = ForwardPlans::new();
    for _ in 0..4 {
        let out = plans
            .run(&[0], &x, &both, |x| {
                let _ = x.to_vec::<f32>()?;
                call(x, &w, &b)
            })
            .unwrap();
        assert_eq!(vals(&out), want);
    }
    assert_eq!(plans.replays(), 0);
    assert!(plans.refusal().is_some_and(|r| r.contains("readback")), "{:?}", plans.refusal());

    // Under autocast (a matrix op would pick other kernels): no plan, nothing recorded.
    let plans = ForwardPlans::new();
    {
        let _a = folo_core::autocast();
        for _ in 0..4 {
            assert_eq!(vals(&plans.run(&[0], &x, &both, |x| call(x, &w, &b)).unwrap()), want);
        }
    }
    assert_eq!(plans.replays(), 0);
    assert_eq!(plans.refusal(), None);
}

#[test]
fn a_panic_while_recording_leaves_the_owner_usable() {
    let _ctx = gpu!();
    let d = crate::init().unwrap();
    let _g = no_grad();
    let (x, w, b, want) = setup(d);
    let plans = ForwardPlans::new();
    let tensors = |f: &mut dyn FnMut(&Tensor)| {
        f(&w);
        f(&b);
    };
    plans.run(&[0], &x, &tensors, |x| call(x, &w, &b)).unwrap();
    // The second call records, and panics half-way.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plans.run(&[0], &x, &tensors, |x| -> folo_core::Result<Vec<Tensor>> {
            let _ = x.mul(&w)?;
            panic!("inside the recorded call")
        })
    }));
    assert!(r.is_err());
    // The thread records nothing any more, and the key is recorded and replayed afresh.
    let ctx = crate::context_for(d).unwrap();
    assert!(!ctx.recording());
    for _ in 0..3 {
        assert_eq!(vals(&plans.run(&[0], &x, &tensors, |x| call(x, &w, &b)).unwrap()), want);
    }
    assert_eq!(plans.replays(), 1, "{:?}", plans.refusal());
}

#[test]
fn a_lost_device_is_an_error_not_a_replay() {
    let _shared = gpu!();
    let ctx = crate::GpuContext::new().unwrap();
    // A device of its own (ordinal 5 is used by no other test).
    let d = Device::Gpu(5);
    crate::register(ctx.clone(), d).unwrap();
    let _g = no_grad();
    let (x, w, b, want) = setup(d);
    let plans = ForwardPlans::new();
    let tensors = |f: &mut dyn FnMut(&Tensor)| {
        f(&w);
        f(&b);
    };
    for _ in 0..3 {
        assert_eq!(vals(&plans.run(&[0], &x, &tensors, |x| call(x, &w, &b)).unwrap()), want);
    }
    assert_eq!(plans.replays(), 1);
    ctx.destroy_device();
    let read = plans.run(&[0], &x, &tensors, |x| call(x, &w, &b)).and_then(|out| out[0].to_vec::<f32>());
    assert!(read.as_ref().is_err_and(crate::is_device_lost), "{read:?}");
    // Once the loss is known no plan answers.
    let replays = plans.replays();
    let again = plans.run(&[0], &x, &tensors, |x| call(x, &w, &b)).and_then(|out| out[0].to_vec::<f32>());
    assert!(again.is_err());
    assert_eq!(plans.replays(), replays);
}

#[test]
fn plan_replays_long_keys_many_outputs_and_reuses_dynamic_binds() {
    let _shared = gpu!();
    let ctx = crate::GpuContext::new().unwrap();
    // Dedicated device ordinal 6 so bind-group counters are isolated from parallel tests.
    let d = Device::Gpu(6);
    crate::register(ctx.clone(), d).unwrap();
    let _g = no_grad();
    let (x, w, b, want) = setup(d);
    let tensors = |f: &mut dyn FnMut(&Tensor)| {
        f(&w);
        f(&b);
    };

    // 1. Long key (>16 u64 signature words -> Heap spill) and 10 outputs (>8 INLINE_OUTS -> Heap spill).
    let long_key = [11u64; 20];
    let plans = ForwardPlans::new();
    let many_out = |x: &Tensor| -> folo_core::Result<Vec<Tensor>> {
        let base = x.mul(&w)?.add(&b)?.relu()?;
        let mut outs = Vec::with_capacity(10);
        for i in 0..10 {
            outs.push(base.add_scalar(i as f32)?);
        }
        Ok(outs)
    };
    for _ in 0..4 {
        let outs = plans.run(&long_key, &x, &tensors, many_out).unwrap();
        assert_eq!(outs.len(), 10);
        for (i, t) in outs.iter().enumerate() {
            let expected: Vec<f32> = want.iter().map(|v| v + i as f32).collect();
            assert_eq!(t.to_vec::<f32>().unwrap(), expected);
        }
    }
    assert_eq!(plans.replays(), 2, "{:?}", plans.refusal());

    // 2. Repeated replay with the same input and recycled output buffer creates zero new bind groups.
    let simple = ForwardPlans::new();
    for _ in 0..3 {
        let out = simple.run(&[42], &x, &tensors, |x| call(x, &w, &b)).unwrap();
        assert_eq!(vals(&out), want);
    }
    assert_eq!(simple.replays(), 1, "{:?}", simple.refusal());
    ctx.sync().unwrap();
    let before = ctx.work_stats().bind_groups;
    for _ in 0..5 {
        let out = simple.run(&[42], &x, &tensors, |x| call(x, &w, &b)).unwrap();
        assert_eq!(vals(&out), want);
    }
    assert_eq!(ctx.work_stats().bind_groups - before, 0, "dynamic bind hit must not recreate bind groups");
    assert_eq!(simple.replays(), 6);
}
