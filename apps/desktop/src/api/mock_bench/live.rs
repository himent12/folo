//! The demo moving on: entries and loops advance with the clock, and a benchmark's event stream.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;

use super::loops::step_loop;
use super::scoring::synthesize_result;
use super::{MockBench, MockBenchData, find_bench, snapshot_bench};

impl MockBench {
    pub fn events(
        this: Arc<Self>,
        id: &str,
        datasets: Vec<DatasetFixture>,
        models: Vec<Model>,
    ) -> ApiResult<BenchSubscription> {
        let initial = this.get(id, &datasets, &models)?;
        let (tx, rx) = channel();
        let (sub, cancel) = BenchSubscription::new(rx);
        let _ = tx.send(BenchEvent::Bench(Box::new(initial)));
        if this.frozen {
            return Ok(sub);
        }
        let bid = id.to_string();
        // What the subscriber has seen, taken now (a change right after this call is streamed): each
        // tick sends what changed since, as the server does (an entry or loop that moved, one that is
        // gone, the benchmark's own record).
        let (bench0, entries0, loops0) = this.with(&datasets, &models, |d, _, _| {
            let b = find_bench(&d.benches, &bid)?;
            Ok((
                snapshot_bench(b),
                b.entries.iter().map(|e| e.entry.clone()).collect::<Vec<_>>(),
                b.loops.clone(),
            ))
        })?;
        let mut seen_bench = Some(bench0);
        let mut seen: HashMap<String, Option<BenchEntry>> =
            entries0.into_iter().map(|e| (e.id.clone(), Some(e))).collect();
        let mut seen_loops: HashMap<String, Option<BenchLoop>> =
            loops0.into_iter().map(|l| (l.id.clone(), Some(l))).collect();
        std::thread::spawn(move || {
            while !cancel.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(450));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let snap = this.with(&datasets, &models, |d, _, _| {
                    let b = find_bench(&d.benches, &bid)?;
                    let entries: Vec<BenchEntry> = b.entries.iter().map(|e| e.entry.clone()).collect();
                    Ok((snapshot_bench(b), entries, b.loops.clone()))
                });
                let Ok((bench_snap, entries, loops)) = snap else { break };
                let mut out = Vec::new();
                for e in &entries {
                    if seen.get(&e.id).and_then(Option::as_ref) != Some(e) {
                        seen.insert(e.id.clone(), Some(e.clone()));
                        out.push(BenchEvent::Entry(Box::new(e.clone())));
                    }
                }
                for l in &loops {
                    if seen_loops.get(&l.id).and_then(Option::as_ref) != Some(l) {
                        seen_loops.insert(l.id.clone(), Some(l.clone()));
                        out.push(BenchEvent::Loop(Box::new(l.clone())));
                    }
                }
                // An entry or loop that is gone: one removal event names it (ids don't collide).
                let gone: Vec<String> = seen
                    .iter()
                    .filter(|(id, v)| v.is_some() && !entries.iter().any(|e| &e.id == *id))
                    .map(|(id, _)| id.clone())
                    .chain(
                        seen_loops
                            .iter()
                            .filter(|(id, v)| v.is_some() && !loops.iter().any(|l| &l.id == *id))
                            .map(|(id, _)| id.clone()),
                    )
                    .collect();
                for id in gone {
                    seen.insert(id.clone(), None);
                    seen_loops.insert(id.clone(), None);
                    out.push(BenchEvent::Removed(BenchRemoved { id }));
                }
                if seen_bench.as_ref() != Some(&bench_snap) {
                    seen_bench = Some(bench_snap.clone());
                    out.push(BenchEvent::Bench(Box::new(bench_snap)));
                }
                for ev in out {
                    if tx.send(ev).is_err() {
                        return;
                    }
                }
            }
        });
        Ok(sub)
    }
}

pub(super) fn advance(data: &mut MockBenchData, dt: f64, now: i64) {
    if dt <= 0.0 {
        return;
    }
    for b in &mut data.benches {
        let mut model_slot_busy = false;
        for ef in &mut b.entries {
            let is_model = matches!(ef.entry.contestant, BenchContestant::Model { .. });
            if is_model && ef.entry.status == BenchEntryStatus::Queued && !model_slot_busy {
                ef.entry.status = BenchEntryStatus::Running;
                ef.live = Some((0, dt));
            }
            if let Some((base, t0)) = ef.live
                && ef.entry.status == BenchEntryStatus::Running
            {
                let rate = if is_model { 180.0 } else { 14.0 };
                let done = (base + ((dt - t0).max(0.0) * rate) as u64).min(ef.entry.progress.total);
                ef.entry.progress.done = done;
                if done >= ef.entry.progress.total {
                    ef.entry.status = BenchEntryStatus::Done;
                    ef.entry.finished = Some(time::format(now));
                    ef.live = None;
                    let res = synthesize_result(&b.bench, &b.images, &ef.entry, ef.skill);
                    ef.entry.score = Some(res.entry.score.clone().unwrap());
                    ef.entry.map50_95 = Some(res.accuracy.map50_95);
                    ef.entry.ci95 = Some(res.accuracy.ci95);
                    ef.entry.conditions_run =
                        res.robustness.as_ref().map_or(0, |r| r.conditions.len() as u32);
                    ef.cached_result = Some(res);
                } else {
                    let (cond, sev) = pass_at(&b.bench.plan, ef.entry.images, ef.entry.conditions, done);
                    ef.entry.progress.condition = cond;
                    ef.entry.progress.severity = sev;
                }
            }
            if is_model && ef.entry.status.is_active() {
                model_slot_busy = true;
            }
        }
        for lp in &mut b.loops {
            if let Some(&t0) = b.loop_live.get(&lp.id)
                && lp.status == BenchLoopStatus::Running
            {
                step_loop(lp, dt - t0, now);
            }
        }
    }
}

pub(super) fn pass_at(
    plan: &BenchPlan,
    images: u64,
    conditions: bool,
    done: u64,
) -> (Option<String>, Option<u8>) {
    let imgs = images.max(1);
    let pass = (done / imgs) as usize;
    if pass == 0 || !conditions {
        return (None, None);
    }
    let mut k = 1usize;
    for c in &plan.conditions {
        for &s in &c.severities {
            if k == pass {
                return (Some(c.condition.clone()), Some(s));
            }
            k += 1;
        }
    }
    (None, None)
}
