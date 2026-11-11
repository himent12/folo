//! Self-improvement loops: listing, starting, stopping and deleting them, and how a started loop runs
//! through its generations on the demo's clock.

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;
use crate::synth::hash;

use super::{MockBench, find_bench, find_bench_mut};

impl MockBench {
    pub fn loops(
        &self,
        id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<Vec<BenchLoop>> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench(&d.benches, id)?;
            let mut out = b.loops.clone();
            out.sort_by(|a, b| b.created.cmp(&a.created));
            Ok(out)
        })
    }

    pub fn start_loop(
        &self,
        id: &str,
        req: &StartBenchLoop,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchLoop> {
        self.with(datasets, models, |d, dt, now| {
            let teacher = models
                .iter()
                .find(|m| m.id == req.teacher)
                .ok_or_else(|| ApiError::new("model_not_found", format!("No model {}", req.teacher)))?;
            d.seq += 1;
            let seq = d.seq;
            let b = find_bench_mut(&mut d.benches, id)?;
            if teacher.classes.len() != b.bench.classes.len()
                || !teacher
                    .classes
                    .iter()
                    .zip(&b.bench.classes)
                    .all(|(a, c)| a.eq_ignore_ascii_case(c))
            {
                return Err(ApiError::new(
                    "bench_class_mismatch",
                    format!("{} does not have {}'s classes in order.", teacher.name, b.bench.name),
                ));
            }
            let gens = req.generations.unwrap_or(1);
            if !(1..=5).contains(&gens) {
                return Err(ApiError::new("invalid_body", "generations must be 1 to 5."));
            }
            let lid = format!("bl_{:04x}", (hash(&format!("bl-{seq}")) & 0xFFFF).max(1));
            let name = req
                .name
                .clone()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| format!("{} loop", teacher.name));
            let teacher_entry = b.entries.iter().find(|e| {
                matches!(&e.entry.contestant, BenchContestant::Model { model_id, .. } if *model_id == teacher.id)
                    && e.entry.status == BenchEntryStatus::Done
                    && e.entry.fingerprint == b.bench.fingerprint
            });
            let g0 = BenchGeneration {
                generation: 0,
                labels: None,
                run_id: teacher.run_id.clone(),
                model_id: Some(teacher.id.clone()),
                model_name: Some(teacher.name.clone()),
                entry_id: teacher_entry.map(|e| e.entry.id.clone()),
                score: teacher_entry.and_then(|e| e.entry.score.as_ref().map(|s| s.total)),
                map50_95: teacher_entry.and_then(|e| e.entry.map50_95),
                ci95: teacher_entry.and_then(|e| e.entry.ci95),
            };
            let phase = if g0.map50_95.is_some() {
                BenchLoopPhase::Labeling
            } else {
                BenchLoopPhase::Benchmarking
            };
            let mut generations = vec![g0];
            if phase == BenchLoopPhase::Labeling {
                generations.push(BenchGeneration {
                    generation: 1,
                    labels: None,
                    run_id: None,
                    model_id: None,
                    model_name: None,
                    entry_id: None,
                    score: None,
                    map50_95: None,
                    ci95: None,
                });
            }
            let mut config = req.clone();
            config.generations = Some(gens);
            if config.student.arch.is_none() {
                config.student.arch = Some(teacher.arch.as_str().to_string());
            }
            if config.student.epochs.is_none() {
                config.student.epochs = Some(30);
            }
            if config.student.imgsz.is_none() {
                config.student.imgsz = Some(teacher.imgsz);
            }
            let lp = BenchLoop {
                id: lid,
                bench_id: b.bench.id.clone(),
                name,
                status: BenchLoopStatus::Running,
                phase: Some(phase),
                created: time::format(now),
                finished: None,
                config,
                generations,
                verdict: None,
                error: None,
                hint: None,
            };
            b.loops.insert(0, lp.clone());
            b.loop_live.insert(lp.id.clone(), dt);
            Ok(lp)
        })
    }

    pub fn stop_loop(
        &self,
        id: &str,
        loop_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<BenchLoop> {
        self.with(datasets, models, |d, _, now| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let lp = b
                .loops
                .iter_mut()
                .find(|l| l.id == loop_id)
                .ok_or_else(|| ApiError::new("bench_loop_not_found", format!("No loop {loop_id}")))?;
            if !lp.status.is_active() {
                return Err(ApiError::new("bench_loop_not_active", "The loop is no longer running."));
            }
            lp.status = BenchLoopStatus::Stopped;
            lp.phase = None;
            lp.finished = Some(time::format(now));
            Ok(lp.clone())
        })
    }

    pub fn delete_loop(
        &self,
        id: &str,
        loop_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<()> {
        self.with(datasets, models, |d, _, _| {
            let b = find_bench_mut(&mut d.benches, id)?;
            let lp = b
                .loops
                .iter()
                .find(|l| l.id == loop_id)
                .ok_or_else(|| ApiError::new("bench_loop_not_found", format!("No loop {loop_id}")))?;
            if lp.status.is_active() {
                return Err(ApiError::new("bench_loop_active", "Stop the loop before deleting it."));
            }
            b.loops.retain(|l| l.id != loop_id);
            Ok(())
        })
    }
}

/// Seconds a demo loop spends on one generation: labeling the pool, training the student, then
/// benchmarking it (a teacher without an entry here is benchmarked first, for [`LOOP_TEACHER_S`]).
pub(super) const LOOP_GEN_S: f64 = 4.0;
pub(super) const LOOP_TEACHER_S: f64 = 2.0;

/// A loop started `t` seconds ago, as the demo runs it: each generation's student scores a little
/// higher than the last, so a long enough loop improves.
pub(super) fn step_loop(lp: &mut BenchLoop, t: f64, now: i64) {
    let gens = lp.config.generations.unwrap_or(1).max(1) as usize;
    let mut t = t.max(0.0);
    if lp.generations[0].map50_95.is_none() {
        if t < LOOP_TEACHER_S {
            lp.phase = Some(BenchLoopPhase::Benchmarking);
            return;
        }
        let g0 = &mut lp.generations[0];
        g0.score = Some(50.0);
        g0.map50_95 = Some(0.40);
        g0.ci95 = Some([0.37, 0.43]);
        t -= LOOP_TEACHER_S;
    }
    let base = lp.generations[0].clone();
    let (base_map, base_score) = (base.map50_95.unwrap_or(0.4), base.score.unwrap_or(50.0));
    let done = ((t / LOOP_GEN_S) as usize).min(gens);
    lp.generations.truncate(1);
    for k in 1..=done {
        let map = base_map + 0.03 * k as f32;
        lp.generations.push(BenchGeneration {
            generation: k as u32,
            labels: Some(BenchLabels {
                threshold: 0.5,
                images: 400,
                labeled_images: 380,
                boxes: 1500 + 20 * k as u64,
                excluded: 0,
                dataset_id: format!("ds_{}g{k}", lp.id),
            }),
            run_id: None,
            model_id: None,
            model_name: Some(format!("{} gen {k}-best", lp.name)),
            entry_id: None,
            score: Some(base_score + 2.0 * k as f32),
            map50_95: Some(map),
            ci95: Some([map - 0.02, map + 0.02]),
        });
    }
    if done == gens {
        // As docs/BENCH.md defines it: the last generation's mAP against the first teacher's interval,
        // its bounds included in Held.
        let last = lp.generations.last().and_then(|g| g.map50_95);
        let ci0 = base.ci95.unwrap_or([base_map, base_map]);
        lp.verdict = Some(match last {
            Some(m) if m > ci0[1] => BenchLoopVerdict::Improved,
            Some(m) if m < ci0[0] => BenchLoopVerdict::Regressed,
            _ => BenchLoopVerdict::Held,
        });
        lp.status = BenchLoopStatus::Done;
        lp.phase = None;
        lp.finished = Some(time::format(now));
        return;
    }
    let within = t - done as f64 * LOOP_GEN_S;
    lp.phase = Some(if within < 1.5 {
        BenchLoopPhase::Labeling
    } else if within < 3.5 {
        BenchLoopPhase::Training
    } else {
        BenchLoopPhase::Benchmarking
    });
    lp.generations.push(BenchGeneration {
        generation: done as u32 + 1,
        labels: None,
        run_id: None,
        model_id: None,
        model_name: None,
        entry_id: None,
        score: None,
        map50_95: None,
        ci95: None,
    });
}

#[cfg(test)]
mod tests {
    use crate::api::mock::MockBackend;
    use crate::api::*;
    #[test]
    fn a_loop_starts_first_in_the_list() {
        let m = MockBackend::new(0, true);
        // The benchmark's best model that the server still has (a loop's students may be gone).
        let models = m.models().unwrap();
        let teacher = m.bench_entries("bn_5e12").unwrap().into_iter().find_map(|e| match e.contestant {
            BenchContestant::Model { model_id, .. }
                if e.status == BenchEntryStatus::Done && models.iter().any(|x| x.id == model_id) =>
            {
                Some(model_id)
            }
            _ => None,
        });
        let ds = m.datasets().unwrap().into_iter().find(|d| d.splits.train > 0).unwrap();
        let req = StartBenchLoop {
            name: None,
            teacher: teacher.unwrap(),
            pool: BenchSource::Dataset { dataset_id: ds.id, split: Some(Split::Train) },
            generations: Some(2),
            min_confidence: None,
            student: BenchStudent { epochs: Some(10), ..BenchStudent::default() },
        };
        let l = m.start_bench_loop("bn_5e12", &req).unwrap();
        assert!(l.status.is_active());
        assert_eq!(m.bench_loops("bn_5e12").unwrap()[0].id, l.id);
    }

    #[test]
    fn a_started_loop_runs_its_generations_and_ends() {
        let mut lp = MockBackend::new(0, true).bench_loops("bn_5e12").unwrap().remove(0);
        lp.status = BenchLoopStatus::Running;
        lp.config.generations = Some(2);
        lp.generations.truncate(1);
        lp.generations[0].map50_95 = None;
        lp.generations[0].score = None;
        // A teacher without an entry is benchmarked first.
        super::step_loop(&mut lp, 1.0, 0);
        assert_eq!(lp.phase, Some(BenchLoopPhase::Benchmarking));
        // Then each generation labels, trains and is benchmarked.
        super::step_loop(&mut lp, super::LOOP_TEACHER_S + 2.0, 0);
        assert_eq!((lp.phase, lp.generations.len()), (Some(BenchLoopPhase::Training), 2));
        assert!(lp.generations[0].map50_95.is_some() && lp.generations[1].map50_95.is_none());
        super::step_loop(&mut lp, super::LOOP_TEACHER_S + 2.0 * super::LOOP_GEN_S, 0);
        assert_eq!(lp.status, BenchLoopStatus::Done);
        assert_eq!(lp.generations.len(), 3);
        assert!(lp.generations.iter().all(|g| g.map50_95.is_some()));
        assert_eq!(lp.verdict, Some(BenchLoopVerdict::Improved));
    }

    #[test]
    fn a_loop_verdict_compares_the_students_map_with_the_teachers_interval() {
        let mut lp = MockBackend::new(0, true).bench_loops("bn_5e12").unwrap().remove(0);
        lp.status = BenchLoopStatus::Running;
        lp.config.generations = Some(1);
        lp.generations.truncate(1);
        // A teacher at 0.624 (0.607–0.642): the student's 0.654 is above the interval, though its own
        // interval's lower bound (0.634) is not.
        lp.generations[0].map50_95 = Some(0.624);
        lp.generations[0].ci95 = Some([0.607, 0.642]);
        lp.generations[0].score = Some(67.0);
        super::step_loop(&mut lp, super::LOOP_GEN_S, 0);
        assert_eq!(lp.verdict, Some(BenchLoopVerdict::Improved));
        // On the bound itself it holds.
        lp.status = BenchLoopStatus::Running;
        lp.generations.truncate(1);
        lp.generations[0].map50_95 = Some(0.40);
        lp.generations[0].ci95 = Some([0.37, 0.43]);
        super::step_loop(&mut lp, super::LOOP_GEN_S, 0);
        assert_eq!(lp.generations[1].map50_95, Some(0.43));
        assert_eq!(lp.verdict, Some(BenchLoopVerdict::Held));
    }
}
