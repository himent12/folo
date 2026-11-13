//! Loops: the models that can teach one, the New loop form, and applying and stopping a loop.

use folo_ui::Context;

use super::create::{SourceChoice, SourceId, SourcePick};
use super::load::route_bench;
use super::merge::upsert_loop;
use super::run::model_teaches;
use super::state::{LOOP_EPOCHS, LOOP_GENERATIONS, NewLoopForm};
use crate::api::*;
use crate::app::FoloApp;
use crate::ui::Tone;

impl FoloApp {
    /// Applies a loop the server answered with, as [`Self::apply_entry`] does.
    pub fn apply_loop(&mut self, bench: &str, l: BenchLoop, since: u64) {
        if self.bench.touched.get(&l.id).is_some_and(|&s| s > since) {
            return;
        }
        self.touch(&l.id);
        if let Some(list) = self.bench.loops.get_mut(bench).and_then(|x| x.ready_mut()) {
            upsert_loop(list, l);
        }
    }

    /// The New loop form's teacher was deleted (its run was) while the form was open: the model
    /// list is in without it.
    pub fn loop_teacher_deleted(&self) -> bool {
        let Some(id) = self.bench.new_loop.as_ref().and_then(|f| f.teacher.as_ref()) else { return false };
        self.store.models.ready().is_some_and(|l| !l.iter().any(|m| m.id == *id))
    }

    /// The models that can teach a loop on `b` (its classes, in its order), newest first.
    pub fn bench_teachers(&self, b: &Bench) -> Vec<Model> {
        let models = self.store.models.ready().into_iter().flatten();
        models.filter(|m| model_teaches(&m.classes, &b.classes)).cloned().collect()
    }

    /// Opens the New loop form for the benchmark on screen: its best model that can teach (else the
    /// newest that can), on its own source's train split (else the first source's).
    pub fn open_new_loop(&mut self, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        let Some(bench) = route_bench(&self.route).map(str::to_string) else { return };
        let b = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == bench)).cloned();
        let teachers = b.as_ref().map(|b| self.bench_teachers(b)).unwrap_or_default();
        let best = self.bench.entries.get(&bench).and_then(|l| l.ready()).and_then(|l| {
            l.iter().filter(|e| e.status == BenchEntryStatus::Done).find_map(|e| match &e.contestant {
                BenchContestant::Model { model_id, .. } if teachers.iter().any(|m| &m.id == model_id) => {
                    Some(model_id.clone())
                }
                _ => None,
            })
        });
        let teacher = best.or_else(|| teachers.first().map(|m| m.id.clone()));
        // The pool: the benchmark's own source while it is listed (the server leaves the benchmark's
        // images out of it), as Studio does, else the first source.
        let sources = self.bench_sources();
        let own = b.as_ref().map(|b| match &b.source {
            BenchSource::Dataset { dataset_id, .. } => SourceId::Dataset(dataset_id.clone()),
            BenchSource::LabelJob { job_id } => SourceId::Job(job_id.clone()),
        });
        let pool = own
            .filter(|own| sources.iter().any(|s| s.id() == *own))
            .or_else(|| sources.first().map(SourcePick::id));
        self.bench.form_seq += 1;
        self.bench.new_loop = Some(NewLoopForm {
            bench,
            seq: self.bench.form_seq,
            teacher,
            pool,
            generations: 0,
            epochs: 1,
            submitting: false,
            error: None,
        });
        cx.notify();
    }

    /// Starts the loop the New loop form describes.
    pub fn submit_new_loop(&mut self, bench: &str, cx: &mut Context<Self>) {
        let sources = self.bench_sources();
        let choice = self.source_choice(&sources, self.bench.new_loop.as_ref().and_then(|f| f.pool.as_ref()));
        let teacher_gone = self.loop_teacher_deleted();
        let Some(f) = self.bench.new_loop.as_mut() else { return };
        if f.submitting {
            return;
        }
        let refuse = |f: &mut NewLoopForm, why: &str, cx: &mut Context<Self>| {
            f.error = Some(ApiError::new("invalid_body", why));
            cx.notify();
        };
        let teacher = match f.teacher.clone() {
            Some(_) if teacher_gone => return refuse(f, "Its teacher was deleted: pick another", cx),
            Some(t) => t,
            None => return refuse(f, "Pick a teacher: a model of this server", cx),
        };
        let pool = match choice {
            SourceChoice::At(i) => &sources[i],
            SourceChoice::Missing => {
                return refuse(f, "Add a dataset first: the teacher labels its images", cx);
            }
            SourceChoice::Deleted => return refuse(f, "Its pool was deleted: pick another", cx),
        };
        let req = StartBenchLoop {
            name: None,
            teacher,
            pool: pool.source(Split::Train),
            generations: Some(LOOP_GENERATIONS[f.generations]),
            min_confidence: None,
            student: BenchStudent { epochs: Some(LOOP_EPOCHS[f.epochs]), ..BenchStudent::default() },
        };
        f.submitting = true;
        f.error = None;
        let seq = f.seq;
        let (a, b) = (bench.to_string(), bench.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.start_bench_loop(&a, &req),
            move |this, r, cx| {
                // Only the form that asked (same benchmark, same opening) closes or shows the error.
                let mine = |f: &NewLoopForm| f.seq == seq && f.bench == b;
                match r {
                    Ok(l) => {
                        if this.bench.new_loop.as_ref().is_some_and(mine) {
                            this.bench.new_loop = None;
                        }
                        let name = l.name.clone();
                        this.apply_loop(&b, l, since);
                        this.toast(
                            Tone::Info,
                            format!("{name} started"),
                            Some(
                                "Its generations appear on its benchmark's Loops tab as they finish.".into(),
                            ),
                            cx,
                        );
                        this.refresh_benches(cx);
                    }
                    Err(e) => match this.bench.new_loop.as_mut().filter(|f| mine(f)) {
                        Some(f) => {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                        None => {
                            this.toast(Tone::Danger, "Couldn't start the loop", Some(e.message.into()), cx)
                        }
                    },
                }
            },
        );
        cx.notify();
    }

    /// Stops a running loop.
    pub fn stop_bench_loop(&mut self, bench: &str, id: &str, cx: &mut Context<Self>) {
        if !self.bench.busy.insert(id.to_string()) {
            return;
        }
        let (a, l1, b, l2) = (bench.to_string(), id.to_string(), bench.to_string(), id.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.stop_bench_loop(&a, &l1),
            move |this, r, cx| {
                this.bench.busy.remove(&l2);
                match r {
                    Ok(l) => this.apply_loop(&b, l, since),
                    Err(e) => this.toast(Tone::Danger, "Couldn't stop the loop", Some(e.message.into()), cx),
                }
            },
        );
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use crate::api::BenchSource;
    use crate::app::Route;
    use crate::app::tests::{drive, open};
    use crate::bench::SourceId;

    #[test]
    fn a_new_loop_labels_the_benchmarks_own_dataset_by_default() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, _, cx| this.ensure_benches(cx));
        let benches = drive(&mut app, window, |this, _, _| this.bench.list.ready().unwrap().clone());
        // Every demo benchmark, whichever dataset comes first in the list.
        for b in benches {
            let BenchSource::Dataset { dataset_id, .. } = &b.source else { continue };
            drive(&mut app, window, |this, _, cx| {
                this.route = Route::Bench(b.id.clone());
                this.open_new_loop(cx);
                let pool = this.bench.new_loop.as_ref().unwrap().pool.clone();
                assert_eq!(pool, Some(SourceId::Dataset(dataset_id.clone())), "{}", b.name);
            });
        }
    }
}
