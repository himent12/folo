//! Loops: the models that can teach one, the New loop form, and applying and stopping a loop.

use gpui::Context;

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

    /// The models that can teach a loop on `b` (its classes, in its order), newest first.
    pub fn bench_teachers(&self, b: &Bench) -> Vec<Model> {
        let models = self.store.models.ready().into_iter().flatten();
        models.filter(|m| model_teaches(&m.classes, &b.classes)).cloned().collect()
    }

    /// Opens the New loop form for the benchmark on screen: its best model that can teach (else the
    /// newest that can), on the first source's train split.
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
        self.bench.form_seq += 1;
        self.bench.new_loop = Some(NewLoopForm {
            bench,
            seq: self.bench.form_seq,
            teacher,
            pool: 0,
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
        let Some(f) = self.bench.new_loop.as_mut() else { return };
        if f.submitting {
            return;
        }
        let Some(teacher) = f.teacher.clone() else {
            f.error = Some(ApiError::new("invalid_body", "Pick a teacher: a model of this server"));
            cx.notify();
            return;
        };
        let Some(pool) = sources.get(f.pool) else {
            f.error =
                Some(ApiError::new("invalid_body", "Add a dataset first: the teacher labels its images"));
            cx.notify();
            return;
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
