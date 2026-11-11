//! Contestants: which models and coding agents can run on a benchmark, running one, and stopping an entry.

use folo_ui::Context;

use super::merge::upsert_entry;
use super::state::Contender;
use crate::api::*;
use crate::app::FoloApp;
use crate::store::Load;
use crate::ui::Tone;

/// A class name as the server compares them: trimmed, without case.
fn class_key(s: &str) -> String {
    s.trim().to_lowercase()
}

/// A model can run on a benchmark when it knows at least one of its classes (by name, without
/// case); the server refuses one that knows none (`bench_class_mismatch`).
pub fn model_fits(model: &[String], bench: &[String]) -> bool {
    model.iter().any(|m| bench.iter().any(|b| class_key(b) == class_key(m)))
}

/// A model can teach a loop on a benchmark when its classes are the benchmark's, the same names in
/// the same order: its labels are written by class index.
pub fn model_teaches(model: &[String], bench: &[String]) -> bool {
    model.len() == bench.len() && model.iter().zip(bench).all(|(m, b)| class_key(m) == class_key(b))
}

impl FoloApp {
    // ---------- actions ----------

    /// The server runs coding agents (`agent_labeling`): agents can be contestants and design plans,
    /// and labeling jobs can be sources.
    pub fn agents_on(&self) -> bool {
        self.store.features.iter().any(|f| f == FEATURE_AGENT_LABELING)
    }

    /// Loads the server's coding agents and labeling jobs, once each (servers with `agent_labeling`).
    pub fn ensure_agents(&mut self, cx: &mut Context<Self>) {
        if !self.agents_on() {
            return;
        }
        if self.bench.agents.needs_fetch() {
            self.bench.agents = Load::Loading;
            self.fetch(cx, |api| api.label_agents(), |this, r, _| this.bench.agents = r.into());
        }
        if self.bench.jobs.needs_fetch() {
            self.bench.jobs = Load::Loading;
            self.fetch(cx, |api| api.label_jobs(), |this, r, _| this.bench.jobs = r.into());
        }
    }

    /// The agents that are installed on the server's machine.
    pub fn bench_agents(&self) -> Vec<LabelAgent> {
        self.bench
            .agents
            .ready()
            .map(|l| l.iter().filter(|a| a.available).cloned().collect())
            .unwrap_or_default()
    }

    /// Runs the picked contestant on a benchmark: a model on every image and the plan's conditions, an
    /// agent as the server runs one by default (its first images, unchanged only).
    pub fn run_bench_model(&mut self, bench: &str, cx: &mut Context<Self>) {
        let Some(b) = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == bench)).cloned() else {
            return;
        };
        let Some(who) = self.bench_run_model(&b) else { return };
        if self.bench.starting {
            return;
        }
        self.bench.starting = true;
        let contestant = match who {
            Contender::Model(model_id) => BenchContestantSpec::Model { model_id },
            Contender::Agent(agent) => BenchContestantSpec::Agent { agent, model: None, effort: None },
        };
        let req = StartBenchEntry { contestant, name: None, max_images: None, conditions: None };
        let (a, b) = (bench.to_string(), bench.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.start_bench_entry(&a, &req),
            move |this, r, cx| {
                this.bench.starting = false;
                match r {
                    Ok(e) => {
                        let name = e.name.clone();
                        this.apply_entry(&b, e, since);
                        this.toast(
                            Tone::Info,
                            format!("{name} started"),
                            Some("It appears on the leaderboard as it runs.".into()),
                            cx,
                        );
                        this.refresh_benches(cx);
                    }
                    Err(e) => {
                        this.toast(Tone::Danger, "Couldn't start the entry", Some(e.message.into()), cx)
                    }
                }
            },
        );
    }

    /// Who Run would start on `b`: the contestant picked while it can still run there, else the newest
    /// model that knows its classes, else the first installed agent.
    pub fn bench_run_model(&self, b: &Bench) -> Option<Contender> {
        let options = self.bench_contenders(b);
        self.bench
            .run_model
            .as_ref()
            .filter(|c| options.contains(c))
            .cloned()
            .or_else(|| options.into_iter().next())
    }

    /// Everyone the leaderboard of `b` can run: the models that know its classes, newest first, then
    /// the installed agents (which are asked for its classes by name).
    pub fn bench_contenders(&self, b: &Bench) -> Vec<Contender> {
        let models = self
            .store
            .models
            .ready()
            .into_iter()
            .flatten()
            .filter(|m| model_fits(&m.classes, &b.classes))
            .map(|m| Contender::Model(m.id.clone()));
        let agents = self.bench_agents().into_iter().map(|a| Contender::Agent(a.id));
        models.chain(agents).collect()
    }

    /// Applies an entry the server answered with (a start or a stop), unless the stream has said
    /// something about it since the request went out (its answer is at least as new).
    pub fn apply_entry(&mut self, bench: &str, e: BenchEntry, since: u64) {
        if self.bench.touched.get(&e.id).is_some_and(|&s| s > since) {
            return;
        }
        self.touch(&e.id);
        if let Some(list) = self.bench.entries.get_mut(bench).and_then(|l| l.ready_mut()) {
            upsert_entry(list, e);
        }
    }

    /// Stops a queued or running entry.
    pub fn stop_bench_entry(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        if !self.bench.busy.insert(entry.to_string()) {
            return;
        }
        let (a, e1, b, e2) = (bench.to_string(), entry.to_string(), bench.to_string(), entry.to_string());
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.stop_bench_entry(&a, &e1),
            move |this, r, cx| {
                this.bench.busy.remove(&e2);
                match r {
                    Ok(e) => this.apply_entry(&b, e, since),
                    Err(e) => this.toast(Tone::Danger, "Couldn't stop the entry", Some(e.message.into()), cx),
                }
            },
        );
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_match_as_the_server_matches_them() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let bench = names(&["Person", "car", "dog"]);
        // A contestant needs one class in common (by name, without case or edge spaces).
        assert!(model_fits(&names(&["cat", " person "]), &bench));
        assert!(!model_fits(&names(&["circle", "square"]), &bench));
        // A teacher needs all of them, in the same order.
        assert!(model_teaches(&names(&["person", "Car", "dog"]), &bench));
        assert!(!model_teaches(&names(&["car", "person", "dog"]), &bench));
        assert!(!model_teaches(&names(&["person", "car"]), &bench));
    }
}
