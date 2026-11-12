//! New benchmark: where images can come from, and the form that creates one.

use folo_ui::{AppContext, Context, Focusable, Window};

use super::state::{BENCH_SPLITS, NewBenchForm};
use crate::api::*;
use crate::app::{FoloApp, Route};
use crate::input::TextInput;
use crate::ui::Tone;

/// Where a benchmark's images, or a loop's pool, can come from.
#[derive(Clone, Debug, PartialEq)]
pub enum SourcePick {
    Dataset {
        id: String,
        name: String,
        splits: Splits,
    },
    /// A labeling job's labeled images.
    Job {
        id: String,
        name: String,
        labeled: u64,
    },
}

/// Which source a form picked, by kind and id: a deletion that shrinks the list doesn't move the
/// choice to another source.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceId {
    Dataset(String),
    Job(String),
}

/// What a form's source choice is in the current [`FoloApp::bench_sources`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceChoice {
    /// The source picked (nothing picked yet: the first) is at this index.
    At(usize),
    /// Nothing to pick from (or its list isn't in yet).
    Missing,
    /// The source picked was deleted while the form was open: it asks for another.
    Deleted,
}

impl SourceChoice {
    pub fn index(self) -> Option<usize> {
        match self {
            SourceChoice::At(i) => Some(i),
            _ => None,
        }
    }
}

impl SourcePick {
    pub fn id(&self) -> SourceId {
        match self {
            SourcePick::Dataset { id, .. } => SourceId::Dataset(id.clone()),
            SourcePick::Job { id, .. } => SourceId::Job(id.clone()),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            SourcePick::Dataset { name, .. } | SourcePick::Job { name, .. } => name,
        }
    }

    /// How many images it holds (a dataset in `split`).
    pub fn images(&self, split: Split) -> u64 {
        match self {
            SourcePick::Dataset { splits, .. } => match split {
                Split::Train => splits.train,
                Split::Val => splits.val,
                Split::Test => splits.test,
            },
            SourcePick::Job { labeled, .. } => *labeled,
        }
    }

    /// `split` (an index into [`BENCH_SPLITS`]) when it has images, else the first split that has
    /// some (an empty one can't make a benchmark), else `split` as it is.
    pub fn split_with_images(&self, split: usize) -> usize {
        let has = |i: usize| self.images(BENCH_SPLITS[i]) > 0;
        if has(split) { split } else { (0..BENCH_SPLITS.len()).find(|&i| has(i)).unwrap_or(split) }
    }

    /// The source a request sends.
    pub fn source(&self, split: Split) -> BenchSource {
        match self {
            SourcePick::Dataset { id, .. } => {
                BenchSource::Dataset { dataset_id: id.clone(), split: Some(split) }
            }
            SourcePick::Job { id, .. } => BenchSource::LabelJob { job_id: id.clone() },
        }
    }

    /// The name the New benchmark form suggests for it.
    pub fn suggested_name(&self, split: Split) -> String {
        match self {
            SourcePick::Dataset { name, .. } => format!("{name} {}", split.as_str()),
            SourcePick::Job { name, .. } => name.clone(),
        }
    }
}

impl FoloApp {
    /// Where images can come from: the datasets, then the labeling jobs with labeled images.
    pub fn bench_sources(&self) -> Vec<SourcePick> {
        let datasets = self.store.datasets.ready().into_iter().flatten().map(|d| SourcePick::Dataset {
            id: d.id.clone(),
            name: d.name.clone(),
            splits: d.splits,
        });
        let jobs =
            self.bench.jobs.ready().into_iter().flatten().filter(|j| j.counts.labeled > 0).map(|j| {
                SourcePick::Job { id: j.id.clone(), name: j.name.clone(), labeled: j.counts.labeled }
            });
        datasets.chain(jobs).collect()
    }

    /// Where a form's `picked` source is in `sources` (from [`Self::bench_sources`]): deleted when
    /// the list it comes from is in without it.
    pub fn source_choice(&self, sources: &[SourcePick], picked: Option<&SourceId>) -> SourceChoice {
        let Some(id) = picked else {
            return if sources.is_empty() { SourceChoice::Missing } else { SourceChoice::At(0) };
        };
        if let Some(i) = sources.iter().position(|s| s.id() == *id) {
            return SourceChoice::At(i);
        }
        let listed = match id {
            SourceId::Dataset(_) => self.store.datasets.ready().is_some(),
            SourceId::Job(_) => self.bench.jobs.ready().is_some(),
        };
        if listed { SourceChoice::Deleted } else { SourceChoice::Missing }
    }

    /// Opens the New benchmark form, named after the first source.
    pub fn open_new_bench(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        // The first source's validation split, or the first of its splits that has images.
        let first = self.bench_sources().into_iter().next();
        let source = first.as_ref().map(SourcePick::id);
        let split = first.as_ref().map_or(0, |s| s.split_with_images(0));
        let name = first.map_or_else(|| "benchmark".to_string(), |s| s.suggested_name(BENCH_SPLITS[split]));
        let input = cx.new(|cx| {
            let mut i = TextInput::new("Benchmark name", cx);
            i.set_text(name, cx);
            i
        });
        let fh = input.read(cx).focus_handle(cx);
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        self.bench.new_bench =
            Some(NewBenchForm { name: input, seq, source, split, submitting: false, error: None });
        window.focus(&fh);
        cx.notify();
    }

    /// Sets the New benchmark form's source and split, and its name when the name still follows them.
    pub fn pick_bench_source(
        &mut self,
        source: Option<SourceId>,
        split: Option<usize>,
        cx: &mut Context<Self>,
    ) {
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_bench.as_ref() else { return };
        let at = |id: Option<&SourceId>| self.source_choice(&sources, id).index().map(|i| &sources[i]);
        let old = at(f.source.as_ref()).map(|x| x.suggested_name(BENCH_SPLITS[f.split]));
        let source_changed = source.is_some();
        let source = source.or_else(|| f.source.clone());
        let next = at(source.as_ref()).cloned();
        let Some(f) = self.bench.new_bench.as_mut() else { return };
        f.source = source;
        f.split = split.unwrap_or(f.split).min(BENCH_SPLITS.len() - 1);
        // Another source: off a split it has no images in (the control disables those).
        if source_changed && let Some(s) = &next {
            f.split = s.split_with_images(f.split);
        }
        f.error = None;
        let current = f.name.read(cx).text().to_string();
        if old.is_none_or(|o| o == current)
            && let Some(next) = next.map(|s| s.suggested_name(BENCH_SPLITS[f.split]))
        {
            f.name.update(cx, |input, cx| input.set_text(next, cx));
        }
        cx.notify();
    }

    /// Creates the benchmark the form describes (the default plan) and opens it.
    pub fn submit_new_bench(&mut self, cx: &mut Context<Self>) {
        let sources = self.bench_sources();
        let picked = self.bench.new_bench.as_ref().and_then(|f| f.source.as_ref());
        let choice = self.source_choice(&sources, picked);
        let Some(f) = self.bench.new_bench.as_mut() else { return };
        if f.submitting {
            return;
        }
        let name = f.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            f.error = Some(ApiError::new("invalid_body", "Name the benchmark"));
            cx.notify();
            return;
        }
        let src = match choice {
            SourceChoice::At(i) => &sources[i],
            SourceChoice::Missing => {
                f.error =
                    Some(ApiError::new("invalid_body", "Add a dataset first: a benchmark copies its images"));
                cx.notify();
                return;
            }
            SourceChoice::Deleted => {
                f.error = Some(ApiError::new("invalid_body", "That source was deleted: pick another"));
                cx.notify();
                return;
            }
        };
        let split = BENCH_SPLITS[f.split];
        if src.images(split) == 0 {
            f.error = Some(ApiError::new("invalid_body", "That source has no images: pick another"));
            cx.notify();
            return;
        }
        let req = CreateBench {
            name,
            description: String::new(),
            source: src.source(split),
            max_images: None,
            seed: 0,
            plan: None,
        };
        f.submitting = true;
        f.error = None;
        let seq = f.seq;
        self.fetch(
            cx,
            move |api| api.create_bench(&req),
            move |this, r, cx| {
                // The form that asked, still open (a late answer leaves a newer form alone).
                let mine = this.bench.new_bench.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(b) => {
                        this.touch(&b.id);
                        if let Some(list) = this.bench.list.ready_mut()
                            && !list.iter().any(|x| x.id == b.id)
                        {
                            list.insert(0, b.clone());
                        }
                        this.toast(Tone::Success, format!("Created {}", b.name), None, cx);
                        if mine {
                            this.bench.new_bench = None;
                            this.route_pending = Some(Route::Bench(b.id));
                        }
                    }
                    Err(e) => {
                        if let Some(f) = this.bench.new_bench.as_mut().filter(|f| f.seq == seq) {
                            f.submitting = false;
                            f.error = Some(e);
                        } else {
                            this.toast(
                                Tone::Danger,
                                "Couldn't create the benchmark",
                                Some(e.message.into()),
                                cx,
                            );
                        }
                    }
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
    fn sources_name_and_count_what_a_benchmark_would_copy() {
        let ds = SourcePick::Dataset {
            id: "ds_1".into(),
            name: "pond".into(),
            splits: Splits { train: 800, val: 200, test: 0 },
        };
        assert_eq!(ds.suggested_name(Split::Val), "pond val");
        assert_eq!(ds.images(Split::Train), 800);
        assert_eq!(ds.images(Split::Test), 0);
        // BENCH_SPLITS is val, test, train: an empty test split gives way to val, an empty val
        // and test to train.
        assert_eq!(ds.split_with_images(1), 0);
        assert_eq!(ds.split_with_images(2), 2);
        let train_only = SourcePick::Dataset {
            id: "ds_2".into(),
            name: "t".into(),
            splits: Splits { train: 9, val: 0, test: 0 },
        };
        assert_eq!(train_only.split_with_images(0), 2);
        assert_eq!(
            ds.source(Split::Test),
            BenchSource::Dataset { dataset_id: "ds_1".into(), split: Some(Split::Test) }
        );
        // A labeling job has no splits: its labeled images, whatever split the form shows.
        let job = SourcePick::Job { id: "lb_1".into(), name: "rejects".into(), labeled: 120 };
        assert_eq!(job.suggested_name(Split::Val), "rejects");
        assert_eq!(job.images(Split::Test), 120);
        assert_eq!(job.source(Split::Val), BenchSource::LabelJob { job_id: "lb_1".into() });
    }
}
