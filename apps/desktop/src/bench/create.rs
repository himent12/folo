//! New benchmark: where images can come from, and the form that creates one.

use folo_ui::{AppContext, Context, Focusable, Window};

use super::state::{BENCH_GEN_IMGSZ, BENCH_SPLITS, BenchSourceKind, NewBenchForm};
use crate::api::*;
use crate::app::{FoloApp, Route, SYNTH_CLASSES};
use crate::input::TextInput;
use crate::ui::Tone;

/// Default benchmark name suggested when creating a generated-source benchmark.
const GENERATED_SUGGESTED_NAME: &str = "generated-shapes";

/// True when `source` is the wire representation of a server-generated synthetic benchmark
/// (`{ "kind": "dataset", "dataset_id": "generated", "split": null }`).
pub fn is_generated_source(source: &BenchSource) -> bool {
    matches!(source, BenchSource::Dataset { dataset_id, split: None } if dataset_id == GENERATED_BENCH_SOURCE_ID)
}

/// Parses a generated benchmark's image count (`1..=5000`), accepting plain digits or thousands
/// grouped by `,`, `_`, or space (rejecting decimal commas like `1,5`).
pub fn parse_bench_gen_images(s: &str) -> Option<u32> {
    let t = s.trim();
    let t = t.strip_prefix('+').unwrap_or(t);
    if t.is_empty() || t.starts_with('+') {
        return None;
    }
    let groups: Vec<&str> = t.split([',', '_', ' ']).collect();
    if let [head, rest @ ..] = groups.as_slice()
        && !rest.is_empty()
        && !((1..=3).contains(&head.len()) && rest.iter().all(|g| g.len() == 3))
    {
        return None;
    }
    groups.concat().parse::<u32>().ok().filter(|v| (1..=folo_bench::source::MAX_IMAGES).contains(v))
}

/// Parses a generated benchmark's deterministic RNG seed (`0..=2^53 - 1`).
pub fn parse_bench_gen_seed(s: &str) -> Option<u64> {
    let t = s.trim();
    let t = t.strip_prefix('+').unwrap_or(t);
    if t.is_empty() || t.starts_with('+') {
        return None;
    }
    t.parse::<u64>().ok().filter(|v| *v <= folo_bench::source::MAX_SEED)
}

/// Parses comma-separated class names (`1..=8` unique non-empty names) for a generated benchmark.
pub fn parse_bench_gen_classes(s: &str) -> Result<Vec<String>, String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err("Enter 1 to 8 comma-separated class names.".into());
    }
    let mut out = Vec::new();
    for part in trimmed.split(',') {
        let name = part.trim();
        if name.is_empty() {
            return Err("Remove empty class names between commas.".into());
        }
        out.push(name.to_string());
    }
    let probe = folo_bench::GeneratedSourceSpec {
        images: 1,
        classes: out.clone(),
        imgsz: folo_bench::MIN_IMGSZ,
        seed: 0,
        vary_aspect: false,
    };
    probe.to_synthetic_config()?;
    Ok(out)
}

/// Field-level validation errors for the New benchmark form's Generated source mode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeneratedBenchErrors {
    pub images: Option<&'static str>,
    pub seed: Option<&'static str>,
    pub classes: Option<String>,
}

impl GeneratedBenchErrors {
    pub fn any(&self) -> bool {
        self.images.is_some() || self.seed.is_some() || self.classes.is_some()
    }
}

/// Validates all Generated-source inputs against `folo-bench::GeneratedSourceSpec` rules.
pub fn validate_generated_bench(
    images_text: &str,
    seed_text: &str,
    classes_text: &str,
    imgsz: u32,
    vary_aspect: bool,
) -> Result<folo_bench::GeneratedSourceSpec, GeneratedBenchErrors> {
    let mut errs = GeneratedBenchErrors::default();
    let images = parse_bench_gen_images(images_text);
    if images.is_none() {
        errs.images = Some("Whole number 1 to 5,000");
    }
    let seed = parse_bench_gen_seed(seed_text);
    if seed.is_none() {
        errs.seed = Some("Whole number 0 to 9,007,199,254,740,991");
    }
    let classes = match parse_bench_gen_classes(classes_text) {
        Ok(c) => Some(c),
        Err(e) => {
            errs.classes = Some(e);
            None
        }
    };
    if errs.any() {
        return Err(errs);
    }
    let spec = folo_bench::GeneratedSourceSpec {
        images: images.unwrap(),
        classes: classes.unwrap(),
        imgsz,
        seed: seed.unwrap(),
        vary_aspect,
    };
    if let Err(e) = spec.to_synthetic_config() {
        errs.classes = Some(e);
        return Err(errs);
    }
    Ok(spec)
}

/// Where a benchmark's images, or a loop's pool, can come from.
#[derive(Clone, Debug, PartialEq)]
pub enum SourcePick {
    Dataset {
        id: String,
        name: String,
        splits: Splits,
    },
    /// A labeling job's labeled images (for benchmarks) and total images (for loop pools).
    Job {
        id: String,
        name: String,
        labeled: u64,
        images: u64,
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

    /// How many images it holds (a dataset in `split`; for a labeling job, its labeled images).
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

    /// How many images a self-improvement loop pool reads from this source (a dataset's `train`
    /// split, or all images of a labeling job including unlabeled/pending ones).
    pub fn pool_images(&self) -> u64 {
        match self {
            SourcePick::Dataset { splits, .. } => splits.train,
            SourcePick::Job { images, .. } => *images,
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
    /// True when the connected server advertises `bench_generated_source` (`POST /api/bench/benchmarks`
    /// with `source.kind == "generated"`).
    pub fn bench_generated_on(&self) -> bool {
        self.store.features.iter().any(|f| f == FEATURE_BENCH_GENERATED_SOURCE)
    }

    /// Where images can come from: the datasets, then the labeling jobs with labeled images.
    pub fn bench_sources(&self) -> Vec<SourcePick> {
        let datasets = self.store.datasets.ready().into_iter().flatten().map(|d| SourcePick::Dataset {
            id: d.id.clone(),
            name: d.name.clone(),
            splits: d.splits,
        });
        let jobs = self.bench.jobs.ready().into_iter().flatten().filter(|j| j.counts.labeled > 0).map(|j| {
            SourcePick::Job {
                id: j.id.clone(),
                name: j.name.clone(),
                labeled: j.counts.labeled,
                images: j.counts.images,
            }
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

    /// Opens the New benchmark form, named after the first source (or Generated when the workspace
    /// has no datasets/labeling jobs yet and the server supports `bench_generated_source`).
    pub fn open_new_bench(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        // The first source's validation split, or the first of its splits that has images.
        let first = self.bench_sources().into_iter().next();
        let source_kind = if first.is_none() && self.bench_generated_on() {
            BenchSourceKind::Generated
        } else {
            BenchSourceKind::Existing
        };
        let source = first.as_ref().map(SourcePick::id);
        let split = first.as_ref().map_or(0, |s| s.split_with_images(0));
        let name = if source_kind == BenchSourceKind::Generated {
            GENERATED_SUGGESTED_NAME.to_string()
        } else {
            first.map_or_else(|| "benchmark".to_string(), |s| s.suggested_name(BENCH_SPLITS[split]))
        };
        let input = cx.new(|cx| {
            let mut i = TextInput::new("Benchmark name", cx);
            i.set_text(name, cx);
            i
        });
        let gen_images = cx.new(|cx| {
            let mut i = TextInput::new("64", cx);
            i.set_text(folo_bench::DEFAULT_IMAGES.to_string(), cx);
            i
        });
        let gen_classes = cx.new(|cx| {
            let mut i = TextInput::new("circle, square, triangle", cx);
            i.set_text(folo_bench::DEFAULT_CLASSES.join(", "), cx);
            i
        });
        let gen_seed = cx.new(|cx| {
            let mut i = TextInput::new("0", cx);
            i.set_text("0".to_string(), cx);
            i
        });
        let fh = input.read(cx).focus_handle(cx);
        self.bench.form_seq += 1;
        let seq = self.bench.form_seq;
        self.bench.new_bench = Some(NewBenchForm {
            name: input,
            seq,
            source_kind,
            source,
            split,
            gen_images,
            gen_imgsz: 2, // 640 px
            gen_classes,
            gen_seed,
            gen_vary_aspect: false,
            submitting: false,
            error: None,
        });
        window.focus(&fh);
        cx.notify();
    }

    /// Switches the New benchmark form between Existing (dataset / labeling job) and Generated
    /// scenes, updating the suggested name when the user hasn't edited it away from the old default.
    pub fn pick_bench_source_kind(&mut self, kind: BenchSourceKind, cx: &mut Context<Self>) {
        if kind == BenchSourceKind::Generated && !self.bench_generated_on() {
            return;
        }
        let sources = self.bench_sources();
        let Some(f) = self.bench.new_bench.as_ref() else { return };
        if f.source_kind == kind {
            return;
        }
        let existing_suggested = self
            .source_choice(&sources, f.source.as_ref())
            .index()
            .map(|i| sources[i].suggested_name(BENCH_SPLITS[f.split]))
            .unwrap_or_else(|| "benchmark".to_string());
        let old_suggested = match f.source_kind {
            BenchSourceKind::Existing => existing_suggested.clone(),
            BenchSourceKind::Generated => GENERATED_SUGGESTED_NAME.to_string(),
        };
        let next_suggested = match kind {
            BenchSourceKind::Existing => existing_suggested,
            BenchSourceKind::Generated => GENERATED_SUGGESTED_NAME.to_string(),
        };
        let Some(f) = self.bench.new_bench.as_mut() else { return };
        f.source_kind = kind;
        f.error = None;
        let current = f.name.read(cx).text().to_string();
        if current == old_suggested || current.trim().is_empty() {
            f.name.update(cx, |input, cx| input.set_text(next_suggested, cx));
        }
        cx.notify();
    }

    /// Fills the generated-source Classes input with one of the built-in shape presets (`3`, `5`, `8`).
    pub fn set_bench_gen_class_preset(&mut self, preset: usize, cx: &mut Context<Self>) {
        let Some(classes) = SYNTH_CLASSES.get(preset) else { return };
        let Some(f) = self.bench.new_bench.as_mut() else { return };
        let joined = classes.join(", ");
        f.gen_classes.update(cx, |input, cx| input.set_text(joined, cx));
        f.error = None;
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
        let generated_on = self.bench_generated_on();
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
        if name.chars().count() > 100 {
            f.error = Some(ApiError::new("invalid_body", "Benchmark name must be at most 100 characters"));
            cx.notify();
            return;
        }
        enum BenchCreateJob {
            Existing(CreateBench),
            Generated(CreateGeneratedBench),
        }
        let job = match f.source_kind {
            BenchSourceKind::Existing => {
                let src = match choice {
                    SourceChoice::At(i) => &sources[i],
                    SourceChoice::Missing => {
                        f.error = Some(ApiError::new(
                            "invalid_body",
                            "Add a dataset or labeling job first: a benchmark copies its images",
                        ));
                        cx.notify();
                        return;
                    }
                    SourceChoice::Deleted => {
                        f.error =
                            Some(ApiError::new("invalid_body", "That source was deleted: pick another"));
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
                BenchCreateJob::Existing(CreateBench {
                    name,
                    description: String::new(),
                    source: src.source(split),
                    max_images: None,
                    seed: 0,
                    plan: None,
                })
            }
            BenchSourceKind::Generated => {
                if !generated_on {
                    f.error = Some(ApiError::new(
                        "not_supported",
                        "This server doesn't advertise generated benchmark sources",
                    ));
                    cx.notify();
                    return;
                }
                let imgsz = BENCH_GEN_IMGSZ[f.gen_imgsz.min(BENCH_GEN_IMGSZ.len() - 1)];
                let images_text = f.gen_images.read(cx).text().to_string();
                let seed_text = f.gen_seed.read(cx).text().to_string();
                let classes_text = f.gen_classes.read(cx).text().to_string();
                let spec = match validate_generated_bench(
                    &images_text,
                    &seed_text,
                    &classes_text,
                    imgsz,
                    f.gen_vary_aspect,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        let msg =
                            e.images.or(e.seed).map(str::to_string).or(e.classes).unwrap_or_else(|| {
                                "Fix the highlighted generated-source fields first".into()
                            });
                        f.error = Some(ApiError::new("invalid_body", msg));
                        cx.notify();
                        return;
                    }
                };
                BenchCreateJob::Generated(CreateGeneratedBench {
                    name,
                    description: String::new(),
                    source: GeneratedBenchSource {
                        kind: "generated",
                        images: spec.images,
                        classes: spec.classes,
                        imgsz: spec.imgsz,
                        seed: spec.seed,
                        vary_aspect: spec.vary_aspect,
                    },
                    plan: None,
                })
            }
        };
        f.submitting = true;
        f.error = None;
        let seq = f.seq;
        self.fetch(
            cx,
            move |api| match &job {
                BenchCreateJob::Existing(req) => api.create_bench(req),
                BenchCreateJob::Generated(req) => api.create_generated_bench(req),
            },
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
        let job = SourcePick::Job { id: "lb_1".into(), name: "rejects".into(), labeled: 120, images: 150 };
        assert_eq!(job.suggested_name(Split::Val), "rejects");
        assert_eq!(job.images(Split::Test), 120);
        assert_eq!(job.pool_images(), 150);
        assert_eq!(job.source(Split::Val), BenchSource::LabelJob { job_id: "lb_1".into() });
    }
}
