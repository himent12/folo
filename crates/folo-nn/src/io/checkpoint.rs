//! Checkpoints: the full training state (model, EMA, optimizer, schedule, RNG streams, epoch, early-stopping
//! state) with the metadata contracts of docs/FORMAT.md.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use folo_core::Tensor;
use folo_core::rng::Rng;
use serde::{Deserialize, Serialize};

use super::file::{LoadOptions, LoadReport, ModelFile, Namespace, Purpose};
use super::meta::{FileKind, Metadata};
use super::safetensors;
use crate::ema::{EmaContract, EmaState};
use crate::module::Module;
use crate::optim::{OptimContract, OptimKind, Optimizer};
use crate::sched::{LrSchedule, SchedContract};
use crate::{Error, Result};

/// Named RNG streams (`folo.rng`): the run seed and each stream's exact position.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RngState {
    /// The run seed.
    pub seed: u64,
    /// Stream name (`data`, `augment`, `init`, ...) → generator.
    pub streams: BTreeMap<String, Rng>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RngJson {
    algo: String,
    seed: u64,
    streams: BTreeMap<String, String>,
}

impl RngState {
    /// The `folo.rng` JSON: each stream's four state words, each as 8 little-endian bytes, as lowercase hex.
    pub fn to_json(&self) -> String {
        let streams = self
            .streams
            .iter()
            .map(|(k, r)| {
                let hex: String =
                    r.state().iter().flat_map(|w| w.to_le_bytes()).map(|b| format!("{b:02x}")).collect();
                (k.clone(), hex)
            })
            .collect();
        serde_json::to_string(&RngJson { algo: Rng::ALGO.into(), seed: self.seed, streams })
            .expect("serialises")
    }

    /// Parse `folo.rng`.
    pub fn from_json(s: &str) -> Result<Self> {
        let bad = |m: String| Error::Metadata(format!("folo.rng: {m}"));
        let j: RngJson = serde_json::from_str(s).map_err(|e| bad(e.to_string()))?;
        if j.algo != Rng::ALGO {
            return Err(bad(format!("generator `{}` is not supported (expected {})", j.algo, Rng::ALGO)));
        }
        let mut streams = BTreeMap::new();
        for (name, hex) in j.streams {
            let digits = hex.as_bytes();
            if digits.len() != 64 || !digits.iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)) {
                return Err(bad(format!("stream `{name}`: expected 64 lowercase hex digits")));
            }
            let byte = |i: usize| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("validated");
            let words: [u64; 4] =
                std::array::from_fn(|w| u64::from_le_bytes(std::array::from_fn(|b| byte(8 * w + b))));
            let rng = Rng::from_state(words)
                .ok_or_else(|| bad(format!("stream `{name}` has the all-zero state")))?;
            streams.insert(name, rng);
        }
        Ok(RngState { seed: j.seed, streams })
    }
}

/// Early-stopping state (`folo.best`, docs/MODEL.md §8.7).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BestState {
    /// Best epoch, −1 if none yet.
    pub epoch: i64,
    /// Best fitness, `None` while it is −∞.
    pub fitness: Option<f64>,
}

impl Default for BestState {
    fn default() -> Self {
        BestState { epoch: -1, fitness: None }
    }
}

/// The parsed, cross-checked checkpoint metadata.
#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    /// `folo.train` (informational).
    pub train: serde_json::Value,
    /// `folo.optim`.
    pub optim: OptimContract,
    /// `folo.sched`.
    pub sched: SchedContract,
    /// `folo.ema`, when the checkpoint has EMA tensors.
    pub ema: Option<EmaContract>,
    /// `ema.__updates`, when it has EMA.
    pub ema_updates: Option<u64>,
    /// `folo.rng`.
    pub rng: RngState,
    /// `folo.epoch`: completed epochs.
    pub epoch: u64,
    /// `folo.best`.
    pub best: BestState,
}

fn json<T: for<'de> Deserialize<'de>>(meta: &Metadata, key: &str) -> Result<T> {
    let s = meta.extra.get(key).ok_or_else(|| Error::Metadata(format!("checkpoint is missing `{key}`")))?;
    serde_json::from_str(s).map_err(|e| Error::Metadata(format!("{key}: {e}")))
}

/// Opening a checkpoint: parse the contracts and check them against the tensors.
pub(crate) fn validate(file: &ModelFile) -> Result<CheckpointMeta> {
    let meta = file.metadata();
    if meta.fused {
        return Err(Error::Metadata(
            "a checkpoint cannot be fused (folo.fused = \"true\" is inference-only)".into(),
        ));
    }
    let optim: OptimContract = json(meta, "folo.optim")?;
    let sched: SchedContract = json(meta, "folo.sched")?;
    let rng = RngState::from_json(
        meta.extra
            .get("folo.rng")
            .ok_or_else(|| Error::Metadata("checkpoint is missing `folo.rng`".into()))?,
    )?;
    let best: BestState = json(meta, "folo.best")?;
    let train: serde_json::Value = json(meta, "folo.train")?;
    let epoch: u64 = meta
        .extra
        .get("folo.epoch")
        .ok_or_else(|| Error::Metadata("checkpoint is missing `folo.epoch`".into()))?
        .parse()
        .map_err(|_| Error::Metadata("folo.epoch is not a non-negative integer".into()))?;

    let (o, s) = (file.counter("optim.__step")?, file.counter("sched.__step")?);
    if o != optim.step || s != sched.step || o != s {
        return Err(Error::Metadata(format!(
            "step counters disagree: optim.__step {o}, folo.optim.step {}, sched.__step {s}, folo.sched.step {}",
            optim.step, sched.step
        )));
    }
    LrSchedule::from_contract(sched.clone()).map_err(|e| Error::Metadata(format!("folo.sched: {e}")))?;
    if let Some(w) = &optim.musgd {
        if optim.kind != OptimKind::MuSgd {
            return Err(Error::Metadata(format!(
                "folo.optim: `musgd` on a `{}` contract",
                optim.kind.as_str()
            )));
        }
        w.validate().map_err(|e| Error::Metadata(format!("folo.optim: {e}")))?;
    }
    if optim.muon_ortho.is_some() && optim.kind != OptimKind::MuSgd {
        return Err(Error::Metadata(format!(
            "folo.optim: `muon_ortho` on a `{}` contract",
            optim.kind.as_str()
        )));
    }
    if optim.muon_geo.is_some() && optim.kind != OptimKind::MuSgd {
        return Err(Error::Metadata(format!(
            "folo.optim: `muon_geo` on a `{}` contract",
            optim.kind.as_str()
        )));
    }
    if !optim.muon_depthwise.is_empty()
        && (optim.kind != OptimKind::MuSgd || optim.muon_geo != Some(crate::optim::MuonGeoMode::Geo))
    {
        return Err(Error::Metadata(
            "folo.optim: `muon_depthwise` is only valid on a `musgd` contract with `muon_geo: \"geo\"`"
                .into(),
        ));
    }
    if !optim.param_steps.is_empty() && optim.kind != OptimKind::AdamW {
        return Err(Error::Metadata(format!(
            "folo.optim: `param_steps` on a `{}` contract",
            optim.kind.as_str()
        )));
    }

    // Every model tensor is in exactly one group, and every group parameter is a model tensor.
    let model: BTreeSet<&str> = file.names(Namespace::Model).into_iter().collect();
    let mut seen = BTreeSet::new();
    let mut muon_params = BTreeSet::new();
    for g in &optim.groups {
        let is_muon = crate::optim::is_muon_group(&g.name);
        for p in &g.params {
            if !seen.insert(p.as_str()) {
                return Err(Error::Metadata(format!("folo.optim: `{p}` is in more than one group")));
            }
            if !model.contains(p.as_str()) {
                return Err(Error::Metadata(format!(
                    "folo.optim: group `{}` lists `{p}`, which is not a model tensor",
                    g.name
                )));
            }
            if is_muon {
                muon_params.insert(p.as_str());
            }
        }
    }
    let mut dw_seen = BTreeSet::new();
    for p in &optim.muon_depthwise {
        if !dw_seen.insert(p.as_str()) {
            return Err(Error::Metadata(format!("folo.optim: duplicate `{p}` in `muon_depthwise`")));
        }
        if !muon_params.contains(p.as_str()) {
            return Err(Error::Metadata(format!(
                "folo.optim: `muon_depthwise` names `{p}`, which is not in a Muon group"
            )));
        }
        let info = file.info(p.as_str()).expect("muon param is a model tensor");
        if !crate::optim::is_depthwise_conv_shape(&info.shape) {
            return Err(Error::Metadata(format!(
                "folo.optim: `muon_depthwise` names `{p}` with shape {:?}, expected depthwise conv weight `[C > 1, 1, kH, kW]`",
                info.shape
            )));
        }
    }
    for (p, &ps) in &optim.param_steps {
        if !seen.contains(p.as_str()) {
            return Err(Error::Metadata(format!(
                "folo.optim: `param_steps` names `{p}`, which is not in any optimizer group"
            )));
        }
        if ps > optim.step {
            return Err(Error::Metadata(format!(
                "folo.optim: `param_steps[{p}]` ({ps}) exceeds optimizer step ({})",
                optim.step
            )));
        }
    }

    let ema_names: BTreeSet<&str> = file
        .names(Namespace::Ema)
        .into_iter()
        .filter(|n| !n.starts_with("ema.__"))
        .map(|n| &n["ema.".len()..])
        .collect();
    let has_ema = !ema_names.is_empty();
    let ema =
        meta.extra.contains_key("folo.ema").then(|| json::<EmaContract>(meta, "folo.ema")).transpose()?;
    if let Some(p) = ema.as_ref().and_then(|e| e.power) {
        p.validate().map_err(|e| Error::Metadata(format!("folo.ema: {e}")))?;
    }
    let ema_updates = match (has_ema, ema.is_some()) {
        (false, false) => {
            if file.info("ema.__updates").is_some() {
                return Err(Error::Metadata("`ema.__updates` without EMA tensors".into()));
            }
            None
        }
        (true, true) => {
            let missing: Vec<&&str> = model.difference(&ema_names).collect();
            let extra: Vec<&&str> = ema_names.difference(&model).collect();
            if !missing.is_empty() || !extra.is_empty() {
                return Err(Error::Metadata(format!(
                    "the ema. set must cover the model exactly; missing {missing:?}, unexpected {extra:?}"
                )));
            }
            for n in &ema_names {
                let (a, b) = (file.info(n).expect("model"), file.info(&format!("ema.{n}")).expect("ema"));
                if a.shape != b.shape {
                    return Err(Error::Metadata(format!(
                        "`ema.{n}` is {:?}, the model tensor is {:?}",
                        b.shape, a.shape
                    )));
                }
            }
            Some(file.counter("ema.__updates")?)
        }
        (true, false) => return Err(Error::Metadata("EMA tensors without `folo.ema`".into())),
        (false, true) => return Err(Error::Metadata("`folo.ema` without EMA tensors".into())),
    };
    Ok(CheckpointMeta { train, optim, sched, ema, ema_updates, rng, epoch, best })
}

/// Everything a checkpoint stores (docs/FORMAT.md: `folo.kind = "checkpoint"`).
pub struct CheckpointParts<'a> {
    /// The live training model.
    pub model: &'a dyn Module,
    /// The EMA, if enabled.
    pub ema: Option<&'a dyn EmaState>,
    /// The optimizer (its step count must equal the schedule's).
    pub optimizer: &'a dyn Optimizer,
    /// The schedule.
    pub schedule: &'a LrSchedule,
    /// RNG streams.
    pub rng: &'a RngState,
    /// Completed epochs.
    pub epoch: u64,
    /// Early-stopping state.
    pub best: BestState,
    /// The resolved TrainConfig (docs/API.md), stored as `folo.train`.
    pub train_config: &'a serde_json::Value,
}

fn counter(v: u64) -> Result<Tensor> {
    let v = i64::try_from(v).map_err(|_| Error::Invalid(format!("counter {v} does not fit an I64")))?;
    Ok(Tensor::from_vec(vec![v], folo_core::Shape::scalar())?)
}

/// A metadata map and a named tensor list: what a `.folo` file holds.
pub type FileContents = (BTreeMap<String, String>, Vec<(String, Tensor)>);

/// The tensors and metadata map of a checkpoint (what [`save_checkpoint`] writes).
pub fn checkpoint_contents(meta: &Metadata, p: &CheckpointParts<'_>) -> Result<FileContents> {
    if meta.fused {
        return Err(Error::Invalid("a checkpoint cannot be fused".into()));
    }
    let step = p.optimizer.step_count();
    if step != p.schedule.step() {
        return Err(Error::Invalid(format!(
            "optimizer step {step} and schedule step {} disagree; they must advance together",
            p.schedule.step()
        )));
    }
    let model = p.model.named_tensors();
    let params: BTreeSet<&str> =
        model.iter().filter(|t| t.kind == crate::TensorKind::Parameter).map(|t| t.name.as_str()).collect();
    let grouped: BTreeSet<&str> =
        p.optimizer.groups().iter().flat_map(|g| &g.params).map(|(n, _)| n.as_str()).collect();
    if params != grouped {
        let missing: Vec<_> = params.difference(&grouped).collect();
        let extra: Vec<_> = grouped.difference(&params).collect();
        return Err(Error::Invalid(format!(
            "every model parameter must be in exactly one optimizer group; ungrouped {missing:?}, not in the model {extra:?}"
        )));
    }

    let mut m = meta.clone();
    m.kind = FileKind::Checkpoint;
    let mut map = m.to_map();
    map.insert("folo.train".into(), enc(p.train_config));
    map.insert("folo.optim".into(), enc(&p.optimizer.contract()));
    map.insert("folo.sched".into(), enc(p.schedule.contract()));
    map.insert("folo.rng".into(), p.rng.to_json());
    map.insert("folo.epoch".into(), p.epoch.to_string());
    map.insert("folo.best".into(), enc(&p.best));
    map.remove("folo.ema");

    let mut tensors: Vec<(String, Tensor)> = model.into_iter().map(|t| (t.name, t.tensor)).collect();
    if let Some(ema) = p.ema {
        map.insert("folo.ema".into(), enc(&ema.contract()));
        tensors.extend(ema.ema_tensors().into_iter().map(|(n, t)| (format!("ema.{n}"), t)));
        tensors.push(("ema.__updates".into(), counter(ema.updates())?));
    }
    tensors.extend(p.optimizer.state_tensors()?.into_iter().map(|(n, t)| (format!("optim.{n}"), t)));
    tensors.push(("optim.__step".into(), counter(step)?));
    tensors.push(("sched.__step".into(), counter(p.schedule.step())?));
    Ok((map, tensors))
}

/// Write a checkpoint to `path` (atomically: a fresh temporary file next to it, then rename). `meta` supplies the model metadata;
/// its kind is set to `checkpoint` and the checkpoint contracts are filled from `parts`.
pub fn save_checkpoint(path: impl AsRef<Path>, meta: &Metadata, parts: &CheckpointParts<'_>) -> Result<()> {
    let (map, tensors) = checkpoint_contents(meta, parts)?;
    safetensors::write_file(path.as_ref(), Some(&map), &tensors)
}

fn enc<T: Serialize + ?Sized>(v: &T) -> String {
    serde_json::to_string(v).expect("contract serialises")
}

/// What [`ModelFile::resume`] restored besides the tensors.
#[derive(Debug, Clone)]
pub struct Resumed {
    /// The schedule at the saved step.
    pub schedule: LrSchedule,
    /// RNG streams.
    pub rng: RngState,
    /// Completed epochs.
    pub epoch: u64,
    /// Early-stopping state.
    pub best: BestState,
    /// `folo.train`.
    pub train_config: serde_json::Value,
    /// The (strict) model load.
    pub report: LoadReport,
}

impl ModelFile {
    /// Resume training: load the model namespace strictly into `model`, check and restore the optimizer
    /// (type, groups and parameter sets must match; state must be complete unless the step is 0), rebuild the
    /// schedule, and restore the EMA. `ema` must be given exactly when the checkpoint has one. `classes` is
    /// the run's class list, if it has one (strict loading requires it to match).
    pub fn resume(
        &self,
        model: &dyn Module,
        optimizer: &mut dyn Optimizer,
        ema: Option<&mut dyn EmaState>,
        classes: Option<Vec<String>>,
    ) -> Result<Resumed> {
        let c = self
            .checkpoint()
            .ok_or_else(|| Error::Load("not a checkpoint (folo.kind is `weights`)".into()))?;
        match (&ema, c.ema) {
            (Some(_), None) => return Err(Error::Load("the run has EMA but the checkpoint has none".into())),
            (None, Some(_)) => return Err(Error::Load("the checkpoint has EMA but the run has none".into())),
            _ => {}
        }
        let mut opts = LoadOptions::strict(Purpose::Training);
        opts.classes = classes;
        let report = self.load(model, &opts)?;

        optimizer.apply_contract(&c.optim)?;
        let state_names = expected_state(optimizer);
        let mut state = HashMap::new();
        let mut unexpected = Vec::new();
        for n in self.names(Namespace::Optim) {
            let key = &n["optim.".len()..];
            if key.starts_with("__") {
                continue;
            }
            if state_names.contains(key) {
                state.insert(key.to_string(), self.tensor(n)?);
            } else {
                unexpected.push(n);
            }
        }
        if !unexpected.is_empty() {
            return Err(Error::Load(format!("unexpected optimizer state: {}", unexpected.join(", "))));
        }
        optimizer.load_state(c.optim.step, &state)?;

        if let (Some(ema), Some(contract)) = (ema, c.ema) {
            let mut t = HashMap::new();
            for (model_name, file_name) in self.source_names(super::file::Source::Ema) {
                t.insert(model_name, self.tensor(file_name)?);
            }
            ema.restore(&t, c.ema_updates.expect("validated with folo.ema"), contract)?;
        }
        Ok(Resumed {
            schedule: LrSchedule::from_contract(c.sched.clone())?,
            rng: c.rng.clone(),
            epoch: c.epoch,
            best: c.best,
            train_config: c.train.clone(),
            report,
        })
    }
}

/// `<param>.<state>` keys the optimizer type can use.
fn expected_state(opt: &dyn Optimizer) -> BTreeSet<String> {
    let states: &[&str] = match opt.kind() {
        OptimKind::Sgd => &["momentum_buffer"],
        OptimKind::AdamW => &["exp_avg", "exp_avg_sq"],
        OptimKind::MuSgd => &["momentum_buffer", "muon_buffer"],
    };
    opt.groups()
        .iter()
        .flat_map(|g| &g.params)
        .flat_map(|(n, _)| states.iter().map(move |s| format!("{n}.{s}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_json() {
        let mut r = Rng::seed(7);
        r.next_u64();
        let s = RngState { seed: 7, streams: [("data".to_string(), r.clone())].into() };
        let j = s.to_json();
        assert!(j.contains("\"algo\":\"xoshiro256pp\""), "{j}");
        let back = RngState::from_json(&j).unwrap();
        assert_eq!(back, s);
        // Little-endian words: the first hex byte is the low byte of word 0.
        let hex = &serde_json::from_str::<serde_json::Value>(&j).unwrap()["streams"]["data"];
        assert_eq!(&hex.as_str().unwrap()[..2], format!("{:02x}", r.state()[0] & 0xff));
        assert!(RngState::from_json(&j.replace("xoshiro256pp", "pcg")).is_err());
        assert!(
            RngState::from_json(&format!(
                r#"{{"algo":"xoshiro256pp","seed":0,"streams":{{"a":"{}"}}}}"#,
                "0".repeat(64)
            ))
            .is_err()
        );
    }
}
