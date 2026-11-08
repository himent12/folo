//! `RealStore`: datasets registered from disk (persisted in `<data_dir>/datasets.json`). Training, inference,
//! export and synthetic datasets return `501 not_implemented` until the engine (`folo-vision`) is wired in.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use folo_api::{
    Dataset, DatasetFormat, ExportFormat, ExportResult, GpuInfo, ImagePage, LogLine, Model, ModelEval,
    PredictResponse, RegisterDataset, Run, RunMetrics, RunSamples, SyntheticDataset, TrainConfig,
};
use serde::{Deserialize, Serialize};

use crate::disk::DiskDataset;
use crate::error::{Error, Result};
use crate::store::{ImageData, ImageQuery, PredictInput, PredictSource, Store, Subscription};
use crate::util::{fnv1a, rfc3339, unique_id, unix_now};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Registration {
    id: String,
    name: String,
    path: String,
    format: DatasetFormat,
    created: String,
}

/// The real store. See the module docs.
pub struct RealStore {
    data_dir: PathBuf,
    datasets: RwLock<Vec<Arc<DiskDataset>>>,
}

impl RealStore {
    /// Opens the store rooted at `data_dir`, re-scanning previously registered datasets. Datasets whose folder is
    /// gone are kept out of the list (and logged) but stay in the registry file.
    pub fn open(data_dir: PathBuf) -> RealStore {
        let mut datasets = Vec::new();
        let regs = Self::read_registry(&data_dir).unwrap_or_else(|e| {
            tracing::error!("{e}; registering and unregistering datasets is disabled until it is fixed");
            Vec::new()
        });
        for r in regs {
            match DiskDataset::scan(r.id.clone(), Some(r.name), &r.path, r.format, r.created) {
                Ok(d) => datasets.push(Arc::new(d)),
                Err(e) => tracing::warn!(id = %r.id, path = %r.path, "skipping registered dataset: {e}"),
            }
        }
        RealStore { data_dir, datasets: RwLock::new(datasets) }
    }

    fn registry_path(dir: &std::path::Path) -> PathBuf {
        dir.join("datasets.json")
    }

    /// The registry. A missing file is an empty registry; an unreadable or malformed one is an error, so it is
    /// never overwritten (and its registrations lost) by the next write.
    fn read_registry(dir: &std::path::Path) -> Result<Vec<Registration>> {
        let path = Self::registry_path(dir);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::internal(format!("reading {}: {e}", path.display()))),
        };
        serde_json::from_str(&text).map_err(|e| {
            Error::internal(format!("{} is malformed: {e}", path.display()))
                .hint("fix the JSON by hand or move the file away; FOLO will not overwrite it")
        })
    }

    fn write_registry(&self, regs: &[Registration]) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)
            .and_then(|_| {
                let tmp = Self::registry_path(&self.data_dir).with_extension("json.tmp");
                std::fs::write(&tmp, serde_json::to_vec_pretty(regs).expect("registrations serialize"))?;
                std::fs::rename(&tmp, Self::registry_path(&self.data_dir))
            })
            .map_err(|e| {
                Error::internal(format!("writing {}: {e}", Self::registry_path(&self.data_dir).display()))
            })
    }

    fn dataset(&self, id: &str) -> Result<Arc<DiskDataset>> {
        self.datasets
            .read()
            .unwrap()
            .iter()
            .find(|d| d.meta.id == id)
            .cloned()
            .ok_or_else(|| Error::not_found("dataset", id))
    }
}

impl Store for RealStore {
    fn is_mock(&self) -> bool {
        false
    }

    fn gpus(&self) -> Vec<GpuInfo> {
        // TODO(gpu): enumerate adapters through folo-gpu once it exists.
        Vec::new()
    }

    fn list_datasets(&self) -> Result<Vec<Dataset>> {
        Ok(self.datasets.read().unwrap().iter().map(|d| d.meta.clone()).collect())
    }

    fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset> {
        let mut ds = DiskDataset::scan(String::new(), req.name, &req.path, req.format, rfc3339(unix_now()))?;
        let mut list = self.datasets.write().unwrap();
        let mut regs = Self::read_registry(&self.data_dir)?;
        // Unique against loaded datasets and registrations whose folder is currently missing.
        ds.meta.id = unique_id("ds", fnv1a(req.path.as_bytes()) ^ unix_now().to_bits(), |id| {
            list.iter().any(|d| d.meta.id == id) || regs.iter().any(|r| r.id == id)
        });
        let out = Dataset { stats: Some(ds.stats.clone()), ..ds.meta.clone() };
        regs.push(Registration {
            id: out.id.clone(),
            name: out.name.clone(),
            path: out.path.clone(),
            format: out.format,
            created: out.created.clone(),
        });
        self.write_registry(&regs)?;
        list.push(Arc::new(ds));
        Ok(out)
    }

    fn create_synthetic(&self, _req: SyntheticDataset) -> Result<Dataset> {
        Err(Error::not_implemented("generating synthetic datasets"))
    }

    fn get_dataset(&self, id: &str) -> Result<Dataset> {
        let d = self.dataset(id)?;
        Ok(Dataset { stats: Some(d.stats.clone()), ..d.meta.clone() })
    }

    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage> {
        Ok(self.dataset(id)?.list_images(q))
    }

    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        self.dataset(id)?.image(image_id, thumb)
    }

    fn delete_dataset(&self, id: &str) -> Result<()> {
        let mut list = self.datasets.write().unwrap();
        let mut regs = Self::read_registry(&self.data_dir)?;
        let found = list.iter().any(|d| d.meta.id == id) || regs.iter().any(|r| r.id == id);
        if !found {
            return Err(Error::not_found("dataset", id));
        }
        regs.retain(|r| r.id != id);
        self.write_registry(&regs)?;
        list.retain(|d| d.meta.id != id);
        Ok(())
    }

    fn list_models(&self) -> Result<Vec<Model>> {
        // TODO(engine): list `.folo` files under `<data_dir>/models` and run outputs once the format reader exists.
        Ok(Vec::new())
    }

    fn get_model(&self, id: &str) -> Result<Model> {
        Err(Error::not_found("model", id))
    }

    fn model_eval(&self, id: &str) -> Result<ModelEval> {
        Err(Error::not_found("model", id))
    }

    fn export_model(&self, id: &str, _format: ExportFormat) -> Result<ExportResult> {
        Err(Error::not_found("model", id))
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        Ok(Vec::new())
    }

    fn create_run(&self, config: TrainConfig) -> Result<Run> {
        self.dataset(&config.dataset_id)?;
        Err(Error::not_implemented("training"))
    }

    fn get_run(&self, id: &str) -> Result<Run> {
        Err(Error::not_found("run", id))
    }

    fn stop_run(&self, id: &str) -> Result<Run> {
        Err(Error::not_found("run", id))
    }

    fn delete_run(&self, id: &str) -> Result<()> {
        Err(Error::not_found("run", id))
    }

    fn run_metrics(&self, id: &str, _every: u64) -> Result<RunMetrics> {
        Err(Error::not_found("run", id))
    }

    fn run_samples(&self, id: &str, _epoch: Option<u32>) -> Result<RunSamples> {
        Err(Error::not_found("run", id))
    }

    fn run_logs(&self, id: &str, _limit: usize) -> Result<Vec<LogLine>> {
        Err(Error::not_found("run", id))
    }

    fn subscribe(&self, id: &str) -> Result<Subscription> {
        Err(Error::not_found("run", id))
    }

    fn predict(&self, input: PredictInput) -> Result<PredictResponse> {
        if let PredictSource::Dataset { dataset_id, .. } = &input.source {
            self.dataset(dataset_id)?;
        }
        Err(Error::not_implemented("inference"))
    }
}
