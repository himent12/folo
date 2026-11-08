//! `folo://` resources (docs/MCP.md "Resources"): URIs, the listing, templates and JSON snapshots.

use std::fmt;

use folo_server::Store;
use rmcp::model::{Resource, ResourceTemplate};
use serde_json::{Value, json};

use crate::errors::Error;
use crate::proxy::encode;

/// `resources/list` page size.
pub const PAGE: usize = 100;

/// The repository docs served as `folo://docs/{name}`.
const DOCS: &[(&str, &str)] = &[
    ("ARCHITECTURE", include_str!("../../../docs/ARCHITECTURE.md")),
    ("API", include_str!("../../../docs/API.md")),
    ("CLI", include_str!("../../../docs/CLI.md")),
    ("MODEL", include_str!("../../../docs/MODEL.md")),
    ("FORMAT", include_str!("../../../docs/FORMAT.md")),
    ("METRICS", include_str!("../../../docs/METRICS.md")),
    ("MCP", include_str!("../../../docs/MCP.md")),
    ("AGENT_COOKBOOK", include_str!("../../../docs/AGENT_COOKBOOK.md")),
];

/// A parsed `folo://` URI. Ids are percent-decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoloUri {
    /// `folo://datasets/{id}`
    Dataset(String),
    /// `folo://datasets/{id}/images/{image_id}`
    Image(String, String),
    /// `folo://runs/{id}`
    Run(String),
    /// `folo://runs/{id}/metrics`
    Metrics(String),
    /// `folo://runs/{id}/epochs`
    Epochs(String),
    /// `folo://runs/{id}/samples/{epoch}`
    Samples(String, u32),
    /// `folo://predictions/{batch_id}`
    Predictions(String),
    /// `folo://docs/{name}`
    Doc(String),
}

fn decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = std::str::from_utf8(b.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    let s = String::from_utf8(out).ok()?;
    (!s.is_empty()).then_some(s)
}

impl FoloUri {
    /// Parses a URI; `None` for anything that is not one of the forms above.
    pub fn parse(uri: &str) -> Option<FoloUri> {
        let rest = uri.strip_prefix("folo://")?;
        let parts: Vec<&str> = rest.split('/').collect();
        Some(match parts.as_slice() {
            ["datasets", id] => FoloUri::Dataset(decode(id)?),
            ["datasets", id, "images", image] => FoloUri::Image(decode(id)?, decode(image)?),
            ["runs", id] => FoloUri::Run(decode(id)?),
            ["runs", id, "metrics"] => FoloUri::Metrics(decode(id)?),
            ["runs", id, "epochs"] => FoloUri::Epochs(decode(id)?),
            ["runs", id, "samples", epoch] => FoloUri::Samples(decode(id)?, epoch.parse().ok()?),
            ["predictions", id] => FoloUri::Predictions(decode(id)?),
            ["docs", name] => FoloUri::Doc(decode(name)?),
            _ => return None,
        })
    }

    /// Run snapshots and metric (or epoch) histories can be subscribed to.
    pub fn subscribable(&self) -> bool {
        matches!(self, FoloUri::Run(_) | FoloUri::Metrics(_) | FoloUri::Epochs(_))
    }
}

impl fmt::Display for FoloUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FoloUri::Dataset(id) => write!(f, "folo://datasets/{}", encode(id)),
            FoloUri::Image(id, image) => write!(f, "folo://datasets/{}/images/{}", encode(id), encode(image)),
            FoloUri::Run(id) => write!(f, "folo://runs/{}", encode(id)),
            FoloUri::Metrics(id) => write!(f, "folo://runs/{}/metrics", encode(id)),
            FoloUri::Epochs(id) => write!(f, "folo://runs/{}/epochs", encode(id)),
            FoloUri::Samples(id, e) => write!(f, "folo://runs/{}/samples/{e}", encode(id)),
            FoloUri::Predictions(id) => write!(f, "folo://predictions/{}", encode(id)),
            FoloUri::Doc(name) => write!(f, "folo://docs/{}", encode(name)),
        }
    }
}

/// A repository doc by name (`API`, `MCP`, ...).
pub fn doc(name: &str) -> Option<&'static str> {
    DOCS.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// Every concrete resource: datasets, runs with their metrics, and docs. (Images, samples and predictions are
/// reached through the templates.)
/// `keep(dataset_id)` filters datasets (`--confine` leaves out those that read outside the data dir).
pub fn list(store: &dyn Store, keep: &dyn Fn(&str) -> bool) -> Result<Vec<Resource>, Error> {
    let mut out = Vec::new();
    for d in store.list_datasets()?.into_iter().filter(|d| keep(&d.id)) {
        out.push(
            Resource::new(FoloUri::Dataset(d.id.clone()).to_string(), format!("dataset {}", d.name))
                .with_description(format!("{} with stats; {} classes", d.id, d.classes.len()))
                .with_mime_type("application/json"),
        );
    }
    for r in store.list_runs()? {
        out.push(
            Resource::new(FoloUri::Run(r.id.clone()).to_string(), format!("run {}", r.name))
                .with_description(format!("{}: status and progress (subscribable)", r.id))
                .with_mime_type("application/json"),
        );
        out.push(
            Resource::new(FoloUri::Metrics(r.id.clone()).to_string(), format!("metrics of {}", r.name))
                .with_description(format!("{}: full step and epoch history (subscribable)", r.id))
                .with_mime_type("application/json"),
        );
        out.push(
            Resource::new(FoloUri::Epochs(r.id.clone()).to_string(), format!("epochs of {}", r.name))
                .with_description(format!("{}: epoch history only, no steps (subscribable)", r.id))
                .with_mime_type("application/json"),
        );
    }
    for (name, text) in DOCS {
        out.push(
            Resource::new(FoloUri::Doc(name.to_string()).to_string(), format!("docs/{name}.md"))
                .with_description(format!("the FOLO {name} documentation"))
                .with_mime_type("text/markdown")
                .with_size(text.len() as u64),
        );
    }
    Ok(out)
}

/// The resource templates.
pub fn templates() -> Vec<ResourceTemplate> {
    let t = |uri: &str, name: &str, description: &str, mime: &str| {
        ResourceTemplate::new(uri, name).with_description(description).with_mime_type(mime)
    };
    vec![
        t("folo://datasets/{id}", "dataset", "A dataset with statistics.", "application/json"),
        t(
            "folo://datasets/{id}/images/{image_id}",
            "dataset image",
            "The image file itself (JPEG / PNG blob), for looking at it.",
            "image/jpeg",
        ),
        t(
            "folo://runs/{id}",
            "run",
            "{ run, last_step, last_epoch, revision }; subscribable (at most one update per second).",
            "application/json",
        ),
        t(
            "folo://runs/{id}/metrics",
            "run metrics",
            "Full step and epoch history; subscribable.",
            "application/json",
        ),
        t(
            "folo://runs/{id}/epochs",
            "run epochs",
            "{ epochs, last_step }: the epoch history without per-step records (about 1 KB per epoch); subscribable.",
            "application/json",
        ),
        t(
            "folo://runs/{id}/samples/{epoch}",
            "validation samples",
            "Validation samples of an epoch as an annotated JPEG grid (truth white, predictions in class colours).",
            "image/jpeg",
        ),
        t(
            "folo://predictions/{batch_id}",
            "batch predictions",
            "Per-image boxes of a predict_batch call.",
            "application/json",
        ),
        t(
            "folo://docs/{name}",
            "docs",
            "ARCHITECTURE, API, CLI, MODEL, FORMAT, METRICS or MCP.",
            "text/markdown",
        ),
    ]
}

/// `{ run, last_step, last_epoch }` of `folo://runs/{id}` (the caller adds `revision`).
pub fn run_snapshot(store: &dyn Store, id: &str) -> Result<Value, Error> {
    let run = store.get_run(id)?;
    let m = store.run_metrics(id, u64::MAX)?;
    Ok(json!({ "run": run, "last_step": m.steps.last(), "last_epoch": m.epochs.last() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris_roundtrip() {
        for u in [
            FoloUri::Dataset("ds_1".into()),
            FoloUri::Image("ds_1".into(), "train/0 1".into()),
            FoloUri::Run("run_1".into()),
            FoloUri::Metrics("run_1".into()),
            FoloUri::Epochs("run_1".into()),
            FoloUri::Samples("run_1".into(), 3),
            FoloUri::Predictions("pb_1".into()),
            FoloUri::Doc("API".into()),
        ] {
            assert_eq!(FoloUri::parse(&u.to_string()), Some(u));
        }
        assert_eq!(FoloUri::parse("folo://runs/x/samples/nope"), None);
        assert_eq!(FoloUri::parse("folo://nothing"), None);
        assert_eq!(FoloUri::parse("file:///etc/passwd"), None);
        assert!(doc("MCP").unwrap().contains("folo mcp"));
    }
}
