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

/// Long side of `folo://datasets/{id}/images/{image_id}` without `?size` (a JPEG thumbnail, tens of KB instead of
/// the original's hundreds).
pub const DEFAULT_IMAGE_SIZE: u32 = 512;

/// `?size` range of an image resource (as `GET /api/datasets/{id}/images/{image_id}?thumb=`).
pub const IMAGE_SIZES: std::ops::RangeInclusive<u32> = 16..=2048;

/// Cell size of `folo://runs/{id}/samples/{epoch}` without `?tile` (a 4-column grid: 640 px wide).
pub const DEFAULT_TILE: u32 = 160;

/// `?tile` range of a sample grid.
pub const TILES: std::ops::RangeInclusive<u32> = 64..=512;

/// The size an image resource is read at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageSize {
    /// The original file (`?size=original`).
    Original,
    /// A JPEG whose long side is at most this (`?size=<px>`, clamped to [`IMAGE_SIZES`]; never upscaled).
    Max(u32),
}

impl Default for ImageSize {
    fn default() -> Self {
        ImageSize::Max(DEFAULT_IMAGE_SIZE)
    }
}

impl ImageSize {
    /// The `thumb` argument of [`Store::image`].
    pub fn thumb(self) -> Option<u32> {
        match self {
            ImageSize::Original => None,
            ImageSize::Max(px) => Some(px),
        }
    }
}

/// A parsed `folo://` URI. Ids are percent-decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FoloUri {
    /// `folo://datasets/{id}`
    Dataset(String),
    /// `folo://datasets/{id}/images/{image_id}{?size}`
    Image(String, String, ImageSize),
    /// `folo://runs/{id}`
    Run(String),
    /// `folo://runs/{id}/metrics`
    Metrics(String),
    /// `folo://runs/{id}/epochs`
    Epochs(String),
    /// `folo://runs/{id}/samples/{epoch}{?tile}`: the epoch and the grid's cell size in pixels.
    Samples(String, u32, u32),
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
    /// Parses a URI; `None` for anything that is not one of the forms above. Images take `?size=<px>|original`
    /// and sample grids `?tile=<px>`; any other query is `None`.
    pub fn parse(uri: &str) -> Option<FoloUri> {
        let rest = uri.strip_prefix("folo://")?;
        let (path, query) = match rest.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (rest, None),
        };
        // The value of the one query parameter `key` a form accepts: `Some(None)` without a query, `None` for
        // any other query.
        let param = |key: &str| -> Option<Option<&str>> {
            match query {
                None => Some(None),
                Some(q) => match q.split_once('=') {
                    Some((k, v)) if k == key && !v.is_empty() && !v.contains('&') => Some(Some(v)),
                    _ => None,
                },
            }
        };
        let parts: Vec<&str> = path.split('/').collect();
        Some(match parts.as_slice() {
            ["datasets", id, "images", image] => {
                let size = match param("size")? {
                    None => ImageSize::default(),
                    Some("original") => ImageSize::Original,
                    Some(px) => ImageSize::Max(
                        px.parse::<u32>().ok()?.clamp(*IMAGE_SIZES.start(), *IMAGE_SIZES.end()),
                    ),
                };
                FoloUri::Image(decode(id)?, decode(image)?, size)
            }
            ["runs", id, "samples", epoch] => {
                let tile = match param("tile")? {
                    None => DEFAULT_TILE,
                    Some(px) => px.parse::<u32>().ok()?.clamp(*TILES.start(), *TILES.end()),
                };
                FoloUri::Samples(decode(id)?, epoch.parse().ok()?, tile)
            }
            _ if query.is_some() => return None,
            ["datasets", id] => FoloUri::Dataset(decode(id)?),
            ["runs", id] => FoloUri::Run(decode(id)?),
            ["runs", id, "metrics"] => FoloUri::Metrics(decode(id)?),
            ["runs", id, "epochs"] => FoloUri::Epochs(decode(id)?),
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
            FoloUri::Image(id, image, size) => {
                write!(f, "folo://datasets/{}/images/{}", encode(id), encode(image))?;
                match size {
                    s if *s == ImageSize::default() => Ok(()),
                    ImageSize::Original => write!(f, "?size=original"),
                    ImageSize::Max(px) => write!(f, "?size={px}"),
                }
            }
            FoloUri::Run(id) => write!(f, "folo://runs/{}", encode(id)),
            FoloUri::Metrics(id) => write!(f, "folo://runs/{}/metrics", encode(id)),
            FoloUri::Epochs(id) => write!(f, "folo://runs/{}/epochs", encode(id)),
            FoloUri::Samples(id, e, tile) => {
                write!(f, "folo://runs/{}/samples/{e}", encode(id))?;
                if *tile != DEFAULT_TILE { write!(f, "?tile={tile}") } else { Ok(()) }
            }
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
            "folo://datasets/{id}/images/{image_id}{?size}",
            "dataset image",
            "The image as a JPEG whose long side is at most 512 px (`?size=16..2048`), or the original file \
             (`?size=original`, JPEG / PNG), for looking at it.",
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
            "folo://runs/{id}/samples/{epoch}{?tile}",
            "validation samples",
            "Validation samples of an epoch as an annotated JPEG grid (truth white, predictions in class colours), \
             4 cells of 160 px per row (`?tile=64..512`).",
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
            FoloUri::Image("ds_1".into(), "train/0 1".into(), ImageSize::default()),
            FoloUri::Image("ds_1".into(), "a".into(), ImageSize::Original),
            FoloUri::Image("ds_1".into(), "a".into(), ImageSize::Max(64)),
            FoloUri::Run("run_1".into()),
            FoloUri::Metrics("run_1".into()),
            FoloUri::Epochs("run_1".into()),
            FoloUri::Samples("run_1".into(), 3, DEFAULT_TILE),
            FoloUri::Samples("run_1".into(), 3, 256),
            FoloUri::Predictions("pb_1".into()),
            FoloUri::Doc("API".into()),
        ] {
            assert_eq!(FoloUri::parse(&u.to_string()), Some(u));
        }
        assert_eq!(FoloUri::parse("folo://runs/x/samples/nope"), None);
        let img = |q: &str| FoloUri::parse(&format!("folo://datasets/d/images/i{q}"));
        assert_eq!(img(""), Some(FoloUri::Image("d".into(), "i".into(), ImageSize::Max(512))));
        assert_eq!(img("?size=9999"), Some(FoloUri::Image("d".into(), "i".into(), ImageSize::Max(2048))));
        assert_eq!(img("?size=original"), Some(FoloUri::Image("d".into(), "i".into(), ImageSize::Original)));
        for bad in ["?size=", "?size=big", "?thumb=64", "?size=64&x=1", "?"] {
            assert_eq!(img(bad), None, "{bad}");
        }
        assert_eq!(
            FoloUri::parse("folo://runs/r/samples/2?tile=8"),
            Some(FoloUri::Samples("r".into(), 2, 64))
        );
        assert_eq!(FoloUri::parse("folo://runs/r?tile=8"), None);
        assert_eq!(FoloUri::parse("folo://nothing"), None);
        assert_eq!(FoloUri::parse("file:///etc/passwd"), None);
        assert!(doc("MCP").unwrap().contains("folo mcp"));
    }
}
