//! The mock's datasets: procedural ones (images rendered on request, labels and issues derived from a seed) and
//! ones registered from disk.

use folo_api::{
    BBox, Dataset, DatasetFormat, DatasetIssue, ImageInfo, ImagePage, IssuePage, IssueSeverity, Split, Splits,
};

use crate::error::{Error, Result};
use crate::index::{Index, read_image};
use crate::stats::StatsBuilder;
use crate::store::{ImageData, ImageQuery, IssueQuery};
use crate::util::mix64;

use super::scene::{Scene, Theme};

/// A procedural dataset.
pub(super) struct ProcDataset {
    pub(super) meta: Dataset,
    theme: Theme,
    w: u32,
    h: u32,
    seed: u64,
    labels: Vec<Vec<BBox>>,
    stats: folo_api::DatasetStats,
}

impl ProcDataset {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        id: &str,
        name: &str,
        theme: Theme,
        classes: &[&str],
        w: u32,
        h: u32,
        splits: Splits,
        seed: u64,
        created: &str,
        broken: u64,
    ) -> Self {
        let classes: Vec<String> = classes.iter().map(|s| s.to_string()).collect();
        ProcDataset::build(id.into(), name.into(), theme, classes, w, h, splits, seed, created.into(), broken)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        id: String,
        name: String,
        theme: Theme,
        classes: Vec<String>,
        w: u32,
        h: u32,
        splits: Splits,
        seed: u64,
        created: String,
        broken: u64,
    ) -> Self {
        let n = splits.train + splits.val + splits.test;
        let labels: Vec<Vec<BBox>> =
            (0..n).map(|i| Scene::generate(theme, &classes, w, h, seed, i).labels()).collect();
        let mut sb = StatsBuilder::new(classes.len());
        let (wf, hf) = (w as f32, h as f32);
        for l in &labels {
            sb.add_image(l.iter().map(|b| (b.class, (b.x2 - b.x1) / wf, (b.y2 - b.y1) / hf)));
        }
        sb.add_broken(broken);
        let meta = Dataset {
            id,
            path: format!("mock://{name}"),
            name,
            format: DatasetFormat::Yolo,
            classes,
            created,
            splits,
            stats: None,
        };
        ProcDataset { meta, theme, w, h, seed, labels, stats: sb.finish() }
    }

    fn split_of(&self, i: u64) -> Split {
        let s = self.meta.splits;
        if i < s.train {
            Split::Train
        } else if i < s.train + s.val {
            Split::Val
        } else {
            Split::Test
        }
    }

    fn index(&self, image_id: &str) -> Result<u64> {
        image_id
            .parse::<u64>()
            .ok()
            .filter(|&i| (i as usize) < self.labels.len() && format!("{i:06}") == image_id)
            .ok_or_else(|| Error::not_found("image", image_id))
    }

    /// Deterministic `folo dataset check` findings consistent with the stats: one `broken_label` per broken label
    /// file (those images are not listed, as the loader skips them), an `empty_image` per image without boxes, and
    /// a sprinkle of warnings (boxes clipped at the frame, a duplicated image) chosen by the seed.
    fn issues(&self) -> Vec<DatasetIssue> {
        let issue = |kind: &str, severity, file: String, line: Option<u32>, message: String| {
            // The fixes `folo dataset check --fix-dry-run` suggests for these kinds.
            let fix = match (kind, line) {
                ("empty_image", _) => None,
                ("clipped_box", Some(l)) => Some(format!("clip the box on line {l} to the image")),
                (_, Some(l)) => Some(format!("delete line {l}")),
                _ => Some(match message.strip_prefix("same bytes as ") {
                    Some(first) => format!("remove this copy (and its label file); keep {first}"),
                    None => "remove the image and its label file".into(),
                }),
            };
            DatasetIssue { kind: kind.into(), severity, file, line, message, fix }
        };
        let mut out = Vec::new();
        let n = self.labels.len() as u64;
        for k in 0..self.stats.broken_labels {
            let message = match k % 3 {
                0 => "expected 5 values (class cx cy w h) or a polygon, got 4 (the loader skips this image)"
                    .into(),
                1 => format!(
                    "class id {} out of range for {} classes (the loader skips this image)",
                    self.meta.classes.len() + k as usize,
                    self.meta.classes.len()
                ),
                _ => "coordinates out of [0, 1]: labels must be normalised (the loader skips this image)"
                    .into(),
            };
            let kind = ["broken_label", "class_overflow", "out_of_range_box"][(k % 3) as usize];
            let file = format!("labels/train/{:06}.txt", n + k);
            out.push(issue(kind, IssueSeverity::Error, file, Some(1 + (k % 4) as u32), message));
        }
        for i in 0..n {
            let split = self.split_of(i).as_str();
            let labels = &self.labels[i as usize];
            if labels.is_empty() {
                let file = format!("images/{split}/{i:06}.jpg");
                out.push(issue(
                    "empty_image",
                    IssueSeverity::Info,
                    file,
                    None,
                    "no label file (background image)".into(),
                ));
                continue;
            }
            let h = mix64(self.seed ^ 0x155e ^ i);
            if h.is_multiple_of(97) {
                let line = 1 + (h >> 8) % labels.len() as u64;
                let file = format!("labels/{split}/{i:06}.txt");
                let message = "box extends slightly past the image edge".into();
                out.push(issue("clipped_box", IssueSeverity::Warning, file, Some(line as u32), message));
            }
            if h % 389 == 1 && i > 0 {
                let j = (h >> 16) % i;
                let file = format!("images/{split}/{i:06}.jpg");
                let message = format!("same bytes as images/{}/{j:06}.jpg", self.split_of(j).as_str());
                out.push(issue("duplicate_image", IssueSeverity::Warning, file, None, message));
            }
        }
        out
    }

    fn info(&self, i: u64) -> ImageInfo {
        let split = self.split_of(i);
        ImageInfo {
            id: format!("{i:06}"),
            file: format!("images/{}/{i:06}.jpg", split.as_str()),
            split,
            width: self.w,
            height: self.h,
            boxes: self.labels[i as usize].clone(),
        }
    }
}

pub(super) enum DatasetEntry {
    Proc(ProcDataset),
    Disk(Dataset, Box<Index>),
}

impl DatasetEntry {
    pub(super) fn meta(&self) -> &Dataset {
        match self {
            DatasetEntry::Proc(d) => &d.meta,
            DatasetEntry::Disk(meta, _) => meta,
        }
    }

    pub(super) fn with_stats(&self) -> Dataset {
        let (meta, stats) = match self {
            DatasetEntry::Proc(d) => (&d.meta, &d.stats),
            DatasetEntry::Disk(meta, index) => (meta, &index.stats),
        };
        Dataset { stats: Some(stats.clone()), ..meta.clone() }
    }

    pub(super) fn list_images(&self, q: &ImageQuery) -> ImagePage {
        match self {
            DatasetEntry::Disk(_, index) => index.list_images(q),
            DatasetEntry::Proc(d) => {
                let matches = |i: u64| {
                    q.split.is_none_or(|s| s == d.split_of(i))
                        && q.class.is_none_or(|c| d.labels[i as usize].iter().any(|b| b.class == c))
                };
                let n = d.labels.len() as u64;
                let total = (0..n).filter(|&i| matches(i)).count() as u64;
                let images = (0..n).filter(|&i| matches(i)).skip(q.offset as usize).take(q.limit as usize);
                ImagePage { total, images: images.map(|i| d.info(i)).collect() }
            }
        }
    }

    pub(super) fn image(&self, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        match self {
            DatasetEntry::Disk(_, index) => read_image(&index.image_path(image_id)?, thumb, None),
            DatasetEntry::Proc(d) => {
                let i = d.index(image_id)?;
                let scene = Scene::generate(d.theme, &d.meta.classes, d.w, d.h, d.seed, i);
                let quality = if thumb.is_some() { 80 } else { 90 };
                let bytes = scene.render(thumb).to_jpeg(quality);
                Ok(ImageData { content_type: "image/jpeg", bytes, immutable: true })
            }
        }
    }

    /// `(width, height, truth)` of an image.
    pub(super) fn truth(&self, image_id: &str) -> Result<(u32, u32, Vec<BBox>)> {
        match self {
            DatasetEntry::Disk(_, index) => index.truth(image_id),
            DatasetEntry::Proc(d) => {
                let i = d.index(image_id)?;
                Ok((d.w, d.h, d.labels[i as usize].clone()))
            }
        }
    }

    pub(super) fn list_issues(&self, q: &IssueQuery) -> IssuePage {
        match self {
            DatasetEntry::Disk(_, index) => index.list_issues(q),
            DatasetEntry::Proc(d) => q.page(d.issues().iter()),
        }
    }

    /// Ids of up to `n` validation images spread over the split (falls back to train when there is no val split).
    pub(super) fn sample_ids(&self, n: usize) -> Vec<String> {
        let split = if self.meta().splits.val > 0 { Split::Val } else { Split::Train };
        let all =
            self.list_images(&ImageQuery { split: Some(split), offset: 0, limit: u64::MAX, class: None });
        let step = (all.images.len() / n.max(1)).max(1);
        all.images.into_iter().step_by(step).take(n).map(|im| im.id).collect()
    }
}
