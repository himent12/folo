//! The dataset tools: list, inspect (issues, images), register and generate synthetic datasets.

use std::collections::HashSet;

use folo_api::{Dataset, IssuePage, IssueSeverity, RegisterDataset, SyntheticDataset};
use folo_server::store::{ImageQuery, IssueQuery};

use super::Progress;
use super::arguments::check_name;
use super::results::{
    MAX_NAME_CHARS, MAX_RESULT_BYTES, Output, fit_page, json_len, out, page_of, page_window, short,
    summary_classes,
};
use crate::errors::{Error, invalid_arguments};
use crate::server::Inner;
use crate::types::*;

/// `get_dataset` lists at most this many issues.
const MAX_ISSUES: usize = 50;

/// `Store::list_issues` for `q`, also against a `folo serve` from before severity filters and counts (proxy
/// mode): it ignores `severity` and leaves `counts` out, so the page is rebuilt from all its issues.
pub(super) fn list_issues_compat(
    fetch: impl Fn(&IssueQuery) -> Result<IssuePage, Error>,
    q: &IssueQuery,
) -> Result<IssuePage, Error> {
    let counted = |p: &IssuePage| p.counts.error + p.counts.warning + p.counts.info > 0;
    let page = fetch(q)?;
    if counted(&page) {
        return Ok(page);
    }
    // No counts: a dataset without issues, or an older server. A filtered answer cannot tell them apart (an old
    // server honours `kind`), so ask for everything once.
    let filtered = q.kind.is_some() || q.severity.is_some();
    let probe =
        if filtered { fetch(&IssueQuery { limit: 1, ..IssueQuery::default() })? } else { page.clone() };
    if probe.total == 0 || counted(&probe) {
        return Ok(page);
    }
    let mut all = Vec::new();
    loop {
        let q = IssueQuery { offset: all.len() as u64, limit: 1000, ..IssueQuery::default() };
        let p = fetch(&q)?;
        let n = p.issues.len();
        all.extend(p.issues);
        if n == 0 || all.len() as u64 >= p.total {
            break;
        }
    }
    Ok(q.page(all.iter()))
}

/// An issue with its file, message and fix cut to 300 characters, so a page always makes progress.
fn short_issue(mut i: folo_api::DatasetIssue) -> folo_api::DatasetIssue {
    i.file = short(&i.file, 300);
    i.message = short(&i.message, 300);
    i.fix = i.fix.map(|f| short(&f, 300));
    i
}

impl Inner {
    pub(super) async fn list_datasets(&self, a: PageArgs) -> Result<Output, Error> {
        let c = self.confinement.clone();
        // With --confine, datasets that read outside the data dir are not listed.
        let all: Vec<Dataset> = self
            .blocking(move |s| {
                Ok(s.list_datasets()?.into_iter().filter(|d| c.dataset(s, &d.id).is_ok()).collect())
            })
            .await?;
        let items: Vec<DatasetSummary> = all
            .into_iter()
            .map(|d: Dataset| DatasetSummary {
                id: d.id,
                name: short(&d.name, MAX_NAME_CHARS),
                format: d.format,
                num_classes: d.classes.len(),
                classes: summary_classes(&d.classes),
                splits: d.splits,
            })
            .collect();
        let page = page_of(items, a.cursor.as_deref(), a.limit)?;
        let summary = format!("{} of {} datasets", page.items.len(), page.total);
        out(&page, summary)
    }

    pub(super) async fn get_dataset(&self, a: DatasetIdArgs) -> Result<Output, Error> {
        self.ensure_dataset(&a.dataset_id).await?;
        let id = a.dataset_id.clone();
        let (dataset, issues, counts) = self
            .blocking(move |s| {
                let dataset = s.get_dataset(&id)?;
                // Errors first, then warnings; infos (background images) are only counted.
                let mut issues = Vec::new();
                let mut counts = None;
                for severity in [IssueSeverity::Error, IssueSeverity::Warning] {
                    let limit = (MAX_ISSUES - issues.len()) as u64;
                    let q = IssueQuery { severity: Some(severity), limit, ..IssueQuery::default() };
                    let page = list_issues_compat(|q| s.list_issues(&id, q), &q)?;
                    issues.extend(
                        page.issues.into_iter().filter(|i| i.severity == severity).take(limit as usize),
                    );
                    counts.get_or_insert(page.counts);
                }
                Ok((dataset, issues, counts.unwrap_or_default()))
            })
            .await?;
        let mut detail = DatasetDetail {
            dataset,
            issues: issues.into_iter().map(short_issue).collect(),
            issues_total: counts.error + counts.warning + counts.info,
            issue_counts: counts,
        };
        while json_len(&detail) > MAX_RESULT_BYTES && detail.issues.pop().is_some() {}
        let s = &detail.dataset;
        let c = &detail.issue_counts;
        let summary = format!(
            "{} ({}): {} classes, {} train / {} val / {} test images; {} errors, {} warnings, {} infos",
            s.name,
            s.id,
            s.classes.len(),
            s.splits.train,
            s.splits.val,
            s.splits.test,
            c.error,
            c.warning,
            c.info
        );
        out(&detail, summary)
    }

    pub(super) async fn list_dataset_issues(&self, a: ListIssuesArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        if let Some(k) = &a.kind
            && !folo_api::ISSUE_KINDS.contains(&k.as_str())
        {
            return Err(invalid_arguments(format!(
                "unknown issue kind `{}`; kind is one of {}",
                short(k, 40),
                folo_api::ISSUE_KINDS.join(", ")
            )));
        }
        self.ensure_dataset(&a.dataset_id).await?;
        let q = IssueQuery { kind: a.kind.clone(), severity: a.severity, offset, limit };
        let id = a.dataset_id.clone();
        let page = self.blocking(move |s| list_issues_compat(|q| s.list_issues(&id, q), &q)).await?;
        let counts = page.counts;
        let page = fit_page(page.issues.into_iter().map(short_issue), offset, limit, page.total);
        let filter: Vec<&str> = a.severity.map(|s| s.as_str()).into_iter().chain(a.kind.as_deref()).collect();
        let summary = format!(
            "{} of {} {}issues of {}; {} errors, {} warnings, {} infos in all",
            page.items.len(),
            page.total,
            if filter.is_empty() { String::new() } else { format!("{} ", filter.join(" ")) },
            a.dataset_id,
            counts.error,
            counts.warning,
            counts.info
        );
        let result =
            IssuesPage { items: page.items, next_cursor: page.next_cursor, total: page.total, counts };
        out(&result, summary)
    }

    pub(super) async fn list_dataset_images(&self, a: ListImagesArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        self.ensure_dataset(&a.dataset_id).await?;
        let q = ImageQuery { split: a.split, offset, limit, class: a.class };
        let id = a.dataset_id.clone();
        let page = self.blocking(move |s| s.list_images(&id, &q)).await?;
        let with_boxes = a.include_boxes;
        let items = page.images.into_iter().map(|i| ImageSummary {
            image_id: i.id,
            file: i.file,
            split: i.split,
            width: i.width,
            height: i.height,
            boxes: with_boxes.then_some(i.boxes),
        });
        let page = fit_page(items, offset, limit, page.total);
        let summary = format!("{} of {} images of {}", page.items.len(), page.total, a.dataset_id);
        out(&page, summary)
    }

    pub(super) async fn register_dataset(&self, a: RegisterArgs) -> Result<Output, Error> {
        if self.opts.confine && self.proxy.is_some() {
            return Err(Error::new(
                http::StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                "registering a dataset is not available with --confine through `folo serve` (the server reads the dataset files without the confinement guard)",
            )
            .hint("run `folo mcp --confine` on the data dir itself (no server), or without --confine"));
        }
        if let Some(n) = &a.name {
            check_name("name", n)?;
        }
        let path = self.resolve(a.path).await?;
        let req = RegisterDataset { name: a.name, path: folo_api::json::path(&path), format: a.format };
        // The store returns the existing dataset for a folder registered before (same canonical path).
        let (d, existed) = self
            .blocking(move |s| {
                let before: HashSet<String> = s.list_datasets()?.into_iter().map(|d| d.id).collect();
                let d = s.register_dataset(req)?;
                let existed = before.contains(&d.id);
                Ok((d, existed))
            })
            .await?;
        // The root is inside the data dir; with --confine its contents must be too (a data.yaml or links can point
        // elsewhere). If not, unregister it again (files are never touched); one registered earlier stays.
        if let Err(e) = self.ensure_dataset(&d.id).await {
            if !existed {
                let (id, c) = (d.id.clone(), self.confinement.clone());
                let _ = self.blocking(move |s| s.delete_dataset(&id)).await;
                c.forget(&d.id);
            }
            return Err(e);
        }
        let summary = if existed {
            format!("{} is already registered as {} ({}); returned it", d.path, d.id, d.name)
        } else {
            format!("registered {} as {} ({} classes)", d.name, d.id, d.classes.len())
        };
        out(&d, summary)
    }

    pub(super) async fn create_synthetic(
        &self,
        a: SyntheticArgs,
        progress: &mut Progress,
    ) -> Result<Output, Error> {
        if !(1..=100_000).contains(&a.images) {
            return Err(invalid_arguments(format!("images must be in 1..=100000, got {}", a.images)));
        }
        check_name("name", &a.name)?;
        if a.classes.len() > 80 || a.classes.iter().any(|c| c.chars().count() > 64) {
            return Err(invalid_arguments("classes must hold at most 80 names of at most 64 characters"));
        }
        let total = a.images as f64;
        progress.report(0.0, Some(total), false).await;
        let req = SyntheticDataset {
            name: a.name,
            images: a.images,
            classes: a.classes,
            imgsz: a.imgsz,
            seed: a.seed,
        };
        let d = self.blocking(move |s| s.create_synthetic(req)).await?;
        progress.report(total, Some(total), true).await;
        let summary = format!(
            "created {} ({}): {} train / {} val images, classes {}",
            d.name,
            d.id,
            d.splits.train,
            d.splits.val,
            d.classes.join(", ")
        );
        out(&d, summary)
    }
}
