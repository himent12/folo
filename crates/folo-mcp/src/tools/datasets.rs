//! The dataset tools: list, inspect (issues, images), register and generate synthetic datasets.

use std::collections::HashSet;

use folo_api::{Dataset, IssueCounts, IssuePage, IssueSeverity, RegisterDataset, SyntheticDataset};
use folo_server::store::{ImageQuery, IssueQuery};

use super::Progress;
use super::arguments::check_name;
use super::results::{
    MAX_NAME_CHARS, MAX_RESULT_BYTES, Output, fit_page, json_len, out, page_window, short, summary_classes,
};
use crate::errors::{Error, invalid_arguments};
use crate::server::Inner;
use crate::types::*;

/// `get_dataset` lists at most this many issues.
const MAX_ISSUES: usize = 50;

fn has_counts(p: &IssuePage) -> bool {
    p.counts.error > 0 || p.counts.warning > 0 || p.counts.info > 0
}

fn fetch_all_issues_from(
    fetch: &impl Fn(&IssueQuery) -> Result<IssuePage, Error>,
    mut all: Vec<folo_api::DatasetIssue>,
) -> Result<Vec<folo_api::DatasetIssue>, Error> {
    loop {
        let q = IssueQuery { offset: all.len() as u64, limit: 1000, ..IssueQuery::default() };
        let p = fetch(&q)?;
        let n = p.issues.len();
        all.extend(p.issues);
        if n == 0 || all.len() as u64 >= p.total {
            break;
        }
    }
    Ok(all)
}

/// `Store::list_issues` for `q`, also against a `folo serve` from before severity filters and counts (proxy
/// mode): it ignores `severity` and leaves `counts` out, so the page is rebuilt from all its issues.
pub(super) fn list_issues_compat(
    fetch: impl Fn(&IssueQuery) -> Result<IssuePage, Error>,
    q: &IssueQuery,
) -> Result<IssuePage, Error> {
    let page = fetch(q)?;
    if has_counts(&page) {
        return Ok(page);
    }
    // No counts: a dataset without issues, or an older server. A filtered answer cannot tell them apart (an old
    // server honours `kind`), so ask for everything once.
    let filtered = q.kind.is_some() || q.severity.is_some();
    let probe =
        if filtered { fetch(&IssueQuery { limit: 1, ..IssueQuery::default() })? } else { page.clone() };
    if probe.total == 0 || has_counts(&probe) {
        return Ok(page);
    }
    let all = fetch_all_issues_from(&fetch, Vec::new())?;
    Ok(q.page(all.iter()))
}

/// Fetches the summary issues for `get_dataset` (up to [`MAX_ISSUES`]: errors first, then warnings; infos are
/// only counted) plus dataset-wide counts. Starts with the error bucket (which also carries dataset-wide counts),
/// so clean, info-only, error-only, and full-error datasets take a single call, mixed datasets take two, and
/// older servers without severity filters or counts (which ignore `severity`) fall back in one pass.
pub(super) fn collect_dataset_issues(
    fetch: impl Fn(&IssueQuery) -> Result<IssuePage, Error>,
) -> Result<(Vec<folo_api::DatasetIssue>, IssueCounts), Error> {
    let first = fetch(&IssueQuery {
        severity: Some(IssueSeverity::Error),
        limit: MAX_ISSUES as u64,
        ..IssueQuery::default()
    })?;
    if !has_counts(&first) {
        if first.total == 0 {
            return Ok((Vec::new(), first.counts));
        }
        // Older `folo serve` without severity filters or counts ignores `severity`, so `first` already holds the
        // initial unfiltered slice; pull any remaining pages once and split locally.
        let all = if first.issues.len() as u64 >= first.total {
            first.issues
        } else {
            fetch_all_issues_from(&fetch, first.issues)?
        };
        let counts = IssueCounts::of(&all);
        let mut issues = Vec::new();
        for severity in [IssueSeverity::Error, IssueSeverity::Warning] {
            let remaining = MAX_ISSUES.saturating_sub(issues.len());
            if remaining == 0 {
                break;
            }
            issues.extend(all.iter().filter(|i| i.severity == severity).take(remaining).cloned());
        }
        return Ok((issues, counts));
    }
    let counts = first.counts;
    let mut issues: Vec<_> =
        first.issues.into_iter().filter(|i| i.severity == IssueSeverity::Error).take(MAX_ISSUES).collect();
    if counts.warning > 0 && issues.len() < MAX_ISSUES {
        let limit = (MAX_ISSUES - issues.len()) as u64;
        let q = IssueQuery { severity: Some(IssueSeverity::Warning), limit, ..IssueQuery::default() };
        let page = fetch(&q)?;
        issues.extend(
            page.issues.into_iter().filter(|i| i.severity == IssueSeverity::Warning).take(limit as usize),
        );
    }
    Ok((issues, counts))
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
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        let c = self.confinement.clone();
        // With --confine, datasets that read outside the data dir are not listed.
        let all: Vec<Dataset> = self
            .blocking(move |s| {
                Ok(s.list_datasets()?.into_iter().filter(|d| c.dataset(s, &d.id).is_ok()).collect())
            })
            .await?;
        let total = all.len() as u64;
        let skip = usize::try_from(offset).unwrap_or(usize::MAX);
        let items = all.into_iter().skip(skip).map(|d: Dataset| DatasetSummary {
            id: d.id,
            name: short(&d.name, MAX_NAME_CHARS),
            format: d.format,
            num_classes: d.classes.len(),
            classes: summary_classes(&d.classes),
            splits: d.splits,
        });
        let page = fit_page(items, offset, limit, total);
        let summary = format!("{} of {} datasets", page.items.len(), page.total);
        out(&page, summary)
    }

    pub(super) async fn get_dataset(&self, a: DatasetIdArgs) -> Result<Output, Error> {
        self.ensure_dataset(&a.dataset_id).await?;
        let id = a.dataset_id.clone();
        let (dataset, issues, counts) = self
            .blocking(move |s| {
                let dataset = s.get_dataset(&id)?;
                let (issues, counts) = collect_dataset_issues(|q| s.list_issues(&id, q))?;
                Ok((dataset, issues, counts))
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use folo_api::{DatasetIssue, IssueCounts, IssuePage, IssueSeverity};

    use super::*;
    use crate::Options;
    use crate::proxy::HttpStore;
    use crate::server::FoloMcp;

    fn make_issue(kind: &str, severity: IssueSeverity) -> DatasetIssue {
        DatasetIssue {
            kind: kind.into(),
            severity,
            file: "images/a.jpg".into(),
            line: None,
            message: "msg".into(),
            fix: None,
        }
    }

    #[test]
    fn collect_dataset_issues_short_circuits_clean_and_empty_buckets() {
        // Clean dataset: 1 call, no warning follow-up.
        let calls = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            calls.borrow_mut().push(q.clone());
            Ok(IssuePage { total: 0, issues: vec![], counts: IssueCounts::default() })
        })
        .unwrap();
        assert!(issues.is_empty());
        assert_eq!(counts, IssueCounts::default());
        assert_eq!(calls.borrow().len(), 1);
        assert_eq!(calls.borrow()[0].severity, Some(IssueSeverity::Error));

        // Info-only dataset: 1 call suffices because error and warning buckets are empty.
        let infos: Vec<_> = (0..5).map(|_| make_issue("empty_image", IssueSeverity::Info)).collect();
        let calls = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            calls.borrow_mut().push(q.clone());
            Ok(q.page(infos.iter()))
        })
        .unwrap();
        assert!(issues.is_empty());
        assert_eq!((counts.error, counts.warning, counts.info), (0, 0, 5));
        assert_eq!(calls.borrow().len(), 1);

        // Errors-only dataset: 1 call returns all errors and counts; warning bucket skipped.
        let errs = [
            make_issue("class_overflow", IssueSeverity::Error),
            make_issue("empty_image", IssueSeverity::Info),
        ];
        let calls = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            calls.borrow_mut().push(q.clone());
            Ok(q.page(errs.iter()))
        })
        .unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
        assert_eq!((counts.error, counts.warning, counts.info), (1, 0, 1));
        assert_eq!(calls.borrow().len(), 1);

        // Warnings-only dataset: error call sees counts.warning > 0, then fetches warnings (2 calls).
        let warns = [
            make_issue("clipped_box", IssueSeverity::Warning),
            make_issue("duplicate_box", IssueSeverity::Warning),
            make_issue("empty_image", IssueSeverity::Info),
        ];
        let calls = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            calls.borrow_mut().push(q.clone());
            Ok(q.page(warns.iter()))
        })
        .unwrap();
        assert_eq!(issues.len(), 2);
        assert!(issues.iter().all(|i| i.severity == IssueSeverity::Warning));
        assert_eq!((counts.error, counts.warning, counts.info), (0, 2, 1));
        assert_eq!(calls.borrow().len(), 2);
        assert_eq!(calls.borrow()[1].severity, Some(IssueSeverity::Warning));

        // 60 errors + 5 warnings: 1 call fills MAX_ISSUES (50) errors, warning bucket skipped.
        let mut many: Vec<_> = (0..60).map(|_| make_issue("class_overflow", IssueSeverity::Error)).collect();
        many.extend((0..5).map(|_| make_issue("clipped_box", IssueSeverity::Warning)));
        let calls = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            calls.borrow_mut().push(q.clone());
            Ok(q.page(many.iter()))
        })
        .unwrap();
        assert_eq!(issues.len(), MAX_ISSUES);
        assert!(issues.iter().all(|i| i.severity == IssueSeverity::Error));
        assert_eq!((counts.error, counts.warning, counts.info), (60, 5, 0));
        assert_eq!(calls.borrow().len(), 1);

        // Errors + warnings (< MAX_ISSUES): 2 calls (error + warning in order).
        let mixed = [
            make_issue("empty_image", IssueSeverity::Info),
            make_issue("clipped_box", IssueSeverity::Warning),
            make_issue("class_overflow", IssueSeverity::Error),
        ];
        let calls = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            calls.borrow_mut().push(q.clone());
            Ok(q.page(mixed.iter()))
        })
        .unwrap();
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
        assert_eq!(issues[1].severity, IssueSeverity::Warning);
        assert_eq!((counts.error, counts.warning, counts.info), (1, 1, 1));
        assert_eq!(calls.borrow().len(), 2);
    }

    #[test]
    fn collect_dataset_issues_handles_older_server_across_pages() {
        // Older server ignoring `severity` and omitting `counts`, with > 1050 issues so pagination advances.
        let mut all: Vec<_> = (0..1050).map(|_| make_issue("empty_image", IssueSeverity::Info)).collect();
        for _ in 0..30 {
            all.push(make_issue("clipped_box", IssueSeverity::Warning));
        }
        for _ in 0..30 {
            all.push(make_issue("class_overflow", IssueSeverity::Error));
        }
        let offsets = RefCell::new(Vec::new());
        let (issues, counts) = collect_dataset_issues(|q| {
            offsets.borrow_mut().push(q.offset);
            let page = all.iter().skip(q.offset as usize).take(q.limit as usize).cloned().collect();
            Ok(IssuePage { total: all.len() as u64, issues: page, counts: IssueCounts::default() })
        })
        .unwrap();
        // Initial slice (0..50), then offset 50 (1000 items -> 1050), then offset 1050 (60 items -> 1110).
        assert_eq!(*offsets.borrow(), vec![0, 50, 1050]);
        assert_eq!(issues.len(), MAX_ISSUES);
        assert_eq!(issues.iter().filter(|i| i.severity == IssueSeverity::Error).count(), 30);
        assert_eq!(issues.iter().filter(|i| i.severity == IssueSeverity::Warning).count(), 20);
        assert!(issues[..30].iter().all(|i| i.severity == IssueSeverity::Error));
        assert!(issues[30..].iter().all(|i| i.severity == IssueSeverity::Warning));
        assert_eq!((counts.error, counts.warning, counts.info), (30, 30, 1050));
    }

    #[tokio::test]
    async fn list_datasets_validates_paging_before_store_access() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_bg = Arc::clone(&hits);
        let accept = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                hits_bg.fetch_add(1, Ordering::SeqCst);
                drop(stream);
            }
        });
        let http = Arc::new(HttpStore::new(&url, None));
        let home = folo_test_support::TestDir::new("mcp-list-datasets-paging");
        let opts = Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        };
        let server = FoloMcp::new(http.clone(), Some(http), opts.clone());
        for args in [
            PageArgs { limit: Some(0), cursor: None },
            PageArgs { limit: Some(MAX_LIMIT + 1), cursor: None },
            PageArgs { limit: None, cursor: Some("bogus".into()) },
            PageArgs { limit: Some(10), cursor: Some("o-1".into()) },
            PageArgs { limit: None, cursor: Some("o18446744073709551616".into()) },
        ] {
            let err = server.inner.list_datasets(args).await.err().unwrap();
            assert_eq!(err.body.code, "invalid_arguments");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 0, "invalid paging must not hit the store or proxy server");
        accept.abort();
        let _ = accept.await;

        // `u64::MAX` cursor returns an empty page with no next_cursor and no overflow.
        let mock = FoloMcp::new(Arc::new(folo_server::MockStore::new()), None, opts);
        let out = mock
            .inner
            .list_datasets(PageArgs { limit: Some(10), cursor: Some(format!("o{}", u64::MAX)) })
            .await
            .unwrap();
        assert_eq!(out.value["items"].as_array().unwrap().len(), 0);
        assert!(out.value["next_cursor"].is_null());
    }
}
