//! The dataset tools: list, inspect (issues, images), register and generate synthetic datasets.

use std::collections::HashSet;

use folo_api::{Dataset, IssueCounts, IssuePage, IssueSeverity, RegisterDataset, SyntheticDataset};
use folo_server::Store as _;
use folo_server::store::{ImageQuery, IssueQuery};

use super::Progress;
use super::arguments::{check_name, closest_match};
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
            let mut msg = format!("unknown issue kind `{}`", short(k, 40));
            if let Some(s) = closest_match(k, &folo_api::ISSUE_KINDS) {
                msg.push_str(&format!("; did you mean `{s}`? kind is one of "));
            } else {
                msg.push_str("; kind is one of ");
            }
            msg.push_str(&folo_api::ISSUE_KINDS.join(", "));
            return Err(invalid_arguments(msg));
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
        if self.confinement.active()
            && self.confinement.has_non_local_proxy_url()
            && let Some(proxy) = &self.proxy
        {
            return Err(Error::new(
                http::StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                format!(
                    "register_dataset with --confine is not available through remote server {}",
                    proxy.base()
                ),
            )
            .hint("run `folo mcp --confine` against a local `folo serve` sharing --data-dir, without --server, or without --confine"));
        }
        let path = self.resolve(a.path).await?;
        // In confined local proxy mode, validate the manifest and every referenced file under MCP's read guard
        // BEFORE sending `POST /api/datasets` so an unguarded `folo serve` never opens outside files while indexing.
        if self.confinement.active() && self.proxy.is_some() {
            let (c, p, fmt) = (self.confinement.clone(), path.clone(), a.format);
            tokio::task::spawn_blocking(move || c.pre_register_local(&p, fmt))
                .await
                .map_err(|e| Error::internal(format!("confinement task failed: {e}")))??;
        }
        let req = RegisterDataset { name: a.name, path: folo_api::json::path(&path), format: a.format };
        // Track whether this call created a brand-new exclusive registration (`rollback` token) so a confinement
        // failure rolls back atomically without ever deleting an already-registered or concurrently-used dataset.
        let (d, rollback, existed) = self
            .blocking(move |s| {
                let before: HashSet<String> = s.list_datasets()?.into_iter().map(|d| d.id).collect();
                let (d, rollback) = s.register_dataset_tracked(req)?;
                let existed = rollback.is_none() && before.contains(&d.id);
                Ok((d, rollback, existed))
            })
            .await?;
        // The root is inside the data dir; with --confine its contents must be too (a data.yaml or links can point
        // elsewhere). If not, roll back only a fresh exclusive registration (files are never touched).
        if let Err(e) = self.ensure_dataset(&d.id).await {
            if let Some(token) = rollback {
                let (id, c) = (d.id.clone(), self.confinement.clone());
                let _ = self.blocking(move |s| s.delete_dataset_if_unused(&id, &token)).await;
                c.forget(&d.id);
            } else if !existed && self.proxy.is_none() {
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
        // In proxy mode with --confine, `folo serve` writes synthetic datasets under its own `<data_dir>/datasets`:
        // allow mock servers (procedural `mock://` datasets), refuse non-loopback servers whose filesystem cannot be
        // checked locally, and verify loopback real servers report a `data_dir` whose `datasets/` lies inside this
        // MCP process's --data-dir before generating anything.
        if self.confinement.active()
            && let Some(proxy) = &self.proxy
        {
            let proxy = proxy.clone();
            let paths = self.paths.clone();
            tokio::task::spawn_blocking(move || -> Result<(), Error> {
                if proxy.is_mock() {
                    return Ok(());
                }
                if !crate::confine::is_local_url(proxy.base()) {
                    return Err(Error::new(
                        http::StatusCode::NOT_IMPLEMENTED,
                        "not_supported",
                        format!(
                            "create_synthetic_dataset with --confine is not available through remote server {}",
                            proxy.base()
                        ),
                    )
                    .hint("run `folo mcp --confine` against a local `folo serve` sharing --data-dir, without --server, or without --confine"));
                }
                let sys = proxy.system()?;
                let Some(remote_dir) = sys.data_dir else {
                    return Err(Error::new(
                        http::StatusCode::FORBIDDEN,
                        "path_outside_data_dir",
                        format!(
                            "the proxied folo serve at {} did not report its data_dir, so --confine cannot verify where synthetic datasets are written",
                            proxy.base()
                        ),
                    )
                    .hint("update folo serve, run `folo mcp --confine` without --server, or restart without --confine"));
                };
                let target = std::path::Path::new(&remote_dir).join("datasets");
                paths.check_dir_inside(&target).map_err(|mut e| {
                    if e.body.code == "path_outside_data_dir" {
                        e.body.message = format!(
                            "the proxied folo serve writes synthetic datasets under {}, which is outside --data-dir {}",
                            folo_api::json::path(&target),
                            folo_api::json::path(&paths.data_dir)
                        );
                        e.body.hint = Some(
                            "pass the server's --data-dir to `folo mcp --confine --server ...`, run `folo mcp --confine` without --server, or restart without --confine"
                                .into(),
                        );
                    }
                    e
                })?;
                Ok(())
            })
            .await
            .map_err(super::join)??;
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
        // Defense-in-depth: if the server placed the dataset outside --data-dir anyway (e.g. a symlink or an older
        // server omitting `data_dir` in `/api/system`), clean up what this call just created before returning the error.
        if let Err(e) = self.ensure_dataset(&d.id).await {
            let (id, c) = (d.id.clone(), self.confinement.clone());
            let _ =
                self.blocking(move |s| s.delete_dataset_files(&id).or_else(|_| s.delete_dataset(&id))).await;
            c.forget(&d.id);
            return Err(e);
        }
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

    #[test]
    fn loopback_url_detection_covers_local_and_remote_hosts() {
        assert!(crate::confine::is_local_url("http://127.0.0.1:7878"));
        assert!(crate::confine::is_local_url("http://localhost:7878/api"));
        assert!(crate::confine::is_local_url("https://[::1]:7878"));
        assert!(!crate::confine::is_local_url("http://192.168.1.5:7878"));
        assert!(!crate::confine::is_local_url("https://gpu.example.com:7878"));
    }

    #[tokio::test]
    async fn confined_remote_proxy_allows_remote_datasets_and_refuses_remote_register() {
        let root = folo_test_support::TestDir::new("mcp-confine-remote-datasets");
        let remote_dir = root.join("remote_server");
        let local_dir = root.join("local_client");
        std::fs::create_dir_all(&remote_dir).unwrap();
        std::fs::create_dir_all(&local_dir).unwrap();
        std::fs::write(root.join("outside.jpg"), b"not-allowed").unwrap();

        // Populate a real store on `remote_dir` (paths exist on disk outside `local_dir` — worst-case collision).
        let remote_store = Arc::new(folo_server::RealStore::open(remote_dir.clone()).device("cpu"));
        let ds = remote_store
            .create_synthetic(SyntheticDataset {
                name: "remote_shapes".into(),
                images: 4,
                classes: vec!["circle".into()],
                imgsz: 64,
                seed: 1,
            })
            .unwrap();

        // Pair it with a non-local proxy URL whose `/api/system` reports a remote `data_dir` that does not exist
        // locally so `Confinement` verifies it as a separate remote host.
        let remote_proxy = Arc::new(
            HttpStore::new("https://gpu.example.com:7878", None)
                .assume_data_dir(Some("/nonexistent/remote-gpu-server/folo".into())),
        );
        let opts = Options {
            data_dir: local_dir.clone(),
            read_only: false,
            confine: true,
            device: "cpu".into(),
            published_weights: None,
        };
        let server = FoloMcp::new(remote_store.clone(), Some(remote_proxy), opts.clone());

        // Remote datasets are visible and readable rather than blanket-hidden with `path_outside_data_dir`.
        let list = server.inner.list_datasets(PageArgs { limit: None, cursor: None }).await.unwrap();
        assert_eq!(list.value["total"], 1, "{}", list.value);
        let got = server.inner.get_dataset(DatasetIdArgs { dataset_id: ds.id.clone() }).await.unwrap();
        assert_eq!(got.value["id"], ds.id);
        let imgs = server
            .inner
            .list_dataset_images(ListImagesArgs {
                dataset_id: ds.id.clone(),
                split: None,
                class: None,
                include_boxes: false,
                limit: None,
                cursor: None,
            })
            .await
            .unwrap();
        assert_eq!(imgs.value["total"], 4);

        // Registering a path on a remote server under `--confine` is refused cleanly as `not_supported`.
        let reg_err = server
            .inner
            .register_dataset(RegisterArgs { path: ".".into(), name: None, format: Default::default() })
            .await
            .err()
            .unwrap();
        assert_eq!(reg_err.body.code, "not_supported");

        // Local file reads by the MCP process remain strictly confined to `local_dir`.
        let pred_err = server
            .inner
            .predict(PredictArgs {
                model_id: "m_1".into(),
                image_path: Some("../outside.jpg".into()),
                image_base64: None,
                dataset_id: None,
                image_id: None,
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
            })
            .await
            .err()
            .unwrap();
        assert_eq!(pred_err.body.code, "path_outside_data_dir");

        // A non-loopback LAN IP (`192.168.1.5`) whose `/api/system` `data_dir` EXISTS on this local filesystem
        // (e.g. a local `folo serve` reached via its LAN interface) fails closed and stays strictly confined.
        let lan_proxy = Arc::new(
            HttpStore::new("http://192.168.1.5:7878", None)
                .assume_data_dir(Some(folo_api::json::path(&remote_dir))),
        );
        let lan_server = FoloMcp::new(remote_store.clone(), Some(lan_proxy), opts);
        let lan_list = lan_server.inner.list_datasets(PageArgs { limit: None, cursor: None }).await.unwrap();
        assert_eq!(lan_list.value["total"], 0, "local LAN server with existing data_dir must stay confined");
        let lan_err =
            lan_server.inner.get_dataset(DatasetIdArgs { dataset_id: ds.id.clone() }).await.err().unwrap();
        assert_eq!(lan_err.body.code, "path_outside_data_dir");
        remote_store.shutdown();
    }

    #[tokio::test]
    async fn confined_local_proxy_enforces_shared_fs_and_safe_register_rollback() {
        let root = folo_test_support::TestDir::new("mcp-confine-local-proxy-rollback");
        let server_dir = root.join("server");
        let other_dir = root.join("other");
        let outside_ds = root.join("outside_ds");
        std::fs::create_dir_all(outside_ds.join("images/train")).unwrap();
        std::fs::create_dir_all(outside_ds.join("labels/train")).unwrap();
        image::RgbImage::from_pixel(16, 16, image::Rgb([1, 2, 3]))
            .save(outside_ds.join("images/train/a.png"))
            .unwrap();
        std::fs::write(outside_ds.join("labels/train/a.txt"), "0 0.5 0.5 0.4 0.4\n").unwrap();
        std::fs::write(outside_ds.join("data.yaml"), "train: images/train\nnames: [box]\n").unwrap();

        // Sneaky dataset inside `server_dir` whose `data.yaml` points at `outside_ds`.
        let sneaky = server_dir.join("sneaky");
        std::fs::create_dir_all(&sneaky).unwrap();
        let yaml =
            format!("path: {}\ntrain: images/train\nnames: [box]\n", folo_api::json::path(&outside_ds));
        std::fs::write(sneaky.join("data.yaml"), yaml).unwrap();
        std::fs::create_dir_all(&other_dir).unwrap();

        // Keep the default 5-second `recheck_every` so the test exercises manifest edits inside the server's
        // recheck window (not `Duration::ZERO`).
        let store = Arc::new(folo_server::RealStore::open(server_dir.clone()).device("cpu"));
        let preexisting = store
            .register_dataset(RegisterDataset {
                name: Some("preexisting".into()),
                path: folo_api::json::path(&outside_ds),
                format: Default::default(),
            })
            .unwrap();
        let inside_synth = store
            .create_synthetic(SyntheticDataset {
                name: "inside".into(),
                images: 4,
                classes: vec!["circle".into()],
                imgsz: 64,
                seed: 2,
            })
            .unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let srv = folo_server::Server::new(store.clone(), Default::default());
        let task = tokio::spawn(srv.serve(listener, async move {
            let _ = stop_rx.await;
        }));

        // 1. Same `--data-dir`: inside dataset is visible, outside preexisting dataset is hidden, and registering
        //    `sneaky` fails confinement and rolls back via `delete_dataset_if_unused` without removing `preexisting`.
        let http = Arc::new(HttpStore::new(&url, None));
        let same_dir = FoloMcp::new(
            http.clone(),
            Some(http.clone()),
            Options {
                data_dir: server_dir.clone(),
                read_only: false,
                confine: true,
                device: "cpu".into(),
                published_weights: None,
            },
        );
        let listed = same_dir.inner.list_datasets(PageArgs { limit: None, cursor: None }).await.unwrap();
        assert_eq!(listed.value["total"], 1);
        assert_eq!(listed.value["items"][0]["id"], inside_synth.id);

        let sneaky_err = same_dir
            .inner
            .register_dataset(RegisterArgs { path: "sneaky".into(), name: None, format: Default::default() })
            .await
            .err()
            .unwrap();
        assert_eq!(sneaky_err.body.code, "path_outside_data_dir");
        let on_server: Vec<String> = store.list_datasets().unwrap().into_iter().map(|d| d.id).collect();
        assert_eq!(on_server.len(), 2, "sneaky registration must be rejected before indexing: {on_server:?}");
        assert!(on_server.contains(&preexisting.id), "preexisting dataset must never be removed");
        let cached_count = std::fs::read_dir(server_dir.join("cache").join("datasets")).unwrap().count();
        assert_eq!(cached_count, 2, "folo serve must never build an index for sneaky");

        // 2. Manifest edited mid-session on the local server must invalidate the cached `Ok(())` verdict
        //    immediately even inside the server's default 5-second recheck window (for both `get_dataset` and
        //    `train`), before `folo serve` has even bumped `Dataset.revision`.
        let edited_yaml = std::path::Path::new(&inside_synth.path).join("data.yaml");
        std::fs::write(
            &edited_yaml,
            format!("path: {}\ntrain: images/train\nnames: [box]\n", folo_api::json::path(&outside_ds)),
        )
        .unwrap();
        let after_edit = same_dir
            .inner
            .get_dataset(DatasetIdArgs { dataset_id: inside_synth.id.clone() })
            .await
            .err()
            .unwrap();
        assert_eq!(after_edit.body.code, "path_outside_data_dir");
        let train_args =
            serde_json::json!({ "dataset_id": inside_synth.id, "epochs": 1 }).as_object().unwrap().clone();
        let train_err = same_dir.inner.train(train_args).await.err().unwrap();
        assert_eq!(train_err.body.code, "path_outside_data_dir");

        // 3. Different `--data-dir` on loopback (shared filesystem): server's `server_dir` datasets stay hidden
        //    so paths outside `other_dir` are never exposed on a shared filesystem.
        let other_mcp = FoloMcp::new(
            http.clone(),
            Some(http),
            Options {
                data_dir: other_dir,
                read_only: false,
                confine: true,
                device: "cpu".into(),
                published_weights: None,
            },
        );
        let other_list = other_mcp.inner.list_datasets(PageArgs { limit: None, cursor: None }).await.unwrap();
        assert_eq!(other_list.value["total"], 0);
        let err = other_mcp
            .inner
            .get_dataset(DatasetIdArgs { dataset_id: inside_synth.id.clone() })
            .await
            .err()
            .unwrap();
        assert_eq!(err.body.code, "path_outside_data_dir");

        let _ = stop_tx.send(());
        let _ = task.await;
        store.shutdown();
    }

    #[tokio::test]
    async fn confined_proxy_rejects_remote_host_and_missing_data_dir() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let home = folo_test_support::TestDir::new("mcp-confine-remote-or-legacy");
        let opts = Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: true,
            device: "cpu".into(),
            published_weights: None,
        };
        let args = || SyntheticArgs {
            name: "s".into(),
            images: 4,
            classes: vec!["circle".into()],
            imgsz: 64,
            seed: 0,
        };

        // Loopback real server that omits `data_dir` in `/api/system`: refused before creating.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let serve = tokio::spawn(async move {
            for _ in 0..4 {
                let Ok((mut s, _)) = listener.accept().await else { break };
                let mut buf = [0u8; 1024];
                let n = s.read(&mut buf).await.unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]);
                let body = if head.starts_with("GET /api/health") {
                    r#"{"status":"ok","version":"0.1.0","mock":false,"features":[]}"#.to_string()
                } else if head.starts_with("GET /api/system") {
                    serde_json::to_string(&folo_api::SystemResponse {
                        system: folo_api::SystemInfo {
                            cpu: folo_api::CpuInfo { name: "c".into(), threads: 1, util: 0.0 },
                            memory: folo_api::MemoryInfo { used_bytes: 1, total_bytes: 2 },
                            gpus: vec![],
                        },
                        data_dir: None,
                        features: vec![],
                        capabilities: None,
                        gpu_pool: None,
                    })
                    .unwrap()
                } else {
                    panic!("unexpected request before confinement refusal: {head}");
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = s.write_all(resp.as_bytes()).await;
            }
        });
        let http = Arc::new(HttpStore::new(&url, None));
        let server = FoloMcp::new(http.clone(), Some(http), opts);
        let err = server.inner.create_synthetic(args(), &mut Progress::none()).await.err().unwrap();
        assert_eq!(err.body.code, "path_outside_data_dir", "{err:?}");
        serve.abort();
        let _ = serve.await;
    }
}
