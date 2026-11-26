//! The dataset, model and run lists: loaded once, then fetched again in place (the 5 s refresh, a
//! reconnect, after a change made here), each answer installed only if nothing newer stands. Every
//! fetch takes a ticket; an answer is dropped when a later fetch's answer is in already, or when the
//! list was changed here (an add or a delete) after it was asked for: it predates that change, and a
//! fresh fetch follows each one.

use folo_ui::Context;

use crate::api::*;
use crate::store::*;

use super::FoloApp;

/// One of the lists [`ListTickets`] orders.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum List {
    Datasets,
    Models,
    Runs,
}

/// The order of the lists' answers: one ticket counter, and per list the ticket of the answer installed
/// last and the last ticket handed out before the list was changed here.
#[derive(Default)]
pub struct ListTickets {
    seq: u64,
    installed: [u64; 3],
    changed: [u64; 3],
    /// Per dataset id, the ticket of the newest detail record (`GET /api/datasets/{id}`) installed,
    /// sharing `seq` with list fetches so out-of-order list and detail responses reconcile cleanly.
    detail_installed: std::collections::HashMap<String, u64>,
    /// Per dataset id, the shared-sequence ticket of the latest in-flight detail read, so a newer
    /// list summary can cancel an older pending detail request without altering `ds_read_seq`.
    detail_in_flight: std::collections::HashMap<String, u64>,
    /// Per list, all fetch tickets still outstanding, so overlapping explicit/poll fetches never
    /// lose track of an earlier in-flight request.
    in_flight: [Vec<u64>; 3],
    /// Per list, the ticket of a failed list's retry still out (0: none): one at a time, so a server
    /// that never answers isn't asked again at every refresh.
    retrying: [u64; 3],
}

impl ListTickets {
    /// A ticket for a fetch asked for now (shared by list and dataset-detail requests).
    pub fn ask(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// The newest installed ticket for `list` (0 if none installed yet).
    pub fn list_installed(&self, list: List) -> u64 {
        self.installed[list as usize]
    }

    /// Allocates and records a shared-sequence ticket for an in-flight dataset detail read.
    pub fn start_detail(&mut self, id: &str) -> u64 {
        let ticket = self.ask();
        self.detail_in_flight.insert(id.to_string(), ticket);
        ticket
    }

    /// Whether the latest in-flight detail read for `id` predates `list_ticket`.
    pub fn detail_in_flight_older_than(&self, id: &str, list_ticket: u64) -> bool {
        self.detail_in_flight.get(id).is_some_and(|&t| t < list_ticket)
    }

    /// Clears the in-flight detail marker for `id` if it matches `ticket` (or unconditionally when `None`).
    pub fn finish_detail(&mut self, id: &str, ticket: Option<u64>) {
        if ticket.is_none_or(|t| self.detail_in_flight.get(id) == Some(&t)) {
            self.detail_in_flight.remove(id);
        }
    }

    /// Records that a detail record for `id` asked for at `ticket` was installed.
    pub fn note_detail(&mut self, id: &str, ticket: u64) {
        self.finish_detail(id, Some(ticket));
        let e = self.detail_installed.entry(id.to_string()).or_insert(0);
        if ticket > *e {
            *e = ticket;
        }
    }

    /// The newest installed detail ticket for `id` (0 if none installed yet).
    pub fn detail_ticket(&self, id: &str) -> u64 {
        self.detail_installed.get(id).copied().unwrap_or(0)
    }

    /// Drops the recorded detail ticket for `id` (e.g. when the dataset is removed).
    pub fn forget_detail(&mut self, id: &str) {
        self.detail_installed.remove(id);
        self.detail_in_flight.remove(id);
    }

    /// A ticket for a fetch of `list`, recorded as in flight.
    pub fn start(&mut self, list: List) -> u64 {
        let ticket = self.ask();
        self.in_flight[list as usize].push(ticket);
        ticket
    }

    /// A ticket for a background in-place refresh of `list`, unless a still-current fetch or retry
    /// is already in flight.
    pub fn refresh_if_idle(&mut self, list: List) -> Option<u64> {
        let i = list as usize;
        if self.retrying[i] != 0 || self.in_flight[i].iter().any(|&t| self.current(list, t)) {
            return None;
        }
        Some(self.start(list))
    }

    /// Whether an answer to `ticket` is still news: no later fetch's answer is installed, and the list
    /// wasn't changed here after it was asked for.
    pub fn current(&self, list: List, ticket: u64) -> bool {
        ticket > self.installed[list as usize] && ticket > self.changed[list as usize]
    }

    pub fn installed(&mut self, list: List, ticket: u64) {
        self.installed[list as usize] = ticket;
    }

    /// `list` was changed here: every answer asked for until now predates it.
    pub fn changed(&mut self, list: List) {
        self.changed[list as usize] = self.seq;
    }

    /// A ticket for a retry of failed `list`, unless the last one is still out.
    pub fn retry(&mut self, list: List) -> Option<u64> {
        if self.retrying[list as usize] != 0 {
            return None;
        }
        let ticket = self.start(list);
        self.retrying[list as usize] = ticket;
        Some(ticket)
    }

    /// The answer to `ticket` is in (news or not): any retry or in-flight mark it held is cleared.
    pub fn answered(&mut self, list: List, ticket: u64) {
        if self.retrying[list as usize] == ticket {
            self.retrying[list as usize] = 0;
        }
        self.in_flight[list as usize].retain(|&t| t != ticket);
    }

    /// `list` is asked for again from scratch (Retry, a server switch): every answer still out is
    /// dropped, a retry among them too, so the new request goes out at once and only its answer counts.
    pub fn restart(&mut self, list: List) {
        self.changed(list);
        self.in_flight[list as usize].clear();
        self.retrying[list as usize] = 0;
        if list == List::Datasets {
            self.detail_installed.clear();
            self.detail_in_flight.clear();
        }
    }
}

impl FoloApp {
    /// Fetches `list`. Loading, its answer fills it (a failure shows); loaded, the answer replaces it in
    /// place (a failure leaves it as it is). Either way only while it is still news.
    pub(super) fn fetch_list(&mut self, list: List, cx: &mut Context<Self>) {
        if self.hold_loading {
            return;
        }
        let ticket = self.list_tickets.start(list);
        self.fetch_list_as(list, ticket, cx);
    }

    /// Refreshes an already-loaded `list` in place (e.g. on window activation or periodic poll) if
    /// no current fetch for it is already in flight.
    pub(super) fn refresh_list_if_idle(&mut self, list: List, cx: &mut Context<Self>) {
        if self.hold_loading {
            return;
        }
        let loaded = match list {
            List::Datasets => self.store.datasets.ready().is_some(),
            List::Models => self.store.models.ready().is_some(),
            List::Runs => self.store.runs.ready().is_some(),
        };
        if !loaded {
            return;
        }
        if let Some(ticket) = self.list_tickets.refresh_if_idle(list) {
            self.fetch_list_as(list, ticket, cx);
        }
    }

    /// Fetches failed `list` again in place: its error stays on screen until an answer replaces it (the
    /// list, or a newer error), as Studio's polls do, not the loading state at every refresh while the
    /// server is down. Nothing is asked while the last retry is still out.
    pub(super) fn retry_list(&mut self, list: List, cx: &mut Context<Self>) {
        if self.hold_loading {
            return;
        }
        if let Some(ticket) = self.list_tickets.retry(list) {
            self.fetch_list_as(list, ticket, cx);
        }
    }

    /// Retry: the three lists are asked for again from scratch ([`ListTickets::restart`]).
    pub fn restart_lists(&mut self) {
        for list in [List::Datasets, List::Models, List::Runs] {
            self.list_tickets.restart(list);
        }
    }

    fn fetch_list_as(&mut self, list: List, ticket: u64, cx: &mut Context<Self>) {
        // Only the answer that is still news counts; a failure shows only on a list not loaded yet.
        fn take<T>(this: &mut FoloApp, list: List, ticket: u64, r: ApiResult<T>, loaded: bool) -> Option<T> {
            this.list_tickets.answered(list, ticket);
            if !this.list_tickets.current(list, ticket) {
                return None;
            }
            match r {
                Ok(v) => {
                    this.list_tickets.installed(list, ticket);
                    Some(v)
                }
                Err(e) => {
                    if !loaded {
                        match list {
                            List::Datasets => this.store.datasets = Load::Failed(e),
                            List::Models => this.store.models = Load::Failed(e),
                            List::Runs => this.store.runs = Load::Failed(e),
                        }
                    }
                    None
                }
            }
        }
        match list {
            List::Datasets => self.fetch(
                cx,
                |api| api.datasets(),
                move |this, r, cx| {
                    let loaded = this.store.datasets.ready().is_some();
                    if let Some(l) = take(this, list, ticket, r, loaded) {
                        this.install_datasets(l, ticket, cx);
                    }
                },
            ),
            List::Models => self.fetch(
                cx,
                |api| api.models(),
                move |this, r, cx| {
                    let loaded = this.store.models.ready().is_some();
                    if let Some(l) = take(this, list, ticket, r, loaded) {
                        this.install_models(l, cx);
                    }
                },
            ),
            List::Runs => self.fetch(
                cx,
                |api| api.runs(),
                move |this, r, cx| {
                    let loaded = this.store.runs.ready().is_some();
                    if let Some(l) = take(this, list, ticket, r, loaded) {
                        this.install_runs(l, cx);
                    }
                },
            ),
        }
    }

    /// `list` was changed here (a dataset added, a run created or deleted, its models with it): older
    /// answers are dropped, and it is fetched again now. A list not asked for yet loads as usual.
    pub(super) fn changed_here(&mut self, list: List, cx: &mut Context<Self>) {
        self.list_tickets.changed(list);
        let asked = match list {
            List::Datasets => !self.store.datasets.needs_fetch(),
            List::Models => !self.store.models.needs_fetch(),
            List::Runs => !self.store.runs.needs_fetch(),
        };
        // A failed list waits for its retry; one loading has just lost its answer and asks again.
        let failed = match list {
            List::Datasets => self.store.datasets.error().is_some(),
            List::Models => self.store.models.error().is_some(),
            List::Runs => self.store.runs.error().is_some(),
        };
        if asked && !failed {
            self.fetch_list(list, cx);
        }
    }

    /// Effective dataset record for cards, filters, sorting, and subtitle counts: uses the loaded
    /// detail record (which carries `stats` and per-dataset refreshes) when cached and fresh.
    pub(crate) fn effective_dataset<'a>(&'a self, d: &'a Dataset) -> &'a Dataset {
        self.store.dataset.get(&d.id).and_then(|l| l.ready()).unwrap_or(d)
    }

    /// Keeps the cached list summary aligned when a per-dataset record (`GET /api/datasets/{id}`)
    /// refreshes first so list filters, format counts, and cards agree on one record. Leaves
    /// `item.revision` (and `item.stats`) untouched because `GET /api/datasets` omits revisions.
    pub(super) fn sync_dataset_summary(&mut self, fresh: &Dataset) {
        if let Some(list) = self.store.datasets.ready_mut()
            && let Some(item) = list.iter_mut().find(|d| d.id == fresh.id)
        {
            item.name = fresh.name.clone();
            item.path = fresh.path.clone();
            item.format = fresh.format;
            item.classes = fresh.classes.clone();
            item.splits = fresh.splits;
        }
    }

    pub(super) fn install_datasets(&mut self, mut list: Vec<Dataset>, ticket: u64, cx: &mut Context<Self>) {
        // Reconcile against any newer per-dataset detail record that already landed after `ticket`
        // was sent: keep its fresher summary fields instead of regressing or evicting it.
        for d in &mut list {
            if self.list_tickets.detail_ticket(&d.id) > ticket
                && let Some(newer) = self.store.dataset.get(&d.id).and_then(|l| l.ready())
            {
                d.name = newer.name.clone();
                d.path = newer.path.clone();
                d.format = newer.format;
                d.classes = newer.classes.clone();
                d.splits = newer.splits;
            }
        }
        if self.store.datasets.ready() == Some(&list) {
            return;
        }
        let changed_ids: Vec<String> = self
            .store
            .datasets
            .ready()
            .into_iter()
            .flat_map(|old| {
                list.iter().filter_map(|d| {
                    if self.list_tickets.detail_ticket(&d.id) > ticket {
                        return None;
                    }
                    let prev = old.iter().find(|x| x.id == d.id)?;
                    dataset_summary_changed(prev, d).then(|| d.id.clone())
                })
            })
            .collect();
        let gone = gone_from(self.store.datasets.ready(), &list, |d| (&d.id, &d.name));
        for (id, name) in gone {
            self.removed_elsewhere(&id, name, cx);
        }
        // Listed again (its folder registered again, after this answer's fetch was asked for): no
        // longer deleted here.
        for d in &list {
            self.deleted.remove(&d.id);
        }
        self.store.datasets = Load::Ready(list);
        for id in changed_ids {
            // Drop any in-flight detail read that predates this newer list summary so its stale
            // answer cannot overwrite `store.datasets` when it arrives.
            if self.list_tickets.detail_in_flight_older_than(&id, ticket) {
                self.ds_reads.remove(&id);
                self.list_tickets.finish_detail(&id, None);
                if self.store.dataset.get(&id).is_some_and(|l| l.is_loading()) {
                    self.store.dataset.remove(&id);
                }
            }
            let cached_stale = self
                .store
                .dataset
                .get(&id)
                .and_then(|l| l.ready())
                .zip(self.store.dataset_summary(&id))
                .is_some_and(|(cached, summary)| dataset_summary_changed(cached, summary));
            if cached_stale {
                self.ds_recheck.insert(id.clone());
                if !self.inspecting(&id) {
                    self.forget_dataset_contents(&id, cx);
                    self.store.dataset.remove(&id);
                }
            }
        }
        // An open select may list them: its highlight is an index into the list.
        self.menu_open = None;
    }

    fn install_models(&mut self, mut models: Vec<Model>, cx: &mut Context<Self>) {
        models.sort_by(|a, b| b.created.cmp(&a.created));
        if self.store.models.ready() != Some(&models) {
            // A model gone from the list (its run deleted elsewhere).
            for (id, name) in gone_from(self.store.models.ready(), &models, |m| (&m.id, &m.name)) {
                self.removed_elsewhere(&id, name, cx);
            }
            self.store.models = Load::Ready(models);
            self.menu_open = None;
        }
        // The Playground's model is one of these, or replaced now.
        self.reconcile_play_model(cx);
    }

    fn install_runs(&mut self, runs: Vec<Run>, cx: &mut Context<Self>) {
        for (id, name) in gone_from(self.store.runs.ready(), &runs, |r| (&r.id, &r.name)) {
            self.removed_elsewhere(&id, name, cx);
        }
        self.store.runs = Load::Ready(newest_first_runs(runs));
        self.store.sync_run_list();
        self.subscribe_active(cx);
        self.settle_lost_streams(cx);
    }
}

impl FoloApp {
    /// `id` (called `name`) is no longer listed, though nothing here deleted it (Studio, the CLI,
    /// another Desktop did): it is remembered as deleted, what was read of it is dropped, and its page,
    /// if open, gives way to its list and says so, as a deletion here does.
    fn removed_elsewhere(&mut self, id: &str, name: String, cx: &mut Context<Self>) {
        self.deleted.insert(id.to_string(), name);
        self.list_tickets.forget_detail(id);
        self.store.dataset.remove(id);
        self.store.images.remove(id);
        self.images_more.remove(id);
        self.failed_pages.remove(id);
        self.store.previews.remove(id);
        self.store.issues.remove(id);
        self.issue_reads.remove(id);
        self.store.evals.remove(id);
        self.store.run_data.remove(id);
        self.store.samples.remove(id);
        self.store.samples_stale.remove(id);
        self.store.samples_refreshing.remove(id);
        self.store.samples_refresh_error.remove(id);
        self.store.logs.remove(id);
        self.forget_history(id);
        self.ds_recheck.remove(id);
        if self.inspector.as_ref().is_some_and(|i| i.dataset == id) {
            self.take_inspector(cx);
            self.inspector_return = None;
            self.restore_pending = true;
        }
        if let Some((name, list)) = super::navigation::deleted_route(&self.deleted, &self.route) {
            let title = super::navigation::gone_title(&name, &self.route);
            self.toast(crate::ui::Tone::Neutral, title, None, cx);
            self.route_pending = Some(list);
        }
    }
}

/// Whether any summary field carried by `GET /api/datasets` changed between two dataset records.
/// List responses omit `revision` (`None`), so revisions only disagree when both records carry one.
pub(super) fn dataset_summary_changed(a: &Dataset, b: &Dataset) -> bool {
    a.name != b.name
        || a.path != b.path
        || a.format != b.format
        || a.classes != b.classes
        || a.splits != b.splits
        || (a.revision.is_some() && b.revision.is_some() && a.revision != b.revision)
}

/// What `old` listed that `new` doesn't, as (id, name).
fn gone_from<T>(
    old: Option<&Vec<T>>,
    new: &[T],
    key: impl Fn(&T) -> (&String, &String),
) -> Vec<(String, String)> {
    let Some(old) = old else { return Vec::new() };
    old.iter()
        .map(&key)
        .filter(|(id, _)| !new.iter().any(|n| key(n).0 == *id))
        .map(|(id, name)| (id.clone(), name.clone()))
        .collect()
}

/// Runs newest first (the server already sends them so; the mock and refreshes may not).
fn newest_first_runs(mut runs: Vec<Run>) -> Vec<Run> {
    runs.sort_by(|a, b| b.created.cmp(&a.created));
    runs
}

#[cfg(test)]
mod tests {
    use super::{List, ListTickets};

    #[test]
    fn an_answer_counts_only_while_nothing_newer_stands() {
        let mut t = ListTickets::default();
        let (a, b) = (t.ask(), t.ask());
        // The later fetch answered first: the earlier one's answer is old news.
        assert!(t.current(List::Datasets, b));
        t.installed(List::Datasets, b);
        assert!(!t.current(List::Datasets, a));
        // A change here drops every answer asked for before it, on that list only.
        let c = t.ask();
        t.changed(List::Datasets);
        assert!(!t.current(List::Datasets, c));
        assert!(t.current(List::Models, c));
        let d = t.ask();
        assert!(t.current(List::Datasets, d), "the fetch that follows the change counts");
    }

    #[test]
    fn refresh_if_idle_deduplicates_concurrent_refreshes() {
        let mut t = ListTickets::default();
        let first = t.refresh_if_idle(List::Datasets).expect("first refresh starts");
        assert!(t.refresh_if_idle(List::Datasets).is_none(), "second refresh deduplicated while in flight");
        assert!(t.refresh_if_idle(List::Models).is_some(), "other lists stay independent");
        t.answered(List::Datasets, first);
        assert!(t.refresh_if_idle(List::Datasets).is_some(), "next refresh allowed once answered");
    }
}
