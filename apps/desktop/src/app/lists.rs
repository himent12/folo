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
    /// Per list, the ticket of a failed list's retry still out (0: none): one at a time, so a server
    /// that never answers isn't asked again at every refresh.
    retrying: [u64; 3],
}

impl ListTickets {
    /// A ticket for a fetch of `list` asked for now.
    pub fn ask(&mut self) -> u64 {
        self.seq += 1;
        self.seq
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
        self.retrying[list as usize] = self.ask();
        Some(self.seq)
    }

    /// The answer to `ticket` is in (news or not): a retry it was is out no longer.
    pub fn answered(&mut self, list: List, ticket: u64) {
        if self.retrying[list as usize] == ticket {
            self.retrying[list as usize] = 0;
        }
    }

    /// `list` is asked for again from scratch (Retry, a server switch): every answer still out is
    /// dropped, a retry among them too, so the new request goes out at once and only its answer counts.
    pub fn restart(&mut self, list: List) {
        self.changed(list);
        self.retrying[list as usize] = 0;
    }
}

impl FoloApp {
    /// Fetches `list`. Loading, its answer fills it (a failure shows); loaded, the answer replaces it in
    /// place (a failure leaves it as it is). Either way only while it is still news.
    pub(super) fn fetch_list(&mut self, list: List, cx: &mut Context<Self>) {
        let ticket = self.list_tickets.ask();
        self.fetch_list_as(list, ticket, cx);
    }

    /// Fetches failed `list` again in place: its error stays on screen until an answer replaces it (the
    /// list, or a newer error), as Studio's polls do, not the loading state at every refresh while the
    /// server is down. Nothing is asked while the last retry is still out.
    pub(super) fn retry_list(&mut self, list: List, cx: &mut Context<Self>) {
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
                        this.install_datasets(l, cx);
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

    fn install_datasets(&mut self, list: Vec<Dataset>, cx: &mut Context<Self>) {
        if self.store.datasets.ready() == Some(&list) {
            return;
        }
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
        self.store.dataset.remove(id);
        self.store.images.remove(id);
        self.store.previews.remove(id);
        self.store.issues.remove(id);
        self.issue_reads.remove(id);
        self.store.evals.remove(id);
        self.store.run_data.remove(id);
        self.store.samples.remove(id);
        self.store.logs.remove(id);
        self.forget_history(id);
        self.ds_recheck.remove(id);
        if self.inspector.as_ref().is_some_and(|i| i.dataset == id) {
            self.take_inspector(cx);
        }
        if let Some((name, list)) = super::navigation::deleted_route(&self.deleted, &self.route) {
            let title = super::navigation::gone_title(&name, &self.route);
            self.toast(crate::ui::Tone::Neutral, title, None, cx);
            self.route_pending = Some(list);
        }
    }
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
}
