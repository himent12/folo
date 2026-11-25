//! Loading: the list, the catalog, a benchmark's entries and loops, one reload at a time, and results.

use folo_ui::Context;

use super::merge::{merge_snapshot, upsert_entry, upsert_loop};
use super::state::{BenchState, BenchTab, Reload};
use crate::api::*;
use crate::app::{FoloApp, Route};
use crate::store::Load;

/// A benchmark route's benchmark id, if the route is one.
pub fn route_bench(route: &Route) -> Option<&str> {
    match route {
        Route::Bench(id) | Route::BenchEntry(id, _) => Some(id),
        _ => None,
    }
}

impl FoloApp {
    /// The server has the `/api/bench` endpoints.
    pub fn bench_on(&self) -> bool {
        self.store.features.iter().any(|f| f == FEATURE_VISION_BENCH)
    }

    /// The server is known not to have them (its system info is in and doesn't list the feature).
    pub fn bench_unsupported(&self) -> bool {
        self.store.system.ready().is_some() && !self.bench_on()
    }

    /// Loads the benchmark list and the catalog, once each.
    pub fn ensure_benches(&mut self, cx: &mut Context<Self>) {
        if !self.bench_on() {
            return;
        }
        if self.bench.list.needs_fetch() {
            // Through the same merge as every later reload: an answer older than the stream's
            // events never undoes them.
            self.bench.list = Load::Loading;
            self.refresh_benches(cx);
        }
        if self.bench.catalog.needs_fetch() {
            self.bench.catalog = Load::Loading;
            self.fetch(cx, |api| api.bench_catalog(), |this, r, _| this.bench.catalog = r.into());
        }
        self.ensure_agents(cx);
    }

    /// Loads a benchmark's entries and loops, once each.
    pub fn ensure_bench(&mut self, id: &str, cx: &mut Context<Self>) {
        self.ensure_benches(cx);
        if !self.bench_on() {
            return;
        }
        if self.bench.entries.get(id).is_none_or(|l| l.needs_fetch()) {
            self.bench.entries.insert(id.to_string(), Load::Loading);
            self.load_bench_entries(id, cx);
        }
        if self.bench.loops.get(id).is_none_or(|l| l.needs_fetch()) {
            self.bench.loops.insert(id.to_string(), Load::Loading);
            self.load_bench_loops(id, cx);
        }
    }

    /// (Re)loads a benchmark's leaderboard, merged with what the stream said meanwhile.
    pub fn load_bench_entries(&mut self, id: &str, cx: &mut Context<Self>) {
        self.reload(id, Reload::Entries, cx);
    }

    /// (Re)loads a benchmark's loops, merged with what the stream said meanwhile.
    pub fn load_bench_loops(&mut self, id: &str, cx: &mut Context<Self>) {
        self.reload(id, Reload::Loops, cx);
    }

    /// Reloads the benchmark list in place (no loading state once it has one).
    pub fn refresh_benches(&mut self, cx: &mut Context<Self>) {
        self.reload("", Reload::List, cx);
    }

    /// One reload per list at a time: asked for while one is out, it runs once more after it (and
    /// so does one whose snapshot disagreed with the stream on what has ended). A failed reload keeps
    /// the list there is.
    fn reload(&mut self, bench: &str, what: Reload, cx: &mut Context<Self>) {
        let key = (bench.to_string(), what);
        if let Some(again) = self.bench.reloading.get_mut(&key) {
            *again = true;
            return;
        }
        self.bench.reloading.insert(key.clone(), false);
        let since = self.bench.ev_seq;
        let b = bench.to_string();
        let finish = move |this: &mut Self, differs: bool, cx: &mut Context<Self>| {
            let again = this.bench.reloading.remove(&key).unwrap_or(false);
            if again || differs {
                this.reload(&key.0, key.1, cx);
            }
        };
        match what {
            Reload::Entries => self.fetch(
                cx,
                {
                    let b = b.clone();
                    move |api| api.bench_entries(&b)
                },
                move |this, r, cx| {
                    let mut differs = false;
                    match r {
                        Ok(fresh) => {
                            let s = &this.bench;
                            let current = s
                                .entries
                                .get(&b)
                                .and_then(|l| l.ready())
                                .or_else(|| s.early_entries.get(&b))
                                .map(Vec::as_slice);
                            let (list, d) = merge_snapshot(
                                current,
                                fresh,
                                since,
                                &s.touched,
                                &s.removed,
                                |e: &BenchEntry| e.id.as_str(),
                                |e| !e.status.is_active(),
                                upsert_entry,
                            );
                            differs = d;
                            this.bench.early_entries.remove(&b);
                            this.bench.entries.insert(b, Load::Ready(list));
                        }
                        Err(e) => {
                            if this.bench.entries.get(&b).is_none_or(|l| l.ready().is_none()) {
                                this.bench.entries.insert(b, Load::Failed(e));
                            }
                        }
                    }
                    finish(this, differs, cx);
                },
            ),
            Reload::Loops => self.fetch(
                cx,
                {
                    let b = b.clone();
                    move |api| api.bench_loops(&b)
                },
                move |this, r, cx| {
                    match r {
                        Ok(fresh) => {
                            let s = &this.bench;
                            let current = s
                                .loops
                                .get(&b)
                                .and_then(|l| l.ready())
                                .or_else(|| s.early_loops.get(&b))
                                .map(Vec::as_slice);
                            let (list, _) = merge_snapshot(
                                current,
                                fresh,
                                since,
                                &s.touched,
                                &s.removed,
                                |l: &BenchLoop| l.id.as_str(),
                                |l| !l.status.is_active(),
                                upsert_loop,
                            );
                            this.bench.early_loops.remove(&b);
                            this.bench.loops.insert(b, Load::Ready(list));
                        }
                        Err(e) => {
                            if this.bench.loops.get(&b).is_none_or(|l| l.ready().is_none()) {
                                this.bench.loops.insert(b, Load::Failed(e));
                            }
                        }
                    }
                    // Loops are listed newest first: a kept version never moves one.
                    finish(this, false, cx);
                },
            ),
            Reload::List => self.fetch(
                cx,
                |api| api.benches(),
                move |this, r, cx| {
                    match r {
                        Ok(fresh) => {
                            let s = &this.bench;
                            let (list, _) = merge_snapshot(
                                s.list.ready().map(Vec::as_slice).or(Some(s.early_list.as_slice())),
                                fresh,
                                since,
                                &s.touched,
                                &s.removed,
                                |b: &Bench| b.id.as_str(),
                                |_| false,
                                |l, b| l.insert(0, b),
                            );
                            this.bench.early_list.clear();
                            this.bench.list = Load::Ready(list);
                        }
                        Err(e) => {
                            if this.bench.list.ready().is_none() {
                                this.bench.list = Load::Failed(e);
                            }
                        }
                    }
                    finish(this, false, cx);
                },
            ),
        }
    }

    /// Notes that `id` is gone (deleted here, or removed on the stream): older snapshots don't bring
    /// it back.
    pub(super) fn forget(&mut self, id: &str) {
        self.touch(id);
        self.bench.removed.insert(id.to_string(), self.bench.ev_seq);
    }

    /// Notes that the stream (or an answer) changed `id` now: older snapshots keep this version.
    pub(super) fn touch(&mut self, id: &str) {
        self.bench.ev_seq += 1;
        self.bench.touched.insert(id.to_string(), self.bench.ev_seq);
    }

    /// Loads a finished entry's full result, once.
    pub fn ensure_bench_result(&mut self, bench: &str, entry: &str, cx: &mut Context<Self>) {
        let key = (bench.to_string(), entry.to_string());
        if self.bench.results.get(&key).is_some_and(|l| !l.needs_fetch()) {
            return;
        }
        self.bench.results.insert(key.clone(), Load::Loading);
        let (b, e) = key.clone();
        self.fetch(
            cx,
            move |api| api.bench_result(&b, &e),
            move |this, r, _| {
                this.bench.results.insert(key, r.into());
            },
        );
    }

    /// Forgets everything benchmark (a new data source, or Retry).
    pub fn clear_bench(&mut self) {
        let (tab, form_seq, image_sort) = (self.bench.tab, self.bench.form_seq, self.bench.image_sort);
        self.bench = BenchState { tab, form_seq, image_sort, ..BenchState::default() };
    }

    /// The Benchmarks section shows the server as it is now (also when going back to it): the list
    /// refreshes in place on the next render; a benchmark's page reloads when its stream opens.
    pub fn after_bench_navigate(&mut self) {
        self.bench.confirm_delete = None;
        // Forms belong to the page they were opened on.
        if self.bench.design.as_ref().is_some_and(|f| route_bench(&self.route) != Some(f.bench.as_str())) {
            self.bench.design = None;
        }
        if !matches!(self.route, Route::Bench(_)) {
            self.bench.new_loop = None;
        }
        if matches!(self.route, Route::Benches) {
            self.bench.list_stale = true;
        }
        if let Route::Bench(_) = self.route {
            // A benchmark opens on its leaderboard, whatever tab the last one was on.
            self.bench.tab = BenchTab::Leaderboard;
        }
        match route_bench(&self.route) {
            Some(id) if self.bench.sort_bench.as_deref() != Some(id) => {
                self.bench.leaderboard_sort = None;
                self.bench.sort_bench = Some(id.to_string());
            }
            None => {
                self.bench.leaderboard_sort = None;
                self.bench.sort_bench = None;
            }
            _ => {}
        }
    }
}
