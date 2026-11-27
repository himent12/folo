//! The open benchmark's event stream: opening it, and applying its events from the app's pump.

use std::sync::mpsc::TryRecvError;

use folo_ui::Context;

use super::load::route_bench;
use super::merge::{upsert_entry, upsert_loop};
use crate::api::*;
use crate::app::FoloApp;
use crate::ui::Tone;

impl FoloApp {
    /// Called from the app's 100 ms pump: applies the open benchmark's events, keeps its stream open
    /// while a benchmark page shows and closes it when none does.
    pub fn bench_pump(&mut self, tick: u64, cx: &mut Context<Self>) -> bool {
        let open = route_bench(&self.route).map(str::to_string);
        if self.bench.events.as_ref().is_some_and(|(id, _)| Some(id) != open.as_ref()) {
            self.bench.events = None;
        }
        let Some(id) = open else {
            self.bench.sub_for = None;
            return false;
        };
        if !self.bench_on() {
            return false;
        }
        let mut changed = false;
        let mut ended = false;
        let mut reload = false;
        loop {
            let ev = match self.bench.events.as_ref().map(|(_, rx)| rx.try_recv()) {
                Some(Ok(ev)) => ev,
                Some(Err(TryRecvError::Empty)) | None => break,
                Some(Err(TryRecvError::Disconnected)) => {
                    ended = true;
                    break;
                }
            };
            changed = true;
            match ev {
                BenchEvent::Bench(b) => {
                    self.touch(&b.id);
                    let list = match self.bench.list.ready_mut() {
                        Some(list) => list,
                        None => &mut self.bench.early_list,
                    };
                    match list.iter_mut().find(|x| x.id == b.id) {
                        Some(old) => *old = *b,
                        None => list.insert(0, *b),
                    }
                    // Sent when the stream opens, at a benchmark's milestones and to a subscriber that
                    // fell behind (whose missed events are gone): the lists are reconciled from the
                    // server.
                    reload = true;
                }
                BenchEvent::Entry(e) => {
                    self.touch(&e.id);
                    let finished = !e.status.is_active();
                    let known = self
                        .bench
                        .entries
                        .get(&id)
                        .and_then(|l| l.ready())
                        .and_then(|l| l.iter().find(|x| x.id == e.id))
                        .map(|x| x.status.is_active());
                    // An entry that ends here, or one first seen ended (predictions sent in one
                    // request are scored at once): its place on the leaderboard comes from the
                    // server's order.
                    if finished && known != Some(false) {
                        reload = true;
                    }
                    if finished && known == Some(true) {
                        if e.status == BenchEntryStatus::Done {
                            let detail = e.score.as_ref().map(|s| format!("Score {:.1}", s.total).into());
                            self.toast(Tone::Success, format!("{} finished", e.name), detail, cx);
                        } else if e.status == BenchEntryStatus::Failed {
                            self.toast(
                                Tone::Danger,
                                format!("{} failed", e.name),
                                e.error.clone().map(Into::into),
                                cx,
                            );
                        }
                        self.bench.results.remove(&(id.clone(), e.id.clone()));
                    }
                    match self.bench.entries.get_mut(&id).and_then(|l| l.ready_mut()) {
                        Some(list) => upsert_entry(list, *e),
                        None => upsert_entry(self.bench.early_entries.entry(id.clone()).or_default(), *e),
                    }
                }
                BenchEvent::Loop(l) => {
                    self.touch(&l.id);
                    match self.bench.loops.get_mut(&id).and_then(|x| x.ready_mut()) {
                        Some(list) => upsert_loop(list, *l),
                        None => upsert_loop(self.bench.early_loops.entry(id.clone()).or_default(), *l),
                    }
                }
                BenchEvent::Removed(r) => {
                    self.forget(&r.id);
                    if let Some(list) = self.bench.entries.get_mut(&id).and_then(|l| l.ready_mut()) {
                        list.retain(|x| x.id != r.id);
                    }
                    if let Some(cmp) = self.bench.compare.get_mut(&id) {
                        cmp.retain(|x| x != &r.id);
                    }
                    if let Some(list) = self.bench.loops.get_mut(&id).and_then(|l| l.ready_mut()) {
                        list.retain(|x| x.id != r.id);
                    }
                }
            }
        }
        if ended {
            // The stream dropped: what happened in the gap comes from a reload; it reopens below.
            self.bench.events = None;
            reload = true;
        }
        if reload {
            self.load_bench_entries(&id, cx);
            self.load_bench_loops(&id, cx);
            self.refresh_benches(cx);
        }
        // Open (or reopen, at most every 2 s) the stream of the benchmark on screen.
        let first = self.bench.sub_for.as_deref() != Some(id.as_str());
        if self.bench.events.is_none()
            && self.bench.subscribing.is_none()
            && (first || tick.is_multiple_of(20))
        {
            self.subscribe_bench(id, cx);
        }
        changed
    }

    fn subscribe_bench(&mut self, id: String, cx: &mut Context<Self>) {
        self.bench.subscribing = Some(id.clone());
        self.bench.sub_for = Some(id.clone());
        let a = id.clone();
        self.fetch(
            cx,
            move |api| api.bench_events(&a),
            move |this, r, _| {
                if this.bench.subscribing.as_deref() == Some(id.as_str()) {
                    this.bench.subscribing = None;
                }
                // Only the benchmark still on screen keeps its stream. What changed before it opened
                // (or while it was closed) comes from the reload its first event, the benchmark,
                // starts.
                if let Ok(sub) = r
                    && route_bench(&this.route) == Some(id.as_str())
                {
                    this.bench.events = Some((id, sub));
                }
            },
        );
    }
}
