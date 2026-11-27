//! Esc and Ctrl+Enter on the Benchmarks screens' inline forms (New benchmark, New loop, Design with
//! an agent) and on a delete waiting for its second click: the form on screen answers, as the
//! dialogs do, before Esc goes back a page.

use folo_ui::{Context, Window};

use super::load::route_bench;
use super::state::BenchTab;
use crate::app::{FoloApp, Route};

/// The bench form the page on screen shows, the one a key goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchForm {
    NewBench,
    NewLoop,
    Design,
    External,
}

impl BenchForm {
    /// The button that opens it: the focus goes back to it when the form closes.
    fn opener(self) -> &'static str {
        match self {
            BenchForm::NewBench => "bench-new",
            BenchForm::NewLoop => "bench-loop-new",
            BenchForm::Design => "bench-design",
            BenchForm::External => "bench-external-open",
        }
    }
}

impl FoloApp {
    /// Closes `form` (Esc, Cancel, Close, Discard) and gives the focus back to the button that
    /// opened it, as the dialogs do: its field had the focus and is gone with it.
    pub fn close_bench_form(&mut self, form: BenchForm, window: &mut Window, cx: &mut Context<Self>) {
        match form {
            BenchForm::Design => self.bench.design = None,
            BenchForm::NewLoop => self.bench.new_loop = None,
            BenchForm::NewBench => self.bench.new_bench = None,
            BenchForm::External => self.bench.external = None,
        }
        let opener = self.fh(form.opener(), cx);
        window.focus(&opener);
        cx.notify();
    }

    /// The inline form shown on the page now, if any (a form left on another tab isn't).
    pub fn shown_bench_form(&self) -> Option<BenchForm> {
        let here = route_bench(&self.route);
        let on_tab = |tab: BenchTab, bench: &str| {
            matches!(&self.route, Route::Bench(_)) && self.bench.tab == tab && here == Some(bench)
        };
        if self.bench.external.as_ref().is_some_and(|f| on_tab(BenchTab::Leaderboard, &f.bench)) {
            Some(BenchForm::External)
        } else if self.bench.design.as_ref().is_some_and(|f| on_tab(BenchTab::Plan, &f.bench)) {
            Some(BenchForm::Design)
        } else if self.bench.new_loop.as_ref().is_some_and(|f| on_tab(BenchTab::Loops, &f.bench)) {
            Some(BenchForm::NewLoop)
        } else if self.bench.new_bench.is_some() && self.route == Route::Benches {
            Some(BenchForm::NewBench)
        } else {
            None
        }
    }

    /// Esc: closes a visible inline entry image preview, stands down a visible delete waiting for
    /// its second click, or closes the inline form on screen, restoring focus to the triggering
    /// control. False when none is open on this screen (Esc then goes back a page).
    pub fn dismiss_bench_form(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if let Route::BenchEntry(bid, eid) = &self.route
            && let Some((_, _, img_id)) =
                self.bench.selected_entry_image.as_ref().filter(|(sb, se, _)| sb == bid && se == eid).cloned()
        {
            let (bid, eid) = (bid.clone(), eid.clone());
            self.bench.selected_entry_image = None;
            self.bench.entry_image_reveal = None;
            let entry_ready = self.bench.list.ready().is_some_and(|l| l.iter().any(|b| b.id == bid))
                && self
                    .bench
                    .entries
                    .get(&bid)
                    .and_then(|l| l.ready())
                    .and_then(|l| l.iter().find(|e| e.id == eid))
                    .is_some_and(|e| e.status == crate::api::BenchEntryStatus::Done)
                && self.bench.results.get(&(bid.clone(), eid.clone())).is_some_and(|l| l.ready().is_some());
            let visible = entry_ready
                && self
                    .bench
                    .entry_images
                    .get(&(bid.clone(), eid))
                    .and_then(|x| x.page.ready())
                    .is_some_and(|p| p.images.iter().any(|im| im.image.id == img_id));
            if visible {
                self.bench.entry_image_reveal = Some(crate::bench::EntryImageReveal::Tile(img_id.clone()));
                let key = if self.bench_image_failed(&bid, &img_id, None) {
                    format!("eimg-retry-{img_id}")
                } else {
                    format!("eimg-open-{img_id}")
                };
                let fh = self.fh(key, cx);
                window.focus(&fh);
                cx.notify();
                return true;
            }
        }
        if let Some((bench, item)) = self.bench.confirm_delete.take()
            && matches!(&self.route, Route::Bench(cur) if cur == &bench)
        {
            let target = match item {
                None => Some("bench-del".to_string()),
                Some(id) => {
                    let on_loop = self.bench.tab == BenchTab::Loops
                        && self
                            .bench
                            .loops
                            .get(&bench)
                            .and_then(|l| l.ready())
                            .is_some_and(|l| l.iter().any(|x| x.id == id));
                    let on_entry = self.bench.tab == BenchTab::Leaderboard
                        && self
                            .bench
                            .entries
                            .get(&bench)
                            .and_then(|l| l.ready())
                            .is_some_and(|l| l.iter().any(|x| x.id == id));
                    if on_loop {
                        Some(format!("bench-loop-{id}-del"))
                    } else if on_entry {
                        Some(format!("bench-entry-{id}-del"))
                    } else {
                        None
                    }
                }
            };
            if let Some(target) = target {
                let fh = self.fh(target, cx);
                window.focus(&fh);
                cx.notify();
                return true;
            }
        }
        let Some(form) = self.shown_bench_form() else { return false };
        self.close_bench_form(form, window, cx);
        true
    }

    /// Ctrl+Enter (and Enter in the New benchmark name): the form on screen does what its primary
    /// button does. False when there was none.
    pub fn submit_bench_form(&mut self, cx: &mut Context<Self>) -> bool {
        match self.shown_bench_form() {
            Some(BenchForm::NewBench) => self.submit_new_bench(cx),
            Some(BenchForm::NewLoop) => {
                let Some(bench) = self.bench.new_loop.as_ref().map(|f| f.bench.clone()) else { return false };
                self.submit_new_loop(&bench, cx);
            }
            // Asking only: a proposal replaces the plan, so it is used with its own button after a
            // look at it, and Ctrl+Enter then does nothing.
            Some(BenchForm::Design) => {
                if self.bench.design.as_ref().is_some_and(|f| f.proposal.is_none()) {
                    self.ask_design(cx);
                }
            }
            Some(BenchForm::External) => self.submit_external_entry(cx),
            None => return false,
        }
        true
    }
}
