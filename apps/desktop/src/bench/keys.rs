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
}

impl BenchForm {
    /// The button that opens it: the focus goes back to it when the form closes.
    fn opener(self) -> &'static str {
        match self {
            BenchForm::NewBench => "bench-new",
            BenchForm::NewLoop => "bench-loop-new",
            BenchForm::Design => "bench-design",
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
        if self.bench.design.as_ref().is_some_and(|f| on_tab(BenchTab::Plan, &f.bench)) {
            Some(BenchForm::Design)
        } else if self.bench.new_loop.as_ref().is_some_and(|f| on_tab(BenchTab::Loops, &f.bench)) {
            Some(BenchForm::NewLoop)
        } else if self.bench.new_bench.is_some() && self.route == Route::Benches {
            Some(BenchForm::NewBench)
        } else {
            None
        }
    }

    /// Esc: a delete waiting for its second click stands down, else the form on screen closes.
    /// False when there was none (Esc then goes back a page).
    pub fn dismiss_bench_form(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.bench.confirm_delete.take().is_some() {
            cx.notify();
            return true;
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
            None => return false,
        }
        true
    }
}
