//! Deleting an entry, a loop or a benchmark after the second click.

use folo_ui::Context;

use super::load::route_bench;
use crate::app::{FoloApp, Route};
use crate::ui::Tone;

impl FoloApp {
    /// Deletes what [`BenchState::confirm_delete`] names: an entry or loop of a benchmark, or the
    /// benchmark itself (which then closes).
    ///
    /// [`BenchState::confirm_delete`]: super::state::BenchState::confirm_delete
    pub fn confirm_bench_delete(&mut self, cx: &mut Context<Self>) {
        let Some((bench, item)) = self.bench.confirm_delete.take() else { return };
        let key = item.clone().unwrap_or_else(|| bench.clone());
        if !self.bench.busy.insert(key.clone()) {
            return;
        }
        let is_loop = item.as_ref().is_some_and(|id| {
            self.bench
                .loops
                .get(&bench)
                .and_then(|l| l.ready())
                .is_some_and(|l| l.iter().any(|x| &x.id == id))
        });
        let (b, i) = (bench.clone(), item.clone());
        self.fetch(
            cx,
            move |api| match &i {
                None => api.delete_bench(&b),
                Some(id) if is_loop => api.delete_bench_loop(&b, id),
                Some(id) => api.delete_bench_entry(&b, id),
            },
            move |this, r, cx| {
                this.bench.busy.remove(&key);
                match (r, &item) {
                    (Ok(()), None) => {
                        this.forget(&bench);
                        if let Some(list) = this.bench.list.ready_mut() {
                            list.retain(|x| x.id != bench);
                        }
                        this.bench.entries.remove(&bench);
                        this.bench.loops.remove(&bench);
                        this.bench.results.retain(|(b, _), _| *b != bench);
                        if route_bench(&this.route) == Some(bench.as_str()) {
                            this.route_pending = Some(Route::Benches);
                        }
                        this.toast(Tone::Info, "Benchmark deleted", None, cx);
                    }
                    (Ok(()), Some(id)) => {
                        this.forget(id);
                        if let Some(list) = this.bench.entries.get_mut(&bench).and_then(|l| l.ready_mut()) {
                            list.retain(|x| &x.id != id);
                        }
                        if let Some(list) = this.bench.loops.get_mut(&bench).and_then(|l| l.ready_mut()) {
                            list.retain(|x| &x.id != id);
                        }
                        this.bench.results.remove(&(bench.clone(), id.clone()));
                        if matches!(&this.route, Route::BenchEntry(b, e) if *b == bench && e == id) {
                            this.route_pending = Some(Route::Bench(bench.clone()));
                        }
                        this.refresh_benches(cx);
                    }
                    (Err(e), _) => this.toast(Tone::Danger, "Couldn't delete it", Some(e.message.into()), cx),
                }
            },
        );
        cx.notify();
    }
}
