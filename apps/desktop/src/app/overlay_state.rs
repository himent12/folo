//! Opening and closing what sits above the page (toasts, the command palette, the stop
//! confirmation, the image inspector), the focus they take and give back, and Esc (`dismiss`).
//! `crate::overlays` draws them.

use std::time::Duration;

use folo_ui::{App, AppContext, Context, FocusHandle, Focusable, ScrollHandle, SharedString, Window};

use crate::input::{InputEvent, TextInput};
use crate::ui::*;

use super::{Action, FoloApp, Inspector, Palette, Route, Toast};

/// Focus target remembered before an overlay opened, with enough page context to reject stale or
/// positionally shifted targets when the overlay closes.
pub(super) struct OverlayReturn {
    handle: FocusHandle,
    route: Route,
    play_pred_seq: u64,
}

/// How long a toast shows before it leaves by itself (not counting time hovered or focused): a
/// toast with an action gets long enough to read it and reach the button.
pub(super) fn toast_ttl(has_action: bool) -> Duration {
    Duration::from_millis(if has_action { 12_000 } else { 4200 })
}

impl FoloApp {
    pub fn toast(
        &mut self,
        tone: Tone,
        title: impl Into<SharedString>,
        body: Option<SharedString>,
        cx: &mut Context<Self>,
    ) {
        self.toast_with(tone, title, body, None, cx);
    }

    pub fn toast_with(
        &mut self,
        tone: Tone,
        title: impl Into<SharedString>,
        body: Option<SharedString>,
        action: Option<(SharedString, Action)>,
        cx: &mut Context<Self>,
    ) {
        self.toast_seq += 1;
        let id = self.toast_seq;
        // Every toast expires, none while hovered or focused; one with an action stays longer, so
        // there is time to use it.
        let ttl = toast_ttl(action.is_some());
        self.toasts.push(Toast { id, tone, title: title.into(), body, action });
        // Over three, the oldest passive toast goes first: one with an action stays for its action
        // (the stack folds them) unless five are waiting. The focused toast stays, and so does the
        // one just added (it would vanish unseen).
        while self.toasts.len() > 3 {
            let keep = |t: &Toast| t.id != id && Some(t.id) != self.toast_focus;
            let gone = self
                .toasts
                .iter()
                .position(|t| t.action.is_none() && keep(t))
                .or_else(|| (self.toasts.len() > 5).then(|| self.toasts.iter().position(keep)).flatten());
            let Some(i) = gone else { break };
            self.toasts.remove(i);
        }
        if !self.frozen {
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(ttl).await;
                loop {
                    let hovered = this
                        .update(cx, |this, _| {
                            this.toast_hover == Some(id)
                                || this.toast_focus == Some(id)
                                || this.toast_stack_focus
                        })
                        .unwrap_or(false);
                    if !hovered {
                        break;
                    }
                    cx.background_executor().timer(Duration::from_millis(800)).await;
                }
                this.update(cx, |this, cx| {
                    this.toasts.retain(|t| t.id != id);
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
        cx.notify();
    }

    pub fn dismiss_toast(&mut self, id: u64, cx: &mut Context<Self>) {
        self.toasts.retain(|t| t.id != id);
        cx.notify();
    }

    pub fn open_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.palette.is_some() {
            self.close_palette(window, cx);
            return;
        }
        // One overlay at a time: the palette doesn't open over a dialog or the inspector.
        if self.modal_open() {
            return;
        }
        self.remember_focus(window, cx);
        self.ensure_benches(cx);
        let input =
            cx.new(|cx| TextInput::new("Search runs, models, datasets, benchmarks and commands…", cx));
        let sub = cx.subscribe(&input, |this: &mut Self, _, ev: &InputEvent, cx| {
            let InputEvent::Changed(q) = ev;
            if let Some(p) = this.palette.as_mut() {
                p.query = q.to_string();
                p.selected = 0;
                p.shown.clear();
            }
            cx.notify();
        });
        let fh = input.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.palette = Some(Palette {
            input,
            query: String::new(),
            selected: 0,
            shown: Vec::new(),
            scroll: ScrollHandle::new(),
            _sub: sub,
        });
        cx.notify();
    }

    pub fn close_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.palette = None;
        self.restore_focus(window);
        cx.notify();
    }

    /// Remember the focused control before an overlay takes focus. When another overlay is already
    /// open and a pre-overlay return target is saved, keep it so overlay-to-overlay hand-offs
    /// (e.g. Command Palette → New run / Delete) never clobber the page control with a dying
    /// overlay's own focus handle.
    pub fn remember_focus(&mut self, window: &Window, cx: &App) {
        if self.any_overlay() && self.overlay_return.is_some() {
            return;
        }
        self.overlay_return = window.focused(cx).map(|handle| OverlayReturn {
            handle,
            route: self.route.clone(),
            play_pred_seq: self.play_pred_seq,
        });
    }

    /// Give focus back to whatever had it before the overlay opened, falling back to `root_focus`
    /// if the route changed, positional Playground detection rows (`det-*`) shifted while the
    /// overlay was open, or the remembered handle is no longer mounted in the frame.
    pub fn restore_focus(&mut self, window: &mut Window) {
        let Some(ret) = self.overlay_return.take() else {
            self.verify_focus = None;
            window.focus(&self.root_focus);
            return;
        };
        let shifted_det = ret.route == Route::Playground
            && self.play_pred_seq != ret.play_pred_seq
            && self.focus.borrow().iter().any(|(k, fh)| fh.id == ret.handle.id && k.starts_with("det-"));
        if ret.route == self.route && !shifted_det && window.is_focus_mounted(&ret.handle) {
            window.focus(&ret.handle);
            // Re-verify once this closing frame prepaints in case the remembered control vanished on
            // the same tick that closed the overlay (e.g. confirming Stop, Disconnect, or Terminate).
            self.verify_focus = Some(ret.handle);
        } else {
            self.verify_focus = None;
            window.focus(&self.root_focus);
        }
    }

    /// Ask before stopping a run; the safe choice has focus.
    pub fn ask_stop(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.remember_focus(window, cx);
        self.confirm_stop = Some(id);
        let fh = self.fh("stop-cancel", cx);
        window.focus(&fh);
        cx.notify();
    }

    /// Ask before deleting a run or a dataset; the safe choice has focus.
    pub fn ask_delete(&mut self, target: super::DeleteTarget, window: &mut Window, cx: &mut Context<Self>) {
        self.remember_focus(window, cx);
        self.palette = None;
        self.confirm_delete = Some(target);
        self.delete_files = false;
        let fh = self.fh("del-cancel", cx);
        window.focus(&fh);
        cx.notify();
    }

    pub fn open_inspector(
        &mut self,
        dataset: String,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ds_scroll_anchor = None;
        self.inspector = Some(Inspector {
            dataset,
            index,
            zoom: 1.0,
            pan: (0., 0.),
            drag_from: None,
            hover_box: None,
            focused_box: None,
            boxes: true,
            labels: true,
            want: None,
            shown: None,
            sample: None,
        });
        self.inspector_return = window.focused(cx);
        window.focus(&self.inspector_focus);
        cx.notify();
    }

    /// Opens the inspector on one image of a dataset by id (a run's sample tile): it lands there
    /// once the image list is in, and ←/→ then walk the dataset from it.
    pub fn open_sample(
        &mut self,
        dataset: String,
        image: String,
        run: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let view = self.sample_view;
        self.open_inspector(dataset.clone(), 0, window, cx);
        if let Some(i) = self.inspector.as_mut() {
            i.want = Some(image);
            i.sample = Some((run, view));
        }
        self.ensure_dataset(&dataset, cx);
    }

    /// Every way an inspector goes away: a read of its dataset held while it was open
    /// (`ds_recheck`) goes now, whatever page is under it (a run's has none that would ask).
    pub(crate) fn take_inspector(&mut self, cx: &mut Context<Self>) -> Option<Inspector> {
        let closed = self.inspector.take();
        if let Some(i) = &closed
            && self.ds_recheck.contains(&i.dataset)
        {
            self.ensure_dataset(&i.dataset, cx);
        }
        closed
    }

    pub fn close_inspector(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Whether closing the inspector will trigger (or join) a dataset record recheck on this page.
        let recheck_pending = self.inspector.as_ref().is_some_and(|i| {
            i.sample.is_none()
                && self.route == super::Route::Dataset(i.dataset.clone())
                && (self.ds_recheck.contains(&i.dataset) || self.ds_reads.contains_key(&i.dataset))
        });
        let closed = self.take_inspector(cx);
        let opener = self.inspector_return.take();
        // Opened from the dataset's grid: back on the image it showed last (scrolled into view and
        // focused), not on the tile it was opened from (←/→ may have walked far from it).
        let last =
            closed.filter(|i| i.sample.is_none() && self.route == super::Route::Dataset(i.dataset.clone()));
        if let Some(i) = last
            && let Some(meta) = self.filtered_image(&i.dataset, i.index).cloned()
        {
            self.grid_scroll.scroll_to_item(i.index / self.grid_cols(), folo_ui::ScrollStrategy::Center);
            let fh = self.fh(format!("tile-{}", meta.id), cx);
            window.focus(&fh);
            if recheck_pending {
                // Anchor on the exact inspected file (not column 0 of its row) until the recheck
                // resolves, and remember the focus handle so a later page never steals focus from
                // an overlay or control opened in the meantime.
                self.ds_scroll_anchor = Some(super::data::DatasetScrollAnchor {
                    dataset: i.dataset,
                    file: meta.file,
                    split: meta.split,
                    intra_row: 0.,
                    strategy: Some(folo_ui::ScrollStrategy::Center),
                    pending_recheck: true,
                    focus_from: Some(fh),
                });
            }
        } else {
            match opener {
                Some(fh) => window.focus(&fh),
                None => window.focus(&self.root_focus),
            }
        }
        cx.notify();
    }

    pub fn dismiss(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.menu_open.is_some() || self.export_open {
            self.menu_open = None;
            self.export_open = false;
        } else if self.confirm_close.is_some() {
            // On top of every other overlay: it goes first, and theirs stay as they were.
            self.keep_open(window, cx);
        } else if self.palette.is_some() {
            self.close_palette(window, cx);
        } else if self.confirm_delete.is_some() {
            self.confirm_delete = None;
            self.restore_focus(window);
        } else if self.confirm_stop.is_some() {
            self.confirm_stop = None;
            self.restore_focus(window);
        } else if self.cloud.disconnect.is_some() || self.cloud.terminate.is_some() {
            self.close_confirm(window, cx);
        } else if self.cloud.connect.is_some() {
            self.close_connect(window, cx);
        } else if self.new_run.is_some() {
            // Kept as a draft, like Cancel.
            self.new_run_draft = self.new_run.take().and_then(|f| self.keep_draft(f));
            self.restore_focus(window);
        } else if self.synth.is_some() {
            self.close_synth(window, cx);
        } else if self.inspector.is_some() {
            self.close_inspector(window, cx);
        } else if self.dismiss_bench_form(window, cx) {
            // A Benchmarks form (or a delete's second click) closed; the page stays.
        } else if let Some(p) = self.parent_route() {
            self.navigate(p, window, cx);
        }
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    use crate::api::RunStatus;
    use crate::app::Route;
    use crate::app::tests::{drive, open};

    #[test]
    fn every_toast_expires_and_action_toasts_last_longer() {
        let (plain, action) = (super::toast_ttl(false), super::toast_ttl(true));
        assert!(action > plain && action.as_secs() >= 10 && action.as_secs() <= 15, "{action:?}");
    }

    #[test]
    fn overlay_handoff_preserves_page_focus_and_falls_back_when_control_vanishes_or_shifts() {
        let (mut app, window) = open();
        app.update(|cx| {
            crate::app::bind_keys(cx);
            crate::input::bind_keys(cx);
        });
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Runs, w, cx);
            this.focus_key("runs-new", w, cx);
        });
        app.frame(window.into());

        // 1. Opening palette then chaining directly into ask_delete must preserve `runs-new`,
        // never clobbering `overlay_return` with the dropped palette input handle.
        drive(&mut app, window, |this, w, cx| {
            this.open_palette(w, cx);
            this.ask_delete(crate::app::DeleteTarget::Run("run_52e8".into()), w, cx);
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.dismiss(w, cx));
        app.frame(window.into());
        assert!(
            drive(&mut app, window, |this, w, cx| this.fh("runs-new", cx).is_focused(w)),
            "dismissing chained overlay restores original pre-palette control"
        );

        // 2. If the remembered control disappears while the overlay is open (e.g. live-mini
        // vanishes when the running job finishes), closing the overlay falls back to `root_focus`
        // instead of stranding focus on an unmounted handle.
        drive(&mut app, window, |this, w, cx| {
            this.focus_key("live-mini", w, cx);
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            this.open_palette(w, cx);
            for r in this.store.runs.ready_mut().unwrap() {
                if r.status == RunStatus::Running {
                    r.status = RunStatus::Completed;
                }
            }
            cx.notify();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.close_palette(w, cx));
        app.frame(window.into());
        assert!(
            drive(&mut app, window, |this, w, _| this.root_focus.is_focused(w)),
            "unmounted live-mini falls back cleanly to root_focus"
        );

        // 3. If a control disappears on the very same tick that closes a dialog (post-prepaint
        // verification), focus falls back to `root_focus` in that same frame.
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Settings, w, cx);
        });
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        let conn_id = drive(&mut app, window, |this, w, cx| {
            let id = this.cloud.connections.ready().unwrap()[0].id.clone();
            this.focus_key(&format!("cloud-disc-{id}"), w, cx);
            id
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            this.ask_disconnect(conn_id.clone(), w, cx);
        });
        app.frame(window.into());
        drive(&mut app, window, |this, _, cx| {
            this.cloud.disconnect = None;
            this.cloud.connections.ready_mut().unwrap().retain(|c| c.id != conn_id);
            this.restore_pending = true;
            cx.notify();
        });
        app.frame(window.into());
        assert!(
            drive(&mut app, window, |this, w, _| this.root_focus.is_focused(w)),
            "same-tick removed control falls back to root_focus after prepaint verification"
        );

        // 4. On Playground, if positional detection rows (`det-*`) change predictions while the
        // palette is open, closing the palette falls back to `root_focus` instead of focusing a
        // shifted detection at the old index.
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Playground, w, cx);
        });
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| {
            let fh = this.fh("det-0", cx);
            w.focus(&fh);
            this.open_palette(w, cx);
            this.play_pred_seq += 1;
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.close_palette(w, cx));
        app.frame(window.into());
        assert!(
            drive(&mut app, window, |this, w, _| this.root_focus.is_focused(w)),
            "shifted positional det-* index falls back to root_focus"
        );
    }
}
