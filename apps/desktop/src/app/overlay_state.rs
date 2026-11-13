//! Opening and closing what sits above the page (toasts, the command palette, the stop
//! confirmation, the image inspector), the focus they take and give back, and Esc (`dismiss`).
//! `crate::overlays` draws them.

use std::time::Duration;

use folo_ui::{App, AppContext, Context, Focusable, ScrollHandle, SharedString, Window};

use crate::input::{InputEvent, TextInput};
use crate::ui::*;

use super::{Action, FoloApp, Inspector, Palette, Toast};

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
        let input = cx.new(|cx| TextInput::new("Search runs, models, datasets and commands…", cx));
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

    /// Remember the focused control before an overlay takes focus.
    pub fn remember_focus(&mut self, window: &Window, cx: &App) {
        self.overlay_return = window.focused(cx);
    }

    /// Give focus back to whatever had it before the overlay opened.
    pub fn restore_focus(&mut self, window: &mut Window) {
        match self.overlay_return.take() {
            Some(fh) => window.focus(&fh),
            None => window.focus(&self.root_focus),
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
        self.inspector = Some(Inspector {
            dataset,
            index,
            zoom: 1.0,
            pan: (0., 0.),
            drag_from: None,
            hover_box: None,
            want: None,
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
        self.ensure_dataset(&dataset, cx);
        let view = self.sample_view;
        self.open_inspector(dataset, 0, window, cx);
        if let Some(i) = self.inspector.as_mut() {
            i.want = Some(image);
            i.sample = Some((run, view));
        }
    }

    pub fn close_inspector(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let closed = self.inspector.take();
        let opener = self.inspector_return.take();
        // Opened from the dataset's grid: back on the image it showed last (scrolled into view and
        // focused), not on the tile it was opened from (←/→ may have walked far from it).
        let last =
            closed.filter(|i| i.sample.is_none() && self.route == super::Route::Dataset(i.dataset.clone()));
        if let Some(i) = last
            && let Some(id) = self.filtered_image(&i.dataset, i.index).map(|m| m.id.clone())
        {
            self.grid_scroll.scroll_to_item(i.index / self.grid_cols(), folo_ui::ScrollStrategy::Center);
            let fh = self.fh(format!("tile-{id}"), cx);
            window.focus(&fh);
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
        } else if let Some(p) = self.parent_route() {
            self.navigate(p, window, cx);
        }
        cx.notify();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_toast_expires_and_action_toasts_last_longer() {
        let (plain, action) = (super::toast_ttl(false), super::toast_ttl(true));
        assert!(action > plain && action.as_secs() >= 10 && action.as_secs() <= 15, "{action:?}");
    }
}
