//! "Design with an agent": opening it, asking for a plan and using the proposal.

use gpui::{AppContext, Context, Focusable, Window};

use super::state::DesignForm;
use crate::api::*;
use crate::app::FoloApp;
use crate::input::TextInput;
use crate::ui::Tone;

impl FoloApp {
    /// Opens "Design with an agent" for a benchmark, with the first installed agent.
    pub fn open_design(&mut self, bench: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.ensure_agents(cx);
        let agent = self.bench_agents().first().map_or(LabelAgentId::Claude, |a| a.id);
        let note = cx.new(|cx| TextInput::new("Optional: what matters most, what to leave out", cx));
        let fh = note.read(cx).focus_handle(cx);
        self.bench.form_seq += 1;
        self.bench.design = Some(DesignForm {
            bench: bench.to_string(),
            agent,
            note,
            asking: false,
            proposal: None,
            applying: false,
            error: None,
            seq: self.bench.form_seq,
        });
        window.focus(&fh);
        cx.notify();
    }

    /// Asks the chosen agent for a plan.
    pub fn ask_design(&mut self, cx: &mut Context<Self>) {
        let Some(f) = self.bench.design.as_mut() else { return };
        if f.asking {
            return;
        }
        let req = DesignBench {
            agent: f.agent,
            model: None,
            effort: None,
            note: f.note.read(cx).text().trim().to_string(),
        };
        f.asking = true;
        f.error = None;
        let (id, seq) = (f.bench.clone(), f.seq);
        self.fetch(
            cx,
            move |api| api.design_bench(&id, &req),
            move |this, r, _| {
                let Some(f) = this.bench.design.as_mut().filter(|f| f.seq == seq) else { return };
                f.asking = false;
                match r {
                    Ok(d) => f.proposal = Some(d),
                    Err(e) => f.error = Some(e),
                }
            },
        );
        cx.notify();
    }

    /// Makes the proposed plan the benchmark's.
    pub fn apply_design(&mut self, cx: &mut Context<Self>) {
        let Some(f) = self.bench.design.as_mut() else { return };
        let Some(plan) = f.proposal.as_ref().map(|d| d.plan.clone()) else { return };
        if f.applying {
            return;
        }
        f.applying = true;
        f.error = None;
        let (id, seq) = (f.bench.clone(), f.seq);
        let req = UpdateBench { name: None, description: None, plan: Some(plan) };
        let since = self.bench.ev_seq;
        self.fetch(
            cx,
            move |api| api.update_bench(&id, &req),
            move |this, r, cx| match r {
                Ok(b) => {
                    if this.bench.design.as_ref().is_some_and(|f| f.seq == seq) {
                        this.bench.design = None;
                    }
                    // The benchmark as the answer has it, unless the stream has said something newer.
                    if !this.bench.touched.get(&b.id).is_some_and(|&s| s > since) {
                        this.touch(&b.id);
                        if let Some(list) = this.bench.list.ready_mut()
                            && let Some(old) = list.iter_mut().find(|x| x.id == b.id)
                        {
                            *old = b;
                        }
                    }
                    this.toast(Tone::Success, "Plan updated", None, cx);
                }
                Err(e) => {
                    if let Some(f) = this.bench.design.as_mut().filter(|f| f.seq == seq) {
                        f.applying = false;
                        f.error = Some(e);
                    }
                }
            },
        );
        cx.notify();
    }
}
