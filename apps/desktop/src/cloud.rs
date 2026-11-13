//! Cloud compute (docs/CLOUD.md) on the app's side: the state Settings › Cloud, New run's Run on
//! and a cloud run's page share (connections, instances and offers as the server has them, the
//! Connect dialog and the two confirmations), the actions on it, and the rules those surfaces agree
//! on: which offer New run preselects, what Stop after defaults to, and the `compute` a form sends.
//!
//! Credentials only ever live in the Connect dialog's masked inputs: they are sent once and dropped
//! with the dialog, and nothing here logs, toasts or copies them.

use std::collections::{HashMap, HashSet};

use folo_ui::{AppContext, Context, Entity, Focusable, Window};

use crate::api::*;
use crate::app::FoloApp;
use crate::input::TextInput;
use crate::store::Load;
use crate::ui::Tone;

/// Everything cloud the app holds, reset with the data source.
#[derive(Default)]
pub struct CloudState {
    /// `GET /api/cloud/providers`: what a connection can be made to, and the fields each takes.
    pub providers: Load<Vec<CloudProvider>>,
    pub connections: Load<Vec<CloudConnection>>,
    pub instances: Load<Vec<CloudInstance>>,
    /// Each connection's offers, by connection id.
    pub offers: HashMap<String, Load<CloudOfferList>>,
    /// Connections a Check is out for.
    pub checking: HashSet<String>,
    pub connect: Option<ConnectForm>,
    pub disconnect: Option<Confirm>,
    pub terminate: Option<Confirm>,
    /// Bumped on every opening of the Connect dialog: a late answer only touches its own dialog.
    pub connect_seq: u64,
}

/// The Connect dialog.
pub struct ConnectForm {
    pub id: u64,
    /// The chosen provider, an index into [`CloudState::providers`].
    pub provider: usize,
    /// One input per field of the provider, in its order (secret ones masked). Switching the
    /// provider makes new ones, so nothing typed for one provider reaches another.
    pub fields: Vec<Entity<TextInput>>,
    /// The connection's optional name.
    pub name: Entity<TextInput>,
    pub submitting: bool,
    pub error: Option<ConnectError>,
}

/// Where a failed Connect is shown.
#[derive(Clone, Debug, PartialEq)]
pub enum ConnectError {
    /// Under the field at this index (a missing value, a rejected key).
    Field(usize, String),
    /// For the dialog as a whole: the server's message and hint.
    Dialog(ApiError),
}

/// A confirmation about one connection or instance: Disconnect, Terminate.
pub struct Confirm {
    pub id: String,
    /// The request is out.
    pub busy: bool,
    /// The server's refusal, shown in the dialog (`409 connection_in_use`).
    pub error: Option<ApiError>,
}

/// The control of New run's compute choice a server error is about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ComputeField {
    RunOn,
    Gpu,
    StopAfter,
}

/// The offer New run preselects: the cheapest one that can start now with at least `need` bytes of
/// GPU memory, or, without an estimate, the cheapest that can start. When none is that large, the
/// one with the most memory (the cheapest of those), so the "won't fit" warning stands against the
/// nearest miss.
pub fn pick_offer(offers: &[CloudOffer], need: Option<u64>) -> Option<&CloudOffer> {
    let open = || offers.iter().filter(|o| o.availability != CloudAvailability::None);
    // Compute that isn't billed (a server of yours) is the cheapest there is.
    let cheapest = |fits: &dyn Fn(&CloudOffer) -> bool| {
        open()
            .filter(|o| fits(o))
            .min_by(|a, b| a.price_per_hour.unwrap_or(0.).total_cmp(&b.price_per_hour.unwrap_or(0.)))
    };
    let Some(need) = need else { return cheapest(&|_| true) };
    cheapest(&|o| o.vram_bytes >= need).or_else(|| {
        let most = open().map(|o| o.vram_bytes).max()?;
        cheapest(&|o| o.vram_bytes == most)
    })
}

/// The offers in the order New run's GPU menu lists them: those that can start now, then those with none in
/// stock, each group as the server sent it (cheapest first). A provider's catalogue is mostly out of stock at its
/// cheap end (RunPod: a dozen of the first twenty), and a menu that opens on rows that can't be picked reads as
/// broken.
pub fn menu_order(offers: &[CloudOffer]) -> Vec<&CloudOffer> {
    let none = |o: &&CloudOffer| o.availability == CloudAvailability::None;
    offers.iter().filter(|o| !none(o)).chain(offers.iter().filter(none)).collect()
}

/// Stop after's default: the high end of the time estimate × 1.5 plus half an hour, rounded up to
/// a whole hour, at least 1; 12 (the API's default) until an estimate exists.
pub fn default_max_hours(est_high_s: Option<f64>) -> f32 {
    match est_high_s {
        Some(s) => ((s / 3600. * 1.5 + 0.5).ceil().max(1.) as f32).min(folo_api::MAX_MAX_HOURS),
        None => folo_api::DEFAULT_MAX_HOURS,
    }
}

/// Stop after as typed: hours from 0.25 to 720.
pub fn parse_hours(s: &str) -> Option<f32> {
    s.trim()
        .trim_end_matches('h')
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|h| (folo_api::MIN_MAX_HOURS..=folo_api::MAX_MAX_HOURS).contains(h))
}

/// The `compute` a New run form sends: only while the server has the `cloud` feature and the form
/// has a connection and one of its offers chosen; otherwise the run trains on this machine.
pub fn form_compute(
    cloud: bool,
    run_on: Option<&str>,
    offer: Option<&str>,
    max_hours: f32,
) -> Option<Compute> {
    let (connection_id, offer_id) = (run_on?, offer?);
    cloud.then(|| Compute {
        connection_id: connection_id.into(),
        offer_id: offer_id.into(),
        max_hours,
        disk_gb: None,
    })
}

/// The field a server message is about (`400 invalid_body` names it by key or label), if one is.
pub fn field_named(fields: &[CloudField], message: &str) -> Option<usize> {
    let m = message.to_lowercase();
    fields.iter().position(|f| m.contains(&f.key.to_lowercase()) || m.contains(&f.label.to_lowercase()))
}

/// The control a refused cloud run belongs on: the offer (gone, or the provider says no), the
/// connection (gone), or the time limit (out of range); `None` for the form's general error.
pub fn compute_error_field(e: &ApiError) -> Option<ComputeField> {
    match e.code.as_str() {
        "offer_not_found" | "cloud_refused" => Some(ComputeField::Gpu),
        "connection_not_found" => Some(ComputeField::RunOn),
        "invalid_config" if e.message.contains("max_hours") => Some(ComputeField::StopAfter),
        _ => None,
    }
}

/// Where a refused Connect shows: credentials the provider rejects at the first secret field ("RunPod
/// rejected this key."), a field the server names (`invalid_body`) under that field, anything else
/// for the dialog with the server's message and hint.
pub fn connect_error(provider: &CloudProvider, e: ApiError) -> ConnectError {
    match (e.code.as_str(), provider.fields.iter().position(|f| f.secret)) {
        ("cloud_auth_failed", Some(i)) => {
            let label = &provider.fields[i].label;
            let what =
                if label.to_lowercase().ends_with("key") { "key".to_string() } else { in_sentence(label) };
            ConnectError::Field(i, format!("{} rejected this {what}.", provider.name))
        }
        ("invalid_body", _) => match field_named(&provider.fields, &e.message) {
            Some(i) => ConnectError::Field(i, e.message),
            None => ConnectError::Dialog(e),
        },
        _ => ConnectError::Dialog(e),
    }
}

/// A field label inside a sentence: "the API key", "the token" (an initialism keeps its capitals).
pub fn in_sentence(label: &str) -> String {
    let mut chars = label.chars();
    match (chars.next(), chars.next()) {
        (Some(a), Some(b)) if a.is_uppercase() && b.is_lowercase() => {
            a.to_lowercase().chain(label.chars().skip(1)).collect()
        }
        _ => label.to_string(),
    }
}

impl FoloApp {
    /// The server has the `cloud` feature: without it nothing cloud renders and no request
    /// carries `compute`.
    pub fn cloud_on(&self) -> bool {
        self.store.features.iter().any(|f| f == folo_api::FEATURE_CLOUD)
    }

    /// Loads what Settings › Cloud and New run need, once each (servers with `cloud`).
    pub fn ensure_cloud(&mut self, cx: &mut Context<Self>) {
        if !self.cloud_on() {
            return;
        }
        if self.cloud.providers.needs_fetch() {
            self.cloud.providers = Load::Loading;
            self.fetch(cx, |api| api.cloud_providers(), |this, r, _| this.cloud.providers = r.into());
        }
        if self.cloud.connections.needs_fetch() {
            self.load_connections(cx);
        }
        if self.cloud.instances.needs_fetch() {
            self.cloud.instances = Load::Loading;
            self.refresh_instances(cx);
        }
    }

    pub fn load_connections(&mut self, cx: &mut Context<Self>) {
        self.cloud.connections = Load::Loading;
        self.fetch(cx, |api| api.cloud_connections(), |this, r, _| this.cloud.connections = r.into());
    }

    /// Reloads the instance list in place (no loading state once it has one).
    pub fn refresh_instances(&mut self, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            |api| api.cloud_instances(),
            |this, r, _| {
                // A refresh that fails keeps the list there is.
                if r.is_ok() || this.cloud.instances.ready().is_none() {
                    this.cloud.instances = r.into();
                }
            },
        );
    }

    /// Settings shows the section as the server has it now (also when going back to it).
    pub fn reload_cloud(&mut self) {
        if !self.cloud.connections.is_loading() {
            self.cloud.connections = Load::Idle;
        }
        if !self.cloud.instances.is_loading() {
            self.cloud.instances = Load::Idle;
        }
    }

    /// Forgets everything cloud (a new data source, or Retry).
    pub fn clear_cloud(&mut self) {
        let seq = self.cloud.connect_seq;
        self.cloud = CloudState { connect_seq: seq, ..CloudState::default() };
    }

    /// Whether an instance may still cost money: the list then refreshes every 5 s.
    pub fn instance_alive(&self) -> bool {
        self.cloud.instances.ready().is_some_and(|l| l.iter().any(|i| i.status.is_alive()))
    }

    /// A connection's offers, fetched once per connection (`refresh` asks the provider again).
    pub fn ensure_offers(&mut self, connection: &str, refresh: bool, cx: &mut Context<Self>) {
        if !refresh && self.cloud.offers.get(connection).is_some_and(|l| !l.needs_fetch()) {
            return;
        }
        self.cloud.offers.insert(connection.to_string(), Load::Loading);
        let (a, b) = (connection.to_string(), connection.to_string());
        self.fetch(
            cx,
            move |api| api.cloud_offers(&a, refresh),
            move |this, r, cx| {
                this.cloud.offers.insert(b, r.into());
                this.sync_compute(cx);
            },
        );
    }

    /// Keeps New run's compute choice whole: the chosen connection's offers loading, an offer
    /// preselected (or the one a cloud run's config asked for), and Stop after following the
    /// estimate until it is typed. A connection or offer that no longer exists (or can't be used)
    /// falls back to this machine, with a note.
    pub fn sync_compute(&mut self, cx: &mut Context<Self>) {
        let Some(conn) = self.new_run.as_ref().and_then(|f| f.run_on.clone()).filter(|_| self.cloud_on())
        else {
            return;
        };
        let usable = self
            .cloud
            .connections
            .ready()
            .map(|l| l.iter().any(|c| c.id == conn && c.status == CloudConnectionStatus::Ok));
        if usable == Some(false) {
            self.compute_falls_back("Its cloud connection is gone or not working");
            return;
        }
        self.ensure_offers(&conn, false, cx);
        let offers = self.cloud.offers.get(&conn).and_then(|l| l.ready()).map(|l| l.offers.clone());
        let need = self.nr_estimate_last.as_ref().and_then(|e| e.est_peak_vram_bytes);
        let high = match &self.nr_estimate {
            Some((_, Load::Ready(e))) if e.device == "cloud" => Some(e.est_total_s_high),
            _ => None,
        };
        let Some(f) = self.new_run.as_mut() else { return };
        if let Some(offers) = &offers {
            let startable =
                |id: &str| offers.iter().any(|o| o.id == id && o.availability != CloudAvailability::None);
            match f.want_offer.take() {
                Some(want) if startable(&want) => f.offer = Some(want),
                Some(_) => {
                    self.compute_falls_back("Its GPU isn't offered right now");
                    return;
                }
                None => {}
            }
            if f.offer.as_deref().is_none_or(|id| !offers.iter().any(|o| o.id == id)) {
                f.offer = pick_offer(offers, need).map(|o| o.id.clone());
            }
        }
        let hours = default_max_hours(high);
        if !f.hours_edited && hours != f.max_hours {
            f.max_hours = hours;
            let input = f.hours_input.clone();
            input.update(cx, |i, cx| i.set_text(crate::ui::hours_value(hours), cx));
        }
    }

    /// New run trains on this machine after all, saying why (`why` ends where ": …" follows).
    fn compute_falls_back(&mut self, why: &str) {
        if let Some(f) = self.new_run.as_mut() {
            f.run_on = None;
            f.offer = None;
            f.want_offer = None;
            f.note = Some(format!("{why}: this run trains on this machine. Run on picks another."));
        }
    }

    /// A provider's display name ("RunPod"), from the server's list; its id until that is in.
    pub fn provider_name(&self, id: &str) -> String {
        self.cloud
            .providers
            .ready()
            .and_then(|ps| ps.iter().find(|p| p.id == id))
            .map_or(id.to_string(), |p| p.name.clone())
    }

    pub fn connection(&self, id: &str) -> Option<&CloudConnection> {
        self.cloud.connections.ready()?.iter().find(|c| c.id == id)
    }

    /// Where a cloud run trains, as its header says it: "RunPod · RTX 4090", then the worker's
    /// device once it reports one ("RunPod · RTX 4090 · Vulkan").
    pub fn cloud_device(&self, r: &Run, c: &RunCloud) -> String {
        let provider = self.provider_name(&c.provider);
        // Until the worker reports its device (and for good on a run that failed before training)
        // the server says "<offer's GPU> at <provider>": the provider is already named, so the GPU
        // once.
        let placeholder = format!("{} at ", c.offer.gpu);
        if r.device.is_empty() || r.device.starts_with(&placeholder) {
            format!("{provider} · {}", crate::ui::device_label(&c.offer.gpu))
        } else {
            format!("{provider} · {}", crate::ui::device_label(&r.device))
        }
    }

    /// Checks a connection's credentials again; the row shows the answer.
    pub fn check_connection(&mut self, id: String, cx: &mut Context<Self>) {
        if !self.cloud.checking.insert(id.clone()) {
            return;
        }
        cx.notify();
        let key = id.clone();
        self.fetch(
            cx,
            move |api| api.check_cloud(&id),
            move |this, r, cx| {
                this.cloud.checking.remove(&key);
                match r {
                    Ok(c) => {
                        if let Some(slot) = this
                            .cloud
                            .connections
                            .ready_mut()
                            .and_then(|l| l.iter_mut().find(|x| x.id == c.id))
                        {
                            *slot = c;
                        }
                        // Its offers may read differently now (a key that works again).
                        this.cloud.offers.remove(&key);
                    }
                    Err(e) => {
                        this.toast(Tone::Danger, "Couldn't check the connection", Some(e.message.into()), cx)
                    }
                }
            },
        );
    }

    /// Opens the Connect dialog on the first provider, its first field focused.
    pub fn open_connect(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.modal_open() || self.cloud.providers.ready().is_none_or(|p| p.is_empty()) {
            return;
        }
        self.remember_focus(window, cx);
        self.cloud.connect_seq += 1;
        let name = cx.new(|cx| TextInput::new("", cx));
        self.cloud.connect = Some(ConnectForm {
            id: self.cloud.connect_seq,
            provider: 0,
            fields: Vec::new(),
            name,
            submitting: false,
            error: None,
        });
        self.set_connect_provider(0, cx);
        if let Some(first) = self.cloud.connect.as_ref().and_then(|f| f.fields.first()) {
            window.focus(&first.read(cx).focus_handle(cx));
        }
        cx.notify();
    }

    /// Chooses the provider: a fresh input per field (masked for secrets), the name's placeholder
    /// its name. Nothing typed for the previous provider is kept.
    pub fn set_connect_provider(&mut self, i: usize, cx: &mut Context<Self>) {
        let Some(p) = self.cloud.providers.ready().and_then(|ps| ps.get(i)).cloned() else { return };
        let fields = p
            .fields
            .iter()
            .map(|f| {
                cx.new(|cx| {
                    let input = TextInput::new(f.placeholder.clone().unwrap_or_default(), cx);
                    if f.secret { input.masked() } else { input }
                })
            })
            .collect();
        let name = cx.new(|cx| TextInput::new(p.name.clone(), cx));
        if let Some(form) = self.cloud.connect.as_mut() {
            form.provider = i;
            form.fields = fields;
            form.name = name;
            form.error = None;
        }
        cx.notify();
    }

    pub fn close_connect(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.cloud.connect = None;
        self.restore_focus(window);
        cx.notify();
    }

    /// Sends the dialog's values to the server, which checks them with the provider. Required
    /// fields are checked here first; the server's refusal lands on the field it is about.
    pub fn submit_connect(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(provider) =
            self.cloud.connect.as_ref().and_then(|f| self.cloud.providers.ready()?.get(f.provider)).cloned()
        else {
            return;
        };
        let Some(form) = self.cloud.connect.as_mut() else { return };
        if form.submitting {
            return;
        }
        let values: Vec<String> = form.fields.iter().map(|i| i.read(cx).text().trim().to_string()).collect();
        if let Some(i) = provider.fields.iter().zip(&values).position(|(f, v)| f.required && v.is_empty()) {
            form.error = Some(ConnectError::Field(i, format!("{} is required.", provider.fields[i].label)));
            window.focus(&form.fields[i].read(cx).focus_handle(cx));
            cx.notify();
            return;
        }
        let req = ConnectCloud {
            provider: provider.id.clone(),
            name: form.name.read(cx).text().trim().to_string(),
            values: provider
                .fields
                .iter()
                .zip(values)
                .filter(|(_, v)| !v.is_empty())
                .map(|(f, v)| (f.key.clone(), v))
                .collect(),
        };
        form.submitting = true;
        form.error = None;
        let form_id = form.id;
        cx.notify();
        self.fetch(
            cx,
            move |api| api.connect_cloud(&req),
            move |this, r, cx| {
                let Some(form) = this.cloud.connect.as_mut().filter(|f| f.id == form_id) else { return };
                match r {
                    Ok(c) => {
                        this.cloud.connect = None;
                        // Back to whatever opened the dialog, on the next frame (needs a window).
                        this.restore_pending = true;
                        this.toast(Tone::Success, format!("Connected {}", c.name), None, cx);
                        if let Some(list) = this.cloud.connections.ready_mut() {
                            list.push(c);
                        } else {
                            this.load_connections(cx);
                        }
                    }
                    Err(e) => {
                        form.submitting = false;
                        form.error = Some(connect_error(&provider, e));
                    }
                }
            },
        );
    }

    /// Asks before forgetting a connection's credentials; the safe choice has focus.
    pub fn ask_disconnect(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.remember_focus(window, cx);
        self.cloud.disconnect = Some(Confirm { id, busy: false, error: None });
        window.focus(&self.fh("cd-cancel", cx));
        cx.notify();
    }

    pub fn disconnect(&mut self, cx: &mut Context<Self>) {
        let Some(c) = self.cloud.disconnect.as_mut().filter(|c| !c.busy) else { return };
        c.busy = true;
        c.error = None;
        let id = c.id.clone();
        let name = self.connection(&id).map_or(id.clone(), |c| c.name.clone());
        cx.notify();
        self.fetch(
            cx,
            {
                let id = id.clone();
                move |api| api.disconnect_cloud(&id)
            },
            move |this, r, cx| {
                let mine = this.cloud.disconnect.as_ref().is_some_and(|c| c.id == id);
                match r {
                    Ok(()) => {
                        if mine {
                            this.cloud.disconnect = None;
                            this.restore_pending = true;
                        }
                        if let Some(list) = this.cloud.connections.ready_mut() {
                            list.retain(|c| c.id != id);
                        }
                        this.cloud.offers.remove(&id);
                        this.toast(Tone::Neutral, format!("Disconnected {name}"), None, cx);
                    }
                    Err(e) if mine => {
                        if let Some(c) = this.cloud.disconnect.as_mut() {
                            c.busy = false;
                            c.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    /// Asks before terminating an instance; the safe choice has focus.
    pub fn ask_terminate(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.remember_focus(window, cx);
        self.cloud.terminate = Some(Confirm { id, busy: false, error: None });
        window.focus(&self.fh("ct-cancel", cx));
        cx.notify();
    }

    pub fn terminate(&mut self, cx: &mut Context<Self>) {
        let Some(c) = self.cloud.terminate.as_mut().filter(|c| !c.busy) else { return };
        c.busy = true;
        c.error = None;
        let id = c.id.clone();
        cx.notify();
        self.fetch(
            cx,
            {
                let id = id.clone();
                move |api| api.terminate_instance(&id)
            },
            move |this, r, cx| {
                let mine = this.cloud.terminate.as_ref().is_some_and(|c| c.id == id);
                match r {
                    Ok(inst) => {
                        if mine {
                            this.cloud.terminate = None;
                            this.restore_pending = true;
                        }
                        let what = format!("{} at {}", inst.offer.gpu, this.provider_name(&inst.provider));
                        let title = if inst.status.is_alive() {
                            format!("Terminating the {what}")
                        } else {
                            format!("Terminated the {what}")
                        };
                        if let Some(slot) = this
                            .cloud
                            .instances
                            .ready_mut()
                            .and_then(|l| l.iter_mut().find(|i| i.id == inst.id))
                        {
                            *slot = inst;
                        }
                        this.toast(Tone::Neutral, title, None, cx);
                        // Its run fails: the list says so without waiting for the next refresh.
                        this.refresh_runs(cx);
                    }
                    Err(e) if mine => {
                        if let Some(c) = this.cloud.terminate.as_mut() {
                            c.busy = false;
                            c.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    pub fn close_confirm(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.cloud.disconnect = None;
        self.cloud.terminate = None;
        self.restore_focus(window);
        cx.notify();
    }
}

/// How far one phase of a cloud run got, as its panel marks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseState {
    Done,
    Now,
    Next,
    Failed,
    Skipped,
}

/// Each phase of a cloud run with its state, as Studio marks them (`web/src/lib/cloud.ts`,
/// `phaseStates`). Live, the phases before the current one are done. Once released, the phases
/// before the one its hand-over ended in (`RunCloud::ended_in`) are done, that one failed (a
/// failed run), skipped (stopped before training) or done, and the later ones never reached; a
/// server that doesn't say (`None`, older servers) has every phase done.
pub fn phase_states(c: &RunCloud, r: &Run) -> Vec<(CloudPhase, PhaseState)> {
    // A newer server's phase has no known place among these: it is the one step, current.
    if c.phase == CloudPhase::Unknown {
        return vec![(CloudPhase::Unknown, PhaseState::Now)];
    }
    if c.phase != CloudPhase::Released {
        return CloudPhase::ALL
            .iter()
            .map(|&p| {
                let s = if p < c.phase {
                    PhaseState::Done
                } else if p == c.phase {
                    PhaseState::Now
                } else {
                    PhaseState::Next
                };
                (p, s)
            })
            .collect();
    }
    let failed = r.status == RunStatus::Failed;
    // Ended while collecting but not by the copy back: training is what failed, the copy is done.
    let collected = c.ended_in == Some(CloudPhase::Collecting)
        && failed
        && r.error_code.as_deref() != Some("cloud_sync_failed");
    let ended = if collected { Some(CloudPhase::Training) } else { c.ended_in };
    let end_state = if failed {
        PhaseState::Failed
    } else if r.status == RunStatus::Stopped && ended < Some(CloudPhase::Training) {
        PhaseState::Skipped
    } else {
        PhaseState::Done
    };
    CloudPhase::ALL
        .iter()
        .map(|&p| {
            let s = match ended {
                None => PhaseState::Done,
                _ if p >= CloudPhase::Releasing => PhaseState::Done,
                Some(e) if p < e => PhaseState::Done,
                _ if collected && p == CloudPhase::Collecting => PhaseState::Done,
                Some(e) if p == e => end_state,
                _ => PhaseState::Skipped,
            };
            (p, s)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offer(id: &str, vram_gib: u64, price: Option<f64>, availability: CloudAvailability) -> CloudOffer {
        CloudOffer {
            id: id.into(),
            gpu: id.into(),
            gpu_count: 1,
            vram_bytes: vram_gib << 30,
            price_per_hour: price,
            region: None,
            detail: None,
            availability,
        }
    }

    #[test]
    fn the_cheapest_offer_that_fits_and_can_start_is_preselected() {
        use CloudAvailability as A;
        let offers = vec![
            offer("4090-community", 24, Some(0.34), A::None),
            offer("a4000", 16, Some(0.17), A::High),
            offer("4090", 24, Some(0.69), A::High),
            offer("3090", 24, Some(0.22), A::Low),
            offer("a100", 80, Some(1.64), A::Medium),
        ];
        let pick = |need: Option<u64>| pick_offer(&offers, need).map(|o| o.id.as_str());
        // Without an estimate: the cheapest that can start.
        assert_eq!(pick(None), Some("a4000"));
        // 20 GiB needed: the 16 GiB card is out, the cheaper 4090 has none in stock.
        assert_eq!(pick(Some(20 << 30)), Some("3090"));
        assert_eq!(pick(Some(40 << 30)), Some("a100"));
        // More than any offer has: the largest that can start, the nearest miss (the form warns).
        assert_eq!(pick(Some(200 << 30)), Some("a100"));
        // A server's own GPU has no price: it is the cheapest.
        let own = vec![offer("own", 16, None, A::Unknown), offer("4090", 24, Some(0.69), A::High)];
        assert_eq!(pick_offer(&own, None).map(|o| o.id.as_str()), Some("own"));
        assert!(pick_offer(&[offer("x", 24, Some(1.), A::None)], None).is_none());
    }

    #[test]
    fn the_gpu_menu_lists_what_can_start_first() {
        use CloudAvailability as A;
        let offers = vec![
            offer("a2000", 6, Some(0.12), A::None),
            offer("3080ti", 12, Some(0.18), A::Low),
            offer("a4500", 20, Some(0.19), A::None),
            offer("3090", 24, Some(0.22), A::Unknown),
            offer("a40", 48, Some(0.59), A::Medium),
        ];
        let ids: Vec<&str> = menu_order(&offers).iter().map(|o| o.id.as_str()).collect();
        assert_eq!(ids, ["3080ti", "3090", "a40", "a2000", "a4500"]);
        assert!(menu_order(&[]).is_empty());
    }

    #[test]
    fn stop_after_defaults_from_the_estimate() {
        assert_eq!(default_max_hours(None), 12.);
        // 0.5 h at the high end: 0.75 + 0.5 = 1.25, rounded up to 2.
        assert_eq!(default_max_hours(Some(1800.)), 2.);
        // A few minutes: at least 1.
        assert_eq!(default_max_hours(Some(60.)), 1.);
        // 5 h: 7.5 + 0.5 = 8.
        assert_eq!(default_max_hours(Some(5. * 3600.)), 8.);
        // Never past the API's limit.
        assert_eq!(default_max_hours(Some(1e7)), 720.);
        assert_eq!(parse_hours(" 12 "), Some(12.));
        assert_eq!(parse_hours("0.5h"), Some(0.5));
        assert_eq!(parse_hours("0.1"), None);
        assert_eq!(parse_hours("721"), None);
        assert_eq!(parse_hours("soon"), None);
    }

    #[test]
    fn compute_is_sent_only_with_the_feature_and_a_choice() {
        let c = form_compute(true, Some("cc_1"), Some("o_1"), 8.).expect("compute");
        assert_eq!(
            (c.connection_id.as_str(), c.offer_id.as_str(), c.max_hours, c.disk_gb),
            ("cc_1", "o_1", 8., None)
        );
        assert!(form_compute(false, Some("cc_1"), Some("o_1"), 8.).is_none(), "feature off");
        assert!(form_compute(true, None, Some("o_1"), 8.).is_none(), "this machine");
        assert!(form_compute(true, Some("cc_1"), None, 8.).is_none(), "no offer yet");
    }

    #[test]
    fn a_refused_connect_lands_on_its_field() {
        let api = crate::api::mock::MockBackend::new(0, true);
        let runpod = api.cloud_providers().unwrap().into_iter().find(|p| p.id == "runpod").expect("runpod");
        let req = |key: &str| ConnectCloud {
            provider: "runpod".into(),
            name: String::new(),
            values: [("api_key".to_string(), key.to_string())].into(),
        };
        let rejected = api.connect_cloud(&req("rpa_bad_1234")).unwrap_err();
        assert_eq!(
            connect_error(&runpod, rejected),
            ConnectError::Field(0, "RunPod rejected this key.".into())
        );
        let missing = api.connect_cloud(&req(" ")).unwrap_err();
        assert!(matches!(connect_error(&runpod, missing), ConnectError::Field(0, _)));
        let down = ApiError::new("cloud_unavailable", "RunPod did not answer");
        assert!(matches!(connect_error(&runpod, down), ConnectError::Dialog(_)));
    }

    #[test]
    fn errors_land_on_their_control() {
        let e = |code: &str, msg: &str| ApiError::new(code, msg);
        assert_eq!(compute_error_field(&e("offer_not_found", "gone")), Some(ComputeField::Gpu));
        assert_eq!(compute_error_field(&e("cloud_refused", "no capacity")), Some(ComputeField::Gpu));
        assert_eq!(compute_error_field(&e("connection_not_found", "gone")), Some(ComputeField::RunOn));
        assert_eq!(
            compute_error_field(&e("invalid_config", "max_hours must be from 0.25 to 720")),
            Some(ComputeField::StopAfter)
        );
        assert_eq!(compute_error_field(&e("invalid_config", "epochs must be 1..=10000")), None);
        let fields = vec![
            CloudField {
                key: "url".into(),
                label: "Server URL".into(),
                secret: false,
                required: true,
                placeholder: None,
                help: None,
            },
            CloudField {
                key: "token".into(),
                label: "Token".into(),
                secret: true,
                required: true,
                placeholder: None,
                help: None,
            },
        ];
        assert_eq!(field_named(&fields, "Token is missing"), Some(1));
        assert_eq!(field_named(&fields, "url: not a URL"), Some(0));
        assert_eq!(field_named(&fields, "something else"), None);
        assert_eq!((in_sentence("Token"), in_sentence("API key")), ("token".into(), "API key".into()));
    }
}
