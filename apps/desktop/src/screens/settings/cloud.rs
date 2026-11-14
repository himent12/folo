//! Settings › Cloud: the cloud connections and the instances FOLO launched.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::page::COLUMN_MAX;
use crate::api::Label;
use crate::app::{FoloApp, Route, act};
use crate::screens::{loading_rows, th};
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// Settings › Cloud: the server's cloud connections, each with its account, balance and status,
    /// Check and Disconnect; Connect in the header (docs/CLOUD.md).
    pub(super) fn cloud_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let now = self.store.api.now();
        let connect = match &self.cloud.providers {
            Load::Ready(p) if !p.is_empty() => self
                .button(
                    "cloud-connect",
                    "Connect",
                    crate::controls::Btn::Secondary,
                    Some("icons/plug.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_connect(w, cx)),
                )
                .into_any_element(),
            Load::Failed(_) => self
                .button_disabled(
                    "cloud-connect",
                    "Connect",
                    crate::controls::Btn::Secondary,
                    Some("icons/plug.svg"),
                    "The server didn't list its cloud providers",
                    window,
                    cx,
                )
                .into_any_element(),
            _ => self
                .button_disabled(
                    "cloud-connect",
                    "Connect",
                    crate::controls::Btn::Secondary,
                    Some("icons/plug.svg"),
                    "Loading the providers…",
                    window,
                    cx,
                )
                .into_any_element(),
        };
        // In the header, or the empty state's one action.
        let mut connect = Some(connect);
        let connections = self.cloud.connections.ready().cloned();
        let body = match &self.cloud.connections {
            Load::Failed(e) => div()
                .p(px(16.))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(
                    text(ty::SM, t.danger_text)
                        .flex_1()
                        .min_w_0()
                        .child(format!("Couldn't load the connections: {}", e.message)),
                )
                .child(self.button(
                    "cloud-retry",
                    "Retry",
                    crate::controls::Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.load_connections(cx)),
                ))
                .into_any_element(),
            Load::Ready(list) if list.is_empty() => empty_state(
                &t,
                "icons/cloud.svg",
                "No cloud connected",
                "Connect RunPod to train on a rented GPU, or another FOLO server of yours.",
                Tone::Neutral,
            )
            .py(px(28.))
            .children(connect.take())
            .into_any_element(),
            Load::Ready(_) => div()
                .flex()
                .flex_col()
                .children(
                    connections.unwrap_or_default().iter().map(|c| self.connection_row(c, now, window, cx)),
                )
                .into_any_element(),
            _ => loading_rows(&t, "cloud-sk", 2).into_any_element(),
        };
        panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Cloud").children(connect))
            .child(body)
            .into_any_element()
    }

    /// One connection: name (and its provider when they differ), status, who and what it is, the
    /// last check's error or a note, then Check and Disconnect.
    fn connection_row(
        &mut self,
        c: &crate::api::CloudConnection,
        now: i64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let failed = c.status == crate::api::CloudConnectionStatus::Error;
        let provider = self.provider_name(&c.provider);
        let mut facts: Vec<String> = c.account.iter().cloned().collect();
        if let Some(b) = c.balance_usd {
            facts.push(format!("Balance {}", fmt_cost(b)));
        }
        if let Some(at) = c.checked.as_deref().and_then(crate::api::time::parse) {
            facts.push(format!("checked {}", crate::api::time::ago(at, now)));
        }
        let (dot_c, status, status_c) = if failed {
            (t.danger_solid, "Error", t.danger_text)
        } else {
            (t.success_solid, "Connected", t.text_secondary)
        };
        let id = c.id.clone();
        let check = if self.cloud.checking.contains(&c.id) {
            self.button_disabled(
                &format!("cloud-check-{}", c.id),
                "Checking…",
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                "Checking the credentials with the provider",
                window,
                cx,
            )
        } else {
            let id = id.clone();
            self.button(
                &format!("cloud-check-{}", c.id),
                if failed { "Check again" } else { "Check" },
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.check_connection(id.clone(), cx)),
            )
            .tooltip(crate::controls::tip("Checks the credentials with the provider again", None))
        };
        let disconnect = self.button(
            &format!("cloud-disc-{}", c.id),
            "Disconnect",
            crate::controls::Btn::Secondary,
            None,
            window,
            cx,
            act(move |this, w, cx| this.ask_disconnect(id.clone(), w, cx)),
        );
        // A line about the connection under its row, at the row's full width: the last check's error, or what a
        // working connection still has against it (an unencrypted URL, a worker that cannot be installed). Not
        // beside the buttons: a wrapped line in the column that shrinks for them paints at the width it was first
        // measured at, under the buttons.
        let line = |mark: &'static str, mark_c: folo_ui::Hsla, ink: folo_ui::Hsla, words: &str| {
            div()
                .flex()
                .items_start()
                .gap(px(6.))
                .pt(px(4.))
                .child(div().pt(px(2.)).flex_none().child(icon(mark, 12., mark_c)))
                .child(rich(&t, ty::SM, ink, words).text_left().flex_1().min_w_0())
        };
        div()
            .flex()
            .flex_col()
            .justify_center()
            .px(px(16.))
            .min_h(px(56.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(24.))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(
                                        text(ty::BASE, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child(c.name.clone()),
                                    )
                                    .when(c.name != provider, |d| {
                                        d.child(text(ty::SM, t.text_tertiary).child(provider))
                                    })
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .gap(px(6.))
                                            .child(div().size(px(6.)).flex_none().rounded_full().bg(dot_c))
                                            .child(text(ty::SM, status_c).child(status)),
                                    ),
                            )
                            .when(!facts.is_empty(), |d| {
                                d.child(
                                    text(ty::SM, t.text_tertiary).tnum().ellipsis().child(facts.join(" · ")),
                                )
                            }),
                    )
                    .child(
                        div().flex().flex_none().items_center().gap(px(8.)).child(check).child(disconnect),
                    ),
            )
            .when_some(c.error.clone().filter(|_| failed), |d, e| {
                d.child(line("icons/circle-alert.svg", t.danger_solid, t.danger_text, &e))
            })
            .when_some(c.note.clone().filter(|_| !failed), |d, n| {
                d.child(line("icons/triangle-alert.svg", t.warning_solid, t.text_secondary, &n))
            })
            .into_any_element()
    }

    /// Settings › Cloud › Instances: every machine FOLO launched (alive first, then the last 7
    /// days'), with its run, cost so far and deadline, and Terminate for the alive ones. Hidden
    /// while there are none and no connection provisions any.
    pub(super) fn instances_panel(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let t = theme(cx);
        let provisions = |p: &str| {
            self.cloud.providers.ready().is_some_and(|ps| ps.iter().any(|x| x.id == p && x.provisions))
        };
        let any_provisioning =
            self.cloud.connections.ready().is_some_and(|l| l.iter().any(|c| provisions(&c.provider)));
        let instances = self.cloud.instances.ready().cloned();
        if instances.as_ref().is_some_and(|l| l.is_empty()) && !any_provisioning {
            return None;
        }
        // Nothing to say until there's something to show (connections still loading included).
        if self.cloud.instances.is_loading() && !any_provisioning {
            return None;
        }
        let now = self.store.api.now();
        let running = instances
            .as_ref()
            .map_or(0, |l| l.iter().filter(|i| i.status == crate::api::CloudInstanceStatus::Running).count());
        // An alive instance's deadline, "—" for the others.
        let deadline = |i: &crate::api::CloudInstance| {
            i.deadline
                .as_deref()
                .and_then(crate::api::time::parse)
                .filter(|_| i.status.is_alive())
                .map_or("—".to_string(), |d| crate::api::time::local_deadline(d, now))
        };
        // Column widths, measured: Stops by holds its longest reading ("2026-11-08 23:24" when a
        // deadline falls on another day). The GPU line takes what the others leave; when that is
        // less than its longest line, Started (the least needed) goes, as a whole column.
        let list = instances.as_deref().unwrap_or_default();
        let measure = |window: &mut Window, s: &str, size: f32| {
            f32::from(crate::charts::text_width(window, s, size)).ceil()
        };
        // Headings are medium weight: ~4 % wider than the regular face `text_width` measures.
        let heading_w = (measure(window, "Stops by", ty::SM.size) * 1.04).ceil();
        let stops_w =
            list.iter().map(|i| measure(window, &deadline(i), ty::SM.size)).fold(heading_w, f32::max);
        let gpu_need =
            list.iter().map(|i| measure(window, &offer_line(&i.offer), ty::SM.size)).fold(0., f32::max);
        const GAP: f32 = 12.;
        // The panel's border and its rows' padding; Run, Status, Cost and the action, each with a gap.
        let fixed = 2. + 32. + (128. + 100. + 64. + 88. + 4. * GAP) + stops_w + GAP;
        let table_w = self.content_width(window).min(COLUMN_MAX);
        let show_started = table_w - fixed - (72. + GAP) >= gpu_need;
        let cell =
            |w: f32, right: bool, el: folo_ui::Div| el.w(px(w)).flex_none().when(right, |d| d.text_right());
        let cols = |d: folo_ui::Div| d.flex().items_center().gap(px(GAP)).px(px(16.)).h(px(36.));
        let head = cols(div())
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "GPU").flex_1().min_w_0())
            .child(cell(128., false, th(&t, "Run")))
            .child(cell(100., false, th(&t, "Status")))
            .when(show_started, |d| d.child(cell(72., true, th(&t, "Started"))))
            .child(cell(64., true, th(&t, "Cost")))
            .child(cell(stops_w, true, th(&t, "Stops by")))
            .child(cell(88., false, div()));
        let body = match &self.cloud.instances {
            Load::Failed(e) => div()
                .p(px(16.))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(
                    text(ty::SM, t.danger_text)
                        .flex_1()
                        .min_w_0()
                        .child(format!("Couldn't load the instances: {}", e.message)),
                )
                .child(self.button(
                    "cloud-inst-retry",
                    "Retry",
                    crate::controls::Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.refresh_instances(cx)),
                ))
                .into_any_element(),
            Load::Ready(list) if list.is_empty() => div()
                .p(px(16.))
                .child(text(ty::SM, t.text_tertiary).child(
                    "No instances. FOLO starts one per cloud run and terminates it when the run ends.",
                ))
                .into_any_element(),
            Load::Ready(list) => {
                let list = list.clone();
                let rows: Vec<AnyElement> = list
                    .iter()
                    .map(|i| {
                        let live = i.status.is_alive();
                        let ink = if live { t.text_primary } else { t.text_tertiary };
                        let quiet = if live { t.text_secondary } else { t.text_tertiary };
                        let line = offer_line(&i.offer);
                        let run = i.run_id.as_ref().map(|r| {
                            (r.clone(), self.store.run(r).map_or(r.clone(), |run| run.name.clone()))
                        });
                        let run_cell = match run {
                            Some((rid, name)) => {
                                let el = div()
                                    .id(SharedString::from(format!("cloud-run-{}", i.id)))
                                    .rounded(px(radius::SM))
                                    .cursor_pointer()
                                    .hover(|s| s.bg(t.bg_hover))
                                    .child(
                                        text(ty::SM, if live { t.text_secondary } else { t.text_tertiary })
                                            .ellipsis()
                                            .child(name),
                                    );
                                self.bind(
                                    el,
                                    &format!("cloud-run-{}", i.id),
                                    radius::SM,
                                    window,
                                    cx,
                                    act(move |this, w, cx| this.navigate(Route::Run(rid.clone()), w, cx)),
                                )
                                .into_any_element()
                            }
                            None => text(ty::SM, t.text_tertiary).child("—").into_any_element(),
                        };
                        let started = crate::api::time::parse(&i.created)
                            .map_or(String::new(), |c| crate::api::time::ago(c, now));
                        // Every instance that may still cost money can be terminated (again, for
                        // one whose termination isn't confirmed: it is idempotent).
                        let action = if !live {
                            div().into_any_element()
                        } else {
                            let id = i.id.clone();
                            self.button(
                                &format!("cloud-term-{}", i.id),
                                "Terminate",
                                crate::controls::Btn::Secondary,
                                None,
                                window,
                                cx,
                                act(move |this, w, cx| this.ask_terminate(id.clone(), w, cx)),
                            )
                            .into_any_element()
                        };
                        cols(div())
                            .id(SharedString::from(format!("cloud-inst-{}", i.id)))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .tooltip(crate::controls::tip(format!("{line} · {}", i.name), None))
                            .child(text(ty::SM, ink).flex_1().min_w_0().ellipsis().child(line))
                            .child(cell(128., false, div().child(run_cell)))
                            .child(cell(
                                100.,
                                false,
                                // The one status mark of every list; a newer server's status is
                                // counted alive (it keeps its Terminate) and waits like a launch.
                                status_cell(
                                    &t,
                                    StatusKind::of_instance(i.status),
                                    i.status.label(),
                                    format!("inst-{}", i.id),
                                ),
                            ))
                            .when(show_started, |d| {
                                d.child(cell(
                                    72.,
                                    true,
                                    text(ty::SM, quiet).tnum().whitespace_nowrap().child(started),
                                ))
                            })
                            .child(cell(
                                64.,
                                true,
                                text(ty::SM, ink).tnum().child(i.cost_usd.map_or("—".to_string(), fmt_cost)),
                            ))
                            .child(cell(
                                stops_w,
                                true,
                                text(ty::SM, quiet).tnum().whitespace_nowrap().child(deadline(i)),
                            ))
                            .child(cell(88., false, div().flex().justify_end().child(action)))
                            .into_any_element()
                    })
                    .collect();
                // A termination the provider hasn't confirmed is a problem, not "done": it leads.
                let stuck: Vec<AnyElement> = list
                    .iter()
                    .filter(|i| i.status == crate::api::CloudInstanceStatus::Terminating && i.error.is_some())
                    .map(|i| {
                        let provider = self.provider_name(&i.provider);
                        // How to find it there.
                        let reference = i
                            .provider_ref
                            .as_ref()
                            .map_or(String::new(), |r| format!(" Its {provider} id is `{r}`."));
                        div()
                            .m(px(12.))
                            .mb(px(4.))
                            .p(px(10.))
                            .flex()
                            .items_start()
                            .gap(px(8.))
                            .rounded(px(radius::MD))
                            .bg(t.warning_muted)
                            .child(icon("icons/triangle-alert.svg", 14., t.warning_solid))
                            .child(
                                rich(
                                    &t,
                                    ty::SM,
                                    t.text_primary,
                                    &format!(
                                        "FOLO could not confirm that `{}` is terminated: {} It keeps trying. Check the provider's console; the instance may still be billed.{reference}",
                                        i.name,
                                        i.error.clone().unwrap_or_default().trim_end_matches('.').to_string() + ".",
                                    ),
                                )
                                .text_left()
                                .flex_1()
                                .min_w_0(),
                            )
                            .into_any_element()
                    })
                    .collect();
                div().flex().flex_col().children(stuck).child(head).children(rows).into_any_element()
            }
            _ => loading_rows(&t, "cloud-inst-sk", 2).into_any_element(),
        };
        Some(
            panel(&t)
                .flex()
                .flex_col()
                .overflow_hidden()
                .child(panel_header(&t, "Instances").when(running > 0, |d| {
                    d.child(
                        text(ty::SM, t.text_tertiary)
                            .child(format!("{running} running · refreshes every 5 s")),
                    )
                }))
                .child(body)
                .into_any_element(),
        )
    }
}
