//! The New run dialog's Compute rows (servers with `cloud`): this machine or a cloud connection,
//! its GPU offers, and Stop after.

use std::rc::Rc;

use gpui::Focusable;
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString, Styled,
    Window, div, px,
};

use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// New run's Run on (servers with `cloud`): this machine or a cloud connection; for a
    /// connection, its GPU offers and Stop after, the most the run can cost said beside it. With no
    /// connection, the way to make one.
    pub(super) fn nr_compute_rows(
        &mut self,
        cloud: Option<&crate::api::CloudConnection>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        use crate::cloud::ComputeField;
        let t = theme(cx);
        let Some(f) = self.new_run.as_ref() else { return div().into_any_element() };
        let (offer_id, hours, hours_input, hours_bad, compute_error) =
            (f.offer.clone(), f.max_hours, f.hours_input.clone(), f.hours_invalid, f.compute_error.clone());
        let error_on =
            |c: ComputeField| compute_error.as_ref().filter(|(k, _)| *k == c).map(|(_, m)| m.clone());
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        let danger = |s: String| text(ty::SM, t.danger_text).whitespace_normal().child(s);
        let quiet = |s: String| text(ty::SM, t.text_tertiary).whitespace_normal().child(s);
        let connections = self.cloud.connections.ready().cloned().unwrap_or_default();
        let here = self
            .store
            .system
            .ready()
            .and_then(|s| s.gpus.first())
            .map_or("this PC".to_string(), |g| crate::ui::short_gpu(&g.name));
        let mut options = vec![(SharedString::from("This machine"), Some(SharedString::from(here)))];
        let mut disabled: Vec<Option<SharedString>> = vec![None];
        for c in &connections {
            let failed = c.status == crate::api::CloudConnectionStatus::Error;
            options.push((
                c.name.clone().into(),
                Some(if failed {
                    "check failed".into()
                } else {
                    c.account.clone().unwrap_or_default().into()
                }),
            ));
            disabled.push(
                failed.then(|| c.error.clone().unwrap_or_else(|| "Its last check failed".into()).into()),
            );
        }
        let selected = cloud.and_then(|c| connections.iter().position(|x| x.id == c.id)).map_or(0, |i| i + 1);
        let ids: Vec<String> = connections.iter().map(|c| c.id.clone()).collect();
        let run_on = self.select_with(
            "nr-run-on",
            cloud.map_or("This machine".into(), |c| c.name.clone().into()),
            options,
            selected,
            &disabled,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                if let Some(f) = this.new_run.as_mut() {
                    f.run_on = i.checked_sub(1).and_then(|i| ids.get(i).cloned());
                    f.offer = None;
                    f.want_offer = None;
                    f.compute_error = None;
                }
                this.sync_compute(cx);
            }),
        );
        let provisions = cloud.is_some_and(|c| {
            self.cloud
                .providers
                .ready()
                .is_some_and(|ps| ps.iter().any(|p| p.id == c.provider && p.provisions))
        });
        let run_on_help = match cloud {
            None => "Trains here, on the device chosen under Model".to_string(),
            Some(c) if provisions => {
                format!("A GPU rented from {} for this run, by the hour", self.provider_name(&c.provider))
            }
            Some(_) => "That server trains it, in its own queue".to_string(),
        };
        let left = div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(label("Run on"))
            .child(run_on)
            .child(match error_on(ComputeField::RunOn) {
                Some(e) => danger(e),
                None => quiet(run_on_help),
            });
        let Some(conn) = cloud else {
            // No connection yet: the way to one, in Settings › Cloud (the form waits as a draft).
            let right = self.cloud.connections.ready().is_some_and(|l| l.is_empty()).then(|| {
                let link = div()
                    .id("nr-cloud-settings")
                    .mt(px(24.))
                    .px(px(4.))
                    .mx(px(-4.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(4.))
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .whitespace_normal()
                                    .child("Connect a cloud in Settings to train on a rented GPU."),
                            )
                            .child(icon("icons/chevron-right.svg", 12., t.text_secondary)),
                    );
                self.bind(
                    link,
                    "nr-cloud-settings",
                    radius::SM,
                    window,
                    cx,
                    act(|this, w, cx| {
                        this.close_new_run(w, cx);
                        this.navigate(Route::Settings, w, cx);
                    }),
                )
            });
            return div()
                .flex()
                .gap(px(16.))
                .child(left)
                .child(div().flex_1().min_w_0().children(right))
                .into_any_element();
        };
        // The GPU: the connection's offers, each with its price; none in stock can't be picked.
        let offers = self.cloud.offers.get(&conn.id).cloned().unwrap_or(Load::Loading);
        let offer =
            offers.ready().and_then(|l| l.offers.iter().find(|o| Some(&o.id) == offer_id.as_ref()).cloned());
        let gpu: AnyElement = match &offers {
            Load::Ready(list) => {
                let shown = crate::cloud::menu_order(&list.offers);
                let opts = shown
                    .iter()
                    .map(|o| {
                        let price = fmt_price(o.price_per_hour);
                        let detail = match o.availability {
                            crate::api::CloudAvailability::None => format!("none available now · {price}"),
                            crate::api::CloudAvailability::Low => format!("low stock · {price}"),
                            _ => price,
                        };
                        (SharedString::from(offer_line(o)), Some(SharedString::from(detail)))
                    })
                    .collect();
                let off: Vec<Option<SharedString>> = shown
                    .iter()
                    .map(|o| {
                        (o.availability == crate::api::CloudAvailability::None)
                            .then(|| "None available now: launching would be refused".into())
                    })
                    .collect();
                let picked =
                    offer.as_ref().and_then(|o| shown.iter().position(|x| x.id == o.id)).unwrap_or(0);
                let ids: Vec<String> = shown.iter().map(|o| o.id.clone()).collect();
                self.select_with(
                    "nr-gpu",
                    offer.as_ref().map_or("Choose a GPU".into(), |o| {
                        format!("{} · {}", offer_line(o), fmt_price(o.price_per_hour)).into()
                    }),
                    opts,
                    picked,
                    &off,
                    window,
                    cx,
                    Rc::new(move |this, i, _, _| {
                        if let Some(f) = this.new_run.as_mut() {
                            f.offer = ids.get(i).cloned();
                            f.compute_error.take_if(|(c, _)| *c == ComputeField::Gpu);
                        }
                    }),
                )
            }
            Load::Failed(e) => div()
                .h(px(crate::theme::size::CONTROL_LG))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(8.))
                .child(text(ty::SM, t.danger_text).flex_1().min_w_0().ellipsis().child(e.message.clone()))
                .child({
                    let id = conn.id.clone();
                    self.button(
                        "nr-offers-retry",
                        "Retry",
                        Btn::Secondary,
                        Some("icons/refresh-cw.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.ensure_offers(&id, true, cx)),
                    )
                })
                .into_any_element(),
            // Shaped like the select it becomes, inert.
            _ => div()
                .h(px(crate::theme::size::CONTROL_LG))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_inset)
                .border_1()
                .border_color(t.line_default)
                .child(text(ty::BASE, t.text_tertiary).child("Loading offers…"))
                .into_any_element(),
        };
        let gpu_help = match (error_on(ComputeField::Gpu), &offer) {
            (Some(e), _) => danger(e),
            (None, Some(o)) if o.price_per_hour.is_none() => quiet("Not billed: a machine of yours".into()),
            (None, Some(o)) if o.availability == crate::api::CloudAvailability::Low => {
                quiet("Low stock: the provider may refuse to start it".into())
            }
            _ => quiet("Billed by the hour while its instance exists".into()),
        };
        let right = div()
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .gap(px(8.))
            .child(label("GPU"))
            .child(gpu)
            .child(gpu_help);
        // Stop after: the run's time limit, and so the most it can cost.
        let fh = hours_input.read(cx).focus_handle(cx);
        let ring = self.ring(&fh, window);
        let focused = fh.is_focused(window);
        let bad = hours_bad || error_on(ComputeField::StopAfter).is_some();
        let hours_box = div()
            .relative()
            .w(px(84.))
            .h(px(crate::theme::size::CONTROL_MD))
            .px(px(8.))
            .flex()
            .flex_none()
            .items_center()
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if bad {
                t.danger_solid
            } else if focused {
                t.accent_line
            } else {
                t.line_control
            })
            .text_size(px(ty::SM.size))
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .child(hours_input);
        let price = offer.as_ref().and_then(|o| o.price_per_hour);
        let limit = match (provisions, price) {
            (true, Some(p)) => format!(
                "The run is stopped and the instance terminated after this long at the latest. It cannot cost more than {}.",
                fmt_cost(p * hours as f64)
            ),
            (true, None) => {
                "The run is stopped and the instance terminated after this long at the latest.".to_string()
            }
            (false, _) => "The run is stopped after this long at the latest.".to_string(),
        };
        let stop_after = div().flex().flex_col().gap(px(8.)).child(label("Stop after")).child(
            div()
                .flex()
                .items_center()
                .gap(px(12.))
                .child(
                    div()
                        .flex()
                        .flex_none()
                        .items_center()
                        .gap(px(6.))
                        .child(hours_box)
                        .child(text(ty::SM, t.text_secondary).child("hours")),
                )
                .child(match error_on(ComputeField::StopAfter) {
                    Some(e) => danger(e).flex_1().min_w_0(),
                    None if hours_bad => danger("Enter hours from 0.25 to 720".into()).flex_1().min_w_0(),
                    None => quiet(limit).flex_1().min_w_0(),
                }),
        );
        div()
            .flex()
            .flex_col()
            .gap(px(14.))
            .child(div().flex().gap(px(16.)).child(left).child(right))
            .child(stop_after)
            .into_any_element()
    }
}
