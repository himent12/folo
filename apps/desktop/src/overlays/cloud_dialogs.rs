//! The cloud dialogs: connect a cloud, disconnect a connection, terminate an instance.

use std::rc::Rc;

use folo_ui::Focusable;
use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

use super::{dialog_footer, error_note};

impl FoloApp {
    /// Connect a cloud: the provider, its fields as the server lists them (secrets masked), and an
    /// optional name. The server checks the values with the provider before it keeps them; what it
    /// refuses shows on the field it is about.
    pub(super) fn connect_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let providers = self.cloud.providers.ready().cloned().unwrap_or_default();
        let Some(f) = self.cloud.connect.as_ref() else { return div().into_any_element() };
        let (chosen, inputs, name, submitting, error) =
            (f.provider, f.fields.clone(), f.name.clone(), f.submitting, f.error.clone());
        let Some(p) = providers.get(chosen).cloned() else { return div().into_any_element() };
        let names: Vec<&str> = providers.iter().map(|p| p.name.as_str()).collect();
        let provider_disabled: Vec<Option<folo_ui::SharedString>> = providers
            .iter()
            .enumerate()
            .map(|(i, _)| {
                (submitting && i != chosen)
                    .then(|| folo_ui::SharedString::from("Wait for the current check to finish"))
            })
            .collect();
        let provider_seg = self.segmented_fill_with(
            "cc-provider",
            &names,
            chosen,
            &provider_disabled,
            window,
            cx,
            Rc::new(|this, i, _, cx| this.set_connect_provider(i, cx)),
        );
        let label =
            |s: &str| text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(s.to_string());
        // A field's box; Enter moves to the next field, and in the last one connects.
        let last = inputs.len();
        let input_box = |this: &Self,
                         input: folo_ui::Entity<crate::input::TextInput>,
                         index: usize,
                         bad: bool,
                         mono: bool,
                         window: &Window,
                         cx: &mut Context<Self>| {
            let fh = input.read(cx).focus_handle(cx);
            let ring = this.ring(&fh, window);
            let focused = fh.is_focused(window);
            div()
                .relative()
                .w_full()
                .h(px(32.))
                .px(px(10.))
                .flex()
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
                .text_size(px(ty::BASE.size))
                .when(mono, |d| d.font_family(crate::theme::MONO).text_size(px(ty::SM.size)))
                .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, w, cx| {
                    if ev.keystroke.key != "enter" || ev.keystroke.modifiers.control {
                        return;
                    }
                    let next = this
                        .cloud
                        .connect
                        .as_ref()
                        .and_then(|f| f.fields.get(index + 1).or((index + 1 == last).then_some(&f.name)));
                    match next {
                        Some(next) => w.focus(&next.read(cx).focus_handle(cx)),
                        None => this.submit_connect(w, cx),
                    }
                    cx.stop_propagation();
                }))
                .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(input)
                .into_any_element()
        };
        let field_error = |i: usize| match &error {
            Some(crate::cloud::ConnectError::Field(j, m)) if *j == i => Some(m.clone()),
            _ => None,
        };
        let first_secret = p.fields.iter().position(|f| f.secret);
        let mut rows: Vec<AnyElement> = Vec::new();
        for (i, (field, input)) in p.fields.iter().zip(inputs).enumerate() {
            let err = field_error(i);
            // Keys and addresses in Geist Mono, where every character counts.
            let mono = field.secret || field.key.ends_with("url");
            let control = input_box(self, input, i, err.is_some(), mono, window, cx);
            // Where the provider's console makes the credential, beside the first secret's help.
            let docs = p.docs_url.clone().filter(|_| Some(i) == first_secret).map(|url| {
                let link = div()
                    .id("cc-docs")
                    .flex_none()
                    .px(px(4.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(4.))
                            .child(
                                text(ty::SM, t.text_secondary).whitespace_nowrap().child("Where to find it"),
                            )
                            .child(icon("icons/external-link.svg", 12., t.text_secondary)),
                    );
                self.bind(link, "cc-docs", radius::SM, window, cx, act(move |_, _, cx| cx.open_url(&url)))
                    .tooltip(crate::controls::tip(p.docs_url.clone().unwrap_or_default(), None))
            });
            rows.push(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(label(&field.label))
                            .when(!field.required, |d| {
                                d.child(text(ty::SM, t.text_tertiary).child("Optional"))
                            }),
                    )
                    .child(control)
                    .when_some(err, |d, e| d.child(text(ty::SM, t.danger_text).whitespace_normal().child(e)))
                    .when(field.help.is_some() || docs.is_some(), |d| {
                        d.child(
                            div()
                                .flex()
                                .items_start()
                                .justify_between()
                                .gap(px(12.))
                                // Providers write commands and flags as `code`.
                                .child(
                                    rich(
                                        &t,
                                        ty::SM,
                                        t.text_tertiary,
                                        field.help.as_deref().unwrap_or_default(),
                                    )
                                    .text_left()
                                    .flex_1()
                                    .min_w_0(),
                                )
                                .children(docs),
                        )
                    })
                    .into_any_element(),
            );
        }
        let name_box = input_box(self, name, last, false, false, window, cx);
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(16.))
            .when_some(
                match &error {
                    Some(crate::cloud::ConnectError::Dialog(e)) => Some(e.clone()),
                    _ => None,
                },
                |d, e| d.child(error_note(&t, e.message, e.hint)),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(label("Provider"))
                    .child(provider_seg)
                    .when(!p.about.is_empty(), |d| {
                        d.child(rich(&t, ty::SM, t.text_tertiary, &p.about).text_left())
                    }),
            )
            .children(rows)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(label("Name"))
                            .child(text(ty::SM, t.text_tertiary).child("Optional")),
                    )
                    .child(name_box),
            );
        let cancel = self.button(
            "cc-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_connect(w, cx)),
        );
        let go = if submitting {
            self.button_disabled(
                "cc-go",
                "Connecting…",
                Btn::Primary,
                Some("icons/plug.svg"),
                "Checking the credentials with the provider…",
                window,
                cx,
            )
        } else {
            self.button(
                "cc-go",
                "Connect",
                Btn::Primary,
                Some("icons/plug.svg"),
                window,
                cx,
                act(|this, w, cx| this.submit_connect(w, cx)),
            )
        };
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "cloud-connect",
            520.,
            "Connect a cloud",
            Some("FOLO checks these with the provider, then keeps them on the server.".into()),
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_connect(w, cx)),
        )
    }

    /// Disconnect a connection: the server forgets its credentials; runs and models stay. A
    /// connection still in use (`409 connection_in_use`) says so here.
    pub(super) fn disconnect_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(c) = self.cloud.disconnect.as_ref() else { return div().into_any_element() };
        let (busy, error) = (c.busy, c.error.clone());
        let conn = self.connection(&c.id).cloned();
        let name = conn.as_ref().map_or(c.id.clone(), |c| c.name.clone());
        // What it forgets, as the provider calls it ("the API key", "the token").
        let secret = conn
            .as_ref()
            .and_then(|c| self.cloud.providers.ready()?.iter().find(|p| p.id == c.provider))
            .and_then(|p| p.fields.iter().find(|f| f.secret))
            .map_or("credentials".to_string(), |f| crate::cloud::in_sentence(&f.label));
        let cancel = self.button(
            "cd-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        );
        let go = if busy {
            self.button_disabled(
                "cd-confirm",
                "Disconnecting…",
                Btn::DangerCommit,
                None,
                "Asking the server…",
                window,
                cx,
            )
        } else {
            self.button(
                "cd-confirm",
                "Disconnect",
                Btn::DangerCommit,
                None,
                window,
                cx,
                act(|this, _, cx| this.disconnect(cx)),
            )
        };
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(12.))
            .child(
                text(ty::BASE, t.text_secondary)
                    .whitespace_normal()
                    .child(format!("FOLO forgets the {secret}. Runs and models stay.")),
            )
            .when_some(error, |d, e| d.child(error_note(&t, e.message, e.hint)));
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "cloud-disconnect",
            440.,
            &format!("Disconnect {name}?"),
            None,
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        )
    }

    /// Terminate an instance by hand: it is shut down now and its run fails, keeping the
    /// checkpoints already copied here.
    pub(super) fn terminate_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(c) = self.cloud.terminate.as_ref() else { return div().into_any_element() };
        let (busy, error) = (c.busy, c.error.clone());
        let inst = self.cloud.instances.ready().and_then(|l| l.iter().find(|i| i.id == c.id)).cloned();
        let what = inst.as_ref().map_or("instance".to_string(), |i| {
            format!("{} at {}", i.offer.gpu, self.provider_name(&i.provider))
        });
        let run = inst
            .as_ref()
            .and_then(|i| i.run_id.as_deref())
            .map(|r| self.store.run(r).map_or(r.to_string(), |run| run.name.clone()));
        let sentence = match run {
            Some(r) => format!(
                "The {what} is shut down now. Its run `{r}` fails; the checkpoints already copied to this machine stay."
            ),
            None => format!("The {what} is shut down now."),
        };
        let cancel = self.button(
            "ct-cancel",
            "Cancel",
            Btn::Secondary,
            None,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        );
        let go = if busy {
            self.button_disabled(
                "ct-confirm",
                "Terminating…",
                Btn::DangerCommit,
                Some("icons/stop.svg"),
                "Asking the provider…",
                window,
                cx,
            )
        } else {
            self.button(
                "ct-confirm",
                "Terminate",
                Btn::DangerCommit,
                Some("icons/stop.svg"),
                window,
                cx,
                act(|this, _, cx| this.terminate(cx)),
            )
        };
        let body = div()
            .px(px(20.))
            .pb(px(20.))
            .flex()
            .flex_col()
            .gap(px(12.))
            .child(rich(&t, ty::BASE, t.text_secondary, &sentence).text_left())
            .when_some(error, |d, e| d.child(error_note(&t, e.message, e.hint)));
        let footer = dialog_footer(&t).justify_end().child(cancel).child(go);
        self.dialog_frame(
            &t,
            "cloud-terminate",
            440.,
            "Terminate this instance?",
            None,
            body,
            footer,
            window,
            cx,
            act(|this, w, cx| this.close_confirm(w, cx)),
        )
    }
}
