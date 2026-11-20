//! A run's Config panel (one column of its settings) and its `folo train` command.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Label, Run, RunExt};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    /// Fetches the run's `folo train` command and puts it on the clipboard.
    fn copy_run_cli(&mut self, id: String, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            move |api| api.run_cli(&id),
            |this, r, cx| match r {
                Ok(c) => {
                    cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(c.cli.clone()));
                    this.toast(
                        Tone::Neutral,
                        "Copied the training command",
                        Some("Paste it in a terminal on the server's machine.".into()),
                        cx,
                    );
                }
                Err(e) => this.toast(Tone::Danger, "Couldn't get the command", Some(e.message.into()), cx),
            },
        );
    }

    pub(super) fn config_panel(
        &self,
        r: &Run,
        ds_name: &str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let c = &r.config;
        let a = &c.augment;
        // A rate's power of ten in a legible raised run (`sci`).
        let mono_v = |s: String| crate::ui::sci(ty::SM, t.text_primary, s);
        let col1 = div()
            .flex()
            .flex_col()
            .child(kv(&t, "Dataset", mono_v(ds_name.to_string())))
            .child(kv(&t, "Architecture", mono_v(c.arch.as_str().to_string())))
            // The start model by name (the config holds its id).
            .child(kv(
                &t,
                "Pretrained weights",
                mono_v(c.weights.as_deref().map_or("none".into(), |w| {
                    self.store
                        .models
                        .ready()
                        .and_then(|ms| ms.iter().find(|m| m.id == w))
                        .map_or(w.to_string(), |m| m.name.clone())
                })),
            ))
            .child(kv(
                &t,
                "Head",
                mono_v(if c.head == crate::api::Head::O2o {
                    "end-to-end (no NMS)".into()
                } else {
                    "one-to-many + NMS".into()
                }),
            ))
            .child(kv(&t, "Seed", mono_v(c.seed.to_string())))
            .child(kv(&t, "Device", mono_v(c.device.clone())));
        let col2 = div()
            .flex()
            .flex_col()
            .child(kv(&t, "Epochs", mono_v(c.epochs.to_string())))
            .child(kv(&t, "Batch", mono_v(c.batch.to_string())))
            .child(kv(&t, "Image size", mono_v(format!("{} px", c.imgsz))))
            .child(kv(&t, "Precision", mono_v(if c.amp { "mixed (f16)".into() } else { "f32".into() })))
            // What the run trains with (`Run.resolved`): Auto's pick and its rate, not "auto · 0.01".
            .child(kv(
                &t,
                "Optimizer",
                mono_v(match &r.resolved {
                    Some(s) => {
                        let opt = match s.optimizer {
                            crate::api::ResolvedOptimizer::Sgd => "SGD",
                            crate::api::ResolvedOptimizer::Adamw => "AdamW",
                            crate::api::ResolvedOptimizer::Musgd => "MuSGD",
                            crate::api::ResolvedOptimizer::Unknown => "Unknown",
                        };
                        let how = if c.optimizer == crate::api::Optimizer::Auto { " (auto)" } else { "" };
                        format!("{opt}{how} · lr0 {}", crate::ui::fmt_lr(s.lr0))
                    }
                    None => format!("{} · lr0 {}", c.optimizer.label(), crate::ui::fmt_lr(c.lr0 as f64)),
                }),
            ))
            // The EMA's window (`Run.resolved.ema_decay`): none for a run without one, or from a
            // server older than the field.
            .when_some(crate::ui::fmt_ema_window(r.resolved.as_ref().and_then(|s| s.ema_decay)), |d, w| {
                d.child(kv(&t, "EMA", mono_v(w)))
            });
        // Augmentation, one setting per row, under the training settings.
        let col3 = div()
            .flex()
            .flex_col()
            .child(kv(
                &t,
                "Mosaic",
                mono_v(if a.mosaic <= 0. { "off".to_string() } else { crate::ui::fmt_aug(a.mosaic) }),
            ))
            .child(kv(
                &t,
                "Close mosaic",
                mono_v({
                    let n = r.close_mosaic_epochs();
                    if a.mosaic <= 0. {
                        "–".to_string()
                    } else if n < a.close_mosaic {
                        format!("last {n} epochs ({} capped at half)", a.close_mosaic)
                    } else {
                        format!("last {n} epochs")
                    }
                }),
            ))
            .child(kv(&t, "Mixup", mono_v(crate::ui::fmt_aug(a.mixup))))
            .child(kv(&t, "Horizontal flip", mono_v(crate::ui::fmt_aug(a.fliplr))))
            .child(kv(
                &t,
                "HSV (h / s / v)",
                // The hue's strength needs three decimals; saturation and value two.
                mono_v(format!("{:.3} / {:.2} / {:.2}", a.hsv[0], a.hsv[1], a.hsv[2])),
            ))
            .child(kv(&t, "Scale", mono_v(crate::ui::fmt_scale(a.scale))));
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(
                panel_header(&t, "Config").child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(12.))
                        .child(
                            text(ty::SM, t.text_tertiary)
                                .font_family(crate::theme::MONO)
                                .min_w_0()
                                .ellipsis()
                                .child(r.id.clone()),
                        )
                        // The `folo train` command for this config on the server's machine (servers
                        // with `run_cli`): train it again from a terminal.
                        .when(self.store.features.iter().any(|f| f == "run_cli"), |d| {
                            let rid = r.id.clone();
                            d.child(
                                self.button(
                                    "run-copy-cli",
                                    "Copy command",
                                    Btn::Secondary,
                                    Some("icons/copy.svg"),
                                    window,
                                    cx,
                                    act(move |this, _, cx| this.copy_run_cli(rid.clone(), cx)),
                                )
                                .tooltip(crate::controls::tip(
                                    "The folo train command for this run's settings, for a terminal on the server's machine",
                                    None,
                                )),
                            )
                        }),
                ),
            )
            .child(div().flex().flex_col().px(px(16.)).py(px(8.)).child(col1).child(col2).child(col3))
            .into_any_element()
    }
}
