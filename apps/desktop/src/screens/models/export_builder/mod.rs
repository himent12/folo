//! Interactive Export & Inference CLI Command Builder UI panel on the Model Detail screen.

pub(crate) mod command;

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Head, Model};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

pub use command::{
    CliExportFormat, MODEL_CLI_CONF, MODEL_CLI_IOU, ModelCliMode, ModelCliState, PredictPrecision,
    QuantMethod, ServerExportSupport, build_model_cli, check_server_export_support, predict_uses_iou,
};

impl FoloApp {
    /// Interactive Export & Inference CLI Command Builder panel on the Model Detail screen.
    pub(super) fn export_cli_panel(
        &mut self,
        m: &Model,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let run_ds_path = m
            .run_id
            .as_deref()
            .and_then(|rid| self.store.run(rid))
            .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
            .map(|d| d.path.clone());
        let built = build_model_cli(m, run_ds_path.as_deref(), &self.model_cli);
        let mode_idx = usize::from(self.model_cli.mode == ModelCliMode::Predict);
        let mode_seg = self.segmented(
            "model-cli-mode",
            &["Export", "Predict"],
            mode_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.model_cli.mode = if i == 0 { ModelCliMode::Export } else { ModelCliMode::Predict };
                cx.notify();
            }),
        );

        let controls = match self.model_cli.mode {
            ModelCliMode::Export => self.export_cli_export_controls(&t, m, window, cx),
            ModelCliMode::Predict => self.export_cli_predict_controls(&t, m, window, cx),
        };

        let server_support = built
            .server_export
            .map(|opts| check_server_export_support(opts, &self.store.features, self.store.api.is_remote()));
        let can_server_export = matches!(server_support, Some(ServerExportSupport::Supported(_)));
        let server_unsupported_reason = match &server_support {
            Some(ServerExportSupport::Unsupported(msg)) => Some(*msg),
            _ => None,
        };

        let copy_cmd = built.command.clone();
        let copy_note = built.quote_note.clone();
        let copy_title = match self.model_cli.mode {
            ModelCliMode::Export if self.model_cli.export_format == CliExportFormat::Int8Quantize => {
                "Copied quantize command"
            }
            ModelCliMode::Export => "Copied export command",
            ModelCliMode::Predict => "Copied predict command",
        };
        let copy_btn = self.button(
            "model-cli-copy",
            "Copy command",
            if can_server_export { Btn::Secondary } else { Btn::Primary },
            Some("icons/copy.svg"),
            window,
            cx,
            act(move |this, _, cx| {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_cmd.clone()));
                this.toast(Tone::Success, copy_title, copy_note.clone().map(Into::into), cx);
            }),
        );

        let is_busy_export = match &server_support {
            Some(ServerExportSupport::Supported(opts)) => {
                self.exporting.contains(&(m.id.clone(), opts.format))
            }
            _ => false,
        };
        let disabled_shown =
            is_busy_export || matches!(server_support, Some(ServerExportSupport::Unsupported(_)));
        let disabled_anchor = self
            .passing_stop("model-cli-server-export-disabled", disabled_shown, cx)
            .filter(|_| !disabled_shown);
        let server_btn = match server_support {
            Some(ServerExportSupport::Supported(opts)) => {
                let mid = m.id.clone();
                if is_busy_export {
                    Some(
                        self.button_disabled(
                            "model-cli-server-export",
                            "Exporting…",
                            Btn::Primary,
                            Some("icons/download.svg"),
                            "An export in this format is already running",
                            window,
                            cx,
                        )
                        .into_any_element(),
                    )
                } else {
                    let label = if opts.format == "onnx" { "Export ONNX now" } else { "Export f16 now" };
                    Some(
                        self.button(
                            "model-cli-server-export",
                            label,
                            Btn::Primary,
                            Some("icons/download.svg"),
                            window,
                            cx,
                            act(move |this, w, cx| {
                                this.export_model_with_options(mid.clone(), opts, cx);
                                if this.exporting.contains(&(mid.clone(), opts.format)) {
                                    // Keep keyboard focus mounted on the disabled trigger while exporting.
                                    let fh = this.fh("model-cli-server-export-disabled", cx);
                                    w.focus(&fh);
                                }
                            }),
                        )
                        .into_any_element(),
                    )
                }
            }
            Some(ServerExportSupport::Unsupported(reason)) => Some(
                self.button_disabled(
                    "model-cli-server-export",
                    "Export ONNX now",
                    Btn::Secondary,
                    Some("icons/download.svg"),
                    reason,
                    window,
                    cx,
                )
                .into_any_element(),
            ),
            None => None,
        };

        let header_badge = match self.model_cli.mode {
            ModelCliMode::Export if self.model_cli.export_format == CliExportFormat::Onnx => {
                Some(badge(&t, "opset 17", Tone::Neutral))
            }
            ModelCliMode::Export if self.model_cli.export_format == CliExportFormat::Int8Quantize => {
                Some(badge(&t, "CPU int8", Tone::Neutral))
            }
            ModelCliMode::Predict if self.model_cli.predict_sahi => Some(badge(&t, "SAHI", Tone::Accent)),
            ModelCliMode::Predict if self.model_cli.predict_tta => Some(badge(&t, "TTA", Tone::Accent)),
            _ => None,
        };

        panel(&t)
            .flex()
            .flex_col()
            .child(
                panel_header(&t, "Export & CLI")
                    .child(div().flex().items_center().gap(px(8.)).children(header_badge).child(mode_seg)),
            )
            .child(
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .gap(px(12.))
                    .child(controls)
                    .child(
                        div()
                            .px(px(12.))
                            .py(px(10.))
                            .rounded(px(radius::MD))
                            .bg(t.bg_inset)
                            .border_1()
                            .border_color(t.line_subtle)
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_family(crate::theme::MONO)
                                    .whitespace_normal()
                                    .child(built.command),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .justify_between()
                            .gap(px(12.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(200.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(2.))
                                    .when_some(built.hint, |d, h| {
                                        d.child(text(ty::SM, t.text_tertiary).whitespace_normal().child(h))
                                    })
                                    .when_some(server_unsupported_reason, |d, reason| {
                                        d.child(
                                            text(ty::SM, t.warning_text).whitespace_normal().child(reason),
                                        )
                                    })
                                    .when_some(built.quote_note, |d, note| {
                                        d.child(text(ty::SM, t.warning_text).whitespace_normal().child(note))
                                    }),
                            )
                            .child(
                                div()
                                    .flex_none()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(
                                        // Zero-gap wrapper keeps Tab order intact when the disabled
                                        // trigger unmounts without inserting a phantom gap.
                                        div()
                                            .flex()
                                            .items_center()
                                            .child(copy_btn)
                                            .children(disabled_anchor.map(|fh| div().track_focus(&fh))),
                                    )
                                    .children(server_btn),
                            ),
                    ),
            )
            .into_any_element()
    }

    fn export_cli_export_controls(
        &self,
        t: &crate::theme::Theme,
        m: &Model,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let fmt_labels: Vec<&str> = CliExportFormat::ALL.iter().map(|f| f.label()).collect();
        let fmt_idx =
            CliExportFormat::ALL.iter().position(|&f| f == self.model_cli.export_format).unwrap_or(0);
        let fmt_seg = self.segmented(
            "model-cli-fmt",
            &fmt_labels,
            fmt_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.model_cli.export_format = CliExportFormat::ALL[i.min(CliExportFormat::ALL.len() - 1)];
                cx.notify();
            }),
        );

        let extra: Option<AnyElement> = match self.model_cli.export_format {
            CliExportFormat::Onnx => {
                let dyn_batch = self
                    .chip(
                        "model-cli-dyn-batch",
                        "Dynamic batch",
                        self.model_cli.onnx_dynamic_batch,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_dynamic_batch = !this.model_cli.onnx_dynamic_batch;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Symbolic batch axis (--dynamic-batch; combines with Dynamic H×W into --dynamic)",
                        None,
                    ));
                let dyn_hw = self
                    .chip(
                        "model-cli-dyn-hw",
                        "Dynamic H×W",
                        self.model_cli.onnx_dynamic_hw,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_dynamic_hw = !this.model_cli.onnx_dynamic_hw;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Symbolic height and width axes (--dynamic-hw; multiples of 32)",
                        None,
                    ));
                let decode = self
                    .chip(
                        "model-cli-decode",
                        "Decoded outputs",
                        self.model_cli.onnx_decode,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_decode = !this.model_cli.onnx_decode;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Decode boxes and class scores inside the ONNX graph (--decode)",
                        None,
                    ));
                let half = self
                    .chip(
                        "model-cli-half",
                        "FP16 (--half)",
                        self.model_cli.onnx_half,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_half = !this.model_cli.onnx_half;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Half-precision ONNX weights, inputs, and activations (--half)",
                        None,
                    ));
                let head_o2m = (m.head == Head::O2o).then(|| {
                    self.chip(
                        "model-cli-head-o2m",
                        "Export o2m head",
                        self.model_cli.onnx_head_o2m,
                        None,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.model_cli.onnx_head_o2m = !this.model_cli.onnx_head_o2m;
                            cx.notify();
                        }),
                    )
                    .tooltip(crate::controls::tip(
                        "Export this o2o model's one-to-many branch instead (--head o2m)",
                        None,
                    ))
                });
                Some(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(6.))
                        .child(dyn_batch)
                        .child(dyn_hw)
                        .child(decode)
                        .child(half)
                        .children(head_o2m)
                        .into_any_element(),
                )
            }
            CliExportFormat::Int8Quantize => {
                let labels: Vec<&str> = QuantMethod::ALL.iter().map(|q| q.label()).collect();
                let idx =
                    QuantMethod::ALL.iter().position(|&q| q == self.model_cli.quant_method).unwrap_or(0);
                let seg = self.segmented(
                    "model-cli-quant-method",
                    &labels,
                    idx,
                    window,
                    cx,
                    Rc::new(|this, i, _, cx| {
                        this.model_cli.quant_method = QuantMethod::ALL[i.min(QuantMethod::ALL.len() - 1)];
                        cx.notify();
                    }),
                );
                Some(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(text(ty::SM, t.text_tertiary).child("Calibration method"))
                        .child(seg)
                        .into_any_element(),
                )
            }
            _ => None,
        };

        div()
            .flex()
            .flex_wrap()
            .items_center()
            .justify_between()
            .gap(px(10.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_tertiary).font_weight(FontWeight::MEDIUM).child("Format"))
                    .child(fmt_seg),
            )
            .children(extra)
            .into_any_element()
    }

    fn export_cli_predict_controls(
        &self,
        t: &crate::theme::Theme,
        m: &Model,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let sahi = self
            .chip(
                "model-cli-sahi",
                "SAHI (--tile 640)",
                self.model_cli.predict_sahi,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.model_cli.predict_sahi = !this.model_cli.predict_sahi;
                    if this.model_cli.predict_sahi {
                        this.model_cli.predict_tta = false;
                    }
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "SAHI tiled inference for small objects in high-res images (mutually exclusive with TTA)",
                None,
            ));
        let tta = self
            .chip(
                "model-cli-tta",
                "TTA (--tta)",
                self.model_cli.predict_tta,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.model_cli.predict_tta = !this.model_cli.predict_tta;
                    if this.model_cli.predict_tta {
                        this.model_cli.predict_sahi = false;
                    }
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "Multi-scale + horizontal flip Test-Time Augmentation (mutually exclusive with SAHI)",
                None,
            ));
        let track = self
            .chip(
                "model-cli-track",
                "ByteTrack (--track)",
                self.model_cli.predict_track,
                None,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.model_cli.predict_track = !this.model_cli.predict_track;
                    cx.notify();
                }),
            )
            .tooltip(crate::controls::tip(
                "Track objects across ordered images or folders with persistent IDs",
                None,
            ));

        let prec_labels: Vec<&str> = PredictPrecision::ALL.iter().map(|p| p.label()).collect();
        let prec_idx =
            PredictPrecision::ALL.iter().position(|&p| p == self.model_cli.predict_precision).unwrap_or(0);
        let prec_seg = self.segmented(
            "model-cli-prec",
            &prec_labels,
            prec_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.model_cli.predict_precision =
                    PredictPrecision::ALL[i.min(PredictPrecision::ALL.len() - 1)];
                cx.notify();
            }),
        );

        let conf = self.model_cli.predict_conf;
        let conf_slider = self.slider(
            "model-cli-conf",
            conf,
            MODEL_CLI_CONF.0,
            MODEL_CLI_CONF.1,
            MODEL_CLI_CONF.2,
            window,
            cx,
        );
        let iou_control = predict_uses_iou(m, &self.model_cli).then(|| {
            let iou = self.model_cli.predict_iou;
            let iou_label =
                if m.head == Head::O2o && self.model_cli.predict_tta { "Merge IoU" } else { "IoU" };
            let iou_slider = self.slider(
                "model-cli-iou",
                iou,
                MODEL_CLI_IOU.0,
                MODEL_CLI_IOU.1,
                MODEL_CLI_IOU.2,
                window,
                cx,
            );
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(text(ty::SM, t.text_tertiary).child(iou_label))
                .child(div().w(px(100.)).child(iou_slider))
                .child(text(ty::SM, t.text_primary).tnum().w(px(30.)).child(command::fmt_thresh(iou)))
        });

        div()
            .flex()
            .flex_col()
            .gap(px(10.))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(10.))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .items_center()
                            .gap(px(6.))
                            .child(sahi)
                            .child(tta)
                            .child(track),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(text(ty::SM, t.text_tertiary).child("Precision"))
                            .child(prec_seg),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap(px(20.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(text(ty::SM, t.text_tertiary).child("Confidence"))
                            .child(div().w(px(100.)).child(conf_slider))
                            .child(
                                text(ty::SM, t.text_primary)
                                    .tnum()
                                    .w(px(30.))
                                    .child(command::fmt_thresh(conf)),
                            ),
                    )
                    .children(iou_control),
            )
            .into_any_element()
    }
}
