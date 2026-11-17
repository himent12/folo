//! Settings › Appearance (design/THEMES.md, Choosing a theme): the mode (Dark / Light / System), the
//! theme each mode uses, the custom themes folder with Reload and the files that failed, and Reduce
//! motion.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, SharedString, Styled, Window, div, px};

use super::page::settings_row;
use crate::app::{FoloApp, act};
use crate::theme::{Mode, ModePref, theme, ty};
use crate::ui::*;

/// The width of the theme selects (the Server address field's, above).
const SELECT_W: f32 = 232.;

impl FoloApp {
    pub(super) fn appearance_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let help = |s: &str| crate::ui::rich(&t, ty::SM, t.text_tertiary, s).text_left().into_any_element();
        let modes = [ModePref::Dark, ModePref::Light, ModePref::System];
        let mode_sel = modes.iter().position(|m| *m == self.theme_prefs.mode).unwrap_or(0);
        let mode_seg = self.segmented(
            "set-theme",
            &["Dark", "Light", "System"],
            mode_sel,
            window,
            cx,
            Rc::new(move |this, i, _, cx| this.set_mode(modes[i], cx)),
        );
        let mode_help = if self.theme_prefs.mode == ModePref::System {
            let now = if self.system_dark { "dark" } else { "light" };
            help(&format!("Follows your operating system, currently {now}."))
        } else {
            div()
                .flex()
                .items_center()
                .gap(px(6.))
                .child(shortcut(&t, &["Ctrl", "Shift", "L"]))
                .child(text(ty::SM, t.text_tertiary).child("switches anywhere"))
                .into_any_element()
        };

        let mut rows = vec![settings_row(&t, "Theme", mode_help, mode_seg.into_any_element())];
        for (mode, key, title, what) in [
            (
                Mode::Dark,
                "set-dark-theme",
                "Dark theme",
                "Used when the theme is Dark, or System while your system is dark.",
            ),
            (
                Mode::Light,
                "set-light-theme",
                "Light theme",
                "Used when the theme is Light, or System while your system is light.",
            ),
        ] {
            let select = self.theme_select(mode, key, window, cx);
            rows.push(settings_row(&t, title, help(what), select));
        }

        // Custom themes: the folder (mono), how many it gave, Reload; the files that failed under it.
        let folder = self.themes.folder.clone().unwrap_or_else(crate::theme::catalog::custom_folder);
        let loaded = self.themes.custom_count();
        let failed = &self.themes.failed;
        let count = match (loaded, failed.len()) {
            (0, 0) => "None yet: put theme files (.json) in this folder, then Reload.".to_string(),
            (n, 0) => format!("{n} {} loaded.", if n == 1 { "theme" } else { "themes" }),
            (n, f) => format!("{n} loaded, {f} failed.", n = n),
        };
        let custom_help = div()
            .flex()
            .flex_col()
            .gap(px(2.))
            .child(
                text(ty::SM, t.text_secondary)
                    .font_family(crate::theme::MONO)
                    .text_left()
                    .child(folder.display().to_string()),
            )
            .child(text(ty::SM, t.text_tertiary).text_left().child(count))
            .into_any_element();
        let reload = self
            .button(
                "set-themes-reload",
                if self.themes_loading { "Reloading…" } else { "Reload" },
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(|this, _, cx| this.reload_themes(cx)),
            )
            .into_any_element();
        let custom_row = settings_row(&t, "Custom themes", custom_help, reload);
        let failures = (!failed.is_empty()).then(|| {
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .px(px(16.))
                .py(px(10.))
                .border_b_1()
                .border_color(t.line_subtle)
                .children(failed.iter().map(|f| {
                    div()
                        .flex()
                        .items_baseline()
                        .gap(px(10.))
                        .when(!f.file.is_empty(), |d| {
                            d.child(
                                text(ty::SM, t.text_secondary)
                                    .flex_none()
                                    .font_family(crate::theme::MONO)
                                    .child(f.file.clone()),
                            )
                        })
                        .child(text(ty::SM, t.danger_text).min_w_0().text_left().child(f.error.clone()))
                }))
        });

        let motion = self.switch(
            "set-motion",
            self.reduce_motion,
            window,
            cx,
            act(|this, _, cx| {
                this.reduce_motion = !this.reduce_motion;
                set_reduce_motion(this.reduce_motion);
                cx.refresh_windows();
            }),
        );
        panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Appearance"))
            .children(rows)
            .child(custom_row)
            .children(failures)
            .child(settings_row(
                &t,
                "Reduce motion",
                help("Turn off pulsing live indicators and transitions."),
                motion.into_any_element(),
            ))
            .into_any_element()
    }

    /// The select of `mode`'s themes: built-ins in tokens.json's order, then custom ones by name; the
    /// theme shown for that mode is selected (the default when the stored id is not installed).
    fn theme_select(
        &self,
        mode: Mode,
        key: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let entries: Vec<_> = self.themes.of_mode(mode).cloned().collect();
        let current = self.themes.entry_for(&self.theme_prefs, mode).id.clone();
        let selected = entries.iter().position(|e| e.id == current).unwrap_or(0);
        let options: Vec<(SharedString, Option<SharedString>)> = entries
            .iter()
            .map(|e| {
                let detail = match (e.custom, e.description.is_empty()) {
                    (true, true) => "Custom theme".to_string(),
                    (true, false) => format!("Custom · {}", e.description),
                    (false, _) => e.description.clone(),
                };
                (e.name.clone().into(), Some(detail.into()))
            })
            .collect();
        let ids: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
        let name: SharedString = entries.get(selected).map(|e| e.name.clone()).unwrap_or_default().into();
        div()
            .w(px(SELECT_W))
            .child(self.select(
                key,
                name,
                options,
                selected,
                window,
                cx,
                Rc::new(move |this, i, _, cx| {
                    if let Some(id) = ids.get(i) {
                        this.pick_theme(id, cx);
                    }
                }),
            ))
            .into_any_element()
    }
}
