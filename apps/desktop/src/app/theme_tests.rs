//! Themes in the running app (design/THEMES.md): Settings › Appearance's rows, picking a theme with
//! the mode / slot rules, System following the OS, custom themes and Reload, the palette's commands.

use folo_test_support::TestDir;
use folo_ui::{HeadlessApp, WindowAppearance, WindowHandle};

use super::select_driving::{click_option, press};
use super::tests::{drive, open};
use super::{FoloApp, Route, follow_system_appearance};
use crate::theme::{Catalog, Mode, ModePref, theme};

/// A valid custom theme file.
fn night(id: &str, name: &str) -> String {
    format!(
        r##"{{"folo_theme": 1, "id": "{id}", "name": "{name}", "mode": "dark", "base": "midnight",
            "color": {{"bg": {{"app": "#0B0D12"}}}}}}"##
    )
}

/// The app with custom themes read from `folder` (as at start), on Settings.
fn on_settings(folder: &std::path::Path) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    let folder = folder.to_path_buf();
    drive(&mut app, window, move |this, w, cx| {
        this.themes = Catalog::load(&folder);
        this.apply_theme(cx);
        this.navigate(Route::Settings, w, cx);
    });
    (app, window)
}

/// The id, mode and colours shown now, and the choice.
fn shown(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> (String, Mode, ModePref) {
    drive(app, window, |this, _, cx| {
        let t = theme(cx);
        assert_eq!(t.mode, this.mode, "the app's mode is the theme's");
        (t.id.as_str().to_string(), t.mode, this.theme_prefs.mode)
    })
}

/// The focus keys drawn in a fresh frame.
fn drawn_keys(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

#[test]
fn settings_shows_the_four_rows_and_the_failed_files() {
    let dir = TestDir::new("desktop-theme-settings");
    let folder = dir.path().join("themes");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("night.json"), night("night-shift", "Night shift")).unwrap();
    std::fs::write(folder.join("typo.json"), r#"{"folo_theme": 1, "id": "t", "name": "T", "mode": "dim"}"#)
        .unwrap();
    let (mut app, window) = on_settings(&folder);
    let keys = drawn_keys(&mut app, window);
    for key in ["set-theme", "set-dark-theme", "set-light-theme", "set-themes-reload", "set-motion"] {
        assert!(keys.iter().any(|k| k == key), "{key} not drawn: {keys:?}");
    }
    let (count, failed, dark_names) = drive(&mut app, window, |this, _, _| {
        let names: Vec<String> = this.themes.of_mode(Mode::Dark).map(|e| e.name.clone()).collect();
        (this.themes.custom_count(), this.themes.failed.clone(), names)
    });
    assert_eq!(count, 1);
    assert_eq!(failed.len(), 1);
    assert_eq!(
        (failed[0].file.as_str(), failed[0].error.as_str()),
        ("typo.json", "mode: must be \"dark\" or \"light\"")
    );
    assert_eq!(dark_names, ["Dark", "Midnight", "Dim", "High contrast dark", "Night shift"]);
}

#[test]
fn picking_a_theme_follows_the_mode_and_slot_rules() {
    let dir = TestDir::new("desktop-theme-pick");
    let (mut app, window) = on_settings(&dir.path().join("themes"));
    assert_eq!(shown(&mut app, window), ("dark".into(), Mode::Dark, ModePref::Dark));
    // A light theme while Dark: switches to Light with it.
    drive(&mut app, window, |this, _, cx| this.pick_theme("paper", cx));
    assert_eq!(shown(&mut app, window), ("paper".into(), Mode::Light, ModePref::Light));
    drive(&mut app, window, |this, _, cx| this.pick_theme("midnight", cx));
    assert_eq!(shown(&mut app, window), ("midnight".into(), Mode::Dark, ModePref::Dark));
    // System: the headless OS is dark; the dark slot shows, and a light pick only fills its slot.
    drive(&mut app, window, |this, _, cx| this.set_mode(ModePref::System, cx));
    assert_eq!(shown(&mut app, window), ("midnight".into(), Mode::Dark, ModePref::System));
    drive(&mut app, window, |this, _, cx| this.pick_theme("contrast-light", cx));
    assert_eq!(shown(&mut app, window), ("midnight".into(), Mode::Dark, ModePref::System));
    // The OS turns light: System follows at once; back to dark, too.
    drive(&mut app, window, |this, _, cx| this.system_appearance_changed(false, cx));
    assert_eq!(shown(&mut app, window), ("contrast-light".into(), Mode::Light, ModePref::System));
    drive(&mut app, window, |this, _, cx| this.system_appearance_changed(true, cx));
    assert_eq!(shown(&mut app, window).0, "midnight");
    // Ctrl Shift L: the other mode, explicit, each with its theme.
    drive(&mut app, window, |this, _, cx| this.toggle_theme(cx));
    assert_eq!(shown(&mut app, window), ("contrast-light".into(), Mode::Light, ModePref::Light));
    drive(&mut app, window, |this, _, cx| this.toggle_theme(cx));
    assert_eq!(shown(&mut app, window), ("midnight".into(), Mode::Dark, ModePref::Dark));
    // Explicit modes ignore the OS.
    drive(&mut app, window, |this, _, cx| this.system_appearance_changed(false, cx));
    assert_eq!(shown(&mut app, window).1, Mode::Dark);
    // An unknown id picks nothing.
    drive(&mut app, window, |this, _, cx| this.pick_theme("nope", cx));
    assert_eq!(shown(&mut app, window).0, "midnight");
    // Each frame draws without trouble in the theme it shows.
    assert!(drawn_keys(&mut app, window).iter().any(|k| k == "set-dark-theme"));
}

#[test]
fn reload_reads_the_folder_again_and_a_removed_theme_falls_back_to_the_default() {
    let dir = TestDir::new("desktop-theme-reload");
    let folder = dir.path().join("themes");
    std::fs::create_dir_all(&folder).unwrap();
    let (mut app, window) = on_settings(&folder);
    assert_eq!(drive(&mut app, window, |this, _, _| this.themes.custom_count()), 0);
    std::fs::write(folder.join("night.json"), night("night-shift", "Night shift")).unwrap();
    drive(&mut app, window, |this, _, cx| this.reload_themes(cx));
    app.run_until_idle();
    assert_eq!(
        drive(&mut app, window, |this, _, _| (this.themes.custom_count(), this.themes_loading)),
        (1, false)
    );
    drive(&mut app, window, |this, _, cx| this.pick_theme("night-shift", cx));
    let bg = drive(&mut app, window, |_, _, cx| theme(cx).bg_app);
    assert_eq!(bg, crate::theme::rgba32(0x0B0D12FF), "the custom colours show");
    assert_eq!(shown(&mut app, window).0, "night-shift");
    // Removed: the default shows, and the stored id stays for when it comes back.
    std::fs::remove_file(folder.join("night.json")).unwrap();
    drive(&mut app, window, |this, _, cx| this.reload_themes(cx));
    app.run_until_idle();
    assert_eq!(shown(&mut app, window).0, "dark");
    assert_eq!(drive(&mut app, window, |this, _, _| this.theme_prefs.dark.clone()), "night-shift");
}

/// System follows the OS through folo-ui's delivery: an appearance change at the platform reaches
/// the handler the app registers at start (`follow_system_appearance`), and the window frame then
/// takes the mode shown.
#[test]
fn system_follows_an_os_change_delivered_through_the_window() {
    let dir = TestDir::new("desktop-theme-system");
    let (mut app, window) = on_settings(&dir.path().join("themes"));
    drive(&mut app, window, |this, w, cx| {
        follow_system_appearance(w, cx.weak_entity(), cx);
        this.theme_prefs.light = "paper".into();
        this.set_mode(ModePref::System, cx);
    });
    assert_eq!(shown(&mut app, window), ("dark".into(), Mode::Dark, ModePref::System));
    app.set_system_appearance(WindowAppearance::Light);
    assert_eq!(shown(&mut app, window), ("paper".into(), Mode::Light, ModePref::System));
    app.frame(window.into());
    assert_eq!(app.platform().frame_appearances(), [Some(WindowAppearance::Light)], "the frame follows");
    app.set_system_appearance(WindowAppearance::Dark);
    assert_eq!(shown(&mut app, window), ("dark".into(), Mode::Dark, ModePref::System));
    app.frame(window.into());
    assert_eq!(app.platform().frame_appearances(), [Some(WindowAppearance::Dark)]);
    // An explicit mode ignores the OS.
    drive(&mut app, window, |this, _, cx| this.set_mode(ModePref::Dark, cx));
    app.set_system_appearance(WindowAppearance::Light);
    assert_eq!(shown(&mut app, window).1, Mode::Dark);
}

/// The choice (mode, both slots) and what shows.
fn state(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> (String, ModePref, String, String) {
    let (id, _, mode) = shown(app, window);
    let (dark, light) =
        drive(app, window, |this, _, _| (this.theme_prefs.dark.clone(), this.theme_prefs.light.clone()));
    (id, mode, dark, light)
}

/// design/THEMES.md, Choosing a theme: a different theme in a slot applies it (an explicit mode
/// switches to the slot's mode); the theme the slot has already does nothing, by pointer or keys,
/// so both input paths agree (review r4: the pointer applied it, Enter and Space didn't). The rule
/// lives in the slot's callback: the select calls back for a click on its chosen option, as other
/// selects need (review r5).
#[test]
fn re_picking_a_slots_theme_does_nothing_by_pointer_or_keys() {
    let dir = TestDir::new("desktop-theme-repick");
    let (mut app, window) = on_settings(&dir.path().join("themes"));
    let start = ("dark".to_string(), ModePref::Dark, "dark".to_string(), "light".to_string());
    assert_eq!(state(&mut app, window), start);
    // The Light slot's own theme (Light, the first option) while Dark: by pointer, then by keys.
    click_option(&mut app, window, "set-light-theme", 0);
    assert_eq!(state(&mut app, window), start, "the pointer on the chosen option changes nothing");
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("set-light-theme", cx)));
    press(&mut app, window, "enter");
    press(&mut app, window, "enter");
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), None, "Enter closes the menu");
    assert_eq!(state(&mut app, window), start, "Enter on the chosen option changes nothing");
    press(&mut app, window, "space");
    press(&mut app, window, "space");
    assert_eq!(state(&mut app, window), start, "nor does Space");
    // Another theme in that slot applies it and switches to Light: by pointer (Paper, the second)...
    click_option(&mut app, window, "set-light-theme", 1);
    let paper = ("paper".to_string(), ModePref::Light, "dark".to_string(), "paper".to_string());
    assert_eq!(state(&mut app, window), paper);
    // ...and by keys (Midnight, below Dark): back to Dark with it.
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("set-dark-theme", cx)));
    press(&mut app, window, "enter");
    press(&mut app, window, "down");
    press(&mut app, window, "enter");
    let midnight = ("midnight".to_string(), ModePref::Dark, "midnight".to_string(), "paper".to_string());
    assert_eq!(state(&mut app, window), midnight);
    // Re-picking Paper (the Light slot's) while Dark, by pointer: still Dark, with Midnight.
    click_option(&mut app, window, "set-light-theme", 1);
    assert_eq!(state(&mut app, window), midnight);
}

/// A headless app `width` wide (tall enough for the whole Settings page) with the app's fonts and
/// the dark theme, as the app starts.
fn headless(width: f32) -> (HeadlessApp, folo_ui::Bounds<folo_ui::Pixels>) {
    use folo_ui::{Bounds, HeadlessOptions, point, px, size};
    let window_size = size(px(width), px(3200.));
    let mut app = HeadlessApp::new(HeadlessOptions { size: window_size, ..Default::default() });
    app.update(|cx| {
        cx.text_system()
            .add_fonts(crate::FONTS.iter().map(|f| std::borrow::Cow::Borrowed(*f)).collect())
            .expect("bundled fonts load");
        cx.set_global(crate::theme::Theme::new(Mode::Dark));
    });
    (app, Bounds::new(point(px(0.), px(0.)), window_size))
}

/// The app on Settings at `width` with the custom themes in `folder`.
fn settings_at(width: f32, folder: &std::path::Path) -> (HeadlessApp, WindowHandle<FoloApp>) {
    use folo_ui::{AppContext, WindowBounds, WindowOptions};
    let (mut app, bounds) = headless(width);
    let window = app.update(|cx| {
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        let api = std::sync::Arc::new(crate::api::mock::MockBackend::new(4200, true));
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
            .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    let folder = folder.to_path_buf();
    drive(&mut app, window, move |this, w, cx| {
        this.themes = Catalog::load(&folder);
        this.navigate(Route::Settings, w, cx);
    });
    (app, window)
}

/// The glyphs a fresh frame paints in the danger and in the secondary ink: (wholly inside their
/// clip, all).
fn glyphs(app: &mut HeadlessApp, window: folo_ui::AnyWindowHandle) -> [(usize, usize); 2] {
    app.frame(window);
    app.run_until_idle();
    app.frame(window);
    let t = app.update(|cx| theme(cx));
    let mut n = [(0, 0); 2];
    for p in &app.scene(window).primitives {
        if let folo_ui::Primitive::MonoSprite(s) = p
            && s.kind == folo_ui::SpriteKind::Glyph
        {
            let ink = if s.color == t.danger_text {
                0
            } else if s.color == t.text_secondary {
                1
            } else {
                continue;
            };
            n[ink].0 += usize::from(s.bounds.is_contained_within(&s.clip));
            n[ink].1 += 1;
        }
    }
    n
}

/// Each failed file's name and reason styled as the row styles them, each on its own unwrapped
/// line: the glyphs the row must show, shaped by the same text backend.
struct Reference(Vec<(String, String)>);

impl folo_ui::Render for Reference {
    fn render(
        &mut self,
        _: &mut folo_ui::Window,
        cx: &mut folo_ui::Context<Self>,
    ) -> impl folo_ui::IntoElement {
        use crate::theme::ty;
        use crate::ui::text;
        use folo_ui::{ParentElement, Styled, div};
        let t = theme(cx);
        div().flex().flex_col().items_start().children(self.0.iter().flat_map(|(file, error)| {
            [
                text(ty::SM, t.text_secondary)
                    .flex_none()
                    .whitespace_nowrap()
                    .font_family(crate::theme::MONO)
                    .child(file.clone()),
                text(ty::SM, t.danger_text).flex_none().whitespace_nowrap().child(error.clone()),
            ]
        }))
    }
}

/// #352: a failed file's name too long for its row pushed the reason past the panel's edge, where
/// it was clipped. At the narrowest window and at 1280, a long name with no spaces and one with
/// spaces each paint whole, and so does each reason: the glyphs the failed files add inside their
/// clip are the glyphs the same names and reasons paint on unclipped lines. (Glyphs, not characters:
/// a font may shape `fi` as one.)
#[test]
fn a_failed_file_with_a_long_name_shows_the_whole_name_and_reason() {
    let dir = TestDir::new("desktop-theme-long-name");
    // Two folders whose shown paths (in the names' ink) are as long.
    let folder = dir.path().join("a").join("themes");
    let empty = dir.path().join("b").join("themes");
    std::fs::create_dir_all(&folder).unwrap();
    let solid = format!("broken-theme-{}case.json", "filename-".repeat(17));
    let spaced = format!("broken theme {}case.json", "file name ".repeat(12));
    std::fs::write(
        folder.join(&solid),
        r#"{"folo_theme": 1, "id": "long-theme-file", "name": "Broken", "mode": "chartreuse"}"#,
    )
    .unwrap();
    std::fs::write(folder.join(&spaced), r#"{"folo_theme": 2, "id": "spaced", "name": "S", "mode": "dark"}"#)
        .unwrap();
    // The reference: every name and reason on one line in a window far wider than any of them.
    // (Not the Settings page in a wide window: its column stops at 880 px, so there the rows still
    // wrap by the very layout under test.)
    let failed = Catalog::load(&folder).failed;
    assert_eq!(failed.len(), 2, "{failed:?}");
    let rows = failed
        .iter()
        .map(|f| (f.file.clone(), f.error.strip_prefix("(file): ").unwrap_or(&f.error).to_string()))
        .collect();
    let (mut app, bounds) = headless(8000.);
    let window = app.update(|cx| {
        use folo_ui::{AppContext, WindowBounds, WindowOptions};
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |_, cx| cx.new(|_| Reference(rows))).expect("open window")
    });
    let [reasons, names] = glyphs(&mut app, window.into());
    assert!(reasons.1 > 0 && names.1 > 0, "the reference paints: {reasons:?} {names:?}");
    assert_eq!(reasons.0, reasons.1, "the reference's reasons are unclipped");
    assert_eq!(names.0, names.1, "the reference's names are unclipped");
    for width in [960., 1280.] {
        let (mut app, window) = settings_at(width, &folder);
        let [with_reasons, with_names] = glyphs(&mut app, window.into());
        // The same page without the failed files: what the rest of it paints in those inks.
        drive(&mut app, window, |this, _, cx| {
            this.themes = Catalog::load(&empty);
            cx.notify();
        });
        let [without_reasons, without_names] = glyphs(&mut app, window.into());
        assert_eq!(with_reasons.0 - without_reasons.0, reasons.1, "every reason's glyph shows at {width}");
        assert_eq!(with_names.0 - without_names.0, names.1, "every name's glyph shows at {width}");
    }
}

/// #382 (after #375): a failed theme file with a short name and a long reason must keep the start of
/// the reason beside the name on the first line (not wrap the entire reason onto the next line under
/// an otherwise empty row), while still wrapping and showing every glyph unclipped.
#[test]
fn a_failed_file_with_a_short_name_and_long_reason_starts_beside_the_name() {
    let dir = TestDir::new("desktop-theme-short-name-long-reason");
    let folder = dir.path().join("a").join("themes");
    let empty = dir.path().join("b").join("themes");
    std::fs::create_dir_all(&folder).unwrap();
    // A multi-word unknown top-level key long enough to force multi-line wrapping at both 960 and
    // 1280 px: the old `flex_wrap` layout dropped the entire reason onto the line below `bad.json`.
    let long_key = format!("invalid option {}", "with detailed reason text ".repeat(6));
    std::fs::write(
        folder.join("bad.json"),
        format!(
            r#"{{"{long_key}": true, "folo_theme": 1, "id": "short-name", "name": "Short", "mode": "dark"}}"#
        ),
    )
    .unwrap();
    let failed = Catalog::load(&folder).failed;
    assert_eq!(failed.len(), 1, "{failed:?}");
    let rows: Vec<(String, String)> = failed
        .iter()
        .map(|f| (f.file.clone(), f.error.strip_prefix("(file): ").unwrap_or(&f.error).to_string()))
        .collect();
    let (mut ref_app, ref_bounds) = headless(8000.);
    let ref_win = ref_app.update(|cx| {
        use folo_ui::{AppContext, WindowBounds, WindowOptions};
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(ref_bounds)), ..Default::default() };
        cx.open_window(options, |_, cx| cx.new(|_| Reference(rows))).expect("open window")
    });
    let [ref_reasons, ref_names] = glyphs(&mut ref_app, ref_win.into());
    assert!(ref_reasons.1 > 0 && ref_names.1 > 0);

    for width in [960., 1280.] {
        let (mut app, window) = settings_at(width, &folder);
        let [with_reasons, with_names] = glyphs(&mut app, window.into());
        let t = app.update(|cx| theme(cx));
        // Snapshot danger and secondary glyph (x, y) positions with and without the failed file.
        let collect_pts = |app: &mut HeadlessApp, window: folo_ui::AnyWindowHandle| {
            let mut d = Vec::new();
            let mut s = Vec::new();
            for p in &app.scene(window).primitives {
                if let folo_ui::Primitive::MonoSprite(sp) = p
                    && sp.kind == folo_ui::SpriteKind::Glyph
                {
                    let pt = (f64::from(sp.bounds.origin.x) as f32, f64::from(sp.bounds.origin.y) as f32);
                    if sp.color == t.danger_text {
                        d.push(pt);
                    } else if sp.color == t.text_secondary {
                        s.push(pt);
                    }
                }
            }
            (d, s)
        };
        let (danger_pts, with_secondary_pts) = collect_pts(&mut app, window.into());
        // Remove the failed file to see what the rest of the page paints.
        drive(&mut app, window, |this, _, cx| {
            this.themes = Catalog::load(&empty);
            cx.notify();
        });
        let [without_reasons, without_names] = glyphs(&mut app, window.into());
        let (_, without_secondary_pts) = collect_pts(&mut app, window.into());
        assert_eq!(with_reasons.0 - without_reasons.0, ref_reasons.1, "every reason glyph shows at {width}");
        assert_eq!(with_names.0 - without_names.0, ref_names.1, "every name glyph shows at {width}");
        // The wrapped reason's continuation lines begin at the text column's left edge (`col_left`),
        // strictly inside the failed row (excluding the sidebar). On line 1, the reason starts to the
        // right of `bad.json`, and every glyph of `bad.json` occupies `[col_left, first_line_danger_min_x)`.
        assert_eq!(without_reasons.1, 0, "expected no other danger glyphs on Settings");
        let line_h = crate::theme::ty::SM.line;
        let col_left = danger_pts.iter().map(|&(x, _)| x).fold(f32::INFINITY, f32::min);
        let reason_top = danger_pts.iter().map(|&(_, y)| y).fold(f32::INFINITY, f32::min);
        let reason_bottom = danger_pts.iter().map(|&(_, y)| y).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            reason_bottom - reason_top >= line_h * 0.75,
            "long reason should wrap across multiple lines at {width}: top={reason_top}, bottom={reason_bottom}"
        );
        let on_first_line = |y: f32| y >= reason_top - 4. && y < reason_top + line_h * 0.85;
        let first_line_danger_min_x = danger_pts
            .iter()
            .filter(|&&(_, y)| on_first_line(y))
            .map(|&(x, _)| x)
            .fold(f32::INFINITY, f32::min);
        assert!(
            first_line_danger_min_x > col_left + 20.,
            "reason must start beside bad.json on line 1 (indented past wrapped line 2's col_left) at {width}: col_left={col_left}, first_line_danger_min_x={first_line_danger_min_x}"
        );
        let in_name_slot =
            |&(x, y): &(f32, f32)| x >= col_left - 2. && x < first_line_danger_min_x && on_first_line(y);
        let baseline_in_slot = without_secondary_pts.iter().filter(|pt| in_name_slot(pt)).count();
        assert_eq!(
            baseline_in_slot, 0,
            "empty baseline should have no secondary glyphs in the failed row slot"
        );
        let name_glyphs_beside_reason = with_secondary_pts.iter().filter(|pt| in_name_slot(pt)).count();
        assert_eq!(
            name_glyphs_beside_reason, ref_names.1,
            "every glyph of bad.json should sit on line 1 in [{col_left}, {first_line_danger_min_x}) at {width}"
        );
    }
}

/// Bidi failed theme filename (`אx.json`) with a long English reason: both inks match the single-line
/// reference and every glyph renders unclipped inside the Settings panel.
#[test]
fn a_failed_file_with_an_rtl_name_keeps_both_inks_and_shows_the_full_reason() {
    let dir = TestDir::new("desktop-theme-rtl-name");
    let folder = dir.path().join("a").join("themes");
    let empty = dir.path().join("b").join("themes");
    std::fs::create_dir_all(&folder).unwrap();
    let long_key = format!("invalid option {}", "with detailed reason text ".repeat(6));
    std::fs::write(
        folder.join("אx.json"),
        format!(
            r#"{{"{long_key}": true, "folo_theme": 1, "id": "rtl-name", "name": "RTL", "mode": "dark"}}"#
        ),
    )
    .unwrap();
    let failed = Catalog::load(&folder).failed;
    assert_eq!(failed.len(), 1, "{failed:?}");
    let rows: Vec<(String, String)> = failed
        .iter()
        .map(|f| (f.file.clone(), f.error.strip_prefix("(file): ").unwrap_or(&f.error).to_string()))
        .collect();
    let (mut ref_app, ref_bounds) = headless(8000.);
    let ref_win = ref_app.update(|cx| {
        use folo_ui::{AppContext, WindowBounds, WindowOptions};
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(ref_bounds)), ..Default::default() };
        cx.open_window(options, |_, cx| cx.new(|_| Reference(rows))).expect("open window")
    });
    let [ref_reasons, ref_names] = glyphs(&mut ref_app, ref_win.into());
    assert!(ref_reasons.1 > 0 && ref_names.1 > 0);
    assert_eq!(ref_reasons.0, ref_reasons.1);
    assert_eq!(ref_names.0, ref_names.1);

    let (mut app, window) = settings_at(960., &folder);
    let [with_reasons, with_names] = glyphs(&mut app, window.into());
    drive(&mut app, window, |this, _, cx| {
        this.themes = Catalog::load(&empty);
        cx.notify();
    });
    let [without_reasons, without_names] = glyphs(&mut app, window.into());
    assert_eq!(with_reasons.0 - without_reasons.0, ref_reasons.1, "every reason glyph shows unclipped");
    assert_eq!(with_names.0 - without_names.0, ref_names.1, "every RTL filename glyph keeps secondary ink");
}
