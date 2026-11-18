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
