//! Command palette ranking, theme commands, and keyboard navigation tests.

use super::*;

#[test]
fn fuzzy_ranks_prefix_and_substring() {
    assert!(fuzzy("run", "Runs").unwrap() > fuzzy("run", "shapes-n-run").unwrap());
    assert!(fuzzy("shn", "shapes-n").is_some());
    assert!(fuzzy("zzz", "shapes").is_none());
    assert_eq!(fuzzy("", "anything"), Some(0));
}

fn item(group: &'static str, title: &str) -> PaletteItem {
    PaletteItem {
        group,
        icon: "",
        title: title.into(),
        detail: String::new(),
        keys: &[],
        action: act(|_, _, _| {}),
    }
}

/// "run" matches a run by its group's name ("shapes-n Runs") better than "New training run": the
/// Runs group showed twice, split around Actions (stress d5-window-input-6).
#[test]
fn a_group_shows_once_with_its_best_match_first() {
    let items = vec![
        item("Actions", "New training run"),
        item("Go to", "Runs"),
        item("Runs", "shapes-n"),
        item("Runs", "shapes-mini-s"),
    ];
    let ranked: Vec<(&str, String)> = rank("run", items).into_iter().map(|i| (i.group, i.title)).collect();
    let groups: Vec<&str> = ranked.iter().map(|r| r.0).collect();
    assert_eq!(groups, ["Go to", "Runs", "Runs", "Actions"], "{ranked:?}");
    assert_eq!(ranked[1].1, "shapes-n", "within a group, best first");
    // No query: the items as listed.
    let all = rank("", vec![item("Actions", "a"), item("Runs", "b"), item("Actions", "c")]);
    assert_eq!(all.iter().map(|i| i.title.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
}

/// The 40 shown are the 40 best before they are grouped: with 40 runs of long names that match
/// "run" weakly (through their group), grouping first pushed "New training run" out of the list
/// (round 1 review of stress d5-window-input-6).
#[test]
fn the_cut_keeps_the_best_matches_of_every_group() {
    let mut items = vec![item("Actions", "New training run"), item("Go to", "Runs")];
    items.push(item("Runs", "a"));
    for n in 0..39 {
        items.push(item("Runs", &format!("experiment-with-a-very-long-name-{n:03}")));
    }
    assert!(fuzzy("run", "a Runs") > fuzzy("run", "New training run Actions"));
    assert!(
        fuzzy("run", "New training run Actions") > fuzzy("run", "experiment-with-a-very-long-name-000 Runs")
    );
    let ranked = rank("run", items);
    assert_eq!(ranked.len(), 40);
    let titles: Vec<&str> = ranked.iter().map(|i| i.title.as_str()).collect();
    assert!(titles.contains(&"New training run"), "{titles:?}");
    let groups: Vec<&str> = ranked.iter().map(|i| i.group).collect();
    assert_eq!(&groups[..3], ["Go to", "Runs", "Runs"], "grouped, the best group first");
    assert_eq!(groups.last(), Some(&"Actions"));
}

#[test]
fn rank_matches_detail_behind_title_and_group() {
    let mut by_title = item("Models", "folo-s-demo");
    by_title.detail = "folo-n · mAP50-95 0.500".into();
    let mut by_detail_same_group = item("Models", "street-detector");
    by_detail_same_group.detail = "folo-s · mAP50-95 0.720".into();
    let other_group_title = item("Runs", "exp-folo-s-baseline");
    let plain = item("Models", "other-model");
    let ranked = rank("folo-s", vec![by_detail_same_group, plain, other_group_title, by_title]);
    let summary: Vec<(&str, &str)> = ranked.iter().map(|i| (i.group, i.title.as_str())).collect();
    // Every primary match (across Models and Runs) precedes the detail-only match, and no heading repeats.
    assert_eq!(
        summary,
        [("Models", "folo-s-demo"), ("Runs", "exp-folo-s-baseline"), ("Details", "street-detector")]
    );
}

mod theme_commands {
    use crate::api::FEATURE_VISION_BENCH;
    use crate::app::Route;
    use crate::app::tests::{drive, open};
    use crate::store::Load;
    use crate::theme::{ModePref, theme};
    use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};

    fn press(app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<crate::app::FoloApp>, key: &str) {
        let ks = Keystroke::parse(key).unwrap();
        app.simulate_input(
            w.into(),
            PlatformInput::KeyDown(KeyDownEvent { keystroke: ks.clone(), is_held: false }),
        );
        app.run_until_idle();
        app.frame(w.into());
        app.simulate_input(w.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke: ks }));
        app.run_until_idle();
        app.frame(w.into());
    }

    /// "Theme: <name>" for every theme, custom ones too, picking it as Settings does; the toggle stays.
    #[test]
    fn every_theme_has_a_command() {
        let (mut app, window) = open();
        let titles = drive(&mut app, window, |this, _, cx| {
            let custom = crate::theme::custom::parse(
                r#"{"folo_theme": 1, "id": "sepia-2", "name": "Sepia", "mode": "light", "base": "paper"}"#,
            )
            .unwrap();
            this.themes = crate::theme::Catalog::with_custom(vec![custom]);
            let items = super::rank("theme", this.palette_items(theme(cx).id));
            items.into_iter().map(|i| (i.title, i.detail)).collect::<Vec<_>>()
        });
        let names: Vec<&str> = titles.iter().filter_map(|(t, _)| t.strip_prefix("Theme: ")).collect();
        assert_eq!(
            names,
            [
                "Dark",
                "Midnight",
                "Dim",
                "High contrast dark",
                "Light",
                "Paper",
                "High contrast light",
                "Sepia"
            ]
        );
        assert!(titles.iter().any(|(t, _)| t == "Switch to light theme"), "the toggle stays: {titles:?}");
        assert!(titles.iter().any(|(t, d)| t == "Theme: Dark" && d.ends_with("· current")));
        assert!(titles.iter().any(|(t, d)| t == "Theme: Sepia" && d == "custom light theme"));
        // The empty palette doesn't offer them (a query does, as in Studio).
        let unqueried = drive(&mut app, window, |this, _, cx| {
            this.filtered_palette(theme(cx).id).iter().filter(|i| i.group == "Themes").count()
        });
        assert_eq!(unqueried, 0);
        // Running one: Sepia's mode with it in the light slot.
        drive(&mut app, window, |this, w, cx| {
            let items = this.palette_items(theme(cx).id);
            let sepia = items.into_iter().find(|i| i.title == "Theme: Sepia").unwrap();
            (sepia.action)(this, w, cx);
        });
        let now = drive(&mut app, window, |this, _, cx| {
            (theme(cx).id.as_str().to_string(), this.theme_prefs.mode, this.theme_prefs.light.clone())
        });
        assert_eq!(now, ("sepia-2".into(), ModePref::Light, "sepia-2".into()));
    }

    #[test]
    fn benchmarks_indexed_and_jump_keys_navigate_palette() {
        let (mut app, window) = open();
        // Register production keybindings (including TextInput Home/End) so jump keys are exercised
        // through the real dispatch path.
        app.update(|cx| {
            crate::app::bind_keys(cx);
            crate::input::bind_keys(cx);
        });
        // Opening the palette before `/system` reports `vision_bench` still loads benchmarks once
        // the feature arrives on the next render.
        drive(&mut app, window, |this, w, cx| {
            this.store.features.retain(|f| f != FEATURE_VISION_BENCH);
            this.bench.list = Load::Idle;
            this.open_palette(w, cx);
            assert!(this.bench.list.needs_fetch(), "no fetch before vision_bench is reported");
        });
        app.frame(window.into());
        drive(&mut app, window, |this, _, cx| {
            this.store.features.push(FEATURE_VISION_BENCH.into());
            cx.notify();
        });
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());

        let (total, first_bench_id, first_bench_name) = drive(&mut app, window, |this, _, cx| {
            let benches = this.bench.list.ready().expect("benchmarks loaded while palette stayed open");
            (this.filtered_palette(theme(cx).id).len(), benches[0].id.clone(), benches[0].name.clone())
        });
        assert!(total > super::PALETTE_PAGE_STEP + 2, "need enough items to page: {total}");

        // PageDown advances by PALETTE_PAGE_STEP; PageUp returns to 0.
        press(&mut app, window, "pagedown");
        assert_eq!(
            drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().selected),
            super::PALETTE_PAGE_STEP
        );
        press(&mut app, window, "pageup");
        assert_eq!(drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().selected), 0);

        // With empty query and production TextInput bindings active, plain End/Home jump the list.
        press(&mut app, window, "end");
        assert_eq!(drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().selected), total - 1);
        press(&mut app, window, "home");
        assert_eq!(drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().selected), 0);

        // With a non-empty query, plain End stays with the text input; Ctrl+End still jumps the list.
        let bench_items = drive(&mut app, window, |this, _, cx| {
            let p = this.palette.as_mut().unwrap();
            p.query = "a".into();
            p.input.update(cx, |i, cx| i.set_text("a", cx));
            p.selected = 0;
            this.filtered_palette(theme(cx).id).len()
        });
        assert!(bench_items > 1);
        app.frame(window.into());
        press(&mut app, window, "end");
        assert_eq!(
            drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().selected),
            0,
            "plain End must not steal cursor movement while typing a query"
        );
        press(&mut app, window, "ctrl-end");
        assert_eq!(
            drive(&mut app, window, |this, _, _| this.palette.as_ref().unwrap().selected),
            bench_items - 1,
            "Ctrl+End still jumps to the last palette item"
        );

        // Alt+Right while palette is open must not navigate underneath and strand palette focus.
        let route_before = drive(&mut app, window, |this, _, _| {
            this.forward_history.push(Route::Runs);
            this.route.clone()
        });
        press(&mut app, window, "alt-right");
        drive(&mut app, window, |this, _, _| {
            assert_eq!(this.route, route_before, "Alt+Right ignored while palette is open");
            assert!(this.palette.is_some());
        });

        // Benchmarks appear in the palette and activate into Route::Bench.
        drive(&mut app, window, |this, w, cx| {
            let items = this.palette_items(theme(cx).id);
            let b = items
                .into_iter()
                .find(|i| i.group == "Benchmarks" && i.title == first_bench_name)
                .expect("benchmark indexed in palette");
            this.close_palette(w, cx);
            (b.action)(this, w, cx);
        });
        assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::Bench(first_bench_id));
    }
}
