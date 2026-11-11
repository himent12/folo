//! FOLO Desktop: the native front end for FOLO, on folo-ui (`ui/`).
//!
//! `folo-desktop [--server URL | --mock] [--open SCREEN] [--scenario NAME] [--width W --height H]
//! [--theme dark|light]`; `--help` and `--version` print and exit (see [`USAGE`]).
//!
//! Scenarios put the app into a fixed, deterministic state (frozen mock clock, a screen, an
//! overlay, a hover) for screenshots; see `scenario.rs` and `tools/shoot-all.ps1`.

mod api;
mod app;
mod bench;
mod charts;
mod cloud;
mod controls;
mod input;
mod local;
mod overlays;
mod scenario;
mod screens;
mod store;
mod synth;
mod theme;
mod ui;
mod viz;
mod window_state;

use std::borrow::Cow;
use std::sync::Arc;

use folo_ui::{
    App, AppContext, Application, AssetSource, Bounds, SharedString, TitlebarOptions, WindowBounds,
    WindowOptions, point, px, size,
};

use crate::api::mock::MockBackend;
use crate::theme::{Mode, Theme};

/// Allocation counts for `--bench-frames` (`cargo build --release --features bench-alloc`).
#[cfg(feature = "bench-alloc")]
#[global_allocator]
static ALLOC: folo_ui::CountingAlloc = folo_ui::CountingAlloc;

mod icons {
    include!(concat!(env!("OUT_DIR"), "/icons.rs"));
}

struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        Ok(icons::ICONS.iter().find(|(name, _)| *name == path).map(|(_, bytes)| Cow::Borrowed(*bytes)))
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(icons::ICONS.iter().map(|(name, _)| (*name).into()).collect())
    }
}

/// Geist and Geist Mono from the shared design system (`design/fonts`, OFL).
const FONTS: &[&[u8]] = &[
    include_bytes!("../../../design/fonts/Geist-Regular.ttf"),
    include_bytes!("../../../design/fonts/Geist-Medium.ttf"),
    include_bytes!("../../../design/fonts/Geist-SemiBold.ttf"),
    include_bytes!("../../../design/fonts/GeistMono-Regular.ttf"),
    include_bytes!("../../../design/fonts/GeistMono-Medium.ttf"),
    include_bytes!("../../../design/fonts/GeistMono-SemiBold.ttf"),
];

/// `folo-desktop --help`.
const USAGE: &str = "FOLO Desktop: datasets, training runs and live inference for FOLO, natively.

Usage: folo-desktop [OPTIONS]

By default it connects to a running `folo serve` ($FOLO_SERVER, else http://127.0.0.1:7878),
or starts a local `folo serve` when none answers. Pass --mock for the built-in demo data.
The window and theme open as they were last closed ($FOLO_HOME or ~/.folo, desktop.json), except
with --scenario, --width or --height.

Options:
  --server URL          Connect to the folo serve at URL
  --mock                Use the built-in demo data
  --open SCREEN         Start on a screen: overview, datasets, dataset:ID, runs, run:ID, models,
                        model:ID, playground, settings, or new-run (the dialog)
  --theme dark|light    Colour theme [default: the last one, else dark]
  --width W             Window width in px [default: the last size, else 1280]
  --height H            Window height in px [default: the last size, else 800]
  --scenario NAME       A fixed, deterministic state for screenshots (implies demo data)
  --list-scenarios      Print the scenario names and exit
  -h, --help            Print this help and exit
  -V, --version         Print the version and exit
";

/// What to print instead of opening the app (`--help`, `--version`), if `args` (without the program name) ask
/// for it.
fn info_text<S: AsRef<str>>(args: &[S]) -> Option<String> {
    let has = |flags: &[&str]| args.iter().any(|a| flags.contains(&a.as_ref()));
    if has(&["-h", "--help"]) {
        Some(USAGE.to_string())
    } else if has(&["-V", "--version"]) {
        Some(format!(
            "folo-desktop {}
",
            env!("CARGO_PKG_VERSION")
        ))
    } else {
        None
    }
}

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(text) = info_text(&args) {
        print!("{text}");
        return;
    }
    if std::env::args().any(|a| a == "--list-scenarios") {
        for s in scenario::ALL {
            println!("{s}");
        }
        return;
    }
    let scenario = arg("--scenario");
    let width: f32 = arg("--width").and_then(|w| w.parse().ok()).unwrap_or(1280.0);
    let height: f32 = arg("--height").and_then(|h| h.parse().ok()).unwrap_or(800.0);
    // The window and theme from the last launch, unless this one is for a screenshot.
    let remember =
        window_state::remembers(scenario.is_some(), arg("--width").is_some() || arg("--height").is_some());
    let state_path = remember.then(window_state::path);
    let saved = state_path.as_deref().map(window_state::load).unwrap_or_default();
    let mode = match arg("--theme").as_deref().or(saved.theme.as_deref()) {
        Some("light") => Mode::Light,
        _ => Mode::Dark,
    };
    let spec = scenario.as_deref().map(scenario::Spec::parse);
    // Data source: `--server URL` connects to a running folo serve; `--mock` uses the built-in demo
    // data; by default the app finds or starts a server (see `FoloApp::autoconnect`).
    let server = arg("--server");
    // `--open run:run_3c02` starts on a screen (also `dataset:ID`, `model:ID`, `runs`, …).
    let open = arg("--open");
    let use_mock = std::env::args().any(|a| a == "--mock");
    let default_url = std::env::var("FOLO_SERVER").unwrap_or_else(|_| "http://127.0.0.1:7878".into());

    Application::new().with_assets(Assets).run(move |cx: &mut App| {
        cx.text_system()
            .add_fonts(FONTS.iter().map(|f| Cow::Borrowed(*f)).collect())
            .expect("bundled fonts load");
        let mode = spec.as_ref().and_then(|s| s.mode).unwrap_or(mode);
        cx.set_global(Theme::new(mode));
        app::bind_keys(cx);
        input::bind_keys(cx);

        let frozen = spec.is_some();
        let use_demo = spec.is_some() || use_mock;
        let auto = !use_demo && server.is_none();
        let connect_url = server.clone().unwrap_or(default_url.clone());
        let api: Arc<dyn api::Backend> = if use_demo {
            let live_step = spec.as_ref().map_or(4180, |s| s.live_step);
            let mut mock = if spec.as_ref().is_some_and(|s| s.cloud_runs) {
                MockBackend::with_cloud_runs(live_step, frozen)
            } else {
                MockBackend::new(live_step, frozen)
            };
            if spec.as_ref().is_some_and(|s| s.fail) {
                mock = mock.failing();
            }
            if spec.as_ref().is_some_and(|s| s.empty) {
                mock = mock.empty().without_cloud();
            }
            if spec.as_ref().is_some_and(|s| s.has("trouble")) {
                mock = mock.cloud_trouble();
            }
            Arc::new(mock)
        } else {
            Arc::new(api::http::HttpBackend::new(&connect_url))
        };
        let server_mode = if use_demo { app::ServerMode::Mock } else { app::ServerMode::Local };
        let spawn_server = saved.spawn_server.unwrap_or(true);

        let (window_bounds, display_id) = match saved.window {
            Some(p) => restored_bounds(p, cx),
            None => (WindowBounds::Windowed(Bounds::centered(None, size(px(width), px(height)), cx)), None),
        };
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(window_bounds),
                    display_id,
                    titlebar: Some(TitlebarOptions {
                        title: Some("FOLO Desktop".into()),
                        appears_transparent: true,
                        traffic_light_position: Some(point(px(12.), px(12.))),
                    }),
                    window_min_size: Some(size(px(960.), px(620.))),
                    app_id: Some("folo-desktop".into()),
                    ..Default::default()
                },
                |window, cx| {
                    cx.new(|cx| {
                        let mut app = app::FoloApp::new(api, frozen, window, cx);
                        app.state_path = state_path.clone();
                        app.server_mode = server_mode;
                        app.spawn_server = spawn_server;
                        app.server_url.update(cx, |i, cx| i.set_text(connect_url.clone(), cx));
                        if let Some(s) = &spec {
                            scenario::apply(&mut app, s, window, cx);
                        }
                        if auto {
                            app.autoconnect(connect_url.clone(), spawn_server, cx);
                        }
                        if open.as_deref() == Some("new-run") {
                            app.open_new_run_pending = true;
                        } else if let Some(route) = open.as_deref().and_then(app::Route::parse) {
                            app.route = route;
                        }
                        let this = cx.weak_entity();
                        window.on_window_should_close(cx, move |window, cx| {
                            this.update(cx, |app, cx| app.save_window_state(window, cx)).ok();
                            true
                        });
                        app
                    })
                },
            )
            .expect("open window");
        let _ = window;
        cx.on_window_closed(|cx| cx.quit()).detach();
        cx.activate(true);
    });
}

/// Where a remembered window opens: its saved bounds on the display it was on (matched by UUID) or
/// another that still shows its title bar, else centred on the primary display
/// (`window_state::place`), maximized or full screen as it was.
fn restored_bounds(p: window_state::Placement, cx: &App) -> (WindowBounds, Option<folo_ui::DisplayId>) {
    use window_state::{Rect, WindowMode};
    let rect = |b: Bounds<folo_ui::Pixels>| Rect {
        x: b.origin.x.into(),
        y: b.origin.y.into(),
        width: b.size.width.into(),
        height: b.size.height.into(),
    };
    let displays = cx.displays();
    let primary = cx.primary_display().map(|p| p.id());
    let primary = displays.iter().position(|d| Some(d.id()) == primary).unwrap_or(0);
    let rects: Vec<Rect> = displays.iter().map(|d| rect(d.bounds())).collect();
    let remembered = p
        .display
        .as_deref()
        .and_then(|uuid| displays.iter().position(|d| d.uuid().is_ok_and(|u| u.to_string() == uuid)));
    let saved = Rect { x: p.x, y: p.y, width: p.width, height: p.height };
    let (i, r) = window_state::place(saved, &rects, primary, remembered, theme::size::TOPBAR);
    let bounds = Bounds::new(point(px(r.x), px(r.y)), size(px(r.width), px(r.height)));
    let window_bounds = match p.mode {
        WindowMode::Windowed => WindowBounds::Windowed(bounds),
        WindowMode::Maximized => WindowBounds::Maximized(bounds),
        WindowMode::Fullscreen => WindowBounds::Fullscreen(bounds),
    };
    (window_bounds, displays.get(i).map(|d| d.id()))
}

#[cfg(test)]
mod source_tests {
    /// Catches text double-encoded by an editor or shell (UTF-8 read as Windows-1252) and stray BOMs.
    #[test]
    fn sources_are_clean_utf8() {
        fn walk(dir: &std::path::Path, bad: &mut Vec<String>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, bad);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let s = std::fs::read_to_string(&p).unwrap();
                    if s.contains(['\u{00C3}', '\u{00C2}', '\u{00E2}', '\u{FEFF}']) {
                        bad.push(p.display().to_string());
                    }
                }
            }
        }
        let mut bad = Vec::new();
        walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut bad);
        assert!(bad.is_empty(), "mojibake or BOM in {bad:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_and_version_print_instead_of_opening() {
        for flag in ["--help", "-h"] {
            let text = info_text(&["--mock", flag]).unwrap();
            assert!(text.starts_with("FOLO Desktop") && text.contains("--server URL"), "{text}");
        }
        for flag in ["--version", "-V"] {
            assert_eq!(
                info_text(&[flag]).unwrap(),
                format!(
                    "folo-desktop {}
",
                    env!("CARGO_PKG_VERSION")
                )
            );
        }
        assert!(info_text(&["--help", "--version"]).unwrap().contains("Usage:"), "help wins");
        assert_eq!(info_text(&["--mock", "--open", "runs"]), None);
        assert_eq!(info_text::<&str>(&[]), None);
    }
}
