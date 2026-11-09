//! FOLO Desktop: the native GPUI front end for FOLO.
//!
//! `folo-desktop [--server URL | --mock] [--open SCREEN] [--scenario NAME] [--width W --height H]
//! [--theme dark|light]`; `--help` and `--version` print and exit (see [`USAGE`]).
//!
//! Scenarios put the app into a fixed, deterministic state (frozen mock clock, a screen, an
//! overlay, a hover) for screenshots; see `scenario.rs` and `tools/shoot-all.ps1`.

mod api;
mod app;
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

use std::borrow::Cow;
use std::sync::Arc;

use gpui::{
    App, AppContext, Application, AssetSource, Bounds, SharedString, TitlebarOptions, WindowBounds,
    WindowOptions, point, px, size,
};

use crate::api::mock::MockBackend;
use crate::theme::{Mode, Theme};

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

By default it connects to a running `folo serve` ($FOLO_SERVER, else http://127.0.0.1:7878);
when none answers it starts `folo serve --mock`, and without `folo` it uses built-in demo data.

Options:
  --server URL          Connect to the folo serve at URL
  --mock                Use the built-in demo data
  --open SCREEN         Start on a screen: overview, datasets, dataset:ID, runs, run:ID, models,
                        model:ID, playground, settings, or new-run (the dialog)
  --theme dark|light    Colour theme [default: dark]
  --width W             Window width in px [default: 1280]
  --height H            Window height in px [default: 800]
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
    let mode = match arg("--theme").as_deref() {
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
        let auto = spec.is_none() && !use_mock && server.is_none();
        let api: Arc<dyn api::Backend> = match (&spec, &server) {
            (None, Some(url)) => Arc::new(api::http::HttpBackend::new(url)),
            _ => Arc::new(mock),
        };
        let server_mode = if server.is_some() { app::ServerMode::Local } else { app::ServerMode::Mock };
        let connect_url = server.clone().unwrap_or(default_url.clone());

        let bounds = Bounds::centered(None, size(px(width), px(height)), cx);
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
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
                        app.server_mode = server_mode;
                        app.server_url.update(cx, |i, cx| i.set_text(connect_url.clone(), cx));
                        if let Some(s) = &spec {
                            scenario::apply(&mut app, s, window, cx);
                        }
                        if auto {
                            app.autoconnect(connect_url.clone(), true, cx);
                        }
                        if open.as_deref() == Some("new-run") {
                            app.open_new_run_pending = true;
                        } else if let Some(route) = open.as_deref().and_then(app::Route::parse) {
                            app.route = route;
                        }
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
