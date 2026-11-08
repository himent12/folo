//! FOLO Desktop: the native GPUI front end for FOLO.
//!
//! `folo-desktop [--scenario NAME] [--width W --height H] [--theme dark|light]`
//!
//! Scenarios put the app into a fixed, deterministic state (frozen mock clock, a screen, an
//! overlay, a hover) for screenshots; see `scenario.rs` and `tools/shoot-all.ps1`.

mod api;
mod app;
mod charts;
mod controls;
mod input;
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

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
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
        let mut mock = MockBackend::new(live_step, frozen);
        if spec.as_ref().is_some_and(|s| s.fail) {
            mock = mock.failing();
        }
        if spec.as_ref().is_some_and(|s| s.empty) {
            mock = mock.empty();
        }
        let api: Arc<dyn api::Backend> = Arc::new(mock);

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
                        if let Some(s) = &spec {
                            scenario::apply(&mut app, s, window, cx);
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
