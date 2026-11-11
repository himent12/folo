//! `--shot <png>`: the app runs headless, settles, writes its window as a PNG and returns from `run`.
//! Owner tag: `"runtime"`.
//!
//! `Application::new` reads the process arguments, so the app needs no code for it: `--shot <png>` with
//! `--width` / `--height` (logical px, default 1280 x 800, the app's defaults), `--scale` (default 1) and
//! `--text portable` (cosmic-text instead of the platform's text backend). Settling: the app's launch
//! callback runs, the queues are pumped until idle (background jobs included; the clock stays frozen), and
//! frames are drawn until two in a row are identical or the window stops being dirty, at most
//! `MAX_SETTLE_FRAMES`. The PNG is `width * scale` x `height * scale` pixels.

use std::{path::PathBuf, rc::Rc};

use anyhow::{Context as _, Result, anyhow};

use crate::{
    AppCell, Framebuffer, HeadlessApp, HeadlessOptions, TextBackend, platform::headless::HeadlessPlatform,
    px, size,
};

/// The most frames a screenshot draws while settling.
const MAX_SETTLE_FRAMES: usize = 30;

/// A `--shot` request from the command line.
pub(crate) struct ShotArgs {
    /// Where the PNG goes.
    pub(crate) path: PathBuf,
    /// The headless window's size, scale and text backend.
    pub(crate) options: HeadlessOptions,
}

impl ShotArgs {
    /// The request in `args` (the process arguments, program name first), if `--shot` is there.
    pub(crate) fn parse(args: &[String]) -> Result<Option<Self>> {
        let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
        let Some(path) = value("--shot") else { return Ok(None) };
        let number = |name: &str, default: f32| -> Result<f32> {
            value(name).map_or(Ok(default), |v| v.parse().with_context(|| format!("{name} {v}")))
        };
        let text = match value("--text").map(String::as_str) {
            None | Some("native") => TextBackend::Native,
            Some("portable") => TextBackend::Portable,
            Some(other) => return Err(anyhow!("--text {other}: expected `native` or `portable`")),
        };
        let scale_factor = number("--scale", 1.0)?;
        anyhow::ensure!(scale_factor > 0.0, "--scale must be positive");
        Ok(Some(ShotArgs {
            path: PathBuf::from(path),
            options: HeadlessOptions {
                size: size(px(number("--width", 1280.0)?), px(number("--height", 800.0)?)),
                scale_factor,
                text,
            },
        }))
    }
}

/// A screenshot run: the request and the headless platform it runs on.
pub(crate) struct Shot {
    /// Where the PNG goes.
    pub(crate) path: PathBuf,
    /// The platform (for its dispatcher).
    pub(crate) platform: Rc<HeadlessPlatform>,
}

impl Shot {
    /// Settles the app's first window and writes it to `self.path`.
    pub(crate) fn take(self, app: Rc<AppCell>) -> Result<()> {
        let mut harness = HeadlessApp::from_parts(app, self.platform);
        harness.run_until_idle();
        let window = *harness.windows().first().context("the app opened no window")?;
        if !harness.settle(window, MAX_SETTLE_FRAMES) {
            eprintln!("folo-ui: --shot did not settle in {MAX_SETTLE_FRAMES} frames; writing the last one");
        }
        let frame = harness.frame(window).clone();
        write_png(&self.path, &frame)
    }
}

/// Writes a framebuffer as a PNG (the raster package's writer), creating the folder.
fn write_png(path: &std::path::Path, frame: &Framebuffer) -> Result<()> {
    anyhow::ensure!(!frame.is_empty(), "the window has no pixels ({}x{})", frame.width, frame.height);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    crate::raster::debug::write_png(frame, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("folo-desktop").chain(list.iter().copied()).map(String::from).collect()
    }

    #[test]
    fn parses_the_shot_arguments() {
        assert!(ShotArgs::parse(&args(&["--scenario", "runs"])).unwrap().is_none());
        let shot = ShotArgs::parse(&args(&[
            "--shot",
            "out/a.png",
            "--width",
            "960",
            "--scale",
            "1.5",
            "--text",
            "portable",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(shot.path, PathBuf::from("out/a.png"));
        assert_eq!(shot.options.size, size(px(960.), px(800.)));
        assert_eq!(shot.options.scale_factor, 1.5);
        assert_eq!(shot.options.text, TextBackend::Portable);
        assert!(ShotArgs::parse(&args(&["--shot", "a.png", "--scale", "x"])).is_err());
        assert!(ShotArgs::parse(&args(&["--shot", "a.png", "--text", "dwrite"])).is_err());
    }
}
