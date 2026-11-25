//! `--shot <png>`: the app runs headless, settles, writes its window as a PNG and returns from `run`.
//! Owner tag: `"runtime"`.
//!
//! `Application::new` reads the process arguments, so the app needs no code for it: `--shot <png>` with
//! `--width` / `--height` (logical px, default 1280 x 800, the app's defaults), `--scale` (default 1) and
//! `--text portable` (cosmic-text instead of the platform's text backend) and `--shot-time <ms>`. Settling:
//! the app's launch callback runs, the queues are pumped until idle (background jobs included), then frames
//! are drawn while the fake clock moves in 50 ms steps until the window stops asking for frames (one-shot
//! fades have finished) or `--shot-time` has passed (default 2200 ms, the delay `tools/shoot.ps1` waited
//! before capturing GPUI: repeating pulses and shimmers are caught at that time). The app's frozen
//! scenarios run no timers of their own, so only animations see the clock move. The PNG is
//! `width * scale` x `height * scale` pixels. With `FOLO_UI_SCENE_DUMP=<file>` the frame's primitives
//! are also written there, one per line (`Debug`) after the glyph blend, for parity work.

use std::{path::PathBuf, rc::Rc, time::Duration};

use anyhow::{Context as _, Result, anyhow};

use crate::{
    AppCell, Framebuffer, HeadlessApp, HeadlessOptions, TextBackend, platform::headless::HeadlessPlatform,
    px, size,
};

/// The fake-clock step while settling.
pub(crate) const SETTLE_STEP: Duration = Duration::from_millis(50);

/// How long animations may run before the capture, by default.
pub(crate) const SETTLE_TIME: Duration = Duration::from_millis(2200);

/// A `--shot` request from the command line.
pub(crate) struct ShotArgs {
    /// Where the PNG goes.
    pub(crate) path: PathBuf,
    /// The headless window's size, scale and text backend.
    pub(crate) options: HeadlessOptions,
    /// How long animations may run before the capture.
    pub(crate) settle_time: Duration,
}

/// Reads a flag's value from `args` (program name first): `Ok(None)` when the flag is absent,
/// `Err` when any occurrence lacks a value (trailing, empty, or followed by another `--` flag).
pub(crate) fn flag_value<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>> {
    let rest = args.get(1..).unwrap_or(&[]);
    let mut found = None;
    for (i, arg) in rest.iter().enumerate() {
        if arg != name {
            continue;
        }
        match rest.get(i + 1).map(String::as_str) {
            None => return Err(anyhow!("{name}: missing its value")),
            Some("") => return Err(anyhow!("{name}: value cannot be empty")),
            Some(v) if v.starts_with("--") => return Err(anyhow!("{name}: missing its value (got `{v}`)")),
            Some(v) => {
                if found.is_none() {
                    found = Some(v);
                }
            }
        }
    }
    Ok(found)
}

impl ShotArgs {
    /// The request in `args` (the process arguments, program name first), if `--shot` is there.
    pub(crate) fn parse(args: &[String]) -> Result<Option<Self>> {
        let Some(path) = flag_value(args, "--shot")? else { return Ok(None) };
        let number = |name: &str, default: f32| -> Result<f32> {
            match flag_value(args, name)? {
                None => Ok(default),
                Some(v) => v.parse().with_context(|| format!("{name} {v}")),
            }
        };
        let text = match flag_value(args, "--text")? {
            None | Some("native") => TextBackend::Native,
            Some("portable") => TextBackend::Portable,
            Some(other) => return Err(anyhow!("--text {other}: expected `native` or `portable`")),
        };
        let scale_factor = number("--scale", 1.0)?;
        let (width, height) = (number("--width", 1280.0)?, number("--height", 800.0)?);
        check_size(width, height, scale_factor)?;
        let settle_ms = number("--shot-time", SETTLE_TIME.as_millis() as f32)?;
        anyhow::ensure!(
            (0.0..=MAX_SETTLE_MS).contains(&settle_ms),
            "--shot-time {settle_ms}: expected 0 to {MAX_SETTLE_MS} ms"
        );
        Ok(Some(ShotArgs {
            settle_time: Duration::from_millis(settle_ms as u64),
            path: PathBuf::from(path),
            options: HeadlessOptions { size: size(px(width), px(height)), scale_factor, text },
        }))
    }
}

/// The longest `--shot-time`: ten minutes of animation (a repeating pulse asks for frames for as long
/// as it may, 50 ms apiece; `1e12` never finished).
const MAX_SETTLE_MS: f32 = 600_000.0;

/// The most device pixels a headless frame may have on a side (a 16384 x 16384 frame is 1 GiB).
pub(crate) const MAX_SIDE: f32 = 16384.0;

/// Checks a headless window's `--width`, `--height` (logical px) and `--scale` up front: finite, at
/// least 1 px, a scale above 0 and at most 8, and 1 to [`MAX_SIDE`] device pixels a side, rounded as
/// the framebuffer is (`Window::resize_framebuffer`). (`inf` asked for a 13 TB frame and aborted, a
/// scale of `inf` panicked; 0, negative, NaN and a scale that rounds a side to 0 pixels failed late.)
pub(crate) fn check_size(width: f32, height: f32, scale: f32) -> Result<()> {
    anyhow::ensure!(scale > 0.0 && scale <= 8.0, "--scale {scale}: expected more than 0 and at most 8");
    for (name, v) in [("--width", width), ("--height", height)] {
        let device = (v * scale).round();
        anyhow::ensure!(
            v >= 1.0 && (1.0..=MAX_SIDE).contains(&device),
            "{name} {v}: expected {} to {} px at --scale {scale}",
            (0.5 / scale).ceil().max(1.0),
            (MAX_SIDE / scale).floor()
        );
    }
    Ok(())
}

/// A screenshot run: the request and the headless platform it runs on.
pub(crate) struct Shot {
    /// Where the PNG goes.
    pub(crate) path: PathBuf,
    /// The platform (for its dispatcher).
    pub(crate) platform: Rc<HeadlessPlatform>,
    /// How long animations may run before the capture.
    pub(crate) settle_time: Duration,
}

impl Shot {
    /// Settles the app's first window and writes it to `self.path`.
    pub(crate) fn take(self, app: Rc<AppCell>) -> Result<()> {
        let mut harness = HeadlessApp::from_parts(app, self.platform);
        harness.run_until_idle();
        let window = *harness.windows().first().context("the app opened no window")?;
        harness.settle_advancing(window, self.settle_time, SETTLE_STEP);
        let frame = harness.frame(window).clone();
        if let Some(dump) = std::env::var_os("FOLO_UI_SCENE_DUMP") {
            // The glyph blend, then one primitive per line, in painter's order (parity work: which
            // primitive drew a pixel, and with what).
            let blend = harness.read(|cx| cx.text_system().glyph_blend());
            let text: String = std::iter::once(format!("{blend:?}\n"))
                .chain(harness.scene(window).primitives.iter().map(|p| format!("{p:?}\n")))
                .collect();
            std::fs::write(&dump, text).with_context(|| format!("writing {}", dump.display()))?;
        }
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

    /// Sizes and times that crashed, hung or failed late (stress d5-window-input-3) are refused up front.
    #[test]
    fn refuses_sizes_and_times_it_cannot_render() {
        let shot = |extra: &[&str]| {
            let mut list = vec!["--shot", "a.png"];
            list.extend_from_slice(extra);
            ShotArgs::parse(&args(&list))
        };
        for bad in [
            &["--width", "inf"][..],
            &["--width", "NaN"],
            &["--width", "0"],
            &["--height", "-5"],
            &["--width", "20000"],
            &["--width", "9000", "--scale", "2"],
            &["--scale", "inf"],
            &["--scale", "NaN"],
            &["--scale", "0"],
            &["--scale", "9"],
            // Rounds to 0 x 0 device pixels: "the window has no pixels" after settling.
            &["--scale", "0.0001"],
            &["--width", "1", "--scale", "0.4"],
            &["--shot-time", "1e12"],
            &["--shot-time", "NaN"],
            &["--shot-time", "-1"],
        ] {
            assert!(shot(bad).is_err(), "{bad:?}");
        }
        for good in [
            &["--width", "1", "--height", "1"][..],
            &["--width", "16384"],
            &["--scale", "8"],
            &["--scale", "0.001"],
            &["--width", "2", "--height", "2", "--scale", "0.25"],
            &["--shot-time", "0"],
        ] {
            assert!(shot(good).unwrap().is_some(), "{good:?}");
        }
    }

    #[test]
    fn fails_closed_when_shot_or_its_options_lack_a_value() {
        for bad in [
            &["--shot"][..],
            &["--scenario", "runs", "--shot"],
            &["--shot", ""],
            &["--shot", "--width", "960"],
            &["--shot", "a.png", "--width"],
            &["--shot", "a.png", "--height", "--scale", "2"],
            &["--shot", "a.png", "--scale"],
            &["--shot", "a.png", "--text"],
            &["--shot", "a.png", "--shot-time"],
            &["--shot", "a.png", "--shot"],
            &["--shot", "a.png", "--width", "960", "--width"],
        ] {
            assert!(ShotArgs::parse(&args(bad)).is_err(), "{bad:?}");
        }
        // Repeated valid flags keep first-value precedence while still validating every occurrence.
        let repeated =
            ShotArgs::parse(&args(&["--shot", "first.png", "--shot", "second.png"])).unwrap().unwrap();
        assert_eq!(repeated.path, PathBuf::from("first.png"));
    }
}
