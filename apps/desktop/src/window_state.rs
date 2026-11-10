//! What the app remembers between launches: the window's size, position and maximized / full-screen
//! state, and the theme. One small JSON file, `<FOLO_HOME or ~/.folo>/desktop.json`
//! (`$FOLO_DESKTOP_STATE` names another, as `tools/window-check.ps1` does).
//!
//! Screenshots stay deterministic: with `--scenario`, `--width` or `--height` nothing is read or
//! written ([`remembers`]). A missing or unreadable file is a first launch; writes go to a temporary
//! file renamed over the old one, so a crash mid-write never leaves half a file.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The file's contents. Every field is optional, so an older or newer file still loads.
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct State {
    pub window: Option<Placement>,
    /// `"dark"` or `"light"`.
    pub theme: Option<String>,
}

/// The window in logical pixels: its restored (windowed) bounds, and how it was showing.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    #[serde(default)]
    pub mode: WindowMode,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum WindowMode {
    #[default]
    Windowed,
    Maximized,
    Fullscreen,
}

/// A rectangle in logical pixels (a window, a display).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    fn intersection(&self, o: &Rect) -> Option<Rect> {
        let (x0, y0) = (self.x.max(o.x), self.y.max(o.y));
        let (x1, y1) = ((self.x + self.width).min(o.x + o.width), (self.y + self.height).min(o.y + o.height));
        (x1 > x0 && y1 > y0).then_some(Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 })
    }
}

/// How much of the title bar must be on a display to grab it: this wide, the bar's height tall.
const GRAB_WIDTH: f32 = 96.;

/// Whether this launch reads and writes the state file: not for a `--scenario`, and not when
/// `--width` / `--height` fix the window size (both are for screenshots).
pub fn remembers(scenario: bool, size_given: bool) -> bool {
    !scenario && !size_given
}

/// The mode to remember on close. A minimized window reports itself windowed (Windows has no
/// "minimized from maximized" in what GPUI returns), and GPUI draws no frame while minimized, so the
/// mode of the last frame drawn (`shown`) is how it will come back: use it when the window says
/// windowed. Full screen survives minimizing in GPUI's own report.
pub fn closing_mode(reported: WindowMode, shown: WindowMode) -> WindowMode {
    match reported {
        WindowMode::Windowed => shown,
        other => other,
    }
}

/// `$FOLO_DESKTOP_STATE`, else `desktop.json` in FOLO's home folder.
pub fn path() -> PathBuf {
    match std::env::var_os("FOLO_DESKTOP_STATE") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => folo_api::weights::home_dir().join("desktop.json"),
    }
}

/// The saved state; a missing or corrupt file reads as nothing saved.
pub fn load(path: &Path) -> State {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// Write `state` atomically: a temporary file next to `path`, then a rename over it.
pub fn save(path: &Path, state: &State) -> io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let json = serde_json::to_vec_pretty(state).map_err(io::Error::other)?;
    let written = std::fs::write(&tmp, json).and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// Where a saved window opens, given the displays (`displays[primary]` the primary one) and the
/// title bar's height: on the display that holds the largest grabbable piece of its title bar,
/// shrunk to fit that display; when no display has one (the monitor it was on is gone, the
/// resolution dropped), centred on the primary display. Returns the display's index and the bounds.
pub fn place(saved: Rect, displays: &[Rect], primary: usize, bar: f32) -> (usize, Rect) {
    let title = Rect { height: bar.min(saved.height), ..saved };
    let on = displays
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            let part = title.intersection(d)?;
            (part.width >= GRAB_WIDTH.min(saved.width) && part.height >= title.height / 2.)
                .then_some((i, part.width))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);
    let index = on.unwrap_or(primary.min(displays.len().saturating_sub(1)));
    let Some(d) = displays.get(index) else {
        return (0, saved);
    };
    let (width, height) = (saved.width.min(d.width), saved.height.min(d.height));
    let rect = match on {
        // Still on screen: keep the position, pulled in only until the centre is on the display
        // (GPUI opens bounds whose centre is elsewhere at its own default place instead).
        // `max` then `min`, not `clamp`, so a tiny display can't panic.
        Some(_) => Rect {
            x: saved.x.max(d.x - width / 2. + 1.).min(d.x + d.width - width / 2. - 1.),
            y: saved.y.max(d.y - bar / 2.).min(d.y + d.height - height / 2. - 1.),
            width,
            height,
        },
        None => Rect { x: d.x + (d.width - width) / 2., y: d.y + (d.height - height) / 2., width, height },
    };
    (index, rect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    const BAR: f32 = 40.;

    fn r(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect { x, y, width, height }
    }

    #[test]
    fn saved_only_without_scenario_or_size() {
        assert!(remembers(false, false));
        assert!(!remembers(true, false));
        assert!(!remembers(false, true));
        assert!(!remembers(true, true));
    }

    #[test]
    fn a_minimized_window_keeps_the_mode_it_was_shown_in() {
        use WindowMode::*;
        // Minimized from maximized: reported windowed, last drawn maximized.
        assert_eq!(closing_mode(Windowed, Maximized), Maximized);
        assert_eq!(closing_mode(Windowed, Windowed), Windowed);
        assert_eq!(closing_mode(Fullscreen, Maximized), Fullscreen);
        assert_eq!(closing_mode(Maximized, Windowed), Maximized);
    }

    #[test]
    fn round_trips_and_tolerates_missing_or_corrupt_files() {
        let dir = TestDir::new("desktop-state");
        let path = dir.path().join("nested").join("desktop.json");
        assert_eq!(load(&path), State::default(), "missing file");
        let state = State {
            window: Some(Placement {
                x: 10.,
                y: -20.5,
                width: 1400.,
                height: 900.,
                mode: WindowMode::Maximized,
            }),
            theme: Some("light".into()),
        };
        save(&path, &state).unwrap();
        assert_eq!(load(&path), state);
        // Saving again replaces the file and leaves no temporary file behind.
        let state2 = State { theme: Some("dark".into()), ..state.clone() };
        save(&path, &state2).unwrap();
        assert_eq!(load(&path), state2);
        let names: Vec<_> =
            std::fs::read_dir(path.parent().unwrap()).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(names, ["desktop.json"]);

        for junk in ["", "{", "[1,2]", "null", "{\"window\": {\"x\": \"left\"}}"] {
            std::fs::write(&path, junk).unwrap();
            assert_eq!(load(&path), State::default(), "{junk:?}");
        }
        // Unknown fields and a missing mode are fine.
        std::fs::write(&path, r#"{"window":{"x":1,"y":2,"width":3,"height":4},"later":true}"#).unwrap();
        assert_eq!(
            load(&path).window,
            Some(Placement { x: 1., y: 2., width: 3., height: 4., mode: WindowMode::Windowed })
        );
    }

    #[test]
    fn mode_is_lowercase_in_the_file() {
        let p = Placement { x: 0., y: 0., width: 1., height: 1., mode: WindowMode::Fullscreen };
        assert!(serde_json::to_string(&p).unwrap().contains("\"mode\":\"fullscreen\""));
    }

    #[test]
    fn a_window_on_screen_stays_where_it_was() {
        let displays = [r(0., 0., 2560., 1400.), r(2560., 0., 1920., 1080.)];
        let w = r(300., 200., 1280., 800.);
        assert_eq!(place(w, &displays, 0, BAR), (0, w));
        // A little above the top edge, but most of the title bar shows.
        let up = r(100., -10., 1280., 800.);
        assert_eq!(place(up, &displays, 0, BAR), (0, up));
        // On the second display, and partly off its right edge.
        let w2 = r(3500., 100., 1280., 800.);
        assert_eq!(place(w2, &displays, 0, BAR), (1, w2));
        // So far off it that the centre isn't on it: pulled back until the centre is.
        assert_eq!(place(r(3900., 100., 1280., 800.), &displays, 0, BAR), (1, r(3839., 100., 1280., 800.)));
        // Straddling both: the display with more of the title bar wins.
        assert_eq!(place(r(2000., 100., 1280., 800.), &displays, 0, BAR).0, 1);
        assert_eq!(place(r(1800., 100., 1280., 800.), &displays, 0, BAR).0, 0);
    }

    #[test]
    fn a_window_off_every_display_is_centred_on_the_primary_one() {
        let displays = [r(-1920., 0., 1920., 1080.), r(0., 0., 2560., 1440.)];
        let w = r(5000., 300., 1280., 800.);
        assert_eq!(place(w, &displays, 1, BAR), (1, r(640., 320., 1280., 800.)));
        // Title bar above the top of every display: the window can't be grabbed.
        assert_eq!(place(r(100., -500., 1280., 800.), &displays, 1, BAR).1, r(640., 320., 1280., 800.));
        // Only a sliver of the title bar showing counts as off screen too.
        assert_eq!(
            place(r(2500., 100., 1280., 800.), &[r(0., 0., 2560., 1440.)], 0, BAR).1,
            r(640., 320., 1280., 800.)
        );
    }

    #[test]
    fn a_window_bigger_than_its_display_shrinks_to_fit() {
        let displays = [r(0., 0., 1366., 768.)];
        assert_eq!(place(r(0., 0., 2000., 1200.), &displays, 0, BAR), (0, r(0., 0., 1366., 768.)));
        let (_, off) = place(r(-3000., 0., 2000., 1200.), &displays, 0, BAR);
        assert_eq!(off, r(0., 0., 1366., 768.));
    }

    #[test]
    fn no_displays_keeps_the_saved_bounds() {
        let w = r(1., 2., 3., 4.);
        assert_eq!(place(w, &[], 0, BAR), (0, w));
    }
}
