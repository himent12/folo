//! What the app remembers between launches: the window's size, position and maximized / full-screen
//! state, and the theme (the mode and each mode's theme). One small JSON file, `<FOLO_HOME or ~/.folo>/desktop.json`
//! (`$FOLO_DESKTOP_STATE` names another, as `tools/window-check.ps1` does).
//!
//! Screenshots stay deterministic: with `--scenario`, `--width` or `--height` nothing is read or
//! written ([`remembers`]). A missing or unreadable file is a first launch; writes go to a temporary
//! file renamed over the old one, so a crash mid-write never leaves half a file.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::theme::{ModePref, ThemePrefs};

/// The file's contents. Every field is optional, so an older or newer file still loads.
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct State {
    pub window: Option<Placement>,
    /// The mode: `"dark"`, `"light"` or `"system"` (design/THEMES.md; older builds wrote the first two).
    pub theme: Option<String>,
    /// The theme Dark uses (a theme id, kept as is when no such theme is installed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dark_theme: Option<String>,
    /// The theme Light uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_theme: Option<String>,
    /// Whether connecting starts a local `folo serve` when none is running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn_server: Option<bool>,
    /// Last server URL configured in Settings › Connection (overridden by `--server` / `$FOLO_SERVER`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    /// Settings › Reduce motion, when picked against the OS's own setting (else it follows the OS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduce_motion: Option<bool>,
    /// Whether the user collapsed the sidebar to the icon rail (`Ctrl+B`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidebar_collapsed: Option<bool>,
}

/// The window in logical pixels: its restored (windowed) bounds, and how it was showing.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Placement {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    #[serde(default)]
    pub mode: WindowMode,
    /// The display it was on: folo-ui's display UUID, stable across launches where the platform has
    /// one (the display id is not). The bounds are in that display's logical pixels, and with
    /// mixed scale factors displays' rectangles overlap, so they alone can't say which one it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
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
/// "minimized from maximized" in what folo-ui returns), and folo-ui draws no frame while minimized, so the
/// mode of the last frame drawn (`shown`) is how it will come back: use it when the window says
/// windowed. Full screen survives minimizing in folo-ui's own report.
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
    std::fs::read(path).ok().map(|b| parse(&b)).unwrap_or_default()
}

/// The state in `bytes`, field by field: a field of the wrong type or with a value this build
/// doesn't know (a window mode from a newer one) is dropped alone, so the theme survives a bad
/// window and the bounds survive an unknown mode. Not JSON at all: nothing saved.
fn parse(bytes: &[u8]) -> State {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) else { return State::default() };
    fn field<T: serde::de::DeserializeOwned>(v: &serde_json::Value, key: &str) -> Option<T> {
        v.get(key).cloned().and_then(|x| serde_json::from_value(x).ok())
    }
    let window = v.get("window").and_then(|w| {
        Some(Placement {
            x: field(w, "x")?,
            y: field(w, "y")?,
            width: field(w, "width")?,
            height: field(w, "height")?,
            mode: field(w, "mode").unwrap_or_default(),
            display: field(w, "display"),
        })
    });
    State {
        window,
        theme: field(&v, "theme"),
        dark_theme: field(&v, "dark_theme"),
        light_theme: field(&v, "light_theme"),
        spawn_server: field(&v, "spawn_server"),
        server_url: field(&v, "server_url"),
        reduce_motion: field(&v, "reduce_motion"),
        sidebar_collapsed: field(&v, "sidebar_collapsed"),
    }
}

impl State {
    /// The theme choice saved here: the mode (Dark when missing or unknown) and each mode's theme
    /// (its default when missing; an id no theme has is kept, and shows the default).
    pub fn theme_prefs(&self) -> ThemePrefs {
        let d = ThemePrefs::default();
        ThemePrefs {
            mode: self.theme.as_deref().and_then(ModePref::parse).unwrap_or(d.mode),
            dark: self.dark_theme.clone().unwrap_or(d.dark),
            light: self.light_theme.clone().unwrap_or(d.light),
        }
    }

    /// Keeps `prefs` in this state.
    pub fn set_theme_prefs(&mut self, prefs: &ThemePrefs) {
        self.theme = Some(prefs.mode.as_str().into());
        self.dark_theme = Some(prefs.dark.clone());
        self.light_theme = Some(prefs.light.clone());
    }
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

/// Where a saved window opens, given the displays (`displays[primary]` the primary one), the one it
/// was on if it is still connected (`remembered`) and the title bar's height: on the remembered
/// display if it still holds a grabbable piece of its title bar, else on the display that holds the
/// largest such piece (a tie goes to the primary display, then the lower index), shrunk to fit that
/// display; when no display has one (the monitor it was on is gone, the resolution dropped),
/// centred on the primary display. Returns the display's index and the bounds.
pub fn place(
    saved: Rect,
    displays: &[Rect],
    primary: usize,
    remembered: Option<usize>,
    bar: f32,
) -> (usize, Rect) {
    let title = Rect { height: bar.min(saved.height), ..saved };
    let grab = |d: &Rect| {
        let part = title.intersection(d)?;
        (part.width >= GRAB_WIDTH.min(saved.width) && part.height >= title.height / 2.).then_some(part.width)
    };
    let on = remembered.filter(|&i| displays.get(i).and_then(grab).is_some()).or_else(|| {
        displays
            .iter()
            .enumerate()
            .filter_map(|(i, d)| Some((i, grab(d)?)))
            // On equal widths `max_by` keeps the later one: rank the primary, then the lower index, up.
            .max_by(|a, b| {
                a.1.total_cmp(&b.1).then((a.0 == primary).cmp(&(b.0 == primary))).then(b.0.cmp(&a.0))
            })
            .map(|(i, _)| i)
    });
    let index = on.unwrap_or(primary.min(displays.len().saturating_sub(1)));
    let Some(d) = displays.get(index) else {
        return (0, saved);
    };
    let (width, height) = (saved.width.min(d.width), saved.height.min(d.height));
    let rect = match on {
        // Still on screen: keep the position, pulled in only until the centre is on the display
        // (folo-ui opens bounds whose centre is elsewhere at its own default place instead).
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

    /// One field this build can't read drops that field alone (stress d5-window-input-4): a window mode
    /// from a newer build kept the theme light and the bounds; a window of the wrong types kept the theme.
    #[test]
    fn a_bad_field_drops_only_itself() {
        let s =
            parse(br#"{"window":{"x":5,"y":6,"width":1300,"height":810,"mode":"sideways"},"theme":"light"}"#);
        assert_eq!(s.theme.as_deref(), Some("light"));
        let w = s.window.expect("bounds kept");
        assert_eq!((w.x, w.y, w.width, w.height, w.mode), (5., 6., 1300., 810., WindowMode::Windowed));
        let s =
            parse(br#"{"window":{"x":"a","y":2,"width":3,"height":4},"theme":"light","spawn_server":"yes"}"#);
        assert_eq!((s.window, s.theme.as_deref(), s.spawn_server), (None, Some("light"), None));
        assert_eq!(
            parse(br#"{"theme":42,"reduce_motion":true}"#),
            State { reduce_motion: Some(true), ..State::default() }
        );
        assert_eq!(parse(b"{not json"), State::default());
        assert_eq!(parse(b"[1,2]"), State::default());
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
                display: Some("display-uuid".into()),
            }),
            theme: Some("system".into()),
            dark_theme: Some("midnight".into()),
            // A custom theme not installed now: kept.
            light_theme: Some("gone-custom".into()),
            spawn_server: Some(false),
            server_url: Some("http://10.0.0.5:7878".into()),
            reduce_motion: Some(true),
            sidebar_collapsed: Some(true),
        };
        save(&path, &state).unwrap();
        assert_eq!(load(&path), state);
        let prefs = load(&path).theme_prefs();
        assert_eq!(
            (prefs.mode, prefs.dark.as_str(), prefs.light.as_str()),
            (ModePref::System, "midnight", "gone-custom")
        );
        let mut again = State::default();
        again.set_theme_prefs(&prefs);
        assert_eq!(again.theme_prefs(), prefs);
        // A file written before Reduce motion was kept: it follows the OS. Before themes: the mode
        // reads the same, each mode's theme is its default.
        std::fs::write(&path, r#"{"theme":"light","spawn_server":true}"#).unwrap();
        assert_eq!(load(&path).reduce_motion, None);
        assert_eq!(load(&path).theme_prefs(), ThemePrefs { mode: ModePref::Light, ..ThemePrefs::default() });
        std::fs::write(&path, r#"{"theme":"sideways","dark_theme":7}"#).unwrap();
        assert_eq!(load(&path).theme_prefs(), ThemePrefs::default(), "unknown values read as the defaults");
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
            Some(Placement {
                x: 1.,
                y: 2.,
                width: 3.,
                height: 4.,
                mode: WindowMode::Windowed,
                display: None
            })
        );
    }

    #[test]
    fn mode_is_lowercase_in_the_file() {
        let p =
            Placement { x: 0., y: 0., width: 1., height: 1., mode: WindowMode::Fullscreen, display: None };
        assert!(serde_json::to_string(&p).unwrap().contains("\"mode\":\"fullscreen\""));
    }

    #[test]
    fn a_window_on_screen_stays_where_it_was() {
        let displays = [r(0., 0., 2560., 1400.), r(2560., 0., 1920., 1080.)];
        let w = r(300., 200., 1280., 800.);
        assert_eq!(place(w, &displays, 0, None, BAR), (0, w));
        // A little above the top edge, but most of the title bar shows.
        let up = r(100., -10., 1280., 800.);
        assert_eq!(place(up, &displays, 0, None, BAR), (0, up));
        // On the second display, and partly off its right edge.
        let w2 = r(3500., 100., 1280., 800.);
        assert_eq!(place(w2, &displays, 0, None, BAR), (1, w2));
        // So far off it that the centre isn't on it: pulled back until the centre is.
        assert_eq!(
            place(r(3900., 100., 1280., 800.), &displays, 0, None, BAR),
            (1, r(3839., 100., 1280., 800.))
        );
        // Straddling both: the display with more of the title bar wins.
        assert_eq!(place(r(2000., 100., 1280., 800.), &displays, 0, None, BAR).0, 1);
        assert_eq!(place(r(1800., 100., 1280., 800.), &displays, 0, None, BAR).0, 0);
    }

    #[test]
    fn a_window_off_every_display_is_centred_on_the_primary_one() {
        let displays = [r(-1920., 0., 1920., 1080.), r(0., 0., 2560., 1440.)];
        let w = r(5000., 300., 1280., 800.);
        assert_eq!(place(w, &displays, 1, None, BAR), (1, r(640., 320., 1280., 800.)));
        // Title bar above the top of every display: the window can't be grabbed.
        assert_eq!(place(r(100., -500., 1280., 800.), &displays, 1, None, BAR).1, r(640., 320., 1280., 800.));
        // Only a sliver of the title bar showing counts as off screen too.
        assert_eq!(
            place(r(2500., 100., 1280., 800.), &[r(0., 0., 2560., 1440.)], 0, None, BAR).1,
            r(640., 320., 1280., 800.)
        );
    }

    #[test]
    fn a_window_bigger_than_its_display_shrinks_to_fit() {
        let displays = [r(0., 0., 1366., 768.)];
        assert_eq!(place(r(0., 0., 2000., 1200.), &displays, 0, None, BAR), (0, r(0., 0., 1366., 768.)));
        let (_, off) = place(r(-3000., 0., 2000., 1200.), &displays, 0, None, BAR);
        assert_eq!(off, r(0., 0., 1366., 768.));
    }

    /// folo-ui gives each display's bounds in its own logical pixels, so with mixed scale factors the
    /// rectangles overlap: a 2560x1440 primary at 100% and a 3840x2160 display to its right at 200%.
    const MIXED: [Rect; 2] = [
        Rect { x: 0., y: 0., width: 2560., height: 1440. },
        Rect { x: 1280., y: 0., width: 1920., height: 1080. },
    ];

    #[test]
    fn mixed_scales_reopen_on_the_remembered_display() {
        // Saved on the primary, in the overlap: back on the primary, its full height kept.
        let w = r(1400., 100., 1000., 1200.);
        assert_eq!(place(w, &MIXED, 0, Some(0), BAR), (0, w));
        assert_eq!(place(w, &MIXED, 0, None, BAR), (0, w));
        // Saved on the 200% display, in the overlap: back there, though the primary ties.
        let w = r(1400., 100., 800., 500.);
        assert_eq!(place(w, &MIXED, 0, Some(1), BAR), (1, w));
    }

    #[test]
    fn a_remembered_display_that_is_gone_or_no_longer_shows_the_bar_falls_back() {
        // Gone: its UUID matched no connected display, so nothing is remembered.
        let w = r(1400., 100., 800., 500.);
        assert_eq!(place(w, &MIXED, 0, None, BAR), (0, w));
        // Still connected, but the title bar is off it (its resolution dropped, say): the usual rule.
        let displays = [r(0., 0., 2560., 1440.), r(2560., 0., 1280., 720.)];
        let w = r(300., 200., 1280., 800.);
        assert_eq!(place(w, &displays, 0, Some(1), BAR), (0, w));
        // An index past the displays is no display.
        assert_eq!(place(w, &displays, 0, Some(5), BAR), (0, w));
        // Off every display: centred on the primary, as without one.
        assert_eq!(
            place(r(9000., 0., 1280., 800.), &displays, 0, Some(1), BAR),
            (0, r(640., 320., 1280., 800.))
        );
    }

    #[test]
    fn the_display_is_kept_in_the_file_and_older_files_load_without_it() {
        let p = Placement {
            x: 1400.,
            y: 100.,
            width: 1000.,
            height: 1200.,
            mode: WindowMode::Windowed,
            display: Some("6ba7b810-9dad-51d1-80b4-00c04fd430c8".into()),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"display\":\"6ba7b810-9dad-51d1-80b4-00c04fd430c8\""), "{json}");
        assert_eq!(serde_json::from_str::<Placement>(&json).unwrap(), p);
        // A file from before the field: no display, and none written back for a window without one.
        let old: Placement =
            serde_json::from_str(r#"{"x":1,"y":2,"width":3,"height":4,"mode":"maximized"}"#).unwrap();
        assert_eq!(old.display, None);
        assert_eq!(old.mode, WindowMode::Maximized);
        assert!(!serde_json::to_string(&old).unwrap().contains("display"));
    }

    #[test]
    fn a_tie_goes_to_the_primary_display_then_the_lower_index() {
        let w = r(1400., 100., 800., 500.);
        assert_eq!(place(w, &MIXED, 0, None, BAR).0, 0);
        assert_eq!(place(w, &MIXED, 1, None, BAR).0, 1);
        // Neither of the tied displays is the primary: the lower index.
        let three = [MIXED[0], MIXED[1], r(-1920., 0., 1920., 1080.)];
        assert_eq!(place(w, &three, 2, None, BAR).0, 0);
    }

    #[test]
    fn no_displays_keeps_the_saved_bounds() {
        let w = r(1., 2., 3., 4.);
        assert_eq!(place(w, &[], 0, None, BAR), (0, w));
    }
}
