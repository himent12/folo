//! The themes to pick from and the user's choice (design/THEMES.md, Choosing a theme): the mode
//! (Dark, Light or System) and, separately, the theme each mode uses. Built-in themes come from
//! tokens.json in its order; custom ones from `<FOLO_HOME or ~/.folo>/themes/*.json`, after them by
//! name. A stored id that no longer exists shows the mode's default and stays stored, so installing
//! that theme again brings it back.

use std::path::{Path, PathBuf};

use super::custom::{self, CustomTheme};
use super::tokens::THEMES;
use super::{Colors, Mode, Theme, ThemeId};

/// The user's mode: Dark, Light, or System (follows the OS appearance live).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModePref {
    Dark,
    Light,
    System,
}

impl ModePref {
    /// `dark`, `light` or `system` (desktop.json's `theme`, `--theme`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "dark" => Some(Self::Dark),
            "light" => Some(Self::Light),
            "system" => Some(Self::System),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
            Self::System => "system",
        }
    }
}

impl From<Mode> for ModePref {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Dark => Self::Dark,
            Mode::Light => Self::Light,
        }
    }
}

/// What the user picked: the mode and the theme of each mode (ids, kept as stored even when no such
/// theme is installed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThemePrefs {
    pub mode: ModePref,
    pub dark: String,
    pub light: String,
}

impl Default for ThemePrefs {
    fn default() -> Self {
        Self {
            mode: ModePref::Dark,
            dark: Mode::Dark.default_theme().into(),
            light: Mode::Light.default_theme().into(),
        }
    }
}

impl ThemePrefs {
    /// The mode in effect, with the OS's appearance now (`system_dark`) for System.
    pub fn mode_now(&self, system_dark: bool) -> Mode {
        match self.mode {
            ModePref::Dark => Mode::Dark,
            ModePref::Light => Mode::Light,
            ModePref::System if system_dark => Mode::Dark,
            ModePref::System => Mode::Light,
        }
    }

    /// The theme id stored for `mode`.
    pub fn slot(&self, mode: Mode) -> &str {
        match mode {
            Mode::Dark => &self.dark,
            Mode::Light => &self.light,
        }
    }

    /// Picks theme `id` of mode `mode`: it becomes that mode's theme, and the mode switches to it
    /// unless it is System (then the choice applies whenever that mode is in effect).
    pub fn pick(&mut self, mode: Mode, id: &str) {
        match mode {
            Mode::Dark => self.dark = id.to_string(),
            Mode::Light => self.light = id.to_string(),
        }
        if self.mode != ModePref::System {
            self.mode = mode.into();
        }
    }

    /// Ctrl Shift L: the other mode than the one in effect, as an explicit Dark or Light.
    pub fn toggle(&mut self, system_dark: bool) {
        self.mode = match self.mode_now(system_dark) {
            Mode::Dark => ModePref::Light,
            Mode::Light => ModePref::Dark,
        };
    }

    /// `--theme`: a mode (`dark`, `light`, `system`), or a theme id of `catalog` (`midnight`: mode
    /// Dark with Midnight). `false` when it is neither, and nothing changes.
    pub fn apply_arg(&mut self, arg: &str, catalog: &Catalog) -> bool {
        if let Some(m) = ModePref::parse(arg) {
            self.mode = m;
        } else if let Some(e) = catalog.get(arg) {
            let mode = e.mode;
            self.pick(mode, arg);
            self.mode = mode.into();
        } else {
            return false;
        }
        true
    }
}

/// One theme the user can pick.
#[derive(Clone, Debug, PartialEq)]
pub struct ThemeEntry {
    pub id: String,
    pub name: String,
    pub mode: Mode,
    pub description: String,
    pub colors: Colors,
    pub custom: bool,
    /// A custom theme's base (a built-in id); empty for a built-in.
    pub base: String,
    /// The file a custom theme was read from (its name); empty for a built-in.
    pub file: String,
}

impl ThemeEntry {
    pub fn theme(&self) -> Theme {
        Theme { mode: self.mode, id: ThemeId::new(&self.id).expect("theme ids are valid"), c: self.colors }
    }
}

/// A theme file that wasn't installed, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failed {
    /// The file's name.
    pub file: String,
    /// `path: what`, from [`custom::parse`], or why the file couldn't be read.
    pub error: String,
}

/// The themes to pick from: built-ins in tokens.json's order, then custom ones by name.
#[derive(Clone, Debug)]
pub struct Catalog {
    entries: Vec<ThemeEntry>,
    /// The folder custom themes were read from (`None`: none were read).
    pub folder: Option<PathBuf>,
    /// The files there that failed, in file name order.
    pub failed: Vec<Failed>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self::builtin()
    }
}

/// A theme file larger than this, 1 MiB (1 048 576 bytes), is refused; at most one byte more is
/// read (a theme is a few kB; Studio and design/contrast.mjs refuse the same size).
const MAX_FILE: u64 = 1 << 20;

/// `path` as the platform writes it, one separator throughout: a `FOLO_HOME` of `D:/x` joined with
/// `themes` reads `D:\x\themes` on Windows, not `D:/x\themes`.
pub fn display_path(path: &Path) -> String {
    path.components().collect::<PathBuf>().display().to_string()
}

/// `<FOLO_HOME or ~/.folo>/themes`: where Desktop reads custom themes from.
pub fn custom_folder() -> PathBuf {
    folo_api::weights::home_dir().join("themes")
}

impl Catalog {
    /// The built-in themes alone.
    pub fn builtin() -> Self {
        let entries = THEMES
            .iter()
            .map(|b| ThemeEntry {
                id: b.id.into(),
                name: b.name.into(),
                mode: b.mode,
                description: b.description.into(),
                colors: (b.colors)(),
                custom: false,
                base: String::new(),
                file: String::new(),
            })
            .collect();
        Self { entries, folder: None, failed: Vec::new() }
    }

    /// The built-ins and `custom` (in load order: a later theme with an id already there replaces it).
    #[cfg(test)]
    pub fn with_custom(custom: Vec<CustomTheme>) -> Self {
        Self::with_files(custom.into_iter().map(|t| (String::new(), t)).collect())
    }

    /// [`Self::with_custom`] of themes read from files: `(file name, theme)`.
    fn with_files(custom: Vec<(String, CustomTheme)>) -> Self {
        let mut c = Self::builtin();
        let mut extra: Vec<ThemeEntry> = Vec::new();
        for (file, t) in custom {
            extra.retain(|e| e.id != t.id);
            extra.push(ThemeEntry {
                id: t.id,
                name: t.name,
                mode: t.mode,
                description: t.description,
                colors: t.colors,
                custom: true,
                base: t.base,
                file,
            });
        }
        // By lower-cased name, then id, both by code point (design/THEMES.md; Studio's themeCatalog).
        extra.sort_by_cached_key(|e| (e.name.to_lowercase(), e.id.clone()));
        c.entries.extend(extra);
        c
    }

    /// The built-ins and every valid `*.json` in `folder`, read in file name order; a missing folder
    /// is no custom themes. Small files: fast enough at start, Reload reads it off the UI thread.
    pub fn load(folder: &Path) -> Self {
        let mut themes = Vec::new();
        let mut failed = Vec::new();
        match std::fs::read_dir(folder) {
            Ok(dir) => {
                let mut files: Vec<PathBuf> = dir
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")))
                    .collect();
                files.sort();
                for p in files {
                    let file = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    match read_theme(&p) {
                        Ok(t) => themes.push((file, t)),
                        Err(error) => failed.push(Failed { file, error }),
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                failed.push(Failed { file: String::new(), error: format!("couldn't read the folder: {e}") })
            }
        }
        let mut c = Self::with_files(themes);
        c.folder = Some(folder.to_path_buf());
        c.failed = failed;
        c
    }

    pub fn entries(&self) -> &[ThemeEntry] {
        &self.entries
    }

    /// The themes of `mode`, in the order they are listed.
    pub fn of_mode(&self, mode: Mode) -> impl Iterator<Item = &ThemeEntry> {
        self.entries.iter().filter(move |e| e.mode == mode)
    }

    pub fn get(&self, id: &str) -> Option<&ThemeEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// How many custom themes are installed.
    #[cfg(test)]
    pub fn custom_count(&self) -> usize {
        self.custom().count()
    }

    /// The custom themes, in the order they are listed (by name).
    pub fn custom(&self) -> impl Iterator<Item = &ThemeEntry> {
        self.entries.iter().filter(|e| e.custom)
    }

    /// What a custom theme is, for its line under Settings › Custom themes, as Studio says it:
    /// `["Dark", "based on Midnight"]`; the base is left out when it is the mode's default.
    pub fn facts(&self, e: &ThemeEntry) -> Vec<String> {
        let mode = match e.mode {
            Mode::Dark => "Dark",
            Mode::Light => "Light",
        };
        let mut facts = vec![mode.to_string()];
        if let Some(base) = self.get(&e.base).filter(|b| !b.custom && b.id != e.mode.default_theme()) {
            facts.push(format!("based on {}", base.name));
        }
        facts
    }

    /// The theme `prefs` shows for `mode`: its stored id when a theme of that mode has it, else the
    /// mode's default.
    pub fn entry_for(&self, prefs: &ThemePrefs, mode: Mode) -> &ThemeEntry {
        let id = prefs.slot(mode);
        self.of_mode(mode)
            .find(|e| e.id == id)
            .or_else(|| self.get(mode.default_theme()))
            .expect("the default themes are built in")
    }

    /// The theme to show for `prefs` with the OS appearance `system_dark`.
    pub fn resolve(&self, prefs: &ThemePrefs, system_dark: bool) -> Theme {
        self.entry_for(prefs, prefs.mode_now(system_dark)).theme()
    }
}

/// Reads and parses one theme file ([`custom::parse_bytes`]), at most [`MAX_FILE`] + 1 bytes of it:
/// the limit holds at the read itself (a file that grew, or was replaced, after the folder was
/// listed still costs no more).
pub(super) fn read_theme(path: &Path) -> Result<CustomTheme, String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| format!("(file): couldn't read ({e})"))?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE + 1).read_to_end(&mut bytes).map_err(|e| format!("(file): couldn't read ({e})"))?;
    if bytes.len() as u64 > MAX_FILE {
        return Err("(file): larger than 1 MB; not a theme".into());
    }
    custom::parse_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    fn custom(id: &str, name: &str, mode: &str) -> String {
        format!(
            r##"{{"folo_theme": 1, "id": "{id}", "name": "{name}", "mode": "{mode}", "color": {{"bg": {{"app": "#010203"}}}}}}"##
        )
    }

    #[test]
    fn builtins_keep_the_token_order_and_defaults_come_first() {
        let c = Catalog::builtin();
        let ids: Vec<&str> = c.entries().iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["dark", "midnight", "dim", "contrast-dark", "light", "paper", "contrast-light"]);
        assert_eq!(c.of_mode(Mode::Dark).next().unwrap().id, "dark");
        assert_eq!(c.of_mode(Mode::Light).next().unwrap().id, "light");
        // The defaults are the base palettes, untouched.
        assert_eq!(c.get("dark").unwrap().colors, super::super::tokens::theme_dark());
        assert_eq!(Theme::new(Mode::Light).id.as_str(), "light");
    }

    /// Ctrl Shift L, the segmented control and the theme selects (design/THEMES.md, Choosing a theme).
    #[test]
    fn mode_and_slot_rules() {
        let mut p = ThemePrefs::default();
        assert_eq!((p.mode, p.dark.as_str(), p.light.as_str()), (ModePref::Dark, "dark", "light"));
        // Picking a light theme while Dark switches to Light (and the other way round).
        p.pick(Mode::Light, "paper");
        assert_eq!((p.mode, p.light.as_str()), (ModePref::Light, "paper"));
        p.pick(Mode::Dark, "midnight");
        assert_eq!((p.mode, p.dark.as_str(), p.light.as_str()), (ModePref::Dark, "midnight", "paper"));
        // With System the mode stays; the choice applies when that mode is in effect.
        p.mode = ModePref::System;
        p.pick(Mode::Light, "contrast-light");
        assert_eq!(p.mode, ModePref::System);
        let c = Catalog::builtin();
        assert_eq!(c.resolve(&p, true).id.as_str(), "midnight");
        assert_eq!(c.resolve(&p, false).id.as_str(), "contrast-light");
        assert_eq!(c.resolve(&p, false).mode, Mode::Light);
        // The toggle leaves System for the other mode than the one in effect.
        p.toggle(true);
        assert_eq!(p.mode, ModePref::Light);
        p.toggle(true);
        assert_eq!(p.mode, ModePref::Dark);
        assert_eq!(p.mode_now(false), Mode::Dark);
    }

    #[test]
    fn an_unknown_stored_id_shows_the_default_and_stays() {
        let p = ThemePrefs { mode: ModePref::Dark, dark: "gone".into(), light: "midnight".into() };
        let c = Catalog::builtin();
        assert_eq!(c.resolve(&p, false).id.as_str(), "dark");
        // A theme of the other mode in a slot doesn't apply there either.
        assert_eq!(c.entry_for(&p, Mode::Light).id, "light");
        assert_eq!(p.dark, "gone", "kept, so installing it again brings it back");
        let back = Catalog::with_custom(vec![custom::parse(&custom("gone", "Gone", "dark")).unwrap()]);
        assert_eq!(back.resolve(&p, false).id.as_str(), "gone");
        assert_eq!(back.resolve(&p, false).c.bg_app, super::super::rgba32(0x010203FF));
    }

    #[test]
    fn theme_argument_takes_a_mode_or_an_id() {
        let c = Catalog::with_custom(vec![custom::parse(&custom("night", "Night", "dark")).unwrap()]);
        let mut p = ThemePrefs { mode: ModePref::System, ..Default::default() };
        assert!(p.apply_arg("midnight", &c));
        assert_eq!((p.mode, p.dark.as_str()), (ModePref::Dark, "midnight"), "mode Dark with Midnight");
        assert!(p.apply_arg("paper", &c));
        assert_eq!((p.mode, p.light.as_str(), p.dark.as_str()), (ModePref::Light, "paper", "midnight"));
        assert!(p.apply_arg("system", &c));
        assert_eq!(p.mode, ModePref::System);
        assert!(p.apply_arg("night", &c), "an installed custom theme");
        assert_eq!((p.mode, p.dark.as_str()), (ModePref::Dark, "night"));
        let before = p.clone();
        assert!(!p.apply_arg("bogus", &c));
        assert_eq!(p, before);
    }

    #[test]
    fn the_folder_loads_valid_files_and_names_the_failed_ones() {
        let dir = TestDir::new("desktop-themes");
        let folder = dir.path().join("themes");
        assert_eq!(Catalog::load(&folder).custom_count(), 0, "no folder: no custom themes, no error");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("b.json"), custom("zeta", "Zeta", "light")).unwrap();
        std::fs::write(folder.join("a.JSON"), custom("alpha", "alpha", "dark")).unwrap();
        // Same id as a.JSON, read after it: replaces it.
        std::fs::write(folder.join("c.json"), custom("alpha", "Alpha two", "dark")).unwrap();
        std::fs::write(folder.join("d.json"), r##"{"folo_theme": 1, "id": "x", "name": "X", "mode": "dark", "color": {"bg": {"ap": "#000000"}}}"##).unwrap();
        std::fs::write(folder.join("e.json"), "{ nope").unwrap();
        std::fs::write(folder.join("notes.txt"), "not a theme").unwrap();
        let c = Catalog::load(&folder);
        let ids: Vec<&str> = c.entries().iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids[7..], ["alpha", "zeta"], "custom themes after the built-ins, by name");
        assert_eq!(c.get("alpha").unwrap().name, "Alpha two");
        assert_eq!(c.custom_count(), 2);
        assert_eq!(c.of_mode(Mode::Dark).last().unwrap().id, "alpha");
        assert_eq!(
            c.failed,
            [
                Failed { file: "d.json".into(), error: "color.bg.ap: not a colour token".into() },
                Failed { file: "e.json".into(), error: "(file): not valid JSON (line 1, column 3)".into() },
            ]
        );
        assert_eq!(c.folder.as_deref(), Some(folder.as_path()));
        // Each custom theme keeps its file (the later one when two share an id) and says what it is.
        let files = (c.get("alpha").unwrap().file.as_str(), c.get("zeta").unwrap().file.as_str());
        assert_eq!(files, ("c.json", "b.json"));
        assert_eq!(c.facts(c.get("zeta").unwrap()), ["Light"], "the mode's default base goes unsaid");
        let night = r##"{"folo_theme": 1, "id": "n", "name": "N", "mode": "dark", "base": "midnight"}"##;
        std::fs::write(folder.join("f.json"), night).unwrap();
        let c = Catalog::load(&folder);
        assert_eq!(c.facts(c.get("n").unwrap()), ["Dark", "based on Midnight"]);
        assert_eq!(c.custom().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["alpha", "n", "zeta"]);
    }

    /// The same names as Studio's test (web/src/lib/design/themes.test.ts): code point order of the
    /// lower-cased names, ties by id; no locale collation.
    #[test]
    fn custom_themes_sort_by_code_point_as_in_studio() {
        let names = ["😀 Smile", "\u{FF21}pple", "éclair", "Ember", "apple", "Same", "same"];
        let themes = names
            .iter()
            .enumerate()
            .map(|(i, name)| custom::parse(&custom(&format!("t{}", names.len() - i), name, "dark")).unwrap())
            .collect();
        let c = Catalog::with_custom(themes);
        let listed: Vec<&str> = c.custom().map(|e| e.name.as_str()).collect();
        assert_eq!(listed, ["apple", "Ember", "same", "Same", "éclair", "\u{FF21}pple", "😀 Smile"]);
    }

    #[test]
    fn a_byte_order_mark_is_skipped_and_large_or_reserved_files_are_refused() {
        let dir = TestDir::new("desktop-themes-edge");
        let folder = dir.path().join("themes");
        std::fs::create_dir_all(&folder).unwrap();
        // PowerShell 5.1's `Set-Content -Encoding utf8` writes a byte order mark.
        std::fs::write(folder.join("bom.json"), format!("\u{feff}{}", custom("bom", "Bom", "dark"))).unwrap();
        // Valid JSON padded past the limit: refused by size, not parsed.
        let big = format!("{}{}", custom("big", "Big", "dark"), " ".repeat(MAX_FILE as usize));
        std::fs::write(folder.join("big.json"), big).unwrap();
        // Exactly the limit is read.
        let mut edge = custom("edge", "Edge", "dark");
        edge.push_str(&" ".repeat(MAX_FILE as usize - edge.len()));
        assert_eq!(edge.len() as u64, MAX_FILE);
        std::fs::write(folder.join("edge.json"), edge).unwrap();
        std::fs::write(folder.join("system.json"), custom("system", "System", "dark")).unwrap();
        let c = Catalog::load(&folder);
        assert_eq!(c.custom().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["bom", "edge"]);
        assert_eq!(
            c.failed,
            [
                Failed { file: "big.json".into(), error: "(file): larger than 1 MB; not a theme".into() },
                Failed { file: "system.json".into(), error: "id: \"system\" is reserved".into() },
            ]
        );
        // So `--theme system` always means the mode.
        let mut p = ThemePrefs::default();
        assert!(p.apply_arg("system", &c));
        assert_eq!(p.mode, ModePref::System);
    }

    /// A file that can't be read says so as the JS readers do: `(file): couldn't read (…)`. A folder
    /// stands in for it (Windows refuses to open one, Linux to read one).
    #[test]
    fn an_unreadable_file_says_so_as_the_js_readers_do() {
        let dir = TestDir::new("desktop-theme-unreadable");
        let e = read_theme(dir.path()).unwrap_err();
        assert!(e.starts_with("(file): couldn't read (") && e.ends_with(')'), "{e}");
    }

    #[test]
    fn folders_show_in_the_platform_form() {
        let shown = display_path(&Path::new("D:/folo-build/x").join("themes"));
        if cfg!(windows) {
            assert_eq!(shown, r"D:\folo-build\x\themes");
        } else {
            assert_eq!(shown, "D:/folo-build/x/themes");
        }
    }
}
