//! Custom theme files (design/THEMES.md, Custom themes): a small JSON file with a header (`folo_theme`,
//! `id`, `name`, `mode`, optional `base` and `description`) and any subset of the colour tokens under
//! `color`, merged over the base theme's resolved palette. Anything the format doesn't know is an
//! error naming its path (`color.bg.ap: not a colour token`), so a typo never fails silently; keys
//! starting with `$` are comments, as in tokens.json.

use serde_json::{Map, Value};

#[cfg(test)]
use super::Theme;
use super::tokens::{THEMES, TOKENS, TokenKind};
use super::{Colors, Mode, ThemeId, rgba32};

/// A valid custom theme, resolved: `colors` is its base's palette with its own colours over it.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomTheme {
    pub id: String,
    pub name: String,
    pub mode: Mode,
    /// The built-in theme it starts from (the mode's default when the file names none).
    pub base: String,
    pub description: String,
    pub colors: Colors,
}

/// The keys of a theme file besides `$` comments.
const KEYS: [&str; 7] = ["folo_theme", "id", "name", "mode", "base", "description", "color"];

/// Parses and validates one theme file. The error is the first problem found, as `path: what`; the
/// messages and their order are the ones `design/theme-cases.json` holds for every validator
/// (`design/contrast.mjs`, Studio, this one): unknown top-level keys, `folo_theme`, `id`, `name`,
/// `description`, `mode`, `base`, then `color` depth first in file order.
pub fn parse(text: &str) -> Result<CustomTheme, String> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| format!("(file): not valid JSON (line {}, column {})", e.line(), e.column()))?;
    parse_value(&v)
}

/// [`parse`] of a JSON value.
pub fn parse_value(v: &Value) -> Result<CustomTheme, String> {
    let root = v.as_object().ok_or("(file): must be a JSON object")?;
    if let Some(k) = root.keys().find(|k| !k.starts_with('$') && !KEYS.contains(&k.as_str())) {
        return Err(format!("{k}: unknown key"));
    }
    match root.get("folo_theme") {
        None => return Err("folo_theme: required".into()),
        Some(v) if v.as_u64() == Some(1) => {}
        Some(_) => return Err("folo_theme: must be 1".into()),
    }
    let id = match root.get("id") {
        None => return Err("id: required".into()),
        Some(v) => v.as_str().filter(|id| ThemeId::new(id).is_some()),
    };
    let Some(id) = id else {
        return Err("id: must be 1 to 40 characters of a-z, 0-9 and -".into());
    };
    if THEMES.iter().any(|b| b.id == id) {
        return Err(format!("id: \"{id}\" is a built-in theme"));
    }
    let name = match root.get("name") {
        None => return Err("name: required".into()),
        Some(v) => v.as_str().map(str::trim).filter(|n| !n.is_empty() && n.chars().count() <= 40),
    };
    let Some(name) = name else {
        return Err("name: must be 1 to 40 characters".into());
    };
    let description = match root.get("description") {
        None => "",
        Some(v) => match v.as_str().filter(|d| d.chars().count() <= 120) {
            Some(d) => d,
            None => return Err("description: must be at most 120 characters".into()),
        },
    };
    let mode = match root.get("mode") {
        None => return Err("mode: required".into()),
        Some(v) if v.as_str() == Some("dark") => Mode::Dark,
        Some(v) if v.as_str() == Some("light") => Mode::Light,
        Some(_) => return Err("mode: must be \"dark\" or \"light\"".into()),
    };
    let builtin = match root.get("base") {
        None => THEMES.iter().find(|b| b.id == mode.default_theme()),
        Some(v) => v.as_str().and_then(|base| THEMES.iter().find(|b| b.id == base)),
    };
    let Some(builtin) = builtin else {
        return Err("base: not a built-in theme".into());
    };
    if builtin.mode != mode {
        return Err(format!(
            "base: \"{}\" is a {} theme, not {}",
            builtin.id,
            builtin.mode.as_str(),
            mode.as_str()
        ));
    }
    let mut colors = (builtin.colors)();
    match root.get("color") {
        None => {}
        Some(Value::Object(m)) => merge(m, "", &mut colors)?,
        Some(_) => return Err("color: must be an object".into()),
    }
    Ok(CustomTheme {
        id: id.to_string(),
        name: name.to_string(),
        mode,
        base: builtin.id.to_string(),
        description: description.to_string(),
        colors,
    })
}

/// Sets every token of `group` (at `prefix`, `""` for `color` itself) on `colors`.
fn merge(group: &Map<String, Value>, prefix: &str, colors: &mut Colors) -> Result<(), String> {
    for (k, v) in group {
        if k.starts_with('$') {
            continue;
        }
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        let at = format!("color.{path}");
        match TOKENS.iter().find(|(p, _)| *p == path).map(|(_, kind)| *kind) {
            Some(TokenKind::Color) => {
                colors.set(&path, colour(v).ok_or_else(|| not_colour(&at))?);
            }
            Some(TokenKind::Colors(n)) => {
                let Some(items) = v.as_array().filter(|items| items.len() == n) else {
                    return Err(format!("{at}: must be a list of {n} colours"));
                };
                for (i, item) in items.iter().enumerate() {
                    let c = colour(item).ok_or_else(|| not_colour(&format!("{at}.{i}")))?;
                    colors.set(&format!("{path}.{i}"), c);
                }
            }
            Some(TokenKind::Number) => {
                let n = v.as_f64().filter(|n| (0.0..=1.0).contains(n));
                colors.set_number(
                    &path,
                    n.ok_or_else(|| format!("{at}: must be a number from 0 to 1"))? as f32,
                );
            }
            None if TOKENS
                .iter()
                .any(|(p, _)| p.strip_prefix(path.as_str()).is_some_and(|r| r.starts_with('.'))) =>
            {
                let inner =
                    v.as_object().ok_or_else(|| format!("{at}: must be an object of colour tokens"))?;
                merge(inner, &path, colors)?;
            }
            None => return Err(format!("{at}: not a colour token")),
        }
    }
    Ok(())
}

fn not_colour(at: &str) -> String {
    format!("{at}: not a #RRGGBB or #RRGGBBAA colour")
}

/// `#RRGGBB` or `#RRGGBBAA`, either case.
fn colour(v: &Value) -> Option<folo_ui::Hsla> {
    let hex = v.as_str()?.strip_prefix('#')?;
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let n = u32::from_str_radix(hex, 16).ok()?;
    match hex.len() {
        6 => Some(rgba32(n << 8 | 0xFF)),
        8 => Some(rgba32(n)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NIGHT: &str = r##"{
        "$comment": "A test theme.",
        "folo_theme": 1,
        "id": "night-shift",
        "name": "  Night shift ",
        "mode": "dark",
        "base": "midnight",
        "description": "Optional, one line.",
        "color": {
            "$comment": "ignored",
            "bg": { "app": "#0B0D12", "panel": "#11141bcc" },
            "text": { "secondary": "#A3AAB6" },
            "focus": "#FFFFFF",
            "chart": {
                "series": ["#5CC8FF", "#F2C14E", "#E879F9", "#2DD4BF", "#A78BFA", "#FB923C"],
                "area_opacity": 0.2
            }
        }
    }"##;

    #[test]
    fn a_valid_file_resolves_over_its_base() {
        let t = parse(NIGHT).unwrap();
        assert_eq!(
            (t.id.as_str(), t.name.as_str(), t.mode, t.base.as_str()),
            ("night-shift", "Night shift", Mode::Dark, "midnight")
        );
        assert_eq!(t.description, "Optional, one line.");
        let base = Theme::builtin("midnight").unwrap().c;
        assert_eq!(t.colors.bg_app, rgba32(0x0B0D12FF));
        assert_eq!(t.colors.bg_panel, rgba32(0x11141BCC), "lower case and alpha");
        assert_eq!(t.colors.text_secondary, rgba32(0xA3AAB6FF));
        assert_eq!(t.colors.focus, rgba32(0xFFFFFFFF), "a top-level token");
        assert_eq!(t.colors.chart_series[5], rgba32(0xFB923CFF));
        assert_eq!(t.colors.chart_area_opacity, 0.2);
        // Everything else is the base's.
        assert_eq!(t.colors.bg_raised, base.bg_raised);
        assert_eq!(t.colors.chart_fill, base.chart_fill);
    }

    #[test]
    fn the_base_defaults_to_the_modes_default() {
        let t = parse(r#"{"folo_theme": 1, "id": "plain", "name": "Plain", "mode": "light"}"#).unwrap();
        assert_eq!(t.base, "light");
        assert_eq!(t.colors, Theme::new(Mode::Light).c);
    }

    /// `design/theme-cases.json`, the cases every validator replays (design/contrast.mjs, Studio,
    /// this one): each invalid file is refused with exactly its message (path first), each valid one
    /// resolves to its base, trimmed name and token values.
    #[test]
    fn the_shared_cases_hold() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../design/theme-cases.json");
        let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let invalid = cases["invalid"].as_array().unwrap();
        let valid = cases["valid"].as_array().unwrap();
        assert!(invalid.len() >= 40 && valid.len() >= 5, "the cases file lost its cases");
        for case in invalid {
            let what = case["what"].as_str().unwrap();
            let got = parse_value(&case["theme"]).expect_err(what);
            assert_eq!(got, case["error"].as_str().unwrap(), "{what}");
        }
        for case in valid {
            let what = case["what"].as_str().unwrap();
            let t = parse_value(&case["theme"]).unwrap_or_else(|e| panic!("{what}: {e}"));
            let expect = &case["expect"];
            assert_eq!(t.base, expect["base"].as_str().unwrap(), "{what}");
            assert_eq!(t.name, expect["name"].as_str().unwrap(), "{what}");
            // Each expected token, set on a copy of the resolved palette: unchanged means it matched.
            for (token, want) in expect["color"].as_object().unwrap() {
                let mut c = t.colors;
                let set = match want {
                    Value::Number(n) => c.set_number(token, n.as_f64().unwrap() as f32),
                    v => c.set(token, colour(v).unwrap_or_else(|| panic!("{what}: {token}"))),
                };
                assert!(set, "{what}: {token} is no token path");
                assert_eq!(c, t.colors, "{what}: {token} is not {want}");
            }
        }
    }

    /// What only a file (not a JSON value) can get wrong.
    #[test]
    fn a_file_that_is_not_json_says_where() {
        assert_eq!(parse("{").unwrap_err(), "(file): not valid JSON (line 1, column 1)");
        assert_eq!(parse("[1]").unwrap_err(), "(file): must be a JSON object");
    }

    #[test]
    fn comments_are_ignored_everywhere() {
        let t = parse(
            r##"{"$schema": 3, "folo_theme": 1, "id": "c", "name": "C", "mode": "dark",
                "color": {"$x": 1, "bg": {"$y": [], "app": "#010203"}}}"##,
        )
        .unwrap();
        assert_eq!(t.colors.bg_app, rgba32(0x010203FF));
    }

    /// The setters reach every token path the format accepts.
    #[test]
    fn every_token_path_can_be_set() {
        let mut c = Theme::new(Mode::Dark).c;
        for (path, kind) in TOKENS {
            match kind {
                TokenKind::Color => assert!(c.set(path, rgba32(0x123456FF)), "{path}"),
                TokenKind::Colors(n) => {
                    for i in 0..n {
                        assert!(c.set(&format!("{path}.{i}"), rgba32(0x123456FF)), "{path}.{i}");
                    }
                    assert!(!c.set(&format!("{path}.{n}"), rgba32(0)), "{path}.{n}");
                }
                TokenKind::Number => assert!(c.set_number(path, 0.5), "{path}"),
            }
        }
        assert!(!c.set("bg.nope", rgba32(0)));
        assert!(!c.set_number("bg.app", 0.5));
        assert_eq!(c.bg_app, rgba32(0x123456FF));
    }
}
