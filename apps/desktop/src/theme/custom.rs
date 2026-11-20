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

/// Not a custom theme id: `--theme system` means the mode (design/THEMES.md, Custom themes).
pub const RESERVED_ID: &str = "system";

/// The keys of a theme file besides `$` comments.
const KEYS: [&str; 7] = ["folo_theme", "id", "name", "mode", "base", "description", "color"];

/// Parses and validates one theme file's bytes. The error is the first problem found, as `path:
/// what`; the messages and their order are the ones `design/theme-cases.json` holds for every
/// validator (`design/contrast.mjs`, Studio, this one): the encoding, JSON, unknown top-level keys,
/// `folo_theme`, `id`, `name`, `description`, `mode`, `base`, then `color` depth first in file
/// order. Exactly one leading UTF-8 byte order mark (EF BB BF, which PowerShell 5.1 and some
/// editors write) is stripped, then the rest must be UTF-8: a second mark is a U+FEFF that is not
/// JSON. serde_json refuses a lone surrogate escape (`\uD800`) while parsing; the JS validators
/// refuse it the same way.
pub fn parse_bytes(bytes: &[u8]) -> Result<CustomTheme, String> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let text = std::str::from_utf8(bytes).map_err(|_| "(file): not UTF-8 text".to_string())?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| format!("(file): not valid JSON (line {}, column {})", e.line(), e.column()))?;
    parse_value(&v)
}

/// [`parse_bytes`] of a file's text (its UTF-8 bytes), for tests.
#[cfg(test)]
pub fn parse(text: &str) -> Result<CustomTheme, String> {
    parse_bytes(text.as_bytes())
}

/// [`parse`] of a JSON value.
pub fn parse_value(v: &Value) -> Result<CustomTheme, String> {
    let root = v.as_object().ok_or("(file): must be a JSON object")?;
    if let Some(k) = root.keys().find(|k| !k.starts_with('$') && !KEYS.contains(&k.as_str())) {
        return Err(format!("{k}: unknown key"));
    }
    match root.get("folo_theme") {
        None => return Err("folo_theme: required".into()),
        // Any JSON number equal to 1 (`1`, `1.0`, `1e0`): JSON.parse can't tell them apart either.
        Some(v) if v.as_f64() == Some(1.0) => {}
        Some(_) => return Err("folo_theme: must be 1".into()),
    }
    let id = match root.get("id") {
        None => return Err("id: required".into()),
        Some(v) => v.as_str().filter(|id| ThemeId::new(id).is_some()),
    };
    let Some(id) = id else {
        return Err(
            "id: must be 1 to 40 characters of a-z, 0-9 and -, starting with a letter or digit".into()
        );
    };
    if THEMES.iter().any(|b| b.id == id) {
        return Err(format!("id: \"{id}\" is a built-in theme"));
    }
    if id == RESERVED_ID {
        return Err(format!("id: \"{RESERVED_ID}\" is reserved"));
    }
    let name = match root.get("name") {
        None => return Err("name: required".into()),
        // Lengths count code points after trimming Unicode white space (`str::trim`), as in the JS
        // validators (`[...s].length` after their White_Space trim).
        Some(v) => v.as_str().map(str::trim).filter(|n| !n.is_empty() && n.chars().count() <= 40),
    };
    let Some(name) = name else {
        return Err("name: must be 1 to 40 characters".into());
    };
    if let Some(e) = name_problem(name) {
        return Err(e.into());
    }
    let description = match root.get("description") {
        None => "",
        Some(v) => match v.as_str().map(str::trim).filter(|d| d.chars().count() <= 120) {
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

/// What is wrong with a (trimmed) name that no list could show as written, in the words of every
/// validator (design/THEMES.md; Studio's `HIDDEN` and `VISIBLE`): it holds a control character, a
/// line or paragraph separator or an invisible format character (not the zero-width joiner and
/// non-joiner, which emoji sequences and some scripts need), or it has no letter, digit, punctuation
/// or symbol (the Hangul fillers and the blank Braille pattern count as none). An unassigned code
/// point counts as visible here; the JavaScript validators know no more than their engine's Unicode.
fn name_problem(name: &str) -> Option<&'static str> {
    use super::name_chars::{FORMAT, MARKS};
    let within = |table: &[(u32, u32)], c: char| {
        let c = c as u32;
        let i = table.partition_point(|&(_, hi)| hi < c);
        table.get(i).is_some_and(|&(lo, _)| lo <= c)
    };
    let format = |c: char| within(&FORMAT, c) && !matches!(c, '\u{200C}' | '\u{200D}');
    if name.chars().any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}') || format(c)) {
        return Some("name: must not hold control or invisible characters");
    }
    // Not a letter, digit, punctuation or symbol: a mark, a space, a joiner, a private-use character
    // (the controls and other format characters are refused above), or a blank one Unicode files as a
    // letter or symbol.
    let blank = |c: char| {
        within(&MARKS, c)
            || c.is_whitespace()
            || within(&FORMAT, c)
            || matches!(c, '\u{E000}'..='\u{F8FF}' | '\u{F0000}'..='\u{FFFFD}' | '\u{100000}'..='\u{10FFFD}')
            || matches!(c, '\u{115F}' | '\u{1160}' | '\u{3164}' | '\u{FFA0}' | '\u{2800}')
    };
    name.chars().all(blank).then_some("name: must have a visible character")
}

/// Sets every token of `group` (at `prefix`, `""` for `color` itself) on `colors`, one level of the
/// palette at a time: a key holding a `.` (`"bg.app"`) is no token, as in the JS validators.
fn merge(group: &Map<String, Value>, prefix: &str, colors: &mut Colors) -> Result<(), String> {
    for (k, v) in group {
        if k.starts_with('$') {
            continue;
        }
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        let at = format!("color.{path}");
        if k.contains('.') {
            return Err(format!("{at}: not a colour token"));
        }
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

    /// A case's file bytes: its `hex`, or its raw `text` as UTF-8; `None` for a parsed `theme`.
    fn case_bytes(case: &Value) -> Option<Vec<u8>> {
        if let Some(hex) = case.get("hex") {
            let hex = hex.as_str().unwrap().as_bytes();
            let digit = |c: u8| (c as char).to_digit(16).unwrap() as u8;
            return Some(hex.chunks(2).map(|p| digit(p[0]) << 4 | digit(p[1])).collect());
        }
        case.get("text").map(|t| t.as_str().unwrap().as_bytes().to_vec())
    }

    /// `design/theme-cases.json`, the cases every validator replays (design/contrast.mjs, Studio,
    /// this one): each invalid file is refused with exactly its message (path first; `(file): not
    /// valid JSON` with any detail after it), each valid one resolves to its base, trimmed name and
    /// token values. A file's bytes (`text`, `hex`) go through the folder reader itself.
    #[test]
    fn the_shared_cases_hold() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../design/theme-cases.json");
        let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let invalid = cases["invalid"].as_array().unwrap();
        let valid = cases["valid"].as_array().unwrap();
        assert!(invalid.len() >= 40 && valid.len() >= 5, "the cases file lost its cases");
        // A case holds the parsed `theme`, the file's raw `text` (number spellings, dotted keys: what
        // only parsing the file itself shows) or its bytes in `hex` (byte order marks, invalid UTF-8,
        // surrogate escapes).
        let dir = folo_test_support::TestDir::new("desktop-theme-cases");
        let file = dir.path().join("case.json");
        let check = |case: &Value| match case_bytes(case) {
            Some(bytes) => {
                std::fs::write(&file, bytes).unwrap();
                super::super::catalog::read_theme(&file)
            }
            None => parse_value(&case["theme"]),
        };
        assert!(
            invalid.iter().chain(valid).filter(|c| c.get("text").is_some()).count() >= 8,
            "the cases file lost its raw cases"
        );
        assert!(
            invalid.iter().chain(valid).filter(|c| c.get("hex").is_some()).count() >= 10,
            "the cases file lost its byte cases"
        );
        const NOT_JSON: &str = "(file): not valid JSON";
        for case in invalid {
            let what = case["what"].as_str().unwrap();
            let got = check(case).expect_err(what);
            let want = case["error"].as_str().unwrap();
            if !(want == NOT_JSON && got.starts_with(&format!("{NOT_JSON} ("))) {
                assert_eq!(got, want, "{what}");
            }
        }
        for case in valid {
            let what = case["what"].as_str().unwrap();
            let t = check(case).unwrap_or_else(|e| panic!("{what}: {e}"));
            let expect = &case["expect"];
            assert_eq!(t.base, expect["base"].as_str().unwrap(), "{what}");
            assert_eq!(t.name, expect["name"].as_str().unwrap(), "{what}");
            if let Some(d) = expect.get("description") {
                assert_eq!(t.description, d.as_str().unwrap(), "{what}");
            }
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

    /// The generated ranges (`name_chars.rs`) are in order and apart, which the lookup needs; the
    /// names themselves are the shared cases' ("raw: name with ...").
    #[test]
    fn the_name_tables_are_ordered_ranges() {
        use super::super::name_chars::{FORMAT, MARKS};
        for table in [&MARKS[..], &FORMAT[..]] {
            assert!(table.iter().all(|&(lo, hi)| lo <= hi));
            assert!(table.windows(2).all(|w| w[0].1 < w[1].0));
        }
        // Each table's ends and a middle range are found, their neighbours not.
        let blank = Some("name: must have a visible character");
        assert_eq!(name_problem("\u{0300}\u{036F}\u{1E94A}\u{E01EF}"), blank);
        let hidden = Some("name: must not hold control or invisible characters");
        assert_eq!(name_problem("\u{00AD}"), hidden);
        assert_eq!(name_problem("A\u{E007F}"), hidden);
        assert_eq!(name_problem("\u{02FF}\u{0370}"), None);
        assert_eq!(name_problem("Night shift"), None);
    }

    /// What only a file (not a JSON value) can get wrong.
    #[test]
    fn a_file_that_is_not_json_says_where() {
        assert_eq!(parse("{").unwrap_err(), "(file): not valid JSON (line 1, column 1)");
        assert_eq!(parse("[1]").unwrap_err(), "(file): must be a JSON object");
    }

    /// The JSON serde_json refuses while parsing, which the JS readers match before `JSON.parse`
    /// (design/json-scan.mjs, `JSON_MAX_DEPTH` = 127): the deepest nesting accepted, found by
    /// trying every depth, counting the top-level object; numbers out of a double's range; and a
    /// lone surrogate escape a later duplicate member would replace. Duplicates otherwise pass,
    /// the last one winning.
    #[test]
    fn serde_json_refuses_what_the_js_readers_scan_for() {
        let nested = |arrays: usize| {
            let open = "[".repeat(arrays);
            let close = "]".repeat(arrays);
            format!(r#"{{"folo_theme": 1, "id": "x", "name": "X", "mode": "dark", "$c": {open}{close}}}"#)
        };
        let deepest = (1..400).take_while(|&n| parse(&nested(n)).is_ok()).last().unwrap() + 1;
        assert_eq!(deepest, 127, "levels accepted, the top-level object counted");
        assert!(parse(&nested(127)).unwrap_err().starts_with("(file): not valid JSON ("));
        let head = r#"{"folo_theme": 1, "id": "x", "name": "X", "mode": "dark", "#;
        // The range limit is the correctly rounded one (`float_roundtrip`), as in JavaScript: the
        // largest double's neighbours above it round to Infinity, those below it to the double.
        for n in ["1e400", "-1e400", "1E309", &format!("1{}", "0".repeat(400)), "1.7976931348623159e308"] {
            let e = parse(&format!(r#"{head}"$c": {n}}}"#)).unwrap_err();
            assert!(e.starts_with("(file): not valid JSON ("), "{n}: {e}");
        }
        for n in
            ["1e-400", "0e999", "1.7976931348623157e308", "1.7976931348623158e308", "17976931348623158e292"]
        {
            assert!(parse(&format!(r#"{head}"$c": {n}}}"#)).is_ok(), "{n}");
        }
        // Every decimal reads as the nearest double, as Rust's own (correctly rounded) parser
        // gives it: the folo_theme and area_opacity checks see the value JavaScript sees.
        for n in [
            "0.99999999999999994",
            "0.99999999999999997",
            "1.00000000000000011",
            "1.0000000000000001110223",
            "0.1",
            "1.7976931348623158e308",
        ] {
            let v: serde_json::Value = serde_json::from_str(n).unwrap();
            assert_eq!(v.as_f64(), Some(n.parse::<f64>().unwrap()), "{n}");
        }
        let hidden = r#"{"name": "\uD800", "folo_theme": 1, "id": "demo", "name": "Demo", "mode": "dark"}"#;
        assert!(parse(hidden).unwrap_err().starts_with("(file): not valid JSON ("));
        let twice = r#"{"folo_theme": 1, "id": "x", "name": "First", "name": "Second", "mode": "dark"}"#;
        assert_eq!(parse(twice).unwrap().name, "Second");
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
