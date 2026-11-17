//! Generates the theme's token tables from `design/tokens.json`, the design system shared with FOLO
//! Studio. Output: `$OUT_DIR/tokens.rs`, included by `src/theme.rs`.
//!
//! Colours become `Hsla` fields of a `Colors` struct: one constructor per built-in theme of `themes`
//! (its mode's base palette, `color.dark` or `color.light`, with its overrides merged over it), a
//! `THEMES` table in tokens.json's order, the token paths (`TOKENS`) and setters by path for custom
//! themes (design/THEMES.md). Both base palettes must have the same keys. Type steps, space, radii,
//! sizes, durations and easings become constants.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use serde_json::Value;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let path = manifest.join("../../design/tokens.json");
    println!("cargo:rerun-if-changed={}", path.display());
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("reading {}: {e} (FOLO Desktop builds from the repo checkout)", path.display())
    });
    let tokens: Value = serde_json::from_str(&text).expect("design/tokens.json is valid JSON");
    let out = generate(&tokens);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out_dir.join("tokens.rs"), out).unwrap();

    // Embed every icon under assets/icons.
    let icons = manifest.join("assets/icons");
    println!("cargo:rerun-if-changed={}", icons.display());
    let mut names: Vec<String> = std::fs::read_dir(&icons)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".svg"))
        .collect();
    names.sort();
    let mut table = String::from(
        "pub const ICONS: &[(&str, &[u8])] = &[
",
    );
    for n in names {
        table.push_str(&format!(
            "    (\"icons/{n}\", include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/assets/icons/{n}\"))),
"
        ));
    }
    table.push_str(
        "];
",
    );
    std::fs::write(out_dir.join("icons.rs"), table).unwrap();

    // The Windows app icon (taskbar, Alt-Tab, Explorer): folo-ui loads icon resource 1 from the
    // executable. And the application manifest (RT_MANIFEST 1: DPI awareness, Common Controls 6).
    // The linker (rust-lld, the root .cargo/config.toml, or link.exe) takes a compiled resource file
    // as an input, so both become a .res here, in plain Rust:
    // no resource compiler or extra crate needed.
    let ico = manifest.join("assets/folo.ico");
    let app_manifest = manifest.join("assets/folo-desktop.manifest");
    println!("cargo:rerun-if-changed={}", ico.display());
    println!("cargo:rerun-if-changed={}", app_manifest.display());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "windows" && target_env == "msvc" {
        let bytes = std::fs::read(&ico).unwrap_or_else(|e| panic!("reading {}: {e}", ico.display()));
        let xml = std::fs::read(&app_manifest)
            .unwrap_or_else(|e| panic!("reading {}: {e}", app_manifest.display()));
        let res = out_dir.join("folo-icon.res");
        std::fs::write(&res, windows_res(&bytes, &xml)).unwrap();
        println!("cargo:rustc-link-arg-bins={}", res.display());
    }
}

/// A compiled resource file (`.res`) holding an `.ico` as icon group 1 (its images as icons 1..n) and
/// an application manifest as `RT_MANIFEST` 1.
fn windows_res(ico: &[u8], app_manifest: &[u8]) -> Vec<u8> {
    const RT_ICON: u16 = 3;
    const RT_GROUP_ICON: u16 = 14;
    const RT_MANIFEST: u16 = 24;
    let u16_at = |i: usize| u16::from_le_bytes([ico[i], ico[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes(ico[i..i + 4].try_into().unwrap());
    assert!(ico.len() >= 6 && u16_at(0) == 0 && u16_at(2) == 1, "assets/folo.ico is not an icon file");
    let count = u16_at(4) as usize;

    // One resource: header (numeric type and name, language neutral), then the data, both
    // padded to 4 bytes.
    fn entry(out: &mut Vec<u8>, kind: u16, id: u16, flags: u16, data: &[u8]) {
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(32u32.to_le_bytes());
        out.extend([0xFF, 0xFF]);
        out.extend(kind.to_le_bytes());
        out.extend([0xFF, 0xFF]);
        out.extend(id.to_le_bytes());
        out.extend(0u32.to_le_bytes()); // data version
        out.extend(flags.to_le_bytes());
        out.extend(0u16.to_le_bytes()); // LANG_NEUTRAL
        out.extend(0u32.to_le_bytes()); // version
        out.extend(0u32.to_le_bytes()); // characteristics
        out.extend(data);
        out.resize(out.len().next_multiple_of(4), 0);
    }
    let mut out = Vec::new();
    entry(&mut out, 0, 0, 0, &[]); // a .res file starts with an empty entry
    let mut group = Vec::new();
    group.extend([0, 0, 1, 0]);
    group.extend((count as u16).to_le_bytes());
    for i in 0..count {
        let e = 6 + 16 * i;
        let (size, offset) = (u32_at(e + 8) as usize, u32_at(e + 12) as usize);
        let image = ico.get(offset..offset + size).expect("assets/folo.ico: an image runs past the end");
        let id = i as u16 + 1;
        entry(&mut out, RT_ICON, id, 0x1010, image);
        // GRPICONDIRENTRY: the ICONDIRENTRY's first 12 bytes, then the icon's resource id.
        group.extend(&ico[e..e + 12]);
        group.extend(id.to_le_bytes());
    }
    entry(&mut out, RT_GROUP_ICON, 1, 0x1030, &group);
    entry(&mut out, RT_MANIFEST, 1, 0x0030, app_manifest);
    out
}

/// Parse `#RRGGBB` / `#RRGGBBAA` into `0xRRGGBBAA`.
fn color(v: &Value, at: &str) -> u32 {
    let s = v.as_str().unwrap_or_else(|| panic!("{at}: expected a colour string"));
    let hex = s.strip_prefix('#').unwrap_or_else(|| panic!("{at}: colour must start with #"));
    match hex.len() {
        6 => u32::from_str_radix(hex, 16).unwrap() << 8 | 0xFF,
        8 => u32::from_str_radix(hex, 16).unwrap(),
        _ => panic!("{at}: colour must be #RRGGBB or #RRGGBBAA, got {s}"),
    }
}

fn ident(key: &str) -> String {
    let mut s: String =
        key.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    s
}

#[derive(Clone, PartialEq)]
enum Field {
    Color(u32),
    Colors(Vec<u32>),
    Number(f64),
}

impl Field {
    fn kind(&self) -> String {
        match self {
            Field::Color(_) => "a colour".into(),
            Field::Colors(v) => format!("an array of {} colours", v.len()),
            Field::Number(_) => "a number".into(),
        }
    }
}

/// One colour token: its dotted path in tokens.json (`bg.app`, `chart.series`) and its value.
struct Token {
    path: String,
    field: Field,
}

/// Flatten a palette's colour tree into `group_key` fields (each with its dotted path), skipping `$`
/// comments. `at` names the tree in panics (`color.dark`, `themes.midnight.color`).
fn flatten(at: &str, prefix: (&str, &str), v: &Value, out: &mut BTreeMap<String, Token>) {
    let obj = v.as_object().unwrap_or_else(|| panic!("{at}: a colour group must be an object"));
    for (k, v) in obj {
        if k.starts_with('$') {
            continue;
        }
        let name = if prefix.0.is_empty() { ident(k) } else { format!("{}_{}", prefix.0, ident(k)) };
        let path = if prefix.1.is_empty() { k.clone() } else { format!("{}.{k}", prefix.1) };
        let here = format!("{at}.{k}");
        let field = match v {
            Value::Object(_) => {
                flatten(&here, (&name, &path), v, out);
                continue;
            }
            Value::Array(items) => Field::Colors(items.iter().map(|c| color(c, &here)).collect()),
            Value::Number(n) => Field::Number(n.as_f64().unwrap()),
            Value::String(_) => Field::Color(color(v, &here)),
            _ => panic!("{here}: unsupported token value"),
        };
        out.insert(name, Token { path, field });
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

/// A built-in theme of tokens.json's `themes`: its mode's base palette with its own `color` merged
/// over it key by key (arrays replace whole). Panics with the token's path on a malformed theme.
struct ThemeDef {
    id: String,
    name: String,
    dark: bool,
    description: String,
    palette: BTreeMap<String, Token>,
}

fn resolve_themes(
    t: &Value,
    dark: &BTreeMap<String, Token>,
    light: &BTreeMap<String, Token>,
) -> Vec<ThemeDef> {
    let themes = t["themes"].as_object().expect("tokens.json: `themes` is an object (design/THEMES.md)");
    let mut out = Vec::new();
    for (id, v) in themes {
        if id.starts_with('$') {
            continue;
        }
        let at = format!("themes.{id}");
        assert!(
            !id.is_empty()
                && id.len() <= 40
                && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                && !id.starts_with('-'),
            "{at}: a theme id is 1 to 40 of a-z, 0-9 and -, starting with a letter or digit"
        );
        let obj = v.as_object().unwrap_or_else(|| panic!("{at}: a theme is an object"));
        for k in obj.keys() {
            assert!(
                k.starts_with('$') || ["name", "mode", "description", "color"].contains(&k.as_str()),
                "{at}.{k}: unknown key (a theme has name, mode, description and color)"
            );
        }
        let text = |k: &str| -> String {
            obj.get(k)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("{at}.{k}: missing, or not a string"))
                .to_string()
        };
        let name = text("name");
        let description = text("description");
        let is_dark = match text("mode").as_str() {
            "dark" => true,
            "light" => false,
            m => panic!("{at}.mode: `{m}` is not dark or light"),
        };
        let base = if is_dark { dark } else { light };
        let mut over = BTreeMap::new();
        if let Some(c) = obj.get("color") {
            flatten(&format!("{at}.color"), ("", ""), c, &mut over);
        }
        let mut palette: BTreeMap<String, Token> = base
            .iter()
            .map(|(k, tk)| (k.clone(), Token { path: tk.path.clone(), field: tk.field.clone() }))
            .collect();
        for (k, tk) in over {
            let Some(b) = palette.get_mut(&k) else {
                panic!("{at}.color.{}: not a colour token (see color.dark)", tk.path);
            };
            let same = match (&b.field, &tk.field) {
                (Field::Color(_), Field::Color(_)) | (Field::Number(_), Field::Number(_)) => true,
                (Field::Colors(a), Field::Colors(b)) => a.len() == b.len(),
                _ => false,
            };
            assert!(same, "{at}.color.{}: expected {}, got {}", tk.path, b.field.kind(), tk.field.kind());
            if let Field::Number(n) = tk.field {
                assert!((0.0..=1.0).contains(&n), "{at}.color.{}: expected a number from 0 to 1", tk.path);
            }
            b.field = tk.field;
        }
        out.push(ThemeDef { id: id.clone(), name, dark: is_dark, description, palette });
    }
    for (mode, default) in [(true, "dark"), (false, "light")] {
        let first =
            out.iter().find(|d| d.dark == mode).unwrap_or_else(|| panic!("themes: no {default} theme"));
        assert!(
            first.id == default,
            "themes: the first {default} theme must be `{default}` (the mode's default), not `{}`",
            first.id
        );
        let base = if mode { dark } else { light };
        assert!(
            first.palette.iter().all(|(k, tk)| base[k].field == tk.field),
            "themes.{default}: the default theme has no overrides (design/THEMES.md)"
        );
    }
    out
}

/// `0xRRGGBBAA` as Rust source.
fn rgba_src(c: u32) -> String {
    format!("rgba32(0x{c:08X})")
}

fn generate_colors(s: &mut String, t: &Value) {
    let palettes = &t["color"];
    let mut dark = BTreeMap::new();
    let mut light = BTreeMap::new();
    flatten("color.dark", ("", ""), &palettes["dark"], &mut dark);
    flatten("color.light", ("", ""), &palettes["light"], &mut light);
    let dk: Vec<_> = dark.iter().map(|(k, tk)| (k, &tk.path, tk.field.kind())).collect();
    let lk: Vec<_> = light.iter().map(|(k, tk)| (k, &tk.path, tk.field.kind())).collect();
    assert_eq!(dk, lk, "dark and light themes must define the same colour tokens");

    s.push_str("/// Every colour token of one theme.\n#[derive(Clone, Copy, Debug, PartialEq)]\npub struct Colors {\n");
    for (k, tk) in &dark {
        let ty = match &tk.field {
            Field::Color(_) => "Hsla".to_string(),
            Field::Colors(v) => format!("[Hsla; {}]", v.len()),
            Field::Number(_) => "f32".to_string(),
        };
        writeln!(s, "    /// `{}`", tk.path).unwrap();
        writeln!(s, "    pub {k}: {ty},").unwrap();
    }
    s.push_str("}\n\n");

    let themes = resolve_themes(t, &dark, &light);
    for th in &themes {
        writeln!(
            s,
            "/// The `{}` theme ({}).\npub fn theme_{}() -> Colors {{\n    Colors {{",
            th.id,
            th.name,
            ident(&th.id)
        )
        .unwrap();
        for (k, tk) in &th.palette {
            let v = match &tk.field {
                Field::Color(c) => rgba_src(*c),
                Field::Colors(cs) => {
                    format!("[{}]", cs.iter().map(|c| rgba_src(*c)).collect::<Vec<_>>().join(", "))
                }
                Field::Number(n) => format!("{n:?}_f32"),
            };
            writeln!(s, "        {k}: {v},").unwrap();
        }
        s.push_str("    }\n}\n\n");
    }

    s.push_str("/// A built-in theme: tokens.json's `themes`, in its order (the first of each mode is its default).\n");
    s.push_str("#[derive(Clone, Copy, Debug)]\npub struct BuiltinTheme {\n    pub id: &'static str,\n    pub name: &'static str,\n    pub mode: Mode,\n    pub description: &'static str,\n    pub colors: fn() -> Colors,\n}\n\n");
    writeln!(s, "pub const THEMES: [BuiltinTheme; {}] = [", themes.len()).unwrap();
    for th in &themes {
        writeln!(
            s,
            "    BuiltinTheme {{ id: {:?}, name: {:?}, mode: Mode::{}, description: {:?}, colors: theme_{} }},",
            th.id,
            th.name,
            if th.dark { "Dark" } else { "Light" },
            th.description,
            ident(&th.id)
        )
        .unwrap();
    }
    s.push_str("];\n\n");

    // The token paths a custom theme may set, and the setters by path.
    s.push_str(
        "/// What a colour token holds: one colour, a fixed number of them, or a number from 0 to 1.\n",
    );
    s.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub enum TokenKind {\n    Color,\n    Colors(usize),\n    Number,\n}\n\n");
    s.push_str("/// Every colour token by its path in tokens.json (`bg.app`, `chart.series`).\n");
    writeln!(s, "pub const TOKENS: [(&str, TokenKind); {}] = [", dark.len()).unwrap();
    for tk in dark.values() {
        let kind = match &tk.field {
            Field::Color(_) => "TokenKind::Color".to_string(),
            Field::Colors(v) => format!("TokenKind::Colors({})", v.len()),
            Field::Number(_) => "TokenKind::Number".to_string(),
        };
        writeln!(s, "    ({:?}, {kind}),", tk.path).unwrap();
    }
    s.push_str("];\n\n");
    s.push_str("impl Colors {\n");
    s.push_str("    /// Sets the colour token at `path` (`bg.app`; an array's item as `chart.series.0`); `false`\n    /// when no colour token has that path.\n");
    s.push_str("    pub fn set(&mut self, path: &str, value: Hsla) -> bool {\n        match path {\n");
    for (k, tk) in &dark {
        match &tk.field {
            Field::Color(_) => writeln!(s, "            {:?} => self.{k} = value,", tk.path).unwrap(),
            Field::Colors(v) => {
                for i in 0..v.len() {
                    writeln!(s, "            \"{}.{i}\" => self.{k}[{i}] = value,", tk.path).unwrap();
                }
            }
            Field::Number(_) => {}
        }
    }
    s.push_str("            _ => return false,\n        }\n        true\n    }\n\n");
    s.push_str(
        "    /// Sets the number token at `path` (`chart.area_opacity`); `false` when there is none.\n",
    );
    s.push_str("    pub fn set_number(&mut self, path: &str, value: f32) -> bool {\n        match path {\n");
    for (k, tk) in &dark {
        if let Field::Number(_) = tk.field {
            writeln!(s, "            {:?} => self.{k} = value,", tk.path).unwrap();
        }
    }
    s.push_str("            _ => return false,\n        }\n        true\n    }\n}\n\n");
}

fn generate(t: &Value) -> String {
    let mut s = String::from("// @generated by build.rs from design/tokens.json. Do not edit.\n\n");
    generate_colors(&mut s, t);

    // Detection classes (theme independent).
    let classes = &t["classes"];
    let palette: Vec<String> = classes["palette"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| format!("0x{:08X}", color(c, "classes.palette")))
        .collect();
    writeln!(s, "pub const CLASS_PALETTE: [u32; {}] = [{}];", palette.len(), palette.join(", ")).unwrap();
    writeln!(s, "pub const CLASS_CHIP_TEXT: u32 = 0x{:08X};", color(&classes["chip_text"], "chip_text"))
        .unwrap();
    writeln!(s, "pub const CLASS_HALO: u32 = 0x{:08X};", color(&classes["halo"], "halo")).unwrap();
    writeln!(s, "pub const CLASS_FILL_OPACITY: f32 = {:?};\n", num(&classes["fill_opacity"])).unwrap();

    // Type scale.
    s.push_str("/// One step of the type scale: size and line height in px, tracking in em.\n");
    s.push_str("#[derive(Clone, Copy, Debug, PartialEq)]\npub struct TypeStep {\n    pub size: f32,\n    pub line: f32,\n    pub tracking: f32,\n    pub weight: u16,\n}\n\n");
    s.push_str("pub mod ty {\n    use super::TypeStep;\n");
    for (k, v) in t["type"].as_object().unwrap() {
        if k.starts_with('$') {
            continue;
        }
        let name = ident(k).to_uppercase();
        let name = match name.strip_prefix('_') {
            Some(rest) => format!("X{rest}"),
            None => name,
        };
        writeln!(
            s,
            "    pub const {name}: TypeStep = TypeStep {{ size: {:?}, line: {:?}, tracking: {:?}, weight: {} }};",
            num(&v["size"]) as f32,
            num(&v["line"]) as f32,
            num(&v["tracking"]) as f32,
            num(&v["weight"]) as u16
        )
        .unwrap();
    }
    s.push_str("}\n\n");

    let flat_group = |s: &mut String, module: &str, v: &Value| {
        writeln!(s, "pub mod {module} {{").unwrap();
        let mut stack = vec![(String::new(), v)];
        while let Some((prefix, v)) = stack.pop() {
            for (k, v) in v.as_object().unwrap() {
                if k.starts_with('$') {
                    continue;
                }
                let name = if prefix.is_empty() { ident(k) } else { format!("{prefix}_{}", ident(k)) };
                match v {
                    Value::Number(n) => {
                        let mut id = name.to_uppercase().replace("__", "_");
                        if id.starts_with('_') {
                            id = format!("S{id}");
                        }
                        writeln!(s, "    pub const {id}: f32 = {:?};", n.as_f64().unwrap() as f32).unwrap();
                    }
                    Value::Object(_) => stack.push((name, v)),
                    _ => {}
                }
            }
        }
        s.push_str("}\n\n");
    };
    flat_group(&mut s, "space", &t["space"]);
    flat_group(&mut s, "radius", &t["radius"]);
    flat_group(&mut s, "size", &t["size"]);
    flat_group(&mut s, "layout", &t["layout"]);
    flat_group(&mut s, "duration", &t["motion"]["duration"]);

    // Shadows: layers of (x, y, blur, spread, 0xRRGGBBAA).
    s.push_str("pub mod shadow {\n");
    for theme in ["dark", "light"] {
        for (k, layers) in t["shadow"][theme].as_object().unwrap() {
            if k.starts_with('$') {
                continue;
            }
            let items: Vec<String> = layers
                .as_array()
                .unwrap()
                .iter()
                .map(|l| {
                    format!(
                        "({:?}, {:?}, {:?}, {:?}, 0x{:08X})",
                        num(&l["x"]) as f32,
                        num(&l["y"]) as f32,
                        num(&l["blur"]) as f32,
                        num(&l["spread"]) as f32,
                        color(&l["color"], "shadow")
                    )
                })
                .collect();
            writeln!(
                s,
                "    pub const {}_{}: &[(f32, f32, f32, f32, u32)] = &[{}];",
                theme.to_uppercase(),
                ident(k).to_uppercase(),
                items.join(", ")
            )
            .unwrap();
        }
    }
    s.push_str("}\n\n");

    s.push_str("pub mod easing {\n");
    for (k, v) in t["motion"]["easing"].as_object().unwrap() {
        if k.starts_with('$') {
            continue;
        }
        let p: Vec<String> = v.as_array().unwrap().iter().map(|x| format!("{:?}", num(x) as f32)).collect();
        writeln!(s, "    pub const {}: [f32; 4] = [{}];", ident(k).to_uppercase(), p.join(", ")).unwrap();
    }
    s.push_str("}\n");
    s
}
