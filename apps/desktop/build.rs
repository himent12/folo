//! Generates the GPUI token tables from `design/tokens.json`, the design system shared with FOLO
//! Studio. Output: `$OUT_DIR/tokens.rs`, included by `src/theme.rs`.
//!
//! Colours become `Hsla` fields of a `Colors` struct (one constructor per theme; both themes must
//! have the same keys). Type steps, space, radii, sizes, durations and easings become constants.

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

    // The Windows app icon (taskbar, Alt-Tab, Explorer): GPUI loads icon resource 1 from the
    // executable. link.exe takes a compiled resource file as an input, so the .ico becomes a .res
    // here, in plain Rust: no resource compiler or extra crate needed.
    let ico = manifest.join("assets/folo.ico");
    println!("cargo:rerun-if-changed={}", ico.display());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "windows" && target_env == "msvc" {
        let bytes = std::fs::read(&ico).unwrap_or_else(|e| panic!("reading {}: {e}", ico.display()));
        let res = out_dir.join("folo-icon.res");
        std::fs::write(&res, icon_res(&bytes)).unwrap();
        println!("cargo:rustc-link-arg-bins={}", res.display());
    }
}

/// A compiled resource file (`.res`) holding an `.ico` as icon group 1 (its images as icons 1..n).
fn icon_res(ico: &[u8]) -> Vec<u8> {
    const RT_ICON: u16 = 3;
    const RT_GROUP_ICON: u16 = 14;
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

enum Field {
    Color(u32),
    Colors(Vec<u32>),
    Number(f64),
}

/// Flatten a theme's colour tree into `group_key` fields, skipping `$` comments.
fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Field>) {
    for (k, v) in v.as_object().expect("colour group is an object") {
        if k.starts_with('$') {
            continue;
        }
        let name = if prefix.is_empty() { ident(k) } else { format!("{prefix}_{}", ident(k)) };
        match v {
            Value::Object(_) => flatten(&name, v, out),
            Value::Array(items) => {
                let cs = items.iter().map(|c| color(c, &name)).collect();
                out.insert(name, Field::Colors(cs));
            }
            Value::Number(n) => {
                out.insert(name, Field::Number(n.as_f64().unwrap()));
            }
            Value::String(_) => {
                let c = color(v, &name);
                out.insert(name, Field::Color(c));
            }
            _ => panic!("{name}: unsupported token value"),
        }
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn generate(t: &Value) -> String {
    let mut s = String::from("// @generated by build.rs from design/tokens.json. Do not edit.\n\n");
    let themes = &t["color"];
    let mut dark = BTreeMap::new();
    let mut light = BTreeMap::new();
    flatten("", &themes["dark"], &mut dark);
    flatten("", &themes["light"], &mut light);
    let dk: Vec<_> = dark.keys().collect();
    let lk: Vec<_> = light.keys().collect();
    assert_eq!(dk, lk, "dark and light themes must define the same colour tokens");

    s.push_str("/// Every colour token of one theme.\n#[derive(Clone, Copy, Debug, PartialEq)]\npub struct Colors {\n");
    for (k, f) in &dark {
        let ty = match f {
            Field::Color(_) => "Hsla".to_string(),
            Field::Colors(v) => format!("[Hsla; {}]", v.len()),
            Field::Number(_) => "f32".to_string(),
        };
        writeln!(s, "    pub {k}: {ty},").unwrap();
    }
    s.push_str("}\n\n");
    for (name, map) in [("dark", &dark), ("light", &light)] {
        writeln!(s, "pub fn {name}() -> Colors {{\n    Colors {{").unwrap();
        for (k, f) in map.iter() {
            let v = match f {
                Field::Color(c) => format!("rgba32(0x{c:08X})"),
                Field::Colors(cs) => {
                    let items: Vec<String> = cs.iter().map(|c| format!("rgba32(0x{c:08X})")).collect();
                    format!("[{}]", items.join(", "))
                }
                Field::Number(n) => format!("{n:?}_f32"),
            };
            writeln!(s, "        {k}: {v},").unwrap();
        }
        s.push_str("    }\n}\n\n");
    }

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
