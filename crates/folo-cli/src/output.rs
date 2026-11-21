//! Output conventions shared by every command (see docs/CLI.md): `--json` puts one JSON object per line on stdout
//! and human text on stderr; errors are `{ "error": { code, message, hint } }`; exit codes are stable.

use std::io::Write;

use serde::Serialize;
use serde_json::{Value, json};

/// A path for a record: [`folo_api::json::path`], except a path that starts with `\\?\` and is not safe to show
/// without it, which stays as it was given. On Windows that is a path Windows' own normalization would change (a
/// volume path, `\\?\Volume{GUID}\out`, is relative without the prefix, and a name ending in a dot or a space
/// loses that ending) or one of 248 UTF-16 units or more (`LONG_PATH`: a program held to `MAX_PATH` may open it
/// only with the prefix). Elsewhere `\\?\` is an ordinary start of a file name, so every such path keeps its
/// spelling. "As it was given" is the path's lossy text, as for every path in a record: JSON holds Unicode only.
pub fn record_path(p: impl AsRef<std::path::Path>) -> String {
    let p = p.as_ref();
    let raw = p.to_string_lossy();
    if !raw.starts_with(r"\\?\") {
        return folo_api::json::path(p);
    }
    if !cfg!(windows) {
        return raw.into_owned();
    }
    let shown = folo_api::json::path(p);
    let native = shown.replace('/', r"\");
    let short = native.encode_utf16().count() < LONG_PATH;
    let same = std::path::absolute(&native).is_ok_and(|a| a.as_os_str() == native.as_str());
    if short && same { shown } else { raw.into_owned() }
}

/// The length from which [`record_path`] keeps a `\\?\` prefix: for a program that has not opted in to long paths,
/// Windows limits a folder's path to 248 characters (`MAX_PATH` less an 8.3 file name).
const LONG_PATH: usize = 248;

/// Stable process exit codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCode {
    /// Success.
    Ok = 0,
    /// The command ran and failed (IO, bind, bad data, ...).
    Runtime = 1,
    /// Invalid arguments or flags.
    Usage = 2,
    /// A named file, dataset, model or run does not exist.
    NotFound = 3,
    /// The command exists but is not implemented yet.
    NotImplemented = 4,
}

impl ExitCode {
    /// All codes in order.
    pub const ALL: [ExitCode; 5] =
        [ExitCode::Ok, ExitCode::Runtime, ExitCode::Usage, ExitCode::NotFound, ExitCode::NotImplemented];

    /// The exit code of an error code (`folo_api::codes`, the vocabulary the CLI shares with the API and MCP).
    pub fn of(code: &str) -> ExitCode {
        match folo_api::codes::exit_code(code) {
            2 => ExitCode::Usage,
            3 => ExitCode::NotFound,
            4 => ExitCode::NotImplemented,
            _ => ExitCode::Runtime,
        }
    }

    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            ExitCode::Ok => "ok",
            ExitCode::Runtime => "runtime_error",
            ExitCode::Usage => "usage_error",
            ExitCode::NotFound => "not_found",
            ExitCode::NotImplemented => "not_implemented",
        }
    }

    /// What it means.
    pub fn meaning(self) -> &'static str {
        match self {
            ExitCode::Ok => "success",
            ExitCode::Runtime => "the command ran and failed",
            ExitCode::Usage => "invalid arguments or flags",
            ExitCode::NotFound => "a named file, dataset, model or run does not exist",
            ExitCode::NotImplemented => "the command exists but is not implemented yet",
        }
    }
}

/// A command failure.
#[derive(Clone, Debug, PartialEq)]
pub struct CliError {
    /// Exit code.
    pub exit: ExitCode,
    /// Stable snake_case code, e.g. `not_implemented`.
    pub code: String,
    /// Human message.
    pub message: String,
    /// How to fix it.
    pub hint: Option<String>,
}

impl CliError {
    /// A new error.
    pub fn new(exit: ExitCode, code: &str, message: impl Into<String>) -> CliError {
        CliError { exit, code: code.into(), message: message.into(), hint: None }
    }

    /// Adds a hint.
    pub fn hint(mut self, hint: impl Into<String>) -> CliError {
        self.hint = Some(hint.into());
        self
    }

    /// The JSON envelope.
    pub fn to_json(&self) -> Value {
        json!({ "error": { "code": self.code, "message": self.message, "hint": self.hint } })
    }
}

/// Description of the error envelope for `folo describe`.
pub fn error_shape() -> Value {
    json!({ "error": { "code": "string (stable, snake_case)", "message": "string", "hint": "string | null" } })
}

/// Where a command writes.
#[derive(Clone, Copy, Debug)]
pub struct Out {
    /// `--json` mode.
    pub json: bool,
}

impl Out {
    /// Human text: stdout normally, stderr in `--json` mode.
    pub fn human(&self, text: impl AsRef<str>) {
        if self.json {
            eprintln!("{}", text.as_ref());
        } else {
            println!("{}", text.as_ref());
        }
    }

    /// A machine record: one compact JSON line on stdout in `--json` mode, nothing otherwise. Floats are written
    /// without f32 → f64 noise (`0.01`, not `0.009999999776482582`; `folo_api::json::clean_floats`).
    pub fn record<T: Serialize>(&self, value: &T) {
        if self.json {
            let Ok(value) = folo_api::json::to_value(value) else { return };
            let mut stdout = std::io::stdout().lock();
            let _ = serde_json::to_writer(&mut stdout, &value);
            let _ = writeln!(stdout);
            let _ = stdout.flush();
        }
    }

    /// Reports an error: the JSON envelope on stdout in `--json` mode, and the human form on stderr always.
    pub fn error(&self, e: &CliError) {
        self.record(&e.to_json());
        eprintln!("error: {}", e.message);
        if let Some(h) = &e.hint {
            eprintln!("  hint: {h}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// Every `.rs` file under `dir`.
    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                sources(&p, out);
            } else if p.extension().is_some_and(|e| e == "rs") {
                out.push(p);
            }
        }
    }

    /// What Windows opens is what the record names: the prefix goes only where that does not change it.
    #[test]
    fn a_verbatim_path_keeps_its_prefix_where_it_matters() {
        assert_eq!(super::record_path("runs/a"), "runs/a");
        if cfg!(windows) {
            assert_eq!(super::record_path(r"C:\data\x"), "C:/data/x");
            assert_eq!(super::record_path(r"\\?\C:\data\x"), "C:/data/x");
            assert_eq!(super::record_path(r"\\?\UNC\server\share\x"), "//server/share/x");
            for kept in [
                r"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\out",
                r"\\?\C:\studio\dist.",
                r"\\?\C:\data\b ",
            ] {
                assert_eq!(super::record_path(kept), kept);
            }
            // A long path keeps its prefix from 248 characters on, and loses it one character below.
            let long = |n: usize| format!(r"\\?\C:\{}", "a".repeat(n - 3));
            assert_eq!(super::record_path(long(248)), long(248));
            assert_eq!(super::record_path(long(247)), format!("C:/{}", "a".repeat(244)));
            // The length is in UTF-16 units: an emoji counts two (one as a character, four as UTF-8 bytes).
            let wide = |n: usize, tail: &str| format!(r"\\?\C:\{}{tail}", "\u{1F600}".repeat(n));
            assert_eq!(super::record_path(wide(122, "a")), wide(122, "a"), "3 + 244 + 1 = 248 units");
            assert_eq!(
                super::record_path(wide(122, "")),
                format!("C:/{}", "\u{1F600}".repeat(122)),
                "247 units"
            );
        } else {
            // Elsewhere the prefix is part of a file name: nothing is dropped or rewritten.
            for kept in [r"\\?\/studio", r"\\?\UNC\server\share\x", r"\\?\C:\data\x"] {
                assert_eq!(super::record_path(kept), kept);
            }
        }
    }

    /// A JSON key that holds a file or folder in the CLI's records.
    fn names_a_path(key: &str) -> bool {
        matches!(
            key,
            "path" | "root" | "file" | "yaml" | "input" | "output" | "model" | "data" | "annotations"
        ) || key == "saved"
            || ["_dir", "_path", "_file", "_to"].iter().any(|s| key.ends_with(s))
    }

    /// docs/CLI.md promises `/` in the paths of records on every OS, which a raw `PathBuf` breaks on Windows only. So
    /// this reads the source, on every OS: a `"key": value` in a `json!`, or a `record["key"] = value`, whose key
    /// names a path must build its value with `folo_api::json::path`, `record_path` or `rel` (the dataset-relative
    /// form). It is a text scan and only a net for the next field; the runtime assertions in `tests/` hold the four
    /// known ones. It cannot see a path under a key not named like one, a value on the line after its key, or a
    /// `Serialize` struct holding a `PathBuf` (none reaches a record today), and text in a string that looks like a
    /// field trips it.
    #[test]
    fn json_records_spell_paths_with_forward_slashes() {
        // Not raw paths: a command's words, and a string `json::path` made a few lines above.
        const FINE: [(&str, &str); 2] = [("describe.rs", "path"), ("stop.rs", "run_dir")];
        let mut files = Vec::new();
        sources(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
        let mut raw = Vec::new();
        for file in &files {
            let name = file.file_name().unwrap().to_string_lossy().into_owned();
            if name == "tests.rs" {
                continue;
            }
            let text = std::fs::read_to_string(file).unwrap();
            let code = text.split("#[cfg(test)]").next().unwrap();
            for line in code.lines().filter(|l| !l.trim_start().starts_with("//")) {
                for sep in ["\": ", "\"] = "] {
                    for (before, after) in line.split(sep).zip(line.split(sep).skip(1)) {
                        let key = before.rsplit('"').next().unwrap();
                        let value = after.split(", \"").next().unwrap();
                        if names_a_path(key)
                            && !value.contains("json::path")
                            && !value.contains("record_path")
                            && !value.starts_with("rel(")
                            && !FINE.contains(&(name.as_str(), key))
                        {
                            raw.push(format!("{name}: \"{key}\": {value}"));
                        }
                    }
                }
            }
        }
        assert!(raw.is_empty(), "use folo_api::json::path for: {raw:#?}");
    }
}
