//! Output conventions shared by every command (see docs/CLI.md): `--json` puts one JSON object per line on stdout
//! and human text on stderr; errors are `{ "error": { code, message, hint } }`; exit codes are stable.

use std::io::Write;

use serde::Serialize;
use serde_json::{Value, json};

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
