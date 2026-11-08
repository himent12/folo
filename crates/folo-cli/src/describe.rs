//! `folo describe`: the CLI surface as JSON, generated from clap's command tree.

use std::any::TypeId;
use std::path::PathBuf;

use clap::{Arg, ArgAction, Command};
use serde_json::{Value, json};

use crate::output::{CliError, ExitCode, error_shape};

fn value_type(arg: &Arg) -> &'static str {
    if matches!(arg.get_action(), ArgAction::SetTrue | ArgAction::SetFalse) {
        return "boolean";
    }
    if matches!(arg.get_action(), ArgAction::Count) {
        return "count";
    }
    if arg.get_action().takes_values() && !arg.get_possible_values().is_empty() {
        return "enum";
    }
    let t = arg.get_value_parser().type_id();
    let is = |id: TypeId| t == id;
    if is(TypeId::of::<String>()) {
        "string"
    } else if is(TypeId::of::<PathBuf>()) {
        "path"
    } else if is(TypeId::of::<bool>()) {
        "boolean"
    } else if [
        TypeId::of::<u8>(),
        TypeId::of::<u16>(),
        TypeId::of::<u32>(),
        TypeId::of::<u64>(),
        TypeId::of::<usize>(),
        TypeId::of::<i32>(),
        TypeId::of::<i64>(),
    ]
    .into_iter()
    .any(is)
    {
        "integer"
    } else if is(TypeId::of::<f32>()) || is(TypeId::of::<f64>()) {
        "number"
    } else {
        "string"
    }
}

fn styled(s: Option<&clap::builder::StyledStr>) -> Value {
    s.map_or(Value::Null, |s| Value::String(s.to_string()))
}

fn describe_arg(arg: &Arg) -> Value {
    let takes_values = arg.get_action().takes_values();
    let multiple = matches!(arg.get_action(), ArgAction::Append)
        || arg.get_num_args().is_some_and(|r| r.max_values() > 1);
    let defaults: Vec<String> =
        arg.get_default_values().iter().map(|v| v.to_string_lossy().into_owned()).collect();
    let default = match (takes_values, defaults.len()) {
        (false, _) => Value::Bool(false),
        (true, 0) => Value::Null,
        (true, 1) if !multiple => Value::String(defaults[0].clone()),
        (true, _) => json!(defaults),
    };
    let is_flag = !takes_values;
    let possible: Vec<Value> = arg
        .get_possible_values()
        .iter()
        .filter(|p| !is_flag && !p.is_hide_set())
        .map(|p| json!({ "value": p.get_name(), "help": styled(p.get_help()) }))
        .collect();
    json!({
        "name": arg.get_id().as_str(),
        "long": arg.get_long().map(|l| format!("--{l}")),
        "short": arg.get_short().map(|s| format!("-{s}")),
        "positional": arg.is_positional(),
        "type": value_type(arg),
        "value_name": arg.get_value_names().filter(|_| !is_flag).and_then(|v| v.first()).map(|v| v.as_str().to_string()),
        "required": arg.is_required_set(),
        "multiple": multiple,
        "default": default,
        "possible_values": possible,
        "env": arg.get_env().map(|e| e.to_string_lossy().into_owned()),
        "help": styled(arg.get_help()),
    })
}

fn describe_command(cmd: &Command, path: &[&str]) -> Value {
    let mut path: Vec<&str> = path.to_vec();
    path.push(cmd.get_name());
    let args: Vec<Value> =
        cmd.get_arguments().filter(|a| !a.is_global_set() && !a.is_hide_set()).map(describe_arg).collect();
    let subs: Vec<Value> =
        cmd.get_subcommands().filter(|c| !c.is_hide_set()).map(|c| describe_command(c, &path)).collect();
    json!({
        "name": cmd.get_name(),
        "path": path.join(" "),
        "about": styled(cmd.get_about()),
        "args": args,
        "subcommands": subs,
    })
}

/// The CLI surface: everything (`path` empty), or the command `path` names (e.g. `["dataset", "check"]`).
/// `mcp_tools` adds the `folo mcp` tool schemas to the full surface.
pub fn describe(root: &Command, path: &[String], mcp_tools: bool) -> Result<Value, CliError> {
    if path.is_empty() {
        let mut all = describe_all(root);
        if mcp_tools {
            // The `folo mcp` tools with the same JSON Schemas `tools/list` publishes (docs/MCP.md).
            all["mcp_tools"] = folo_mcp::describe_tools();
        }
        return Ok(all);
    }
    let mut cmd = root;
    let mut names = vec![root.get_name()];
    for name in path {
        cmd = cmd.get_subcommands().filter(|c| !c.is_hide_set()).find(|c| c.get_name() == name).ok_or_else(
            || {
                let have: Vec<&str> =
                    cmd.get_subcommands().filter(|c| !c.is_hide_set()).map(Command::get_name).collect();
                CliError::new(
                    ExitCode::Usage,
                    "usage",
                    format!("`{} {name}` is not a command", names.join(" ")),
                )
                .hint(format!("`{}` has: {}", names.join(" "), have.join(", ")))
            },
        )?;
        names.push(cmd.get_name());
    }
    names.pop();
    Ok(describe_command(cmd, &names))
}

/// The whole CLI surface, without the MCP tools.
fn describe_all(root: &Command) -> Value {
    let globals: Vec<Value> = root.get_arguments().filter(|a| a.is_global_set()).map(describe_arg).collect();
    let commands: Vec<Value> = root
        .get_subcommands()
        .filter(|c| !c.is_hide_set())
        .map(|c| describe_command(c, &[root.get_name()]))
        .collect();
    json!({
        "name": root.get_name(),
        "version": env!("CARGO_PKG_VERSION"),
        "about": styled(root.get_about()),
        "global_args": globals,
        "conventions": {
            "json": "with --json, stdout carries one JSON object per line and human text goes to stderr",
            "error": error_shape(),
            "exit_codes": ExitCode::ALL.iter().map(|c| json!({ "code": *c as u8, "name": c.name(), "meaning": c.meaning() })).collect::<Vec<_>>(),
            "docs": "docs/CLI.md",
        },
        "commands": commands,
    })
}
