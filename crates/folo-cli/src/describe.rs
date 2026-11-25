//! `folo describe`: the CLI surface as JSON, generated from clap's command tree.

use std::any::TypeId;
use std::collections::BTreeSet;
use std::path::PathBuf;

use clap::{Arg, ArgAction, ArgGroup, Command};
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

fn arg_label(arg: &Arg) -> String {
    if let Some(long) = arg.get_long() {
        format!("--{long}")
    } else if let Some(short) = arg.get_short() {
        format!("-{short}")
    } else {
        arg.get_id().as_str().to_string()
    }
}

fn group_is_multiple(g: &ArgGroup) -> bool {
    g.clone().is_multiple()
}

fn is_constraint_group(g: &ArgGroup) -> bool {
    g.is_required_set() || !group_is_multiple(g)
}

/// Symmetric conflict set for `arg` on `cmd`: forward `conflicts_with`, reverse `conflicts_with` declared on
/// peer args, and peer args in any mutually exclusive `ArgGroup` containing `arg`.
fn arg_conflicts(cmd: &Command, arg: &Arg) -> Vec<String> {
    let mut out = BTreeSet::new();
    let self_id = arg.get_id();
    for other in cmd.get_arg_conflicts_with(arg) {
        if !other.is_hide_set() && other.get_id() != self_id {
            out.insert(arg_label(other));
        }
    }
    for other in cmd.get_arguments().filter(|a| !a.is_hide_set() && a.get_id() != self_id) {
        if cmd.get_arg_conflicts_with(other).iter().any(|c| c.get_id() == self_id) {
            out.insert(arg_label(other));
        }
    }
    for group in cmd.get_groups().filter(|g| !group_is_multiple(g)) {
        if group.get_args().any(|id| id == self_id) {
            for id in group.get_args().filter(|id| *id != self_id) {
                if let Some(other) = cmd.get_arguments().find(|a| a.get_id() == id && !a.is_hide_set()) {
                    out.insert(arg_label(other));
                }
            }
        }
    }
    out.into_iter().collect()
}

fn describe_group(cmd: &Command, group: &ArgGroup) -> Value {
    let args: Vec<String> = group
        .get_args()
        .filter_map(|id| cmd.get_arguments().find(|a| a.get_id() == id && !a.is_hide_set()))
        .map(arg_label)
        .collect();
    json!({
        "name": group.get_id().as_str(),
        "required": group.is_required_set(),
        "exclusive": !group_is_multiple(group),
        "args": args,
    })
}

fn describe_arg(cmd: &Command, arg: &Arg) -> Value {
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
        .map(|p| {
            let aliases: Vec<&str> = p.get_name_and_aliases().skip(1).collect();
            json!({ "value": p.get_name(), "aliases": aliases, "help": styled(p.get_help()) })
        })
        .collect();
    let mut aliases: Vec<String> =
        arg.get_visible_aliases().unwrap_or_default().into_iter().map(|a| format!("--{a}")).collect();
    aliases.extend(arg.get_visible_short_aliases().unwrap_or_default().into_iter().map(|s| format!("-{s}")));
    json!({
        "name": arg.get_id().as_str(),
        "long": arg.get_long().map(|l| format!("--{l}")),
        "short": arg.get_short().map(|s| format!("-{s}")),
        "aliases": aliases,
        "positional": arg.is_positional(),
        "type": value_type(arg),
        "value_name": arg.get_value_names().filter(|_| !is_flag).and_then(|v| v.first()).map(|v| v.as_str().to_string()),
        "required": arg.is_required_set(),
        "multiple": multiple,
        "default": default,
        "possible_values": possible,
        "env": arg.get_env().map(|e| e.to_string_lossy().into_owned()),
        "conflicts_with": arg_conflicts(cmd, arg),
        "help": styled(arg.get_help()),
    })
}

fn describe_command(cmd: &Command, path: &[&str]) -> Value {
    let mut path: Vec<&str> = path.to_vec();
    path.push(cmd.get_name());
    let aliases: Vec<&str> = cmd.get_visible_aliases().collect();
    let groups: Vec<Value> =
        cmd.get_groups().filter(|g| is_constraint_group(g)).map(|g| describe_group(cmd, g)).collect();
    let args: Vec<Value> = cmd
        .get_arguments()
        .filter(|a| !a.is_global_set() && !a.is_hide_set())
        .map(|a| describe_arg(cmd, a))
        .collect();
    let subs: Vec<Value> =
        cmd.get_subcommands().filter(|c| !c.is_hide_set()).map(|c| describe_command(c, &path)).collect();
    json!({
        "name": cmd.get_name(),
        "aliases": aliases,
        "path": path.join(" "),
        "about": styled(cmd.get_about()),
        "groups": groups,
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
        cmd = cmd
            .get_subcommands()
            .filter(|c| !c.is_hide_set())
            .find(|c| c.get_name() == name || c.get_visible_aliases().any(|a| a == name))
            .ok_or_else(|| {
                let have: Vec<&str> =
                    cmd.get_subcommands().filter(|c| !c.is_hide_set()).map(Command::get_name).collect();
                CliError::new(
                    ExitCode::Usage,
                    "usage",
                    format!("`{} {name}` is not a command", names.join(" ")),
                )
                .hint(format!("`{}` has: {}", names.join(" "), have.join(", ")))
            })?;
        names.push(cmd.get_name());
    }
    names.pop();
    Ok(describe_command(cmd, &names))
}

/// The whole CLI surface, without the MCP tools.
fn describe_all(root: &Command) -> Value {
    let globals: Vec<Value> =
        root.get_arguments().filter(|a| a.is_global_set()).map(|a| describe_arg(root, a)).collect();
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
