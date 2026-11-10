//! `folo cloud`: cloud providers, connections, offers and instances (docs/CLI.md, Cloud; docs/CLOUD.md) on the data
//! dir, through `folo_server::cloud`, with or without a server running. A connection's secrets come from the
//! environment or stdin, never from the command line.

use std::collections::BTreeMap;
use std::io::BufRead;

use folo_api::{CloudConnection, CloudInstance, CloudOffer, CloudProvider, ConnectCloud};
use folo_server::cloud::Cloud;
use serde_json::json;

use crate::cli::{CloudCommand, CloudConnectArgs, CloudDirArgs};
use crate::output::{CliError, ExitCode, Out};

pub fn run(cmd: CloudCommand, out: Out) -> Result<(), CliError> {
    match cmd {
        CloudCommand::Providers(dir) => {
            let providers = open(&dir)?.providers();
            for p in &providers {
                out.record(p);
            }
            providers_table(&providers, out);
            Ok(())
        }
        CloudCommand::Connect(args) => connect(args, out),
        CloudCommand::Connections(dir) => {
            let list = open(&dir)?.connections().map_err(api_error)?;
            for c in &list {
                out.record(c);
            }
            connections_table(&list, out);
            Ok(())
        }
        CloudCommand::Check(args) => {
            let c = open(&args.dir)?.check(&args.id).map_err(api_error)?;
            out.record(&c);
            connections_table(std::slice::from_ref(&c), out);
            Ok(())
        }
        CloudCommand::Disconnect(args) => {
            open(&args.dir)?.disconnect(&args.id).map_err(api_error)?;
            out.record(&json!({ "event": "disconnected", "id": args.id }));
            out.human(format!("disconnected {}", args.id));
            Ok(())
        }
        CloudCommand::Offers(args) => {
            let list = open(&args.dir)?.offers(&args.connection, args.refresh).map_err(api_error)?;
            for o in &list.offers {
                out.record(o);
            }
            offers_table(&list.offers, out);
            Ok(())
        }
        CloudCommand::Instances(dir) => {
            let list = open(&dir)?.instances();
            for i in &list {
                out.record(i);
            }
            instances_table(&list, out);
            Ok(())
        }
        CloudCommand::Terminate(args) => {
            let i = open(&args.dir)?.terminate(&args.id).map_err(api_error)?;
            out.record(&i);
            instances_table(std::slice::from_ref(&i), out);
            Ok(())
        }
    }
}

/// The control plane of the command's data dir.
fn open(dir: &CloudDirArgs) -> Result<Cloud, CliError> {
    Ok(Cloud::open(&crate::data_dir(dir.data_dir.clone())?))
}

/// A store error as the CLI reports it: the same code, its exit code.
fn api_error(e: folo_server::error::Error) -> CliError {
    let err = CliError::new(ExitCode::of(&e.body.code), &e.body.code, e.body.message);
    match e.body.hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

fn usage(message: impl Into<String>) -> CliError {
    CliError::new(ExitCode::Usage, "usage", message).hint("see `folo cloud connect --help`")
}

/// `KEY=VALUE` split at its first `=`.
fn pair<'a>(flag: &str, s: &'a str) -> Result<(&'a str, &'a str), CliError> {
    s.split_once('=').filter(|(k, _)| !k.trim().is_empty()).map(|(k, v)| (k.trim(), v)).ok_or_else(|| {
        match flag {
            // What was given may be the secret itself: it is not repeated.
            "--secret" => usage("--secret takes KEY=env:VAR or KEY=-"),
            _ => usage(format!("{flag} takes KEY=VALUE, got `{s}`")),
        }
    })
}

/// The value of a `--secret KEY=SOURCE`: the environment variable of `env:VAR`, or a line of stdin for `-`
/// (`stdin_used` refuses a second one). A literal is a usage error: the command line is not a place for secrets.
fn secret(key: &str, source: &str, stdin_used: &mut bool) -> Result<String, CliError> {
    if let Some(var) = source.strip_prefix("env:") {
        return match std::env::var(var) {
            Ok(v) if !v.trim().is_empty() => Ok(v),
            _ => Err(usage(format!("--secret {key}: the environment variable {var} is not set"))),
        };
    }
    if source == "-" {
        if std::mem::replace(stdin_used, true) {
            return Err(usage("only one --secret can be read from stdin"));
        }
        let mut line = String::new();
        std::io::stdin()
            .lock()
            .read_line(&mut line)
            .map_err(|e| CliError::new(ExitCode::Runtime, "io_error", format!("reading stdin: {e}")))?;
        return Ok(line.trim_end_matches(['\r', '\n']).to_string());
    }
    Err(usage(format!(
        "--secret {key} takes env:VAR or - (stdin), not the secret itself: a command line is visible to other \
         processes and kept in shell history"
    )))
}

fn connect(args: CloudConnectArgs, out: Out) -> Result<(), CliError> {
    let cloud = open(&args.dir)?;
    let provider = cloud.providers().into_iter().find(|p| p.id == args.provider);
    let mut values = BTreeMap::new();
    for s in &args.set {
        let (key, value) = pair("--set", s)?;
        if provider.as_ref().and_then(|p| p.fields.iter().find(|f| f.key == key)).is_some_and(|f| f.secret) {
            return Err(usage(format!(
                "`{key}` is a secret: pass it with --secret {key}=env:VAR or --secret {key}=-"
            )));
        }
        values.insert(key.to_string(), value.to_string());
    }
    let mut stdin_used = false;
    for s in &args.secret {
        let (key, source) = pair("--secret", s)?;
        values.insert(key.to_string(), secret(key, source, &mut stdin_used)?);
    }
    let req = ConnectCloud { provider: args.provider, name: args.name, values };
    let c = cloud.connect(req).map_err(api_error)?;
    out.record(&c);
    connections_table(std::slice::from_ref(&c), out);
    if let Some(note) = &c.note {
        out.human(format!("note: {note}"));
    }
    Ok(())
}

/// `rows` as a table: a header, then columns padded to their widest cell.
fn table(header: &[&str], rows: Vec<Vec<String>>) -> String {
    let mut widths: Vec<usize> = header.iter().map(|h| h.chars().count()).collect();
    for row in &rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let line = |cells: Vec<String>| {
        let padded: Vec<String> = cells.iter().zip(&widths).map(|(c, w)| format!("{c:<w$}")).collect();
        padded.join("  ").trim_end().to_string()
    };
    let mut out = vec![line(header.iter().map(|h| h.to_string()).collect())];
    out.extend(rows.into_iter().map(line));
    out.join("\n")
}

fn or_dash(v: Option<String>) -> String {
    v.unwrap_or_else(|| "-".into())
}

fn usd(v: Option<f64>) -> String {
    or_dash(v.map(|v| format!("${v:.2}")))
}

/// A value's wire spelling (`ok`, `terminated`, `high`, ...).
fn word<T: serde::Serialize>(v: T) -> String {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default()
}

fn providers_table(providers: &[CloudProvider], out: Out) {
    let rows = providers
        .iter()
        .map(|p| {
            let fields: Vec<String> = p
                .fields
                .iter()
                .map(|f| match (f.secret, f.required) {
                    (true, _) => format!("{} (secret)", f.key),
                    (false, true) => f.key.clone(),
                    (false, false) => format!("[{}]", f.key),
                })
                .collect();
            vec![p.id.clone(), p.name.clone(), fields.join(", ")]
        })
        .collect();
    out.human(table(&["ID", "NAME", "FIELDS"], rows));
}

fn connections_table(list: &[CloudConnection], out: Out) {
    if list.is_empty() {
        out.human("no connections; make one with `folo cloud connect <provider>`");
        return;
    }
    let rows = list
        .iter()
        .map(|c| {
            let status = match &c.error {
                Some(e) => format!("{}: {e}", word(c.status)),
                None => word(c.status),
            };
            vec![
                c.id.clone(),
                c.provider.clone(),
                c.name.clone(),
                or_dash(c.account.clone()),
                usd(c.balance_usd),
                status,
            ]
        })
        .collect();
    out.human(table(&["ID", "PROVIDER", "NAME", "ACCOUNT", "BALANCE", "STATUS"], rows));
}

fn offers_table(offers: &[CloudOffer], out: Out) {
    let rows = offers
        .iter()
        .map(|o| {
            let vram = if o.vram_bytes > 0 { format!("{} GB", o.vram_bytes >> 30) } else { "-".into() };
            let price = or_dash(o.price_per_hour.map(|p| format!("${p:.2}/h")));
            vec![o.id.clone(), o.gpu.clone(), vram, price, word(o.availability)]
        })
        .collect();
    out.human(table(&["OFFER", "GPU", "MEMORY", "PRICE", "AVAILABILITY"], rows));
}

fn instances_table(list: &[CloudInstance], out: Out) {
    if list.is_empty() {
        out.human("no instances");
        return;
    }
    let rows = list
        .iter()
        .map(|i| {
            let status = match &i.error {
                Some(e) => format!("{} ({e})", word(i.status)),
                None => word(i.status),
            };
            vec![
                i.id.clone(),
                or_dash(i.run_id.clone()),
                i.offer.gpu.clone(),
                usd(i.cost_usd),
                i.created.clone(),
                status,
            ]
        })
        .collect();
    out.human(table(&["ID", "RUN", "GPU", "COST", "CREATED", "STATUS"], rows));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_align_and_secrets_stay_off_the_command_line() {
        let t = table(&["A", "LONGER"], vec![vec!["xyz".into(), "1".into()], vec!["q".into(), "".into()]]);
        assert_eq!(t, "A    LONGER\nxyz  1\nq");
        let mut stdin_used = false;
        let e = secret("api_key", "rpa_SECRET", &mut stdin_used).unwrap_err();
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "usage"));
        assert!(!e.message.contains("rpa_SECRET") && e.message.contains("env:VAR"), "{}", e.message);
        let e = secret("api_key", "env:FOLO_CLI_TEST_SURELY_UNSET", &mut stdin_used).unwrap_err();
        assert!(e.message.contains("is not set"), "{}", e.message);
        assert_eq!(pair("--set", "url=https://a=b").unwrap(), ("url", "https://a=b"));
        assert!(pair("--set", "=x").is_err() && pair("--set", "nothing").is_err());
        // A secret given without its key is not repeated in the error (which `--json` prints to stdout).
        for given in ["rpa_SECRET", "=rpa_SECRET"] {
            let e = pair("--secret", given).unwrap_err();
            assert!(!e.message.contains("rpa_SECRET") && e.message.contains("KEY="), "{}", e.message);
        }
    }
}
