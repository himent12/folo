//! Codex model catalog parsing, `config.toml` conflict detection, and lean catalog stripping.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use folo_api::LabelModel;
use serde_json::Value;

use super::super::agent::Output;
use super::{COSTS_MORE, FAST_TIER};

/// Top-level keys of Codex's `config.toml` that can make `codex debug models` print another catalog than the one
/// `codex exec --ignore-user-config` uses (another catalog file, a profile, another server); see
/// [`catalog_conflict`]. With one set, FOLO cannot tell which models a call has, nor which offer Fast.
const CATALOG_KEYS: [&str; 4] = ["model_catalog_json", "profile", "openai_base_url", "chatgpt_base_url"];

/// What Codex's model catalog (`codex debug models`) gives FOLO.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Catalog {
    /// The version of Codex that printed it: the snapshot's generation.
    pub(super) version: String,
    /// Whether `codex debug models` printed a catalog FOLO could read.
    pub(super) read: bool,
    /// The executable that printed it, which a call with it runs (`None`: Codex is not installed).
    pub(super) path: Option<PathBuf>,
    /// The models that take images ([`catalog_models`]).
    pub(super) models: Vec<LabelModel>,
    /// Codex's default model: the catalog's first listed model.
    pub(super) default: Option<String>,
    /// The models that offer the [`FAST_TIER`].
    pub(super) fast: Vec<String>,
    /// The catalog with its models' tools taken off ([`lean_catalog`]); `None` when there is none to use.
    pub(super) lean: Option<Arc<String>>,
    /// Why there is no lean catalog, for a run's log; `None` when there is one, or Codex is not installed.
    pub(super) no_lean: Option<String>,
    /// Why no model is offered Fast although the catalog lists it ([`catalog_conflict`]).
    pub(super) no_fast: Option<String>,
}

impl Catalog {
    /// The catalog `codex debug models` printed, `conflict` saying how the user's config may have made it another
    /// catalog than a call's ([`catalog_conflict`]): the lean catalog is then left out and no model is offered
    /// Fast, and the models are only suggestions.
    pub(super) fn read(json: &str, conflict: Option<&str>) -> Catalog {
        let Ok(catalog) = serde_json::from_str::<Value>(json) else {
            return Catalog::unread("`codex debug models` printed no catalog FOLO can read");
        };
        let entries = catalog["models"].as_array().cloned().unwrap_or_default();
        let slug = |m: &Value| m["slug"].as_str().map(str::to_string);
        let mut listed: Vec<&Value> = entries.iter().filter(|m| m["visibility"] == "list").collect();
        listed.sort_by_key(|m| m["priority"].as_u64().unwrap_or(u64::MAX));
        let fast: Vec<String> = if conflict.is_some() {
            Vec::new()
        } else {
            entries
                .iter()
                .filter(|m| {
                    m["service_tiers"].as_array().is_some_and(|t| t.iter().any(|t| t["id"] == FAST_TIER))
                })
                .filter_map(slug)
                .collect()
        };
        let no_fast = conflict.map(|how| {
            format!("Codex's config.toml {how}, so FOLO cannot tell which models offer Fast to its calls")
        });
        let (lean, no_lean) = match conflict {
            Some(how) => (
                None,
                Some(format!(
                    "Codex's config.toml {how}, so the model catalog FOLO reads may not be the one its calls use: \
                     {COSTS_MORE}"
                )),
            ),
            None => match lean_catalog(json) {
                Some(lean) => (Some(Arc::new(lean)), None),
                None => (None, Some(format!("`codex debug models` printed no model catalog: {COSTS_MORE}"))),
            },
        };
        Catalog {
            version: String::new(),
            // An object without a model list is no catalog either.
            read: catalog["models"].is_array(),
            path: None,
            models: catalog_models(&entries, &fast),
            default: listed.first().and_then(|m| slug(m)),
            fast,
            lean,
            no_lean,
            no_fast,
        }
    }

    /// No catalog, for `why`.
    pub(super) fn unread(why: &str) -> Catalog {
        Catalog { no_lean: Some(format!("{why}: {COSTS_MORE}")), ..Catalog::default() }
    }

    /// [`Agent::check_fast`](crate::label::agent::Agent::check_fast) for this catalog.
    pub(super) fn check_fast(&self, model: Option<&str>) -> Result<(), String> {
        if let Some(why) = &self.no_fast {
            return Err(why.clone());
        }
        match model.or(self.default.as_deref()) {
            Some(model) if self.fast.iter().any(|m| m == model) => Ok(()),
            Some(model) => Err(format!("Codex's model catalog lists no faster service tier for {model}")),
            None => Err(
                "Codex's model catalog could not be read, so FOLO cannot tell whether the model offers a \
                         faster service tier"
                    .into(),
            ),
        }
    }
}

/// Codex's home folder: `CODEX_HOME`, else `.codex` in the user's home.
pub(super) fn codex_home() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("CODEX_HOME").filter(|h| !h.is_empty()) {
        return Some(PathBuf::from(home));
    }
    let home =
        std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".codex"))
}

/// How Codex's `config.toml` at `path` may point Codex at another catalog or server than a call's
/// ([`catalog_conflict`]). Only a file that is not there is no config: one that cannot be read (no permission, an
/// I/O error, not UTF-8) may set anything, so it counts as a conflict.
pub(super) fn config_conflict(path: &Path) -> Option<String> {
    match std::fs::read_to_string(path) {
        Ok(config) => catalog_conflict(&config),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => Some(format!("cannot be read ({e})")),
    }
}

/// How Codex's `config.toml` (its text) may point Codex at another catalog or server than a call's: "sets `key`"
/// for the first of [`CATALOG_KEYS`] it sets at the top level (a bare or a quoted key alike), or that it cannot be
/// read as TOML; `None` when it does neither.
pub(super) fn catalog_conflict(config: &str) -> Option<String> {
    match config.parse::<toml::Table>() {
        Ok(table) => CATALOG_KEYS.iter().find(|k| table.contains_key(**k)).map(|key| format!("sets `{key}`")),
        Err(_) => Some("cannot be read as TOML".into()),
    }
}

/// Whether a call's `out` is Codex refusing the lean catalog as it started: it failed before any event (so before
/// it asked the model, and with no tokens used), naming `model_catalog_json`.
pub(super) fn refused_catalog(out: &Output) -> bool {
    let started =
        out.stdout.lines().any(|l| serde_json::from_str::<Value>(l).is_ok_and(|v| v["type"].is_string()));
    !out.success && !started && format!("{}{}", out.stderr, out.stdout).contains("model_catalog_json")
}

/// Codex's model catalog (`codex debug models`) as `model_catalog_json` for a labeling call: every model as Codex
/// describes it, but without the tools its entry brings into every request. `tool_mode` "direct" instead of
/// "code_mode_only" drops the JavaScript `exec` tool and its helpers, an empty `experimental_supported_tools` the
/// `clock`, `send_message_to_user_async` and `request_user_input_async` tools, and no `apply_patch_tool_type` the
/// patch tool. Only fields the catalog has are changed, so the file stays in the shape this version reads; `None`
/// when the text is not a catalog.
pub(super) fn lean_catalog(json: &str) -> Option<String> {
    let mut catalog: Value = serde_json::from_str(json).ok()?;
    let models = catalog.get_mut("models")?.as_array_mut()?;
    for model in models.iter_mut().filter_map(Value::as_object_mut) {
        for (key, lean) in [
            ("tool_mode", Value::from("direct")),
            ("experimental_supported_tools", Value::Array(Vec::new())),
            ("apply_patch_tool_type", Value::Null),
        ] {
            if let Some(v) = model.get_mut(key) {
                *v = lean;
            }
        }
    }
    serde_json::to_string(&catalog).ok()
}

/// The listed models of a catalog's `entries` that accept images, in the catalog's order of preference; those in
/// `fast` offer the [`FAST_TIER`].
fn catalog_models(entries: &[Value], fast: &[String]) -> Vec<LabelModel> {
    let mut models: Vec<(u64, LabelModel)> = entries
        .iter()
        .filter(|m| m["visibility"] == "list")
        .filter(|m| m["input_modalities"].as_array().is_some_and(|a| a.iter().any(|x| x == "image")))
        .filter_map(|m| {
            let id = m["slug"].as_str()?.to_string();
            let label = m["display_name"].as_str().unwrap_or(&id).to_string();
            let fast = fast.contains(&id);
            Some((m["priority"].as_u64().unwrap_or(u64::MAX), LabelModel { id, label, fast }))
        })
        .collect();
    models.sort_by_key(|(priority, _)| *priority);
    models.into_iter().map(|(_, m)| m).collect()
}
