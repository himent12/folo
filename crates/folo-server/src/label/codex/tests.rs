//! Unit tests for the Codex labeling agent, catalog filtering, and JSON event parsing.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use folo_api::{LabelEffort, LabelModel, LabelUsage};
use folo_test_support::TestDir;
use serde_json::Value;

use super::catalog::{Catalog, catalog_conflict, config_conflict, lean_catalog};
use super::parse::{parse, usage_of};
use super::{Codex, Files, command, toml_string, with_lean_fallback};
use crate::label::agent::{Agent, AgentError, AgentRequest, Output};

fn output(stdout: &str, stderr: &str, success: bool) -> Output {
    Output { success, stdout: stdout.into(), stderr: stderr.into() }
}

/// The events codex-cli 0.167.0 prints for a successful call, with a warning about an unknown config key.
const SUCCESS: &str = r#"{"type":"thread.started","thread_id":"01a1c20c"}
{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Codex is ignoring 1 unrecognized configuration setting."}}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"{\"images\":[{\"image\":1,\"boxes\":[]}]}"}}
{"type":"turn.completed","usage":{"input_tokens":8993,"cached_input_tokens":4096,"cache_write_input_tokens":0,"output_tokens":287,"reasoning_output_tokens":150}}"#;

#[test]
fn the_last_agent_message_is_the_reply_and_warnings_are_not_failures() {
    let reply = parse(&output(SUCCESS, "Reading prompt from stdin...\n", true)).unwrap();
    assert_eq!(reply.text, r#"{"images":[{"image":1,"boxes":[]}]}"#);
    assert_eq!(
        reply.usage,
        LabelUsage {
            calls: 1,
            input_tokens: 8993,
            cached_input_tokens: 4096,
            output_tokens: 287,
            cost_usd: None
        }
    );
}

#[test]
fn failed_turns_and_silent_exits_are_failures() {
    let stdout = r#"{"type":"thread.started","thread_id":"t"}
{"type":"error","message":"unexpected status 401 Unauthorized"}
{"type":"turn.failed","error":{"message":"unexpected status 401 Unauthorized: sign in again"}}"#;
    let err = parse(&output(stdout, "", false)).unwrap_err();
    assert_eq!(err, AgentError::failed("unexpected status 401 Unauthorized: sign in again"));
    assert!(Codex::new().hint("unexpected status 401 Unauthorized").unwrap().contains("codex login"));
    let err = parse(&output("", "Error: Unknown feature flag: nope\n", false)).unwrap_err();
    assert_eq!(err, AgentError::failed("Error: Unknown feature flag: nope"));
    let err = parse(&output("", "", true)).unwrap_err();
    let AgentError::Unavailable(message) = &err else { panic!("{err:?}") };
    assert!(Codex::new().hint(message).unwrap().contains("`codex exec hi`"));
    // A turn that completed without a message, on a normal exit: no reply, but its tokens were used.
    let no_message = SUCCESS.lines().filter(|l| !l.contains("agent_message")).collect::<Vec<_>>().join("\n");
    let err = parse(&output(&no_message, "", true)).unwrap_err();
    let AgentError::Failed(failure) = &err else { panic!("{err:?}") };
    assert_eq!((failure.message.as_str(), failure.used.input_tokens), ("`codex` printed no reply", 8993));
    assert!(Codex::new().hint(&failure.message).unwrap().contains("`codex exec hi`"));
    // ... and a call killed after it printed that event (it hung on its way out) used them too.
    assert_eq!(usage_of(&no_message).map(|u| u.output_tokens), Some(287));
    assert_eq!(usage_of("{\"type\":\"turn.started\"}"), None);
    // A usage limit says when it resets; a server error that outlasted Codex's own retries is not a limit.
    let limit = "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), or try again \
                 at 3:04 PM.";
    let hint = Codex::new().hint(limit).unwrap();
    assert!(hint.contains("usage limit was reached (try again at 3:04 PM): start again then"), "{hint}");
    assert_eq!(
        Codex::new().hint("exceeded retry limit, last status: 503 Service Unavailable").unwrap(),
        crate::label::agent::BUSY
    );
    // A rate limit is the provider's, in its own words; one that asks for a long wait is that wait.
    let hint = Codex::new().hint("Rate limit reached for gpt-6.2-sol. Please try again in 30s.").unwrap();
    assert!(hint.starts_with("the provider's rate limit was reached (try again in 30s)"), "{hint}");
    let hint = Codex::new().hint("Rate limit reached for gpt-6.2-sol. Please try again in 20m0s.").unwrap();
    assert!(hint.starts_with("the provider asked for a long wait (try again in 20m0s)"), "{hint}");
}

#[test]
fn the_command_attaches_the_images_and_switches_the_tools_off() {
    let dir = TestDir::new("label-codex-command");
    let (schema, cancel) = (serde_json::json!({ "type": "object" }), AtomicBool::new(false));
    let req = AgentRequest {
        system: "label things",
        prompt: "Classes: car",
        images: &[vec![1], vec![2, 2]],
        schema: &schema,
        model: Some("gpt-6-luna"),
        effort: LabelEffort::Low,
        fast: true,
        scratch: dir.path(),
        cancel: &cancel,
    };
    let files = Files::write(&req, Some(r#"{"models":[]}"#)).unwrap();
    assert_eq!(std::fs::read_to_string(&files.instructions).unwrap(), "label things");
    assert_eq!(std::fs::read(&files.images[1]).unwrap(), vec![2, 2]);
    assert_eq!(std::fs::read_to_string(files.catalog.as_ref().unwrap()).unwrap(), r#"{"models":[]}"#);
    assert_eq!(std::fs::read_dir(&files.work).unwrap().count(), 0);

    let cmd = command(Path::new("codex"), &req, &files);
    let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
    assert_eq!(args[0], "exec");
    assert_eq!(args.last().map(String::as_str), Some("-"));
    for flag in ["--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config"] {
        assert!(args.iter().any(|a| a == flag), "{flag} missing");
    }
    let values = |flag: &str| -> Vec<&str> {
        args.windows(2).filter(|w| w[0] == flag).map(|w| w[1].as_str()).collect()
    };
    assert_eq!(values("--sandbox"), ["read-only"]);
    assert_eq!(values("--model"), ["gpt-6-luna"]);
    assert_eq!(values("--image").len(), 2);
    assert!(values("--image")[0].ends_with("1.jpg"));
    let config = values("-c");
    assert!(config.contains(&"model_reasoning_effort='low'"));
    assert!(config.contains(&"features.shell_tool=false") && config.contains(&"web_search='disabled'"));
    for key in ["skills.include_instructions=false", "agents.enabled=false", "service_tier='priority'"] {
        assert!(config.contains(&key), "{key} missing from {config:?}");
    }
    let instructions = config.iter().find(|c| c.starts_with("model_instructions_file='")).unwrap();
    assert!(instructions.ends_with("instructions.md'") && !instructions.contains('\\'), "{instructions}");
    let catalog = config.iter().find(|c| c.starts_with("model_catalog_json='")).unwrap();
    assert!(catalog.ends_with("models.json'"), "{catalog}");
    // No argument needs quoting a batch shim could get wrong.
    assert!(args.iter().all(|a| !a.contains('"') && !a.contains('\n')));
    assert_eq!(toml_string("D:/data/x.md"), "'D:/data/x.md'");
    assert_eq!(toml_string(r#"C:/O'Brien/a "b".md"#), r#""C:/O'Brien/a \"b\".md""#);

    // Without a catalog (it could not be read) and without the fast tier, neither is set.
    let dir = TestDir::new("label-codex-command-plain");
    let req = AgentRequest { fast: false, scratch: dir.path(), ..req };
    let files = Files::write(&req, None).unwrap();
    let cmd = command(Path::new("codex"), &req, &files);
    let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
    assert!(!args.iter().any(|a| a.starts_with("model_catalog_json") || a.starts_with("service_tier")));
}

#[test]
fn the_lean_catalog_takes_off_the_models_tools_and_nothing_else() {
    let json = r#"{"models":[
        {"slug":"a","tool_mode":"code_mode_only","experimental_supported_tools":["clock"],
         "apply_patch_tool_type":"freeform","base_instructions":"x","priority":1},
        {"slug":"old","priority":2}],"etag":"e"}"#;
    let lean: Value = serde_json::from_str(&lean_catalog(json).unwrap()).unwrap();
    let expected = serde_json::json!({"models":[
        {"slug":"a","tool_mode":"direct","experimental_supported_tools":[],
         "apply_patch_tool_type":null,"base_instructions":"x","priority":1},
        {"slug":"old","priority":2}],"etag":"e"});
    assert_eq!(lean, expected);
    assert_eq!(lean_catalog("not json"), None);
    assert_eq!(lean_catalog(r#"{"data":[]}"#), None);
}

#[test]
fn an_error_before_a_completed_turn_was_a_retry() {
    let retried =
        "{\"type\":\"turn.started\"}\n{\"type\":\"error\",\"message\":\"stream disconnected; retrying 1/5\"}";
    let stdout = SUCCESS.replacen("{\"type\":\"turn.started\"}", retried, 1);
    assert!(parse(&output(&stdout, "", true)).is_ok());
}

const FAST: &str = r#"[{"id":"priority","name":"Fast","description":"2x speed"}]"#;

/// A catalog as `codex debug models` prints it: `b` and `hidden` offer Fast, `text` takes no images.
fn catalog_json() -> String {
    format!(
        r#"{{"models":[
        {{"slug":"b","display_name":"B","visibility":"list","input_modalities":["text","image"],"priority":2,"service_tiers":{FAST}}},
        {{"slug":"hidden","display_name":"H","visibility":"hide","input_modalities":["text","image"],"priority":0,"service_tiers":{FAST}}},
        {{"slug":"text","display_name":"T","visibility":"list","input_modalities":["text"],"priority":1,"service_tiers":[]}},
        {{"slug":"a","visibility":"list","input_modalities":["image"],"priority":1,"tool_mode":"code_mode_only"}}]}}"#
    )
}

#[test]
fn catalog_lists_visible_image_models_by_priority() {
    let catalog = Catalog::read(&catalog_json(), None);
    assert_eq!(
        catalog.models,
        [
            LabelModel { id: "a".into(), label: "a".into(), fast: false },
            LabelModel { id: "b".into(), label: "B".into(), fast: true }
        ]
    );
    assert_eq!(catalog.default.as_deref(), Some("text"), "the first listed model, images or not");
    assert!(catalog.lean.is_some() && catalog.no_lean.is_none());
    let unread = Catalog::read("not json", None);
    assert!(unread.models.is_empty() && unread.lean.is_none() && !unread.read);
    assert!(catalog.read);
    assert!(unread.no_lean.unwrap().contains("7800 more input tokens"));
    assert!(Catalog::read("{}", None).models.is_empty());
    // No model list: unread, so a Fast call goes without the tier instead of failing.
    assert!(!Catalog::read("{}", None).read && Catalog::read(r#"{"models":[]}"#, None).read);
}

#[test]
fn fast_is_offered_for_the_models_whose_entry_lists_the_priority_tier() {
    let catalog = Catalog::read(&catalog_json(), None);
    assert_eq!(catalog.check_fast(Some("b")), Ok(()));
    assert_eq!(catalog.check_fast(Some("hidden")), Ok(()), "a model the list hides can still be named");
    assert_eq!(
        catalog.check_fast(Some("text")).unwrap_err(),
        "Codex's model catalog lists no faster service tier for text"
    );
    assert!(catalog.check_fast(Some("not-in-the-catalog")).is_err());
    // The default model (`text`) does not offer it, so neither does a run that names no model.
    assert!(catalog.check_fast(None).is_err());
    let b_first = catalog_json().replace(r#""priority":2"#, r#""priority":0"#);
    assert_eq!(Catalog::read(&b_first, None).check_fast(None), Ok(()));
    assert!(Catalog::default().check_fast(None).unwrap_err().contains("could not be read"));
}

#[test]
fn a_config_that_may_point_codex_elsewhere_keeps_the_lean_catalog_and_fast_out() {
    let config = "model_auto_compact_token_limit = 400000\n# profile = \"x\"\n\n[profiles.fast]\nmodel_catalog_json = \"x\"\n";
    assert_eq!(catalog_conflict(config), None, "comments and keys in tables do not count");
    assert_eq!(
        catalog_conflict("openai_base_url = \"https://proxy\"\n[x]\n").as_deref(),
        Some("sets `openai_base_url`")
    );
    assert_eq!(catalog_conflict("  profile=\"work\"").as_deref(), Some("sets `profile`"));
    // Quoted keys are the same keys.
    assert_eq!(
        catalog_conflict("'model_catalog_json' = 'x.json'").as_deref(),
        Some("sets `model_catalog_json`")
    );
    assert_eq!(catalog_conflict("\"chatgpt_base_url\" = \"x\"").as_deref(), Some("sets `chatgpt_base_url`"));
    assert_eq!(catalog_conflict("profile = ").as_deref(), Some("cannot be read as TOML"));
    assert_eq!(catalog_conflict(""), None);
    let catalog = Catalog::read(&catalog_json(), Some("sets `profile`"));
    assert_eq!(catalog.lean, None);
    assert!(catalog.no_lean.as_deref().unwrap().starts_with("Codex's config.toml sets `profile`"));
    assert_eq!(catalog.models.len(), 2, "the models are still offered");
    // ... but no model is offered Fast, and asking for it says why.
    assert!(catalog.models.iter().all(|m| !m.fast));
    let why = catalog.check_fast(Some("b")).unwrap_err();
    assert!(why.contains("sets `profile`") && why.contains("cannot tell which models offer Fast"), "{why}");
}

#[test]
fn a_refused_lean_catalog_is_left_out_of_that_call_and_by_that_version_only() {
    // What codex-cli prints when it cannot read the file: no event, nothing asked of the model.
    let refusal = || {
        output(
            "",
            "Error: failed to parse model_catalog_json path `x` as JSON: missing field `slug`\n",
            false,
        )
    };
    let mut seen = Vec::new();
    let mut refused = false;
    let out = with_lean_fallback(
        Some("{}"),
        || refused = true,
        |catalog| {
            seen.push(catalog.is_some());
            Ok(if catalog.is_some() { refusal() } else { output(SUCCESS, "", true) })
        },
    )
    .unwrap();
    assert!(parse(&out).is_ok() && seen == [true, false] && refused);
    // A call that started (it printed events) and failed is the call's failure, whatever its message names:
    // no second call.
    let started = output(
        "{\"type\":\"thread.started\",\"thread_id\":\"t\"}\n{\"type\":\"turn.failed\",\"error\":{\"message\":\"model_catalog_json says no\"}}",
        "",
        false,
    );
    let mut calls = 0;
    let mut refused = false;
    let out = with_lean_fallback(
        Some("{}"),
        || refused = true,
        |_| {
            calls += 1;
            Ok(started.clone())
        },
    )
    .unwrap();
    assert!(parse(&out).is_err() && calls == 1 && !refused);
    // Without the lean catalog nothing is retried.
    let mut calls = 0;
    let _ = with_lean_fallback(
        None,
        || {},
        |_| {
            calls += 1;
            Ok(refusal())
        },
    );
    assert_eq!(calls, 1);

    // The refusal holds for the version that refused, across reads of its catalog, and not for another.
    let codex = Codex::new();
    let snapshot =
        |version: &str| Catalog { version: version.into(), ..Catalog::read(&catalog_json(), None) };
    codex.cli.remember("codex", "0.169.0");
    let _ = codex.catalog.get(|| snapshot("0.169.0"));
    codex.refuse(&snapshot("0.169.0"));
    assert!(codex.refused(&snapshot("0.169.0")));
    // ... and the catalog is read again (with the probe): the next call may run a Codex updated in place.
    assert_eq!(codex.catalog.get(|| snapshot("0.170.0")).version, "0.170.0");
    codex.cli.remember("codex", "0.170.0");
    // A late refusal of the older version leaves the newer snapshot in place, the probe's and the catalog's.
    codex.refuse(&snapshot("0.169.0"));
    assert_eq!(codex.cli.version().as_deref(), Some("0.170.0"), "the probe of the newer Codex stays");
    assert_eq!(codex.catalog.get(|| snapshot("0.171.0")).version, "0.170.0");
    assert!(!codex.refused(&snapshot("0.170.0")), "a newer Codex is given the lean catalog again");
    // Refusals interleave: the newer version refuses too, then a late refusal from a call that still held
    // the older snapshot comes in. Neither clears the other.
    codex.refuse(&snapshot("0.170.0"));
    codex.refuse(&snapshot("0.169.0"));
    assert!(codex.refused(&snapshot("0.170.0")) && codex.refused(&snapshot("0.169.0")));
    assert!(!codex.refused(&snapshot("0.171.0")));
}

#[test]
fn a_config_that_cannot_be_read_counts_as_one_that_may_point_codex_elsewhere() {
    let dir = TestDir::new("label-codex-config");
    let path = dir.path().join("config.toml");
    assert_eq!(config_conflict(&path), None, "no config at all");
    std::fs::write(&path, "model = \"x\"\n").unwrap();
    assert_eq!(config_conflict(&path), None);
    std::fs::write(&path, "profile = \"work\"\n").unwrap();
    assert_eq!(config_conflict(&path).as_deref(), Some("sets `profile`"));
    // Not UTF-8: not read, so it may set anything.
    std::fs::write(&path, b"profile = \"\xff\xfe\"\n").unwrap();
    assert!(config_conflict(&path).unwrap().starts_with("cannot be read ("));
    // A folder where the file should be: an error other than its absence.
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(config_conflict(&path).unwrap().starts_with("cannot be read ("));
    let catalog = Catalog::read(&catalog_json(), config_conflict(&path).as_deref());
    assert!(catalog.lean.is_none() && catalog.check_fast(Some("b")).is_err());
}
