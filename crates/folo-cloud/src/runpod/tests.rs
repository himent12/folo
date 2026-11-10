use super::*;
use folo_api::CloudAvailability;

use crate::bootstrap;
use crate::fixture::{Answer, Fixture, bash_syntax};
use crate::{InstanceState, LaunchSpec, Provider, Values, WORKER_BASE_URL_FIELD, WORKER_PORT};

const KEY: &str = "rpa_TESTKEY0123456789abcdef";
const TOKEN: &str = "worker-token-0123456789";
const NO_PODS: &str = r#"{"pods":[],"pagination":{"nextCursor":null,"hasNextPage":false}}"#;

fn values() -> Values {
    Values::from(BTreeMap::from([(API_KEY.to_string(), KEY.to_string())]))
}

fn runpod(f: &Fixture) -> RunPod {
    RunPod::with_urls(&f.url("/v2"), &f.url("/graphql"), &format!("{}/proxy/{{id}}-{{port}}", f.base()))
}

fn spec(offer_id: &str) -> LaunchSpec {
    LaunchSpec {
        name: "folo-run_7d1e-4c1a".into(),
        offer_id: offer_id.into(),
        disk_gb: 40,
        token: TOKEN.into(),
        lifetime_s: 3600,
        worker_base_url: "https://github.com/himent12/folo/releases/download/v0.1.0".into(),
    }
}

fn problem(status: u16, detail: &str) -> Answer {
    let body = serde_json::json!({ "title": "Error", "status": status, "detail": detail });
    Answer::json(status, body.to_string())
}

/// Every request carried the key as a bearer header and never in its URL.
fn assert_key_only_in_headers(f: &Fixture) {
    for r in f.requests() {
        assert!(!r.path.contains(KEY), "{}", r.path);
        assert_eq!(r.header("authorization"), Some(format!("Bearer {KEY}").as_str()), "{}", r.path);
    }
}

/// The error has `kind`, mentions `part` and shows neither the key nor the token.
fn assert_error(e: &Error, kind: ErrorKind, part: &str) {
    assert_eq!(e.kind, kind, "{e:?}");
    assert!(e.message.contains(part), "`{part}` not in: {}", e.message);
    for shown in [e.message.clone(), format!("{e:?}")] {
        assert!(!shown.contains(KEY) && !shown.contains(TOKEN), "{shown}");
    }
}

#[test]
fn info_describes_the_connection() {
    let info = RunPod::new().info();
    assert_eq!((info.id.as_str(), info.provisions, info.self_terminates), ("runpod", true, true));
    let fields: Vec<(&str, bool, bool)> =
        info.fields.iter().map(|f| (f.key.as_str(), f.secret, f.required)).collect();
    assert_eq!(fields, [("api_key", true, true), ("worker_base_url", false, false)]);
    assert!(info.fields[0].help.as_deref().unwrap().contains("write access"));
}

#[test]
fn check_reads_the_account() {
    let f = Fixture::serve(vec![
        Answer::json(200, NO_PODS),
        Answer::json(
            200,
            r#"{"data":{"myself":{"id":"u1","email":"you@example.com","clientBalance":"42.17"}}}"#,
        ),
    ]);
    let account = runpod(&f).check(&values()).unwrap();
    assert_eq!(
        account,
        Account { label: Some("you@example.com".into()), balance_usd: Some(42.17), note: None }
    );
    let r = f.requests();
    assert_eq!((r[0].method.as_str(), r[0].path.as_str()), ("GET", "/v2/pods?limit=1"));
    assert_eq!((r[1].method.as_str(), r[1].path.as_str()), ("POST", "/graphql"));
    assert!(r[1].json()["query"].as_str().unwrap().contains("myself"));
    assert_key_only_in_headers(&f);
}

#[test]
fn check_notes_a_worker_download_that_is_not_there() {
    let pods = || Answer::json(200, NO_PODS);
    let me = || Answer::json(200, r#"{"data":{"myself":{"email":"you@example.com","clientBalance":5}}}"#);
    // Pods, the account, then the two files of the download: the installer is there, the archive is not.
    let f = Fixture::serve(vec![pods(), me(), Answer::json(200, "ok"), Answer::json(404, "{}")]);
    let mut map = values().as_map().clone();
    map.insert(WORKER_BASE_URL_FIELD.into(), f.url("/mirror"));
    let probing = RunPod { probe: true, ..runpod(&f) };
    let account = probing.check(&Values::from(map.clone())).unwrap();
    assert_eq!(account.label.as_deref(), Some("you@example.com"));
    let note = account.note.expect("a note");
    assert!(note.starts_with("Instances cannot install FOLO: ") && note.ends_with("_URL."), "{note}");
    assert!(note.contains("/mirror/folo-x86_64-unknown-linux-gnu.tar.gz.sha256 answers 404"), "{note}");
    let r = f.requests();
    assert_eq!(r.len(), 4);
    // The key goes to RunPod only, never to the place the worker is downloaded from.
    assert!(r[2..].iter().all(|r| r.header("authorization").is_none() && r.path.starts_with("/mirror/")));

    // A download that is there leaves no note.
    let f = Fixture::serve(vec![pods(), me(), Answer::json(200, "ok"), Answer::json(200, "ok")]);
    map.insert(WORKER_BASE_URL_FIELD.into(), f.url("/mirror"));
    let probing = RunPod { probe: true, ..runpod(&f) };
    assert_eq!(probing.check(&Values::from(map)).unwrap().note, None);
}

#[test]
fn launch_rents_nothing_when_the_worker_cannot_be_downloaded() {
    let f = Fixture::serve(vec![Answer::json(404, "{}")]);
    let probing = RunPod { probe: true, ..runpod(&f) };
    let missing = LaunchSpec { worker_base_url: f.url("/v9.9.9"), ..spec("NVIDIA L4@secure") };
    let e = probing.launch(&values(), &missing).unwrap_err();
    assert_error(&e, ErrorKind::NoWorker, "/v9.9.9/install.sh answers 404");
    assert_eq!(e.code(), "cloud_boot_failed");
    let r = f.requests();
    assert_eq!((r.len(), r[0].method.as_str()), (1, "GET"), "no pod was asked for");

    // With the download there, the pod is created.
    let created = Answer::json(201, r#"{"id":"abc123xyz","status":"PROVISIONING"}"#);
    let f = Fixture::serve(vec![Answer::json(200, "ok"), Answer::json(200, "ok"), created]);
    let probing = RunPod { probe: true, ..runpod(&f) };
    let there = LaunchSpec { worker_base_url: f.url("/v0.1.0"), ..spec("NVIDIA L4@secure") };
    assert_eq!(probing.launch(&values(), &there).unwrap(), "abc123xyz");
    assert_eq!(f.requests()[2].path, "/v2/pods");
}

#[test]
fn check_maps_a_rejected_key_and_one_without_pod_access() {
    let f = Fixture::serve(vec![problem(401, "invalid token")]);
    assert_error(&runpod(&f).check(&values()).unwrap_err(), ErrorKind::Auth, "RunPod rejected this API key");
    assert_eq!(f.requests().len(), 1, "no GraphQL call after a rejected key");
    let f = Fixture::serve(vec![problem(403, "forbidden")]);
    assert_error(&runpod(&f).check(&values()).unwrap_err(), ErrorKind::Auth, "cannot manage pods");
}

#[test]
fn check_passes_without_graphql() {
    for graphql in [
        problem(503, "down"),
        Answer::json(200, r#"{"data":{"myself":null},"errors":[{"message":"unauthorized"}]}"#),
        Answer::json(200, "not json"),
    ] {
        let f = Fixture::serve(vec![Answer::json(200, NO_PODS), graphql]);
        assert_eq!(runpod(&f).check(&values()).unwrap(), Account::default());
    }
}

#[test]
fn check_refuses_malformed_values_before_any_request() {
    let f = Fixture::serve(Vec::new());
    let p = runpod(&f);
    assert_error(&p.check(&Values::default()).unwrap_err(), ErrorKind::Invalid, "api_key");
    let spaced = Values::from(BTreeMap::from([(API_KEY.to_string(), "rpa_a b".to_string())]));
    assert_error(&p.check(&spaced).unwrap_err(), ErrorKind::Invalid, "api_key");
    let mut map = values().as_map().clone();
    map.insert(WORKER_BASE_URL_FIELD.into(), "ftp://mirror".into());
    assert_error(&p.check(&Values::from(map)).unwrap_err(), ErrorKind::Invalid, "worker_base_url");
    assert!(f.requests().is_empty());
}

#[test]
fn offers_merge_both_clouds_cheapest_first() {
    let secure = r#"{"gpus":[
        {"id":"NVIDIA GeForce RTX 4090","name":"RTX 4090","manufacturer":"NVIDIA","memory":24,"secure":true,
         "community":true,"price":{"secure":0.69,"community":0.34},"availability":"HIGH","extra":1},
        {"id":"NVIDIA A100 80GB PCIe","name":"A100 PCIe","manufacturer":"NVIDIA","memory":"80","secure":true,
         "community":false,"price":{"secure":"1.64","community":null},"availability":"LOW"},
        {"id":"AMD Instinct MI300X OAM","name":"MI300X","manufacturer":"AMD","memory":192,"secure":true,
         "community":false,"price":{"secure":2.49,"community":0}},
        {"id":"NVIDIA H100 80GB HBM3","name":"H100 SXM","manufacturer":"NVIDIA","memory":80,"secure":true,
         "community":false,"price":{"secure":2.99,"community":0}},
        {"id":"NVIDIA L4","name":"L4","manufacturer":"NVIDIA","memory":24,"secure":false,"community":true,
         "price":{"secure":0.5,"community":0.29},"availability":"NONE"}]}"#;
    let community = r#"{"gpus":[
        {"id":"NVIDIA GeForce RTX 4090","name":"RTX 4090","manufacturer":"NVIDIA","memory":24,"secure":true,
         "community":true,"price":{"secure":0.69,"community":0.34},"availability":"MEDIUM"},
        {"id":"NVIDIA A100 80GB PCIe","name":"A100 PCIe","manufacturer":"NVIDIA","memory":80,"secure":true,
         "community":false,"price":{"secure":1.64,"community":null},"availability":"NONE"},
        {"id":"AMD Instinct MI300X OAM","name":"MI300X","manufacturer":"AMD","memory":192,"secure":false,
         "community":true,"price":{"secure":2.49,"community":1.99},"availability":"HIGH"},
        {"id":"NVIDIA L4","name":"L4","manufacturer":"NVIDIA","memory":24,"secure":false,"community":true,
         "price":{"secure":0.5,"community":0.29},"availability":"NONE"}]}"#;
    let f = Fixture::serve(vec![Answer::json(200, secure), Answer::json(200, community)]);
    let offers = runpod(&f).offers(&values()).unwrap();
    let shown: Vec<(&str, &str, f64, CloudAvailability)> = offers
        .iter()
        .map(|o| (o.id.as_str(), o.detail.as_deref().unwrap(), o.price_per_hour.unwrap(), o.availability))
        .collect();
    assert_eq!(
        shown,
        [
            ("NVIDIA L4@community", "Community Cloud", 0.29, CloudAvailability::None),
            ("NVIDIA GeForce RTX 4090@community", "Community Cloud", 0.34, CloudAvailability::Medium),
            ("NVIDIA GeForce RTX 4090@secure", "Secure Cloud", 0.69, CloudAvailability::High),
            ("NVIDIA A100 80GB PCIe@secure", "Secure Cloud", 1.64, CloudAvailability::Low),
            ("NVIDIA H100 80GB HBM3@secure", "Secure Cloud", 2.99, CloudAvailability::Unknown),
        ]
    );
    let rtx = &offers[2];
    assert_eq!((rtx.gpu.as_str(), rtx.gpu_count, rtx.vram_bytes), ("RTX 4090", 1, 24 << 30));
    assert_eq!(offers[3].vram_bytes, 80 << 30);
    let paths: Vec<String> = f.requests().into_iter().map(|r| r.path).collect();
    assert_eq!(
        paths,
        [
            "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1",
            "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=COMMUNITY&count=1",
        ]
    );
    assert_key_only_in_headers(&f);
}

#[test]
fn offers_fail_as_their_requests_do() {
    let f = Fixture::serve(vec![problem(401, "nope")]);
    assert_error(&runpod(&f).offers(&values()).unwrap_err(), ErrorKind::Auth, "rejected");
    let f = Fixture::serve(vec![Answer::json(200, r#"{"gpus":[]}"#), problem(500, "boom")]);
    assert_error(
        &runpod(&f).offers(&values()).unwrap_err(),
        ErrorKind::Unavailable,
        "RunPod answered 500: boom",
    );
}

#[test]
fn launch_sends_exactly_the_documented_fields() {
    let created = r#"{"id":"abc123xyz","name":"folo-run_7d1e-4c1a","status":"PROVISIONING","cost":0.34}"#;
    let f = Fixture::serve(vec![Answer::json(201, created)]);
    let spec = spec("NVIDIA GeForce RTX 4090@community");
    assert_eq!(runpod(&f).launch(&values(), &spec).unwrap(), "abc123xyz");
    let r = &f.requests()[0];
    assert_eq!((r.method.as_str(), r.path.as_str()), ("POST", "/v2/pods"));
    let body = r.json();
    let keys: Vec<&str> = body.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["name", "image", "gpu", "cloud", "disk", "ports", "env", "entrypoint", "cmd"]);
    let image = std::env::var(IMAGE_ENV).ok().filter(|i| !i.trim().is_empty());
    assert_eq!(body["image"], image.as_deref().unwrap_or(IMAGE).trim());
    assert_eq!(body["name"], "folo-run_7d1e-4c1a");
    assert_eq!(body["gpu"], serde_json::json!({ "id": "NVIDIA GeForce RTX 4090", "count": 1 }));
    assert_eq!((&body["cloud"], &body["disk"]), (&"COMMUNITY".into(), &40.into()));
    assert_eq!(body["ports"], serde_json::json!(["7878/http"]));
    assert_eq!(body["env"], serde_json::json!({ "FOLO_TOKEN": TOKEN, "NVIDIA_DRIVER_CAPABILITIES": "all" }));
    assert_eq!(body["entrypoint"], serde_json::json!(["/bin/bash", "-c"]));
    let script = bootstrap::script(&bootstrap::Spec {
        base_url: &spec.worker_base_url,
        port: WORKER_PORT,
        lifetime_s: 3600,
        terminate: SELF_TERMINATE,
    })
    .unwrap();
    assert_eq!(body["cmd"], serde_json::json!([script]));
    // The token is in `env` and nowhere else.
    let mut rest = body.clone();
    rest.as_object_mut().unwrap().remove("env");
    assert!(!rest.to_string().contains(TOKEN));
    assert_key_only_in_headers(&f);
}

#[test]
fn the_pod_script_parses() {
    let s = bootstrap::script(&bootstrap::Spec {
        base_url: "https://example.com/folo",
        port: WORKER_PORT,
        lifetime_s: 3600,
        terminate: SELF_TERMINATE,
    })
    .unwrap();
    assert!(s.contains("$RUNPOD_POD_ID") && s.contains("${RUNPOD_API_KEY:-}"));
    match bash_syntax(&s) {
        None => eprintln!("skipped: no bash on PATH"),
        Some(r) => assert_eq!(r, Ok(()), "{s}"),
    }
}

#[test]
fn launch_maps_each_refusal() {
    let no_capacity = "There are no longer any instances available with the requested specifications.";
    let invalid = r#"{"title":"Unprocessable Entity","status":422,"detail":"validation failed",
                      "errors":["$: additional properties 'bogus' not allowed"]}"#;
    let busy = problem(429, "slow down").header("Retry-After", "0");
    let cases = [
        (vec![problem(400, no_capacity)], ErrorKind::Refused, "no longer any instances available"),
        (vec![problem(401, "invalid")], ErrorKind::Auth, "RunPod rejected this API key"),
        (vec![problem(402, "Insufficient balance.")], ErrorKind::Refused, "balance is too low"),
        (vec![problem(403, "no access to this pool")], ErrorKind::Refused, "no access to this pool"),
        (vec![problem(404, "resource not found")], ErrorKind::NoOffer, "`NVIDIA L4`"),
        (
            vec![Answer::json(422, invalid)],
            ErrorKind::Unavailable,
            "validation failed ($: additional properties 'bogus' not allowed)",
        ),
        (vec![problem(500, "internal")], ErrorKind::Unavailable, "RunPod answered 500: internal"),
        (vec![Answer::empty(502)], ErrorKind::Unavailable, "RunPod answered 502"),
        (
            vec![Answer::json(503, r#"{"title":"Service Unavailable","detail":null,"errors":null}"#)],
            ErrorKind::Unavailable,
            "RunPod answered 503: Service Unavailable",
        ),
        (vec![busy.clone(), busy.clone(), busy], ErrorKind::Unavailable, "rate limiting"),
    ];
    for (answers, kind, part) in cases {
        let f = Fixture::serve(answers);
        let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, kind, part);
        assert_key_only_in_headers(&f);
    }
}

#[test]
fn launch_refuses_what_it_cannot_send() {
    let f = Fixture::serve(Vec::new());
    let p = runpod(&f);
    for offer in ["NVIDIA L4", "NVIDIA L4@spot", "@secure", ""] {
        assert_error(&p.launch(&values(), &spec(offer)).unwrap_err(), ErrorKind::NoOffer, "has no offer");
    }
    let no_token = LaunchSpec { token: String::new(), ..spec("NVIDIA L4@secure") };
    assert_error(&p.launch(&values(), &no_token).unwrap_err(), ErrorKind::Invalid, "token");
    let no_disk = LaunchSpec { disk_gb: 0, ..spec("NVIDIA L4@secure") };
    assert_error(&p.launch(&values(), &no_disk).unwrap_err(), ErrorKind::Invalid, "disk");
    let bad_url = LaunchSpec { worker_base_url: "mirror.example".into(), ..spec("NVIDIA L4@secure") };
    assert_error(&p.launch(&values(), &bad_url).unwrap_err(), ErrorKind::Invalid, "worker_base_url");
    assert!(f.requests().is_empty());
}

#[test]
fn a_created_pod_without_an_id_is_unavailable() {
    let f = Fixture::serve(vec![Answer::json(201, r#"{"status":"PROVISIONING"}"#)]);
    let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
    assert_error(&e, ErrorKind::Unavailable, "did not answer with its id");
}

#[test]
fn state_maps_every_status() {
    let pod = |status: &str| {
        Answer::json(200, format!(r#"{{"id":"abc123xyz","status":"{status}","runtime":null}}"#))
    };
    let f = Fixture::serve(vec![
        pod("PROVISIONING"),
        pod("STARTING"),
        pod("RUNNING"),
        pod("EXITED"),
        pod("ERROR"),
        pod("TERMINATED"),
        problem(404, "resource not found"),
    ]);
    let p = runpod(&f);
    let running = InstanceState::Running { url: format!("{}/proxy/abc123xyz-7878", f.base()) };
    for want in [
        InstanceState::Starting,
        InstanceState::Starting,
        running,
        InstanceState::Stopped,
        InstanceState::Stopped,
        InstanceState::Gone,
        InstanceState::Gone,
    ] {
        assert_eq!(p.state(&values(), "abc123xyz").unwrap(), want);
    }
    assert!(f.requests().iter().all(|r| r.method == "GET" && r.path == "/v2/pods/abc123xyz"));
    assert_key_only_in_headers(&f);
}

#[test]
fn the_real_proxy_url_names_the_pod_and_port() {
    let f = Fixture::serve(vec![Answer::json(200, r#"{"id":"abc123xyz","status":"RUNNING"}"#)]);
    let p = RunPod::with_urls(&f.url("/v2"), &f.url("/graphql"), PROXY_URL);
    let url = "https://abc123xyz-7878.proxy.runpod.net".to_string();
    assert_eq!(p.state(&values(), "abc123xyz").unwrap(), InstanceState::Running { url });
}

#[test]
fn state_does_not_guess_an_unknown_status() {
    let f = Fixture::serve(vec![Answer::json(200, r#"{"id":"abc123xyz","status":"PAUSED"}"#)]);
    assert_error(&runpod(&f).state(&values(), "abc123xyz").unwrap_err(), ErrorKind::Unavailable, "`PAUSED`");
    assert_error(&runpod(&f).state(&values(), "../pods").unwrap_err(), ErrorKind::Invalid, "pod id");
    assert_eq!(f.requests().len(), 1);
}

#[test]
fn find_follows_pages_and_skips_terminated_namesakes() {
    let f = Fixture::serve(vec![
        Answer::json(
            200,
            r#"{"pods":[{"id":"a1","name":"folo-x","status":"RUNNING"},
            {"id":"b2","name":"folo-y","status":"RUNNING"}],"pagination":{"nextCursor":"c 2","hasNextPage":true}}"#,
        ),
        Answer::json(
            200,
            r#"{"pods":[{"id":"c3","name":"folo-x","status":"TERMINATED"},
            {"id":"d4","name":"folo-x","status":"EXITED"}],"pagination":{"nextCursor":null,"hasNextPage":false}}"#,
        ),
    ]);
    assert_eq!(runpod(&f).find(&values(), "folo-x").unwrap(), ["a1", "d4"]);
    let paths: Vec<String> = f.requests().into_iter().map(|r| r.path).collect();
    assert_eq!(paths, ["/v2/pods?limit=100", "/v2/pods?limit=100&cursor=c%202"]);
    assert_key_only_in_headers(&f);
}

#[test]
fn terminate_is_ok_once_the_pod_is_gone() {
    let f =
        Fixture::serve(vec![Answer::empty(204), problem(404, "resource not found"), problem(500, "oops")]);
    let p = runpod(&f);
    p.terminate(&values(), "abc123xyz").unwrap();
    p.terminate(&values(), "abc123xyz").unwrap();
    assert_error(&p.terminate(&values(), "abc123xyz").unwrap_err(), ErrorKind::Unavailable, "500");
    assert!(f.requests().iter().all(|r| r.method == "DELETE" && r.path == "/v2/pods/abc123xyz"));
    assert_key_only_in_headers(&f);
}

#[test]
fn a_key_echoed_by_runpod_is_not_repeated() {
    let f = Fixture::serve(vec![problem(400, &format!("bad key {KEY}"))]);
    let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
    assert_error(&e, ErrorKind::Refused, "bad key [redacted]");
}

#[test]
fn a_worker_token_echoed_by_runpod_is_not_repeated() {
    // A validation error names the value it refused, and `env` carries the worker's token.
    let invalid = format!(
        r#"{{"title":"Unprocessable Entity","status":422,"detail":"validation failed","errors":[
            {{"message":"unexpected property","location":"body.env",
              "value":{{"FOLO_TOKEN":"{TOKEN}","NVIDIA_DRIVER_CAPABILITIES":"all"}}}}]}}"#
    );
    let f = Fixture::serve(vec![Answer::json(422, invalid)]);
    let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
    assert_error(&e, ErrorKind::Unavailable, r#""FOLO_TOKEN":"[redacted]""#);
    let f = Fixture::serve(vec![problem(400, &format!("bad env FOLO_TOKEN={TOKEN}"))]);
    let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
    assert_error(&e, ErrorKind::Refused, "bad env FOLO_TOKEN=[redacted]");
}

/// `s` with every character as a JSON `\uXXXX` escape: the same string to a JSON reader, other bytes.
fn escaped(s: &str) -> String {
    s.chars().map(|c| format!("\\u{:04x}", c as u32)).collect()
}

#[test]
fn identifiers_are_used_as_the_provider_sent_them() {
    let with_key = |key: &str| Values::from(BTreeMap::from([(API_KEY.to_string(), key.to_string())]));
    // Short secrets (below the length that is scrubbed at all), and the real ones, inside the identifiers.
    for (key, token, pod, gpu) in [
        ("abc".to_string(), "abc".to_string(), "abc123xyz".to_string(), "NVIDIA abc".to_string()),
        (KEY.to_string(), TOKEN.to_string(), format!("p{KEY}"), format!("NVIDIA {KEY}")),
    ] {
        let values = with_key(&key);
        // A launch answered with a pod id that holds the worker's token succeeds, with that id.
        let f =
            Fixture::serve(vec![Answer::json(201, format!(r#"{{"id":"{pod}","status":"PROVISIONING"}}"#))]);
        let launching = LaunchSpec { token: token.clone(), ..spec("NVIDIA L4@secure") };
        assert_eq!(runpod(&f).launch(&values, &launching).unwrap(), pod);
        let f = Fixture::serve(vec![Answer::json(
            201,
            format!(r#"{{"id":"p{token}","status":"PROVISIONING"}}"#),
        )]);
        assert_eq!(runpod(&f).launch(&values, &launching).unwrap(), format!("p{token}"));
        // `find` answers the id as it is, and terminating it sends DELETE for that id.
        let page =
            format!(r#"{{"pods":[{{"id":"{pod}","name":"folo-x","status":"RUNNING"}}],"pagination":{{}}}}"#);
        let f = Fixture::serve(vec![Answer::json(200, page), Answer::empty(204)]);
        let p = runpod(&f);
        let ids = p.find(&values, "folo-x").unwrap();
        assert_eq!(ids, std::slice::from_ref(&pod));
        p.terminate(&values, &ids[0]).unwrap();
        let r = &f.requests()[1];
        assert_eq!((r.method.as_str(), r.path.clone()), ("DELETE", format!("/v2/pods/{pod}")));
        // An offer's id goes back to the provider in a launch as the catalog named it.
        let gpus = format!(
            r#"{{"gpus":[{{"id":"{gpu}","name":"G","manufacturer":"NVIDIA","memory":24,"secure":true,
                "community":false,"price":{{"secure":0.5}},"availability":"HIGH"}}]}}"#
        );
        let created = Answer::json(201, r#"{"id":"abc999","status":"PROVISIONING"}"#);
        let f = Fixture::serve(vec![Answer::json(200, gpus), Answer::json(200, r#"{"gpus":[]}"#), created]);
        let p = runpod(&f);
        let offer = p.offers(&values).unwrap().remove(0);
        assert_eq!(offer.id, format!("{gpu}@secure"));
        p.launch(&values, &LaunchSpec { token: token.clone(), ..spec(&offer.id) }).unwrap();
        assert_eq!(f.requests()[2].json()["gpu"]["id"], gpu.as_str());
    }
}

#[test]
fn fields_of_successful_answers_never_carry_the_key_into_a_message_or_an_offer() {
    let key = escaped(KEY);
    let shows = |text: &str| text.contains(KEY);
    // A pod in a status FOLO does not know, spelled with the key.
    let f = Fixture::serve(vec![Answer::json(200, format!(r#"{{"id":"abc123xyz","status":"{key}"}}"#))]);
    assert_error(&runpod(&f).state(&values(), "abc123xyz").unwrap_err(), ErrorKind::Unavailable, "a status");
    // A GPU type named with it: its name is shown (and logged, "launching <gpu>"); its id is the provider's
    // reference and is kept as it came (identifiers_are_used_as_the_provider_sent_them).
    let gpus = format!(
        r#"{{"gpus":[{{"id":"NVIDIA {key}","name":"{key}","manufacturer":"NVIDIA","memory":24,"secure":true,
            "community":false,"price":{{"secure":0.5}},"availability":"HIGH"}}]}}"#
    );
    let f = Fixture::serve(vec![Answer::json(200, gpus), Answer::json(200, r#"{"gpus":[]}"#)]);
    let offers = runpod(&f).offers(&values()).unwrap();
    assert!(offers.len() == 1 && !shows(&offers[0].gpu), "{offers:?}");
    // The account's email.
    let me = format!(r#"{{"data":{{"myself":{{"email":"{key}","clientBalance":1}}}}}}"#);
    let f = Fixture::serve(vec![Answer::json(200, NO_PODS), Answer::json(200, me)]);
    let label = runpod(&f).check(&values()).unwrap().label.unwrap();
    assert!(!shows(&label), "{label}");
}

#[test]
fn secrets_echoed_in_any_spelling_are_not_repeated() {
    for (key, token) in [(KEY.to_string(), TOKEN.to_string()), (escaped(KEY), escaped(TOKEN))] {
        let refused =
            format!(r#"{{"title":"Bad Request","status":400,"detail":"bad key {key} and token {token}"}}"#);
        let invalid = format!(
            r#"{{"title":"Unprocessable Entity","status":422,"detail":"validation failed for {key}","errors":[
                {{"message":"unexpected {token}","location":"body.env","value":{{"FOLO_TOKEN":"{token}"}}}}]}}"#
        );
        let f = Fixture::serve(vec![Answer::json(400, refused), Answer::json(422, invalid)]);
        let p = runpod(&f);
        let e = p.launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, ErrorKind::Refused, "bad key [redacted] and token [redacted]");
        let e = p.launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, ErrorKind::Unavailable, "validation failed for [redacted]");
        assert!(e.message.contains(r#""FOLO_TOKEN":"[redacted]""#), "{}", e.message);
    }
    // A long entry is shortened after the scrub: a token across the cut leaves no part of itself behind.
    let long = format!(r#"{{"status":422,"detail":"x","errors":["{}{TOKEN}"]}}"#, "a".repeat(290));
    let f = Fixture::serve(vec![Answer::json(422, long)]);
    let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
    assert!(!e.message.contains(&TOKEN[..8]), "{}", e.message);
    // An answer that is not an error is read as it came, also where the token's text happens to be.
    let created = format!(r#"{{"id":"abc123xyz","name":"{TOKEN}","status":"PROVISIONING"}}"#);
    let f = Fixture::serve(vec![Answer::json(201, created)]);
    assert_eq!(runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap(), "abc123xyz");
}
