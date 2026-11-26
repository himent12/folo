use super::*;

/// A server answering each connection with the next canned response (and recording requests).
fn canned_server(responses: Vec<Vec<u8>>) -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    canned_server_on(listener, std::time::Duration::ZERO, responses)
}

/// [`canned_server`] on `listener`, taking its first connection only `after` that long.
fn canned_server_on(
    listener: std::net::TcpListener,
    after: std::time::Duration,
    responses: Vec<Vec<u8>>,
) -> (String, std::sync::mpsc::Receiver<String>) {
    use std::io::{Read, Write};
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        std::thread::sleep(after);
        let mut responses = responses.into_iter();
        let mut resp = responses.next();
        while let Some(answer) = &resp {
            let Ok((mut sock, _)) = listener.accept() else { return };
            // The whole request, body included (answering before a POST's body is read would
            // reset the connection under the client).
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = sock.read(&mut buf).unwrap_or(0);
                req.extend_from_slice(&buf[..n]);
                let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n") else {
                    if n == 0 { break } else { continue }
                };
                let head = String::from_utf8_lossy(&req[..end]).to_ascii_lowercase();
                let body = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if n == 0 || req.len() >= end + 4 + body {
                    break;
                }
            }
            // A connection closed before sending anything is no request (the probe's check that a
            // port on this machine is not refused, `refused.rs`): it gets no answer of the list.
            if req.is_empty() {
                continue;
            }
            let _ = tx.send(String::from_utf8_lossy(&req).into_owned());
            let _ = sock.write_all(answer);
            let _ = sock.flush();
            // Dropping the socket closes the connection (cutting a short body).
            resp = responses.next();
        }
    });
    (format!("http://{addr}"), rx)
}

#[test]
fn weights_revalidate_by_etag_and_a_cut_download_is_unreachable() {
    let (url, requests) = canned_server(vec![
        b"HTTP/1.1 200 OK
etag: \"w1\"
content-length: 4
connection: close

FOLO"
            .to_vec(),
        b"HTTP/1.1 304 Not Modified
etag: \"w1\"
connection: close

"
        .to_vec(),
        // Announces 100 bytes, sends 4, hangs up.
        b"HTTP/1.1 200 OK
etag: \"w2\"
content-length: 100
connection: close

FOLO"
            .to_vec(),
    ]);
    let api = HttpBackend::new(&url);
    let w = api.model_weights("m_1", None).unwrap().expect("weights");
    assert_eq!((w.bytes.as_slice(), w.etag.as_str()), (&b"FOLO"[..], "\"w1\""));
    let first = requests.recv().unwrap();
    assert!(first.starts_with("GET /api/models/m_1/weights?format=f16 "), "{first}");
    assert!(!first.to_ascii_lowercase().contains("if-none-match"));
    assert!(api.model_weights("m_1", Some("\"w1\"")).unwrap().is_none(), "304 is unchanged");
    assert!(requests.recv().unwrap().to_ascii_lowercase().contains("if-none-match: \"w1\""));
    let cut = api.model_weights("m_1", Some("\"w1\"")).err().expect("a cut body fails");
    assert_eq!(cut.code, "server_unreachable");
}

/// A canned JSON response with its status line.
fn json(status: &str, body: &str) -> Vec<u8> {
    format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
}

#[test]
fn cloud_calls_hit_their_paths_and_decode() {
    let offer = r#"{ "id": "NVIDIA GeForce RTX 4090@secure", "gpu": "RTX 4090", "vram_bytes": 25769803776,
            "price_per_hour": 0.69, "region": null, "detail": "Secure Cloud", "availability": "high" }"#;
    let conn = r#"{ "id": "cc_1", "provider": "runpod", "name": "RunPod", "created": "2026-11-08T06:00:00Z",
            "status": "ok", "account": "you@example.com", "balance_usd": 42.17, "error": null, "note": null,
            "checked": "2026-11-08T07:00:00Z", "values": { "api_key": "••••a1b2" } }"#;
    let instance = format!(
        r#"{{ "id": "ci_1", "connection_id": "cc_1", "provider": "runpod", "provider_ref": "abc123",
            "name": "folo-run_1-4c1a", "run_id": "run_1", "offer": {offer}, "status": "terminating",
            "created": "2026-11-08T06:00:00Z", "ended": null, "deadline": "2026-11-08T18:00:00Z",
            "cost_usd": 0.42, "error": null }}"#
    );
    let (url, requests) = canned_server(vec![
        json(
            "200 OK",
            r#"{ "providers": [{ "id": "runpod", "name": "RunPod", "about": "GPU pods rented by the hour.",
                    "docs_url": "https://www.runpod.io/console/user/settings",
                    "fields": [{ "key": "api_key", "label": "API key", "secret": true, "required": true }],
                    "provisions": true, "self_terminates": true }] }"#,
        ),
        json("200 OK", &format!(r#"{{ "connections": [{conn}] }}"#)),
        json("201 Created", conn),
        json("200 OK", conn),
        b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_vec(),
        json("200 OK", &format!(r#"{{ "offers": [{offer}], "fetched": "2026-11-08T07:00:00Z" }}"#)),
        json("200 OK", &format!(r#"{{ "instances": [{instance}] }}"#)),
        json("200 OK", &instance),
        json(
            "409 Conflict",
            r#"{ "error": { "code": "connection_in_use", "message": "this connection has a live instance", "hint": "terminate it first" } }"#,
        ),
    ]);
    let api = HttpBackend::new(&url);
    let line = || requests.recv().unwrap().lines().next().unwrap_or_default().to_string();

    let providers = api.cloud_providers().unwrap();
    assert!(providers[0].fields[0].secret && providers[0].provisions);
    assert_eq!(line(), "GET /api/cloud/providers HTTP/1.1");
    assert_eq!(api.cloud_connections().unwrap()[0].balance_usd, Some(42.17));
    assert_eq!(line(), "GET /api/cloud/connections HTTP/1.1");
    let req = ConnectCloud {
        provider: "runpod".into(),
        name: String::new(),
        values: [("api_key".to_string(), "rpa_x".to_string())].into(),
    };
    assert_eq!(api.connect_cloud(&req).unwrap().id, "cc_1");
    assert_eq!(line(), "POST /api/cloud/connections HTTP/1.1");
    assert_eq!(api.check_cloud("cc_1").unwrap().status, CloudConnectionStatus::Ok);
    assert_eq!(line(), "POST /api/cloud/connections/cc_1/check HTTP/1.1");
    api.disconnect_cloud("cc_1").unwrap();
    assert_eq!(line(), "DELETE /api/cloud/connections/cc_1 HTTP/1.1");
    let offers = api.cloud_offers("cc_1", true).unwrap();
    assert_eq!((offers.offers[0].gpu_count, offers.offers[0].availability), (1, CloudAvailability::High));
    assert_eq!(line(), "GET /api/cloud/connections/cc_1/offers?refresh=true HTTP/1.1");
    assert_eq!(api.cloud_instances().unwrap()[0].status, CloudInstanceStatus::Terminating);
    assert_eq!(line(), "GET /api/cloud/instances HTTP/1.1");
    assert_eq!(api.terminate_instance("ci_1").unwrap().cost_usd, Some(0.42));
    assert_eq!(line(), "DELETE /api/cloud/instances/ci_1 HTTP/1.1");
    // The server's message and hint, as sentences (as Studio shows them).
    let busy = api.disconnect_cloud("cc_1").unwrap_err();
    assert_eq!(
        (busy.code.as_str(), busy.message.as_str(), busy.hint.as_deref()),
        ("connection_in_use", "This connection has a live instance.", Some("Terminate it first."))
    );
}

#[test]
fn a_pasted_or_bare_server_address_still_connects() {
    // As Settings › Connection hands it over: pasted with the line's end, or typed without
    // the scheme.
    let (url, requests) =
        canned_server(vec![json("200 OK", r#"{"providers":[]}"#), json("200 OK", r#"{"providers":[]}"#)]);
    let pasted = format!(" {url}/\r\n");
    assert!(HttpBackend::new(&pasted).cloud_providers().is_ok(), "{pasted:?}");
    assert!(requests.recv().unwrap().starts_with("GET /api/cloud/providers "));
    let bare = url.trim_start_matches("http://").to_string();
    assert!(HttpBackend::new(&bare).cloud_providers().is_ok(), "{bare:?}");
    assert_eq!(HttpBackend::new(&bare).describe(), bare);
}

#[test]
fn unreachable_server_is_a_clean_error() {
    // A server that hangs up without answering (a refused connection takes Windows about
    // 2 s of retries to report, and goes the same way).
    let (url, _requests) = canned_server(vec![Vec::new()]);
    let api = HttpBackend::new(&url);
    let e = api.health().unwrap_err();
    assert_eq!(e.code, "server_unreachable");
    assert!(!e.is_bad_address(), "a server that is down is not a bad address");
    // For a person: no transport jargon ("timeout: connect", "io: …").
    let hint = e.hint.unwrap();
    assert!(hint.starts_with("Start it with `folo serve`") && !hint.contains('('), "{hint}");
}

/// A body cut off midway (the server killed, a proxy dropping the link after the headers) is the
/// server out of reach, so the reconnect sweep asks for it again; it read as "Unexpected
/// response" and a page such as a finished benchmark's result kept its error with the server
/// back. A body that arrived whole but is no JSON, or JSON cut short by the server, stays a bad
/// response. (folo serve sends JSON with a content-length; a chunked body cut by a hang-up ends
/// cleanly in ureq and reads as JSON cut short, a bad response.)
#[test]
fn a_body_cut_off_midway_is_unreachable_and_a_bad_body_is_not() {
    let (url, _requests) = canned_server(vec![
            // Announces 100 bytes, sends 7, hangs up.
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: 100\r\n\r\n[{\"id\":"
                .to_vec(),
            json("200 OK", "not json"),
            json("200 OK", "[{\"id\":"),
        ]);
    let api = HttpBackend::new(&url);
    let errors: Vec<ApiError> = (0..3).map(|_| api.datasets().unwrap_err()).collect();
    let codes: Vec<&str> = errors.iter().map(|e| e.code.as_str()).collect();
    assert_eq!(codes, ["server_unreachable", "bad_response", "bad_response"]);
    // Only the health check says "not a FOLO server" (#349): a call other than health keeps its words.
    assert!(
        errors[1].message.starts_with("Unexpected response from folo serve: ") && errors[1].hint.is_none()
    );
}

/// #349: an address with a path after the port ("http://127.0.0.1:7878/subpath") reached folo
/// serve's Studio page, and Settings said "Unexpected response from folo serve: json: expected
/// value at line 1 column 1". An answer to the health check is folo serve's exactly when its body
/// has FOLO's shape, whatever its status: one that hasn't (a web page, another JSON, an unhealthy
/// health) says so with its status and the address to check; one that has reads as on any call.
#[test]
fn an_address_that_answers_but_not_as_folo_serve_says_so() {
    let page = |status: &str| {
        let body = "<!doctype html><title>FOLO Studio</title>";
        format!(
                "HTTP/1.1 {status}\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_bytes()
    };
    let health = |status: &str| format!(r#"{{ "status": "{status}", "version": "0.1.0", "mock": false }}"#);
    let forbidden = r#"{ "error": { "code": "forbidden_host", "message": "this server listens on loopback",
            "hint": "reach it as localhost or 127.0.0.1" } }"#;
    // Headers, then the link cut inside the body: announces 100 bytes, sends 8.
    let cut = |status: &str| {
        format!("HTTP/1.1 {status}\r\ncontent-length: 100\r\nconnection: close\r\n\r\n{{\"error\"")
            .into_bytes()
    };
    let (url, requests) = canned_server(vec![
        page("200 OK"),
        json("200 OK", r#"{ "ok": true }"#),
        page("404 Not Found"),
        json("200 OK", &health("ok")),
        json("200 OK", &health("down")),
        page("403 Forbidden"),
        page("500 Internal Server Error"),
        json("403 Forbidden", forbidden),
        json("401 Unauthorized", r#"{ "error": { "code": "unauthorized", "message": "no token" } }"#),
        json("500 Internal Server Error", r#"{ "error": { "code": "internal", "message": "boom" } }"#),
        cut("403 Forbidden"),
        cut("401 Unauthorized"),
        cut("200 OK"),
    ]);
    let typed = format!("{url}/subpath/");
    let api = HttpBackend::new(&typed);
    for status in [200, 200, 404] {
        let e = api.health().unwrap_err();
        assert_eq!(e.code, "not_folo");
        assert_eq!(e.message, format!("{} answered HTTP {status}, but not as a FOLO server", api.describe()));
        // The address to try instead: the one without the path.
        assert!(e.hint.unwrap().ends_with(&format!("Try {url}.")));
        assert!(requests.recv().unwrap().starts_with("GET /subpath/api/health "));
    }
    let api = HttpBackend::new(&url);
    assert_eq!(api.health().unwrap().version, "0.1.0");
    // An unhealthy health, a sign-in page's 403 and a proxy's 500 page: not FOLO's answers.
    for status in [200, 403, 500] {
        let e = api.health().unwrap_err();
        assert_eq!(e.message, format!("{} answered HTTP {status}, but not as a FOLO server", api.describe()));
        // No path to drop: what a base address looks like.
        assert!(e.hint.unwrap().ends_with("base address, like http://127.0.0.1:7878."));
    }
    // FOLO's own refusals keep their words: the server's explanation and fix, the token message.
    let e = api.health().unwrap_err();
    assert_eq!((e.code.as_str(), e.message.as_str()), ("forbidden_host", "This server listens on loopback."));
    assert_eq!(e.hint.as_deref(), Some("Reach it as localhost or 127.0.0.1."));
    assert_eq!(api.health().unwrap_err(), needs_token());
    assert_eq!(api.health().unwrap_err().code, "internal");
    // A body cut off midway has no shape to judge: what any call says (main's words).
    let e = api.health().unwrap_err();
    assert_eq!((e.code.as_str(), e.message.as_str()), ("http_error", "folo serve answered HTTP 403"));
    assert_eq!(api.health().unwrap_err(), needs_token());
    assert_eq!(api.health().unwrap_err().code, "server_unreachable");
    let addr = url.trim_start_matches("http://");
    let shouted = HttpBackend::new(&format!("HTTP://{addr}/x"));
    assert!(shouted.not_folo(200).hint.unwrap().ends_with(&format!("Try HTTP://{addr}.")));
    // The log names scheme, host and port, the status and a fixed reason: a pasted URL's user,
    // path or query can hold a token, and an answer's body can quote them.
    assert_eq!(
        HttpBackend::new("http://me:pw@127.0.0.1:7865/x?token=s").not_folo_log(403),
        "folo-desktop: http://127.0.0.1:7865 answered HTTP 403, not as a FOLO server"
    );
}

#[test]
fn an_address_that_is_not_one_says_so_instead_of_blaming_the_server() {
    // Typed in Settings › Connection: nothing can be sent to these, so no server is "down".
    // "http://" and "foo://some:42/123" passed for one (review round 1): trimmed to "http:" and
    // given a scheme, the first became the host "http"; ureq refuses the second's scheme unsent.
    let not = ["abc def", "http://[::", "http:// 127.0.0.1:1", "http://a b:7878", "http://", "http://:7878"];
    // Review round 2: a port ureq can't use sent the request to port 80.
    let ports = ["http://127.0.0.1:99999", "127.0.0.1:70000", "http://h:abc", "localhost:", "http://h:0"];
    let odd = ["foo://some:42/123", "ftp://127.0.0.1:21", " ", "https:///", "http://[::1]:65536"];
    // Review round 3: the URI parser counts a bracketed host's colons but doesn't read it.
    let v6 = ["http://[:::]:7878", "[1::2::3]:7878", "http://[127.0.0.1]:7878", "http://[::g]"];
    for typed in not.into_iter().chain(ports).chain(odd).chain(v6) {
        let api = HttpBackend::new(typed);
        assert!(!api.address_ok(), "{typed:?}");
        assert!(api.url("/health").starts_with("folo-not-sent:"), "never sent: {typed:?}");
        let e = api.health().unwrap_err();
        assert_eq!(e.code, "server_unreachable", "{typed:?}");
        assert!(e.message.ends_with("is not a server address"), "{typed:?}: {}", e.message);
        assert!(!e.message.contains("reach"), "{typed:?}: {}", e.message);
        assert!(e.is_bad_address(), "the sidebar says so too: {typed:?}");
        assert!(e.hint.unwrap().contains("http://host:port"));
    }
    assert_eq!(
        HttpBackend::new("foo://x:1").health().unwrap_err().message,
        "foo://x:1 is not a server address"
    );
    for fine in
        ["127.0.0.1:7878", "http://localhost:18090/", "https://[::1]:443", "http://my-box", "HTTP://a:1"]
            .into_iter()
            .chain(["http://[::1]", "localhost:65535", "http://u@h:8080", "[fe80::1]:7878", "http://[::]:1"])
    {
        let api = HttpBackend::new(fine);
        assert!(api.address_ok(), "{fine:?}");
        assert!(api.url("/health").to_ascii_lowercase().starts_with("http"), "{fine:?}");
    }
}

#[test]
fn a_token_server_says_what_works_instead_of_asking_for_a_header() {
    let body = r#"{ "error": { "code": "unauthorized", "message": "this server needs a token: send `Authorization: Bearer <token>`", "hint": "send the token the server was started with" } }"#;
    let (url, _requests) = canned_server(vec![
        json("200 OK", r#"{ "status": "ok", "version": "0.1.0", "mock": false }"#),
        json("401 Unauthorized", body),
        b"HTTP/1.1 401 Unauthorized
content-length: 0
connection: close

"
        .to_vec(),
    ]);
    // `/api/health` is open on such a server: the probe passes, every other call is refused.
    assert_eq!(probe(&url), Probe::Folo);
    // Any web server answers 200; only a FOLO server's health counts.
    let (page, _r) = canned_server(vec![
            b"HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: 15\r\nconnection: close\r\n\r\n<html></html>\r\n"
                .to_vec(),
        ]);
    assert_eq!(probe(&page), Probe::Answered);
    // Nor an unhealthy health: the app starts its own server only where nothing healthy answers.
    let (down, _r) =
        canned_server(vec![json("200 OK", r#"{ "status": "down", "version": "0.1.0", "mock": false }"#)]);
    assert_eq!(probe(&down), Probe::Answered);
    let api = HttpBackend::new(&url);
    for e in [api.system().unwrap_err(), api.datasets().unwrap_err()] {
        assert_eq!((e.code.as_str(), e.message.as_str()), ("unauthorized", "This server needs a token"));
        let hint = e.hint.unwrap();
        assert!(hint.contains("--token") && !hint.contains("Authorization"), "{hint}");
    }
}

/// `FOLO_SERVER` on this machine reached through a proxy (`ALL_PROXY` without a matching `NO_PROXY`):
/// nothing listens on that port here, but the proxy answers for it, so the probe finds the server
/// (the refused-at-once check would say nobody is there, and the app would start a second server).
#[test]
fn a_server_behind_a_proxy_is_found_although_its_port_is_closed_here() {
    use std::io::{BufRead, BufReader, Write};
    // A closed loopback port, held through both checks: on Windows a socket bound there that never
    // listens, so no other process can take the port meanwhile. Elsewhere (no refused-at-once check,
    // and std can't bind without listening) a port bound for a moment and let go: another process may
    // take it, so the direct probe below is asked only on Windows.
    #[cfg(windows)]
    let (_held, closed) = {
        let held = super::refused::ports::closed_port("127.0.0.1:0".parse().unwrap()).unwrap();
        let port = held.port;
        (held, std::net::SocketAddr::from(([127, 0, 0, 1], port)))
    };
    #[cfg(not(windows))]
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    // The proxy: takes the CONNECT to that port and answers the health check through the tunnel.
    let proxy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let proxy_url = format!("http://{}", proxy.local_addr().unwrap());
    let (tx, connects) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let Ok((mut sock, _)) = proxy.accept() else { return };
        let mut reader = BufReader::new(sock.try_clone().unwrap());
        // A request's head: its first line, then the headers up to the blank line.
        let mut head = || {
            let mut first = String::new();
            let _ = reader.read_line(&mut first);
            loop {
                let mut l = String::new();
                if reader.read_line(&mut l).unwrap_or(0) == 0 || l == "\r\n" {
                    break first;
                }
            }
        };
        let _ = tx.send(head());
        let _ = sock.write_all(b"HTTP/1.1 200 Connection established\r\n\r\n");
        let _get = head();
        let _ = sock.write_all(&json("200 OK", r#"{ "status": "ok", "version": "0.1.0", "mock": false }"#));
    });
    let url = format!("http://{closed}");
    let probe_through = super::server::probe_through;
    assert_eq!(probe_through(&url, Some(ureq::Proxy::new(&proxy_url).unwrap())), Probe::Folo);
    let connect = connects.recv().unwrap();
    assert!(connect.starts_with(&format!("CONNECT {closed} ")), "{connect}");
    // Without the proxy the same address is closed (where it is still held: Windows).
    #[cfg(windows)]
    assert_eq!(probe_through(&url, None), Probe::Nothing);
}

#[test]
fn closing_asks_the_apps_server_for_its_runs_and_tells_no_answer_apart() {
    let (url, requests) = canned_server(vec![json("200 OK", r#"{ "runs": [] }"#), Vec::new()]);
    assert_eq!(runs_now(&url).map(|r| r.len()), Some(0));
    assert!(requests.recv().unwrap().starts_with("GET /api/runs "));
    // Hangs up without answering: unknown, not "nothing is training".
    assert!(runs_now(&url).is_none());
}

/// Every benchmark call the app makes, against a real `folo serve` with a benchmark that has a
/// finished entry and a model: `FOLO_SERVER=… cargo test real_server_bench -- --ignored
/// --nocapture`. It makes a benchmark, starts and stops an entry and deletes both again.
#[test]
#[ignore]
fn real_server_bench_calls() {
    let api = HttpBackend::new(&std::env::var("FOLO_SERVER").unwrap_or("http://127.0.0.1:7878".into()));
    let features = api.system().unwrap().features;
    assert!(features.iter().any(|f| f == FEATURE_VISION_BENCH), "{features:?}");
    let catalog = api.bench_catalog().unwrap();
    assert_eq!(catalog.conditions.len(), 12);
    let benches = api.benches().unwrap();
    let (b, done) = benches
        .iter()
        .find_map(|b| {
            let entries = api.bench_entries(&b.id).unwrap();
            entries.into_iter().find(|e| e.status == BenchEntryStatus::Done).map(|e| (b.clone(), e))
        })
        .expect("a benchmark with a finished entry");
    let result = api.bench_result(&b.id, &done.id).unwrap();
    println!("{} / {}: score {:?}", b.name, done.name, result.entry.score.map(|s| s.total));
    let page = api.bench_entry_images(&b.id, &done.id, "worst", 0, 24).unwrap();
    assert!(!page.images.is_empty() && page.total >= page.images.len() as u64);
    let bad: Vec<u32> = page.images.iter().map(|i| i.missed + i.wrong).collect();
    assert!(bad.windows(2).all(|w| w[0] >= w[1]), "{bad:?}");
    let first = api.bench_images(&b.id, 0, 1, None).unwrap().images.remove(0);
    let decode = |d: ImageData| match d {
        ImageData::Encoded(bytes) => image::load_from_memory(&bytes).unwrap().to_rgb8(),
        ImageData::Rgba { .. } => panic!("the server sends files"),
    };
    let plain = decode(api.bench_image(&b.id, &first.id, None, Some(320)).unwrap());
    let cond = b.plan.conditions.first().map(|c| (c.condition.clone(), c.severities[0]));
    if let Some((c, s)) = &cond {
        let changed = decode(api.bench_image(&b.id, &first.id, Some((c, *s)), Some(320)).unwrap());
        assert_eq!(plain.dimensions(), changed.dimensions());
        assert_ne!(plain.as_raw(), changed.as_raw(), "{c} {s} changes the pixels");
    }
    assert!(plain.width().max(plain.height()) <= 320);
    let _ = api.bench_loops(&b.id).unwrap();
    let _ = api.label_agents().unwrap();
    let _ = api.label_jobs().unwrap();
    drop(api.bench_events(&b.id).unwrap());
    // The write path, on a benchmark of its own.
    let ds =
        api.datasets().unwrap().into_iter().find(|d| d.splits.val > 0).expect("a dataset with val images");
    let created = api
        .create_bench(&CreateBench {
            name: "desktop real-server check".into(),
            description: String::new(),
            source: BenchSource::Dataset { dataset_id: ds.id.clone(), split: Some(Split::Val) },
            max_images: Some(8),
            seed: 0,
            plan: None,
        })
        .unwrap();
    // Whatever happens next, the benchmark goes again.
    let run = || -> ApiResult<()> {
        let model = api
            .models()?
            .into_iter()
            .find(|m| crate::bench::model_fits(&m.classes, &created.classes))
            .expect("a model that knows the dataset's classes");
        let req = StartBenchEntry {
            contestant: BenchContestantSpec::Model { model_id: model.id },
            name: None,
            max_images: None,
            conditions: Some(false),
        };
        let events = api.bench_events(&created.id)?;
        let e = api.start_bench_entry(&created.id, &req)?;
        // The entry comes over the stream too (its progress, or its end).
        let streamed = std::iter::from_fn(|| events.recv_timeout(Duration::from_secs(20)).ok())
            .any(|ev| matches!(ev, BenchEvent::Entry(x) if x.id == e.id));
        assert!(streamed, "the entry is streamed");
        if e.status.is_active() {
            let _ = api.stop_bench_entry(&created.id, &e.id);
        }
        for _ in 0..100 {
            if api.bench_entries(&created.id)?.iter().all(|x| !x.status.is_active()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        api.delete_bench_entry(&created.id, &e.id)
    };
    let ran = run();
    api.delete_bench(&created.id).unwrap();
    ran.unwrap();
    assert!(!api.benches().unwrap().iter().any(|x| x.id == created.id));
}
