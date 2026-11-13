//! A FOLO server that already runs (docs/CLOUD.md, `remote`): `folo serve --host 0.0.0.0 --token <secret>` on a
//! machine of yours. Nothing is launched or terminated; the connection is its URL and token, and its one offer is
//! the device it trains on.

use std::net::IpAddr;
use std::time::Duration;

use folo_api::{CloudAvailability, CloudField, CloudOffer, CloudProvider, Health, SystemResponse};

use crate::{Account, Endpoint, Error, ErrorKind, Provider, Result, Values};

/// The features a server needs to be a worker ([`folo_api::SystemResponse::features`]).
pub const REQUIRED_FEATURES: [&str; 3] =
    [folo_api::FEATURE_UPLOADS, folo_api::FEATURE_RUN_CURSORS, folo_api::FEATURE_RUN_CHECKPOINT];

/// The id of the offer of a server that trains on its GPU, and of one that has none and trains on its CPU.
pub const GPU_OFFER: &str = "gpu";
pub const CPU_OFFER: &str = "cpu";

const URL_FIELD: &str = "url";
const TOKEN_FIELD: &str = "token";
/// Time to reach the server, and to get an answer: these calls answer someone waiting in a form.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// A FOLO server that already runs.
#[derive(Clone, Debug, Default)]
pub struct Remote;

impl Remote {
    /// The provider.
    pub fn new() -> Remote {
        Remote
    }
}

impl Provider for Remote {
    fn info(&self) -> CloudProvider {
        CloudProvider {
            id: "remote".into(),
            name: "Remote FOLO server".into(),
            about: "A FOLO server you run yourself (`folo serve --host 0.0.0.0 --token …`), on a workstation or a \
                    VM at any cloud."
                .into(),
            docs_url: None,
            fields: vec![
                CloudField {
                    key: URL_FIELD.into(),
                    label: "Server URL".into(),
                    secret: false,
                    required: true,
                    placeholder: Some("https://gpu-box.example.com:7878".into()),
                    help: Some(
                        "The address of its `folo serve`; use https:// (a TLS proxy or a tunnel) across networks you \
                         do not control."
                            .into(),
                    ),
                },
                CloudField {
                    key: TOKEN_FIELD.into(),
                    label: "Token".into(),
                    secret: true,
                    required: false,
                    placeholder: None,
                    help: Some("Its `--token` (or FOLO_TOKEN); empty when it runs without one.".into()),
                },
            ],
            provisions: false,
            self_terminates: false,
        }
    }

    fn check(&self, values: &Values) -> Result<Account> {
        let server = Server::of(values)?;
        let client = server.client();
        let health: Health = client
            .get("/health")
            .map_err(|e| if e.is_unreachable() { server.unreachable(e) } else { server.not_folo(e) })?;
        if health.status != "ok" {
            return Err(server.not_folo(format!("its health is `{}`", health.status)));
        }
        let system = server.system(&client)?;
        let missing: Vec<&str> =
            REQUIRED_FEATURES.into_iter().filter(|f| !system.features.iter().any(|g| g == f)).collect();
        if !missing.is_empty() {
            return Err(Error::new(
                ErrorKind::Refused,
                format!(
                    "the FOLO server at {} runs version {}, which cannot train cloud runs (it lacks {}); update it \
                     to FOLO {} or later",
                    server.host,
                    server.scrub(&health.version),
                    missing.join(", "),
                    env!("CARGO_PKG_VERSION")
                ),
            ));
        }
        Ok(Account { note: server.note(), label: Some(server.host), balance_usd: None })
    }

    fn offers(&self, values: &Values) -> Result<Vec<CloudOffer>> {
        let server = Server::of(values)?;
        let system = server.system(&server.client())?;
        // Offers are logged when a run starts.
        let mut offer = offer(&system);
        offer.gpu = server.scrub(&offer.gpu);
        offer.detail = offer.detail.map(|d| server.scrub(d));
        Ok(vec![offer])
    }

    fn fixed_endpoint(&self, values: &Values) -> Option<Endpoint> {
        let server = Server::of(values).ok()?;
        Some(Endpoint { url: server.url, token: server.token })
    }
}

/// A connection's server: its URL, checked and normalised.
struct Server {
    /// Without a trailing slash.
    url: String,
    /// The URL's host, without the brackets of an IPv6 address.
    host: String,
    https: bool,
    token: String,
}

impl Server {
    fn of(values: &Values) -> Result<Server> {
        let raw = values.require(URL_FIELD)?;
        let invalid = |why: &str| Error::new(ErrorKind::Invalid, format!("`url` {why}"));
        if !raw.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(invalid("must not contain spaces or control characters"));
        }
        let uri: http::Uri = raw.parse().map_err(|_| invalid("is not a URL"))?;
        let https = match uri.scheme_str() {
            Some("https") => true,
            Some("http") => false,
            _ => return Err(invalid("must start with https:// or http://")),
        };
        let authority = uri.authority().ok_or_else(|| invalid("has no host"))?;
        // Credentials belong in the token field, not in a URL that is shown and logged.
        if authority.as_str().contains('@') {
            return Err(invalid("must not contain a user name or password"));
        }
        if uri.query().is_some() || raw.contains('#') {
            return Err(invalid("must not have a query or a fragment"));
        }
        let host = authority.host().trim_start_matches('[').trim_end_matches(']').to_string();
        if host.is_empty() {
            return Err(invalid("has no host"));
        }
        let url = raw.trim_end_matches('/').to_string();
        let token = values.get(TOKEN_FIELD).unwrap_or_default().to_string();
        Ok(Server { url, host, https, token })
    }

    /// The warning about an `http://` URL whose traffic leaves the user's own networks.
    fn note(&self) -> Option<String> {
        (!self.https && !is_local(&self.host)).then(|| {
            format!(
                "{} is reached over http://: the token and your data cross the network unencrypted. Use https:// \
                 (a TLS proxy or a tunnel) unless this network is yours.",
                self.host
            )
        })
    }

    fn client(&self) -> folo_client::Client {
        folo_client::Client::with_timeouts(&self.url, CONNECT_TIMEOUT, REQUEST_TIMEOUT)
            .token(self.token.clone())
    }

    /// `GET /api/system` with the token.
    fn system(&self, client: &folo_client::Client) -> Result<SystemResponse> {
        client.get("/system").map_err(|e| match e.status.map(|s| s.as_u16()) {
            Some(401) if self.token.is_empty() => Error::new(
                ErrorKind::Auth,
                format!("the server at {} needs a token (it runs with `--token`)", self.host),
            ),
            Some(401) => {
                Error::new(ErrorKind::Auth, format!("the server at {} rejected the token", self.host))
            }
            None => self.unreachable(e),
            Some(_) => {
                Error::new(ErrorKind::Unavailable, format!("the server at {}: {}", self.host, self.scrub(e)))
            }
        })
    }

    /// The server does not answer: said as a sentence about it (never starting with the URL, which a front end would
    /// capitalise), the transport's own words once (without the URL they repeat), and the likely cause.
    fn unreachable(&self, e: impl std::fmt::Display) -> Error {
        let why = self.detail(e);
        // An https:// URL of a server that speaks plain HTTP fails in the TLS handshake (rustls: a record that is not
        // TLS); `folo serve` itself serves plain HTTP.
        let tls = self.https && (why.contains("InvalidContentType") || why.contains("corrupt message"));
        let cause = if tls {
            "it seems to speak plain HTTP: use http://, or put a TLS proxy in front of it".to_string()
        } else {
            "is `folo serve --host 0.0.0.0` running there?".to_string()
        };
        Error::new(
            ErrorKind::Unavailable,
            format!("the server at {} does not answer ({why}); {cause}", self.url),
        )
    }

    /// The error of a health check that found something other than FOLO.
    fn not_folo(&self, why: impl std::fmt::Display) -> Error {
        Error::new(
            ErrorKind::Unavailable,
            format!(
                "the server at {} does not look like a FOLO server (GET /api/health: {})",
                self.url,
                self.detail(why)
            ),
        )
    }

    /// A transport or server message about this server, scrubbed, without the URL it may start with.
    fn detail(&self, e: impl std::fmt::Display) -> String {
        let text = self.scrub(e);
        text.strip_prefix(&format!("{}: ", self.url)).map(str::to_string).unwrap_or(text)
    }

    /// `text`, made of what the server answered (decoded), with its token blanked out: a server may say anything.
    fn scrub(&self, text: impl std::fmt::Display) -> String {
        crate::scrub(&text.to_string(), [self.token.as_str()])
    }
}

/// The server's one offer: the device `device: auto` trains on there.
fn offer(system: &SystemResponse) -> CloudOffer {
    let gpus = &system.system.gpus;
    let ordinal = match &system.capabilities {
        Some(c) => c.device.strip_prefix("gpu:").and_then(|n| n.parse::<u32>().ok()),
        // A server from before `capabilities` trains on its first GPU, if it has one.
        None => gpus.first().map(|g| g.ordinal),
    };
    let (id, gpu, vram_bytes, detail) = match ordinal {
        Some(n) => match gpus.iter().find(|g| g.ordinal == n) {
            Some(g) => (GPU_OFFER, g.name.clone(), g.memory_bytes, None),
            None => (GPU_OFFER, format!("gpu:{n}"), 0, None),
        },
        None => (CPU_OFFER, "CPU".to_string(), 0, Some(system.system.cpu.name.clone())),
    };
    CloudOffer {
        id: id.into(),
        gpu,
        gpu_count: 1,
        vram_bytes,
        price_per_hour: None,
        region: None,
        detail,
        availability: CloudAvailability::High,
    }
}

/// `host` is this machine or an address of a private network (loopback, private or link-local), where `http://`
/// does not cross networks the user does not control.
fn is_local(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        Ok(IpAddr::V6(ip)) => match ip.to_ipv4_mapped() {
            Some(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
            // Loopback, unique local (fc00::/7), link-local (fe80::/10).
            None => {
                ip.is_loopback() || ip.segments()[0] & 0xfe00 == 0xfc00 || ip.segments()[0] & 0xffc0 == 0xfe80
            }
        },
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::fixture::{Answer, Fixture};

    const TOKEN: &str = "remote-token-0123456789";
    const ALL_FEATURES: &str = r#"["queue_control","uploads","run_cursors","run_checkpoint","cloud"]"#;

    fn values(url: &str, token: &str) -> Values {
        Values::from(BTreeMap::from([
            (URL_FIELD.to_string(), url.to_string()),
            (TOKEN_FIELD.to_string(), token.to_string()),
        ]))
    }

    fn health(version: &str) -> Answer {
        Answer::json(200, format!(r#"{{"status":"ok","version":"{version}","api":"v1","mock":false}}"#))
    }

    /// `/api/system` of a server with `gpus` (JSON), `capabilities` (JSON or nothing) and `features` (JSON).
    fn system(gpus: &str, capabilities: Option<&str>, features: &str) -> Answer {
        let capabilities = capabilities.map(|c| format!(r#","capabilities":{c}"#)).unwrap_or_default();
        Answer::json(
            200,
            format!(
                r#"{{"cpu":{{"name":"AMD EPYC 7B13","threads":16,"util":0.1}},
                    "memory":{{"used_bytes":1,"total_bytes":2}},"gpus":{gpus},"features":{features}{capabilities}}}"#
            ),
        )
    }

    const ONE_GPU: &str = r#"[{"ordinal":0,"name":"NVIDIA GeForce RTX 4090","backend":"vulkan",
                              "memory_bytes":25769803776,"util":null}]"#;
    const ON_GPU: &str = r#"{"device":"gpu:0","shader_f16":true}"#;
    const ON_CPU: &str = r#"{"device":"cpu"}"#;

    fn assert_error(e: &Error, kind: ErrorKind, part: &str) {
        assert_eq!(e.kind, kind, "{e:?}");
        assert!(e.message.contains(part), "`{part}` not in: {}", e.message);
        assert!(!e.message.contains(TOKEN) && !format!("{e:?}").contains(TOKEN), "{e:?}");
    }

    #[test]
    fn info_describes_the_connection() {
        let info = Remote::new().info();
        assert_eq!((info.id.as_str(), info.provisions, info.self_terminates), ("remote", false, false));
        let fields: Vec<(&str, bool, bool)> =
            info.fields.iter().map(|f| (f.key.as_str(), f.secret, f.required)).collect();
        assert_eq!(fields, [("url", false, true), ("token", true, false)]);
    }

    #[test]
    fn what_a_server_answers_never_carries_its_token_into_a_message_or_an_offer() {
        // The token as JSON escapes: the same text once decoded.
        let token: String = TOKEN.chars().map(|c| format!("\\u{:04x}", c as u32)).collect();
        let f = Fixture::serve(vec![Answer::json(
            200,
            format!(r#"{{"status":"{token}","version":"0.1.0","api":"v1","mock":false}}"#),
        )]);
        assert_error(&Remote.check(&values(&f.base(), TOKEN)).unwrap_err(), ErrorKind::Unavailable, "health");
        // A server too old to be a worker, whose version says the token.
        let f = Fixture::serve(vec![health(&token), system(ONE_GPU, None, r#"["queue_control"]"#)]);
        assert_error(
            &Remote.check(&values(&f.base(), TOKEN)).unwrap_err(),
            ErrorKind::Refused,
            "cannot train",
        );
        // An error of its own that quotes it.
        let refused = format!(r#"{{"error":{{"code":"internal","message":"bad {token}","hint":null}}}}"#);
        let f = Fixture::serve(vec![health("0.1.0"), Answer::json(500, refused)]);
        assert_error(&Remote.check(&values(&f.base(), TOKEN)).unwrap_err(), ErrorKind::Unavailable, "bad");
        // A GPU named with it: the offer is logged when a run starts.
        let gpus =
            format!(r#"[{{"ordinal":0,"name":"{token}","backend":"vulkan","memory_bytes":1,"util":null}}]"#);
        let f = Fixture::serve(vec![system(&gpus, Some(ON_GPU), ALL_FEATURES)]);
        let offers = Remote.offers(&values(&f.base(), TOKEN)).unwrap();
        assert!(!offers[0].gpu.contains(TOKEN), "{offers:?}");
    }

    #[test]
    fn check_accepts_a_current_server_with_a_gpu() {
        let f = Fixture::serve(vec![health("0.1.0"), system(ONE_GPU, Some(ON_GPU), ALL_FEATURES)]);
        let account = Remote.check(&values(&format!("{}/", f.base()), TOKEN)).unwrap();
        assert_eq!(account, Account { label: Some("127.0.0.1".into()), balance_usd: None, note: None });
        let r = f.requests();
        assert_eq!((r[0].path.as_str(), r[1].path.as_str()), ("/api/health", "/api/system"));
        assert_eq!(r[1].header("authorization"), Some(format!("Bearer {TOKEN}").as_str()));
    }

    #[test]
    fn check_maps_the_token_answers() {
        let unauthorized =
            || Answer::json(401, r#"{"error":{"code":"unauthorized","message":"no","hint":null}}"#);
        let f = Fixture::serve(vec![health("0.1.0"), unauthorized()]);
        assert_error(
            &Remote.check(&values(&f.base(), TOKEN)).unwrap_err(),
            ErrorKind::Auth,
            "rejected the token",
        );
        let f = Fixture::serve(vec![health("0.1.0"), unauthorized()]);
        assert_error(&Remote.check(&values(&f.base(), "")).unwrap_err(), ErrorKind::Auth, "needs a token");
        assert_eq!(f.requests()[1].header("authorization"), None);
    }

    #[test]
    fn check_names_the_version_of_a_server_that_is_too_old() {
        let f =
            Fixture::serve(vec![health("0.0.9"), system(ONE_GPU, None, r#"["queue_control","uploads"]"#)]);
        let e = Remote.check(&values(&f.base(), TOKEN)).unwrap_err();
        assert_error(&e, ErrorKind::Refused, "runs version 0.0.9");
        assert!(
            e.message.contains("run_cursors, run_checkpoint") && e.message.contains("update it"),
            "{}",
            e.message
        );
    }

    #[test]
    fn check_refuses_what_is_not_folo() {
        let cases = [
            Answer::json(404, "<html>Not Found</html>"),
            Answer::json(200, r#"{"status":"ok"}"#),
            Answer::json(200, r#"{"status":"degraded","version":"0.1.0","api":"v1","mock":false}"#),
        ];
        for answer in cases {
            let f = Fixture::serve(vec![answer]);
            let e = Remote.check(&values(&f.base(), TOKEN)).unwrap_err();
            assert_error(&e, ErrorKind::Unavailable, "does not look like a FOLO server");
        }
        let silent = Fixture::hang_up();
        let e = Remote.check(&values(&silent.base(), TOKEN)).unwrap_err();
        assert_error(&e, ErrorKind::Unavailable, "does not answer");
        // A sentence about the server (a front end capitalises its first letter: never "Http://…"), its URL once.
        assert!(e.message.starts_with("the server at http://"), "{}", e.message);
        assert_eq!(e.message.matches(&silent.base()).count(), 1, "{}", e.message);
        assert!(e.message.ends_with("is `folo serve --host 0.0.0.0` running there?"), "{}", e.message);
        // https:// to a server that speaks plain HTTP: it answers the TLS hello with an HTTP 400, as `folo serve`
        // (hyper) does. Said so, with what to do.
        let plain = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = plain.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut s, _) = plain.accept().unwrap();
            let _ = s.read(&mut [0u8; 1024]);
            let _ =
                s.write_all(b"HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
        });
        let e = Remote.check(&values(&format!("https://{addr}"), TOKEN)).unwrap_err();
        server.join().unwrap();
        assert_error(&e, ErrorKind::Unavailable, "it seems to speak plain HTTP: use http://");
    }

    #[test]
    fn http_is_noted_only_beyond_private_networks() {
        let note = |url: &str| Server::of(&values(url, "")).unwrap().note();
        let public = note("http://203.0.113.7:7878").unwrap();
        assert!(
            public.starts_with("203.0.113.7 is reached over http://") && public.contains("unencrypted"),
            "{public}"
        );
        assert_eq!(note("http://gpu-box.example.com:7878").map(|n| n.contains("gpu-box")), Some(true));
        for quiet in
            ["https://203.0.113.7", "http://127.0.0.1:7878", "http://192.168.1.20:7878", "http://[::1]:7878"]
        {
            assert_eq!(note(quiet), None, "{quiet}");
        }
        for local in [
            "127.0.0.1",
            "localhost",
            "192.168.1.20",
            "10.1.2.3",
            "172.16.0.9",
            "169.254.1.1",
            "::1",
            "fd12:3456::1",
            "fe80::1",
            "::ffff:192.168.0.2",
        ] {
            assert!(is_local(local), "{local}");
        }
        for public in ["203.0.113.7", "8.8.8.8", "gpu-box.example.com", "2001:db8::1", "100.64.0.1"] {
            assert!(!is_local(public), "{public}");
        }
        // A real check: the note names the host and says why.
        let f = Fixture::serve(vec![health("0.1.0"), system(ONE_GPU, Some(ON_GPU), ALL_FEATURES)]);
        let account = Remote.check(&values(&f.base(), TOKEN)).unwrap();
        assert_eq!(account.note, None, "loopback needs no note");
    }

    #[test]
    fn offers_are_the_device_the_server_trains_on() {
        let f = Fixture::serve(vec![
            system(ONE_GPU, Some(ON_GPU), ALL_FEATURES),
            system("[]", Some(ON_CPU), ALL_FEATURES),
            system(ONE_GPU, None, "[]"),
        ]);
        let v = values(&f.base(), TOKEN);
        let gpu = Remote.offers(&v).unwrap();
        assert_eq!(gpu.len(), 1);
        let o = &gpu[0];
        assert_eq!(
            (o.id.as_str(), o.gpu.as_str(), o.vram_bytes),
            ("gpu", "NVIDIA GeForce RTX 4090", 25769803776)
        );
        assert_eq!((o.price_per_hour, o.availability, o.gpu_count), (None, CloudAvailability::High, 1));
        let cpu = &Remote.offers(&v).unwrap()[0];
        assert_eq!((cpu.id.as_str(), cpu.gpu.as_str()), ("cpu", "CPU"));
        assert_eq!(cpu.detail.as_deref(), Some("AMD EPYC 7B13"));
        let old = &Remote.offers(&v).unwrap()[0];
        assert_eq!((old.id.as_str(), old.gpu.as_str()), ("gpu", "NVIDIA GeForce RTX 4090"));
    }

    #[test]
    fn urls_are_checked_and_normalised() {
        let s = Server::of(&values("https://gpu-box.example.com:7878/folo/", TOKEN)).unwrap();
        assert_eq!(
            (s.url.as_str(), s.host.as_str(), s.https),
            ("https://gpu-box.example.com:7878/folo", "gpu-box.example.com", true)
        );
        assert_eq!(Server::of(&values("http://[::1]:7878", "")).unwrap().host, "::1");
        for bad in [
            "gpu-box:7878",
            "ftp://gpu-box",
            "https://",
            "https://user:pw@gpu-box",
            "https://gpu-box/?a=1",
            "https://gpu-box/#x",
            "https://gpu box",
        ] {
            let e = Server::of(&values(bad, "")).err().unwrap_or_else(|| panic!("{bad} was accepted"));
            assert_error(&e, ErrorKind::Invalid, "`url`");
        }
        assert_error(&Remote.check(&Values::default()).unwrap_err(), ErrorKind::Invalid, "url");
    }

    #[test]
    fn the_fixed_endpoint_is_the_url_and_token() {
        let e = Remote.fixed_endpoint(&values("https://gpu-box.example.com/", TOKEN)).unwrap();
        assert_eq!((e.url.as_str(), e.token.as_str()), ("https://gpu-box.example.com", TOKEN));
        assert_eq!(Remote.fixed_endpoint(&values("nope", TOKEN)), None);
        assert_eq!(
            Remote
                .launch(
                    &values("https://x.example", ""),
                    &crate::LaunchSpec {
                        name: String::new(),
                        offer_id: "gpu".into(),
                        disk_gb: 1,
                        token: String::new(),
                        lifetime_s: 1,
                        worker_base_url: String::new(),
                    }
                )
                .unwrap_err()
                .kind,
            ErrorKind::Invalid
        );
    }
}
