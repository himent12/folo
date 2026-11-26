//! `cloud/connections.json`: the cloud connections with their credentials, rewritten atomically and readable by the
//! owner only ([`super::files::write_private`]). The file is read on every call, so a connection made with `folo
//! cloud connect` is there for a running server at once. Credentials never leave this file: the API's
//! [`CloudConnection`] carries a hint of each secret ([`redact`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::http::StatusCode;
use folo_api::{CloudConnection, CloudConnectionStatus, CloudProvider};
use folo_cloud::{Account, Values};
use serde::{Deserialize, Serialize};

use super::files;
use crate::error::{Error, Result};
use crate::util::{fnv1a, rfc3339, unique_id, unix_now};

/// Longest connection name.
const MAX_NAME_CHARS: usize = 100;

/// A connection as stored: what the API shows plus the values themselves.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Stored {
    pub id: String,
    pub provider: String,
    pub name: String,
    pub created: String,
    pub status: CloudConnectionStatus,
    #[serde(default)]
    pub account: Option<String>,
    #[serde(default)]
    pub balance_usd: Option<f64>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub checked: Option<String>,
    /// Every value by field key, secrets in clear.
    values: BTreeMap<String, String>,
}

// Not derived: the values are credentials, which must not reach a log through `{:?}`.
impl std::fmt::Debug for Stored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stored")
            .field("id", &self.id)
            .field("provider", &self.provider)
            .field("values", &self.values.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Stored {
    /// The values, for the provider.
    pub fn values(&self) -> Values {
        Values::from(self.values.clone())
    }

    /// The API record: fields that are not secret as entered, secrets as a hint. Values of a field the provider
    /// does not declare (or of a provider this version does not know) count as secret.
    pub fn api(&self, provider: Option<&CloudProvider>) -> CloudConnection {
        let plain = |key: &str| {
            provider.and_then(|p| p.fields.iter().find(|f| f.key == key)).is_some_and(|f| !f.secret)
        };
        let values = self
            .values
            .iter()
            .map(|(k, v)| (k.clone(), if plain(k) { v.clone() } else { redact(v) }))
            .collect();
        CloudConnection {
            id: self.id.clone(),
            provider: self.provider.clone(),
            name: self.name.clone(),
            created: self.created.clone(),
            status: self.status,
            account: self.account.clone(),
            balance_usd: self.balance_usd,
            error: self.error.clone(),
            note: self.note.clone(),
            checked: self.checked.clone(),
            values,
        }
    }

    /// Values of fields marked secret by `provider` (or unknown to it), used for defense-in-depth scrubbing.
    fn secret_values(&self, provider: Option<&CloudProvider>) -> Vec<String> {
        let plain = |key: &str| {
            provider.and_then(|p| p.fields.iter().find(|f| f.key == key)).is_some_and(|f| !f.secret)
        };
        self.values.iter().filter(|(k, _)| !plain(k)).map(|(_, v)| v.clone()).collect()
    }

    /// Records a check's result: the account when it passed, the error when it did not. Scrubs stored secret
    /// values as defense-in-depth so no provider error, account label or note can persist a credential.
    pub fn checked(
        &mut self,
        result: std::result::Result<Account, folo_cloud::Error>,
        provider: Option<&CloudProvider>,
    ) {
        let secrets = self.secret_values(provider);
        let scrub = |s: String| folo_cloud::scrub(&s, secrets.iter().map(String::as_str));
        self.checked = Some(rfc3339(unix_now()));
        match result {
            Ok(a) => {
                self.status = CloudConnectionStatus::Ok;
                self.account = a.label.map(scrub);
                self.balance_usd = a.balance_usd;
                self.note = a.note.map(scrub);
                self.error = None;
            }
            Err(e) => {
                self.status = CloudConnectionStatus::Error;
                self.error = Some(scrub(e.message));
            }
        }
    }
}

/// A hint of secret `value`: `••••` and its last 4 characters, or just `••••` for one shorter than 8 characters
/// (so a short secret is not mostly shown).
pub(crate) fn redact(value: &str) -> String {
    let n = value.chars().count();
    if n < 8 {
        return "••••".into();
    }
    let tail: String = value.chars().skip(n - 4).collect();
    format!("••••{tail}")
}

/// The file's shape.
#[derive(Default, Serialize, Deserialize)]
struct ConnectionsFile {
    connections: Vec<Stored>,
}

/// The connections of one data dir.
pub(crate) struct Connections {
    path: PathBuf,
    lock: PathBuf,
}

/// `404 connection_not_found`.
pub(crate) fn not_found(id: &str) -> Error {
    Error::not_found("connection", id)
        .hint("GET /api/cloud/connections (`folo cloud connections`, MCP cloud_connections) lists them")
}

impl Connections {
    /// The connections in `cloud_dir`, guarded by the cloud lock `lock`.
    pub fn new(cloud_dir: &Path, lock: PathBuf) -> Connections {
        Connections { path: cloud_dir.join("connections.json"), lock }
    }

    /// The file; a missing one is empty, a malformed one an error (it is never overwritten).
    fn read(&self) -> Result<ConnectionsFile> {
        let bytes = match std::fs::read(&self.path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ConnectionsFile::default()),
            Err(e) => return Err(Error::io(format!("reading {}: {e}", self.path.display()))),
        };
        serde_json::from_slice(&bytes).map_err(|e| {
            Error::internal(format!("{} is malformed: {e}", self.path.display()))
                .hint("fix the JSON by hand or move the file away; FOLO will not overwrite it")
        })
    }

    fn write(&self, file: &ConnectionsFile) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(file).expect("connections serialize");
        files::write_private(&self.path, &bytes)
            .map_err(|e| Error::io(format!("writing {}: {e}", self.path.display())))
    }

    /// Runs `change` on the file under the cloud lock and writes it back.
    fn modify<T>(&self, change: impl FnOnce(&mut ConnectionsFile) -> Result<T>) -> Result<T> {
        let _lock = files::lock(&self.lock)
            .map_err(|e| Error::io(format!("locking {}: {e}", self.lock.display())))?;
        let mut file = self.read()?;
        let out = change(&mut file)?;
        self.write(&file)?;
        Ok(out)
    }

    /// Every connection, oldest first.
    pub fn list(&self) -> Result<Vec<Stored>> {
        Ok(self.read()?.connections)
    }

    /// Connection `id`.
    pub fn get(&self, id: &str) -> Result<Stored> {
        self.list()?.into_iter().find(|c| c.id == id).ok_or_else(|| not_found(id))
    }

    /// Stores a new connection to `provider` whose `values` passed the provider's check (`account`). An empty
    /// `name` is the provider's name, with ` 2`, ` 3`, ... when taken.
    pub fn add(
        &self,
        provider: &CloudProvider,
        name: &str,
        values: Values,
        account: Account,
    ) -> Result<Stored> {
        let name = name.trim();
        if name.chars().count() > MAX_NAME_CHARS {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                format!("name must be at most {MAX_NAME_CHARS} characters"),
            ));
        }
        self.modify(|file| {
            let taken = |n: &str| file.connections.iter().any(|c| c.name == n);
            let name = match name {
                "" => (1..)
                    .map(|i| if i == 1 { provider.name.clone() } else { format!("{} {i}", provider.name) })
                    .find(|n| !taken(n))
                    .expect("some suffix is free"),
                n => n.to_string(),
            };
            let seed = fnv1a(name.as_bytes()) ^ unix_now().to_bits();
            let id = unique_id("cc", seed, |id| file.connections.iter().any(|c| c.id == id));
            let mut c = Stored {
                id,
                provider: provider.id.clone(),
                name,
                created: rfc3339(unix_now()),
                status: CloudConnectionStatus::Ok,
                account: None,
                balance_usd: None,
                error: None,
                note: None,
                checked: None,
                values: values.as_map().clone(),
            };
            c.checked(Ok(account), Some(provider));
            file.connections.push(c.clone());
            Ok(c)
        })
    }

    /// Changes connection `id` (a check's result) and stores it.
    pub fn update(&self, id: &str, change: impl FnOnce(&mut Stored)) -> Result<Stored> {
        self.modify(|file| {
            let c = file.connections.iter_mut().find(|c| c.id == id).ok_or_else(|| not_found(id))?;
            change(c);
            Ok(c.clone())
        })
    }

    /// Forgets connection `id` once `in_use` (asked under the lock) finds nothing that still needs it.
    pub fn remove(&self, id: &str, in_use: impl FnOnce() -> Result<()>) -> Result<()> {
        self.modify(|file| {
            let before = file.connections.len();
            file.connections.retain(|c| c.id != id);
            if file.connections.len() == before {
                return Err(not_found(id));
            }
            in_use()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_api::CloudField;
    use folo_test_support::TestDir;

    fn provider() -> CloudProvider {
        let field = |key: &str, secret| CloudField {
            key: key.into(),
            label: key.into(),
            secret,
            required: false,
            placeholder: None,
            help: None,
        };
        CloudProvider {
            id: "p".into(),
            name: "Prov".into(),
            about: String::new(),
            docs_url: None,
            fields: vec![field("api_key", true), field("region", false)],
            provisions: true,
            self_terminates: false,
        }
    }

    #[test]
    fn short_secrets_are_hidden_whole() {
        assert_eq!(redact("rpa_0123456789abcd"), "••••abcd");
        assert_eq!(redact("12345678"), "••••5678");
        assert_eq!(redact("1234567"), "••••");
        assert_eq!(redact(""), "••••");
        assert_eq!(redact("ééééééééé1"), "••••ééé1");
    }

    #[test]
    fn names_get_numbers_and_values_stay_in_the_file() {
        let dir = TestDir::new("cloud-connections");
        let c = Connections::new(&dir.join("cloud"), dir.join("cloud").join(".lock"));
        let values = |k: &str| {
            Values::from(BTreeMap::from([
                ("api_key".to_string(), k.to_string()),
                ("region".to_string(), "eu".to_string()),
            ]))
        };
        let a = c.add(&provider(), "", values("secret-aaaa-1111"), Account::default()).unwrap();
        let b = c.add(&provider(), " ", values("secret-bbbb-2222"), Account::default()).unwrap();
        let own = c.add(&provider(), "mine", values("secret-cccc-3333"), Account::default()).unwrap();
        assert_eq!((a.name.as_str(), b.name.as_str(), own.name.as_str()), ("Prov", "Prov 2", "mine"));
        assert_ne!(a.id, b.id);
        let shown = b.api(Some(&provider()));
        assert_eq!(shown.values["api_key"], "••••2222");
        assert_eq!(shown.values["region"], "eu");
        assert!(!format!("{b:?}").contains("secret"));
        assert_eq!(c.get(&b.id).unwrap().values().get("api_key"), Some("secret-bbbb-2222"));

        let e = c.remove(&a.id, || Err(Error::conflict("connection_in_use", "busy"))).unwrap_err();
        assert_eq!(e.body.code, "connection_in_use");
        assert_eq!(c.list().unwrap().len(), 3, "a refused removal changes nothing");
        c.remove(&a.id, || Ok(())).unwrap();
        assert_eq!(c.get(&a.id).unwrap_err().body.code, "connection_not_found");
        let long = "x".repeat(MAX_NAME_CHARS + 1);
        assert_eq!(
            c.add(&provider(), &long, values("k"), Account::default()).unwrap_err().body.code,
            "invalid_body"
        );

        std::fs::write(dir.join("cloud").join("connections.json"), b"{ nope").unwrap();
        assert_eq!(c.list().unwrap_err().body.code, "internal");
        assert!(c.add(&provider(), "", values("k"), Account::default()).is_err());
        assert_eq!(
            std::fs::read(dir.join("cloud").join("connections.json")).unwrap(),
            b"{ nope",
            "never overwritten"
        );
    }

    #[test]
    fn checked_redacts_stored_secret_values_without_touching_plain_fields() {
        let p = provider();
        let mut stored = Stored {
            id: "cc_1".into(),
            provider: p.id.clone(),
            name: "Prov".into(),
            created: "2026-11-26T00:00:00Z".into(),
            status: CloudConnectionStatus::Ok,
            account: None,
            balance_usd: None,
            error: None,
            note: None,
            checked: None,
            values: BTreeMap::from([
                ("api_key".into(), "rpa_SUPER_SECRET+KEY/99".into()),
                ("region".into(), "eu-west-region-1".into()),
            ]),
        };
        stored.checked(
            Ok(Account {
                label: Some("eu-west-region-1 / rpa_SUPER_SECRET+KEY/99".into()),
                balance_usd: Some(5.0),
                note: Some(format!("note {}", folo_client::encode("rpa_SUPER_SECRET+KEY/99"))),
            }),
            Some(&p),
        );
        assert_eq!(stored.account.as_deref(), Some("eu-west-region-1 / [redacted]"));
        assert_eq!(stored.note.as_deref(), Some("note [redacted]"));

        stored.checked(
            Err(folo_cloud::Error::new(
                folo_cloud::ErrorKind::Auth,
                "rejected rpa_SUPER_SECRET+KEY/99 in eu-west-region-1",
            )),
            Some(&p),
        );
        assert_eq!(stored.error.as_deref(), Some("rejected [redacted] in eu-west-region-1"));
    }
}
