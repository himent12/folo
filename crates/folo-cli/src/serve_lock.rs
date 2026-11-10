//! One `folo serve` per data dir. Two servers on one data dir would each treat the other's active runs as
//! interrupted (the store stops them when it opens) and write the same registries, so `folo serve` (real store)
//! holds an exclusive OS lock on `<data dir>/serve.lock` while it runs, and writes its address to
//! `<data dir>/serve.json` so a second start can say where the running one is (FOLO Desktop connects there).

use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

/// The data dir's serve lock, held until dropped; dropping it also removes `serve.json`.
pub struct ServeLock {
    /// Locked; dropping it unlocks (the OS also releases it when the process dies).
    _file: File,
    info: PathBuf,
}

/// What [`acquire`] found.
pub enum Acquire {
    /// This process owns the data dir now.
    Held(ServeLock),
    /// Another process holds the lock; `url` is where it listens, when it has said so.
    InUse { url: Option<String> },
}

/// Takes `dir`'s serve lock (creating `dir` when missing) without waiting.
pub fn acquire(dir: &Path) -> std::io::Result<Acquire> {
    std::fs::create_dir_all(dir)?;
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("serve.lock"))?;
    let info = dir.join("serve.json");
    match file.try_lock() {
        Ok(()) => {
            // A killed owner's address is stale: nobody listens there any more.
            let _ = std::fs::remove_file(&info);
            Ok(Acquire::Held(ServeLock { _file: file, info }))
        }
        Err(TryLockError::WouldBlock) => {
            let url = std::fs::read(&info)
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| v["url"].as_str().map(str::to_string));
            Ok(Acquire::InUse { url })
        }
        Err(TryLockError::Error(e)) => Err(e),
    }
}

impl ServeLock {
    /// Records where this server listens, for a second `folo serve` on the same data dir to report.
    pub fn publish(&self, url: &str) -> std::io::Result<()> {
        let body = serde_json::json!({ "url": url, "pid": std::process::id() });
        std::fs::write(&self.info, serde_json::to_vec(&body).expect("json"))
    }
}

impl Drop for ServeLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.info);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    #[test]
    fn a_second_server_sees_the_first_and_its_address() {
        let dir = TestDir::new("serve-lock");
        let Acquire::Held(first) = acquire(dir.path()).unwrap() else { panic!("the first takes the lock") };
        assert!(matches!(acquire(dir.path()).unwrap(), Acquire::InUse { url: None }));
        first.publish("http://127.0.0.1:4242").unwrap();
        match acquire(dir.path()).unwrap() {
            Acquire::InUse { url } => assert_eq!(url.as_deref(), Some("http://127.0.0.1:4242")),
            Acquire::Held(_) => panic!("the lock is held"),
        }
        drop(first);
        assert!(!dir.path().join("serve.json").exists());
        assert!(matches!(acquire(dir.path()).unwrap(), Acquire::Held(_)));
    }

    #[test]
    fn a_stale_address_is_dropped_by_the_next_owner() {
        let dir = TestDir::new("serve-lock-stale");
        std::fs::write(dir.path().join("serve.json"), r#"{"url":"http://127.0.0.1:1"}"#).unwrap();
        let Acquire::Held(_lock) = acquire(dir.path()).unwrap() else { panic!("free") };
        assert!(!dir.path().join("serve.json").exists());
    }
}
