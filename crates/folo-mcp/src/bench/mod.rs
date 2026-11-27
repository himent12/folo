//! Vision benchmarks for the MCP tools and `folo bench` (docs/BENCH.md; docs/MCP.md and docs/CLI.md, Vision
//! benchmarks).
//!
//! [`Benchmarks`] reaches a data dir's benchmarks with one set of calls: in process (a store's
//! [`Benches`](folo_server::bench::Benches), as `folo mcp` embeds it and `folo bench` opens it), or through a running
//! `folo serve` (`/api/bench`, docs/API.md). This file holds those calls, one per endpoint; `ops` builds the
//! operations both front ends share on them (the compact results of [`crate::types::bench`], following an entry,
//! sending a file of external predictions), and `hints` words each front end's hints.

pub mod hints;
mod ops;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use folo_api::{
    Bench, BenchCatalog, BenchEntry, BenchEntryList, BenchList, BenchPrediction, BenchPredictions,
    BenchResult, CreateBench, StartBenchEntry,
};
use folo_client::encode;
use folo_server::Store;
use folo_server::bench::Benches;
use folo_server::store::FEATURE_BENCH_UPLOAD_SAFE;
use http::StatusCode;

pub use ops::{Contestant, MAX_PART_BOXES, read_predictions};

use crate::errors::Error;
use crate::proxy::{HttpStore, store_error};

/// Where benchmark calls go. Cheap to clone.
#[derive(Clone)]
pub enum Benchmarks {
    /// A store's benchmarks, in this process.
    Store(Arc<dyn Store>),
    /// A running `folo serve`.
    Http(Arc<HttpStore>),
}

/// One call's way: a store's benchmarks here, or a server.
enum Via<'a> {
    Here(&'a Benches, &'a dyn Store),
    Http(&'a HttpStore),
}

impl Benchmarks {
    fn via(&self) -> Result<Via<'_>, Error> {
        Ok(match self {
            Benchmarks::Store(s) => Via::Here(s.benches()?, &**s),
            Benchmarks::Http(h) => {
                if !h.has_feature(folo_api::FEATURE_VISION_BENCH)? {
                    let message = format!("the folo serve at {} has no vision benchmarks", h.base());
                    return Err(Error::new(StatusCode::NOT_IMPLEMENTED, "not_implemented", message)
                        .hint("a `folo serve --mock` has none, nor one from before them: use a real server of this version"));
                }
                Via::Http(h)
            }
        })
    }

    /// These calls with every request to a server ending by `deadline` (in process, the same calls).
    pub fn until(&self, deadline: Instant) -> Benchmarks {
        match self {
            Benchmarks::Http(h) => Benchmarks::Http(Arc::new(h.until(deadline))),
            here => here.clone(),
        }
    }

    /// Fails as every call does where there are no benchmarks (the mock, a server without `vision_bench`): for a
    /// command to check before it changes anything else.
    pub fn check(&self) -> Result<(), Error> {
        self.via().map(|_| ())
    }

    /// `GET /api/bench/catalog`.
    pub fn catalog(&self) -> Result<BenchCatalog, Error> {
        match self.via()? {
            Via::Http(h) => get(h, "/bench/catalog"),
            Via::Here(b, _) => Ok(b.catalog()),
        }
    }

    /// `GET /api/bench/benchmarks`: newest first.
    pub fn list(&self) -> Result<Vec<Bench>, Error> {
        match self.via()? {
            Via::Http(h) => Ok(get::<BenchList>(h, "/bench/benchmarks")?.benchmarks),
            Via::Here(b, _) => Ok(b.list()),
        }
    }

    /// `GET /api/bench/benchmarks/{id}`.
    pub fn get(&self, id: &str) -> Result<Bench, Error> {
        match self.via()? {
            Via::Http(h) => get(h, &bench_path(id, "")?),
            Via::Here(b, _) => b.get(id),
        }
    }

    /// `POST /api/bench/benchmarks`.
    pub fn create(&self, req: CreateBench) -> Result<Bench, Error> {
        match self.via()? {
            Via::Http(h) => h.client().post("/bench/benchmarks", &req).map_err(store_error),
            Via::Here(b, s) => b.create(s, req),
        }
    }

    /// `DELETE /api/bench/benchmarks/{id}`.
    pub fn delete(&self, id: &str) -> Result<(), Error> {
        match self.via()? {
            Via::Http(h) => h.client().delete(&bench_path(id, "")?).map_err(store_error),
            Via::Here(b, _) => b.delete(id),
        }
    }

    /// `GET /api/bench/benchmarks/{id}/entries`: the leaderboard.
    pub fn entries(&self, id: &str) -> Result<Vec<BenchEntry>, Error> {
        match self.via()? {
            Via::Http(h) => Ok(get::<BenchEntryList>(h, &bench_path(id, "/entries")?)?.entries),
            Via::Here(b, _) => b.entries(id),
        }
    }

    /// `GET …/entries/{entry_id}`.
    pub fn entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => get(h, &entry_path(id, entry_id, "")?),
            Via::Here(b, _) => b.entry(id, entry_id),
        }
    }

    /// Whether this backend supports client-chosen `entry_id` and conditional `?only_if_open=true[&delete=true]` stops.
    pub(super) fn supports_upload_safe(&self) -> bool {
        match self {
            Benchmarks::Store(s) => s.benches().is_ok(),
            Benchmarks::Http(h) => h.has_feature(FEATURE_BENCH_UPLOAD_SAFE).unwrap_or(false),
        }
    }

    /// `POST /api/bench/benchmarks/{id}/entries`.
    pub fn start_entry(&self, id: &str, req: StartBenchEntry) -> Result<BenchEntry, Error> {
        self.start_entry_with_id(id, req, None)
    }

    /// `POST /api/bench/benchmarks/{id}/entries[?entry_id=<entry_id>]`.
    pub(super) fn start_entry_with_id(
        &self,
        id: &str,
        req: StartBenchEntry,
        entry_id: Option<&str>,
    ) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => {
                let path = match entry_id {
                    Some(eid) if h.has_feature(FEATURE_BENCH_UPLOAD_SAFE)? => {
                        format!("{}?entry_id={}", bench_path(id, "/entries")?, encode(eid))
                    }
                    _ => bench_path(id, "/entries")?,
                };
                h.client().post(&path, &req).map_err(store_error)
            }
            Via::Here(b, s) => b.start_entry_with_id(s, id, req, entry_id.map(str::to_string)),
        }
    }

    /// `POST …/entries/{entry_id}/predictions`: a part of an open external entry.
    pub fn add_predictions(
        &self,
        id: &str,
        entry_id: &str,
        predictions: Vec<BenchPrediction>,
    ) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => {
                let body = BenchPredictions { predictions };
                h.client().post(&entry_path(id, entry_id, "/predictions")?, &body).map_err(store_error)
            }
            Via::Here(b, _) => b.add_predictions(id, entry_id, predictions),
        }
    }

    /// `POST …/entries/{entry_id}/finish`: scores an open external entry.
    pub fn finish(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => h.client().post_empty(&entry_path(id, entry_id, "/finish")?).map_err(store_error),
            Via::Here(b, _) => b.finish_open(id, entry_id),
        }
    }

    /// `GET …/entries/{entry_id}/result`.
    pub fn result(&self, id: &str, entry_id: &str) -> Result<BenchResult, Error> {
        match self.via()? {
            Via::Http(h) => get(h, &entry_path(id, entry_id, "/result")?),
            Via::Here(b, _) => b.result(id, entry_id),
        }
    }

    /// `POST …/entries/{entry_id}/stop`.
    pub fn stop_entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => h.client().post_empty(&entry_path(id, entry_id, "/stop")?).map_err(store_error),
            Via::Here(b, _) => b.stop_entry(id, entry_id),
        }
    }

    /// `POST …/entries/{entry_id}/stop?only_if_open=true&delete=true`: atomically stops and deletes an external
    /// entry only while it is still open and collecting parts (requires `bench_upload_safe`).
    pub(super) fn discard_open_entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => h
                .client()
                .post_empty(&entry_path(id, entry_id, "/stop?only_if_open=true&delete=true")?)
                .map_err(store_error),
            Via::Here(b, _) => b.discard_open_entry(id, entry_id),
        }
    }

    /// `DELETE …/entries/{entry_id}`.
    pub fn delete_entry(&self, id: &str, entry_id: &str) -> Result<(), Error> {
        match self.via()? {
            Via::Http(h) => h.client().delete(&entry_path(id, entry_id, "")?).map_err(store_error),
            Via::Here(b, _) => b.delete_entry(id, entry_id),
        }
    }
}

fn get<T: serde::de::DeserializeOwned>(h: &HttpStore, path: &str) -> Result<T, Error> {
    h.client().get(path).map_err(store_error)
}

/// Generates a unique client-chosen benchmark entry id (`be_` + 16 lowercase hex chars) so a lost response to the
/// request that opens an external entry still lets the client look up and clean up that exact entry.
pub(super) fn fresh_entry_id(bench_id: &str, name: &str) -> String {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let pid = std::process::id() as u64;
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01B3);
        }
    };
    mix(&seq.to_le_bytes());
    mix(&nanos.to_le_bytes());
    mix(&pid.to_le_bytes());
    mix(bench_id.as_bytes());
    mix(name.as_bytes());
    // SplitMix64 finalizer for uniform hex distribution across rapid sequential calls.
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    format!("be_{h:016x}")
}

/// `/bench/benchmarks/{id}{rest}`.
fn bench_path(id: &str, rest: &str) -> Result<String, Error> {
    if id.is_empty() || matches!(id, "." | "..") {
        return Err(Error::new(
            http::StatusCode::NOT_FOUND,
            "bench_not_found",
            format!("benchmark '{id}' does not exist"),
        ));
    }
    Ok(format!("/bench/benchmarks/{}{rest}", encode(id)))
}

/// `/bench/benchmarks/{id}/entries/{entry_id}{rest}`.
fn entry_path(id: &str, entry_id: &str, rest: &str) -> Result<String, Error> {
    let base = bench_path(id, "")?;
    if entry_id.is_empty() || matches!(entry_id, "." | "..") {
        return Err(Error::new(
            http::StatusCode::NOT_FOUND,
            "bench_entry_not_found",
            format!("benchmark '{id}' has no entry '{entry_id}'"),
        ));
    }
    Ok(format!("{base}/entries/{}{rest}", encode(entry_id)))
}
