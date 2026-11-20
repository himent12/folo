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

use folo_api::{
    Bench, BenchCatalog, BenchEntry, BenchEntryList, BenchList, BenchPrediction, BenchPredictions,
    BenchResult, CreateBench, StartBenchEntry,
};
use folo_client::encode;
use folo_server::Store;
use folo_server::bench::Benches;
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
            Via::Http(h) => get(h, &bench_path(id, "")),
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
            Via::Http(h) => h.client().delete(&bench_path(id, "")).map_err(store_error),
            Via::Here(b, _) => b.delete(id),
        }
    }

    /// `GET /api/bench/benchmarks/{id}/entries`: the leaderboard.
    pub fn entries(&self, id: &str) -> Result<Vec<BenchEntry>, Error> {
        match self.via()? {
            Via::Http(h) => Ok(get::<BenchEntryList>(h, &bench_path(id, "/entries"))?.entries),
            Via::Here(b, _) => b.entries(id),
        }
    }

    /// `GET …/entries/{entry_id}`.
    pub fn entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => get(h, &entry_path(id, entry_id, "")),
            Via::Here(b, _) => b.entry(id, entry_id),
        }
    }

    /// `POST /api/bench/benchmarks/{id}/entries`.
    pub fn start_entry(&self, id: &str, req: StartBenchEntry) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => h.client().post(&bench_path(id, "/entries"), &req).map_err(store_error),
            Via::Here(b, s) => b.start_entry(s, id, req),
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
                h.client().post(&entry_path(id, entry_id, "/predictions"), &body).map_err(store_error)
            }
            Via::Here(b, _) => b.add_predictions(id, entry_id, predictions),
        }
    }

    /// `POST …/entries/{entry_id}/finish`: scores an open external entry.
    pub fn finish(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => h.client().post_empty(&entry_path(id, entry_id, "/finish")).map_err(store_error),
            Via::Here(b, _) => b.finish_open(id, entry_id),
        }
    }

    /// `GET …/entries/{entry_id}/result`.
    pub fn result(&self, id: &str, entry_id: &str) -> Result<BenchResult, Error> {
        match self.via()? {
            Via::Http(h) => get(h, &entry_path(id, entry_id, "/result")),
            Via::Here(b, _) => b.result(id, entry_id),
        }
    }

    /// `POST …/entries/{entry_id}/stop`.
    pub fn stop_entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry, Error> {
        match self.via()? {
            Via::Http(h) => h.client().post_empty(&entry_path(id, entry_id, "/stop")).map_err(store_error),
            Via::Here(b, _) => b.stop_entry(id, entry_id),
        }
    }

    /// `DELETE …/entries/{entry_id}`.
    pub fn delete_entry(&self, id: &str, entry_id: &str) -> Result<(), Error> {
        match self.via()? {
            Via::Http(h) => h.client().delete(&entry_path(id, entry_id, "")).map_err(store_error),
            Via::Here(b, _) => b.delete_entry(id, entry_id),
        }
    }
}

fn get<T: serde::de::DeserializeOwned>(h: &HttpStore, path: &str) -> Result<T, Error> {
    h.client().get(path).map_err(store_error)
}

/// `/bench/benchmarks/{id}{rest}`.
fn bench_path(id: &str, rest: &str) -> String {
    format!("/bench/benchmarks/{}{rest}", encode(id))
}

/// `/bench/benchmarks/{id}/entries/{entry_id}{rest}`.
fn entry_path(id: &str, entry_id: &str, rest: &str) -> String {
    format!("/bench/benchmarks/{}/entries/{}{rest}", encode(id), encode(entry_id))
}
