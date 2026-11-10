//! The SSE events of a benchmark's stream (`GET /api/bench/benchmarks/{id}/events`).

use serde::{Deserialize, Serialize};

use crate::{Bench, BenchEntry, BenchLoop};

/// SSE `removed` payload of a benchmark's stream: an entry or a loop was deleted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchRemoved {
    /// The deleted entry's or loop's id.
    pub id: String,
}

/// One event of `GET /api/bench/benchmarks/{id}/events`. On the wire the SSE `event:` field is
/// [`BenchEvent::name`] and `data:` is the JSON of the payload alone.
#[derive(Clone, Debug, PartialEq)]
pub enum BenchEvent {
    /// `bench`: the benchmark after a change (name, plan, entry count, best entry).
    Bench(Box<Bench>),
    /// `entry`: an entry after a change (created, progress, done, failed).
    Entry(Box<BenchEntry>),
    /// `loop`: a loop after a change.
    Loop(Box<BenchLoop>),
    /// `removed`: an entry or a loop was deleted.
    Removed(BenchRemoved),
}

impl BenchEvent {
    /// The SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            BenchEvent::Bench(_) => "bench",
            BenchEvent::Entry(_) => "entry",
            BenchEvent::Loop(_) => "loop",
            BenchEvent::Removed(_) => "removed",
        }
    }

    /// The JSON `data:` payload.
    pub fn data_json(&self) -> String {
        let r = match self {
            BenchEvent::Bench(p) => serde_json::to_string(p),
            BenchEvent::Entry(p) => serde_json::to_string(p),
            BenchEvent::Loop(p) => serde_json::to_string(p),
            BenchEvent::Removed(p) => serde_json::to_string(p),
        };
        r.expect("event payloads always serialize")
    }

    /// Parses an SSE event from its `event:` name and `data:` JSON. Returns `None` for unknown event names (new
    /// event types may be added; clients should skip them) and an error for malformed data.
    pub fn from_sse(name: &str, data: &str) -> Option<Result<BenchEvent, serde_json::Error>> {
        Some(match name {
            "bench" => serde_json::from_str(data).map(BenchEvent::Bench),
            "entry" => serde_json::from_str(data).map(BenchEvent::Entry),
            "loop" => serde_json::from_str(data).map(BenchEvent::Loop),
            "removed" => serde_json::from_str(data).map(BenchEvent::Removed),
            _ => return None,
        })
    }
}
