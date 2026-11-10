//! The cloud control plane end to end (docs/CLOUD.md) against in-process workers: a test provider whose `launch`
//! starts a real store behind the launch's token on a loopback port, with every provider call recorded and any of
//! them scriptable (fail, panic, lose its answer, hold until released). Tiny synthetic datasets, one- to three-epoch
//! runs on the CPU, and short timings, so every lifecycle path runs in seconds: whole runs, stops in every phase,
//! resumes (on a new instance and here), refused and lost launches, boot failures, a worker that dies, terminations
//! the provider refuses, crash recovery, the deadline, a `remote` connection, and `/api/cloud` over HTTP.

mod failures;
mod harness;
mod lifecycle;
mod lost_instances;
mod remote;
mod routes;
