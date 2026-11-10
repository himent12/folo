//! Agent labeling end to end over HTTP (docs/API.md, Agent labeling), with scripted agents in place of the
//! Claude Code and Codex CLIs: jobs, uploads and folder imports, runs and what they do when calls fail, stopping,
//! hand edits, events, and the dataset a job becomes.

mod failures;
mod harness;
mod jobs;
mod runs;
