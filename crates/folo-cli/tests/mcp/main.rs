//! `folo mcp` end to end (docs/MCP.md "Tests"): spawns the binary and speaks raw JSON-RPC over stdio, for each
//! supported revision. Every line the server writes to stdout must be a JSON-RPC 2.0 message.

mod confine;
mod harness;
mod protocol;
mod proxy;
mod real_store;
