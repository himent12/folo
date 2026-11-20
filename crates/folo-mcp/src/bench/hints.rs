//! The hints of benchmark errors in each front end's words. The server's hints name HTTP requests; [`decorate`]
//! replaces them by code, naming the MCP tools or the `folo bench` commands that help.

use crate::errors::Error;
pub use crate::label::hints::Front;

/// `e` with a hint in `front`'s words (an error whose code has nothing to add keeps its hint).
pub fn decorate(mut e: Error, front: Front) -> Error {
    let mcp = front == Front::Mcp;
    let hint = match e.body.code.as_str() {
        "bench_not_found" => {
            Some(if mcp { "list_benchmarks lists them" } else { "`folo bench list` lists the benchmarks" })
        }
        "bench_entry_not_found" => Some(if mcp {
            "get_benchmark lists its entries"
        } else {
            "`folo bench show BENCH` lists its entries"
        }),
        "bench_active" => Some(if mcp {
            "wait for its entries to end (bench_entry), or stop them (stop_bench_entry)"
        } else {
            "wait for its entries to end (`folo bench status`), or stop them (`folo bench stop`)"
        }),
        "bench_entry_not_active" => Some(if mcp {
            "bench_entry says how it ended"
        } else {
            "`folo bench status BENCH ENTRY` says how it ended"
        }),
        "model_not_found" => Some(if mcp {
            "list_models lists the models; published weights are models once pulled (`folo weights pull NAME`)"
        } else {
            "`folo weights pull NAME` makes published weights a model; the models are those Studio lists"
        }),
        // Only when the error has none: a failed submit's hint says what became of its entry.
        "invalid_arguments" | "invalid_body" if !mcp && e.body.hint.is_none() => {
            Some("see `folo bench <command> --help`")
        }
        _ => None,
    };
    if let Some(h) = hint {
        e.body.hint = Some(h.to_string());
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_name_the_front_ends_own_commands() {
        let e =
            || Error::conflict("bench_active", "benchmark 'bn_1' has an active entry").hint("POST /api/…");
        assert!(decorate(e(), Front::Mcp).body.hint.unwrap().contains("stop_bench_entry"));
        assert!(decorate(e(), Front::Cli).body.hint.unwrap().contains("folo bench stop"));
        let other = Error::conflict("bench_frozen", "x").hint("kept");
        assert_eq!(decorate(other, Front::Cli).body.hint.unwrap(), "kept");
    }
}
