//! The `folo` command-line tool. See `docs/CLI.md`.

fn main() {
    println!("folo {}", env!("CARGO_PKG_VERSION"));
}
