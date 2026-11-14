//! The `folo` binary's command tree as a library: the binary (`src/main.rs`) and the tests that parse commands with
//! the real clap parser (tests/studio_cli.rs, tests/cli_values.rs) share it.
//!
//! Only the parts those tests need live here: [`cli`] (the clap definitions, which `folo describe` serialises),
//! [`VERSION`] and [`WEIGHTS_REPO`]. Every command's implementation is a private module of the binary; its
//! `src/main.rs` doc lists them, and `crates/folo-cli/README.md` maps each subcommand to its file.

pub mod cli;

/// `folo --version`: the workspace version, the commit and the build date (see build.rs).
pub const VERSION: &str =
    concat!(env!("CARGO_PKG_VERSION"), " (", env!("FOLO_GIT_SHA"), " ", env!("FOLO_BUILD_DATE"), ")");

/// The GitHub repository whose releases carry the published weights (`folo weights pull --repo`'s default).
pub const WEIGHTS_REPO: &str = "himent12/folo";
