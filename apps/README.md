# apps

Native applications built on the FOLO engine. There is one: [desktop/](desktop/README.md), FOLO Desktop, the
native front end, written in Rust on its own UI framework, folo-ui ([desktop/ui/](desktop/ui/README.md)). The web
front end, FOLO Studio, is [web/](../web/README.md); both talk to `folo serve` over [docs/API.md](../docs/API.md).

`apps/desktop` is its own cargo workspace (the app and folo-ui), excluded from the root workspace
(`exclude = ["apps/desktop"]` in the root `Cargo.toml`). The app is a slow build and needs neither the server
nor the CLI to compile, so `cargo build` and `cargo test` at the root leave it out, and its own CI
(`.github/workflows/desktop.yml`) runs when the app, the design tokens or the engine crates it uses change. It
takes the engine crates it needs from `crates/` by path. Build it from `apps/desktop`
([desktop/README.md](desktop/README.md), Workspace, build, test).
