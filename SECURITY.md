# Security policy

## Reporting a vulnerability

Please report it privately, not in a public issue: on GitHub, open the repository's **Security** tab and choose
**Report a vulnerability** (GitHub's private vulnerability reporting). Say what is affected, how to reproduce it and
what an attacker gains. You will get an answer there, and credit in the fix's release notes if you want it.

Fixes go into `main` and the next release; only the latest release is supported.

## What to keep in mind when running FOLO

FOLO is a local tool first, and its defaults assume that:

- **`folo serve` listens on loopback** (`127.0.0.1`) and, there, answers only requests whose `Host` is a loopback
  name, which stops a web page from reaching it through DNS rebinding. It has no user accounts. A server bound beyond
  loopback (`--host 0.0.0.0`, a LAN address, a cloud worker) should be started with `--token`: without one FOLO
  serves anyway and only logs a warning. Its traffic is plain HTTP unless you put TLS in front of it
  ([docs/API.md, Conventions](docs/API.md#conventions), [docs/CLOUD.md](docs/CLOUD.md)).
- **The server reads and writes files** for the person running it: it registers dataset folders by path, writes runs
  and models under its data dir and exports files. Anyone who can reach its API can do the same, so treat access to
  the port as access to that account's files.
- **`folo mcp` acts for the agent that drives it.** `--read-only` leaves out every tool that changes state and
  `--confine` keeps file reads inside the data dir ([docs/MCP.md](docs/MCP.md)).
- **Agent labeling and agent benchmark entries start `claude` or `codex` on the server's machine**, with tools,
  configuration files and sessions switched off, and send them your images ([docs/LABELING.md](docs/LABELING.md)).
- **Cloud runs spend money** on the provider account whose key you connect. The key is stored in the data dir
  (mode 0600 on Unix; on Windows under the folder's own permissions, which for the default `~/.folo` are the user's
  profile's) and is never returned by the API ([docs/CLOUD.md](docs/CLOUD.md)).
- **`.folo` files are data**: a JSON header and raw tensors in a safetensors-compatible layout, with no code in them
  ([docs/FORMAT.md](docs/FORMAT.md)). Images and label files from an untrusted source go through the image decoders
  and the parsers; a crash or a hang on a crafted file is a bug worth reporting.

A report is in scope when it breaks one of these: a request that gets past the `Host`, `Origin` or token checks, a
path that escapes `--confine` or the data dir where one is promised, a file that makes FOLO execute code or write
outside the folders it was given, a secret that reaches a log or an API response.
