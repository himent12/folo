#!/usr/bin/env bash
# Type-checks folo-ui (the winit backend and the cosmic-text text backend, which are always on off
# Windows) for Linux and macOS from any host, and lints it there with clippy. No C toolchain is needed:
# every dependency of these targets builds its scripts for the host and binds the system libraries at run
# time (winit's x11 / wayland dlopen, x11rb, zbus for the portal, AppKit through objc2).
#
#   bash apps/desktop/tools/cross-check.sh            # checks + clippy for all targets
#   CROSS_TARGETS="aarch64-apple-darwin" bash ...     # a subset
#   CROSS_NO_CLIPPY=1 bash ...                        # checks only
#
# This is a compile check: nothing gated on Linux or macOS runs. Set CARGO_TARGET_DIR off C: on the dev
# machine (e.g. D:/folo-build/<dir>).
set -euo pipefail

cd "$(dirname "$0")/.."

targets=${CROSS_TARGETS:-"x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-apple-darwin"}

installed=$(rustup target list --installed)
for target in $targets; do
  if ! grep -qx "$target" <<<"$installed"; then
    rustup target add "$target"
  fi
done

status=0
for target in $targets; do
  echo "== cargo check -p folo-ui --target $target"
  cargo check --locked -p folo-ui --all-targets --target "$target" || status=1
  if [[ -z "${CROSS_NO_CLIPPY:-}" ]]; then
    echo "== cargo clippy -p folo-ui --all-targets --target $target -- -D warnings"
    cargo clippy --locked -p folo-ui --all-targets --target "$target" -- -D warnings || status=1
  fi
done

if [[ $status -ne 0 ]]; then
  echo "cross-check: FAILED" >&2
else
  echo "cross-check: ok ($targets)"
fi
exit $status
