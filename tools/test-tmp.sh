#!/usr/bin/env bash
# Run a test command with its temp folders pointed at one fresh folder, then fail if the run left `folo-*` entries
# there (docs/CI.md, Test temp dirs). FOLO_TEST_TMP (where folo_test_support::TestDir puts test folders) and the
# system temp dir (TMPDIR, TMP, TEMP) all point at that folder. So the check also catches a test that bypasses
# TestDir, and a `folo` binary that writes to the temp dir.
#   bash tools/test-tmp.sh cargo test --workspace
#   TEST_TMP=D:/folo-build/test-tmp bash tools/test-tmp.sh cargo test -p folo-cli
# The folder is a new `test-tmp.XXXXXX` (mktemp) under $TEST_TMP, else $RUNNER_TEMP (CI), else $TMPDIR, so concurrent
# runs never share one. It is removed afterwards when nothing was left behind.
set -uo pipefail
parent="${TEST_TMP:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}}"
parent="${parent//\\//}"
mkdir -p "$parent" || exit 1
# Absolute: cargo runs each test binary in its own package folder, where a relative path would point elsewhere.
parent="$(cd "$parent" && pwd)" || exit 1
dir="$(mktemp -d "$parent/test-tmp.XXXXXX")" || exit 1
# The native form for the processes under test (on Windows, cygpath -w turns /d/a/b into D:\a\b).
native="$dir"
command -v cygpath >/dev/null && native="$(cygpath -w "$dir")"
FOLO_TEST_TMP="$native" TMPDIR="$native" TMP="$native" TEMP="$native" "$@"
rc=$?
shopt -s nullglob dotglob
left=("$dir"/folo-*)
if [ "${#left[@]}" -gt 0 ]; then
  echo "::error::the test run left ${#left[@]} folo-* entries in $dir; tests must remove what they create (folo_test_support::TestDir, docs/CI.md):"
  for p in "${left[@]}"; do echo "  ${p##*/}"; done
  [ "$rc" -eq 0 ] && rc=1
  exit "$rc"
fi
other=("$dir"/*)
if [ "${#other[@]}" -gt 0 ]; then
  echo "note: ${#other[@]} other entries in $dir (not FOLO's):"
  for p in "${other[@]:0:20}"; do echo "  ${p##*/}"; done
fi
echo "test temp dir $dir: no folo-* leftovers"
rm -rf "$dir"
exit "$rc"
