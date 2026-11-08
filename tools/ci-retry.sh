#!/usr/bin/env bash
# Run a command; on the unstable self-hosted Windows host (docs/CI.md) retry it once, but only when the failure was
# a crashed process (Windows access violation / illegal instruction / stack-buffer overrun exit codes in the output).
# Ordinary failures (compile errors, failing or flaky tests) are never retried.
#   ATTEMPTS=2 bash tools/ci-retry.sh cargo test --workspace
set -uo pipefail
attempts="${ATTEMPTS:-1}"
log="${RUNNER_TEMP:-/tmp}/ci-retry-$$.log"
crash='0xc0000005|0xc000001d|0xc0000409|STATUS_ACCESS_VIOLATION|STATUS_ILLEGAL_INSTRUCTION|STATUS_STACK_BUFFER_OVERRUN'
for i in $(seq 1 "$attempts"); do
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && exit 0
  if [ "$i" -lt "$attempts" ] && grep -qE "$crash" "$log"; then
    echo "::warning::'$*' crashed (attempt $i, exit $rc); retrying once on the unstable self-hosted host (docs/CI.md)"
    continue
  fi
  exit "$rc"
done
exit "$rc"
