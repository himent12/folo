#!/usr/bin/env bash
# Run a command; on the unstable self-hosted host (docs/CI.md) retry it once, but only when the failure was a crashed
# process: a Windows access violation / illegal instruction / stack-buffer overrun code in the output, or on Linux a
# process killed by SIGILL, SIGTRAP, SIGBUS or SIGSEGV (exit 132 / 133 / 135 / 139, npm's "signal SIG..." line, or
# V8's "# Fatal error in" banner). Ordinary failures (compile errors, failing or flaky tests) are never retried.
#   ATTEMPTS=2 bash tools/ci-retry.sh cargo test --workspace
set -uo pipefail
attempts="${ATTEMPTS:-1}"
log="${RUNNER_TEMP:-/tmp}/ci-retry-$$.log"
crash='0xc0000005|0xc000001d|0xc0000409|STATUS_ACCESS_VIOLATION|STATUS_ILLEGAL_INSTRUCTION|STATUS_STACK_BUFFER_OVERRUN'
crash="$crash|^# Fatal error in |signal SIG(ILL|TRAP|BUS|SEGV)\b"
crashed() {
  case "$1" in 132 | 133 | 135 | 139) return 0 ;; esac
  grep -qE "$crash" "$log"
}
for i in $(seq 1 "$attempts"); do
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && exit 0
  if [ "$i" -lt "$attempts" ] && crashed "$rc"; then
    echo "::warning::'$*' crashed (attempt $i, exit $rc); retrying once on the unstable self-hosted host (docs/CI.md)"
    continue
  fi
  exit "$rc"
done
exit "$rc"
