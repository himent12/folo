#!/usr/bin/env bash
# Run a command; on the unstable self-hosted host (docs/CI.md) retry it once, but only when the failure was a crashed
# process and nothing else failed:
# - crashed: the command died of SIGILL, SIGTRAP, SIGBUS or SIGSEGV (exit 132 / 133 / 135 / 139), or a whole line of
#   its output is what the dying tool's runtime writes: cargo's "process didn't exit successfully: ... (exit code:
#   0xc0000005 / 0xc000001d / 0xc0000409" (Windows access violation, illegal instruction, stack-buffer overrun),
#   npm's "npm error signal SIGILL|SIGTRAP|SIGBUS|SIGSEGV", V8's "# Fatal error in" banner. The same words printed
#   inside a line (a test's message) are no crash.
# - and nothing else failed: no failing test, compile or type error, or QA problem in the output (vitest's summary,
#   "test result: FAILED", "error[E...]", "error TS...", ...). When there is one, the step fails as it is, crash or
#   not, so a failing or flaky test can never be retried into a pass. tools/ci-retry-test.sh holds the cases.
#   ATTEMPTS=2 bash tools/ci-retry.sh cargo test --workspace
set -uo pipefail
attempts="${ATTEMPTS:-1}"
log="$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ci-retry.XXXXXX")" || exit 1
trap 'rm -f "$log" "$log.txt"' EXIT
crash="process didn't exit successfully: .*\(exit code: 0xc000(0005|001d|0409)\b"
crash="$crash|^npm (ERR!|error) signal SIG(ILL|TRAP|BUS|SEGV)\$|^# Fatal error in "
failed='^ *(Test Files|Tests|Errors) +[0-9]+ (failed|errors?)\b|^ *FAIL +[^ ]|error TS[0-9]+:'
failed="$failed|^test result: FAILED|^failures:\$|^error\[E[0-9]+\]|^error: aborting due to|due to [0-9]+ previous error"
failed="$failed|^[0-9]+ problem\(s\):|^[0-9]+ state\(s\) not shot:"
crashed() {
  # Without colours and carriage returns, so whole lines match whatever the tool's colour or line-end setting.
  sed -E 's/\x1b\[[0-9;?]*[A-Za-z]//g; s/\r$//' "$log" >"$log.txt"
  grep -qE "$failed" "$log.txt" && return 1
  case "$1" in 132 | 133 | 135 | 139) return 0 ;; esac
  grep -qE "$crash" "$log.txt"
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
