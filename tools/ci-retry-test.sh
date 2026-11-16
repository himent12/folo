#!/usr/bin/env bash
# The cases of tools/ci-retry.sh: a crashed tool is retried once; a failing test, a type or compile error or a QA
# problem never is, whatever crash words its output prints. Runs a fake command that plays one script per attempt.
#   bash tools/ci-retry-test.sh        (CI: web.yml, job check)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ci-retry-test.XXXXXX")" || exit 1
trap 'rm -rf "$dir"' EXIT
fails=0
summary=$'\n Test Files  1 failed | 3 passed (4)\n      Tests  1 failed | 20 passed (21)'
# expect <name> <exit> <runs> <attempt 1> [<attempt 2>]: run ci-retry.sh (ATTEMPTS, default 2) on the fake command.
expect() {
  local name="$1" want_rc="$2" want_runs="$3"
  printf '%s\n' "$4" >"$dir/a1"
  printf '%s\n' "${5:-echo ok; exit 0}" >"$dir/a2"
  echo 0 >"$dir/n"
  ATTEMPTS="${ATTEMPTS:-2}" bash "$here/ci-retry.sh" \
    bash -c 'n=$(($(cat "$0/n") + 1)); echo "$n" >"$0/n"; . "$0/a$n"' "$dir" >"$dir/out" 2>&1
  local rc=$? runs
  runs="$(cat "$dir/n")"
  if [ "$rc" -eq "$want_rc" ] && [ "$runs" -eq "$want_runs" ]; then
    echo "ok   $name"
  else
    echo "FAIL $name: exit $rc after $runs run(s), expected exit $want_rc after $want_runs"
    sed 's/^/     | /' "$dir/out"
    fails=$((fails + 1))
  fi
}
p() { printf 'printf %q; exit %s' "$1" "$2"; }

# A crashed tool: retried once, the second attempt's exit code is the step's.
expect 'killed by SIGTRAP (exit 133)' 0 2 'exit 133'
expect 'killed by SIGSEGV (exit 139)' 0 2 'exit 139'
expect "npm's signal line" 0 2 "$(p $'> vitest run\nnpm error command sh -c vitest run\nnpm error signal SIGTRAP\n' 1)"
expect "npm's signal line, coloured, CRLF" 0 2 "$(p $'\e[31mnpm error\e[39m signal SIGILL\r\n' 1)"
expect "V8's banner" 0 2 "$(p $'\n#\n# Fatal error in , line 0\n# unreachable code\n#\n' 1)"
expect 'rustc access violation' 0 2 "$(p $'error: could not compile `x` (lib)\n\nCaused by:\n  process didn\'t exit successfully: `rustc -` (exit code: 0xc0000005, STATUS_ACCESS_VIOLATION)\n' 101)"
expect 'a crash on the last attempt fails' 133 2 'exit 133' 'exit 133'
ATTEMPTS=1 expect 'ATTEMPTS=1 never retries' 133 1 'exit 133'

# An ordinary failure: never retried, its exit code is the step's.
expect 'a failing test' 1 1 "$(p "$summary" 1)"
expect 'any exit code passes through' 3 1 'exit 3'
expect "a test's message names a signal" 1 1 "$(p $'AssertionError: expected signal SIGSEGV to be undefined\nsignal SIGSEGV\n' 1)"
expect 'a test prints a mid-line crash code' 1 1 "$(p $'got (exit code: 0xc0000005, STATUS_ACCESS_VIOLATION)\n' 1)"
expect "a failing test prints npm's signal line" 1 1 "$(p $'npm error signal SIGSEGV'"$summary"$'\n' 1)"
expect "a failing test prints V8's banner" 1 1 "$(p $'# Fatal error in , line 0'"$summary"$'\n' 1)"
expect 'a failing test prints a cargo crash line' 101 1 \
  "$(p $'  process didn\'t exit successfully: `t` (exit code: 0xc0000005, STATUS_ACCESS_VIOLATION)\ntest result: FAILED. 1 passed; 1 failed\n' 101)"
expect 'coloured vitest summary' 1 1 "$(p $'npm error signal SIGTRAP\n      Tests  \e[1m\e[31m1 failed\e[39m\e[22m | 2 passed (3)\n' 1)"
expect 'a type error' 2 1 "$(p $'src/a.ts(1,7): error TS2322: Type\n# Fatal error in , line 0\n' 2)"
expect 'a compile error' 101 1 "$(p $'error[E0308]: mismatched types\n  process didn\'t exit successfully: `rustc -` (exit code: 0xc0000005, STATUS_ACCESS_VIOLATION)\n' 101)"
expect 'a QA problem' 1 1 "$(p $'1 problem(s):\n  runs @ phone: overflow\nnpm error signal SIGSEGV\n' 1)"
expect 'killed by a signal after a failing test' 133 1 "$(p "$summary" 133)"

[ "$fails" -eq 0 ] && echo "ci-retry: all cases pass" && exit 0
echo "ci-retry: $fails case(s) failed"
exit 1
