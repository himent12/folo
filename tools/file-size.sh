#!/usr/bin/env bash
# No god files (AGENTS.md, Ground rules): every tracked source file stays under 1000 lines, every line counted.
# Lists the files at or over the limit and fails if there is one. Vendored code (apps/desktop/vendor) is exempt,
# and so is everything that is not source code (docs, data, lock files). CI runs it on every pull request (site.yml).
#
#   bash tools/file-size.sh          the check
#   bash tools/file-size.sh 800      the same with another limit, to see what is getting close
set -euo pipefail
limit="${1:-1000}"
cd "$(git rev-parse --show-toplevel)"
over="$(
  git ls-files -z -- \
    '*.rs' '*.ts' '*.svelte' '*.mjs' '*.js' '*.cjs' '*.py' '*.css' '*.wgsl' '*.sh' '*.ps1' '*.html' \
    ':!apps/desktop/vendor' |
    xargs -0 wc -l |
    awk -v limit="$limit" '!($2 == "total" && NF == 2) && $1 + 0 >= limit + 0' |
    sort -rn
)"
if [ -n "$over" ]; then
  echo "$over"
  echo "these files have $limit lines or more: split them by responsibility (AGENTS.md, Ground rules)" >&2
  exit 1
fi
echo "every source file is under $limit lines"
