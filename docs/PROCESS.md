# Agent process: implement → PR → review → fix → merge

Every engine change follows this loop. The orchestrator thread starts implementer agents; implementers drive their
own review loop; the orchestrator merges.

## Implementer (Claude Opus 5.5, effort high)
1. You are in your own git worktree. Create the branch from your task (`git switch -c <area>/<topic>`) if you are
   not on it yet.
2. `export CARGO_TARGET_DIR=D:/folo-build/<branch-with-dashes>` for every cargo command (see AGENTS.md).
3. Implement with tests. Before pushing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
   `cargo test --workspace`. All green.
4. Commit (message `area: summary`, ending with the co-author line your harness gives you), `git push -u origin HEAD`,
   then `gh pr create --base main --title "area: summary" --body-file <file>`. The body says what / why / how tested /
   benchmark numbers / follow-ups.
5. **Review loop** (at most 4 rounds):
   - Start a reviewer with the T3 `delegate_task` tool (call `orchestrator_capabilities` once first):
     `target = { providerInstanceId: "codex", model: "gpt-6.1-sol", options: { reasoningEffort: "high", serviceTier: "priority" } }`,
     `role: "review"`, `isolation: "shared"`, `mode: "wait"`, `timeoutMs: 1800000`. Its task: the PR number and URL,
     the absolute path of your checkout, the branch, "review the diff against origin/main for correctness bugs,
     soundness (unsafe), numerical errors, missing tests, API problems and performance traps; do not edit files;
     return findings as a numbered list, each with severity (blocking / should-fix / nit), file:line, the failure
     scenario and a suggested fix; say `NO BLOCKING FINDINGS` explicitly if there are none".
   - If the wait times out, follow the tool's `next` (wait again on that task), don't start a second reviewer.
   - Fix every blocking and should-fix finding yourself (you are the Opus fixer). Nits at your judgement. Push.
   - Each re-review is a **new** `delegate_task` with the previous findings and what you changed.
   - Stop when the reviewer reports no blocking findings.
6. Post the final review summary as a PR comment (`gh pr comment`), including which findings were fixed or declined
   and why.
7. Do **not** merge. Finish with one line: `PR <full URL>, head <SHA>, checks and reviews <state>`.

## Orchestrator
- Watches each delivered PR, merges with `gh pr merge --squash --delete-branch` once CI is green, updates
  `docs/ROADMAP.md`, and starts the next wave. On a merge conflict it asks the implementer (or a new one) to rebase.

## UI threads
`web/` (Studio) and `apps/desktop/` (GPUI) each have a long-running top-level thread that hill-climbs its UI:
build → screenshot every screen and state → two independent design reviews (Claude Opus 5.5 and GPT-6.1 Sol) that
score a fixed rubric 1–10 → fix → repeat until **both reviewers give ≥ 9 on every rubric line**. Each round lands as a
PR through the same review loop above.
