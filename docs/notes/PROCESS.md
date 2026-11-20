# Agent process: orchestrate → implement → play-test & review → write → merge

How the maintainers' agents collaborate on FOLO. It is specific to their setup (T3 Code orchestration tools, a Windows
machine with build output on `D:`, and the four subagent roles below); an external contributor needs only
[AGENTS.md](../../AGENTS.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Subagent roster

Call `orchestrator_capabilities` once per parent thread before the first `delegate_task`. Every role runs as a
dedicated subagent with a clear boundary of responsibility:

| Role | Model | T3 Code `delegate_task` `target` |
|---|---|---|
| **Main orchestrator** | Claude Opus 5.5 or Claude Fable 5.2 (`effort: high`) | `{ providerInstanceId: "claudeAgent", model: "claude-opus-5-5", options: { effort: "high" } }` or `{ providerInstanceId: "claudeAgent", model: "claude-fable-5-2", options: { effort: "high" } }` |
| **Main code implementer & doer** | Claude Opus 5.5 (`effort: high`) | `{ providerInstanceId: "claudeAgent", model: "claude-opus-5-5", options: { effort: "high" } }` |
| **App play-tester & PR reviewer** | GPT-6.2 Sol (`high`, **fast**) or GPT-6.2 Astra (`high`, **standard — not fast**) | Fast: `{ providerInstanceId: "codex", model: "gpt-6.2-sol", options: { reasoningEffort: "high", serviceTier: "priority" } }` <br> Standard (not fast): `{ providerInstanceId: "codex", model: "gpt-6.2-astra", options: { reasoningEffort: "high", serviceTier: "default" } }` |
| **Writer & creative writer** | Gemini 4 Neon (`thought_level: high`) via Devin CLI in T3 Code | `{ providerInstanceId: "acpRegistry_devin", model: "gemini-4-neon-high", options: { thought_level: "high" } }` |

Keep `serviceTier: "default"` (standard, never `"priority"` / fast) when running **GPT-6.2 Astra**; use
`serviceTier: "priority"` (fast) only for **GPT-6.2 Sol**.

## 1. Main orchestrator (Claude Opus 5.5 or Claude Fable 5.2)

- Plans the wave, breaks work into focused, independent PRs, and dispatches subagents with disjoint worktrees
  (`isolation: "worktree"` for independent work).
- Dispatches **Claude Opus 5.5** as the main implementer/doer for code changes, bug fixes, and refactors.
- Dispatches **GPT-6.2 Sol (high, fast)** or **GPT-6.2 Astra (high, standard — not fast)** to play-test the real app
  (Studio, Desktop, CLI, MCP) and to review PRs.
- Dispatches **Gemini 4 Neon** (via Devin CLI in T3 Code) for documentation rewrites, README storytelling, release
  notes, guides, and creative/UX writing.
- Watches each delivered PR, merges with `gh pr merge --squash --delete-branch` once CI is green and review/play-test
  sign-off is clean, updates `docs/ROADMAP.md`, and starts the next wave. On a merge conflict, asks the implementer
  (or a fresh Opus 5.5 subagent) to rebase.

## 2. Main code implementer & doer (Claude Opus 5.5, effort high)

1. Work in your own git worktree. Create the branch for your task (`git switch -c <area>/<topic>`) if you are not on
   it yet.
2. `export CARGO_TARGET_DIR=D:/folo-build/<branch-with-dashes>` and `export FOLO_TEST_TMP=D:/folo-build/test-tmp` for
   every cargo command (see [AGENTS.md](../../AGENTS.md)).
3. Implement with tests. Before pushing, verify every relevant check is green:
   - Rust workspace: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`
   - File sizes: `bash tools/file-size.sh`
   - Studio (`web/`): `npm run check && npm test`
   - Desktop (`apps/desktop/`): `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
     `cargo test --workspace`
   - Docs / Markdown: `(cd site && npm run check && npm run build)`
4. Commit (`area: summary`, with your harness co-author trailer), `git push -u origin HEAD`, then
   `gh pr create --base main --title "area: summary" --body-file <file>`. The body states what changed, why, how it
   was tested, benchmark numbers if hot paths changed, and any follow-ups.
5. **Review & play-test loop** (at most 4 rounds):
   - Start a fresh reviewer / play-tester subagent via `delegate_task` using either:
     - **GPT-6.2 Sol (high, fast)**:
       `target = { providerInstanceId: "codex", model: "gpt-6.2-sol", options: { reasoningEffort: "high", serviceTier: "priority" } }`
     - **GPT-6.2 Astra (high, standard — not fast)**:
       `target = { providerInstanceId: "codex", model: "gpt-6.2-astra", options: { reasoningEffort: "high", serviceTier: "default" } }`
   - Pass `role: "review"`, `isolation: "shared"`, `mode: "wait"`, `timeoutMs: 1800000`, the PR number and URL, the
     absolute path of your checkout, and the branch.
   - Ask it to review the diff against `origin/main` for correctness bugs, soundness (`unsafe`), numerical errors,
     missing tests, API problems, and performance traps without editing files, and to return findings as a numbered
     list with severity (`blocking` / `should-fix` / `nit`), `file:line`, failure scenario, and suggested fix — or
     state `NO BLOCKING FINDINGS` explicitly if there are none.
   - If the wait times out, follow the tool's `next` instruction (wait again on that task); do not start a second
     reviewer.
   - Fix every `blocking` and `should-fix` finding yourself (as the Opus 5.5 implementer/fixer). Handle nits at your
     judgement. Push.
   - Each re-review is a **new** `delegate_task` summarizing the previous findings and what changed.
   - Stop when the reviewer reports `NO BLOCKING FINDINGS`.
6. Post the final review summary as a PR comment (`gh pr comment`), noting which findings were fixed or declined and
   why.
7. Do **not** merge. Finish with one line: `PR <full URL>, head <SHA>, checks and reviews <state>`.

## 3. App play-tester & reviewer (GPT-6.2 Sol high fast / GPT-6.2 Astra high standard)

Used in two complementary ways, always as an independent subagent that does not silently patch over bugs it finds:

- **App play-testing:** Exercise FOLO Studio (`web/`), FOLO Desktop (`apps/desktop/`), `folo` CLI, and `folo mcp` like
  a real, demanding user — starting servers, running real and `--mock` flows, testing bad inputs, empty/error states,
  disconnects, keyboard navigation, and cross-app parity. Return concrete, reproducible findings (steps, observed vs
  expected, screenshots/logs, severity) for **Opus 5.5** to fix with regression tests.
  - Use **GPT-6.2 Sol (`reasoningEffort: "high"`, `serviceTier: "priority"`)** for fast play-testing passes and
    tight PR review loops.
  - Use **GPT-6.2 Astra (`reasoningEffort: "high"`, `serviceTier: "default"`, standard — not fast)** for deeper,
    thorough play-testing sweeps, tricky multi-step user journeys, and careful architectural reviews.
- **PR code review:** Inspect diffs against `origin/main` read-only and report blocking, should-fix, and nit findings
  with exact `file:line` references and concrete failure scenarios.

## 4. Writer & creative writer (Gemini 4 Neon via Devin CLI in T3 Code)

Delegated via `target = { providerInstanceId: "acpRegistry_devin", model: "gemini-4-neon-high", options: { thought_level: "high" } }`:

- Owns high-craft writing and creative writing across the repository: `README.md`, user guides (`docs/QUICKSTART.md`,
  `docs/AGENT_COOKBOOK.md`, etc.), `CONTRIBUTING.md`, `CHANGELOG.md`, release announcements, UI/empty-state microcopy,
  and architectural storytelling that makes FOLO clear and enjoyable to read.
- Verifies all technical facts against the code and benchmarks before writing (no invented numbers or features).
- Runs `(cd site && npm run check && npm run build)` and `bash tools/file-size.sh` before opening a docs PR so every
  link, anchor, image, and landing-page section stays valid.

## UI threads
`web/` (Studio) and `apps/desktop/` (Desktop) each have a long-running top-level thread that hill-climbs its UI:
build → play-test and screenshot every screen and state → two independent design reviews (Claude Opus 5.5 and
GPT-6.2 Sol / GPT-6.2 Astra) scoring a fixed rubric 1–10 → Opus 5.5 fixes (with Gemini 4 Neon polishing copy when
needed) → repeat until **both reviewers give ≥ 9 on every rubric line**. Each round lands as a PR through the same
review loop above.
