# Agent process: orchestrate → implement → play-test & review → write → merge

How the maintainers' agents collaborate on FOLO. It is specific to their setup (T3 Code orchestration tools, a Windows
machine with build output on `D:`, and the four subagent roles below); an external contributor needs only
[AGENTS.md](../../AGENTS.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Subagent roster

Call `orchestrator_capabilities` once per parent thread before the first `delegate_task`. Every role runs as a
dedicated subagent with a clear boundary of responsibility:

| Role | Model | T3 Code `delegate_task` `target` |
|---|---|---|
| **Main orchestrator** | **Gemini 4 Neon** (`thought_level: high`), Claude Opus 5.5, or Claude Fable 5.2 (`effort: high`) | `{ providerInstanceId: "antigravity", model: "gemini-4-neon", options: { thought_level: "high" } }` <br> or `{ providerInstanceId: "claudeAgent", model: "claude-opus-5-5", options: { effort: "high" } }` <br> or `{ providerInstanceId: "claudeAgent", model: "claude-fable-5-2", options: { effort: "high" } }` |
| **Main code implementer & doer** | **Gemini 4 Neon** (`thought_level: high`) or Claude Opus 5.5 (`effort: high`) | `{ providerInstanceId: "antigravity", model: "gemini-4-neon", options: { thought_level: "high" } }` <br> or `{ providerInstanceId: "claudeAgent", model: "claude-opus-5-5", options: { effort: "high" } }` |
| **PR reviewer & app play-tester** | **Dual PR review (both required):** **GPT-6.2 Sol** (`high`, **fast**) **+** **Gemini 4 Neon** (`high`); GPT-6.2 Astra (`high`, **standard — not fast**) for deep play-testing | Sol (fast): `{ providerInstanceId: "codex", model: "gpt-6.2-sol", options: { reasoningEffort: "high", serviceTier: "priority" } }` <br> Neon (high): `{ providerInstanceId: "antigravity", model: "gemini-4-neon", options: { thought_level: "high" } }` <br> Astra (standard, not fast): `{ providerInstanceId: "codex", model: "gpt-6.2-astra", options: { reasoningEffort: "high", serviceTier: "default" } }` |
| **Writer & creative writer** | **Gemini 4 Neon** (`thought_level: high`) via the Antigravity CLI in T3 Code | `{ providerInstanceId: "antigravity", model: "gemini-4-neon", options: { thought_level: "high" } }` |

Keep `serviceTier: "default"` (standard, never `"priority"` / fast) when running **GPT-6.2 Astra**; use
`serviceTier: "priority"` (fast) for **GPT-6.2 Sol**.

**Gemini 4 Neon (`high`) is good for everything.** Gemini 4 Neon (`thought_level: high`) is a first-class default across
every role — main orchestrator, implementer and fixer, researcher, PR reviewer, and writer — because it performs
strongly across the whole stack. No special owner opt-in is required to dispatch it for any task. A Neon agent may split
its own implementation work with Antigravity's agent spawning and verify what its sub-agents return before reporting.
Every change still lands as a **PR to `main`**, worktrees stay disjoint, and every PR goes through the mandatory **dual
review loop** below (fresh **GPT-6.2 Sol `high`, fast** + fresh **Gemini 4 Neon `high`**) until both sign off with zero
blocking findings.

## 1. Main orchestrator (Gemini 4 Neon high / Claude Opus 5.5 / Claude Fable 5.2)

- Plans the wave, breaks work into focused, independent PRs against `main`, and dispatches subagents with disjoint
  worktrees (`isolation: "worktree"` for independent work).
- Dispatches **Gemini 4 Neon (`thought_level: high`)** or **Claude Opus 5.5 (`effort: high`)** as the main
  implementer/doer for code changes, research, bug fixes, and refactors.
- Requires **dual PR review** on every PR to `main` — both a fresh **GPT-6.2 Sol (`high`, fast)** reviewer and a fresh
  **Gemini 4 Neon (`high`)** reviewer — and dispatches **GPT-6.2 Sol (`high`, fast)** or **GPT-6.2 Astra (`high`,
  standard — not fast)** to play-test the real app (Studio, Desktop, CLI, MCP).
- Dispatches **Gemini 4 Neon (`thought_level: high`)** for documentation rewrites, README storytelling, release notes,
  guides, and creative/UX writing.
- Watches each delivered PR, merges with `gh pr merge --squash --delete-branch` (or `pull_request_merge`) only once CI
  is green and **both GPT-6.2 Sol (`high`, fast) and Gemini 4 Neon (`high`)** report no blocking findings, updates
  `docs/ROADMAP.md`, and starts the next wave. On a merge conflict, asks the implementer (or a fresh Neon / Opus 5.5
  subagent) to rebase.

## 2. Main code implementer & doer (Gemini 4 Neon high / Claude Opus 5.5 high)

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
4. **Always open a PR to `main`:** commit (`area: summary`, with your harness co-author trailer), `git push -u origin
   HEAD`, then open a PR against `main` (`pull_request_create` or `gh pr create --base main --title "area: summary"
   --body-file <file>`). The body states what changed, why, how it was tested, benchmark numbers if hot paths changed,
   and any follow-ups.
5. **Mandatory dual review & play-test loop** (repeat rounds until both reviewers sign off):
   - In every review round, launch **two fresh, independent read-only reviewers** against the PR diff (`origin/main..HEAD`):
     1. **GPT-6.2 Sol (`high`, fast)** via `delegate_task`:
        `target = { providerInstanceId: "codex", model: "gpt-6.2-sol", options: { reasoningEffort: "high", serviceTier: "priority" } }`
        *(Optional: also run **GPT-6.2 Astra (`high`, standard — not fast)** with `serviceTier: "default"` for deep
        app play-testing sweeps.)*
     2. **Gemini 4 Neon (`high`)** via a fresh `delegate_task` (`target = { providerInstanceId: "antigravity", model:
        "gemini-4-neon", options: { thought_level: "high" } }`) or, when orchestrating inside Antigravity, a fresh
        independent read-only Neon reviewer sub-agent. Never let the implementing agent grade its own PR.
   - Pass `role: "review"`, `isolation: "shared"`, the PR number and URL, the absolute path of your checkout, and the
     branch.
   - Ask both reviewers to inspect the diff against `origin/main` for correctness bugs, soundness (`unsafe`), numerical
     errors, missing tests, API problems, and performance traps without editing files, and to return findings as a
     numbered list with severity (`blocking` / `should-fix` / `nit`), `file:line`, failure scenario, and suggested fix
     — or state `NO BLOCKING FINDINGS` explicitly if there are none.
   - Fix every `blocking` and `should-fix` finding from either reviewer yourself (handle nits at your judgement), re-run
     relevant checks, and push.
   - Each re-review round starts **fresh** reviewer instances summarizing the previous findings and what changed (never
     reusing an old review thread with `t3_thread_send`).
   - Stop when **both GPT-6.2 Sol (`high`, fast) and Gemini 4 Neon (`high`)** report `NO BLOCKING FINDINGS` on the
     final PR head — or, if **Gemini 4 Neon (`high`)** reports `NO BLOCKING FINDINGS`, you agree, CI is green, and
     Sol's only remaining findings are minor, low-impact, or contrived extreme edge cases, note why they are declined
     in the PR comment and proceed to merge rather than looping indefinitely.
6. Post the final review summary as a PR comment (`pull_request_respond` or `gh pr comment`), noting which findings from
   Sol and Neon were fixed or declined and why.
7. Do **not** merge unless your task says so. Finish with one line: `PR <full URL>, head <SHA>, checks and reviews <state>`.

## 3. PR reviewers & app play-testers (GPT-6.2 Sol high fast + Gemini 4 Neon high / GPT-6.2 Astra high standard)

Used in two complementary ways, always as independent subagents that do not silently patch over bugs they find:

- **Dual PR code review (every PR to `main`):** Both **GPT-6.2 Sol (`reasoningEffort: "high"`, `serviceTier: "priority"`)**
  and **Gemini 4 Neon (`thought_level: "high"`)** inspect diffs against `origin/main` read-only and report blocking,
  should-fix, and nit findings with exact `file:line` references and concrete failure scenarios. Cross-model review
  catches distinct classes of bugs; a PR is ready to merge only when **both** report `NO BLOCKING FINDINGS`.
- **App play-testing:** Exercise FOLO Studio (`web/`), FOLO Desktop (`apps/desktop/`), `folo` CLI, and `folo mcp` like
  a real, demanding user — starting servers, running real and `--mock` flows, testing bad inputs, empty/error states,
  disconnects, keyboard navigation, and cross-app parity. Return concrete, reproducible findings (steps, observed vs
  expected, screenshots/logs, severity) for the implementer (**Gemini 4 Neon `high`** or **Opus 5.5**) to fix with
  regression tests.
  - Use **GPT-6.2 Sol (`reasoningEffort: "high"`, `serviceTier: "priority"`)** for fast play-testing passes and tight PR
    review loops.
  - Use **GPT-6.2 Astra (`reasoningEffort: "high"`, `serviceTier: "default"`, standard — not fast)** for deeper,
    thorough play-testing sweeps, tricky multi-step user journeys, and careful architectural reviews.

## 4. Writer & creative writer (Gemini 4 Neon high via the Antigravity CLI in T3 Code)

Delegated via `target = { providerInstanceId: "antigravity", model: "gemini-4-neon", options: { thought_level: "high" } }`:

- Owns high-craft writing and creative writing across the repository: `README.md`, user guides (`docs/QUICKSTART.md`,
  `docs/AGENT_COOKBOOK.md`, etc.), `CONTRIBUTING.md`, `CHANGELOG.md`, release announcements, UI/empty-state microcopy,
  and architectural storytelling that makes FOLO clear and enjoyable to read.
- Verifies all technical facts against the code and benchmarks before writing (no invented numbers or features).
- Runs `(cd site && npm run check && npm run build)` and `bash tools/file-size.sh` before opening a docs PR so every
  link, anchor, image, and landing-page section stays valid.

## UI threads
`web/` (Studio) and `apps/desktop/` (Desktop) each have a long-running top-level thread that hill-climbs its UI:
build → play-test and screenshot every screen and state → two independent design reviews (Gemini 4 Neon `high` / Claude
Opus 5.5 and GPT-6.2 Sol `high`, fast / GPT-6.2 Astra) scoring a fixed rubric 1–10 → implementer fixes (Gemini 4 Neon
`high` or Opus 5.5) → repeat until **both reviewers give ≥ 9 on every rubric line**. Each round lands as a PR to `main`
through the same dual review loop above.
