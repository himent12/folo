# Stress round, lane s5-label: Studio labeling and benchmarks

Base `1436ad68`. Studio driven headless (playwright-core, Chrome, 1440x900 and 390x844) against `folo serve` on the
CPU with an isolated data dir copied from the seed kit; the agent CLI was a fake `claude` (no network, no account)
answering ok, garbage, error, slow, huge and edge replies. Scripts and shots live in the lane dir
(`D:/folo-build/stress/lanes/s5-label/`).

## What was tested

- Labeling: jobs from a server folder (60 COCO photos), the 22 hostile files of the seed (`odd/`: zero-byte,
  truncated, CMYK, 16-bit, EXIF orientations 1-8, 8000x6000, unicode and spaced names) and an 8-image folder;
  agent runs in every reply mode; stop mid-run; edits and deletes of images in and out of the agent call while a run
  is active; delete of a running job; server killed mid-run and restarted; the editor: boxes drawn left-to-right,
  right-to-left, bottom-up, from the letterbox, out of the image, tiny, a click; Add box, class keys 1-9 (9 with four
  classes), Delete, 40 J and 45 K presses held, 30 adds + 100 undo + 100 redo, reload mid-edit (beforeunload asks),
  Enter on the last image; save with the server killed, then after its restart; the same image in two tabs;
  datasets made from two jobs (val 0.2 and 0.5) with every label line compared to the API boxes (0 differences).
- Benchmarks: from the shapes val split, from a photos job and from the odd job; model entries on the CPU
  (folo-n-coco, the shapes models), followed live in two tabs, reloaded mid-run, deleted while the entry page was
  open; external entries over the API with 20 malformed shapes; a 35-entry leaderboard (25 externals with long CJK
  names) checked row by row against `GET …/entries`; compare of four entries with one deleted under it; 390 px; the
  predictions-file upload through Add entry; the plan editor with weights 0, -1, empty, `1e400`, 0.0001, 10.5.

## Findings

| id | severity | surface | what | status |
|---|---|---|---|---|
| s5-label-1 | annoying | Studio labeling | The same image open in two tabs: when one saves, the other (with unsaved edits) shows nothing; its Save silently replaces the other's labels and its Discard goes back to stale boxes. | fixed |
| s5-label-2 | annoying | server labeling | A server killed mid-run leaves the agent CLI processes running to the end of their (paid) call: 4 of 4 alive 2 s after the kill. | deferred: Windows fix on branch `stress/s5-label-tree`, not compiled (no build slot); Unix open |
| s5-label-3 | annoying | server + Studio bench | External boxes with `x2 < x1` / `y2 < y1` (most often `x, y, w, h` sent as corners) or an f32-overflowing coordinate were taken and scored as wrong guesses: `201 done`, a real-looking leaderboard row. | fixed |
| s5-label-4 | polish | Studio bench | A plan weight field emptied read as valid, and Save plan stored the last number typed into it. | fixed |
| s5-label-5 | polish | server labeling | A dataset made from a job and named `nul` answers `500 internal`. | deferred: s1-datasets fixes `slug` |

### s5-label-1: two tabs, one image
Repro: open `#/datasets/label/<job>?image=7` in tabs A and B. In B, Add box (unsaved). In A, delete every box and
Ctrl+S. Before: B showed its three boxes, no notice; the server had `[]`; B's Save would write its boxes over A's,
B's Discard went back to what B opened with. Root cause: `web/src/routes/label/Editor.tsx`, the effect that follows
the server's image kept the draft and the old `base` whenever the draft was dirty. Fix: a server change under
unsaved edits that is not this editor's own save on its way becomes `base` (what Discard returns to), and the status
line says "Changed elsewhere while you edited ... Save replaces them with yours; Discard shows them". The PUT still
has no version check (last writer wins): the user now knows before choosing. Test:
`web/src/routes/label/editor.test.tsx` "says so when another tab saves the image under unsaved edits" (fails
without the fix: no status). Checked end to end with the built Studio served by `--web-dir`.

### s5-label-2: agent processes outlive a killed server
Repro: a slow agent, start a run, `taskkill /F` the server: the CLI shims and their node children run on (4 of 4
after 2 s). Root cause: `crates/folo-server/src/label/tree.rs` made the call's job object without
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Proposed fix, NOT in this PR: the limit is set when the job is made, so the
handle Windows closes for a dead server ends the tree; `Drop` clears it before closing, so a call that finished
normally still leaves what it started alone (the module's promise). Test: `label::tree::windows::tests` (a job closed
without its drop ends its process; a dropped one does not). It is commit `3263d76d` on branch `stress/s5-label-tree`
and has never been compiled: in three turns no build slot came free (queues of four to eight). Deferred: Unix (the call leads a process group, which nothing ends when the
server is killed; `PR_SET_PDEATHSIG` or a reaper would be needed).

### s5-label-3: external boxes with reversed corners
Repro: `POST /api/bench/benchmarks/{id}/entries` with an external contestant whose boxes have `x1: 50, x2: 10` (or
`x1: 1e308`, read as f32 infinity): `201 done score=0`. Root cause: `crates/folo-server/src/bench/external.rs`
`Filling::add` checked the class and the score only; Studio's file check (`web/src/lib/bench/external.ts`
`checkPredictions`) the same. Fix: both refuse a coordinate that is not a finite f32 and corners the wrong way round
(comparing as f32, as the server reads them), naming the record and box, with a hint about `x, y, w, h`; a box of no
area is still taken. Tests: `external.rs` `every_wrong_record_is_named`, `external.test.ts`. `docs/API.md` lists
both rules.

### s5-label-4: an emptied plan weight saves the last number typed
Repro: Conditions tab of a benchmark with no entries; type 3 into Weight of Haze, then clear the field. Before: the
field was empty and not flagged, Save plan was enabled and stored weight 3 (0, -1 and 10.5 were flagged; `1e400`,
which the number input reads as empty, was not). Root cause: `web/src/routes/bench/PlanTab.tsx` keeps the draft at
the last number while the text is blank, and the field's error and the Save gate looked at the draft only. Fix: a
weight text that is not a number flags the field and gives the plan error "Haze: the weight is empty ...", which holds
Save plan. Test: `web/src/routes/bench/bench.test.tsx` "does not save a weight field left empty with the last number
typed in it" (fails without the fix).

Review round 1 found a second way to the same save: leaving the page with the field empty and choosing Save plan in
"Save the plan?" called `savePlan` (`web/src/routes/Benchmark.tsx`) with no check. The typed weights now live in the
page next to the draft (they also survive a change of tab), `$lib/bench/plan` `editedPlanError` is the one check,
`savePlan` refuses while it fails, and the dialog's Save plan is held with the reason. Test: "does not save a weight
field left empty from the leave dialog either, also after a change of tab" (fails without the fix).

### Siblings of s5-label-1 (review round 1)
- The plan editor in two tabs: B with unsaved edits kept them silently when A saved the plan, so B's Save plan undid
  A's change unseen. `Benchmark.tsx` now says "Changed elsewhere while you edited ... Save plan replaces it with
  yours; Discard shows it" when a plan arrives under edits that is not the page's own save. Test: "says so when
  another tab saves the plan under unsaved edits, and keeps them".
- Edit job in two tabs (`web/src/routes/label/JobDialog.tsx`): B, changing only the name, sent its stale class list
  and reverted A's class rename. A field still at what the dialog took in now follows the job's changes from
  elsewhere; one edited here is kept and the dialog says so. Tests in `labeling.test.tsx` ("Edit job does not put
  back a class another tab renamed ...", "Edit job says so when another tab saves a field edited here").
- Neither save has a server-side version check: last writer wins once the user is told.

### Review round 2 (re-review 1)
- The plan's "Changed elsewhere" note lived only in the Conditions tab: B on the Leaderboard tab, leaving with edits,
  saw no warning in "Save the plan?" and its Save plan undid A's change. The leave dialog now says so too. Test: "says
  so in the leave dialog when another tab saved the plan while another view was open".
- A plan saved by another tab while this page's save was on its way was taken for the page's own (told by `saving`,
  i.e. timing) and never warned; the late answer then said "Plan saved". The page's own update is now told by the plan
  it sent, and a save whose answer finds another plan on the server keeps the edits, warns and does not say saved.
  Test: "says so when another tab saves the plan while this one saves, and does not say Plan saved".
- An emptied weight lost its text and error at any edit of another condition, so the draft's old number came back.
  The typed text that holds no number now stays until Discard, a plan from the server or the condition turned off.
  Test: "keeps an emptied weight empty, with its error, across edits of other conditions". All three fail without the
  fix.
- Desktop's demo backend (`apps/desktop/src/api/mock_bench/scoring.rs` `check_external_boxes`, called from
  `entries.rs`) now refuses external boxes with reversed corners or a non-finite coordinate, as the server does
  (s5-label-3's sibling). Test: `mock_bench` `external_boxes_with_reversed_corners_or_no_number_are_refused_not_scored`.

### Review round 3 (re-review 2)
- Round 2's check took the server's own rounding for another tab: a weight such as `2.00000001` comes back as `2`
  (the server keeps it as an f32), so "Plan saved, then changed elsewhere" showed and the leave dialog never let go.
  `samePlan` now compares plans as the server keeps them (weight as f32, reasons trimmed) and a save is checked
  against the PUT answer. Test: "saves a weight the server rounds from the leave dialog and leaves ...".
- The image editor's "Changed elsewhere" lived only in its status line, under the inert editor when "Save your
  changes?" was open. The leave dialog now says it too, also when the other tab saves while the dialog is open.
  Test: "says so in the leave dialog too when another tab saves the image ...".
- Use this plan (a proposal from Design with an agent) kept an emptied weight field of the old plan, so the proposal
  showed an empty field and could not be saved. It now starts the fields afresh. Test: "takes a proposal used in
  place of the plan with its own weights ...". All three fail without the fix (lane log `r3-nofix.log`).

## Pushed hard, no bug
Agent garbage / error / edge / huge replies (clear run errors, sub-pixel and unknown-label boxes dropped); the
in-call image is `409` to PUT and DELETE while a waiting one leaves the run; save with the server down (clear error,
edits kept, saved after the restart); job state after a server killed mid-run (idle, `stopped`, images back to
labeled); dataset export line by line; bench live progress, reload, delete under an open entry ("Entry not found");
leaderboard order and scores against the API with 35 entries; compare with a deleted entry; 20 malformed external
shapes (all refused with clear messages except s5-label-3); dataset names `../../evil`, `..`, `日本語`, blank, `CON`
(folders `evil`, `labeled`, `labeled-2`, the job's name, `con`); Enter held 80 times on "Looks right" (31 images
reviewed in order, none twice or skipped, no request error); editor, job page and leaderboard at 390×844 and
800×600 (no sideways overflow, every editor button reachable).

### s5-label-5: a dataset named "nul" (deferred, SIBLING)
`POST /api/label/jobs/{id}/dataset {"name":"nul"}` answers `500 internal` "writing the dataset .../datasets/nul: The
system cannot find the path specified": `crates/folo-server/src/util.rs` `slug` keeps `nul`, a Windows device name.
The same root cause is claimed and fixed by lane s1-datasets (`slug`, shared by synthetic, upload, label export and
loop datasets); nothing changed here.

## Not done
Loops (a tiny student run on the CPU), agent benchmark entries, plan design by an agent.
