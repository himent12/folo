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
  names) checked row by row against `GET …/entries`; compare of four entries with one deleted under it; 390 px.

## Findings

| id | severity | surface | what | status |
|---|---|---|---|---|
| s5-label-1 | annoying | Studio labeling | The same image open in two tabs: when one saves, the other (with unsaved edits) shows nothing; its Save silently replaces the other's labels and its Discard goes back to stale boxes. | fixed |
| s5-label-2 | annoying | server labeling | A server killed mid-run leaves the agent CLI processes running to the end of their (paid) call: 4 of 4 alive 2 s after the kill. | fixed (Windows); Unix deferred |
| s5-label-3 | annoying | server + Studio bench | External boxes with `x2 < x1` / `y2 < y1` (most often `x, y, w, h` sent as corners) or an f32-overflowing coordinate were taken and scored as wrong guesses: `201 done`, a real-looking leaderboard row. | fixed |

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
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Fix: the limit is set when the job is made, so the handle Windows closes for a
dead server ends the tree; `Drop` clears it before closing, so a call that finished normally still leaves what it
started alone (the module's promise). Test: `label::tree::windows::tests` (a job closed without its drop ends its
process; a dropped one does not). Deferred: Unix (the call leads a process group, which nothing ends when the
server is killed; `PR_SET_PDEATHSIG` or a reaper would be needed).

### s5-label-3: external boxes with reversed corners
Repro: `POST /api/bench/benchmarks/{id}/entries` with an external contestant whose boxes have `x1: 50, x2: 10` (or
`x1: 1e308`, read as f32 infinity): `201 done score=0`. Root cause: `crates/folo-server/src/bench/external.rs`
`Filling::add` checked the class and the score only; Studio's file check (`web/src/lib/bench/external.ts`
`checkPredictions`) the same. Fix: both refuse a coordinate that is not a finite f32 and corners the wrong way round
(comparing as f32, as the server reads them), naming the record and box, with a hint about `x, y, w, h`; a box of no
area is still taken. Tests: `external.rs` `every_wrong_record_is_named`, `external.test.ts`. `docs/API.md` lists
both rules.

## Pushed hard, no bug
Agent garbage / error / edge / huge replies (clear run errors, sub-pixel and unknown-label boxes dropped); the
in-call image is `409` to PUT and DELETE while a waiting one leaves the run; save with the server down (clear error,
edits kept, saved after the restart); job state after a server killed mid-run (idle, `stopped`, images back to
labeled); dataset export line by line; bench live progress, reload, delete under an open entry ("Entry not found");
leaderboard order and scores against the API with 35 entries; compare with a deleted entry; 20 malformed external
shapes (all refused with clear messages except s5-label-3).

## Not done
Loops (a tiny student run on the CPU), the Studio predictions-file upload driven through the dialog (only its
checker was tested), the benchmark plan editor with odd weights, agent benchmark entries.
