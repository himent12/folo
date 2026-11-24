# Agent labeling

You have images and no labels. FOLO can have a coding agent draw the boxes: it drives **Claude Code**, **Codex** or
**Antigravity** on your machine, shows you every box for review, and turns the result into a dataset that trains like
any other.

```
unlabeled images ──► labeling job ──► agent run ──► review and correct ──► dataset ──► folo train
   (upload or a          (classes +      (Claude Code,     (Studio, or an    (YOLO folder,
    folder)              instructions)    Codex or Agy)     agent via MCP)    registered)
```

The agents' boxes are a draft. On everyday photos they find most objects and place most boxes well (numbers
below), and they also miss things and box things that are not there. Plan to look through the result: the review
view is built for that.

## What you need
- `folo serve` (the real server, not `--mock`), or `folo mcp` / `folo label` on a data dir.
- At least one of the three CLIs on the machine that runs `folo serve`, signed in:
  - Claude Code: install it from <https://claude.com/claude-code>, then run `claude` once and sign in.
  - Codex: install it from <https://developers.openai.com/codex/cli>, then run `codex login`.
  - Antigravity: install it from <https://antigravity.google/docs/getting-started>, then run `agy` (or `agy acp --login`) once and sign in.

  FOLO finds `claude`, `codex` and `agy` on the `PATH`; `FOLO_CLAUDE_BIN`, `FOLO_CODEX_BIN` and
  `FOLO_ANTIGRAVITY_BIN` name another executable.
  Studio shows which agents were found. Labeling uses that CLI's account: its plan limits and its bill.

The images are sent to the agent's provider (Anthropic, OpenAI or Google), as with any use of those tools. Do not label
images you may not share with them.

## In Studio
1. **Datasets → Label images → New labeling job.** Give it a name and the classes to find, and say in the
   instructions what a person would need to know: what counts as each class, what to ignore, how to treat partly
   hidden objects.
2. **Add images.** Drop files on the page, or add a folder that is on the server's machine (its images are read in
   place, not copied). A job holds up to 10000 images; JPEG, PNG, BMP and WebP.
3. **Pick the agent and press Label.** Images turn from pending to labeled as the calls come back; the panel
   shows progress, the tokens the agent reported, and the log. Stop ends the run at once and keeps what is done.
4. **Review.** Open an image to move, resize, relabel, delete or draw boxes (Ctrl+Z or ⌘Z undoes an edit,
   Ctrl+Shift+Z redoes it); saving marks it as labeled by hand
   and reviewed (Enter saves and opens the next image to review, Ctrl+S or ⌘S saves and stays; its tile then says
   Edited and Reviewed). When the agent's boxes are right, **Looks right** (Enter) marks them reviewed and opens the next
   image to review; the To review and Reviewed filters and the count in the header show how far you are. A save or
   a Looks right is made on the boxes you see: if another tab, the CLI or an agent changed them first, nothing is
   saved or marked, your edits stay, and the editor shows the new boxes to look at again. If the agent got an image wrong in a way you can describe, ask it again with a note ("the boxes are too
   loose", "that is a van, not a truck"). If the agent cannot answer, the image keeps the boxes it had, reviewed or
   not, and the editor says why. Failed images wait for the next run.
5. **Create dataset.** The labeled images become a YOLO dataset under the server's data dir, split into train and
   val, registered and ready for a training run. Images without labels are left out; a labeled image without
   boxes is kept as a background image. The job stays, so you can add images, label them and export again.

## From a script or an agent
An agent (Claude Code, Codex, Antigravity, any MCP client) can run the whole loop itself through FOLO's MCP server
([MCP.md](MCP.md), Labeling): `create_label_job` on a folder, `start_labeling` (a sample of about 20 images first,
with `image_ids`), `label_status` until the run is idle, then for each image `view_label_image` (the image with its
boxes drawn and numbered, as an image the model sees, plus the boxes as text and their revision) and either
`review_label_images` ("Looks right") or `set_label_boxes` (relabel, remove or add boxes by number, saved as a hand
edit), and finally `export_label_dataset` and `train`. Boxes are `[class, x1, y1, x2, y2]` with the corners in
pixels of the view the agent looked at (1024 px on the long side by default; the view states its size), and an
edit or a review names the revision of the boxes it was made on, so a run that changed them since makes it fail
instead of landing on other boxes. The `label_folder` prompt walks an agent through it:

```sh
claude mcp add folo -- folo mcp        # then ask: "label ~/photos/street for cars and people, then train on it"
```

Scripts get the same operations, with `--json` and stable exit codes, from `folo label` ([CLI.md](CLI.md),
Labeling). It goes through the `folo serve` that owns the data dir, so Studio shows the job as it changes:

```sh
folo label create street --classes car,person --instructions "Label every car and pedestrian." \
  --path /data/photos/street --json                       # {"job_id":"lb_3f9a","images":240,"added":240,...}
folo label start lb_3f9a --agent claude --wait --json      # progress records, then the job's status
folo label images lb_3f9a --status labeled --reviewed false --boxes --json
folo label view lb_3f9a -o next.jpg                        # the next image to review, boxes numbered
folo label set lb_3f9a 17 --expect 3-5d01c2e7 --relabel 2=person --remove 3   # the revision view printed
folo label review lb_3f9a 18 19 20
folo label export lb_3f9a --json                           # {"dataset_id":"ds_8c1e",...,"train":192,"val":48}
```

### CLI vs MCP cheat sheet
`folo label` and the 10 MCP labeling tools ([MCP.md](MCP.md#labeling)) share the same underlying operations (pass
`--json` to any CLI subcommand for machine-readable output; list envelopes differ by front end — CLI `jobs` returns
`{jobs}` and `images` uses `--offset`/`--limit`, while MCP list tools use cursor paging `{items, next_cursor,
total}`):

| step | CLI (`folo label ...`) | MCP tool (`folo mcp`) |
|---|---|---|
| Check installed CLIs | `folo label agents` | *(none; `GET /api/label/agents`, or `start_labeling` error hint)* |
| List jobs | `folo label jobs` | `list_label_jobs` |
| Create job (+ initial paths) | `folo label create NAME --classes A,B [--instructions T] [--path P]...` | `create_label_job { name, classes, instructions?, paths? }` |
| Add images to existing job | `folo label add JOB PATH...` | `create_label_job { job_id, paths }` |
| Start agent run | `folo label start JOB --agent claude\|codex\|antigravity [--model M] [--effort low\|medium\|high] [--fast] [--images ID,...\|--all] [--note T] [--batch N] [--parallel N] [--wait]` | `start_labeling { job_id, agent, model?, effort?, fast?, scope?, image_ids?, note?, batch?, parallel? }` |
| Stop agent run | `folo label stop JOB` | `stop_labeling { job_id }` |
| Check progress, tokens & log | `folo label status JOB [--log 5]` | `label_status { job_id, log_lines? }` |
| List / filter images | `folo label images JOB [--status ...] [--reviewed true\|false] [--boxes]` | `list_label_images { job_id, status?, reviewed?, include_boxes? }` |
| View next or specific image | `folo label view JOB [IMAGE] [-o FILE] [--size 1024] [--from 1]` | `view_label_image { job_id, image_id?, size?, from? }` |
| Edit boxes (manual + reviewed) | `folo label set JOB IMAGE (--box ... \| --clear \| --relabel/--remove/--add) --expect REV [--size WxH]` *(required with `--box`/`--add`)* | `set_label_boxes { job_id, image_id, expect, size?, boxes? \| relabel/remove/add }` *(`size` required with non-empty `boxes`/`add`)* |
| Mark reviewed ("Looks right") | `folo label review JOB IMAGE... [--undo] [--expect ID=REV]...` | `review_label_images { job_id, image_ids, reviewed?, expect? }` |
| Export registered YOLO dataset | `folo label export JOB [--name N] [--val-fraction 0.2] [--seed 0]` | `export_label_dataset { job_id, name?, val_fraction?, seed? }` |

Everything Studio does is also the HTTP API ([API.md](API.md), Agent labeling):

```sh
api=http://127.0.0.1:7878/api
job=$(curl -s $api/label/jobs -H 'Content-Type: application/json' \
  -d '{"name": "street", "classes": ["car", "person"], "instructions": "Label every car and pedestrian.",
       "path": "/data/photos/street"}' | jq -r .id)
curl -s $api/label/jobs/$job/start -H 'Content-Type: application/json' -d '{"agent": "claude"}'
curl -sN $api/label/jobs/$job/events          # job / image / log events until you stop reading
curl -s $api/label/jobs/$job                  # status "idle" once the run has ended; counts, usage, run.outcome
curl -s "$api/label/jobs/$job/images?status=labeled&reviewed=false&limit=20"   # what is left to review
curl -s $api/label/jobs/$job/dataset -H 'Content-Type: application/json' -d '{"val_fraction": 0.2}'
```

The last call answers with the new `Dataset`; its `id` goes into `POST /api/runs` or `folo train --data <path>`.

## What a run costs
FOLO makes **one model request per batch of images, with no tools**. That is the whole trick: an agent CLI left
to its defaults loads its coding instructions, tool definitions, your project's notes and MCP servers into every
request, and would read each image through a tool call, one request per step. FOLO switches all of that off and
attaches the images to the prompt.

| | Claude Code | Codex | Antigravity |
|---|---|---|---|
| Images per call (default) | 4 | 8 | 8 |
| Images sent at, long side at most | 1024 px | 768 px | 1024 px |
| Fixed input tokens per call | about 900 (FOLO's instructions) | about 700 (FOLO's instructions and reply schema; 20000 with Codex's defaults) | FOLO's instructions via ACP (`--include-default-customizations=false`) |
| Input tokens per photo, all told | about 1280 for a photo of 1024 px or more, 620 for a 640 px one | about 650 for a photo of 768 px or more, 460 for a 640 px one | depends on the model (`gemini-4-neon`, `gemini-3.8-flash`, ...) |
| Output tokens per photo | about 165 (Haiku at low effort, mostly its reasoning) | about 40 | low effort by default (effort-suffixed slugs like `gemini-4-neon-xhigh` also work) |
| Calls in flight | 3 | 3 | 3 |

Larger images cost more, up to a cap: images are scaled down before they are sent, so a 12-megapixel photo costs
what a 1024 px one does (768 px for Codex). The prompt gives each image's size as sent, the boxes come back in
pixels of that, and they are stored in pixels of the original file. (Codex passes its images on at the size FOLO
sends them: asked for a red square's corners in images of 768 x 576 up to 2048 x 1536 pixels, portrait too,
gpt-6.2-sol answered in exact pixels of each.) The sizes are where labels stop improving
for their cost: Codex labels as well at 768 px as at 1024 for 40% fewer tokens, while Haiku labels worse on
smaller images and better on larger ones (below).

The fixed cost is per call, however few images it has, and it is small for both agents: asking again about a
single image costs about one image's tokens plus the fixed cost.

What made it this lean (November 2026, measured with the eval below; per photo on the 60 test photos, the
defaults before and after):

| | Claude Code before | after | Codex before | after |
|---|---:|---:|---:|---:|
| Input tokens per photo | 686 | 621 | 1456 | 462 |
| Output tokens per photo | 290 | 165 | 73 | 40 |
| Found / right / IoU | 74% / 87% / 0.85 | 80% / 90% / 0.86 | 83% / 80% / 0.89 | 86% / 79% / 0.89 |
| Reported cost of the 60 photos | $0.020 | $0.009 | not reported | not reported |
| Time for the 60 photos | 42 s | 28 s | 54 s | 35 s |

- Claude Code is started without prompt caching (on a subscription it wrote every request, images included, to a
  one-hour cache at twice the input price, and nothing but a short preamble was ever read back), without the
  extra request that titles a session, and without the advisor tool. `--bare` would save a little more, but it
  takes only an API key, never a subscription's sign-in.
- Codex gets its own model catalog back with the models' tools taken off (since codex-cli 0.16x every model
  carried a code-mode tool into every request, whatever the features said), and the skills list, subagent tools
  and `request_user_input` switched off: a call's own overhead went from about 7800 tokens to about 50. When the
  catalog cannot be read, your Codex `config.toml` sets a key that can point Codex at another catalog or server
  (`model_catalog_json`, `profile`, `openai_base_url` or `chatgpt_base_url` at its top level) or exists but
  cannot be read, or Codex refuses the file as it starts, calls go without it at the old cost, and the run's log
  says so once. A refusal holds for the version of Codex that refused, until another one is installed.
- Antigravity runs `agy acp --agent folo-labeler --cli_dir_name folo-label --sandbox --include-default-customizations=false --disable-telemetry`
  over ACP v1 JSON-RPC on stdio in an isolated scratch workspace with no MCP servers: FOLO writes a restricted
  `folo-labeler` custom agent (native file/shell tools and inherited customizations disabled) and a strict deny-all
  `folo-label/settings.json` profile so user auto-approve settings cannot execute tools, enforces a non-auto ACP
  mode, rejects any `session/request_permission` request, and fails closed if a `tool_call` update ever appears.
  The images go inline as base64 JPEGs in a single `session/prompt` turn, FOLO's instructions attach through
  `_meta.antigravity.systemPrompt.append`, and the model and `thought_level` are selected and verified via
  `session/set_config_option` (accepting base ids like `gemini-4-neon` as well as effort-suffixed slugs like
  `gemini-4-neon-xhigh`). Models come from `agy models` with a curated fallback (`gemini-4-neon`,
  `gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-4-argon`, `gemini-3.8-flash`).
- A box is a list, `["car", x1, y1, x2, y2]`, instead of `{"label": "car", "box": [x1, y1, x2, y2]}`: Codex
  writes 25% fewer output tokens for the same labels.
- Haiku runs at low effort and Codex sends images at 768 px (the reasons are in the next section).

**Fast (Codex).** Codex offers a faster service tier (what it calls Fast: priority processing, which uses more of
the plan's usage) for the models whose catalog entry lists it, which on codex-cli 0.169.0 is every one. Studio's
agent panel shows a Fast switch when the Codex model picked offers it (it is set for each agent, so it does not
follow you to another), and the API takes `"fast": true` (refused, before any call, with a model that does not
offer it). When your Codex `config.toml` sets one of the keys above, FOLO cannot tell which models a call has, so
it offers Fast for none (asking for it says why). On the 60 photos it took 28 s instead of 35, with the same
labels and tokens. Claude Code and Antigravity offer no fast mode to FOLO's calls.

A run stops spending as soon as spending is pointless. If the CLI is missing, signed out, refuses the model or
hits its plan's usage limit, or a rate limit (or any fault) that asks for a wait of more than a minute, the run
ends on that call with the CLI's own message, and the images it did not reach stay as they were; the hint names
the limit the message names (the account's usage limit or the provider's rate limit) and when to start again, in
the CLI's words ("resets 3pm", "try again in 20m"; of several waits in one message, the longest counts). A
client error (a 4xx status other than 429) is never asked
again, whatever else its message says. A passing fault of the provider (it is overloaded, a server error, the
connection dropped, a rate limit that asks for a minute or less) is asked once more after 15 seconds, or after the
wait it asks for if that is longer, and a reply that cannot be read is asked for once more; but a batch makes two
calls at most, whatever made the first one fail, and when the second fails too the batch's images fail. What
every call used counts in the run's usage, whatever came of it (a stop during the wait, a timeout, a call that
reported its usage and then hung). One bad box
in a reply costs only that box; an image whose every box was dropped (no class of the job, or nothing left inside
the image) fails with that reason instead of passing for an empty one. Three calls in a row that time out, return
garbage or fail twice on a passing fault end the run.

## How good the labels are
Measured with `crates/folo-server/examples/label_eval.rs` on 60 COCO val2017 photos with 133 objects of 10
everyday classes (person, car, chair, dog, bottle, cup, tv, couch, bicycle, bus), against COCO's own boxes;
Claude Code 2.1.326 and codex-cli 0.169.0, 18 November 2026. The photos are a fixed, seeded pick (every class in
it, no crowd or tiny boxes). *Found* is the share of true boxes matched by a box of the same class at IoU 0.5,
*right* the share of the agent's boxes that match one, *IoU* the mean overlap of the matches. With 133 objects a
difference of 3 points is noise: two passes of the same setup differ by that much.

| Agent, model, effort | Found | Right | IoU | Input / output tokens per photo | Reported cost (60 photos) | Time |
|---|---:|---:|---:|---:|---:|---:|
| Claude Code, Haiku, low (**default**) | 80% | 90% | 0.86 | 621 / 165 | $0.009 | 28 s |
| Claude Code, Haiku, medium | 77% | 87% | 0.85 | 623 / 202 | $0.010 | 29 s |
| Claude Code, Haiku, low, 8 photos a call | 72-74% | 85-88% | 0.85 | 511 / 135 | $0.007 | 18 s |
| Claude Code, Sonnet, low | 76% | 79% | 0.89 | 616 / 44 | $0.10 | 15 s |
| Claude Code, Sonnet, medium | 79% | 81% | 0.89 | 618 / 47 | $0.10 | 14 s |
| Claude Code, Opus, medium | 87% | 70% | 0.89 | 617 / 56 | $0.22 | 24 s |
| Codex, gpt-6.2-sol (its default), low (**default**) | 86% | 79% | 0.89 | 462 / 40 | not reported | 35 s |
| Codex, gpt-6.2-sol, low, Fast | 85% | 80% | 0.89 | 462 / 38 | not reported | 28 s |
| Codex, gpt-6.2-sol, medium | 86% | 79% | 0.89 | 454 / 88 | not reported | 61 s |
| Codex, gpt-6.2-sol, low, 16 photos a call | 84-85% | 77-79% | 0.89 | 412-422 / 36 | not reported | 30 s |
| Codex, gpt-6.2-astra, low | 88% | 82% | 0.89 | 454 / 36 | not reported | 33 s |

The Codex rows at medium effort and with gpt-6.2-astra were measured with the boxes in thousandths of the image
(the shape before November 2026), which Codex labels as well with as with pixels; the Haiku medium, 8-a-call,
Sonnet medium and Opus rows with an earlier wording of the image sizes in the prompt. Antigravity (`agy acp`) was
added after that November 2026 sweep, so it is not in the table above; run `label_eval --agent antigravity` below to
measure your own Antigravity model (such as `gemini-4-neon` or `gemini-3.8-flash`) on the same 60 COCO photos.

That is where the defaults come from:
- **Haiku at low effort**: as good as at medium (two passes at low found 77% and 78% before the final one), with
  less output and in less time. Eight photos a call would save 18% of the input but cost Haiku 4 to 5 points of
  found. Sonnet finds as much for twelve times the cost; Opus finds more but draws more boxes that are wrong.
- **Codex's default model at low effort**: medium adds nothing but output tokens and time. gpt-6.2-astra did as
  well as gpt-6.2-sol (within the noise), so the default stays Codex's own. Sixteen photos a call saved 9% of the
  input and found 1 to 2 points fewer in both passes.
- **Image size**: COCO's photos are at most 640 px, so the eval measured the effect of size at that scale. Sent at
  480 px (as 768 is to 1024), Codex labelled as well (85% / 80% / 0.89) and Haiku worse (75% / 87% / 0.80, from
  79% / 85% / 0.84). Scaled up to 1024 px, Haiku found 83% (at twice the input tokens), and Codex at 768 px did as
  well as at 1024 (87% against 86%) for 650 input tokens a photo instead of 1050.
- **Pixels, not thousandths.** Until November 2026 a reply's coordinates were thousandths of the image's width and
  height. The current Sonnet and Opus answer in pixels whatever they are told: Sonnet then found 5% of the objects
  and Opus 62%, Haiku and Codex as many as now. Asked for pixels of the image as sent, every model does well. A
  reply in the older object shape (`{"label", "box"}`) is read in pixels too: the prompt asks for pixels whatever
  shape a model answers in.

To measure your own setup, run the eval against the real CLIs (it spends about 60 images' worth of tokens a
pass; COCO val2017 and its annotations are a free download):

```sh
cargo run -p folo-server --release --example label_eval -- --coco /data/coco --agent claude --out eval.jsonl
cargo run -p folo-server --release --example label_eval -- --coco /data/coco --agent codex --out eval.jsonl
cargo run -p folo-server --release --example label_eval -- --coco /data/coco --agent antigravity --model gemini-4-neon --out eval.jsonl
cargo run -p folo-server --release --example label_eval -- --table eval.jsonl
```

Pass `--model`, `--effort`, `--batch`, `--side`, `--parallel` or `--fast` (Codex only) to try other settings. On
clean synthetic shapes, `cargo test -p folo-server --test label_live -- --ignored --nocapture claude` (or `codex` /
`antigravity`) checks each real CLI end to end (`FOLO_LABEL_LIVE_MODEL` picks a model). `npm run label-smoke` in
`web/` drives the whole flow through Studio against a running `folo serve` and a real agent, with a folder of your
own images, and prints what the agent reported using.

## Troubleshooting
| symptom / error | cause | what to do |
|---|---|---|
| Agent unavailable (`not_supported`) | `claude`, `codex` or `agy` is not on the server's `PATH`, or `--version` did not succeed | Install the CLI on the server's machine, or point `FOLO_CLAUDE_BIN`, `FOLO_CODEX_BIN` or `FOLO_ANTIGRAVITY_BIN` at the executable. Verify with `folo label agents` (probes are cached for 60 s). |
| Auth / sign-in failure (`401`, `/login`, `authentication required`, `printed no result` / `printed no reply` / `exited before replying`) | The CLI is signed out, expired, or blocked on an interactive prompt/update | Sign in on the server's machine: `claude`, `codex login`, or `agy` (`agy acp --login`). To see raw CLI output, run `claude -p hi`, `codex exec hi`, or `agy -p hi`. If Claude Code reports `unknown option`, run `claude update` (FOLO requires 2.1+). |
| Codex logs *"calls go without FOLO's lean model catalog and cost about 7800 more input tokens each"*, or Fast is disabled | `$CODEX_HOME/config.toml` (default `~/.codex/config.toml`) sets a top-level `model_catalog_json`, `profile`, `openai_base_url` or `chatgpt_base_url` or cannot be read (disables both the lean catalog and Fast); `codex debug models` failed or printed no readable catalog; that Codex version refused the generated `model_catalog_json` on start (remembered for that version; falls back without the lean catalog while Fast eligibility still follows the catalog); or the catalog lists no `priority` tier for the chosen model | For a config conflict, remove or move those top-level keys (keys inside `[profiles.*]` tables do not conflict), fix `config.toml` permissions/syntax, or use a clean `CODEX_HOME` (and run `codex login` under it unless using an API key). For a catalog probe failure or version-specific refusal, run `codex debug models` to inspect the error or update `codex`; for a model without `priority`, omit `--fast` / `fast` or pick a Fast-capable model from `folo label agents`. |
| Antigravity fails before or during a turn (settings, non-restricted mode, or `attempted to invoke a tool`) | FOLO writes `<home>/.gemini/antigravity-cli/folo-label/settings.json` (needs writable `HOME` / `USERPROFILE`) and fails closed if `agy acp` leaves an allowlisted restricted mode (`strict`, `default`, `request-review`, `ask`, `read-only`, `plan`) or emits `tool_call` / `tool_call_update` | Ensure `HOME` (Unix) or `USERPROFILE` (Windows) is set and writable by the server user, and update `agy` if it ignores `--agent folo-labeler --cli_dir_name folo-label --sandbox --include-default-customizations=false`. |
| Antigravity rejects model + effort (`already specifies effort ...`) | An effort-suffixed slug such as `gemini-4-neon-xhigh` was passed alongside a different explicit `--effort` / `effort` | Omit `--effort` / `effort` when using an effort-suffixed slug, or pass the base slug (e.g. `gemini-4-neon`) with `--effort`. |
| Run stops on account usage / quota limit (`usage limit`, `session limit`, `weekly limit`, `hour limit`, `credit balance`, `spending cap`, `out of extra usage`, or Google daily `*PerDay*` quota) | Hard plan or daily quota exhaustion; FOLO never retries these so it does not burn remaining quota | Wait until the reset time quoted in `run.hint` (from `label_status` or `folo label status --json`), or switch to another agent. |
| Run pauses or stops on rate limit (`429`, `RESOURCE_EXHAUSTED`) or `500`/`502`/`503`/`504`/`529` / overloaded | Per-minute rate limits with a parseable wait ≤ 60 s and transient server/connection faults retry once after `max(15 s, retry-after)` (at most 2 calls per batch); a rate limit with no parseable delay or a wait > 60 s stops immediately without retrying, and 3 consecutive calls that time out, return unreadable output, or fail again on retry end the run | Wait for any quoted backoff and restart with the default `unlabeled` scope and fewer calls in flight (`--parallel 1` or `2`), or switch agents. Previously labeled images stay saved unless you pass `--all` or explicit `--images` / `image_ids`. |
| `label_boxes_changed` (`409`) or `label_job_active` (`409`) | Boxes were modified since the `revision` passed in `expect`; or the job has an active agent run, an export in progress (even while `status` is `idle`), or an in-flight call holding that image | For `label_boxes_changed`, call `view_label_image` / `folo label view` again to get the new `revision` and box numbers. For `label_job_active`, wait for the export or in-flight call to finish, or stop the run with `stop_labeling` / `folo label stop`. |

## Getting better labels
- Write the instructions for someone who has never seen your data. Name look-alikes ("vans count as car"), say
  what to skip ("ignore cars on posters and in mirrors"), and say how tight a box should be if it matters.
- Keep class names plain and distinct. The agent sees only the name and your instructions.
- Label a few dozen images first, review them, adjust the instructions, then label the rest.
- Small or crowded objects are where the agents are weakest. Crop or tile such images before labeling if those
  objects are the point.
- Use "ask again" with a note for images that came back wrong for a reason you can say in a sentence; draw the
  box yourself when that is quicker.

## Limits
- Boxes only (no masks or keypoints), up to 80 classes and 10000 images per job.
- One agent run per job at a time; several jobs can run side by side.
- `folo serve --mock` has no agent labeling. Studio's own demo mode (`?mock=1`) simulates it for a look around.
- Not yet in FOLO Desktop: use Studio, `folo label`, the MCP tools or the HTTP API.
- A run lives in the process that started it: `folo label start` without a `folo serve` on the data dir, and
  `folo mcp` in embedded mode, keep running until it ends (the MCP server stops it when the client disconnects).
- An agent reviewing through MCP sees each image at 1024 px on the long side by default (`size` goes up to 1568):
  enough for most objects, but small ones in large photos may need a person, or a smaller `size` to save tokens
  where they do not matter.
- Label paths name files on the machine that runs the labeling (the server's, with `--server`); the MCP and CLI
  commands do not upload images. Upload them in Studio, or copy them there first. `folo mcp --confine` through a
  `folo serve` cannot check the files the server reads, so it does not add, label, view or export there.
- Against a `folo serve` older than this release (no `label_revisions` feature), a revision is checked just before
  the change is sent, not under the server's lock, and a review is one request per image; without
  `label_image_filters`, finding one image reads the job's image list 1000 at a time.
