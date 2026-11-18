# Agent labeling

You have images and no labels. FOLO can have a coding agent draw the boxes: it drives **Claude Code** or **Codex**
on your machine, shows you every box for review, and turns the result into a dataset that trains like any other.

```
unlabeled images ──► labeling job ──► agent run ──► review and correct ──► dataset ──► folo train
   (upload or a          (classes +      (Claude Code      (Studio)          (YOLO folder,
    folder)              instructions)    or Codex)                           registered)
```

The agents' boxes are a draft. On everyday photos they find most objects and place most boxes well (numbers
below), and they also miss things and box things that are not there. Plan to look through the result: the review
view is built for that.

## What you need
- `folo serve` (the real server, not `--mock`).
- At least one of the two CLIs on the machine that runs `folo serve`, signed in:
  - Claude Code: install it from <https://claude.com/claude-code>, then run `claude` once and sign in.
  - Codex: install it from <https://developers.openai.com/codex/cli>, then run `codex login`.

  FOLO finds `claude` and `codex` on the `PATH`; `FOLO_CLAUDE_BIN` and `FOLO_CODEX_BIN` name another executable.
  Studio shows which agents were found. Labeling uses that CLI's account: its plan limits and its bill.

The images are sent to the agent's provider (Anthropic or OpenAI), as with any use of those tools. Do not label
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
   image to review; the To review and Reviewed filters and the count in the header show how far you are. If the agent got an image wrong in a way you can describe, ask it again with a note ("the boxes are too
   loose", "that is a van, not a truck"). Failed images wait for the next run.
5. **Create dataset.** The labeled images become a YOLO dataset under the server's data dir, split into train and
   val, registered and ready for a training run. Images without labels are left out; a labeled image without
   boxes is kept as a background image. The job stays, so you can add images, label them and export again.

## From a script or an agent
Everything Studio does is the HTTP API ([API.md](API.md), Agent labeling):

```sh
api=http://127.0.0.1:7878/api
job=$(curl -s $api/label/jobs -H 'Content-Type: application/json' \
  -d '{"name": "street", "classes": ["car", "person"], "instructions": "Label every car and pedestrian.",
       "path": "/data/photos/street"}' | jq -r .id)
curl -s $api/label/jobs/$job/start -H 'Content-Type: application/json' -d '{"agent": "claude"}'
curl -sN $api/label/jobs/$job/events          # job / image / log events until you stop reading
curl -s $api/label/jobs/$job                  # status "idle" once the run has ended; counts, usage, run.outcome
curl -s $api/label/jobs/$job/dataset -H 'Content-Type: application/json' -d '{"val_fraction": 0.2}'
```

The last call answers with the new `Dataset`; its `id` goes into `POST /api/runs` or `folo train --data <path>`.

## What a run costs
FOLO makes **one model request per batch of images, with no tools**. That is the whole trick: an agent CLI left
to its defaults loads its coding instructions, tool definitions, your project's notes and MCP servers into every
request, and would read each image through a tool call, one request per step. FOLO switches all of that off and
attaches the images to the prompt.

| | Claude Code | Codex |
|---|---|---|
| Images per call (default) | 4 | 8 |
| Images sent at, long side at most | 1024 px | 768 px |
| Fixed input tokens per call | about 900 (FOLO's instructions) | about 700 (FOLO's instructions and reply schema; 20000 with Codex's defaults) |
| Input tokens per photo, all told | about 1280 for a photo of 1024 px or more, 620 for a 640 px one | about 650 for a photo of 768 px or more, 460 for a 640 px one |
| Output tokens per photo | about 165 (Haiku at low effort, mostly its reasoning) | about 40 |
| Calls in flight | 3 | 3 |

Larger images cost more, up to a cap: images are scaled down before they are sent, so a 12-megapixel photo costs
what a 1024 px one does (768 px for Codex). The prompt gives each image's size as sent, the boxes come back in
pixels of that, and they are stored in pixels of the original file. The sizes are where labels stop improving
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
  and `request_user_input` switched off: a call's own overhead went from about 7800 tokens to about 50.
- A box is a list, `["car", x1, y1, x2, y2]`, instead of `{"label": "car", "box": [x1, y1, x2, y2]}`: Codex
  writes 25% fewer output tokens for the same labels.
- Haiku runs at low effort and Codex sends images at 768 px (the reasons are in the next section).

**Fast (Codex).** Codex offers a faster service tier (what it calls Fast: priority processing, which uses more of
the plan's usage). Studio's agent panel has a Fast switch for Codex, and the API takes `"fast": true`. On the 60
photos it took 28 s instead of 35, with the same labels and tokens. Claude Code offers no fast mode to `claude -p`.

A run stops spending as soon as spending is pointless. If the CLI is missing, signed out, refuses the model or
hits a rate limit or its plan's usage limit, the run ends on that call with the CLI's own message (and, for a
limit, when it resets), and the images it did not reach stay as they were. A passing fault of the provider (it is
overloaded, a server error, the connection dropped) is asked once more after 15 seconds; failing that way twice,
the call's images fail. A reply that cannot be read is asked for once more, never in a loop, and one bad box in a
reply costs only that box. Three calls in a row that time out, return garbage or fail twice on a passing fault
end the run.

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
Sonnet medium and Opus rows with an earlier wording of the image sizes in the prompt.

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
  and Opus 62%, Haiku and Codex as many as now. Asked for pixels of the image as sent, every model does well.

To measure your own setup, run the eval against the real CLIs (it spends about 60 images' worth of tokens a
pass; COCO val2017 and its annotations are a free download):
`cargo run -p folo-server --release --example label_eval -- --coco <coco dir> --agent claude --out eval.jsonl`,
with `--model`, `--effort`, `--batch`, `--side` and `--fast` to try other settings, and `--table eval.jsonl` to
print the passes side by side. On clean synthetic shapes `cargo test -p folo-server --test label_live --
--ignored --nocapture` checks both CLIs end to end (`FOLO_LABEL_LIVE_MODEL` picks a model). `npm run
label-smoke` in `web/` drives the whole flow through Studio against a running `folo serve` and a real agent, with
a folder of your own images, and prints what the agent reported using.

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
- Not yet in FOLO Desktop, the `folo` CLI or the MCP server: use Studio or the HTTP API.
