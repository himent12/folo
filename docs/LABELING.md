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
4. **Review.** Open an image to move, resize, relabel, delete or draw boxes; saving marks it as labeled by hand
   and reviewed. When the agent's boxes are right, **Looks right** (Enter) marks them reviewed and opens the next
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
       "path": "D:/photos/street"}' | jq -r .id)
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
| Fixed input tokens per call | a few hundred (FOLO's instructions) | about 8000 (Codex's own; 20000 without FOLO's switches) |
| Input tokens per image | about 600 for a 640 px photo | about 400 for a 640 px photo |
| Calls in flight | 3 | 3 |

Larger images cost more, up to a cap: images are scaled down to at most 1024 px on the long side before they are
sent, so a 12-megapixel photo costs what a 1024 px one does. The boxes come back in relative coordinates and are
stored in pixels of the original file.

The fixed cost is per call, however few images it has: asking Codex again about a single image costs its 8000
tokens for that one image, where Claude Code's call costs little more than the image.

A run stops spending as soon as spending is pointless. If the CLI is missing, signed out, refuses the model or
hits a rate limit, the run ends on that call with the CLI's own message, and the images it did not reach stay as
they were. A reply that cannot be read is asked for once more, never in a loop, and one bad box in a reply costs
only that box. Three calls in a row that time out or return garbage end the run.

## How good the labels are
Measured through `folo serve` on 24 COCO val2017 photos with 73 objects of 10 everyday classes (person, car,
chair, dog, bottle, cup, tv, couch, bicycle, bus), against COCO's own boxes; Claude Code 2.1.302 and codex-cli
0.167.0, November 2026. *Found* is the share of true boxes matched by a box of the same class at IoU 0.5,
*right* the share of the agent's boxes that match one, *IoU* the mean overlap of the matches. With 73 objects a
difference of a few points is noise.

| Agent, model, effort | Found | Right | IoU | Input / output tokens | Reported cost |
|---|---:|---:|---:|---:|---:|
| Claude Code, Haiku, medium (**default**) | 79% | 92% | 0.88 | 16300 / 8000 | $0.008 |
| Claude Code, Haiku, high | 84% | 90% | 0.88 | 16300 / 11800 | $0.008 |
| Claude Code, Sonnet, medium | 67% | 69% | 0.85 | 16200 / 2200 | $0.064 |
| Claude Code, Opus, medium | 89% | 88% | 0.88 | 16100 / 2900 | $0.19 |
| Codex, its default model, low (**default**) | 88% | 85% | 0.91 | 32800 / 1700 | not reported |
| Codex, its default model, medium | 90% | 83% | 0.92 | 32800 / 3100 | not reported |
| Codex, GPT-6-Luna, low | 64% | 82% | 0.80 | 32800 / 1100 | not reported |

That is where the defaults come from: Haiku at medium effort labels about as well as the larger Claude models at
a small fraction of their cost (at low effort it found 47% in a smaller sample, so low is not the default), and
Codex's default model is already good at low effort. On clean synthetic shapes
(`crates/folo-server/tests/label_live.rs`) both defaults find every object at a mean IoU of 0.90 and 0.93.

To measure your own setup, run that test against the real CLIs:
`cargo test -p folo-server --test label_live -- --ignored --nocapture` (`FOLO_LABEL_LIVE_MODEL` picks a model).
`npm run label-smoke` in `web/` drives the whole flow through Studio against a running `folo serve` and a real
agent, with a folder of your own images, and prints what the agent reported using.

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
