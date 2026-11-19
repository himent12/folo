// Agent labeling: agents, jobs, images and their events (docs/API.md, Agent labeling).

import type { Box } from './datasets';
import type { LogLine } from './runs';
import type { Time } from './time';

// ---------- Agent labeling (docs/API.md); servers that have it list `agent_labeling` in `SystemInfo.features` ----------

/** A coding-agent CLI on the server's machine that FOLO can ask for labels. */
export type LabelAgentId = 'claude' | 'codex';

/** A model of a `LabelAgent`: `id` is what the CLI takes, `label` the display name. */
export interface LabelModel {
  id: string;
  label: string;
  /** It offers the agent's faster service tier (`StartLabeling.fast`); absent when it does not. */
  fast?: boolean;
}

/** `GET /api/label/agents`: one agent CLI and whether the server found it. */
export interface LabelAgent {
  id: LabelAgentId;
  /** Product name: "Claude Code", "Codex". */
  name: string;
  /** Its CLI is installed and answered `--version`; whether it is signed in is only known once a run calls it. */
  available: boolean;
  version: string | null;
  /** Suggestions: a run takes any model name the CLI accepts. */
  models: LabelModel[];
  /** The model a run uses when it names none; null: the CLI's own default. */
  default_model: string | null;
  /** The effort a run uses when it names none: what labels best for its cost with this agent. */
  default_effort: LabelEffort;
  /**
   * Its default model offers a faster service tier that uses more of the account's usage (`StartLabeling.fast`):
   * Codex's Fast. Which listed models offer it is `LabelModel.fast`. Absent from older servers.
   */
  fast?: boolean;
  /** How to make the agent available (install, sign in); null when it is. */
  hint: string | null;
}

export type LabelJobStatus = 'idle' | 'running' | 'stopping';

/** `pending + labeled + failed == images`. */
export interface LabelCounts {
  images: number;
  pending: number;
  labeled: number;
  /** Images whose status is `failed` (a labeled image whose call fails stays labeled, with an `error`). */
  failed: number;
  boxes: number;
}

export type LabelEffort = 'low' | 'medium' | 'high';
export type LabelOutcome = 'completed' | 'stopped' | 'failed';

/** Tokens the agent CLIs reported. `cost_usd` is null when no call reported one (Codex reports none). */
export interface LabelUsage {
  calls: number;
  /** Cached ones included. */
  input_tokens: number;
  cached_input_tokens: number;
  /** Reasoning included. */
  output_tokens: number;
  cost_usd: number | null;
}

/** One agent run over some images of a `LabelJob`. */
export interface LabelRun {
  /** Unique per run; absent from older servers (`started` has one-second resolution). */
  id?: string;
  agent: LabelAgentId;
  /** null: the CLI's default. */
  model: string | null;
  effort: LabelEffort;
  /** It asked for the faster service tier; absent when it did not. */
  fast?: boolean;
  started: Time;
  /** null while the run is active. */
  finished: Time | null;
  total: number;
  done: number;
  /** Images whose call failed in this run, labeled ones that kept their labels too (unlike `counts.failed`). */
  failed: number;
  /** null while the run is active. */
  outcome: LabelOutcome | null;
  /** Why it failed (`outcome: failed`), else null. */
  error: string | null;
  /** How to fix `error`, when there is something useful to say. */
  hint?: string | null;
  usage: LabelUsage;
}

/** A labeling job: a set of images, the classes to find in them, and the labels so far. */
export interface LabelJob {
  id: string;
  name: string;
  /** A box's `class` indexes this list. */
  classes: string[];
  /** What the agent is told, in the user's words (may be empty). */
  instructions: string;
  created: Time;
  status: LabelJobStatus;
  counts: LabelCounts;
  /** The active run, else the most recent one; null before the first. */
  run: LabelRun | null;
  /** Every run of the job. */
  usage: LabelUsage;
  /** The dataset most recently made from the job. */
  dataset_id: string | null;
}

export type LabelImageStatus = 'pending' | 'labeling' | 'labeled' | 'failed';
/** Who drew an image's current boxes. */
export type LabelSource = 'agent' | 'manual';

export interface LabelImage {
  id: string;
  /** Unique within the job; the dataset keeps it. */
  file: string;
  width: number;
  height: number;
  status: LabelImageStatus;
  /** Pixel xyxy; kept while the image is labeled again. */
  boxes: Box[];
  /** null until labeled. */
  source: LabelSource | null;
  /** Why the last agent call for it failed: a `failed` image, or a `labeled` one that keeps the labels (and review) it had. */
  error: string | null;
  /**
   * A person checked its labels (saved boxes drawn by hand, or marked them right); only a `labeled` image is. An
   * agent that labels it again clears it. Absent from older servers.
   */
  reviewed?: boolean;
}

/** `POST /api/label/jobs`. */
export interface CreateLabelJob {
  name: string;
  classes: string[];
  instructions?: string;
  /** A folder of images on the server's machine to start with. */
  path?: string;
}

/** `PUT /api/label/jobs/{id}`: absent fields stay. Once the job has boxes, classes may only be renamed or appended. */
export interface UpdateLabelJob {
  name?: string;
  classes?: string[];
  instructions?: string;
}

/** A file `POST /api/label/jobs/{id}/images` did not add, and why. */
export interface SkippedImage {
  file: string;
  reason: string;
}

/** `POST /api/label/jobs/{id}/images`. */
export interface LabelImagesAdded {
  added: LabelImage[];
  skipped: SkippedImage[];
  job: LabelJob;
}

export interface LabelImagesQuery {
  status?: LabelImageStatus;
  offset?: number;
  /** Default 100, at most 1000. */
  limit?: number;
  /** Only these images (feature `label_image_filters`, which every server with `label_revisions` has). */
  ids?: readonly string[];
}

/** `GET /api/label/jobs/{id}/images`: `total` counts every match before offset / limit; in the order added. */
export interface LabelImagePage {
  total: number;
  images: LabelImage[];
}

/** `unlabeled`: pending and failed images (the default); `all`: every image, replacing its labels. */
export type LabelScope = 'unlabeled' | 'all';

/** `POST /api/label/jobs/{id}/start`: only `agent` is required. */
export interface StartLabeling {
  agent: LabelAgentId;
  model?: string | null;
  /** Absent or null: the agent's `default_effort`. */
  effort?: LabelEffort | null;
  /**
   * The agent's faster service tier, using more of the account's usage; only with a model that offers it
   * (`LabelModel.fast`, or `LabelAgent.fast` for the default model).
   */
  fast?: boolean;
  scope?: LabelScope;
  /** Exactly these images, whatever their status, instead of `scope`. */
  image_ids?: string[] | null;
  /** An extra instruction for this run only (at most 2000 characters). */
  note?: string;
  batch?: number | null;
  parallel?: number | null;
}

/** `POST /api/label/jobs/{id}/dataset`. */
export interface LabelDatasetRequest {
  /** Defaults to the job's name. */
  name?: string | null;
  /** In 0..=0.5; default 0.2. */
  val_fraction?: number;
  seed?: number;
}

/** SSE events on GET /api/label/jobs/{id}/events, keyed by the SSE `event:` name. */
export interface LabelEventMap {
  job: LabelJob;
  image: LabelImage;
  removed: { id: string };
  /** More changed than the stream carries (a bulk add, a client that fell behind): load the images again. */
  reload: Record<string, never>;
  log: LogLine;
}
