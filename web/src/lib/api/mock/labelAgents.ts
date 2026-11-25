// Agent catalog, request validation, constants, and internal state types for mock agent labeling (`./label.ts`).
import { AGENT_NAME } from '../../label';
import { ApiError } from '../client';
import type { Box, LabelAgent, LabelAgentId, LabelEffort, LabelImageStatus, LabelOutcome, LabelUsage, LogLine } from '../types';
import { round } from './boxes';
import type { MockDatasetSpec } from './datasets';

/** One agent call ends every this many ms. */
export const BATCH_MS = 1500;
export const MIN = 60_000;
export const HOUR = 60 * MIN;
/** A job's image limit (docs/API.md), and why an image past it is skipped. */
export const MAX_IMAGES = 10_000;
export const FULL = `a job holds at most ${MAX_IMAGES} images`;
/** Over this many images added at once, the stream says `reload` instead of one `image` event each (docs/API.md). */
export const BULK = 100;

/** What `start` takes at most (docs/API.md): a model name's characters, images per call, calls at a time. */
export const MODEL_MAX = 100;
export const BATCH_MAX = 16;
export const PARALLEL_MAX = 8;

/** Whether an optional count is absent or a whole number in `1..=max`. */
export const inRange = (v: number | null | undefined, max: number) => v === null || v === undefined || (Number.isInteger(v) && v >= 1 && v <= max);

const EFFORT_SUFFIXES = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'] as const;

/** Mirrors the server's `check_model`: trims a model name and refuses invalid characters or length. */
export function checkMockModel(rawModel: string | null | undefined): string | null {
  const model = rawModel?.trim() || null;
  if (model && !/^[A-Za-z0-9._:/[\]-]{1,100}$/.test(model))
    throw new ApiError(400, 'invalid_body', `model must be at most ${MODEL_MAX} characters of A-Z a-z 0-9 . _ : / - [ ]`);
  return model;
}

/** Mirrors the server's `Agent::resolve_request`: validates model, maps Antigravity effort-suffixed slugs, and rejects conflicts. */
export function resolveAgentRequest(
  agent: LabelAgentId,
  rawModel: string | null | undefined,
  explicitEffort: LabelEffort | null | undefined,
  defaultModel: string | null,
  defaultEffort: LabelEffort,
): { model: string | null; effort: LabelEffort } {
  const model = checkMockModel(rawModel);
  if (agent === 'antigravity' && model) {
    const lower = model.toLowerCase();
    for (const suffix of EFFORT_SUFFIXES) {
      const tail = `-${suffix}`;
      if (lower.endsWith(tail) && model.length > tail.length) {
        const base = model.slice(0, model.length - tail.length);
        const mapped: LabelEffort = suffix === 'minimal' || suffix === 'low' ? 'low' : suffix === 'medium' ? 'medium' : 'high';
        if (explicitEffort && explicitEffort !== mapped) {
          throw new ApiError(
            400,
            'invalid_body',
            `model '${model}' already specifies effort '${suffix}', which conflicts with effort '${explicitEffort}': omit effort or pass the base model '${base}'`,
          );
        }
        return { model, effort: mapped };
      }
    }
  }
  return { model: model ?? defaultModel, effort: explicitEffort ?? defaultEffort };
}

/** Images per call when a run names none (docs/API.md: Claude Code 4, Codex 8, Antigravity 8). */
export const DEFAULT_BATCH: Record<LabelAgentId, number> = { claude: 4, codex: 8, antigravity: 8 };
/** Claude Code's list prices per million tokens, input and output (Codex reports no cost). */
export const PRICE: Record<string, [number, number]> = { sonnet: [3, 15], haiku: [1, 5], opus: [15, 75] };
/** How well each model finds the objects (the share a call gets right, roughly). */
export const QUALITY: Record<string, number> = {
  sonnet: 0.86,
  haiku: 0.74,
  opus: 0.92,
  'gemini-4-neon': 0.91,
  'gemini-4-hydrogen': 0.88,
  'gemini-4-carbon': 0.84,
  'gemini-4-argon': 0.8,
  'gemini-3.8-flash': 0.78,
};

/**
 * The agents a server finds, with the real server's models and defaults: Claude Code installed; Codex and Antigravity
 * not installed (their hints say how), except in `all-agents`, where all three are.
 */
export function mockAgents(codex: boolean): LabelAgent[] {
  return [
    {
      id: 'claude',
      name: AGENT_NAME.claude,
      available: true,
      version: '2.1.302',
      models: [
        { id: 'haiku', label: 'Haiku' },
        { id: 'sonnet', label: 'Sonnet' },
        { id: 'opus', label: 'Opus' },
      ],
      default_model: 'haiku',
      default_effort: 'low',
      fast: false,
      hint: null,
    },
    {
      id: 'codex',
      name: AGENT_NAME.codex,
      available: codex,
      version: codex ? '0.58.0' : null,
      models: [
        { id: 'gpt-6.1-sol', label: 'GPT-6.1-Sol', fast: true },
        { id: 'gpt-6-astra', label: 'GPT-6-Astra', fast: true },
        { id: 'gpt-6.1-terra', label: 'GPT-6.1 Terra', fast: true },
        { id: 'gpt-6-luna', label: 'GPT-6-Luna', fast: true },
      ],
      default_model: null,
      default_effort: 'low',
      fast: true,
      hint: codex ? null : 'Install Codex (npm install -g @openai/codex) and sign in with `codex login`, or set FOLO_CODEX_BIN to its path.',
    },
    {
      id: 'antigravity',
      name: AGENT_NAME.antigravity,
      available: codex,
      version: codex ? '1.5.13' : null,
      models: [
        { id: 'gemini-4-neon', label: 'Gemini 4 Neon' },
        { id: 'gemini-4-hydrogen', label: 'Gemini 4 Hydrogen' },
        { id: 'gemini-4-carbon', label: 'Gemini 4 Carbon' },
        { id: 'gemini-4-argon', label: 'Gemini 4 Argon' },
        { id: 'gemini-3.8-flash', label: 'Gemini 3.8 Flash' },
      ],
      default_model: null,
      default_effort: 'low',
      fast: false,
      hint: codex
        ? null
        : 'Install Antigravity CLI (https://antigravity.google/docs/getting-started) and sign in by running `agy` once, or set FOLO_ANTIGRAVITY_BIN to its path.',
    },
  ];
}

export interface Img {
  id: string;
  file: string;
  width: number;
  height: number;
  status: LabelImageStatus;
  boxes: Box[];
  source: 'agent' | 'manual' | null;
  error: string | null;
  /** A person checked its labels (reported for a labeled image only). */
  reviewed: boolean;
  /** Its pixels: an image of a mock dataset, or a file uploaded in this page. */
  from: { spec: MockDatasetSpec; index: number } | { blob: Blob; key: string };
  /** What it was before an agent call took it: a stopped call gives it back. */
  before: { status: LabelImageStatus; error: string | null } | null;
  /** Agent calls that have failed for it: the mock's agent fails an image once at most. */
  failures: number;
}

export interface Run {
  agent: LabelAgentId;
  model: string | null;
  effort: LabelEffort;
  fast: boolean;
  note: string;
  startedAt: number;
  finishedAt: number | null;
  ids: string[];
  /** Images changed by hand while they waited for their call: they left the run. */
  left: Set<string>;
  batch: number;
  parallel: number;
  /** Calls answered so far. */
  applied: number;
  done: number;
  failed: number;
  outcome: LabelOutcome | null;
  error: string | null;
  hint: string | null;
  usage: LabelUsage;
  /** Set by a stop: when the calls in flight have been ended. */
  stopAt: number | null;
}

export interface Job {
  id: string;
  name: string;
  classes: string[];
  instructions: string;
  createdAt: number;
  images: Img[];
  run: Run | null;
  usage: LabelUsage;
  datasetId: string | null;
  logs: LogLine[];
  nextImage: number;
  /** Runs started so far: what seeds a run's boxes (never the clock, so every load draws the same ones). */
  runs: number;
}

export const zeroUsage = (): LabelUsage => ({ calls: 0, input_tokens: 0, cached_input_tokens: 0, output_tokens: 0, cost_usd: null });

export function addUsage(a: LabelUsage, b: LabelUsage) {
  a.calls += b.calls;
  a.input_tokens += b.input_tokens;
  a.cached_input_tokens += b.cached_input_tokens;
  a.output_tokens += b.output_tokens;
  if (b.cost_usd !== null) a.cost_usd = round((a.cost_usd ?? 0) + b.cost_usd, 6);
}

export const iso = (ms: number) => new Date(ms).toISOString();
/** `word` for `k` of them, as the server's log writes it. */
export const plural = (k: number, word: string) => (k === 1 ? word : word === 'box' ? 'boxes' : `${word}s`);
/** Why an image the reply did not mention failed (the server's words). */
export const LEFT_OUT = "the agent's reply left this image out";
