// The trainer's warnings that a run is too short to learn from (folo_vision::train::schedule_warnings, docs/MODEL.md
// §8.4). It writes them as `warn` log lines straight after the run's first line (and as `warnings` in config.json, which
// the API does not serve), so the run page finds them in the log.

import type { LogLine } from '$lib/api/types';

/** How each schedule warning starts, as the engine words it. */
const SCHEDULE = [/^this run takes only \d+ optimizer steps\b/, /^warmup takes \d+ of the run's \d+ optimizer steps\b/];

/** The schedule warnings among `logs`, once each, in the order the trainer wrote them. */
export function scheduleWarnings(logs: readonly LogLine[]): string[] {
  const out: string[] = [];
  for (const l of logs) if (l.level === 'warn' && SCHEDULE.some((re) => re.test(l.message)) && !out.includes(l.message)) out.push(l.message);
  return out;
}

/** The optimizer steps a run from scratch needs before it learns anything (folo_vision::train::SHORT_RUN_STEPS). */
export const SHORT_RUN_STEPS = 200;

/** The short-run warning's numbers: "this run takes only 125 optimizer steps (5 epochs × 25: …)". */
const SHORT = /^this run takes only (\d+) optimizer steps \((\d+) epochs × (\d+):/;
/** The warmup warning's numbers: "warmup takes 75 of the run's 125 optimizer steps". */
const WARMUP = /^warmup takes (\d+) of the run's (\d+) optimizer steps\b/;

/**
 * The epochs that clear the short-run warning: ⌈200 / optimizer steps per epoch⌉, the engine's rule
 * (`SHORT_RUN_STEPS.div_ceil(spe)`), else the epochs its sentence names ("Train for at least N epochs"); null if none.
 */
export function minimumEpochs(warnings: readonly string[]): number | null {
  for (const w of warnings) {
    const s = SHORT.exec(w);
    if (s && Number(s[3]) > 0) return Math.ceil(SHORT_RUN_STEPS / Number(s[3]));
    const m = /\bTrain for at least (\d+) epochs\b/i.exec(w);
    if (m) return Number(m[1]);
  }
  return null;
}

/** The epoch counts a suggestion picks from: round ones, up to the New run form's limit. */
const ROUND_EPOCHS = [10, 20, 30, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000];

/**
 * The epochs the warning's action suggests for a run of `epochs`: a clearly larger, round count, at least the minimum
 * that clears the warning (`minimumEpochs`) and twice the run's own, never one epoch more ("Train 16 epochs" after 15
 * reads like a rounding glitch). Null when the warning names no minimum.
 */
export function suggestedEpochs(warnings: readonly string[], epochs: number): number | null {
  const min = minimumEpochs(warnings);
  if (min === null) return null;
  const want = Math.max(min, 2 * epochs);
  return ROUND_EPOCHS.find((n) => n >= want) ?? ROUND_EPOCHS[ROUND_EPOCHS.length - 1]!;
}

/**
 * A run whose best mAP50-95 reached this learned: its short-run warning says more epochs would likely improve it, not
 * that it was too short to learn, nor that its mAP will be near 0.
 */
export const LEARNED_MAP = 0.2;

/**
 * The banner's title: "Too short to learn" only when the engine says so (too few optimizer steps from scratch) and the
 * run did not prove it wrong (`learned`: "More epochs would likely improve this"); a run warned only about its warmup
 * (e.g. fine-tuned, with enough steps) is "Mostly warmup".
 */
export function scheduleTitle(warnings: readonly string[], learned = false): string {
  if (!warnings.some((w) => SHORT.test(w))) return 'Mostly warmup:';
  return learned ? 'More epochs would likely improve this:' : 'Too short to learn:';
}

/** The banner's one-line cause: "125 optimizer steps", or the warmup's share when only that warning is present. */
export function shortCause(warnings: readonly string[]): string {
  for (const w of warnings) {
    const s = SHORT.exec(w);
    if (s) return `${s[1]} optimizer steps`;
  }
  for (const w of warnings) {
    const m = WARMUP.exec(w);
    if (m) return `warmup takes ${m[1]} of ${m[2]} optimizer steps`;
  }
  return '';
}

/**
 * An engine warning in plain words, as sentences: the config keys it names become what they mean ("lr0" the initial
 * learning rate, "warmup_epochs" the warmup), so the "Why?" behind a banner needs no knowledge of the config file. For a
 * run that `learned`, without the claim that it learns nothing.
 */
export function plain(message: string, learned = false): string {
  const text = (learned ? message.replace(/\bto learn anything, so expect an mAP near 0\b/, 'before it stops improving') : message)
    .replace(/\(\d+ epochs × (\d+): /, (_, spe: string) => `(${spe} per epoch: `)
    .replace(/\bthe learning rate reaches lr0\b/g, 'the learning rate reaches its initial value')
    .replace(/\blr0\b/g, 'the initial learning rate')
    .replace(/\blower warmup_epochs \(now ([\d.]+)\)/g, (_, n: string) => `shorten the warmup (now ${n} ${n === '1' ? 'epoch' : 'epochs'})`)
    .replace(/\bwarmup_epochs\b/g, 'the warmup');
  return sentences(text);
}

/** An engine message as sentences: "a; b" becomes "A. B.", capitalised, ending on a full stop. */
export function sentences(message: string): string {
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
  const body = message
    .trim()
    .split(/;\s+/)
    .map((s) => cap(s.replace(/[.\s]+$/, '')))
    .join('. ');
  return body ? `${body}.` : '';
}
