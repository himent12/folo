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
export function suggestedEpochs(warnings: readonly string[]): number | null {
  for (const w of warnings) {
    const s = SHORT.exec(w);
    if (s && Number(s[3]) > 0) return Math.ceil(SHORT_RUN_STEPS / Number(s[3]));
    const m = /\bTrain for at least (\d+) epochs\b/i.exec(w);
    if (m) return Number(m[1]);
  }
  return null;
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
 * learning rate, "warmup_epochs" the warmup), so the "Why?" behind a banner needs no knowledge of the config file.
 */
export function plain(message: string): string {
  const text = message
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
