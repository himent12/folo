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

/** The epochs the trainer suggests ("Train for at least N epochs"), if any warning names them. */
export function suggestedEpochs(warnings: readonly string[]): number | null {
  for (const w of warnings) {
    const m = /\bTrain for at least (\d+) epochs\b/i.exec(w);
    if (m) return Number(m[1]);
  }
  return null;
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
