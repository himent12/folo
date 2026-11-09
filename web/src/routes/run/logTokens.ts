// A log message as the pieces a line may wrap between. Mono log lines wrap only at spaces, and a `key value` pair
// ("mAP50-95 0.588", "loss 2.221") or a flag with its value ("--workers 12") stays whole, so no metric is split from
// its value and no hyphen ("mAP50-⏎95", "--⏎workers") ends a line. Pieces too long for any line (a path) may still
// break inside, as a last resort.

export interface LogPiece {
  text: string;
  /** Whitespace between pieces: a place the line may wrap. */
  space?: boolean;
  /** Longer than LONG: may break inside if no line holds it. */
  long?: boolean;
}

/** Pieces longer than this may break inside: a phone's log line holds about 40 mono characters. */
export const LONG = 32;

const VALUE = /^[-+]?(\d[\d.,:/]*|\.\d+)(e[-+]?\d+)?[a-z%]*[,;)]?$/i;
const KEY = /^[(]?[a-z][\w.\-/]*[:=]?$/i;
const FLAG = /^--?[a-z][\w-]*$/i;

export function logPieces(message: string): LogPiece[] {
  const parts = message.split(/(\s+)/).filter((p) => p !== '');
  const out: LogPiece[] = [];
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i]!;
    if (/^\s+$/.test(p)) {
      out.push({ text: p, space: true });
      continue;
    }
    const sep = parts[i + 1];
    const next = parts[i + 2];
    // A key and its value, or a flag and its argument, joined by one space: one piece.
    const pair = sep === ' ' && next !== undefined && ((FLAG.test(p) && !next.startsWith('-')) || (KEY.test(p) && !FLAG.test(p) && VALUE.test(next)));
    const text = pair ? `${p} ${next}` : p;
    if (pair) i += 2;
    out.push(text.length > LONG ? { text, long: true } : { text });
  }
  return out;
}
