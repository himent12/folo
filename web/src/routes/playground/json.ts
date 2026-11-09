// Compact, readable JSON for the Playground's JSON tab: like JSON.stringify(v, null, 2), except that small flat
// objects (a box, the timing) stay on one line, so a list of detections reads as one row per box.

export type JsonToken = { t: 'key' | 'str' | 'num' | 'lit' | 'punct' | 'ws'; v: string };

const flat = (v: unknown): v is Record<string, string | number | boolean | null> =>
  typeof v === 'object' && v !== null && !Array.isArray(v) && Object.values(v).every((x) => x === null || typeof x !== 'object');

function scalar(v: unknown, out: JsonToken[]) {
  if (typeof v === 'string') out.push({ t: 'str', v: JSON.stringify(v) });
  else if (typeof v === 'number') out.push({ t: 'num', v: Number.isFinite(v) ? String(v) : 'null' });
  else out.push({ t: 'lit', v: String(v) });
}

function emit(v: unknown, indent: string, out: JsonToken[]) {
  if (v === null || typeof v !== 'object') return scalar(v, out);
  const isArr = Array.isArray(v);
  const entries: [string | null, unknown][] = isArr ? v.map((x) => [null, x]) : Object.entries(v);
  const [open, close] = isArr ? ['[', ']'] : ['{', '}'];
  if (entries.length === 0) return out.push({ t: 'punct', v: open + close });
  if (!isArr && flat(v) && entries.length <= 10) {
    out.push({ t: 'punct', v: '{ ' });
    entries.forEach(([k, x], i) => {
      out.push({ t: 'key', v: JSON.stringify(k) }, { t: 'punct', v: ': ' });
      scalar(x, out);
      if (i < entries.length - 1) out.push({ t: 'punct', v: ', ' });
    });
    out.push({ t: 'punct', v: ' }' });
    return;
  }
  const inner = indent + '  ';
  out.push({ t: 'punct', v: open }, { t: 'ws', v: '\n' });
  entries.forEach(([k, x], i) => {
    out.push({ t: 'ws', v: inner });
    if (k !== null) out.push({ t: 'key', v: JSON.stringify(k) }, { t: 'punct', v: ': ' });
    emit(x, inner, out);
    if (i < entries.length - 1) out.push({ t: 'punct', v: ',' });
    out.push({ t: 'ws', v: '\n' });
  });
  out.push({ t: 'ws', v: indent }, { t: 'punct', v: close });
}

/** Tokens for syntax tinting. Joining their `v` gives valid JSON. */
export function jsonTokens(v: unknown): JsonToken[] {
  const out: JsonToken[] = [];
  emit(v, '', out);
  return out;
}

export const jsonText = (v: unknown) =>
  jsonTokens(v)
    .map((t) => t.v)
    .join('');

export interface JsonLine {
  /** Leading spaces: a wrapped line's continuation hangs one step deeper than this. */
  indent: number;
  tokens: JsonToken[];
}

/** The tokens split into lines (no newline tokens), each with its indent, for a view that wraps long lines. */
export function jsonLines(tokens: readonly JsonToken[]): JsonLine[] {
  const lines: JsonLine[] = [{ indent: 0, tokens: [] }];
  for (const tk of tokens) {
    if (tk.t === 'ws' && tk.v === '\n') lines.push({ indent: 0, tokens: [] });
    else if (tk.t === 'ws' && lines.at(-1)!.tokens.length === 0) lines.at(-1)!.indent += tk.v.length;
    else lines.at(-1)!.tokens.push(tk);
  }
  return lines;
}
