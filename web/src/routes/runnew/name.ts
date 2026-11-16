// A run's name: what was typed becomes a tidy one (the server takes any text, but the name travels into file names,
// commands and every list of runs and models), at most as long as a dataset's.

/** The longest run name the server takes (docs/API.md, POST /api/runs), as for a dataset's. */
export const MAX_RUN_NAME = 64;

/**
 * What a typed name is saved as: each run of characters other than letters (any script, with their accents), digits,
 * ".", "_" and "-" becomes one "-", and dashes at the ends go ("my first run!" → "my-first-run", "名前 v2" →
 * "名前-v2"). Empty when nothing usable is left. FOLO Desktop and the server keep such letters too.
 */
export function runName(raw: string): string {
  return raw
    .trim()
    .replace(/[^\p{L}\p{M}\p{N}_.-]+/gu, '-')
    .replace(/^-+|-+$/g, '');
}

/** A name's length as the server counts it (characters, crates/folo-server/src/store.rs), not UTF-16 units. */
export const nameLength = (n: string) => [...n].length;

/** The first `max` characters of `n`, never half of one (a letter outside the BMP is two UTF-16 units). */
export const clipName = (n: string, max: number) => [...n].slice(0, Math.max(0, max)).join('');

/** `base`, else `base-2`, `base-3`, … : the first not `taken`, shortened to fit the length limit. */
export function uniqueName(base: string, taken: Set<string>): string {
  const fit = (sfx: string) => `${clipName(base, MAX_RUN_NAME - nameLength(sfx)).replace(/-+$/, '')}${sfx}`;
  let name = fit('');
  for (let k = 2; taken.has(name); k++) name = fit(`-${k}`);
  return name;
}

/**
 * The line under Run name: a run of that name exists already (two runs no one can tell apart, and two models of the
 * same name), or what the typed name is saved as. Null when neither.
 */
export function nameHint(raw: string, taken: Set<string>, edited: boolean): string | null {
  const n = runName(raw);
  if (!n) return null;
  if (edited && taken.has(n)) return `A run called ${n} already exists. Pick another name to tell them apart.`;
  return n !== raw.trim() && nameLength(n) <= MAX_RUN_NAME ? `Saved as ${n}: letters, digits, dot, dash and underscore.` : null;
}
