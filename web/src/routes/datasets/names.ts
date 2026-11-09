// A dataset's name as the server takes it (docs/API.md, POST /api/datasets): letters, digits, ".", "_" and "-", starting
// with a letter or digit, at most 64 characters.

export const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

/**
 * The nearest valid name to what was typed, for a one-click fix: each run of other characters becomes one "-", and
 * the ends are trimmed ("my dataset!" → "my-dataset"); null when nothing usable is left.
 */
export function suggestName(raw: string): string | null {
  const n = raw
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^[^A-Za-z0-9]+/, '')
    .slice(0, 64)
    .replace(/[-._]+$/, '');
  return NAME_RE.test(n) ? n : null;
}
