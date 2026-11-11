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

/**
 * A folder path as typed or pasted: surrounding whitespace and one pair of matching quotes removed (Windows' "Copy as
 * path" adds them), as the server does.
 */
export function cleanPath(raw: string): string {
  const t = raw.trim();
  const m = /^(["'])(.*)\1$/s.exec(t);
  return (m ? m[2]! : t).trim();
}

/** A full path the server can find whatever its working directory: a drive (`D:\`, `D:/`), a share (`\\host`) or `/`. */
export function isFullPath(path: string): boolean {
  return /^([A-Za-z]:[\\/]|[\\/])/.test(path);
}

/**
 * A tidy default name for the dataset at `path`: its folder's name (for a COCO file in `annotations/`, the folder above
 * it; for a `data.yaml`, its folder), a file's extension dropped, made a valid name with no doubled or trailing dashes
 * ("blood cells copy (2)" → "blood-cells-copy-2", ".../bccd/annotations/instances_test.json" → "bccd").
 */
export function nameFromPath(path: string): string {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  let last = parts.at(-1) ?? '';
  if (/\.(json|ya?ml)$/i.test(last)) {
    const dir = parts.at(-2);
    if (/\.ya?ml$/i.test(last) && dir) last = dir;
    else if (dir?.toLowerCase() === 'annotations' && parts.length > 2) last = parts.at(-3)!;
    else last = last.replace(/\.json$/i, '');
  }
  return suggestName(last.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/-{2,}/g, '-')) ?? '';
}

/** `base`, or `base-2`, `base-3`, ... when a dataset is already called that (names compared ignoring case). */
export function uniqueName(base: string, taken: ReadonlySet<string>): string {
  if (!base || !taken.has(base.toLowerCase())) return base;
  for (let n = 2; ; n++) {
    const name = `${base.slice(0, 64 - String(n).length - 1)}-${n}`;
    if (!taken.has(name.toLowerCase())) return name;
  }
}
