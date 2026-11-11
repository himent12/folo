// Agent labeling's rules (docs/API.md, Agent labeling), shared by the forms that check what a person typed and the
// in-browser mock that answers as the server would.

import type { LabelAgentId } from './api/types';

/** The feature a server lists in `/api/system` when it has the `/api/label` endpoints. */
export const LABEL_FEATURE = 'agent_labeling';

/** The agents' product names. */
export const AGENT_NAME: Record<LabelAgentId, string> = { claude: 'Claude Code', codex: 'Codex' };

export const JOB_NAME_MAX = 100;
export const CLASSES_MAX = 80;
export const CLASS_NAME_MAX = 64;
export const INSTRUCTIONS_MAX = 4000;
export const NOTE_MAX = 2000;

/**
 * A folder path as people paste it, without what is not part of it: surrounding whitespace and one pair of matching
 * quotes (Windows' "Copy as path" adds them), as the server reads it.
 */
export function folderPath(raw: string): string {
  const p = raw.trim();
  const q = p[0];
  return p.length >= 2 && (q === '"' || q === "'") && p.at(-1) === q ? p.slice(1, -1).trim() : p;
}

/**
 * Why a cleaned folder path cannot be a folder on the server's machine, or null: a relative one would be read from
 * wherever the server was started (a drive path, a `/` path and a `\\server` share are full paths).
 */
export function folderPathError(path: string): string | null {
  if (!path || /^([A-Za-z]:[\\/]|[\\/])/.test(path)) return null;
  return 'Give the full path, such as D:/photos: a relative one would be read from wherever folo serve was started.';
}

/** What is wrong with one class name, or null: not blank, at most 64 characters, no `,`, `:` or line break. */
export function classNameError(name: string): string | null {
  const n = name.trim();
  if (!n) return 'A class needs a name.';
  if (n.length > CLASS_NAME_MAX) return `"${n.slice(0, 24)}…" is longer than ${CLASS_NAME_MAX} characters.`;
  if (/[,:\r\n]/.test(n)) return `"${n}" has a comma, a colon or a line break.`;
  return null;
}

/** What is wrong with a class list, or null: 1 to 80 valid names, distinct when compared without case. */
export function classesError(classes: readonly string[]): string | null {
  if (classes.length === 0) return 'Add at least one class.';
  if (classes.length > CLASSES_MAX) return `At most ${CLASSES_MAX} classes.`;
  const seen = new Set<string>();
  for (const c of classes) {
    const e = classNameError(c);
    if (e) return e;
    const k = c.trim().toLowerCase();
    if (seen.has(k)) return `"${c.trim()}" is listed twice.`;
    seen.add(k);
  }
  return null;
}

/**
 * Once a job has boxes, which refer to classes by position, its class list may only be renamed or appended to: why
 * `next` is not such a change of `prev`, or null when it is.
 */
export function classChangeError(prev: readonly string[], next: readonly string[], hasBoxes: boolean): string | null {
  if (!hasBoxes) return null;
  if (next.length < prev.length) return 'Labels refer to classes by their place in the list, so classes can be renamed or added, not removed.';
  return null;
}
