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
