// Types of json-scan.mjs, for Studio (web/src/lib/design/themes.ts imports it).

/** The deepest nesting serde_json accepts: 127 arrays and objects, the top-level object counted (128 is refused). */
export const JSON_MAX_DEPTH: number;

/**
 * Why `text` is not JSON Desktop would accept, though JSON.parse might (the detail of `(file): not valid JSON (…)`), or
 * undefined. Text that is not JSON at all may pass: JSON.parse refuses it next.
 */
export function jsonProblem(text: string): string | undefined;
