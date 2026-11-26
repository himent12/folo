// Classifies rejected dynamic `import()` errors so recovery picks Retry (network/preload drop) vs Reload (module
// evaluation error cached in the browser's module map until the document reloads).

/**
 * Whether a rejected dynamic `import()` looks like a transient fetch/preload failure that a later `import()` can retry,
 * rather than a module evaluation error (which browsers cache in the module map until the page reloads).
 */
export function isChunkFetchError(error: unknown): boolean {
  let cur: unknown = error;
  for (let depth = 0; cur instanceof Error && depth < 5; depth++) {
    if (cur.name === 'AbortError') return true;
    const msg = cur.message;
    // Vite's modulepreload helper rejects with this when a chunk's companion stylesheet fails to load.
    if (/^Unable to preload CSS\b/i.test(msg)) return true;
    if (
      cur.name === 'TypeError' &&
      /failed to fetch|error loading dynamically imported module|importing a module script failed|load failed|networkerror|chunkloaderror/i.test(msg)
    ) {
      return true;
    }
    cur = (cur as { cause?: unknown }).cause;
  }
  return false;
}

/**
 * Whether a rejected dynamic `import()` failed while evaluating the module (SyntaxError, ReferenceError, top-level
 * runtime throw, etc.). Browsers cache evaluation errors per module URL, so retrying `import()` sends no request and
 * only reloading the page can recover.
 */
export function isChunkEvalError(error: unknown): boolean {
  return !isChunkFetchError(error);
}
