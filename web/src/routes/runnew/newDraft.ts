// Draft persistence for New run: reads the tab's stored draft for the active server/query, debounces edits into
// sessionStorage, flushes pending writes on unmount/pagehide, and clears only this form's draft on Start.
import { effect, onCleanup, onWindow } from 'fw';
import { clearOwnDraft, DRAFT_KEY, readDraft, saveDraft, type Draft } from './draft';

/** Delay before persisting edits to storage so fast typing writes once instead of on every keystroke. */
export const DRAFT_DEBOUNCE_MS = 300;

/** Tracks the latest mounted New run form in this tab so an older form's Start never clears a newer form's draft. */
let activeOwner: symbol | null = null;

/** Resolves the stored draft for the current server and URL query. */
export function initialDraftState(query: string, server: string, fromId: string | null) {
  const stored = readDraft(server);
  const own = stored?.query === query ? stored : null;
  const draft = own ?? (fromId ? null : stored);
  return { stored, own, draft };
}

export interface DraftPersistenceOptions {
  /** Builds the current draft snapshot (reads reactive form signals). */
  build: () => Draft;
  /** True once the form has finished any initial link/run prefill and may persist edits. */
  ready: () => boolean;
  /** Debounce delay in ms for edits after the initial snapshot (default `DRAFT_DEBOUNCE_MS`). */
  delayMs?: number;
  /** Storage provider override for tests. */
  store?: () => Storage;
}

export interface DraftPersistence {
  /** Flushes any pending debounced draft write synchronously (unless the run has already started). */
  flush: () => void;
  /** Marks the form as started, cancels pending writes, and clears the draft if this form still owns it. */
  markStarted: () => void;
}

function readRaw(store: () => Storage): string | null {
  try {
    return store().getItem(DRAFT_KEY);
  } catch {
    return null;
  }
}

/**
 * Persists the New run form's draft reactively. The first ready snapshot writes immediately; subsequent edits are
 * debounced by `delayMs` and flushed synchronously on cleanup or `pagehide` while storage still belongs to this form.
 */
export function useDraftPersistence(opts: DraftPersistenceOptions): DraftPersistence {
  const delayMs = opts.delayMs ?? DRAFT_DEBOUNCE_MS;
  const store = opts.store ?? (() => sessionStorage);
  const owner = Symbol('runNewDraft');
  activeOwner = owner;

  let started = false;
  let initialized = false;
  let mine: string | null = null;
  let pending: Draft | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function clearTimer() {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  }

  function writeIfOwned(d: Draft) {
    // Do not clobber a draft overwritten externally or by a newer form instance.
    if (activeOwner !== owner) return;
    if (mine !== null) {
      const current = readRaw(store);
      if (current !== null && current !== mine) return;
    }
    mine = saveDraft(d, store);
  }

  function flush() {
    clearTimer();
    if (started || !pending) return;
    const d = pending;
    pending = null;
    writeIfOwned(d);
  }

  function markStarted() {
    started = true;
    clearTimer();
    pending = null;
    if (activeOwner === owner && mine !== null) clearOwnDraft(mine, store);
  }

  effect(() => {
    const d = opts.build();
    if (!opts.ready() || started) return;
    const raw = JSON.stringify(d);
    if (raw === mine && !pending) return;
    if (!initialized) {
      initialized = true;
      writeIfOwned(d);
      return;
    }
    pending = d;
    clearTimer();
    timer = setTimeout(flush, delayMs);
  });

  if (typeof window !== 'undefined') onWindow('pagehide', flush);
  onCleanup(flush);

  return { flush, markStarted };
}
