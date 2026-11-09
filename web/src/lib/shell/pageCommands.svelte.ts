// Commands a page offers in the palette while it is shown ("Stop run", "Resume", "Export", "Copy run id"): the page
// registers them with `pageCommands(() => [...])` in its script, and they leave with it. The palette lists them first,
// under "This page", with no query and for a matching one.

import { untrack } from 'svelte';
import { isTyping } from '$lib/keys';
import { toast } from '$lib/ui/toast.svelte';
import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
import type { IconType } from '$lib/ui/types';

export interface PageCommand {
  id: string;
  label: string;
  icon: IconType;
  /** Other phrasings it answers to. */
  match?: string[];
  /** The key that does the same on the page, shown beside it. */
  keys?: string;
  run: () => void;
}

export const pageCommandList = $state<{ items: PageCommand[] }>({ items: [] });

/**
 * Registers the page's commands for as long as the calling component is mounted. `get` is reactive: a command that
 * stops applying (Stop once the run has ended) leaves the list.
 */
export function pageCommands(get: () => PageCommand[]) {
  $effect(() => {
    const mine = get();
    // Only `get` is tracked: the list is read and written here, so reading it as a dependency would loop.
    untrack(() => (pageCommandList.items = [...pageCommandList.items.filter((c) => !mine.some((m) => m.id === c.id)), ...mine]));
    return () => untrack(() => (pageCommandList.items = pageCommandList.items.filter((c) => !mine.includes(c))));
  });
}

/**
 * Page keys, for a page that offers some (Run's 1–6): `handler` sees single-key presses with the same guards as the
 * global ones (not while typing, a dialog is open, a modifier is held, or single-key shortcuts are off) and returns
 * true when it used the key. Call from a component's script; it stops when the component goes.
 */
export function pageKeys(handler: (key: string) => boolean) {
  $effect(() => {
    const onkeydown = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
      if (!keyPrefs.singleKeys || document.querySelector('dialog[open]')) return;
      if (handler(e.key)) e.preventDefault();
    };
    window.addEventListener('keydown', onkeydown);
    return () => window.removeEventListener('keydown', onkeydown);
  });
}

/** Copies `text` and says so (a toast), or says it could not (no clipboard permission). */
export async function copyText(text: string, what: string) {
  try {
    await navigator.clipboard.writeText(text);
    toast.success(`Copied ${what}`, text);
  } catch (e) {
    toast.error(e, `Could not copy the ${what}`);
  }
}

const RECENT_KEY = 'folo.palette.recent';
const RECENT_MAX = 5;

/** Ids of the palette items chosen last, newest first (none when storage is unavailable). */
export function recentIds(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]');
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string').slice(0, RECENT_MAX) : [];
  } catch {
    return [];
  }
}

/** Remembers a chosen item (page commands are about one page, so they are not kept). */
export function rememberRecent(id: string) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify([id, ...recentIds().filter((x) => x !== id)].slice(0, RECENT_MAX)));
  } catch {
    // Private window or blocked storage: no recents, nothing else changes.
  }
}
