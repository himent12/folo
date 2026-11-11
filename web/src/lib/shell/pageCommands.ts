// fw port of pageCommands.svelte.ts (the same registry, on an fw signal). One difference: a page's commands leave the
// list when the page goes. The Svelte registry keeps them (its cleanup looks for the page's own objects in a list of
// $state proxies of them, and finds none), so a page left behind still offered its "Stop run" until another page
// registered commands with the same ids.
//
// Commands a page offers in the palette while it is shown ("Stop run", "Resume", "Export", "Copy run id"): the page
// registers them with `pageCommands(() => [...])` in its body, and they leave with it. The palette lists them first,
// under "This page", with no query and for a matching one.

import { effect, signal, untrack } from 'fw';
import { isTyping } from '$lib/keys';
import { toast } from '$lib/ui/toast';
import { keyPrefs } from '$lib/ui/keyPrefs';
import type { IconType } from '$lib/ui/types.fw';

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

const items = signal<PageCommand[]>([]);
export const pageCommandList = {
  get items(): PageCommand[] {
    return items.get();
  },
  set items(v: PageCommand[]) {
    items.set(v);
  },
};

/**
 * Registers the page's commands for as long as the calling component is mounted. `get` is reactive: a command that
 * stops applying (Stop once the run has ended) leaves the list.
 */
export function pageCommands(get: () => PageCommand[]) {
  effect(() => {
    const mine = get();
    // Only `get` is tracked: the list is read and written here, so reading it as a dependency would loop.
    untrack(() => {
      pageCommandList.items = [...pageCommandList.items.filter((c) => !mine.some((m) => m.id === c.id)), ...mine];
    });
    return () =>
      untrack(() => {
        pageCommandList.items = pageCommandList.items.filter((c) => !mine.includes(c));
      });
  });
}

/**
 * Page keys, for a page that offers some (Run's 1–6): `handler` sees single-key presses with the same guards as the
 * global ones (not while typing, a dialog is open, a modifier is held, or single-key shortcuts are off) and returns
 * true when it used the key. Call from a component's body; it stops when the component goes.
 */
export function pageKeys(handler: (key: string) => boolean) {
  effect(() => {
    const onkeydown = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
      if (!keyPrefs.singleKeys || document.querySelector('dialog[open]')) return;
      if (handler(e.key)) e.preventDefault();
    };
    window.addEventListener('keydown', onkeydown);
    return () => window.removeEventListener('keydown', onkeydown);
  });
}

/**
 * A dialog's own keys (the labeling editor's): whether a key press is for `dialog`, with the page keys' guards (not
 * while typing, with Ctrl or Cmd held, or once something else used it) and `dialog` the topmost one, which is where
 * focus is: a dialog opened over it (the palette, a confirmation) takes focus, and the page under a modal cannot have
 * it. Alt and Shift are the caller's to read. Since they only work with focus in the dialog, they work whether or not
 * single-key shortcuts are on, as the image inspector's do.
 */
export function dialogKey(e: KeyboardEvent, dialog: Element | null | undefined): boolean {
  if (e.defaultPrevented || e.metaKey || e.ctrlKey || isTyping(e)) return false;
  return !!dialog && dialog instanceof HTMLDialogElement && dialog.open && e.target instanceof Node && dialog.contains(e.target);
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
