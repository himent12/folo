// Global keyboard shortcuts. Two-key "go to" sequences (G then R) like Linear and GitHub. The single-character ones
// (everything but Mod+K) can be turned off in Settings (ui/keyPrefs).

import { isTyping } from '$lib/keys';
import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
import { sidebar } from './sidebarState.svelte';
import { router } from '$lib/router.svelte';
import { mainNav, settingsNav } from './nav';

export interface ShortcutHandlers {
  palette: () => void;
  shortcuts: () => void;
}

/** Opens the shortcuts dialog from anywhere (Settings' button), whether or not the `?` key is on. */
export const SHOW_SHORTCUTS = 'folo:shortcuts';

export function showShortcuts() {
  window.dispatchEvent(new Event(SHOW_SHORTCUTS));
}

export const shortcutList: { keys: string; label: string; group: string }[] = [
  { keys: 'Mod+K', label: 'Command palette', group: 'General' },
  { keys: '?', label: 'Keyboard shortcuts', group: 'General' },
  { keys: '[', label: 'Collapse or expand the sidebar', group: 'General' },
  { keys: 'N', label: 'New training run', group: 'General' },
  ...[...mainNav, settingsNav].map((n) => ({ keys: n.keys, label: `Go to ${n.label}`, group: 'Navigation' })),
  // Rows of the Runs, Models and Datasets tables (ui/rowNav.ts). J / K enter the list when nothing has focus.
  { keys: 'J K', label: 'Next / previous row', group: 'Lists' },
  { keys: 'ArrowDown ArrowUp', label: 'Next / previous row (focused list)', group: 'Lists' },
  { keys: 'Enter', label: 'Open the row', group: 'Lists' },
  { keys: 'X', label: 'Select the run to compare (Runs)', group: 'Lists' },
  { keys: 'ArrowLeft ArrowRight', label: 'Move the chart cursor (focused chart)', group: 'Charts' },
  { keys: 'J K', label: 'Next / previous image (inspector)', group: 'Datasets' },
  { keys: '+ -', label: 'Zoom in / out (inspector)', group: 'Datasets' },
  { keys: '0', label: 'Fit image (inspector)', group: 'Datasets' },
];

export function installShortcuts(h: ShortcutHandlers): () => void {
  let pendingG = 0;

  const onkeydown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      h.palette();
      return;
    }
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
    if (!keyPrefs.singleKeys || document.querySelector('dialog[open]')) return;
    const k = e.key.toLowerCase();
    if (pendingG && Date.now() - pendingG < 1200) {
      pendingG = 0;
      const target = [...mainNav, settingsNav].find((n) => n.keys === `G ${k.toUpperCase()}`);
      if (target) {
        e.preventDefault();
        router.go(target.path);
      }
      return;
    }
    if (k === 'g') {
      pendingG = Date.now();
      return;
    }
    if (e.key === '?') {
      e.preventDefault();
      h.shortcuts();
    } else if (e.key === '[') {
      e.preventDefault();
      sidebar.toggle();
    } else if (k === 'n') {
      e.preventDefault();
      router.go('/runs/new');
    }
  };

  const onshow = () => h.shortcuts();
  window.addEventListener('keydown', onkeydown);
  window.addEventListener(SHOW_SHORTCUTS, onshow);
  return () => {
    window.removeEventListener('keydown', onkeydown);
    window.removeEventListener(SHOW_SHORTCUTS, onshow);
  };
}
