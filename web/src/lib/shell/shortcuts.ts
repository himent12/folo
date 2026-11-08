// Global keyboard shortcuts. Two-key "go to" sequences (G then R) like Linear and GitHub.

import { isTyping } from '$lib/keys';
import { sidebar } from './sidebarState.svelte';
import { router } from '$lib/router.svelte';
import { mainNav, settingsNav } from './nav';

export interface ShortcutHandlers {
  palette: () => void;
  shortcuts: () => void;
}

export const shortcutList: { keys: string; label: string; group: string }[] = [
  { keys: 'Mod+K', label: 'Command palette', group: 'General' },
  { keys: '?', label: 'Keyboard shortcuts', group: 'General' },
  { keys: '[', label: 'Collapse or expand the sidebar', group: 'General' },
  { keys: 'N', label: 'New training run', group: 'General' },
  ...[...mainNav, settingsNav].map((n) => ({ keys: n.keys, label: `Go to ${n.label}`, group: 'Navigation' })),
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
    if (document.querySelector('dialog[open]')) return;
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

  window.addEventListener('keydown', onkeydown);
  return () => window.removeEventListener('keydown', onkeydown);
}
