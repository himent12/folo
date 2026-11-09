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

/**
 * A touch-only device (`(hover: none) and (pointer: coarse)`: a phone or tablet without a mouse). It gets no shortcuts
 * reference (DESIGN_DIRECTION §9): no palette entry, no "Show shortcuts", and `?` does not open it.
 */
export const TOUCH_ONLY = '(hover: none) and (pointer: coarse)';
export const touchOnly = () => typeof matchMedia !== 'undefined' && matchMedia(TOUCH_ONLY).matches;

/**
 * Every shortcut the reference lists. `single`: a single-character key that Settings can turn off (ui/keyPrefs); the
 * rest (Mod+K, the arrow keys and Enter of a focused list or chart, the image inspector's own keys) always work.
 */
export const shortcutList: { keys: string; label: string; group: string; single?: boolean }[] = [
  { keys: 'Mod+K', label: 'Command palette', group: 'General' },
  { keys: '?', label: 'Keyboard shortcuts', group: 'General', single: true },
  { keys: '[', label: 'Collapse or expand the sidebar', group: 'General', single: true },
  { keys: 'N', label: 'New training run', group: 'General', single: true },
  ...[...mainNav, settingsNav].map((n) => ({ keys: n.keys, label: `Go to ${n.label}`, group: 'Navigation', single: true })),
  // A run, model or dataset page (and Compare): back to its list, when nothing on the page is using Escape.
  { keys: 'Escape', label: 'Back to the list (detail pages)', group: 'Navigation', single: true },
  // Rows of the Runs, Models and Datasets tables (ui/rowNav.ts). J / K enter the list when nothing has focus.
  { keys: 'J K', label: 'Next / previous row', group: 'Lists', single: true },
  { keys: 'ArrowDown ArrowUp', label: 'Next / previous row (focused list)', group: 'Lists' },
  { keys: 'Enter', label: 'Open the row', group: 'Lists' },
  { keys: 'X', label: 'Select the run to compare (Runs)', group: 'Lists', single: true },
  { keys: 'ArrowLeft ArrowRight', label: 'Move the chart cursor (focused chart)', group: 'Charts' },
  { keys: '1–6', label: 'Pick a metric tab (run page)', group: 'Charts', single: true },
  // Compare's side by side: B moves the baseline to the next run, as its Swap buttons do one column at a time.
  { keys: 'B', label: 'Make the next run the baseline (Compare)', group: 'Charts', single: true },
  // The confusion matrix's cells: arrows move the focused cell and the columns page with it, as its ‹ › do.
  { keys: 'ArrowLeft ArrowRight', label: "Move through the confusion matrix's cells, paging its columns (focused matrix)", group: 'Charts' },
  { keys: 'Home End', label: "First / last column of the confusion matrix (focused matrix)", group: 'Charts' },
  // Legend toggles are buttons (aria-pressed): Space on a focused one, whatever the single-key setting.
  { keys: 'Space', label: "Trace or untrace a run's line (Compare's chips)", group: 'Charts' },
  { keys: 'Space', label: "Trace or untrace a class's curve (Model's per class)", group: 'Charts' },
  // The dataset image inspector's own keys: they work while it is open, whatever the setting.
  { keys: 'J K', label: 'Next / previous image (inspector)', group: 'Datasets' },
  { keys: '+ -', label: 'Zoom in / out (inspector)', group: 'Datasets' },
  { keys: '0', label: 'Fit image (inspector)', group: 'Datasets' },
];

/** Where Escape goes back to from a detail page (its list), or null where it does nothing. */
export function backPath(route: { name: string }): string | null {
  return ({ run: '/runs', 'run-compare': '/runs', model: '/models', dataset: '/datasets' } as Record<string, string>)[route.name] ?? null;
}

/**
 * True when nothing that might use Escape itself has focus: the page (body), the main region, or the page heading new
 * pages focus (`data-route-focus`). A focused chart, menu, field or popover keeps its own Escape.
 */
function atRest(el: Element | null): boolean {
  return !el || el === document.body || (el instanceof HTMLElement && (el.id === 'content' || el.hasAttribute('data-route-focus')));
}

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
    if (e.key === 'Escape') {
      const back = backPath(router.route);
      if (back && atRest(document.activeElement) && !document.querySelector('[aria-expanded="true"][aria-haspopup]')) {
        e.preventDefault();
        router.go(back);
      }
      return;
    }
    if (e.key === '?') {
      if (touchOnly()) return;
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

  const onshow = () => !touchOnly() && h.shortcuts();
  window.addEventListener('keydown', onkeydown);
  window.addEventListener(SHOW_SHORTCUTS, onshow);
  return () => {
    window.removeEventListener('keydown', onkeydown);
    window.removeEventListener(SHOW_SHORTCUTS, onshow);
  };
}
