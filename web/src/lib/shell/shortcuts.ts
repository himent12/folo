// Global keyboard shortcuts. Two-key "go to" sequences (G then R) like Linear and GitHub. The single-character ones
// (everything but Mod+K) can be turned off in Settings (ui/keyPrefs).

import { signal } from 'fw';
import { isTyping } from '$lib/keys';
import { keyPrefs } from '$lib/ui/keyPrefs';
import { sidebar } from './sidebarState';
import { router } from '$lib/router';
import { mainNav, settingsNav } from './nav';

export interface ShortcutHandlers {
  palette: () => void;
  shortcuts: () => void;
  /** Called when a valid G-chord navigation sequence starts (never while typing, modified, or inside a dialog). */
  gChord?: () => void;
  /** Called on Escape before detail-page back navigation; return true if handled (e.g. cancelling a pending dialog load). */
  escape?: () => boolean;
}

/** How long a G-chord stays armed waiting for the second navigation key. */
export const G_CHORD_MS = 1200;

/** Reactive flag: true while a G-chord ("go to") sequence is armed and waiting for its second key. */
export const gChord = signal(false);

const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta']);

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
 * Every shortcut the reference lists, one row per action. `single`: `keys` is a single-character key that Settings can
 * turn off (ui/keyPrefs); the rest (Mod+K, the arrow keys and Enter of a focused list or chart, the image inspector's
 * own keys) always work. `also`: other keys for the same action, which always work (a row's J K and its ↓ ↑), so an
 * action is listed once with both, never as two rows with one name.
 */
export const shortcutList: { keys: string; also?: string; label: string; group: string; single?: boolean }[] = [
  { keys: 'Mod+K', label: 'Command palette', group: 'General' },
  { keys: '?', label: 'Keyboard shortcuts', group: 'General', single: true },
  { keys: '[', label: 'Collapse or expand the sidebar', group: 'General', single: true },
  { keys: 'N', label: 'New training run', group: 'General', single: true },
  { keys: 'Mod+Enter', label: 'Start the run (New run)', group: 'General' },
  ...[...mainNav, settingsNav].map((n) => ({ keys: n.keys, label: `Go to ${n.label}`, group: 'Navigation', single: true })),
  // A run, model, dataset or benchmark page (and Compare): back to its list, when nothing on the page is using Escape.
  { keys: 'Escape', label: 'Back to the list (detail pages)', group: 'Navigation', single: true },
  // Rows of the Runs, Models and Datasets tables (ui/rowNav.ts). J / K enter the list when nothing has focus; the
  // arrows move within a focused one.
  { keys: 'J K', also: 'ArrowDown ArrowUp', label: 'Next / previous row', group: 'Lists', single: true },
  { keys: 'Enter', label: 'Open the row', group: 'Lists' },
  { keys: 'X', label: 'Select the run to compare (Runs)', group: 'Lists', single: true },
  { keys: 'ArrowLeft ArrowRight', label: 'Move the chart cursor (focused chart)', group: 'Charts' },
  // Compare's side by side: B moves the baseline to the next run or benchmark entry; 1–4 pin an entry on benchmark compare, Escape unpins.
  { keys: 'B', label: 'Make the next run or entry the baseline (Compare)', group: 'Charts', single: true },
  { keys: '1–4', label: 'Pin or unpin an entry across charts (Benchmark compare)', group: 'Charts', single: true },
  { keys: 'Escape', label: 'Unpin the highlighted entry (Benchmark compare)', group: 'Charts', single: true },
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
  { keys: 'B', label: 'Show or hide boxes (inspector)', group: 'Datasets' },
  // The labeling editor's own keys: they work while it is the topmost dialog, whatever the setting.
  { keys: 'J K', label: 'Next / previous image (labeling editor)', group: 'Datasets' },
  { keys: '1–9', label: "Set the active drawing or selected box's class (labeling editor)", group: 'Datasets' },
  { keys: 'Delete', also: 'Backspace', label: 'Delete the selected box (labeling editor)', group: 'Datasets' },
  { keys: 'ArrowLeft ArrowRight ArrowUp ArrowDown', label: 'Move the selected box a pixel, with Shift ten (labeling editor)', group: 'Datasets' },
  { keys: 'Alt ArrowLeft ArrowRight ArrowUp ArrowDown', label: 'Resize the selected box from its bottom-right corner, with Shift by ten pixels (labeling editor)', group: 'Datasets' },
  { keys: 'Mod+Z', label: 'Undo an edit of the boxes (labeling editor)', group: 'Datasets' },
  { keys: 'Mod+Shift+Z', also: 'Mod+Y', label: 'Redo it (labeling editor)', group: 'Datasets' },
  { keys: 'Escape', label: 'Drop the selection, then close (labeling editor)', group: 'Datasets' },
];

/** Where Escape goes back to from a detail page (its list), or null where it does nothing. */
export function backPath(route: { name: string; params?: Record<string, string> }): string | null {
  const bench = `/benchmarks/${encodeURIComponent(route.params?.id ?? '')}`;
  const back: Record<string, string> = {
    run: '/runs',
    'run-compare': '/runs',
    model: '/models',
    dataset: '/datasets',
    'label-jobs': '/datasets',
    'label-job': '/datasets/label',
    benchmark: '/benchmarks',
    // An entry's result and a comparison go back to their benchmark, a loop to its benchmark's loops.
    'bench-entry': bench,
    'bench-compare': bench,
    'bench-loop': `${bench}?tab=loops`,
  };
  return back[route.name] ?? null;
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
  let chordTimer: ReturnType<typeof setTimeout> | undefined;

  const clearChord = () => {
    if (chordTimer !== undefined) {
      clearTimeout(chordTimer);
      chordTimer = undefined;
    }
    pendingG = 0;
    if (gChord.peek()) gChord.set(false);
  };

  const onkeydown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === 'k') {
      clearChord();
      e.preventDefault();
      h.palette();
      return;
    }
    if (MODIFIER_KEYS.has(e.key)) return;
    if (e.defaultPrevented || e.isComposing || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) {
      if (pendingG) clearChord();
      return;
    }
    if (document.querySelector('dialog[open]')) {
      if (pendingG) clearChord();
      return;
    }
    if (!pendingG && !e.repeat && e.key === 'Escape' && h.escape?.()) {
      e.preventDefault();
      return;
    }
    if (!keyPrefs.singleKeys) {
      if (pendingG) clearChord();
      return;
    }
    if (e.repeat) return;
    const k = e.key.toLowerCase();
    if (pendingG) {
      const live = Date.now() - pendingG < G_CHORD_MS;
      clearChord();
      if (live) {
        if (e.key === 'Escape') {
          e.preventDefault();
          return;
        }
        const target = [...mainNav, settingsNav].find((n) => n.keys === `G ${k.toUpperCase()}`);
        if (target) {
          e.preventDefault();
          router.go(target.path);
        }
        return;
      }
      if (e.key === 'Escape' && h.escape?.()) {
        e.preventDefault();
        return;
      }
    }
    if (k === 'g' && !e.shiftKey) {
      pendingG = Date.now();
      gChord.set(true);
      clearTimeout(chordTimer);
      chordTimer = setTimeout(clearChord, G_CHORD_MS);
      h.gChord?.();
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
  window.addEventListener('blur', clearChord);
  window.addEventListener(SHOW_SHORTCUTS, onshow);
  return () => {
    clearChord();
    window.removeEventListener('keydown', onkeydown);
    window.removeEventListener('blur', clearChord);
    window.removeEventListener(SHOW_SHORTCUTS, onshow);
  };
}
