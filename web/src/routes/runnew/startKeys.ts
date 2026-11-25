// The keys of New run: Mod+Enter starts the run, Enter in a field never does, and Escape returns when no nested
// dialog, popup, or select is active.
import { onDocument } from 'fw';

/** The shortcut that starts the run, as the Start buttons' tooltip and the shortcuts reference name it. */
export const START_KEYS = 'Mod+Enter';

export interface StartKeysOptions {
  /** Called on Escape when no nested dialog, menu/popup, or native select owns the key. */
  onCancel?: () => void;
}

/** True when a native Popover API element (e.g. InfoButton's Class balance popover) is currently open. */
function hasOpenPopover(): boolean {
  try {
    return !!document.querySelector('[popover]:popover-open');
  } catch {
    return false;
  }
}

/**
 * Enter in a field commits its value and never starts a run (a run may queue, or rent a cloud GPU): the browser's
 * implicit submission is turned off for the form's inputs. Mod+Enter (Ctrl, ⌘ on Apple) starts it from anywhere on
 * the page, as the Start button does, unless another dialog (the command palette) has the keyboard. Escape cancels
 * and returns cleanly when no nested dialog, popup, tooltip, or select is open.
 */
export function startKeys(form: () => HTMLFormElement | undefined, opts?: StartKeysOptions) {
  onDocument('keydown', (e) => {
    const f = form();
    if (!f || e.isComposing || e.defaultPrevented) return;
    const el = e.target as HTMLElement | null;
    if (e.key === 'Escape') {
      if (!opts?.onCancel || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
      if (
        document.querySelector('dialog[open], [aria-expanded="true"][aria-haspopup], #folo-tooltip[data-show], .info-pop.open') ||
        hasOpenPopover()
      ) {
        return;
      }
      if (el?.tagName === 'SELECT' || el?.closest?.('select, [role="dialog"], [role="listbox"], [role="menu"]')) return;
      e.preventDefault();
      opts.onCancel();
      return;
    }
    if (e.key !== 'Enter') return;
    if (e.ctrlKey || e.metaKey) {
      const dialog = el?.closest?.('dialog[open]');
      if (dialog && !f.contains(dialog)) return;
      e.preventDefault();
      f.requestSubmit();
    } else if (el?.tagName === 'INPUT' && f.contains(el)) e.preventDefault();
  });
}
