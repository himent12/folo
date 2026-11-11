import { onDocument } from 'fw';

/** The shortcut that starts the run, as the Start buttons' tooltip and the shortcuts reference name it. */
export const START_KEYS = 'Mod+Enter';

/**
 * Enter in a field commits its value and never starts a run (a run may queue, or rent a cloud GPU): the browser's
 * implicit submission is turned off for the form's inputs. Mod+Enter (Ctrl, ⌘ on Apple) starts it from anywhere on
 * the page, as the Start button does, unless another dialog (the command palette) has the keyboard.
 */
export function startKeys(form: () => HTMLFormElement | undefined) {
  onDocument('keydown', (e) => {
    const f = form();
    if (!f || e.key !== 'Enter' || e.isComposing || e.defaultPrevented) return;
    const el = e.target as HTMLElement | null;
    if (e.ctrlKey || e.metaKey) {
      const dialog = el?.closest?.('dialog[open]');
      if (dialog && !f.contains(dialog)) return;
      e.preventDefault();
      f.requestSubmit();
    } else if (el?.tagName === 'INPUT' && f.contains(el)) e.preventDefault();
  });
}
