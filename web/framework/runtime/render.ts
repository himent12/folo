// Mounting: `render(() => <App />, el)` appends the app to `el` and returns the function that removes it.

import { createRoot, untrack } from './core';
import { clearLeaves, insert } from './dom';
import { delegateRoot } from './events';

/**
 * Renders `code()` at the end of `container` (port of Svelte's `mount`) and makes `container` handle delegated
 * events. Render effects have run when it returns; user effects run at the next flush (`flush()` or a microtask).
 * The returned function disposes everything and removes the nodes (port of `unmount`).
 */
export function render(code: () => JSX.Element, container: Element | DocumentFragment): () => void {
  const undo = delegateRoot(container);
  const start = document.createTextNode('');
  const end = document.createTextNode('');
  container.appendChild(start);
  container.appendChild(end);
  const removeNodes = () => {
    let n: Node | null = start;
    while (n !== null) {
      const next: Node | null = n === end ? null : n.nextSibling;
      clearLeaves(n);
      n.parentNode?.removeChild(n);
      n = next;
    }
  };
  let dispose!: () => void;
  try {
    createRoot((d) => {
      dispose = d;
      insert(container, untrack(code), end);
    });
  } catch (e) {
    try {
      dispose?.();
    } catch {
      // Keep the original render error.
    } finally {
      undo();
      removeNodes();
    }
    throw e;
  }
  let unmounted = false;
  return () => {
    if (unmounted) return;
    unmounted = true;
    try {
      dispose();
    } finally {
      undo();
      removeNodes();
    }
  };
}
